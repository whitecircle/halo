"""The ``full_determinism`` backward-replay body the EP determinism GPU suite runs: every gradient must match bit for
bit across repeats. DeepEP's default atomic receive order changes each expert's gradient summation, so expert weights
drift and two runs diverge in a few steps."""

import argparse

import torch
from accelerate.state import GradientState
from transformers.trainer_utils import enable_full_determinism

from halo.distributed.expert_parallel.deepep.backends import _ElasticBackend
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from tests.common.distributed import world_all
from tests.common.ep_reference import ep_layers, random_token_batch
from tests.common.rank_log import log, log_all
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

WORLD_SIZE = 2
SEED = 1234
BATCH, SEQ = 2, 1024
REPEATS = 3
# ParallelismConfig kwargs and tiny-config overrides per mode.
MODES = {
    "ep2": ({"ep_size": 2}, {}),
    "ep2_top1": ({"ep_size": 2}, {"num_experts_per_tok": 1}),
    "ep2_loop": ({"ep_size": 2, "use_grouped_gemm": False}, {}),
    "ep2_legacy": ({"ep_size": 2, "ep_buffer_backend": "legacy"}, {}),
    "ep1": ({"ep_size": 1}, {}),
    "etp2": ({"ep_size": 1, "expert_tp_size": 2}, {}),
}


def determinism_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(TINY_MOE_FAMILIES), required=True)
    parser.add_argument("--mode", choices=sorted(MODES), default="ep2")
    return parser


def _backward_pass(model, ids, labels) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, labels=labels).loss
    loss.backward()
    grads = {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.grad is not None}
    return loss.detach().clone(), grads


def _mismatched(reference: dict[str, torch.Tensor], repeat: dict[str, torch.Tensor], names: set[str]) -> set[str]:
    """Names whose gradient one pass lacks or that differ in any bit; a gradient both passes lack (an expert no token
    reached) matches."""
    return {
        name
        for name in names
        if (name in reference) != (name in repeat)
        or (name in reference and not torch.equal(reference[name], repeat[name]))
    }


def run_backward_replay(ctx, *, family: str, mode: str) -> dict:
    """``REPEATS`` forward/backward passes under ``mode``; returns the harness result."""
    parallelism, overrides = MODES[mode]
    GradientState()._set_sync_gradients(True)
    torch.manual_seed(SEED)
    model = tiny_family_model(TINY_MOE_FAMILIES[family], overrides=overrides).to(torch.bfloat16)
    model = apply_ep_layers(model.to(ctx.device), ParallelismConfig(**parallelism).ep_config())
    model.train()
    layers = ep_layers(model)
    expert_params = {id(param) for layer in layers for _, param in layer.expert_named_params()}
    expert_names = {name for name, param in model.named_parameters() if id(param) in expert_params}

    # After the load, as Trainer.__init__ does, so the mode is on before the first dispatch builds a buffer.
    enable_full_determinism(SEED)
    vocab_size = model.get_input_embeddings().num_embeddings
    ids, labels = random_token_batch(vocab_size, BATCH, SEQ, ctx.device, seed=SEED + ctx.rank)

    reference_loss, reference = _backward_pass(model, ids, labels)
    other_names = set(reference) - expert_names
    checks = {
        # Vacuous unless this rank's experts trained. Not every expert: a random-init router can starve one (Zaya's
        # tiny model).
        "expert_grads_present": any(name in reference and bool(reference[name].any()) for name in expert_names),
    }
    # Checked directly as well: at this size the atomic receive order can repeat by chance.
    elastic = [layer.dispatcher.backend for layer in layers if isinstance(layer.dispatcher.backend, _ElasticBackend)]
    if elastic:
        checks["dispatch_buffers_deterministic"] = all(backend._arena.deterministic for backend in elastic)
    expert_diffs, other_diffs, loss_equal = set(), set(), True
    for _ in range(REPEATS - 1):
        loss, grads = _backward_pass(model, ids, labels)
        loss_equal &= torch.equal(loss, reference_loss)
        expert_diffs |= _mismatched(reference, grads, expert_names)
        other_diffs |= _mismatched(reference, grads, other_names)
    checks["loss_bit_identical"] = loss_equal
    checks["expert_grads_bit_identical"] = not expert_diffs
    checks["other_grads_bit_identical"] = not other_diffs
    if expert_diffs or other_diffs:
        log_all(f"  differing expert grads {sorted(expert_diffs)[:6]}, other grads {sorted(other_diffs)[:6]}")
    log(f"  {family} --mode {mode}: {len(expert_names)} expert / {len(other_names)} other grads compared")

    checks["every_rank_passed"] = world_all(all(checks.values()), ctx.device)
    return {"checks": checks, "metrics": {"expert_grads": len(expert_names), "differing": len(expert_diffs)}}
