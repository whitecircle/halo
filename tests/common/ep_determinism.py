"""The ``full_determinism`` backward-replay body the EP determinism GPU suites run.

DeepEP's default dispatch claims each receive slot with an atomic, so the order an expert's tokens
arrive in changes from one dispatch to the next, and with it the summation order of that expert's
weight gradient. Attention, router, norm and embedding gradients stay bit-identical while every expert
weight drifts in its last bits, and two runs diverge within a few steps. Under deterministic-algorithms
mode the dispatcher builds its buffer in DeepEP's deterministic mode, and switches off the
uninitialized-memory fill that DeepEP refuses to run beside (left on, the first dispatch asserts).

The body turns determinism on exactly as HF's ``Trainer`` does for ``full_determinism``
(``enable_full_determinism``, after the load), then runs the same forward and backward on the same
per-rank batch ``REPEATS`` times on a family's tiny model and requires the loss and every gradient to
match the first pass bit for bit. ``MODES`` are the two-rank layouts, each a distinct dispatch or
expert-compute path:

  ep2          — DeepEP V2 dispatch, grouped GEMM, atomic-free fused permute (``top_k >= ep_size``).
  ep2_top1     — top-1 routing, so the grouped path permutes through ``index_select`` / ``index_add_``
                 (families whose config spells the top-k ``num_experts_per_tok``).
  ep2_loop     — the per-expert loop (``use_grouped_gemm: false``).
  ep2_legacy   — the DeepEP V1 buffer (``ep_buffer_backend: legacy``).
  ep1          — experts replicated on every rank, no dispatch.
  etp2         — pure expert-TP: expert FFNs sharded, token-space all-reduce, no dispatch.
"""

import argparse
from collections.abc import Iterable

import torch
from accelerate.state import GradientState
from transformers.trainer_utils import enable_full_determinism

from src.distributed.expert_parallel.dispatcher import _ElasticBackend
from src.distributed.expert_parallel.patching import create_ep_buffers, patch_moe_model_for_ep
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.distributed import world_all
from tests.common.ep_reference import ep_layers, random_token_batch
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model
from tests.common.utils import log, log_all

WORLD_SIZE = 2
SEED = 1234
BATCH, SEQ = 2, 1024
REPEATS = 3
# The families the representative suite runs; the sweep runs ep2 on every other one.
REPRESENTATIVE_FAMILIES = ("gpt_oss", "qwen3_moe")
# ParallelismConfig kwargs and tiny-config overrides per mode.
MODES = {
    "ep2": ({"ep_size": 2}, {}),
    "ep2_top1": ({"ep_size": 2}, {"num_experts_per_tok": 1}),
    "ep2_loop": ({"ep_size": 2, "use_grouped_gemm": False}, {}),
    "ep2_legacy": ({"ep_size": 2, "ep_buffer_backend": "legacy"}, {}),
    "ep1": ({"ep_size": 1}, {}),
    "etp2": ({"ep_size": 1, "expert_tp_size": 2}, {}),
}


def determinism_parser(families: Iterable[str]) -> argparse.ArgumentParser:
    """The CLI a determinism suite takes, over the ``families`` it runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(families), required=True)
    parser.add_argument("--mode", choices=sorted(MODES), default="ep2")
    return parser


def _backward_pass(model, ids, labels) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, labels=labels).loss
    loss.backward()
    grads = {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.grad is not None}
    return loss.detach().clone(), grads


def _mismatched(reference: dict[str, torch.Tensor], repeat: dict[str, torch.Tensor], names: set[str]) -> set[str]:
    """Names in ``names`` whose gradient one pass has and the other lacks, or that differ in any bit. A
    gradient both passes lack (an expert no token reached) matches."""
    return {
        name
        for name in names
        if (name in reference) != (name in repeat)
        or (name in reference and not torch.equal(reference[name], repeat[name]))
    }


def run_backward_replay(ctx, *, family: str, mode: str) -> dict:
    """``REPEATS`` forward/backward passes of ``family``'s tiny model under ``mode``; returns the harness
    result."""
    parallelism, overrides = MODES[mode]
    GradientState()._set_sync_gradients(True)
    torch.manual_seed(SEED)
    model = tiny_family_model(TINY_MOE_FAMILIES[family], overrides=overrides).to(torch.bfloat16)
    model = patch_moe_model_for_ep(model.to(ctx.device), ParallelismConfig(**parallelism).create_ep_config())
    create_ep_buffers(model)
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
        "deterministic_algorithms_on": torch.are_deterministic_algorithms_enabled(),
        # Vacuous unless this rank's experts received tokens and trained. Not every expert: a random-init
        # router can starve one (Zaya's tiny model sends nothing to rank 1 in its last layers).
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
