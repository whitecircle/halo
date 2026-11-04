#!/usr/bin/env python
"""Gradient checkpointing must not silence the bias-update balancer, on a real DeepEP EP layer.

Grad mode can't pick which pass counts. Reentrant runs its original with grad off and recompute with grad on;
non-reentrant has it on for both. A ``torch.is_grad_enabled()`` gate moves the recording to the recompute, so a skipped
backward drops the microbatch, and double-counts in the second case.
"""

import sys

import torch
import torch.distributed as dist
from transformers import GptOssConfig, GptOssForCausalLM, TrainerState

from halo.callbacks.balancing import BalancingBiasCallback, apply_balancing_mode
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.distributed.expert_parallel.parallelize import (
    apply_ep_layers,
)
from halo.distributed.parallelism_config import ParallelismConfig
from halo.models.gradient_checkpointing import enable_scoped_gradient_checkpointing
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_GPTOSS_CONFIG
from tests.common.rank_log import log

EP_SIZE = 2
BATCH = 2
SEQ_LEN = 32
TOP_K = TINY_GPTOSS_CONFIG["num_experts_per_tok"]
NUM_EXPERTS = TINY_GPTOSS_CONFIG["num_local_experts"]
NUM_LAYERS = TINY_GPTOSS_CONFIG["num_hidden_layers"]
TOKENS = BATCH * SEQ_LEN
# Large enough that a bf16-rounded write would show; the update is a pure sign step.
GAMMA = 0.125
SEED = 1234


def build_model(device):
    """bf16, the DeepEP dispatch dtype every EP run trains in."""
    torch.manual_seed(SEED)
    model = GptOssForCausalLM(GptOssConfig(**TINY_GPTOSS_CONFIG)).to(torch.bfloat16).to(device)
    return model


def build_batch(device, rank: int) -> dict:
    """EP is orthogonal to DP, so each rank routes its own tokens."""
    generator = torch.Generator(device="cpu").manual_seed(SEED + rank)
    input_ids = torch.randint(
        0, TINY_GPTOSS_CONFIG["vocab_size"], (BATCH, SEQ_LEN), generator=generator, device="cpu"
    ).to(device)
    return {"input_ids": input_ids, "labels": input_ids}


def read_counters(ep_layers) -> list[torch.Tensor | None]:
    return [None if l.expert_load_counter is None else l.expert_load_counter.clone() for l in ep_layers]


def train_step(model, ep_layers, batch, *, after_forward=None) -> tuple[list[torch.Tensor | None], float]:
    """``after_forward`` sees the counters between forward and backward, the only way to tell which pass recorded
    them."""
    for layer in ep_layers:
        layer.expert_load_counter = None
    model.train()
    model.zero_grad(set_to_none=True)
    outputs = model(**batch)
    if after_forward is not None:
        after_forward(read_counters(ep_layers))
    outputs.loss.backward()
    return read_counters(ep_layers), float(outputs.loss)


def frozen_forward(model, ep_layers, batch) -> list[torch.Tensor | None]:
    for layer in ep_layers:
        layer.expert_load_counter = None
    model.train()
    with torch.no_grad():
        model(**batch)
    return read_counters(ep_layers)


def all_checkpoint_funcs_scoped(model) -> bool:
    funcs = [
        module._gradient_checkpointing_func
        for module in model.modules()
        if getattr(module, "_gradient_checkpointing_func", None) is not None
    ]
    return bool(funcs) and all(getattr(func, "_ep_scoped", False) for func in funcs)


def counts_match_expected(counters: list[torch.Tensor | None]) -> bool:
    """Exactly one count per (token, selected expert) of this microbatch."""
    return len(counters) == NUM_LAYERS and all(
        c is not None and int(c.sum().item()) == TOKENS * TOP_K for c in counters
    )


def counters_equal(left: list[torch.Tensor | None], right: list[torch.Tensor | None]) -> bool:
    return len(left) == len(right) and all(
        a is not None and b is not None and torch.equal(a, b) for a, b in zip(left, right, strict=True)
    )


@gpu_test_main(exact_world_size=2, prefix="ep_gc_bias")
def run(ctx):
    checks, metrics = {}, {}

    parallelism_config = ParallelismConfig(ep_size=EP_SIZE)
    model = build_model(ctx.device)
    apply_ep_layers(model, parallelism_config.ep_config())

    ep_layers = [module for module in model.modules() if isinstance(module, EPMoELayerBase)]
    checks["ep_layers_patched"] = len(ep_layers) == NUM_LAYERS
    checks["each_rank_holds_an_expert_shard"] = bool(ep_layers) and all(
        layer.experts_per_rank == NUM_EXPERTS // EP_SIZE for layer in ep_layers
    )

    # GptOss adopts the hub router's `router.bias`, so this measures the tensor an export carries.
    apply_balancing_mode(model, "bias_update", is_moe=True)
    checks["balancing_bias_is_the_native_router_bias"] = all(
        layer.balancing_biases is layer.router.bias for layer in ep_layers
    )

    # Without it, "counted the same" would also pass if the recompute never ran.
    forward_calls = {"n": 0}

    def count_forward(_module, _args):
        forward_calls["n"] += 1

    # A pre-hook: the non-reentrant recompute stops at the last saved tensor and can leave without returning.
    ep_layers[0].register_forward_pre_hook(count_forward)

    batch = build_batch(ctx.device, ctx.rank)

    forward_calls["n"] = 0
    plain_counts, plain_loss = train_step(model, ep_layers, batch)
    plain_calls = forward_calls["n"]
    checks["no_gc_counts_every_routed_token"] = counts_match_expected(plain_counts)
    checks["no_gc_ran_one_forward"] = plain_calls == 1
    checks["no_gc_loss_finite"] = bool(torch.isfinite(torch.tensor(plain_loss)))
    metrics["tokens_times_topk"] = float(TOKENS * TOP_K)

    # A frozen reference/teacher pass drives no optimizer step, so its load must not skew the balance.
    checks["frozen_no_grad_pass_records_nothing"] = all(c is None for c in frozen_forward(model, ep_layers, batch))

    # Reentrant, what every non-PP EP/CP run is forced onto.
    enable_scoped_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": True})
    checks["gc_scopes_installed"] = all_checkpoint_funcs_scoped(model)
    forward_calls["n"] = 0
    mid_step: list[list[torch.Tensor | None]] = []
    reentrant_counts, reentrant_loss = train_step(model, ep_layers, batch, after_forward=mid_step.append)
    reentrant_calls = forward_calls["n"]
    # The contract is the original pass: a counter filled by the recompute is only complete once backward has run.
    checks["reentrant_gc_counts_before_backward"] = counts_match_expected(mid_step[0])
    # The body really ran twice, so a gate counting both passes would double and one counting neither would be at zero.
    checks["reentrant_gc_recomputed_the_body"] = reentrant_calls == 2
    checks["reentrant_gc_counts_every_routed_token"] = counts_match_expected(reentrant_counts)
    checks["reentrant_gc_counts_identically_to_no_gc"] = counters_equal(plain_counts, reentrant_counts)
    checks["reentrant_gc_loss_finite"] = bool(torch.isfinite(torch.tensor(reentrant_loss)))
    metrics["reentrant_counter_sum"] = (
        float(reentrant_counts[0].sum().item()) if reentrant_counts[0] is not None else -1
    )

    # Non-reentrant, the mode PP accepts. Re-enabling through the model's own method also exercises
    # `_rescope_on_reenable`: HF installs a bare checkpoint function here, and losing the scope would make the EP layer
    # raise.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    checks["gc_scopes_survive_reenable"] = all_checkpoint_funcs_scoped(model)
    forward_calls["n"] = 0
    non_reentrant_counts, non_reentrant_loss = train_step(model, ep_layers, batch)
    checks["non_reentrant_gc_recomputed_the_body"] = forward_calls["n"] == 2
    checks["non_reentrant_gc_counts_every_routed_token"] = counts_match_expected(non_reentrant_counts)
    checks["non_reentrant_gc_counts_identically_to_no_gc"] = counters_equal(plain_counts, non_reentrant_counts)
    checks["non_reentrant_gc_loss_finite"] = bool(torch.isfinite(torch.tensor(non_reentrant_loss)))

    # Eval tokens would fold the eval set's balance into the training bias. HF gates checkpointing on ``training``, so
    # this is the no-scope path even with checkpointing enabled.
    for layer in ep_layers:
        layer.expert_load_counter = None
    model.eval()
    with torch.no_grad():
        model(**batch)
    checks["eval_pass_records_nothing"] = all(layer.expert_load_counter is None for layer in ep_layers)

    # GC on + bias_update: the exported bias moves, back on the reentrant mode.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    step_counts, _ = train_step(model, ep_layers, batch)
    checks["final_step_recorded_load_under_gc"] = counts_match_expected(step_counts)

    # The callback all-reduces the counters over the world before the sign step, so the reference does too.
    global_counts = torch.stack([c.clone() for c in step_counts])
    dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
    expected_delta = GAMMA * torch.sign(global_counts.mean(dim=-1, keepdim=True) - global_counts)
    # A perfectly uniform load would make every sign 0 and the movement check vacuous.
    checks["load_is_imbalanced_enough_to_move_the_bias"] = bool((expected_delta[0] != 0).any())

    before = [layer.balancing_biases.clone() for layer in ep_layers]
    BalancingBiasCallback(update_rate=GAMMA).on_step_end(None, TrainerState(global_step=1), None, model=model)
    after = [layer.balancing_biases for layer in ep_layers]

    moved = [(a - b) for a, b in zip(after, before, strict=True)]
    checks["bias_moved_under_gradient_checkpointing"] = bool(torch.stack(moved).abs().max().item() > 0)
    checks["bias_moved_by_the_exact_sign_step"] = all(
        torch.allclose(delta, expected, atol=1e-6)
        for delta, expected in zip(moved, expected_delta.unbind(0), strict=True)
    )
    checks["counters_zeroed_after_the_update"] = all(
        float(layer.expert_load_counter.abs().sum().item()) == 0.0 for layer in ep_layers
    )
    metrics["max_bias_movement"] = float(torch.stack(moved).abs().max().item())

    # `state_dict` is what the gathered save, the weight sync and `save_pretrained` all read.
    exported = model.state_dict()
    bias_keys = [key for key in exported if key.endswith("mlp.router.bias")]
    checks["bias_is_in_the_exported_state_dict"] = len(bias_keys) == NUM_LAYERS
    checks["exported_bias_carries_the_movement"] = bool(bias_keys) and all(
        torch.equal(exported[key], layer.balancing_biases)
        for key, layer in zip(sorted(bias_keys), ep_layers, strict=False)
    )

    log(f"[rank {ctx.rank}] forward calls: no-gc={plain_calls} reentrant={reentrant_calls}")
    log(f"[rank {ctx.rank}] max |bias movement| = {metrics['max_bias_movement']:.4f} (gamma={GAMMA})")
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
