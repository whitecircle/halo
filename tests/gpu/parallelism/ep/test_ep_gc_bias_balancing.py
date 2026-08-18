#!/usr/bin/env python
"""Gradient checkpointing must not silence the bias-update balancer, on a REAL DeepEP EP layer.

``moe_balancing: bias_update`` steers routing from ``expert_load_counter``, which every EP layer
fills in its forward. Under activation checkpointing that forward runs TWICE per microbatch, so the
gate has to pick exactly ONE of the two passes — and grad mode cannot make that call. The REENTRANT
checkpoint (forced on every non-PP EP/CP run) executes its original pass inside an
``autograd.Function``, i.e. with grad DISABLED, and its recompute inside backward with grad ENABLED;
the NON-reentrant one (the mode pipeline parallelism accepts) runs both passes with grad enabled. A
gate written as ``torch.is_grad_enabled()`` therefore silently relocates the recording to the
recompute in the first case — leaving the counter incomplete until backward has run, so any step
whose backward is skipped or pruned drops the microbatch — and DOUBLE-counts in the second. Every
shipped balancing config sets ``gradient_checkpointing: true``, and nothing about either failure
surfaces as an error.

:func:`~src.distributed.expert_parallel.gc_scope.counts_toward_expert_load` is unit-pinned on stubs
in ``tests/cpu/parallelism/test_expert_load_recording.py``. What only a GPU can show is the same
matrix through the real thing: a patched ``EPGptOssMoELayer`` with a live DeepEP ``ElasticBuffer``,
HF's own checkpoint function, and the native adopted ``router.bias`` the export path serves. So this
runs one model through four modes —

  1. no checkpointing (plus a frozen ``no_grad`` reference/teacher pass on that same path),
  2. reentrant checkpointing (the EP/CP default),
  3. non-reentrant checkpointing (the mode pipeline parallelism accepts),
  4. eval.

Modes 1-3 must record EXACTLY ``tokens * top_k`` per layer and agree tensor-for-tensor; the frozen
pass and eval must record nothing. A forward-entry counter proves the recompute actually ran —
without it "counts did not double" is vacuous — and the finale drives ``RouterBiasBalancingCallback``
after a checkpointed step and asserts the bias MOVED by the exact DeepSeek-V3 sign step, in
``router.bias``: the tensor ``state_dict`` exports and vLLM/SGLang route on.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_ep_gc_bias_balancing.py
"""

import sys

import torch
import torch.distributed as dist
from transformers import GptOssConfig, GptOssForCausalLM

from src.callbacks.router_bias_balancing import RouterBiasBalancingCallback
from src.distributed.expert_parallel.balancing_strategy import apply_balancing_strategy
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.patching import (
    create_ep_buffers,
    enable_ep_gradient_checkpointing,
    patch_moe_model_for_ep,
)
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.harness import gpu_test_main, log
from tests.common.models import TINY_GPTOSS_CONFIG

EP_SIZE = 2
BATCH = 2
SEQ_LEN = 32
TOP_K = TINY_GPTOSS_CONFIG["num_experts_per_tok"]
NUM_EXPERTS = TINY_GPTOSS_CONFIG["num_local_experts"]
NUM_LAYERS = TINY_GPTOSS_CONFIG["num_hidden_layers"]
TOKENS = BATCH * SEQ_LEN
# Large enough that a bf16-rounded write would still be visible; the update is a pure sign step, so
# the magnitude only has to be exactly reproducible.
GAMMA = 0.125
SEED = 1234


def build_model(device):
    """The seeded tiny GptOss MoE. bf16 — the DeepEP dispatch dtype every EP run trains in."""
    torch.manual_seed(SEED)
    model = GptOssForCausalLM(GptOssConfig(**TINY_GPTOSS_CONFIG)).to(torch.bfloat16).to(device)
    return model


def build_batch(device, rank: int) -> dict:
    """One rank-local microbatch. EP is orthogonal to DP, so each rank routes its own tokens."""
    generator = torch.Generator(device="cpu").manual_seed(SEED + rank)
    input_ids = torch.randint(
        0, TINY_GPTOSS_CONFIG["vocab_size"], (BATCH, SEQ_LEN), generator=generator, device="cpu"
    ).to(device)
    return {"input_ids": input_ids, "labels": input_ids}


def read_counters(ep_layers) -> list[torch.Tensor | None]:
    return [None if l.expert_load_counter is None else l.expert_load_counter.clone() for l in ep_layers]


def train_step(model, ep_layers, batch, *, after_forward=None) -> tuple[list[torch.Tensor | None], float]:
    """One training forward+backward; returns each EP layer's freshly recorded counter.

    ``after_forward`` observes the counters between forward and backward — the only way to tell
    WHICH pass recorded them, since both passes have finished by the time the step is over.
    """
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
    """A reference/teacher-shaped pass: train mode, no grad, no optimizer step behind it."""
    for layer in ep_layers:
        layer.expert_load_counter = None
    model.train()
    with torch.no_grad():
        model(**batch)
    return read_counters(ep_layers)


def counts_match_expected(counters: list[torch.Tensor | None]) -> bool:
    """Every layer recorded exactly one count per (token, selected expert) of this microbatch."""
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
    patch_moe_model_for_ep(model, parallelism_config.create_ep_config())
    create_ep_buffers(model)

    ep_layers = [module for module in model.modules() if isinstance(module, EPMoELayerBase)]
    checks["ep_layers_patched"] = len(ep_layers) == NUM_LAYERS
    checks["each_rank_holds_an_expert_shard"] = bool(ep_layers) and all(
        layer.experts_per_rank == NUM_EXPERTS // EP_SIZE for layer in ep_layers
    )

    # bias_update through the real strategy seam: GptOss adopts the hub router's own `router.bias`,
    # so everything below is measured on the tensor an export actually carries.
    apply_balancing_strategy(model, "bias_update", is_moe=True)
    checks["balancing_bias_is_the_native_router_bias"] = all(
        layer.balancing_biases is layer.router.bias for layer in ep_layers
    )

    # A forward-call counter on one EP layer: without it, "the checkpointed step counted the same as
    # the unchecked one" would also pass if the recompute never ran at all.
    forward_calls = {"n": 0}

    def count_forward(_module, _args):
        forward_calls["n"] += 1

    # A PRE-hook: the non-reentrant recompute stops as soon as it has reproduced the last saved
    # tensor, so it can leave the layer without ever returning from it.
    ep_layers[0].register_forward_pre_hook(count_forward)

    batch = build_batch(ctx.device, ctx.rank)

    # ── 1. no gradient checkpointing ────────────────────────────────────────────────────────────
    forward_calls["n"] = 0
    plain_counts, plain_loss = train_step(model, ep_layers, batch)
    plain_calls = forward_calls["n"]
    checks["no_gc_counts_every_routed_token"] = counts_match_expected(plain_counts)
    checks["no_gc_ran_one_forward"] = plain_calls == 1
    checks["no_gc_loss_finite"] = bool(torch.isfinite(torch.tensor(plain_loss)))
    metrics["tokens_times_topk"] = float(TOKENS * TOP_K)

    # A frozen reference/teacher pass drives no optimizer step, so its load must not skew the
    # balance. Outside a checkpoint scope that is exactly the grad-mode gate.
    checks["frozen_no_grad_pass_records_nothing"] = all(c is None for c in frozen_forward(model, ep_layers, batch))

    # ── 2. REENTRANT checkpointing — what every non-PP EP/CP run is forced onto ──────────────────
    enable_ep_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": True})
    checks["gc_scopes_installed"] = all(
        getattr(module._gradient_checkpointing_func, "_ep_scoped", False)
        for module in model.modules()
        if getattr(module, "_gradient_checkpointing_func", None) is not None
    )
    forward_calls["n"] = 0
    mid_step: list[list[torch.Tensor | None]] = []
    reentrant_counts, reentrant_loss = train_step(model, ep_layers, batch, after_forward=mid_step.append)
    reentrant_calls = forward_calls["n"]
    # WHICH pass recorded. The reentrant checkpoint runs its original forward inside an
    # ``autograd.Function``, i.e. with grad disabled, and its recompute inside backward with grad
    # ENABLED — so a grad-mode gate silently moves the recording to the recompute. The totals still
    # look right, but the counter is then only complete once backward has run, and any step whose
    # backward is skipped or pruned loses the microbatch entirely. The contract is the original pass.
    checks["reentrant_gc_counts_before_backward"] = counts_match_expected(mid_step[0])
    # The premise for the whole file: the body really did run twice, so a gate that counted both
    # passes would have doubled and one that counted neither would be at zero.
    checks["reentrant_gc_recomputed_the_body"] = reentrant_calls == 2
    checks["reentrant_gc_counts_every_routed_token"] = counts_match_expected(reentrant_counts)
    checks["reentrant_gc_counts_identically_to_no_gc"] = counters_equal(plain_counts, reentrant_counts)
    checks["reentrant_gc_loss_finite"] = bool(torch.isfinite(torch.tensor(reentrant_loss)))
    metrics["reentrant_counter_sum"] = (
        float(reentrant_counts[0].sum().item()) if reentrant_counts[0] is not None else -1
    )

    # ── 3. NON-reentrant checkpointing — the mode pipeline parallelism accepts. Re-enabling through
    # the model's own method also exercises `_rescope_on_reenable`: HF installs a BARE checkpoint
    # function here, and losing the scope would make the EP layer raise rather than replay.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    checks["gc_scopes_survive_reenable"] = all(
        getattr(module._gradient_checkpointing_func, "_ep_scoped", False)
        for module in model.modules()
        if getattr(module, "_gradient_checkpointing_func", None) is not None
    )
    forward_calls["n"] = 0
    non_reentrant_counts, non_reentrant_loss = train_step(model, ep_layers, batch)
    checks["non_reentrant_gc_recomputed_the_body"] = forward_calls["n"] == 2
    checks["non_reentrant_gc_counts_every_routed_token"] = counts_match_expected(non_reentrant_counts)
    checks["non_reentrant_gc_counts_identically_to_no_gc"] = counters_equal(plain_counts, non_reentrant_counts)
    checks["non_reentrant_gc_loss_finite"] = bool(torch.isfinite(torch.tensor(non_reentrant_loss)))

    # ── 4. eval — an evaluation loop routes real tokens through the same layers, and counting them
    # would fold the eval set's balance into the training bias. HF also gates checkpointing on
    # ``training``, so this is the no-scope path even with checkpointing enabled.
    for layer in ep_layers:
        layer.expert_load_counter = None
    model.eval()
    with torch.no_grad():
        model(**batch)
    checks["eval_pass_records_nothing"] = all(layer.expert_load_counter is None for layer in ep_layers)

    # ── the shipped shape end to end: GC on + bias_update ⇒ the exported bias MOVES ──────────────
    # Back to the reentrant mode the EP/CP default forces, then one real step and one callback tick.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    step_counts, _ = train_step(model, ep_layers, batch)
    checks["final_step_recorded_load_under_gc"] = counts_match_expected(step_counts)

    # The callback all-reduces the counters over the world before deriving the sign step, so the
    # reference must be reduced the same way.
    global_counts = torch.stack([c.clone() for c in step_counts])
    dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
    expected_delta = GAMMA * torch.sign(global_counts.mean(dim=-1, keepdim=True) - global_counts)
    # A perfectly uniform load would make every sign 0 and the movement check vacuous.
    checks["load_is_imbalanced_enough_to_move_the_bias"] = bool((expected_delta[0] != 0).any())

    before = [layer.balancing_biases.clone() for layer in ep_layers]
    RouterBiasBalancingCallback(update_rate=GAMMA).on_step_end(None, None, None, model=model)
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

    # The moved tensor is the one an export carries: `state_dict` is what the gathered save, the
    # vLLM/SGLang weight sync and `save_pretrained` all read.
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
