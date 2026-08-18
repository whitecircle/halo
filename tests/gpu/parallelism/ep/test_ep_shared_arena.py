#!/usr/bin/env python
"""One DeepEP arena for every MoE layer: same gradients, a fraction of the memory.

The DeepEP receive arena is sized for every peer sending its whole batch, so it is linear in the
dispatch-group width and independent of layer count — one per MoE layer multiplies a 4 GiB arena by
60 on a wide-EP job. This test pins the two properties that let a single arena serve every layer:

  1. Every EP layer dispatches through the SAME ``ElasticBuffer`` object, and the arena's device
     footprint is one buffer's worth rather than one per layer.
  2. Sharing it changes nothing numerically. A later layer's dispatch must not disturb the handle an
     earlier layer's backward still holds, so loss and gradients must match a private-arena run.

Both halves are needed: (1) alone would pass if sharing silently corrupted gradients, and (2) alone
would pass if the arenas were never actually shared.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_ep_shared_arena.py
"""

import deep_ep
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from src.distributed.expert_parallel import dispatcher as dispatcher_mod
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.distributed import ensure_model_downloaded
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_30B_A3B
from tests.common.utils import log

MODEL_NAME = QWEN3_30B_A3B
EP_SIZE = 2
SEQ_LEN = 256
SEED = 42

# DeepEP packs received tokens in a run-to-run nondeterministic order, so bf16 gradients reassociate
# between any two runs — shared arena or not. The test therefore measures that noise floor from two
# same-mode runs and requires the cross-mode comparison to be no worse than it, rather than guessing a
# constant. An invalidated handle is not a rounding difference: it decorrelates the gradient outright,
# so it lands far below both the floor and the absolute guard.
NOISE_MARGIN = 5e-3
GRAD_COSINE_FLOOR = 0.99
LOSS_ABS_TOL = 0.02


def live_arenas(model):
    """Distinct ElasticBuffer objects currently backing the model's EP layers."""
    seen = {}
    for layer in ep_layers(model):
        buf = layer.dispatcher.backend.buffer
        if buf is not None:
            seen[id(buf)] = buf
    return seen


def arena_bytes(model):
    """Device bytes held by the distinct arenas, from DeepEP's own sizing.

    Read from ``get_buffer_size_hint`` rather than a ``mem_get_info`` delta: the arena is allocated
    outside the caching allocator, which recycles around it and swamps the measurement (the hint
    tracked the real device-free delta to within 5% when checked directly).
    """
    total, counted = 0, set()
    for layer in ep_layers(model):
        backend = layer.dispatcher.backend
        if backend.buffer is None or id(backend.buffer) in counted:
            continue
        counted.add(id(backend.buffer))
        total += deep_ep.ElasticBuffer.get_buffer_size_hint(
            layer.dispatcher.ep_group, backend._capacity, backend._padded_hidden, backend._num_topk
        )
    return total


def fixed_batch(tokenizer, device):
    torch.manual_seed(SEED)
    text = "Expert parallelism dispatches tokens to experts. " * 40
    tokens = tokenizer(text, return_tensors="pt", padding="max_length", max_length=SEQ_LEN, truncation=True)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    dist.broadcast(input_ids, src=0)
    dist.broadcast(attention_mask, src=0)
    dist.broadcast(labels, src=0)
    return input_ids, attention_mask, labels


def forward_backward(model, batch):
    """One fwd+bwd on a fixed batch; returns (loss, {param name -> grad})."""
    model.zero_grad(set_to_none=True)
    input_ids, attention_mask, labels = batch
    torch.manual_seed(SEED)
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    out.loss.backward()
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
    return out.loss.item(), grads


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()


def worst_cosine(left, right):
    """The least-correlated gradient between two runs, as (name, cosine)."""
    assert set(left) == set(right), "gradient key sets differ"
    worst_name, worst = None, 1.0
    for name, g in left.items():
        c = cosine(g, right[name])
        if c < worst:
            worst_name, worst = name, c
    return worst_name, worst


def run(ctx):
    checks: dict[str, bool] = {}
    device = f"cuda:{ctx.local_rank}"
    ensure_model_downloaded(MODEL_NAME, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = ParallelismConfig(ep_size=EP_SIZE)
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.train()
    batch = fixed_batch(tokenizer, device)
    num_layers = len(ep_layers(model))
    log(f"EP layers: {num_layers}")
    # Premise: one MoE layer cannot show sharing.
    checks["model_has_multiple_moe_layers"] = num_layers > 1

    # ---- shared arenas (default) -----------------------------------------
    loss_shared, grads_shared = forward_backward(model, batch)
    shared = live_arenas(model)
    shared_arena_bytes = arena_bytes(model)
    log(
        f"shared:  {len(shared)} distinct ElasticBuffer across {num_layers} layers, "
        f"loss={loss_shared:.6f}, arena={shared_arena_bytes / 2**30:.2f} GiB"
    )
    checks["one_arena_serves_every_layer"] = len(shared) == 1

    # Noise floor: a second identical run. Any spread here is EP's own nondeterminism.
    _, grads_repeat = forward_backward(model, batch)
    floor_name, floor_cos = worst_cosine(grads_shared, grads_repeat)
    log(f"noise floor from two identical shared runs: {floor_cos:.6f} ({floor_name})")

    # ---- private arenas (pre-sharing behaviour) --------------------------
    # Capacity dedup is what pins one capacity per forward; without it a later layer may grow the
    # arena mid-forward, so each layer falls back to its own. Toggling it here reproduces the
    # per-layer arena that sharing replaces.
    for layer in ep_layers(model):
        layer.dispatcher.destroy()
    dedup_was = dispatcher_mod._CAPACITY_DEDUP_ENABLED
    dispatcher_mod._CAPACITY_DEDUP_ENABLED = False
    try:
        loss_private, grads_private = forward_backward(model, batch)
        private = live_arenas(model)
        private_arena_bytes = arena_bytes(model)
    finally:
        dispatcher_mod._CAPACITY_DEDUP_ENABLED = dedup_was

    log(
        f"private: {len(private)} distinct ElasticBuffer across {num_layers} layers, "
        f"loss={loss_private:.6f}, arena={private_arena_bytes / 2**30:.2f} GiB"
    )
    checks["private_mode_builds_one_arena_per_layer"] = len(private) == num_layers
    # The arena is per-layer-count-independent, so sharing must divide it by exactly the layer count.
    checks["sharing_divides_the_arena_by_layer_count"] = private_arena_bytes == num_layers * shared_arena_bytes

    # ---- equivalence ------------------------------------------------------
    checks["loss_matches_private_arena"] = abs(loss_shared - loss_private) < LOSS_ABS_TOL
    worst_name, worst_cos = worst_cosine(grads_shared, grads_private)
    log(f"worst gradient cosine shared-vs-private {worst_cos:.6f} ({worst_name}) over {len(grads_shared)} tensors")
    checks["grads_match_private_arena"] = worst_cos >= GRAD_COSINE_FLOOR and worst_cos >= floor_cos - NOISE_MARGIN
    if not checks["grads_match_private_arena"]:
        log(
            f"gradient {worst_name} decorrelated under a shared arena: cosine {worst_cos:.6f}, against a "
            f"same-mode noise floor of {floor_cos:.6f} — a later layer's dispatch is invalidating an "
            f"earlier layer's handle"
        )

    log(
        f"\nloss Δ={abs(loss_shared - loss_private):.2e}, worst grad cosine={worst_cos:.6f} "
        f"(noise floor {floor_cos:.6f}), arena {private_arena_bytes / 2**30:.2f} -> "
        f"{shared_arena_bytes / 2**30:.2f} GiB ({num_layers}x smaller at {num_layers} layers)"
    )
    return {"checks": checks}


main = gpu_test_main(min_world_size=EP_SIZE, prefix="ep_shared_arena")(run)

if __name__ == "__main__":
    main()
