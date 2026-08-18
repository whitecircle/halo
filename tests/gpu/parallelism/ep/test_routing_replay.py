#!/usr/bin/env python
"""
Routing replay (R2) correctness on a real EP=2 MoE model.

Validates the full capture → arm → replay cycle on GptOss-20B under DeepEP:

1. Capture: a no-grad forward records per-layer top-k masks of shape [B, S, L, K].
2. Natural equivalence: replaying the captured (natural) mask reproduces the unforced forward
   within the DeepEP repeat-forward noise floor (measured in-run, not assumed), and the replay
   flip-rate is ~0.
3. Forced gradients: replaying a shuffled mask still yields finite, nonzero router gradients on
   every EP layer (the RSPO failure mode — dead router grads — would fail here), flip-rate ~1.
4. Gradient checkpointing: the armed backward completes under EP's reentrant GC with finite grads
   (the recompute reads the same mask by construction).

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_routing_replay.py
"""

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.patching import enable_ep_gradient_checkpointing
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.grpo.rollout.routing_replay import build_routing_replay_injector
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.utils import cleanup_memory, log, log_all

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 2
BATCH, SEQ_LEN = 2, 128
SEED = 42


def _fixed_batch(tokenizer, device):
    torch.manual_seed(SEED)
    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "What is 42 + 58?"},
            {"role": "assistant", "content": "The sum of 42 and 58 is 100."},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    tokens = tokenizer([text] * BATCH, return_tensors="pt", padding="max_length", max_length=SEQ_LEN, truncation=True)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    dist.broadcast(input_ids, src=0)
    dist.broadcast(attention_mask, src=0)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return input_ids, attention_mask, labels


def _forward_loss(model, batch, grad=False):
    input_ids, attention_mask, labels = batch
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return out.loss


def _router_grad_norms(model):
    norms = []
    for module in model.modules():
        if isinstance(module, EPMoELayerBase):
            router = getattr(module, "router", None) or getattr(module, "gate", None)
            grads = [p.grad for p in router.parameters() if p.grad is not None]
            norms.append(sum(g.norm().item() for g in grads) if grads else None)
    return norms


def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    device = ctx.device

    log(f"\n{'#' * 70}\n  Routing Replay (R2) Correctness — EP={EP_SIZE} {MODEL_NAME}\n{'#' * 70}")

    ensure_model_downloaded(MODEL_NAME, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=ParallelismConfig(ep_size=EP_SIZE, cp_size=1, tp_size=1),
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    injector = build_routing_replay_injector(model)
    num_layers = len(injector._layers)
    num_experts = injector._layers[0].num_experts
    log(f"  EP MoE layers: {num_layers}, experts: {num_experts}")
    batch = _fixed_batch(tokenizer, device)

    # --- 1. Capture ---
    with injector.capture(BATCH, SEQ_LEN) as cap:
        _forward_loss(model, batch)
    masks = cap[0]
    assert masks.shape[:3] == (BATCH, SEQ_LEN, num_layers), f"mask shape {tuple(masks.shape)}"
    assert masks.dtype == torch.int16 and (masks >= 0).all() and (masks < num_experts).all()
    log(f"  [1] capture: mask {tuple(masks.shape)} OK")

    # --- 2. Natural equivalence (noise floor measured from two unforced forwards) ---
    loss_a = _forward_loss(model, batch).item()
    loss_b = _forward_loss(model, batch).item()
    noise = abs(loss_a - loss_b)
    injector.arm(masks)
    loss_forced = _forward_loss(model, batch).item()
    injector.disarm()
    flip = injector.flip_rate()
    tol = max(noise * 4, 5e-2)  # DeepEP repeat-forward accumulation noise, measured in-run
    eq_ok = abs(loss_forced - loss_a) <= tol and (flip is None or flip < 0.01)
    log_all(
        f"  [2] natural replay: loss unforced={loss_a:.6f} forced={loss_forced:.6f} "
        f"noise={noise:.6f} flip_rate={flip} -> {'PASS' if eq_ok else 'FAIL'}"
    )
    checks["natural_replay_equivalent"] = eq_ok

    # --- 3. Shuffled mask: router grads must stay alive ---
    model.train()
    shuffled = ((masks.long() + 1) % num_experts).to(torch.int16).to(device)
    injector.arm(shuffled)
    loss = _forward_loss(model, batch, grad=True)
    loss.backward()
    injector.disarm()
    flip = injector.flip_rate()
    loss_shuffled = loss.item()
    norms = _router_grad_norms(model)
    grads_ok = (
        torch.isfinite(loss).item()
        and all(n is not None and n > 0 and torch.isfinite(torch.tensor(n)) for n in norms)
        and flip is not None
        and flip > 0.9
    )
    log_all(
        f"  [3] shuffled replay: loss={loss_shuffled:.4f} flip_rate={flip:.3f} "
        f"router-grad norms nonzero={sum(1 for n in norms if n and n > 0)}/{len(norms)} "
        f"-> {'PASS' if grads_ok else 'FAIL'}"
    )
    checks["shuffled_router_grads_alive"] = grads_ok
    model.zero_grad(set_to_none=True)

    # --- 4. Gradient checkpointing: armed backward completes, grads finite ---
    enable_ep_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": True})
    injector.arm(masks)
    loss_gc = _forward_loss(model, batch, grad=True)
    loss_gc.backward()
    injector.disarm()
    injector.flip_rate()  # drain
    norms_gc = _router_grad_norms(model)
    gc_ok = (
        torch.isfinite(loss_gc).item()
        and abs(loss_gc.item() - loss_a) <= tol
        and all(n is not None and n > 0 for n in norms_gc)
    )
    log_all(f"  [4] GC replay: loss={loss_gc.item():.6f} vs {loss_a:.6f} -> {'PASS' if gc_ok else 'FAIL'}")
    checks["gc_replay_equivalent"] = gc_ok
    model.zero_grad(set_to_none=True)

    # --- 5. Shuffled mask UNDER GC: the backward recompute must read the same forced slice its
    # original forward used (a recompute falling back to natural routing would move the loss back
    # toward the unforced value and corrupt grads silently) ---
    injector.arm(shuffled)
    loss_gc_shuffled = _forward_loss(model, batch, grad=True)
    loss_gc_shuffled.backward()
    injector.disarm()
    flip = injector.flip_rate()
    norms_gc_shuffled = _router_grad_norms(model)
    gc_shuffled_ok = (
        torch.isfinite(loss_gc_shuffled).item()
        and abs(loss_gc_shuffled.item() - loss_shuffled) <= tol
        and all(n is not None and n > 0 for n in norms_gc_shuffled)
        and flip is not None
        and flip > 0.9
    )
    log_all(
        f"  [5] GC shuffled replay: loss={loss_gc_shuffled.item():.6f} vs non-GC {loss_shuffled:.6f} "
        f"flip_rate={flip:.3f} -> {'PASS' if gc_shuffled_ok else 'FAIL'}"
    )
    checks["gc_shuffled_replay_equivalent"] = gc_shuffled_ok

    del model
    cleanup_memory()
    return {"checks": checks}


main = gpu_test_main(exact_world_size=EP_SIZE, prefix="routing_replay")(run)

if __name__ == "__main__":
    main()
