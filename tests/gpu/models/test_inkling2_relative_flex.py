#!/usr/bin/env python
"""Inkling 2 on ``inkling2_relative_flex``: the tiny model's logits and parameter gradients match the eager path on a
batch whose rows outrun the sliding window and the relative extent and one of which is left-padded; a packed row of
three documents scores each document as that document alone, in its logits and its gradients; and one Inkling-2-Small
attention call at 8,192 tokens runs forward and backward faster and in less memory than eager."""

import time

import torch
import transformers.models.inkling2.modeling_inkling2 as inkling2_modeling

from halo.models.attention.inkling2_relative_flex import (
    INKLING2_RELATIVE_FLEX,
    inkling2_relative_flex_attention,
    register_inkling2_relative_flex,
)
from halo.models.families.inkling2 import apply_inkling2_patches
from tests.common.harness import gpu_test_main, record_check
from tests.common.rank_log import log
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

DEV = "cuda"
# Past the tiny config's 16-token window and relative extent.
LENGTH = 48
DOCUMENTS = (20, 33, 11)
TOLERANCE = 2e-4


def _model():
    torch.manual_seed(0)
    return tiny_family_model(TINY_MOE_FAMILIES["inkling2_text"]).to(DEV)


def _use(model, attn_implementation: str) -> None:
    model.config._attn_implementation = attn_implementation


def _forward_backward(model, input_ids, **kwargs):
    model.zero_grad(set_to_none=True)
    logits = model(input_ids=input_ids, use_cache=False, **kwargs).logits
    mask = kwargs.get("attention_mask")
    weights = torch.ones_like(logits[..., 0]) if mask is None else mask.to(logits.dtype)
    (logits.float().square().mean(-1) * weights).sum().backward()
    grads = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
    return logits.detach(), grads


def _max_gap(a: dict, b: dict) -> tuple[str, float]:
    assert a.keys() == b.keys(), sorted(a.keys() ^ b.keys())
    gaps = {name: ((a[name] - b[name]).abs().max() / b[name].abs().max().clamp(min=1e-6)).item() for name in a}
    worst = max(gaps, key=gaps.get)
    return worst, gaps[worst]


def check_matches_eager() -> None:
    model = _model()
    generator = torch.Generator(device=DEV).manual_seed(1)
    input_ids = torch.randint(3, model.config.vocab_size, (2, LENGTH), device=DEV, generator=generator)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, :7] = 0
    _use(model, "eager")
    eager_logits, eager_grads = _forward_backward(model, input_ids, attention_mask=attention_mask)
    _use(model, INKLING2_RELATIVE_FLEX)
    flex_logits, flex_grads = _forward_backward(model, input_ids, attention_mask=attention_mask)
    real = attention_mask.bool()
    gap = ((flex_logits - eager_logits)[real].abs().max() / eager_logits[real].abs().max()).item()
    name, grad_gap = _max_gap(flex_grads, eager_grads)
    log(f"  logits gap {gap:.2e}, worst gradient gap {grad_gap:.2e} ({name})")
    assert gap < TOLERANCE, f"logits differ from eager by {gap:.2e}"
    assert grad_gap < TOLERANCE, f"{name} gradient differs from eager by {grad_gap:.2e}"


def check_packed_documents_stay_apart() -> None:
    model = _model()
    _use(model, INKLING2_RELATIVE_FLEX)
    generator = torch.Generator(device=DEV).manual_seed(2)
    documents = [torch.randint(3, model.config.vocab_size, (1, n), device=DEV, generator=generator) for n in DOCUMENTS]
    position_ids = torch.cat([torch.arange(n, device=DEV) for n in DOCUMENTS]).unsqueeze(0)
    packed_logits, packed_grads = _forward_backward(model, torch.cat(documents, dim=1), position_ids=position_ids)

    alone_logits, alone_grads = [], {}
    for document in documents:
        logits, grads = _forward_backward(model, document)
        alone_logits.append(logits)
        for name, grad in grads.items():
            alone_grads[name] = alone_grads.get(name, 0) + grad
    alone = torch.cat(alone_logits, dim=1)
    gap = ((packed_logits - alone).abs().max() / alone.abs().max()).item()
    name, grad_gap = _max_gap(packed_grads, alone_grads)
    log(f"  packed vs alone: logits gap {gap:.2e}, worst gradient gap {grad_gap:.2e} ({name})")
    assert gap < TOLERANCE, f"a packed document's logits differ from the document alone by {gap:.2e}"
    assert grad_gap < TOLERANCE, (
        f"{name} gradient of the packed row differs from the documents alone by {grad_gap:.2e}"
    )


class _Layer(torch.nn.Module):
    num_key_value_groups = 6
    training = True


def _time_and_peak(fn) -> tuple[float, float]:
    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start = time.perf_counter()
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / 3 * 1e3, (torch.cuda.max_memory_allocated() - base) / 2**30


def check_speed_and_memory() -> None:
    """One full-attention call at Inkling-2-Small's shape: 48 query and 8 KV heads of 128, relative extent 1024, a
    sink per query; bf16, forward and backward."""
    heads, kv_heads, head_dim, extent, length = 48, 8, 128, 1024, 8192
    generator = torch.Generator(device=DEV).manual_seed(3)

    def tensor(*shape):
        return torch.randn(*shape, device=DEV, dtype=torch.bfloat16, generator=generator).requires_grad_()

    q, k, v = (
        tensor(1, heads, length, head_dim),
        tensor(1, kv_heads, length, head_dim),
        tensor(1, kv_heads, length, head_dim),
    )
    rel = tensor(1, heads, length, extent)
    sinks = tensor(1, heads, length, 1)
    positions = torch.arange(length, device=DEV)
    causal = torch.full((length, length), float("-inf"), device=DEV, dtype=torch.bfloat16).triu(1)[None, None]
    layer = _Layer()

    def eager():
        distance = positions[:, None] - positions[None, :]
        index = distance.clamp(0, extent - 1)[None, None].expand(1, heads, -1, -1)
        bias = rel.gather(-1, index).masked_fill((distance < 0) | (distance >= extent), 0.0)
        out, _ = inkling2_modeling.eager_attention_forward(
            layer, q, k, v, causal, scaling=head_dim**-0.5, position_bias=bias, sinks=sinks
        )
        out.float().square().mean().backward()

    def flex():
        out, _ = inkling2_relative_flex_attention(
            layer, q, k, v, None, scaling=head_dim**-0.5, position_bias=rel, sinks=sinks
        )
        out.float().square().mean().backward()

    times = {}
    for name, fn in (("eager", eager), ("flex", flex)):
        times[name] = _time_and_peak(fn)
        q.grad = k.grad = v.grad = rel.grad = sinks.grad = None
    log(
        f"  8,192 tokens fwd+bwd: eager {times['eager'][0]:.1f} ms / {times['eager'][1]:.1f} GiB peak, "
        f"flex {times['flex'][0]:.1f} ms / {times['flex'][1]:.1f} GiB peak"
    )
    assert times["flex"][0] < times["eager"][0], "flex is slower than eager"
    assert times["flex"][1] < times["eager"][1], "flex peaks above eager"


@gpu_test_main(exact_world_size=1, prefix="inkling2_relative_flex", partial_state=False)
def run(ctx) -> dict:
    apply_inkling2_patches()
    register_inkling2_relative_flex()
    checks: dict[str, bool] = {}
    record_check(checks, "matches_eager", check_matches_eager)
    record_check(checks, "packed_documents_stay_apart", check_packed_documents_stay_apart)
    record_check(checks, "speed_and_memory", check_speed_and_memory)
    return {"checks": checks}


if __name__ == "__main__":
    run()
