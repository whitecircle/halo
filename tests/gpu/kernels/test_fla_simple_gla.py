#!/usr/bin/env python
"""FLA's Lightning Attention-2 ops, the two modes of Ring's linear attention, are exact and repeatable on the real Triton
kernels against the naive recurrence per document: ``chunk`` on one document as Ring calls it and on a packed row, and
``fused_recurrent``, which Ring takes at or under 64 tokens.

Nothing else holds these ops to a reference. The state crossing documents with ``cu_seqlens`` dropped must miss the
reference far past the bar, or the packed comparison shows nothing about the boundaries.
"""

import os

import torch
from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla
from fla.ops.simple_gla.naive import naive_recurrent_simple_gla

from tests.common.harness import gpu_test_main, record_check
from tests.common.numerics import cos_sim, fro_rel_err, max_abs_rel_err
from tests.common.rank_log import log

SEED = 1234
# Ring-mini-linear-2.0's linear attention after the key/value repeat, and its slope ratio at this head count.
HEADS, HEAD_DIM = 16, 128
SLOPE_RATIO = 2**-0.5
# One packed row of four documents, the layout the padding-free collators emit; the same tokens as one document.
CU_SEQLENS = (0, 896, 2048, 3584, 4096)
ONE_DOCUMENT = (0, 4096)
# The longest row Ring runs on ``fused_recurrent``.
SHORT_ROW = (0, 64)
REPEATS = 20
# bf16 inputs against the fp32 recurrence, as the GatedDeltaNet op's own row bars it.
TOL_O_MAXREL = 2e-2
TOL_GRAD_COS = 0.9999
# A cosine is blind to a gradient's scale.
TOL_GRAD_FRO = 2e-2
# The other documents' state, when it crosses, moves the output by many times the bar.
CROSSING_O_MAXREL_MIN = 1e-1
OP_INPUTS = ("q", "k", "v", "g")
GRADS = tuple(f"d{name}" for name in OP_INPUTS)


def _inputs(device) -> dict[str, torch.Tensor]:
    """``g`` is the log decay of the state, one value per token and head: Ring's slope per head."""
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    total = CU_SEQLENS[-1]
    slopes = -(SLOPE_RATIO ** torch.arange(1.0, HEADS + 1))
    leaves = {
        "q": torch.randn(1, total, HEADS, HEAD_DIM, generator=gen).to(torch.bfloat16),
        "k": torch.randn(1, total, HEADS, HEAD_DIM, generator=gen).to(torch.bfloat16),
        "v": torch.randn(1, total, HEADS, HEAD_DIM, generator=gen).to(torch.bfloat16),
        "g": slopes[None, None].expand(1, total, HEADS).to(torch.bfloat16),
        "do": torch.randn(1, total, HEADS, HEAD_DIM, generator=gen),
    }
    return {name: tensor.to(device) for name, tensor in leaves.items()}


def _fresh(leaves, dtype=None) -> dict[str, torch.Tensor]:
    """Independent grad-carrying leaves, so no call reads or accumulates into another's tensors."""
    return {name: leaves[name].detach().to(dtype or leaves[name].dtype).clone().requires_grad_() for name in OP_INPUTS}


def _run(leaves, cu_seqlens: tuple[int, ...] | None, op=chunk_simple_gla) -> dict[str, torch.Tensor]:
    inputs = _fresh(leaves)
    device = leaves["q"].device
    o, _ = op(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        g=inputs["g"],
        cu_seqlens=None if cu_seqlens is None else torch.tensor(cu_seqlens, device=device, dtype=torch.long),
    )
    (o.float() * leaves["do"]).sum().backward()
    return {"o": o.detach(), **{f"d{name}": inputs[name].grad for name in OP_INPUTS}}


def _reference(leaves, cu_seqlens: tuple[int, ...]) -> dict[str, torch.Tensor]:
    """FLA's naive recurrence in fp32, one document at a time from a zero state."""
    inputs = _fresh(leaves, torch.float32)
    outputs = [
        naive_recurrent_simple_gla(*(inputs[name][:, s:e] for name in OP_INPUTS), output_final_state=False)[0]
        for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True)
    ]
    o = torch.cat(outputs, dim=1)
    (o * leaves["do"]).sum().backward()
    return {"o": o.detach(), **{f"d{name}": inputs[name].grad for name in OP_INPUTS}}


def _check_exact_and_repeatable(leaves, reference, cu_seqlens: tuple[int, ...] | None, op=chunk_simple_gla) -> None:
    first = _run(leaves, cu_seqlens, op)
    stable, worst_rel, worst_cos, worst_fro = True, 0.0, 1.0, 0.0
    for repeat in range(REPEATS):
        result = first if repeat == 0 else _run(leaves, cu_seqlens, op)
        stable &= all(torch.equal(result[name], first[name]) for name in first)
        worst_rel = max(worst_rel, max_abs_rel_err(result["o"], reference["o"]))
        worst_cos = min(worst_cos, *(cos_sim(result[name], reference[name], label=name) for name in GRADS))
        worst_fro = max(worst_fro, *(fro_rel_err(result[name], reference[name]) for name in GRADS))
    log(
        f"  stable={stable} over {REPEATS} calls, o max-rel={worst_rel:.2e}, worst grad cos={worst_cos:.6f}, "
        f"worst grad fro={worst_fro:.2e}"
    )
    assert stable, f"the op is not repeatable over {REPEATS} calls"
    assert worst_rel < TOL_O_MAXREL, f"o max-rel {worst_rel:.2e} exceeds {TOL_O_MAXREL:.0e}"
    assert worst_cos > TOL_GRAD_COS, f"worst gradient cosine {worst_cos:.6f} under {TOL_GRAD_COS}"
    assert worst_fro < TOL_GRAD_FRO, f"worst gradient error {worst_fro:.2e} exceeds {TOL_GRAD_FRO:.0e}"


def _check_dropped_boundaries_miss(leaves, reference) -> None:
    crossing = max_abs_rel_err(_run(leaves, None)["o"], reference["o"])
    log(f"  no cu_seqlens against the per-document reference: o max-rel={crossing:.2e}")
    assert crossing > CROSSING_O_MAXREL_MIN, (
        f"the state crossing documents moves o by {crossing:.2e} only, so the packed comparison shows nothing"
    )


@gpu_test_main(exact_world_size=1, partial_state=False, prefix="fla_simple_gla")
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    # An empty cache, as on a node's first run. Triton reads it at first launch.
    os.environ["TRITON_CACHE_DIR"] = os.path.join(ctx.cache_dir, "triton")
    torch.cuda.set_device(ctx.device)
    leaves = _inputs(ctx.device)
    packed_reference = _reference(leaves, CU_SEQLENS)
    log("\n--- packed row ---")
    record_check(
        checks,
        "packed_exact_and_repeatable",
        lambda: _check_exact_and_repeatable(leaves, packed_reference, CU_SEQLENS),
    )
    log("\n--- one document, no cu_seqlens as Ring calls it ---")
    record_check(
        checks,
        "one_document_exact_and_repeatable",
        lambda: _check_exact_and_repeatable(leaves, _reference(leaves, ONE_DOCUMENT), None),
    )
    log("\n--- fused_recurrent on a short row ---")
    short = {name: tensor[:, : SHORT_ROW[-1]] for name, tensor in leaves.items()}
    record_check(
        checks,
        "fused_recurrent_exact_and_repeatable",
        lambda: _check_exact_and_repeatable(short, _reference(short, SHORT_ROW), None, fused_recurrent_simple_gla),
    )
    log("\n--- boundaries dropped ---")
    record_check(checks, "dropped_boundaries_miss", lambda: _check_dropped_boundaries_miss(leaves, packed_reference))
    return {"checks": checks}


if __name__ == "__main__":
    run()
