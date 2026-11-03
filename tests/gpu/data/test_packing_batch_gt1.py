#!/usr/bin/env python
"""Packing at batch > 1: only a B>1 row reaches a varlen kernel and FSDP2 backward in the flattened shape.

A wrong position_ids restart at the row seam shifts logits silently. A wider row changes the shape every rank hands its
kernel, and that failure is a hang, so the manifest timeout is the assertion.
"""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM, AutoTokenizer

from halo.data.collators.packing import DataCollatorWithPacking
from halo.hardware import is_blackwell_gpu
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log

MODEL = QWEN3_0_6B
# Unequal lengths: collation pads row 0 and the flatten must drop that padding, so the seam is real token against real
# token.
DOCS_ROW0 = ["The capital of France is Paris.", "Water boils at one hundred degrees."]
DOCS_ROW1 = [
    "Quantum mechanics describes the behaviour of subatomic particles in detail.",
    "Bananas are yellow and grow on trees in large hanging clusters near the top.",
]


def _varlen_impls() -> list[str]:
    impls = []
    if is_blackwell_gpu():
        # Unguarded: the Blackwell image always ships FA4, so swallowing an import failure would cover FA2 alone and
        # pass.
        from flash_attn.cute import flash_attn_func  # noqa: PLC0415, F401

        impls.append("flash_attention_4")
    try:
        import flash_attn  # noqa: PLC0415, F401

        impls.append("flash_attention_2")
    except ImportError:
        pass
    return impls


def _rows(tok) -> list[list[list[int]]]:
    return [[tok(d, add_special_tokens=False)["input_ids"] for d in row] for row in (DOCS_ROW0, DOCS_ROW1)]


def _feature(docs: list[list[int]]) -> dict:
    flat = [t for d in docs for t in d]
    return {"input_ids": flat, "labels": list(flat), "seq_lengths": [len(d) for d in docs]}


def _logits_for(model, collator, rows: list[list[list[int]]], device) -> torch.Tensor:
    batch = collator.torch_call([_feature(r) for r in rows])
    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        return model(input_ids=batch["input_ids"], position_ids=batch["position_ids"]).logits


def test_row_seam_does_not_change_documents(rank: int, impl: str) -> bool:
    """Row 0's logits must not depend on the content of the row after it. The flattened shape stays fixed: B=2 against
    B=1 re-tiles the kernel and reorders bf16 accumulation, so a leak would be indistinguishable there."""
    device = f"cuda:{torch.cuda.current_device()}"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation=impl).to(device)
    model.eval()
    collator = DataCollatorWithPacking(tokenizer=tok)
    rows = _rows(tok)
    row0, row1 = rows[0], rows[1]

    # Token for token as long as the original, so the pack layout and flattened width are byte-identical.
    alt = [[(t + 977) % tok.vocab_size for t in doc] for doc in row1]
    assert [len(d) for d in alt] == [len(d) for d in row1], "variant must not change the packed layout"

    try:
        with_orig = _logits_for(model, collator, [row0, row1], device)
        with_alt = _logits_for(model, collator, [row0, alt], device)
        assert with_orig.shape == with_alt.shape, "shape control failed; the comparison is not isolating content"
        n = sum(len(d) for d in row0)
        drift = (with_orig[0, :n].float() - with_alt[0, :n].float()).abs().max().item()
    finally:
        del model
        cleanup_memory()

    # Identical shape and work over row 0: a correct kernel recomputes it bit for bit.
    ok = drift == 0.0
    log(f"[{impl}] row-seam max |Δlogit| over row 0 = {drift:.6f} -> {'PASS' if ok else 'FAIL'}", rank=rank)
    if not ok:
        log(
            f"FAIL: rewriting row 1's tokens moved row 0's logits by {drift:.6f} — attention crosses "
            f"the batch>1 concatenation seam",
            rank=rank,
        )
    return ok


def test_fsdp2_backward_completes_at_b2(rank: int, world_size: int, impl: str) -> bool:
    """A kernel or collective mishandling the wider row stalls rather than erroring, so reaching the end is the
    assertion."""
    device = f"cuda:{torch.cuda.current_device()}"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation=impl).to(device)
    for layer in model.model.layers:
        fully_shard(layer)
    fully_shard(model)
    collator = DataCollatorWithPacking(tokenizer=tok)
    rows = _rows(tok)

    ok = True
    try:
        # Different row orders per rank: a shared kernel-compile or collective assumption would desync on them.
        local = rows if rank % 2 == 0 else list(reversed(rows))
        batch = collator.torch_call([_feature(r) for r in local])
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        for step in range(3):
            out = model(input_ids=batch["input_ids"], position_ids=batch["position_ids"], labels=batch["labels"])
            out.loss.backward()
            finite = all(p.grad is None or torch.isfinite(p.grad.to_local()).all() for p in model.parameters())
            if not finite:
                log(f"FAIL: [{impl}] non-finite gradient at step {step}", rank=rank)
                ok = False
            model.zero_grad(set_to_none=True)
            # Every rank runs every iteration: an early exit would strand the others in this barrier, turning a failure
            # into a hang.
            dist.barrier()
        log(f"[{impl}] fsdp2 packed-B2 loop completed on {world_size} ranks -> {'PASS' if ok else 'FAIL'}", rank=rank)
    finally:
        del model
        cleanup_memory()
    return ok


def run(ctx) -> dict:
    impls = _varlen_impls()
    if not impls:
        log("FAIL: no varlen attention kernel available; this test exists to cover them", rank=ctx.rank)
        return {"checks": {"varlen_kernel_available": False}}
    log(f"covering varlen kernels: {impls}", rank=ctx.rank)

    checks = {}
    for impl in impls:
        checks[f"row_seam[{impl}]"] = test_row_seam_does_not_change_documents(ctx.rank, impl)
        checks[f"fsdp2_b2[{impl}]"] = test_fsdp2_backward_completes_at_b2(ctx.rank, ctx.world_size, impl)
    ctx.barrier()
    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="packing_batch_gt1", partial_state=False)(run)

if __name__ == "__main__":
    main()
