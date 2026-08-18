#!/usr/bin/env python
"""Packing at ``per_device_train_batch_size > 1``: real kernel, real ranks.

The CPU suite covers the collator's B>1 flattening thoroughly (shape, loss equivalence, completion
masking, pad tails). This covers what happens AFTER it: ``flatten_packed_batch`` concatenates the
mini-batch into one ``[1, B*max_length]`` row, and a single-row collate — what every other GPU
packing test does — makes that reshape a no-op, so only a B>1 row reaches a varlen attention kernel
and an FSDP2 backward across ranks in that shape.

Two things only a GPU run can check:

1. **The row seam.** Concatenating two packed rows creates a boundary the B=1 path never builds —
   row 0's last document butting directly against row 1's first (the flatten drops inter-row
   padding). Isolation there rests on row 1's position_ids restarting at the seam. If that is
   wrong, a document's logits shift depending on what preceded it in the batch, silently, with the
   loss still looking plausible.
2. **The collective path.** A packed B>1 step changes the attention shape every rank hands its
   kernel. The failure mode is not a wrong number but a hang, so the assertion is that a full
   forward/backward/step loop completes and produces finite gradients — the manifest timeout is
   what fails when it does not.

Both varlen kernels available on the host are covered, because the production default differs by
architecture (FA4 on Blackwell, FA2 elsewhere) and the shape handling is per-kernel.

Usage::

    torchrun --nproc_per_node=2 tests/gpu/data/test_packing_batch_gt1.py
"""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.collators.packing import DataCollatorWithPacking
from src.hardware import is_blackwell_gpu
from tests.common.harness import gpu_test_main
from tests.common.utils import cleanup_memory, log

MODEL = "Qwen/Qwen3-0.6B"
# Rows of UNEQUAL length: collation pads row 0 and the flatten must drop that padding, so the seam
# is row 0's last real token against row 1's first. Equal-length rows would sidestep that path.
DOCS_ROW0 = ["The capital of France is Paris.", "Water boils at one hundred degrees."]
DOCS_ROW1 = [
    "Quantum mechanics describes the behaviour of subatomic particles in detail.",
    "Bananas are yellow and grow on trees in large hanging clusters near the top.",
]


def _varlen_impls() -> list[str]:
    """The varlen kernels this host can actually dispatch, production default first."""
    impls = []
    if is_blackwell_gpu():
        try:
            from flash_attn.cute import flash_attn_func  # noqa: F401

            impls.append("flash_attention_4")
        except Exception:
            pass
    try:
        import flash_attn  # noqa: F401

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
    """Row 0's logits must not depend on the CONTENT of the row concatenated after it.

    The comparison holds the flattened shape fixed and swaps only row 1's tokens. Comparing B=2
    against B=1 instead would prove nothing: the shorter row re-tiles the kernel and reorders bf16
    accumulation, so a difference there is expected and a leak would be indistinguishable from it.
    With the shape pinned, every remaining difference over row 0 is attention reaching across the
    seam — the one thing B>1 packing adds that the single-row path never builds.
    """
    device = f"cuda:{torch.cuda.current_device()}"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation=impl).to(device)
    model.eval()
    collator = DataCollatorWithPacking(tokenizer=tok)
    rows = _rows(tok)
    row0, row1 = rows[0], rows[1]

    # A same-length, different-content row 1: token-for-token as long as the original (so the pack
    # layout and the flattened width are byte-identical), built from other vocabulary.
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

    # Identical shape and identical work over row 0: a correct kernel recomputes it bit-for-bit.
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
    """A packed B=2 forward/backward/step must complete on every rank with finite grads.

    Regression guard for the shape the trainer actually runs: packing raises tokens per row, and a
    kernel or collective that mishandles the wider row stalls rather than erroring. Reaching the end
    of this loop at all is the assertion the timeout enforces; finite grads make it more than a
    liveness check.
    """
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
        # Ranks hold DIFFERENT row orders: divergent per-rank document layouts are what a shared
        # kernel-compile or collective assumption would desync on, and identical batches would hide it.
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
            # Every rank runs every iteration: an early exit on one rank would strand the others in
            # this barrier, turning a reported failure into the hang this test exists to detect.
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
