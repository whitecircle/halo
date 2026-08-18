#!/usr/bin/env python
"""Behavioral test: DataCollatorWithPacking must isolate documents under Flash Attention.

The collator packs several documents into one row, restarts ``position_ids`` at every document
boundary and drops the dense all-ones ``attention_mask`` so Flash Attention derives a block-diagonal
``cu_seqlens`` from those positions. If that holds, a document's logits do not depend on the content
of the documents packed BEFORE it.

Direction is the whole test. Measuring the FIRST document while the second changes proves nothing —
causality already forbids that dependence, so the measurement reads ~0 whether isolation works or
not. The leak is the later document attending BACK into the earlier one, so the perturbed document
must precede the measured one.

The control re-runs the identical tokens with un-isolated inputs (contiguous ``position_ids`` and a
dense mask) and requires the drift to be large: without it, "no drift" would be equally consistent
with a model that ignores its context entirely.

Single GPU::

    CUDA_VISIBLE_DEVICES=0 python tests/gpu/data/test_packing_isolation.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.collators.packing import DataCollatorWithPacking

MODEL = "Qwen/Qwen3-0.6B"
# The isolated path recomputes doc B over exactly the same keys/values, so bf16 accumulation is
# reproduced bit-for-bit; the control moves the logits by O(1).
LEAK_THRESHOLD = 1e-2


def _packed_batch(collator, doc_a, doc_b):
    feats = [{"input_ids": doc_a + doc_b, "labels": doc_a + doc_b, "seq_lengths": [len(doc_a), len(doc_b)]}]
    return collator.torch_call(feats)


def _doc_b_logits(model, batch, device, la, lb, *, isolated: bool):
    """Doc B's logits, either from the collator's isolating inputs or from un-isolated dense ones."""
    input_ids = batch["input_ids"].to(device)
    if isolated:
        kwargs = {"position_ids": batch["position_ids"].to(device)}
    else:
        # Withhold the isolation: one continuous position ramp across the boundary plus the dense
        # all-ones mask — what the kernel sees when the collator emits no document boundaries.
        kwargs = {
            "position_ids": torch.arange(input_ids.shape[1], device=device).unsqueeze(0),
            "attention_mask": torch.ones_like(input_ids),
        }
    with torch.no_grad():
        return model(input_ids=input_ids, **kwargs).logits[0, la : la + lb].float()


def main() -> None:
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device)
    model.eval()

    collator = DataCollatorWithPacking(tokenizer=tok)

    # Doc A is PERTURBED and comes first; doc B is MEASURED and follows it.
    doc_a = tok("The capital of France is Paris.", add_special_tokens=False)["input_ids"]
    doc_b = tok("Quantum mechanics describes subatomic particles.", add_special_tokens=False)["input_ids"]
    # Same length, different vocabulary: the packed layout and every slice offset stay byte-identical,
    # so the only thing that changes between the two runs is doc A's content.
    doc_a_variant = [(t + 977) % tok.vocab_size for t in doc_a]
    la, lb = len(doc_a), len(doc_b)

    batch1 = _packed_batch(collator, doc_a, doc_b)
    batch2 = _packed_batch(collator, doc_a_variant, doc_b)

    assert "attention_mask" not in batch1, "fix: packing collator must drop the dense attention_mask"
    assert batch1["position_ids"][0, la].item() == 0, "position_ids must restart at the document boundary"
    assert torch.equal(batch1["input_ids"][0, la:], batch2["input_ids"][0, la:]), "doc B must be identical"
    assert not torch.equal(batch1["input_ids"][0, :la], batch2["input_ids"][0, :la]), "doc A must differ"

    isolated1 = _doc_b_logits(model, batch1, device, la, lb, isolated=True)
    isolated2 = _doc_b_logits(model, batch2, device, la, lb, isolated=True)
    max_diff = (isolated1 - isolated2).abs().max().item()
    print(f"[isolation] max|Δlogits| on doc B when the PRECEDING doc A changes = {max_diff:.6f}")
    assert max_diff == 0.0, f"Documents not isolated: doc B logits moved by {max_diff} when doc A changed"

    # Control: same tokens, isolation withheld. Doc B now attends back into doc A, so the drift the
    # assertion above requires to be zero must here be large — otherwise that zero is vacuous.
    leaky1 = _doc_b_logits(model, batch1, device, la, lb, isolated=False)
    leaky2 = _doc_b_logits(model, batch2, device, la, lb, isolated=False)
    ctl_diff = (leaky1 - leaky2).abs().max().item()
    print(f"[control]   max|Δlogits| on doc B with a dense mask (leak expected) = {ctl_diff:.6f}")
    assert ctl_diff > LEAK_THRESHOLD, (
        f"Control failed: doc B's logits moved by only {ctl_diff} when doc A changed under a dense "
        f"all-ones mask — the perturbation is not measurable, so the isolation assertion proves nothing"
    )

    print("PACKING ISOLATION TEST PASSED")


if __name__ == "__main__":
    main()
