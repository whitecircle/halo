#!/usr/bin/env python
"""DataCollatorWithPacking isolates documents under Flash Attention: a document's logits don't depend on those packed
before it.

The perturbed document must precede the measured one, since causality already shields an earlier document. The control
runs un-isolated inputs and requires large drift, or "no drift" also fits a model ignoring its context.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from halo.data.collators.packing import DataCollatorWithPacking
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log

MODEL = QWEN3_0_6B
# The isolated path recomputes doc B over the same keys and values, so bf16 accumulation repeats bit for bit; the
# control moves logits by O(1).
LEAK_THRESHOLD = 1e-2


def _packed_batch(collator, doc_a, doc_b):
    feats = [{"input_ids": doc_a + doc_b, "labels": doc_a + doc_b, "seq_lengths": [len(doc_a), len(doc_b)]}]
    return collator.torch_call(feats)


def _doc_b_logits(model, batch, device, la, lb, *, isolated: bool):
    input_ids = batch["input_ids"].to(device)
    if isolated:
        kwargs = {"position_ids": batch["position_ids"].to(device)}
    else:
        # One continuous position ramp plus the dense mask: what the kernel sees with no document boundaries.
        kwargs = {
            "position_ids": torch.arange(input_ids.shape[1], device=device).unsqueeze(0),
            "attention_mask": torch.ones_like(input_ids),
        }
    with torch.no_grad():
        return model(input_ids=input_ids, **kwargs).logits[0, la : la + lb].float()


@gpu_test_main(exact_world_size=1, prefix="packing_isolation", partial_state=False)
def run(ctx) -> dict:
    device = ctx.device
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device)
    model.eval()

    collator = DataCollatorWithPacking(tokenizer=tok)

    doc_a = tok("The capital of France is Paris.", add_special_tokens=False)["input_ids"]
    doc_b = tok("Quantum mechanics describes subatomic particles.", add_special_tokens=False)["input_ids"]
    # Same length, so the packed layout and every slice offset stay byte-identical.
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
    log(f"[isolation] max|Δlogits| on doc B when the PRECEDING doc A changes = {max_diff:.6f}")

    # Same tokens, isolation withheld: doc B attends back into doc A, so the drift must be large.
    leaky1 = _doc_b_logits(model, batch1, device, la, lb, isolated=False)
    leaky2 = _doc_b_logits(model, batch2, device, la, lb, isolated=False)
    ctl_diff = (leaky1 - leaky2).abs().max().item()
    log(f"[control]   max|Δlogits| on doc B with a dense mask (leak expected) = {ctl_diff:.6f}")

    return {
        "checks": {
            "doc_b_unmoved_by_preceding_doc": max_diff == 0.0,
            "dense_mask_control_leaks": ctl_diff > LEAK_THRESHOLD,
        }
    }


if __name__ == "__main__":
    run()
