#!/usr/bin/env python
"""Packed rows on the real kernels for families whose flash path or isolation is a toolkit patch.

Doc A is perturbed and precedes the measured doc B, since causality alone shields an earlier document. The control
withholds the isolation and must move doc B far more, or the zero is vacuous. Ring's Lightning Attention-2 crosses
documents by design.
"""

import argparse
from dataclasses import dataclass, field

import torch

from halo.data.collators.packing import DataCollatorWithPacking
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.attention.selection import is_flash_attn_implementation
from halo.models.segment_markers import segment_markers_for
from tests.common.harness import gpu_test_main
from tests.common.rank_log import log
from tests.common.tiny_models import (
    TINY_BAILING_RING_FAMILY,
    TINY_MOE_FAMILIES,
    TinyFamily,
    shared_tiny_family_checkpoint,
    word_level_tokenizer,
)

SEED = 0
DOC_A, DOC_B = 48, 40
INIT_STD = 0.1
# Fraction of doc B's largest |logit|. Isolated, only bf16 reduction order differs; a leak, a dropped causal flag or a
# wrong scale moves the logits by O(1) of their scale.
ISOLATED_REL_MAX = 2e-2
LEAK_REL_MIN = 1e-1
# On a dense label the packed row (masked SDPA) and lone document (causal) take different kernels, whose bf16 rounding
# moves doc B by ~1e-1, onto the leak's bar; at fp32 they agree to ~1e-6.
FP32_ISOLATED_REL_MAX = 1e-4
# Packed and lone doc B run different kernels, and GatedDeltaNet Triton kernels take dots at TF32 even on fp32 inputs,
# so they part at 1.06e-4 of scale while changing doc A moves doc B by exactly 0 and the control by ~1.5.
QWEN4_EXP_FP32_ISOLATED_REL_MAX = 5e-4


@dataclass(frozen=True)
class _FamilyRow:
    family: TinyFamily
    overrides: dict = field(default_factory=dict)
    isolates: bool = True
    fp32_isolated_rel_max: float = FP32_ISOLATED_REL_MAX


FAMILIES = {
    "bailing_moe": _FamilyRow(TINY_MOE_FAMILIES["bailing_moe"]),
    "bailing_moe_linear": _FamilyRow(TINY_BAILING_RING_FAMILY, isolates=False),
    # Zaya's tiny head_dim 16 is below what the FA3/FA4 kernels take.
    "zaya": _FamilyRow(TINY_MOE_FAMILIES["zaya"], {"head_dim": 64}),
    # Its class declares no flash kernel: SDPA under the tiled QSA.
    "qwen4_exp_text": _FamilyRow(
        TINY_MOE_FAMILIES["qwen4_exp_text"], fp32_isolated_rel_max=QWEN4_EXP_FP32_ISOLATED_REL_MAX
    ),
}


def _randomize(model) -> None:
    """Widen the random init so a leak reads well above bf16 noise."""
    for param in model.parameters():
        if param.dim() >= 2:
            torch.nn.init.normal_(param, std=INIT_STD)


def _load(ckpt: str, attn_implementation: str | None, dtype: torch.dtype = torch.bfloat16):
    model, _ = load_distributed_model(
        model_name_or_path=ckpt,
        parallelism_config=ParallelismConfig(max_concurrent_loading=0),
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        use_liger_kernel=False,
    )
    return model.eval()


def _row_inputs(batch, device, keys) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items() if key in keys}


def _doc_b_logits(model, batch, device, *, isolated: bool) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    if isolated:
        kwargs = _row_inputs(batch, device, set(batch) - {"input_ids", "labels"})
    else:
        kwargs = {
            "position_ids": torch.arange(input_ids.shape[1], device=device)[None],
            "attention_mask": torch.ones_like(input_ids),
        }
    with torch.no_grad():
        return model(input_ids=input_ids, use_cache=False, **kwargs).logits[0, DOC_A:].float()


def _one_document_logits(model, doc, device) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=torch.tensor([doc], device=device), use_cache=False).logits[0].float()


def _fp32_doc_b_versus_alone(ckpt: str, label: str, batch, doc_b, device) -> tuple[torch.Tensor, torch.Tensor]:
    model = _load(ckpt, label, dtype=torch.float32)
    return _doc_b_logits(model, batch, device, isolated=True), _one_document_logits(model, doc_b, device)


def _packed_step_is_finite(model, batch, device) -> bool:
    """A finite loss and finite gradients; an expert no token routed to has none."""
    model.train()
    model.zero_grad(set_to_none=True)
    loss = model(**_row_inputs(batch, device, set(batch)), use_cache=False).loss
    loss.backward()
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    model.eval()
    return bool(torch.isfinite(loss)) and bool(grads) and all(bool(torch.isfinite(grad).all()) for grad in grads)


@gpu_test_main(exact_world_size=1, prefix="family_packing_isolation", partial_state=False)
def run(ctx) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=sorted(FAMILIES), required=True)
    p.add_argument("--attn", default=None, help="attn_implementation; unset = the loader's auto selection")
    args = p.parse_args()
    row = FAMILIES[args.family]

    tokenizer = word_level_tokenizer([f"w{i}" for i in range(500)])
    ckpt = shared_tiny_family_checkpoint(
        ctx,
        row.family,
        f"{args.family}_packing_isolation",
        tokenizer,
        SEED,
        edit=_randomize,
        overrides=row.overrides,
    )
    model = _load(ckpt, args.attn)
    label = model.config._attn_implementation
    log(f"=== {args.family}: requested {args.attn!r}, resolved {label!r}")

    g = torch.Generator().manual_seed(1)
    doc_a, doc_a_variant, doc_b = (
        torch.randint(2, len(tokenizer), (n,), generator=g).tolist() for n in (DOC_A, DOC_A, DOC_B)
    )
    markers = segment_markers_for(model.config)
    collator = DataCollatorWithPacking(
        tokenizer=tokenizer, return_seq_idx=markers.seq_idx, emit_varlen_always=markers.cu_seq_lens
    )
    batches = [
        collator.torch_call([{"input_ids": a + doc_b, "labels": a + doc_b, "seq_lengths": [DOC_A, DOC_B]}])
        for a in (doc_a, doc_a_variant)
    ]

    alone = _one_document_logits(model, doc_b, ctx.device)
    scale = alone.abs().max().item()
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    if args.attn is None or is_flash_attn_implementation(args.attn):
        checks["runs_a_flash_label"] = is_flash_attn_implementation(label)
    if is_flash_attn_implementation(label):
        sdpa_alone = _one_document_logits(_load(ckpt, "sdpa"), doc_b, ctx.device)
        metrics["flash_vs_sdpa_rel"] = (alone - sdpa_alone).abs().max().item() / scale
        log(f"  one document, {label} vs sdpa: {metrics['flash_vs_sdpa_rel']:.3e} (of max |logit| {scale:.3f})")
        checks["flash_matches_sdpa_on_one_document"] = metrics["flash_vs_sdpa_rel"] <= ISOLATED_REL_MAX

    isolated = [_doc_b_logits(model, batch, ctx.device, isolated=True) for batch in batches]
    metrics["drift_rel"] = (isolated[0] - isolated[1]).abs().max().item() / scale
    log(f"  doc B drift when doc A changes: {metrics['drift_rel']:.3e}")
    if row.isolates:
        leaky = [_doc_b_logits(model, batch, ctx.device, isolated=False) for batch in batches]
        if is_flash_attn_implementation(label):
            packed_b, alone_b, versus_alone_max, scored_on = isolated[0], alone, ISOLATED_REL_MAX, "bf16"
        else:
            packed_b, alone_b = _fp32_doc_b_versus_alone(ckpt, label, batches[0], doc_b, ctx.device)
            versus_alone_max, scored_on = row.fp32_isolated_rel_max, "fp32"
        metrics["versus_alone_rel"] = (packed_b - alone_b).abs().max().item() / alone_b.abs().max().item()
        metrics["leak_rel"] = (leaky[0] - leaky[1]).abs().max().item() / scale
        log(
            f"  control drift {metrics['leak_rel']:.3e}; packed doc B vs alone {metrics['versus_alone_rel']:.3e} "
            f"({scored_on}, bar {versus_alone_max:.0e})"
        )
        checks["packed_row_leaves_doc_b_unmoved"] = metrics["drift_rel"] <= ISOLATED_REL_MAX
        checks["packed_doc_b_matches_doc_b_alone"] = metrics["versus_alone_rel"] <= versus_alone_max
        checks["control_without_isolation_leaks"] = metrics["leak_rel"] >= LEAK_REL_MIN

    checks["packed_backward_is_finite"] = _packed_step_is_finite(model, batches[0], ctx.device)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
