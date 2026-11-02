#!/usr/bin/env python
"""Every head_dim-256 family trains on FA4, as the loader patches it, as accurately as on FA2.

One family per launch (``--family``), at its real attention geometry on a tiny random-init model: Qwen3.5/3.6
dense and MoE and Qwen3-Next (16:2 heads, ``head_dim`` 256, partial rotary), GLM-4.7-Flash (MLA, 192 + 64 query/key
and 256 value width). Full-attention layers only, and every expert routed, so the comparison is a continuous
function of the attention output. ``apply_family_patches`` installs FA4's backend fixes exactly as the loader does;
the bf16 model then runs FA4 and FA2 against the same weights in fp32 on SDPA, on four rows: one unpacked row, three
packed documents by ``position_ids`` alone and again with the ``cu_seq_lens`` markers the packing and padding-free
collators emit, and a right-padded batch of two (FA's unpad path).

Per row: logits, the attention parameters' gradients (worst one) and every gradient (median) must stay within
1.5x of FA2's error, all finite. On the rows whose q/k/v reach the kernel strided (unpacked, ``cu_seq_lens``), a
forward on the unwrapped FA4 kernel shows the family meets the head_dim-256 stride defect at all (its error at
least 10x FA2's); transformers copies q/k/v on the ``position_ids``-only and unpad paths. Fails if the loader stops
installing the contiguity fix, if the fix misses a tensor this family hands the kernel, or if FA4 trails FA2 on
any row.

    torchrun --nproc_per_node=1 tests/gpu/kernels/test_fa4_hd256_family_parity.py --family qwen3_5_moe_text
"""

import argparse
import contextlib
import copy
import os
import statistics

import torch
import torch.nn.functional as F
from transformers import modeling_flash_attention_utils as flash_utils

from halo.model_loading.dtype import configure_float32_matmul_precision
from halo.model_loading.model_preparation import apply_family_patches
from halo.models.segment_markers import SegmentMarkers, segment_marker_kwargs
from tests.common.harness import gpu_test_main
from tests.common.numerics import max_abs_rel_err
from tests.common.rank_log import log
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    TINY_UNWRAPPED_MOE_FAMILY,
    tiny_family_model,
)

SEED = 0
SEQ = 1024
DOCUMENT_LENGTHS = (400, 300, 324)
PADDED_LENGTHS = (SEQ, SEQ - 200)
# Two bf16 kernels accumulate in different orders; FA4 may trail FA2 by this factor.
FA2_ERROR_RATIO = 1.5
# The unwrapped kernel reads another head's rows: its logits error is far past bf16 round-off.
MISREAD_RATIO = 10.0
# The rows whose q/k/v transformers hands the kernel strided; its position_ids-only and unpad paths copy them.
STRIDED_ROWS = ("unpacked", "packed_cu_seq_lens")

_GQA_256 = {"num_hidden_layers": 2, "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256}
_FULL_ATTENTION = {"layer_types": ["full_attention", "full_attention"]}
_ALL_EIGHT_EXPERTS = {"num_experts": 8, "num_experts_per_tok": 8}
FAMILIES = {
    "qwen3_5": (TINY_DENSE_FAMILIES["qwen3_5"], {**_GQA_256, **_FULL_ATTENTION}),
    "qwen3_5_moe_text": (TINY_MOE_FAMILIES["qwen3_5_moe_text"], {**_GQA_256, **_FULL_ATTENTION, **_ALL_EIGHT_EXPERTS}),
    "qwen3_next": (TINY_UNWRAPPED_MOE_FAMILY, {**_GQA_256, **_FULL_ATTENTION, **_ALL_EIGHT_EXPERTS}),
    "glm4_moe_lite": (
        TINY_MOE_FAMILIES["glm4_moe_lite"],
        {
            "num_attention_heads": 20,
            "num_key_value_heads": 20,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "v_head_dim": 256,
            "kv_lora_rank": 64,
            "n_routed_experts": 8,
            "num_experts_per_tok": 8,
        },
    ),
}


def _rows(vocab_size: int, device) -> dict[str, dict]:
    generator = torch.Generator().manual_seed(SEED)
    ids = torch.randint(0, vocab_size, (2, SEQ), generator=generator).to(device)
    packed_positions = torch.cat([torch.arange(length) for length in DOCUMENT_LENGTHS]).unsqueeze(0).to(device)
    padding_mask = torch.stack([torch.arange(SEQ) < length for length in PADDED_LENGTHS]).long().to(device)
    cu_seq_lens = segment_marker_kwargs(packed_positions, SegmentMarkers(cu_seq_lens=True))
    return {
        "unpacked": {"input_ids": ids[:1], "position_ids": torch.arange(SEQ, device=device).unsqueeze(0)},
        "packed_position_ids": {"input_ids": ids[:1], "position_ids": packed_positions},
        "packed_cu_seq_lens": {"input_ids": ids[:1], "position_ids": packed_positions, **cu_seq_lens},
        "padded": {"input_ids": ids, "attention_mask": padding_mask},
    }


def _forward_backward(model, row: dict, *, backward: bool = True):
    """Logits over the real tokens and every parameter's gradient of a next-token loss over them."""
    model.zero_grad(set_to_none=True)
    logits = model(**row, use_cache=False).logits.float()
    real = row.get("attention_mask", torch.ones_like(row["input_ids"])).bool()
    logits = logits[real]
    if not backward:
        return logits.detach(), {}
    labels = row["input_ids"].roll(-1, dims=1)[real]
    F.cross_entropy(logits, labels).backward()
    grads = {name: p.grad.detach().float() for name, p in model.named_parameters() if p.grad is not None}
    return logits.detach(), grads


def _errors(model, row, reference) -> dict[str, float]:
    logits, grads = _forward_backward(model, row)
    ref_logits, ref_grads = reference
    per_param = {name: max_abs_rel_err(grads[name], ref) for name, ref in ref_grads.items()}
    attention = [error for name, error in per_param.items() if "self_attn" in name]
    finite = torch.isfinite(logits).all().item() and all(torch.isfinite(g).all().item() for g in grads.values())
    return {
        "logits": max_abs_rel_err(logits, ref_logits),
        "attention_grad_max": max(attention),
        "grad_median": statistics.median(per_param.values()),
        "finite": float(finite),
    }


@contextlib.contextmanager
def _unwrapped_fa4():
    """FA4 with Halo's entry-point wrappers peeled off, for the duration of the block."""
    import flash_attn.cute as cute  # noqa: PLC0415 - optional Blackwell-only dependency

    wrapped = cute.flash_attn_func, cute.flash_attn_varlen_func
    cute.flash_attn_func, cute.flash_attn_varlen_func = (fn.__wrapped__ for fn in wrapped)
    flash_utils._loaded_implementation = None  # transformers re-imports the entry points on its next call
    try:
        yield
    finally:
        cute.flash_attn_func, cute.flash_attn_varlen_func = wrapped
        flash_utils._loaded_implementation = None


@gpu_test_main(exact_world_size=1, partial_state=False, prefix="fa4_hd256_family_parity")
def run(ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    family_name = parser.parse_args().family
    family, overrides = FAMILIES[family_name]
    if os.environ.get("FLASH_ATTENTION_DETERMINISTIC") == "1":
        raise RuntimeError("FLASH_ATTENTION_DETERMINISTIC=1 makes FA4's head_dim-256 backward assert; unset it")
    configure_float32_matmul_precision()

    torch.manual_seed(SEED)
    base = tiny_family_model(family, overrides=overrides)
    reference_model = copy.deepcopy(base).to(ctx.device, torch.float32)
    reference_model.config._attn_implementation = "sdpa"
    model = copy.deepcopy(base).to(ctx.device, torch.bfloat16)
    rows = _rows(base.config.vocab_size, ctx.device)

    checks, metrics = {}, {}
    for row_name, row in rows.items():
        reference = _forward_backward(reference_model, row)
        model.config._attn_implementation = apply_family_patches(model.config, "flash_attention_4")
        fa4 = _errors(model, row, reference)
        with _unwrapped_fa4():
            misread = max_abs_rel_err(_forward_backward(model, row, backward=False)[0], reference[0])
        model.config._attn_implementation = "flash_attention_2"
        fa2 = _errors(model, row, reference)
        log(f"  [{family_name}/{row_name}] fa4 {fa4}  fa2 {fa2}  unwrapped-fa4 logits {misread:.2e}")
        checks[f"{row_name}_finite"] = bool(fa4["finite"] and fa2["finite"])
        for metric in ("logits", "attention_grad_max", "grad_median"):
            checks[f"{row_name}_{metric}_within_fa2_error"] = fa4[metric] <= FA2_ERROR_RATIO * fa2[metric]
            metrics[f"{row_name}_{metric}_fa4"] = fa4[metric]
            metrics[f"{row_name}_{metric}_fa2"] = fa2[metric]
        if row_name in STRIDED_ROWS:
            checks[f"{row_name}_family_reaches_the_stride_defect"] = not (misread < MISREAD_RATIO * fa2["logits"])
        metrics[f"{row_name}_logits_unwrapped_fa4"] = misread
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
