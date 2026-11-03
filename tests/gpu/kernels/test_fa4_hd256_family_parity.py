#!/usr/bin/env python
"""Every head_dim-256 family, and each family the loader leaves on FA4 at another width, trains on FA4, as the loader
patches it, as accurately as on FA2, on four row shapes.

Full-attention layers and every expert routed, so the comparison is continuous in the attention output. The unpacked
and ``cu_seq_lens`` rows must reach FA4 strided, the layout flash-attn-4 4.0.0b16 read scrambled. GLM-4.7-Flash holds
the kernel path alone: the resolver keeps its latent attention on FA2 as a precaution.
"""

import argparse
import contextlib
import copy
import functools
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
    tiny_family_model,
)

SEED = 0
SEQ = 1024
DOCUMENT_LENGTHS = (400, 300, 324)
PADDED_LENGTHS = (SEQ, SEQ - 200)
# Two bf16 kernels accumulate in different orders.
FA2_ERROR_RATIO = 1.5
# Transformers' position_ids-only and unpad paths copy q/k/v; these rows hand them strided.
STRIDED_ROWS = ("unpacked", "packed_cu_seq_lens")

_GQA_256 = {"num_hidden_layers": 2, "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256}
_FULL_ATTENTION = {"layer_types": ["full_attention", "full_attention"]}
_ALL_EIGHT_EXPERTS = {"num_experts": 8, "num_experts_per_tok": 8}
FAMILIES = {
    "qwen3_5": (TINY_DENSE_FAMILIES["qwen3_5"], {**_GQA_256, **_FULL_ATTENTION}),
    "qwen3_5_moe_text": (TINY_MOE_FAMILIES["qwen3_5_moe_text"], {**_GQA_256, **_FULL_ATTENTION, **_ALL_EIGHT_EXPERTS}),
    "qwen3_next": (TINY_MOE_FAMILIES["qwen3_next"], {**_GQA_256, **_FULL_ATTENTION, **_ALL_EIGHT_EXPERTS}),
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
    "cohere2_moe": (TINY_MOE_FAMILIES["cohere2_moe"], {**_FULL_ATTENTION, **_ALL_EIGHT_EXPERTS}),
    "laguna": (
        TINY_MOE_FAMILIES["laguna"],
        {
            "num_hidden_layers": 2,
            "num_attention_heads_per_layer": [4, 4],
            "mlp_layer_types": ["sparse", "sparse"],
            **_FULL_ATTENTION,
            **_ALL_EIGHT_EXPERTS,
        },
    ),
    "lfm2_moe": (
        TINY_MOE_FAMILIES["lfm2_moe"],
        {"num_hidden_layers": 2, "num_dense_layers": 0, **_FULL_ATTENTION, **_ALL_EIGHT_EXPERTS},
    ),
    "mistral4": (
        TINY_MOE_FAMILIES["mistral4"],
        {
            "num_hidden_layers": 2,
            "qk_nope_head_dim": 32,
            "qk_rope_head_dim": 32,
            "qk_head_dim": 64,
            "v_head_dim": 64,
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


def _forward_backward(model, row: dict):
    model.zero_grad(set_to_none=True)
    logits = model(**row, use_cache=False).logits.float()
    real = row.get("attention_mask", torch.ones_like(row["input_ids"])).bool()
    logits = logits[real]
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
def _fa4_strided_calls():
    import flash_attn.cute as cute  # noqa: PLC0415 - optional Blackwell-only dependency

    strided = []
    entry_points = cute.flash_attn_func, cute.flash_attn_varlen_func

    def recording(fn):
        @functools.wraps(fn)
        def wrapped(q, k, v, *args, **kwargs):
            strided.append(not all(t.is_contiguous() for t in (q, k, v)))
            return fn(q, k, v, *args, **kwargs)

        return wrapped

    cute.flash_attn_func, cute.flash_attn_varlen_func = (recording(fn) for fn in entry_points)
    flash_utils._loaded_implementation = None  # transformers re-imports the entry points on its next call.
    try:
        yield strided
    finally:
        cute.flash_attn_func, cute.flash_attn_varlen_func = entry_points
        flash_utils._loaded_implementation = None


@gpu_test_main(exact_world_size=1, partial_state=False, prefix="fa4_hd256_family_parity")
def run(ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    family_name = parser.parse_args().family
    family, overrides = FAMILIES[family_name]
    if os.environ.get("FLASH_ATTENTION_DETERMINISTIC") == "1":
        raise RuntimeError("FLASH_ATTENTION_DETERMINISTIC=1 keeps head_dim-256 attention off FA4; unset it")
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
        with _fa4_strided_calls() as strided:
            fa4 = _errors(model, row, reference)
        model.config._attn_implementation = "flash_attention_2"
        fa2 = _errors(model, row, reference)
        log(f"  [{family_name}/{row_name}] fa4 {fa4}  fa2 {fa2}  strided fa4 calls {sum(strided)}/{len(strided)}")
        checks[f"{row_name}_finite"] = bool(fa4["finite"] and fa2["finite"])
        for metric in ("logits", "attention_grad_max", "grad_median"):
            checks[f"{row_name}_{metric}_within_fa2_error"] = fa4[metric] <= FA2_ERROR_RATIO * fa2[metric]
            metrics[f"{row_name}_{metric}_fa4"] = fa4[metric]
            metrics[f"{row_name}_{metric}_fa2"] = fa2[metric]
        if row_name in STRIDED_ROWS:
            checks[f"{row_name}_hands_fa4_strided_inputs"] = any(strided)
        metrics[f"{row_name}_fa4_strided_calls"] = sum(strided)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
