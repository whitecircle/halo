#!/usr/bin/env python
"""Gemma 2 and Gemma 3 train each row shape on the backend its script resolves, as accurately as eager. Blackwell only.

FA4's head_dim-256 backward asserts on a softcap and a window, and a softcap stays off its varlen kernels, so only
Gemma 2 27B's unpacked row may stay on FA4. On Gemma 2 SDPA's logits error must be at least 10x eager's, showing the
softcap it drops matters at this geometry.
"""

import argparse
import copy
import statistics
import types

import torch
import torch.nn.functional as F
from transformers import Gemma2Config, Gemma2ForCausalLM, Gemma3ForCausalLM, Gemma3TextConfig

from halo.entrypoints.model_setup import default_padded_attn_implementation, default_training_rows_attn_implementation
from halo.model_loading.dtype import configure_float32_matmul_precision
from halo.model_loading.model_preparation import apply_family_patches
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.models.attention.selection import resolve_attn_implementation
from tests.common.harness import gpu_test_main, skip_unless_blackwell
from tests.common.numerics import max_abs_rel_err
from tests.common.rank_log import log

SEED = 0
SEQ = 1024
DOCUMENT_LENGTHS = (400, 300, 324)
PADDED_LENGTHS = (SEQ, SEQ - 200)
WINDOW = 256
SOFTCAP = 2.0
# Reaches well past SOFTCAP, where the softcap changes the softmax.
SCORE_SCALE = 8.0
# Two bf16 kernels accumulate in different orders.
EAGER_ERROR_RATIO = 1.5
# SDPA without the softcap is a different function, not a rounding difference.
SOFTCAP_DROP_RATIO = 10.0
# What FA4's varlen entry point raises once the family patches installed its refusal.
VARLEN_SOFTCAP_REFUSAL = "reached FlashAttention-4's varlen kernels"
FA2, FA4, SDPA = "flash_attention_2", "flash_attention_4", "sdpa"

_GEOMETRY = {
    "vocab_size": 1024,
    "hidden_size": 512,
    "intermediate_size": 1024,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "query_pre_attn_scalar": 256,
    "sliding_window": WINDOW,
    "layer_types": ["sliding_attention", "full_attention"],
}
_GEMMA2_SOFTCAP = {"attn_logit_softcapping": SOFTCAP}
# family -> (config class, model class, geometry overrides, the backend each row shape resolves to)
FAMILIES = {
    "gemma2": (Gemma2Config, Gemma2ForCausalLM, _GEMMA2_SOFTCAP, {"unpacked": FA2, "packed": FA2, "padded": FA2}),
    "gemma2_hd128": (
        Gemma2Config,
        Gemma2ForCausalLM,
        {**_GEMMA2_SOFTCAP, "head_dim": 128, "query_pre_attn_scalar": 128},
        {"unpacked": FA4, "packed": FA2, "padded": FA2},
    ),
    "gemma3_text": (Gemma3TextConfig, Gemma3ForCausalLM, {}, {"unpacked": FA2, "packed": FA2, "padded": FA2}),
}


def _rows(vocab_size: int, device) -> dict[str, dict]:
    generator = torch.Generator().manual_seed(SEED)
    ids = torch.randint(0, vocab_size, (2, SEQ), generator=generator).to(device)
    packed_positions = torch.cat([torch.arange(length) for length in DOCUMENT_LENGTHS]).unsqueeze(0).to(device)
    padding_mask = torch.stack([torch.arange(SEQ) < length for length in PADDED_LENGTHS]).long().to(device)
    return {
        "unpacked": {"input_ids": ids[:1], "position_ids": torch.arange(SEQ, device=device).unsqueeze(0)},
        "packed": {"input_ids": ids[:1], "position_ids": packed_positions},
        "padded": {"input_ids": ids, "attention_mask": padding_mask},
    }


def _request(row_name: str, config) -> str | None:
    """The request the script building ``row_name`` rows makes when the YAML pins none."""
    unset = types.SimpleNamespace(attn_implementation=None)
    if row_name == "padded":
        return default_padded_attn_implementation(unset, sinks_reset=True, checkpoint_config=config)
    return default_training_rows_attn_implementation(
        unset,
        packing=row_name == "packed",
        padding_free=False,
        per_device_train_batch_size=1,
        dtype=torch.bfloat16,
        checkpoint_config=config,
    )


def _forward_backward(model, row: dict, attn_implementation: str):
    model.config._attn_implementation = attn_implementation
    model.zero_grad(set_to_none=True)
    logits = model(**row, use_cache=False).logits.float()
    real = row.get("attention_mask", torch.ones_like(row["input_ids"])).bool()
    logits = logits[real]
    labels = row["input_ids"].roll(-1, dims=1)[real]
    F.cross_entropy(logits, labels).backward()
    grads = {name: p.grad.detach().float() for name, p in model.named_parameters() if p.grad is not None}
    return logits.detach(), grads


def _errors(model, row, attn_implementation, reference) -> dict[str, float]:
    logits, grads = _forward_backward(model, row, attn_implementation)
    ref_logits, ref_grads = reference
    finite = torch.isfinite(logits).all().item() and all(torch.isfinite(g).all().item() for g in grads.values())
    return {
        "logits": max_abs_rel_err(logits, ref_logits),
        "grad_median": statistics.median(max_abs_rel_err(grads[name], ref) for name, ref in ref_grads.items()),
        "finite": float(finite),
    }


def _fa4_refuses(model, row) -> bool:
    """Whether FA4 raises on this row, its kernel's assert or the patches' softcap refusal: the limit a route off FA4
    exists for."""
    model.config._attn_implementation = apply_family_patches(model.config, FA4)
    model.zero_grad(set_to_none=True)
    try:
        model(**row, use_cache=False).logits.float().sum().backward()
    except AssertionError as error:
        log(f"  FA4 refused: {error}")
        return True
    except RuntimeError as error:
        if VARLEN_SOFTCAP_REFUSAL not in str(error):
            raise
        log(f"  FA4's varlen entry point refused the forward: {error}")
        return True
    return False


@gpu_test_main(exact_world_size=1, partial_state=False, prefix="softcap_window_attention")
def run(ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    family_name = parser.parse_args().family
    config_cls, model_cls, overrides, expected = FAMILIES[family_name]
    configure_float32_matmul_precision()

    torch.manual_seed(SEED)
    base = model_cls(config_cls(**{**_GEOMETRY, **overrides}))
    with torch.no_grad():
        for layer in base.model.layers:
            layer.self_attn.q_proj.weight.mul_(SCORE_SCALE)
            layer.self_attn.k_proj.weight.mul_(SCORE_SCALE)

    reference_model = copy.deepcopy(base).to(ctx.device, torch.float32)
    model = copy.deepcopy(base).to(ctx.device, torch.bfloat16)
    rows = _rows(base.config.vocab_size, ctx.device)

    checks, metrics = {}, {}
    for row_name, row in rows.items():
        request = _request(row_name, base.config)
        backend = resolve_attn_implementation(
            copy.deepcopy(base.config), request, torch.bfloat16, sinks_policy=SinksPolicy.NEUTRALIZED
        )
        checks[f"{row_name}_resolves_as_expected"] = backend == expected[row_name]
        backend = apply_family_patches(model.config, backend)
        reference = _forward_backward(reference_model, row, "eager")
        eager = _errors(model, row, "eager", reference)
        chosen = _errors(model, row, backend, reference)
        log(f"  [{family_name}/{row_name}] request {request!r} -> {backend} {chosen}  bf16 eager {eager}")
        checks[f"{row_name}_finite"] = bool(chosen["finite"])
        for metric in ("logits", "grad_median"):
            checks[f"{row_name}_{metric}_within_eager_error"] = chosen[metric] <= EAGER_ERROR_RATIO * eager[metric]
            metrics[f"{row_name}_{metric}_{backend}"] = chosen[metric]
            metrics[f"{row_name}_{metric}_eager"] = eager[metric]
        if config_cls is Gemma2Config:
            sdpa = _errors(model, row, SDPA, reference)
            checks[f"{row_name}_sdpa_drops_the_softcap"] = sdpa["logits"] >= SOFTCAP_DROP_RATIO * eager["logits"]
            metrics[f"{row_name}_logits_sdpa"] = sdpa["logits"]
    if base.config.head_dim == 256:
        checks["fa4_backward_refuses_this_model"] = _fa4_refuses(model, rows["unpacked"])
    else:
        for row_name in ("packed", "padded"):
            checks[f"fa4_refuses_the_{row_name}_row"] = _fa4_refuses(model, rows[row_name])
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    skip_unless_blackwell("the softcap and window routes off FA4")
    run()
