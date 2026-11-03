#!/usr/bin/env python
"""The load-time attention warm-ups leave the first step nothing to compile on models with per-layer head counts.

FA4 keys its kernels on the query/KV head ratio. A softcap stays off FA4's varlen kernels (not validated there), so a
varlen call carrying one is refused before launch and fails the check.
"""

import functools
from unittest.mock import patch

import torch
from transformers import (
    Gemma2Config,
    Gemma2ForCausalLM,
    LagunaConfig,
    LagunaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
)
from transformers import modeling_flash_attention_utils as flash_utils
from transformers.models.step3p7.configuration_step3p7 import Step3p7TextConfig
from transformers.models.step3p7.modeling_step3p7 import Step3p7TextModel

from halo.hardware import is_blackwell_gpu
from halo.kernels.jit_cache import ensure_fa4_kernel_cache_env
from halo.models.attention import sdpa_flex_sliding
from halo.models.attention.backend_setup import patch_transformers_flash_varlen_int_seqlen, warm_fa4_kernels
from halo.models.attention.layout import attention_calls
from halo.models.attention.sdpa_flex_sliding import register_flex_sliding_attention, warm_flex_sliding_kernels
from tests.common.flex_sliding import FLEX_ROWS, flex_layers, train_row
from tests.common.harness import gpu_test_main, record_check
from tests.common.models import TINY_LAGUNA_CONFIG, TINY_STEP3P7_CONFIG
from tests.common.rank_log import log

WINDOW = 256
HEAD_DIM = 128
# The longest document stays above FA4's 128-row tile and, times the head ratio, its query tile (at most 192 rows):
# below either, FA4 keys a one-block backward or single-stage forward the warm-up never compiles. A packed row drops
# the window when its length is within it.
ROWS = ([600], [200], [300, 90, 200], [200, 40])

STEP3P7 = {
    **TINY_STEP3P7_CONFIG,
    "num_attention_heads": 16,
    "num_sliding_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": HEAD_DIM,
    "sliding_window": WINDOW,
}
LAGUNA = {
    **TINY_LAGUNA_CONFIG,
    "num_attention_heads": 12,
    "num_attention_heads_per_layer": [12, 18, 18, 18],
    "num_key_value_heads": 2,
    "head_dim": HEAD_DIM,
    "sliding_window": WINDOW,
}
MISTRAL = {
    "vocab_size": 1024,
    "hidden_size": 256,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": HEAD_DIM,
    "sliding_window": WINDOW,
}
GEMMA2 = {
    **MISTRAL,
    "num_hidden_layers": 4,
    "attn_logit_softcapping": 50.0,
    "query_pre_attn_scalar": HEAD_DIM,
}
PER_LAYER_HEADS = {
    "step3p7": lambda: Step3p7TextModel(Step3p7TextConfig(**STEP3P7)),
    "laguna": lambda: LagunaForCausalLM(LagunaConfig(**LAGUNA)),
}
FA4_MODELS = {
    "step3p7": (PER_LAYER_HEADS["step3p7"], {8, 12}),
    "laguna": (PER_LAYER_HEADS["laguna"], {6, 9}),
    "gemma2": (lambda: Gemma2ForCausalLM(Gemma2Config(**GEMMA2)), {2}),
}
FLEX_MODELS = {**PER_LAYER_HEADS, "mistral": lambda: MistralForCausalLM(MistralConfig(**MISTRAL))}


def _model(build, attn: str) -> torch.nn.Module:
    torch.manual_seed(0)
    model = build().cuda().to(torch.bfloat16).train()
    model.config._attn_implementation = attn
    return model


def _counted(fn, calls: list):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        calls.append(1)
        return fn(*args, **kwargs)

    return wrapped


def _fa4_entry(fn, calls: list, *, varlen: bool):
    """Records ``(varlen, softcap)`` and refuses a varlen call with a softcap before it launches."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        softcap = kwargs.get("softcap") or None
        calls.append((varlen, softcap))
        if varlen and softcap:
            raise AssertionError(f"an FA4 varlen call with softcap {softcap} reached the kernel")
        return fn(*args, **kwargs)

    return wrapped


def _jit_keys(interface) -> dict[str, set]:
    """In-memory key set of each FA4 JIT cache (persistent hits populate it too)."""
    return {
        name: set(getattr(interface, name).compile_cache.cache)
        for name in ("_flash_attn_fwd", "_flash_attn_bwd", "_bwd_preprocess", "_bwd_postprocess_convert")
    }


def check_fa4_compiles_nothing_after_the_warmup(name: str) -> None:
    import flash_attn.cute as cute  # noqa: PLC0415 - optional Blackwell-only dependency

    build, ratios = FA4_MODELS[name]
    model = _model(build, "flash_attention_4")
    assert {call.heads // call.kv_heads for call in attention_calls(model)} == ratios  # Premise.
    softcaps = {call.softcap for call in attention_calls(model) if call.softcap}
    rows = [lengths for lengths in ROWS if len(lengths) == 1] if softcaps else list(ROWS)
    calls: list = []
    # transformers binds the entry points on its first flash call; unbinding makes it bind the recording ones.
    with (
        patch.object(cute, "flash_attn_func", _fa4_entry(cute.flash_attn_func, calls, varlen=False)),
        patch.object(cute, "flash_attn_varlen_func", _fa4_entry(cute.flash_attn_varlen_func, calls, varlen=True)),
        patch.object(flash_utils, "_loaded_implementation", None),
    ):
        warm_fa4_kernels(model, dtype=torch.bfloat16)
        warmup_calls = list(calls)
        # The warm-up logs and swallows a refused call, so its record is the verdict.
        refused = [call for call in warmup_calls if call[0] and call[1]]
        assert not refused, f"the warm-up made FA4 varlen calls with a softcap: {refused}"
        dense_softcaps = {softcap for varlen, softcap in warmup_calls if not varlen and softcap}
        assert dense_softcaps == softcaps, f"the warm-up's dense softcaps {dense_softcaps}, want {softcaps}"
        warmed = _jit_keys(cute.interface)
        assert warmed["_flash_attn_fwd"], "the warm-up compiled no forward kernel"
        calls.clear()
        for lengths in rows:
            train_row(model, lengths)
    layers = sum(1 for m in model.modules() if hasattr(m, "num_key_value_groups"))
    assert len(calls) == layers * len(rows), f"{len(calls)} FA4 calls; want every layer on every row"
    after = _jit_keys(cute.interface)
    new = {cache: after[cache] - warmed[cache] for cache in after if after[cache] - warmed[cache]}
    assert not new, f"compiled after the warm-up: {new}"
    log(f"  [{name}] {len(warmup_calls)} warm-up and {len(calls)} row FA4 calls, none varlen with a softcap")


def check_flex_compiles_nothing_after_the_warmup(name: str, build) -> None:
    flex = register_flex_sliding_attention()
    model = _model(build, flex)
    torch._dynamo.reset()
    warm_flex_sliding_kernels(model, dtype=torch.bfloat16)
    calls: list = []
    with (
        patch.object(torch._dynamo.config, "error_on_recompile", True),
        patch.object(sdpa_flex_sliding, "_compiled_flex", _counted(sdpa_flex_sliding._compiled_flex, calls)),
    ):
        for lengths, train in FLEX_ROWS:
            before = len(calls)
            train_row(model, lengths, train=train)
            want = flex_layers(model, lengths)
            assert len(calls) - before == want, f"{lengths}: {len(calls) - before} flex calls; want {want}"
    log(f"  [{name}] {len(calls)} FlexAttention calls, no recompile")


@gpu_test_main(exact_world_size=1, prefix="warmup_per_layer_heads")
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    for name, build in FLEX_MODELS.items():
        record_check(
            checks,
            f"{name}_flex_compiles_nothing_after_the_warmup",
            lambda name=name, build=build: check_flex_compiles_nothing_after_the_warmup(name, build),
        )
    if is_blackwell_gpu():
        ensure_fa4_kernel_cache_env()
        patch_transformers_flash_varlen_int_seqlen()
        for name in FA4_MODELS:
            record_check(
                checks,
                f"{name}_fa4_compiles_nothing_after_the_warmup",
                lambda name=name: check_fa4_compiles_nothing_after_the_warmup(name),
            )
    else:
        log("  FA4 checks skipped: FlashAttention 4 needs a Blackwell GPU")
    return {"checks": checks}


if __name__ == "__main__":
    run()
