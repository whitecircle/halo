#!/usr/bin/env python
"""The load-time attention warm-ups leave nothing for the first training step to compile on models whose head
count varies by layer.

Step-3.7 (64 full / 96 sliding heads over 8 KV heads) and Laguna (48 / 72 over 8) give each layer type its own
query/KV head ratio, which FA4 keys its compiled kernels on; the reduced models below keep those ratios and the
128-wide heads, so they compile the same kernels as the released checkpoints. Each model is warmed, then trains
on single-document and packed rows past and within the sliding window:

* FA4 (Blackwell): no new key lands in FA4's forward or backward JIT caches. Gemma 2 at the 27B's geometry
  (128-wide heads, ratio 2) rides along on its single-document rows: its logit softcap is part of the kernel key,
  value included, and a softcap stays off FA4's varlen kernels (not validated there; on B300 their forward with one
  did not finish), so a varlen call carrying a softcap, the warm-up's included, is refused before it launches and
  fails the check;
* ``sdpa_flex_sliding``: Dynamo raises on any recompile, and the sliding layers are seen to run FlexAttention.
  Mistral, whose modules keep no window and pass the config's, rides along.

A warm-up keyed on the config's global head count raised on Step-3.7 (the count is per layer) and compiled
Laguna's sliding kernel at the full layers' ratio, so its first sliding call compiled inside the step.

Run: torchrun --nproc_per_node=1 tests/gpu/kernels/test_attention_warmup_per_layer_heads.py
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
from tests.common.harness import gpu_test_main, record_check
from tests.common.models import TINY_LAGUNA_CONFIG, TINY_STEP3P7_CONFIG
from tests.common.rank_log import log

WINDOW = 256
HEAD_DIM = 128
# Single-document rows past and within the window, then packed rows of both kinds (a packed row drops the window
# when its total length is within it). The longest document of each row stays above FA4's 128-row tile, and
# times the head ratio above its query tile (at most 192 rows): below either, FA4 keys a one-block backward or a
# single-stage forward the 512-token warm-up rows never compile, a length gap apart from the head counts pinned.
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
# Each FA4 model and the query/KV head ratios its layers attend at.
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


def _step(model: torch.nn.Module, lengths: list[int], *, train: bool = True) -> None:
    """A forward over one row of ``lengths`` packed documents, and its backward when ``train``."""
    generator = torch.Generator().manual_seed(sum(lengths))
    ids = torch.randint(3, 1024, (1, sum(lengths)), generator=generator).cuda()
    position_ids = torch.cat([torch.arange(n) for n in lengths]).unsqueeze(0).cuda()
    with torch.set_grad_enabled(train):
        out = model(input_ids=ids, position_ids=position_ids, use_cache=False)
    if train:
        hidden = out.logits if hasattr(out, "logits") else out.last_hidden_state
        hidden.float().pow(2).mean().backward()


def _counted(fn, calls: list):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        calls.append(1)
        return fn(*args, **kwargs)

    return wrapped


def _fa4_entry(fn, calls: list, *, varlen: bool):
    """``fn`` recording each call as ``(varlen, softcap)``, and refusing a varlen call with a softcap before it
    launches, since the softcap stays off FA4's varlen kernels."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        softcap = kwargs.get("softcap") or None
        calls.append((varlen, softcap))
        if varlen and softcap:
            raise AssertionError(f"an FA4 varlen call with softcap {softcap} reached the kernel")
        return fn(*args, **kwargs)

    return wrapped


def _jit_keys(interface) -> dict[str, set]:
    """The in-memory key set of each FA4 JIT cache (persistent hits populate it too)."""
    return {
        name: set(getattr(interface, name).compile_cache.cache)
        for name in ("_flash_attn_fwd", "_flash_attn_bwd", "_bwd_preprocess", "_bwd_postprocess_convert")
    }


def check_fa4_compiles_nothing_after_the_warmup(name: str) -> None:
    import flash_attn.cute as cute  # noqa: PLC0415 - optional Blackwell-only dependency

    build, ratios = FA4_MODELS[name]
    model = _model(build, "flash_attention_4")
    assert {call.heads // call.kv_heads for call in attention_calls(model)} == ratios  # premise
    softcaps = {call.softcap for call in attention_calls(model) if call.softcap}
    rows = [lengths for lengths in ROWS if len(lengths) == 1] if softcaps else list(ROWS)
    calls: list = []
    # The warm-up resolves the entry points per call, and transformers binds them on its first flash call;
    # unbinding makes it bind the recording ones.
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
            _step(model, lengths)
    layers = sum(1 for m in model.modules() if hasattr(m, "num_key_value_groups"))
    assert len(calls) == layers * len(rows), f"{len(calls)} FA4 calls; want every layer on every row"
    after = _jit_keys(cute.interface)
    new = {cache: after[cache] - warmed[cache] for cache in after if after[cache] - warmed[cache]}
    assert not new, f"compiled after the warm-up: {new}"
    log(f"  [{name}] {len(warmup_calls)} warm-up and {len(calls)} row FA4 calls, none varlen with a softcap")


def check_flex_compiles_nothing_after_the_warmup(name: str, build) -> None:
    flex = register_flex_sliding_attention()
    model = _model(build, flex)
    sliding = sum(
        1
        for m in model.modules()
        if hasattr(m, "num_key_value_groups") and getattr(m, "sliding_window", model.config.sliding_window)
    )
    torch._dynamo.reset()
    warm_flex_sliding_kernels(model, dtype=torch.bfloat16)
    calls: list = []
    with (
        patch.object(torch._dynamo.config, "error_on_recompile", True),
        patch.object(sdpa_flex_sliding, "_compiled_flex", _counted(sdpa_flex_sliding._compiled_flex, calls)),
    ):
        _step(model, [300, 150, 40])
        _step(model, [600])
        _step(model, [400], train=False)
    assert len(calls) == 3 * sliding, f"{len(calls)} flex calls; want {sliding} sliding layers x 3 forwards"
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
