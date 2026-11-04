"""What the flash-attention probes ask the kernel, and what the FA4 warm-up compiles.
Both must ask the varlen entry point: transformers maps capabilities off it, and packed batches dispatch it."""

from __future__ import annotations

import contextlib
import inspect
import sys
import types

import pytest
import torch
from transformers import Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM, LlamaConfig, LlamaForCausalLM
from transformers import modeling_flash_attention_utils as flash_utils

import halo.model_loading.parallel.whole_model as warmup_mod
from halo.model_loading.parallel.whole_model import warm_attention_kernels
from halo.models.attention.backend_setup import warm_fa4_kernels
from halo.models.attention.selection import FLASH_SINK_PARAM_NAMES, _attn_impl_handles_sinks, flash_kernel
from tests.common.models import TINY_GLM4_MOE_LITE_CONFIG


def test_transformers_still_keys_capability_off_the_varlen_function():
    """If an upgrade builds the capability map from the dense function, ``flash_kernel`` must be re-pointed with it."""
    source = inspect.getsource(flash_utils.lazy_import_flash_attention)
    assert "_lazy_define_process_function(_flash_varlen_fn)" in source, source


def test_the_probe_returns_the_function_transformers_would_bind():
    cute = pytest.importorskip("flash_attn.cute")
    assert flash_kernel("flash_attention_4") is cute.flash_attn_varlen_func


def test_the_sink_argument_names_match_transformers_alias_table():
    """The probe must test exactly these two, or a kernel exposing only one spelling is misread."""
    assert set(FLASH_SINK_PARAM_NAMES) == {"s_aux", flash_utils._flash_api_alternative_names["s_aux"]}


def _fake_cute_module(*, dense_sink: bool, varlen_sink: bool) -> types.ModuleType:
    mod = types.ModuleType("flash_attn.cute")

    if dense_sink:

        def flash_attn_func(q, k, v, learnable_sink=None):
            raise NotImplementedError
    else:

        def flash_attn_func(q, k, v):
            raise NotImplementedError

    if varlen_sink:

        def flash_attn_varlen_func(q, k, v, cu_seqlens_q=None, cu_seqlens_k=None, learnable_sink=None):
            raise NotImplementedError
    else:

        def flash_attn_varlen_func(q, k, v, cu_seqlens_q=None, cu_seqlens_k=None):
            raise NotImplementedError

    mod.flash_attn_func = flash_attn_func
    mod.flash_attn_varlen_func = flash_attn_varlen_func
    return mod


@pytest.mark.parametrize(
    ("dense_sink", "varlen_sink", "handles"),
    [(True, False, False), (False, True, True)],
    ids=["dense-only-sink-is-not-support", "varlen-sink-is-support"],
)
def test_the_sink_probe_reads_the_varlen_signature(monkeypatch, dense_sink, varlen_sink, handles):
    """A dense-only sink is no support: transformers would never pass the tensor, for either shape."""
    monkeypatch.setitem(
        sys.modules, "flash_attn.cute", _fake_cute_module(dense_sink=dense_sink, varlen_sink=varlen_sink)
    )
    assert _attn_impl_handles_sinks("flash_attention_4") is handles


def test_an_absent_build_reports_no_sink_support(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn.cute", None)  # a None entry makes the import raise
    assert _attn_impl_handles_sinks("flash_attention_4") is False


class _RecordingCute(types.ModuleType):
    def __init__(self):
        super().__init__("flash_attn.cute")
        self.calls: list[dict] = []
        self.flash_attn_func = self._record("dense")
        self.flash_attn_varlen_func = self._record("varlen")

    def _record(self, name):
        def call(q, k, v, **kwargs):
            self.calls.append({"entry": name, "q": q, "k": k, "v": v, "kwargs": kwargs})
            return q

        return call


def _fa4_model() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=64, num_hidden_layers=1, hidden_size=64, num_attention_heads=4, num_key_value_heads=2
    )
    model = LlamaForCausalLM(config)
    model.config._attn_implementation = "flash_attention_4"
    return model


@pytest.fixture
def warmed(monkeypatch):
    cute = _RecordingCute()
    monkeypatch.setitem(sys.modules, "flash_attn.cute", cute)
    monkeypatch.setattr(warmup_mod, "get_global_world_size", lambda: 2)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    return cute


def test_the_warmup_compiles_both_entry_points(warmed):
    """Warming only the dense kernel leaves the varlen one to JIT-compile on the first packed step.
    One rank stalls ~10s while its peers run ahead into the next collective."""
    warm_fa4_kernels(_fa4_model(), dtype=torch.bfloat16, device=torch.device("cpu"))
    assert [call["entry"] for call in warmed.calls] == ["dense", "varlen"]

    varlen = next(call for call in warmed.calls if call["entry"] == "varlen")
    # A real two-document cu_seqlens, so the compiled kernel is the one packing dispatches.
    assert torch.equal(varlen["kwargs"]["cu_seqlens_q"], torch.tensor([0, 512, 1024], dtype=torch.int32))
    assert varlen["kwargs"]["max_seqlen_q"] == 512
    assert varlen["q"].dim() == 3, "varlen takes flat [total_tokens, heads, dim] inputs"


def test_the_warmup_compiles_the_runs_dtype(warmed):
    """The kernel is keyed by dtype: a hardcoded bf16 warm-up under an fp16 run leaves the one it calls cold."""
    warm_fa4_kernels(_fa4_model(), dtype=torch.float16, device=torch.device("cpu"))
    assert warmed.calls
    for call in warmed.calls:
        assert {call[side].dtype for side in ("q", "k", "v")} == {torch.float16}


@pytest.mark.parametrize(
    ("qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"),
    [(192, 64, 256), (128, 64, 128)],
    ids=["glm-4.7-flash-256", "qk-wider-than-v"],
)
def test_the_warmup_compiles_the_mla_head_widths(warmed, qk_nope_head_dim, qk_rope_head_dim, v_head_dim):
    """MLA's ``head_dim`` is only its RoPE slice (GLM-4.7-Flash: 64 against 256-wide q/k/v).
    Sized off it, the run's kernel compiles on every rank in the first step, outside the fence.
    transformers' flash path pads a narrower value up to the q/k width, so that is the value width the kernel sees."""
    config = Glm4MoeLiteConfig(
        **{
            **TINY_GLM4_MOE_LITE_CONFIG,
            "num_hidden_layers": 1,
            "qk_nope_head_dim": qk_nope_head_dim,
            "qk_rope_head_dim": qk_rope_head_dim,
            "v_head_dim": v_head_dim,
        }
    )
    model = Glm4MoeLiteForCausalLM(config)
    model.config._attn_implementation = "flash_attention_4"
    warm_fa4_kernels(model, dtype=torch.bfloat16, device=torch.device("cpu"))

    qk_width = qk_nope_head_dim + qk_rope_head_dim
    assert len(warmed.calls) == 2
    for call in warmed.calls:
        assert (call["q"].shape[-1], call["k"].shape[-1], call["v"].shape[-1]) == (qk_width, qk_width, qk_width)
        assert call["kwargs"]["softmax_scale"] == qk_width**-0.5


def test_the_warmup_stays_inert_for_another_backend(warmed):
    model = _fa4_model()
    model.config._attn_implementation = "sdpa"
    warm_fa4_kernels(model, dtype=torch.bfloat16, device=torch.device("cpu"))
    assert warmed.calls == []


def _recorded_fences(monkeypatch) -> list[str]:
    fences: list[str] = []

    @contextlib.contextmanager
    def fence(what: str):
        yield
        fences.append(what)

    monkeypatch.setattr(warmup_mod, "barrier_on_exit", fence)
    return fences


@pytest.mark.parametrize(
    ("mutate", "case"),
    [
        (lambda m: setattr(m.config, "_attn_implementation", "sdpa"), "another backend"),
        (lambda m: setattr(m, "config", None), "no config"),
    ],
)
def test_every_rank_reaches_the_fence_whatever_it_resolved(monkeypatch, warmed, mutate, case):
    """The fence is world-wide, but each rank resolves its own backend (auto-detection, FA4 on some nodes only).
    A rank returning early leaves peers that warmed stuck in it, so the fence sits above every per-rank verdict."""
    fences = _recorded_fences(monkeypatch)

    model = _fa4_model()
    mutate(model)
    warm_attention_kernels(model, dtype=torch.bfloat16)

    assert warmed.calls == [], f"{case}: nothing should compile here"
    assert len(fences) == 1, f"{case}: the rank skipped the fence its peers are waiting in"


def test_a_single_process_run_does_not_fence(monkeypatch, warmed):
    """world==1 is the one verdict every rank shares, so it may skip the fence."""
    fences = _recorded_fences(monkeypatch)
    monkeypatch.setattr(warmup_mod, "get_global_world_size", lambda: 1)

    warm_attention_kernels(_fa4_model(), dtype=torch.bfloat16)

    assert fences == [] and warmed.calls == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
