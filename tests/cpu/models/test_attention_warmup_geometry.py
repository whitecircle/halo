"""The attention warm-ups compile each layer's own call, so the first real forward compiles nothing.
FA4 keys on the query/KV head ratio, so a global head count leaves the real compile inside the first DeepEP barrier."""

from __future__ import annotations

import sys
import types
from functools import partial
from typing import NamedTuple
from unittest import mock

import pytest
import torch
from torch.nn.attention import flex_attention as flex_attention_module
from torch.nn.attention.flex_attention import flex_attention
from transformers import (
    Gemma2Config,
    Gemma2ForCausalLM,
    Glm4MoeLiteConfig,
    Glm4MoeLiteForCausalLM,
    InklingForCausalLM,
    InklingTextConfig,
    LagunaConfig,
    LagunaForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    MiniCPM3Config,
    MiniCPM3ForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    PreTrainedModel,
)
from transformers import modeling_flash_attention_utils as flash_utils
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.step3p7.configuration_step3p7 import Step3p7TextConfig
from transformers.models.step3p7.modeling_step3p7 import Step3p7TextModel

from halo.models.attention import sdpa_flex_sliding
from halo.models.attention.backend_setup import WARMUP_MIN_SEQ_LEN, warm_fa4_kernels
from halo.models.attention.layout import AttentionCall, attention_calls
from halo.models.attention.sdpa_flex_sliding import (
    flex_attention_calls,
    register_flex_sliding_attention,
    warm_flex_sliding_kernels,
)
from tests.common.flex_sliding import FLEX_ROWS, flex_layers, flex_sliding_on_cpu, no_device_check, train_row
from tests.common.models import (
    TINY_GEMMA2_CONFIG,
    TINY_GLM4_MOE_LITE_CONFIG,
    TINY_INKLING_CONFIG,
    TINY_LAGUNA_CONFIG,
    TINY_MINICPM3_CONFIG,
    TINY_STEP3P7_CONFIG,
)
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TinyFamily,
    tiny_family_model,
)

FA4 = "flash_attention_4"
# The transformers module globals the flash path caches the bound kernel and its kwarg map in.
_FLASH_CACHE_GLOBALS = (
    "_loaded_implementation",
    "_flash_fn",
    "_flash_varlen_fn",
    "_flash_with_kvcache_fn",
    "_pad_fn",
    "_unpad_fn",
    "_process_flash_kwargs_fn",
    "_process_with_kvcache_kwargs_fn",
)


class _KernelKey(NamedTuple):
    """The call arguments FA4's compile key reads, for a forward kernel or the backward one."""

    dtype: torch.dtype
    qk_dim: int
    v_dim: int
    head_ratio: int
    causal: bool
    window: bool
    varlen: bool
    sink: bool
    softcap: bool
    lse: bool  # FA4's forward allocates the log-sum-exp only when its inputs require grad
    backward: bool
    deterministic: bool  # read by the backward's key alone


class _RecordedBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, record):
        ctx.record = record
        return q * 1.0

    @staticmethod
    def backward(ctx, grad):
        ctx.record()
        return grad, None, None, None


class _CompileCountingCute(types.ModuleType):
    """Compiles once per key, as FA4's JIT cache does, and raises on a varlen call with a softcap once recorded."""

    def __init__(self):
        super().__init__("flash_attn.cute")
        self.compiled: set[tuple] = set()
        self.calls: list[tuple] = []

        def flash_attn_func(
            q,
            k,
            v,
            softmax_scale=None,
            causal=False,
            window_size=(None, None),
            learnable_sink=None,
            softcap=0.0,
            deterministic=False,
        ):
            return self._run(
                q, k, v, causal, window_size, learnable_sink, softcap, varlen=False, deterministic=deterministic
            )

        def flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            softmax_scale=None,
            causal=False,
            window_size=(None, None),
            learnable_sink=None,
            softcap=0.0,
            deterministic=False,
        ):
            varlen = cu_seqlens_q is not None
            return self._run(
                q, k, v, causal, window_size, learnable_sink, softcap, varlen=varlen, deterministic=deterministic
            )

        self.flash_attn_func = flash_attn_func
        self.flash_attn_varlen_func = flash_attn_varlen_func

    def _run(self, q, k, v, causal, window_size, learnable_sink, softcap, *, varlen, deterministic):
        grad = any(t.requires_grad for t in (q, k, v))
        key = _KernelKey(
            q.dtype,
            q.shape[-1],
            v.shape[-1],
            q.shape[-2] // k.shape[-2],
            causal,
            window_size[0] is not None,
            varlen,
            learnable_sink is not None,
            bool(softcap),
            lse=grad,
            backward=False,
            deterministic=False,
        )
        self.calls.append(key)
        if varlen and softcap:
            raise AssertionError("an FA4 varlen call with a softcap, which stays off those kernels")
        self.compiled.add(key)
        if not grad:
            return q * 1.0
        backward_key = key._replace(backward=True, deterministic=deterministic)
        return _RecordedBackward.apply(q, k, v, partial(self._dispatch_backward, backward_key))

    def _dispatch_backward(self, key: _KernelKey) -> None:
        self.calls.append(key)
        self.compiled.add(key)


@pytest.fixture
def cute(monkeypatch) -> _CompileCountingCute:
    fake = _CompileCountingCute()
    monkeypatch.setitem(sys.modules, "flash_attn.cute", fake)
    for name in _FLASH_CACHE_GLOBALS:
        monkeypatch.setattr(flash_utils, name, getattr(flash_utils, name))
    flash_utils._loaded_implementation = None
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    return fake


def _step3p7() -> Step3p7TextModel:
    return Step3p7TextModel(Step3p7TextConfig(**TINY_STEP3P7_CONFIG))


def _laguna() -> LagunaForCausalLM:
    return LagunaForCausalLM(LagunaConfig(**TINY_LAGUNA_CONFIG))


def _gemma2() -> Gemma2ForCausalLM:
    """Gemma 2 with a window the single-document rows pass, so both its dense softcap kernels run."""
    return Gemma2ForCausalLM(Gemma2Config(**{**TINY_GEMMA2_CONFIG, "sliding_window": 64}))


def _mistral() -> MistralForCausalLM:
    return MistralForCausalLM(
        MistralConfig(
            vocab_size=64,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            sliding_window=16,
        )
    )


def _tiny(family: TinyFamily) -> torch.nn.Module:
    """A remote-code family builds on FA4: its file picks the attention class at build (Ling 2.0's v4 flash template).
    Only the build's device check is lifted; its ``_supports_flash_attn`` check stays live."""
    if not family.trust_remote_code:
        return tiny_family_model(family)
    with mock.patch.object(PreTrainedModel, "_flash_attn_import_error", lambda self, **_kwargs: None):
        return tiny_family_model(family, overrides={"_attn_implementation": FA4})


# The roster's tiny models, and generic families carrying a call argument no roster model does.
_FLASH_SWEEP = {
    **{name: partial(_tiny, family) for name, family in TINY_MOE_FAMILIES.items()},
    **{name: partial(_tiny, family) for name, family in TINY_MOE_VLM_FAMILIES.items()},
    **{name: partial(_tiny, family) for name, family in TINY_DENSE_FAMILIES.items()},
    "gemma2": _gemma2,
    "minicpm3": lambda: MiniCPM3ForCausalLM(MiniCPM3Config(**TINY_MINICPM3_CONFIG)),
    "mistral": _mistral,
}


def _on_fa4(model: torch.nn.Module) -> torch.nn.Module:
    torch.manual_seed(0)
    model = model.to(torch.bfloat16).train()
    model.config._attn_implementation = FA4
    return model


def _train_step(model: torch.nn.Module, lengths: list[int]) -> None:
    ids = torch.randint(3, 64, (1, sum(lengths)), generator=torch.Generator().manual_seed(sum(lengths)))
    position_ids = torch.cat([torch.arange(n) for n in lengths]).unsqueeze(0)
    model(input_ids=ids, position_ids=position_ids, use_cache=False)[0].float().sum().backward()


# Rows past the window and within it, single-document (dense kernel) and packed (varlen kernel).
_ROWS = ([150], [40], [100, 30, 90], [20, 30])


def _fa4_rows(model: torch.nn.Module) -> tuple[list[int], ...]:
    """A softcapped model runs single-document rows only, since a softcap stays off FA4's varlen kernels."""
    if any(call.softcap for call in attention_calls(model)):
        return tuple(lengths for lengths in _ROWS if len(lengths) == 1)
    return _ROWS


def test_step3p7_has_no_global_head_count():
    """A ``RuntimeError``, which ``getattr``'s default does not catch."""
    with pytest.raises(RuntimeError):
        _ = _step3p7().config.num_attention_heads


def test_laguna_global_head_count_is_the_full_layers():
    model = _laguna()
    assert model.config.num_attention_heads == 4
    assert {m.num_heads for m in model.modules() if getattr(m, "sliding_window", None)} == {8}


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        pytest.param(
            _step3p7,
            {AttentionCall(4, 2, 64, 64, None, 0.125, None), AttentionCall(2, 2, 64, 64, 64, 0.125, None)},
            id="step3p7",
        ),
        pytest.param(
            _laguna,
            {AttentionCall(4, 2, 64, 64, None, 0.125, None), AttentionCall(8, 2, 64, 64, 64, 0.125, None)},
            id="laguna",
        ),
        pytest.param(
            lambda: InklingForCausalLM(InklingTextConfig(**TINY_INKLING_CONFIG)),
            {AttentionCall(4, 2, 64, 64, None, 1 / 64, None), AttentionCall(64, 16, 128, 128, 32, 1 / 128, None)},
            id="inkling",
        ),
    ],
)
def test_each_layer_reports_its_own_head_count(build, expected):
    assert attention_calls(build()) == expected


def test_mla_reports_its_query_key_and_value_widths():
    """MLA keeps no ``head_dim`` on the module (GLM-4.7-Flash's config aliases it to the rope slice); flash runs
    at the query/key width, with the value padded up to it, and SDPA at the value's own width."""
    config = Glm4MoeLiteConfig(**{**TINY_GLM4_MOE_LITE_CONFIG, "v_head_dim": 16})
    assert config.head_dim != config.qk_head_dim
    assert config.v_head_dim != config.qk_head_dim
    calls = attention_calls(Glm4MoeLiteForCausalLM(config))
    assert {(call.head_dim, call.v_head_dim) for call in calls} == {(config.qk_head_dim, config.v_head_dim)}


def test_a_config_level_window_is_the_call_window():
    """Mistral keeps no window on its modules and passes the config's on every call."""
    assert {call.window for call in attention_calls(_mistral())} == {16}


def test_a_logit_softcap_is_part_of_the_call():
    config = Gemma2Config(**TINY_GEMMA2_CONFIG)
    assert config.attn_logit_softcapping
    calls = attention_calls(Gemma2ForCausalLM(config))
    assert {call.softcap for call in calls} == {config.attn_logit_softcapping}


def test_the_flex_warmup_reads_per_layer_heads():
    assert flex_attention_calls(_step3p7()) == {
        AttentionCall(4, 2, 64, 64, None, 0.125, None),
        AttentionCall(2, 2, 64, 64, 64, 0.125, None),
    }
    assert flex_attention_calls(_laguna()) == {
        AttentionCall(4, 2, 64, 64, None, 0.125, None),
        AttentionCall(8, 2, 64, 64, 64, 0.125, None),
    }


def _traced_flex(calls: list):
    """The kernel as ``sdpa_flex_sliding`` builds it, on Dynamo's eager backend, which a CPU runs."""
    compiled = torch.compile(flex_attention, dynamic=True, backend="aot_eager")

    def kernel(query, key, value, **kwargs):
        calls.append(1)
        return compiled(*(t.as_subclass(torch.Tensor) for t in (query, key, value)), **kwargs)

    return kernel


@pytest.fixture
def fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.usefixtures("fresh_dynamo")
@pytest.mark.parametrize("build", [_step3p7, _laguna, _mistral], ids=["step3p7", "laguna", "mistral"])
def test_a_flex_warmed_model_compiles_nothing_on_its_first_rows(build):
    """A packed row's full layers run FlexAttention under the document mask; a single document's keep SDPA.
    Each row runs FlexAttention on exactly the layers it masks."""
    flex = register_flex_sliding_attention()
    torch.manual_seed(0)
    model = build().to(torch.bfloat16).train()
    model.config._attn_implementation = flex
    calls: list = []
    with (
        mock.patch.object(flex_attention_module, "_validate_device", no_device_check),
        mock.patch.object(sdpa_flex_sliding, "_compiled_flex", _traced_flex(calls)),
    ):
        warm_flex_sliding_kernels(model, dtype=torch.bfloat16, device=torch.device("cpu"))
        assert len(calls) == 2 * len(flex_attention_calls(model))  # each call, training and forward-only
        with (
            mock.patch.object(torch._dynamo.config, "error_on_recompile", True),
            mock.patch.dict(ALL_ATTENTION_FUNCTIONS._global_mapping, {flex: flex_sliding_on_cpu}),
        ):
            for lengths, train in FLEX_ROWS:
                calls.clear()
                train_row(model, lengths, train=train)
                assert len(calls) == flex_layers(model, lengths), f"{lengths}: {len(calls)} FlexAttention calls"


def test_the_flex_warmup_skips_softcapped_calls():
    """A softcapped call keeps SDPA."""
    model = Gemma2ForCausalLM(Gemma2Config(**TINY_GEMMA2_CONFIG))
    assert any(call.window and call.softcap for call in attention_calls(model))
    assert flex_attention_calls(model) == set()


@pytest.mark.parametrize(
    ("build", "full_ratio", "sliding_ratio"), [(_step3p7, 2, 1), (_laguna, 2, 4)], ids=["step3p7", "laguna"]
)
def test_a_warmed_model_compiles_nothing_on_its_first_step(cute, build, full_ratio, sliding_ratio):
    model = _on_fa4(build())
    warm_fa4_kernels(model, dtype=torch.bfloat16, device=torch.device("cpu"))
    warmed = set(cute.compiled)
    cute.calls.clear()
    for lengths in _ROWS:
        _train_step(model, lengths)
    dispatched = set(cute.calls)
    # The rows reach six distinct calls, two of them windowless sliding-layer ones, each forward and backward.
    assert {(key.head_ratio, key.window, key.varlen, key.backward) for key in dispatched} == {
        (ratio, window, varlen, backward)
        for ratio, window in ((full_ratio, False), (sliding_ratio, True), (sliding_ratio, False))
        for varlen in (False, True)
        for backward in (False, True)
    }
    assert dispatched <= warmed, f"compiled mid-run: {sorted(dispatched - warmed, key=str)}"


def _warm_then_run_rows(cute: _CompileCountingCute, name: str) -> tuple[set[tuple], set[tuple]]:
    model = _FLASH_SWEEP[name]()
    if not model._supports_flash_attn:
        pytest.skip(f"{type(model).__name__} declares no flash support, so no run reaches FA4")
    model = _on_fa4(model)
    warm_fa4_kernels(model, dtype=torch.bfloat16, device=torch.device("cpu"))
    warmed = set(cute.compiled)
    cute.calls.clear()
    for lengths in _fa4_rows(model):
        _train_step(model, lengths)
    return warmed, set(cute.calls)


@pytest.mark.parametrize("name", sorted(_FLASH_SWEEP))
def test_every_flash_model_compiles_nothing_on_its_first_step(cute, name):
    warmed, dispatched = _warm_then_run_rows(cute, name)
    assert any(key.backward for key in dispatched), "no backward reached the flash kernel"
    assert dispatched <= warmed, f"compiled mid-run: {sorted(dispatched - warmed, key=str)}"


@pytest.mark.parametrize("name", sorted(_FLASH_SWEEP))
def test_a_deterministic_run_compiles_nothing_on_its_first_backward(cute, monkeypatch, name):
    """Every training call asks FA4 for its deterministic backward, a kernel of its own."""
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    warmed, dispatched = _warm_then_run_rows(cute, name)
    assert any(key.deterministic for key in dispatched), "no deterministic backward reached the flash kernel"
    assert dispatched <= warmed, f"compiled mid-run: {sorted(dispatched - warmed, key=str)}"


def test_a_softcapped_layer_warms_its_dense_call_only(cute):
    """A softcap stays off FA4's varlen kernels, so the warm-up covers the dense call one unpadded row trains with."""
    model = _on_fa4(_gemma2())
    assert {bool(call.window) for call in attention_calls(model) if call.softcap} == {True, False}
    warm_fa4_kernels(model, dtype=torch.bfloat16, device=torch.device("cpu"))
    softcapped = {(key.varlen, key.window, key.lse, key.backward) for key in cute.calls if key.softcap}
    assert softcapped == {(False, window, True, backward) for window in (False, True) for backward in (False, True)}


def test_the_softcap_kernel_is_warmed_at_the_layers_cap(cute):
    """FA4 hashes the cap's value into the kernel key, so the warm-up passes each layer's own."""
    caps = []
    record = cute._run

    def run(q, k, v, causal, window_size, learnable_sink, softcap, **kwargs):
        caps.append(softcap)
        return record(q, k, v, causal, window_size, learnable_sink, softcap, **kwargs)

    cute._run = run
    model = _on_fa4(Gemma2ForCausalLM(Gemma2Config(**TINY_GEMMA2_CONFIG)))
    warm_fa4_kernels(model, dtype=torch.bfloat16, device=torch.device("cpu"))
    assert set(caps) == {model.config.attn_logit_softcapping}


def test_the_warmup_sizes_each_sink_to_its_layer(cute, monkeypatch):
    """A sink tensor carries one value per query head of its layer.
    The sinks family is uniform, so its sinks ride on Laguna's per-layer heads here."""
    monkeypatch.setattr("halo.models.attention.backend_setup.uses_gpt_oss_sinks", lambda config: True)
    shapes = []
    record = cute._run

    def run(q, k, v, causal, window_size, learnable_sink, softcap, **kwargs):
        shapes.append((q.shape[-2], None if learnable_sink is None else tuple(learnable_sink.shape)))
        return record(q, k, v, causal, window_size, learnable_sink, softcap, **kwargs)

    cute._run = run
    warm_fa4_kernels(_on_fa4(_laguna()), dtype=torch.bfloat16, device=torch.device("cpu"))
    assert {heads for heads, _ in shapes} == {4, 8}  # both layer types warmed
    assert all(sink == (heads,) for heads, sink in shapes)


def test_a_window_longer_than_the_floor_lengthens_the_warmup_rows(cute):
    """The warm-up rows run past the longest window, so the windowed kernel compiles on a row it masks."""
    config = LlamaConfig(vocab_size=64, hidden_size=64, num_hidden_layers=1, num_attention_heads=4)
    model = LlamaForCausalLM(config)
    model.config._attn_implementation = FA4
    for module in model.modules():
        if hasattr(module, "num_key_value_groups"):
            module.sliding_window = 2 * WARMUP_MIN_SEQ_LEN
    lengths = []
    record = cute._run

    def run(q, *args, **kwargs):
        lengths.append(q.shape[-3])
        return record(q, *args, **kwargs)

    cute._run = run
    warm_fa4_kernels(model, dtype=torch.bfloat16, device=torch.device("cpu"))
    assert min(lengths) > 2 * WARMUP_MIN_SEQ_LEN


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
