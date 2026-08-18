"""What the flash-attention probes ask the kernel, and what the FA4 warm-up compiles.

Two decisions in ``patches/attention.py`` are read off a flash kernel's signature, and both must ask
the VARLEN entry point rather than the DENSE one:

  * the sink-capability probe. transformers builds its whole per-argument capability map
    (``_process_flash_kwargs_fn``) by inspecting ``_flash_varlen_fn`` and then applies that map to
    the dense call as well, so the VARLEN signature is what decides whether ``s_aux`` /
    ``learnable_sink`` reaches the kernel — and it drops the tensor with no warning when it does not.
    A dense-signature probe can therefore approve a backend that silently drops the sinks a
    ``reset_sinks: false`` run keeps live, which is nats of logprob against the served policy.
  * the pre-compile warm-up, whose whole job is that no rank meets a first-use JIT compile after the
    others have moved on. The dense and varlen kernels compile separately and a packed /
    padding-free batch dispatches the varlen one, so warming only the dense kernel leaves the shape
    production actually runs cold — the deadlock the warm-up exists to prevent.

    python tests/cpu/models/test_flash_sink_probe.py
"""

from __future__ import annotations

import inspect
import sys
import types

import pytest
import torch
from accelerate import PartialState
from transformers import LlamaConfig
from transformers import modeling_flash_attention_utils as flash_utils

import src.distributed.loading.warmup as warmup_mod
from src.distributed.loading.warmup import warm_attention_kernels
from src.models.patches.attention import (
    FLASH_SINK_PARAM_NAMES,
    _attn_impl_handles_sinks,
    flash_varlen_fn,
    warmup_fa4_kernels,
)

PartialState()  # the warm-up logs through accelerate's logger


# The probed entry point must stay the one transformers reads


def test_transformers_still_keys_capability_off_the_varlen_function():
    """Drift pin. If a transformers upgrade builds the capability map from the dense function
    instead, this fails and ``flash_varlen_fn`` has to be re-pointed with it — otherwise the probe
    would quietly answer a question the library no longer asks."""
    source = inspect.getsource(flash_utils.lazy_import_flash_attention)
    assert "_lazy_define_process_function(_flash_varlen_fn)" in source, source


def test_the_probe_returns_the_function_transformers_would_bind():
    """Identity, not just spelling: the probe and transformers must reach the same object for the
    installed FA4 build."""
    pytest.importorskip("flash_attn.cute")
    from flash_attn.cute import flash_attn_varlen_func

    assert flash_varlen_fn("flash_attention_4") is flash_attn_varlen_func


def test_the_sink_argument_names_match_transformers_alias_table():
    """transformers accepts the sink tensor under ``s_aux`` or its own alternative-name entry; the
    probe must test exactly those two, or a kernel exposing only one spelling is misread."""
    assert set(FLASH_SINK_PARAM_NAMES) == {"s_aux", flash_utils._flash_api_alternative_names["s_aux"]}


def _fake_cute_module(*, dense_sink: bool, varlen_sink: bool) -> types.ModuleType:
    """A stand-in ``flash_attn.cute`` whose dense and varlen signatures disagree about the sink."""
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
    """A build whose dense function takes the sink and whose varlen function does not must report
    NO sink support: transformers would never pass the tensor, for either shape."""
    monkeypatch.setitem(
        sys.modules, "flash_attn.cute", _fake_cute_module(dense_sink=dense_sink, varlen_sink=varlen_sink)
    )
    assert _attn_impl_handles_sinks("flash_attention_4") is handles


def test_an_absent_build_reports_no_sink_support(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn.cute", None)  # a None entry makes the import raise
    assert _attn_impl_handles_sinks("flash_attention_4") is False


# The FA4 warm-up


class _RecordingCute(types.ModuleType):
    """``flash_attn.cute`` whose two entry points record their inputs and stay differentiable."""

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


def _fa4_model() -> types.SimpleNamespace:
    config = LlamaConfig(num_hidden_layers=1, hidden_size=64, num_attention_heads=4, num_key_value_heads=2)
    config._attn_implementation = "flash_attention_4"
    return types.SimpleNamespace(config=config)


@pytest.fixture
def warmed(monkeypatch):
    """Run the warm-up against a recording kernel on CPU, with a world of 2 faked."""
    cute = _RecordingCute()
    monkeypatch.setitem(sys.modules, "flash_attn.cute", cute)
    monkeypatch.setattr(warmup_mod, "barrier", lambda: None)
    monkeypatch.setattr(warmup_mod, "get_global_world_size", lambda: 2)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    return cute


def test_the_warmup_compiles_both_entry_points(warmed):
    """Only the dense kernel was warmed, so the first packed step JIT-compiled the varlen one
    mid-forward — one rank stalling ~10s while its peers ran ahead into the next collective."""
    warmup_fa4_kernels(_fa4_model(), dtype=torch.bfloat16, device=torch.device("cpu"))
    assert [call["entry"] for call in warmed.calls] == ["dense", "varlen"]

    varlen = next(call for call in warmed.calls if call["entry"] == "varlen")
    # A real two-document cu_seqlens, so the compiled kernel is the one packing dispatches.
    assert torch.equal(varlen["kwargs"]["cu_seqlens_q"], torch.tensor([0, 512, 1024], dtype=torch.int32))
    assert varlen["kwargs"]["max_seqlen_q"] == 512
    assert varlen["q"].dim() == 3, "varlen takes flat [total_tokens, heads, dim] inputs"


def test_the_warmup_compiles_the_runs_dtype(warmed):
    """The kernel is keyed by dtype: a hardcoded bf16 warm-up under an fp16 run compiles a kernel
    the run never calls and leaves the one it does call cold — the deadlock, plus the compile."""
    warmup_fa4_kernels(_fa4_model(), dtype=torch.float16, device=torch.device("cpu"))
    assert warmed.calls
    for call in warmed.calls:
        assert {call[side].dtype for side in ("q", "k", "v")} == {torch.float16}


def test_the_warmup_stays_inert_for_another_backend(warmed):
    model = _fa4_model()
    model.config._attn_implementation = "sdpa"
    warmup_fa4_kernels(model, dtype=torch.bfloat16, device=torch.device("cpu"))
    assert warmed.calls == []


@pytest.mark.parametrize(
    ("mutate", "case"),
    [
        (lambda m: setattr(m.config, "_attn_implementation", "sdpa"), "another backend"),
        (lambda m: setattr(m, "config", None), "no config"),
    ],
)
def test_every_rank_reaches_the_barrier_whatever_it_resolved(monkeypatch, warmed, mutate, case):
    """The barrier is world-wide, so it cannot sit behind a PER-RANK verdict.

    The backend a rank resolves is its own (auto-detection, an FA4 build present on some nodes only,
    a composite config that keeps the field elsewhere), and a rank returning before the barrier
    leaves every peer that DID warm sitting in it — the same desync the warm-up exists to prevent,
    moved from the JIT to the barrier. Hence the fence sits at the distributed caller, above every
    per-rank verdict the compile itself takes.
    """
    barriers: list[int] = []
    monkeypatch.setattr(warmup_mod, "barrier", lambda: barriers.append(1))

    model = _fa4_model()
    mutate(model)
    warm_attention_kernels(model, dtype=torch.bfloat16)

    assert warmed.calls == [], f"{case}: nothing should compile here"
    assert barriers == [1], f"{case}: the rank skipped the barrier its peers are waiting in"


def test_a_single_process_run_does_not_barrier(monkeypatch, warmed):
    """Anti-vacuity: world==1 is the one verdict every rank shares, so it may skip the barrier."""
    barriers: list[int] = []
    monkeypatch.setattr(warmup_mod, "barrier", lambda: barriers.append(1))
    monkeypatch.setattr(warmup_mod, "get_global_world_size", lambda: 1)

    warm_attention_kernels(_fa4_model(), dtype=torch.bfloat16)

    assert barriers == [] and warmed.calls == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
