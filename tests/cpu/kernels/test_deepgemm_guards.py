"""The two silent-corruption guards around native DeepGEMM: an ``offs`` not ending at ``T`` returns uninitialized
memory, and the warning key excludes what changes every forward."""

import ast
import pathlib
import re
import types

import pytest
import torch

import halo.kernels.lowp.deepgemm as deepgemm

E, K, N, T = 4, 8, 16, 40


def _stub_resolver(monkeypatch):
    """Only ``utils.align`` is reached before the ``offs`` guard."""
    utils = types.SimpleNamespace(align=lambda value, alignment: -(-value // alignment) * alignment)
    monkeypatch.setattr(deepgemm, "_deep_gemm", lambda: (None, utils))


def _forward(monkeypatch, ends: list[int]):
    _stub_resolver(monkeypatch)
    return deepgemm._deepgemm_forward(
        torch.zeros(T, K),
        torch.zeros(E, K, N),
        torch.tensor(ends, dtype=torch.int32),
        "mxfp8",
        False,
    )


def test_offs_that_stops_short_of_t_is_rejected(monkeypatch):
    """The tail rows of ``out`` are never written."""
    with pytest.raises(ValueError, match="ending at T="):
        _forward(monkeypatch, [10, 20, 30, T - 1])


def test_offs_overrunning_t_is_rejected(monkeypatch):
    """A cumulative count past ``T`` slices the input out of range."""
    with pytest.raises(ValueError, match="ending at T="):
        _forward(monkeypatch, [10, 20, 30, T + 1])


@pytest.mark.parametrize(
    "ends",
    [
        [10, 20, T],
        [10, 5, 20, T],
        [-1, 10, 20, T],
    ],
    ids=["short", "backwards", "negative"],
)
def test_length_ordering_and_negative_offsets_still_rejected(monkeypatch, ends):
    """The guard raises one message naming every invariant, so the echoed offs pin which input was rejected."""
    with pytest.raises(ValueError, match=re.escape(f"got {ends}")):
        _forward(monkeypatch, ends)


def test_full_coverage_passes_the_guard(monkeypatch):
    """The stub then fails on its own."""
    with pytest.raises(AttributeError):  # The stub resolver has no per_token_cast_to_fp8.
        _forward(monkeypatch, [10, 20, 30, T])


def _forward_on_a_kernel_raising(monkeypatch, error: Exception):
    def kernel(*args, **kwargs):
        raise error

    utils = types.SimpleNamespace(
        align=lambda value, alignment: -(-value // alignment) * alignment, per_token_cast_to_fp8=lambda x, **kw: x
    )
    monkeypatch.setattr(
        deepgemm, "_deep_gemm", lambda: (types.SimpleNamespace(m_grouped_fp8_gemm_nt_contiguous=kernel), utils)
    )
    monkeypatch.setattr(deepgemm, "_quant_weight", lambda *args, **kwargs: None)
    return deepgemm._deepgemm_forward(
        torch.zeros(T, K), torch.zeros(E, K, N), torch.tensor([10, 20, 30, T]), "mxfp8", False
    )


def test_an_oom_in_the_native_kernel_is_not_taken_for_a_rejected_shape(monkeypatch):
    """OutOfMemoryError is a RuntimeError, so without the re-raise the step retries in bf16 on a full GPU."""
    with pytest.raises(torch.cuda.OutOfMemoryError):
        _forward_on_a_kernel_raising(monkeypatch, torch.cuda.OutOfMemoryError("CUDA out of memory"))


def test_a_shape_the_native_kernel_rejects_falls_back(monkeypatch):
    key = ("mxfp8", N, K, E)
    try:
        assert _forward_on_a_kernel_raising(monkeypatch, RuntimeError("narrow N")) is None
    finally:
        deepgemm._warned_fallback_shapes.discard(key)


def test_a_direct_call_without_deep_gemm_raises_import_error(monkeypatch):
    """The resolver returns None when ``deep_gemm`` does not import."""
    monkeypatch.setattr(deepgemm, "_deep_gemm", lambda: None)
    with pytest.raises(ImportError, match="deep_gemm"):
        deepgemm.deepgemm_grouped_gemm(
            torch.zeros(T, K), torch.zeros(E, K, N), offs=torch.tensor([10, 20, 30, T]), quant_format="mxfp8"
        )


def test_fallback_warning_fires_once_per_weight_shape(caplog):
    """A key with the routed token count would re-warn each step and grow the seen-set without bound."""
    key = ("mxfp8", N, K, E)
    deepgemm._warned_fallback_shapes.discard(key)
    before = len(deepgemm._warned_fallback_shapes)
    try:
        with caplog.at_level("WARNING", logger=deepgemm.__name__):
            for tokens in (128, 256, 384):
                deepgemm._warn_fallback_once(key, f"m_grouped mxfp8 GEMM on M={tokens}", RuntimeError("narrow N"))
        rejected = [r for r in caplog.records if "DeepGEMM native path rejected" in r.getMessage()]
        assert len(rejected) == 1, [r.getMessage() for r in rejected]
        # The token count is outside the key, so the set cannot grow per step.
        assert len(deepgemm._warned_fallback_shapes) == before + 1
    finally:
        deepgemm._warned_fallback_shapes.discard(key)


def test_the_call_site_keys_only_on_the_weight_shape():
    """The tests above build their own key, so they pin ``set.add``; this pins what ``_deepgemm_forward``
    constructs."""
    tree = ast.parse(pathlib.Path(deepgemm.__file__).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_warn_fallback_once"
    ]
    assert calls, "the fallback warning is no longer issued from deepgemm.py"
    for call in calls:
        key = call.args[0]
        assert isinstance(key, ast.Tuple), f"the dedupe key must be a literal tuple, got {ast.dump(key)}"
        names = [getattr(elt, "id", None) for elt in key.elts]
        assert names == ["fmt", "N", "K", "E"], (
            f"dedupe key is {names}; anything per-forward (M, the exception text) makes the "
            f"once-per-shape promise false and the seen-set unbounded"
        )


def test_a_genuinely_different_shape_still_warns(caplog):
    """Anti-vacuity: per weight shape, not a one-shot global mute."""
    keys = [("mxfp8", N, K, E), ("nvfp4", N, K, E), ("mxfp8", N * 2, K, E)]
    for key in keys:
        deepgemm._warned_fallback_shapes.discard(key)
    try:
        with caplog.at_level("WARNING", logger=deepgemm.__name__):
            for key in keys:
                deepgemm._warn_fallback_once(key, "m_grouped GEMM", RuntimeError("narrow N"))
        assert len([r for r in caplog.records if "DeepGEMM native path rejected" in r.getMessage()]) == len(keys)
    finally:
        for key in keys:
            deepgemm._warned_fallback_shapes.discard(key)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
