"""The two silent-corruption guards on the native DeepGEMM grouped path.

``deep_gemm`` ships in neither image, so the kernel itself is GPU-and-opt-in only
(``tests/gpu/kernels/test_deepgemm.py``). Both behaviors checked here sit *around* the kernel and
are reachable with the resolver stubbed:

- ``offs`` validation: the output buffer is uninitialized and only the covered span is written, so an
  ``offs`` that does not end at ``T`` returns garbage memory as activations,
- the fallback warning's once-per-shape claim, which holds only while its key excludes the routed
  token count and the exception text — both of which change every forward.
"""

import ast
import pathlib
import re
import sys
import types

import pytest
import torch

import src.kernels.lowp.deepgemm as deepgemm

E, K, N, T = 4, 8, 16, 40


def _stub_resolver(monkeypatch):
    """Stand in for ``_deep_gemm()``: only ``utils.align`` is reached before the ``offs`` guard."""
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
    """The tail rows of ``out`` are never written, so a short ``offs`` would return uninitialized
    memory as activations — the exact silent corruption the guard exists to prevent."""
    with pytest.raises(ValueError, match="ending at T="):
        _forward(monkeypatch, [10, 20, 30, T - 1])


def test_offs_overrunning_t_is_rejected(monkeypatch):
    """The mirror case: a cumulative count past ``T`` slices the input out of range."""
    with pytest.raises(ValueError, match="ending at T="):
        _forward(monkeypatch, [10, 20, 30, T + 1])


@pytest.mark.parametrize(
    "ends",
    [
        [10, 20, T],  # wrong length
        [10, 5, 20, T],  # goes backwards
        [-1, 10, 20, T],  # negative first offset
    ],
    ids=["short", "backwards", "negative"],
)
def test_length_ordering_and_negative_offsets_still_rejected(monkeypatch, ends):
    """Anti-vacuity: the coverage check is added to the length/monotonicity guard, not in place of it.

    Matching the echoed offs pins WHICH input was rejected — the guard raises one message naming
    every invariant, so a phrase from it would pass for any of these three.
    """
    with pytest.raises(ValueError, match=re.escape(f"got {ends}")):
        _forward(monkeypatch, ends)


def test_full_coverage_passes_the_guard(monkeypatch):
    """Anti-vacuity: a well-formed ``offs`` reaches the kernel (the stub then fails on its own)."""
    with pytest.raises(AttributeError):  # the stub resolver has no per_token_cast_to_fp8
        _forward(monkeypatch, [10, 20, 30, T])


def test_fallback_warning_fires_once_per_weight_shape(caplog):
    """The warning promises once per shape, and the routed token count varies every forward — so a
    key carrying it would both re-warn each step and grow the seen-set without bound."""
    key = ("mxfp8", N, K, E)
    deepgemm._WARNED_FALLBACK_SHAPES.discard(key)
    before = len(deepgemm._WARNED_FALLBACK_SHAPES)
    try:
        with caplog.at_level("WARNING", logger=deepgemm.__name__):
            for tokens in (128, 256, 384):
                deepgemm._warn_fallback_once(key, f"m_grouped mxfp8 GEMM on M={tokens}", RuntimeError("narrow N"))
        rejected = [r for r in caplog.records if "DeepGEMM native path rejected" in r.getMessage()]
        assert len(rejected) == 1, [r.getMessage() for r in rejected]
        # Three calls, one entry: the token count is outside the key, so the set cannot grow per step.
        assert len(deepgemm._WARNED_FALLBACK_SHAPES) == before + 1
    finally:
        deepgemm._WARNED_FALLBACK_SHAPES.discard(key)


def test_the_call_site_keys_only_on_the_weight_shape():
    """The two tests above hand ``_warn_fallback_once`` a key they built, so they pin ``set.add`` —
    not the key ``_deepgemm_forward`` actually constructs. That key is the whole promise: adding the
    padded routed-token count ``M`` (which changes every forward) re-warns each step and grows the
    seen-set without bound, and neither test above would notice."""
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
    """Anti-vacuity: the dedupe is per weight shape, not a one-shot global mute."""
    keys = [("mxfp8", N, K, E), ("nvfp4", N, K, E), ("mxfp8", N * 2, K, E)]
    for key in keys:
        deepgemm._WARNED_FALLBACK_SHAPES.discard(key)
    try:
        with caplog.at_level("WARNING", logger=deepgemm.__name__):
            for key in keys:
                deepgemm._warn_fallback_once(key, "m_grouped GEMM", RuntimeError("narrow N"))
        assert len([r for r in caplog.records if "DeepGEMM native path rejected" in r.getMessage()]) == len(keys)
    finally:
        for key in keys:
            deepgemm._WARNED_FALLBACK_SHAPES.discard(key)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
