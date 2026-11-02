"""The two silent-corruption guards on the native DeepGEMM grouped path.

``deep_gemm`` is Blackwell-only and opt-in (``HALO_DEEPGEMM_NATIVE=1``), so the kernel itself is
GPU-tested (``tests/gpu/kernels/test_deepgemm.py``). Both behaviors checked here sit *around* the
kernel and are reachable with the resolver stubbed:

- ``offs`` validation: the output buffer is uninitialized and only the covered span is written, so an
  ``offs`` that does not end at ``T`` returns garbage memory as activations,
- the fallback warning's once-per-shape claim, which holds only while its key excludes the routed
  token count and the exception text — both of which change every forward.
"""

import ast
import pathlib
import re
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


_WEIGHT_K = 256  # divisible by the fp8 block (128) and the fp4 packing (256)
# Aligned and unaligned N, and E=1 where a slice of the padded layout would keep the padded stride.
_WEIGHT_SHAPES = [("mxfp8", 4, 256), ("mxfp8", 4, 100), ("mxfp8", 1, 100), ("nvfp4", 4, 100), ("nvfp4", 1, 100)]
_CHUNKS = pytest.mark.parametrize("chunk_mb", [256, 0], ids=["one-chunk", "chunk-per-expert"])


def _weight_cast(fmt, utils):
    """The cast ``_quant_weight`` applies per format, on whichever ``utils`` it is handed."""
    if fmt == "nvfp4":
        return lambda t: utils.per_token_cast_to_fp4(t, use_ue8m0=True, gran_k=32)
    return lambda t: utils.per_block_cast_to_fp8(t, use_ue8m0=True, gran_k=128)


def _per_expert_loop(b, cast):
    """The pre-batching reference: one cast per expert into dense preallocated outputs."""
    data0, sf0 = cast(b[0])
    data = b.new_empty((b.shape[0], *data0.shape), dtype=data0.dtype)
    sf = sf0.new_empty((b.shape[0], *sf0.shape))
    data[0], sf[0] = data0, sf0
    for i in range(1, b.shape[0]):
        data[i], sf[i] = cast(b[i])
    return data, sf


def _bits(t):
    return t.view(torch.uint8) if t.element_size() == 1 else t


def _assert_matches_the_loop(fmt, w, utils):
    data, sf = deepgemm._quant_weight(w, fmt, _WEIGHT_K, cacheable=False)
    ref_data, ref_sf = _per_expert_loop(w.transpose(-2, -1).contiguous(), _weight_cast(fmt, utils))
    assert data.shape == ref_data.shape and sf.shape == ref_sf.shape
    assert data.dtype == ref_data.dtype and sf.dtype == ref_sf.dtype
    assert torch.equal(_bits(data), _bits(ref_data))
    assert torch.equal(_bits(sf), _bits(ref_sf))
    assert data.stride() == ref_data.stride(), "DeepGEMM asserts the weight's expert stride is N * K"


@_CHUNKS
@pytest.mark.parametrize(("fmt", "E", "N"), _WEIGHT_SHAPES)
def test_batched_weight_cast_is_bitwise_the_per_expert_loop(monkeypatch, fmt, E, N, chunk_mb):
    """The real DeepGEMM casts, not a stand-in: a per-row stand-in cannot show that the fp8 cast scales
    128 x 128 blocks, which is what makes an unpadded flatten mix scales across experts."""
    # Inside the test: a module-level skip would also skip every guard above in the Hopper image.
    utils = pytest.importorskip("deep_gemm.utils")
    monkeypatch.setattr(deepgemm, "_deep_gemm", lambda: (None, utils))
    monkeypatch.setattr(deepgemm, "_CAST_CHUNK_MB", chunk_mb)
    torch.manual_seed(0)
    _assert_matches_the_loop(fmt, torch.randn(E, _WEIGHT_K, N, dtype=torch.bfloat16), utils)


def _stand_in_utils(cast_rows: list[int]):
    """Block-scaled fp8 and row-scaled fp4 casts with the real signatures and layouts, recording the
    row count of every call. Unlike a per-row stand-in, the fp8 scale spans 128 rows, so a flatten that
    lets a block cross two experts changes the result."""

    def per_block_cast_to_fp8(t, use_ue8m0, gran_k):
        cast_rows.append(t.shape[0])
        m, n = t.shape
        pm, pn = -(-m // gran_k) * gran_k, -(-n // gran_k) * gran_k
        padded = t.new_zeros(pm, pn)
        padded[:m, :n] = t
        sf = padded.float().abs().reshape(pm // gran_k, gran_k, pn // gran_k, gran_k).amax((1, 3))
        return (t.float() * 10).clamp(-127, 127).to(torch.int8), sf

    def per_token_cast_to_fp4(t, use_ue8m0, gran_k):
        cast_rows.append(t.shape[0])
        m, n = t.shape
        sf = t.float().abs().reshape(m, n // gran_k, gran_k).amax(-1)
        return (t[:, ::2].float() * 10).clamp(-127, 127).to(torch.int8), sf

    return types.SimpleNamespace(
        align=lambda value, alignment: -(-value // alignment) * alignment,
        per_block_cast_to_fp8=per_block_cast_to_fp8,
        per_token_cast_to_fp4=per_token_cast_to_fp4,
    )


@_CHUNKS
@pytest.mark.parametrize(("fmt", "E", "N"), _WEIGHT_SHAPES)
def test_batched_weight_cast_matches_the_loop_and_stays_in_bounded_chunks(monkeypatch, fmt, E, N, chunk_mb):
    """Runs without ``deep_gemm``: the pad, chunk and copy-back bookkeeping, and the chunk bound."""
    cast_rows: list[int] = []
    utils = _stand_in_utils(cast_rows)
    monkeypatch.setattr(deepgemm, "_deep_gemm", lambda: (None, utils))
    monkeypatch.setattr(deepgemm, "_CAST_CHUNK_MB", chunk_mb)
    torch.manual_seed(0)
    w = torch.randn(E, _WEIGHT_K, N, dtype=torch.bfloat16)

    _assert_matches_the_loop(fmt, w, utils)

    # _assert_matches_the_loop ran the batched cast first, then E reference casts of N rows each.
    batched_rows = cast_rows[:-E]
    n_cast = -(-N // 128) * 128 if fmt == "mxfp8" else N
    experts_per_call = 1 if chunk_mb == 0 else E
    assert batched_rows == [experts_per_call * n_cast] * (E // experts_per_call)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
