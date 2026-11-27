"""Native DeepGEMM's guards and its one decision per run: ``offs`` must cover ``T``, ``native`` reaches every grouped
call whatever its shape, a kernel rejection raises, the 4-bit route casts MXFP4, a non-finite weight never reaches the
cast that hides it, and the FP8 backward runs dgrad on the forward's weight and wgrad on 128-token tiles."""

import contextlib
import re
import sys
import types
import weakref

import pytest
import torch
import torch.nn as nn

from halo.kernels import expert_gemm as expert_gemm_module
from halo.kernels.expert_gemm import expert_gemm
from halo.kernels.lowp import deepgemm
from halo.kernels.lowp.deepgemm_cast_triton import GRADIENT_AMAX_FLOOR, REFERENCE_AMAX_FLOOR
from halo.kernels.lowp.quantization import fake_quant
from halo.kernels.lowp.spec import LOWP_RECIPES, LowpSpec

E, K, N, T = 4, 8, 16, 40
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _native(quant_format: str, backward: str = "bf16") -> LowpSpec:
    precision = next(p for p, recipe in LOWP_RECIPES.items() if recipe.quant_format == quant_format)
    return LowpSpec(precision=precision, backend="native", backward_precision=backward)


def _align(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


def _layout(ends: list[int]):
    return deepgemm._row_layout(torch.tensor(ends, dtype=torch.int32), T, E)


def _gathered(x: torch.Tensor, src: torch.Tensor, width: int) -> torch.Tensor:
    """The operand the Triton casts read through their row map, which the GPU test pins them bitwise against."""
    out = x.new_zeros(src.numel(), width)
    live = src >= 0
    out[live, : x.shape[1]] = x[src[live].long()]
    return out


def _identity_cast(tensor: torch.Tensor, **_) -> tuple[torch.Tensor, torch.Tensor]:
    return tensor, tensor.new_zeros(1)


def _block_cast(tensor: torch.Tensor, *, gran_k: int, **_) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepGEMM's FP8 weight cast in miniature: one amax scale per ``gran_k`` x ``gran_k`` block, so a flatten that
    lets a block span two experts changes the result."""
    m, n = tensor.shape
    rows, cols = _align(m, gran_k), _align(n, gran_k)
    padded = tensor.new_zeros(rows, cols)
    padded[:m, :n] = tensor
    scales = padded.float().abs().reshape(rows // gran_k, gran_k, cols // gran_k, gran_k).amax((1, 3))
    return (tensor.float() * 10).clamp(-127, 127).to(torch.int8), scales


def _row_cast(tensor: torch.Tensor, *, gran_k: int, **_) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepGEMM's MXFP4 weight cast in miniature: one amax scale per row and ``gran_k`` of K, two values per byte."""
    scales = tensor.float().abs().reshape(tensor.shape[0], -1, gran_k).amax(-1)
    return (tensor[:, ::2].float() * 10).clamp(-127, 127).to(torch.int8), scales


@pytest.fixture
def native(monkeypatch):
    """The flag on, over kernels that record their calls, and a spy on the simulated path's matmul (``F.grouped_mm``
    does not run on CPU). The activation cast is an identity and the weight casts are stand-ins with DeepGEMM's scale
    layouts, unless a test swaps in the real utils."""
    kernel_calls: list[dict] = []
    wgrad_calls: list[dict] = []
    simulated_calls: list[torch.Tensor] = []
    cast_floors: list[tuple[str, float]] = []

    def kernel(a, b, d, m_indices, **kwargs):
        kernel_calls.append({"a": a, "b": b, "m_indices": m_indices, **kwargs})
        d.zero_()

    def wgrad(a, b, d, ks, ks_dev, c, **kwargs):
        wgrad_calls.append({"a": a, "b": b, "d": d, "ks": ks, "ks_dev": ks_dev, "c": c, **kwargs})

    kernels = types.SimpleNamespace(
        m_grouped_fp8_gemm_nt_contiguous=kernel,
        m_grouped_fp8_fp4_gemm_nt_contiguous=kernel,
        m_grouped_fp8_gemm_nn_contiguous=kernel,
        k_grouped_fp8_gemm_tn_contiguous=wgrad,
    )
    utils = types.SimpleNamespace(
        set_mk_alignment_for_contiguous_layout=lambda align: None,
        per_token_cast_to_fp8=_identity_cast,
        per_channel_cast_to_fp8=_identity_cast,
        per_block_cast_to_fp8=_block_cast,
        per_token_cast_to_fp4=_row_cast,
    )

    def simulated(mat_a, mat_b, *, offs=None):
        simulated_calls.append(mat_b)
        return mat_a.new_zeros(mat_a.shape[0], mat_b.shape[2])

    state = types.SimpleNamespace(
        kernels=kernels,
        utils=utils,
        kernel_calls=kernel_calls,
        wgrad_calls=wgrad_calls,
        simulated=simulated_calls,
        cast_floors=cast_floors,
    )

    def cast_rows(x, src, width, amax_floor=REFERENCE_AMAX_FLOOR):
        cast_floors.append(("rows", amax_floor))
        return state.utils.per_token_cast_to_fp8(_gathered(x, src, width), use_ue8m0=True, gran_k=128)

    def cast_tiles(x, src, amax_floor=REFERENCE_AMAX_FLOOR):
        cast_floors.append(("tiles", amax_floor))
        return state.utils.per_channel_cast_to_fp8(_gathered(x, src, x.shape[1]), use_ue8m0=True, gran_k=128)

    monkeypatch.setattr(deepgemm, "_deep_gemm", lambda: (kernels, utils))
    monkeypatch.setattr(expert_gemm_module, "grouped_mm", simulated)
    monkeypatch.setattr(deepgemm, "cast_rows_to_fp8", cast_rows)
    monkeypatch.setattr(deepgemm, "cast_tiles_to_fp8", cast_tiles)
    return state


@pytest.fixture
def real_casts(native, monkeypatch):
    """``native`` over DeepGEMM's own Python casts, which run on CPU; its wheel ships in the Blackwell image only."""
    native.utils = pytest.importorskip("deep_gemm.utils")
    monkeypatch.setattr(deepgemm, "_deep_gemm", lambda: (native.kernels, native.utils))
    return native


def _operands(
    rows_per_expert: int, k: int, n: int, experts: int = 2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.zeros(experts * rows_per_expert, k, dtype=torch.bfloat16)
    w = torch.zeros(experts, k, n, dtype=torch.bfloat16)
    offs = torch.arange(1, experts + 1, dtype=torch.int32) * rows_per_expert
    return x, w, offs


def _spread_weight(experts: int, k: int, n: int) -> torch.Tensor:
    """Blocks of widely different magnitude, the case block scaling exists for."""
    torch.manual_seed(0)
    spread = torch.exp(torch.empty(experts, k // 32, n).uniform_(-3.0, 3.0)).repeat_interleave(32, dim=1)
    return (torch.randn(experts, k, n) * 0.02 * spread).to(torch.bfloat16)


def test_offs_that_stops_short_of_t_is_rejected():
    """The tail rows of ``out`` are never written."""
    with pytest.raises(ValueError, match="ending at T="):
        _layout([10, 20, 30, T - 1])


def test_offs_overrunning_t_is_rejected():
    """A cumulative count past ``T`` slices the input out of range."""
    with pytest.raises(ValueError, match="ending at T="):
        _layout([10, 20, 30, T + 1])


@pytest.mark.parametrize(
    "ends",
    [
        [10, 20, T],
        [10, 5, 20, T],
        [-1, 10, 20, T],
    ],
    ids=["short", "backwards", "negative"],
)
def test_length_ordering_and_negative_offsets_still_rejected(ends):
    """The guard raises one message naming every invariant, so the echoed offs pin which input was rejected."""
    with pytest.raises(ValueError, match=re.escape(f"got {ends}")):
        _layout(ends)


def test_the_layout_puts_each_expert_at_the_start_of_its_128_row_segment():
    """An empty expert keeps a zero-length segment, so the next one starts where it would have."""
    layout = _layout([10, 10, 30, T])
    assert layout.segments == [128, 0, 128, 128] and layout.rows == 384
    starts = [0] * 10 + [128] * 20 + [256] * 10
    within = [*range(10), *range(20), *range(10)]
    assert layout.dst.tolist() == [s + i for s, i in zip(starts, within, strict=True)]
    expected = torch.full((384,), -1, dtype=torch.int32)
    expected[:10], expected[128:148], expected[256:266] = 0, 2, 3
    assert torch.equal(layout.m_indices, expected)
    assert layout.segments_dev.tolist() == layout.segments


def _forward_on_a_kernel_raising(monkeypatch, error: Exception):
    def kernel(*args, **kwargs):
        raise error

    kernels = types.SimpleNamespace(m_grouped_fp8_gemm_nn_contiguous=kernel)
    monkeypatch.setattr(deepgemm, "_deep_gemm", lambda: (kernels, None))
    monkeypatch.setattr(deepgemm, "_quant_weight", lambda *args, **kwargs: None)
    x = torch.zeros(T, K)
    return deepgemm._deepgemm_forward(x, torch.zeros(E, K, N), _layout([10, 20, 30, T]), (x, None), "mxfp8", False)


def test_an_oom_in_the_native_kernel_is_not_taken_for_a_rejected_shape(monkeypatch):
    """OutOfMemoryError is a RuntimeError, so without the re-raise a full GPU would read as a shape the kernel refused."""
    with pytest.raises(torch.cuda.OutOfMemoryError):
        _forward_on_a_kernel_raising(monkeypatch, torch.cuda.OutOfMemoryError("CUDA out of memory"))


def test_a_direct_call_without_deep_gemm_raises_import_error(monkeypatch):
    """The resolver returns None when ``deep_gemm`` does not import."""
    monkeypatch.setattr(deepgemm, "_deep_gemm", lambda: None)
    with pytest.raises(ImportError, match="deep_gemm"):
        deepgemm.deepgemm_grouped_gemm(
            torch.zeros(T, K), torch.zeros(E, K, N), offs=torch.tensor([10, 20, 30, T]), lowp=_native("mxfp8")
        )


def test_only_an_absent_package_reads_unavailable_and_a_broken_install_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "deep_gemm", types.ModuleType("deep_gemm"))  # imports, and has no ``utils``
    deepgemm._deep_gemm.cache_clear()
    try:
        with pytest.raises(ImportError, match="utils"):
            deepgemm.deepgemm_available()
        monkeypatch.setitem(sys.modules, "deep_gemm", None)  # ``import deep_gemm`` raises ModuleNotFoundError
        assert not deepgemm.deepgemm_available()
    finally:
        deepgemm._deep_gemm.cache_clear()


def test_a_direct_call_without_offs_raises(native):
    """The kernel's ``m_indices`` are built from the group ends; ``F.grouped_mm`` alone can run without them."""
    x, w, _ = _operands(8, 64, 64)
    with pytest.raises(ValueError, match="needs offs"):
        expert_gemm(x, w, lowp=_native("mxfp8"))


@pytest.mark.parametrize(
    ("rows_per_expert", "n"),
    [(8, 64), (1024, 4096)],
    ids=["tiny", "wide-many-rows"],
)
@pytest.mark.parametrize("quant_format", ["mxfp8", "mxfp4"])
def test_native_runs_every_shape_native(native, quant_format, rows_per_expert, n):
    """A gate on tokens per expert or width flips one expert's numerics between forwards as routing moves."""
    x, w, offs = _operands(rows_per_expert, 64, n)
    out = expert_gemm(x, w, offs=offs, lowp=_native(quant_format))
    assert out.shape == (x.shape[0], n)
    assert len(native.kernel_calls) == 1
    assert not native.simulated


def test_simulated_runs_at_a_shape_native_would_take(native):
    x, w, offs = _operands(1024, 64, 4096)
    expert_gemm(x, w, offs=offs, lowp=LowpSpec(precision="fp8"))
    assert not native.kernel_calls
    assert len(native.simulated) == 1


def test_an_absent_package_raises_rather_than_run_simulated(native, monkeypatch):
    """The Hopper image ships no ``deep_gemm``."""
    monkeypatch.setattr(deepgemm, "_deep_gemm", lambda: None)
    x, w, offs = _operands(8, 64, 64)
    with pytest.raises(ImportError, match="deep_gemm"):
        expert_gemm(x, w, offs=offs, lowp=_native("mxfp8"))
    assert not native.simulated


@pytest.mark.parametrize("quant_format", ["mxfp8", "mxfp4"])
def test_a_kernel_rejection_raises_naming_the_shape(native, quant_format):
    """A bf16 fallback would change one expert's numerics mid-run behind a log line."""

    def reject(*args, **kwargs):
        raise RuntimeError("narrow N")

    native.kernels.m_grouped_fp8_gemm_nn_contiguous = reject
    native.kernels.m_grouped_fp8_fp4_gemm_nt_contiguous = reject
    x, w, offs = _operands(8, 64, 32)
    with pytest.raises(RuntimeError, match=rf"{quant_format} grouped GEMM on M=256 N=32 K=64 E=2: narrow N"):
        expert_gemm(x, w, offs=offs, lowp=_native(quant_format))
    assert not native.simulated


def test_the_4bit_route_casts_mxfp4_and_not_nvfp4(real_casts):
    """Equals Halo's ``mxfp4`` element for element, e2m1 midpoints included (both round ties to even), on blocks
    whose largest element is at least 1e-4: DeepGEMM floors a block's scale there and ``fake_quant`` at float32's
    tiny. It differs from ``nvfp4`` nearly everywhere."""
    experts, k, n = 2, 2880, 64
    w = _spread_weight(experts, k, n)
    x, _, offs = _operands(8, k, n)
    expert_gemm(x, w, offs=offs, lowp=_native("mxfp4"))

    assert len(real_casts.kernel_calls) == 1
    call = real_casts.kernel_calls[0]
    assert call["recipe_b"] == (1, 32)
    packed, scales = call["b"]
    assert packed.shape == (experts, n, 3072 // 2), "K zero-pads to 3072, two e2m1 per byte"
    assert call["a"][1].shape == (experts * 128, 3072 // 128), "the activation keeps one scale per row and 128 of K"

    for expert in range(experts):
        decoded = real_casts.utils.cast_back_from_fp4(packed[expert], scales[expert], gran_k=32)[:, :k]
        master = w[expert].t().float()
        as_mxfp4 = fake_quant(w[expert].t(), "mxfp4", axis=-1).float()
        as_nvfp4 = fake_quant(w[expert].t(), "nvfp4", axis=-1).float()

        scaled = master.abs() / scales[expert][:, torch.arange(k) // 32]
        assert torch.isin(scaled, torch.tensor(_E2M1_MIDPOINTS)).any(), "premise: bf16 over pow2 scales hits midpoints"
        assert torch.equal(decoded, as_mxfp4), "the native weight is not Halo's MXFP4 1x32, ties to even"
        assert (decoded != as_nvfp4).float().mean() > 0.5, "the native weight reads as NVFP4"


def test_the_8bit_route_keeps_128_by_128_weight_blocks(real_casts):
    x, _, offs = _operands(8, 256, 256)
    w = _spread_weight(2, 256, 256)
    expert_gemm(x, w, offs=offs, lowp=_native("mxfp8"))
    data, scales = real_casts.kernel_calls[0]["b"]
    assert data.dtype == torch.float8_e4m3fn
    assert scales.shape == (2, 2, 2), "one scale per 128x128 block of each expert's weight"
    assert real_casts.kernel_calls[0]["a"][1].shape == (2 * 128, 2), (
        "the activation keeps one scale per row and 128 of K"
    )


@pytest.mark.parametrize(
    ("poison", "weight_cacheable"),
    [
        pytest.param(float("nan"), True, id="nan-cached"),
        pytest.param(float("inf"), False, id="inf-fsdp2-managed"),
    ],
)
def test_a_non_finite_weight_raises_before_the_mxfp4_cast(native, poison, weight_cacheable):
    x, w, offs = _operands(8, 64, 64)
    w[1, 3, 5] = poison
    with pytest.raises(ValueError, match="Non-finite expert weight"):
        expert_gemm(x, w, offs=offs, lowp=_native("mxfp4"), weight_cacheable=weight_cacheable)
    assert not native.kernel_calls


def test_the_fp8_route_needs_no_guard_as_its_cast_keeps_nan(real_casts):
    w = _spread_weight(2, 128, 128)
    w[0, 0, 0] = float("nan")
    x, _, offs = _operands(8, 128, 128)
    expert_gemm(x, w, offs=offs, lowp=_native("mxfp8"))
    assert real_casts.kernel_calls[0]["b"][0].float().isnan().any()


@pytest.mark.parametrize("quant_format", deepgemm.DEEPGEMM_FORMATS)
def test_an_fsdp2_managed_weight_never_reads_the_cache(native, quant_format):
    """FSDP2 pins its unsharded param's version counter, so an entry keyed on it would serve the step-0 cast all run."""
    x, _, offs = _operands(8, 64, 64)
    w = nn.Parameter(torch.ones(2, 64, 64, dtype=torch.bfloat16))

    def weights_the_kernel_saw(weight_cacheable: bool) -> tuple:
        native.kernel_calls.clear()
        for _ in range(2):
            expert_gemm(x, w, offs=offs, lowp=_native(quant_format), weight_cacheable=weight_cacheable)
        return tuple(call["b"] for call in native.kernel_calls)

    first, second = weights_the_kernel_saw(True)
    assert first is second, "test premise: a cacheable weight reuses its cast"
    first, second = weights_the_kernel_saw(False)
    assert first is not second


@pytest.mark.parametrize("fmt", deepgemm.DEEPGEMM_FORMATS)
def test_the_weight_cache_requantizes_after_the_write_and_only_then(real_casts, fmt):
    k = 256
    w = nn.Parameter(_spread_weight(2, k, 64))
    first = deepgemm._quant_weight(w, fmt)
    assert deepgemm._quant_weight(w, fmt) is first, "a second call in the step must reuse the cast"

    with torch.no_grad():
        w.mul_(2.0).add_(0.01)
    second = deepgemm._quant_weight(w, fmt)
    fresh = deepgemm._quant_weight(w, fmt, cacheable=False)
    assert second is not first, "the version bump must miss the cache"
    assert fresh is not second, "cacheable=False must not read the cache"
    for served, recast in zip(second, fresh, strict=True):
        assert torch.equal(served, recast)
    assert not torch.equal(second[1], first[1])


_WEIGHT_K = 256  # a multiple of the fp8 block (128) and of the fp4 packing (256)
# N aligned and not to the 128-row block, and E=1, where a slice of the padded layout would keep the padded stride.
_WEIGHT_CASES = pytest.mark.parametrize(
    ("fmt", "experts", "n"),
    [("mxfp8", 4, 256), ("mxfp8", 4, 100), ("mxfp8", 1, 100), ("mxfp4", 4, 100), ("mxfp4", 1, 100)],
)
_CHUNK_BYTES = pytest.mark.parametrize("chunk_bytes", [256 << 20, 0], ids=["one-chunk", "chunk-per-expert"])


def _per_expert_loop(w: torch.Tensor, fmt: str, utils) -> tuple[torch.Tensor, torch.Tensor]:
    """One cast per expert into dense preallocated outputs, which the batched cast must reproduce."""
    cast = (
        (lambda t: utils.per_token_cast_to_fp4(t, use_ue8m0=True, gran_k=32))
        if fmt == "mxfp4"
        else (lambda t: utils.per_block_cast_to_fp8(t, use_ue8m0=True, gran_k=128))
    )
    # The FP8 operand keeps the weight's [E, K, N] (the GEMMs read it N-major), the MXFP4 one is [E, N, K].
    b = w.contiguous() if fmt == "mxfp8" else w.transpose(-2, -1).contiguous()
    data0, sf0 = cast(b[0])
    data = b.new_empty((b.shape[0], *data0.shape), dtype=data0.dtype)
    sf = sf0.new_empty((b.shape[0], *sf0.shape))
    data[0], sf[0] = data0, sf0
    for i in range(1, b.shape[0]):
        data[i], sf[i] = cast(b[i])
    return data, sf


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.uint8) if tensor.element_size() == 1 else tensor


def _assert_same_operand(batched, reference) -> None:
    for got, expected in zip(batched, reference, strict=True):
        assert got.shape == expected.shape and got.dtype == expected.dtype
        assert torch.equal(_bits(got), _bits(expected))
    assert batched[0].stride() == reference[0].stride(), "DeepGEMM asserts the weight's expert stride is N * K"


@_CHUNK_BYTES
@_WEIGHT_CASES
def test_the_batched_weight_cast_is_bitwise_the_per_expert_loop(real_casts, monkeypatch, fmt, experts, n, chunk_bytes):
    """Over DeepGEMM's own casts, as a per-row stand-in cannot show that the FP8 cast scales 128 x 128 blocks, which an
    unpadded flatten mixes across experts whenever N is not a multiple of 128."""
    monkeypatch.setattr(deepgemm, "_CAST_CHUNK_BYTES", chunk_bytes)
    torch.manual_seed(0)
    w = torch.randn(experts, _WEIGHT_K, n, dtype=torch.bfloat16)
    _assert_same_operand(deepgemm._quant_weight(w, fmt, cacheable=False), _per_expert_loop(w, fmt, real_casts.utils))


def _recording(cast, rows: list[int]):
    def recorded(tensor, **kwargs):
        rows.append(tensor.shape[0])
        return cast(tensor, **kwargs)

    return recorded


@_CHUNK_BYTES
@_WEIGHT_CASES
def test_the_batched_cast_pads_chunks_and_copies_back_without_deep_gemm(
    native, monkeypatch, fmt, experts, n, chunk_bytes
):
    """The pad, chunk and copy-back bookkeeping over scale-faithful stand-ins, on the Hopper image too."""
    rows: list[int] = []
    for name in ("per_block_cast_to_fp8", "per_token_cast_to_fp4"):
        monkeypatch.setattr(native.utils, name, _recording(getattr(native.utils, name), rows))
    monkeypatch.setattr(deepgemm, "_CAST_CHUNK_BYTES", chunk_bytes)
    torch.manual_seed(0)
    w = torch.randn(experts, _WEIGHT_K, n, dtype=torch.bfloat16)

    batched = deepgemm._quant_weight(w, fmt, cacheable=False)
    batched_rows, rows[:] = list(rows), []
    _assert_same_operand(batched, _per_expert_loop(w, fmt, native.utils))

    rows_per_expert = _align(_WEIGHT_K, 128) if fmt == "mxfp8" else n
    experts_per_call = 1 if chunk_bytes == 0 else experts
    assert batched_rows == [experts_per_call * rows_per_expert] * (experts // experts_per_call)


@pytest.mark.parametrize("fmt", deepgemm.DEEPGEMM_FORMATS)
def test_a_chunk_is_copied_out_and_freed_before_the_next_is_cast(native, monkeypatch, fmt):
    """Holding every chunk beside the full outputs would peak at two copies of the quantized weight."""
    monkeypatch.setattr(deepgemm, "_CAST_CHUNK_BYTES", 0)
    casts: list[weakref.ref] = []
    live_at_each_cast: list[int] = []

    def recording(cast):
        def recorded(tensor, **kwargs):
            live_at_each_cast.append(sum(cast_out() is not None for cast_out in casts))
            data, scales = cast(tensor, **kwargs)
            casts.append(weakref.ref(data))
            return data, scales

        return recorded

    for name in ("per_block_cast_to_fp8", "per_token_cast_to_fp4"):
        monkeypatch.setattr(native.utils, name, recording(getattr(native.utils, name)))
    deepgemm._quant_weight(torch.randn(4, _WEIGHT_K, 100, dtype=torch.bfloat16), fmt, cacheable=False)
    assert live_at_each_cast == [0, 0, 0, 0]


def _fp8_backward(native, x, w, offs, quant_format="mxfp8"):
    x.requires_grad_(True)
    w.requires_grad_(True)
    expert_gemm(x, w, offs=offs, lowp=_native(quant_format, "fp8")).sum().backward()


def test_the_fp8_dgrad_runs_on_the_forward_weight_operand(native):
    """The backward takes the forward's cast from the per-step cache: a second cast of the weight per backward would
    cost what the forward's costs."""
    x, w, offs = _operands(8, 128, 256)
    _fp8_backward(native, x, w, offs)
    forward, dgrad = native.kernel_calls
    assert [t.data_ptr() for t in dgrad["b"]] == [t.data_ptr() for t in forward["b"]]
    assert torch.equal(dgrad["m_indices"], forward["m_indices"])
    assert dgrad["a"][0].shape == (256, 256), "the gradient enters in the forward's padded row layout"
    assert x.grad.shape == x.shape and w.grad.shape == w.shape and w.grad.dtype == w.dtype


@pytest.mark.parametrize(
    ("weight_cacheable", "hooks", "casts"),
    [(True, False, 1), (False, False, 2), (True, True, 1)],
    ids=["cached", "fsdp2-managed", "cached-under-saved-tensor-hooks"],
)
def test_an_uncached_weight_is_cast_again_for_the_dgrad_and_never_held_in_the_graph(
    native, weight_cacheable, hooks, casts
):
    """An FP8 copy saved for the backward would hold half a bf16 bank per projection for as long as the graph lives.
    Saved-tensor hooks (a non-reentrant checkpoint) hand the backward another object for the weight, which must not
    read as an uncached one."""
    cast, weight_casts = native.utils.per_block_cast_to_fp8, []
    native.utils.per_block_cast_to_fp8 = lambda t, **kwargs: weight_casts.append(t.shape) or cast(t, **kwargs)
    x, w, offs = _operands(8, 128, 256)
    x.requires_grad_(True)
    w.requires_grad_(True)
    with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t) if hooks else contextlib.nullcontext():
        out = expert_gemm(x, w, offs=offs, lowp=_native("mxfp8", "fp8"), weight_cacheable=weight_cacheable)
    assert {t.dtype for t in out.grad_fn.saved_tensors} <= {x.dtype, w.dtype, offs.dtype}, "an FP8 operand is saved"
    assert len(out.grad_fn.saved_tensors) == 3
    assert (out.grad_fn.w_fp8 is None) == (not weight_cacheable), "only a cached cast rides on the node"
    out.sum().backward()
    assert len(weight_casts) == casts


def test_the_gradient_casts_drop_the_reference_floor_and_the_activation_casts_keep_it(native):
    """A gradient group's largest element sits far under the reference's 1e-4 floor, which would scale it into fp8's
    subnormals; the activation casts stay the reference's."""
    x, w, offs = _operands(8, 128, 256)
    _fp8_backward(native, x, w, offs)
    assert native.cast_floors == [
        ("rows", REFERENCE_AMAX_FLOOR),  # the forward's activation
        ("tiles", REFERENCE_AMAX_FLOOR),  # the wgrad's activation
        ("rows", GRADIENT_AMAX_FLOOR),  # the dgrad's gradient
        ("tiles", GRADIENT_AMAX_FLOOR),  # the wgrad's gradient
    ]
    assert GRADIENT_AMAX_FLOOR == 0.0 < REFERENCE_AMAX_FLOOR


def test_an_activations_fp8_casts_are_dropped_after_their_last_reader(native):
    """Gate and up read one activation: each cast runs once, only the memo holds the forward's rows, and the tiles
    do not outlive the second backward."""
    x, w, offs = _operands(8, 128, 256)
    banks = [w.clone().requires_grad_(True) for _ in range(2)]
    gate_out, up_out = (expert_gemm(x, bank, offs=offs, lowp=_native("mxfp8", "fp8")) for bank in banks)
    shared = gate_out.grad_fn.tiles
    assert up_out.grad_fn.tiles is shared and shared.tiles is None
    gate_out.sum().backward()
    assert deepgemm._row_casts._value is None and shared.tiles is not None
    up_out.sum().backward()
    assert shared.tiles is None
    assert [kind for kind, floor in native.cast_floors if floor == REFERENCE_AMAX_FLOOR] == ["rows", "tiles"]


def test_no_node_holds_the_forwards_fp8_activation_rows(native):
    """A non-reentrant checkpoint never frees a node's attributes, so the rows live in the one-entry memo alone:
    the next activation's cast replaces them while the first graph is still alive."""
    rows = []
    cast = native.utils.per_token_cast_to_fp8

    def cast_rows(tensor, **kwargs):
        data = tensor.clone()
        rows.append(weakref.ref(data))
        return data, cast(tensor, **kwargs)[1]

    native.utils.per_token_cast_to_fp8 = cast_rows
    _, w, offs = _operands(8, 128, 256)
    w.requires_grad_(True)
    outs = [
        expert_gemm(torch.full((16, 128), fill, dtype=torch.bfloat16), w, offs=offs, lowp=_native("mxfp8", "fp8"))
        for fill in (1.0, 2.0)
    ]
    native.kernel_calls.clear()  # the recording kernel holds its operands
    assert len(rows) == 2 and rows[0]() is None and rows[1]() is not None
    assert all(out.grad_fn is not None for out in outs)


def test_a_second_backward_over_a_retained_graph_drops_the_tiles_again(native):
    x, w, offs = _operands(8, 128, 256)
    banks = [w.clone().requires_grad_(True) for _ in range(2)]
    gate_out, up_out = (expert_gemm(x, bank, offs=offs, lowp=_native("mxfp8", "fp8")) for bank in banks)
    loss = gate_out.sum() + up_out.sum()
    loss.backward(retain_graph=True)
    loss.backward()
    assert gate_out.grad_fn.tiles.tiles is None


def test_an_activation_wider_than_the_weights_contraction_is_refused(native):
    x, w, offs = _operands(8, 256, 128)
    with pytest.raises(ValueError, match="contracts over K=128"):
        deepgemm.deepgemm_grouped_gemm(x, w[:, :128], offs=offs, lowp=_native("mxfp8"))


def test_the_memo_derives_once_per_contents_and_pins_no_graph():
    derived = []
    memo = deepgemm._Memo(lambda t, width: derived.append(width) or t.sum())
    x = torch.ones(4, requires_grad=True) * 2
    memo(x, 8), memo(x, 8)
    assert derived == [8]
    assert memo._held[0].grad_fn is None and memo._held[0].data_ptr() == x.data_ptr(), "the storage, not the graph"
    memo(x, 16)
    with torch.no_grad():
        x.add_(1)
    memo(x, 16)
    assert derived == [8, 16, 16], "another extra or an in-place write derives again"


def test_the_memo_derives_afresh_for_an_inference_tensor():
    """An inference tensor has no version counter to key on, and reading one raises."""
    derived = []
    memo = deepgemm._Memo(lambda t: derived.append(1) or t.sum())
    with torch.inference_mode():
        x = torch.ones(4)
        memo(x), memo(x)
    assert len(derived) == 2 and memo._held is None


def test_the_fp8_wgrad_takes_128_experts_per_call_on_whole_token_tiles(native):
    """The kernel's scale transform asserts at most 128 groups, and each group's tokens must fill whole 128-row tiles."""
    experts = 130
    x, w, offs = _operands(3, 64, 32, experts=experts)
    _fp8_backward(native, x, w, offs)
    first, second = native.wgrad_calls
    assert (first["ks"], second["ks"]) == ([128] * 128, [128] * 2)
    assert (first["a"][0].shape[0], second["a"][0].shape[0]) == (128 * 128, 2 * 128)
    assert first["d"].shape == (128, 64, 32) and first["d"].dtype == w.dtype, "written in the weight's own layout"
    assert first["c"] is None and second["c"] is None, (
        "the kernel writes its output whole, with no zero-filled accumulator"
    )
    assert first["recipe"] == (1, 1, 128)
    assert w.grad.shape == (experts, 64, 32)


def test_the_mxfp4_dgrad_runs_on_an_fp8_block_cast_kept_beside_the_forward_cast(native):
    """FP8 x FP4 issues at the FP8 rate, so an FP8 weight costs the dgrad nothing over a transposed FP4 cast and is the
    finer of the two. Each format keeps its own cache entry, or every microbatch would evict the other's."""
    x, w, offs = _operands(8, 256, 256)
    w = nn.Parameter(_spread_weight(2, 256, 256))
    operands = []
    for _ in range(2):
        native.kernel_calls.clear()
        _fp8_backward(native, x, w, offs, quant_format="mxfp4")
        forward, dgrad = native.kernel_calls
        assert forward["recipe_b"] == (1, 32) and forward["b"][0].shape == (2, 256, 128), "MXFP4, two e2m1 per byte"
        assert dgrad["b"][0].shape == (2, 256, 256) and dgrad["b"][1].shape == (2, 2, 2), "FP8 in 128x128 blocks"
        operands.append((forward["b"], dgrad["b"]))
    assert operands[1][0] is operands[0][0] and operands[1][1] is operands[0][1], "the second microbatch recast"
    assert w.grad.shape == w.shape


def test_a_bf16_backward_holds_no_fp8_operand_in_the_graph(native):
    x, w, offs = _operands(8, 128, 256)
    w.requires_grad_(True)
    node = expert_gemm(x, w, offs=offs, lowp=_native("mxfp8")).grad_fn
    assert node.tiles is None and node.w_fp8 is None


def test_a_format_no_kernel_takes_is_refused_at_the_entry(native):
    """A direct caller's spec has not been validated: nvfp4 would run as FP8 on the wrong weight layout."""
    x, w, offs = _operands(8, 128, 256)
    with pytest.raises(ValueError, match="deepgemm_grouped_gemm takes"):
        deepgemm.deepgemm_grouped_gemm(x, w, offs=offs, lowp=LowpSpec(precision="fp4", backend="native"))
    assert not native.kernel_calls


def test_the_non_finite_flag_raises_once_at_its_one_read_and_drops_the_cached_casts(monkeypatch):
    """A caller that catches the raise must not find the non-finite weight's cast in the cache at its next forward."""
    monkeypatch.setitem(deepgemm._weight_caches, "mxfp4", deepgemm.WeightVersionCache())
    flag = deepgemm._nonfinite_weight.setdefault(torch.device("cpu"), torch.zeros(1, dtype=torch.int32))
    weight = nn.Parameter(torch.zeros(1, 32, 32))
    deepgemm._weight_caches["mxfp4"].put(weight, weight._version, object())
    deepgemm.verify_finite_mxfp4_weights()
    assert len(deepgemm._weight_caches["mxfp4"]) == 1
    flag.fill_(1)
    with pytest.raises(ValueError, match="Non-finite expert weight bad: "):
        deepgemm.verify_finite_mxfp4_weights([("bad", torch.tensor(float("inf"))), ("fine", torch.zeros(1))])
    assert len(deepgemm._weight_caches["mxfp4"]) == 0
    deepgemm.verify_finite_mxfp4_weights()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
