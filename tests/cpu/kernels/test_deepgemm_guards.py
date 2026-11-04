"""Native DeepGEMM's guards and its one decision per run: ``offs`` must cover ``T``, the flag reaches every grouped call
whatever its shape, a kernel rejection raises, the 4-bit route casts MXFP4 and a non-finite weight never reaches the
cast that hides it."""

import re
import types

import pytest
import torch
import torch.nn as nn

from halo.kernels import expert_gemm as expert_gemm_module
from halo.kernels.expert_gemm import expert_gemm
from halo.kernels.lowp import deepgemm
from halo.kernels.lowp.quantization import fake_quant

E, K, N, T = 4, 8, 16, 40
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _align(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


def _stub_resolver(monkeypatch):
    """Only ``utils.align`` is reached before the ``offs`` guard."""
    utils = types.SimpleNamespace(align=_align)
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


def _identity_cast(tensor: torch.Tensor, **_) -> tuple[torch.Tensor, torch.Tensor]:
    return tensor, tensor.new_zeros(1)


@pytest.fixture
def native(monkeypatch):
    """The flag on, over kernels that record their calls, and a spy on the simulated path's matmul (``F.grouped_mm``
    does not run on CPU). Casts are identities unless a test swaps in the real utils."""
    kernel_calls: list[dict] = []
    simulated_calls: list[torch.Tensor] = []

    def kernel(a, b, d, m_indices, **kwargs):
        kernel_calls.append({"a": a, "b": b, "m_indices": m_indices, **kwargs})
        d.zero_()

    kernels = types.SimpleNamespace(
        m_grouped_fp8_gemm_nt_contiguous=kernel, m_grouped_fp8_fp4_gemm_nt_contiguous=kernel
    )
    utils = types.SimpleNamespace(
        align=_align,
        per_token_cast_to_fp8=_identity_cast,
        per_block_cast_to_fp8=_identity_cast,
        per_token_cast_to_fp4=_identity_cast,
    )

    def simulated(mat_a, mat_b, *, offs=None):
        simulated_calls.append(mat_b)
        return mat_a.new_zeros(mat_a.shape[0], mat_b.shape[2])

    monkeypatch.setenv("HALO_DEEPGEMM_NATIVE", "1")
    monkeypatch.setattr(deepgemm, "_deep_gemm", lambda: (kernels, utils))
    monkeypatch.setattr(expert_gemm_module, "grouped_mm", simulated)
    return types.SimpleNamespace(kernels=kernels, utils=utils, kernel_calls=kernel_calls, simulated=simulated_calls)


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
    """OutOfMemoryError is a RuntimeError, so without the re-raise a full GPU would read as a shape the kernel refused."""
    with pytest.raises(torch.cuda.OutOfMemoryError):
        _forward_on_a_kernel_raising(monkeypatch, torch.cuda.OutOfMemoryError("CUDA out of memory"))


def test_a_direct_call_without_deep_gemm_raises_import_error(monkeypatch):
    """The resolver returns None when ``deep_gemm`` does not import."""
    monkeypatch.setattr(deepgemm, "_deep_gemm", lambda: None)
    with pytest.raises(ImportError, match="deep_gemm"):
        deepgemm.deepgemm_grouped_gemm(
            torch.zeros(T, K), torch.zeros(E, K, N), offs=torch.tensor([10, 20, 30, T]), quant_format="mxfp8"
        )


def test_a_direct_call_without_offs_raises(native):
    """The kernel's ``m_indices`` are built from the group ends; ``F.grouped_mm`` alone can run without them."""
    x, w, _ = _operands(8, 64, 64)
    with pytest.raises(ValueError, match="needs offs"):
        expert_gemm(x, w, quant_format="mxfp8")


@pytest.mark.parametrize(
    ("rows_per_expert", "n"),
    [(8, 64), (8, 4096), (1024, 64), (1024, 4096)],
    ids=["tiny", "wide-few-rows", "narrow-many-rows", "wide-many-rows"],
)
@pytest.mark.parametrize("quant_format", ["mxfp8", "mxfp4"])
def test_opt_in_runs_every_shape_native(native, quant_format, rows_per_expert, n):
    """A gate on tokens per expert or width flips one expert's numerics between forwards as routing moves."""
    x, w, offs = _operands(rows_per_expert, 64, n)
    out = expert_gemm(x, w, offs=offs, quant_format=quant_format)
    assert out.shape == (x.shape[0], n)
    assert len(native.kernel_calls) == 1
    assert not native.simulated


def test_the_flag_off_runs_simulated_at_a_shape_the_opt_in_would_take(native, monkeypatch):
    monkeypatch.delenv("HALO_DEEPGEMM_NATIVE")
    x, w, offs = _operands(1024, 64, 4096)
    expert_gemm(x, w, offs=offs, quant_format="mxfp8")
    assert not native.kernel_calls
    assert len(native.simulated) == 1


def test_nvfp4_stays_simulated_under_the_flag_and_says_so(native, caplog):
    """No kernel takes its e4m3 weight scales."""
    deepgemm._warn_no_native_format_once.cache_clear()
    x, w, offs = _operands(8, 64, 64)
    with caplog.at_level("WARNING", logger=deepgemm.__name__):
        for _ in range(2):
            expert_gemm(x, w, offs=offs, quant_format="nvfp4")
    assert not native.kernel_calls
    assert len(native.simulated) == 2
    assert len([r for r in caplog.records if "no kernel for nvfp4" in r.getMessage()]) == 1


def test_an_absent_package_leaves_the_opt_in_simulated_and_says_so(native, monkeypatch, caplog):
    """The Hopper image ships no ``deep_gemm``."""
    monkeypatch.setattr(deepgemm, "_deep_gemm", lambda: None)
    deepgemm._warn_absent_once.cache_clear()
    x, w, offs = _operands(8, 64, 64)
    with caplog.at_level("WARNING", logger=deepgemm.__name__):
        expert_gemm(x, w, offs=offs, quant_format="mxfp8")
    assert len(native.simulated) == 1
    assert [r for r in caplog.records if "deep_gemm package is not importable" in r.getMessage()]


@pytest.mark.parametrize("quant_format", ["mxfp8", "mxfp4"])
def test_a_kernel_rejection_raises_naming_the_shape(native, quant_format):
    """A bf16 fallback would change one expert's numerics mid-run behind a log line."""

    def reject(*args, **kwargs):
        raise RuntimeError("narrow N")

    native.kernels.m_grouped_fp8_gemm_nt_contiguous = reject
    native.kernels.m_grouped_fp8_fp4_gemm_nt_contiguous = reject
    x, w, offs = _operands(8, 64, 32)
    with pytest.raises(RuntimeError, match=rf"{quant_format} grouped GEMM on M=256 N=32 K=64 E=2: narrow N"):
        expert_gemm(x, w, offs=offs, quant_format=quant_format)
    assert not native.simulated


def test_the_4bit_route_casts_mxfp4_and_not_nvfp4(real_casts):
    """Differs from Halo's ``mxfp4`` only where an element sits exactly on an e2m1 midpoint, as DeepGEMM rounds those
    down and the hardware to even, and from ``nvfp4`` nearly everywhere."""
    experts, k, n = 2, 2880, 64
    w = _spread_weight(experts, k, n)
    x, _, offs = _operands(8, k, n)
    expert_gemm(x, w, offs=offs, quant_format="mxfp4")

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
        on_midpoint = torch.isin(scaled, torch.tensor(_E2M1_MIDPOINTS))
        differs = decoded != as_mxfp4
        assert differs.any(), "test premise: bf16 over power-of-two scales lands on midpoints"
        assert not (differs & ~on_midpoint).any(), "the native weight is not MXFP4 1x32 away from rounding ties"
        assert (decoded != as_nvfp4).float().mean() > 0.5, "the native weight reads as NVFP4"


def test_the_8bit_route_keeps_128_by_128_weight_blocks(real_casts):
    x, _, offs = _operands(8, 256, 256)
    w = _spread_weight(2, 256, 256)
    expert_gemm(x, w, offs=offs, quant_format="mxfp8")
    data, scales = real_casts.kernel_calls[0]["b"]
    assert data.dtype == torch.float8_e4m3fn
    assert scales.shape == (2, 2, 2), "one scale per 128x128 block of each expert's weight"
    assert real_casts.kernel_calls[0]["a"][1].shape == (2 * 128, 2), (
        "the activation keeps one scale per row and 128 of K"
    )


@pytest.mark.parametrize("weight_cacheable", [True, False], ids=["cached", "fsdp2-managed"])
@pytest.mark.parametrize("poison", [float("nan"), float("inf")])
def test_a_non_finite_weight_raises_before_the_mxfp4_cast(native, poison, weight_cacheable):
    x, w, offs = _operands(8, 64, 64)
    w[1, 3, 5] = poison
    with pytest.raises(ValueError, match="Non-finite expert weight"):
        expert_gemm(x, w, offs=offs, quant_format="mxfp4", weight_cacheable=weight_cacheable)
    assert not native.kernel_calls


def test_the_mxfp4_cast_itself_decodes_a_nan_block_as_zeros(real_casts):
    """Why the guard exists; the day this stops holding the guard can go."""
    w = _spread_weight(1, 64, 64)[0].t().contiguous()
    w[0, 0] = float("nan")
    packed, scales = real_casts.utils.per_token_cast_to_fp4(w, use_ue8m0=True, gran_k=32)
    decoded = real_casts.utils.cast_back_from_fp4(packed, scales, gran_k=32)
    assert decoded.isfinite().all()
    assert decoded[0, :32].abs().max() == 0.0


def test_the_fp8_route_needs_no_guard_as_its_cast_keeps_nan(real_casts):
    w = _spread_weight(2, 128, 128)
    w[0, 0, 0] = float("nan")
    x, _, offs = _operands(8, 128, 128)
    expert_gemm(x, w, offs=offs, quant_format="mxfp8")
    assert real_casts.kernel_calls[0]["b"][0].float().isnan().any()


@pytest.mark.parametrize("quant_format", deepgemm.DEEPGEMM_FORMATS)
def test_an_fsdp2_managed_weight_never_reads_the_cache(native, quant_format):
    """FSDP2 pins its unsharded param's version counter, so an entry keyed on it would serve the step-0 cast all run."""
    x, _, offs = _operands(8, 64, 64)
    w = nn.Parameter(torch.ones(2, 64, 64, dtype=torch.bfloat16))

    def weights_the_kernel_saw(weight_cacheable: bool) -> tuple:
        native.kernel_calls.clear()
        for _ in range(2):
            expert_gemm(x, w, offs=offs, quant_format=quant_format, weight_cacheable=weight_cacheable)
        return tuple(call["b"] for call in native.kernel_calls)

    first, second = weights_the_kernel_saw(True)
    assert first is second, "test premise: a cacheable weight reuses its cast"
    first, second = weights_the_kernel_saw(False)
    assert first is not second


@pytest.mark.parametrize("fmt", deepgemm.DEEPGEMM_FORMATS)
def test_the_weight_cache_requantizes_after_the_write_and_only_then(real_casts, fmt):
    k = 256
    w = nn.Parameter(_spread_weight(2, k, 64))
    first = deepgemm._quant_weight(w, fmt, k)
    assert deepgemm._quant_weight(w, fmt, k) is first, "a second call in the step must reuse the cast"

    with torch.no_grad():
        w.mul_(2.0).add_(0.01)
    second = deepgemm._quant_weight(w, fmt, k)
    fresh = deepgemm._quant_weight(w, fmt, k, cacheable=False)
    assert second is not first, "the version bump must miss the cache"
    assert fresh is not second, "cacheable=False must not read the cache"
    for served, recast in zip(second, fresh, strict=True):
        assert torch.equal(served, recast)
    assert not torch.equal(second[1], first[1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
