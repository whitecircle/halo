"""The ``halo/kernels/grouped_mm.py`` autograd wrapper, with a CPU reference for the CUDA-only ``F.grouped_mm``.
The reference enforces the kernel's matching-dtype and int32-offsets contracts, so a leak fails as on device."""

import pytest
import torch
import torch.nn.functional

from halo.kernels.grouped_mm import grouped_mm


def _reference_grouped_mm(mat_a, mat_b, offs=None):
    """Mimics the kernel's contracts."""
    if mat_a.dtype != mat_b.dtype:
        raise RuntimeError(f"grouped_mm requires matching operand dtypes, got {mat_a.dtype} vs {mat_b.dtype}")
    assert offs is not None and offs.dtype == torch.int32
    ends = offs.tolist()
    starts = [0] + ends[:-1]
    spans = list(zip(starts, ends, strict=True))
    if mat_a.dim() == 2 and mat_b.dim() == 3:
        return torch.cat([mat_a[s:e] @ mat_b[g] for g, (s, e) in enumerate(spans)], dim=0)
    if mat_a.dim() == 2 and mat_b.dim() == 2:
        # Offsets segment the shared dim (a's dim-1, b's dim-0) → stacked per-group [M, N].
        return torch.stack([mat_a[:, s:e] @ mat_b[s:e] for s, e in spans], dim=0)
    raise AssertionError(f"unexpected reference shapes: {mat_a.shape} x {mat_b.shape}")


@pytest.fixture(autouse=True)
def _patch_grouped_mm(monkeypatch):
    monkeypatch.setattr(torch.nn.functional, "grouped_mm", _reference_grouped_mm)


def _inputs():
    torch.manual_seed(0)
    mat_a = torch.randn(6, 8, dtype=torch.bfloat16, requires_grad=True)
    mat_b = torch.randn(2, 8, 4, dtype=torch.bfloat16, requires_grad=True)
    offs = torch.tensor([3, 6], dtype=torch.int64)  # int64 on purpose: forward must normalize to int32.
    return mat_a, mat_b, offs


def test_int64_offsets_are_normalized_and_grads_keep_input_dtype():
    mat_a, mat_b, offs = _inputs()
    out = grouped_mm(mat_a, mat_b, offs=offs)
    assert out.dtype == torch.bfloat16
    out.float().sum().backward()
    assert mat_a.grad is not None and mat_a.grad.dtype == torch.bfloat16
    assert mat_b.grad is not None and mat_b.grad.dtype == torch.bfloat16


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
