#!/usr/bin/env python
"""Newton-Schulz orthogonality of Muon's matrix update (``src/optimizers/muon.py``), on CPU.

Muon replaces a matrix's momentum with an approximation of its polar factor ``U V^T``: five quintic
Newton-Schulz steps after a Frobenius normalization push every singular value to ~1. The band the
iteration reaches (``TOL.muon_orthogonal_sv_*``) holds for singular values of at least 1.4e-3 of the
Frobenius norm, so every input here carries a prescribed spectrum inside that domain; a raw
gradient's smallest singular values sit below it, where the iteration is not meant to converge.

The orthogonalizer under test is the one ``create_muon_optimizer`` configures, on its pure-torch
backend: square matrices take the standard iteration, rectangular ones the Gram iteration. The fused
Triton step around it is GPU-only, so ``tests/gpu/optimizers/test_muon.py`` checks the update that
step writes, on the CUDA-kernel backend.

Run: pytest tests/cpu/optimizers/test_muon_orthogonalization.py
"""

import pytest
import torch
import torch.nn as nn

from src.optimizers.muon import create_muon_optimizer
from tests.common.tolerances import TOL
from tests.common.utils import assert_orthogonalized, matrix_with_spectrum

# Singular values log-spaced over one decade: at the shapes below the smallest stays above 1e-2 of the
# Frobenius norm, well inside the band's domain, while the input itself is far from orthogonal.
SPECTRUM_DECADES = 1.0


@pytest.fixture(scope="module")
def newton_schulz():
    """The orthogonalizer a toolkit-built Muon runs, on the pure-torch backend."""
    holder = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    return create_muon_optimizer(holder, ns_use_kernels=False).newton_schulz


def _momentum(rows: int, cols: int, generator: torch.Generator) -> torch.Tensor:
    spectrum = torch.logspace(0, -SPECTRUM_DECADES, min(rows, cols), dtype=torch.float64)
    return matrix_with_spectrum(rows, cols, spectrum, generator)


@pytest.mark.parametrize(("rows", "cols"), [(128, 128), (64, 192), (192, 64)], ids=["square", "wide", "tall"])
def test_matrix_update_is_orthogonal(newton_schulz, rows, cols):
    """bf16 momentum in (what the fused momentum kernel hands over), bf16 update out, in the band."""
    momentum = _momentum(rows, cols, torch.Generator().manual_seed(rows * 1000 + cols)).to(torch.bfloat16)
    # Premise: the input is far from orthogonal, so an orthogonalizer returning it unchanged fails.
    assert torch.linalg.svdvals(momentum.double()).min() < TOL.muon_orthogonal_sv_min / 2

    update = newton_schulz(momentum.unsqueeze(0))[0]

    assert update.shape == momentum.shape and update.dtype == torch.bfloat16
    assert_orthogonalized(update, momentum, f"{rows}x{cols}")


def test_expert_stack_orthogonalizes_each_expert_on_its_own_scale(newton_schulz):
    """A 3-D expert stack is a batch of matrices, each normalized by its own Frobenius norm.

    Experts receive token counts orders of magnitude apart, and their momenta differ in scale the same
    way: one normalization across the stack would leave the small experts' singular values near zero.
    An expert no token reached carries zero momentum and must get a zero update, not NaN.
    """
    generator = torch.Generator().manual_seed(0)
    scales = (1e-3, 1.0, 1e3, 0.0)
    stack = torch.stack([_momentum(64, 96, generator) * scale for scale in scales]).to(torch.bfloat16)

    update = newton_schulz(stack)

    assert update.shape == stack.shape and update.dtype == torch.bfloat16
    for index, scale in enumerate(scales):
        if scale == 0.0:
            assert torch.equal(update[index], torch.zeros_like(update[index])), "a zero-momentum expert moved"
        else:
            assert_orthogonalized(update[index], stack[index], f"expert {index} (scale {scale:g})")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
