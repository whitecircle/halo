#!/usr/bin/env python
"""Newton-Schulz orthogonality of Muon's matrix update (``halo/optimizers/muon.py``). The band holds only inside
``TOL.muon_band_domain_*``, so inputs carry a prescribed spectrum."""

import pytest
import torch
import torch.nn as nn

from halo.optimizers.muon import build_muon_optimizer
from tests.common.numerics import assert_orthogonalized, log_spectrum_matrix
from tests.common.tolerances import TOL


@pytest.fixture(scope="module")
def newton_schulz():
    holder = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    return build_muon_optimizer(holder, ns_use_kernels=False).newton_schulz


@pytest.mark.parametrize(
    ("rows", "cols", "decades"),
    [(128, 128, 1.0), (128, 128, 2.0), (64, 192, 1.0), (192, 64, 1.0)],
    ids=["square", "square-two-decades", "wide", "tall"],
)
def test_matrix_update_is_orthogonal(newton_schulz, rows, cols, decades):
    """bf16 momentum in, bf16 update out."""
    generator = torch.Generator().manual_seed(rows * 1000 + cols)
    momentum = log_spectrum_matrix(rows, cols, generator, decades).to(torch.bfloat16)
    # Premise: an orthogonalizer returning a far-from-orthogonal input unchanged fails.
    assert torch.linalg.svdvals(momentum.double()).min() < TOL.muon_orthogonal_sv_min / 2

    update = newton_schulz(momentum.unsqueeze(0))[0]

    assert update.shape == momentum.shape and update.dtype == torch.bfloat16
    assert_orthogonalized(update, momentum, f"{rows}x{cols} over {decades:g} decades")


def test_expert_stack_orthogonalizes_each_expert_on_its_own_scale(newton_schulz):
    """One normalization across the stack leaves small experts near zero; an expert no token reached gets a zero
    update, not NaN."""
    generator = torch.Generator().manual_seed(0)
    scales = (1e-3, 1.0, 1e3, 0.0)
    stack = torch.stack([log_spectrum_matrix(64, 96, generator) * scale for scale in scales]).to(torch.bfloat16)

    update = newton_schulz(stack)

    assert update.shape == stack.shape and update.dtype == torch.bfloat16
    for index, scale in enumerate(scales):
        if scale == 0.0:
            assert torch.equal(update[index], torch.zeros_like(update[index])), "a zero-momentum expert moved"
        else:
            assert_orthogonalized(update[index], stack[index], f"expert {index} (scale {scale:g})")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
