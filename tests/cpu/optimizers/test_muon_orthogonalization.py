#!/usr/bin/env python
"""Newton-Schulz orthogonality of Muon's matrix update (``src/optimizers/muon.py``), on CPU.

Muon replaces a matrix's momentum with an approximation of its polar factor ``U V^T``: five quintic
Newton-Schulz steps after a Frobenius normalization push every singular value into the band
``TOL.muon_orthogonal_sv_*``. The band holds only for inputs whose smallest singular value clears a
fraction of the Frobenius norm that depends on the path (``TOL.muon_band_domain_*``): under the
default ``ns_algorithm`` square matrices take the standard iteration and rectangular ones the Gram
iteration, which needs a larger margin. A raw gradient's smallest singular values sit below either,
where the iteration is not meant to converge, so every input here carries a prescribed spectrum
inside its path's domain.

The square case also runs at two decades: the first three steps only lift singular values below
~1e-2 of the Frobenius norm, so a one-decade input passes with any of them dropped.

The orthogonalizer under test is the one ``create_muon_optimizer`` configures, on its pure-torch
backend. The fused Triton step around it is GPU-only, so ``tests/gpu/optimizers/test_muon.py``
checks the update that step writes, on the CUDA-kernel backend where it runs.

Run: pytest tests/cpu/optimizers/test_muon_orthogonalization.py
"""

import pytest
import torch
import torch.nn as nn

from src.optimizers.muon import create_muon_optimizer
from tests.common.tolerances import TOL
from tests.common.utils import assert_orthogonalized, log_spectrum_matrix


@pytest.fixture(scope="module")
def newton_schulz():
    """The orthogonalizer a toolkit-built Muon runs, on the pure-torch backend."""
    holder = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    return create_muon_optimizer(holder, ns_use_kernels=False).newton_schulz


@pytest.mark.parametrize(
    ("rows", "cols", "decades"),
    [(128, 128, 1.0), (128, 128, 2.0), (64, 192, 1.0), (192, 64, 1.0)],
    ids=["square", "square-two-decades", "wide", "tall"],
)
def test_matrix_update_is_orthogonal(newton_schulz, rows, cols, decades):
    """bf16 momentum in (what the fused momentum kernel hands over), bf16 update out, in the band."""
    generator = torch.Generator().manual_seed(rows * 1000 + cols)
    momentum = log_spectrum_matrix(rows, cols, generator, decades).to(torch.bfloat16)
    # Premise: the input is far from orthogonal, so an orthogonalizer returning it unchanged fails.
    assert torch.linalg.svdvals(momentum.double()).min() < TOL.muon_orthogonal_sv_min / 2

    update = newton_schulz(momentum.unsqueeze(0))[0]

    assert update.shape == momentum.shape and update.dtype == torch.bfloat16
    assert_orthogonalized(update, momentum, f"{rows}x{cols} over {decades:g} decades")


def test_expert_stack_orthogonalizes_each_expert_on_its_own_scale(newton_schulz):
    """A 3-D expert stack is a batch of matrices, each normalized by its own Frobenius norm.

    Experts receive token counts orders of magnitude apart, and their momenta differ in scale the same
    way: one normalization across the stack would leave the small experts' singular values near zero.
    An expert no token reached carries zero momentum and must get a zero update, not NaN.
    """
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
