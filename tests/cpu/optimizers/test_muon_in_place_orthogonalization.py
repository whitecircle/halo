#!/usr/bin/env python
"""CPU test: Muon orthogonalizes its matrix updates in place, chunk by chunk.

The matrix step writes each Newton-Schulz chunk straight back into the Nesterov updates instead of
collecting the results per shape and rebuilding the updates from them, so it holds one chunk beyond the
updates rather than a second full copy. The written updates must be bitwise the ones upstream's
split → stack → scale → reconstruct composition yields, on a group mixing 2D matrices with 3D expert
stacks (the 2D-first order the toolkit builds) and on a 3D-only group (which upstream splits into its 2D
experts). The fused Triton momentum and weight-decay kernels around it are GPU-only
(``tests/gpu/optimizers/test_muon.py``).

Run: pytest tests/cpu/optimizers/test_muon_in_place_orthogonalization.py
"""

import pytest
import torch
import torch.nn as nn
from gram_newton_schulz.muon.muon_utils.muon_matrix_split_utils import (
    get_newton_schulz_inputs_from_gradients,
    reconstruct_update_from_newton_schulz_outputs,
    scale_newton_schulz_outputs_with_adjusted_lr,
)
from gram_newton_schulz.muon.muon_utils.muon_opt_utils import adjust_lr_rms_norm

from halo.optimizers import muon
from halo.optimizers.muon import build_muon_optimizer
from tests.common.memory import LiveTensorBytes

CHUNK = 2
LR = 0.02
MIXED_SHAPES = [(16, 24), (16, 24), (24, 16), (16, 24), (16, 24), (3, 16, 24), (3, 24, 16)]
EXPERT_SHAPES = [(4, 16, 24), (4, 16, 24), (4, 24, 16)]


@pytest.fixture(autouse=True)
def _small_chunks(monkeypatch):
    monkeypatch.setattr(muon, "_GNS_CHUNK_SIZE", CHUNK)


@pytest.fixture(scope="module")
def optimizer():
    """A toolkit-built Muon on the pure-torch Newton-Schulz backend."""
    return build_muon_optimizer(nn.Linear(8, 8, bias=False, dtype=torch.bfloat16), ns_use_kernels=False)


def _updates(shapes, seed):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(shape, generator=generator).to(torch.bfloat16) for shape in shapes]


def _upstream_updates(newton_schulz, updates, lr, adjust_lr_fn):
    """Upstream's composition at the same chunk size: per-shape results collected and concatenated, scaled,
    then reconstructed into each parameter's shape."""
    by_shape, shape_indices, metadata = get_newton_schulz_inputs_from_gradients(updates, None)
    orthogonalized = {
        shape: torch.cat(
            [
                newton_schulz(torch.stack(inputs[start : start + CHUNK])).clone()
                for start in range(0, len(inputs), CHUNK)
            ]
        )
        for shape, inputs in by_shape.items()
    }
    orthogonalized = scale_newton_schulz_outputs_with_adjusted_lr(orthogonalized, lr, adjust_lr_fn)
    return reconstruct_update_from_newton_schulz_outputs(orthogonalized, shape_indices, metadata, None)


@pytest.mark.parametrize("shapes", [MIXED_SHAPES, EXPERT_SHAPES], ids=["mixed-2d-first", "experts-only"])
@pytest.mark.parametrize("adjust_lr_fn", [adjust_lr_rms_norm, None], ids=["rms-norm", "plain-lr"])
def test_in_place_updates_are_bitwise_upstreams(optimizer, shapes, adjust_lr_fn):
    updates = _updates(shapes, seed=len(shapes))
    expected = _upstream_updates(optimizer.newton_schulz, [u.clone() for u in updates], LR, adjust_lr_fn)
    storages = [u.untyped_storage().data_ptr() for u in updates]

    optimizer._orthogonalize_in_place(updates, LR, adjust_lr_fn)

    assert [u.untyped_storage().data_ptr() for u in updates] == storages, "the updates were replaced, not written"
    for index, (got, want) in enumerate(zip(updates, expected, strict=True)):
        assert got.shape == want.shape and got.dtype == want.dtype
        assert torch.equal(got, want), f"update {index} {tuple(got.shape)} differs from upstream's"


def test_step_holds_one_chunk_beyond_the_updates(optimizer, monkeypatch):
    """With Newton-Schulz stubbed to a fresh same-size tensor, the region holds the stacked chunk and its
    result and nothing else; upstream's composition holds every shape's results on top."""
    monkeypatch.setattr(optimizer, "newton_schulz", lambda batch: batch.flip(-1))
    shapes = [(16, 24)] * 12 + [(3, 16, 24)] * 4
    chunk_bytes = CHUNK * 16 * 24 * 3 * torch.bfloat16.itemsize  # largest stacked chunk: two 3D stacks
    total_bytes = sum(torch.Size(shape).numel() for shape in shapes) * torch.bfloat16.itemsize

    updates, upstream_inputs = _updates(shapes, seed=0), _updates(shapes, seed=0)
    with LiveTensorBytes() as in_place:
        optimizer._orthogonalize_in_place(updates, LR, adjust_lr_rms_norm)
    with LiveTensorBytes() as upstream:
        _upstream_updates(optimizer.newton_schulz, upstream_inputs, LR, adjust_lr_rms_norm)

    assert in_place.peak <= 2 * chunk_bytes, f"held {in_place.peak} B beyond the updates (bound {2 * chunk_bytes} B)"
    assert upstream.peak > total_bytes, "premise: upstream's composition holds a second full copy of the updates"


@pytest.mark.parametrize("hook", ["param_split_fn", "param_recombine_fn"])
def test_split_hooks_are_refused(hook):
    model = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    optimizer = build_muon_optimizer(model, ns_use_kernels=False)
    optimizer.param_groups[0][hook] = lambda tensor: tensor
    with pytest.raises(ValueError, match="param_split_fn / param_recombine_fn is not supported"):
        optimizer._muon_step(optimizer.param_groups)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
