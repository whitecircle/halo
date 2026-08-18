#!/usr/bin/env python
"""Accessor-contract tests for ``ParallelDims`` (src/distributed/mesh.py) and the
``has_tp_dim`` predicate its TP accessors share with the other two TP-mesh consumers.

``ParallelDims`` is the single typed view over a run's device mesh. These tests pin its branching
logic — an inactive axis returns None, and the accessors resolve the TP/DP sub-groups by ``MeshDim``
name. A regression in any branch (e.g. a wrong dim name, or a DP-only mesh reported as a TP group)
fails here without needing GPUs.

Run: python tests/cpu/parallelism/test_parallel_dims.py
"""

from types import SimpleNamespace

import pytest

from src.distributed.mesh import MeshDim, ParallelDims, get_tp_submesh, has_tp_dim
from src.distributed.tensor_parallel.state_dict import get_tp_mesh


class _FakeSubmesh:
    def __init__(self, group, local_rank=0):
        self._group = group
        self._local_rank = local_rank

    def get_group(self):
        return self._group

    def get_local_rank(self):
        return self._local_rank


class _FakeMesh:
    """Mimics the bits of torch DeviceMesh that ParallelDims touches."""

    def __init__(self, dim_names, submeshes=None, group=None):
        self.mesh_dim_names = dim_names
        self._submeshes = submeshes or {}
        self._group = group

    def __getitem__(self, name):
        return self._submeshes[name]

    def get_group(self):
        return self._group


def test_inactive_returns_none():
    pd = ParallelDims()
    assert pd.tp_group() is None
    assert pd.tp_local_rank() is None
    assert pd.dp_group() is None
    assert pd.mesh_dims == ()


def test_2d_mesh_tp_and_dp_groups():
    mesh = _FakeMesh(
        (MeshDim.DP, MeshDim.TP),
        submeshes={
            MeshDim.DP: _FakeSubmesh("DP_GROUP"),
            MeshDim.TP: _FakeSubmesh("TP_GROUP", local_rank=1),
        },
    )
    pd = ParallelDims(device_mesh=mesh)
    assert pd.tp_group() == "TP_GROUP"
    assert pd.tp_local_rank() == 1
    assert pd.dp_group() == "DP_GROUP"
    assert pd.mesh_dims == (MeshDim.DP, MeshDim.TP)


def test_1d_mesh_without_tp_dim_is_the_tp_group():
    """A bare 1D mesh (no 'tp' dim name) is itself the TP group; no DP sub-group."""
    mesh = _FakeMesh((), group="ONE_D_GROUP")
    pd = ParallelDims(device_mesh=mesh)
    assert pd.tp_group() == "ONE_D_GROUP"
    assert pd.tp_local_rank() is None
    assert pd.dp_group() is None


def test_named_mesh_without_tp_dim_has_no_tp_group():
    """A NAMED mesh without a 'tp' dim has no TP group: a (dp,) mesh must return None — not its DP
    group relabeled as TP — and an HSDP (dp_replicate, dp_shard) mesh must return None, not raise."""
    dp_mesh = _FakeMesh((MeshDim.DP,), submeshes={MeshDim.DP: _FakeSubmesh("DP_GROUP")}, group="DP_GROUP")
    pd = ParallelDims(device_mesh=dp_mesh)
    assert pd.tp_group() is None
    assert pd.dp_group() == "DP_GROUP"

    hsdp_mesh = _FakeMesh((MeshDim.DP_REPLICATE, MeshDim.DP_SHARD))
    assert ParallelDims(device_mesh=hsdp_mesh).tp_group() is None


@pytest.mark.parametrize(
    ("dim_names", "carries_tp"),
    [
        ((MeshDim.DP, MeshDim.TP), True),
        ((MeshDim.TP,), True),
        ((MeshDim.DP,), False),
        ((MeshDim.DP_REPLICATE, MeshDim.DP_SHARD), False),
        ((), False),
    ],
)
def test_every_tp_mesh_consumer_reads_one_predicate(dim_names, carries_tp):
    """Three places ask whether a mesh carries a ``tp`` dim — ``ParallelDims.tp_group``,
    ``mesh.get_tp_submesh`` and the state dict's ``get_tp_mesh`` — and all three must read
    :func:`has_tp_dim`. Private spellings drift: one answering "yes" for an unnamed mesh while
    another answers "no" makes a run gather a TP save through a mesh its group lookup calls TP-less.
    Each assert below ties one consumer's verdict to the predicate.
    """
    mesh = _FakeMesh(
        dim_names,
        submeshes={
            MeshDim.DP: _FakeSubmesh("DP_GROUP"),
            MeshDim.TP: _FakeSubmesh("TP_GROUP", local_rank=1),
        },
        group="MESH_GROUP",
    )

    assert has_tp_dim(mesh) is carries_tp

    # The state-dict half: a mesh the predicate rejects has no TP mesh, never a DP-only fallback.
    assert (get_tp_mesh(SimpleNamespace(_device_mesh=mesh)) is not None) is carries_tp

    # The indexing half: descend into the tp dim exactly when the predicate says the dim is there
    # (a 1D (tp,) mesh IS its own sub-mesh, so only a multi-dim mesh is indexed).
    assert (get_tp_submesh(mesh) is not mesh) is (carries_tp and len(dim_names) > 1)

    # The group half: the predicate, plus this accessor's documented allowance that an UNNAMED mesh
    # is a bare TP mesh — the one place the three verdicts legitimately differ.
    assert (ParallelDims(device_mesh=mesh).tp_group() is not None) is (carries_tp or not dim_names)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
