"""Torch ``DeviceMesh`` construction and the typed view over one (torch-native DP / HSDP / TP only).

EP/CP use hand-built process groups (``group_layout.py``); their rank orderings don't map to a
row-major DeviceMesh, and their groups live on ``EPConfig`` rather than being mirrored here.
"""

from __future__ import annotations

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


class MeshDim:
    """Canonical ``mesh_dim_names``. Row-major DeviceMesh shards on the last dim, so HSDP is
    ``(DP_REPLICATE, DP_SHARD)``: shard within a domain, replicate across domains."""

    DP = "dp"
    DP_REPLICATE = "dp_replicate"
    DP_SHARD = "dp_shard"
    TP = "tp"


def create_dp_mesh(
    dp_size: int,
    dp_replicate_size: int = 1,
    dp_group=None,
    device_type: str = "cuda",
) -> DeviceMesh:
    """Build the data-parallel mesh for FSDP2 ``fully_shard``.

    - ``dp_replicate_size > 1`` → 2D HSDP mesh ``(dp_replicate, dp_shard)`` with
      ``dp_shard = dp_size // dp_replicate_size``: shard within each NVLink domain,
      replicate across domains.
    - ``dp_group`` given (1D only) → wrap that process group as a ``(dp,)`` mesh.
    - otherwise → 1D ``(dp,)`` full-shard mesh.
    """
    if dp_replicate_size > 1:
        if dp_group is not None:
            raise ValueError("create_dp_mesh: dp_group is 1D-only and cannot be combined with dp_replicate_size > 1.")
        if dp_size % dp_replicate_size != 0:
            raise ValueError(f"HSDP dp_replicate_size ({dp_replicate_size}) must divide the DP world ({dp_size}).")
        # init_device_mesh numbers the mesh 0..dp_size-1 and does NOT check it against the world, so a
        # dp_size below the world silently hands EVERY rank a mesh made of the FIRST dp_size ranks —
        # the rest would then shard over a mesh they are not in. Reachable only by a caller whose DP
        # scope is a sub-block of the world (a pipeline stage), which needs an outer pipeline mesh
        # dimension instead: init_device_mesh((pp, dp_replicate, dp_shard))[dp_replicate, dp_shard].
        world_size = dist.get_world_size() if dist.is_initialized() else dp_size
        if dp_size != world_size:
            raise ValueError(
                f"create_dp_mesh: a 2-D HSDP mesh must span the whole world; got dp_size={dp_size} "
                f"with world_size={world_size}. A sub-world DP scope (a pipeline stage) needs the "
                f"stage dimension in the mesh itself, not a smaller 2-D mesh."
            )
        dp_shard_size = dp_size // dp_replicate_size
        return init_device_mesh(
            device_type,
            (dp_replicate_size, dp_shard_size),
            mesh_dim_names=(MeshDim.DP_REPLICATE, MeshDim.DP_SHARD),
        )
    if dp_group is not None:
        return DeviceMesh.from_group(dp_group, device_type, mesh_dim_names=(MeshDim.DP,))
    # Same sub-world hazard as the 2-D branch, checked here too: init_device_mesh numbers a 1-D mesh
    # 0..dp_size-1, so a dp_size below the world builds a mesh of the FIRST dp_size ranks. torch does
    # not reject it up front — the ranks left out raise a bare "IndexError: list index out of range"
    # while the ranks inside proceed and then block on collectives their peers will never reach.
    world_size = dist.get_world_size() if dist.is_initialized() else dp_size
    if dp_size != world_size:
        raise ValueError(
            f"create_dp_mesh: a 1-D DP mesh must span the whole world; got dp_size={dp_size} with "
            f"world_size={world_size}. Pass the sub-world's process group as dp_group (a pipeline "
            f"stage's DP scope) so the mesh is built from ITS ranks, not ranks 0..{dp_size - 1}."
        )
    return init_device_mesh(device_type, (dp_size,), mesh_dim_names=(MeshDim.DP,))


def create_dp_tp_mesh(tp_size: int, dp_size: int = 1, device_type: str = "cuda") -> DeviceMesh:
    """Build the TP mesh, adding a data-parallel dim when ``dp_size > 1``.

    ``dp_size == 1`` → 1D ``(tp,)``; else 2D ``(dp, tp)`` row-major (EP groups align with TP rows).
    Example ``TP=4`` on 8 GPUs (``dp_size=2``)::

        2D mesh: [[0, 1, 2, 3], [4, 5, 6, 7]]
        DP pairs:  (0, 4), (1, 5), (2, 6), (3, 7)
    """
    if dp_size == 1:
        return init_device_mesh(device_type, (tp_size,), mesh_dim_names=(MeshDim.TP,))
    return init_device_mesh(device_type, (dp_size, tp_size), mesh_dim_names=(MeshDim.DP, MeshDim.TP))


def mesh_dim_names(device_mesh: DeviceMesh | None) -> tuple[str, ...]:
    """A mesh's named dims, ``()`` for an unnamed mesh or none at all."""
    return tuple(getattr(device_mesh, "mesh_dim_names", None) or ())


def has_tp_dim(device_mesh: DeviceMesh | None) -> bool:
    """Whether ``device_mesh`` carries a named ``tp`` dim.

    The one predicate every TP-mesh consumer branches on, so a DP-only ``(dp,)`` / HSDP mesh can
    never have its DP group relabelled as TP. An UNNAMED mesh answers False here and is treated as a
    bare TP mesh by the callers that accept one — that is the caller's call, not the predicate's.
    """
    return MeshDim.TP in mesh_dim_names(device_mesh)


def get_tp_submesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """Return the TP sub-mesh of a 1D ``(tp,)`` or 2D ``(dp, tp)`` mesh.

    A mesh with no named dims is treated as a bare TP mesh; one whose names omit ``tp`` (a DP-only
    or HSDP mesh) is returned unchanged rather than indexed into a dim it does not have.
    """
    if has_tp_dim(device_mesh) and len(mesh_dim_names(device_mesh)) > 1:
        return device_mesh[MeshDim.TP]
    return device_mesh


class ParallelDims:
    """Typed accessors for the active device mesh of one trainer."""

    def __init__(self, device_mesh: DeviceMesh | None = None):
        self.device_mesh = device_mesh

    @property
    def mesh_dims(self) -> tuple[str, ...]:
        return mesh_dim_names(self.device_mesh)

    def tp_group(self) -> dist.ProcessGroup | None:
        """Attention-TP process group, or None when TP is inactive.

        A named mesh must carry a ``tp`` dim — a DP-only mesh (``(dp,)`` / HSDP) returns None, never
        its DP group relabeled as TP. An UNNAMED mesh is treated as a bare TP mesh.
        """
        if self.device_mesh is None:
            return None
        if has_tp_dim(self.device_mesh):
            return self.device_mesh[MeshDim.TP].get_group()
        if not self.mesh_dims:
            return self.device_mesh.get_group()
        return None

    def tp_local_rank(self) -> int | None:
        """Rank within the attention-TP group from the mesh, or None if not 2D/1D-TP."""
        if self.device_mesh is not None and has_tp_dim(self.device_mesh):
            return self.device_mesh[MeshDim.TP].get_local_rank()
        return None

    def dp_group(self) -> dist.ProcessGroup | None:
        """Data-parallel sub-group of a 2D ``(dp, tp)`` mesh, else None."""
        if self.device_mesh is not None and MeshDim.DP in self.mesh_dims:
            return self.device_mesh[MeshDim.DP].get_group()
        return None
