"""Torch ``DeviceMesh`` construction and TP/DP accessors. EP/CP use hand-built groups (``group_layout.py``), since
their rank orderings do not map to a row-major mesh."""

from __future__ import annotations

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from halo.runtime.ranks import has_process_group


class MeshDim:
    """Canonical ``mesh_dim_names``. A row-major mesh shards on the last dim, hence HSDP's order."""

    DP = "dp"
    DP_REPLICATE = "dp_replicate"
    DP_SHARD = "dp_shard"
    TP = "tp"


def _require_whole_world(dp_size: int, mesh_kind: str, remedy: str) -> None:
    """``init_device_mesh`` numbers a mesh ``0..dp_size-1`` unchecked, so ranks left out raise a bare ``IndexError``
    while the rest block on collectives."""
    world_size = dist.get_world_size() if has_process_group() else dp_size
    if dp_size != world_size:
        raise ValueError(
            f"build_dp_mesh: a {mesh_kind} must span the whole world; got dp_size={dp_size} with "
            f"world_size={world_size}. {remedy}"
        )


def build_dp_mesh(
    dp_size: int,
    dp_replicate_size: int = 1,
    dp_group: dist.ProcessGroup | None = None,
    device_type: str = "cuda",
) -> DeviceMesh:
    """The FSDP2 mesh: 2D HSDP ``(dp_replicate, dp_shard)`` when ``dp_replicate_size > 1``, else 1D ``(dp,)`` over
    ``dp_group`` or the world."""
    if dp_replicate_size > 1:
        if dp_group is not None:
            raise ValueError("build_dp_mesh: dp_group is 1D-only and cannot be combined with dp_replicate_size > 1.")
        if dp_size % dp_replicate_size != 0:
            raise ValueError(f"HSDP dp_replicate_size ({dp_replicate_size}) must divide the DP world ({dp_size}).")
        _require_whole_world(
            dp_size,
            "2-D HSDP mesh",
            "A sub-world DP scope (a pipeline stage) needs the stage dimension in the mesh itself, not a "
            "smaller 2-D mesh.",
        )
        return init_device_mesh(
            device_type,
            (dp_replicate_size, dp_size // dp_replicate_size),
            mesh_dim_names=(MeshDim.DP_REPLICATE, MeshDim.DP_SHARD),
        )
    if dp_group is not None:
        return DeviceMesh.from_group(dp_group, device_type, mesh_dim_names=(MeshDim.DP,))
    _require_whole_world(
        dp_size,
        "1-D DP mesh",
        f"Pass the sub-world's process group as dp_group (a pipeline stage's DP scope) so the mesh is "
        f"built from ITS ranks, not ranks 0..{dp_size - 1}.",
    )
    return init_device_mesh(device_type, (dp_size,), mesh_dim_names=(MeshDim.DP,))


def build_dp_tp_mesh(tp_size: int, dp_size: int = 1, device_type: str = "cuda") -> DeviceMesh:
    """The TP mesh: 1D ``(tp,)`` at ``dp_size == 1``, else 2D row-major ``(dp, tp)``, so EP groups align with TP rows
    (TP=4 on 8 GPUs: rows ``[0..3]``, ``[4..7]``)."""
    if dp_size == 1:
        return init_device_mesh(device_type, (tp_size,), mesh_dim_names=(MeshDim.TP,))
    return init_device_mesh(device_type, (dp_size, tp_size), mesh_dim_names=(MeshDim.DP, MeshDim.TP))


def mesh_dim_names(device_mesh: DeviceMesh | None) -> tuple[str, ...]:
    return tuple(getattr(device_mesh, "mesh_dim_names", None) or ())


def has_tp_dim(device_mesh: DeviceMesh | None) -> bool:
    """Whether the mesh has a named ``tp`` dim. An unnamed mesh returns False; callers that accept a bare TP mesh treat
    it as one themselves."""
    return MeshDim.TP in mesh_dim_names(device_mesh)


def tp_submesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """The ``tp`` slice of a 2D ``(dp, tp)`` mesh; any other mesh is returned unchanged."""
    if has_tp_dim(device_mesh) and len(mesh_dim_names(device_mesh)) > 1:
        return device_mesh[MeshDim.TP]
    return device_mesh


def tp_group(device_mesh: DeviceMesh | None) -> dist.ProcessGroup | None:
    """The attention-TP group, or None when TP is inactive (a DP-only or HSDP mesh). Unnamed means a bare TP mesh."""
    if has_tp_dim(device_mesh):
        return device_mesh[MeshDim.TP].get_group()
    if device_mesh is not None and not mesh_dim_names(device_mesh):
        return device_mesh.get_group()
    return None


def tp_local_rank(device_mesh: DeviceMesh | None) -> int | None:
    return device_mesh[MeshDim.TP].get_local_rank() if has_tp_dim(device_mesh) else None


def dp_group(device_mesh: DeviceMesh | None) -> dist.ProcessGroup | None:
    return device_mesh[MeshDim.DP].get_group() if MeshDim.DP in mesh_dim_names(device_mesh) else None


def model_tp_mesh(model) -> DeviceMesh | None:
    """The TP mesh ``model`` was parallelized on, or ``None`` where :func:`tp_submesh` would return a DP-only mesh."""
    device_mesh = getattr(model, "_device_mesh", None)
    if device_mesh is None or not has_tp_dim(device_mesh):
        return None
    return tp_submesh(device_mesh)
