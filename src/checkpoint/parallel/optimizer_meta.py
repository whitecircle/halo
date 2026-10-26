"""``optimizer_meta.pt``: the rank count and topology fingerprint that vouch for a per-rank optimizer shard set.

The shards are raw local views (FSDP2 DTensor shards, plain per-rank EP tensors) that restore only into
the sharding layout that wrote them, so a resume under a differing fingerprint drops them (a warm
restart, or a raise under PP, whose stage-local FQNs would map moments onto the wrong layers).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.distributed as dist

from src.checkpoint.atomic import atomic_torch_save
from src.checkpoint.format import OPTIMIZER_META_FILE
from src.runtime.ranks import get_global_rank, get_global_world_size

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizerStateFingerprint:
    """Topology and optimizer identity of a per-rank shard save, its config fields named as
    :class:`~src.distributed.parallelism_config.ParallelismConfig`'s.

    Every field changes the per-rank shard layout or, for ``optimizer_class``, the state schema. The
    shape-preserving ones (``ep_scope``, ``use_grouped_gemm``, ``use_hsdp``, ``nvlink_domain_size``,
    ``expert_replica_size``) move slice ownership, expert parameter names, the FSDP mesh rank or the
    replica writer at identical shapes, so without them a restore reports success over permuted state.
    """

    world_size: int
    ep_size: int
    expert_tp_size: int
    cp_size: int
    tp_size: int
    pp_size: int
    fsdp_shard_ep1_experts: bool
    optimizer_class: str
    ep_scope: str
    use_grouped_gemm: bool
    use_hsdp: bool
    nvlink_domain_size: int
    # How many shards the expert-replica dedup collapses into one, 1 wherever it does not run.
    expert_replica_size: int

    @classmethod
    def capture(cls, parallelism_config: Any, optimizer: Any, world_size: int) -> OptimizerStateFingerprint:
        """Fingerprint the live run; rank-uniform, since ``parallelism_config`` is."""
        pc = parallelism_config
        # The dedup runs only where the experts are the FSDP-ignored plain tensors every EP group
        # replicates; elsewhere the EP-group count describes no shard layout, and stamping it would
        # mismatch every checkpoint of a topology that never changed.
        experts_replicated = not getattr(pc, "experts_fsdp_managed", True)
        return cls(
            world_size=world_size,
            ep_size=getattr(pc, "ep_size", 1) or 1,
            expert_tp_size=getattr(pc, "expert_tp_size", 1) or 1,
            cp_size=getattr(pc, "cp_size", 1) or 1,
            tp_size=getattr(pc, "tp_size", 1) or 1,
            pp_size=getattr(pc, "pp_size", 1) or 1,
            fsdp_shard_ep1_experts=bool(getattr(pc, "fsdp_shard_ep1_experts", True)),
            optimizer_class=type(optimizer).__name__,
            ep_scope=str(getattr(pc, "ep_scope", "node")),
            use_grouped_gemm=bool(getattr(pc, "use_grouped_gemm", True)),
            use_hsdp=bool(getattr(pc, "use_hsdp", False)),
            nvlink_domain_size=int(getattr(pc, "nvlink_domain_size", 0) or 0),
            expert_replica_size=int(getattr(pc, "num_ep_groups", 1) or 1) if experts_replicated else 1,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def mismatches(self, saved: Mapping[str, Any]) -> list[str]:
        """The fields where the ``saved`` fingerprint dict differs from this live one; empty means its
        shards are layout-compatible. The key sets must match exactly: a field either side lacks is a
        mismatch, so a meta written under another fingerprint schema never restores."""
        live = self.to_dict()
        absent = "<absent>"
        return [
            f"{name}: saved={saved.get(name, absent)!r} current={live.get(name, absent)!r}"
            for name in [*live, *(name for name in saved if name not in live)]
            if name not in saved or name not in live or saved[name] != live[name]
        ]


@dataclass(frozen=True)
class SavedOptimizerMeta:
    """What ``optimizer_meta.pt`` says about the save, as this rank could read it.

    ``torn`` is why the meta cannot vouch for the shards beside it (missing, or unreadable), None when
    it can. ``topology_ok`` is the rank-count gate's local verdict, False on a torn meta too. The rest
    are the raw saved fields, ``None`` / ``-1`` where the meta did not carry them.
    """

    topology_ok: bool
    torn: str | None = None
    num_ranks: int = -1
    fingerprint: dict | None = None
    pp_stage_partition: object | None = None


def read_optimizer_meta(checkpoint: str, *, shard_any: bool) -> SavedOptimizerMeta:
    """``checkpoint``'s ``optimizer_meta.pt``, read rank-locally; every gate on it is consensus'd.

    A missing meta beside shards is a failed gate (no rank-count or fingerprint gate can run on
    ungated state), and a torn one never raises here, which would desync the caller's consensus.
    """
    meta_path = os.path.join(checkpoint, OPTIMIZER_META_FILE)
    if not os.path.exists(meta_path):
        torn = f"{OPTIMIZER_META_FILE} is missing beside the per-rank optimizer shards" if shard_any else None
        return SavedOptimizerMeta(topology_ok=not shard_any, torn=torn)
    saved_ranks = -1
    try:
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        saved_ranks = meta.get("num_ranks", -1)
        fingerprint = meta.get("fingerprint")
        return SavedOptimizerMeta(
            topology_ok=saved_ranks == get_global_world_size(),
            num_ranks=saved_ranks,
            fingerprint=fingerprint if isinstance(fingerprint, dict) else None,
            pp_stage_partition=meta.get("pp_stage_partition"),
        )
    except Exception as e:
        logger.warning(f"[rank {get_global_rank()}] Unreadable {OPTIMIZER_META_FILE} at {meta_path}: {e}")
        return SavedOptimizerMeta(
            topology_ok=False,
            torn=f"{OPTIMIZER_META_FILE} is unreadable ({type(e).__name__})",
            num_ranks=saved_ranks,
        )


def capture_pp_stage_partition(stage, parallelism_config) -> list[list[int] | None]:
    """COLLECTIVE, on every rank. Each PP stage's ``[lo, hi)`` layer range, read off the LIVE stages, never
    the split formula the resume gate must catch."""
    lo, hi = stage.layer_range
    entries: list[tuple[int, int, int] | None] = [None] * get_global_world_size()
    dist.all_gather_object(entries, (parallelism_config.pp_rank, lo, hi))
    partition: list[list[int] | None] = [None] * parallelism_config.pp_size
    for pp_rank, stage_lo, stage_hi in entries:
        partition[pp_rank] = [stage_lo, stage_hi]
    return partition


def write_optimizer_meta(output_dir: str, parallelism_config, optimizer, pp_stage_partition) -> None:
    """Write ``optimizer_meta.pt``: the rank count, the topology fingerprint and, under PP, the stage
    partition. Staged, so an interrupted write leaves no meta rather than a truncated one."""
    fingerprint = OptimizerStateFingerprint.capture(parallelism_config, optimizer, get_global_world_size())
    meta = {"num_ranks": get_global_world_size(), "fingerprint": fingerprint.to_dict()}
    if pp_stage_partition is not None:
        meta["pp_stage_partition"] = pp_stage_partition
    atomic_torch_save(os.path.join(output_dir, OPTIMIZER_META_FILE), lambda: meta)
