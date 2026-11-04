"""``optimizer_meta.pt``: the rank count and topology fingerprint vouching for a shard set. Shards are raw local views
that restore only into the layout that wrote them, so a differing fingerprint drops them (a warm restart, or a raise
under PP)."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from halo.checkpoint.atomic import atomic_torch_save
from halo.checkpoint.format import OPTIMIZER_META_FILE
from halo.runtime.ranks import get_global_rank, get_global_world_size

if TYPE_CHECKING:
    from halo.distributed.parallelism_config import ParallelismConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizerStateFingerprint:
    """Topology and optimizer identity of a shard save. The shape-preserving fields (``ep_scope``,
    ``use_grouped_gemm``, ``use_hsdp``, ``nvlink_domain_size``, ``expert_replica_size``) move slice ownership, expert
    names, the FSDP mesh rank or the replica writer at identical shapes, so without them a restore reports success over
    permuted state."""

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
    def capture(
        cls, parallelism_config: ParallelismConfig, optimizer: Any, world_size: int
    ) -> OptimizerStateFingerprint:
        """Rank-uniform, since ``parallelism_config`` is."""
        pc = parallelism_config
        # The dedup runs only where experts are FSDP-ignored plain tensors every EP group replicates; stamping the
        # group count elsewhere would mismatch every checkpoint of a topology that never changed.
        experts_replicated = not pc.experts_fsdp_managed
        return cls(
            world_size=world_size,
            ep_size=pc.ep_size,
            expert_tp_size=pc.expert_tp_size,
            cp_size=pc.cp_size,
            tp_size=pc.tp_size,
            pp_size=pc.pp_size,
            fsdp_shard_ep1_experts=bool(pc.fsdp_shard_ep1_experts),
            optimizer_class=type(optimizer).__name__,
            ep_scope=str(pc.ep_scope),
            use_grouped_gemm=bool(pc.use_grouped_gemm),
            use_hsdp=bool(pc.use_hsdp),
            nvlink_domain_size=int(pc.nvlink_domain_size),
            expert_replica_size=int(pc.num_ep_groups) if experts_replicated else 1,
        )

    def to_dict(self) -> dict[str, Any]:
        """The saved form, ``use_hsdp`` under ``hsdp``: the key every 1.x ``optimizer_meta.pt`` carries."""
        return {("hsdp" if name == "use_hsdp" else name): value for name, value in asdict(self).items()}

    def mismatches(self, saved: Mapping[str, Any]) -> list[str]:
        """Keys where ``saved`` differs from this fingerprint; empty means layout-compatible. A key either side lacks
        is a mismatch, except ``expert_replica_size``: a meta without it holds expert moments on every rank, restorable
        anywhere."""
        live = self.to_dict()
        saved = {"expert_replica_size": live["expert_replica_size"], **saved}
        absent = "<absent>"
        return [
            f"{name}: saved={saved.get(name, absent)!r} current={live.get(name, absent)!r}"
            for name in [*live, *(name for name in saved if name not in live)]
            if name not in saved or name not in live or saved[name] != live[name]
        ]


@dataclass(frozen=True)
class SavedOptimizerMeta:
    """What ``optimizer_meta.pt`` says, as this rank could read it. ``torn`` is why it cannot vouch for its shards
    (missing or unreadable); ``topology_ok`` is the rank-count gate's local verdict, False when torn."""

    topology_ok: bool
    torn: str | None = None
    num_ranks: int = -1
    fingerprint: dict | None = None
    pp_stage_partition: object | None = None


def read_optimizer_meta(checkpoint: str, *, shard_any: bool) -> SavedOptimizerMeta:
    """Read the meta rank-locally; every gate on it is consensus'd. A missing meta beside shards is a failed gate, and
    a torn one never raises here, which would desync the caller's consensus."""
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
    """Staged, so an interrupted write leaves no meta rather than a truncated one."""
    fingerprint = OptimizerStateFingerprint.capture(parallelism_config, optimizer, get_global_world_size())
    meta = {"num_ranks": get_global_world_size(), "fingerprint": fingerprint.to_dict()}
    if pp_stage_partition is not None:
        meta["pp_stage_partition"] = pp_stage_partition
    atomic_torch_save(os.path.join(output_dir, OPTIMIZER_META_FILE), lambda: meta)
