"""Topology fingerprint gating sharded optimizer resume.

Per-rank optimizer shards (``optimizer_shard_XXXXX.pt``) are raw local views of the optimizer state:
FSDP2 DTensor shards for sharded params, plain per-rank tensors for EP expert params. They restore
correctly only into the sharding layout that produced them, so the save records this fingerprint in
``optimizer_meta.pt`` and the load rejects a differing one instead of loading tensors into the wrong
ranks/shapes — a warm restart, or a raise under PP, whose stage-local shard FQNs would map moments
onto the wrong layers. Shards carrying no fingerprint are rejected in every mode.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, fields
from typing import Any


def _expert_replica_size(parallelism_config: Any) -> int:
    """How many shards the expert-replica dedup collapses into one; 1 wherever it does not run.

    The dedup runs only where the expert params are the FSDP-ignored plain tensors every EP group
    replicates, which ``experts_fsdp_managed`` decides, so the count is read behind that predicate.
    Dense runs and ep1 runs at the default deduplicate nothing, and the EP-group count is not part of
    their shard layout; reading ``num_ep_groups`` unconditionally would stamp them with a replica
    layout no writer used, so every existing checkpoint would report a mismatch and warm-restart its
    optimizer for a topology that never changed.
    """
    pc = parallelism_config
    if getattr(pc, "experts_fsdp_managed", True):
        return 1
    return int(getattr(pc, "num_ep_groups", 1) or 1)


@dataclass(frozen=True)
class OptimizerStateFingerprint:
    """Topology and optimizer identity captured when per-rank optimizer shards are written.

    Every field changes the per-rank shard layout (or, for ``optimizer_class``, the state-tensor
    schema). A mismatch on resume means the shards cannot be mapped onto the live optimizer, so the
    optimizer state is dropped (warm restart), or the resume raises under PP.
    """

    world_size: int
    ep_size: int
    expert_tp_size: int
    cp_size: int
    tp_size: int
    pp_size: int
    fsdp_shard_ep1_experts: bool
    optimizer_class: str
    # Knobs that leave every tensor shape intact while changing slice ownership, expert parameter
    # names, or the FSDP mesh rank: without them a restore reports success over permuted or
    # reinitialized state.
    ep_scope: str
    use_grouped_gemm: bool
    # Field names are keys inside already-written ``optimizer_meta.pt`` files, and ``from_dict``
    # returns None when one is missing, which rejects the resume. Renaming this to follow the config
    # knob (``use_hsdp``) would therefore reject every existing checkpoint.
    hsdp: bool
    nvlink_domain_size: int
    # How many shards the dedup collapsed into one (:func:`_expert_replica_size`), i.e. which rank's
    # shard carries the replicated expert moments. Defaults to None ("not recorded") and is optional
    # in ``from_dict``; a required field would make every already-written meta incomplete. None is
    # not compared either: a pre-dedup shard set duplicates the expert moments on every rank, so it
    # restores under any replica size. See :meth:`mismatches`.
    expert_replica_size: int | None = None

    @classmethod
    def capture(cls, parallelism_config: Any, optimizer: Any, world_size: int) -> OptimizerStateFingerprint:
        """Fingerprint the live run. ``parallelism_config`` fields are rank-uniform (validated at
        config time), so the captured fingerprint is identical on every rank."""
        pc = parallelism_config
        return cls(
            world_size=world_size,
            ep_size=getattr(pc, "ep_size", 1) or 1,
            expert_tp_size=getattr(pc, "expert_tp_size", 1) or 1,
            cp_size=getattr(pc, "cp_size", 1) or 1,
            tp_size=getattr(pc, "tp_size", 1) or 1,
            # PP shards by stage (local FQNs): a different pp_size maps shards onto the wrong layers.
            pp_size=getattr(pc, "pp_size", 1) or 1,
            fsdp_shard_ep1_experts=bool(getattr(pc, "fsdp_shard_ep1_experts", True)),
            optimizer_class=type(optimizer).__name__,
            # node vs global EP groups permute ep_rank at identical shapes: wrong experts, no error.
            ep_scope=str(getattr(pc, "ep_scope", "node")),
            # Flips expert parameter names (gate_up_proj vs gate_proj_gmm/up_proj_gmm); strict=False
            # then skips the missing FQNs and every expert moment reinitializes.
            use_grouped_gemm=bool(getattr(pc, "use_grouped_gemm", True)),
            hsdp=bool(getattr(pc, "use_hsdp", False)),
            nvlink_domain_size=int(getattr(pc, "nvlink_domain_size", 0) or 0),
            expert_replica_size=_expert_replica_size(pc),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> OptimizerStateFingerprint | None:
        """Rebuild from a saved ``optimizer_meta.pt`` entry.

        Returns ``None`` for anything that is not a complete fingerprint dict (absent entry,
        pre-fingerprint checkpoint, foreign content); the caller then rejects the resume. Only fields
        without a default are required, so a defaulted addition still reads back on existing metas;
        unknown extra keys are ignored, keeping older readers compatible with future additions.
        """
        if not isinstance(data, dict):
            return None
        required = [f.name for f in fields(cls) if f.default is MISSING]
        if any(name not in data for name in required):
            return None
        return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})

    def mismatches(self, saved: OptimizerStateFingerprint) -> list[str]:
        """Fields where ``saved`` differs from this (live) fingerprint; empty means the saved shards
        are layout-compatible with the current run.

        A field the saved meta did not record reads as ``None`` and is skipped, since its absence is
        compatible with any live value. Only fields whose absence is layout-compatible may default to
        ``None``; every other field is required by :meth:`from_dict`.
        """
        return [
            f"{f.name}: saved={getattr(saved, f.name)!r} current={getattr(self, f.name)!r}"
            for f in fields(self)
            if getattr(saved, f.name) is not None and getattr(saved, f.name) != getattr(self, f.name)
        ]
