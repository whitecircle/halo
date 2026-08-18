"""Optimizer-state topology fingerprint — the same-topology gate for sharded optimizer resume.

Per-rank optimizer shards (``optimizer_shard_XXXXX.pt``) are raw local views of the optimizer
state: FSDP2 DTensor shards for sharded params and plain per-rank tensors for EP expert params.
They only restore correctly into the exact sharding layout that produced them, so the save path
records this fingerprint inside ``optimizer_meta.pt`` and the load path refuses a differing one
instead of loading tensors into the wrong ranks/shapes: a loud warm restart, or — under PP, whose
stage-LOCAL shard FQNs would map moments onto the wrong layers — a raise that stands until the
shards are deleted. Shards carrying no fingerprint at all are refused outright in every mode.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, fields
from typing import Any


def _expert_replica_size(parallelism_config: Any) -> int:
    """How many shards the expert-replica dedup collapses into one — 1 wherever it does not run.

    The dedup has a writer only where the expert params are the FSDP-IGNORED plain tensors every EP
    group replicates — ``experts_fsdp_managed`` is the one predicate that decides that, so the count
    is read behind it rather than beside it. Everywhere else — every dense run, and every ep1 run at
    the default — nothing is deduplicated and the EP-group count is not part of the shard layout at
    all. Reading ``num_ep_groups`` unconditionally would stamp those runs with a replica layout no
    writer used, and the count is a derived quantity (one singleton group per rank at
    ``ep_group_size == 1``): every existing checkpoint would then report a mismatch and warm-restart
    its optimizer for a topology that never changed.
    """
    pc = parallelism_config
    if getattr(pc, "experts_fsdp_managed", True):
        return 1
    return int(getattr(pc, "num_ep_groups", 1) or 1)


@dataclass(frozen=True)
class OptimizerStateFingerprint:
    """Topology + optimizer identity captured when per-rank optimizer shards are written.

    Every field changes the per-rank layout (or the state-tensor schema, for
    ``optimizer_class``) of the saved shards; a mismatch on resume means the shards cannot be
    mapped onto the live optimizer, so the optimizer state is dropped (warm restart) — or, under
    PP, the resume raises while those shards are still on disk.
    """

    world_size: int
    ep_size: int
    expert_tp_size: int
    cp_size: int
    tp_size: int
    pp_size: int
    fsdp_shard_ep1_experts: bool
    optimizer_class: str
    # Below: knobs that leave every tensor SHAPE intact while changing slice ownership, expert
    # parameter names, or the FSDP mesh rank — a restore without them reports success while the
    # state is permuted or silently reinitialized.
    ep_scope: str
    use_grouped_gemm: bool
    # Spelled as the config knob was when this schema was written: these names are keys inside
    # already-written ``optimizer_meta.pt`` files, and ``from_dict`` returns None the moment one is
    # missing — which now REFUSES the resume. Renaming it here would reject every existing
    # checkpoint, so the config knob (``use_hsdp``) moved on alone.
    hsdp: bool
    nvlink_domain_size: int
    # How many shards the dedup collapsed into one (:func:`_expert_replica_size`), i.e. which rank's
    # shard carries the replicated expert moments. Defaulted to None ("the meta did not record it"),
    # and therefore optional in ``from_dict``: a REQUIRED addition would make every already-written
    # meta incomplete and refuse every existing checkpoint. ``None`` must NOT be read as a value
    # either — a pre-dedup shard set duplicates the expert moments on every rank, so it restores
    # under any replica size, and comparing it against a live count would discard the optimizer state
    # of every multi-node EP checkpoint written before the dedup (a hard raise under PP). See
    # :meth:`mismatches`.
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
            # PP shards by STAGE (local FQNs): a different pp_size maps shards onto the wrong layers.
            pp_size=getattr(pc, "pp_size", 1) or 1,
            fsdp_shard_ep1_experts=bool(getattr(pc, "fsdp_shard_ep1_experts", True)),
            optimizer_class=type(optimizer).__name__,
            # node vs global EP groups permute ep_rank at identical shapes — wrong experts, no error.
            ep_scope=str(getattr(pc, "ep_scope", "node")),
            # Flips expert parameter NAMES (gate_up_proj vs gate_proj_gmm/up_proj_gmm), and
            # strict=False skips the now-missing FQNs — every expert moment silently reinitializes.
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
        pre-fingerprint checkpoint, foreign content); the caller refuses to resume an unfingerprinted
        shard set. Only fields without a default are required, so a defaulted addition reads back on
        every existing meta instead of refusing it; unknown extra keys are ignored so older readers
        stay compatible with future additions.
        """
        if not isinstance(data, dict):
            return None
        required = [f.name for f in fields(cls) if f.default is MISSING]
        if any(name not in data for name in required):
            return None
        return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})

    def mismatches(self, saved: OptimizerStateFingerprint) -> list[str]:
        """Human-readable list of fields where ``saved`` differs from this (live) fingerprint;
        empty means the saved shards are layout-compatible with the current run.

        A field the saved meta did not record reads as ``None`` and is skipped: absence is
        "compatible with any live value", never a mismatch. Only fields whose ABSENCE is
        layout-compatible may default to ``None`` — every other field is required by
        :meth:`from_dict`, so an unfingerprinted shard set is refused rather than read as compatible.
        """
        return [
            f"{f.name}: saved={getattr(saved, f.name)!r} current={getattr(self, f.name)!r}"
            for f in fields(self)
            if getattr(saved, f.name) is not None and getattr(saved, f.name) != getattr(self, f.name)
        ]
