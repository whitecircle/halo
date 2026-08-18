"""Per-rank optimizer-state shards — the other half of a resume, beside :mod:`.loader`'s weights.

:class:`OptimizerShardStore` owns both directions of the sharded optimizer state: the per-rank
``optimizer_shard_XXXXX.pt`` write with its ``optimizer_meta.pt`` topology fingerprint, and the
gated restore that reads them back. The LR scheduler rides along, restored on every resume path.
Reads a :class:`CheckpointLoadContext` (never the trainer) so collective/rank invariants stay
explicit.
"""

from __future__ import annotations

import glob
import logging
import os
import traceback
from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.tensor import DTensor

from src.checkpoint.format import OPTIMIZER_STATE_FILES, SCHEDULER_STATE_FILE
from src.distributed.checkpoint.context import CheckpointLoadContext
from src.distributed.checkpoint.coordination import KEY_PREVIEW_COUNT, all_ranks_ok, consensus_read
from src.distributed.checkpoint.fingerprint import OptimizerStateFingerprint
from src.distributed.expert_parallel.base_layer import find_ep_layers
from src.distributed.filesystem import sequential_load_within_node
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.runtime import (
    DeferredRankFailure,
    barrier,
    broadcast_from_rank0,
    fs_aware_save_rank,
    get_global_rank,
    get_global_world_size,
    is_global_main_process,
    is_output_shared_filesystem,
    rank_consensus,
)
from src.hardware import available_host_ram_bytes
from src.models.structure import unwrap_model

logger = logging.getLogger(__name__)

# The warm-restart remedy names these, so writer, reader and message must mean the same files.
_OPTIMIZER_SHARD_FMT = "optimizer_shard_{rank:05d}.pt"
_OPTIMIZER_SHARD_GLOB = _OPTIMIZER_SHARD_FMT.replace("{rank:05d}", "*")  # derived — cannot drift
_OPTIMIZER_META_FILE = "optimizer_meta.pt"


def expert_replica_writer(model) -> tuple[int, frozenset[str]]:
    """``(rank whose shard carries the replicated expert state, the FQNs it carries)``.

    Under multi-group EP the EP groups are DP replicas: ranks sharing an ``ep_rank`` hold the same
    expert slice, their gradients are averaged across ``expert_replica_group``, and the optimizer is
    deterministic (AdamWBF16 draws stochastic rounding from the rank-synchronized ``_SR_RNG``), so
    their moments are bit-identical. Those params are the FSDP-IGNORED ones — the whole EP module,
    router included — so no reduce-scatter splits them and every rank would otherwise write the same
    bytes. The lowest rank of the replica group writes them; its peers strip them and read them back
    from that rank's shard.

    ``(this rank, empty)`` whenever nothing is replicated: no EP layers, experts FSDP-managed
    (``ep_group_size == 1`` with ``fsdp_shard_ep1_experts`` — already sharded), or one EP group.
    """
    ep_layers = find_ep_layers(model)
    ep_config = ep_layers[0][1].ep_config if ep_layers else None
    if ep_config is None or ep_config.experts_fsdp_managed or not ep_config.needs_expert_grad_sync:
        return get_global_rank(), frozenset()
    keys = frozenset(
        f"{layer_name}.{param_name}"
        for layer_name, module in ep_layers
        for param_name, _param in module.named_parameters()
    )
    return min(ep_config.expert_replica_ranks), keys


def _tracked_fqns(osd) -> set[str]:
    """The param FQNs a saved shard's ``param_groups`` say the saving optimizer tracked.

    ``param_groups`` stay whole in every shard — the expert-replica dedup strips ``state`` only — so
    this is the one key space that survives both halves of a deduplicated save, and both gates that
    need "what the optimizer knew about" read it here rather than re-walking it.
    """
    shard = osd if isinstance(osd, dict) else {}
    return {
        fqn
        for group in (shard.get("param_groups") or ())
        if isinstance(group, dict)
        for fqn in (group.get("params") or ())
        if isinstance(fqn, str)
    }


def _warn_if_low_host_ram(optimizer, doing: str) -> None:
    """Warn — never raise — when host RAM looks too small for this rank's optimizer-state host copy.

    Both the shard save (``cpu_offload=True``) and the shard restore (``torch.load`` to cpu)
    materialize the full local optimizer state in host RAM, where running out surfaces as an opaque
    OOM kill mid-checkpoint. Upper-bound estimate: two moment tensors per param at the param's own
    dtype — AdamWBF16 keeps bf16 moments (4 B/param), fp32 AdamW fp32 ones (8 B/param) — with
    DTensor params counted at their LOCAL shard. Errs high for momentum-only (Muon) and packed
    (FlashAdamW) states.
    """
    available = available_host_ram_bytes()
    if optimizer is None or available is None:
        return
    estimated = sum(
        2 * (p.data.to_local().numel() if isinstance(p.data, DTensor) else p.data.numel()) * p.data.element_size()
        for group in optimizer.param_groups
        for p in group["params"]
    )
    if available < estimated:
        logger.warning(
            f"[rank {get_global_rank()}] Host MemAvailable ({available / 1e9:.1f} GB) is below the "
            f"~{estimated / 1e9:.1f} GB upper-bound estimate of this rank's optimizer state for "
            f"{doing}; the host copy may OOM-kill the run (peer ranks on this node add their own)."
        )


@dataclass(frozen=True)
class _SavedOptimizerMeta:
    """What ``optimizer_meta.pt`` says about the save, as this rank could read it.

    ``topology_ok`` is the rank-count gate's local verdict — False also when the meta is missing
    beside shards or unreadable, since neither the rank-count nor the fingerprint gate can run on
    state it cannot read. The rest are the raw saved fields, absent (``None`` / ``-1``) when the
    meta did not carry them, which the caller's gates each read differently.
    """

    topology_ok: bool
    num_ranks: int = -1
    fingerprint: OptimizerStateFingerprint | None = None
    pp_stage_partition: object | None = None


def _apply_scheduler_lr(lr_scheduler, optimizer) -> None:
    """Push the restored schedule's LR into the live optimizer's param groups.

    ``LRScheduler.load_state_dict`` only updates the scheduler's own ``__dict__``, and HF steps the
    optimizer BEFORE the scheduler — so without this the first resumed step runs at the LR
    ``_initial_step()`` left in ``param_groups``: ~0 under a warmup (wasted), the full base LR for a
    warmup-free schedule resumed deep into decay. The shard cannot supply it either, since
    :meth:`OptimizerShardStore._apply_shard_state` deliberately re-imposes this run's param-group
    settings over the saved ones.
    """
    last_lr = getattr(lr_scheduler, "_last_lr", None)
    if optimizer is None or not last_lr:
        return
    if len(last_lr) != len(optimizer.param_groups):
        logger.warning(
            f"Restored LR schedule tracks {len(last_lr)} param group(s) but the optimizer has "
            f"{len(optimizer.param_groups)} — leaving the live learning rates untouched; the first "
            f"resumed step runs at the schedule's step-0 LR."
        )
        return
    for group, lr in zip(optimizer.param_groups, last_lr, strict=True):
        group["lr"] = lr


class OptimizerShardStore:
    """Per-rank optimizer-state shards: the sharded save, the gated resume, the LR scheduler.

    Consumes the same :class:`CheckpointLoadContext` as
    :class:`~src.distributed.checkpoint.loader.CheckpointLoader`, whose weight restore is the other
    half of a resume. Sharded modes — mixin-managed FSDP2 (EP/CP runs included) and pure TP — write
    one shard per rank plus an FS-aware ``optimizer_meta.pt``, and restore them only under a matching
    topology fingerprint; every other mode falls through to the base Trainer's ``optimizer.pt``.
    """

    def __init__(self, ctx: CheckpointLoadContext):
        self.ctx = ctx

    @staticmethod
    def _reject_sharded_optimizer_resume(checkpoint: str | None) -> None:
        """Raise uniformly (rank-0 check, broadcast) when a non-sharded resume targets per-rank
        optimizer shards. The base Trainer's loader recognizes only ``optimizer.pt``/``.bin``, so
        it would restore NO optimizer state while weights, trainer step and LR schedule resume —
        Adam moments silently reset, with no warning anywhere."""
        if checkpoint is None:
            return
        local_present = is_global_main_process() and bool(
            glob.glob(os.path.join(checkpoint, _OPTIMIZER_SHARD_GLOB))
            or os.path.isfile(os.path.join(checkpoint, _OPTIMIZER_META_FILE))
        )
        if broadcast_from_rank0(local_present):
            raise RuntimeError(
                f"{checkpoint} carries per-rank optimizer shards (optimizer_shard_*.pt / "
                f"optimizer_meta.pt), but this run resumes without FSDP2/pure-TP sharding and the "
                f"base Trainer's loader cannot read them — it would silently restore no optimizer "
                f"state while weights, step and LR schedule resume. Resume under the sharded "
                f"topology that wrote the checkpoint, or delete every optimizer_shard_*.pt and "
                f"optimizer_meta.pt to explicitly accept a cold optimizer."
            )

    def load(self, checkpoint: str | None) -> None:
        """Restore optimizer state and the LR scheduler on resume.

        Sharded modes (mixin-managed FSDP2, incl. EP/CP runs, and pure TP) resume from per-rank
        ``optimizer_shard_XXXXX.pt``, gated by the topology fingerprint in ``optimizer_meta.pt``:
        match → full restore via ``set_optimizer_state_dict``; mismatch → warm restart naming the
        differing fields; shards with no fingerprint at all (a pre-fingerprint checkpoint) → raise.
        Shards absent on every rank = warm restart only when nothing proves state was written; other
        ranks' shard files or a fingerprint-matched meta = misplaced/permuted shards → raise, as does
        a subset absent under a matching fingerprint (torn). Under PP the gates that would
        warm-restart raise instead once any shard is present (see ``pp_strict`` below), so deleting
        every shard is the explicit opt-in. Non-sharded modes (single process, replicated DDP) fall
        through to the base Trainer's ``optimizer.pt`` path.

        Reads as the gate sequence it is: every step below is entered on every rank in this order,
        because each verdict is consensus'd and the restore itself issues DTensor collectives — a
        rank that skips one strands its peers in the next.
        """
        ctx = self.ctx
        # Pure TP (dp=1, no FSDP2) also needs per-rank shards: the base path loads rank 0's optimizer.pt
        # into every TP rank — same shapes, different weight shards.
        pure_tp = ctx.is_tp_mode and ctx.tp_size > 1 and not ctx.fsdp_wrapped
        if (not ctx.fsdp_wrapped and not pure_tp) or checkpoint is None:
            return self._resume_unsharded(checkpoint)

        rank = get_global_rank()
        shard_path = os.path.join(checkpoint, _OPTIMIZER_SHARD_FMT.format(rank=rank))

        # Every restore decision below is collective (set_optimizer_state_dict issues DTensor collectives),
        # and topology gates run BEFORE the presence check so a topology change reads as change, not torn.
        shard_all, shard_any = rank_consensus(os.path.exists(shard_path))
        saved = self._read_saved_meta(os.path.join(checkpoint, _OPTIMIZER_META_FILE), shard_any=shard_any)

        # Under PP the gates below RAISE instead of warm-restarting: shards keyed by stage-LOCAL FQNs let
        # topology drift map moments onto the wrong layers.
        pp_strict = ctx.is_pp_mode and shard_any

        if not all_ranks_ok(saved.topology_ok):
            return self._strict_or_warm_restart(
                checkpoint,
                pp_strict=pp_strict,
                strict_msg=(
                    f"PP optimizer resume from {checkpoint}: optimizer_meta.pt is unreadable or its "
                    f"rank-shard count does not match the current world_size "
                    f"({get_global_world_size()}; this rank saw {saved.num_ranks}) on at least one rank. "
                    f"PP resume requires the identical topology that wrote the checkpoint. Delete "
                    f"every optimizer_shard_*.pt and optimizer_meta.pt to explicitly accept an "
                    f"optimizer warm restart."
                ),
                warm_msg=(
                    f"Optimizer checkpoint rank-shard count does not match the current world_size "
                    f"({get_global_world_size()}) on at least one rank (this rank saw {saved.num_ranks}). "
                    f"Warm restart."
                ),
            )

        # A shard set whose meta carries no fingerprint is a pre-fingerprint checkpoint: unsupported,
        # because nothing then proves the shards were written under this run's sharding and the
        # rank-count gate above admits a permuted restore at the same world size. Consensus'd as a
        # WORLD fact before any rank branches on it: on a non-shared filesystem the meta is written
        # once per node, so a heterogeneous set would otherwise split the ranks across the gates
        # below — a watchdog hang instead of a diagnostic.
        fp_all, fp_any = rank_consensus(saved.fingerprint is not None)
        if shard_any and not fp_all:
            raise RuntimeError(
                f"Optimizer resume from {checkpoint}: the per-rank optimizer shards carry no topology "
                f"fingerprint in optimizer_meta.pt on at least one rank. Pre-fingerprint checkpoints "
                f"are not supported — the shards are raw local layouts (DTensor shards + plain EP "
                f"tensors) and nothing records the sharding that produced them, so a matching rank "
                f"count alone would restore another topology's moments onto this run's params. Delete "
                f"every optimizer_shard_*.pt and optimizer_meta.pt to resume the weights, step and LR "
                f"schedule with a fresh optimizer."
            )

        # Same-topology gate: per-rank shards are raw local layouts, valid only under the sharding
        # that wrote them.
        live_fp = OptimizerStateFingerprint.capture(ctx.parallelism_config, ctx.optimizer, get_global_world_size())
        mismatched = live_fp.mismatches(saved.fingerprint) if saved.fingerprint is not None else []
        if not all_ranks_ok(not mismatched):
            return self._strict_or_warm_restart(
                checkpoint,
                pp_strict=pp_strict,
                strict_msg=(
                    f"PP optimizer resume from {checkpoint}: topology fingerprint mismatch — the "
                    f"saved per-rank optimizer shards were written under a different "
                    f"topology/optimizer and cannot be mapped onto this run's stages. Mismatched "
                    f"fields: {mismatched or ['(reported by another rank)']}. Resume under the "
                    f"identical topology, or delete every optimizer_shard_*.pt and "
                    f"optimizer_meta.pt to explicitly accept an optimizer warm restart."
                ),
                warm_msg=(
                    f"Optimizer topology fingerprint mismatch at {checkpoint} — the saved per-rank "
                    f"optimizer shards were written under a different topology/optimizer and cannot be "
                    f"mapped onto this run. Warm restart: optimizer reinitialized from scratch "
                    f"(weights, LR scheduler, and trainer state still resume). Mismatched fields: "
                    f"{mismatched or ['(reported by another rank)']}"
                ),
            )

        if pp_strict:
            self._reject_pp_partition_drift(checkpoint, saved)

        if not shard_all:
            return self._resolve_absent_shards(checkpoint, shard_any=shard_any, fp_any=fp_any)

        osd, local_load_ok = self._read_local_state(checkpoint, shard_path)
        if not all_ranks_ok(local_load_ok):
            return self._strict_or_warm_restart(
                checkpoint,
                pp_strict=pp_strict,
                strict_msg=(
                    f"PP optimizer resume from {checkpoint}: torn/unreadable optimizer shard on at "
                    f"least one rank under a matching topology. Resume from a complete checkpoint, "
                    f"or delete every optimizer_shard_*.pt and optimizer_meta.pt to explicitly "
                    f"accept an optimizer warm restart."
                ),
                warm_msg="Torn/unreadable optimizer shard on at least one rank. Warm restart.",
            )

        missing_fqns = self._fqns_without_saved_state(osd)
        if not all_ranks_ok(not missing_fqns):
            raise RuntimeError(
                f"Optimizer resume from {checkpoint}: trainable parameter(s) have no saved optimizer "
                f"state on at least one rank — resume would silently reinitialize their moments "
                f"(Adam's step counter included) while the rest restore. This rank: "
                f"{missing_fqns[:KEY_PREVIEW_COUNT] or '(none; reported by another rank)'}. "
                f"This usually means the checkpoint was written "
                f"under a different parameter layout (e.g. a different use_grouped_gemm, model "
                f"revision, or pipeline stage split). Resume from a checkpoint saved under the "
                f"identical topology, or delete every optimizer_shard_*.pt and optimizer_meta.pt to "
                f"explicitly accept an optimizer warm restart."
            )

        if not all_ranks_ok(self._apply_shard_state(osd)):
            return self._warm_restart(
                checkpoint,
                "Failed to restore sharded optimizer state on at least one rank (its log carries "
                "the exception). Warm restart.",
            )
        if is_global_main_process():
            logger.info(f"✓ Restored per-rank optimizer state from {checkpoint}")

        self.restore_lr_scheduler(checkpoint)

        barrier()

    def _resume_unsharded(self, checkpoint: str | None) -> None:
        """Hand a non-sharded resume (single process, replicated DDP) to the base Trainer's
        ``optimizer.pt`` path, refusing a checkpoint whose per-rank shards it cannot read."""
        self._reject_sharded_optimizer_resume(checkpoint)
        self.ctx.super_load_optimizer_and_scheduler(checkpoint)
        # The base Trainer restores scheduler.pt only alongside an optimizer file; without this the
        # LR re-warms from step 0 while the dataloader skips ahead. Rank 0's verdict, not a local
        # stat: restore_lr_scheduler all-reduces, so entering it must be a collective decision.
        optimizer_file_present = checkpoint is not None and any(
            os.path.isfile(os.path.join(checkpoint, name)) for name in OPTIMIZER_STATE_FILES
        )
        if checkpoint is not None and not broadcast_from_rank0(optimizer_file_present):
            self.restore_lr_scheduler(checkpoint)

    @staticmethod
    def _read_saved_meta(meta_path: str, *, shard_any: bool) -> _SavedOptimizerMeta:
        """Read ``optimizer_meta.pt`` — rank-local, no collective; every gate on it is consensus'd."""
        # Shards WITHOUT a meta file are ungated state (no rank-count or fingerprint gate can run), so a
        # missing meta is a failed gate, not a passed one.
        if not os.path.exists(meta_path):
            return _SavedOptimizerMeta(topology_ok=not shard_any)
        saved_ranks = -1
        # A torn meta file must not raise bare here and desync the consensus in the caller (hang).
        try:
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            saved_ranks = meta.get("num_ranks", -1)
            return _SavedOptimizerMeta(
                topology_ok=saved_ranks == get_global_world_size(),
                num_ranks=saved_ranks,
                fingerprint=OptimizerStateFingerprint.from_dict(meta.get("fingerprint")),
                pp_stage_partition=meta.get("pp_stage_partition"),
            )
        except Exception as e:
            logger.warning(f"[rank {get_global_rank()}] Unreadable optimizer_meta.pt at {meta_path}: {e}")
            return _SavedOptimizerMeta(topology_ok=False, num_ranks=saved_ranks)

    def _strict_or_warm_restart(self, checkpoint: str, *, pp_strict: bool, strict_msg: str, warm_msg: str) -> None:
        """The verdict every failed restore gate shares: raise under PP, warm-restart everywhere else.

        PP shards are keyed by stage-LOCAL FQNs, so a drifted topology maps moments onto the wrong
        layers at identical shapes — there is no safe warm arm, only the explicit opt-in of deleting
        the shards. Callers reach this from a consensus'd verdict, so both arms are rank-uniform and
        the collectives inside :meth:`_warm_restart` stay in step.
        """
        if pp_strict:
            raise RuntimeError(strict_msg)
        self._warm_restart(checkpoint, warm_msg)

    def _reject_pp_partition_drift(self, checkpoint: str, saved: _SavedOptimizerMeta) -> None:
        """COLLECTIVE. Same-partition gate: with stage-LOCAL FQNs, an identical world/pp_size at a
        different layer split maps moments onto the wrong layers at identical shapes. Each rank
        checks its own range."""
        config = self.ctx.parallelism_config
        live_range = list(unwrap_model(self.ctx.model).layer_range)
        saved_entry = None
        saved_partition = saved.pp_stage_partition
        if isinstance(saved_partition, list | tuple) and len(saved_partition) == config.pp_size:
            saved_entry = saved_partition[config.pp_rank]
        partition_ok = saved.fingerprint is not None and saved_entry is not None and list(saved_entry) == live_range
        if not all_ranks_ok(partition_ok):
            raise RuntimeError(
                f"PP optimizer resume from {checkpoint}: the saved pp_stage_partition "
                f"({saved_partition}) is absent or does not match the live layer partition "
                f"(stage {config.pp_rank} holds layers {live_range}) on at least one rank. The "
                f"shards only restore under the identical layer split that wrote them. Resume "
                f"under the saving topology, or delete every optimizer_shard_*.pt and "
                f"optimizer_meta.pt to explicitly accept an optimizer warm restart."
            )

    def _read_local_state(self, checkpoint: str, shard_path: str) -> tuple[object | None, bool]:
        """This rank's optimizer state — its own shard plus the replica writer's — read under the
        node's load throttle.

        Both reads land in host RAM (``map_location="cpu"``), so an unthrottled resume peaks at
        ``local_world_size`` × (own shard + writer shard) per node, the same host-RAM wall
        :func:`_warn_if_low_host_ram` warns about one rank at a time. The writer shard adds the
        second pressure the throttle bounds: under multi-group EP every follower of a replica group
        reads the SAME file, so at ``ep8`` on 512 GPUs each of the 8 writer shards is opened by 63
        followers — the throttle turns that fan-in into ``max_concurrent_loading`` readers per node
        instead of one per rank, and the followers of one writer are spread one-per-node by the
        replica layout, so nothing serializes behind a single node's slot.

        Entered on EVERY rank, unconditionally: the throttle is a store phase whose participants are
        the node's local ranks, and the branches inside it are rank-dependent (only followers merge,
        and under PP a stage can hold no EP layer at all) — gating entry on any of them would leave
        a peer waiting out the store timeout on a key nobody writes.
        """
        with sequential_load_within_node(
            "optimizer_shard", max_concurrent=getattr(self.ctx.parallelism_config, "max_concurrent_loading", None)
        ):
            osd, local_load_ok = self._read_shard(shard_path)
            return osd, local_load_ok and self._merge_replicated_state(checkpoint, osd)

    def _merge_replicated_state(self, checkpoint: str, osd) -> bool:
        """Fold the replica writer's expert moments into this rank's shard; False when it could not.

        Rank-local by design — it adds no gate to the consensus'd sequence in :meth:`load`; a False
        here rides the shard-readability verdict that already follows, so an unreachable writer shard
        warm-restarts the whole world (raises under PP) instead of leaving this rank alone with a
        half-restored optimizer. The rank that hit it logs the exact file.

        Self-describing, by an ALL-or-nothing test: the save strips every replicated FQN at once, so a
        shard holding state for none of them was deduplicated, while one holding state for any of them
        carries its own copy — every shard set written before the dedup, and every
        non-shared-filesystem save — and reads nothing extra. "Any missing" would be the wrong test: a
        tracked param that never received a gradient is legitimately absent from both shards.

        The merge is then verified against the same shard's ``param_groups``, which the save keeps
        whole: a merge that restored NONE of the replicated FQNs the saving optimizer TRACKED means
        the writer's shard spells them differently from :func:`expert_replica_writer`, which the
        missing-FQN gate cannot see (it subtracts every tracked FQN out) and which would otherwise
        resume every expert param at zero moments while logging a full restore. Tracked-none is the
        legitimate arm (frozen base experts under expert-LoRA), and it merges nothing either.
        """
        writer_rank, keys = expert_replica_writer(unwrap_model(self.ctx.model))
        state = osd.get("state") if isinstance(osd, dict) else None
        if writer_rank == get_global_rank() or not isinstance(state, dict):
            return True
        # No state at all is a pre-first-step checkpoint, not a deduplicated shard.
        if not keys or not state or keys & set(state):
            return True
        path = os.path.join(checkpoint, _OPTIMIZER_SHARD_FMT.format(rank=writer_rank))
        if not os.path.exists(path):
            logger.warning(
                f"[rank {get_global_rank()}] This shard carries no state for its replicated expert "
                f"params — rank {writer_rank} holds them for the whole EP replica group — but "
                f"{os.path.basename(path)} is not in {checkpoint}. On a non-shared filesystem the "
                f"restart mapped this rank onto a node without the writer's shard; restore the "
                f"original rank->node placement or resume from a shared filesystem."
            )
            return False
        writer_osd, ok = self._read_shard(path)
        if not ok:
            return False
        merged = {key: value for key, value in (writer_osd.get("state") or {}).items() if key in keys}
        state.update(merged)
        tracked = keys & _tracked_fqns(osd)
        if not merged and tracked:
            logger.warning(
                f"[rank {get_global_rank()}] Restored NONE of the {len(tracked)} replicated expert "
                f"parameter(s) this shard was deduplicated against from {os.path.basename(path)}: the "
                f"writer's shard carries no state under any of them (first: {sorted(tracked)[:KEY_PREVIEW_COUNT]}). "
                f"The writer's state is keyed differently from this run's expert parameter names, so "
                f"resuming would silently reinitialize every expert moment."
            )
            return False
        return True

    def _resolve_absent_shards(self, checkpoint: str, *, shard_any: bool, fp_any: bool) -> None:
        """COLLECTIVE. No rank found its OWN shard: raise where the state provably exists, warm-restart
        where nothing proves it was ever written."""
        if shard_any and fp_any:
            # A fingerprint-matched save wrote one shard per rank, so a missing subset is torn —
            # restoring here while peers warm-restart diverges the replicas.
            raise RuntimeError(
                f"Optimizer shards at {checkpoint} are torn: present on some ranks, missing on "
                f"others, under a matching topology fingerprint. On non-shared filesystems this "
                f"also happens when a restart maps ranks onto different nodes (shards are "
                f"node-local). Resume from a complete checkpoint, restore the original rank→node "
                f"placement, or delete every optimizer_shard_*.pt (and optimizer_meta.pt) to "
                f"explicitly accept an optimizer warm restart."
            )
        # The checkpoint dir holds other ranks' shard files or a meta whose fingerprint MATCHED this run
        # (the earlier gates warm-restart on any mismatch): the optimizer state was written and is merely
        # misplaced — on a non-shared filesystem, a restart that permuted the rank→node placement
        # wholesale. Consensus'd because each node sees different files; any single rank's evidence must
        # raise the whole world.
        local_evidence = fp_any or bool(glob.glob(os.path.join(checkpoint, _OPTIMIZER_SHARD_GLOB)))
        if rank_consensus(local_evidence)[1]:
            raise RuntimeError(
                f"No rank found its own optimizer shard at {checkpoint}, but optimizer state WAS "
                f"written there (other ranks' optimizer_shard_*.pt files and/or a topology-matched "
                f"optimizer_meta.pt are present). On a non-shared filesystem this means the restart "
                f"mapped every rank onto a node holding none of its shards (shards are node-local); "
                f"warm-restarting would silently reset the optimizer moments. Restore the original "
                f"rank→node placement, or delete every optimizer_shard_*.pt and optimizer_meta.pt "
                f"to explicitly accept an optimizer warm restart."
            )
        self._warm_restart(
            checkpoint,
            f"No optimizer shards at {checkpoint} — the checkpoint carries no optimizer state "
            f"(saved with save_only_model; a shard save that could not produce state fails the "
            f"checkpoint instead of writing one without). Warm restart: optimizer reinitialized "
            f"from scratch.",
        )

    def _read_shard(self, shard_path: str) -> tuple[object | None, bool]:
        """This rank's optimizer shard and whether it read — a crash mid-save leaves a shard that
        exists but is truncated, so presence is not readability. Rank-local; the caller consensus'd."""
        _warn_if_low_host_ram(self.ctx.optimizer, f"restoring {os.path.basename(shard_path)}")
        try:
            return torch.load(shard_path, map_location="cpu", weights_only=False), True
        except Exception as e:
            logger.warning(f"[rank {get_global_rank()}] Unreadable optimizer shard {shard_path}: {e}")
            return None, False

    def _fqns_without_saved_state(self, osd) -> list[str]:
        """Trainable FQNs the shard restores nothing for — the layout-drift signal.

        ``set_optimizer_state_dict(strict=False)`` quietly reinitializes any param absent from the
        shard, so a rename or layer-count change restores partially and still logs success. A param
        the saving optimizer TRACKED but gave no state (never received a gradient) is a legitimate
        absence — the shard's own param_groups name those, so layout drift is what lands in neither
        set, and an empty state at all is a pre-first-step checkpoint.
        """
        shard = osd if isinstance(osd, dict) else {}
        state_keys = set(shard.get("state") or {})
        if not state_keys:
            return []
        required = {
            name for name, param in unwrap_model(self.ctx.model).named_parameters() if param.requires_grad
        } - _tracked_fqns(shard)
        return sorted(required - state_keys)

    def _apply_shard_state(self, osd) -> bool:
        """COLLECTIVE (DTensor). Write this rank's shard into the live optimizer; False when it failed
        here, which the caller turns into a world-wide warm restart — a rank-local fallback would
        leave some replicas restored and others reinitialized."""
        try:
            # strict=False: params that require grad but never accrue one (GptOss sinks) are absent from
            # the shard and strict=True would abort the whole restore over them.
            options = StateDictOptions(full_state_dict=False, broadcast_from_rank0=False, strict=False)
            # Hyperparameters belong to THIS run: set_optimizer_state_dict rebuilds param_groups from the
            # shard and can drop keys the live optimizer reads each step (AdamWBF16's ``betas``).
            live_group_settings = [
                {k: v for k, v in group.items() if k != "params"} for group in self.ctx.optimizer.param_groups
            ]
            # unwrap_model as on the save side: the shard's FQNs are relative to the inner model.
            set_optimizer_state_dict(
                unwrap_model(self.ctx.model), self.ctx.optimizer, optim_state_dict=osd, options=options
            )
            # strict: a rebuilt group count that drifted would leave tail groups on the shard's stale LR
            # with no error; the raise lands in the consensus-gated except below.
            for group, settings in zip(self.ctx.optimizer.param_groups, live_group_settings, strict=True):
                group.update(settings)
            # strict= governs key presence, not shape, so a wrong-shape moment would surface at the first
            # step() — or truncate silently on the bf16 triton path, which sizes its mask off the PARAM.
            # Moments only: packed/quantized states (FlashAdamW) legitimately differ in shape.
            for param, state in self.ctx.optimizer.state.items():
                for key in ("exp_avg", "exp_avg_sq"):
                    value = state.get(key)
                    if torch.is_tensor(value) and value.shape != param.shape:
                        raise RuntimeError(
                            f"restored optimizer moment {key!r} has shape {tuple(value.shape)} but its "
                            f"parameter has {tuple(param.shape)} — the checkpoint does not match the model"
                        )
            return True
        except Exception as e:
            logger.warning(f"[rank {get_global_rank()}] Failed to restore sharded optimizer state: {e}")
            return False

    def _warm_restart(self, checkpoint: str, msg: str) -> None:
        """Skip optimizer restore uniformly; the scheduler is still restored (structure-independent —
        skipping it re-warms the LR from step 0 while the dataloader skips ahead).

        Clear any optimizer state first: a failed ``set_optimizer_state_dict`` attempt runs torch's
        ``_init_optim_state`` (a zero-LR step) before it raises, which materializes exp_avg/exp_avg_sq
        for every param — leaving a fully zeroed optimizer that contradicts "reinitialized from
        scratch". The fallback is consensus-gated (rank-uniform), so clearing is safe across replicas.
        """
        if self.ctx.optimizer is not None:
            self.ctx.optimizer.state.clear()
            # Muon keeps 1D-param moments on a nested scalar optimizer, which the zero-LR init step
            # materializes too — clear both or "from scratch" is half-true.
            scalar = getattr(self.ctx.optimizer, "scalar_optimizer", None)
            if scalar is not None:
                scalar.state.clear()
        if is_global_main_process():
            logger.warning(msg)
        self.restore_lr_scheduler(checkpoint)
        barrier()

    def restore_lr_scheduler(self, checkpoint: str) -> bool:
        """Restore the LR scheduler from ``scheduler.pt`` if present; returns True on success.

        Decoupled from optimizer-state restore so it runs on every resume path, including EP/CP.
        """
        ctx = self.ctx

        def _restore(path: str) -> None:
            """Read AND apply inside the joined read: a state dict this scheduler refuses is exactly
            as unusable as a torn file, and both must fail the world the same way."""
            if ctx.lr_scheduler is None:
                return  # nothing to restore into; a present file is then not an error
            ctx.lr_scheduler.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))

        # Absent everywhere is a legitimate warm restart (save_only_model); present on a subset, or
        # unreadable, would re-warm the LR from step 0 while the dataloader skips ahead — those are
        # the two raises consensus_read owns.
        _state, path = consensus_read(
            os.path.join(checkpoint, SCHEDULER_STATE_FILE),
            _restore,
            what=SCHEDULER_STATE_FILE,
            checkpoint=checkpoint,
            remedy=(
                f" Or delete {SCHEDULER_STATE_FILE} to explicitly accept an LR-schedule warm restart from step 0."
            ),
        )
        if path is None or ctx.lr_scheduler is None:
            return False
        # Rank-uniform, and only once every rank restored: the LR the schedule says this step runs at.
        _apply_scheduler_lr(ctx.lr_scheduler, ctx.optimizer)
        if is_global_main_process():
            logger.info(f"✓ Restored LR scheduler state from {checkpoint}")
        return True

    def _drop_replicated_state(self, osd) -> None:
        """Strip the replicated expert moments from every rank but their writer's shard.

        ``param_groups`` stay whole in every shard — FQN lists plus this run's hyperparameters, not
        per-replica bytes — so the missing-FQN gate on resume still reads the key space it always
        did. Skipped on a non-shared output filesystem, where the writer's shard is not on the
        follower's node: the duplication stands there, warned about rather than turned into a shard
        set that resumes on one node only.
        """
        writer_rank, keys = expert_replica_writer(unwrap_model(self.ctx.model))
        state = osd.get("state") if isinstance(osd, dict) else None
        if not keys or not isinstance(state, dict):
            return
        if not is_output_shared_filesystem():
            if is_global_main_process():
                logger.warning(
                    "Non-shared output filesystem: the FSDP-ignored expert optimizer state is "
                    "replicated across the EP groups (they are DP replicas), but a follower cannot "
                    "read the writer's shard off another node — every rank writes its own copy. "
                    "Write checkpoints to a shared output directory to deduplicate them."
                )
            return
        if writer_rank == get_global_rank():
            return
        for key in keys:
            state.pop(key, None)

    def save(self, output_dir: str) -> None:
        """Save per-rank optimizer state shards (mixin FSDP2 — incl. EP/CP runs — and pure TP).

        Each rank writes its local view (FSDP2 DTensor shards + plain EP expert tensors, keyed by
        param FQN via ``get_optimizer_state_dict``) to ``optimizer_shard_XXXXX.pt``; the FS-aware
        save rank(s) also write ``optimizer_meta.pt`` carrying the rank count and the topology
        fingerprint the resume gate validates. Shards live inside the ``checkpoint-N`` directory,
        so ``save_total_limit`` rotation removes them with the checkpoint.

        Under multi-group EP the EP groups are DP replicas holding identical expert moments in
        FSDP-ignored plain tensors, so only the replica group's lowest rank keeps them in its shard
        (:func:`expert_replica_writer`); its peers strip them and read them back from that shard.

        A rank that cannot produce its optimizer state fails the whole checkpoint, uniformly: the
        caller's stale-``optimizer.pt`` delete and its deferred rotation both run only on a save that
        returned, so raising here is what keeps the previous — complete — checkpoint.
        """
        ctx = self.ctx
        # try/finally: a rank returning early while peers block on the closing barrier() hangs the job.
        try:
            rank = get_global_rank()
            osd = None
            if ctx.optimizer is not None:
                _warn_if_low_host_ram(ctx.optimizer, "the cpu_offload of the shard save")
                options = StateDictOptions(full_state_dict=False, cpu_offload=True)
                try:
                    reshard_fsdp2_modules(ctx.model)
                    # unwrap_model: under CP the FQNs are relative to the inner model, and resolving them
                    # against the wrapper raises.
                    osd = get_optimizer_state_dict(unwrap_model(ctx.model), ctx.optimizer, options=options)
                except Exception as e:
                    logger.warning(
                        f"[rank {rank}] Failed to get sharded optimizer state dict: "
                        f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                    )
            # All-or-nothing, an ABSENT optimizer included: a rank writing while a peer failed leaves a
            # set the resume gate must call torn. Uniform because the verdict is a world all-reduce —
            # and a RAISE, not a skip: the caller deletes the base Trainer's optimizer.pt and rotates
            # the previous checkpoint away right after this returns, so a silent skip trades the last
            # complete checkpoint for an optimizer-less one at exit code 0.
            if not all_ranks_ok(osd is not None):
                raise RuntimeError(
                    f"Optimizer shard save to {output_dir} failed: at least one rank could not produce "
                    f"its optimizer state (that rank's log carries the exception). This checkpoint would "
                    f"carry no optimizer state while rotation removed the previous complete one. Known "
                    f"trigger: an optimizer whose state_dict refuses the live sharding (FlashAdamW on "
                    f"unevenly-sharded DTensors). Set save_only_model: true to write weights-only "
                    f"checkpoints deliberately."
                )

            self._drop_replicated_state(osd)

            # A raise on one shard write would take that rank straight to the closing barrier() while its
            # peers still have this one plus the meta ahead — they would vouch for an incomplete set, then hang.
            guard = DeferredRankFailure(f"optimizer shard write to {output_dir}")
            guard.run(partial(torch.save, osd, os.path.join(output_dir, _OPTIMIZER_SHARD_FMT.format(rank=rank))))

            # The collective that makes "meta present ⇒ complete shard set" true, which is what lets
            # resume call "fingerprint matches but shard missing" torn.
            guard.reject()

            # PP: partition read off the LIVE stages, never the split formula the gate must catch.
            pp_partition = None
            if ctx.is_pp_mode:
                config = ctx.parallelism_config
                lo, hi = unwrap_model(ctx.model).layer_range
                entries: list[tuple[int, int, int] | None] = [None] * get_global_world_size()
                dist.all_gather_object(entries, (config.pp_rank, lo, hi))
                pp_partition = [None] * config.pp_size
                for pp_rank, stage_lo, stage_hi in entries:
                    pp_partition[pp_rank] = [stage_lo, stage_hi]

            # optimizer_meta.pt is read per-node on resume — one writer per node, for non-shared FS.
            # Fenced like the shard write: a meta failure (ENOSPC on one node's writer) must reach
            # every rank at the closing reject, not raise on the writer alone mid-collective-region.
            def _write_meta() -> None:
                fingerprint = OptimizerStateFingerprint.capture(
                    ctx.parallelism_config, ctx.optimizer, get_global_world_size()
                )
                meta = {"num_ranks": get_global_world_size(), "fingerprint": fingerprint.to_dict()}
                if pp_partition is not None:
                    meta["pp_stage_partition"] = pp_partition
                torch.save(meta, os.path.join(output_dir, _OPTIMIZER_META_FILE))

            if fs_aware_save_rank():
                guard.run(_write_meta)
                if guard.reason is None:
                    logger.info(f"✓ Saved per-rank optimizer shards ({get_global_world_size()} ranks) to {output_dir}")
            guard.reject()
        finally:
            barrier()
