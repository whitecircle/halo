"""Per-rank optimizer-state shards: :class:`OptimizerShardStore` writes them with their ``optimizer_meta.pt``
and restores them behind its gates."""

from __future__ import annotations

import glob
import logging
import os
from functools import partial

import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.tensor import DTensor

from halo.checkpoint.format import (
    OPTIMIZER_META_FILE,
    OPTIMIZER_SHARD_GLOB,
    OPTIMIZER_STATE_FILES,
    optimizer_shard_filename,
)
from halo.checkpoint.parallel.context import CheckpointLoadContext
from halo.checkpoint.parallel.optimizer_meta import (
    OptimizerStateFingerprint,
    SavedOptimizerMeta,
    capture_pp_stage_partition,
    read_optimizer_meta,
    write_optimizer_meta,
)
from halo.checkpoint.parallel.sidecars import restore_lr_scheduler
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.hardware import available_host_ram_bytes
from halo.log import KEY_PREVIEW_COUNT
from halo.models.unwrapping import unwrap_toolkit_wrappers
from halo.runtime.collectives import barrier, broadcast_from_rank0
from halo.runtime.consensus import (
    DeferredRankFailure,
    RankConsensus,
    gather_rank_reasons,
    gathered_failure_summary,
    rank0_verdict,
    rank_consensus,
)
from halo.runtime.filesystem import is_fs_save_rank, is_output_shared_filesystem
from halo.runtime.load_coordination import joined_node_load, resolve_load_concurrency
from halo.runtime.ranks import get_global_rank, get_global_world_size, get_local_world_size, is_global_main_process

logger = logging.getLogger(__name__)


def _tracked_fqns(osd) -> set[str]:
    """The param FQNs a saved shard's ``param_groups`` say the saving optimizer tracked: the one key space
    that survives both halves of a deduplicated save, which strips ``state`` only."""
    shard = osd if isinstance(osd, dict) else {}
    return {
        fqn
        for group in (shard.get("param_groups") or ())
        if isinstance(group, dict)
        for fqn in (group.get("params") or ())
        if isinstance(fqn, str)
    }


def _reimpose_group_settings(param_groups: list[dict], settings: list[dict]) -> str | None:
    """Write this run's ``settings`` back over the param groups ``set_optimizer_state_dict`` rebuilt; why a
    group was left on the shard's values (a drifted group count), or None. Restores every pair it can
    either way, so a caller already unwinding another error keeps it."""
    for group, live in zip(param_groups, settings, strict=False):
        group.update(live)
    if len(param_groups) == len(settings):
        return None
    return (
        f"the restored optimizer holds {len(param_groups)} param group(s) but this run built "
        f"{len(settings)}, so the unmatched groups would step on the checkpoint's hyperparameters"
    )


def _concurrent_node_ranks(max_concurrent: int | None) -> int:
    """How many of this node's ranks :func:`~halo.runtime.load_coordination.joined_node_load` admits at once."""
    local_world_size = get_local_world_size()
    admitted = resolve_load_concurrency(max_concurrent, local_world_size)
    return local_world_size if admitted == 0 else min(admitted, local_world_size)


def _warn_if_low_host_ram(optimizer, doing: str, max_concurrent: int | None) -> None:
    """Warn, never raise, when host RAM is below an upper bound on the optimizer-state host copies this node holds
    at once: this rank's (two moments per param, a DTensor at its local shard) times the ranks the node's load
    throttle admits together (``max_concurrent``). Running out is an opaque OOM kill mid-checkpoint."""
    available = available_host_ram_bytes()
    if optimizer is None or available is None:
        return
    per_rank = sum(
        2 * (p.data.to_local().numel() if isinstance(p.data, DTensor) else p.data.numel()) * p.data.element_size()
        for group in optimizer.param_groups
        for p in group["params"]
    )
    ranks = _concurrent_node_ranks(max_concurrent)
    if available < ranks * per_rank:
        logger.warning(
            f"[rank {get_global_rank()}] Host MemAvailable ({available / 1e9:.1f} GB) is below the "
            f"~{ranks * per_rank / 1e9:.1f} GB upper-bound estimate of the optimizer state {ranks} rank(s) of this "
            f"node hold at once for {doing} (~{per_rank / 1e9:.1f} GB each); the host copies may OOM-kill the "
            f"run. Lower max_concurrent_loading to admit fewer ranks at a time."
        )


def _reject_unsharded_resume_over_optimizer_shards(checkpoint: str | None) -> None:
    """COLLECTIVE (rank 0's probe). Refuse per-rank shards on a non-sharded resume, whose base loader would
    silently restore no optimizer state."""
    if checkpoint is None:
        return
    if rank0_verdict(
        lambda: bool(
            glob.glob(os.path.join(checkpoint, OPTIMIZER_SHARD_GLOB))
            or os.path.isfile(os.path.join(checkpoint, OPTIMIZER_META_FILE))
        ),
        f"Probing {checkpoint} for per-rank optimizer shards",
    ):
        raise RuntimeError(
            f"{checkpoint} carries per-rank optimizer shards (optimizer_shard_*.pt / optimizer_meta.pt), but "
            "this run resumes without FSDP2/pure-TP sharding and the base Trainer's loader cannot read them — "
            "it would silently restore no optimizer state while weights, step and LR schedule resume. Resume "
            "under the sharded topology that wrote the checkpoint, or delete every optimizer_shard_*.pt and "
            "optimizer_meta.pt to explicitly accept a cold optimizer."
        )


def expert_replica_writer(model) -> tuple[int, frozenset[str]]:
    """``(rank whose shard carries the replicated expert state, the FQNs it carries)``; ``(this rank, empty)``
    when nothing is replicated. Multi-group EP's groups are DP replicas whose FSDP-ignored experts hold
    bit-identical moments, so the replica group's lowest rank writes them for its peers.
    """
    ep_layers = named_ep_layers(model)
    ep_config = ep_layers[0][1].ep_config if ep_layers else None
    if ep_config is None or ep_config.experts_fsdp_managed or not ep_config.needs_expert_grad_sync:
        return get_global_rank(), frozenset()
    keys = frozenset(
        f"{layer_name}.{param_name}"
        for layer_name, module in ep_layers
        for param_name, _param in module.named_parameters()
    )
    return min(ep_config.expert_replica_ranks), keys


class OptimizerShardStore:
    """Per-rank optimizer-state shards: the sharded save and the gated resume (:attr:`writes_shards` modes);
    every other mode keeps the base Trainer's ``optimizer.pt``."""

    def __init__(self, ctx: CheckpointLoadContext):
        self.ctx = ctx
        # The shards' FQNs are relative to the model inside the toolkit wrappers.
        self.model = unwrap_toolkit_wrappers(ctx.model)

    @property
    def writes_shards(self) -> bool:
        """Whether this run keeps its optimizer state as per-rank shards: mixin-managed FSDP2, and pure TP,
        whose base ``optimizer.pt`` would load rank 0's state into every TP rank."""
        return self.ctx.fsdp_wrapped or self.ctx.parallelism_config.is_tp_mode

    def save(self, output_dir: str) -> None:
        """COLLECTIVE. Write this rank's optimizer view to its shard, a node's ranks a batch at a time as the
        restore reads them (``max_concurrent_loading``), and ``optimizer_meta.pt`` once every shard is on disk. Any
        rank's failure raises on every rank: the caller rotates the previous checkpoint away only after a save
        that returned."""
        ctx = self.ctx
        max_concurrent = getattr(ctx.parallelism_config, "max_concurrent_loading", None)
        # A rank leaving early while its peers block on the closing barrier() hangs the job.
        try:
            _warn_if_low_host_ram(ctx.optimizer, "the cpu_offload of the shard save", max_concurrent)
            # Deferred: a raise on one rank would take it to the closing barrier() while its peers vouch for an
            # incomplete set with the meta. The node's ranks enter the throttle in turn, so nothing in it may
            # be a collective.
            guard = DeferredRankFailure(f"optimizer shard save to {output_dir}")
            with joined_node_load(f"Writing the per-rank optimizer shards to {output_dir}", max_concurrent):
                guard.run(partial(self._write_shard, output_dir))

            # What makes "meta present ⇒ complete shard set" true, which lets resume call a missing shard torn.
            # Over the store: the throttled batches and the slowest shard on shared storage outlast any collective.
            guard.store_reject("checkpoint/optimizer_shards")

            pp_partition = (
                capture_pp_stage_partition(self.model, ctx.parallelism_config)
                if ctx.parallelism_config.is_pp_mode
                else None
            )

            # One meta writer per node (read per node on resume), fenced like the shard write.
            if is_fs_save_rank():
                guard.run(
                    partial(write_optimizer_meta, output_dir, ctx.parallelism_config, ctx.optimizer, pp_partition)
                )
                if guard.reason is None:
                    logger.info(f"✓ Saved per-rank optimizer shards ({get_global_world_size()} ranks) to {output_dir}")
            guard.reject()
        finally:
            barrier()

    def _write_shard(self, output_dir: str) -> None:
        """This rank's optimizer view, offloaded to host memory, written to its shard without the replicated expert
        state it does not write (:meth:`_drop_replicated_state`); the host copy is freed on return. Rank-local.

        Raises:
            RuntimeError: this rank holds no optimizer, or its state cannot be produced under the live sharding.
        """
        ctx = self.ctx
        if ctx.optimizer is None:
            raise RuntimeError("This rank holds no optimizer to save.")
        try:
            reshard_fsdp2_modules(ctx.model)
            osd = get_optimizer_state_dict(
                self.model, ctx.optimizer, options=StateDictOptions(full_state_dict=False, cpu_offload=True)
            )
        except Exception as e:
            raise RuntimeError(
                f"This rank could not produce its sharded optimizer state ({type(e).__name__}: {e}), so the "
                "checkpoint would carry none while rotation removed the previous complete one. Known trigger: an "
                "optimizer whose state_dict refuses the live sharding (FlashAdamW on unevenly-sharded DTensors). Set "
                "save_only_model: true to write weights-only checkpoints deliberately."
            ) from e
        self._drop_replicated_state(osd)
        torch.save(osd, os.path.join(output_dir, optimizer_shard_filename(get_global_rank())))

    def _drop_replicated_state(self, osd) -> None:
        """Strip the replicated expert moments (``state`` only) from every shard but their writer's; skipped,
        warned, on a non-shared output filesystem, where a follower could not read the writer's shard."""
        writer_rank, keys = expert_replica_writer(self.model)
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

    def load(self, checkpoint: str | None) -> None:
        """COLLECTIVE. Restore the optimizer state and the LR scheduler on resume.

        A fingerprint mismatch or no shards anywhere warm-restarts; a meta without fingerprint and misplaced
        or torn shards raise; state that provably exists but cannot be restored (a failed read or apply, an
        interrupted save, a base ``optimizer.pt`` with no shards) raises unless ``allow_optimizer_warm_restart``.
        Under PP every warm restart raises once any shard is present. Each verdict is consensus'd.
        """
        if not self.writes_shards or checkpoint is None:
            return self._resume_unsharded(checkpoint)
        shard_path = os.path.join(checkpoint, optimizer_shard_filename(get_global_rank()))
        shards = rank_consensus(os.path.exists(shard_path))
        # Stage-local FQNs map drifted moments onto the wrong layers, so PP raises instead of warm-restarting.
        pp_strict = self.ctx.parallelism_config.is_pp_mode and shards.any_ok
        if not self._gate_saved_shards(checkpoint, shards, pp_strict=pp_strict):
            return
        osd, read_failure = self._read_local_state(checkpoint, shard_path)
        if self._settle_restore_failure(
            checkpoint, read_failure, f"Reading the per-rank optimizer shards from {checkpoint}", pp_strict=pp_strict
        ):
            return
        self._apply_restored_state(checkpoint, osd, pp_strict=pp_strict)

    def _resume_unsharded(self, checkpoint: str | None) -> None:
        """Hand a non-sharded resume to the base Trainer's ``optimizer.pt`` path, refusing a checkpoint whose
        per-rank shards it cannot read."""
        _reject_unsharded_resume_over_optimizer_shards(checkpoint)
        self.ctx.super_load_optimizer_and_scheduler(checkpoint)
        # The base Trainer restores scheduler.pt only alongside an optimizer file. Rank 0's verdict:
        # restore_lr_scheduler is collective.
        optimizer_file_present = checkpoint is not None and any(
            os.path.isfile(os.path.join(checkpoint, name)) for name in OPTIMIZER_STATE_FILES
        )
        if checkpoint is not None and not broadcast_from_rank0(optimizer_file_present):
            restore_lr_scheduler(self.ctx.lr_scheduler, self.ctx.optimizer, checkpoint)

    def _gate_saved_shards(self, checkpoint: str, shards: RankConsensus, *, pp_strict: bool) -> bool:
        """COLLECTIVE. The meta and fingerprint gates ahead of the shard read: True when every rank holds a shard
        the saved topology vouches for; a failed gate raises or warm-restarts and returns False."""
        ctx = self.ctx
        # Topology gates run before the presence verdict, so a topology change reads as change, not torn.
        saved = read_optimizer_meta(checkpoint, shard_any=shards.any_ok)

        # A meta that cannot vouch for its shards is an interrupted save, not a topology change.
        if not rank_consensus(saved.torn is None).all_ok:
            self._refuse_unless_opted_in(
                checkpoint,
                f"Optimizer resume from {checkpoint}: {saved.torn or 'reported by another rank'} on at least one rank",
                "The meta is written only after every rank's shard, so these shards are the torn set of an "
                "interrupted save (or of one whose meta was deleted alone), which no gate can tell from "
                "another run's. Resume from an earlier complete checkpoint",
                pp_strict=pp_strict,
            )
            return False

        if not rank_consensus(saved.topology_ok).all_ok:
            self._strict_or_warm_restart(
                checkpoint,
                strict=pp_strict,
                strict_msg=(
                    f"PP optimizer resume from {checkpoint}: optimizer_meta.pt's rank-shard count does not match "
                    f"the current world_size ({get_global_world_size()}; this rank saw {saved.num_ranks}) on at "
                    "least one rank. PP resume requires the identical topology that wrote the checkpoint. Delete "
                    "every optimizer_shard_*.pt and optimizer_meta.pt to explicitly accept an optimizer warm "
                    "restart."
                ),
                warm_msg=(
                    f"Optimizer checkpoint rank-shard count does not match the current world_size "
                    f"({get_global_world_size()}) on at least one rank (this rank saw {saved.num_ranks}). "
                    f"Warm restart."
                ),
            )
            return False

        # A world fact before any rank branches on it: a non-shared filesystem holds one meta per node.
        fingerprints = rank_consensus(saved.fingerprint is not None)
        if shards.any_ok and not fingerprints.all_ok:
            raise RuntimeError(
                f"Optimizer resume from {checkpoint}: the per-rank optimizer shards carry no topology fingerprint "
                "in optimizer_meta.pt on at least one rank. Pre-fingerprint checkpoints are not supported — the "
                "shards are raw local layouts (DTensor shards + plain EP tensors) and nothing records the sharding "
                "that produced them, so a matching rank count alone would restore another topology's moments onto "
                "this run's params. Delete every optimizer_shard_*.pt and optimizer_meta.pt to resume the weights, "
                "step and LR schedule with a fresh optimizer."
            )

        live_fp = OptimizerStateFingerprint.capture(ctx.parallelism_config, ctx.optimizer, get_global_world_size())
        mismatched = live_fp.mismatches(saved.fingerprint) if saved.fingerprint is not None else []
        if not rank_consensus(not mismatched).all_ok:
            self._strict_or_warm_restart(
                checkpoint,
                strict=pp_strict,
                strict_msg=(
                    f"PP optimizer resume from {checkpoint}: topology fingerprint mismatch — the saved per-rank "
                    "optimizer shards were written under a different topology/optimizer and cannot be mapped onto "
                    f"this run's stages. Mismatched fields: {mismatched or ['(reported by another rank)']}. Resume "
                    "under the identical topology, or delete every optimizer_shard_*.pt and optimizer_meta.pt to "
                    "explicitly accept an optimizer warm restart."
                ),
                warm_msg=(
                    f"Optimizer topology fingerprint mismatch at {checkpoint} — the saved per-rank optimizer shards "
                    "were written under a different topology/optimizer and cannot be mapped onto this run. Warm "
                    "restart: optimizer reinitialized from scratch (weights, LR scheduler, and trainer state still "
                    f"resume). Mismatched fields: {mismatched or ['(reported by another rank)']}"
                ),
            )
            return False

        if pp_strict:
            self._reject_pp_partition_drift(checkpoint, saved)

        if not shards.all_ok:
            self._resolve_absent_shards(checkpoint, shard_any=shards.any_ok, fp_any=fingerprints.any_ok)
            return False
        return True

    def _refuse_unless_opted_in(self, checkpoint: str, failure: str, consequence: str, *, pp_strict: bool) -> None:
        """Raise ``failure`` with its consequence and remedy, or warm-restart under ``allow_optimizer_warm_restart``
        (outside PP): the state provably exists, so dropping it is opt-in. Reached from a consensus'd verdict."""
        remedy = (
            "delete every optimizer_shard_*.pt and optimizer_meta.pt to explicitly accept an optimizer warm restart"
            if pp_strict
            else "set allow_optimizer_warm_restart: true to resume with a freshly initialized optimizer"
        )
        self._strict_or_warm_restart(
            checkpoint,
            strict=pp_strict or not self.ctx.allow_optimizer_warm_restart,
            strict_msg=f"{failure}. {consequence}, or {remedy}.",
            warm_msg=(
                f"{failure}. Warm restart (allow_optimizer_warm_restart): optimizer reinitialized from "
                f"scratch on every rank; weights, LR scheduler and trainer state still resume."
            ),
        )

    def _strict_or_warm_restart(self, checkpoint: str, *, strict: bool, strict_msg: str, warm_msg: str) -> None:
        """A failed gate's outcome: raise when ``strict``, warm-restart otherwise. Reached from a consensus'd
        verdict, so the collectives inside :meth:`_warm_restart` stay in step."""
        if strict:
            raise RuntimeError(strict_msg)
        self._warm_restart(checkpoint, warm_msg)

    def _warm_restart(self, checkpoint: str, msg: str) -> None:
        """COLLECTIVE. Drop the optimizer state on every rank; the structure-independent LR schedule still
        restores."""
        self._discard_optimizer_state()
        if is_global_main_process():
            logger.warning(msg)
        restore_lr_scheduler(self.ctx.lr_scheduler, self.ctx.optimizer, checkpoint)
        barrier()

    def _discard_optimizer_state(self) -> None:
        """Clear the moments and gradients a ``set_optimizer_state_dict`` attempt materialized: its zero-LR init
        step leaves even a failed attempt fully zeroed, Muon's nested scalar optimizer included."""
        optimizer = self.ctx.optimizer
        if optimizer is None:
            return
        optimizer.state.clear()
        scalar = getattr(optimizer, "scalar_optimizer", None)
        if scalar is not None:
            scalar.state.clear()
        optimizer.zero_grad(set_to_none=True)

    def _reject_pp_partition_drift(self, checkpoint: str, saved: SavedOptimizerMeta) -> None:
        """COLLECTIVE. Refuse a saved PP layer split other than the live one, which stage-local FQNs would map
        onto the wrong layers at identical shapes."""
        config = self.ctx.parallelism_config
        live_range = list(self.model.layer_range)
        saved_entry = None
        saved_partition = saved.pp_stage_partition
        if isinstance(saved_partition, list | tuple) and len(saved_partition) == config.pp_size:
            saved_entry = saved_partition[config.pp_rank]
        partition_ok = saved.fingerprint is not None and saved_entry is not None and list(saved_entry) == live_range
        if not rank_consensus(partition_ok).all_ok:
            raise RuntimeError(
                f"PP optimizer resume from {checkpoint}: the saved pp_stage_partition ({saved_partition}) is absent "
                f"or does not match the live layer partition (stage {config.pp_rank} holds layers {live_range}) on "
                "at least one rank. The shards only restore under the identical layer split that wrote them. "
                "Resume under the saving topology, or delete every optimizer_shard_*.pt and optimizer_meta.pt to "
                "explicitly accept an optimizer warm restart."
            )

    def _resolve_absent_shards(self, checkpoint: str, *, shard_any: bool, fp_any: bool) -> None:
        """COLLECTIVE. Not every rank found its own shard: raise where the state provably exists, warm-restart
        where nothing proves it was ever written."""
        if shard_any and fp_any:
            # A fingerprint-matched save wrote one shard per rank, so a missing subset is torn.
            raise RuntimeError(
                f"Optimizer shards at {checkpoint} are torn: present on some ranks, missing on others, under a "
                "matching topology fingerprint. On non-shared filesystems this also happens when a restart maps "
                "ranks onto different nodes (shards are node-local). Resume from a complete checkpoint, restore "
                "the original rank→node placement, or delete every optimizer_shard_*.pt (and optimizer_meta.pt) to "
                "explicitly accept an optimizer warm restart."
            )
        # Other ranks' shards or a matching meta: the state was written and is misplaced (a permuted
        # rank→node placement). Any one rank's evidence raises the world.
        local_evidence = fp_any or bool(glob.glob(os.path.join(checkpoint, OPTIMIZER_SHARD_GLOB)))
        if rank_consensus(local_evidence).any_ok:
            raise RuntimeError(
                f"No rank found its own optimizer shard at {checkpoint}, but optimizer state WAS written there "
                "(other ranks' optimizer_shard_*.pt files and/or a topology-matched optimizer_meta.pt are "
                "present). On a non-shared filesystem this means the restart mapped every rank onto a node holding "
                "none of its shards (shards are node-local); warm-restarting would silently reset the optimizer "
                "moments. Restore the original rank→node placement, or delete every optimizer_shard_*.pt and "
                "optimizer_meta.pt to explicitly accept an optimizer warm restart."
            )
        # A sharded save writes no base optimizer file: this one is another layout's state.
        replicated = any(os.path.isfile(os.path.join(checkpoint, name)) for name in OPTIMIZER_STATE_FILES)
        if rank_consensus(replicated).any_ok:
            return self._refuse_unless_opted_in(
                checkpoint,
                f"Optimizer resume from {checkpoint}: it holds a replicated "
                f"{' / '.join(OPTIMIZER_STATE_FILES)} but no per-rank optimizer shards",
                "That file is a non-sharded run's state, or the rank-0 view an earlier Halo release's sharded "
                "save wrote before its per-rank shards and was interrupted there; neither restores into this "
                "run's shards. Resume from an earlier complete checkpoint",
                pp_strict=False,
            )
        self._warm_restart(
            checkpoint,
            f"No optimizer shards at {checkpoint} — the checkpoint carries no optimizer state (saved with "
            "save_only_model; a shard save that could not produce state fails the checkpoint instead of writing "
            "one without). Warm restart: optimizer reinitialized from scratch.",
        )

    def _read_local_state(self, checkpoint: str, shard_path: str) -> tuple[object | None, str | None]:
        """COLLECTIVE. This rank's optimizer state (its own shard plus the replica writer's) and why it could
        not be read, under the node's load throttle (:func:`~halo.runtime.load_coordination.joined_node_load`):
        both reads land in host RAM, and every follower of a replica group opens the writer's file."""
        max_concurrent = getattr(self.ctx.parallelism_config, "max_concurrent_loading", None)
        _warn_if_low_host_ram(self.ctx.optimizer, f"restoring {os.path.basename(shard_path)}", max_concurrent)
        with joined_node_load(f"Reading the per-rank optimizer shards from {checkpoint}", max_concurrent):
            osd, failure = self._read_shard(shard_path)
            if failure is None:
                failure = self._merge_replicated_state(checkpoint, osd)
        if failure is not None:
            logger.warning(f"[rank {get_global_rank()}] {failure}")
        return osd, failure

    def _read_shard(self, shard_path: str) -> tuple[object | None, str | None]:
        """An optimizer shard, or why it could not be read (a crash mid-save leaves a truncated one).
        Rank-local; the caller joins it."""
        try:
            return torch.load(shard_path, map_location="cpu", weights_only=False), None
        except Exception as e:
            return None, f"Unreadable optimizer shard {shard_path}: {type(e).__name__}: {e}"

    def _merge_replicated_state(self, checkpoint: str, osd) -> str | None:
        """Fold the replica writer's expert moments into this rank's shard; why it could not, or None. Rank-local:
        a failure rides the shard-read verdict. A shard holding state for none of the replicated FQNs was
        deduplicated (the save strips them all at once); merging none of the FQNs it tracked means the writer
        spells them differently, which would resume every expert param at zero moments.
        """
        writer_rank, keys = expert_replica_writer(self.model)
        state = osd.get("state") if isinstance(osd, dict) else None
        if writer_rank == get_global_rank() or not isinstance(state, dict):
            return None
        # No state at all is a pre-first-step checkpoint, not a deduplicated shard.
        if not keys or not state or keys & set(state):
            return None
        path = os.path.join(checkpoint, optimizer_shard_filename(writer_rank))
        if not os.path.exists(path):
            return (
                f"This shard carries no state for its replicated expert params — rank {writer_rank} holds them for "
                f"the whole EP replica group — but {os.path.basename(path)} is not in {checkpoint}. On a non-shared "
                "filesystem the restart mapped this rank onto a node without the writer's shard; restore the "
                "original rank->node placement or resume from a shared filesystem."
            )
        writer_osd, failure = self._read_shard(path)
        if failure is not None:
            return failure
        merged = {key: value for key, value in (writer_osd.get("state") or {}).items() if key in keys}
        state.update(merged)
        tracked = keys & _tracked_fqns(osd)
        if not merged and tracked:
            return (
                f"Restored none of the {len(tracked)} replicated expert parameter(s) this shard was deduplicated "
                f"against from {os.path.basename(path)}: the writer's shard carries no state under any of them "
                f"(first: {sorted(tracked)[:KEY_PREVIEW_COUNT]}). The writer's state is keyed differently from this "
                "run's expert parameter names, so resuming would silently reinitialize every expert moment."
            )
        return None

    def _settle_restore_failure(
        self, checkpoint: str, local_failure: str | None, what: str, *, pp_strict: bool
    ) -> bool:
        """COLLECTIVE. Whether ``what`` failed on any rank; when it did, every rank raises, or warm-restarts under
        the opt-in (:meth:`_refuse_unless_opted_in`)."""
        failure = gathered_failure_summary(gather_rank_reasons(local_failure), what)
        if failure is None:
            return False
        self._refuse_unless_opted_in(
            checkpoint,
            failure,
            "The shards match this run's topology, so resuming without them would reset every optimizer "
            "moment while weights, step and LR schedule resume. Fix the cause (each failing rank's log "
            "carries its error) and resume again",
            pp_strict=pp_strict,
        )
        return True

    def _apply_restored_state(self, checkpoint: str, osd, *, pp_strict: bool) -> None:
        """COLLECTIVE. Write the read state into the live optimizer once every rank's covers its trainable
        params, then restore the LR scheduler."""
        # Trainable FQNs the shard restores nothing for, which strict=False would quietly reinitialize. Tracked
        # but stateless params (never received a gradient) and an empty state (pre-first-step) are legitimate.
        shard = osd if isinstance(osd, dict) else {}
        state_keys = set(shard.get("state") or {})
        trainable = {name for name, param in self.model.named_parameters() if param.requires_grad}
        missing_fqns = sorted(trainable - _tracked_fqns(shard) - state_keys) if state_keys else []
        if not rank_consensus(not missing_fqns).all_ok:
            raise RuntimeError(
                f"Optimizer resume from {checkpoint}: trainable parameter(s) have no saved optimizer state on at "
                "least one rank — resume would silently reinitialize their moments (Adam's step counter included) "
                "while the rest restore. This rank: "
                f"{missing_fqns[:KEY_PREVIEW_COUNT] or '(none; reported by another rank)'}. This usually means the "
                "checkpoint was written under a different parameter layout (e.g. a different use_grouped_gemm, "
                "model revision, or pipeline stage split). Resume from a checkpoint saved under the identical "
                "topology, or delete every optimizer_shard_*.pt and optimizer_meta.pt to explicitly accept an "
                "optimizer warm restart."
            )

        restore = DeferredRankFailure(f"Restoring the per-rank optimizer state from {checkpoint}")
        restore.run(partial(self._apply_shard_state, osd))
        if restore.reason is not None:
            # Freed first, so a rank that ran out of memory can allocate the gather below.
            self._discard_optimizer_state()
        if self._settle_restore_failure(checkpoint, restore.reason, restore.what, pp_strict=pp_strict):
            return
        if is_global_main_process():
            logger.info(f"✓ Restored per-rank optimizer state from {checkpoint}")

        restore_lr_scheduler(self.ctx.lr_scheduler, self.ctx.optimizer, checkpoint)

        barrier()

    def _apply_shard_state(self, osd) -> None:
        """COLLECTIVE (DTensor). Write this rank's shard into the live optimizer; the caller defers a raise to
        its gathered verdict."""
        # strict=False: params that require grad but never accrue one (GptOss sinks) are absent.
        options = StateDictOptions(full_state_dict=False, broadcast_from_rank0=False, strict=False)
        # Hyperparameters belong to this run: the restore rebuilds param_groups from the shard and can
        # drop keys the live optimizer reads (AdamWBF16's ``betas``).
        live_group_settings = [
            {k: v for k, v in group.items() if k != "params"} for group in self.ctx.optimizer.param_groups
        ]
        try:
            set_optimizer_state_dict(self.model, self.ctx.optimizer, optim_state_dict=osd, options=options)
        except BaseException as exc:
            # A failed init step leaves every group at lr=0; the restore error stays the one raised.
            if drift := _reimpose_group_settings(self.ctx.optimizer.param_groups, live_group_settings):
                exc.add_note(drift)
            raise
        if drift := _reimpose_group_settings(self.ctx.optimizer.param_groups, live_group_settings):
            raise RuntimeError(drift)
        # strict= governs key presence, not shape. Moments only: packed states (FlashAdamW) differ legitimately.
        for param, state in self.ctx.optimizer.state.items():
            for key in ("exp_avg", "exp_avg_sq"):
                value = state.get(key)
                if torch.is_tensor(value) and value.shape != param.shape:
                    raise RuntimeError(
                        f"restored optimizer moment {key!r} has shape {tuple(value.shape)} but its "
                        f"parameter has {tuple(param.shape)} — the checkpoint does not match the model"
                    )
