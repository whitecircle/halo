"""Gradient synchronization and global-norm clipping for the distributed trainers.

The collectives that make a step's gradients correct before the optimizer sees them: the FSDP2 wrap
each parallel mode needs, the QLoRA / deferred-EP / TP-replicated post-backward sweeps, the
``accelerator.clip_grad_norm_`` replacements that compute a true global norm across EP/TP/FSDP
shards, and the step-pre-hooks that run the sweeps on steps where clipping is disabled.

Mixed into :class:`~src.trainers.mixins.base.DistributedTrainerMixin` as a mixin class rather than a
function module, because every method here reads the trainer's live parallelism state
(``parallelism_config``, ``_ep_config``, ``_device_mesh``, ``_fsdp_wrapped``, the PP groups). The
topology-independent pieces shared by every clip path live in :mod:`src.trainers.mixins.grad_clip`.
"""

from collections.abc import Callable

import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from torch.distributed.tensor import DTensor
from transformers.trainer_utils import IntervalStrategy

from src.distributed.fsdp import (
    IdentityParamSet,
    apply_fsdp2_per_layer,
    create_mixed_precision_policy_v2,
    fsdp2_modules,
    reshard_label,
)
from src.distributed.grad_reduce import reduce_grads_bucketed
from src.distributed.mesh import MeshDim, mesh_dim_names
from src.distributed.runtime import current_device, get_global_world_size
from src.distributed.tensor_parallel.state_dict import tp_sharded_non_dtensor_suffixes
from src.models.structure import model_has_quantized_params
from src.trainers.mixins.grad_clip import (
    bucketed_grad_norm_sq,
    clipping_enabled,
    scale_shards_to_max_norm_,
    trainable_clip_params,
)

logger = get_logger(__name__, log_level="info")


def register_grad_sync_step_hook(
    trainer, flag_attr: str, enabled: Callable[[], bool], action: Callable[[], None]
) -> None:
    """Register ``action`` as a once-per-trainer optimizer step-pre-hook, marked by ``flag_attr``.

    The three grad-sync backstops (QLoRA, deferred EP, TP-replicated) share this shape. Each exists
    because HF skips ``clip_grad_norm_`` at ``max_grad_norm == 0``, and each must sit out torch's
    zero-LR ``_init_optim_state`` step during optimizer-state restore, which carries no gradients.
    ``enabled`` is a callable so its state reads stay behind the ``optimizer is None`` check.
    """
    if trainer.optimizer is None or getattr(trainer, flag_attr, False) or not enabled():
        return

    def _pre_step_hook(optimizer, args, kwargs):
        if getattr(trainer, "_restoring_optimizer_state", False):
            return
        action()

    trainer.optimizer.register_step_pre_hook(_pre_step_hook)
    setattr(trainer, flag_attr, True)


class GradientSyncMixin:
    """Gradient sync and global-norm clipping for :class:`DistributedTrainerMixin`."""

    def _setup_qlora_gradient_sync(self):
        """Enable the post-backward DP average for QLoRA (FSDP2 unusable: quantized base weights
        conflict with DTensor inputs), which :meth:`_sync_qlora_grads` performs.

        A flag, not a hook per parameter: ``register_post_accumulate_grad_hook`` fires only for
        params that received a grad on this rank, so its all-reduce would have rank-local
        membership and hang whenever a microbatch skips a branch on one rank. Per-microbatch firing
        also ignores ``no_sync``, paying a full all-reduce per accumulation step.
        """
        if get_global_world_size() <= 1:
            return
        self._qlora_grad_sync = True

    def _sync_qlora_grads(self) -> None:
        """DP-average every trainable grad once per optimizer step, with structural membership.

        Grad presence is rank-local, so which params to reduce is agreed first with a single
        all-reduce over a presence mask: a param some rank produced a grad for is reduced on every
        rank (zero-filled where absent, which is its true contribution), and one no rank touched is
        skipped everywhere. Materializing a zero grad there would apply weight decay and momentum to
        a parameter the step should have left alone.
        """
        if not getattr(self, "_qlora_grad_sync", False):
            return
        current_step = self.state.global_step
        if getattr(self, "_qlora_sweep_last_step", None) == current_step:
            return
        self._qlora_sweep_last_step = current_step

        params = [p for _name, p in self._top_level_model().named_parameters() if p.requires_grad]
        if not params:
            return
        present = torch.tensor([p.grad is not None for p in params], dtype=torch.uint8, device=current_device())
        dist.all_reduce(present, op=dist.ReduceOp.MAX)

        grads = []
        for param, any_rank_has_grad in zip(params, present.tolist(), strict=True):
            if not any_rank_has_grad:
                continue
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            grads.append(param.grad)
        reduce_grads_bucketed(grads, op=dist.ReduceOp.AVG, fp32=self.parallelism_config.fp32_grad_reduce)

    def _setup_backward_reshard_window(self) -> None:
        """Cache the FSDP2 modules whose post-backward reshard toggles per grad-accum window.

        Empty for every run but ``fsdp_reshard_after_backward=False`` on a wrap applied by this
        mixin, which leaves :meth:`_set_backward_reshard` inert on torch's always-reshard default.
        Call once, after every wrap: ``fully_shard`` is what makes a module an ``FSDPModule``.
        """
        config = self.parallelism_config
        pin = self._fsdp_wrapped and not config.fsdp_reshard_after_backward
        # The outermost model, not ``_top_level_model()``: every mode wraps at or below it.
        self._backward_reshard_modules = fsdp2_modules(self.model) if pin else []
        # fully_shard's own default, and what the modules carry until the first microstep toggles them.
        self._backward_reshard_armed = True
        if self._backward_reshard_modules:
            logger.info(
                f"  ✓ fsdp_reshard_after_backward=False: {len(self._backward_reshard_modules)} FSDP2 "
                f"modules stay unsharded across the grad-accum window (its LAST backward still reshards)"
            )

    def _set_backward_reshard(self, reshard: bool) -> None:
        """Arm FSDP2's post-backward reshard for the microstep about to run.

        torch's ``set_reshard_after_backward`` contract is per grad-accum window: off for microbatches
        1..n-1, on for the last. Left off, ``post_backward`` clears the unsharded params' ``.grad``
        before reduce-scattering onto the sharded DTensors and never re-registers those, so
        ``model.parameters()`` yields grad-less objects the optimizer never captured while
        ``unshard()`` no-ops, hiding the optimizer's update from the next forward.
        """
        if not self._backward_reshard_modules or reshard == self._backward_reshard_armed:
            return
        for module in self._backward_reshard_modules:
            module.set_reshard_after_backward(reshard, recurse=False)
        self._backward_reshard_armed = reshard

    def _setup_ep_gradient_sync(self) -> None:
        """FSDP2 for the EP (and EP+CP) gradient sync: experts FSDP-ignored, everything else sharded.

        One module-tree walk feeds both the presence check and the wrap: ``_ep_fsdp_ignored_modules``
        inspects every parameter's dtype, so deriving it twice doubles that pass over the model.
        """
        config = self.parallelism_config
        # Rank-block width, not the global world (identical without PP).
        if config.stage_world_size <= 1:
            return
        ignored = self._ep_fsdp_ignored_modules()
        if not ignored[0] and config.is_ep_mode:
            raise RuntimeError(
                "EP mode is active but no EP-patched modules found in the model. "
                "This means expert gradient synchronization will not work — experts "
                "on different ranks will silently diverge. Ensure the model was loaded "
                "with EP patching (via load_distributed_model)."
            )
        self._apply_ep_aware_dp_fsdp2(
            self.model,
            ignored=ignored,
            fallback_dp_size=config.stage_world_size,
            dp_replicate_size=config.dp_replicate_size,
            topo=f", HSDP {config.dp_replicate_size}×{config.dp_shard_size}" if config.is_hsdp else "",
        )

    def _setup_ep_tp_gradient_sync(self) -> None:
        """FSDP2 over the loader's 2D (dp, tp) mesh: the EP+TP DP leg."""
        config = self.parallelism_config
        if config.stage_world_size <= 1:
            return
        dp_size = config.data_parallel_size
        if dp_size <= 1:
            logger.info("  DP=1, gradient sync handled by DTensor (TP) and EP hooks")
            return

        # Reuse the loader's 2D (dp, tp) mesh; a fresh 1D mesh would be a duplicate communicator.
        device_mesh = getattr(self.model, "_device_mesh", None)
        if MeshDim.DP not in mesh_dim_names(device_mesh):
            raise RuntimeError(
                "EP+TP gradient sync needs the loader's 2D (dp, tp) DeviceMesh, but the model "
                "carries none — the load/trainer contract broke (load_distributed_model attaches "
                "the mesh when it TP-shards). A hand-rolled DP-group fallback here would mint a "
                "duplicate communicator and silently diverge from the mesh the DTensors shard "
                "over; load the model through load_distributed_model."
            )
        mp_policy = create_mixed_precision_policy_v2(self.args, fp32_master_weights=config.fp32_non_ep_params)
        ignored_set = IdentityParamSet(self._ignored_params(self._ep_fsdp_ignored_modules()[2]) or ())
        apply_fsdp2_per_layer(
            self.model, device_mesh[MeshDim.DP], mp_policy, config.fsdp_reshard_after_forward, ignored_set
        )
        self._fsdp_wrapped = True
        logger.info(
            f"  ✓ FSDP2 ({reshard_label(config.fsdp_reshard_after_forward)}) applied for EP+TP gradient "
            f"sync (DP={dp_size}, reused 2D mesh)"
        )

    def _setup_cp_gradient_sync(self) -> None:
        """FSDP2 (or the QLoRA all-reduce) for the CP-only gradient sync."""
        config = self.parallelism_config
        if config.stage_world_size <= 1:
            return
        # QLoRA skips FSDP2 (quantized weights cannot coexist with DTensor inputs) → AllReduce.
        if model_has_quantized_params(self.model):
            self._reject_fsdp_knobs_under_qlora()
            self._setup_qlora_gradient_sync()
            self._patch_gradient_clipping_for_qlora()
            logger.info("✓ QLoRA gradient sync applied for CP gradient sync")
            return
        self._apply_dp_fsdp2(
            self._top_level_model(),
            config.stage_world_size,
            ignored_modules=self._find_fsdp_incompatible_modules(),
            dp_replicate_size=config.dp_replicate_size,
            topo=f", HSDP {config.dp_replicate_size}×{config.dp_shard_size}" if config.is_hsdp else "",
            detail="applied for CP gradient sync",
        )

    def _patch_gradient_clipping_for_qlora(self):
        """Run the QLoRA DP average before the norm, so the clip coefficient matches on every rank.

        The per-parameter hooks this replaces synced during backward, so clipping already saw
        averaged grads. A post-backward sweep has to be ordered ahead of the norm explicitly, or
        each rank would clip by its own local coefficient and the weights would diverge.
        """
        trainer = self
        base_clip = self.accelerator.clip_grad_norm_

        def qlora_clip_grad_norm_(parameters, max_norm, norm_type=2):
            trainer._sync_qlora_grads()
            return base_clip(parameters, max_norm, norm_type)

        self.accelerator.clip_grad_norm_ = qlora_clip_grad_norm_

    def _patch_gradient_clipping_for_ep(self):
        """Patch gradient clipping to compute true global gradient norm for EP."""
        trainer = self

        def ep_clip_grad_norm_(parameters, max_norm, norm_type=2):
            all_params = trainable_clip_params(parameters)
            # Structural, never grad-presence: the EP branch issues collectives, so the early return
            # must be rank-uniform.
            if not all_params:
                return torch.tensor(0.0, device=current_device())

            device = current_device()

            if trainer._has_ep_layers:
                # Before the norm, so the clip runs on fully-synced grads and matches on every rank.
                trainer._sync_deferred_expert_grads()

                # EP+TP: TP-average replicated grads first, else norms / the shared expert drift across the TP axis.
                if trainer.parallelism_config.is_tp_mode:
                    expert_ids = trainer._get_sharded_expert_param_ids()
                    trainer._sync_tp_replicated_grads([p for p in all_params if id(p) not in expert_ids])

                global_norm = trainer._compute_global_grad_norm()

                # Device-resident: reading the norm back stalls the launch queue; max_norm <= 0 disables clipping (HF).
                # Scale local shards: _foreach_mul_ refuses DTensor + plain EP tensors together.
                if clipping_enabled(max_norm):
                    shards = [
                        g.to_local() if isinstance(g, DTensor) else g
                        for g in (p.grad for p in all_params)
                        if g is not None
                    ]
                    if shards:
                        scale_shards_to_max_norm_(shards, float(max_norm), global_norm)

                return global_norm
            else:
                # No EP layers → torch clip issues no collectives, so a rank-local filter is safe.
                params = [p for p in all_params if p.grad is not None]
                if not params:
                    return torch.tensor(0.0, device=device)
                return torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type=norm_type, foreach=False)

        self.accelerator.clip_grad_norm_ = ep_clip_grad_norm_

    def _sync_deferred_expert_grads(self) -> None:
        """Post-backward EP grad sync for every deferred topology (``EPConfig.defer_grad_sync``).

        Deferred because in-backward hooks would race the EP group's DeepEP combine (multi-group EP
        across nodes) or re-fire per microbatch (PP). Each grad ends as the ``/world_size`` DP
        average over its rank block:
        - expert FFN shards: combine already summed the dispatch group → ``SUM`` over replicas then
          ``/(world_size // expert_tp_size)`` (ETP partners share a batch, so are not DP replicas);
          at R==1 that sum already spans every replica, so only the storage-dtype divide remains;
        - router / replicated EP submodules / plain non-expert params: ``AVG`` over the DP scope;
        - non-expert FSDP shards: replica-group ``AVG`` of the local slice, but only when sharded
          within the EP group — otherwise reduce-scatter already produced the average.

        Collectives issue in a fixed ``named_parameters`` order identical on every rank. No-op
        unless ``defer_grad_sync``.
        """
        ep_cfg = self._ep_config
        if ep_cfg is None or not getattr(ep_cfg, "defer_grad_sync", False):
            return
        replica_group = ep_cfg.expert_replica_group  # None ⇔ one EP group per rank block (R==1)
        # The expert leg (SUM/world_size) is not idempotent: a second pass halves every expert grad.
        current_step = self.state.global_step
        if getattr(self, "_deferred_sweep_last_step", None) == current_step:
            raise RuntimeError(
                f"_sync_deferred_expert_grads ran twice at global_step={current_step}: the expert "
                f"SUM/world_size leg is not idempotent — a second pass halves every expert gradient."
            )
        self._deferred_sweep_last_step = current_step
        world_size = ep_cfg.world_size
        # ETP partners slice one expert and share a batch, so the DP divisor drops the ETP factor.
        expert_divisor = world_size // ep_cfg.expert_tp_size
        fp32 = ep_cfg.fp32_grad_reduce
        expert_ids = self._get_sharded_expert_param_ids()
        ep_ids = self._get_ep_param_ids()

        # Membership must be structural: a grad can be None on some ranks only, so params enter zero-filled.
        expert_sum_grads: list = []
        world_avg_grads: list = []
        replica_avg_grads: list = []

        for _name, param in self._top_level_model().named_parameters():
            if not param.requires_grad:
                continue
            pid = id(param)
            target = world_avg_grads
            if pid in expert_ids:
                target = expert_sum_grads
            elif pid not in ep_ids and isinstance(param.data, DTensor):
                if not ep_cfg.is_deferred_dp:
                    # PP stage: sharded over the block's full DP scope, so reduce-scatter averaged it.
                    continue
                # Valid only when sharded over the EP group (1D mesh); any other mesh averages different slices.
                mesh = param.data.device_mesh
                if mesh.ndim != 1 or mesh.size() != ep_cfg.ep_group_size:
                    raise RuntimeError(
                        f"Deferred cross-replica DP sync expects FSDP sharding over the EP group "
                        f"(1D mesh of {ep_cfg.ep_group_size}), got a {tuple(mesh.shape)} mesh for "
                        f"'{_name}'. This topology must not set is_deferred_dp."
                    )
                target = replica_avg_grads
            # Zero-fill only params that enter a collective, else AdamW gets a zero grad where it had None.
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            target.append(param.grad._local_tensor if target is replica_avg_grads else param.grad)

        if replica_group is not None:
            reduce_grads_bucketed(
                expert_sum_grads, op=dist.ReduceOp.SUM, divisor=expert_divisor, group=replica_group, fp32=fp32
            )
        elif expert_sum_grads:
            # R==1: the combine already summed every replica, so only the storage-dtype divide remains.
            torch._foreach_div_(expert_sum_grads, expert_divisor)
        # DP-scope AVG, not data_parallel_size: the per-rank loss is mean-normalized.
        reduce_grads_bucketed(world_avg_grads, op=dist.ReduceOp.AVG, group=ep_cfg.dp_scope_group, fp32=fp32)
        if replica_avg_grads and replica_group is None:
            # group=None is the world group, which under PP spans stages holding different layers.
            # Unreachable in practice (these grads are collected only under is_deferred_dp, which is
            # what builds the replica group), so this is a contract check rather than a fallback.
            raise RuntimeError(
                f"Deferred cross-replica DP sync collected {len(replica_avg_grads)} sharded expert "
                f"gradient(s) but has no expert replica group to average them over. Reducing over "
                f"the default world group would average across pipeline stages / dispatch groups "
                f"that hold different parameters."
            )
        reduce_grads_bucketed(replica_avg_grads, op=dist.ReduceOp.AVG, group=replica_group, fp32=fp32)

    @staticmethod
    def _sharded_grad_bucket(grad: torch.Tensor, tp_disjoint: bool) -> str:
        """Classify a grad by the group whose ranks tile the whole parameter exactly once.

        - ``"tp1d"``: a 1D ``tp`` mesh (DTensor attention TP), or a plain TP-disjoint slice with no
          FSDP on top (pure TP) → TP group;
        - ``"tp2d"``: a 2D ``(dp, tp)`` mesh, or a hand-sliced sink FSDP-sharded over ``dp``
          (TP + DP: the slices tile over TP, each slice's shards tile over DP) → both axes;
        - ``"dp"``: sharded over a 1D ``dp`` mesh and replicated across TP → DP group only (a
          world/TP reduce would over-count by ``tp_size``).

        ``tp_disjoint`` separates the last two: after FSDP2 wraps a run, a hand-sliced sink and a TP
        replica are both 1D ``dp`` DTensors (see :meth:`_tp_sharded_plain_param_ids`).
        """
        dims = mesh_dim_names(getattr(grad, "device_mesh", None))
        if MeshDim.TP in dims:
            return "tp2d" if len(dims) > 1 else "tp1d"
        if tp_disjoint:
            return "tp2d" if dims else "tp1d"
        return "dp"

    def _reduce_shard_norm_buckets(
        self, tp1d_sq: torch.Tensor, tp2d_sq: torch.Tensor, dp_sq: torch.Tensor
    ) -> torch.Tensor:
        """Sum each TP/DP shard-norm bucket over the ranks that tile it; return the total.

        ``tp1d`` reduces over the TP group, ``tp2d`` over the ``(dp, tp)`` plane (2D shards tile it
        once), ``dp`` over the DP group only (replicated across TP, so a wider reduce over-counts by
        ``tp_size``). Shared by the TP and EP+TP grad-norm paths.

        The ``(dp, tp)`` plane is the pipeline stage, not the world: under PP the other stages hold
        different layers, so a world reduce would add their norms into this stage's bucket and the
        chain reduction would count them again. ``_pp_stage_group`` is ``None`` without PP, i.e. the
        default world group, since the stage is the world at ``pp_size == 1``.
        """
        pdims = self.parallel_dims
        tp_group = pdims.tp_group()
        if tp_group is not None:
            dist.all_reduce(tp1d_sq, op=dist.ReduceOp.SUM, group=tp_group)
        dp_group = pdims.dp_group()
        if dp_group is not None:
            dist.all_reduce(tp2d_sq, op=dist.ReduceOp.SUM, group=self._pp_stage_group)
            dist.all_reduce(dp_sq, op=dist.ReduceOp.SUM, group=dp_group)
        return tp1d_sq + tp2d_sq + dp_sq

    @staticmethod
    def _fsdp_shard_group(mesh) -> dist.ProcessGroup:
        """Process group over which to sum FSDP shard norms for a mesh-sharded parameter.

        1D full-shard mesh → whole DP group. 2D HSDP ``(dp_replicate, dp_shard)`` mesh → the
        ``dp_shard`` sub-group only: each rank holds a shard of the already-replica-reduced grad, so
        summing over the shard dim yields the full norm, and replicas hold identical shards (a
        dp_replicate reduce would over-count).

        Raises on any other shape rather than falling back to the default world group, which is
        wrong wherever it differs from the mesh: under PP it spans every stage, so the reduce would
        mix stages and the chain reduce would count them again.
        """
        if mesh.ndim == 1:
            return mesh.get_group()
        dim_names = mesh_dim_names(mesh)
        if MeshDim.DP_SHARD in dim_names:
            return mesh[MeshDim.DP_SHARD].get_group()
        raise RuntimeError(
            f"Cannot pick an FSDP shard group from a {mesh.ndim}D mesh with dims {dim_names}: "
            f"expected a 1D full-shard mesh or a 2D HSDP mesh carrying {MeshDim.DP_SHARD!r}."
        )

    def _canonical_fsdp_shard_group(self, model, sharded_expert_ids) -> dist.ProcessGroup | None:
        """FSDP shard group for the EP-only / EP+CP grad-norm reduce, derived grad-independently.

        Uses the first non-expert DTensor *parameter*: params exist in identical order on every rank,
        unlike the first non-None grad. With no FSDP-sharded backbone param the reduce has nothing to
        sum and falls back to the pipeline stage group (the world group at ``pp_size == 1``), never a
        cross-stage reduce.
        """
        for _name, param in model.named_parameters():
            if id(param) in sharded_expert_ids:
                continue
            if isinstance(param, DTensor):
                return self._fsdp_shard_group(param.device_mesh)
        return self._pp_stage_group

    def _compute_global_grad_norm(self) -> torch.Tensor:
        """Global gradient norm across all EP ranks for clipping, as a 0-dim device tensor.

        Expert grads are local (different experts per rank); non-expert grads are FSDP2-synced. EP+TP:
        non-expert params are Shard DTensors, so shard norms are local and then all-reduced over the
        TP group. Expert-TP: expert shards summed within the expert-TP group; expert norms aggregate
        across sub-EP (dispatch) groups. Iterates the full top-level model so ``lm_head`` (untied
        models) contributes. Kept on device: reading it back would block the launch queue every step.
        """
        model = self._top_level_model()

        # Only EP/ETP-distributed expert weights: replicated router / shared-expert params would be over-counted.
        sharded_expert_ids = self._get_sharded_expert_param_ids()
        # Plain tensors the TP plan sharded by hand (GptOss sinks): disjoint slices, not replicas.
        tp_sharded_plain_ids = self._tp_sharded_plain_param_ids()

        device = current_device()
        shards: dict[str, list[torch.Tensor]] = {
            name: [] for name in ("expert", "fsdp_full", "tp1d", "tp2d", "dp", "other")
        }

        for _name, param in model.named_parameters():
            if param.grad is None:
                continue

            is_expert = id(param) in sharded_expert_ids
            is_dtensor = isinstance(param.grad, DTensor)
            tp_disjoint = id(param) in tp_sharded_plain_ids
            if is_expert:
                bucket = "expert"
            elif is_dtensor and self._device_mesh is None:
                # EP-only / EP+CP: non-expert params FSDP2-sharded over the DP world.
                bucket = "fsdp_full"
            elif is_dtensor or tp_disjoint:
                # Route by mesh dims plus TP-plan disjointness; the mesh alone cannot distinguish it.
                bucket = self._sharded_grad_bucket(param.grad, tp_disjoint)
            else:
                bucket = "other"
            shards[bucket].append(param.grad._local_tensor if is_dtensor else param.grad)

        norm_sq = bucketed_grad_norm_sq(shards, device=device)
        other_norm_sq = norm_sq["other"]

        # Config-gated (identical branch on all ranks) so collectives never desync.
        if self._device_mesh is not None:
            other_norm_sq = other_norm_sq + self._reduce_shard_norm_buckets(
                norm_sq["tp1d"], norm_sq["tp2d"], norm_sq["dp"]
            )
        elif self._fsdp_wrapped:
            fsdp_dp_group = self._canonical_fsdp_shard_group(model, sharded_expert_ids)
            dist.all_reduce(norm_sq["fsdp_full"], op=dist.ReduceOp.SUM, group=fsdp_dp_group)
            other_norm_sq = other_norm_sq + norm_sq["fsdp_full"]

        # Expert legs read straight off the EP config, which holds those process groups. A dense or
        # non-EP run has no ``_ep_config`` and reduces nothing here.
        global_expert_norm_sq = norm_sq["expert"]
        ep_cfg = self._ep_config
        if ep_cfg is not None:
            if ep_cfg.expert_tp_size > 1 and ep_cfg.expert_tp_group is not None:
                dist.all_reduce(global_expert_norm_sq, op=dist.ReduceOp.SUM, group=ep_cfg.expert_tp_group)

            if ep_cfg.dispatch_ep_group is not None:
                dist.all_reduce(global_expert_norm_sq, op=dist.ReduceOp.SUM, group=ep_cfg.dispatch_ep_group)

            # Experts replicated across EP groups: avoid counting duplicates.
            if ep_cfg.num_ep_groups > 1 and ep_cfg.expert_replica_group is not None:
                dist.all_reduce(global_expert_norm_sq, op=dist.ReduceOp.SUM, group=ep_cfg.expert_replica_group)
                global_expert_norm_sq.div_(ep_cfg.num_ep_groups)

        total_norm_sq = global_expert_norm_sq + other_norm_sq
        if self._pp_chain_group is not None:
            # Every reduce above is stage-scoped; the chain sum over disjoint stages gives the whole model's norm.
            dist.all_reduce(total_norm_sq, op=dist.ReduceOp.SUM, group=self._pp_chain_group)

        return total_norm_sq.sqrt()

    def _patch_gradient_clipping_for_tp(self):
        """Patch gradient clipping for TP-only mode.

        TP-only mixes DTensor params with plain replicated params (experts, norms, embeddings,
        lm_head); accelerate's default ``clip_grad_norm_`` runs ``torch._foreach_norm`` over all at
        once → "mixed torch.Tensor and DTensor". Compute the global L2 norm manually.
        """
        trainer = self

        def tp_clip_grad_norm_(parameters, max_norm, norm_type=2):
            if norm_type != 2:
                raise NotImplementedError(
                    f"TP grad clipping computes an L2 norm over DTensor shards; norm_type={norm_type} "
                    f"would silently return the wrong norm."
                )
            all_params = trainable_clip_params(parameters)
            # Structural, never grad-presence: the sync + norm below issue TP/DP collectives that would desync.
            if not all_params:
                return torch.tensor(0.0, device=current_device())

            # Replicated grads must match across the TP group, else the weights and norm diverge.
            trainer._sync_tp_replicated_grads(all_params)
            params = [p for p in all_params if p.grad is not None]

            total_norm = trainer._compute_tp_grad_norm(params)

            # Device-resident like the EP clip path; scales local shards, mixing DTensor and plain TP grads.
            if clipping_enabled(max_norm) and params:
                shards = [g.to_local() if isinstance(g, DTensor) else g for g in (p.grad for p in params)]
                scale_shards_to_max_norm_(shards, float(max_norm), total_norm)

            return total_norm

        self.accelerator.clip_grad_norm_ = tp_clip_grad_norm_

    def _tp_sharded_plain_param_ids(self) -> set:
        """Ids of params TP sharded as plain tensors: disjoint slices that look like replicas.

        Every plan-driven TP shard is a DTensor whose grad reduces itself; the plain slices are the
        hand-sliced params in ``model._tp_sharded_non_dtensor`` (GptOss sinks). Their grads must not
        be TP-averaged, which would blend another rank's heads into every slice; their norms are
        summed over the TP group instead of counted as replicated. Memoized.
        """
        cached = getattr(self, "_tp_sharded_plain_ids_cache", None)
        if cached is not None:
            return cached
        suffixes = tp_sharded_non_dtensor_suffixes(self._top_level_model())
        ids = {
            id(param)
            for name, param in self._top_level_model().named_parameters()
            if suffixes and name.endswith(suffixes)
        }
        self._tp_sharded_plain_ids_cache = ids
        return ids

    def _tp_per_head_norm_param_ids(self) -> set:
        """Ids of the attention norms whose gradient covers only this rank's heads.

        Registered by :func:`apply_tp_to_attention_only` as names, since FSDP2 replaces Parameter
        objects after TP is applied. Their true gradient is the sum over the TP group, not the
        average the replicated bucket applies: a colwise projection's ``use_local_output=True`` ends
        the DTensor graph, so nothing else reduces them. Memoized.
        """
        cached = getattr(self, "_tp_per_head_norm_ids_cache", None)
        if cached is not None:
            return cached
        model = self._top_level_model()
        names = set(getattr(model, "_tp_per_head_norm_params", None) or ())
        ids = {id(param) for name, param in model.named_parameters() if name in names}
        if names and not ids:
            raise RuntimeError(
                f"TP registered {len(names)} per-head attention norms for the step-time gradient SUM, "
                f"but none of those names resolve against the trainer's model (e.g. {sorted(names)[:3]}). "
                "Their gradients would stay divided across the TP group and the norms would train on a "
                "1/tp_size gradient. The registry is keyed on the model apply_tp_to_attention_only saw."
            )
        self._tp_per_head_norm_ids_cache = ids
        return ids

    def _sync_tp_replicated_grads(self, params: list) -> None:
        """Reduce the grads the TP graph leaves un-reduced, once per optimizer step. Two buckets:

        * **AVG over replicated (non-DTensor) params**: LayerNorms, embeddings, lm_head, pure-TP
          replicated experts. DTensor-sharded grads reduce themselves; these are plain tensors whose
          per-rank grads nothing reduces, so the replicated weights drift apart without this.
        * **SUM over the per-head attention norms** (:meth:`_tp_per_head_norm_param_ids`): each
          rank's gradient covers only its own heads, and DP-sharded ones are reduced on their local
          shard (TP siblings hold the same shard of the same replica).

        Plain tensors the TP plan sharded by hand (:meth:`_tp_sharded_plain_param_ids`) are excluded
        from both, since their per-rank slices are disjoint. No-op without a TP group. A param
        missing a grad gets a zero one (a router tie-break can leave grad=None on one rank only) so
        the collective count stays structural.
        """
        tp_group = self._get_tp_process_group()
        if tp_group is None or dist.get_world_size(group=tp_group) <= 1:
            return
        # Once per step: max_grad_norm == 0 still reaches the clip via _get_grad_norm, and the SUM is not idempotent.
        step = getattr(getattr(self, "state", None), "global_step", None)
        if step is not None and getattr(self, "_tp_sync_last_step", None) == step:
            return
        self._tp_sync_last_step = step
        sharded_plain_ids = self._tp_sharded_plain_param_ids()
        per_head_norm_ids = self._tp_per_head_norm_param_ids()
        replicated_grads = []
        per_head_norm_grads = []
        for p in params:
            if not p.requires_grad or id(p) in sharded_plain_ids:
                continue
            is_per_head_norm = id(p) in per_head_norm_ids
            if (
                not is_per_head_norm
                and isinstance(p.data, DTensor)
                and MeshDim.TP in mesh_dim_names(p.data.device_mesh)
            ):
                # Plan-sharded: the DTensor graph reduces these over the TP mesh itself. A DTensor
                # without a tp dim is FSDP's dp shard of a TP-replicated param; its reduce-scatter
                # runs per TP column, so without the AVG below the TP siblings' replicas are never
                # re-synced and drift apart on any nondeterministic backward.
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            if is_per_head_norm:
                # SUM, not AVG: each rank's gradient covers its own heads, and DP shards the norm identically.
                per_head_norm_grads.append(p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad)
            else:
                # TP siblings hold the same dp shard of the same replica, so reducing local shards
                # is the exact cross-replica sync.
                replicated_grads.append(p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad)
        fp32 = bool(self.parallelism_config.fp32_grad_reduce)
        reduce_grads_bucketed(replicated_grads, op=dist.ReduceOp.AVG, group=tp_group, fp32=fp32)
        reduce_grads_bucketed(per_head_norm_grads, op=dist.ReduceOp.SUM, group=tp_group, fp32=fp32)

    def _compute_tp_grad_norm(self, params: list) -> torch.Tensor:
        """Global L2 gradient norm for TP / TP+DP mode (mixed DTensor + replicated), on device.

        Each DTensor grad's local-shard norm is summed over the ranks tiling it, by sharding:
        - 1D ``tp`` mesh → TP group;
        - 2D ``(dp, tp)`` mesh → the whole ``(dp, tp)`` plane (shards tile it once);
        - 1D ``dp`` mesh (FSDP2-sharded replica) → DP group only (a wider reduce over-counts by tp_size);
        - plain tensor hand-sliced under TP (GptOss sinks) → TP group;
        - plain replicated tensor (DP=1) → counted locally, no reduce.

        Reduces run even with empty ``params`` (zero contribution) so the collectives stay
        rank-uniform. Under PP every reduce is stage-scoped; the chain sum below covers the rest.
        """
        device = current_device()
        shards: dict[str, list[torch.Tensor]] = {name: [] for name in ("tp1d", "tp2d", "dp", "replicated")}
        sharded_plain_ids = self._tp_sharded_plain_param_ids()

        for p in params:
            grad = p.grad
            is_dtensor = isinstance(grad, DTensor)
            tp_disjoint = id(p) in sharded_plain_ids
            # A disjoint slice tiles the tensor once across the TP group; counting it as replicated
            # skews the norm. What lands in ``replicated`` is reduced by nothing.
            sharded = is_dtensor or tp_disjoint
            bucket = self._sharded_grad_bucket(grad, tp_disjoint) if sharded else "replicated"
            shards[bucket].append(grad._local_tensor if is_dtensor else grad)

        norm_sq = bucketed_grad_norm_sq(shards, device=device)
        sharded_norm_sq = self._reduce_shard_norm_buckets(norm_sq["tp1d"], norm_sq["tp2d"], norm_sq["dp"])
        total_norm_sq = sharded_norm_sq + norm_sq["replicated"]
        if self._pp_chain_group is not None:
            dist.all_reduce(total_norm_sq, op=dist.ReduceOp.SUM, group=self._pp_chain_group)
        return total_norm_sq.sqrt()

    def _logs_after_this_step(self) -> bool:
        """Whether ``_maybe_log_save_evaluate`` will log the step now being completed.

        Replicates ``DefaultFlowCallback.on_step_end``, which runs after ``global_step`` is
        incremented, hence ``global_step + 1``. Everything it cannot predict from replicated state
        (an epoch-strategy log, a flag another callback already raised, the final step) answers True
        so no logged metric is dropped, and the predicate reads only rank-uniform trainer state, so
        every rank answers alike.
        """
        args, state = self.args, self.state
        if getattr(self.control, "should_log", False) or args.logging_strategy != IntervalStrategy.STEPS:
            return True
        step = state.global_step + 1
        if args.logging_first_step and step == 1:
            return True
        if state.max_steps and step >= state.max_steps:
            return True
        return step % max(int(state.logging_steps), 1) == 0

    def _get_grad_norm(self, model, grad_norm=None):
        """HF's post-step gradient norm, computed only on the steps that log it.

        With clipping on, the clip's own by-product is passed in and this is a pass-through. With
        ``max_grad_norm <= 0`` HF instead asks for an unclipped norm purely for the log line, which
        costs an all-reduce over the DP/EP/TP/PP groups that is discarded on non-logging steps. The
        grad sweeps that clip path also carries (deferred EP, TP-replicated, QLoRA) are covered by
        their optimizer step-pre-hook backstops.
        """
        if grad_norm is None and not self._logs_after_this_step():
            return None
        return super()._get_grad_norm(model, grad_norm=grad_norm)

    def _register_qlora_grad_sync_hook(self) -> None:
        """Cover the QLoRA sweep when clipping is disabled.

        Idempotent: the sweep's per-step marker makes the second caller a no-op.
        """
        register_grad_sync_step_hook(
            self,
            "_qlora_grad_sync_hook_registered",
            lambda: getattr(self, "_qlora_grad_sync", False),
            self._sync_qlora_grads,
        )

    def _register_deferred_ep_grad_sync_hook(self) -> None:
        """Run the deferred cross-replica sweep every step even when clipping is disabled.

        Backstop for steps that never reach ``ep_clip_grad_norm_``. It must gate on whether the sweep
        already ran this step, never on ``max_grad_norm``: at ``max_grad_norm == 0`` transformers
        still calls ``_get_grad_norm`` → the patched ``clip_grad_norm_``, so the sweep already ran and
        a second pass would trip the non-idempotency guard.
        """

        def _sweep() -> None:
            if getattr(self, "_deferred_sweep_last_step", None) == self.state.global_step:
                return  # already swept this step (ep_clip_grad_norm_ ran)
            self._sync_deferred_expert_grads()

        register_grad_sync_step_hook(
            self,
            "_deferred_ep_grad_sync_hook_registered",
            lambda: getattr(self._ep_config, "defer_grad_sync", False),
            _sweep,
        )

    def _register_tp_replicated_grad_sync_hook(self) -> None:
        """Ensure TP replicated-grad sync runs every step even when clipping is disabled.

        The sync normally lives inside ``tp_clip_grad_norm_``, but HF Trainer skips ``clip_grad_norm_``
        when ``max_grad_norm == 0``, which would let the replicated weights drift across TP ranks.
        Syncs over ``model.parameters()`` (robust to Muon's nested optimizer).
        """

        def _sync() -> None:
            if clipping_enabled(getattr(self.args, "max_grad_norm", None)):
                return  # the clip path (tp_clip_grad_norm_ / ep_clip_grad_norm_) already synced
            # EP experts are rank-owned, not TP replicas, so exclude them; the selection stays
            # structural, never rank-local grad presence.
            expert_ids = self._get_sharded_expert_param_ids()
            self._sync_tp_replicated_grads([p for p in self.model.parameters() if id(p) not in expert_ids])

        register_grad_sync_step_hook(
            self,
            "_tp_grad_sync_hook_registered",
            lambda: self.parallelism_config.is_tp_mode,
            _sync,
        )
