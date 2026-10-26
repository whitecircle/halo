"""Gradient clipping for the distributed trainers: the ``clip_grad_norm_`` each parallel mode installs, the
true global L2 norm they reduce across EP/TP/FSDP/PP shards, and the primitives every clip path shares.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor
from transformers.trainer_utils import IntervalStrategy

from halo.distributed.mesh import MeshDim, dp_group, mesh_dim_names, tp_group
from halo.hardware import current_device
from halo.trainers.parallelism.ep_layers import require_ep_config


def require_l2(norm_type, path: str) -> None:
    """Raise unless ``norm_type`` is 2: every distributed clip sums squared L2 shard norms across
    ranks, so any other norm would come back silently wrong. ``path`` names the clip in the error."""
    if norm_type != 2:
        raise ValueError(
            f"{path} gradient clipping sums squared L2 norms across shards and supports norm_type=2 only, "
            f"got {norm_type}."
        )


def trainable_clip_params(parameters) -> list[nn.Parameter]:
    """The trainable parameters of a clip call, whose ``parameters`` HF/accelerate pass as a module or a
    one-shot iterable. Empty is returned, not short-circuited on: the distributed clips issue
    collectives, so the caller answers with a zero norm rather than skip."""
    if hasattr(parameters, "parameters"):
        parameters = parameters.parameters()
    return [p for p in parameters if p.requires_grad]


def is_clipping_enabled(max_norm) -> bool:
    """HF's convention: ``max_grad_norm <= 0`` (or unset) disables clipping, where scaling by
    ``0 / norm`` would zero every gradient."""
    return bool(max_norm) and max_norm > 0


def local_grad_shards(params) -> list[torch.Tensor]:
    """The local tensors of ``params``' gradients: ``_foreach`` ops refuse DTensors and plain tensors together."""
    return [g.to_local() if isinstance(g, DTensor) else g for g in (p.grad for p in params) if g is not None]


def local_grad_norm_sq(shards: list[torch.Tensor], *, device) -> torch.Tensor:
    """Sum of the squared L2 norms of ``shards`` — this rank's contribution, 0-dim fp32 on ``device``.

    Each per-tensor norm accumulates in fp32 rather than the shard's bf16. An empty list returns a zero
    rather than skipping, since every caller reduces the result across ranks.
    """
    if not shards:
        return torch.zeros((), device=device, dtype=torch.float32)
    norms = torch._foreach_norm(shards, dtype=torch.float32)
    return torch.linalg.vector_norm(torch.stack(norms)).pow(2).to(device)


def clip_coefficient(max_norm: float | torch.Tensor, total_norm: torch.Tensor) -> torch.Tensor:
    """``min(1, max_norm / total_norm)``, device-resident. ``nan_to_num`` runs before the clamp, so a NaN
    norm leaves the gradients untouched; the epsilon guards an all-zero step, as torch's own clip does."""
    return torch.nan_to_num(max_norm / (total_norm + 1e-6), nan=1.0).clamp(max=1.0)


def scale_shards_to_max_norm_(
    shards: list[torch.Tensor], max_norm: float | torch.Tensor, total_norm: torch.Tensor
) -> None:
    """Scale local grad shards in place by the clamped clip coefficient, unconditionally, so no path pays
    a host sync to decide whether to clip; ``max_norm`` is taken as given for the same reason."""
    if shards:
        torch._foreach_mul_(shards, clip_coefficient(max_norm, total_norm))


def pp_clip_grad_norm(parameters, max_norm, norm_type=2) -> torch.Tensor:
    """The pipeline stage's ``clip_grad_norm_`` when every parameter is a DTensor (dense and ep1-FSDP stages).

    Every parameter is sharded across exactly one stage's DP group, so the world sum of the local squared
    norms is the global one; plain-tensor EP expert shards take the EP clip instead.
    """
    require_l2(norm_type, "Pipeline-parallel")
    shards = local_grad_shards(trainable_clip_params(parameters))
    local = local_grad_norm_sq(shards, device=current_device())
    dist.all_reduce(local, op=dist.ReduceOp.SUM)
    total_norm = local.sqrt()
    if is_clipping_enabled(max_norm):
        scale_shards_to_max_norm_(shards, max_norm, total_norm)
    return total_norm


class GradientClippingMixin:
    """The ``clip_grad_norm_`` replacements of :class:`DistributedTrainerMixin` and the global norm they reduce."""

    def _install_ep_clip(self) -> None:
        """Install the EP clip: the deferred sweep, the EP+TP replicated sync, then the true global norm.

        A model without EP layers here is a load that skipped EP patching, whose experts would drift apart
        silently, so it raises once at install, as does a missing EP config the norm's expert legs read.
        """
        if not self._ep_layers():
            raise RuntimeError(
                "EP gradient clipping is being installed on a model with no EP-patched layers, so expert "
                "gradient synchronization will not work and experts on different ranks would silently "
                "diverge. Load the model with EP patching (via load_distributed_model), or drop "
                "expert_parallel_size for a dense model."
            )
        require_ep_config(self._ep_config)

        def sync(params: list) -> None:
            # Before the norm, so the clip runs on fully-synced grads and matches on every rank.
            self._sync_deferred_expert_grads()
            # EP+TP: TP-average replicated grads first, else norms / the shared expert drift across the TP axis.
            if self.parallelism_config.is_tp_mode:
                self._sync_tp_replicated_grads(self._tp_replicated_params(params))

        self.accelerator.clip_grad_norm_ = self._global_norm_clip("Expert-parallel", sync, defer_to_optimizer=True)

    def _install_tp_clip(self) -> None:
        """Install the TP clip: TP mixes DTensor params with plain replicated ones (norms, embeddings,
        ``lm_head``), which accelerate's ``torch._foreach_norm`` refuses together. Replicated grads are
        TP-averaged first, else the weights and the norm diverge across the TP group."""
        self.accelerator.clip_grad_norm_ = self._global_norm_clip(
            "Tensor-parallel", self._sync_tp_replicated_grads, defer_to_optimizer=False
        )

    def _install_qlora_clip(self) -> None:
        """Run the QLoRA DP average, a post-backward sweep, ahead of the norm, so every rank clips by the
        same coefficient instead of its own local one."""
        base_clip = self.accelerator.clip_grad_norm_

        def qlora_clip_grad_norm_(parameters, max_norm, norm_type=2):
            self._sync_qlora_grads()
            return base_clip(parameters, max_norm, norm_type)

        self.accelerator.clip_grad_norm_ = qlora_clip_grad_norm_

    def _global_norm_clip(self, path: str, sync: Callable[[list], None], *, defer_to_optimizer: bool):
        """A ``clip_grad_norm_`` that runs ``sync`` on the trainable params, then clips them to their global
        L2 norm; with ``defer_to_optimizer`` the coefficient goes to :meth:`_grad_scale_deferring_optimizer`
        instead of a pass over the grads. ``path`` names the clip in the norm-type refusal."""

        def clip_grad_norm_(parameters, max_norm, norm_type=2) -> torch.Tensor:
            require_l2(norm_type, path)
            params = trainable_clip_params(parameters)
            # Structural, never grad-presence: the sync and the norm issue collectives.
            if not params:
                return torch.tensor(0.0, device=current_device())
            sync(params)
            global_norm = self._compute_global_grad_norm(params)
            # Device-resident: reading the norm back stalls the launch queue.
            if not is_clipping_enabled(max_norm):
                return global_norm
            deferred = self._grad_scale_deferring_optimizer(params) if defer_to_optimizer else None
            if deferred is not None:
                deferred.defer_grad_scale(clip_coefficient(float(max_norm), global_norm))
            else:
                scale_shards_to_max_norm_(local_grad_shards(params), float(max_norm), global_norm)
            return global_norm

        return clip_grad_norm_

    def _grad_scale_deferring_optimizer(self, params: list):
        """The optimizer that will apply the clip coefficient inside its own step, or ``None``.

        Only an optimizer exposing ``defer_grad_scale`` (AdamWBF16) whose parameters with gradients are
        exactly the clipped ones qualifies, read off its current ``param_groups`` on every call: any other
        set would leave a clipped parameter unclipped or scale one the clip did not select.
        """
        optimizer = getattr(self, "optimizer", None)
        optimizer = getattr(optimizer, "optimizer", optimizer)  # accelerate's AcceleratedOptimizer wrapper
        if optimizer is None or not hasattr(optimizer, "defer_grad_scale"):
            return None
        clipped = {id(p) for p in params if p.grad is not None}
        stepped = {id(p) for group in optimizer.param_groups for p in group["params"] if p.grad is not None}
        return optimizer if clipped == stepped else None

    def _compute_global_grad_norm(self, params: list[nn.Parameter]) -> torch.Tensor:
        """COLLECTIVE: the global L2 gradient norm of ``params`` for every EP and TP clip, a 0-dim device tensor.

        Each bucket of :meth:`_bucket_grad_shards` reduces over the group that tiles its shards once:
        ``expert`` over the expert-TP then the dispatch group (EP-group replicas are already identical,
        since the clip sweeps first), ``fsdp_full`` over the FSDP shard group, the TP buckets per
        :meth:`_reduce_shard_norm_buckets`, ``other`` locally. Every reduce is gated on config, never on
        which grads exist; under PP each is stage-scoped and the chain sum adds the disjoint stages.
        """
        device = current_device()
        norm_sq = {
            name: local_grad_norm_sq(shards, device=device)
            for name, shards in self._bucket_grad_shards(params).items()
        }
        total_norm_sq = norm_sq["other"]
        if self._device_mesh is not None:
            total_norm_sq = total_norm_sq + self._reduce_shard_norm_buckets(
                norm_sq["tp1d"], norm_sq["tp2d"], norm_sq["dp"]
            )
        elif self._fsdp_wrapped:
            # Off the first non-expert DTensor *parameter*, which exists in identical order on every rank
            # (unlike the first non-None grad); without one, the stage group, never a cross-stage reduce.
            expert_ids = self._get_sharded_expert_param_ids()
            sharded = next(
                (
                    p
                    for p in self._top_level_model().parameters()
                    if id(p) not in expert_ids and isinstance(p, DTensor)
                ),
                None,
            )
            group = self._pp_stage_group if sharded is None else self._fsdp_shard_group(sharded.device_mesh)
            dist.all_reduce(norm_sq["fsdp_full"], op=dist.ReduceOp.SUM, group=group)
            total_norm_sq = total_norm_sq + norm_sq["fsdp_full"]

        # The EP config holds the expert groups; a dense TP run has none and no expert bucket.
        ep_cfg = self._ep_config
        if ep_cfg is not None:
            expert_norm_sq = norm_sq["expert"]
            if ep_cfg.expert_tp_size > 1 and ep_cfg.expert_tp_group is not None:
                dist.all_reduce(expert_norm_sq, op=dist.ReduceOp.SUM, group=ep_cfg.expert_tp_group)
            if ep_cfg.dispatch_ep_group is not None:
                dist.all_reduce(expert_norm_sq, op=dist.ReduceOp.SUM, group=ep_cfg.dispatch_ep_group)
            total_norm_sq = total_norm_sq + expert_norm_sq

        if self._pp_chain_group is not None:
            dist.all_reduce(total_norm_sq, op=dist.ReduceOp.SUM, group=self._pp_chain_group)
        return total_norm_sq.sqrt()

    def _bucket_grad_shards(self, params: list[nn.Parameter]) -> dict[str, list[torch.Tensor]]:
        """``params``' local grad shards by bucket, every bucket declared so an empty one still joins its reduce:
        ``expert`` (EP/ETP-distributed expert shards only), ``fsdp_full`` (EP-only / EP+CP FSDP2 shards, no
        device mesh), the TP buckets (DTensor or hand-sliced grads under a mesh), ``other`` (replicated)."""
        expert_ids = self._get_sharded_expert_param_ids()
        # Plain tensors the TP plan sharded by hand (GptOss sinks): disjoint slices, not replicas.
        tp_sharded_plain_ids = self._tp_sharded_plain_param_ids()
        shards: dict[str, list[torch.Tensor]] = {
            name: [] for name in ("expert", "fsdp_full", "tp1d", "tp2d", "dp", "other")
        }
        for param in params:
            grad = param.grad
            if grad is None:
                continue
            is_dtensor = isinstance(grad, DTensor)
            tp_disjoint = id(param) in tp_sharded_plain_ids
            if id(param) in expert_ids:
                bucket = "expert"
            elif is_dtensor and self._device_mesh is None:
                bucket = "fsdp_full"
            elif is_dtensor or tp_disjoint:
                bucket = self._sharded_grad_bucket(grad, tp_disjoint)
            else:
                bucket = "other"
            shards[bucket].append(grad._local_tensor if is_dtensor else grad)
        return shards

    @staticmethod
    def _sharded_grad_bucket(grad: torch.Tensor, tp_disjoint: bool) -> str:
        """The TP bucket whose group tiles ``grad`` once: ``"tp1d"`` (1D ``tp`` mesh, or a plain TP-disjoint
        slice under pure TP), ``"tp2d"`` (2D ``(dp, tp)`` mesh, or a hand-sliced sink FSDP-sharded over
        ``dp``), ``"dp"`` (a 1D ``dp`` shard replicated across TP). ``tp_disjoint`` separates the last two,
        both 1D ``dp`` DTensors after the FSDP2 wrap."""
        dims = mesh_dim_names(getattr(grad, "device_mesh", None))
        if MeshDim.TP in dims:
            return "tp2d" if len(dims) > 1 else "tp1d"
        if tp_disjoint:
            return "tp2d" if dims else "tp1d"
        return "dp"

    def _reduce_shard_norm_buckets(
        self, tp1d_sq: torch.Tensor, tp2d_sq: torch.Tensor, dp_sq: torch.Tensor
    ) -> torch.Tensor:
        """COLLECTIVE: each TP/DP shard-norm bucket summed over the ranks that tile it, and their total:
        ``tp1d`` over the TP group, ``dp`` over the DP group, ``tp2d`` over the ``(dp, tp)`` plane, which is
        the pipeline stage (``None``, the world, without PP), since other stages hold other layers.
        """
        if (group := tp_group(self._device_mesh)) is not None:
            dist.all_reduce(tp1d_sq, op=dist.ReduceOp.SUM, group=group)
        if (group := dp_group(self._device_mesh)) is not None:
            dist.all_reduce(tp2d_sq, op=dist.ReduceOp.SUM, group=self._pp_stage_group)
            dist.all_reduce(dp_sq, op=dist.ReduceOp.SUM, group=group)
        return tp1d_sq + tp2d_sq + dp_sq

    @staticmethod
    def _fsdp_shard_group(mesh) -> dist.ProcessGroup:
        """The group summing FSDP shard norms over ``mesh``: the DP group of a 1D full-shard mesh, the
        ``dp_shard`` sub-group of a 2D HSDP mesh (replicas hold identical shards). Any other shape raises
        rather than fall back to the world group, which under PP spans every stage."""
        if mesh.ndim == 1:
            return mesh.get_group()
        dim_names = mesh_dim_names(mesh)
        if MeshDim.DP_SHARD in dim_names:
            return mesh[MeshDim.DP_SHARD].get_group()
        raise RuntimeError(
            f"Cannot pick an FSDP shard group from a {mesh.ndim}D mesh with dims {dim_names}: "
            f"expected a 1D full-shard mesh or a 2D HSDP mesh carrying {MeshDim.DP_SHARD!r}."
        )

    def _get_grad_norm(self, model, grad_norm=None):
        """HF's post-step gradient norm, computed only on the steps that log it.

        With clipping on, the clip's by-product passes through. With ``max_grad_norm <= 0`` HF asks for an
        unclipped norm for the log line alone, a collective discarded on non-logging steps; the sweeps
        that path carries are covered by their step-pre-hooks.
        """
        if grad_norm is None and not self._logs_after_this_step():
            return None
        return super()._get_grad_norm(model, grad_norm=grad_norm)

    def _logs_after_this_step(self) -> bool:
        """Whether ``_maybe_log_save_evaluate`` will log the step now being completed.

        Replicates ``DefaultFlowCallback.on_step_end``, which runs after ``global_step`` is incremented,
        from rank-uniform state only; whatever it cannot predict (an epoch-strategy log, a flag another
        callback raised, the final step) answers True.
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

    def _has_mixed_mesh_grads(self, model) -> bool:
        """Always ``False``, so HF's clip and its unclipped norm both go through ``accelerator.clip_grad_norm_``:
        HF would send mixed DTensor/plain gradients to its own per-mesh norm instead, bypassing the EP/TP/QLoRA
        clips that carry the global norm and the grad sweeps."""
        return False
