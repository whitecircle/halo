"""Pipeline-parallel trainer integration: split, schedule-driven step, clip, save.

The pipeline schedule replaces the inherited HF step: it drives every microbatch's forward and
backward itself, so under PP the trainer must not call ``accelerator.backward`` and must run with
``gradient_accumulation_steps=1``. The pipeline's microbatches are the accumulation, and the user's
configured accumulation is consumed as the microbatch count.

Composition scope: PP + FSDP2 inside each stage, and PP + EP (grad-norm / checkpoint / balancing
paths are stage-scoped with a chain reduction). Every other axis is rejected by ``ParallelismConfig``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from torch.distributed.tensor import DTensor

from src.callbacks.moe_metrics import MoELoadMetricsCallback
from src.checkpoint.format import save_dtype_caster
from src.data.collators.fixed_shape import FixedShapeCollator
from src.data.collators.packing import DataCollatorWithFlattening, DataCollatorWithPacking, pad_tail_positions
from src.data.vlm import VLM_IMAGE_COLUMNS, dataset_image_evidence
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.pipeline_parallel.groups import (
    broadcast_scalar_from_last_stage,
    broadcast_tensor_from_last_stage,
    create_pipeline_group,
    create_stage_group,
)
from src.distributed.pipeline_parallel.losses import PPLossAdapter, loss_token_count
from src.distributed.pipeline_parallel.runtime import PP_BATCH_PAD_VALUES, PipelineRuntime
from src.distributed.pipeline_parallel.split import resolve_module_attr, resolve_pp_spec
from src.distributed.pipeline_parallel.stage import PipelineStageModule, build_pipeline_stage, module_path
from src.distributed.runtime import (
    current_device,
    fs_aware_save_rank,
    is_global_main_process,
    reject_across_ranks,
)
from src.env import env_flag
from src.models.modality import is_vlm_model
from src.models.structure import backbone_with_layers, resolve_tokenizer
from src.trainers.mixins.grad_clip import (
    clip_parameters,
    clipping_enabled,
    local_grad_norm_sq,
    scale_shards_to_max_norm_,
)
from src.trainers.mixins.pp_gates import (
    reject_pp_activation_offloading,
    reject_pp_compute_metrics,
    reject_pp_peft,
)

logger = get_logger(__name__, log_level="info")

# 1F1B bubble fraction = (pp - 1) / (microbatches + pp - 1): _BUBBLE_TARGET_FACTOR·(pp-1) microbatches
# put it at 1/10, and below _BUBBLE_WARN_FACTOR·pp it grows large enough to warn.
_BUBBLE_WARN_FACTOR = 4
_BUBBLE_TARGET_FACTOR = 9


def pp_pad_values(pad_token_id: int, extra: Mapping[str, int] | None = None) -> dict[str, int]:
    """The runtime's batch pad values resolved against this run's tokenizer.

    ``PP_BATCH_PAD_VALUES`` leaves the pad id as ``None`` because it is tokenizer-specific, and an
    adapter's ``pad_spec`` adds the seq-shaped extra targets it ships.
    """
    resolved = {key: pad_token_id if value is None else value for key, value in PP_BATCH_PAD_VALUES.items()}
    resolved.update(extra or {})
    return resolved


def wrapper_state_outside_stages(model) -> dict[str, torch.Tensor]:
    """The tensors of ``model`` that no pipeline stage holds, by their unsplit (global) names.

    A stage is the backbone plus the task head; a composite wrapper's other children (a vision
    tower and projector) are dropped at the split. Empty for a plain causal LM, whose every tensor
    sits under one of the two.
    """
    backbone = backbone_with_layers(model)
    backbone_prefix = module_path(model, backbone)
    head_attr = resolve_module_attr(model, resolve_pp_spec(backbone).HEAD_ATTRS, "task head")
    owned = tuple(f"{prefix}." for prefix in (backbone_prefix, head_attr) if prefix)
    # An empty backbone prefix means the model is the backbone: every tensor belongs to a stage.
    if not backbone_prefix:
        return {}
    return {name: tensor for name, tensor in model.state_dict().items() if not name.startswith(owned)}


def collator_image_evidence(collator) -> str | None:
    """Why ``collator`` counts as an image consumer, or ``None``.

    Read off the columns it declares it needs, using the same
    :data:`~src.data.vlm.VLM_IMAGE_COLUMNS` spellings as the dataset probe, so a preprocessed-VLM
    collator whose dataset stores pixels rather than images is caught too.
    """
    consumed = set(getattr(collator, "required_dataset_columns", ()) or ()) & set(VLM_IMAGE_COLUMNS)
    return f"the data collator consuming {sorted(consumed)}" if consumed else None


def admit_multimodal_wrapper(model, kwargs: dict) -> dict[str, torch.Tensor]:
    """Admit a multimodal wrapper under PP only for a run that feeds it no images.

    The split keeps the text backbone and the task head, and the backbone probe descends to
    ``language_model``, so the vision tower and projector are in no stage. A run that feeds images
    therefore breaks three ways: the tower gets no gradient, ``pixel_values`` is pruned by the
    runtime's column pin, and the placeholder tokens train as text. Such a run is refused, naming
    the evidence: an image column or embedded image parts in any dataset, or a collator that
    consumes an image column (:data:`~src.data.vlm.VLM_IMAGE_COLUMNS`).

    A text-only run of the same checkpoint runs the wrapper's text tower as the pipeline. The vision
    tower's untouched tensors are kept on the save rank (returned here; the trainer holds them as
    ``_pp_wrapper_state``) so every checkpoint re-emits them under the wrapper layout, letting the
    export reload as the composite class with no reattach step.
    """
    evidence = next(
        (
            found
            for found in (
                dataset_image_evidence(kwargs.get("train_dataset")),
                dataset_image_evidence(kwargs.get("eval_dataset")),
                collator_image_evidence(kwargs.get("data_collator")),
            )
            if found
        ),
        None,
    )
    reason = (
        (
            f"Vision-language training is not supported under pipeline parallelism: the stage split "
            f"keeps only the text backbone and the task head, so the vision tower and projector are "
            f"in no stage — {evidence} means this run feeds images, which would never reach the "
            f"model (pixel inputs are pruned by the runtime's column pin) while their placeholder "
            f"tokens train as text. Train the VLM without pipeline parallelism, or drop the image "
            f"data for a text-only run of its text tower."
        )
        if evidence
        else None
    )
    # Joined across ranks: the dataset probe is rank-local (a pre-sharded dataset can carry images
    # on one rank only), and a lone raise would leave the peers at the split's collectives.
    reject_across_ranks(reason, "Pipeline-parallel multimodal gate", exc_type=ValueError)
    # Every checkpoint re-emits these from the rank that writes the shared files; the other ranks
    # free them with the wrapper right after the split.
    stash = stash_wrapper_state(model) if fs_aware_save_rank() else {}
    if is_global_main_process():
        params = sum(t.numel() for t in stash.values())
        logger.info(
            "Text-only run of a multimodal checkpoint under pipeline parallelism: %d tensors "
            "(%.1fM values) outside the text backbone — the vision tower and projector — are held "
            "by no stage. They stay untrained and are re-emitted unchanged in every checkpoint, so "
            "the export keeps the %s layout.",
            len(stash),
            params / 1e6,
            type(model).__name__,
        )
    return stash


def pp_ep_coverage_reason(config: ParallelismConfig, ep_modules) -> str | None:
    """Why this rank's stage cannot run PP+EP, or ``None``.

    Rank-local by nature (a hybrid MoE with leading dense layers can give one stage no EP modules),
    so the verdict is joined across ranks by the caller; a lone raise would leave the MoE stages in
    the chain broadcast that follows, waiting out the NCCL timeout with no diagnostic.
    """
    if not config.is_ep_mode or ep_modules:
        return None
    return (
        f"PP stage {config.pp_rank}/{config.pp_size} carries no EP-patched modules. Either the model "
        "was loaded without EP patching (load_distributed_model / patch_moe_model_for_ep must run "
        "before the trainer), or the layer partition gave this stage only dense layers (hybrid MoE "
        "with leading dense layers): a dense stage would run a different collective program than its "
        "MoE peers and deadlock at the gradient norm. Pick a pipeline_parallel_size whose partition "
        "puts at least one MoE layer on every stage."
    )


def stash_wrapper_state(model) -> dict[str, torch.Tensor]:
    """Host copies of :func:`wrapper_state_outside_stages`, at the artifact's save dtype.

    Cast through the caster built on the whole wrapper: the artifact contract's keep-sets (its
    norms, fp32 pins) are tree-derived, and the stage held at save time no longer has the tower's
    modules to derive them from.
    """
    cast = save_dtype_caster(model)
    return {
        name: cast(name, tensor.detach()).to("cpu", copy=True)
        for name, tensor in wrapper_state_outside_stages(model).items()
    }


class PipelineTrainerMixin:
    """PP hooks for :class:`DistributedTrainerMixin`; inert unless ``pp_size > 1``."""

    _pp_runtime: PipelineRuntime | None = None
    _pp_adapter: PPLossAdapter | None = None
    _pp_n_microbatches: int = 0
    _pp_stage_group = None
    _pp_chain_group = None
    # Tensors outside every stage (a multimodal wrapper's vision tower), re-emitted by the PP save
    # from the save rank; ``None`` off PP or for a plain causal LM, which drops nothing.
    _pp_wrapper_state: dict[str, torch.Tensor] | None = None
    # Key order of the adapter's per-step metrics, pinned at setup so the per-step share is a
    # values-only broadcast; empty when the adapter declares no ``metrics_fn``.
    _pp_metric_keys: tuple[str, ...] = ()

    def _validate_pp_mode(self, ctor_args: tuple, kwargs: dict) -> None:
        """Hook: reject this trainer's own PP-incompatible options, before the model is split.

        Called by :meth:`_maybe_prepare_pipeline_model` once the PP early-out has passed, so an
        override needs no ``is_pp_mode`` guard. Every ctor parameter is read out of ``kwargs``
        (``require_model_and_args_kwargs`` forces the positional ones there); ``ctor_args`` carries
        the trainer's remaining positionals. A trainer whose PP-relevant options are explicit ctor
        parameters cannot use this hook and gates them in its own ``__init__`` instead.
        """

    def _maybe_prepare_pipeline_model(self, kwargs: dict, training_args, ctor_args: tuple = ()) -> dict:
        """Split the model into this rank's stage before ``super().__init__`` sees it.

        Runs the PP-incompatibility rejections that must precede any wrapping, converts the user's
        gradient accumulation into the pipeline's microbatch count, and pins the collator to fixed
        shapes. No-op at ``pp_size == 1``.
        """
        config: ParallelismConfig = self.parallelism_config
        if not config.is_pp_mode:
            return kwargs

        self._validate_pp_mode(ctor_args, kwargs)

        model = kwargs.get("model")
        # Fixed shapes: the collator (wrapped after super().__init__) must pad every batch to one
        # length. max_length is the natural bound.
        if not getattr(training_args, "max_length", None):
            raise ValueError(
                "Pipeline parallelism requires a fixed sequence length (the P2P buffer shapes "
                "freeze on the first step). Set max_length in the training config."
            )
        if self.save_sharded_ep:
            raise ValueError(
                "save_sharded_ep is not supported under pipeline parallelism: the per-rank shard "
                "format (index.metadata.format ep_sharded) and merge_ep_shards.py key tensors by "
                "unsplit-model names with no pipeline-stage layer offset, so shards from different "
                "stages would collide or merge under wrong names. The PP save already writes one "
                "complete-tensor shard per stage under global names."
            )
        gc_kwargs = training_args.gradient_checkpointing_kwargs or {}
        if training_args.gradient_checkpointing and gc_kwargs.get("use_reentrant", False):
            # The stage's gradient_checkpointing_enable enforces this too; rejecting here names it at
            # config time rather than mid-run.
            raise ValueError(
                "gradient_checkpointing_kwargs={'use_reentrant': True} is not supported under "
                "pipeline parallelism: the reentrant checkpoint runs its forward under no_grad, so "
                "FSDP2 never registers pre-backward hooks and recompute re-enters its full forward "
                "from inside the schedule's backward. Drop the kwarg (the default use_reentrant="
                "False path is supported and validated)."
            )
        evaluating = training_args.eval_strategy not in ("no", None)
        if evaluating and training_args.per_device_eval_batch_size != training_args.per_device_train_batch_size:
            raise ValueError(
                f"per_device_eval_batch_size ({training_args.per_device_eval_batch_size}) must "
                f"equal per_device_train_batch_size ({training_args.per_device_train_batch_size}) "
                f"under pipeline parallelism: the pipeline's P2P buffer shapes are frozen to the "
                f"training microbatch, and an eval batch of any other size cannot be split into "
                f"matching chunks."
            )
        reject_pp_peft(kwargs.get("peft_config"))
        # Here rather than per trainer: SFTConfig and RewardConfig also carry the flag, and the
        # schedule-driven training_step bypasses the wrapper TRL applies it with either way.
        reject_pp_activation_offloading(training_args)
        if model is None or not hasattr(model, "config"):
            raise ValueError(
                "Pipeline parallelism requires the model to be passed as the `model` keyword (a "
                "loaded PreTrainedModel), so the trainer can split it into this rank's stage."
            )
        # The name catches an unregistered VLM (remote code, a custom class) whose config declares
        # nothing; passing the config keeps the probe off the hub.
        if is_vlm_model(getattr(model.config, "_name_or_path", "") or "", config=model.config):
            self._pp_wrapper_state = admit_multimodal_wrapper(model, kwargs)
        if training_args.torch_compile:
            raise ValueError(
                "torch_compile is not supported under pipeline parallelism: the pipeline schedule "
                "captures the stage module at setup, so the compiled wrapper installed afterwards "
                "would never run — a silent no-op. Disable torch_compile."
            )

        # The user's accumulation becomes the microbatch count: the schedule runs all microbatches in
        # one optimizer step, so leaving HF's accumulation loop on would square the effective batch.
        if config.pp_microbatches and training_args.gradient_accumulation_steps > 1:
            raise ValueError(
                f"pipeline_microbatches={config.pp_microbatches} and gradient_accumulation_steps="
                f"{training_args.gradient_accumulation_steps} are both set: under PP the microbatches "
                f"ARE the accumulation, so the accumulation would be silently discarded and the "
                f"effective batch shrink by that factor. Set exactly one of them."
            )
        n_microbatches = config.pp_microbatches or training_args.gradient_accumulation_steps
        if config.pp_schedule == "1f1b" and n_microbatches < config.pp_size:
            logger.warning(
                "pipeline_microbatches=%d is below pp_size=%d (Schedule1F1B needs a full pipeline); raising to %d.",
                n_microbatches,
                config.pp_size,
                config.pp_size,
            )
            n_microbatches = config.pp_size
        if n_microbatches < _BUBBLE_WARN_FACTOR * config.pp_size:
            logger.warning(
                "pipeline_microbatches=%d gives a large pipeline bubble at pp_size=%d (idle fraction "
                "(pp-1)/(m+pp-1): ~22%% of the step measured at m=2*pp on 2 stages). For a 10%% bubble "
                "use at least %d microbatches (rule: %d*(pp_size-1)).",
                n_microbatches,
                config.pp_size,
                _BUBBLE_TARGET_FACTOR * (config.pp_size - 1),
                _BUBBLE_TARGET_FACTOR,
            )
        if training_args.per_device_train_batch_size % n_microbatches != 0:
            raise ValueError(
                f"per_device_train_batch_size ({training_args.per_device_train_batch_size}) must be "
                f"divisible by the pipeline microbatch count ({n_microbatches})."
            )
        training_args.gradient_accumulation_steps = 1
        self._pp_n_microbatches = n_microbatches

        kwargs["model"] = build_pipeline_stage(
            model, config.pp_rank, config.pp_size, moe_balancing=self._moe_balancing, pp_split=config.pp_split
        )

        context_window = getattr(kwargs["model"].config.get_text_config(), "max_position_embeddings", None)
        if context_window and training_args.max_length >= context_window:
            # `max_length: null` resolves to the context window, so every row pays full-window
            # compute with no visibly larger batch shape to explain it.
            logger.warning(
                "Pipeline parallelism pads EVERY batch to max_length=%d, which is the model's full "
                "context window — each row pays full-window compute regardless of its true length. "
                "Set max_length to a cap matched to the data.",
                training_args.max_length,
            )
        return kwargs

    def _pp_wrap_collator(self):
        """Wrap the trainer's final collator for fixed shapes (called from ``_setup_pipeline_parallel``).

        Runs after ``super().__init__``, so it covers user-passed, trainer-default, and
        late-constructed collators alike; dataloaders are built lazily at ``train()``, so the wrap
        is always in time. Idempotent.
        """
        inner = self.data_collator
        if isinstance(inner, FixedShapeCollator):
            return
        if inner is None:
            raise ValueError("Pipeline parallelism requires a data collator; the trainer has none.")
        if isinstance(inner, DataCollatorWithFlattening):
            # Rejected on three counts: its width is the summed document length (variable by
            # construction, so there is no constant to pad to); its cu_seq_lens / max_length_* side
            # outputs are neither splittable on dim 0 nor re-basable per microbatch; and its single
            # row cannot be microbatched. Padding it to a constant would reintroduce the padding it
            # exists to avoid, so packing supplies the fixed shape instead.
            raise ValueError(
                f"{type(inner).__name__} (padding-free) is incompatible with pipeline parallelism: "
                f"its flattened width is the summed document length, which varies every step, and "
                f"the pipeline's P2P buffers are frozen after the first. Use packing instead "
                f"(`packing: true`) — under PP it delivers the same throughput with fixed shapes."
            )
        if isinstance(inner, DataCollatorWithPacking):
            # Packed rows padded to max_length are a fixed shape. The collator normally flattens
            # [B, L] -> [1, B*L] because transformers only engages packed-sequence handling at batch
            # size 1; under PP the microbatch split along dim 0 does that flattening instead, so the
            # rows must survive the collator and each microbatch must carry exactly one of them.
            if self.args.per_device_train_batch_size != self._pp_n_microbatches:
                raise ValueError(
                    f"Packing under pipeline parallelism needs one packed row per microbatch, but "
                    f"per_device_train_batch_size={self.args.per_device_train_batch_size} and the "
                    f"microbatch count is {self._pp_n_microbatches}. A microbatch holding >1 packed "
                    f"row silently leaks attention ACROSS documents under FlashAttention (which "
                    f"derives cu_seqlens from position_ids only at batch size 1) — measured at ~1.9 "
                    f"nats of logprob error. Set per_device_train_batch_size equal to the microbatch "
                    f"count."
                )
            inner.flatten_to_single_row = False
        tokenizer = resolve_tokenizer(self.processing_class)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            raise ValueError(
                "Pipeline parallelism pads every batch to max_length and needs the tokenizer's "
                "pad_token_id; the processing_class passed to the trainer has none."
            )
        self.data_collator = FixedShapeCollator(
            inner, self.args.max_length, pp_pad_values(pad_id, self._pp_adapter.pad_spec)
        )

    def _setup_pipeline_parallel(self):
        """Post-``__init__`` PP setup: stage-scoped (EP-aware) FSDP, the runtime, clip, and column pinning."""
        config: ParallelismConfig = self.parallelism_config
        stage: PipelineStageModule = self.model

        # The trainer's declared PP contract (loss, pair layout, side tensors) — resolved once.
        self._pp_adapter = adapter = self._pp_loss_adapter()
        token_loss_fn = adapter.token_loss_fn

        # Wrap first: the collator declares which dataset columns it needs, and the pin below must
        # union them in. Pinning first would prune a packing collator's `seq_lengths`, collapsing
        # every packed row into a single attended document instead of failing.
        self._pp_wrap_collator()

        # HF prunes dataset columns against the forward signature, and a stage's forward takes
        # hidden_states; without this pin, input_ids and labels are dropped.
        if adapter.pin_runtime_columns:
            collator_columns = getattr(self.data_collator, "required_dataset_columns", ())
            self._signature_columns = list(
                dict.fromkeys(
                    list(PP_BATCH_PAD_VALUES)
                    + list(adapter.extra_target_keys)
                    + list(adapter.extra_signature_columns)
                    + list(collator_columns)
                )
            )
        else:
            self._signature_columns = None
            self._set_signature_columns_if_needed()

        self._pp_chain_group = create_pipeline_group(config)

        # Also captures self._ep_config from the wrappers.
        ep_modules = self._find_ep_modules()
        # Deliberately unconditional: create_stage_group issues pp_size collective dist.new_group
        # calls while `ep_modules` is rank-dependent (a hybrid MoE with leading dense layers can give
        # one stage no EP modules). Gating this deadlocks inside new_group, before any raise prints.
        self._pp_stage_group = create_stage_group(config)
        reject_across_ranks(pp_ep_coverage_reason(config, ep_modules), "PP stage EP coverage")
        topo = f", PP stage {config.pp_rank}/{config.pp_size}"
        if ep_modules:
            # The stage registered the EP capacity-generation hook on itself (PipelineStageModule); only
            # the FSDP2 wrap is left to do here.
            self._apply_ep_aware_dp_fsdp2(
                stage,
                ignored=self._ep_fsdp_ignored_modules(),
                fallback_dp_size=config.stage_world_size,
                fallback_dp_group=self._pp_stage_group,
                topo=topo,
            )
        else:
            self._apply_dp_fsdp2(
                stage,
                config.stage_world_size,
                dp_group=self._pp_stage_group,
                topo=topo,
                detail=f"stage-scoped data parallel ({config.stage_world_size} ranks per stage)",
            )

        # Auto-enabled wherever the objective decomposes over token chunks (the causal-LM contract;
        # see PPLossAdapter.supports_fused_head_loss). Loss and activation grads are unchanged; the
        # head weight's gradient is reassociated (see fused_causal_lm_token_loss) in exchange for
        # dropping the logits plane. HALO_PP_FUSED_HEAD_LOSS=0 restores the logits path.
        fused_head_loss = adapter.supports_fused_head_loss and env_flag("HALO_PP_FUSED_HEAD_LOSS", True)
        self._pp_runtime = PipelineRuntime(
            stage,
            config,
            self._pp_chain_group,
            torch.device("cuda", torch.cuda.current_device()),
            self._pp_n_microbatches,
            token_loss_fn=token_loss_fn,
            paired_examples=adapter.paired_examples,
            fused_head_loss=fused_head_loss,
        )
        # Plain-tensor expert shards need the EP-aware clip (deferred sweep + dispatch-group norm
        # aggregation); dense and ep1-FSDP-sharded stages are all DTensors → direct PP clip.
        if ep_modules and not config.experts_fsdp_managed:
            self._patch_gradient_clipping_for_ep()
        else:
            self.accelerator.clip_grad_norm_ = self._pp_clip_grad_norm

        # Each stage holds different routers, so the callback's expert-load reduce must stay inside
        # this stage's rank block; a world reduce is a shape mismatch or a cross-stage blend.
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, MoELoadMetricsCallback):
                callback.reduce_group = self._pp_stage_group

        self._pp_adapter = adapter = self._pp_hoist_metrics_reduction(adapter)
        self._pp_pin_metric_keys()

        logger.info(
            "✓ Pipeline parallelism: %d stages × %d ranks, %d microbatches, %s schedule%s%s",
            config.pp_size,
            config.stage_world_size,
            self._pp_n_microbatches,
            config.pp_schedule,
            f", {len(ep_modules)} EP MoE layers" if ep_modules else "",
            ", fused head loss (no logits plane)" if fused_head_loss else "",
        )

    def _reshard_after_evaluation_loop(self) -> None:
        """Re-register the sharded params once the eval loop ends.

        The forward-only drives leave the stage gathered across the loop's batches (one all-gather
        per eval, not per batch). Everything after the loop — grad clipping, callbacks, the next
        optimizer step — must read the shards, which carry the pending gradients. Per-rank, no
        collective.
        """
        if self._pp_runtime is not None:
            reshard_fsdp2_modules(self.model)

    def _pp_clip_grad_norm(self, parameters, max_norm, norm_type=2):
        """Gradient clipping with the norm accumulated across the whole pipeline.

        Under PP+FSDP every parameter is sharded across exactly one stage's DP group and stages are
        disjoint, so the sum of every rank's local squared norm is the global squared norm: one
        world all-reduce, then a locally applied, globally identical clip coefficient. DTensor
        experts (ep1 ``fsdp_shard_ep1_experts`` grouped-GEMM MoE) partition the same way and need no
        separate expert leg; stages holding plain-tensor EP expert shards use
        ``ep_clip_grad_norm_`` instead.
        """
        if norm_type != 2:
            raise ValueError(f"Pipeline-parallel clipping supports norm_type=2 only, got {norm_type}")
        grads = [p.grad for p in clip_parameters(parameters) if p.grad is not None]
        shards = [g.to_local() if isinstance(g, DTensor) else g for g in grads]
        local = local_grad_norm_sq(shards, device=current_device())
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
        total_norm = local.sqrt()
        # max_norm <= 0 disables clipping (HF's max_grad_norm=0 convention); scaling by 0/norm would
        # zero every grad. The clamped coefficient is applied unconditionally: no host sync.
        if clipping_enabled(max_norm) and shards:
            scale_shards_to_max_norm_(shards, max_norm, total_norm)
        return total_norm

    def _pp_eval_token_normalizer(self, labels) -> torch.Tensor:
        """Eval denominator for the causal-LM contract: this rank's non-ignored target count.

        Eval loss is the per-batch mean (HF averages the gathered per-rank values), so the
        denominator spans the rank's own batch.
        """
        return loss_token_count(labels).to(dtype=torch.float32).clamp(min=1.0)

    def _pp_default_normalizer(self, inputs) -> torch.Tensor:
        """DP-global loss-token count / dp: the causal-LM (SFT) normalizer.

        FSDP averages gradients over the dp replicas, so per-replica losses normalized by G/dp make
        the averaged gradient exactly the global token mean d(S_total/G). HF's own
        ``num_items_in_batch`` cannot supply this: a stage's forward has no loss kwargs, and its
        world-gathered form would count every chain pp_size times. One scalar all-reduce over the
        stage group yields the stage sum; dividing by stage_world_size gives G/dp for any
        intra-stage composition. Trainers normalizing differently declare it on their adapter.
        """
        count = loss_token_count(inputs["labels"]).to(dtype=torch.float32)
        if self._pp_stage_group is not None:
            dist.all_reduce(count, group=self._pp_stage_group)
        return (count / self.parallelism_config.stage_world_size).clamp(min=1.0)

    def _pp_training_step(self, inputs, num_items_in_batch=None) -> torch.Tensor:
        """One optimizer step's microbatches through the pipeline; loss broadcast for logging.

        The schedule already ran backward for every microbatch, so there is no
        ``accelerator.backward``: returning here hands HF a detached loss with gradients in place.
        ``num_items_in_batch`` is accepted for signature compatibility but unused, since HF never
        computes it for a stage; the normalizer below is derived directly.
        """
        del num_items_in_batch
        self.model.train()
        inputs = self._prepare_inputs(inputs)

        adapter = self._pp_adapter
        if adapter.batch_transform is not None:
            inputs = adapter.batch_transform(inputs)
        normalizer = self._pp_default_normalizer(inputs) if adapter.normalizer is None else adapter.normalizer(inputs)
        extras = {key: inputs[key] for key in adapter.extra_target_keys} or None

        with self.compute_loss_context_manager():
            loss = self._pp_runtime.step(
                inputs["input_ids"],
                inputs["labels"],
                attention_mask=inputs.get("attention_mask"),
                position_ids=inputs.get("position_ids"),
                num_items_in_batch=normalizer,
                extra_targets=extras,
            )

        self._pp_share_step_metrics("train")
        return self._pp_broadcast_loss_from_last_stage(loss)

    @property
    def _pp_last_stage_rank(self) -> int:
        """Global rank of this chain's last stage: the source of every chain broadcast."""
        return self.parallelism_config.get_pp_group_ranks()[-1]

    def _pp_broadcast_loss_from_last_stage(self, tensor) -> torch.Tensor:
        """The last stage's scalar loss, broadcast down the chain so every rank logs the same value."""
        return broadcast_scalar_from_last_stage(tensor, self._pp_last_stage_rank, self._pp_chain_group)

    def _pp_broadcast_output_from_last_stage(self, tensor: torch.Tensor | None) -> torch.Tensor:
        """The last stage's prediction tensor, broadcast down the chain (shape carried with it).

        ``preprocess_logits_for_metrics`` cannot shrink what crosses here, since HF applies it after
        ``prediction_step`` returns. The adapter's ``predictions_fn``, which
        :meth:`_pp_hoist_metrics_reduction` fills from that same function, reduces on the last stage.
        """
        return broadcast_tensor_from_last_stage(tensor, self._pp_last_stage_rank, self._pp_chain_group)

    def _pp_hoist_metrics_reduction(self, adapter: PPLossAdapter) -> PPLossAdapter:
        """Move the user's ``preprocess_logits_for_metrics`` onto the last stage, or refuse the metric.

        With ``compute_metrics`` and no adapter reduction, what crosses the chain is the last stage's
        raw ``[rows, max_length, vocab]`` output (several GB of bf16 per eval batch at a large vocab
        and window), allocated again on every other stage to receive it. HF applies
        ``preprocess_logits_for_metrics`` after ``prediction_step`` returns, too late to shrink that
        hop; run here it is the same function one stage earlier, and an argmax leaves 1/vocab of the
        plane on the wire. It is cleared afterwards so HF does not reduce twice.

        The reduction runs on the pipeline's frozen batch (filler rows are trimmed after the
        broadcast), so it must be row-independent; argmax, top-k and per-token reductions are.
        """
        if self.compute_metrics is None or adapter.predictions_fn is not None:
            return adapter
        reduction = self.preprocess_logits_for_metrics
        reject_pp_compute_metrics(
            None if reduction is not None else self.compute_metrics,
            "this trainer's predictions are the raw logits plane, and a pipeline would broadcast it "
            "whole down the chain (rows x max_length x vocab, allocated on every stage). Pass "
            "preprocess_logits_for_metrics (e.g. `lambda logits, labels: logits.argmax(-1)`) — under "
            "PP it runs on the last stage before the broadcast, so only its result crosses — or "
            "evaluate without pipeline parallelism.",
        )
        self.preprocess_logits_for_metrics = None
        logger.info(
            "compute_metrics under pipeline parallelism: preprocess_logits_for_metrics runs on the "
            "last stage, before the chain broadcast, and is not re-applied by the evaluation loop."
        )
        return replace(
            adapter,
            predictions_fn=lambda outputs, inputs: reduction(outputs, inputs["labels"]),
        )

    def _pp_pin_metric_keys(self) -> None:
        """Pin the adapter's per-step metric names and check that every rank agrees on them.

        The per-step share broadcasts values only, so a key set differing across the chain would
        pair one stage's names with another's numbers, or hang on a length mismatch. The adapter's
        contract is that the names come from config, which makes them rank-uniform; this checks it
        once at setup rather than every step.
        """
        adapter = self._pp_adapter
        if adapter.metrics_fn is None:
            return
        self._pp_metric_keys = keys = tuple(sorted(adapter.metrics_fn()))
        shared = [keys]
        dist.broadcast_object_list(shared, src=self._pp_last_stage_rank, group=self._pp_chain_group)
        reason = (
            None
            if shared[0] == keys
            else (
                f"This rank's per-step PP metric names {keys} differ from the last stage's "
                f"{shared[0]}. The adapter's metrics_fn must name its metrics from configuration, "
                f"not from the batch — the per-step share carries values only."
            )
        )
        reject_across_ranks(reason, "Pipeline metric key consensus", exc_type=ValueError)
        # Checks the trainer has somewhere to put them before the first step rather than at it.
        self._pp_record_metrics({}, "train")

    def _pp_record_metrics(self, metrics: dict[str, torch.Tensor], train_eval: str) -> None:
        """Hand the chain-shared metrics to this trainer's own store.

        The default is ``StoredMetricsMixin.store_metrics``; a trainer keeping its metrics in a
        different structure overrides this.
        """
        self.store_metrics(metrics, train_eval=train_eval)

    def _pp_share_step_metrics(self, train_eval: str) -> None:
        """Broadcast the last stage's per-step metrics down the chain and record them everywhere.

        Only the last stage runs the loss closure, so only it has values; every other rank enters
        with its own (zero) reading and leaves with the last stage's. One fp32 vector per step over
        the chain group — not per microbatch, and not an object hop, because the names are pinned.
        """
        if not self._pp_metric_keys:
            return
        local = self._pp_adapter.metrics_fn()
        missing = [key for key in self._pp_metric_keys if key not in local]
        if missing:
            raise KeyError(
                f"metrics_fn dropped the pinned per-step PP metrics {missing}; the names are pinned "
                f"at setup and every step must report the same set."
            )
        # Per value, not on the stack: a trainer may report a python float beside device tensors,
        # and a mixed-device stack raises rather than moving them.
        device = torch.cuda.current_device()
        values = torch.stack(
            [torch.as_tensor(local[key], dtype=torch.float32).to(device) for key in self._pp_metric_keys]
        )
        dist.broadcast(values, src=self._pp_last_stage_rank, group=self._pp_chain_group)
        self._pp_record_metrics(dict(zip(self._pp_metric_keys, values.unbind(), strict=True)), train_eval)

    def _pp_stage_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-gather ``tensor`` over this stage's ranks — the DP scope under PP.

        A world gather would mix stages: only the last stage's ranks hold sample-level values at
        all, and the other stages would enter with nothing (a hang) or with duplicates.
        """
        stage_size = dist.get_world_size(self._pp_stage_group) if self._pp_stage_group is not None else 1
        if stage_size == 1:
            return tensor
        buffers = [torch.empty_like(tensor) for _ in range(stage_size)]
        dist.all_gather(buffers, tensor.contiguous(), group=self._pp_stage_group)
        return torch.cat(buffers)

    def _pp_broadcast_log_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        """Share a log-step metric dict from the last stage down the chain.

        For metrics whose names depend on the data (offline GRPO drops an advantage-sign bucket no
        sample fell into), which rules out the pinned-key vector above. Global rank 0, the rank HF
        logs from, sits on the first stage. One object hop per log step, not per training step.
        """
        shared = [metrics]
        dist.broadcast_object_list(shared, src=self._pp_last_stage_rank, group=self._pp_chain_group)
        return shared[0]

    def _pp_frozen_row_pads(self) -> dict[str, int]:
        """Fill values for the inert rows :meth:`_pp_pad_rows_to_frozen` appends to a partial batch.

        The fixed-shape collator's pads (pad id, ignore label, the adapter's ``pad_spec``), except
        that ``attention_mask`` fills with ones, since a fully-masked row would NaN the softmax.
        Every extra target fills with 0 so it stays row-aligned with the padded batch.
        """
        pads = dict(self.data_collator.pad_values)
        pads["attention_mask"] = 1
        for key in self._pp_adapter.extra_target_keys:
            pads.setdefault(key, 0)
        return pads

    def _pp_pad_rows_to_frozen(self, inputs: dict, frozen_rows: int, pad_values) -> dict:
        """Row-pad a partial eval batch to the pipeline's frozen row count with inert filler rows.

        Only keys listed in ``pad_values`` are padded; an over-frozen batch is an error, since the
        P2P buffer shapes cannot grow after the first step.
        """
        rows = inputs["input_ids"].size(0)
        if rows > frozen_rows:
            raise ValueError(f"Eval batch of {rows} rows exceeds the pipeline's frozen batch size {frozen_rows}.")
        if rows == frozen_rows:
            return inputs
        out = dict(inputs)
        for key, pad_value in pad_values.items():
            tensor = inputs.get(key)
            if tensor is None:
                continue
            if key == "position_ids":
                # Same rule as FixedShapeCollator's tail: a constant-0 filler row is one varlen
                # segment per token, and the FA4 backward pays a fixed per-segment cost, so a full
                # filler row is orders of magnitude more expensive. The ramp keeps each filler row a
                # handful of no-op documents; their labels are all ignore regardless.
                row = pad_tail_positions(tensor.shape[1], tensor.dtype).to(tensor.device)
                fill = row.unsqueeze(0).expand(frozen_rows - rows, -1)
            else:
                fill = torch.full(
                    (frozen_rows - rows, *tensor.shape[1:]), pad_value, dtype=tensor.dtype, device=tensor.device
                )
            out[key] = torch.cat([tensor, fill], dim=0)
        return out

    def _pp_prediction_step(self, inputs, prediction_loss_only: bool, ignore_keys=None):
        """Evaluation step through the pipeline schedule. Collective over the chain.

        Two paths, chosen by whether anything consumes predictions. Without ``compute_metrics`` the
        schedule runs its own microbatched loss and no rank holds the merged output. With it, one
        forward-only pass yields the last stage's merged output, from which both the loss and the
        predictions derive.

        The adapter's ``predictions_fn`` shrinks what crosses the chain, running on the last stage
        before the broadcast; ``preprocess_logits_for_metrics`` is hoisted onto it at setup
        (:meth:`_pp_hoist_metrics_reduction`) and must not be re-applied here.

        A partial final eval batch is row-padded to the frozen batch size with the inert rows of
        :meth:`_pp_frozen_row_pads`, which contribute nothing to the loss.
        """
        del ignore_keys
        adapter = self._pp_adapter
        inputs = self._prepare_inputs(inputs)
        if adapter.batch_transform is not None:
            inputs = adapter.batch_transform(inputs)
        # Per-batch mean, matching non-PP eval. Computed before the row padding so pair/row-count
        # normalizers see the true batch (the causal-LM token count is pad-invariant either way).
        count = (
            adapter.eval_normalizer(inputs)
            if adapter.eval_normalizer is not None
            else self._pp_eval_token_normalizer(inputs["labels"])
        )

        rows = inputs["input_ids"].size(0)
        frozen_rows = self.args.per_device_train_batch_size * adapter.rows_per_example
        inputs = self._pp_pad_rows_to_frozen(inputs, frozen_rows, self._pp_frozen_row_pads())
        extras = {key: inputs[key] for key in adapter.extra_target_keys} or None

        if prediction_loss_only or self.compute_metrics is None:
            loss = self._pp_runtime.eval_loss(
                inputs["input_ids"],
                inputs["labels"],
                attention_mask=inputs.get("attention_mask"),
                position_ids=inputs.get("position_ids"),
                num_items_in_batch=count,
                extra_targets=extras,
            )
            self._pp_share_step_metrics("eval")
            return self._pp_broadcast_loss_from_last_stage(loss), None, None

        outputs = self._pp_runtime.forward_only(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            position_ids=inputs.get("position_ids"),
        )
        loss, predictions = None, None
        if outputs is not None:
            target = {"labels": inputs["labels"], **extras} if extras else inputs["labels"]
            loss = (adapter.token_loss_fn(outputs, target) / count).detach()
            predictions = outputs if adapter.predictions_fn is None else adapter.predictions_fn(outputs, inputs)
            predictions = predictions.detach()
        self._pp_share_step_metrics("eval")
        loss = self._pp_broadcast_loss_from_last_stage(loss)
        predictions = self._pp_broadcast_output_from_last_stage(predictions)
        labels = inputs["labels"] if adapter.eval_labels_fn is None else adapter.eval_labels_fn(inputs)
        # Trim filler rows in the eval output's own row unit, which a predictions_fn may change: a
        # preference adapter emits one row per pair, so slicing by the batch's row count would keep
        # inert pad pairs, which score as ties and inflate every partial final batch's metric.
        # Predictions and labels trim together or not at all: an adapter reducing to a fixed-size
        # batch summary emits the same shape on both legs, and trimming one alone hands
        # compute_metrics mismatched lengths and desyncs gather_for_metrics across replicas.
        if adapter.row_aligned_eval_outputs:
            labels = labels[: labels.size(0) * rows // frozen_rows]
            predictions = predictions[: predictions.size(0) * rows // frozen_rows]
        return loss, predictions, labels
