"""Pipeline-parallel trainer integration: the split, the schedule-driven step, clipping and eval.

The schedule replaces the inherited HF step and drives every microbatch's forward and backward, so
the trainer never calls ``accelerator.backward`` and runs with ``gradient_accumulation_steps=1``: the
user's accumulation becomes the microbatch count. Composes with FSDP2 and EP inside each stage.
"""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from torch.distributed.tensor import DTensor

from src.callbacks.moe_metrics import MoELoadMetricsCallback
from src.checkpoint.writing import save_dtype_caster
from src.data.collators.fixed_shape import FixedShapeCollator
from src.data.collators.packing import DataCollatorWithFlattening, DataCollatorWithPacking, pad_tail_positions
from src.data.vlm import VLM_IMAGE_COLUMNS, dataset_image_evidence
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.pipeline_parallel.groups import (
    broadcast_scalar_from_last_stage,
    broadcast_tensor_from_last_stage,
    build_pipeline_group,
    build_stage_group,
)
from src.distributed.pipeline_parallel.runner import (
    PP_BATCH_PAD_VALUES,
    PipelineRunner,
    resolve_batch_pad_values,
    resolve_microbatch_count,
)
from src.distributed.pipeline_parallel.stage import (
    PipelineStageModule,
    StageLayout,
    build_pipeline_stage,
    reject_reentrant_checkpointing,
)
from src.distributed.pipeline_parallel.stage_losses import FusedHeadLoss
from src.env import env_flag
from src.hardware import current_device
from src.kernels.logprobs import loss_token_counts_per_row
from src.model_loading.tokenizer_setup import resolve_tokenizer
from src.models.modality import is_vlm_model
from src.runtime.consensus import reject_across_ranks
from src.runtime.filesystem import is_fs_save_rank, is_output_shared_filesystem
from src.runtime.ranks import get_num_nodes, is_global_main_process
from src.trainers.mixins.grad_clip import (
    clip_parameters,
    clipping_enabled,
    local_grad_norm_sq,
    require_l2,
    scale_shards_to_max_norm_,
)
from src.trainers.mixins.pp_gates import (
    reject_pp_activation_offloading,
    reject_pp_compute_metrics,
    reject_pp_peft,
)
from src.trainers.mixins.validation import evaluation_runs
from src.trainers.parallelism.pipeline_loss_adapter import PipelineLossAdapter

logger = get_logger(__name__, log_level="info")


def wrapper_state_outside_stages(model) -> dict[str, torch.Tensor]:
    """The tensors of ``model`` that no pipeline stage holds (a composite wrapper's vision tower and
    projector), by their unsplit names; empty for a plain causal LM."""
    layout = StageLayout.of(model)
    # An empty backbone prefix means the model is the backbone: every tensor belongs to a stage.
    if not layout.backbone_prefix:
        return {}
    owned = (f"{layout.backbone_prefix}.", f"{layout.head_attr}.")
    return {name: tensor for name, tensor in model.state_dict().items() if not name.startswith(owned)}


def collator_image_evidence(collator) -> str | None:
    """Why ``collator`` counts as an image consumer, or ``None``: it declares a column of
    :data:`~src.data.vlm.VLM_IMAGE_COLUMNS` among those it needs."""
    consumed = set(getattr(collator, "required_dataset_columns", ()) or ()) & set(VLM_IMAGE_COLUMNS)
    return f"the data collator consuming {sorted(consumed)}" if consumed else None


def stash_wrapper_state(model) -> dict[str, torch.Tensor]:
    """Host copies of :func:`wrapper_state_outside_stages`, at the artifact's save dtype.

    Cast through the caster built on the whole wrapper: its keep-sets (norms, fp32 pins) are
    tree-derived, and the stage held at save time no longer has the tower's modules.
    """
    cast = save_dtype_caster(model)
    return {
        name: cast(name, tensor.detach()).to("cpu", copy=True)
        for name, tensor in wrapper_state_outside_stages(model).items()
    }


def admit_multimodal_wrapper(model, kwargs: dict) -> dict[str, torch.Tensor]:
    """Admit a multimodal wrapper under PP only for a run that feeds it no images.

    The split keeps the text backbone and the task head, so the vision tower and projector are in no
    stage; a run whose datasets or collator carry images is refused, naming the evidence. A text-only
    run trains the text tower, and the tower's untouched tensors are returned (on the save rank) for
    every checkpoint to re-emit under the wrapper layout.
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
            f"keeps only the text backbone and the task head, and {evidence} means this run feeds "
            f"images the model would never see. Train the VLM without pipeline parallelism, or drop "
            f"the image data for a text-only run of its text tower."
        )
        if evidence
        else None
    )
    # Joined across ranks: the dataset probe is rank-local (a pre-sharded dataset can carry images
    # on one rank only), and a lone raise would leave the peers at the split's collectives.
    reject_across_ranks(reason, "Pipeline-parallel multimodal gate", exc_type=ValueError)
    # Every checkpoint re-emits these from the rank that writes the shared files; the other ranks
    # free them with the wrapper right after the split.
    stash = stash_wrapper_state(model) if is_fs_save_rank() else {}
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
    """Why this rank's stage cannot run PP+EP, or ``None``; rank-local, so the caller joins it across
    ranks (a hybrid MoE with leading dense layers can give one stage no EP modules)."""
    if not config.has_ep_group or ep_modules:
        return None
    return (
        f"PP stage {config.pp_rank}/{config.pp_size} carries no EP-patched modules: the model was "
        "loaded without EP patching (load_distributed_model must run before the trainer), or the "
        "partition gave this stage only dense layers, whose collectives would deadlock against its MoE "
        "peers. Pick a pipeline_parallel_size or pipeline_split that puts a MoE layer on every stage."
    )


class PipelineTrainerMixin:
    """PP hooks for :class:`DistributedTrainerMixin`; inert unless ``pp_size > 1``."""

    # Sequence rows one example becomes in the forward (2 for a chosen/rejected pair): the pipeline's
    # frozen row count is per_device_train_batch_size times this.
    _forward_rows_per_example: ClassVar[int] = 1
    _pp_runner: PipelineRunner | None = None
    _pp_adapter: PipelineLossAdapter | None = None
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

        Runs only under PP. Ctor parameters are read out of ``kwargs``; ``ctor_args`` carries the
        trainer's remaining positionals.
        """

    def _maybe_prepare_pipeline_model(self, kwargs: dict, training_args, ctor_args: tuple = ()) -> dict:
        """Split the model into this rank's stage before ``super().__init__`` sees it, after the
        rejections that must precede any wrapping. No-op at ``pp_size == 1``."""
        config: ParallelismConfig = self.parallelism_config
        if not config.is_pp_mode:
            return kwargs

        self._validate_pp_mode(ctor_args, kwargs)

        model = kwargs.get("model")
        if not getattr(training_args, "max_length", None):
            raise ValueError(
                "Pipeline parallelism requires a fixed sequence length (the P2P buffer shapes "
                "freeze on the first step). Set max_length in the training config."
            )
        if self.save_sharded_ep:
            raise ValueError(
                "save_sharded_ep is not supported under pipeline parallelism: the per-rank shard format "
                "keys tensors with no pipeline-stage layer offset, so stages would collide. The PP save "
                "already writes one complete-tensor shard per stage under the unsplit model's names."
            )
        if getattr(training_args, "push_to_hub", False) and not is_output_shared_filesystem() and get_num_nodes() > 1:
            raise ValueError(
                "push_to_hub is not supported under pipeline parallelism with a per-node output "
                "filesystem (DIST_OUTPUT_SHARED_FILESYSTEM=0) across nodes: global rank 0 uploads, and its "
                "node lacks the other nodes' stage shards. Write checkpoints to a shared output "
                "filesystem, or gather every node's checkpoint directory into one and upload that."
            )
        if training_args.gradient_checkpointing:
            # The stage's gradient_checkpointing_enable refuses it too; here it fails before the split.
            reject_reentrant_checkpointing(training_args.gradient_checkpointing_kwargs)
        if (
            evaluation_runs(training_args)
            and training_args.per_device_eval_batch_size != training_args.per_device_train_batch_size
        ):
            raise ValueError(
                f"per_device_eval_batch_size ({training_args.per_device_eval_batch_size}) must "
                f"equal per_device_train_batch_size ({training_args.per_device_train_batch_size}) "
                f"under pipeline parallelism: the P2P buffer shapes are frozen to the training batch."
            )
        reject_pp_peft(kwargs.get("peft_config"))
        # Here rather than per trainer: SFTConfig and RewardConfig carry the flag too.
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
                "torch_compile is not supported under pipeline parallelism: the schedule captures the "
                "stage module at setup, so a compiled wrapper installed afterwards never runs. Disable "
                "torch_compile."
            )

        self._pp_n_microbatches = resolve_microbatch_count(
            config, training_args.gradient_accumulation_steps, training_args.per_device_train_batch_size
        )
        training_args.gradient_accumulation_steps = 1

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

    def _setup_pipeline_parallel(self):
        """Post-``__init__`` PP setup: stage-scoped (EP-aware) FSDP, the runner, clip, and column pinning."""
        config: ParallelismConfig = self.parallelism_config
        stage: PipelineStageModule = self.model

        self._pp_adapter = adapter = self._pp_loss_adapter()

        # Wrap first: the collator declares which dataset columns it needs, and the pin below must
        # union them in. Pinning first would prune a packing collator's `seq_lengths`, collapsing
        # every packed row into a single attended document instead of failing.
        self._pp_wrap_collator()

        # HF prunes dataset columns against the forward signature, and a stage's forward takes
        # hidden_states; without this pin, input_ids and labels are dropped.
        if adapter.pin_runner_columns:
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

        self._pp_chain_group = build_pipeline_group(config)

        ep_modules = self._find_ep_modules()
        # Deliberately unconditional: build_stage_group issues pp_size collective dist.new_group
        # calls while `ep_modules` is rank-dependent (a hybrid MoE with leading dense layers can give
        # one stage no EP modules). Gating this deadlocks inside new_group, before any raise prints.
        self._pp_stage_group = build_stage_group(config)
        reject_across_ranks(pp_ep_coverage_reason(config, ep_modules), "PP stage EP coverage")
        topo = f", PP stage {config.pp_rank}/{config.pp_size}"
        if ep_modules:
            # The stage registered the EP capacity-generation hook on itself (PipelineStageModule); only
            # the FSDP2 wrap is left to do here.
            self._apply_ep_aware_dp_fsdp2(
                stage,
                exclusions=self._fsdp_exclusions(),
                fallback_dp_size=config.stage_world_size,
                fallback_dp_group=self._pp_stage_group,
                topo=topo,
            )
        else:
            self._apply_dp_fsdp2(
                stage,
                config.stage_world_size,
                excluded_params=self._fsdp_exclusions().params,
                dp_group=self._pp_stage_group,
                topo=topo,
                detail=f"stage-scoped data parallel ({config.stage_world_size} ranks per stage)",
            )

        self._pp_runner = PipelineRunner(
            stage,
            config,
            self._pp_chain_group,
            torch.device("cuda", torch.cuda.current_device()),
            self._pp_n_microbatches,
            adapter.stage_loss(fuse_head=env_flag("HALO_PP_FUSED_HEAD_LOSS", True)),
            rows_per_example=type(self)._forward_rows_per_example,
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
            ", fused head loss (no logits plane)" if isinstance(self._pp_runner.loss, FusedHeadLoss) else "",
        )

    def _pp_wrap_collator(self):
        """Wrap the trainer's final collator for fixed shapes, after ``super().__init__`` so every
        collator source is covered. Idempotent."""
        inner = self.data_collator
        if isinstance(inner, FixedShapeCollator):
            return
        if inner is None:
            raise ValueError("Pipeline parallelism requires a data collator; the trainer has none.")
        if isinstance(inner, DataCollatorWithFlattening):
            # Its cu_seq_lens side outputs cannot be split per microbatch either, and its one row
            # cannot be microbatched at all.
            raise ValueError(
                f"{type(inner).__name__} (padding-free) is incompatible with pipeline parallelism: "
                f"its flattened width is the summed document length, which varies every step, and "
                f"the pipeline's P2P buffers are frozen after the first. Use packing instead "
                f"(`packing: true`), whose packed rows are a fixed shape."
            )
        if isinstance(inner, DataCollatorWithPacking):
            # The collator flattens [B, L] -> [1, B*L] for transformers' batch-size-1 packed path;
            # under PP the dim-0 microbatch split does that instead, one packed row per microbatch.
            if self.args.per_device_train_batch_size != self._pp_n_microbatches:
                raise ValueError(
                    f"Packing under pipeline parallelism needs one packed row per microbatch, but "
                    f"per_device_train_batch_size={self.args.per_device_train_batch_size} and the "
                    f"microbatch count is {self._pp_n_microbatches}: a microbatch holding >1 packed row "
                    f"leaks attention across documents under FlashAttention. Set "
                    f"per_device_train_batch_size equal to the microbatch count."
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
            inner, self.args.max_length, resolve_batch_pad_values(pad_id, self._pp_adapter.pad_spec)
        )

    def _pp_clip_grad_norm(self, parameters, max_norm, norm_type=2):
        """Gradient clipping with the norm accumulated across the whole pipeline.

        Every parameter is sharded across exactly one stage's DP group, so the world sum of the local
        squared norms is the global one (ep1 DTensor experts included; plain-tensor EP expert shards
        use ``ep_clip_grad_norm_`` instead).
        """
        require_l2(norm_type, "Pipeline-parallel")
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

    def _pp_hoist_metrics_reduction(self, adapter: PipelineLossAdapter) -> PipelineLossAdapter:
        """Move the user's ``preprocess_logits_for_metrics`` onto the last stage, or refuse the metric.

        Unreduced, the raw ``[rows, max_length, vocab]`` output would cross the chain and be allocated
        on every stage; HF applies the reduction too late to shrink that hop, so it runs here instead
        and is cleared from HF. It sees the frozen batch, so it must be row-independent.
        """
        if self.compute_metrics is None or adapter.predictions_fn is not None:
            return adapter
        reduction = self.preprocess_logits_for_metrics
        reject_pp_compute_metrics(
            None if reduction is not None else self.compute_metrics,
            "this trainer's predictions are the raw logits plane, which a pipeline would broadcast "
            "whole down the chain. Pass preprocess_logits_for_metrics (e.g. `lambda logits, labels: "
            "logits.argmax(-1)`), which under PP runs on the last stage before the broadcast, or "
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
        """COLLECTIVE over the chain: pin the adapter's per-step metric names and check every rank
        agrees, since the per-step share broadcasts values only."""
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
        # Fails before the first step if the trainer has nowhere to store them.
        self.store_metrics({}, train_eval="train")

    @property
    def _pp_last_stage_rank(self) -> int:
        """Global rank of this chain's last stage: the source of every chain broadcast."""
        return self.parallelism_config.pp_chain_ranks()[-1]

    def _pp_training_step(self, inputs) -> torch.Tensor:
        """One optimizer step's microbatches through the pipeline, the loss broadcast for logging.

        The schedule already ran every backward, so HF gets a detached loss with gradients in place.
        """
        self.model.train()
        inputs = self._prepare_inputs(inputs)

        adapter = self._pp_adapter
        if adapter.batch_transform is not None:
            inputs = adapter.batch_transform(inputs)
        if adapter.step_state_fn is not None:
            adapter.step_state_fn(inputs)
        normalizer = self._pp_default_normalizer(inputs) if adapter.normalizer is None else adapter.normalizer(inputs)
        extras = {key: inputs[key] for key in adapter.extra_target_keys} or None

        with self.compute_loss_context_manager():
            loss = self._pp_runner.step(
                inputs["input_ids"],
                inputs["labels"],
                attention_mask=inputs.get("attention_mask"),
                position_ids=inputs.get("position_ids"),
                num_items_in_batch=normalizer,
                extra_targets=extras,
            )

        self._pp_share_step_metrics("train")
        return self._pp_broadcast_loss_from_last_stage(loss)

    def _pp_default_normalizer(self, inputs) -> torch.Tensor:
        """COLLECTIVE over the stage group: the DP-global loss-token count / dp, the causal-LM normalizer.

        FSDP averages gradients over the replicas, so per-replica losses normalized by G/dp make the
        averaged gradient the global token mean. HF's ``num_items_in_batch`` cannot supply it: a
        stage's forward takes no loss kwargs.
        """
        count = loss_token_counts_per_row(inputs["labels"]).sum().to(dtype=torch.float32)
        dist.all_reduce(count, group=self._pp_stage_group)
        return (count / self.parallelism_config.stage_world_size).clamp(min=1.0)

    def _pp_share_step_metrics(self, train_eval: str, rows: int = 1) -> None:
        """COLLECTIVE over the chain: broadcast the last stage's per-step metrics (only it runs the
        loss) as one fp32 vector and record them on every rank, weighed by ``rows``."""
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
        self.store_metrics(
            dict(zip(self._pp_metric_keys, values.unbind(), strict=True)), train_eval=train_eval, rows=rows
        )

    def _pp_broadcast_loss_from_last_stage(self, tensor) -> torch.Tensor:
        """The last stage's scalar loss, broadcast down the chain so every rank logs the same value."""
        return broadcast_scalar_from_last_stage(tensor, self._pp_last_stage_rank, self._pp_chain_group)

    def _pp_prediction_step(self, inputs, prediction_loss_only: bool, ignore_keys=None):
        """COLLECTIVE over the chain: one evaluation step through the pipeline schedule.

        Without ``compute_metrics`` the schedule runs its own microbatched loss; with it, one
        forward-only pass yields the last stage's merged output, from which the loss and the
        predictions derive. The examples padding an eval split's final round leave first and come
        back as inert filler rows, so the loss, its normalizer and the metrics read the real examples
        alone, while the predictions keep the batch's rows for the evaluation gather to cut.
        """
        del ignore_keys
        adapter = self._pp_adapter
        inputs = self._prepare_inputs(inputs)
        if adapter.batch_transform is not None:
            inputs = adapter.batch_transform(inputs)

        rows_per_example = type(self)._forward_rows_per_example
        rows = inputs["input_ids"].size(0)
        frozen_rows = self.args.per_device_train_batch_size * rows_per_example
        pad_values = self._pp_frozen_row_pads()
        real_examples = self.eval_split_rows(rows // rows_per_example)
        real_rows = real_examples * rows_per_example
        inputs = {key: value[:real_rows] if key in pad_values else value for key, value in inputs.items()}
        if adapter.step_state_fn is not None:
            adapter.step_state_fn(inputs)
        # Before the row padding, so pair/row-count normalizers see the real examples alone; a step of
        # padding alone sums to 0 and divides by 1.
        if not real_examples:
            count = 1.0
        elif adapter.eval_normalizer is not None:
            count = adapter.eval_normalizer(inputs)
        else:
            # Eval loss is HF's mean over per-rank batch losses, so the rank's own count divides.
            count = loss_token_counts_per_row(inputs["labels"]).sum().to(dtype=torch.float32).clamp(min=1.0)
        inputs = self._pp_pad_rows_to_frozen(inputs, frozen_rows, pad_values)
        extras = {key: inputs[key] for key in adapter.extra_target_keys} or None

        if prediction_loss_only or self.compute_metrics is None:
            loss = self._pp_runner.eval_loss(
                inputs["input_ids"],
                inputs["labels"],
                attention_mask=inputs.get("attention_mask"),
                position_ids=inputs.get("position_ids"),
                num_items_in_batch=count,
                extra_targets=extras,
            )
            self._pp_share_step_metrics("eval", rows=real_examples)
            return self._pp_broadcast_loss_from_last_stage(loss), None, None

        outputs = self._pp_runner.forward_only(
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
        self._pp_share_step_metrics("eval", rows=real_examples)
        loss = self._pp_broadcast_loss_from_last_stage(loss)
        predictions = self._pp_broadcast_output_from_last_stage(predictions)
        labels = inputs["labels"] if adapter.eval_labels_fn is None else adapter.eval_labels_fn(inputs)
        # Trim filler rows in the eval output's own row unit, which a predictions_fn may change: a
        # preference adapter emits one row per pair, so slicing by the batch's row count would keep
        # inert pad pairs, which score as ties and inflate every partial final batch's metric.
        # Predictions and labels trim together: trimming one alone hands compute_metrics mismatched
        # lengths and desyncs gather_for_metrics across replicas.
        labels = labels[: labels.size(0) * rows // frozen_rows]
        predictions = predictions[: predictions.size(0) * rows // frozen_rows]
        return loss, predictions, labels

    def _pp_frozen_row_pads(self) -> dict[str, int]:
        """Fill values for the inert rows :meth:`_pp_pad_rows_to_frozen` appends: the collator's pads,
        but ``attention_mask`` fills with ones (a fully-masked row NaNs the softmax) and every extra
        target with 0."""
        pads = dict(self.data_collator.pad_values)
        pads["attention_mask"] = 1
        for key in self._pp_adapter.extra_target_keys:
            pads.setdefault(key, 0)
        return pads

    def _pp_pad_rows_to_frozen(self, inputs: dict, frozen_rows: int, pad_values) -> dict:
        """Row-pad a partial eval batch to the pipeline's frozen row count with inert filler rows,
        padding only the keys in ``pad_values``."""
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
                # A constant-0 filler row is one varlen segment per token, each paying the FA4
                # backward's fixed per-segment cost; the ramp keeps it a handful of no-op documents.
                row = pad_tail_positions(tensor.shape[1], tensor.dtype).to(tensor.device)
                fill = row.unsqueeze(0).expand(frozen_rows - rows, -1)
            else:
                fill = torch.full(
                    (frozen_rows - rows, *tensor.shape[1:]), pad_value, dtype=tensor.dtype, device=tensor.device
                )
            out[key] = torch.cat([tensor, fill], dim=0)
        return out

    def _pp_broadcast_output_from_last_stage(self, tensor: torch.Tensor | None) -> torch.Tensor:
        """The last stage's prediction tensor, broadcast down the chain (shape carried with it)."""
        return broadcast_tensor_from_last_stage(tensor, self._pp_last_stage_rank, self._pp_chain_group)

    def _reshard_after_evaluation_loop(self) -> None:
        """Re-register the sharded params once the eval loop ends: the forward-only drives leave the
        stage gathered, and everything after the loop must read the shards. Per-rank, no collective."""
        if self._pp_runner is not None:
            reshard_fsdp2_modules(self.model)

    def _pp_stage_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """COLLECTIVE over the stage group: all-gather ``tensor`` over this stage's ranks, the DP scope
        under PP (a world gather would mix stages)."""
        stage_size = dist.get_world_size(self._pp_stage_group)
        if stage_size == 1:
            return tensor
        buffers = [torch.empty_like(tensor) for _ in range(stage_size)]
        dist.all_gather(buffers, tensor.contiguous(), group=self._pp_stage_group)
        return torch.cat(buffers)

    def _pp_broadcast_log_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        """COLLECTIVE over the chain: the last stage's log-step metric dict, for metrics whose names
        depend on the data (one object hop per log step)."""
        shared = [metrics]
        dist.broadcast_object_list(shared, src=self._pp_last_stage_rank, group=self._pp_chain_group)
        return shared[0]
