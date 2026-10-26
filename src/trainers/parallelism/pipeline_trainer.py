"""Pipeline-parallel trainer integration: the split, the schedule-driven step, clipping and eval.

The schedule replaces the inherited HF step and drives every microbatch's forward and backward, so
the trainer never calls ``accelerator.backward`` and runs with ``gradient_accumulation_steps=1``: the
user's accumulation becomes the microbatch count. Composes with FSDP2 and EP inside each stage.
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.distributed as dist
from accelerate.logging import get_logger

from src.callbacks.moe_metrics import MoELoadMetricsCallback
from src.checkpoint.writing import save_dtype_caster
from src.data.collators.fixed_shape import FixedShapeCollator
from src.data.collators.packing import DataCollatorWithPacking, pad_tail_positions
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
)
from src.distributed.pipeline_parallel.stage_losses import FusedHeadLoss, causal_lm_token_loss
from src.env import env_flag
from src.kernels.logprobs import loss_token_counts_per_row
from src.models.context_window import context_window_from_config
from src.models.modality import is_vlm_model
from src.runtime.consensus import reject_across_ranks
from src.runtime.filesystem import is_fs_save_rank
from src.runtime.ranks import is_global_main_process
from src.trainers.parallelism.grad_clipping import pp_clip_grad_norm
from src.trainers.parallelism.pipeline_loss_adapter import PipelineLossAdapter
from src.trainers.parallelism.pipeline_rules import (
    reject_pp_collator,
    reject_pp_compute_metrics,
    reject_pp_peft,
    reject_pp_vlm,
    reject_uncovered_pp_ep_layers,
    reject_unsupported_pp_training_args,
    require_pad_token_id,
    require_pp_stage_model,
)
from src.trainers.training_args_overrides import uses_evaluation

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


class PipelineTrainerMixin:
    """PP hooks for :class:`DistributedTrainerMixin`; inert unless ``pp_size > 1``."""

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

    def _reject_unsupported_pp_options(self, ctor_args: tuple, kwargs: dict) -> None:
        """Hook: reject this trainer's own PP-incompatible options, under PP only, before the split; ctor
        parameters are read out of ``kwargs``, the remaining positionals out of ``ctor_args``."""

    def _split_pipeline_stage(self, kwargs: dict, training_args, ctor_args: tuple = ()) -> dict:
        """Split the model into this rank's stage before ``super().__init__`` sees it, after the
        refusals that must precede any wrapping. No-op at ``pp_size == 1``."""
        config: ParallelismConfig = self.parallelism_config
        if not config.is_pp_mode:
            return kwargs

        self._reject_unsupported_pp_options(ctor_args, kwargs)
        reject_unsupported_pp_training_args(
            training_args, save_sharded_ep=self.save_sharded_ep, evaluates=uses_evaluation(training_args)
        )
        reject_pp_peft(kwargs.get("peft_config"))
        model = require_pp_stage_model(kwargs)
        # The name catches an unregistered VLM (remote code, a custom class) whose config declares
        # nothing; passing the config keeps the probe off the hub.
        if is_vlm_model(getattr(model.config, "_name_or_path", "") or "", config=model.config):
            reject_pp_vlm(kwargs.get("train_dataset"), kwargs.get("eval_dataset"), kwargs.get("data_collator"))
            # Held on the rank that writes the shared files, for every checkpoint to re-emit under the
            # wrapper layout.
            self._pp_wrapper_state = stash_wrapper_state(model) if is_fs_save_rank() else {}
            if is_global_main_process():
                logger.info(
                    "Text-only run of a multimodal checkpoint under pipeline parallelism: %d tensors "
                    "(%.1fM values) outside the text backbone — the vision tower and projector — are held "
                    "by no stage. They stay untrained and are re-emitted unchanged in every checkpoint, so "
                    "the export keeps the %s layout.",
                    len(self._pp_wrapper_state),
                    sum(t.numel() for t in self._pp_wrapper_state.values()) / 1e6,
                    type(model).__name__,
                )

        self._pp_n_microbatches = resolve_microbatch_count(
            config, training_args.gradient_accumulation_steps, training_args.per_device_train_batch_size
        )
        training_args.gradient_accumulation_steps = 1

        kwargs["model"] = build_pipeline_stage(
            model, config.pp_rank, config.pp_size, moe_balancing=self._moe_balancing, pp_split=config.pp_split
        )

        context_window = context_window_from_config(kwargs["model"].config)
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

    def _configure_pipeline_parallel(self):
        """Post-``__init__`` PP setup: the fixed-shape collator, the signature-column pin, the stage-scoped
        (EP-aware) FSDP2 wrap, the runner and its clip.

        ``_pp_loss_adapter`` is called once, here, before anything reads the adapter, so a trainer's
        adapter builder may seed the per-step state its loss closes over.
        """
        config: ParallelismConfig = self.parallelism_config
        stage: PipelineStageModule = self.model
        self._pp_adapter = adapter = self._pp_loss_adapter()
        # Wrap first: the collator declares which dataset columns it needs, and the pin must union them
        # in. Pinning first would prune a packing collator's `seq_lengths`, collapsing every packed row
        # into a single attended document instead of failing.
        self._pp_wrap_collator()
        self._pin_pp_signature_columns(adapter)
        self._pp_chain_group = build_pipeline_group(config)
        ep_layers = self._ep_layers()
        self._wrap_pp_stage(stage, ep_layers)
        self._pp_runner = PipelineRunner(
            stage,
            config,
            self._pp_chain_group,
            torch.device("cuda", torch.cuda.current_device()),
            self._pp_n_microbatches,
            adapter.stage_loss(fuse_head=env_flag("HALO_PP_FUSED_HEAD_LOSS", True)),
            rows_per_example=type(self).FORWARD_ROWS_PER_EXAMPLE,
        )
        # Plain-tensor expert shards need the EP-aware clip (deferred sweep + dispatch-group norm
        # aggregation); dense and ep1-FSDP-sharded stages are all DTensors → direct PP clip.
        if ep_layers and not config.experts_fsdp_managed:
            self._install_ep_clip()
        else:
            self.accelerator.clip_grad_norm_ = pp_clip_grad_norm
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
            f", {len(ep_layers)} EP MoE layers" if ep_layers else "",
            ", fused head loss (no logits plane)" if isinstance(self._pp_runner.loss, FusedHeadLoss) else "",
        )

    def _pp_wrap_collator(self):
        """Wrap the trainer's final collator for fixed shapes, after ``super().__init__`` so every
        collator source is covered. Idempotent."""
        collator = self.data_collator
        if isinstance(collator, FixedShapeCollator):
            return
        reject_pp_collator(
            collator,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            n_microbatches=self._pp_n_microbatches,
        )
        if isinstance(collator, DataCollatorWithPacking):
            # The collator flattens [B, L] -> [1, B*L] for transformers' batch-size-1 packed path;
            # under PP the dim-0 microbatch split does that instead, one packed row per microbatch.
            collator.flatten_to_single_row = False
        pad_id = require_pad_token_id(self.processing_class, "Pipeline parallelism pads every batch to max_length")
        self.data_collator = FixedShapeCollator(
            collator, self.args.max_length, resolve_batch_pad_values(pad_id, self._pp_adapter.pad_spec)
        )

    def _pin_pp_signature_columns(self, adapter: PipelineLossAdapter) -> None:
        """Pin HF's signature columns to what the runner and the collator read: HF prunes dataset columns
        against the forward signature, and a stage's forward takes hidden_states, so input_ids and labels
        would be dropped. An adapter that keeps the trainer's own columns leaves HF's derivation."""
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

    def _wrap_pp_stage(self, stage: PipelineStageModule, ep_layers: list) -> None:
        """COLLECTIVE: build the stage group, refuse a PP+EP stage without EP layers, and FSDP2-wrap the
        stage over the stage group, EP-aware when it carries EP layers (their capacity-generation hook the
        stage already registered on itself)."""
        config = self.parallelism_config
        # Deliberately unconditional: build_stage_group issues pp_size collective dist.new_group calls
        # while `ep_layers` is rank-dependent (a hybrid MoE with leading dense layers can give one stage
        # none). Gating it deadlocks inside new_group, before any raise prints.
        self._pp_stage_group = build_stage_group(config)
        reject_uncovered_pp_ep_layers(config, ep_layers)
        if ep_layers:
            self._apply_ep_aware_dp_fsdp2(
                stage,
                exclusions=self._fsdp_exclusions(),
                fallback_dp_size=config.stage_world_size,
                fallback_dp_group=self._pp_stage_group,
            )
        else:
            self._apply_dp_fsdp2(
                stage,
                config.stage_world_size,
                excluded_params=self._fsdp_exclusions().params,
                dp_group=self._pp_stage_group,
            )

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

        Without ``compute_metrics`` the schedule runs its own microbatched loss; with it, one forward-only
        pass yields the last stage's output, from which the loss and the predictions derive. The examples
        padding an eval split's final round come back as inert filler rows the loss and metrics never read.
        """
        del ignore_keys
        adapter = self._pp_adapter
        inputs = self._prepare_inputs(inputs)
        if adapter.batch_transform is not None:
            inputs = adapter.batch_transform(inputs)

        rows_per_example = type(self).FORWARD_ROWS_PER_EXAMPLE
        rows = inputs["input_ids"].size(0)
        frozen_rows = self.args.per_device_train_batch_size * rows_per_example
        pad_values = self._pp_frozen_row_pads()
        real_examples = self.eval_split_rows(rows // rows_per_example)
        real_rows = real_examples * rows_per_example
        inputs = {key: value[:real_rows] if key in pad_values else value for key, value in inputs.items()}
        if adapter.step_state_fn is not None:
            adapter.step_state_fn(inputs)
        count = self._pp_eval_count(inputs, real_examples)
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
        # Trim filler rows in the eval output's own row unit, which a predictions_fn may change (a
        # preference adapter emits one row per pair), and trim both together, or gather_for_metrics
        # desyncs across replicas.
        labels = labels[: labels.size(0) * rows // frozen_rows]
        predictions = predictions[: predictions.size(0) * rows // frozen_rows]
        return loss, predictions, labels

    def _pp_eval_count(self, inputs: dict, real_examples: int) -> torch.Tensor | float:
        """The eval loss denominator, taken before the row padding so a pair/row-count normalizer sees
        the real examples alone; a step of padding alone divides by 1."""
        if not real_examples:
            return 1.0
        if self._pp_adapter.eval_normalizer is not None:
            return self._pp_adapter.eval_normalizer(inputs)
        # Eval loss is HF's mean over per-rank batch losses, so the rank's own count divides.
        return loss_token_counts_per_row(inputs["labels"]).sum().to(dtype=torch.float32).clamp(min=1.0)

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

    def _pp_broadcast_log_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        """COLLECTIVE over the chain: the last stage's log-step metric dict, for metrics whose names
        depend on the data (one object hop per log step)."""
        shared = [metrics]
        dist.broadcast_object_list(shared, src=self._pp_last_stage_rank, group=self._pp_chain_group)
        return shared[0]

    def _pp_loss_adapter(self) -> PipelineLossAdapter:
        """This trainer's pipeline-loss contract: the causal-LM one (summed shifted CE, token-count normalizer)
        SFT needs; a PP-enabled trainer with another objective overrides it."""
        return PipelineLossAdapter(token_loss_fn=causal_lm_token_loss)
