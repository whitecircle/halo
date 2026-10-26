"""Distributed SFT trainer: TRL's SFTTrainer under the trainer spine, with tree packing, router aux-loss weighting
and its own CP loss and metrics."""

import inspect
import logging
from functools import cached_property
from typing import ClassVar

import torch
from transformers import Trainer
from trl import SFTTrainer
from trl.trainer.utils import entropy_from_logits

from halo.distributed.context_parallel.sequence_chunks import cp_chunk_bounds, cp_shift_against_full_labels
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.model_loading.tokenizer_setup import resolve_tokenizer
from halo.models.attention.tree_plan import TREE_ATTENTION_KEY
from halo.models.moe.balancing import expert_load_token_weights
from halo.models.moe.config_fields import require_router_aux_loss_coef
from halo.models.unwrapping import base_transformers_model
from halo.runtime.ranks import is_global_main_process
from halo.trainers.ctor_slots import get_ctor_arg, get_ctor_config, resolve_ctor_positions
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.parallelism.trainer_rules import reject_compressed_kv_collator, reject_cp_packed_collator
from halo.trainers.world_metrics import WorldMetrics

logger = logging.getLogger(__name__)

# TRL SFTTrainer positional slots, for ctor params arriving via *args — derived from the installed signature.
_CTOR_POSITIONS = resolve_ctor_positions(SFTTrainer, "model", "args", "data_collator")


def _replace_chunked_nll_loss_type(training_args) -> None:
    """Replace TRL's ``chunked_nll`` loss type with ``nll``.

    TRL defaults an unset ``loss_type`` to ``chunked_nll`` whenever ``use_liger_kernel`` is off, and that type
    replaces the model's forward with one that projects ``lm_head`` itself: it skips the family's head transform,
    the router aux loss the forward folds in, and the logits a subclass's loss reads.
    """
    if training_args is None or getattr(training_args, "loss_type", None) != "chunked_nll":
        return
    if is_global_main_process():
        logger.warning(
            "Setting loss_type to 'nll': TRL's 'chunked_nll' replaces the model's forward with its own lm_head "
            "projection, which skips the family's head transform and the router aux loss."
        )
    training_args.loss_type = "nll"


@torch.no_grad()
def _record_cp_metrics(
    metrics: WorldMetrics, outputs, full_labels, full_attention_mask, cp_rank: int, cp_size: int, *, training: bool
) -> None:
    """Record one micro-batch's CP metrics over this rank's chunk, nothing without labels or logits. A non-final
    rank's last logit predicts the next chunk's first token, as in the loss."""
    if full_labels is None or outputs.logits is None:
        return
    shift_logits, shift_labels = cp_shift_against_full_labels(outputs.logits, full_labels, cp_rank, cp_size)
    mask = shift_labels != LABEL_IGNORE_INDEX
    correct = (shift_logits.argmax(dim=-1) == shift_labels) & mask
    metrics.fraction("mean_token_accuracy", correct.sum(), mask.sum())

    per_token_entropy = entropy_from_logits(outputs.logits)
    if full_attention_mask is not None:
        start, end = cp_chunk_bounds(full_attention_mask.size(1), cp_rank, cp_size)
        local_attention_mask = full_attention_mask[:, start:end]
        attended_tokens = local_attention_mask.sum()
        metrics.fraction("entropy", torch.sum(per_token_entropy * local_attention_mask), attended_tokens)
    else:
        attended_tokens = outputs.logits.shape[0] * outputs.logits.shape[1]
        metrics.fraction("entropy", per_token_entropy.sum(), per_token_entropy.numel())
    # Only train accumulates the run's token counter; eval contributes a structural zero.
    metrics.sum("attended_tokens", attended_tokens * int(training))

    aux_loss = getattr(outputs, "aux_loss", None)
    if aux_loss is not None:
        # mean(): a family whose aux loss arrives per layer must still record one value.
        metrics.fraction("aux_loss", aux_loss.detach().mean(), 1)


class DistributedSFTTrainer(DistributedTrainerMixin, SFTTrainer):
    """TRL's SFT trainer under EP, CP, TP, ETP and PP."""

    SUPPORTS_CP = True
    # Canonical last-stage loss; the PP-incompatible knobs are rejected in _split_pipeline_stage.
    PP_REFUSAL = None
    SUPPORTS_LOWP_PRECISION = True
    # Forwards with ``labels``, so ``*ForCausalLM.forward`` folds the router aux loss into outputs.loss.
    ROUTER_AUX_LOSS_ADDED_BY = "model_forward"
    # False lets loss-only eval take Liger's fused-CE ``skip_logits`` path; a subclass whose loss reads
    # ``outputs.logits`` must flip it (lce_forward returns None there, and raises without labels).
    LOSS_READS_LOGITS: ClassVar[bool] = False

    def __init__(self, *args, **kwargs):
        _replace_chunked_nll_loss_type(get_ctor_config(args, kwargs, _CTOR_POSITIONS))
        kwargs = self._init_distributed_config(kwargs, ctor_args=args, ctor_positions=_CTOR_POSITIONS)
        collator = get_ctor_arg(args, kwargs, "data_collator", _CTOR_POSITIONS)
        reject_cp_packed_collator(collator, is_cp_mode=self.parallelism_config.is_cp_mode)
        model = get_ctor_arg(args, kwargs, "model", _CTOR_POSITIONS)
        reject_compressed_kv_collator(collator, getattr(model, "config", None))
        super().__init__(*args, **kwargs)
        if self.parallelism_config.is_cp_mode:
            # The CP loss is its own mean: HF's num_items_in_batch scaling would inflate it cp_size x.
            self.model_accepts_loss_kwargs = False
        self._configure_distributed_modes()

        # Per-mode CP metrics awaiting their once-per-log flush (:meth:`_drain_cp_metrics`).
        self._cp_metrics = {"train": WorldMetrics(), "eval": WorldMetrics()}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """TRL's loss with the router aux term weighed at its coefficient and the attended-token counter corrected
        for replicated ranks; under CP, the wrapper's chunk loss (:meth:`_compute_cp_loss`)."""
        self._warn_once_on_empty_labels(inputs)
        if self.parallelism_config.is_cp_mode:
            loss, outputs = self._compute_cp_loss(model, inputs)
            return (loss, outputs) if return_outputs else loss

        prev_tokens = self._total_train_tokens

        tree_plan = inputs.get(TREE_ATTENTION_KEY)
        with expert_load_token_weights(None if tree_plan is None else tree_plan["leaf_count"]):
            loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
        loss = self._weigh_router_aux_loss(model, loss, outputs, num_items_in_batch)
        # TRL's gather().sum() counts a TP/ETP group's identical batch once per rank.
        non_dp_factor = self.parallelism_config.non_dp_replication_factor
        added = self._total_train_tokens - prev_tokens
        if non_dp_factor > 1 and self.model.training and added > 0:
            self._total_train_tokens = prev_tokens + added // non_dp_factor
            self._metrics["train"]["num_tokens"] = [self._total_train_tokens]
        # TRL's num_tokens counts attended (non-padding) tokens, apart from num_input_tokens_seen (padding
        # included) and num_unmasked_output_tokens_seen (loss tokens only).
        for mode_metrics in self._metrics.values():
            if "num_tokens" in mode_metrics:
                mode_metrics["num_attended_tokens_seen"] = mode_metrics.pop("num_tokens")
        return (loss, outputs) if return_outputs else loss

    def _compute_cp_loss(self, model, inputs):
        """The CP wrapper's loss on this rank's chunk, its metrics folded into the per-log sums; no
        ``num_items_in_batch``, since per-token loss scaling is wrong under CP."""
        self._pad_to_cp_multiple(inputs)
        full_labels, full_attention_mask = inputs.get("labels"), inputs.get("attention_mask")
        inputs["use_cache"] = False
        loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True)
        training = self.model.training
        _record_cp_metrics(
            self._cp_metrics["train" if training else "eval"],
            outputs,
            full_labels,
            full_attention_mask,
            self.cp_config.cp_rank,
            self.parallelism_config.cp_size,
            training=training,
        )
        return loss, outputs

    def _pad_to_cp_multiple(self, inputs) -> None:
        """Pad every sequence column in place to a multiple of ``context_parallel_size``, with the tokenizer's
        pad id (never vocabulary token 0) on ``input_ids`` and the ignore index on ``labels``."""
        cp_size = self.parallelism_config.cp_size
        labels = inputs.get("labels")
        if labels is None or labels.shape[1] % cp_size == 0:
            return
        pad_len = cp_size - labels.shape[1] % cp_size
        pad_token_id = resolve_tokenizer(self.processing_class).pad_token_id
        if pad_token_id is None:
            raise ValueError(
                "Context parallelism pads every sequence to a multiple of "
                "context_parallel_size and needs the tokenizer's pad_token_id; the "
                "processing_class passed to the trainer has none. Set one (commonly the "
                "EOS token) rather than padding with vocabulary token 0."
            )
        fills = {"input_ids": pad_token_id, "labels": LABEL_IGNORE_INDEX, "attention_mask": 0, "position_ids": 0}
        for key, fill in fills.items():
            if key in inputs:
                inputs[key] = torch.nn.functional.pad(inputs[key], (0, pad_len), value=fill)

    def _weigh_router_aux_loss(self, model, loss, outputs, num_items_in_batch):
        """Weigh the router aux loss at its coefficient: ``coef x mean(aux)`` per optimizer step, ``coef x aux`` in an
        eval batch's loss.

        HF's ``num_items_in_batch`` path scales a micro-batch loss by the process count and sums the accumulation
        window, which makes the token-summed LM term the window's mean but weighs the per-micro-batch aux mean
        ``processes x accumulation`` times. The forward adds the term only when handed the labels, which HF
        withholds under a label smoother or a ``compute_loss_func``; it is then added here alone.
        """
        aux = outputs.get("aux_loss") if isinstance(outputs, dict) else None
        if aux is None or num_items_in_batch is None:
            return loss
        weighted = require_router_aux_loss_coef(base_transformers_model(model).config) * aux.to(loss.dtype)
        forward_added = weighted if self.label_smoother is None and self.compute_loss_func is None else 0.0
        target = weighted / self.current_gradient_accumulation_steps if model.training else weighted
        # The factor Trainer.compute_loss multiplies a loss by on the num_items_in_batch path.
        if not self.args.average_tokens_across_devices:
            hf_loss_scale = 1
        elif self.args.n_gpu > 1:
            hf_loss_scale = self.args.n_gpu
        else:
            hf_loss_scale = self.accelerator.num_processes
            if (pc := getattr(self.accelerator, "parallelism_config", None)) is not None:
                hf_loss_scale //= pc.tp_size
        return loss - forward_added * hf_loss_scale + target

    def log(self, logs: dict[str, float], start_time: float | None = None):
        """Fold the deferred CP metric sums into ``_metrics`` before the base trainer averages them."""
        self._drain_cp_metrics()
        super().log(logs, start_time)

    def _drain_cp_metrics(self):
        """COLLECTIVE under CP: fold this log window's CP metrics across the world in one flush and emit them. A
        window with no CP forward anywhere (the final log of a run, or a mode that never ran) emits nothing rather
        than a 0.0 accuracy, a verdict every rank reads off the same flushed result."""
        if not self.parallelism_config.is_cp_mode:
            return
        mode = "train" if self.model.training else "eval"
        world = self._cp_metrics[mode].flush()
        if not world:
            return
        # One token per CP rank, distinct sequences per DP group and no TP+CP, so this cannot double-count.
        self._total_train_tokens += int(world.pop("attended_tokens"))
        self._metrics[mode]["num_attended_tokens_seen"] = [self._total_train_tokens]
        for key, value in world.items():
            self._metrics[mode][key].append(value)

    @cached_property
    def _forward_accepts_skip_logits(self) -> bool:
        """Whether forward was patched with Liger ``lce_forward`` (exposes ``skip_logits``); read at the first eval
        step, so both the class-level patch and TRL's instance-level re-application are visible."""
        model = self._top_level_model()
        try:
            return "skip_logits" in inspect.signature(model.forward).parameters
        except (TypeError, ValueError):
            return False

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Route loss-only eval through Liger's fused CE (``skip_logits``) rather than materialize full
        ``[B, S, vocab]`` fp32 logits; every other clause names a consumer that still needs real logits."""
        if (
            prediction_loss_only
            and not self.LOSS_READS_LOGITS
            and not self.parallelism_config.is_cp_mode
            and self.args.use_liger_kernel
            and "skip_logits" not in inputs
            and inputs.get("labels") is not None
            and self._forward_accepts_skip_logits
        ):
            inputs = {**inputs, "skip_logits": True}
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
