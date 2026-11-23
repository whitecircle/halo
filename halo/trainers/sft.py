"""TRL's SFTTrainer under the trainer spine, with tree packing, router aux-loss weighting, the CP loss and its own loss
metrics, folded across the world once per log."""

import inspect
import logging
from functools import cached_property
from typing import ClassVar

import torch
from transformers import Trainer
from trl import SFTTrainer
from trl.trainer.utils import entropy_from_logits

from halo.distributed.context_parallel.sequence_chunks import cp_local_targets, cp_shift_against_full_labels
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.model_loading.tokenizer_setup import resolve_tokenizer
from halo.models.attention.tree_plan import TREE_ATTENTION_KEY
from halo.models.introspection.unwrapping import base_transformers_model
from halo.models.moe.balancing import expert_load_token_weights
from halo.models.moe.config_fields import require_router_aux_loss_coef
from halo.models.mtp.acceptance import MTP_ACCEPTANCE_GREEDY, MTP_ACCEPTANCE_TV, MTP_SCORED_POSITIONS, MTP_TOP1
from halo.runtime.ranks import is_global_main_process
from halo.trainers.construction.ctor_slots import get_ctor_arg, get_ctor_config, resolve_ctor_positions
from halo.trainers.construction.trainer_rules import reject_cp_packed_collator, reject_multi_document_collator
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.metrics.world_metrics import WorldMetrics

logger = logging.getLogger(__name__)

_CTOR_POSITIONS = resolve_ctor_positions(SFTTrainer, "model", "args", "data_collator")


def _replace_chunked_nll_loss_type(training_args) -> None:
    """TRL defaults an unset ``loss_type`` to ``chunked_nll`` when ``use_liger_kernel`` is off, whose forward projects
    ``lm_head`` itself: it skips the family's head transform, the router aux loss and the logits a subclass's loss
    reads."""
    if training_args is None or getattr(training_args, "loss_type", None) != "chunked_nll":
        return
    if is_global_main_process():
        logger.warning(
            "Setting loss_type to 'nll': TRL's 'chunked_nll' replaces the model's forward with its own lm_head "
            "projection, which skips the family's head transform and the router aux loss."
        )
    training_args.loss_type = "nll"


@torch.no_grad()
def _record_loss_metrics(
    metrics: WorldMetrics, outputs, inputs, labels, cp_shard: tuple[int, int] | None, *, training: bool
) -> None:
    """No collective and no host read. ``cp_shard`` is ``(cp_rank, cp_size)`` when the forward covers its chunk alone;
    a non-final rank's last position then predicts the next chunk's first token, as in the loss."""
    logits = getattr(outputs, "logits", None)
    if labels is not None and logits is not None:
        if cp_shard is not None:
            shift_logits, shift_labels = cp_shift_against_full_labels(logits, labels, *cp_shard)
        else:
            # Prompt-learning adapters prepend virtual-token logits the labels do not cover.
            shift_logits, shift_labels = logits[:, logits.size(1) - labels.size(1) : -1], labels[:, 1:]
        mask = shift_labels != LABEL_IGNORE_INDEX
        tokens = mask.sum()
        metrics.fraction("mean_token_accuracy", ((shift_logits.argmax(dim=-1) == shift_labels) & mask).sum(), tokens)
        metrics.fraction("entropy", (entropy_from_logits(shift_logits) * mask).sum(), tokens)
    elif labels is not None and getattr(outputs, "token_accuracy", None) is not None:
        targets = labels[:, 1:] if cp_shard is None else cp_local_targets(labels, *cp_shard)
        tokens = (targets != LABEL_IGNORE_INDEX).sum()
        metrics.fraction("mean_token_accuracy", outputs.token_accuracy * tokens, tokens)
    # The whole batch on every rank, padding-free rows by their length: the flush divides the world sum by the ranks
    # sharing a batch. Only train advances the run's counter; eval contributes a structural zero.
    attention_mask = inputs.get("attention_mask")
    attended = attention_mask.sum() if attention_mask is not None else inputs["input_ids"].numel()
    metrics.sum("attended_tokens", attended * int(training))

    aux_loss = getattr(outputs, "aux_loss", None)
    if aux_loss is not None:
        # mean(): a family whose aux loss arrives per layer must still record one value.
        metrics.fraction("aux_loss", aux_loss.detach().mean(), 1)

    mtp_split = getattr(outputs, "mtp_loss_split", None)
    if mtp_split is not None:
        metrics.fraction("ntp_ce", mtp_split.next_token_sum, mtp_split.next_token_targets)
        metrics.fraction("mtp_ce", mtp_split.head_sum, mtp_split.head_targets)

    acceptance = getattr(outputs, "mtp_acceptance", None)
    if acceptance is not None:
        metrics.fraction(MTP_ACCEPTANCE_TV, acceptance.overlap_sum, acceptance.positions)
        metrics.fraction(MTP_ACCEPTANCE_GREEDY, acceptance.greedy_sum, acceptance.positions)
        metrics.fraction(MTP_TOP1, acceptance.top1_sum, acceptance.positions)
        metrics.sum(MTP_SCORED_POSITIONS, acceptance.positions)


class DistributedSFTTrainer(DistributedTrainerMixin, SFTTrainer):
    SUPPORTS_CP = True
    SUPPORTS_HYBRID_CP = True
    # Canonical last-stage loss; the PP-incompatible knobs are rejected in _split_pipeline_stage.
    PP_REFUSAL = None
    SUPPORTS_LOWP_PRECISION = True
    # Forwards with ``labels``, so ``*ForCausalLM.forward`` folds the router aux loss into outputs.loss.
    ROUTER_AUX_LOSS_ADDED_BY = "model_forward"
    # False lets loss-only eval take Liger's fused-CE ``skip_logits`` path; a subclass whose loss reads
    # ``outputs.logits`` must flip it (lce_forward returns None there, and raises without labels).
    LOSS_READS_LOGITS: ClassVar[bool] = False
    # False for a subclass whose own loss records none of the loss metrics below: their per-log flush would be a
    # world collective carrying nothing.
    RECORDS_LOSS_METRICS: ClassVar[bool] = True

    def __init__(self, *args, **kwargs):
        _replace_chunked_nll_loss_type(get_ctor_config(args, kwargs, _CTOR_POSITIONS))
        kwargs = self._init_distributed_config(kwargs, ctor_args=args, ctor_positions=_CTOR_POSITIONS)
        collator = get_ctor_arg(args, kwargs, "data_collator", _CTOR_POSITIONS)
        reject_cp_packed_collator(collator, is_cp_mode=self.parallelism_config.is_cp_mode)
        super().__init__(*args, **kwargs)
        # On the model the base built, so a model id handed to the constructor is checked too.
        reject_multi_document_collator(collator, getattr(self.model, "config", None))
        if self.parallelism_config.is_cp_mode:
            # The CP loss is its own mean: HF's num_items_in_batch scaling would inflate it cp_size x.
            self.model_accepts_loss_kwargs = False
        self._configure_distributed_modes()

        self._loss_metrics = {"train": WorldMetrics(), "eval": WorldMetrics()}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Under CP, the wrapper's loss on this rank's chunk. Metrics are recorded for the once-per-log flush, where
        TRL gathers and reads them per micro-batch."""
        self._warn_once_on_empty_labels(inputs)
        prediction_loss_only = inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        config = self.parallelism_config
        cp_shard = (self.cp_config.cp_rank, config.cp_size) if config.is_cp_mode else None
        if cp_shard is not None:
            self._pad_to_cp_multiple(inputs)
        # Set aside: HF pops the labels under a label smoother or a compute_loss_func.
        labels = inputs.get("labels")
        if cp_shard is not None:
            # No num_items_in_batch: per-token loss scaling is wrong under CP.
            loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True)
        else:
            self._add_trl_forward_inputs(inputs, prediction_loss_only)
            tree_plan = inputs.get(TREE_ATTENTION_KEY)
            with expert_load_token_weights(None if tree_plan is None else tree_plan["leaf_count"]):
                loss, outputs = Trainer.compute_loss(
                    self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
                )
            loss = self._weigh_router_aux_loss(model, loss, outputs, num_items_in_batch)
        training = self.model.training
        _record_loss_metrics(
            self._loss_metrics["train" if training else "eval"], outputs, inputs, labels, cp_shard, training=training
        )
        return (loss, outputs) if return_outputs else loss

    def _add_trl_forward_inputs(self, inputs, prediction_loss_only: bool | None) -> None:
        """The forward inputs ``SFTTrainer.compute_loss`` adds: router logits where TRL weighs the aux loss and, under
        Liger, the skip-logits verdict, token accuracy and DFT's token scaling."""
        if self.aux_loss_enabled:
            inputs["output_router_logits"] = True
        if self.args.use_liger_kernel:
            inputs["skip_logits"] = (
                self.model.training
                or self.args.prediction_loss_only
                or (
                    self.compute_metrics is None
                    and self.preprocess_logits_for_metrics is None
                    and prediction_loss_only is not False
                )
            )
            inputs["return_token_accuracy"] = True
            inputs["use_token_scaling"] = self.args.loss_type == "dft"

    def _pad_to_cp_multiple(self, inputs) -> None:
        """Pad sequence columns in place to a multiple of ``cp_size``, with the tokenizer's pad id (never vocabulary
        token 0) on ``input_ids`` and the ignore index on ``labels``."""
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
        """``coef x mean(aux)`` per optimizer step, ``coef x aux`` in an eval batch; HF's ``num_items_in_batch`` path
        would weigh the aux mean ``processes x accumulation`` times. The forward adds the term only when handed labels,
        which HF withholds under a label smoother or ``compute_loss_func``; it is then added here alone."""
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
            hf_loss_scale = self.accelerator.num_processes // self.get_tp_size()
        return loss - forward_added * hf_loss_scale + target

    def log(self, logs: dict[str, float], start_time: float | None = None):
        """Fold the deferred loss-metric sums into ``_metrics`` before the base trainer averages them."""
        self._drain_loss_metrics()
        super().log(logs, start_time)

    def _drain_loss_metrics(self):
        """COLLECTIVE outside PP. A window with no loss forward anywhere emits nothing rather than a 0.0 accuracy. PP
        records nothing here, so it skips the flush, as does a subclass with :attr:`RECORDS_LOSS_METRICS` off."""
        if self._pp_runner is not None or not self.RECORDS_LOSS_METRICS:
            return
        mode = "train" if self.model.training else "eval"
        world = self._loss_metrics[mode].flush()
        if not world:
            return
        # Every rank records its whole batch, which the ranks of a TP/ETP/CP group share.
        attended = int(world.pop("attended_tokens")) // self.parallelism_config.non_dp_replication_factor
        self._total_train_tokens += attended
        self._metrics[mode]["num_attended_tokens_seen"] = [self._total_train_tokens]
        for key, value in world.items():
            self._metrics[mode][key].append(value)

    @cached_property
    def _forward_accepts_skip_logits(self) -> bool:
        """Whether forward was patched with Liger ``lce_forward``; read at the first eval step, so both the class-level
        patch and TRL's instance-level re-application are visible."""
        model = self._top_level_model()
        try:
            return "skip_logits" in inspect.signature(model.forward).parameters
        except (TypeError, ValueError):
            return False

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Route loss-only eval through Liger's fused CE rather than materialize full ``[B, S, vocab]`` fp32 logits;
        every other clause names a consumer that still needs real logits."""
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
