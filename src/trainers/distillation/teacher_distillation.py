"""Off-policy teacher→student knowledge distillation (a separate frozen teacher scores fixed dataset
completions; 8 loss types in ``teacher_losses``). EP/TP on the student; CP unsupported (would require
wrapping both models). Same-model variants live in ``self_distillation`` and ``sdpg``.
"""

from collections.abc import Callable

import torch
import torch.nn as nn
from accelerate.logging import get_logger
from datasets import Dataset, IterableDataset
from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
from transformers.trainer_callback import TrainerCallback

from src.configs.distillation_config import DistillationConfig
from src.data.spans import LABEL_IGNORE_INDEX
from src.distributed.loading.frozen_models import place_and_freeze
from src.distributed.loading.model_loading import load_model_from_pretrained
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.distillation.losses import logits_forward_inputs, shifted_token_cross_entropy
from src.trainers.distillation.teacher_losses import (
    apply_hard_labels_mask,
    call_distillation_loss,
    consumes_hard_labels,
    get_distillation_loss_fn,
    hard_labels_coefficient,
)
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.mixins.stored_metrics import StoredMetricsMixin

logger = get_logger(__name__, log_level="info")


class DistributedDistillationTrainer(StoredMetricsMixin, DistributedTrainerMixin, Trainer):
    """Knowledge distillation from teacher to student, with EP/TP on the student.

    Loss is ``distill_alpha * L_distill + (1 - distill_alpha) * L_clm`` (``distill_alpha=1.0`` drops CLM).
    The teacher is not parallelized and runs under ``torch.no_grad()``.
    """

    _tag_names = ["trl", "distillation", "knowledge-distillation"]
    # compute_loss returns the batch's own mean over the distillation + CLM terms.
    _loss_is_own_mean = True

    _supports_pp = False
    _pp_unsupported_reason = (
        "each microbatch needs a second forward through a distinct frozen teacher network whose "
        "FULL-vocabulary logits every distillation loss consumes (KL, JSD, alpha-beta, MSE, cosine, "
        "EMD, SLIM and soft CE all read the whole [tokens, vocab] plane), and no single pipeline "
        "stage holds a whole model to run it. A precomputed per-token target cannot stand in: the "
        "exact cache is the full plane per row, and a top-k cache changes the objective. Supporting "
        "it means a second, frozen, stage-split teacher pipeline whose last-stage logits feed the "
        "student's last-stage loss"
    )

    def __init__(
        self,
        student_model: str | PreTrainedModel | nn.Module,
        teacher_model: PreTrainedModel | nn.Module,
        args: DistillationConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        compute_metrics: Callable | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        parallelism_config: ParallelismConfig = None,
        save_sharded_ep: bool = False,
        dataset_presharded: bool = False,
        **kwargs,
    ):
        if processing_class is None:
            raise ValueError("processing_class (tokenizer) must be provided")

        if isinstance(teacher_model, str):
            raise TypeError(
                "teacher_model must be an already-loaded module. The frozen teacher is loaded by the "
                "caller through load_frozen_auxiliary_model (scripts/training/distillation/"
                "teacher_distill.py::_load_distill_teacher), which is where the run's revision pin, "
                "dtype, sinks policy and VLM device placement are resolved — a path handed here "
                "would be fetched with none of them."
            )

        dist_kwargs = self._init_distributed_config(
            kwargs,
            training_args=args,
            parallelism_config=parallelism_config,
            save_sharded_ep=save_sharded_ep,
            dataset_presharded=dataset_presharded,
        )

        # The method knobs live on ``args`` (DistillationConfig) and are read there; only the
        # resolved loss callable is worth holding.
        self.distillation_loss_fn = get_distillation_loss_fn(args.distill_loss)

        self.teacher_model = teacher_model

        student_model, _ = load_model_from_pretrained(student_model, args)

        super().__init__(
            model=student_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **dist_kwargs,
        )

        self.model.add_model_tags(self._tag_names)

        self._setup_distributed_modes()
        self._setup_teacher_model()

    def _setup_teacher_model(self):
        """Move teacher to the student device, set eval, and freeze gradients."""
        student_vocab = self.model.config.get_text_config().vocab_size
        teacher_vocab = self.teacher_model.config.get_text_config().vocab_size
        if student_vocab != teacher_vocab:
            raise ValueError(
                f"Teacher distillation requires matching vocabularies: student vocab_size={student_vocab} "
                f"vs teacher vocab_size={teacher_vocab}. Use a teacher with the same tokenizer/vocab."
            )
        device = place_and_freeze(self.teacher_model, self.model)
        logger.info(f"Teacher model moved to {device} and set to eval mode")

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Distillation loss: distill term (KL/MSE/...) between student/teacher logits + optional CLM on hard labels."""
        self._validate_inputs(inputs)

        hard_labels = inputs["labels"][..., 1:].contiguous()
        distillation_coef = None

        # Full-vocab logits even under fused-LCE; the CLM loss below reuses them.
        model_inputs = logits_forward_inputs(inputs)

        # Shifted logits stay VIEWS: a [B, S, V] .contiguous() would copy gigabytes per microbatch.
        student_outputs = model(**model_inputs)
        student_logits = student_outputs.logits[..., :-1, :]

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**model_inputs)
            teacher_logits = teacher_outputs.logits[..., :-1, :]

        distillation_loss = call_distillation_loss(
            self.distillation_loss_fn, student_logits, teacher_logits, self.args.distill_temperature, hard_labels
        )

        # distillation_coef is [B, S, 1]; squeeze for per-token [B, S] losses (cosine), keep 3-D for [B, S, V].
        # A loss that takes hard_labels applies its own gold-token coefficient (slim), so the shared
        # one would be counted twice — the same signature the dispatcher reads.
        if self.args.apply_hard_labels and not consumes_hard_labels(self.distillation_loss_fn):
            distillation_coef = hard_labels_coefficient(student_logits, teacher_logits, hard_labels)
            coef = (
                distillation_coef
                if distillation_loss.dim() == distillation_coef.dim()
                else distillation_coef.squeeze(-1)
            )
            distillation_loss = coef * distillation_loss

        distillation_loss = apply_hard_labels_mask(distillation_loss, hard_labels)

        # Clamped count, not a mean reduction: a mean is NaN on an all-masked microbatch, and
        # 0.0 * NaN still reaches every rank through the gradient reduce.
        valid_label_count = (hard_labels != LABEL_IGNORE_INDEX).sum().clamp(min=1)
        # The CLM term is METRIC-ONLY unless it carries weight: at the default distill_alpha=1.0 (or
        # with use_clm_loss off) a grad-carrying [B, S, V] fp32 cross-entropy is built and backward-ed
        # for a summand of exactly zero. Rank-uniform config, so no rank builds a different graph.
        # The distillation leg is deliberately not branched to match: its forward IS the reported
        # distillation_loss metric at every alpha, so a gate would save only the backward graph, and
        # only at the degenerate distill_alpha=0.0 (a distillation run distilling nothing).
        clm_in_loss = self.args.use_clm_loss and self.args.distill_alpha != 1.0
        with torch.enable_grad() if clm_in_loss else torch.no_grad():
            sft_loss = shifted_token_cross_entropy(student_logits, hard_labels).sum() / valid_label_count

        loss = self.args.distill_alpha * distillation_loss
        if clm_in_loss:
            loss = loss + (1 - self.args.distill_alpha) * sft_loss

        metrics = {
            "distillation_loss": distillation_loss.detach(),
            "sft_loss": sft_loss.detach(),
        }
        if distillation_coef is not None:
            # Reduce over the SAME valid-label tokens the loss uses, not the attention mask.
            distillation_coef_masked = apply_hard_labels_mask(distillation_coef, hard_labels)
            metrics["distillation_coef"] = distillation_coef_masked.detach()

        self.store_metrics(metrics, train_eval="train" if self.model.training else "eval")

        if return_outputs:
            return loss, student_outputs
        return loss
