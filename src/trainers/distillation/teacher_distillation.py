"""Off-policy teacher→student knowledge distillation (a separate frozen teacher scores fixed dataset
completions through a ``losses.DIVERGENCES`` entry). EP/TP on the student; CP unsupported (would
require wrapping both models). Same-model variants live in ``self_distillation`` and ``sdpg``.
"""

from collections.abc import Callable
from contextlib import nullcontext

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
from src.distributed.loading.peft_setup import peft_bf16_autocast, prepare_peft_model
from src.distributed.parallelism_config import ParallelismConfig
from src.models.structure import resolve_tokenizer
from src.trainers.distillation.losses import (
    call_divergence,
    consumes_hard_labels,
    get_divergence,
    global_token_mean,
    hard_labels_coefficient,
    logits_forward_inputs,
    shared_vocab_width,
    shifted_token_cross_entropy,
)
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.mixins.stored_metrics import StoredMetricsMixin

logger = get_logger(__name__, log_level="info")


class DistributedDistillationTrainer(StoredMetricsMixin, DistributedTrainerMixin, Trainer):
    """Knowledge distillation from teacher to student, with EP/TP on the student.

    Loss is ``distill_alpha * L_distill + (1 - distill_alpha) * L_clm`` (``distill_alpha=1.0`` drops
    CLM), each term one mean over the micro-batch's supervised tokens. The teacher is not parallelized
    and runs under ``torch.no_grad()``.
    """

    _tag_names = ["trl", "distillation", "knowledge-distillation"]
    # compute_loss returns the batch's own token mean over the distillation + CLM terms.
    _loss_is_own_mean = True

    _supports_pp = False
    _pp_unsupported_reason = (
        "each microbatch needs a second forward through a distinct frozen teacher network whose "
        "FULL-vocabulary logits every distillation loss consumes (KL, JSD, MSE, cosine, SLIM and "
        "soft CE all read the whole [tokens, vocab] plane), and no single pipeline "
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
        peft_config=None,
        parallelism_config: ParallelismConfig = None,
        save_sharded_ep: bool = False,
        dataset_presharded: bool = False,
        *,
        teacher_tokenizer: PreTrainedTokenizerBase,
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

        # The student rides through the seam: the reentrant-checkpointing and Liger gates read its config.
        student_model, _ = load_model_from_pretrained(
            student_model, args, keep_fp32=parallelism_config is not None and parallelism_config.fp32_non_ep_params
        )
        dist_kwargs = self._init_distributed_config(
            kwargs,
            training_args=args,
            model=student_model,
            parallelism_config=parallelism_config,
            save_sharded_ep=save_sharded_ep,
            dataset_presharded=dataset_presharded,
        )
        student_model = dist_kwargs.pop("model")

        self._peft_has_been_casted_to_bf16 = False
        if peft_config is not None:
            student_model, self._peft_has_been_casted_to_bf16 = prepare_peft_model(student_model, peft_config, args)

        # The method knobs live on ``args`` (DistillationConfig) and are read there; only the
        # resolved loss callable is worth holding.
        self.distillation_loss_fn = get_divergence(
            args.distill_loss, jsd_beta=args.distill_jsd_beta, topk=args.distill_topk
        )
        if args.apply_hard_labels and consumes_hard_labels(self.distillation_loss_fn):
            raise ValueError(
                f"apply_hard_labels cannot combine with distill_loss={args.distill_loss!r}, which applies its "
                f"own gold-token weight in place of the hard-label gate. Set apply_hard_labels: false."
            )

        self.teacher_model = teacher_model

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
        self._setup_teacher_model(teacher_tokenizer)

    def _setup_teacher_model(self, teacher_tokenizer: PreTrainedTokenizerBase):
        """Check the teacher scores the student's token ids, then move it to the student device frozen."""
        self._vocab_width = shared_vocab_width(
            self.model.config, self.teacher_model.config, resolve_tokenizer(self.processing_class), teacher_tokenizer
        )
        # DistillationConfig can't see the vocabulary, so the top-k bound is checked here.
        vocab_size = self._vocab_width or self.model.config.get_text_config().vocab_size
        if self.args.distill_topk is not None and self.args.distill_topk >= vocab_size:
            raise ValueError(
                f"distill_topk={self.args.distill_topk} covers the whole vocabulary (vocab_size={vocab_size}). "
                f"Set distill_topk: null for the full-vocab loss, or a k below the vocab size."
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

        hard_labels = inputs["labels"][..., 1:]
        supervised = hard_labels != LABEL_IGNORE_INDEX
        # Rows padding an eval split's final round repeat its first rows: they leave every mean.
        real_rows = self.eval_split_rows(hard_labels.size(0))
        supervised[real_rows:] = False

        # Full-vocab logits even under fused-LCE; the CLM loss below reuses them.
        model_inputs = logits_forward_inputs(inputs)

        with peft_bf16_autocast(self._peft_has_been_casted_to_bf16, self.accelerator.device):
            student_outputs = model(**model_inputs)
        # Shifted logits stay VIEWS: a [B, S, V] .contiguous() would copy gigabytes per microbatch.
        student_logits = student_outputs.logits[..., :-1, : self._vocab_width]

        with torch.no_grad():
            teacher_logits = self.teacher_model(**model_inputs).logits[..., :-1, : self._vocab_width]

        distillation_loss = call_divergence(
            self.distillation_loss_fn, student_logits, teacher_logits, self.args.distill_temperature, hard_labels
        )
        metrics = {}

        if self.args.apply_hard_labels:
            gate = hard_labels_coefficient(student_logits, teacher_logits, hard_labels)
            distillation_loss = (gate.unsqueeze(-1) if distillation_loss.dim() == gate.dim() + 1 else gate) * (
                distillation_loss
            )
            metrics["distillation_coef"] = global_token_mean(gate, supervised)

        distillation_loss = global_token_mean(distillation_loss, supervised)

        # The CLM term is METRIC-ONLY at distill_alpha=1.0: a grad-carrying [B, S, V] fp32
        # cross-entropy there would be built and backward-ed for a summand of exactly zero.
        # Rank-uniform config, so no rank builds a different graph.
        clm_in_loss = self.args.distill_alpha != 1.0
        with nullcontext() if clm_in_loss else torch.no_grad():
            sft_loss = global_token_mean(shifted_token_cross_entropy(student_logits, hard_labels), supervised)

        loss = self.args.distill_alpha * distillation_loss
        if clm_in_loss:
            loss = loss + (1 - self.args.distill_alpha) * sft_loss

        metrics.update(distillation_loss=distillation_loss.detach(), sft_loss=sft_loss.detach())
        self.store_batch_metrics(metrics, "train" if self.model.training else "eval", real_rows)

        if return_outputs:
            return loss, student_outputs
        return loss
