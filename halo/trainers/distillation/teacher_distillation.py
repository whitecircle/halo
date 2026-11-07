"""Off-policy teacher→student knowledge distillation: a separate frozen teacher scores fixed dataset completions. EP/TP
on the student; CP unsupported (would require wrapping both models)."""

import logging
from collections.abc import Callable

import torch
import torch.nn as nn
from datasets import Dataset, IterableDataset
from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
from transformers.trainer_callback import TrainerCallback

from halo.configs.distillation import TeacherDistillationConfig
from halo.distributed.parallelism_config import ParallelismConfig
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.model_loading.parallel.peft_setup import peft_bf16_autocast, prepare_policy_model
from halo.model_loading.parallel.whole_model import resolve_trainer_model
from halo.model_loading.tokenizer_setup import resolve_tokenizer
from halo.models.attention.gpt_oss_sinks import (
    NEUTRALIZED_TEACHER_SINKS_COST,
    SINK_CARRYING_ATTN_HINT,
    SinksPolicy,
    stamped_sinks_policy,
)
from halo.runtime.ranks import is_global_main_process
from halo.trainers.distillation.divergences import (
    hard_labels_coefficient,
    logits_forward_inputs,
    place_frozen_scorer,
    pooled_token_mean,
    reject_topk_covering_vocab,
    require_divergence,
    shifted_token_cross_entropy,
    token_divergence,
    token_divergence_and_cross_entropy,
    uses_hard_labels,
)
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.metrics.stored_metrics import StoredMetricsMixin

logger = logging.getLogger(__name__)


def require_distill_loss(args: TeacherDistillationConfig) -> Callable:
    """The ``distill_loss`` divergence at ``distill_jsd_beta`` and ``distill_topk``, refused under
    ``apply_hard_labels`` when it weighs the gold token itself."""
    loss_fn = require_divergence(args.distill_loss, jsd_beta=args.distill_jsd_beta, topk=args.distill_topk)
    if args.apply_hard_labels and uses_hard_labels(loss_fn):
        raise ValueError(
            f"apply_hard_labels cannot combine with distill_loss={args.distill_loss!r}, which applies its "
            f"own gold-token weight in place of the hard-label gate. Set apply_hard_labels: false."
        )
    return loss_fn


class DistributedTeacherDistillationTrainer(StoredMetricsMixin, DistributedTrainerMixin, Trainer):
    """Loss is ``distill_alpha * L_distill + (1 - distill_alpha) * L_clm``, each term one mean over the micro-batch's
    supervised tokens. The teacher is not parallelized and runs no-grad."""

    _tag_names = ["trl", "distillation", "knowledge-distillation"]
    LOSS_IS_OWN_MEAN = True

    PP_REFUSAL = (
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
        args: TeacherDistillationConfig | None = None,
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
        # Read off the load's stamp, not the tensors: a checkpoint trained without sinks holds them at the
        # neutralized value and is a faithful teacher under a live policy.
        if stamped_sinks_policy(teacher_model) is SinksPolicy.NEUTRALIZED:
            raise ValueError(
                f"The GptOss teacher was loaded with its attention sinks neutralized (reset_sinks=True), and "
                f"{NEUTRALIZED_TEACHER_SINKS_COST}, so the student would fit a degenerate target. Load the teacher with "
                f"reset_sinks=False (load_frozen_auxiliary_model), on a sink-carrying attention implementation "
                f"({SINK_CARRYING_ATTN_HINT})."
            )

        # The student rides through the seam: the reentrant-checkpointing and Liger gates read its config.
        student_model = resolve_trainer_model(student_model, args, parallelism_config=parallelism_config)
        dist_kwargs = self._init_distributed_config(
            kwargs,
            training_args=args,
            model=student_model,
            parallelism_config=parallelism_config,
            save_sharded_ep=save_sharded_ep,
            dataset_presharded=dataset_presharded,
        )
        student_model = dist_kwargs.pop("model")

        student_model, self._peft_has_been_casted_to_bf16 = prepare_policy_model(
            student_model, peft_config, args, disable_dropout=False
        )

        self.distillation_loss_fn = require_distill_loss(args)

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

        self._configure_distributed_modes()
        self._vocab_width, device = place_frozen_scorer(
            self.teacher_model, self.model, resolve_tokenizer(self.processing_class), teacher_tokenizer
        )
        reject_topk_covering_vocab(self.args.distill_topk, self._vocab_width, self.model)
        if is_global_main_process():
            logger.info(f"Teacher model moved to {device} and set to eval mode")

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        self._warn_once_on_empty_labels(inputs)

        hard_labels = inputs["labels"][..., 1:]
        supervised = hard_labels != LABEL_IGNORE_INDEX
        # Rows padding an eval split's final round repeat its first rows: they leave every mean.
        real_rows = self.eval_split_rows(hard_labels.size(0))
        supervised[real_rows:] = False

        model_inputs = logits_forward_inputs(inputs)

        with peft_bf16_autocast(self._peft_has_been_casted_to_bf16, self.accelerator.device):
            student_outputs = model(**model_inputs)
        # Shifted logits stay VIEWS: a [B, S, V] .contiguous() would copy gigabytes per microbatch.
        student_logits = student_outputs.logits[..., :-1, : self._vocab_width]

        with torch.no_grad():
            teacher_logits = self.teacher_model(**model_inputs).logits[..., :-1, : self._vocab_width]

        # The CLM term is METRIC-ONLY at distill_alpha=1.0: a grad-carrying cross-entropy there would be built and
        # backward-ed for a summand of zero. In the loss it shares the divergence's chunked pass, so backward builds
        # one gradient plane of the student logits.
        clm_in_loss = self.args.distill_alpha != 1.0
        divergence_args = (self.distillation_loss_fn, student_logits, teacher_logits, self.args.distill_temperature)
        if clm_in_loss:
            distillation_loss, token_cross_entropy = token_divergence_and_cross_entropy(*divergence_args, hard_labels)
        else:
            distillation_loss = token_divergence(*divergence_args, hard_labels)
            with torch.no_grad():
                token_cross_entropy = shifted_token_cross_entropy(student_logits, hard_labels)
        metrics = {}

        if self.args.apply_hard_labels:
            gate = hard_labels_coefficient(student_logits, teacher_logits, hard_labels)
            distillation_loss = gate * distillation_loss
            metrics["distillation_coef"] = pooled_token_mean(gate, supervised)

        distillation_loss = pooled_token_mean(distillation_loss, supervised)
        sft_loss = pooled_token_mean(token_cross_entropy, supervised)

        loss = self.args.distill_alpha * distillation_loss
        if clm_in_loss:
            loss = loss + (1 - self.args.distill_alpha) * sft_loss

        metrics.update(distillation_loss=distillation_loss.detach(), sft_loss=sft_loss.detach())
        self.store_batch_metrics(metrics, "train" if self.model.training else "eval", real_rows)

        if return_outputs:
            return loss, student_outputs
        return loss
