# Copyright 2024 White Circle
#
# Licensed under the Halo License (the "License"): the Apache License, Version 2.0
# (a copy is provided in APACHE-2.0.txt), as modified by, and subject to, the
# Supplemental Terms in the LICENSE file at the root of this repository.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Distributed binary/multi-label classification trainer: EP, TP and PP; CP unsupported (pooling).

Under PP the whole sequence lives on the last stage, so pooling and every loss variant are last-stage functions,
and evaluation keeps ``compute_metrics``: the pooled ``[B, num_labels]`` logits broadcast down the chain.
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalPrediction

from src.configs.classification import ClassificationConfig
from src.data.collators.classification import ClassificationDataCollatorWithPadding
from src.data.pipeline.coordinated_map import coordinated_filter, coordinated_map, resolve_map_num_proc
from src.data.pipeline.rejected_rows import report_rejected_rows
from src.distributed.parallelism_config import ParallelismConfig
from src.model_loading.parallel.peft_setup import peft_bf16_autocast, prepare_policy_model
from src.model_loading.tokenizer_setup import resolve_tokenizer
from src.trainers.distributed_trainer import DistributedTrainerMixin
from src.trainers.parallelism.pipeline_loss_adapter import (
    PipelineLossAdapter,
    decode_pooling_plane,
    encode_pooling_plane,
    pooled_logits,
)
from src.trainers.parallelism.pipeline_rules import require_pooling_pad_id
from src.trainers.reward.classification_loss import ClassificationLoss, build_classification_loss

logger = logging.getLogger(__name__)

# The one metric this trainer may omit from an eval slice (undefined, not zero): a one-class slice
# makes the AUC meaningless. Ranking checkpoints on it is checked at construction and again wherever
# the key is dropped.
_OMITTABLE_METRIC = "auc_roc"


def _ranks_checkpoints_on_omittable_metric(args: Any) -> bool:
    return (getattr(args, "metric_for_best_model", None) or "").removeprefix("eval_") == _OMITTABLE_METRIC


def reject_unsupported_classification_args(args: Any, *, default_metrics: bool) -> None:
    """Refuse HF's ``label_smoothing_factor``, whose label_smoother never runs under this trainer's own loss; under
    the default metrics, a ``metric_for_best_model`` no evaluation reports, which transformers' best-metric lookup
    would raise on as a ``KeyError`` at the first eval; and a ``max_length`` unresolved by construction, which
    bounds the raw dataset's over-length filter."""
    if args.label_smoothing_factor:
        raise ValueError(
            f"label_smoothing_factor={args.label_smoothing_factor} has no effect on "
            f"ClassificationTrainer: it builds its own loss and never calls HF's label_smoother. "
            f"Use loss_type: label_smoothing_ce with the label_smoothing knob instead."
        )
    # A caller's own compute_metrics decides its own keys.
    if default_metrics and _ranks_checkpoints_on_omittable_metric(args) and not args.compute_auc_roc:
        raise ValueError(
            f"metric_for_best_model={args.metric_for_best_model!r} but compute_auc_roc is off, so no "
            f"evaluation ever reports it and checkpoint ranking would die on a KeyError mid-training. "
            f"Set compute_auc_roc: true, or rank on a metric computed on every slice "
            f"(accuracy / f1 / mcc)."
        )
    if args.max_length is None or args.max_length <= 0:
        raise ValueError(
            "ClassificationConfig.max_length must be a positive int by the time the trainer is "
            "constructed — it bounds the over-length filter applied to a raw text/label dataset. "
            "Set it in the config, or resolve it to the model context window first "
            "(resolve_length_to_context, as scripts/training/classification.py does)."
        )


def _require_class_ids(labels: torch.Tensor) -> torch.Tensor:
    """A single-label batch's ``[B]`` class ids, or a raise: a ``[B, C]`` float target would train
    soft-label cross-entropy, silently. Shape metadata only, so no host sync."""
    if labels.dim() != 1:
        raise ValueError(
            f"Single-label classification expects [B] integer class ids, got labels of shape "
            f"{tuple(labels.shape)}. For multi-hot targets pass is_multi_label=True."
        )
    return labels


def _is_within_length(example: dict[str, Any], max_length: int) -> bool:
    return len(example["input_ids"]) <= max_length


def _filter_over_length(dataset: Dataset, max_length: int, num_proc: int | None, split_name: str) -> Dataset:
    """Drop rows whose tokenized length exceeds ``max_length``, reporting the drop rate.

    Never raises — long-tailed datasets are legitimate — but a high drop rate is escalated to a
    warning by the shared reporter, which holds that threshold for every drop-and-continue filter.
    """
    original = len(dataset)
    dataset = coordinated_filter(
        dataset,
        _is_within_length,
        desc=f"Filtering {split_name} over max_length={max_length}",
        num_proc=num_proc,
        fn_kwargs={"max_length": max_length},
    )
    report_rejected_rows(original, len(dataset), f"the classification {split_name} split's max_length={max_length}")
    return dataset


def _tokenize(batch: dict[str, list[Any]], tokenizer: "PreTrainedTokenizerBase") -> dict[str, list[Any]]:
    new_examples = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    for text, label in zip(batch["text"], batch["label"], strict=False):
        tokenized = tokenizer(text)
        new_examples["input_ids"].append(tokenized["input_ids"])
        new_examples["attention_mask"].append(tokenized["attention_mask"])
        new_examples["labels"].append(label)

    return new_examples


def _tokenized_split(dataset: Dataset, tokenizer, max_length: int, num_proc: int | None, split_name: str) -> Dataset:
    """A raw text/label split tokenized by one coordinated map, its over-length rows dropped."""
    dataset = coordinated_map(
        dataset,
        _tokenize,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=num_proc,
        desc=f"Tokenizing {split_name} dataset",
    )
    return _filter_over_length(dataset, max_length, num_proc, split_name)


class ClassificationTrainer(DistributedTrainerMixin, Trainer):
    """Binary/multi-label classification with EP/TP/PP over 'text'/'label' rows (tokenized when ``input_ids`` is
    absent); CP excluded, since the head needs the full pooled representation."""

    _tag_names = ["trl", "classification"]

    PP_REFUSAL = None
    # compute_loss returns the batch's own mean (per-replica; DP averages gradients).
    LOSS_IS_OWN_MEAN = True

    _pp_pool_pad_id: int | None = None

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | None = None,
        args: ClassificationConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: dict | None = None,
        is_binary: bool = True,
        is_multi_label: bool = False,
        label_names_list: list[str] | None = None,
        parallelism_config: ParallelismConfig = None,
        save_sharded_ep: bool = False,
        dataset_presharded: bool = False,
        **kwargs,
    ):
        # model/peft_config ride through for the PP split; compute_metrics stays out (pooled here).
        dist_kwargs = self._init_distributed_config(
            kwargs,
            training_args=args,
            parallelism_config=parallelism_config,
            save_sharded_ep=save_sharded_ep,
            dataset_presharded=dataset_presharded,
            model=model,
            peft_config=peft_config,
        )
        model = dist_kwargs.pop("model")
        peft_config = dist_kwargs.pop("peft_config")
        model, self._peft_has_been_casted_to_bf16 = prepare_policy_model(
            model, peft_config, args, disable_dropout=args.disable_dropout, merge_existing=False
        )

        reject_unsupported_classification_args(args, default_metrics=compute_metrics is None)
        if compute_metrics is None:
            compute_metrics = self._default_compute_metrics
        max_length = args.max_length

        if data_collator is None:
            if processing_class is None:
                raise ValueError("A processing_class must be specified when using the default DataCollatorWithPadding")
            data_collator = ClassificationDataCollatorWithPadding(processing_class, max_length=max_length)

        if "input_ids" not in train_dataset.column_names:
            num_proc = resolve_map_num_proc(args.dataset_num_proc)
            train_dataset = _tokenized_split(train_dataset, processing_class, max_length, num_proc, "train")
            if eval_dataset is not None:
                eval_dataset = _tokenized_split(eval_dataset, processing_class, max_length, num_proc, "eval")

        self.is_binary = is_binary
        self.is_multi_label = is_multi_label
        self.label_names_list = label_names_list
        self._classification_loss = build_classification_loss(args, model, train_dataset, multi_label=is_multi_label)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **dist_kwargs,
        )

        self.model.add_model_tags(self._tag_names)

        self._configure_distributed_modes()

    def compute_loss(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """The configured objective on the head's pooled logits, in fp32.

        No ``_warn_once_on_empty_labels``: that check reads ``labels`` as token ids, and class ids collide with
        the tokenizer's pad/eos ids (Gemma's pad=0 / eos=1 is every binary label).
        """
        with peft_bf16_autocast(self._peft_has_been_casted_to_bf16, self.accelerator.device):
            logits = model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], return_dict=True
            ).logits
            labels = inputs["labels"] if self.is_multi_label else _require_class_ids(inputs["labels"])
            loss_fn = self._loss_on(logits.device)
            real_rows = self.eval_split_rows(logits.size(0))
            if real_rows < logits.size(0):
                # Rows padding an eval split's final round repeat its first rows: they leave the mean as
                # the pipeline's inert rows do, and a rank of padding alone scores 0.
                real = torch.arange(logits.size(0), device=logits.device) < real_rows
                loss = loss_fn.masked_sum(logits, labels, real) / loss_fn.normalizer(labels, real)
            else:
                loss = loss_fn.mean(logits, labels)
        return (loss, {"logits": logits}) if return_outputs else loss

    def prediction_step(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """``ignore_keys`` is inert: the loss returns ``logits`` directly."""
        if self._pp_runner is not None:
            return self._pp_prediction_step(inputs, prediction_loss_only, ignore_keys)
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        return loss.detach(), nested_detach(outputs["logits"]), self._prepare_inputs(inputs["labels"])

    def _default_compute_metrics(self, eval_pred: EvalPrediction) -> dict:
        logits, labels = eval_pred
        args = self.args
        metrics = {}

        if self.is_multi_label:
            probs = 1.0 / (1.0 + np.exp(-logits))
            preds = (probs >= args.multi_label_threshold).astype(int)
            average = "macro"
            metrics["exact_match_accuracy"] = accuracy_score(labels, preds)
        else:
            preds = logits.argmax(axis=-1)
            average = "binary" if self.is_binary else "weighted"
            metrics["accuracy"] = accuracy_score(labels, preds)

        metrics["precision"] = precision_score(labels, preds, average=average, zero_division=0)
        metrics["recall"] = recall_score(labels, preds, average=average, zero_division=0)
        metrics["f1"] = f1_score(labels, preds, average=average, zero_division=0)

        if args.compute_mcc and not self.is_multi_label:
            metrics["mcc"] = matthews_corrcoef(labels, preds)

        if args.compute_auc_roc:
            try:
                if self.is_multi_label:
                    auc_roc = roc_auc_score(labels, logits, average="macro")
                else:
                    # AUC is rank-based: sigmoid(logit1) ignores logit0 and mis-ranks when it varies.
                    exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
                    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
                    if self.is_binary:
                        auc_roc = roc_auc_score(labels, probs[:, 1])
                    else:
                        auc_roc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
            except ValueError as error:
                self._omit_auc_roc(str(error))
            else:
                # An eval slice carrying one class only warns and returns NaN rather than raising, and
                # a NaN reaching metric_for_best_model / early stopping compares False against every
                # checkpoint. Undefined is reported by omitting the key, as the raising path does.
                if np.isnan(auc_roc):
                    self._omit_auc_roc("undefined (a class has no samples)")
                else:
                    metrics["auc_roc"] = auc_roc

        if args.compute_per_class_metrics and not self.is_multi_label:
            per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
            per_class_prec = precision_score(labels, preds, average=None, zero_division=0)
            per_class_rec = recall_score(labels, preds, average=None, zero_division=0)
            for i in range(len(per_class_f1)):
                name = self.label_names_list[i] if self.label_names_list and i < len(self.label_names_list) else str(i)
                metrics[f"f1_class_{name}"] = per_class_f1[i]
                metrics[f"precision_class_{name}"] = per_class_prec[i]
                metrics[f"recall_class_{name}"] = per_class_rec[i]

        return metrics

    def _omit_auc_roc(self, reason: str) -> None:
        """Leave ``auc_roc`` out of this evaluation's metrics, warning why.

        Both undefined paths (sklearn raising, and sklearn warning + returning NaN) land here so the
        omission is one behavior. It raises instead when the run ranks checkpoints on the key, where
        transformers' best-metric lookup would otherwise fail with a bare ``KeyError``.
        """
        if _ranks_checkpoints_on_omittable_metric(self.args):
            raise ValueError(
                f"auc_roc is undefined for this evaluation ({reason}) and metric_for_best_model ranks "
                f"checkpoints by it, so this checkpoint cannot be ranked. Give the eval split every "
                f"class (stratify it, or enlarge eval_accumulation/batch), or rank on a metric defined "
                f"on every slice (accuracy / f1 / mcc)."
            )
        logger.warning(f"auc_roc skipped for this evaluation: {reason}")

    def _loss_on(self, device) -> ClassificationLoss:
        """The configured loss, its class weights moved onto ``device`` once."""
        self._classification_loss = self._classification_loss.to(device)
        return self._classification_loss

    def _pp_loss_adapter(self) -> PipelineLossAdapter:
        """Pooled sequence classification as a last-stage pipeline loss.

        The batch transform rewrites the collated ``labels`` into the runner's ``[B, S]`` plane, ``-100``
        except at the pooled position, which carries the class id. Multi-hot targets cannot ride a long plane;
        they ship as the ``class_targets`` extra beside an inert ``0`` marker. Every loss variant decomposes
        into microbatch sums over a batch-level denominator (:meth:`ClassificationLoss.masked_sum`).
        """
        tokenizer = resolve_tokenizer(self.processing_class)
        self._pp_pool_pad_id = require_pooling_pad_id(self.model.config, getattr(tokenizer, "pad_token_id", None))
        return PipelineLossAdapter(
            token_loss_fn=self._pp_classification_token_loss,
            batch_transform=self._pp_classification_batch_transform,
            normalizer=self._pp_classification_normalizer,
            # Same denominator at eval: the causal-LM token-count fallback would score another quantity.
            eval_normalizer=self._pp_classification_normalizer,
            extra_target_keys=("class_targets",) if self.is_multi_label else (),
            # Pooled on the last stage, so the chain hop carries [B, num_labels], not [B, S, num_labels].
            predictions_fn=lambda logits, inputs: pooled_logits(
                logits, decode_pooling_plane(inputs["labels"])
            ).float(),
            # The real class targets: the multi-hot extra, or the class ids the labels plane carries.
            eval_labels_fn=(lambda inputs: inputs["class_targets"])
            if self.is_multi_label
            else (lambda inputs: decode_pooling_plane(inputs["labels"]).values),
        )

    def _pp_classification_batch_transform(self, batch: dict) -> dict:
        """Collated class labels → the synthesized labels plane (+ multi-hot extras)."""
        input_ids = batch["input_ids"]
        class_labels = batch["labels"]
        out = dict(batch)
        if self.is_multi_label:
            # The collator right-pads 'labels' to max_length; the first num_labels columns are intact.
            out["class_targets"] = class_labels[:, : self.model.config.num_labels].float()
            markers = 0
        else:
            markers = _require_class_ids(class_labels).to(input_ids.dtype)
        out["labels"] = encode_pooling_plane(input_ids, self._pp_pool_pad_id, markers)
        return out

    def _pp_classification_token_loss(self, logits: torch.Tensor, target: torch.Tensor | dict) -> torch.Tensor:
        """Summed pooled loss over the microbatch (the runner divides by the batch normalizer).

        Free of host syncs: inert rows are carried through and masked out of the sum, so an all-inert
        microbatch is a real zero still wired to the graph and the rank stays in the schedule.
        """
        plane = decode_pooling_plane(target["labels"] if isinstance(target, dict) else target)
        pooled = pooled_logits(logits, plane)
        # Inert rows decode to LABEL_IGNORE_INDEX; clamp keeps the CE gather in range before masking.
        targets = target["class_targets"] if self.is_multi_label else plane.values.clamp_min(0)
        return self._loss_on(pooled.device).masked_sum(pooled, targets, plane.valid)

    def _pp_classification_normalizer(self, inputs: dict) -> torch.Tensor:
        """Chain-local batch denominator: the loss is a per-replica mean whose gradients DP averages, so no
        cross-rank reduction is involved."""
        plane = decode_pooling_plane(inputs["labels"])
        targets = inputs["class_targets"] if self.is_multi_label else plane.values
        return self._loss_on(targets.device).normalizer(targets, plane.valid)
