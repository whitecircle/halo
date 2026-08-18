# Copyright 2024 White Circle
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Distributed Classification Trainer (binary/multi-label) with EP/TP/PP support.

CP is incompatible: the classification head needs the full pooled representation
from the complete sequence, which CP's per-rank sequence split cannot provide
without a full gather. PP is compatible: the whole sequence lives on the last
stage, so pooling and every per-example loss variant are valid last-stage
functions; evaluation even keeps compute_metrics — the pooled ``[B, num_labels]``
logits are small enough to broadcast down the pipeline chain.
"""

import logging
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalPrediction
from trl.trainer import disable_dropout_in_model

from src.configs.classification_config import ClassificationConfig
from src.data.collators.classification import ClassificationDataCollatorWithPadding
from src.data.pipeline.processing import (
    coordinated_filter,
    coordinated_map,
    report_rejected_rows,
)
from src.distributed.loading.peft_setup import peft_bf16_autocast, prepare_peft_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.pipeline_parallel.losses import PPLossAdapter
from src.distributed.runtime import current_device
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.reward.pooling import (
    decode_pooling_plane,
    encode_pooling_plane,
    pooled_outputs,
    pooling_pad_id_for,
)

logger = logging.getLogger(__name__)

# The one metric this trainer may legitimately OMIT from an eval slice (undefined, not zero): a
# one-class slice makes the AUC meaningless. Naming it in ``metric_for_best_model`` is therefore a
# checkpoint-ranking hazard, checked at construction and again wherever the key is dropped.
_OMITTABLE_METRIC = "auc_roc"


def _best_model_metric(args: Any) -> str | None:
    """``metric_for_best_model`` without transformers' ``eval_`` prefix (``None`` when unset)."""
    name = getattr(args, "metric_for_best_model", None)
    return name.removeprefix("eval_") if name else None


def _validate_best_model_metric(args: Any) -> None:
    """Refuse a ``metric_for_best_model`` no evaluation will ever report.

    Guaranteed ``KeyError`` on the first eval otherwise: transformers looks the metric up in the dict
    the trainer returns, and with ``compute_auc_roc`` off the key is never written. Config time, so
    the run does not die hours in at its first checkpoint.
    """
    if _best_model_metric(args) == _OMITTABLE_METRIC and not args.compute_auc_roc:
        raise ValueError(
            f"metric_for_best_model={args.metric_for_best_model!r} but compute_auc_roc is off, so no "
            f"evaluation ever reports it and checkpoint ranking would die on a KeyError mid-training. "
            f"Set compute_auc_roc: true, or rank on a metric computed on every slice "
            f"(accuracy / f1 / mcc)."
        )


def _loss_inputs_fp32(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(logits, targets)`` in the fp32 every classification objective evaluates in.

    The head's logits are the toolkit's bf16 storage dtype, but a (weighted) cross-entropy is
    ``w_y * (logit_y - logsumexp(logits))`` — a cancelling difference whose bf16 8-bit mantissa
    resolves to ~1e-2 absolute, enough to reorder two near-tied classes and to round the class
    weights themselves. Pooled logits are ``[B, num_labels]``, so the upcast is free (unlike a
    causal-LM ``[B, S, vocab]``, which is why fused CE kernels exist there instead).

    Float targets are upcast too: ``binary_cross_entropy_with_logits`` takes its OUTPUT dtype from
    the target, so a bf16 multi-hot target would keep the loss in bf16 whatever the logits are.
    Integer class ids pass through — ``F.cross_entropy`` requires them as ``long``.
    """
    return logits.float(), targets.float() if targets.is_floating_point() else targets


def _is_within_length(example: dict[str, Any], max_length: int) -> bool:
    """Whether a tokenized row fits the length budget."""
    return len(example["input_ids"]) <= max_length


def _filter_over_length(dataset: Dataset, max_length: int, num_proc: int | None, split_name: str) -> Dataset:
    """Drop rows whose tokenized length exceeds ``max_length``, reporting the drop rate.

    Never a raise — legitimately long-tailed datasets exist — but a high rate is escalated to a
    WARNING by the shared reporter, which owns that threshold for every drop-and-continue filter.
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
    """Tokenize a batch from a classification dataset."""
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


class ClassificationTrainer(DistributedTrainerMixin, Trainer):
    """Binary/multi-label classification with EP/TP (CP excluded; the head needs
    the full pooled representation, incompatible with CP's sequence split).

    Expects train_dataset with 'text' and 'label' columns (auto-tokenized if
    'input_ids' is absent).
    """

    _tag_names = ["trl", "classification"]

    _supports_pp = True
    # compute_loss returns the batch's own mean (per-replica; DP averages gradients).
    _loss_is_own_mean = True

    _pp_pool_pad_id: int | None = None
    # Set from prepare_peft_model's verdict; False whenever no PEFT config was passed.
    _peft_has_been_casted_to_bf16: bool = False

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | None = None,
        args: ClassificationConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
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

        if peft_config is not None:
            model, casted = prepare_peft_model(model, peft_config, args, merge_existing=False)
            self._peft_has_been_casted_to_bf16 = casted

        if args.disable_dropout:
            disable_dropout_in_model(model)

        # This trainer overrides compute_loss wholesale, so HF's label_smoother — the only consumer
        # of label_smoothing_factor — is never consulted. Refuse it rather than smooth nothing.
        if args.label_smoothing_factor:
            raise ValueError(
                f"label_smoothing_factor={args.label_smoothing_factor} has no effect on "
                f"ClassificationTrainer: it builds its own loss and never calls HF's label_smoother. "
                f"Use loss_type: label_smoothing_ce with the label_smoothing knob instead."
            )

        if compute_metrics is None:
            compute_metrics = self._default_compute_metrics
            # Only for the default metrics: a caller's own compute_metrics owns which keys it emits.
            _validate_best_model_metric(args)

        max_length = args.max_length
        if max_length is None or max_length <= 0:
            raise ValueError(
                "ClassificationConfig.max_length must be a positive int by the time the trainer is "
                "constructed — it bounds the over-length filter applied to a raw text/label dataset. "
                "Set it in the config, or resolve it to the model context window first "
                "(resolve_length_to_context, as scripts/training/classification.py does)."
            )

        if data_collator is None:
            if processing_class is None:
                raise ValueError("A processing_class must be specified when using the default DataCollatorWithPadding")
            data_collator = ClassificationDataCollatorWithPadding(processing_class, max_length=max_length)

        if "input_ids" not in train_dataset.column_names:
            # Coordinated map: one rank tokenizes to a deterministic cache file, the rest load it.
            num_proc = self._dataset_map_num_proc(args.dataset_num_proc)
            fn_kwargs = {"tokenizer": processing_class}
            train_dataset = coordinated_map(
                train_dataset,
                _tokenize,
                batched=True,
                fn_kwargs=fn_kwargs,
                num_proc=num_proc,
                desc="Tokenizing train dataset",
            )
            train_dataset = _filter_over_length(train_dataset, max_length, num_proc, "train")
            if eval_dataset is not None:
                eval_dataset = coordinated_map(
                    eval_dataset,
                    _tokenize,
                    batched=True,
                    fn_kwargs=fn_kwargs,
                    num_proc=num_proc,
                    desc="Tokenizing eval dataset",
                )
                eval_dataset = _filter_over_length(eval_dataset, max_length, num_proc, "eval")

        self.is_binary = is_binary
        self.is_multi_label = is_multi_label
        self.label_names_list = label_names_list

        self._loss_fn = self._build_loss_fn(args, train_dataset, model)

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

        self._setup_distributed_modes()

    def _default_compute_metrics(self, eval_pred: EvalPrediction) -> dict:
        """Compute classification metrics with proper multi-label handling."""
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
                # checkpoint. Undefined is reported by OMITTING the key, as the raising path does.
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
        omission is one behavior. It becomes a RAISE when the run ranks checkpoints on the key:
        transformers' best-metric lookup would otherwise die on a bare ``KeyError`` deep in the save
        path, naming only the missing key and not the eval slice that made it undefined.
        """
        if _best_model_metric(self.args) == _OMITTABLE_METRIC:
            raise ValueError(
                f"auc_roc is undefined for this evaluation ({reason}) and metric_for_best_model ranks "
                f"checkpoints by it, so this checkpoint cannot be ranked. Give the eval split every "
                f"class (stratify it, or enlarge eval_accumulation/batch), or rank on a metric defined "
                f"on every slice (accuracy / f1 / mcc)."
            )
        logger.warning(f"auc_roc skipped for this evaluation: {reason}")

    @staticmethod
    def _focal_loss(logits, labels, gamma=2.0, alpha=None, weight=None, reduction="mean"):
        """Focal loss for single-label and multi-label classification.

        ``reduction="none"`` is the pipeline-parallel form: per-microbatch sums stay additive, and
        the runtime divides by the full batch's element count to recover the mean, so the seam masks
        the inert rows out of the per-element values before summing them itself.

        ``alpha`` is the RetinaNet balancing factor ``alpha_t = alpha*y + (1-alpha)*(1-y)``, so it is
        meaningful only on the multi-label (sigmoid) branch where an element can be positive or
        negative. It is ignored on the single-label branch, where the caller refuses it instead.

        The modulating factor ``(1 - p_t) ** gamma`` is derived from the UNWEIGHTED cross-entropy.
        ``F.cross_entropy(weight=w)`` returns ``w_y * -log p_y``, so ``exp(-ce)`` on the weighted
        value is ``p_y ** w_y`` rather than ``p_y`` — a class weight would then distort the
        modulator instead of scaling the loss, over-focusing the down-weighted classes and
        under-focusing the up-weighted ones (the two levers fighting each other).
        """
        if labels.dim() == 1:
            ce_loss = F.cross_entropy(logits, labels, reduction="none")
            weighted = ce_loss if weight is None else F.cross_entropy(logits, labels, weight=weight, reduction="none")
            # Every softmax element has exactly one target, so a scalar alpha would be a uniform
            # rescale, not balancing; the caller rejects it there and routes to ``weight``.
            alpha_t = None
        else:
            ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            weighted = (
                ce_loss
                if weight is None
                else F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight, reduction="none")
            )
            alpha_t = None if alpha is None else alpha * labels + (1.0 - alpha) * (1.0 - labels)
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** gamma) * weighted
        if alpha_t is not None:
            focal = alpha_t * focal
        if reduction == "none":
            return focal
        if reduction != "mean":
            raise ValueError(f"_focal_loss takes reduction 'none' (the pipeline seam) or 'mean', got {reduction!r}")
        return focal.mean()

    @staticmethod
    def _balanced_class_weights(train_dataset: Dataset, model: nn.Module | None) -> torch.Tensor:
        """Per-class 'balanced' CE weights derived from *global* label counts.

        With a pre-sharded dataset every DP rank holds a different shard, so weights derived from
        the local shard alone diverge across ranks (silently rank-divergent loss weighting). Label
        counts are therefore summed over the WORLD group before deriving weights. WORLD is correct
        in every topology: DP shards are disjoint (true global sum), while TP siblings share their
        DP rank's shard and replicated (non-presharded) datasets are identical on every rank — both
        contribute a uniform scale on every class count, and balanced weights
        ``n_samples / (n_present_classes * count_c)`` are invariant to a uniform count scale.
        """
        columns = train_dataset.column_names if hasattr(train_dataset, "column_names") else list(train_dataset)
        label_column = next((c for c in ("labels", "label") if c in columns), None)
        if label_column is None:
            raise ValueError(
                f"derive_class_weights: the train dataset has no 'labels'/'label' column (columns: {list(columns)})"
            )
        labels_array = np.asarray(train_dataset[label_column])
        counts = torch.from_numpy(np.bincount(labels_array)).to(torch.float64)

        if dist.is_available() and dist.is_initialized():
            counts = counts.to(current_device())
            # Ranks can disagree on the local max label, so pad to a rank-uniform length first.
            length = torch.tensor(counts.numel(), dtype=torch.int64, device=counts.device)
            dist.all_reduce(length, op=dist.ReduceOp.MAX)
            padded = torch.zeros(int(length.item()), dtype=counts.dtype, device=counts.device)
            padded[: counts.numel()] = counts
            dist.all_reduce(padded, op=dist.ReduceOp.SUM)
            counts = padded.cpu()

        num_labels = getattr(getattr(model, "config", None), "num_labels", None) or counts.numel()
        if counts.numel() > num_labels:
            raise ValueError(
                f"derive_class_weights: train labels reach class {counts.numel() - 1} "
                f"but the model has num_labels={num_labels}"
            )
        present = torch.zeros(num_labels, dtype=torch.bool)
        present[: counts.numel()] = counts > 0
        if not present.any():
            raise ValueError("derive_class_weights: the train dataset 'labels' column is empty")

        full = torch.ones(num_labels, dtype=torch.float32)
        full[present] = (counts.sum() / (present.sum() * counts[counts > 0])).to(torch.float32)
        return full

    def _build_loss_fn(self, args, train_dataset, model=None):
        """Build the loss function based on config. Returns None to use model's default."""
        # Multi-label builds its own loss: the head's dtype sniffing crashes on int multi-hot labels
        # and latches problem_type into config.json, while the PP path is dtype-blind BCE.
        has_custom_loss = (
            args.loss_type != "cross_entropy"
            or args.class_weights is not None
            or args.derive_class_weights
            or self.is_multi_label
        )
        if not has_custom_loss:
            return None

        if self.is_multi_label:
            if args.loss_type == "label_smoothing_ce":
                raise ValueError(
                    "loss_type='label_smoothing_ce' is single-label only (softmax CE). "
                    "Use 'focal' or class_weights (weighted BCE) with multi-label datasets."
                )
            if args.derive_class_weights:
                raise ValueError(
                    "derive_class_weights is not supported with multi-label datasets — pass class_weights explicitly."
                )

        weight = None
        if args.class_weights is not None:
            weight = torch.tensor(args.class_weights, dtype=torch.float32)
        elif args.derive_class_weights:
            weight = self._balanced_class_weights(train_dataset, model)

        if args.loss_type == "focal":
            if args.focal_alpha is not None and not self.is_multi_label:
                raise ValueError(
                    "focal_alpha balances positives against negatives (alpha_t = alpha*y + "
                    "(1-alpha)*(1-y)), which only exists on a multi-label (sigmoid) head. This is a "
                    "single-label softmax head, where every element has exactly one target — a "
                    "scalar alpha there is a uniform rescale of the loss (a learning-rate change in "
                    "disguise), not balancing. Use class_weights or derive_class_weights for per-class "
                    "balancing, and focal_gamma to down-weight easy examples."
                )
            return partial(
                self._focal_loss,
                gamma=args.focal_gamma,
                alpha=args.focal_alpha,
                weight=weight,
            )
        elif args.loss_type == "label_smoothing_ce":
            return nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
        elif self.is_multi_label:
            return nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            return nn.CrossEntropyLoss(weight=weight)

    def _move_loss_weights_to(self, device) -> None:
        """Move the configured loss's class-weight tensors onto ``device`` lazily (idempotent).

        Device only — the weights keep the fp32 they were built with. Every objective here evaluates
        in fp32 (:func:`_loss_inputs_fp32`), so ``F.cross_entropy``'s "weight must carry the input's
        dtype" rule is satisfied without ever rounding the weight vector itself, whose entries span
        orders of magnitude under ``derive_class_weights``.
        """
        if isinstance(self._loss_fn, nn.CrossEntropyLoss) and self._loss_fn.weight is not None:
            self._loss_fn.weight = self._loss_fn.weight.to(device=device)
        elif isinstance(self._loss_fn, nn.BCEWithLogitsLoss) and self._loss_fn.pos_weight is not None:
            self._loss_fn.pos_weight = self._loss_fn.pos_weight.to(device=device)
        keywords = getattr(self._loss_fn, "keywords", None)
        if keywords is not None and keywords.get("weight") is not None:
            keywords["weight"] = keywords["weight"].to(device=device)

    def _pooled_loss(
        self, pooled_logits: torch.Tensor, targets: torch.Tensor, valid_rows: torch.Tensor
    ) -> torch.Tensor:
        """The configured objective on POOLED logits, summed over ``valid_rows``.

        The pipeline seam needs the SUM form: per-microbatch sums are additive, and the runtime
        divides by :meth:`_pooled_loss_normalizer` to recover exactly the non-PP per-batch mean.
        Covers every ``_build_loss_fn`` outcome: model-default CE/BCE (``_loss_fn is None``),
        weighted / label-smoothed ``nn.CrossEntropyLoss``, weighted ``nn.BCEWithLogitsLoss``, and
        the focal partial (which accepts ``reduction`` itself).

        The inert rows are zeroed in VALUE space rather than indexed out: a boolean index would
        force a device→host sync on every micro-batch. Their LOGITS are neutralized before the
        objective, not just their loss after it — masking only the output still runs the objective's
        backward over the row, where a saturated logit makes ``softmax`` NaN and the chain rule
        propagates it through the zero. The zeros change no individual addition, but they lengthen
        the reduction, so the result matches summing the surviving rows alone to ~2 fp32 ULP rather
        than bit-exactly.
        """
        logits, targets = _loss_inputs_fp32(pooled_logits, targets)
        row_mask = valid_rows.reshape(-1, *(1,) * (logits.dim() - 1))
        logits = torch.where(row_mask, logits, logits.new_zeros(()))
        if self._loss_fn is None:
            if self.is_multi_label:
                per_element = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            else:
                per_element = F.cross_entropy(logits, targets, reduction="none")
        else:
            self._move_loss_weights_to(logits.device)
            if isinstance(self._loss_fn, nn.CrossEntropyLoss):
                per_element = F.cross_entropy(
                    logits,
                    targets,
                    weight=self._loss_fn.weight,
                    label_smoothing=self._loss_fn.label_smoothing,
                    reduction="none",
                )
            elif isinstance(self._loss_fn, nn.BCEWithLogitsLoss):
                per_element = F.binary_cross_entropy_with_logits(
                    logits, targets, pos_weight=self._loss_fn.pos_weight, reduction="none"
                )
            else:
                per_element = self._loss_fn(logits, targets, reduction="none")
        mask = valid_rows.reshape(-1, *(1,) * (per_element.dim() - 1))
        return torch.where(mask, per_element, per_element.new_zeros(())).sum()

    def _pooled_loss_normalizer(self, targets: torch.Tensor, valid_rows: torch.Tensor) -> torch.Tensor:
        """Denominator turning the summed pooled loss back into this trainer's per-batch mean.

        Per variant (torch reduction semantics): element count ``B*C`` for BCE / multi-label focal
        (``pos_weight`` weights the summands, not the denominator); the sum of the target classes'
        weights for weighted / label-smoothed CE (``F.cross_entropy(weight=..., reduction='mean')``
        divides by it); the example count ``B`` for unweighted CE and single-label focal (focal is
        a plain mean over per-example values even when class-weighted).

        ``valid_rows`` masks the inert rows the same way :meth:`_pooled_loss` does, so numerator and
        denominator agree on which rows exist without either of them indexing the device.
        """
        rows = valid_rows.sum()
        if self.is_multi_label:
            return (rows * targets.size(-1)).float().clamp(min=1.0)
        if isinstance(self._loss_fn, nn.CrossEntropyLoss) and self._loss_fn.weight is not None:
            self._move_loss_weights_to(targets.device)
            # Inert rows carry LABEL_IGNORE_INDEX; clamp only to keep the gather in range.
            per_row = self._loss_fn.weight[targets.clamp_min(0)]
            weight_sum = torch.where(valid_rows, per_row, per_row.new_zeros(())).sum()
            # A batch of only inert rows sums to zero, and zero numerator / zero denominator is NaN —
            # which then poisons the whole reduced loss. Substituted only at zero, so a real batch's
            # denominator is untouched.
            return torch.where(weight_sum > 0, weight_sum, weight_sum.new_ones(()))
        return rows.float().clamp(min=1.0)

    def _setup_pipeline_parallel(self):
        """Seed the PP-only pooling pad id and refuse a head this adapter cannot score, then run the
        mixin's setup.

        Kept out of ``_pp_loss_adapter``, which the mixin may call as a declarative accessor.
        """
        self._pp_pool_pad_id = pooling_pad_id_for(self)
        if not self.is_multi_label and self.model.config.num_labels < 2:
            raise ValueError(
                f"Sequence classification under pipeline parallelism needs num_labels >= 2 (got "
                f"{self.model.config.num_labels}): transformers treats a single-logit head as "
                f"regression (MSE), which the PP loss adapter does not implement."
            )
        super()._setup_pipeline_parallel()

    def _pp_loss_adapter(self) -> PPLossAdapter:
        """Pooled sequence classification as a last-stage pipeline loss.

        The batch transform rewrites the collated ``labels`` into the runtime's ``[B, S]`` plane:
        ``-100`` everywhere except the pooled position (transformers' rightmost-non-pad rule), which
        carries the class id — so single-label needs no side tensors at all. Multi-hot targets are
        float ``[B, num_labels]`` and cannot ride a long plane; they ship as the ``class_targets``
        extra with an inert ``0`` marker in the plane. Every loss variant is microbatch-invariant in
        its sum form with a batch-level denominator (see ``_pooled_loss`` / ``_pooled_loss_normalizer``),
        so none is rejected; ``num_labels == 1`` is rejected in ``_setup_pipeline_parallel`` because
        transformers' default head loss degenerates to regression MSE there, which this adapter does
        not wire.
        """
        return PPLossAdapter(
            token_loss_fn=self._pp_classification_token_loss,
            batch_transform=self._pp_classification_batch_transform,
            normalizer=self._pp_classification_normalizer,
            # Same denominator at eval: the causal-LM token-count fallback would score another quantity.
            eval_normalizer=self._pp_classification_normalizer,
            extra_target_keys=("class_targets",) if self.is_multi_label else (),
            predictions_fn=self._pp_classification_predictions,
            eval_labels_fn=self._pp_classification_eval_labels,
        )

    def _pp_classification_predictions(self, logits: torch.Tensor, inputs: dict) -> torch.Tensor:
        """Per-token head outputs → the pooled ``[B, num_labels]`` that ``compute_metrics`` consumes.

        Pooling here rather than after the broadcast is what keeps the chain hop small: the raw
        stage output is ``[B, S, num_labels]``.
        """
        _valid, positions, _values = decode_pooling_plane(inputs["labels"])
        return pooled_outputs(logits, positions).float()

    def _pp_classification_eval_labels(self, inputs: dict) -> torch.Tensor:
        """The real class targets, recovered from the runtime-shaped labels plane."""
        if self.is_multi_label:
            return inputs["class_targets"]
        _valid, _positions, plane_values = decode_pooling_plane(inputs["labels"])
        return plane_values

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
            if class_labels.dim() != 1:
                raise ValueError(
                    f"Single-label classification under pipeline parallelism expects [B] integer "
                    f"labels, got shape {tuple(class_labels.shape)}. For multi-hot targets pass "
                    f"is_multi_label=True."
                )
            markers = class_labels.to(input_ids.dtype)
        out["labels"] = encode_pooling_plane(input_ids, self._pp_pool_pad_id, markers)
        return out

    def _pp_classification_token_loss(self, logits: torch.Tensor, target: torch.Tensor | dict) -> torch.Tensor:
        """Summed pooled loss over the microbatch (the runtime divides by the batch normalizer).

        Runs on every micro-batch, so it stays free of ``.item()``/``bool()``/boolean indexing: the
        inert rows are carried through the loss and masked out of the sum instead (:meth:`_pooled_loss`).
        An all-inert micro-batch needs no special case — the masked sum is a real zero still wired to
        the graph, so backward produces zero grads rather than dropping the rank out of the schedule.
        """
        plane = target["labels"] if isinstance(target, dict) else target
        valid, positions, plane_values = decode_pooling_plane(plane)
        pooled = pooled_outputs(logits, positions)
        # Inert rows decode to LABEL_IGNORE_INDEX; clamp keeps the CE gather in range before masking.
        targets = target["class_targets"] if self.is_multi_label else plane_values.clamp_min(0)
        return self._pooled_loss(pooled, targets, valid)

    def _pp_classification_normalizer(self, inputs: dict) -> torch.Tensor:
        """Chain-local batch denominator. This trainer's loss is a per-replica mean
        (``_loss_is_own_mean``; DP averages gradients — a mean of per-replica
        means), so unlike SFT's DP-global token count no cross-rank reduction is involved."""
        valid, _, plane_values = decode_pooling_plane(inputs["labels"])
        targets = inputs["class_targets"] if self.is_multi_label else plane_values
        return self._pooled_loss_normalizer(targets, valid)

    def compute_loss(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute classification loss via the custom loss fn if configured, else the model's built-in head."""
        self._validate_inputs(inputs)

        with peft_bf16_autocast(self._peft_has_been_casted_to_bf16, self.accelerator.device):
            return self._compute_loss_inner(model, inputs, return_outputs)

    def _compute_loss_inner(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self._loss_fn is None:
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                return_dict=True,
            )
            loss = outputs.loss
            logits = outputs.logits
        else:
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )
            logits = outputs.logits

            self._move_loss_weights_to(logits.device)
            loss = self._loss_fn(*_loss_inputs_fp32(logits, inputs["labels"]))

        if return_outputs:
            return loss, {"logits": logits}
        return loss

    def prediction_step(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Perform a prediction step (``ignore_keys`` is inert — the loss returns ``logits`` directly)."""
        if self._pp_runtime is not None:
            return self._pp_prediction_step(inputs, prediction_loss_only, ignore_keys)
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()
        logits = outputs["logits"]
        logits = nested_detach(logits)
        labels = inputs["labels"]
        labels = self._prepare_inputs(labels)

        return loss, logits, labels
