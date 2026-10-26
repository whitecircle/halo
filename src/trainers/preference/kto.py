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
"""Distributed KTO trainer (unpaired desirable/undesirable completions): EP, TP and PP; CP unsupported.

CP would split the full-sequence log-prob sums and KL reference term KTO needs. Under EP/TP the reference is
not parallelized: use PEFT/LoRA or ``precompute_ref_log_probs``. Under PP the trainer is precompute-only and
``apo_zero_unpaired``-only (no KL term), so the objective is a per-row function of the last stage's logits.
"""

import torch
from trl import KTOTrainer

from src.data.collators.fixed_shape import pad_to_fixed_length
from src.data.completion_spans import completion_labels
from src.kernels.logprobs import LABEL_IGNORE_INDEX, sequence_logprobs
from src.trainers.ctor_slots import get_ctor_arg
from src.trainers.distributed_trainer import DistributedTrainerMixin
from src.trainers.parallelism.pipeline_loss_adapter import PipelineLossAdapter, row_count_normalizer, rows_with_labels
from src.trainers.parallelism.pipeline_rules import (
    reject_missing_precomputed_reference,
    reject_positional_model_or_args,
    reject_pp_compute_metrics,
    reject_pp_ref_model,
)
from src.trainers.preference.preference_rules import reject_kto_pp_loss_type
from src.trainers.preference.trl_preference import TRLPreferenceMixin
from src.trainers.trl_dataset_prep import TRLDatasetPrepMixin


class DistributedKTOTrainer(TRLPreferenceMixin, DistributedTrainerMixin, TRLDatasetPrepMixin, KTOTrainer):
    """TRL's KTOTrainer plus EP/TP/PP via DistributedTrainerMixin. CP unsupported (see module docstring)."""

    _tag_names = ["trl", "kto"]

    PP_REFUSAL = None
    REFERENCE_COLUMNS = ("ref_logps", "ref_KL_logps")
    REFERENCE_TOKEN_SETTINGS = "per_device_train_batch_size or dataset_num_proc, which pair the KL completions"
    LIGER_DISABLE_REASON = (
        "Disabling TRL's use_liger_kernel for KTO: TRL's chunked KTO log-prob path refuses the "
        "router aux loss. Liger kernels are still applied at the model level by "
        "load_distributed_model."
    )

    def _reject_unsupported_pp_options(self, ctor_args: tuple, kwargs: dict) -> None:
        """Refuse the KTO options the PP last-stage loss cannot honor (it reproduces ``beta`` and both
        desirability weights); ctor parameters are read from ``kwargs`` only, as in DPO."""
        reject_positional_model_or_args(kwargs)
        training_args = kwargs["args"]
        reject_kto_pp_loss_type(training_args)
        reject_pp_ref_model(
            get_ctor_arg(ctor_args, kwargs, "ref_model", self.CTOR_POSITIONS), "the ref_logps dataset column"
        )
        reject_missing_precomputed_reference(
            "KTO",
            training_args,
            kwargs.get("train_dataset"),
            kwargs.get("eval_dataset"),
            ("ref_logps",),
        )
        reject_pp_compute_metrics(
            kwargs.get("compute_metrics"),
            (
                "TRL's KTO "
                "prediction_step hands metrics the RAW logits plane (outputs.logits, paired with "
                "input_ids), and a pipeline can only serve that by broadcasting "
                "[rows, max_length, vocab] logits from the last stage to every rank in the chain — "
                "the eval hop the adapter's predictions_fn exists to shrink, and multiple GB per "
                "row at a production vocabulary. Any smaller tensor would be a different "
                "prediction than the non-PP path's. Evaluate on eval_loss alone, or compute the "
                "metric in a non-PP run."
            ),
        )

    def _reference_columns(self) -> tuple[str, ...]:
        """``ref_KL_logps`` only when the loss carries a KL term (``calculate_KL``, never under PP)."""
        return self.REFERENCE_COLUMNS if self.calculate_KL else ("ref_logps",)

    def _reference_settings(self) -> dict:
        """The collator's truncation, which keeps the start of each assembled sequence."""
        return {"max_length": self.args.max_length}

    def _pp_loss_adapter(self) -> PipelineLossAdapter:
        """KTO's pipeline-loss contract: unpaired rows + dataset-precomputed reference log-probs.

        Rows are independent, so the microbatch split is safe at any size; ``ref_logps`` and the
        per-row desirability ``label`` ship as extra targets to stay row-aligned. ``completion_mask``
        rides ``pad_spec`` so the batch transform sees it at the width the fixed-shape collator pads
        ``input_ids`` to. ``pin_runner_columns=False`` because KTO's dataset columns feed
        TRL's unpaired-preference collator, which needs TRL's own signature set.
        """
        return PipelineLossAdapter(
            token_loss_fn=self._pp_kto_token_loss,
            batch_transform=self._pp_kto_batch_transform,
            normalizer=row_count_normalizer,
            extra_target_keys=("ref_logps", "label"),
            pad_spec={"completion_mask": 0},
            eval_normalizer=row_count_normalizer,
            pin_runner_columns=False,
        )

    def _pp_kto_batch_transform(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """TRL's unpaired-preference collator batch → the pipeline's fixed-shape batch contract.

        Labels are the completion token ids at their positions (ignore elsewhere) — exactly the
        positions TRL's shifted ``completion_mask`` scores. Right-pads to ``args.max_length``
        because the P2P buffer shapes freeze on the first step; ``label`` (a list of bools from
        the collator) becomes a per-row float tensor so the microbatch split can carry it.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        completion_mask = batch["completion_mask"]
        labels = completion_labels(input_ids, completion_mask)
        # Over-length is unreachable with TRL's collator; the helper's raise guards a custom one.
        padded = pad_to_fixed_length(
            {
                "input_ids": (input_ids, self._tokenizer.pad_token_id),
                "attention_mask": (attention_mask, 0),
                "labels": (labels, LABEL_IGNORE_INDEX),
            },
            self.args.max_length,
            "Truncate in the collator, or raise max_length.",
        )
        return {
            **padded,
            "ref_logps": batch["ref_logps"].float(),
            "label": torch.as_tensor(batch["label"], dtype=torch.float32, device=input_ids.device),
        }

    def _pp_kto_token_loss(self, logits: torch.Tensor, target: dict[str, torch.Tensor]) -> torch.Tensor:
        """Summed per-row apo_zero_unpaired losses over one microbatch (÷ row count by the runner).

        Reproduces TRL's ``_compute_loss`` for the PP subset: per-row log-ratio against the
        precomputed reference, then ``desirable_weight · (1 − σ(β·logratio))`` for desirable rows
        and ``undesirable_weight · σ(β·logratio)`` for undesirable ones. The per-row ``where``
        makes a single-sided microbatch (all-desirable / all-undesirable) a plain partial sum, with
        no empty-tensor branch. Deviation from TRL's reduction: TRL takes ``losses.nanmean()``;
        sum ÷ row count is equivalent when no row is NaN, and lets a NaN from corrupted data or
        references surface instead of being dropped from the mean.
        """
        seq_logps = sequence_logprobs(logits, target["labels"])
        logratios = seq_logps - target["ref_logps"].float()
        desirable = target["label"].bool()
        per_row = torch.where(
            desirable,
            self.desirable_weight * (1 - torch.sigmoid(self.beta * logratios)),
            self.undesirable_weight * torch.sigmoid(self.beta * logratios),
        )
        return (per_row * rows_with_labels(target["labels"]).float()).sum()
