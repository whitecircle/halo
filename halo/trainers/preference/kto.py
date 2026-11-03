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
"""Distributed KTO trainer: EP, TP and PP; CP unsupported (it would split the full-sequence log-prob sums and KL term).
Under EP/TP the reference is not parallelized. PP is precompute-only and ``apo_zero_unpaired``-only, a per-row function
of the last stage's logits."""

import torch
from trl import KTOTrainer

from halo.data.completion_spans import completion_labels
from halo.kernels.logprobs import sequence_logprobs
from halo.trainers.ctor_slots import get_ctor_arg
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.parallelism.pipeline_loss_adapter import PipelineLossAdapter, row_count_normalizer, rows_with_labels
from halo.trainers.parallelism.pipeline_rules import (
    reject_missing_precomputed_reference,
    reject_positional_model_or_args,
    reject_pp_compute_metrics,
    reject_pp_ref_model,
)
from halo.trainers.preference.preference_rules import reject_kto_pp_loss_type
from halo.trainers.preference.trl_preference import TRLPreferenceMixin
from halo.trainers.trl_dataset_prep import TRLDatasetPrepMixin


class DistributedKTOTrainer(TRLPreferenceMixin, DistributedTrainerMixin, TRLDatasetPrepMixin, KTOTrainer):
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
        """The PP loss reproduces ``beta`` and both desirability weights; ctor parameters are read from ``kwargs``
        only, as in DPO."""
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
        """Rows are independent, so the microbatch split is safe at any size. ``pin_runner_columns=False`` because
        KTO's columns feed TRL's unpaired-preference collator, which needs TRL's own signature set."""
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
        """Labels are the completion token ids at the positions TRL's shifted ``completion_mask`` scores; ``label``
        becomes a per-row float tensor so the microbatch split can carry it."""
        input_ids = batch["input_ids"]
        return {
            "input_ids": input_ids,
            "attention_mask": batch["attention_mask"],
            "labels": completion_labels(input_ids, batch["completion_mask"]),
            "ref_logps": batch["ref_logps"].float(),
            "label": torch.as_tensor(batch["label"], dtype=torch.float32, device=input_ids.device),
        }

    def _pp_kto_token_loss(self, logits: torch.Tensor, target: dict[str, torch.Tensor]) -> torch.Tensor:
        """Summed per-row apo_zero_unpaired losses: ``desirable_weight · (1 − σ(β·logratio))`` and ``undesirable_weight
        · σ(β·logratio)``. TRL takes ``losses.nanmean()``; sum ÷ row count is equivalent when no row is NaN, and lets a
        NaN from corrupted data or references surface instead of being dropped from the mean."""
        seq_logps = sequence_logprobs(logits, target["labels"])
        logratios = seq_logps - target["ref_logps"].float()
        desirable = target["label"].bool()
        per_row = torch.where(
            desirable,
            self.desirable_weight * (1 - torch.sigmoid(self.beta * logratios)),
            self.undesirable_weight * torch.sigmoid(self.beta * logratios),
        )
        return (per_row * rows_with_labels(target["labels"]).float()).sum()
