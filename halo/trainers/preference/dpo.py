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
"""Distributed DPO trainer: EP, TP and PP; CP unsupported (the Bradley-Terry loss needs full-sequence sums). Under
EP/TP the reference is not parallelized: use PEFT/LoRA or ``precompute_ref_log_probs``. PP is precompute-only and
interleaves the concat into pairs."""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from trl import DPOTrainer

from halo.data.completion_spans import completion_labels
from halo.kernels.logprobs import LABEL_IGNORE_INDEX, sequence_logprobs
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.parallelism.pipeline_loss_adapter import (
    PipelineLossAdapter,
    interleave_halves,
    interleave_pairs,
    pair_count_normalizer,
    split_pairs,
    valid_pairs,
)
from halo.trainers.parallelism.pipeline_rules import (
    reject_missing_precomputed_reference,
    reject_positional_model_or_args,
    reject_pp_compute_metrics,
    reject_pp_ref_model,
)
from halo.trainers.preference.preference_rules import reject_dpo_pp_objective
from halo.trainers.preference.trl_preference import TRLPreferenceMixin
from halo.trainers.trl_dataset_prep import TRLDatasetPrepMixin


def _ipo_pair_loss(
    beta: float, chosen_scores: torch.Tensor, rejected_scores: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """IPO normalizes by the unshifted completion-token count, matching TRL's ``completion_mask.sum(dim=1)``."""
    counts = (labels != LABEL_IGNORE_INDEX).sum(dim=-1).clamp(min=1).float()
    chosen_counts, rejected_counts = split_pairs(counts)
    ipo_delta = chosen_scores / chosen_counts - rejected_scores / rejected_counts
    return (ipo_delta - 1 / (2 * beta)) ** 2


# ``(beta, chosen_scores, rejected_scores, labels) -> per-pair loss``, shared by the PP gate and the
# last-stage loss so the two cannot drift apart; score-only types drop the labels.
_PP_PAIR_LOSSES: dict[str, Callable[[float, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "sigmoid": lambda beta, chosen, rejected, labels: -F.logsigmoid(beta * (chosen - rejected)),
    "hinge": lambda beta, chosen, rejected, labels: torch.relu(1 - beta * (chosen - rejected)),
    "ipo": _ipo_pair_loss,
}


class DistributedDPOTrainer(TRLPreferenceMixin, DistributedTrainerMixin, TRLDatasetPrepMixin, DPOTrainer):
    _tag_names = ["trl", "dpo"]

    PP_REFUSAL = None
    # TRL concatenates chosen ⧺ rejected into one forward, so every MoE layer sees twice the rows.
    FORWARD_ROWS_PER_EXAMPLE = 2
    REFERENCE_COLUMNS = ("ref_chosen_logps", "ref_rejected_logps")
    LIGER_DISABLE_REASON = (
        "Disabling TRL's use_liger_kernel for DPO: TRL's chunked DPO log-prob path refuses the router "
        "aux loss and skips the family's head transform. Liger kernels are still applied at the model "
        "level by load_distributed_model."
    )

    def _reject_unsupported_pp_options(self, kwargs: dict) -> None:
        """Ctor parameters are read from ``kwargs`` only, since a positional argument would rebind TRL's ``model``
        slot."""
        reject_positional_model_or_args(kwargs)
        training_args = kwargs["args"]
        reject_pp_ref_model(kwargs.get("ref_model"), "ref_chosen_logps/ref_rejected_logps dataset columns")
        reject_missing_precomputed_reference(
            "DPO",
            training_args,
            kwargs.get("train_dataset"),
            kwargs.get("eval_dataset"),
            self.REFERENCE_COLUMNS,
        )
        reject_pp_compute_metrics(
            kwargs.get("compute_metrics"),
            (
                "TRL's DPO "
                "prediction_step hands metrics the RAW logits plane of the [chosen ⧺ rejected] "
                "concat (outputs.logits, paired with input_ids), and a pipeline cannot serve that "
                "convention. Its rows are interleaved [c0, r0, c1, r1, ...] so the concat's halves "
                "are not where a metric expects them; re-chunking them back would mis-slice a "
                "partial eval batch (the trailing inert rows are trimmed off the interleaved axis, "
                "which in concat order cuts into the chosen half); and the plane itself — "
                "[rows, max_length, vocab] — would be broadcast from the last stage to every rank "
                "in the chain. Any smaller tensor would be a different prediction than the non-PP "
                "path's. Evaluate on eval_loss alone, or compute the metric in a non-PP run."
            ),
        )
        reject_dpo_pp_objective(training_args, supported=_PP_PAIR_LOSSES)

    def _reference_settings(self) -> dict:
        """The collator's truncation and ``ld_alpha``'s reshaping of the summed reference log-probs."""
        return {
            "max_length": self.args.max_length,
            "truncation_mode": self.args.truncation_mode,
            "ld_alpha": self.ld_alpha,
        }

    def _pp_loss_adapter(self) -> PipelineLossAdapter:
        """``completion_mask`` rides ``pad_spec`` for full-length tensors; reference log-probs ship interleaved to stay
        row-aligned. ``pin_runner_columns=False`` because DPO's columns feed TRL's preference collator, which needs
        TRL's own signature set."""
        return PipelineLossAdapter(
            token_loss_fn=self._pp_dpo_token_loss,
            batch_transform=self._pp_dpo_batch_transform,
            normalizer=pair_count_normalizer,
            extra_target_keys=("ref_logps",),
            pad_spec={"completion_mask": 0},
            eval_normalizer=pair_count_normalizer,
            pin_runner_columns=False,
        )

    def _pp_dpo_batch_transform(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """``tensor_split`` of the concat would compare unrelated examples. Labels are the completion token ids at the
        positions TRL's ``shift_completion_mask`` scores."""
        input_ids = interleave_halves(batch["input_ids"])
        attention_mask = interleave_halves(batch["attention_mask"])
        completion_mask = interleave_halves(batch["completion_mask"])
        labels = completion_labels(input_ids, completion_mask)
        ref_logps = interleave_pairs(batch["ref_chosen_logps"].float(), batch["ref_rejected_logps"].float())
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "ref_logps": ref_logps,
        }

    def _pp_dpo_token_loss(self, logits: torch.Tensor, target: dict[str, torch.Tensor]) -> torch.Tensor:
        """Summed per-pair DPO losses over one interleaved microbatch; reproduces TRL's ``_compute_loss`` for the
        subset ``_reject_unsupported_pp_options`` gates on."""
        labels = target["labels"]
        seq_logps = sequence_logprobs(logits, labels)
        chosen_logps, rejected_logps = split_pairs(seq_logps)
        ref_chosen_logps, ref_rejected_logps = split_pairs(target["ref_logps"].float())
        chosen_scores = chosen_logps - ref_chosen_logps
        rejected_scores = rejected_logps - ref_rejected_logps
        pair_valid = valid_pairs(labels)

        loss = seq_logps.new_zeros(())
        for loss_type, loss_weight in zip(self.loss_types, self.loss_weights, strict=True):
            per_pair = _PP_PAIR_LOSSES[loss_type](self.beta, chosen_scores, rejected_scores, labels)
            loss = loss + (per_pair * pair_valid).sum() * loss_weight
        return loss
