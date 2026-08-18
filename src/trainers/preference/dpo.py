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
Distributed DPO (Direct Preference Optimization) trainer: EP, TP, and PP.

CP is NOT supported: the chosen/rejected forward needs global log-prob sums over full
sequences for the Bradley-Terry model, incompatible with sequence splitting.

Under EP/TP the reference model is NOT parallelized — use PEFT/LoRA (ref_model=None)
or precompute_ref_log_probs=True (then free the ref model).

Under PP the trainer is PRECOMPUTE-ONLY: the ``ref_chosen_logps`` / ``ref_rejected_logps`` values
must already be dataset columns, since no pipeline rank holds a full model to run a reference
forward. The ``[chosen ⧺ rejected]`` concat is re-laid out as interleaved pairs so no microbatch
split separates a pair, and only the per-pair loss types the PP last-stage loss re-implements are
allowed (see ``_validate_pp_mode`` for every rejection and its mechanism).
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from trl import DPOTrainer

from src.data.spans import LABEL_IGNORE_INDEX
from src.distributed.pipeline_parallel.losses import (
    PPLossAdapter,
    completion_labels,
    interleave_pairs,
    pair_count_normalizer,
    rows_with_labels,
    sequence_logprobs,
    split_pairs,
)
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.mixins.pp_gates import (
    reject_pp_compute_metrics,
    reject_pp_ref_model,
    require_model_and_args_kwargs,
    require_precomputed_reference,
)
from src.trainers.mixins.validation import ctor_config, ctor_positions, ctor_value, disable_trl_liger
from src.trainers.preference.precompute import PrecomputeRefLogpsRankConsistentMixin

# TRL DPOTrainer positional slot for the EP/TP reference gate — derived from the installed signature.
_CTOR_POSITIONS = ctor_positions(DPOTrainer, "ref_model")

_REF_LOGPS_COLUMNS = ("ref_chosen_logps", "ref_rejected_logps")

_PP_BATCH_COUPLED_LOSS_TYPES = {
    "aot": "it sorts the paired log-ratios across the whole batch (torch.sort over dim 0), so a "
    "microbatch split re-pairs the sorted quantiles",
    "aot_unpaired": "it sorts the chosen and rejected log-ratios across the whole batch "
    "(torch.sort over dim 0), so a microbatch split re-pairs the sorted quantiles",
    "sft": "its cross-entropy is a mean over the whole batch's chosen completion tokens — a "
    "batch-global token denominator that changes with the microbatch split",
}


# Uniform signature so the registry dispatches on the loss type alone; score-only types drop labels.
def _sigmoid_pair_loss(
    beta: float, chosen_scores: torch.Tensor, rejected_scores: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    del labels
    return -F.logsigmoid(beta * (chosen_scores - rejected_scores))


def _hinge_pair_loss(
    beta: float, chosen_scores: torch.Tensor, rejected_scores: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    del labels
    return torch.relu(1 - beta * (chosen_scores - rejected_scores))


def _ipo_pair_loss(
    beta: float, chosen_scores: torch.Tensor, rejected_scores: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """IPO's per-sequence normalization uses the UNSHIFTED completion-token count, matching TRL's
    ``completion_mask.sum(dim=1)`` (labels are built from ``completion_mask``, so the label count is
    that count)."""
    counts = (labels != LABEL_IGNORE_INDEX).sum(dim=-1).clamp(min=1).float()
    chosen_counts, rejected_counts = split_pairs(counts)
    ipo_delta = chosen_scores / chosen_counts - rejected_scores / rejected_counts
    return (ipo_delta - 1 / (2 * beta)) ** 2


# One mapping for both the PP gate and the last-stage loss, so neither drifts into another branch.
_PP_PAIR_LOSSES: dict[str, Callable[[float, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "sigmoid": _sigmoid_pair_loss,
    "hinge": _hinge_pair_loss,
    "ipo": _ipo_pair_loss,
}


class DistributedDPOTrainer(DistributedTrainerMixin, PrecomputeRefLogpsRankConsistentMixin, DPOTrainer):
    """TRL's DPOTrainer plus EP/TP/PP via DistributedTrainerMixin. CP unsupported (see module docstring).

    ``PrecomputeRefLogpsRankConsistentMixin`` keeps ``precompute_ref_log_probs`` (the EP/TP
    full-finetune reference path) from deadlocking on its rank-divergent disk cache. PP is
    precompute-only with the reference log-probs shipped as dataset columns; the PP contract lives
    in ``_pp_loss_adapter``.
    """

    _tag_names = ["trl", "dpo"]

    _supports_pp = True
    # TRL concatenates chosen ⧺ rejected into ONE forward, so every MoE layer sees twice the rows.
    _forward_rows_per_example = 2

    def __init__(self, *args, **kwargs):
        config = kwargs.get("parallelism_config")
        if config is not None and config.is_pp_mode:
            # MUST run before _init_distributed_config, which drives the PP gate and the split.
            disable_trl_liger(
                ctor_config(args, kwargs),
                "DPO under pipeline parallelism: disabling use_liger_kernel — TRL's Liger DPO loss "
                "has no precompute_ref_log_probs branch, and the PP last-stage loss replaces TRL's "
                "loss path entirely.",
            )
        kwargs = self._init_distributed_config(kwargs, ctor_args=args)
        self._validate_reference_model(ctor_value(args, kwargs, "ref_model", _CTOR_POSITIONS))
        super().__init__(*args, **kwargs)
        # Post-super: TRL builds its own reference inside __init__ when beta != 0 without PEFT.
        self._validate_implicit_reference_model()
        self._setup_distributed_modes()

    def _validate_pp_mode(self, ctor_args: tuple, kwargs: dict) -> None:
        """Reject DPO configurations pipeline parallelism cannot honor (fail loud at construction).

        Runs BEFORE the model is split, so every raise here is cheap and rank-uniform. The remaining
        ctor parameters are read from ``kwargs`` only: ``require_model_and_args_kwargs`` forces
        ``model`` into kwargs, and any positional argument would rebind TRL's ``model`` slot and
        fail at ``super().__init__``.
        """
        require_model_and_args_kwargs(kwargs)
        training_args = kwargs["args"]
        reject_pp_ref_model(
            ctor_value(ctor_args, kwargs, "ref_model", _CTOR_POSITIONS),
            "ref_chosen_logps/ref_rejected_logps dataset columns",
        )
        require_precomputed_reference(
            "DPO",
            training_args,
            kwargs.get("train_dataset"),
            kwargs.get("eval_dataset"),
            _REF_LOGPS_COLUMNS,
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
        loss_types = training_args.loss_type
        loss_types = loss_types if isinstance(loss_types, list) else [loss_types]
        for loss_type in loss_types:
            if loss_type in _PP_PAIR_LOSSES:
                continue
            mechanism = _PP_BATCH_COUPLED_LOSS_TYPES.get(
                loss_type,
                "its per-pair loss is not re-implemented in the PP last-stage loss, and running it "
                "unaudited would train a silently different objective",
            )
            raise ValueError(
                f"loss_type '{loss_type}' is not supported under pipeline parallelism: {mechanism}. "
                f"Supported under PP: {', '.join(_PP_PAIR_LOSSES)}."
            )
        if training_args.f_divergence_type != "reverse_kl":
            raise ValueError(
                f"f_divergence_type='{training_args.f_divergence_type}' is not supported under "
                f"pipeline parallelism: the PP last-stage loss implements the standard reverse-KL "
                f"scores only; another divergence would silently be computed as reverse-KL."
            )
        if training_args.use_weighting:
            raise ValueError(
                "use_weighting (WPO) is not supported under pipeline parallelism: the per-pair "
                "weights need a full-vocab logsumexp over the shifted logits inside the loss, which "
                "the PP last-stage loss does not implement."
            )
        if training_args.ld_alpha is not None:
            raise ValueError(
                "ld_alpha (length-desensitized DPO) is not supported under pipeline parallelism: it "
                "reshapes the sequence log-probs themselves, the PP last-stage loss does not "
                "implement that shaping, and the precomputed reference columns would have to be "
                "produced with the identical shaping to stay comparable."
            )

    def _required_ref_logps_columns(self) -> tuple[str, ...]:
        """Both columns TRL's DPO sweep would write; present ⇒ the sweep is skipped."""
        return _REF_LOGPS_COLUMNS

    def _pp_loss_adapter(self) -> PPLossAdapter:
        """DPO's pipeline-loss contract: interleaved pairs + dataset-precomputed reference log-probs.

        ``completion_mask`` rides ``pad_spec`` so the batch transform sees full-length tensors; the
        reference log-probs ship interleaved as a per-example extra target so the microbatch split
        keeps them row-aligned with their pairs. ``pin_runtime_columns=False``: DPO's dataset columns
        feed TRL's preference collator, so HF's column pruning must keep TRL's own signature set, not
        the runtime contract. ``eval_normalizer`` is the same pre-pad pair count as training.
        """
        return PPLossAdapter(
            token_loss_fn=self._pp_dpo_token_loss,
            paired_examples=True,
            batch_transform=self._pp_dpo_batch_transform,
            normalizer=pair_count_normalizer,
            extra_target_keys=("ref_logps",),
            pad_spec={"completion_mask": 0},
            rows_per_example=2,
            eval_normalizer=pair_count_normalizer,
            pin_runtime_columns=False,
        )

    def _pp_dpo_batch_transform(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """``[chosen ⧺ rejected]`` → interleaved ``[c0, r0, ...]``, labels from the completion mask.

        TRL's collator emits the chunk-style concat, which ``tensor_split`` would break into
        microbatches comparing unrelated examples; interleaving keeps every pair together. Labels
        are the completion token ids at their positions (ignore elsewhere) — exactly the positions
        TRL's ``shift_completion_mask`` scores.
        """
        input_ids = interleave_pairs(*batch["input_ids"].chunk(2, dim=0))
        attention_mask = interleave_pairs(*batch["attention_mask"].chunk(2, dim=0))
        completion_mask = interleave_pairs(*batch["completion_mask"].chunk(2, dim=0))
        labels = completion_labels(input_ids, completion_mask)
        ref_logps = interleave_pairs(batch["ref_chosen_logps"].float(), batch["ref_rejected_logps"].float())
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "ref_logps": ref_logps,
        }

    def _pp_dpo_token_loss(self, logits: torch.Tensor, target: dict[str, torch.Tensor]) -> torch.Tensor:
        """Summed per-pair DPO losses over one INTERLEAVED microbatch (÷ pair count by the runtime).

        Reproduces TRL's ``_compute_loss`` for the supported subset: reverse-KL scores against the
        precomputed reference, then the weighted sum over the configured loss types of their
        ``_PP_PAIR_LOSSES`` implementations — the same mapping ``_validate_pp_mode`` gates on.
        """
        labels = target["labels"]
        seq_logps = sequence_logprobs(logits, labels)
        chosen_logps, rejected_logps = split_pairs(seq_logps)
        ref_chosen_logps, ref_rejected_logps = split_pairs(target["ref_logps"].float())
        chosen_scores = chosen_logps - ref_chosen_logps
        rejected_scores = rejected_logps - ref_rejected_logps
        # Inert eval-padding pairs must contribute 0: at zero scores every type is nonzero.
        pair_valid = rows_with_labels(labels)[0::2].float()

        loss = seq_logps.new_zeros(())
        for loss_type, loss_weight in zip(self.loss_types, self.loss_weights, strict=True):
            per_pair = _PP_PAIR_LOSSES[loss_type](self.beta, chosen_scores, rejected_scores, labels)
            loss = loss + (per_pair * pair_valid).sum() * loss_weight
        return loss
