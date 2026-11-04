"""SMPO as a last-stage pipeline loss over interleaved pairs, each term divided by whole-batch constants."""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import torch

from halo.data.collators.fixed_shape import pad_to_fixed_length
from halo.data.collators.smpo import concatenate_pairs
from halo.kernels.logprobs import LABEL_IGNORE_INDEX, loss_token_counts_per_row, token_logprobs
from halo.trainers.parallelism.pipeline_loss_adapter import (
    PipelineLossAdapter,
    interleave_halves,
    pair_placeholder_labels,
    split_pairs,
    valid_pairs,
)
from halo.trainers.preference.smpo_objective import SMPOObjective, SMPOStepConstants, metric_sums, metrics_from_sums


@dataclass
class SMPOPipelineLoss:
    """A microbatch cannot derive SMPO's divisors or read the live margin, so :meth:`refresh_constants` records them
    from each step's whole batch and metrics accumulate as sums, equal to the non-PP step's over the microbatches."""

    objective: SMPOObjective
    pad_token_id: int
    max_length: int
    current_margin: Callable[[], float]
    report_margin: bool
    constants: SMPOStepConstants = field(default_factory=SMPOStepConstants)
    step_sums: dict[str, torch.Tensor] = field(default_factory=dict)

    def build(self) -> PipelineLossAdapter:
        """``pin_runner_columns=False``: SMPO's dataset columns feed the pair transform, so HF's pruning keeps
        SMPO's own signature set. The runner's divisor is 1: each term divides by its own constant."""
        return PipelineLossAdapter(
            token_loss_fn=self.token_loss,
            batch_transform=self.batch_transform,
            step_state_fn=self.refresh_constants,
            normalizer=lambda inputs: 1.0,
            eval_normalizer=lambda inputs: 1.0,
            pin_runner_columns=False,
            predictions_fn=self.predictions,
            eval_labels_fn=partial(pair_placeholder_labels, dtype=torch.float32),
            metrics_fn=self.step_metrics,
        )

    def batch_transform(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """The pair-interleaved concat, right-padded to ``max_length`` (the P2P shapes freeze on the first step)."""
        concat = concatenate_pairs(batch, self.pad_token_id)
        return pad_to_fixed_length(
            {
                key: (interleave_halves(concat[key]), fill)
                for key, fill in (
                    ("input_ids", self.pad_token_id),
                    ("attention_mask", 0),
                    ("labels", LABEL_IGNORE_INDEX),
                )
            },
            self.max_length,
            "Lower max_prompt_length/max_completion_length so their sum fits, or raise max_length.",
        )

    def refresh_constants(self, inputs: dict[str, torch.Tensor]) -> None:
        """Each token count is at least 1, so a step of padding alone scores 0."""
        counts = loss_token_counts_per_row(inputs["labels"])
        self.constants = SMPOStepConstants(
            pair_count=max(inputs["input_ids"].size(0) // 2, 1),
            chosen_token_count=counts[0::2].sum().float().clamp(min=1),
            rejected_token_count=counts[1::2].sum().float().clamp(min=1),
            target_margin=self.current_margin(),
        )

    def token_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """The microbatch's share of the step loss: margin losses over the step's pairs plus the SFT term over
        each side's step tokens."""
        constants = self.constants
        token_logps, loss_mask = token_logprobs(logits, labels)
        token_logps = token_logps.masked_fill(~loss_mask, 0)

        chosen_logps, rejected_logps = split_pairs(token_logps)
        # The SFT anchor reads the pre-clip likelihood, as the non-PP forward's does.
        chosen_nll_sum = -chosen_logps.sum()
        rejected_nll_sum = -rejected_logps.sum()

        chosen_mean_logps, rejected_mean_logps = self._pair_mean_logps(token_logps, loss_mask)
        margin_losses, chosen_rewards, rejected_rewards = self.objective.margin_loss(
            chosen_mean_logps, rejected_mean_logps, constants.target_margin
        )
        pair_valid = valid_pairs(labels)
        ratio = self.objective.chosen_sft_ratio
        sft_term = (
            ratio * chosen_nll_sum / constants.chosen_token_count
            + (1 - ratio) * rejected_nll_sum / constants.rejected_token_count
        )
        # The logit means carry their own token counts: a microbatch holds no fixed share of them.
        shift_logits = logits[:, :-1, :].detach()
        chosen_tokens, rejected_tokens = loss_mask[0::2], loss_mask[1::2]
        self._accumulate_metrics(
            metric_sums(
                chosen_rewards=chosen_rewards,
                rejected_rewards=rejected_rewards,
                chosen_logps=chosen_mean_logps,
                rejected_logps=rejected_mean_logps,
                pair_valid=pair_valid,
                chosen_logits_sum=(shift_logits[0::2].sum(-1, dtype=torch.float32) * chosen_tokens).sum(),
                rejected_logits_sum=(shift_logits[1::2].sum(-1, dtype=torch.float32) * rejected_tokens).sum(),
                chosen_logit_tokens=chosen_tokens.sum() * shift_logits.size(-1),
                rejected_logit_tokens=rejected_tokens.sum() * shift_logits.size(-1),
                chosen_sft_sum=chosen_nll_sum,
                rejected_sft_sum=rejected_nll_sum,
            )
        )
        return (margin_losses * pair_valid).sum() / constants.pair_count + sft_term

    def predictions(self, logits: torch.Tensor, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-pair ``[pairs, 2]`` rewards, reduced on the last stage instead of sending the ``[rows, S, vocab]`` plane
        down the chain."""
        token_logps, loss_mask = token_logprobs(logits, inputs["labels"])
        chosen_logps, rejected_logps = self._pair_mean_logps(token_logps.masked_fill(~loss_mask, 0), loss_mask)
        _, chosen_rewards, rejected_rewards = self.objective.margin_loss(
            chosen_logps, rejected_logps, self.constants.target_margin
        )
        return torch.stack([chosen_rewards, rejected_rewards], dim=1)

    def step_metrics(self) -> dict[str, torch.Tensor]:
        """The step's metrics from the accumulated sums, then drained; the same names on a stage that ran none."""
        metrics = metrics_from_sums(self.step_sums, self.constants, report_margin=self.report_margin)
        self.step_sums.clear()
        return metrics

    def _pair_mean_logps(
        self, token_logps: torch.Tensor, loss_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Each pair's clipped per-sequence mean log-probs ``(chosen, rejected)``; ``token_logps`` arrive masked."""
        row_is_chosen = torch.arange(token_logps.size(0), device=token_logps.device) % 2 == 0
        chosen_logps, rejected_logps = split_pairs(
            self.objective.clip_log_probs(token_logps, loss_mask, row_is_chosen.unsqueeze(1))
        )
        chosen_mask, rejected_mask = split_pairs(loss_mask)
        return (
            chosen_logps.sum(dim=-1) / chosen_mask.sum(dim=-1).clamp(min=1),
            rejected_logps.sum(dim=-1) / rejected_mask.sum(dim=-1).clamp(min=1),
        )

    def _accumulate_metrics(self, contributions: dict[str, torch.Tensor]) -> None:
        for key, value in contributions.items():
            previous = self.step_sums.get(key)
            self.step_sums[key] = value.detach().float() if previous is None else previous + value.detach().float()
