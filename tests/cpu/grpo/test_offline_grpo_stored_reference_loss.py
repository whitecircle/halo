"""Stored run-start scores and the scheduled floor determine the actual trainer loss."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from src.trainers.grpo.offline import OfflineGRPOTrainer


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = nn.Parameter(torch.arange(96, dtype=torch.float32).reshape(3, 4, 8).remainder(11) / 2)

    def forward(self, input_ids, attention_mask, logits_to_keep):
        return SimpleNamespace(logits=self.scores[:, -logits_to_keep:])


def _batch():
    return {
        "prompt_input_ids": torch.tensor([[1, 2], [3, 4], [0, 5]]),
        "prompt_attention_mask": torch.tensor([[1, 1], [1, 1], [0, 1]]),
        "completion_input_ids": torch.tensor([[2, 4, 6], [1, 3, 0], [5, 0, 0]]),
        "completion_attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]]),
        "advantage": torch.tensor([1.5, -0.8, -1.3]),
        "group_size": torch.tensor([2, 2, 3]),
        REF_PER_TOKEN_LOGPS_COLUMN: torch.tensor([[-7.2, -0.7, -5.8], [-8.4, -4.2, 0.0], [-6.7, 0.0, 0.0]]),
    }


def _oracle(scores, batch, *, floor, loss_type, formulation):
    target = batch["completion_input_ids"]
    policy = scores[:, :-1].log_softmax(-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    negative = batch["advantage"].unsqueeze(1) < 0
    reference = batch[REF_PER_TOKEN_LOGPS_COLUMN]
    if floor is not None:
        policy = torch.where(negative, policy.clamp(min=floor), policy)
        reference = torch.where(negative, reference.clamp(min=floor), reference)
    delta = torch.minimum(reference, policy.detach() + 5.0) - policy
    weights = policy.exp() if formulation == "prob_weighted" else policy
    tokens = -weights * batch["advantage"].unsqueeze(1) + 0.2 * (delta.exp() - delta - 1)
    mask = batch["completion_attention_mask"]
    group_weights = batch["group_size"].float().reciprocal()
    rows = (tokens * mask).sum(1) * group_weights
    counts = mask.sum(1).float()
    if loss_type == "grpo":
        return (rows / counts.clamp(min=1)).sum() / group_weights.sum()
    if loss_type == "bnpo":
        return rows.sum() / (counts * group_weights).sum().clamp(min=1)
    return rows.sum() / (group_weights.sum() * 4)


@pytest.mark.parametrize("loss_type", ["grpo", "bnpo", "dr_grpo"])
@pytest.mark.parametrize("formulation", ["reinforce", "prob_weighted"])
@pytest.mark.parametrize("floor", [-1.0, -3.0, -6.0, None])
def test_loss_consumes_stored_raw_reference_at_the_live_scheduled_floor(loss_type, formulation, floor):
    trainer = OfflineGRPOTrainer.__new__(OfflineGRPOTrainer)
    trainer.model = _Policy()
    trainer.model.min_log_prob = floor
    trainer.min_log_prob = -2.5
    trainer.parallelism_config = SimpleNamespace(is_cp_mode=False)
    trainer._use_chunked_grpo_logprobs = False
    trainer._precompute_reference = True
    trainer.ref_model = None
    trainer.beta = 0.2
    trainer.loss_type = loss_type
    trainer.policy_gradient_formulation = formulation
    trainer.max_completion_length = 4
    trainer._sign_metric_buffer = {"train": defaultdict(list), "eval": defaultdict(list)}
    batch = _batch()
    raw_reference = batch[REF_PER_TOKEN_LOGPS_COLUMN].clone()
    oracle_scores = trainer.model.scores.detach().clone().requires_grad_()
    expected = _oracle(oracle_scores, batch, floor=floor, loss_type=loss_type, formulation=formulation)
    actual = trainer._compute_loss_inner(trainer.model, batch)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(trainer.model.scores.grad, oracle_scores.grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(batch[REF_PER_TOKEN_LOGPS_COLUMN], raw_reference, rtol=0, atol=0)
    assert not batch[REF_PER_TOKEN_LOGPS_COLUMN].requires_grad


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
