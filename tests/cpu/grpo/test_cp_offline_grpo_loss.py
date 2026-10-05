"""The CP trainer consumes right-padded labels and raw checkpointed KL scores."""

import datetime
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch import nn

from src.data.collators.offline_grpo import (
    REF_PER_TOKEN_LOGPS_COLUMN,
    OfflineGRPOCPDataCollatorWithPadding,
)
from src.data.spans import LABEL_IGNORE_INDEX
from src.trainers.grpo.objective.logratio import KL_LOGRATIO_CLAMP
from src.trainers.grpo.offline import OfflineGRPOTrainer
from tests.common.gloo import run_gloo_ranks

_KL_SCORES = [[-2.0, -2.3, -1.8, -2.2, -2.6, -1.7, -3.2], [-1.2, -2.7, -3.1, -1.9, -2.8, -1.5, -2.4]]
# CP2 loss and full gradient over _KL_SCORES (reinforce, kl_beta 0.2), pinned bit for bit: integer
# token counts are exact in fp32 wherever they come from.
_CP2_FROZEN = {
    "grpo": (
        1.4572901725769043,
        [
            [
                -0.09460652619600296,
                -0.17928069829940796,
                -0.09499384462833405,
                -0.17928069829940796,
                -0.10344026982784271,
                -0.11641031503677368,
                -0.0957413837313652,
            ],
            [0.0, 0.0, 0.0, 0.0, -0.01850668340921402, 0.11364273726940155, -0.04843444377183914],
        ],
    ),
    "bnpo": (
        2.3402340412139893,
        [
            [
                -0.13244913518428802,
                -0.25099295377731323,
                -0.13299137353897095,
                -0.25099295377731323,
                -0.14481636881828308,
                -0.16297443211078644,
                -0.13403794169425964,
            ],
            [0.0, 0.0, 0.0, 0.0, -0.011104006320238113, 0.06818564236164093, -0.029060665518045425],
        ],
    ),
    "dr_grpo": (
        1.6715956926345825,
        [
            [
                -0.09460652619600296,
                -0.17928069829940796,
                -0.09499384462833405,
                -0.17928069829940796,
                -0.10344026982784271,
                -0.11641031503677368,
                -0.0957413837313652,
            ],
            [0.0, 0.0, 0.0, 0.0, -0.007931429892778397, 0.048704031854867935, -0.02075761929154396],
        ],
    ),
}


def _inputs(with_reference: bool):
    rows = [
        {
            "prompt_input_ids": [11, 12],
            "completion_input_ids": [21, 22],
            "group_id": 0,
            "group_size": 2,
            "advantage": 1.5,
        },
        {
            "prompt_input_ids": [13],
            "completion_input_ids": [23],
            "group_id": 0,
            "group_size": 2,
            "advantage": -0.5,
        },
    ]
    if with_reference:
        rows[0][REF_PER_TOKEN_LOGPS_COLUMN] = [-0.8, -0.7]
        rows[1][REF_PER_TOKEN_LOGPS_COLUMN] = [-0.6]
    return OfflineGRPOCPDataCollatorWithPadding(pad_token_id=0, cp_size=1)(rows)


def _trainer(logps, *, beta: float, loss_type: str):
    trainer = OfflineGRPOTrainer.__new__(OfflineGRPOTrainer)
    trainer.parallelism_config = SimpleNamespace(is_cp_mode=True)
    trainer.cp_config = SimpleNamespace(cp_size=1, cp_rank=0, process_group=None)
    trainer.model = nn.Linear(1, 1)
    trainer.min_log_prob = None
    trainer.beta = beta
    trainer.loss_type = loss_type
    trainer.max_completion_length = 4
    trainer.policy_gradient_formulation = "reinforce"
    trainer._sign_metric_buffer = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer._cp_chunked_logps = lambda model, ids, mask, labels: (logps, labels[:, 1:])
    return trainer


@pytest.mark.parametrize("loss_type", ["grpo", "bnpo", "dr_grpo"])
@pytest.mark.parametrize("beta", [0.0, 0.2])
def test_cp_dispatch_loss_and_gradient_use_only_completion_targets(loss_type, beta):
    batch = _inputs(with_reference=beta != 0.0)
    logps = torch.tensor([[-0.3, -0.4, -9.0], [-0.5, -8.0, -8.0]], requires_grad=True)
    trainer = _trainer(logps, beta=beta, loss_type=loss_type)
    loss = trainer._compute_loss_inner(trainer.model, batch)

    valid = batch["labels"][:, 1:] != -100
    token_loss = -logps * batch["advantage"].unsqueeze(1)
    if beta:
        ref = batch[REF_PER_TOKEN_LOGPS_COLUMN][:, 1:]
        # The shared offline objective caps the reference-policy log-ratio. The
        # detached ceiling preserves the policy gradient on capped tokens.
        delta = torch.minimum(ref, logps.detach() + KL_LOGRATIO_CLAMP) - logps
        token_loss = token_loss + beta * (delta.exp() - delta - 1)
    weights = batch["group_size"].float().reciprocal()
    numerators = (token_loss * valid).sum(1) * weights
    counts = valid.sum(1).float()
    if loss_type == "grpo":
        expected = (numerators / counts.clamp(min=1)).sum() / weights.sum()
    elif loss_type == "bnpo":
        expected = numerators.sum() / (counts * weights).sum().clamp(min=1)
    else:
        expected = numerators.sum() / (weights.sum() * trainer.max_completion_length)
    torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-6)
    actual_grad = torch.autograd.grad(loss, logps, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logps)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-6)
    assert torch.all(actual_grad[~valid] == 0)


def test_kl_cannot_run_without_original_reference_column():
    batch = _inputs(with_reference=False)
    logps = torch.zeros_like(batch["labels"][:, 1:], dtype=torch.float32, requires_grad=True)
    trainer = _trainer(logps, beta=0.2, loss_type="grpo")
    with pytest.raises(RuntimeError, match="run-start raw reference"):
        trainer._compute_cp_loss_inner(trainer.model, batch)


def _kl_rows(cp_size: int) -> dict[str, torch.Tensor]:
    rows = [
        {
            "prompt_input_ids": [1],
            "completion_input_ids": [2, 3, 4, 5, 6, 7, 8],
            "group_id": 0,
            "group_size": 2,
            "advantage": 1.5,
            REF_PER_TOKEN_LOGPS_COLUMN: [-4.1, -0.5, -3.7, -0.4, -2.9, -1.2, -4.8],
        },
        {
            "prompt_input_ids": [9, 10, 11, 12, 13],
            "completion_input_ids": [14, 15, 16],
            "group_id": 0,
            "group_size": 2,
            "advantage": -0.5,
            REF_PER_TOKEN_LOGPS_COLUMN: [-1.4, -3.9, -0.8],
        },
    ]
    return OfflineGRPOCPDataCollatorWithPadding(pad_token_id=0, cp_size=cp_size)(rows)


def _kl_oracle(batch: dict, logps: torch.Tensor, loss_type: str, formulation: str) -> torch.Tensor:
    valid = batch["labels"][:, 1:] != LABEL_IGNORE_INDEX
    policy = logps if formulation == "reinforce" else logps.exp()
    token_loss = -policy * batch["advantage"].unsqueeze(1)
    reference = batch[REF_PER_TOKEN_LOGPS_COLUMN][:, 1:]
    delta = torch.minimum(reference, logps.detach() + KL_LOGRATIO_CLAMP) - logps
    token_loss += 0.2 * (delta.exp() - delta - 1)
    weights = batch["group_size"].float().reciprocal()
    numerators = (token_loss * valid).sum(dim=1) * weights
    counts = valid.sum(dim=1).float()
    if loss_type == "grpo":
        return (numerators / counts.clamp(min=1)).sum() / weights.sum()
    if loss_type == "bnpo":
        return numerators.sum() / (counts * weights).sum().clamp(min=1)
    return numerators.sum() / (weights.sum() * 7)


def _ranked_kl_oracle(rank: int, cp_size: int) -> None:
    batch = _kl_rows(cp_size)
    scores = torch.tensor(_KL_SCORES)
    chunk = batch["labels"].size(1) // cp_size
    start, end = rank * chunk, min((rank + 1) * chunk, scores.size(1))
    trainer = _trainer(scores, beta=0.2, loss_type="grpo")
    trainer.cp_config = SimpleNamespace(cp_size=cp_size, cp_rank=rank, process_group=dist.group.WORLD)
    trainer.max_completion_length = 7
    for loss_type in ("grpo", "bnpo", "dr_grpo"):
        for formulation in ("reinforce", "prob_weighted"):
            policy = scores.clone().requires_grad_()
            trainer.loss_type = loss_type
            trainer.policy_gradient_formulation = formulation
            trainer._cp_chunked_logps = lambda model, ids, mask, labels: (
                policy[:, start:end],
                labels[:, start + 1 : end + 1],
            )
            loss = trainer._compute_loss_inner(trainer.model, batch)
            oracle_policy = scores.clone().requires_grad_()
            expected = _kl_oracle(batch, oracle_policy, loss_type, formulation)
            torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-6)
            # Each CP sibling evaluates the same logical loss; the trainer's gradient sync
            # averages the replicas after the autograd-aware sum's backward collective.
            (loss / cp_size).backward()
            dist.all_reduce(policy.grad)
            expected.backward()
            torch.testing.assert_close(policy.grad, oracle_policy.grad, atol=1e-6, rtol=1e-6)

            # Removing cp_rank from reference slicing means every shard reads rank 0's columns.
            # Keep scorer ownership correct while changing only the slice's rank.
            trainer.cp_config.cp_rank = 0
            wrong_loss = trainer._compute_cp_loss_inner(trainer.model, batch)
            trainer.cp_config.cp_rank = rank
            assert abs(float(wrong_loss.detach() - expected.detach())) > 1e-4


@pytest.mark.parametrize("cp_size", [2, 4])
def test_cp_kl_reference_rank_slice_matches_full_loss_and_gradient_and_rejects_mutation(cp_size):
    run_gloo_ranks(_ranked_kl_oracle, cp_size, cp_size, pg_timeout=datetime.timedelta(seconds=30))


def _cp_sum_count_worker(rank: int) -> None:
    cp_size = 2
    batch = _kl_rows(cp_size)
    scores = torch.tensor(_KL_SCORES)
    chunk = batch["labels"].size(1) // cp_size
    start, end = rank * chunk, min((rank + 1) * chunk, scores.size(1))
    trainer = _trainer(scores, beta=0.2, loss_type="grpo")
    trainer.cp_config = SimpleNamespace(cp_size=cp_size, cp_rank=rank, process_group=dist.group.WORLD)
    trainer.max_completion_length = 7
    for loss_type, (frozen_loss, frozen_grad) in _CP2_FROZEN.items():
        policy = scores.clone().requires_grad_()
        trainer.loss_type = loss_type
        trainer._cp_chunked_logps = lambda model, ids, mask, labels, policy=policy: (
            policy[:, start:end],
            labels[:, start + 1 : end + 1],
        )
        with patch.object(dist_nn, "all_reduce", wraps=dist_nn.all_reduce) as cp_sums:
            loss = trainer._compute_loss_inner(trainer.model, batch)
        # The loss numerator and the sign diagnostics; the row token counts come from the labels.
        assert cp_sums.call_count == 2, (loss_type, cp_sums.call_count)
        (loss / cp_size).backward()
        dist.all_reduce(policy.grad)
        assert torch.equal(loss.detach(), torch.tensor(frozen_loss)), (loss_type, loss.item())
        assert torch.equal(policy.grad, torch.tensor(frozen_grad)), (loss_type, policy.grad.tolist())


def test_cp2_loss_reduces_twice_per_microbatch_and_keeps_its_exact_values():
    run_gloo_ranks(_cp_sum_count_worker, 2, pg_timeout=datetime.timedelta(seconds=30))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
