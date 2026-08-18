#!/usr/bin/env python
"""Offline GRPO's KL term under pipeline parallelism must be the SAME objective as off it.

Under PP the reference is scored once by a sweep through the pipeline and shipped per completion
token as a dataset column; the last-stage loss then re-applies every step of ``compute_loss`` — the
negative-advantage ``min_log_prob`` clamp on both policy and reference, the capped k3 KL, the pg
formulation, group weighting and the per-loss-type denominator. Each seam yields a silently wrong
number, not a crash, so the PP loss is pinned against the real non-PP ``_compute_loss_inner`` on
identical logits, with both clamps genuinely binding.

Run: python tests/cpu/grpo/test_offline_grpo_pp_kl.py
"""

import types
from collections import defaultdict

import pytest
import torch
from accelerate import PartialState

from src.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN, OfflineGRPODataCollatorWithPadding
from src.distributed.pipeline_parallel.losses import token_logprobs
from src.trainers.grpo.objective.logratio import KL_LOGRATIO_CLAMP
from src.trainers.grpo.offline import OfflineGRPOTrainer

PartialState()  # the collator's accelerate logger requires an initialized state

BATCH, PROMPT, COMPLETION, VOCAB, MAX_LENGTH = 4, 3, 4, 16, 10
PAD = 0
MIN_LOG_PROB = -3.0
# Per-row completion lengths (row 2 and 3 are short) and advantages of both signs.
COMPLETION_LENGTHS = [4, 4, 2, 3]
ADVANTAGES = [1.0, -0.5, 0.7, -1.2]
GROUP_SIZES = [2, 2, 3, 3]


class _FixedLogits:
    """A model whose forward returns stored logits, honoring ``logits_to_keep`` like HF does."""

    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def __call__(self, input_ids, attention_mask, logits_to_keep):
        assert input_ids.shape[1] == PROMPT + COMPLETION
        return types.SimpleNamespace(logits=self.logits[:, : PROMPT + COMPLETION][:, -logits_to_keep:])


def _collated_batch() -> dict[str, torch.Tensor]:
    """What the offline collator emits: left-padded prompts, right-padded completions, metadata."""
    torch.manual_seed(0)
    prompt_ids = torch.randint(1, VOCAB, (BATCH, PROMPT))
    prompt_mask = torch.ones(BATCH, PROMPT, dtype=torch.long)
    prompt_ids[1, 0], prompt_mask[1, 0] = PAD, 0
    completion_ids = torch.randint(1, VOCAB, (BATCH, COMPLETION))
    completion_mask = torch.zeros(BATCH, COMPLETION, dtype=torch.long)
    for row, length in enumerate(COMPLETION_LENGTHS):
        completion_mask[row, :length] = 1
        completion_ids[row, length:] = PAD
    return {
        "prompt_input_ids": prompt_ids,
        "prompt_attention_mask": prompt_mask,
        "completion_input_ids": completion_ids,
        "completion_attention_mask": completion_mask,
        "advantage": torch.tensor(ADVANTAGES),
        "group_size": torch.tensor(GROUP_SIZES),
    }


def _logits_pair() -> tuple[torch.Tensor, torch.Tensor]:
    """Policy and reference logits over the fixed shape; the spread makes the -3 floor bind, and row
    0 is built so the reference exceeds the policy by far more than the k3 cap at its labels."""
    torch.manual_seed(1)
    policy = torch.randn(BATCH, MAX_LENGTH, VOCAB) * 3
    reference = torch.randn(BATCH, MAX_LENGTH, VOCAB) * 3
    return policy, reference


def _bind_row0_extremes(policy, reference, completion_ids):
    for j in range(COMPLETION_LENGTHS[0]):
        policy[0, PROMPT - 1 + j, completion_ids[0, j]] = -8.0
        reference[0, PROMPT - 1 + j, completion_ids[0, j]] = 10.0


def _reference_column(reference: torch.Tensor, batch: dict) -> torch.Tensor:
    """The collated ``REF_PER_TOKEN_LOGPS_COLUMN``: one log-prob per completion token, from first
    principles (completion token ``j`` is scored by the logits at input position ``P + j - 1``)."""
    column = torch.zeros(BATCH, COMPLETION)
    logps = torch.log_softmax(reference.float(), dim=-1)
    for row, length in enumerate(COMPLETION_LENGTHS):
        for j in range(length):
            column[row, j] = logps[row, PROMPT - 1 + j, batch["completion_input_ids"][row, j]]
    return column


def _stub(*, beta, loss_type, pg, min_log_prob, policy, reference) -> types.SimpleNamespace:
    me = types.SimpleNamespace(
        beta=beta,
        loss_type=loss_type,
        policy_gradient_formulation=pg,
        min_log_prob=min_log_prob,
        max_completion_length=COMPLETION,
        padding_value=PAD,
        args=types.SimpleNamespace(max_length=MAX_LENGTH),
        model=types.SimpleNamespace(training=True),
        ref_model=_FixedLogits(reference),
        _use_chunked_grpo_logprobs=False,
        _sign_metric_buffer={"train": defaultdict(list), "eval": defaultdict(list)},
        _pp_ref_sweep=None,
        _pp_min_log_prob=min_log_prob,
    )
    for name in (
        "_get_per_token_logps",
        "_compute_loss_inner",
        "_pp_batch_transform",
        "_buffer_sign_metrics",
        "_pp_normalizer",
        "_pp_token_loss",
    ):
        setattr(me, name, types.MethodType(getattr(OfflineGRPOTrainer, name), me))
    me.policy = _FixedLogits(policy)
    return me


def _pp_loss(me, policy, batch) -> torch.Tensor:
    inputs = me._pp_batch_transform(batch)
    target = {key: inputs[key] for key in ("labels", "advantage", "group_size")}
    if me.beta != 0.0:
        target[REF_PER_TOKEN_LOGPS_COLUMN] = inputs[REF_PER_TOKEN_LOGPS_COLUMN]
    return me._pp_token_loss(policy, target) / me._pp_normalizer(inputs)


@pytest.mark.parametrize("loss_type", ["grpo", "bnpo", "dr_grpo"])
@pytest.mark.parametrize("pg", ["reinforce", "prob_weighted"])
@pytest.mark.parametrize("min_log_prob", [None, MIN_LOG_PROB])
def test_pp_kl_loss_matches_compute_loss(loss_type, pg, min_log_prob):
    batch = _collated_batch()
    policy, reference = _logits_pair()
    _bind_row0_extremes(policy, reference, batch["completion_input_ids"])
    batch[REF_PER_TOKEN_LOGPS_COLUMN] = _reference_column(reference, batch)

    # Both clamps are live in this fixture, or the test would pass with either one deleted.
    column = batch[REF_PER_TOKEN_LOGPS_COLUMN]
    policy_logps = _reference_column(policy, batch)
    assert bool(
        (column[0, : COMPLETION_LENGTHS[0]] > policy_logps[0, : COMPLETION_LENGTHS[0]] + KL_LOGRATIO_CLAMP).all()
    )
    if min_log_prob is not None:
        assert bool((policy_logps[1] < min_log_prob).any()), "the floor must bind on a negative-advantage row"

    beta = 0.3
    me = _stub(beta=beta, loss_type=loss_type, pg=pg, min_log_prob=min_log_prob, policy=policy, reference=reference)
    non_pp = me._compute_loss_inner(me.policy, {k: v for k, v in batch.items() if k != REF_PER_TOKEN_LOGPS_COLUMN})
    buffered_off_pp = {key: torch.cat(chunks).clone() for key, chunks in me._sign_metric_buffer["train"].items()}
    me._sign_metric_buffer["train"].clear()
    pp = _pp_loss(me, policy, batch)
    assert torch.allclose(pp, non_pp, atol=1e-5, rtol=1e-5), f"PP {pp.item()} vs non-PP {non_pp.item()}"

    # The same per-sample diagnostics the non-PP loss buffers must reach the buffer under PP too —
    # they are the only thing offline GRPO logs per step, and the last stage is where they exist.
    buffered_under_pp = {key: torch.cat(chunks) for key, chunks in me._sign_metric_buffer["train"].items()}
    assert buffered_under_pp.keys() == buffered_off_pp.keys()
    for key, values in buffered_off_pp.items():
        assert torch.allclose(buffered_under_pp[key], values, atol=1e-5, rtol=1e-5), key

    # The KL term genuinely enters: the kl_beta=0 objective must NOT match.
    me_zero = _stub(
        beta=0.0, loss_type=loss_type, pg=pg, min_log_prob=min_log_prob, policy=policy, reference=reference
    )
    assert not torch.allclose(_pp_loss(me_zero, policy, batch), non_pp, atol=1e-3)


def test_batch_transform_places_reference_on_the_shifted_grid():
    """Completion token ``j`` lands at ``P + j - 1``; every other position is 0."""
    batch = _collated_batch()
    column = torch.arange(1, BATCH * COMPLETION + 1, dtype=torch.float32).view(BATCH, COMPLETION)
    batch[REF_PER_TOKEN_LOGPS_COLUMN] = column
    policy, reference = _logits_pair()
    me = _stub(beta=0.1, loss_type="grpo", pg="reinforce", min_log_prob=None, policy=policy, reference=reference)
    shifted = me._pp_batch_transform(batch)[REF_PER_TOKEN_LOGPS_COLUMN]
    assert shifted.shape == (BATCH, MAX_LENGTH - 1) and shifted.dtype == torch.float32
    assert torch.equal(shifted[:, PROMPT - 1 : PROMPT - 1 + COMPLETION], column)
    assert shifted[:, : PROMPT - 1].eq(0).all() and shifted[:, PROMPT - 1 + COMPLETION :].eq(0).all()
    # And the placement agrees with the grid token_logprobs scores: the mask is exactly the placed span.
    _, mask = token_logprobs(policy, me._pp_batch_transform(batch)["labels"])
    for row, length in enumerate(COMPLETION_LENGTHS):
        assert mask[row].nonzero().flatten().tolist() == list(range(PROMPT - 1, PROMPT - 1 + length))


def test_batch_transform_requires_the_column_outside_the_sweep():
    batch = _collated_batch()
    policy, reference = _logits_pair()
    me = _stub(beta=0.1, loss_type="grpo", pg="reinforce", min_log_prob=None, policy=policy, reference=reference)
    with pytest.raises(RuntimeError, match=REF_PER_TOKEN_LOGPS_COLUMN):
        me._pp_batch_transform(batch)
    # The sweep is the one caller that feeds batches through before the column exists.
    me._pp_ref_sweep = []
    assert REF_PER_TOKEN_LOGPS_COLUMN not in me._pp_batch_transform(batch)
    # kl_beta == 0 never asks for it.
    me_zero = _stub(beta=0.0, loss_type="grpo", pg="reinforce", min_log_prob=None, policy=policy, reference=reference)
    assert REF_PER_TOKEN_LOGPS_COLUMN not in me_zero._pp_batch_transform(batch)


def test_token_loss_stashes_raw_logps_during_the_sweep():
    """The sweep branch scores, does not train: raw (unclamped) per-token log-probs, zero loss."""
    batch = _collated_batch()
    policy, reference = _logits_pair()
    me = _stub(
        beta=0.1, loss_type="grpo", pg="reinforce", min_log_prob=MIN_LOG_PROB, policy=policy, reference=reference
    )
    me._pp_ref_sweep = []
    inputs = me._pp_batch_transform(batch)
    target = {key: inputs[key] for key in ("labels", "advantage", "group_size")}
    loss = me._pp_token_loss(policy, target)
    assert loss.item() == 0.0 and not loss.requires_grad
    expected, _ = token_logprobs(policy, inputs["labels"])
    assert len(me._pp_ref_sweep) == 1 and torch.equal(me._pp_ref_sweep[0], expected)


def test_adapter_ships_the_reference_only_with_a_kl_term():
    policy, reference = _logits_pair()
    with_kl = _stub(beta=0.1, loss_type="grpo", pg="reinforce", min_log_prob=None, policy=policy, reference=reference)
    without = _stub(beta=0.0, loss_type="grpo", pg="reinforce", min_log_prob=None, policy=policy, reference=reference)
    assert OfflineGRPOTrainer._pp_loss_adapter(with_kl).extra_target_keys == (
        "advantage",
        "group_size",
        REF_PER_TOKEN_LOGPS_COLUMN,
    )
    assert OfflineGRPOTrainer._pp_loss_adapter(without).extra_target_keys == ("advantage", "group_size")


def test_collator_carries_reference_logps_aligned_with_completions():
    collator = OfflineGRPODataCollatorWithPadding(pad_token_id=PAD)
    rows = [
        {
            "prompt_input_ids": [5, 6],
            "completion_input_ids": [7, 8, 9],
            "group_id": 0,
            "group_size": 2,
            "advantage": 1.0,
        },
        {"prompt_input_ids": [5], "completion_input_ids": [7], "group_id": 0, "group_size": 2, "advantage": -1.0},
        # An empty completion is substituted by a masked pad token; it carries no reference values.
        {"prompt_input_ids": [5], "completion_input_ids": [], "group_id": 1, "group_size": 1, "advantage": 0.0},
    ]
    assert REF_PER_TOKEN_LOGPS_COLUMN not in collator(rows)

    for row, values in zip(rows, ([-0.1, -0.2, -0.3], [-0.4], []), strict=True):
        row[REF_PER_TOKEN_LOGPS_COLUMN] = values
    out = collator(rows)
    assert out[REF_PER_TOKEN_LOGPS_COLUMN].dtype == torch.float32
    assert torch.equal(
        out[REF_PER_TOKEN_LOGPS_COLUMN], torch.tensor([[-0.1, -0.2, -0.3], [-0.4, 0.0, 0.0], [0.0, 0.0, 0.0]])
    )
    assert out[REF_PER_TOKEN_LOGPS_COLUMN].shape == out["completion_input_ids"].shape

    # A column produced under other caps or another tokenizer is refused, not silently misaligned.
    rows[0][REF_PER_TOKEN_LOGPS_COLUMN] = [-0.1, -0.2]
    with pytest.raises(ValueError, match="one per completion token"):
        collator(rows)


def test_the_pp_ctor_gate_accepts_a_kl_term():
    """The construction gate no longer refuses ``kl_beta != 0`` under PP."""
    trainer = types.SimpleNamespace(max_prompt_length=PROMPT, max_completion_length=COMPLETION)
    args = types.SimpleNamespace(kl_beta=0.1, use_chunked_grpo_logprobs=False, max_length=None)
    OfflineGRPOTrainer._reject_pp_explicit_options(trainer, args, types.SimpleNamespace(is_pp_mode=True), None, None)
    assert args.max_length == PROMPT + COMPLETION


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
