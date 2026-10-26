#!/usr/bin/env python
"""``DistributedSFTTrainer``'s CP metrics fold ONCE per log, not once per micro-batch.

Under context parallelism the trainer computes its own mean_token_accuracy / entropy / aux_loss /
num_attended_tokens_seen. Each micro-batch records sums into a ``WorldMetrics`` window that one flush per
log folds across the world — fold-of-sums equals sum-of-folds, which is what these tests prove against a
fake 2-rank group. Gathering and ``.item()``-ing each quantity per micro-batch instead puts five to six
small collectives and host stalls per micro-batch on the critical path of every optimizer step, which at
512 GPUs is hundreds per step.

Run: python tests/cpu/trainers/test_cp_metric_batching.py
"""

import types
from collections import defaultdict
from unittest import mock

import pytest
import torch
from transformers import Trainer
from trl.trainer.utils import entropy_from_logits

from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.trainers import world_metrics
from halo.trainers.sft import DistributedSFTTrainer
from halo.trainers.world_metrics import WorldMetrics
from tests.common.world_metrics import fake_world

CP_SIZE = 2
VOCAB = 32
SEQ = 8


def _rank(cp_rank, *, training=True, is_cp_mode=True):
    me = types.SimpleNamespace(
        model=types.SimpleNamespace(training=training),
        cp_config=types.SimpleNamespace(cp_rank=cp_rank),
        parallelism_config=types.SimpleNamespace(cp_size=CP_SIZE, is_cp_mode=is_cp_mode),
        _cp_metrics={"train": WorldMetrics(), "eval": WorldMetrics()},
        _total_train_tokens=0,
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
    )
    me._pad_to_cp_multiple = types.MethodType(DistributedSFTTrainer._pad_to_cp_multiple, me)
    me._drain_cp_metrics = types.MethodType(DistributedSFTTrainer._drain_cp_metrics, me)
    return me


def _no_collective():
    return mock.patch.object(world_metrics, "_gathered", side_effect=AssertionError("a collective ran"))


def _accumulate(rank, outputs, full_labels, full_attention_mask):
    """One micro-batch through the trainer's real CP loss path, the wrapped forward replaced by ``outputs``."""
    inputs = {"labels": full_labels, "attention_mask": full_attention_mask}
    with mock.patch.object(Trainer, "compute_loss", return_value=(torch.zeros(()), outputs)), _no_collective():
        DistributedSFTTrainer._compute_cp_loss(rank, None, inputs)


def _microbatch(seed, rows=2, aux_loss=None, attended=SEQ):
    """(outputs, full_labels, full_attention_mask) for one micro-batch of a CP step.

    ``attended`` is how many of the SEQ positions are attended, so a caller can make the
    micro-batches carry UNEQUAL token counts.
    """
    generator = torch.Generator().manual_seed(seed)
    chunk = SEQ // CP_SIZE
    outputs = types.SimpleNamespace(logits=torch.randn(rows, chunk, VOCAB, generator=generator))
    if aux_loss is not None:
        outputs.aux_loss = torch.tensor(aux_loss)
    labels = torch.randint(0, VOCAB, (rows, SEQ), generator=generator)
    mask = torch.zeros(rows, SEQ, dtype=torch.long)
    mask[:, :attended] = 1
    return outputs, labels, mask


def _drain_in_lockstep(ranks):
    """Every rank's drain, each flush gathering every rank's window as the world's."""
    with fake_world(*(rank._cp_metrics["train" if rank.model.training else "eval"] for rank in ranks)):
        for rank in ranks:
            rank._drain_cp_metrics()


def _run(ranks, batches, *, drain_every_microbatch):
    """Accumulate ``batches`` on both ranks, draining once at the end or after every micro-batch.

    Draining per micro-batch is the unbatched formulation (one fold per micro-batch, its ratio appended
    for TRL to average); draining once is the shipped behaviour. Both go through the SAME code, so the
    comparison cannot drift.
    """
    for batch in batches:
        for rank in ranks:
            _accumulate(rank, *batch)
        if drain_every_microbatch:
            _drain_in_lockstep(ranks)
    if not drain_every_microbatch:
        _drain_in_lockstep(ranks)


def _metric(rank, key):
    return rank._metrics["train"][key]


def test_accumulation_runs_no_collective_and_emits_nothing():
    """The whole point: the micro-batch path must not reach the wire."""
    rank = _rank(0)
    for seed in range(4):
        _accumulate(rank, *_microbatch(seed, aux_loss=0.25))
    assert rank._cp_metrics["train"], "nothing was recorded"
    assert not rank._metrics["train"], "accumulation emitted a metric, which means it reduced"


def test_token_counter_is_exactly_fold_of_sums_equals_sum_of_folds():
    """``num_attended_tokens_seen`` is a pure sum, so batching it is an identity — asserted EXACTLY,
    across a fake 2-rank group, not to a tolerance."""
    batches = [_microbatch(seed) for seed in range(5)]

    batched = [_rank(0), _rank(1)]
    _run(batched, batches, drain_every_microbatch=False)
    per_microbatch = [_rank(0), _rank(1)]
    _run(per_microbatch, batches, drain_every_microbatch=True)

    assert batched[0]._total_train_tokens == per_microbatch[0]._total_train_tokens == 2 * 5 * 2 * (SEQ // CP_SIZE)
    assert _metric(batched[0], "num_attended_tokens_seen") == _metric(per_microbatch[0], "num_attended_tokens_seen")
    # Both ranks of the group agree — the fold is global, not rank-local.
    assert batched[0]._total_train_tokens == batched[1]._total_train_tokens


def test_ratio_metrics_match_the_per_microbatch_formulation_on_equal_token_counts():
    """Accuracy and entropy become token-weighted over the log window instead of an unweighted mean
    of per-micro-batch ratios. The two coincide exactly when the micro-batches carry equal token
    counts — which is the CP case, since CP pads every sequence to a multiple of cp_size.
    """
    batches = [_microbatch(seed, aux_loss=0.1 * seed) for seed in range(4)]

    batched = [_rank(0), _rank(1)]
    _run(batched, batches, drain_every_microbatch=False)
    per_microbatch = [_rank(0), _rank(1)]
    _run(per_microbatch, batches, drain_every_microbatch=True)

    for key in ("mean_token_accuracy", "entropy", "aux_loss"):
        new = _metric(batched[0], key)
        old = _metric(per_microbatch[0], key)
        assert len(new) == 1 and len(old) == len(batches)
        # TRL's log() averages the list; that average is what actually reaches the run's logs.
        assert new[0] == pytest.approx(sum(old) / len(old), rel=1e-12), key


def test_ratio_metrics_are_token_weighted_when_microbatches_differ_in_length():
    """With unequal micro-batches the batched value is the window's token-weighted mean, which sits
    between the per-micro-batch ratios rather than tracking either end."""
    batches = [_microbatch(0, attended=SEQ), _microbatch(1, attended=SEQ // 4)]

    batched = [_rank(0), _rank(1)]
    _run(batched, batches, drain_every_microbatch=False)
    per_microbatch = [_rank(0), _rank(1)]
    _run(per_microbatch, batches, drain_every_microbatch=True)

    entropies = _metric(per_microbatch[0], "entropy")
    assert min(entropies) <= _metric(batched[0], "entropy")[0] <= max(entropies)
    # The token counter is a sum, so it stays exact even here.
    assert batched[0]._total_train_tokens == per_microbatch[0]._total_train_tokens


def test_drain_is_one_collective_per_log_on_every_rank_however_many_microbatches():
    """Participation is structural: a rank whose window is empty (its first log, or a step whose logits
    Liger skipped) enters the flush too, never skips it — one rank sitting out is a hang for the other
    511 — and a full window costs the same one collective."""
    full, empty = _rank(0), _rank(1)
    for seed in range(8):
        _accumulate(full, *_microbatch(seed, aux_loss=0.3))
    for rank in (full, empty):
        with mock.patch.object(world_metrics, "_gathered", side_effect=lambda payload, group: [payload]) as gather:
            rank._drain_cp_metrics()
        assert gather.call_count == 1


def test_records_carry_the_quantities_their_names_claim():
    """Independent recomputation, not another pass through the same code.

    Every other test here compares the batched path against a per-micro-batch drain of the SAME
    implementation, so an error shared by both is invisible to them — swapping the attended tokens
    (attention mask, padding included) for the loss-token count is exactly the confusion the name
    ``num_attended_tokens_seen`` guards against, and it survives that comparison.
    """
    chunk = SEQ // CP_SIZE
    # Attended < chunk, so the attention mask and the loss mask cover DIFFERENT token counts —
    # without that, swapping the two is invisible.
    outputs, labels, mask = _microbatch(0, rows=2, attended=chunk // 2)
    # Row 0's local logits predict their next tokens 0 and 3 and skip two ignored ones: accuracy is over the
    # loss tokens alone, and nonzero.
    labels[0, 2:4] = LABEL_IGNORE_INDEX
    for position in (0, 3):
        outputs.logits[0, position, labels[0, position + 1]] += 100.0
    rank = _rank(0)
    _accumulate(rank, outputs, labels, mask)
    rank._drain_cp_metrics()

    expected_attended = int(mask[:, :chunk].sum())
    loss_tokens = int((labels[:, :chunk] != LABEL_IGNORE_INDEX).sum())
    assert expected_attended != loss_tokens, "fixture does not separate attended tokens from loss tokens"
    assert _metric(rank, "num_attended_tokens_seen") == [expected_attended]

    expected_entropy = ((entropy_from_logits(outputs.logits) * mask[:, :chunk]).sum() / mask[:, :chunk].sum()).item()
    assert _metric(rank, "entropy")[0] == pytest.approx(expected_entropy, rel=1e-6)

    # Rank 0 is not the last rank: every local logit is scored, its last one against chunk 1's first token.
    shift_logits, shift_labels = outputs.logits, labels[:, 1 : chunk + 1]
    loss_mask = shift_labels != LABEL_IGNORE_INDEX
    expected_accuracy = (((shift_logits.argmax(-1) == shift_labels) & loss_mask).sum() / loss_mask.sum()).item()
    assert _metric(rank, "mean_token_accuracy")[0] == pytest.approx(expected_accuracy, rel=1e-6)
    assert "aux_loss" not in rank._metrics["train"], "a forward without an aux loss logs none"


def test_empty_window_emits_nothing_rather_than_a_zero():
    """A run's final ``log()`` fires after the window was already drained. Appending a 0.0 accuracy
    there puts a bogus last point on every CP run's curves; the emit is skipped on the FLUSHED
    result, so every rank skips together.
    """
    ranks = [_rank(0), _rank(1)]
    _run(ranks, [_microbatch(0)], drain_every_microbatch=False)
    after_real = {key: list(values) for key, values in ranks[0]._metrics["train"].items()}
    assert after_real["mean_token_accuracy"]

    _drain_in_lockstep(ranks)

    assert {key: list(values) for key, values in ranks[0]._metrics["train"].items()} == after_real


def test_drain_is_a_noop_without_cp():
    """Non-CP runs take TRL's own metric path; a flush here would be a collective no other trainer
    in the job is making."""
    rank = _rank(0, is_cp_mode=False)
    with _no_collective():
        rank._drain_cp_metrics()
    assert not rank._metrics["train"]


def test_train_and_eval_windows_do_not_contaminate_each_other():
    """``_metrics`` is keyed by mode and so is the accumulator; eval must not add to the run's
    train-token counter."""
    rank = _rank(0)
    _accumulate(rank, *_microbatch(0))
    rank.model.training = False
    _accumulate(rank, *_microbatch(1))

    # Eval's attended tokens are a structural zero, so draining eval leaves the counter alone.
    _drain_in_lockstep([rank])
    assert rank._total_train_tokens == 0
    assert rank._metrics["eval"]["mean_token_accuracy"] and not rank._metrics["train"]
    assert rank._cp_metrics["train"], "the train window waits for its own log"


def test_accumulator_resets_after_a_drain():
    """A window must not be counted twice: the next log folds only what came after this one."""
    ranks = [_rank(0), _rank(1)]
    _run(ranks, [_microbatch(0)], drain_every_microbatch=False)
    first = ranks[0]._total_train_tokens
    assert not ranks[0]._cp_metrics["train"]

    _drain_in_lockstep(ranks)
    assert ranks[0]._total_train_tokens == first


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
