#!/usr/bin/env python
"""``DistributedSFTTrainer``'s CP metrics reduce ONCE per log, not once per micro-batch.

Under context parallelism the trainer computes its own mean_token_accuracy / entropy / aux_loss /
num_attended_tokens_seen. Every quantity is a plain SUM, so the micro-batches fold into one
fixed-width on-device row that is reduced once per log — reduce-of-sums equals sum-of-reduces, which
is what these tests prove against a fake 2-rank group. Gathering and ``.item()``-ing each quantity
per micro-batch instead puts five to six small collectives and host stalls per micro-batch on the
critical path of every optimizer step, which at 512 GPUs is hundreds per step.

Run: python tests/cpu/trainers/test_cp_metric_batching.py
"""

import types
from collections import defaultdict

import pytest
import torch
from trl.trainer.utils import entropy_from_logits

from src.data.spans import LABEL_IGNORE_INDEX
from src.distributed.context_parallel.config import cp_boundary_shift
from src.trainers import sft as sft_trainer
from src.trainers.sft import _CP_METRIC_COLUMNS, _CP_METRIC_DTYPE, DistributedSFTTrainer

_accumulate = DistributedSFTTrainer._compute_cp_metrics
_drain = DistributedSFTTrainer._drain_cp_metrics

CP_SIZE = 2
VOCAB = 32
SEQ = 8


class _NoCollective:
    """An accelerator whose reduce raises — proves accumulation never touches the wire."""

    device = torch.device("cpu")

    @staticmethod
    def reduce(tensor, reduction="sum"):
        raise AssertionError("a collective ran during per-micro-batch accumulation")


def _rank(cp_rank, *, training=True, is_cp_mode=True, accelerator=None):
    me = types.SimpleNamespace(
        model=types.SimpleNamespace(training=training),
        cp_config=types.SimpleNamespace(cp_rank=cp_rank),
        cp_size=CP_SIZE,
        is_cp_mode=is_cp_mode,
        accelerator=accelerator or _NoCollective(),
        _cp_metric_accum={},
        _total_train_tokens=0,
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
    )
    me._compute_cp_metrics = types.MethodType(_accumulate, me)
    me._drain_cp_metrics = types.MethodType(_drain, me)
    return me


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


def _run(ranks, batches, *, drain_every_microbatch):
    """Accumulate ``batches`` on both ranks, draining once at the end or after every micro-batch.

    Draining per micro-batch reproduces exactly what the pre-batching code did (one reduce per
    quantity per micro-batch, its ratio appended for TRL to average); draining once is the shipped
    behaviour. Both go through the SAME code, so the comparison cannot drift.
    """
    for batch in batches:
        for rank in ranks:
            rank._compute_cp_metrics(*batch)
        if drain_every_microbatch:
            _drain_in_lockstep(ranks)
    if not drain_every_microbatch:
        _drain_in_lockstep(ranks)


def _drain_in_lockstep(ranks):
    """Give every rank a reduce that adds its PEERS' rows to the one it submits itself.

    A rank that skipped its contribution, or entered with the wrong shape, changes the answer — the
    fake group does not paper over either.
    """
    width = len(_CP_METRIC_COLUMNS)
    rows = []
    for rank in ranks:
        row = rank._cp_metric_accum.get("train" if rank.model.training else "eval")
        rows.append(torch.zeros(width, dtype=_CP_METRIC_DTYPE) if row is None else row)
    for index, rank in enumerate(ranks):
        peers = sum(row for position, row in enumerate(rows) if position != index)
        rank.accelerator = types.SimpleNamespace(
            device=torch.device("cpu"),
            reduce=lambda tensor, reduction="sum", _peers=peers: tensor + _peers,
        )
        rank._drain_cp_metrics()


def _metric(rank, key):
    return rank._metrics["train"][key]


def test_accumulation_runs_no_collective_and_no_host_sync():
    """The whole point: the micro-batch path must not reach the wire or the host."""
    rank = _rank(0)
    for seed in range(4):
        rank._compute_cp_metrics(*_microbatch(seed, aux_loss=0.25))
    assert rank._cp_metric_accum["train"].shape == (len(_CP_METRIC_COLUMNS),)
    assert not rank._metrics["train"], "accumulation emitted a metric, which means it reduced"


def test_buffer_width_is_fixed_however_many_microbatches_folded_in():
    """The reduce's shape must not depend on rank-local work: mismatched shapes across 512 ranks
    is a hang, not an error."""
    few, many = _rank(0), _rank(1)
    for seed in range(2):
        few._compute_cp_metrics(*_microbatch(seed))
    for seed in range(9):
        many._compute_cp_metrics(*_microbatch(seed))
    assert few._cp_metric_accum["train"].shape == many._cp_metric_accum["train"].shape


def test_token_counter_is_exactly_reduce_of_sums_equals_sum_of_reduces():
    """``num_attended_tokens_seen`` is a pure sum, so batching it is an identity — asserted EXACTLY,
    across a fake 2-rank group, not to a tolerance."""
    batches = [_microbatch(seed) for seed in range(5)]

    batched = [_rank(0), _rank(1)]
    _run(batched, batches, drain_every_microbatch=False)
    per_microbatch = [_rank(0), _rank(1)]
    _run(per_microbatch, batches, drain_every_microbatch=True)

    assert batched[0]._total_train_tokens == per_microbatch[0]._total_train_tokens
    assert _metric(batched[0], "num_attended_tokens_seen") == _metric(per_microbatch[0], "num_attended_tokens_seen")
    # Both ranks of the group agree — the reduce is global, not rank-local.
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


def test_drain_reduces_on_a_rank_that_accumulated_nothing():
    """Participation is structural. A rank whose window is empty (its first log, or a step whose
    logits Liger skipped) must enter the reduce with zeros, never skip it — one rank sitting out is
    a hang for the other 511.
    """
    reduced = []
    empty = _rank(0)
    empty.accelerator = types.SimpleNamespace(
        device=torch.device("cpu"),
        reduce=lambda tensor, reduction="sum": reduced.append(tensor.clone()) or tensor,
    )
    empty._drain_cp_metrics()

    assert len(reduced) == 1
    assert reduced[0].shape == (len(_CP_METRIC_COLUMNS),)
    assert torch.count_nonzero(reduced[0]) == 0


def test_drain_is_one_collective_per_log_however_many_microbatches():
    """One reduce per log window, whatever the micro-batch count — what the fixed-width row buys."""
    calls = []
    rank = _rank(0)
    for seed in range(8):
        rank._compute_cp_metrics(*_microbatch(seed, aux_loss=0.3))
    rank.accelerator = types.SimpleNamespace(
        device=torch.device("cpu"),
        reduce=lambda tensor, reduction="sum": calls.append(reduction) or tensor,
    )
    rank._drain_cp_metrics()
    assert calls == ["sum"]


def test_columns_carry_the_quantities_their_names_claim():
    """Independent recomputation, not another pass through the same code.

    Every other test here compares the batched path against a per-micro-batch drain of the SAME
    implementation, so an error shared by both is invisible to them — swapping ``attended_tokens``
    (attention mask, padding included) for the loss-token count is exactly the confusion
    ``num_attended_tokens_seen`` was renamed to prevent, and it survives that comparison.
    """
    chunk = SEQ // CP_SIZE
    # Attended < chunk, so the attention mask and the loss mask cover DIFFERENT token counts —
    # without that, swapping the two columns is invisible.
    outputs, labels, mask = _microbatch(0, rows=2, attended=chunk // 2)
    rank = _rank(0, accelerator=types.SimpleNamespace(device=torch.device("cpu"), reduce=lambda t, reduction="sum": t))
    rank._compute_cp_metrics(outputs, labels, mask)
    rank._drain_cp_metrics()

    expected_attended = int(mask[:, :chunk].sum())
    loss_tokens = int((labels[:, :chunk] != LABEL_IGNORE_INDEX).sum())
    assert expected_attended != loss_tokens, "fixture does not separate attended tokens from loss tokens"
    assert _metric(rank, "num_attended_tokens_seen") == [expected_attended]

    expected_entropy = ((entropy_from_logits(outputs.logits) * mask[:, :chunk]).sum() / mask[:, :chunk].sum()).item()
    assert _metric(rank, "entropy")[0] == pytest.approx(expected_entropy, rel=1e-6)

    shift_logits, shift_labels = cp_boundary_shift(
        outputs.logits, labels[:, :chunk], labels[:, chunk : chunk + 1], False
    )
    loss_mask = shift_labels != LABEL_IGNORE_INDEX
    expected_accuracy = (((shift_logits.argmax(-1) == shift_labels) & loss_mask).sum() / loss_mask.sum()).item()
    assert _metric(rank, "mean_token_accuracy")[0] == pytest.approx(expected_accuracy, rel=1e-6)


def test_a_renamed_column_raises_instead_of_transposing_the_row(monkeypatch):
    """The row is stacked BY NAME against ``_CP_METRIC_COLUMNS``.

    Built positionally, a rename or a mid-list insertion keeps the width — which is all the drain's
    ``strict=True`` zip checks — and silently feeds ``mean_token_accuracy`` the entropy sum.
    """
    monkeypatch.setattr(sft_trainer, "_CP_METRIC_COLUMNS", (*_CP_METRIC_COLUMNS[:-1], "aux_loss_total"))
    rank = _rank(0)

    with pytest.raises(KeyError, match="aux_loss_total"):
        rank._compute_cp_metrics(*_microbatch(0, aux_loss=0.3))


def test_empty_window_emits_nothing_rather_than_a_zero():
    """A run's final ``log()`` fires after the window was already drained. Appending a 0.0 accuracy
    there puts a bogus last point on every CP run's curves; the emit is skipped on the REDUCED
    micro-batch count, so every rank skips together.
    """
    ranks = [_rank(0), _rank(1)]
    _run(ranks, [_microbatch(0)], drain_every_microbatch=False)
    after_real = {key: list(values) for key, values in ranks[0]._metrics["train"].items()}
    assert after_real["mean_token_accuracy"]

    _drain_in_lockstep(ranks)

    assert {key: list(values) for key, values in ranks[0]._metrics["train"].items()} == after_real


def test_drain_is_a_noop_without_cp():
    """Non-CP runs take TRL's own metric path; a reduce here would be a collective no other trainer
    in the job is making."""
    rank = _rank(0, is_cp_mode=False)
    rank.accelerator = _NoCollective()
    rank._drain_cp_metrics()
    assert not rank._metrics["train"]


def test_train_and_eval_windows_do_not_contaminate_each_other():
    """``_metrics`` is keyed by mode and so is the accumulator; eval must not add to the run's
    train-token counter."""
    rank = _rank(0)
    rank._compute_cp_metrics(*_microbatch(0))
    rank.model.training = False
    rank._compute_cp_metrics(*_microbatch(1))

    assert set(rank._cp_metric_accum) == {"train", "eval"}
    # Eval's attended-token column is a structural zero, so draining eval leaves the counter alone.
    _drain_in_lockstep([rank])
    assert rank._total_train_tokens == 0
    assert rank._cp_metric_accum["train"] is not None


def test_accumulator_resets_after_a_drain():
    """A window must not be counted twice: the next log reduces only what came after this one."""
    ranks = [_rank(0), _rank(1)]
    _run(ranks, [_microbatch(0)], drain_every_microbatch=False)
    first = ranks[0]._total_train_tokens
    assert ranks[0]._cp_metric_accum["train"] is None

    _drain_in_lockstep(ranks)
    assert ranks[0]._total_train_tokens == first


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
