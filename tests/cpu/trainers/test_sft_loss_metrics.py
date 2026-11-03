#!/usr/bin/env python
"""``DistributedSFTTrainer``'s loss metrics fold once per log, not once per micro-batch, under every mode.
TRL's per-micro-batch fold costs four collectives and three host stalls each: hundreds per step at 512 GPUs."""

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
from tests.common.world_metrics import fake_world, rank_states

CP_SIZE = 2
VOCAB = 32
SEQ = 8


def _rank(cp_rank=0, *, training=True, cp=True, replication=None):
    """A bare trainer running the real ``compute_loss`` and drain: one rank of a CP pair, or with ``cp=False`` a
    rank whose batch ``replication`` ranks share (a TP/ETP group; 1 for plain data parallelism)."""
    trainer = object.__new__(DistributedSFTTrainer)
    trainer.model = types.SimpleNamespace(training=training)
    trainer.cp_config = types.SimpleNamespace(cp_rank=cp_rank)
    trainer.parallelism_config = types.SimpleNamespace(
        cp_size=CP_SIZE if cp else 1,
        is_cp_mode=cp,
        non_dp_replication_factor=(CP_SIZE if cp else 1) if replication is None else replication,
    )
    trainer.args = types.SimpleNamespace(use_liger_kernel=False)
    trainer.aux_loss_enabled = False
    trainer._warned_empty_labels = True
    trainer._loss_metrics = {"train": WorldMetrics(), "eval": WorldMetrics()}
    trainer._total_train_tokens = 0
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    return trainer


def _no_collective():
    return mock.patch.object(world_metrics, "_world_states", side_effect=AssertionError("a collective ran"))


def _accumulate(rank, outputs, labels, attention_mask):
    """One micro-batch through the trainer's real loss path, the forward replaced by ``outputs``."""
    inputs = {"input_ids": torch.zeros_like(labels), "labels": labels, "attention_mask": attention_mask}
    with mock.patch.object(Trainer, "compute_loss", return_value=(torch.zeros(()), outputs)), _no_collective():
        DistributedSFTTrainer.compute_loss(rank, None, inputs)


def _microbatch(seed, rows=2, aux_loss=None, attended=SEQ, cp=True):
    """``(outputs, labels, attention_mask)``; under ``cp`` the logits cover rank 0's chunk.
    ``attended`` lets a caller give the micro-batches unequal token counts."""
    generator = torch.Generator().manual_seed(seed)
    width = SEQ // CP_SIZE if cp else SEQ
    outputs = types.SimpleNamespace(logits=torch.randn(rows, width, VOCAB, generator=generator))
    if aux_loss is not None:
        outputs.aux_loss = torch.tensor(aux_loss)
    labels = torch.randint(0, VOCAB, (rows, SEQ), generator=generator)
    mask = torch.zeros(rows, SEQ, dtype=torch.long)
    mask[:, :attended] = 1
    labels[mask == 0] = LABEL_IGNORE_INDEX
    return outputs, labels, mask


def _drain_in_lockstep(ranks):
    """Every rank's drain, each flush folding every rank's window as the world's."""
    with fake_world(*(rank._loss_metrics["train" if rank.model.training else "eval"] for rank in ranks)):
        for rank in ranks:
            rank._drain_loss_metrics()


def _run(ranks, batches, *, drain_every_microbatch):
    """Drain once at the end (shipped) or after every micro-batch (the unbatched formulation).
    Both go through the same code, so the comparison cannot drift."""
    for batch in batches:
        for rank in ranks:
            _accumulate(rank, *batch)
        if drain_every_microbatch:
            _drain_in_lockstep(ranks)
    if not drain_every_microbatch:
        _drain_in_lockstep(ranks)


def _metric(rank, key):
    return rank._metrics["train"][key]


@pytest.mark.parametrize("cp", [True, False], ids=["cp", "no-cp"])
def test_accumulation_runs_no_collective_and_emits_nothing(cp):
    rank = _rank(cp=cp)
    for seed in range(4):
        _accumulate(rank, *_microbatch(seed, aux_loss=0.25, cp=cp))
    assert rank._loss_metrics["train"], "nothing was recorded"
    assert not rank._metrics["train"], "accumulation emitted a metric, which means it reduced"


def test_token_counter_is_exactly_fold_of_sums_equals_sum_of_folds():
    """``num_attended_tokens_seen`` is a pure sum, so batching it is an identity and takes no tolerance."""
    batches = [_microbatch(seed) for seed in range(5)]

    batched = [_rank(0), _rank(1)]
    _run(batched, batches, drain_every_microbatch=False)
    per_microbatch = [_rank(0), _rank(1)]
    _run(per_microbatch, batches, drain_every_microbatch=True)

    assert batched[0]._total_train_tokens == per_microbatch[0]._total_train_tokens == 5 * 2 * SEQ
    assert _metric(batched[0], "num_attended_tokens_seen") == _metric(per_microbatch[0], "num_attended_tokens_seen")
    # The fold is global, so both ranks of the group agree.
    assert batched[0]._total_train_tokens == batched[1]._total_train_tokens


@pytest.mark.parametrize("replication", [1, 2], ids=["data-parallel", "tp-pair"])
def test_ranks_sharing_a_batch_count_it_once(replication):
    """Two DP ranks hold distinct batches and both count; a TP/ETP pair holds one batch, which counts once."""
    outputs, labels, mask = _microbatch(0, attended=SEQ // 2, cp=False)
    ranks = [_rank(cp=False, replication=replication) for _ in range(2)]
    for rank in ranks:
        _accumulate(rank, outputs, labels, mask)
    _drain_in_lockstep(ranks)
    attended = int(mask.sum())
    assert _metric(ranks[0], "num_attended_tokens_seen") == [attended * 2 // replication]


def test_ratio_metrics_match_the_per_microbatch_formulation_on_equal_token_counts():
    """Accuracy and entropy are token-weighted over the log window, which equals the unweighted mean only here."""
    batches = [_microbatch(seed, aux_loss=0.1 * seed) for seed in range(4)]

    batched = [_rank(0), _rank(1)]
    _run(batched, batches, drain_every_microbatch=False)
    per_microbatch = [_rank(0), _rank(1)]
    _run(per_microbatch, batches, drain_every_microbatch=True)

    for key in ("mean_token_accuracy", "entropy", "aux_loss"):
        new = _metric(batched[0], key)
        old = _metric(per_microbatch[0], key)
        assert len(new) == 1 and len(old) == len(batches)
        # TRL's log() averages the list, and that average reaches the run's logs.
        assert new[0] == pytest.approx(sum(old) / len(old), rel=1e-12), key


def test_ratio_metrics_are_token_weighted_when_microbatches_differ_in_length():
    batches = [_microbatch(0, attended=SEQ, cp=False), _microbatch(1, attended=SEQ // 4, cp=False)]

    batched = [_rank(cp=False)]
    _run(batched, batches, drain_every_microbatch=False)
    per_microbatch = [_rank(cp=False)]
    _run(per_microbatch, batches, drain_every_microbatch=True)

    entropies = _metric(per_microbatch[0], "entropy")
    assert min(entropies) <= _metric(batched[0], "entropy")[0] <= max(entropies)
    # The token counter is a sum, so it stays exact even here.
    assert batched[0]._total_train_tokens == per_microbatch[0]._total_train_tokens


@pytest.mark.parametrize("cp", [True, False], ids=["cp", "no-cp"])
def test_drain_folds_once_per_log_on_every_rank_however_many_microbatches(cp):
    """A rank whose window is empty (its first log) enters the flush too; one rank sitting out hangs the other 511."""
    full, empty = _rank(0, cp=cp), _rank(1, cp=cp)
    for seed in range(8):
        _accumulate(full, *_microbatch(seed, aux_loss=0.3, cp=cp))
    for rank in (full, empty):
        with mock.patch.object(
            world_metrics,
            "_world_states",
            side_effect=lambda records, layout, group, key_mismatch_error: (rank_states(records), layout),
        ) as fold:
            rank._drain_loss_metrics()
        assert fold.call_count == 1


def _oracle(logits, labels):
    """TRL's ``SFTTrainer.compute_loss`` metrics: accuracy and entropy over the supervised next-token positions."""
    shift_labels = labels[:, 1:]
    loss_mask = shift_labels != LABEL_IGNORE_INDEX
    accuracy = (((logits.argmax(-1) == shift_labels) & loss_mask).sum() / loss_mask.sum()).item()
    entropy = ((entropy_from_logits(logits) * loss_mask).sum() / loss_mask.sum()).item()
    return accuracy, entropy


@pytest.mark.parametrize("cp", [True, False], ids=["cp", "no-cp"])
def test_records_carry_the_quantities_their_names_claim(cp):
    """An independent recomputation: two passes of the same code would miss attended tokens swapped for the loss-token
    count, or entropy weighted by the attended tokens."""
    chunk = SEQ // CP_SIZE if cp else SEQ
    outputs, labels, mask = _microbatch(0, rows=2, attended=SEQ - 1, cp=cp)
    # Row 0 skips two ignored labels: accuracy covers the loss tokens alone, while the attended tokens count them.
    labels[0, 2:4] = LABEL_IGNORE_INDEX
    for position in (0, 3):
        outputs.logits[0, position, labels[0, position + 1]] += 100.0
    rank = _rank(0, cp=cp, replication=1)
    _accumulate(rank, outputs, labels, mask)
    _drain_in_lockstep([rank])

    expected_attended = int(mask.sum())
    assert expected_attended != int((labels != LABEL_IGNORE_INDEX).sum()), "fixture does not separate the counts"
    assert _metric(rank, "num_attended_tokens_seen") == [expected_attended]

    # Rank 0 of a CP pair is not the last rank: every local logit is scored, its last one against chunk 1's first
    # token. Without CP the last logit predicts nothing and is dropped.
    shift_logits = outputs.logits if cp else outputs.logits[:, :-1]
    accuracy, entropy = _oracle(shift_logits, labels[:, : chunk + 1])
    assert _metric(rank, "mean_token_accuracy")[0] == pytest.approx(accuracy, rel=1e-6)
    assert _metric(rank, "entropy")[0] == pytest.approx(entropy, rel=1e-6)
    assert "aux_loss" not in rank._metrics["train"], "a forward without an aux loss logs none"


def test_liger_token_accuracy_is_weighted_by_its_supervised_tokens():
    """A fused-loss forward skips the logits and reports its own accuracy; the window weighs it by its tokens."""
    rank = _rank(cp=False)
    for accuracy, attended in ((1.0, SEQ), (0.0, SEQ // 4)):
        _, labels, mask = _microbatch(0, attended=attended, cp=False)
        outputs = types.SimpleNamespace(logits=None, token_accuracy=torch.tensor(accuracy))
        _accumulate(rank, outputs, labels, mask)
    _drain_in_lockstep([rank])
    long_tokens, short_tokens = SEQ - 1, SEQ // 4 - 1
    assert _metric(rank, "mean_token_accuracy") == [pytest.approx(long_tokens / (long_tokens + short_tokens))]
    assert "entropy" not in rank._metrics["train"]


def test_empty_window_emits_nothing_rather_than_a_zero():
    """A run's final ``log()`` fires after the drain, and a 0.0 there would put a bogus last point on every curve.
    The skip reads the flushed result, so every rank skips together."""
    ranks = [_rank(0), _rank(1)]
    _run(ranks, [_microbatch(0)], drain_every_microbatch=False)
    after_real = {key: list(values) for key, values in ranks[0]._metrics["train"].items()}
    assert after_real["mean_token_accuracy"]

    _drain_in_lockstep(ranks)

    assert {key: list(values) for key, values in ranks[0]._metrics["train"].items()} == after_real


def test_drain_is_a_noop_under_pipeline_parallelism():
    """The schedule computes the loss on the last stage and records nothing here, so a flush would be wasted."""
    rank = _rank(cp=False)
    rank._pp_runner = object()
    with _no_collective():
        rank._drain_loss_metrics()
    assert not rank._metrics["train"]


def test_train_and_eval_windows_do_not_contaminate_each_other():
    rank = _rank(0)
    _accumulate(rank, *_microbatch(0))
    rank.model.training = False
    _accumulate(rank, *_microbatch(1))

    # Eval's attended tokens are a structural zero, so draining eval leaves the counter alone.
    _drain_in_lockstep([rank])
    assert rank._total_train_tokens == 0
    assert rank._metrics["eval"]["mean_token_accuracy"] and not rank._metrics["train"]
    assert rank._loss_metrics["train"], "the train window waits for its own log"


def test_accumulator_resets_after_a_drain():
    ranks = [_rank(0), _rank(1)]
    _run(ranks, [_microbatch(0)], drain_every_microbatch=False)
    first = ranks[0]._total_train_tokens
    assert not ranks[0]._loss_metrics["train"]

    _drain_in_lockstep(ranks)
    assert ranks[0]._total_train_tokens == first


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
