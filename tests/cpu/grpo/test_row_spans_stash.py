#!/usr/bin/env python
"""Row spans are read once per call and once per round: the hoist over a whole batch, the round's stash through TRL's
shuffle and split, its device-side validation against the mask the loss forward reads, and the replay arm's use of it."""

from types import SimpleNamespace

import pytest
import torch
from trl import GRPOTrainer
from trl.trainer.utils import shuffle_sequence_dict, split_tensor_dict

import halo.trainers.grpo.environmental.routing_replay as routing_replay
from halo.trainers.grpo.environmental.routing_replay import RoutingReplayMixin
from halo.trainers.grpo.environmental.step_batch import ROUTING_ROWS_KEY
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.logprob_scoring import (
    STASHED_COLUMNS,
    RowSpan,
    dense_row_spans,
    row_spans_on_device,
)
from halo.trainers.grpo.online import DistributedOnlineGRPOTrainer
from halo.trainers.grpo.row_order import permute_rows
from halo.trainers.grpo.row_span_stash import ROW_SPANS_KEY, RowSpanStashMixin
from tests.common.chunked_logprobs import EmbeddingBackboneHarness
from tests.common.gloo import run_gloo_ranks

LTK = 4


def _oracle(mask: torch.Tensor, completion_width: int) -> tuple[list[tuple[int, int]], list[int]]:
    """The row-by-row host computation the device one replaces."""
    spans, completions = [], []
    for row in mask.bool():
        real = row.nonzero().view(-1)
        lo, hi = int(real[0]), int(real[-1]) + 1
        if hi - lo < 2:
            lo, hi = (lo - 1, hi) if lo > 0 else (lo, hi + 1)
        spans.append((lo, hi))
        completions.append(int(row[-completion_width:].sum()) if completion_width else 0)
    return spans, completions


def _ragged_mask(seed: int, rows: int = 12, width: int = 14) -> torch.Tensor:
    """Left- and right-padded rows of every real length from 1, at left padding 0 and above."""
    generator = torch.Generator().manual_seed(seed)
    mask = torch.zeros(rows, width, dtype=torch.long)
    for i in range(rows):
        left = int(torch.randint(0, 4, (1,), generator=generator)) if i % 3 else 0
        length = 1 if i % 4 == 0 else int(torch.randint(1, width - left, (1,), generator=generator))
        mask[i, left : left + length] = 1
    return mask


@pytest.mark.parametrize("seed", range(4))
def test_the_device_spans_equal_the_row_by_row_computation(seed):
    mask = _ragged_mask(seed)
    spans, completions = dense_row_spans(mask, LTK)
    assert (spans, completions) == _oracle(mask, LTK)
    assert dense_row_spans(mask)[1] == [0] * mask.size(0)


@pytest.mark.parametrize("bad_row", [[0, 0, 0, 0, 0], [1, 0, 1, 0, 0]], ids=["empty row", "hole in the real tokens"])
def test_a_malformed_row_is_refused_where_the_spans_are_read(bad_row):
    mask = torch.tensor([[1, 1, 1, 1, 1], bad_row])
    with pytest.raises(ValueError, match="Row 1 has an empty or non-contiguous attention mask"):
        dense_row_spans(mask)
    assert row_spans_on_device(mask)[:, RowSpan.MALFORMED].tolist() == [0, 1]


class _Harness(EmbeddingBackboneHarness):
    """The backbone at batch size 1, so every row forwards densely."""

    def __init__(self):
        super().__init__("sdpa")

    def score(self, ids, mask):
        return self._chunked_logps_impl(self.model, ids, mask, LTK, 1, False)[0]


def _batch(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    mask = _ragged_mask(seed)
    return torch.randint(1, 40, mask.shape), mask


@pytest.fixture
def host_reads(monkeypatch):
    """The device-to-host copies of a block: ``Tensor.cpu`` is how the spans leave the device."""
    reads = []
    real = torch.Tensor.cpu

    def counting(tensor, *args, **kwargs):
        reads.append(tuple(tensor.shape))
        return real(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counting)
    return reads


def test_a_batch_of_dense_rows_reads_its_spans_once_and_scores_as_it_did_row_by_row(host_reads):
    ids, mask = _batch()
    harness = _Harness()
    hoisted = harness.score(ids, mask)
    assert len(host_reads) == 1, f"{len(host_reads)} host reads for {ids.size(0)} rows"

    per_row = torch.cat([harness.score(ids[i : i + 1], mask[i : i + 1]) for i in range(ids.size(0))])
    assert len(host_reads) == 1 + ids.size(0), "one read per row when the rows are scored one call at a time"
    torch.testing.assert_close(hoisted, per_row, atol=0.0, rtol=0.0)


def test_an_armed_table_replaces_the_read_and_matches_the_unarmed_scores(host_reads):
    ids, mask = _batch()
    harness = _Harness()
    unarmed = harness.score(ids, mask)
    harness._armed_row_spans = row_spans_on_device(mask, LTK)[:, STASHED_COLUMNS].cpu()
    host_reads.clear()
    armed = harness.score(ids, mask)
    assert host_reads == [], "an armed forward copies nothing to the host"
    torch.testing.assert_close(armed, unarmed, atol=0.0, rtol=0.0)
    assert int(harness._row_span_mismatches) == 0


def test_a_table_built_from_another_mask_is_tallied_per_misread_row():
    ids, mask = _batch()
    harness = _Harness()
    table = row_spans_on_device(mask, LTK)[:, STASHED_COLUMNS].clone()
    table[3, 1] += 1
    table[7, 0] -= 1
    harness._armed_row_spans = table
    harness.score(ids, mask)
    assert int(harness._row_span_mismatches) == 2

    harness._armed_row_spans = row_spans_on_device(mask, LTK)[:, STASHED_COLUMNS]
    harness.score(ids, mask)
    assert int(harness._row_span_mismatches) == 2, "a matching table adds nothing"


@pytest.mark.parametrize("rows", [5, 12], ids=["too few rows", "span past the row"])
def test_a_table_that_does_not_fit_the_batch_raises(rows):
    ids, mask = _batch()
    harness = _Harness()
    table = row_spans_on_device(mask, LTK)[:, STASHED_COLUMNS].clone()
    if rows == 5:
        table = table[:rows]
    else:
        table[0, 1] = mask.size(1) + 1
    harness._armed_row_spans = table
    with pytest.raises(RuntimeError, match="do not fit"):
        harness.score(ids, mask)


def test_a_table_off_the_host_is_refused_rather_than_synced():
    ids, mask = _batch()
    harness = _Harness()
    harness._armed_row_spans = row_spans_on_device(mask, LTK)[:, STASHED_COLUMNS].to("meta")
    with pytest.raises(RuntimeError, match="not the host"):
        harness.score(ids, mask)


class _Loop:
    """The loop below the mixin: records what the loss forward would find armed."""

    _armed_row_spans = None
    _row_span_mismatches = None

    def __init__(self, *, dense: bool = True, tp_group: object | None = None, training: bool = True):
        self._dense, self._tp_group = dense, tp_group
        self.model = SimpleNamespace(training=training)
        self.accelerator = SimpleNamespace(sync_gradients=True)
        self.armed_during_loss = []

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self.armed_during_loss.append(self._armed_row_spans)
        if inputs.get("fail"):
            raise RuntimeError("loss")
        return torch.tensor(0.0)

    def training_step(self, model, inputs, num_items_in_batch=None):
        return torch.tensor(0.0)

    @property
    def _trains_rows_densely(self):
        return self._dense

    def _get_tp_or_etp_process_group(self):
        return self._tp_group


class _Host(RowSpanStashMixin, _Loop):
    pass


def _stashing_host(**kwargs) -> _Host:
    host = _Host(**kwargs)
    host._stashes_row_spans = True
    return host


def _round(rows: int = 8, seed: int = 0) -> dict:
    """TRL's layout: prompts left-padded, completions right-padded, each at least one token."""
    generator = torch.Generator().manual_seed(seed)
    prompt_lengths = torch.randint(1, 11, (rows,), generator=generator)
    completion_lengths = torch.randint(1, LTK + 1, (rows,), generator=generator)
    prompt = torch.arange(10).expand(rows, 10) >= (10 - prompt_lengths)[:, None]
    completion = torch.arange(LTK).expand(rows, LTK) < completion_lengths[:, None]
    return {
        "prompt_mask": prompt,
        "completion_mask": completion,
        "prompt_ids": torch.randint(0, 40, prompt.shape),
        "advantages": torch.randn(rows),
        "old_per_token_logps": torch.randn(rows, LTK),
    }


def _attended(batch: dict) -> torch.Tensor:
    return torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1)


def _fresh(batch: dict) -> torch.Tensor:
    return row_spans_on_device(_attended(batch), batch["completion_mask"].size(1))[:, STASHED_COLUMNS]


def test_the_stash_resolves_where_rows_forward_densely_and_no_tp_group_re_syncs_them():
    for dense, group, expected in [(True, None, True), (False, None, False), (True, object(), False)]:
        host = _Host(dense=dense, tp_group=group)
        host._configure_row_span_stash()
        assert host._stashes_row_spans is expected


def test_the_round_carries_one_host_table_of_its_final_masks():
    batch = _stashing_host()._with_row_spans(_round())
    table = batch[ROW_SPANS_KEY]
    assert table.device.type == "cpu" and table.dtype == torch.int64 and table.shape == (8, 3)
    assert torch.equal(table, _fresh(batch).cpu())


def test_no_table_is_attached_in_eval_or_where_the_stash_is_off():
    assert ROW_SPANS_KEY not in _stashing_host(training=False)._with_row_spans(_round())
    assert ROW_SPANS_KEY not in _Host()._with_row_spans(_round())


def test_the_round_build_reads_the_spans_once(host_reads):
    _stashing_host()._with_row_spans(_round())
    assert len(host_reads) == 1


def test_the_table_follows_every_row_reorder_and_each_key_stays_row_aligned():
    batch = _stashing_host()._with_row_spans(_round(rows=16))
    order = torch.randperm(16, generator=torch.Generator().manual_seed(1))
    torch.manual_seed(1)
    for reordered in (shuffle_sequence_dict(dict(batch)), permute_rows(batch, order)):
        assert all(value.size(0) == 16 for value in reordered.values()), "every key keeps one row per example"
        for micro in split_tensor_dict(reordered, 4):
            assert torch.equal(micro[ROW_SPANS_KEY], _fresh(micro).cpu())
    assert not torch.equal(permute_rows(batch, order)[ROW_SPANS_KEY], batch[ROW_SPANS_KEY]), "the order moved rows"


def test_rows_reordered_without_their_table_no_longer_match_it():
    batch = _stashing_host()._with_row_spans(_round(rows=16))
    stale = {key: value for key, value in batch.items() if key != ROW_SPANS_KEY}
    shuffled = permute_rows(stale, torch.randperm(16, generator=torch.Generator().manual_seed(3)))
    assert not torch.equal(batch[ROW_SPANS_KEY], _fresh(shuffled).cpu())


def test_the_loss_forward_is_armed_with_the_micro_batchs_table_and_disarmed_after():
    host = _Host()
    table = torch.zeros(2, 3, dtype=torch.long)
    host.compute_loss(None, {ROW_SPANS_KEY: table})
    host.compute_loss(None, {})
    with pytest.raises(RuntimeError, match="loss"):
        host.compute_loss(None, {ROW_SPANS_KEY: table, "fail": True})
    assert host.armed_during_loss == [table, None, table]
    assert host._armed_row_spans is None


def test_a_window_end_refuses_the_round_whose_table_disagreed_with_a_mask():
    host = _stashing_host()
    host._row_span_mismatches = torch.tensor(2)
    host.accelerator.sync_gradients = False
    host.training_step(None, {})
    assert int(host._row_span_mismatches) == 2, "inside a window nothing is read"

    host.accelerator.sync_gradients = True
    with pytest.raises(RuntimeError, match="2 micro-batch row"):
        host.training_step(None, {})
    assert host._row_span_mismatches is None
    host.training_step(None, {})


def _reject_worker(rank: int) -> None:
    host = _stashing_host()
    host._row_span_mismatches = torch.tensor(1 if rank == 1 else 0)
    with pytest.raises(RuntimeError, match=r"The row-span check failed on 1 of 2 rank\(s\) \[1\]"):
        host.training_step(None, {})


def test_a_misaligned_micro_batch_on_one_rank_raises_on_every_rank():
    run_gloo_ranks(_reject_worker, 2)


def _tallied(batch: dict, narrow_third: bool) -> int:
    """Mismatches the loss forward tallies over a reordered round's four micro-batches."""
    _stashing_host()._with_row_spans(batch)
    harness = _Harness()
    for index, micro in enumerate(split_tensor_dict(permute_rows(batch, torch.arange(8).flip(0)), 4)):
        if narrow_third and index == 2:
            micro["completion_mask"] = micro["completion_mask"].clone()
            micro["completion_mask"][0] = False
        harness._armed_row_spans = micro[ROW_SPANS_KEY]
        harness.score(torch.randint(1, 40, (2, 10 + LTK)), _attended(micro).long())
    return int(harness._row_span_mismatches or 0)


def test_a_mask_narrowed_after_the_table_was_built_is_caught_in_its_micro_batch():
    assert _tallied(_round(), narrow_third=False) == 0
    assert _tallied(_round(), narrow_third=True) == 1


def test_the_replay_arm_reads_the_stashed_spans_and_falls_back_to_the_mask(monkeypatch):
    armed = []
    host = SimpleNamespace(
        _replay_row_spans=True,
        _rollout_routing=object(),
        _routing_injector=SimpleNamespace(arm_rows=lambda routing, rows, attended, spans: armed.append(spans)),
    )
    batch = _round()
    batch[ROUTING_ROWS_KEY] = torch.arange(8)
    table = _fresh(batch).cpu()

    with monkeypatch.context() as patch:
        patch.setattr(routing_replay, "dense_row_spans", lambda *_: pytest.fail("read the mask holding its spans"))
        RoutingReplayMixin._arm_loss_forward_replay(host, {**batch, ROW_SPANS_KEY: table})
    assert armed[0] == [(lo, hi) for lo, hi, _ in table.tolist()]

    RoutingReplayMixin._arm_loss_forward_replay(host, batch)
    assert armed[1] == dense_row_spans(_attended(batch))[0]


@pytest.mark.parametrize("trainer_cls", [DistributedAsyncEnvironmentalGRPOTrainer, DistributedOnlineGRPOTrainer])
def test_the_table_is_attached_after_the_tp_broadcast_and_never_sent_through_it(monkeypatch, trainer_cls):
    """The broadcast moves every tensor of the batch over NCCL, which a host table cannot ride."""
    seen = []
    batch = _round()
    trainer = object.__new__(trainer_cls)
    trainer.model = SimpleNamespace(training=True)
    trainer._stashes_row_spans = True
    trainer._broadcast_tensors_from_tp_leader = lambda result: seen.append(set(result)) or result
    monkeypatch.setattr(GRPOTrainer, "_generate_and_score_completions", lambda self, inputs: batch)
    trainer._collect_and_score_rollouts = lambda inputs: batch
    trainer._rlrr_config, trainer._scale_rewards_std_floor = None, 0.0
    trainer._drop_degenerate_groups, trainer._balance_token_mass = False, False
    trainer._apply_token_mass_balance = trainer._clamp_kl_reference = lambda result: None

    result = trainer._generate_and_score_completions([])

    assert seen and ROW_SPANS_KEY not in seen[0]
    assert ROW_SPANS_KEY in result


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
