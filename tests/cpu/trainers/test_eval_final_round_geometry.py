#!/usr/bin/env python
"""Every eval row the loader draws is scored exactly once, on a real 4-rank gloo world.

accelerate's even-batches shard fills every rank's last eval batch by repeating the split's first
rows, so all ranks run the same rounds. Its own trim (``gather_for_metrics`` keeps the gathered
final round's first ``remainder`` rows) counts that remainder over the dataset, which is wrong in
three geometries: TRL's ``RepeatSampler`` draws each prompt ``num_generations`` times, and a
one-process loader (offline GRPO's self-sharding sampler, a pre-sharded dataset) pads nothing yet
is trimmed anyway. sentence-transformers' no-duplicates sampler iterates its own way, but draws
every row once, which the embedding trainer declares. ``eval_split_rows`` and the toolkit gather read the padding off the loader's own
``BatchSamplerShard`` instead. The pins, per geometry:

* the rows ``eval_split_rows`` keeps, over every DP rank, are the drawn rows exactly once, and
  TP-style siblings of one DP rank keep the same rows;
* the toolkit gather returns exactly those rows on every rank, and HF's eval loss over it (a
  real-row mean repeated ``eval_batch_size`` times) is the per-row mean wherever every batch is
  full — a one-process loader's short last batch is weighted by HF's repeat, not by padding;
* the object gather (``eval_use_gather_object``, or a non-tensor input) returns those rows too, in
  the tensor gather's order, as a list;
* accelerate's own trim, tensor and object alike, gets the geometries above wrong, so the comparison
  is not vacuous;
* where HF's loss repeat is narrower than the loader batch (env GRPO's rollout round), every rank
  holding a real row keeps its loss entries and no padding rank keeps any.

    python tests/cpu/trainers/test_eval_final_round_geometry.py
"""

import math
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.data_loader import DataLoaderDispatcher, prepare_data_loader
from accelerate.utils import gather_object
from datasets import Dataset
from sentence_transformers.base.sampler import NoDuplicatesBatchSampler
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler
from trl.trainer.utils import RepeatSampler

from src.trainers.embedding.trainer import EmbeddingTrainer
from src.trainers.mixins.dataloader import (
    DataParallelDataLoaderMixin,
    dp_representative_ranks,
    final_round_split_rows,
    split_rows_head,
    split_rows_mean,
    trim_to_split_rows,
)
from tests.common.gloo import run_gloo_ranks

WORLD = 4
BATCH = 2
# An env-GRPO-style eval round: the loader batch is a rollout round, HF repeats the loss fewer times.
ROUND = 8


class _EvalHost(DataParallelDataLoaderMixin):
    """The state ``eval_split_rows`` and the evaluation gather read: the accelerator, the eval mode,
    the DP scope the install agreed on, and the embedding trainer's declared batch samplers."""

    _dataset_drawing_batch_samplers = (NoDuplicatesBatchSampler,)

    def __init__(self, accelerator, scope=None, use_gather_object=False):
        self.accelerator = accelerator
        self.args = SimpleNamespace(eval_use_gather_object=use_gather_object)
        self.model = SimpleNamespace(training=False)
        self._dp_metric_gather_scope = scope


def _row_ids(rows: list[dict]) -> torch.Tensor:
    return torch.tensor([row["row"] for row in rows])


def _dp_loader(dataset, *, dp_size, dp_rank, sampler=None, batch_sampler=None, collate_fn=None):
    loader = (
        DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)
        if batch_sampler is not None
        else DataLoader(dataset, batch_size=BATCH, sampler=sampler)
    )
    return prepare_data_loader(loader, num_processes=dp_size, process_index=dp_rank, put_on_device=False)


def _geometries(rank: int) -> dict[str, tuple]:
    """``name -> (loader, dp_rank, scope, rows the loader draws over the world)`` for this rank."""
    plain = {
        "dp_rank0_holds_the_tail": list(range(10)),
        "dp_half_padded_rank": list(range(11)),
    }
    cases = {name: (_dp_loader(rows, dp_size=WORLD, dp_rank=rank), rank, None, rows) for name, rows in plain.items()}

    prompts, generations = list(range(5)), 2
    sampler = RepeatSampler(prompts, mini_repeat_count=generations, seed=0, shuffle=False)
    cases["repeat_sampler"] = (
        _dp_loader(prompts, dp_size=WORLD, dp_rank=rank, sampler=sampler),
        rank,
        None,
        [prompt for prompt in prompts for _ in range(generations)],
    )

    # sentence-transformers' no_duplicates over distinct rows: it draws every row once, its own way,
    # in one seeded order on every rank, as its trainer builds it.
    distinct = Dataset.from_dict({"row": list(range(11))})
    no_duplicates = NoDuplicatesBatchSampler(
        distinct, batch_size=BATCH, drop_last=False, generator=torch.Generator(), seed=0
    )
    cases["no_duplicates"] = (
        _dp_loader(distinct, dp_size=WORLD, dp_rank=rank, batch_sampler=no_duplicates, collate_fn=_row_ids),
        rank,
        None,
        list(range(11)),
    )

    # Offline GRPO's loader: its sampler hands each rank a disjoint full-batch slice of the whole
    # split, and accelerate is asked for device placement only.
    split = list(range(13))
    starts = [r * (len(split) // WORLD) + min(r, len(split) % WORLD) for r in range(WORLD)]
    mine = split[starts[rank] : starts[rank] + BATCH]
    cases["offline_one_process"] = (
        _dp_loader(split, dp_size=1, dp_rank=0, batch_sampler=BatchSampler(mine, BATCH, drop_last=True)),
        rank,
        None,
        [row for start in starts for row in split[start : start + BATCH]],
    )

    shards = [[r * 100 + i for i in range(5)] for r in range(WORLD)]
    cases["presharded"] = (
        _dp_loader(shards[rank], dp_size=1, dp_rank=0),
        rank,
        None,
        [row for shard in shards for row in shard],
    )

    # TP-style siblings: two DP replicas of two ranks each; siblings draw the same rows.
    dp_rank_of = [r // 2 for r in range(WORLD)]
    rows = list(range(7))
    cases["dp_scoped_siblings"] = (
        _dp_loader(rows, dp_size=2, dp_rank=dp_rank_of[rank]),
        dp_rank_of[rank],
        (dp_representative_ranks(dp_rank_of), WORLD),
        rows,
    )
    return cases


def _one_pass(host: _EvalHost, loader) -> dict[str, list[float]]:
    """One evaluation pass, per stream: the rows kept, the toolkit gather's rows and loss entries, its
    object gather's rows (a non-tensor input, then the flag on tensors), and accelerate's tensor and
    object gathers."""
    streams = ("kept", "gathered", "loss", "objects", "flagged", "accelerate", "accelerate_objects")
    out: dict[str, list[float]] = {stream: [] for stream in streams}
    flagged = _EvalHost(host.accelerator, host._dp_metric_gather_scope, use_gather_object=True)
    for batch in loader:
        batch = batch.float()
        own = host.eval_split_rows(len(batch))
        out["kept"].extend(batch[:own].tolist())
        out["gathered"].extend(host._dp_gather_for_metrics(batch).tolist())
        out["objects"].extend(host._dp_gather_for_metrics(batch.tolist()))
        flagged_rows = flagged._dp_gather_for_metrics(batch)
        assert isinstance(flagged_rows, list), "args.eval_use_gather_object ignored: the tensor gather ran"
        out["flagged"].extend(map(float, flagged_rows))
        # HF: the batch's loss repeated eval_batch_size times. Over the real rows; an all-padding
        # batch has none, and its NaN must be cut, never averaged.
        real_mean = batch[:own].mean() if own else torch.tensor(float("nan"))
        out["loss"].extend(host._dp_gather_for_metrics(real_mean.repeat(BATCH)).tolist())
        out["accelerate"].extend(host.accelerator.gather_for_metrics(batch).tolist())
        out["accelerate_objects"].extend(host.accelerator.gather_for_metrics(batch.tolist()))
    return out


def _worker(rank: int) -> None:
    accelerator = Accelerator(cpu=True)
    accelerate_wrong: dict[str, set[str]] = {"accelerate": set(), "accelerate_objects": set()}
    for name, (loader, dp_rank, scope, drawn) in _geometries(rank).items():
        host = _EvalHost(accelerator, scope)
        out = _one_pass(host, loader)

        per_rank = gather_object([(dp_rank, out["kept"])])
        by_dp: dict[int, list[float]] = {}
        for peer_dp_rank, peer_kept in per_rank:
            assert by_dp.setdefault(peer_dp_rank, peer_kept) == peer_kept, f"{name}: siblings kept different rows"
        scored = [row for rows in by_dp.values() for row in rows]
        assert Counter(scored) == Counter(map(float, drawn)), f"{name}: eval_split_rows kept {sorted(scored)}"

        gathered = out["gathered"]
        assert Counter(gathered) == Counter(map(float, drawn)), f"{name}: the gather returned {sorted(gathered)}"
        for stream in ("objects", "flagged"):
            assert out[stream] == gathered, f"{name}: the {stream} object gather returned {out[stream]}"
        if name != "presharded":
            mean = sum(drawn) / len(drawn)
            assert sum(out["loss"]) / len(out["loss"]) == pytest.approx(mean), f"{name}: eval loss {out['loss']}"

        for stream, wrong in accelerate_wrong.items():
            if Counter(out[stream]) != Counter(map(float, drawn)):
                wrong.add(name)
    for stream, wrong in accelerate_wrong.items():
        assert wrong >= {"repeat_sampler", "offline_one_process", "presharded", "dp_scoped_siblings"}, (
            f"accelerate's own {stream} got only {sorted(wrong)} wrong: the geometries no longer pin the defect"
        )
    dist.barrier()


def test_every_drawn_eval_row_is_scored_exactly_once():
    run_gloo_ranks(_worker, WORLD)


@pytest.mark.parametrize(
    ("split_rows", "chunk", "kept_per_rank"),
    [
        (4, 1, [1, 0]),  # a one-round split on rank 0 alone: a global floor kept nothing
        (6, 2, [2, 0]),
        (12, 1, [1, 1]),  # rank 1 half real still keeps its entry
        (12, 8, [8, 4]),  # one entry per row: exact
        (4, 16, [8, 0]),  # two entries per row: the head scales with the chunk
    ],
)
def test_the_cut_keeps_each_ranks_own_head_rounded_up(split_rows, chunk, kept_per_rank):
    gathered = torch.cat([torch.full((chunk,), float(rank)) for rank in range(2)])

    cut = trim_to_split_rows(gathered, split_rows=split_rows, rows_per_rank=ROUND, num_ranks=2)

    assert cut.tolist() == [float(rank) for rank, kept in enumerate(kept_per_rank) for _ in range(kept)]


def test_a_batch_of_real_rows_alone_is_handed_back_unsliced():
    """Autograd records even a full-range slice, and its backward allocates and copies a gradient the
    size of the whole tensor, a logits plane per train step at the consumers' call sites."""
    logits = torch.randn(3, 4, requires_grad=True) * 1

    assert split_rows_head(logits, 3) is logits
    assert split_rows_head(logits, 5) is logits
    assert split_rows_head(logits, 2).shape == (2, 4) and split_rows_head(logits, 2).grad_fn is not logits.grad_fn


def test_a_split_rows_mean_is_the_plain_mean_over_every_row_and_zero_over_none():
    """Over every row it is ``mean`` itself, bit for bit, so a train batch reduces to its plain mean; over
    the split's first rows it is their mean; over none (a rank of padding alone) it is 0, not NaN."""
    values = torch.randn(3, 5, generator=torch.Generator().manual_seed(0))

    assert torch.equal(split_rows_mean(values, 5, dim=1), values.mean(dim=1))
    assert torch.equal(split_rows_mean(values, 5, dim=1, whole=True), values.mean())
    torch.testing.assert_close(split_rows_mean(values, 2, dim=1), values[:, :2].mean(dim=1))
    torch.testing.assert_close(split_rows_mean(values, 2, dim=1, whole=True), values[:, :2].mean())
    assert torch.equal(split_rows_mean(values, 0, dim=1), torch.zeros(3))
    assert split_rows_mean(values, 0, whole=True).item() == 0.0


def _rollout_round_worker(rank: int) -> None:
    """Loss entries over a RepeatSampler loader of ``ROUND`` rows per rank, repeated once per batch."""
    accelerator = Accelerator(cpu=True)
    host = _EvalHost(accelerator)
    for prompts in (2, 11):
        sampler = RepeatSampler(list(range(prompts)), mini_repeat_count=2, seed=0, shuffle=False)
        loader = prepare_data_loader(
            DataLoader(list(range(prompts)), batch_size=ROUND, sampler=sampler),
            num_processes=WORLD,
            process_index=rank,
            put_on_device=False,
        )
        entries, real_means = [], []
        for batch in loader:
            batch = batch.float()
            own = host.eval_split_rows(len(batch))
            real_mean = batch[:own].mean() if own else torch.tensor(float("nan"))
            entries.extend(host._dp_gather_for_metrics(real_mean.repeat(1)).tolist())
            real_means.append(None if math.isnan(real_mean) else float(real_mean))
        expected = [
            mean
            for step in zip(*(peer for peer in gather_object([real_means])), strict=True)
            for mean in step
            if mean is not None
        ]
        assert entries == pytest.approx(expected), f"{prompts} prompts: kept {entries}, real ranks' means {expected}"


def test_a_loss_repeated_fewer_times_than_rows_keeps_every_real_rank():
    run_gloo_ranks(_rollout_round_worker, WORLD)


def _short_last_batch_worker(rank: int) -> None:
    """``even_batches`` off: 30 rows at batch 4 over 4 ranks are 8 batches, so every rank runs two
    rounds and rank 3's last batch holds rows 28 and 29 alone. HF still repeats each rank's scalar loss
    ``eval_batch_size`` times, through the object gather too under ``eval_use_gather_object``."""
    accelerator = Accelerator(cpu=True)
    host, flagged = _EvalHost(accelerator), _EvalHost(accelerator, use_gather_object=True)
    rows, batch_size = list(range(30)), 4
    loader = prepare_data_loader(
        DataLoader(rows, batch_size=batch_size),
        num_processes=WORLD,
        process_index=rank,
        put_on_device=False,
        even_batches=False,
    )
    entries, flagged_entries, objects = [], [], []
    for batch in loader:
        batch = batch.float()
        assert host.eval_split_rows(len(batch)) == len(batch), "a short batch holds real rows alone"
        entries.extend(host._dp_gather_for_metrics(batch.mean().repeat(batch_size)).tolist())
        flagged_entries.extend(map(float, flagged._dp_gather_for_metrics(batch.mean().repeat(batch_size))))
        objects.extend(host._dp_gather_for_metrics(batch.tolist()))
    for name, kept in (("tensor", entries), ("object", flagged_entries)):
        assert sum(kept) / len(kept) == pytest.approx(sum(rows) / len(rows)), f"{name} loss entries {kept}"
    assert sorted(objects) == list(map(float, rows)), f"per-row objects {objects}"


def test_a_short_last_batch_weighs_its_real_rows():
    """The loss a short final batch scores is over fewer rows than the ``eval_batch_size`` entries HF
    repeats it into; the cut keeps one entry per real row, as a padded round's does."""
    run_gloo_ranks(_short_last_batch_worker, WORLD)


def _sharded(batch_sampler) -> DataLoader:
    return prepare_data_loader(
        DataLoader(list(range(23)), batch_sampler=batch_sampler), num_processes=2, process_index=0, put_on_device=False
    )


class _NoStreamBatchSampler(BatchSampler):
    """A batch sampler with torch's iteration but no ``sampler``: it names no stream to count."""

    def __init__(self):
        self.batch_size, self.drop_last = 4, False

    def __len__(self):
        return 6


class _FullBatchesOnly(BatchSampler):
    """A batch sampler over a counted stream that iterates its own way, here dropping a short tail."""

    def __iter__(self):
        yield from (batch for batch in super().__iter__() if len(batch) == self.batch_size)


def _st_batch_sampler(kind: str, rows):
    """The batch sampler sentence-transformers' own trainer builds for ``batch_sampler: kind``."""
    st_trainer = pytest.importorskip("sentence_transformers.base.trainer")
    st_sampler = pytest.importorskip("sentence_transformers.base.sampler")
    host = SimpleNamespace(args=SimpleNamespace(batch_sampler=st_sampler.BatchSamplers(kind)))
    return st_trainer.BaseTrainer.get_batch_sampler(
        host, rows, batch_size=4, drop_last=False, valid_label_columns=["label"], generator=None, seed=0
    )


def test_only_a_known_stream_is_counted():
    """The count is ``len(sampler)``, true only where torch's iteration draws a known sampler's rows,
    or ``len(dataset)`` where a trainer declares its batch sampler draws each row once.
    sentence-transformers' default batch sampler is torch's over a ``RandomSampler``, so it counts;
    the embedding trainer declares no-duplicates; group-by-label draws 20 of 23 rows here, so a count
    off it would cut real rows. A batch sampler with no ``sampler`` must read as no trim rather than
    raise at the final eval batch."""
    rows = Dataset.from_dict({"anchor": [f"a{i}" for i in range(23)], "label": [i % 3 for i in range(23)]})
    grouped = _st_batch_sampler("group_by_label", rows)
    assert sum(len(batch) for batch in grouped) < len(rows), "premise: it draws fewer rows"

    assert final_round_split_rows(_sharded(_st_batch_sampler("batch_sampler", rows))) == 23 % 8
    assert final_round_split_rows(_sharded(grouped)) is None
    assert final_round_split_rows(_sharded(_st_batch_sampler("no_duplicates", rows))) is None, "undeclared"
    declared = EmbeddingTrainer._dataset_drawing_batch_samplers
    assert final_round_split_rows(_sharded(_st_batch_sampler("no_duplicates", rows)), declared) == 23 % 8
    assert final_round_split_rows(_sharded(grouped), declared) is None
    assert final_round_split_rows(_sharded(_NoStreamBatchSampler())) is None
    assert final_round_split_rows(_sharded(BatchSampler(list(range(23)), 4, drop_last=False))) is None
    assert final_round_split_rows(_sharded(_FullBatchesOnly(SequentialSampler(range(23)), 4, drop_last=False))) is None
    for stream in (SequentialSampler(range(23)), RepeatSampler(list(range(23)), mini_repeat_count=1, shuffle=False)):
        assert final_round_split_rows(_sharded(BatchSampler(stream, 4, drop_last=False))) == 23 % 8, stream


class _FinalRoundAccelerator:
    """accelerate's surface the gather reads, at the final round of ``loader``, with no backend: the
    tensor gather concatenates fixed per-rank chunks, and each gather records its calls."""

    def __init__(self, loader, per_rank_rows=()):
        self.gradient_state = SimpleNamespace(end_of_dataloader=True, active_dataloader=loader)
        self._per_rank_rows = list(per_rank_rows)
        self.gather_calls = 0
        self.gather_for_metrics_calls = []

    def gather(self, tensor):
        del tensor
        self.gather_calls += 1
        return torch.cat(self._per_rank_rows)

    def gather_for_metrics(self, input_data, use_gather_object=False):
        self.gather_for_metrics_calls.append(use_gather_object)
        return "accelerate's"


def _two_rank_loader(rows: int, process_index: int = 0, **prepare_kwargs):
    """DP rank ``process_index`` of 2, batch BATCH: the final round holds ``rows % (2 * BATCH)`` real rows."""
    return prepare_data_loader(
        DataLoader(list(range(rows)), batch_size=BATCH),
        num_processes=2,
        process_index=process_index,
        put_on_device=False,
        **prepare_kwargs,
    )


def _single_process_host(loader, per_rank_rows=(), *, use_gather_object=False) -> _EvalHost:
    # The DP scope names both chunks, as the install agrees it for a 2-replica loader.
    return _EvalHost(_FinalRoundAccelerator(loader, per_rank_rows), ([0, 1], 2), use_gather_object=use_gather_object)


def test_objects_and_the_flag_skip_the_tensor_gather():
    """Only an all-tensor input without the flag reaches the tensor gather; with no peer to gather
    from, the object gather is its input, as accelerate's is."""
    rows = [torch.tensor([0.0, 1.0]), torch.tensor([2.0, 3.0])]
    host = _single_process_host(_two_rank_loader(3), rows)  # 3 real rows: rank 0's two, rank 1's first
    flagged = _single_process_host(_two_rank_loader(3), rows, use_gather_object=True)

    assert host._dp_gather_for_metrics(torch.zeros(2)).tolist() == [0.0, 1.0, 2.0]
    assert host._dp_gather_for_metrics([0.0, 1.0]) == [0.0, 1.0]
    assert host._dp_gather_for_metrics(torch.zeros(2), use_gather_object=True).tolist() == [0.0, 0.0]
    assert flagged._dp_gather_for_metrics(torch.ones(2)).tolist() == [1.0, 1.0]
    assert host.accelerator.gather_calls == 1
    assert flagged.accelerator.gather_calls == 0, "args.eval_use_gather_object ignored"
    assert host.accelerator.gather_for_metrics_calls == flagged.accelerator.gather_for_metrics_calls == []


def test_a_dispatching_loader_keeps_accelerates_measured_trim():
    dispatcher = object.__new__(DataLoaderDispatcher)
    host = _EvalHost(_FinalRoundAccelerator(dispatcher), None)

    assert host._dp_gather_for_metrics(torch.zeros(2)) == "accelerate's"
    assert host.accelerator.gather_for_metrics_calls == [False]


def test_a_dispatching_loader_under_a_dp_scope_raises():
    """It slices by global rank, so the siblings the scope drops as duplicates drew other rows."""
    host = _EvalHost(_FinalRoundAccelerator(object.__new__(DataLoaderDispatcher)), ([0, 2], WORLD))

    with pytest.raises(RuntimeError, match="dispatches rank 0's batches by global rank"):
        host._dp_gather_for_metrics(torch.zeros(2))
    assert host.accelerator.gather_for_metrics_calls == []


def test_a_dropped_tail_is_not_cut():
    loader = prepare_data_loader(
        DataLoader(list(range(3)), batch_size=BATCH, drop_last=True), num_processes=2, put_on_device=False
    )
    assert final_round_split_rows(loader) is None
    assert final_round_split_rows(_two_rank_loader(3)) == 3, "premise: the padded round is counted"


def test_without_even_batches_a_rank_s_rows_are_all_real():
    """5 rows at batch 2 over 2 ranks: rank 1 ends on its only batch, a full one from the first round,
    while rank 0 runs a second. The rank-order count would read rank 1's rows as padding; without
    even batches no batch carries any, and unequal batch counts take the identity gather."""
    loader = _two_rank_loader(5, process_index=1, even_batches=False)
    assert final_round_split_rows(loader) == 1, "premise: the short round is counted"
    host = _single_process_host(loader)

    assert host.eval_split_rows(BATCH) == BATCH


def test_a_padding_rank_keeps_its_rows_in_train_mode():
    """In eval, rank 1 of a 3-row split's final round holds one real row of two; in train mode the
    batch is a training batch, every row of which is the rank's own."""
    host = _single_process_host(_two_rank_loader(3, process_index=1))
    assert host.eval_split_rows(BATCH) == 1
    host.model.training = True
    assert host.eval_split_rows(BATCH) == BATCH


def test_the_install_arms_the_toolkit_gather_on_an_unscoped_loader_too():
    """Plain DP needs no DP scope, but its final round still owes the geometry's cut."""
    host = _EvalHost(accelerator=None)
    host._needs_custom_dataloader = lambda: False

    host._install_dp_metric_gather()

    assert host.gather_function == host._dp_gather_for_metrics
    assert host._dp_metric_gather_scope is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
