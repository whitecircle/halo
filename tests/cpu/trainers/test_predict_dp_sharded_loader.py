#!/usr/bin/env python
"""``predict()`` under TP returns every test row exactly once, on a real tp2 x dp2 gloo world.

``predict()`` runs the evaluation gather over ``get_test_dataloader``. The gather keeps one chunk
per DP replica; HF's own test loader shards by world rank, so TP siblings would draw different
rows and the scope would keep one sibling's: 7 rows at batch 2 would come back as ``[0, 1, 4, 5]``,
and 11 rows at batch 3 would fail the cut. The mixin's test loader is the eval loader's DP-sharded body, and a
loader that bypasses it fails loud on its first batch rather than being gathered off the wrong
geometry, padded final round or not (8 rows at batch 2 pad nothing, and would come back
as ``[0, 1, 4, 5]``). The pins: the toolkit test loader returns each row once on every rank; a
world-sharded loader raises.

An iterable split is accelerate's dispatching loader by default, which slices rank 0's batches by
global rank whatever DP geometry it is handed: TP siblings would draw different rows. The toolkit loader
shards it by DP rank instead, so siblings draw the same rows, and refuses an explicit
``dispatch_batches: true``.

    python tests/cpu/trainers/test_predict_dp_sharded_loader.py
"""

import datetime
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from accelerate.data_loader import DataLoaderDispatcher, prepare_data_loader
from accelerate.utils import DataLoaderConfiguration, gather_object
from torch.utils.data import DataLoader, IterableDataset, SequentialSampler

from src.trainers.mixins.dataloader import DataParallelDataLoaderMixin, dp_representative_ranks
from tests.common.gloo import run_gloo_ranks

WORLD, TP = 4, 2


class _TpHost(DataParallelDataLoaderMixin):
    """A tp2 x dp2 rank's loader and gather state; the HF seams the build calls are identities."""

    def __init__(self, accelerator, rank: int, batch_size: int):
        self.accelerator = accelerator
        self.model = SimpleNamespace(training=False)
        self.data_collator = torch.tensor
        self._dataset_presharded = False
        self._dp_rank = rank // TP
        dp_rank_of = [r // TP for r in range(WORLD)]
        self._dp_metric_gather_scope = (dp_representative_ranks(dp_rank_of), WORLD)
        self.args = SimpleNamespace(
            eval_batch_size=batch_size,
            eval_use_gather_object=False,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            dataloader_persistent_workers=False,
            dataloader_drop_last=False,
            dataloader_prefetch_factor=None,
            dataloader_multiprocessing_context=None,
            dataloader_in_order=True,
        )

    def _needs_custom_dataloader(self):
        return True

    def get_data_parallel_size(self):
        return WORLD // TP

    def get_data_parallel_rank(self):
        return self._dp_rank

    def _get_collator_with_removed_columns(self, collator, description=None):
        return collator

    def _get_eval_sampler(self, dataset):
        return SequentialSampler(dataset)


def _predict(host, loader) -> list[float]:
    return [value for batch in loader for value in host._dp_gather_for_metrics(batch.float()).tolist()]


def _worker(rank: int) -> None:
    accelerator = Accelerator(cpu=True)
    for rows, batch_size in ((7, 2), (11, 3), (8, 2)):
        host = _TpHost(accelerator, rank, batch_size)
        returned = _predict(host, host.get_test_dataloader(list(range(rows))))
        assert Counter(returned) == Counter(map(float, range(rows))), f"{rows} rows at batch {batch_size}: {returned}"

        world_sharded = prepare_data_loader(
            DataLoader(list(range(rows)), batch_size=batch_size), num_processes=WORLD, process_index=rank
        )
        with pytest.raises(RuntimeError, match="not built through the DP-aware eval loader"):
            _predict(host, world_sharded)


def test_predict_returns_every_test_row_once_under_tp():
    run_gloo_ranks(_worker, WORLD, pg_timeout=datetime.timedelta(seconds=60))


class _Stream(IterableDataset):
    def __init__(self, rows: int):
        self.rows = rows

    def __iter__(self):
        yield from range(self.rows)


def _iterable_worker(rank: int) -> None:
    rows, batch_size = 13, 2
    host = _TpHost(Accelerator(cpu=True), rank, batch_size)
    loader = host.get_test_dataloader(_Stream(rows))
    assert not isinstance(loader, DataLoaderDispatcher), "the iterable split dispatched by global rank"

    drawn = [batch.tolist() for batch in loader]
    by_dp: dict[int, list] = {}
    for dp_rank, peer_drawn in gather_object([(rank // TP, drawn)]):
        assert by_dp.setdefault(dp_rank, peer_drawn) == peer_drawn, (
            f"TP siblings drew {by_dp[dp_rank]} and {peer_drawn}"
        )
    assert {row for batches in by_dp.values() for batch in batches for row in batch} == set(range(rows))

    dispatching = _TpHost(
        Accelerator(cpu=True, dataloader_config=DataLoaderConfiguration(dispatch_batches=True)), rank, 2
    )
    with pytest.raises(ValueError, match="dispatch_batches: false"):
        dispatching.get_test_dataloader(list(range(rows)))


def test_an_iterable_split_is_sharded_by_dp_rank_not_dispatched_by_global_rank():
    run_gloo_ranks(_iterable_worker, WORLD, pg_timeout=datetime.timedelta(seconds=60))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
