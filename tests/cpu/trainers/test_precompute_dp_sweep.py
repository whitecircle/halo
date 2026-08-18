"""``PrecomputeRefLogpsRankConsistentMixin`` pins TRL's reference sweep to the data-parallel axis.

TRL builds the sweep's loader with ``accelerator.prepare`` and reassembles the results with
``accelerator.gather``, both keyed on the GLOBAL rank. The sweep runs inside TRL's ``__init__``, when
the model already carries its load-time TP attention shards and EP/ETP expert wrappers, so siblings
forwarding different rows hit shape-coupled collectives. These tests drive the mixin over a stub base
shaped like TRL's ``_precompute_ref_logps``.
"""

from types import SimpleNamespace

import pytest
import torch
from accelerate import PartialState
from torch.utils.data import DataLoader, SequentialSampler

PartialState()

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.mixins.dataloader import DataParallelDataLoaderMixin
from src.trainers.preference.dpo import DistributedDPOTrainer
from src.trainers.preference.kto import DistributedKTOTrainer
from src.trainers.preference.precompute import PrecomputeRefLogpsRankConsistentMixin
from tests.common.parallelism import make_parallelism_config

WORLD_SIZE = 8
TP_SIZE = 2
DATASET_SIZE = 32
BATCH_SIZE = 2
# Global rank 2 has DP rank 1 under tp2, so DP sharding and global-rank sharding disagree on it —
# rank 0 would pass either way.
PROBE_RANK = 2


def _parallelism_config(rank: int) -> ParallelismConfig:
    return make_parallelism_config(world_size=WORLD_SIZE, gpus_per_node=WORLD_SIZE, rank=rank, tp_size=TP_SIZE)


def _dataloader() -> DataLoader:
    dataset = list(range(DATASET_SIZE))
    return DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SequentialSampler(dataset), collate_fn=torch.tensor)


def _fake_dataset() -> SimpleNamespace:
    """The two attributes the mixin reads: the cache key it pins, and the columns it checks for
    already-supplied reference log-probs (absent here, so the sweep runs)."""
    return SimpleNamespace(_fingerprint="deadbeef", column_names=["prompt"])


class _TrlLikeBase:
    """Stands in for TRL's ``_precompute_ref_logps``: prepare a loader, gather every batch."""

    def _precompute_ref_logps(self, dataset, name, batch_size):
        loader = self.accelerator.prepare(_dataloader())
        self.observed_rows = [int(row) for batch in loader for row in batch]
        self.observed_gathers = [[int(row) for row in self.accelerator.gather(batch)] for batch in loader]
        return dataset


class _Trainer(PrecomputeRefLogpsRankConsistentMixin, DataParallelDataLoaderMixin, _TrlLikeBase):
    """Real mixin over a TRL-shaped base, with a world gather simulated from all ranks' shards."""

    def __init__(self, rank: int, *, presharded: bool = False, world_chunks=None):
        self.parallelism_config = _parallelism_config(rank)
        self._dataset_presharded = presharded
        self._world_chunks = world_chunks or []
        self._gather_step = 0
        self.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            split_batches=False,
            rng_types=[],
            dispatch_batches=False,
            even_batches=True,
            use_seedable_sampler=True,
            dataloader_config=SimpleNamespace(data_seed=None),
            non_blocking=False,
            use_stateful_dataloader=False,
            prepare=lambda loader: self.accelerator.prepare_data_loader(loader),
            prepare_data_loader=self._unpinned_prepare_data_loader,
            gather=self._simulated_world_gather,
        )

    def _required_ref_logps_columns(self) -> tuple[str, ...]:
        return ("ref_logps",)

    def get_data_parallel_size(self) -> int:
        return self.parallelism_config.data_parallel_size

    def get_data_parallel_rank(self) -> int:
        return self.parallelism_config.get_data_parallel_rank()

    def _data_parallel_rank_by_global_rank(self) -> list[int]:
        return [_parallelism_config(rank).get_data_parallel_rank() for rank in range(WORLD_SIZE)]

    def _unpinned_prepare_data_loader(self, loader, device_placement=None, slice_fn_for_dispatch=None):
        """What accelerate would do untouched: shard by GLOBAL rank over the whole world."""
        return self._prepare_dataloader(
            loader, num_processes=WORLD_SIZE, process_index=self.parallelism_config.global_rank
        )

    def _simulated_world_gather(self, tensor):
        """Concatenate every rank's chunk for this step, in global-rank order (what NCCL returns)."""
        chunks = self._world_chunks[self._gather_step]
        self._gather_step += 1
        return torch.cat(chunks)


def _dp_shard_batches(rank: int) -> list[torch.Tensor]:
    trainer = _Trainer(rank)
    return list(trainer._prepare_dataloader(_dataloader()))


def _world_chunks_per_step() -> list[list[torch.Tensor]]:
    per_rank = [_dp_shard_batches(rank) for rank in range(WORLD_SIZE)]
    return [[per_rank[rank][step] for rank in range(WORLD_SIZE)] for step in range(len(per_rank[0]))]


def test_sweep_loader_shards_by_dp_rank_not_global_rank():
    trainer = _Trainer(PROBE_RANK, world_chunks=_world_chunks_per_step())
    trainer._precompute_ref_logps(_fake_dataset(), "train", BATCH_SIZE)

    dp_size = trainer.get_data_parallel_size()
    expected = [
        row
        for step in range(DATASET_SIZE // (BATCH_SIZE * dp_size))
        for row in range(
            (step * dp_size + trainer.get_data_parallel_rank()) * BATCH_SIZE,
            (step * dp_size + trainer.get_data_parallel_rank() + 1) * BATCH_SIZE,
        )
    ]
    assert trainer.observed_rows == expected
    # The un-pinned path would have handed this rank the global-rank shard instead.
    unpinned = trainer._unpinned_prepare_data_loader(_dataloader())
    assert trainer.observed_rows != [int(row) for batch in unpinned for row in batch]


def test_sweep_gather_deduplicates_siblings_into_dataset_order():
    trainer = _Trainer(PROBE_RANK, world_chunks=_world_chunks_per_step())
    trainer._precompute_ref_logps(_fake_dataset(), "train", BATCH_SIZE)

    flat = [row for step in trainer.observed_gathers for row in step]
    assert flat == list(range(DATASET_SIZE)), "gathered log-probs must land in dataset order"


def test_sweep_restores_the_accelerator_hooks():
    trainer = _Trainer(PROBE_RANK, world_chunks=_world_chunks_per_step())
    original_prepare = trainer.accelerator.prepare_data_loader
    original_gather = trainer.accelerator.gather

    trainer._precompute_ref_logps(_fake_dataset(), "train", BATCH_SIZE)

    assert trainer.accelerator.prepare_data_loader is original_prepare
    assert trainer.accelerator.gather is original_gather


def test_presharded_dataset_is_rejected():
    trainer = _Trainer(PROBE_RANK, presharded=True)
    with pytest.raises(ValueError, match="pre-sharded dataset"):
        trainer._precompute_ref_logps(_fake_dataset(), "train", BATCH_SIZE)


@pytest.mark.parametrize("trainer_cls", [DistributedDPOTrainer, DistributedKTOTrainer])
def test_preference_trainers_supply_the_sweep_the_mixin_enters(trainer_cls):
    """``_Trainer`` above stubs only the TRL base — it composes the REAL two mixins, so everything
    this file proves is proved about that pairing, and the production trainers must be that pairing.
    ``_precompute_ref_logps`` enters ``self.data_parallel_sweep()``, which the rank-consistent mixin
    does not define: a trainer without the dataloader mixin dies with an AttributeError inside TRL's
    ``__init__``, and one that shadowed the sweep would silently run on the global-rank axis again.

    The other half of the wiring — the mixin's MRO position ahead of the concrete TRL trainer — is
    pinned by ``test_precompute_rank_consistent.py::test_distributed_trainers_use_the_mixin``.
    """
    assert issubclass(trainer_cls, DataParallelDataLoaderMixin)
    assert trainer_cls.data_parallel_sweep is DataParallelDataLoaderMixin.data_parallel_sweep


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
