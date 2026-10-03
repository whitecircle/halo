"""``PrecomputeRefLogpsRankConsistentMixin`` pins TRL's reference sweep to the data-parallel axis.

The sweep builds its loader with ``accelerator.prepare`` and reassembles the results with
``accelerator.gather_for_metrics``, both keyed on the GLOBAL rank. It runs inside TRL's ``__init__``,
when the model already carries its load-time TP attention shards and EP/ETP expert wrappers, so
siblings forwarding different rows hit shape-coupled collectives. These tests drive the mixin's sweep
over a stub reference forward.
"""

from types import SimpleNamespace

import pytest
import torch
from accelerate import PartialState
from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler

PartialState()

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.mixins.dataloader import DataParallelDataLoaderMixin
from src.trainers.preference.logprobs import FP32LogprobsMixin
from src.trainers.preference.precompute import PrecomputeRefLogpsRankConsistentMixin
from tests.common.parallelism import make_parallelism_config
from tests.common.preference_precompute import TRAINERS

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


def _fake_dataset() -> Dataset:
    """Token rows without reference log-probs, so the sweep runs."""
    dataset = Dataset.from_dict({"prompt_ids": [[row] for row in range(DATASET_SIZE)]})
    dataset._fingerprint = "deadbeef"
    return dataset


class _ReferenceForward:
    """Stands in for TRL's per-batch reference forward: each row's log-prob is its row index, and
    the rows this rank forwarded are recorded. The KL term is ``None``, as on a KL-free KTO loss."""

    _signature_columns = ["prompt_ids", "ref_logps"]
    ref_model = None
    args = SimpleNamespace(dataloader_num_workers=0, dataloader_pin_memory=False, resume_from_checkpoint=None)

    def _set_signature_columns_if_needed(self):
        pass

    def data_collator(self, rows):
        return torch.tensor([row["prompt_ids"][0] for row in rows])

    def compute_ref_log_probs(self, batch):
        self.observed_rows.extend(int(row) for row in batch)
        return batch.float(), None


class _Trainer(PrecomputeRefLogpsRankConsistentMixin, DataParallelDataLoaderMixin, _ReferenceForward):
    """Real mixin over a stub reference forward, with a world gather simulated from all ranks' shards."""

    # The stub forward stands in for FP32LogprobsMixin's, so it declares that mixin's precision.
    logprob_precision = FP32LogprobsMixin.logprob_precision

    def __init__(self, rank: int, *, presharded: bool = False, world_chunks=None):
        self.parallelism_config = _parallelism_config(rank)
        self._dataset_presharded = presharded
        self._init_reference_resume({})
        self._world_chunks = world_chunks or []
        self._gather_step = 0
        self.observed_rows: list[int] = []
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
            # accelerate's gather_for_metrics gathers through ``gather``, so the DP scoping applies.
            gather_for_metrics=lambda data: self.accelerator.gather(data),
        )

    def _required_ref_logps_columns(self) -> tuple[str, ...]:
        return ("ref_logps",)

    def _reference_settings(self) -> dict:
        return {}

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

    def _simulated_world_gather(self, outputs):
        """Concatenate every rank's chunk for this step, in global-rank order (what NCCL returns)."""
        chunks = self._world_chunks[self._gather_step]
        self._gather_step += 1
        return tuple(torch.cat(chunks).float() for _ in outputs)


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
    prepared = trainer._precompute_ref_logps(_fake_dataset(), "train", BATCH_SIZE)

    attached = [int(value) for value in prepared["ref_logps"]]
    assert attached == list(range(DATASET_SIZE)), "gathered log-probs must land in dataset order"


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


@pytest.mark.parametrize("kind", sorted(TRAINERS))
def test_preference_trainers_supply_the_sweep_the_mixin_enters(kind):
    """``_Trainer`` above stubs only the TRL base — it composes the REAL two mixins, so everything
    this file proves is proved about that pairing, and the production trainers must be that pairing.
    ``_precompute_ref_logps`` enters ``self.data_parallel_sweep()``, which the precompute mixin does
    not define: a trainer without the dataloader mixin dies with an AttributeError inside TRL's
    ``__init__``, and one that shadowed the sweep would silently run on the global-rank axis.

    The other half of the wiring — the mixin's MRO position ahead of the concrete TRL trainer — is
    pinned by ``test_precompute_in_memory_attach.py::test_the_trainers_route_the_precompute_through_the_mixin``.
    """
    trainer_cls, _ = TRAINERS[kind]
    assert issubclass(trainer_cls, DataParallelDataLoaderMixin)
    assert trainer_cls.data_parallel_sweep is DataParallelDataLoaderMixin.data_parallel_sweep


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
