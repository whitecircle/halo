"""Every rank of one data-parallel split draws the same shuffle, whatever its own RNG went through.

accelerate shards ONE permutation by batch index across the data-parallel ranks, and TP/CP/ETP
siblings and pipeline peers replay their replica's rows, so all of them must draw the same
permutation. Once training starts the ranks' global torch RNGs no longer agree — a pipeline stage
builds other layers than its peers, an EP rank other experts — so any shuffle drawn from that RNG
splits the ranks: HF's group-by-length train sampler always does, and so does a ``RandomSampler`` on a
one-process loader (``data_parallel_size == 1``) with ``use_seedable_sampler: false``. Proven on a real
4-rank gloo group with each rank's RNG deliberately consumed differently before the first epoch.

    python tests/cpu/data/test_rank_uniform_shuffle.py
"""

import datetime
import json
import os
from types import SimpleNamespace

import pytest
import torch
from accelerate import PartialState
from datasets import Dataset
from transformers import Trainer

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.mixins.dataloader import DataParallelDataLoaderMixin, set_sampler_epoch
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 4
DATASET_SIZE = 32
BATCH_SIZE = 2
EPOCHS = 2
SEED = 42

# A shuffle that diverges ends in mismatched batch shapes or a stuck collective; never sit on one.
PG_TIMEOUT_SEC = 60

# (axes, GPUs per node, train_sampling_strategy, use_seedable_sampler): each a path whose shuffle drew
# from the rank's own RNG. pp2+etp2 (one two-GPU node per stage) and tp4 are one replica over all four
# ranks, a one-process loader; pp2 and tp2 are two replicas of two ranks each.
CASES = {
    "pp2-etp2-random-unseedable": ({"pp_size": 2, "expert_tp_size": 2}, 2, "random", False),
    "tp4-random-unseedable": ({"tp_size": 4}, WORLD_SIZE, "random", False),
    "pp2-group_by_length": ({"pp_size": 2}, 2, "group_by_length", True),
    "tp2-group_by_length": ({"tp_size": 2}, WORLD_SIZE, "group_by_length", True),
}


class _Host(DataParallelDataLoaderMixin, Trainer):
    """The mixin over HF's own train sampler, with exactly what the loader build reads off a trainer."""

    def __init__(self, parallelism_config: ParallelismConfig, strategy: str, seedable: bool):
        self.parallelism_config = parallelism_config
        self._dataset_presharded = False
        self._train_batch_size = BATCH_SIZE
        self.processing_class = None
        self.data_collator = lambda rows: torch.tensor([row["id"] for row in rows])
        self.train_dataset = Dataset.from_dict(
            {"id": list(range(DATASET_SIZE)), "length": [1 + (7 * i) % 13 for i in range(DATASET_SIZE)]}
        )
        self.args = SimpleNamespace(
            seed=SEED,
            data_seed=None,
            train_sampling_strategy=strategy,
            length_column_name="length",
            train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=1,
            remove_unused_columns=False,
            dataloader_drop_last=False,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            dataloader_persistent_workers=False,
            dataloader_multiprocessing_context=None,
            dataloader_prefetch_factor=None,
            dataloader_in_order=True,
        )
        self.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            split_batches=False,
            rng_types=["generator"],
            dispatch_batches=False,
            even_batches=True,
            use_seedable_sampler=seedable,
            dataloader_config=SimpleNamespace(data_seed=None),
            non_blocking=False,
            use_stateful_dataloader=False,
        )

    def get_data_parallel_size(self) -> int:
        return self.parallelism_config.data_parallel_size

    def get_data_parallel_rank(self) -> int:
        return self.parallelism_config.get_data_parallel_rank()


def _worker(rank: int, tmp_dir: str, case: str) -> None:
    PartialState()
    axes, gpus_per_node, strategy, seedable = CASES[case]
    torch.manual_seed(SEED)
    torch.rand(97 * (rank + 1))  # this rank's own consumption before the first epoch
    config = ParallelismConfig(
        world_size=WORLD_SIZE, gpus_per_node=gpus_per_node, nvlink_domain_size=gpus_per_node, **axes
    )
    loader = _Host(config, strategy, seedable).get_train_dataloader()
    epochs = []
    for epoch in range(EPOCHS):
        set_sampler_epoch(loader, epoch)
        epochs.append([int(row) for batch in loader for row in batch])
    with open(os.path.join(tmp_dir, f"rank{rank}.json"), "w") as handle:
        json.dump({"dp_rank": config.get_data_parallel_rank(), "epochs": epochs}, handle)


@pytest.mark.parametrize("case", sorted(CASES))
def test_every_rank_of_a_replica_draws_its_rows_and_replicas_partition_the_split(tmp_path, case):
    run_gloo_ranks(_worker, WORLD_SIZE, str(tmp_path), case, pg_timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC))
    outcomes = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(WORLD_SIZE)]

    by_replica: dict[int, list[list[list[int]]]] = {}
    for outcome in outcomes:
        by_replica.setdefault(outcome["dp_rank"], []).append(outcome["epochs"])
    assert len(by_replica) < WORLD_SIZE, f"{case} must give some replica a second holder to mean anything"
    for replica, holders in by_replica.items():
        assert all(epochs == holders[0] for epochs in holders), f"{case}: replica {replica} split: {holders}"
    for epoch in range(EPOCHS):
        drawn = sorted(row for holders in by_replica.values() for row in holders[0][epoch])
        assert drawn == list(range(DATASET_SIZE)), f"{case}: epoch {epoch} replicas overlap or skip rows: {drawn}"
    assert outcomes[0]["epochs"][0] != sorted(outcomes[0]["epochs"][0]), f"{case}: the split was never shuffled"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
