"""DP row-ownership contract of ``DataParallelDataLoaderMixin._prepare_dataloader``.

Covers the two halves that must agree for any DP < world_size topology: the loader hands TP/CP/ETP
siblings IDENTICAL rows while DP ranks partition the dataset, and the gather inverse
(``dp_representative_ranks`` + ``select_gathered_chunks``) turns a world-order gather of those rows
back into dataset order. Also pins ``data_seed`` to the sampler, not to the ambient torch seed.
"""

from types import SimpleNamespace

import pytest
import torch
from accelerate import PartialState
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# prepare_data_loader resolves its defaults through the accelerate state.
PartialState()

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.mixins.dataloader import (
    DataParallelDataLoaderMixin,
    dp_representative_ranks,
    select_gathered_chunks,
)
from tests.common.parallelism import make_parallelism_config

WORLD_SIZE = 8
DATASET_SIZE = 32
BATCH_SIZE = 2

# Topologies where data_parallel_size < world_size, i.e. every rank has siblings that must replay its
# rows: attention TP, Ulysses CP, and expert-TP (whose DP rank is a dispatch-coordinate, not a
# floor-division — so the sibling pairs are not adjacent global ranks).
SIBLING_TOPOLOGIES = {
    "tp2": {"tp_size": 2},
    "tp4": {"tp_size": 4},
    "cp2": {"cp_size": 2},
    "ep4_etp2": {"ep_size": 4, "expert_tp_size": 2},
}


def _parallelism_config(rank: int, **kwargs) -> ParallelismConfig:
    return make_parallelism_config(world_size=WORLD_SIZE, gpus_per_node=WORLD_SIZE, rank=rank, **kwargs)


class _Trainer(DataParallelDataLoaderMixin):
    """Minimal host exposing exactly what ``_prepare_dataloader`` reads off a trainer."""

    def __init__(self, parallelism_config: ParallelismConfig, data_seed: int | None = None):
        self.parallelism_config = parallelism_config
        self._dataset_presharded = False
        self.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            split_batches=False,
            rng_types=[],
            dispatch_batches=False,
            even_batches=True,
            use_seedable_sampler=True,
            dataloader_config=SimpleNamespace(data_seed=data_seed),
            non_blocking=False,
            use_stateful_dataloader=False,
        )

    def get_data_parallel_size(self) -> int:
        return self.parallelism_config.data_parallel_size

    def get_data_parallel_rank(self) -> int:
        return self.parallelism_config.get_data_parallel_rank()


def _rank_batches(trainer: _Trainer, *, shuffle: bool) -> list[torch.Tensor]:
    """Batches one rank consumes from a DP-prepared loader over ``range(DATASET_SIZE)``."""
    dataset = list(range(DATASET_SIZE))
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=torch.tensor)
    return list(trainer._prepare_dataloader(loader))


def _rank_rows(trainer: _Trainer, *, shuffle: bool = False) -> list[int]:
    return [int(row) for batch in _rank_batches(trainer, shuffle=shuffle) for row in batch]


@pytest.mark.parametrize("topology", sorted(SIBLING_TOPOLOGIES))
def test_siblings_share_rows_and_dp_ranks_partition_the_dataset(topology):
    """Siblings must see identical rows (their collectives are shape-coupled); DP ranks must not."""
    configs = [_parallelism_config(rank, **SIBLING_TOPOLOGIES[topology]) for rank in range(WORLD_SIZE)]
    dp_size = configs[0].data_parallel_size
    assert dp_size < WORLD_SIZE, "topology must have siblings for this test to mean anything"

    rows = [_rank_rows(_Trainer(config)) for config in configs]
    rows_by_dp_rank: dict[int, list[list[int]]] = {}
    for config, rank_rows in zip(configs, rows, strict=True):
        rows_by_dp_rank.setdefault(config.get_data_parallel_rank(), []).append(rank_rows)

    assert sorted(rows_by_dp_rank) == list(range(dp_size))
    for dp_rank, sibling_rows in rows_by_dp_rank.items():
        assert len(sibling_rows) == WORLD_SIZE // dp_size
        for sibling in sibling_rows[1:]:
            assert sibling == sibling_rows[0], f"DP rank {dp_rank} siblings disagree: {sibling_rows}"

    # Every example exactly once across the DP ranks — no rank silently replays another's slice.
    covered = [row for sibling_rows in rows_by_dp_rank.values() for row in sibling_rows[0]]
    assert sorted(covered) == list(range(DATASET_SIZE))


@pytest.mark.parametrize("topology", sorted(SIBLING_TOPOLOGIES))
def test_sibling_deduplicated_gather_reproduces_dataset_order(topology):
    """A world gather of the DP-sharded rows, deduplicated, is the dataset-ordered block per step.

    This is the contract TRL's precompute sweep indexes by: ``cat(per-step gathers)[i]`` must be
    example ``i``.
    """
    configs = [_parallelism_config(rank, **SIBLING_TOPOLOGIES[topology]) for rank in range(WORLD_SIZE)]
    dp_size = configs[0].data_parallel_size
    keep = dp_representative_ranks([config.get_data_parallel_rank() for config in configs])
    assert len(keep) == dp_size

    per_rank = [_rank_batches(_Trainer(config), shuffle=False) for config in configs]
    steps = len(per_rank[0])
    assert steps == DATASET_SIZE // (BATCH_SIZE * dp_size)

    reassembled: list[int] = []
    for step in range(steps):
        gathered = torch.cat([per_rank[rank][step] for rank in range(WORLD_SIZE)])
        assert gathered.numel() == WORLD_SIZE * BATCH_SIZE
        reassembled += [int(row) for row in select_gathered_chunks(gathered, keep, WORLD_SIZE)]

    assert reassembled == list(range(DATASET_SIZE))


def test_dp_representative_ranks_keeps_the_first_holder_in_dp_order():
    # tp2 on 8 ranks: dp rank r is held by global ranks 2r and 2r+1.
    assert dp_representative_ranks([0, 0, 1, 1, 2, 2, 3, 3]) == [0, 2, 4, 6]
    # A layout whose DP ranks are neither contiguous nor sorted by global rank (ETP dispatch coords).
    assert dp_representative_ranks([0, 2, 1, 3, 0, 2, 1, 3]) == [0, 2, 1, 3]
    # No siblings: the gather is already in DP order and the selection is the identity.
    assert dp_representative_ranks(list(range(WORLD_SIZE))) == list(range(WORLD_SIZE))


def test_select_gathered_chunks_rejects_a_non_world_multiple():
    with pytest.raises(ValueError, match="not a multiple of world_size"):
        select_gathered_chunks(torch.arange(7), [0, 1], 4)


def test_data_seed_owns_the_shuffle_order_not_the_ambient_torch_seed():
    """``data_seed`` must reach the seedable sampler: same seed → same order under any global seed."""
    config = _parallelism_config(0, tp_size=2)

    torch.manual_seed(0)
    order = _rank_rows(_Trainer(config, data_seed=1234), shuffle=True)
    torch.manual_seed(999)
    same_seed_other_global = _rank_rows(_Trainer(config, data_seed=1234), shuffle=True)
    assert same_seed_other_global == order, "order must follow data_seed, not the ambient torch seed"

    torch.manual_seed(0)
    other_seed = _rank_rows(_Trainer(config, data_seed=4321), shuffle=True)
    assert other_seed != order, "a different data_seed must produce a different order"

    # Sanity: shuffling actually happened, so the assertions above are not comparing sorted ranges.
    assert order != sorted(order)


def test_data_seed_unset_falls_back_to_the_ambient_torch_seed():
    """No data_seed keeps accelerate's documented fallback — the sharding must not hardcode a seed."""
    config = _parallelism_config(0, tp_size=2)

    torch.manual_seed(0)
    order = _rank_rows(_Trainer(config), shuffle=True)
    torch.manual_seed(0)
    assert _rank_rows(_Trainer(config), shuffle=True) == order
    torch.manual_seed(999)
    assert _rank_rows(_Trainer(config), shuffle=True) != order


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
