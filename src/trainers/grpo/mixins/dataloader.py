"""Train-dataloader machinery for the GRPO trainers.

:class:`GRPOTrainDataLoaderMixin` sizes the shared ``DataParallelDataLoaderMixin`` loader to the
GRPO train batch (``steps_per_generation`` micro-batches per generation round) and rebuilds the
sampler to match.

:class:`MultiGroupSampler` is offline GRPO's own sampler: it flattens the completions group by group
in dataset order and cuts the DP slice POSITIONALLY, so a group may straddle ranks — the advantages
are precomputed at tokenization and the ``1/group_size`` loss weight rides each row, so group
locality is not what makes the objective correct.
"""

from __future__ import annotations

import random
from collections import defaultdict

from accelerate.logging import get_logger
from torch.utils.data import Dataset, Sampler
from trl.trainer.utils import RepeatSampler

logger = get_logger(__name__, log_level="INFO")


class GRPOTrainDataLoaderMixin:
    """GRPO's train-loader geometry. Mixed in ahead of ``DistributedTrainerMixin``."""

    def _train_loader_batch_size(self) -> int:
        """One generation round per fetch: ``steps_per_generation`` micro-batches are drawn together."""
        return self._train_batch_size * self.args.steps_per_generation

    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler:
        """``RepeatSampler`` sized to the custom DP-sharded dataloader's actual consumption rate.

        TRL derives the sampler geometry from ``generation_batch_size`` (world-rate: one full
        prompt block consumed per generation round across ``accelerator.num_processes``). The
        custom dataloader consumes only ``data_parallel_size`` batches per fetch (TP/ETP siblings
        replay the same DP slice), so under the world-rate geometry each generation round re-rolls
        only the leading ``dp/world`` fraction of every prompt block and silently skips the rest.
        Rebuild the sampler from the loader's real per-round consumption; pre-sharded datasets
        consume per DP rank (each rank iterates its own disjoint shard), so the rate is one rank's.
        """
        if not self._needs_custom_dataloader():
            return super()._get_train_sampler(dataset)
        if dataset is None:
            dataset = self.train_dataset
        consumers = 1 if self._dataset_presharded else self.get_data_parallel_size()
        per_rank_rows = self._train_loader_batch_size()
        completions_per_round = per_rank_rows * consumers
        if completions_per_round % self.num_generations != 0:
            raise ValueError(
                f"The DP-rate generation batch ({completions_per_round} = per_device_train_batch_size "
                f"({self._train_batch_size}) * steps_per_generation ({self.args.steps_per_generation}) "
                f"* {'1 (pre-sharded per DP rank)' if self._dataset_presharded else f'data_parallel_size ({consumers})'}) "
                f"must be divisible by num_generations ({self.num_generations}), so every generation "
                f"round covers whole prompt groups."
            )
        if self.accelerator.num_processes != consumers and per_rank_rows % self.num_generations != 0:
            # TP/ETP siblings duplicate their DP slice into TRL's world-order gather; only whole groups regroup.
            raise ValueError(
                f"With TP/ETP active, the per-rank generation rows (per_device_train_batch_size "
                f"({self._train_batch_size}) * steps_per_generation ({self.args.steps_per_generation}) "
                f"= {per_rank_rows}) must be divisible by num_generations ({self.num_generations}): "
                f"a prompt group spanning DP ranks breaks TRL's world-gathered advantage grouping "
                f"once TP siblings inject duplicate blocks."
            )
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=completions_per_round // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )


class MultiGroupSampler(Sampler):
    """Per-completion indices, flattened group-by-group, split across DP ranks (remainder to first ranks).

    Per-rank counts can differ; callers must equalize (see ``_build_grouped_dataloader``).
    """

    def __init__(self, group_ids, rank, world_size, shuffle=True, seed=0):
        self.group_ids = group_ids
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        # Fixed seed: TP/CP siblings share one DP slice and must iterate it in identical order.
        self.seed = seed
        self._epoch = 0

        self.groups = defaultdict(list)
        for idx, gid in enumerate(group_ids):
            self.groups[gid].append(idx)

        self.indices_sequence = []
        self._create_indices_sequence()

        logger.info(
            f"MultiGroupSampler (rank {self.rank}): "
            f"Created sequence of {len(self.indices_sequence)} indices from {len(self.groups)} groups"
        )

    def _create_indices_sequence(self):
        """Build the per-rank index sequence: contiguous rank split, then seeded shuffle."""
        for indices in self.groups.values():
            self.indices_sequence.extend(indices)

        if self.world_size > 1:
            total_indices = len(self.indices_sequence)
            indices_per_rank = total_indices // self.world_size
            remainder = total_indices % self.world_size

            start_idx = self.rank * indices_per_rank + min(self.rank, remainder)
            end_idx = start_idx + indices_per_rank + (1 if self.rank < remainder else 0)

            self.indices_sequence = self.indices_sequence[start_idx:end_idx]

        if self.shuffle:
            random.Random(self.seed).shuffle(self.indices_sequence)

    def set_epoch(self, epoch: int) -> None:
        """Re-seed the per-iteration shuffle for a new epoch (DistributedSampler convention)."""
        self._epoch = epoch

    def __iter__(self):
        indices = self.indices_sequence.copy()
        if self.shuffle:
            # +1 offset so epoch 0 differs from the init shuffle.
            random.Random(self.seed + self._epoch + 1).shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.indices_sequence)
