"""Offline GRPO's DP-sharded group sampler and the train and eval loaders built on it."""

import logging
import random
from collections import defaultdict
from functools import partial

import datasets
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler
from transformers.trainer_utils import seed_worker

from halo.runtime.consensus import rank_min
from halo.runtime.ranks import get_global_world_size, is_global_main_process
from halo.trainers.dataloaders import resolve_data_seed

logger = logging.getLogger(__name__)


class MultiGroupSampler(Sampler):
    """Per-completion indices, flattened group by group and cut contiguously across DP ranks (remainder to
    the first ranks), then shuffled with a rank-independent seed.

    A group may straddle ranks: the advantages are precomputed at tokenization and the ``1/group_size``
    loss weight rides each row, so group locality is not what makes the objective correct. Per-rank counts
    can differ; :class:`GroupedLoaderMixin` equalizes them.
    """

    def __init__(self, group_ids, rank, world_size, shuffle=True, seed=0):
        self.shuffle = shuffle
        # Rank-independent seed: the ranks sharing one DP slice (TP/ETP/CP siblings, a pipeline chain) must
        # iterate it in identical order.
        self.seed = seed
        self._epoch = 0
        groups = defaultdict(list)
        for idx, gid in enumerate(group_ids):
            groups[gid].append(idx)
        indices = [idx for group in groups.values() for idx in group]
        if world_size > 1:
            per_rank, remainder = divmod(len(indices), world_size)
            start = rank * per_rank + min(rank, remainder)
            indices = indices[start : start + per_rank + (1 if rank < remainder else 0)]
        if shuffle:
            random.Random(seed).shuffle(indices)
        self.indices_sequence = indices
        if is_global_main_process():
            logger.info(
                f"MultiGroupSampler (rank {rank}): "
                f"Created sequence of {len(self.indices_sequence)} indices from {len(groups)} groups"
            )

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


class GroupedLoaderMixin:
    """Offline GRPO's train and eval loaders over :class:`MultiGroupSampler`; mixed in ahead of the
    spine's DP loader mixin, whose loaders these replace."""

    def _cache_group_ids(self, train_dataset, eval_dataset) -> None:
        """Read each split's ``group_id`` column once, for the samplers every loader build takes."""
        if self.accelerator.is_main_process:
            logger.info("Extracting group_ids from training dataset...")
        self._cached_train_group_ids = list(train_dataset["group_id"])
        if self.accelerator.is_main_process:
            logger.info(
                f"Training dataset info: {len(train_dataset)} examples, {len(set(self._cached_train_group_ids))} unique groups"
            )

        if eval_dataset is not None:
            if self.accelerator.is_main_process:
                logger.info("Extracting group_ids from evaluation dataset...")
            self._cached_eval_group_ids = list(eval_dataset["group_id"])
            if self.accelerator.is_main_process:
                logger.info(
                    f"Eval dataset info: {len(eval_dataset)} examples, {len(set(self._cached_eval_group_ids))} unique groups"
                )
        else:
            self._cached_eval_group_ids = None

    def get_train_dataloader(self) -> DataLoader:
        """The training loader over the shuffled grouped sampler."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        dataloader = self._build_grouped_dataloader(
            self.train_dataset,
            self._cached_train_group_ids,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
        )
        return self._prepare_dataloader(dataloader, num_processes=1, process_index=0)

    def get_eval_dataloader(self, eval_dataset: str | Dataset | None = None) -> DataLoader:
        """The evaluation loader over the unshuffled grouped sampler, cached per split."""

        def build(dataset):
            group_ids = self._cached_eval_group_ids if dataset is self.eval_dataset else None
            if group_ids is None and dataset is not None and "group_id" in dataset.column_names:
                group_ids = list(dataset["group_id"])
            eval_dataloader = self._build_grouped_dataloader(
                dataset, group_ids, batch_size=self.args.per_device_eval_batch_size, shuffle=False
            )
            # Already sharded by the sampler; num_processes=1 avoids accelerator re-sharding.
            return self._prepare_dataloader(eval_dataloader, num_processes=1, process_index=0)

        if eval_dataset is None or isinstance(eval_dataset, str) or eval_dataset is self.eval_dataset:
            return self._cached_eval_dataloader(eval_dataset, build)
        if isinstance(eval_dataset, datasets.Dataset):
            # Not the constructor's "eval" split: keyed by its fingerprint, which an attached reference
            # column derives from the scores' digest.
            return self._cached_eval_dataloader(eval_dataset, build, key=f"eval/{eval_dataset._fingerprint}")
        return build(eval_dataset)

    def _build_grouped_dataloader(self, dataset, group_ids, *, batch_size: int, shuffle: bool) -> DataLoader:
        """COLLECTIVE when the world has peers. A DataLoader over this DP rank's :class:`MultiGroupSampler` slice
        (TP/ETP/CP siblings see the same batch), its batch count equalized to the world minimum, else
        faster ranks exit and block the next all-reduce. The sampler already shards, so accelerate places
        devices only."""
        dp_size, dp_rank = self.dp_shard_geometry()

        dataloader_params = {
            "collate_fn": self._loader_collate_fn(self.data_collator),
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            sampler = MultiGroupSampler(
                group_ids=group_ids,
                rank=dp_rank,
                world_size=dp_size,
                shuffle=shuffle,
                seed=resolve_data_seed(self.args) if shuffle else 0,
            )
            if get_global_world_size() > 1:
                # compute_loss gathers every batch: a rank with an extra batch blocks its peers.
                (min_batches,) = rank_min([len(sampler) // batch_size])
                if min_batches == 0:
                    raise ValueError(
                        f"Batch-count equalization left 0 full batches on the smallest rank "
                        f"(batch_size={batch_size}, world min over ranks) — the dataset split is too "
                        f"small for this world size; eval/training would silently run on nothing."
                    )
                if min_batches * batch_size < len(sampler.indices_sequence):
                    sampler.indices_sequence = sampler.indices_sequence[: min_batches * batch_size]
            dataloader_params["batch_sampler"] = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=True)
            if self.args.dataloader_num_workers > 0:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=dp_rank
                )
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        else:
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return DataLoader(dataset, **dataloader_params)
