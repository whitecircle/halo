"""Data-parallel-aware train, eval and test loaders for TP/CP/ETP/PP and pre-sharded datasets.

The base ``Trainer`` shards by world size, wrong when model-parallel siblings need identical slices or the
dataset is pre-sharded per DP rank; this passes the DP size/rank to ``prepare_data_loader``, equalizes
pre-sharded lengths and stamps the epoch on the running loader.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

import datasets
import torch
import torch.distributed as dist
from accelerate.data_loader import BatchSamplerShard, prepare_data_loader
from accelerate.utils import recursively_apply
from torch.utils.data import DataLoader
from transformers.trainer_utils import seed_worker

from src.hardware import current_device
from src.runtime.collectives import all_gather_cat
from src.runtime.consensus import rank_consensus, rank_min
from src.runtime.ranks import get_global_world_size, has_process_group, is_global_main_process, is_multi_rank_run
from src.trainers.parallelism.trainer_rules import reject_batch_rebalance_on_dp_sharded_loader, reject_dispatch_batches

logger = logging.getLogger(__name__)

# "absent from the instance __dict__", distinct from a legitimately stored ``None``.
_UNSET = object()


def find_batch_sampler_shard(loader) -> BatchSamplerShard | None:
    """The ``BatchSamplerShard`` accelerate planted under ``loader`` (as ``batch_sampler``, or as ``sampler``
    over a caller's batch sampler); ``None`` for a one-process loader or an iterable dataset."""
    for candidate in (getattr(loader, "batch_sampler", None), getattr(loader, "sampler", None)):
        if isinstance(candidate, BatchSamplerShard):
            return candidate
    return None


def dp_representative_ranks(dp_rank_by_global_rank: Sequence[int]) -> list[int]:
    """Global ranks whose chunks make a world gather one chunk per DP rank, in DP-rank order: the first
    holder of each DP rank, since its TP/CP/ETP siblings contribute byte-identical duplicates. That turns
    the world-order concatenation back into the dataset order a DP-sharded loader produced."""
    first_holder: dict[int, int] = {}
    for global_rank, dp_rank in enumerate(dp_rank_by_global_rank):
        first_holder.setdefault(dp_rank, global_rank)
    return [first_holder[dp_rank] for dp_rank in sorted(first_holder)]


def _select_gathered_chunks(tensor: torch.Tensor, keep: Sequence[int], world_size: int) -> torch.Tensor:
    """Keep only ``keep``'s equal-sized dim-0 chunks of a world-gathered tensor, in ``keep`` order."""
    if tensor.shape[0] % world_size != 0:
        raise ValueError(
            f"Gathered tensor has {tensor.shape[0]} rows, not a multiple of world_size {world_size}: "
            f"it did not come from a world all-gather of equal per-rank chunks, so the "
            f"data-parallel chunks cannot be identified."
        )
    chunks = tensor.reshape(world_size, -1, *tensor.shape[1:])
    return chunks[keep].reshape(-1, *tensor.shape[1:])


@contextlib.contextmanager
def _instance_override(obj, name: str, value):
    """Shadow ``obj.name`` with ``value`` for the block, then restore exactly what ``obj``'s instance
    ``__dict__`` held: the earlier instance attribute, or none, so the class attribute shows again."""
    saved = obj.__dict__.get(name, _UNSET)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if saved is _UNSET:
            delattr(obj, name)
        else:
            setattr(obj, name, saved)


@contextlib.contextmanager
def dp_scoped_gather(accelerator, keep: Sequence[int], world_size: int):
    """Bind ``accelerator.gather`` to keep one chunk per DP rank for the block. The gather stays a world
    collective; only its result is cut back to the chunks a DP-sharded loader produced."""
    original = accelerator.gather

    def _gather(input_data):
        return recursively_apply(
            partial(_select_gathered_chunks, keep=keep, world_size=world_size), original(input_data)
        )

    with _instance_override(accelerator, "gather", _gather):
        yield


def set_sampler_epoch(dataloader: DataLoader, epoch: int) -> int:
    """Set ``epoch`` on every sampler under ``dataloader``; returns how many took it.

    accelerate probes for the epoch-seeded sampler at fixed depths, and a mid-epoch resume's
    ``SkipBatchSampler`` adds a level a dispatching loader (or any chain accelerate did not re-plant)
    falls out of, so the resumed epoch would redraw epoch 0's permutation. Walking ``.batch_sampler``
    and ``.sampler`` finds the sampler at whatever depth the wrappers leave it.
    """
    pending = [dataloader]
    visited = {id(dataloader)}
    applied = 0
    while pending:
        parent = pending.pop()
        for attr in ("batch_sampler", "sampler"):
            node = getattr(parent, attr, None)
            if node is None or id(node) in visited:
                continue
            visited.add(id(node))
            set_epoch = getattr(node, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch)
                applied += 1
            pending.append(node)
    return applied


def resolve_data_seed(args) -> int:
    """The seed a run's data order derives from: ``data_seed`` when set, else ``seed`` (HF's convention)."""
    return args.data_seed if args.data_seed is not None else args.seed


def uses_dp_sharded_loader(parallelism_config, dataset_presharded: bool) -> bool:
    """True when DP < world_size (TP/CP/ETP/PP) or the dataset is pre-sharded per DP rank.

    EP alone keeps the base flow (EP is orthogonal to DP) unless pre-sharded, where accelerate's
    re-sharding would drop most of a disjoint slice. accelerate shards by GLOBAL rank, which would hand
    the stages of one pipeline chain different rows, so the last stage would score another row set's labels.
    """
    return parallelism_config.non_dp_replication_factor > 1 or dataset_presharded


class DataParallelLoaderMixin:
    """Train, eval and test loaders sharded over the data-parallel replicas. Mixed into the trainer."""

    def dp_shard_geometry(self) -> tuple[int, int]:
        """``(size, rank)`` for splitting a dataset across DP replicas; ``(1, 0)`` for a pre-sharded one,
        already this rank's disjoint slice."""
        if self._dataset_presharded:
            return 1, 0
        return self.parallelism_config.data_parallel_size, self.parallelism_config.get_data_parallel_rank()

    def get_train_dataloader(self) -> DataLoader:
        """The train loader, sharded over the DP replicas where the run shares batches across ranks."""
        if not self._needs_custom_dataloader():
            return super().get_train_dataloader()

        if self.train_dataset is None:
            raise ValueError("Training requires a train_dataset.")
        reject_batch_rebalance_on_dp_sharded_loader(self.args)

        # Unequal per-rank lengths run different step counts and hang on the next all-reduce.
        if self._dataset_presharded:
            self.train_dataset = self._equalize_presharded_length(self.train_dataset)

        train_dataset, params = self._loader_params(self.train_dataset, self._train_loader_batch_size(), "training")

        if self.parallelism_config.is_pp_mode and not self.args.dataloader_drop_last:
            # PP's P2P buffer shapes freeze on step 1, so a short final batch crashes mid-epoch.
            if is_global_main_process():
                logger.info("Pipeline parallelism: forcing train dataloader drop_last=True (frozen P2P batch shape).")
            params["drop_last"] = True

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            params["sampler"] = self._get_train_sampler()
            params.setdefault("drop_last", self.args.dataloader_drop_last)
            if self.args.dataloader_num_workers > 0:
                # Seed by DP rank, not global rank: TP/CP siblings must not desync on a stochastic transform.
                params["worker_init_fn"] = partial(
                    seed_worker,
                    num_workers=self.args.dataloader_num_workers,
                    rank=self.parallelism_config.get_data_parallel_rank(),
                )

        return self._prepare_dataloader(DataLoader(train_dataset, **params))

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """The eval loader, sharded over the DP replicas where the run shares batches across ranks."""
        if not self._needs_custom_dataloader():
            with self._eval_batch_size_as(self._eval_loader_batch_size()):
                return super().get_eval_dataloader(eval_dataset=eval_dataset)
        # Whether the caller named a split, decided before the cache resolves it to a dataset.
        caller_supplied = eval_dataset is not None

        def build(dataset):
            # Collective: unequal per-rank lengths hang the metrics gather.
            if self._dataset_presharded:
                dataset = self._equalize_presharded_length(dataset, split="eval")
                if not caller_supplied:
                    self.eval_dataset = dataset
            return self._dp_eval_loader(dataset, self._eval_loader_batch_size(), "evaluation")

        return self._cached_eval_dataloader(eval_dataset, build)

    def get_test_dataloader(self, test_dataset) -> DataLoader:
        """The ``predict()`` loader, DP-sharded like the eval one (HF sizes it ``eval_batch_size``)."""
        if not self._needs_custom_dataloader():
            return super().get_test_dataloader(test_dataset)
        if self._dataset_presharded:
            test_dataset = self._equalize_presharded_length(test_dataset, split="test")
        return self._dp_eval_loader(test_dataset, self.args.eval_batch_size, "test")

    @contextlib.contextmanager
    def data_parallel_sweep(self):
        """COLLECTIVE. Pin a third-party accelerate sweep, its loader and its gather, to the DP axis.

        A sweep that prepares its own loader and reassembles the per-rank results with
        ``accelerator.gather`` (TRL's ``precompute_ref_log_probs``) keys both on the global rank, handing
        TP/CP/ETP siblings different rows and carrying one duplicate per sibling. Routing the loader
        through ``_prepare_dataloader`` and keeping one chunk per DP rank restores the dataset order.
        """
        accelerator = self.accelerator
        dp_rank_by_global_rank = self._data_parallel_rank_by_global_rank()

        def _dp_prepare_data_loader(data_loader, device_placement=None, slice_fn_for_dispatch=None):
            return self._prepare_dataloader(data_loader)

        with (
            _instance_override(accelerator, "prepare_data_loader", _dp_prepare_data_loader),
            dp_scoped_gather(
                accelerator, dp_representative_ranks(dp_rank_by_global_rank), len(dp_rank_by_global_rank)
            ),
        ):
            yield

    def _set_signature_columns_if_needed(self):
        """Union the collator's ``required_dataset_columns`` into HF's signature-column set, so a collator
        input that is no model-forward parameter (packing's ``seq_lengths``) survives column pruning."""
        super()._set_signature_columns_if_needed()
        if self._signature_columns:
            for column in getattr(getattr(self, "data_collator", None), "required_dataset_columns", ()):
                if column not in self._signature_columns:
                    self._signature_columns.append(column)

    def _run_epoch(self, model, epoch, train_dataloader, *args, **kwargs):
        """Stamp ``epoch`` onto the loader's whole sampler chain (:func:`set_sampler_epoch`), which the base's
        own probes can miss on a resumed chain, then run the base's epoch unchanged."""
        set_sampler_epoch(train_dataloader, epoch)
        return super()._run_epoch(model, epoch, train_dataloader, *args, **kwargs)

    def _needs_custom_dataloader(self) -> bool:
        """:func:`uses_dp_sharded_loader` for this trainer's run."""
        return uses_dp_sharded_loader(self.parallelism_config, self._dataset_presharded)

    def _equalize_presharded_length(self, dataset, split: str = "train"):
        """COLLECTIVE. Truncate a pre-sharded dataset to the global-minimum length."""
        if not is_multi_rank_run():
            return dataset
        try:
            n = len(dataset)
        except TypeError:
            return dataset  # IterableDataset
        (min_len,) = rank_min([n])
        if min_len == 0:
            if split == "train":
                raise ValueError(
                    "Pre-sharded dataset: at least one data-parallel rank holds zero examples "
                    "(num_shards < data_parallel_size). Re-preprocess with --num_shards >= the "
                    "data-parallel degree, or use a non-sharded dataset."
                )
            raise ValueError(
                f"Pre-sharded {split} dataset: at least one data-parallel rank holds zero {split} examples "
                f"(the {split} split has fewer non-empty shards than data_parallel_size — sharding skips "
                f"empty shards, so a small split produces fewer of them). Unequal {split} lengths make "
                f"ranks run different step counts (NCCL hang at the metrics gather) or report wrong "
                f"metrics. Re-preprocess the {split} split with at least data_parallel_size non-empty "
                f"shards{', or disable evaluation' if split == 'eval' else ''}."
            )
        if min_len < n:
            dropped = n - min_len
            if is_global_main_process():
                logger.warning(
                    f"⚠ Pre-sharded {split} dataset DATA LOSS: this rank holds {n} examples but is truncated to "
                    f"the global min {min_len} ({dropped} examples / {100.0 * dropped / n:.1f}% of this rank "
                    f"DROPPED each pass over the split) so all data-parallel ranks run the same number of steps. "
                    f"Cause: num_shards is not a multiple of data_parallel_size, so some ranks got more shards. "
                    f"Re-preprocess with num_shards a multiple of the DP degree (ideally >> it, e.g. "
                    f"k×world_size), so the truncation is even and negligible."
                )
            return dataset.select(range(min_len))
        return dataset

    def _loader_params(self, dataset, batch_size: int, description: str) -> tuple[Any, dict]:
        """Column-pruned dataset plus the ``DataLoader`` kwargs the train, eval and test paths share (HF
        ``_get_dataloader``'s); sampler, ``drop_last`` and worker seeding stay with each caller."""
        collator = self.data_collator
        # Pruning reads column_names, which only datasets.Dataset has.
        if isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description=description)
        else:
            collator = self._get_collator_with_removed_columns(collator, description=description)

        params = {
            "batch_size": batch_size,
            "collate_fn": collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "multiprocessing_context": self.args.dataloader_multiprocessing_context,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
            "in_order": self.args.dataloader_in_order,
        }
        return dataset, params

    def _train_loader_batch_size(self) -> int:
        """Rows one train-loader fetch returns. GRPO overrides it to draw a whole generation round."""
        return self._train_batch_size

    def _get_train_sampler(self, *args, **kwargs):
        """The base's train sampler, given a generator seeded with the run's data seed where its slot is empty.

        An empty slot shuffles off the rank's own torch RNG, which pipeline stages and EP ranks consume
        differently, while every rank of one DP split must draw the same permutation; accelerate's seedable
        sampler restores that by itself, but never wraps HF's group-by-length sampler.
        """
        sampler = super()._get_train_sampler(*args, **kwargs)
        if getattr(sampler, "generator", _UNSET) is None:
            sampler.generator = torch.Generator().manual_seed(resolve_data_seed(self.args))
        return sampler

    def _prepare_dataloader(
        self, dataloader: DataLoader, *, num_processes: int | None = None, process_index: int | None = None
    ) -> DataLoader:
        """Prepare ``dataloader`` sharded over the DP replicas; ``num_processes``/``process_index`` override
        the geometry (offline GRPO's self-sharding sampler passes 1/0 for device placement only).

        accelerate's dispatching loader (its default for an iterable dataset) slices rank 0's batches by
        GLOBAL rank, handing TP/CP/ETP siblings and pipeline peers different rows, so a loader sharded
        fewer ways than the world never dispatches: an iterable dataset is sharded by DP rank instead.
        """
        if getattr(dataloader, "_is_accelerate_prepared", False):
            return dataloader

        if num_processes is not None:
            data_parallel_size = num_processes
            data_parallel_rank = process_index or 0
        else:
            # A pre-sharded dataset reports (1, 0): device placement only, no re-sharding.
            data_parallel_size, data_parallel_rank = self.dp_shard_geometry()

        dispatch_batches = self.accelerator.dispatch_batches
        world_size = get_global_world_size()
        reject_dispatch_batches(dispatch_batches, data_parallel_size, world_size)
        if data_parallel_size != world_size:
            dispatch_batches = False

        prepared = prepare_data_loader(
            dataloader,
            self.accelerator.device,
            num_processes=data_parallel_size,
            process_index=data_parallel_rank,
            split_batches=self.accelerator.split_batches,
            put_on_device=True,
            rng_types=self.accelerator.rng_types.copy(),
            dispatch_batches=dispatch_batches,
            even_batches=self.accelerator.even_batches,
            use_seedable_sampler=self.accelerator.use_seedable_sampler,
            # Without it the seedable sampler falls back to the ambient torch seed (data_seed no-op).
            data_seed=self.accelerator.dataloader_config.data_seed,
            non_blocking=self.accelerator.non_blocking,
            use_stateful_dataloader=self.accelerator.use_stateful_dataloader,
        )
        prepared._is_accelerate_prepared = True
        return prepared

    @contextlib.contextmanager
    def _eval_batch_size_as(self, rows: int):
        """Present ``rows`` as the per-device eval batch while the base ``Trainer`` builds a loader, whose
        builder takes no batch-size argument; every other reader sees the configured value."""
        per_device = self.args.per_device_eval_batch_size
        self.args.per_device_eval_batch_size = rows
        try:
            yield
        finally:
            self.args.per_device_eval_batch_size = per_device

    def _eval_loader_batch_size(self) -> int:
        """Rows one eval-loader fetch returns. Env-GRPO overrides it — it draws a whole rollout round."""
        return self.args.eval_batch_size

    def _cached_eval_dataloader(self, eval_dataset, build, key: str | None = None) -> DataLoader:
        """Resolve the eval split, then hand it to ``build``, or reuse the prepared loader already built.

        The persistent-worker cache holds the PREPARED loader: rebuilding one per ``evaluate()`` leaks a
        worker pool. The key is the split name, so a hit is the same on every rank and ``build`` (which
        issues collectives) stays in lockstep. An explicit ``key`` may be process-local (a fingerprint a
        non-picklable transform randomized), so it hits only when every rank holds it; a rank that held it
        keeps its own loader, since one replaced in the cache would keep its worker pool alive.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Evaluation requires an eval_dataset.")

        persistent = self.args.dataloader_persistent_workers
        if key is None:
            key = eval_dataset if isinstance(eval_dataset, str) else "eval"
            hit = persistent and key in self._eval_dataloaders
        else:
            hit = persistent and rank_consensus(key in self._eval_dataloaders).all_ok
        if hit:
            return self._eval_dataloaders[key]

        if isinstance(eval_dataset, str):
            eval_dataset = self.eval_dataset[eval_dataset]
        prepared = build(eval_dataset if eval_dataset is not None else self.eval_dataset)
        if persistent:
            return self._eval_dataloaders.setdefault(key, prepared)
        return prepared

    def _dp_eval_loader(self, dataset, batch_size: int, description: str) -> DataLoader:
        """The eval sampler's DP-sharded, prepared loader over ``dataset``, for the eval and test paths. An
        iterable split has no length to count its final round's padding by, so this warns."""
        dataset, params = self._loader_params(dataset, batch_size, description)
        if not isinstance(dataset, torch.utils.data.IterableDataset):
            params["sampler"] = self._get_eval_sampler(dataset)
            params["drop_last"] = self.args.dataloader_drop_last
        elif (dp_size := self.dp_shard_geometry()[0]) > 1 and is_global_main_process():
            logger.warning(
                f"The {description} split is an iterable dataset sharded {dp_size} ways by data-parallel rank: "
                f"its final round is padded with the split's first rows, which it has no length to count, so "
                f"its metrics count up to {dp_size * batch_size - 1} repeated rows. A map-style split is cut "
                f"exactly."
            )
        return self._prepare_dataloader(DataLoader(dataset, **params))

    def _data_parallel_rank_by_global_rank(self) -> list[int]:
        """COLLECTIVE. The DP rank of every global rank, indexed by global rank."""
        dp_rank = self.parallelism_config.get_data_parallel_rank()
        if not has_process_group():
            return [dp_rank]
        local = torch.tensor([dp_rank], dtype=torch.long, device=current_device())
        return all_gather_cat(local, 0, None, dist.get_world_size()).tolist()
