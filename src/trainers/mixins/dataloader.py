"""Data-parallel-aware train/eval dataloaders for TP/CP/ETP and pre-sharded datasets.

Base ``Trainer`` shards by world size, wrong when DP < world_size (TP/CP/ETP siblings need identical
slices) or the dataset is pre-sharded per DP rank. Passes parallelism-aware DP size/rank to
``accelerate.prepare_data_loader`` and equalizes pre-sharded lengths; defers to ``super()`` otherwise.
The epoch loop is intercepted here too, to stamp the epoch onto whichever loader the trainer runs,
and so is the evaluation gather, which reads the final round's padding off the loader's geometry.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from functools import partial
from typing import Any

import datasets
import torch
import torch.distributed as dist
from accelerate.data_loader import BatchSamplerShard, DataLoaderDispatcher, prepare_data_loader
from accelerate.logging import get_logger
from accelerate.utils import recursively_apply
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import seed_worker
from trl.trainer.utils import RepeatSampler

from src.distributed.runtime import current_device, get_global_world_size, is_multi_rank_run, rank_consensus

logger = get_logger(__name__, log_level="info")

# "absent from the instance __dict__", distinct from a legitimately stored ``None``.
_UNSET = object()

# Samplers whose ``len`` is the number of indices they yield, the streams a final round is counted off:
# accelerate's seedable sampler is a ``RandomSampler``, HF's group-by-length eval sampler permutes every
# index once, and TRL's ``RepeatSampler`` counts its repeats.
_COUNTED_SAMPLERS = (SequentialSampler, RandomSampler, SubsetRandomSampler, LengthGroupedSampler, RepeatSampler)


def batch_sampler_shard(loader) -> BatchSamplerShard | None:
    """The ``BatchSamplerShard`` accelerate planted under ``loader``, or ``None`` where it planted none.

    ``prepare_data_loader`` sets it as ``batch_sampler``, or as ``sampler`` when the caller's sampler
    was itself a batch sampler. A one-process loader (a pre-sharded dataset, offline GRPO's
    self-sharding sampler) and an iterable dataset get none.
    """
    for candidate in (getattr(loader, "batch_sampler", None), getattr(loader, "sampler", None)):
        if isinstance(candidate, BatchSamplerShard):
            return candidate
    return None


def final_round_split_rows(loader, dataset_drawing: tuple[type[BatchSampler], ...] = ()) -> int | None:
    """How many of ``loader``'s final-round rows, over every rank in rank order, are the split's own.

    ``None`` when no round falls short: no ``BatchSamplerShard``, or ``drop_last``. Otherwise the
    split's rows are the ones the sampler draws, modulo one round, filling it in rank order. With
    ``even_batches`` (accelerate's default) every rank's last batch is filled up with the split's
    first rows; without it the last batch is simply short, yet HF still repeats each rank's scalar
    eval loss ``eval_batch_size`` times, so its entries need the same cut. (Ranks that run unequal
    batch counts take the identity gather instead.) accelerate's own ``remainder`` counts the dataset,
    which miscounts TRL's ``RepeatSampler`` (``num_generations`` draws per prompt) and trims real rows
    from a one-process loader.

    The count is trusted only where the rows drawn are known: a batch sampler iterating as torch's
    ``BatchSampler`` does (every row of its sampler, once) over a sampler whose length is the rows it
    yields (:data:`_COUNTED_SAMPLERS`), or iterating as one of ``dataset_drawing`` does, the batch
    samplers a trainer declares to draw every row of their ``dataset`` once (sentence-transformers'
    no-duplicates sampler; exact while only its last batch is short, as accelerate's even batches
    assume). Anything else is ``None`` too, so nothing is cut: group-by-label draws fewer rows than
    its dataset holds, and a batch sampler with no ``sampler`` names no stream at all.
    """
    shard = batch_sampler_shard(loader)
    if shard is None or shard.drop_last:
        return None
    batch_sampler = shard.batch_sampler
    iterate = type(batch_sampler).__iter__
    if iterate is BatchSampler.__iter__ and isinstance(getattr(batch_sampler, "sampler", None), _COUNTED_SAMPLERS):
        drawn = len(batch_sampler.sampler)
    elif any(iterate is declared.__iter__ for declared in dataset_drawing):
        drawn = len(batch_sampler.dataset)
    else:
        return None
    return (drawn % loader.total_batch_size) or None


def split_rows_head(tensor: torch.Tensor, real_rows: int) -> torch.Tensor:
    """``tensor``'s first ``real_rows`` rows (:meth:`DataParallelDataLoaderMixin.eval_split_rows`), or
    ``tensor`` itself when that is every row: autograd records even a full-range slice, whose backward
    allocates and copies a gradient the size of the whole tensor."""
    return tensor if real_rows >= tensor.size(0) else tensor[:real_rows]


def split_rows_mean(values: torch.Tensor, real_rows: int, dim: int = 0, *, whole: bool = False) -> torch.Tensor:
    """The mean over ``values``' first ``real_rows`` rows along ``dim`` — over every element of them
    under ``whole`` — and 0 over none (a rank holding only an eval split's final-round padding).

    Every row is ``values.mean(dim)`` (``values.mean()``) itself, so a train batch reduces bit for bit
    like a plain mean.
    """
    if real_rows >= values.size(dim):
        return values.mean() if whole else values.mean(dim)
    head = values.narrow(dim, 0, real_rows)
    return head.sum() / max(head.numel(), 1) if whole else head.sum(dim) / max(real_rows, 1)


def rank_split_rows(split_rows: int, rows_per_rank: int, rank: int) -> int:
    """How many of ``rank``'s ``rows_per_rank`` final-round rows are real, the split's ``split_rows``
    filling the round in rank order."""
    return max(0, min(rows_per_rank, split_rows - rank * rows_per_rank))


def split_head(entries: int, *, split_rows: int, rows_per_rank: int, rank: int) -> int:
    """How many of ``rank``'s ``entries`` gathered final-round entries stand for its real rows: its
    share of ``rows_per_rank``, scaled to the entries and rounded up.

    A chunk need not hold one entry per row — HF repeats each rank's scalar eval loss
    ``eval_batch_size`` times, fewer than env GRPO's rollout round — and a rank holding any real row
    keeps one at least.
    """
    return -(-rank_split_rows(split_rows, rows_per_rank, rank) * entries // rows_per_rank)


def trim_to_split_rows(tensor: torch.Tensor, *, split_rows: int, rows_per_rank: int, num_ranks: int) -> torch.Tensor:
    """The split's share of a gathered final round: ``num_ranks`` equal chunks in rank order, each
    standing for ``rows_per_rank`` rows, of which the first ``split_rows`` are real. Each chunk keeps
    its own :func:`split_head`."""
    chunk, ragged = divmod(tensor.shape[0], num_ranks)
    if ragged:
        raise ValueError(
            f"Gathered tensor has {tensor.shape[0]} rows, not a multiple of the {num_ranks} data-parallel "
            f"ranks: it did not come from a gather of equal per-rank chunks, so its padding cannot be cut."
        )
    parts = tensor.reshape(num_ranks, chunk, *tensor.shape[1:])
    return torch.cat(
        [
            part[: split_head(chunk, split_rows=split_rows, rows_per_rank=rows_per_rank, rank=rank)]
            for rank, part in enumerate(parts)
        ]
    )


def _all_tensors(data) -> bool:
    """Whether ``data`` is tensors throughout, the shape accelerate's tensor gather takes."""
    try:
        recursively_apply(lambda tensor: tensor, data, error_on_other_type=True)
    except TypeError:
        return False
    return True


def dp_representative_ranks(dp_rank_by_global_rank: Sequence[int]) -> list[int]:
    """Global ranks whose chunks make a world gather one chunk per DP rank, in DP-rank order.

    A world gather concatenates one chunk per global rank; when DP < world_size the TP/CP/ETP
    siblings of a DP rank contribute byte-identical duplicates, and the chunk order follows the
    global-rank layout rather than the DP one. Keeping the first holder of each DP rank, ordered by
    DP rank, turns that concatenation back into the dataset order a DP-sharded loader produced.
    """
    first_holder: dict[int, int] = {}
    for global_rank, dp_rank in enumerate(dp_rank_by_global_rank):
        first_holder.setdefault(dp_rank, global_rank)
    return [first_holder[dp_rank] for dp_rank in sorted(first_holder)]


def select_gathered_chunks(tensor: torch.Tensor, keep: Sequence[int], world_size: int) -> torch.Tensor:
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
def dp_scoped_gather(accelerator, keep: Sequence[int], world_size: int):
    """Bind ``accelerator.gather`` to keep one chunk per DP rank for the duration of the block.

    The gather itself stays a world collective — every rank must enter it — but the concatenation it
    returns is cut back to the chunks a DP-sharded loader actually produced. An identity at
    ``dp_size == world_size``.
    """
    original = accelerator.gather
    saved = accelerator.__dict__.get("gather", _UNSET)

    def _gather(input_data):
        return recursively_apply(
            partial(select_gathered_chunks, keep=keep, world_size=world_size), original(input_data)
        )

    accelerator.gather = _gather
    try:
        yield
    finally:
        if saved is _UNSET:
            del accelerator.gather
        else:
            accelerator.gather = saved


def set_sampler_epoch(dataloader: DataLoader, epoch: int) -> int:
    """Set ``epoch`` on every sampler under ``dataloader``; returns how many took it.

    accelerate reaches the epoch-seeded sampler through fixed probe shapes (three levels for
    ``DataLoaderShard``, one for ``DataLoaderDispatcher``), and a mid-epoch resume adds a level:
    ``skip_first_batches`` wraps the chain in a ``SkipBatchSampler``. The stock DP chain survives
    that, but a dispatching loader does not, nor does any chain whose sampler accelerate did not
    re-plant; the sampler's epoch then stays 0 and the resumed epoch redraws epoch 0's permutation.
    Walking ``.batch_sampler`` and ``.sampler`` finds it at whatever depth the wrappers leave it.
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


def run_data_seed(args) -> int:
    """The seed a run's data order derives from: ``data_seed`` when set, else ``seed`` (HF's convention)."""
    return args.data_seed if args.data_seed is not None else args.seed


def seed_unseeded_sampler(sampler, seed: int):
    """Give ``sampler`` a generator seeded with ``seed`` when its ``generator`` slot is empty.

    An empty slot shuffles off the rank's own torch RNG, which pipeline stages and EP ranks consume
    differently, while every rank of one data-parallel split must draw the same permutation. Only
    accelerate's seedable sampler restores that by itself, and it never wraps HF's group-by-length
    sampler, nor syncs a one-process loader under ``use_seedable_sampler: false``. A seedable
    ``RandomSampler`` reseeds per epoch, so its order is unchanged.
    """
    if getattr(sampler, "generator", _UNSET) is None:
        sampler.generator = torch.Generator().manual_seed(seed)
    return sampler


def needs_dp_sharded_loader(parallelism_config, dataset_presharded: bool) -> bool:
    """True when DP < world_size (TP/CP/ETP/PP) or the dataset is pre-sharded per DP rank.

    EP alone uses the base flow (EP ⊥ DP) unless pre-sharded, where base accelerate re-sharding
    would drop most of an already-disjoint slice. PP must take the custom path: accelerate's
    default shards by GLOBAL rank, which would hand every rank of a pipeline chain a DIFFERENT
    batch — stage 0 then forwards one row set while the last stage scores another's labels,
    silently training on garbage pairs. ``get_data_parallel_rank`` is stage-local, so the custom
    path gives all chain members the same shard.
    """
    return parallelism_config.non_dp_replication_factor > 1 or dataset_presharded


class DataParallelDataLoaderMixin:
    """Parallelism-aware train/eval dataloaders. Mixed into the trainer."""

    # ``(keep_ranks, world_size)`` once the eval gather is DP-scoped, None while it is the world's.
    _dp_metric_gather_scope: tuple[list[int], int] | None = None
    # Raised by ``suspended_dp_metric_gather`` for the unequal-eval-batch escape hatch, whose
    # identity gather must survive the loop's re-arm.
    _dp_metric_gather_suspended: bool = False
    # Batch samplers this trainer builds whose own iteration draws every row of their ``dataset`` once;
    # a final round over one is counted off that dataset (:func:`final_round_split_rows`).
    _dataset_drawing_batch_samplers: tuple[type[BatchSampler], ...] = ()

    def dp_shard_geometry(self) -> tuple[int, int]:
        """``(size, rank)`` for splitting a dataset across DP replicas.

        A pre-sharded dataset is already this rank's disjoint slice, so it reports ``(1, 0)``;
        cutting it again would drop ``(dp-1)/dp`` of the rows.
        """
        if self._dataset_presharded:
            return 1, 0
        return self.get_data_parallel_size(), self.get_data_parallel_rank()

    def _get_train_sampler(self, *args, **kwargs):
        """The base's train sampler, rank-uniformly seeded (:func:`seed_unseeded_sampler`) on every path."""
        return seed_unseeded_sampler(super()._get_train_sampler(*args, **kwargs), run_data_seed(self.args))

    def _train_loader_batch_size(self) -> int:
        """Rows one train-loader fetch returns. GRPO overrides it to draw a whole generation round."""
        return self._train_batch_size

    def _eval_loader_batch_size(self) -> int:
        """Rows one eval-loader fetch returns. Env-GRPO overrides it — it draws a whole rollout round."""
        return self.args.eval_batch_size

    @contextlib.contextmanager
    def _eval_batch_size_as(self, rows: int):
        """Present ``rows`` as the per-device eval batch while the base ``Trainer`` builds a loader: its
        builder reads ``args.eval_batch_size`` and takes no batch-size argument. Every other reader of
        the field (the eval loss forward's chunk, the metrics) sees the configured value again after."""
        per_device = self.args.per_device_eval_batch_size
        self.args.per_device_eval_batch_size = rows
        try:
            yield
        finally:
            self.args.per_device_eval_batch_size = per_device

    def get_train_dataloader(self) -> DataLoader:
        """Create train dataloader with correct DP sharding for TP/CP modes."""
        if not self._needs_custom_dataloader():
            return super().get_train_dataloader()

        if self.train_dataset is None:
            raise ValueError("Training requires a train_dataset.")

        # Unequal per-rank lengths run different step counts and hang on the next all-reduce.
        if self._dataset_presharded:
            self.train_dataset = self._equalize_presharded_length(self.train_dataset)

        train_dataset, params = self._loader_params(self.train_dataset, self._train_loader_batch_size(), "training")

        if self.parallelism_config.is_pp_mode and not self.args.dataloader_drop_last:
            # PP's P2P buffer shapes freeze on step 1, so a short final batch crashes mid-epoch.
            logger.info("Pipeline parallelism: forcing train dataloader drop_last=True (frozen P2P batch shape).")
            params["drop_last"] = True

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            params["sampler"] = self._get_train_sampler()
            params.setdefault("drop_last", self.args.dataloader_drop_last)
            if self.args.dataloader_num_workers > 0:
                # Seed by DP rank, not global rank: TP/CP siblings must not desync on a stochastic transform.
                params["worker_init_fn"] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.get_data_parallel_rank()
                )

        return self._prepare_dataloader(DataLoader(train_dataset, **params))

    def _cached_eval_dataloader(self, eval_dataset, build, key: str | None = None) -> DataLoader:
        """Resolve the eval split, then hand it to ``build`` — or reuse the loader already built.

        The persistent-worker cache holds the PREPARED loader (base ``Trainer`` semantics):
        rebuilding one per ``evaluate()`` leaks a worker pool. Shared by both eval paths so the two
        agree on the key, on what is cached, and on the resolution order. The key is the split name,
        so a hit or a miss is the same on every rank and ``build`` — which issues collectives —
        stays in lockstep. A caller passing an explicit dataset names its loader by ``key``, which may
        be process-local (a fingerprint a non-picklable transform randomized), so its hit counts only
        when every rank holds it; a rank that held it then keeps and returns its own loader, since one
        replaced in the cache would keep its worker pool alive under accelerate's reference for the run.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Evaluation requires an eval_dataset.")

        persistent = self.args.dataloader_persistent_workers
        if key is None:
            key = eval_dataset if isinstance(eval_dataset, str) else "eval"
            hit = persistent and key in self._eval_dataloaders
        else:
            hit = persistent and rank_consensus(key in self._eval_dataloaders)[0]
        if hit:
            return self._eval_dataloaders[key]

        if isinstance(eval_dataset, str):
            eval_dataset = self.eval_dataset[eval_dataset]
        prepared = build(eval_dataset if eval_dataset is not None else self.eval_dataset)
        if persistent:
            return self._eval_dataloaders.setdefault(key, prepared)
        return prepared

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """Create eval dataloader with correct DP sharding for TP/CP modes."""
        if not self._needs_custom_dataloader():
            with self._eval_batch_size_as(self._eval_loader_batch_size()):
                return super().get_eval_dataloader(eval_dataset=eval_dataset)
        # Whether the caller named a split, decided before the cache resolves it to a dataset.
        caller_supplied = eval_dataset is not None

        def build(dataset):
            # Same hang as train (metrics gather); collective, every rank must reach it.
            if self._dataset_presharded:
                dataset = self._equalize_presharded_length(dataset, split="eval")
                if not caller_supplied:
                    self.eval_dataset = dataset
            return self._dp_eval_loader(dataset, self._eval_loader_batch_size(), "evaluation")

        return self._cached_eval_dataloader(eval_dataset, build)

    def get_test_dataloader(self, test_dataset) -> DataLoader:
        """The ``predict()`` loader, DP-sharded like the eval one: the evaluation gather it feeds is
        DP-scoped wherever the loader must be. HF's batch size for it is ``eval_batch_size``."""
        if not self._needs_custom_dataloader():
            return super().get_test_dataloader(test_dataset)
        if self._dataset_presharded:
            test_dataset = self._equalize_presharded_length(test_dataset, split="test")
        return self._dp_eval_loader(test_dataset, self.args.eval_batch_size, "test")

    def _dp_eval_loader(self, dataset, batch_size: int, description: str) -> DataLoader:
        """The eval sampler's DP-sharded, prepared loader over ``dataset``: the eval and test paths' one body.

        An iterable split sharded across DP ranks pads its final round with its first rows, and has no
        length to count them by: nothing can cut them, so its metrics count them, and this warns.
        """
        dataset, params = self._loader_params(dataset, batch_size, description)
        if not isinstance(dataset, torch.utils.data.IterableDataset):
            params["sampler"] = self._get_eval_sampler(dataset)
            params["drop_last"] = self.args.dataloader_drop_last
        elif (dp_size := self.dp_shard_geometry()[0]) > 1:
            logger.warning(
                f"The {description} split is an iterable dataset sharded {dp_size} ways by data-parallel rank: "
                f"its final round is padded with the split's first rows, which it has no length to count, so "
                f"its metrics count up to {dp_size * batch_size - 1} repeated rows. A map-style split is cut "
                f"exactly."
            )
        return self._prepare_dataloader(DataLoader(dataset, **params))

    def _run_epoch(self, model, epoch, train_dataloader, *args, **kwargs):
        """Stamp ``epoch`` onto the loader's sampler chain, then run the base's epoch unchanged.

        Every toolkit trainer reaches this seam, including the plain FSDP2/DP path whose loader is
        the base's own. The base stamps the epoch itself, but only through probes a resumed chain
        can fall out of (see :func:`set_sampler_epoch`); stamping here first is idempotent where
        those probes reach, and survives the base's ``skip_first_batches`` rebuild because the
        wrapper keeps the same sampler object.
        """
        set_sampler_epoch(train_dataloader, epoch)
        return super()._run_epoch(model, epoch, train_dataloader, *args, **kwargs)

    def _set_signature_columns_if_needed(self):
        """Union the collator's ``required_dataset_columns`` into HF's signature-column set.

        Column pruning keeps only ``_signature_columns``, and a collator input like packing's
        ``seq_lengths`` is not a model-forward parameter, so it survives pruning only if the
        trainer's signature set names it (TRL's SFT hard-codes it; the other bases do not). The
        pipeline path pins the same attribute into its column set in ``mixins/pipeline.py``.
        """
        super()._set_signature_columns_if_needed()
        if self._signature_columns:
            for column in getattr(getattr(self, "data_collator", None), "required_dataset_columns", ()):
                if column not in self._signature_columns:
                    self._signature_columns.append(column)

    def _loader_params(self, dataset, batch_size: int, description: str) -> tuple[Any, dict]:
        """Column-pruned dataset plus the ``DataLoader`` kwargs the train, eval and test paths share.

        Unused columns are dropped from the dataset when the trainer prunes datasets, else from the
        collator — the base ``Trainer`` contract. The worker kwargs are HF ``_get_dataloader``'s.
        Sampler, ``drop_last`` and worker seeding legitimately differ per split and stay with each
        caller.
        """
        collator = self.data_collator
        # Branch on the dataset type like the base does: pruning reads column_names, only datasets.Dataset has it.
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

    def _needs_custom_dataloader(self) -> bool:
        """:func:`needs_dp_sharded_loader` for this trainer's run."""
        return needs_dp_sharded_loader(self.parallelism_config, self._dataset_presharded)

    def _equalize_presharded_length(self, dataset, split: str = "train"):
        """Truncate a pre-sharded dataset to the global-minimum length. Collective: every rank must call it."""
        if not is_multi_rank_run():
            return dataset
        try:
            n = len(dataset)
        except TypeError:
            return dataset  # IterableDataset
        t = torch.tensor([n], device=current_device())
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        min_len = int(t.item())
        if min_len == 0:
            if split == "train":
                raise ValueError(
                    "Pre-sharded dataset: at least one data-parallel rank holds zero examples "
                    "(num_shards < data_parallel_size). Re-preprocess with --num-shards >= the "
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

    def _prepare_dataloader(
        self, dataloader: DataLoader, *, num_processes: int | None = None, process_index: int | None = None
    ) -> DataLoader:
        """Prepare dataloader with correct DP parameters for TP/CP modes.

        ``num_processes``/``process_index`` override DP sharding (e.g. offline GRPO's sampler already
        shards, passing 1/0 for device placement only); ``None`` resolves them from the config.

        accelerate's dispatching loader (its default for an iterable dataset) ignores both: it slices
        rank 0's batches by GLOBAL rank, handing TP/CP/ETP siblings and pipeline peers different rows
        and every rank of a pre-sharded split rank 0's. A loader sharded fewer ways than the world
        therefore never dispatches — an iterable dataset is sharded by DP rank instead — and an
        explicit ``dispatch_batches: true`` raises.
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
        if data_parallel_size != get_global_world_size():
            if dispatch_batches:
                raise ValueError(
                    f"accelerator_config.dispatch_batches: true slices rank 0's batches by global rank, but "
                    f"this loader is sharded {data_parallel_size} ways over {get_global_world_size()} ranks "
                    f"(TP/CP/ETP siblings and pipeline peers share a data-parallel shard; a pre-sharded "
                    f"dataset is each rank's own), so ranks would get the wrong rows. Set "
                    f"accelerator_config.dispatch_batches: false."
                )
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
    def data_parallel_sweep(self):
        """Pin a third-party accelerate sweep — both its loader and its gather — to the DP axis.

        A sweep that builds its own loader with ``accelerator.prepare`` and reassembles the per-rank
        results with ``accelerator.gather`` (TRL's ``precompute_ref_log_probs``) keys both on the
        global rank. Under TP/CP/ETP that hands siblings different rows, though they must forward
        identical ones, and the world-order concatenation the caller indexes by dataset position
        carries one duplicate per sibling. Routing the loader through ``_prepare_dataloader``
        restores the DP contract, and keeping one chunk per DP rank restores the dataset order. Both
        are identities at ``dp_size == world_size``.
        """
        accelerator = self.accelerator
        dp_rank_by_global_rank = self._data_parallel_rank_by_global_rank()

        def _dp_prepare_data_loader(data_loader, device_placement=None, slice_fn_for_dispatch=None):
            return self._prepare_dataloader(data_loader)

        saved = accelerator.__dict__.get("prepare_data_loader", _UNSET)
        accelerator.prepare_data_loader = _dp_prepare_data_loader
        try:
            with dp_scoped_gather(
                accelerator, dp_representative_ranks(dp_rank_by_global_rank), len(dp_rank_by_global_rank)
            ):
                yield
        finally:
            if saved is _UNSET:
                delattr(accelerator, "prepare_data_loader")
            else:
                accelerator.prepare_data_loader = saved

    def eval_split_rows(self, num_rows: int) -> int:
        """How many of this rank's ``num_rows`` batch rows are the eval split's own; the rest pad it.

        accelerate fills every rank's last eval batch by repeating the split's first rows, so all
        ranks run the same rounds; scored as well, a repeat counts its row twice. The split's rows
        fill the final round in rank order (:func:`final_round_split_rows`), and this rank's share is
        what is left of them past the ranks before it. ``num_rows`` in train mode, before the final
        round, wherever the round pads nothing, without ``even_batches`` (a short batch holds real
        rows alone, and a rank's last batch need not even sit in the final round), and on an iterable
        split, whose loader holds no ``BatchSamplerShard`` to count its padding by. Read off the
        loader, not the trainer: TP/CP/ETP siblings share their DP rank's ``process_index``, and a
        pre-sharded or one-process loader pads nothing. Per-rank, no collective.
        """
        final_round = None if self.model.training else self._active_final_round()
        if final_round is None or not final_round[2].even_batches:
            return num_rows
        split_rows, rows_per_rank, shard = final_round
        return min(num_rows, rank_split_rows(split_rows, rows_per_rank, shard.process_index))

    def _active_eval_shard(self) -> BatchSamplerShard | None:
        """The active loader's ``BatchSamplerShard``, checked against the chunks the evaluation gather holds.

        Raises on any batch, padded or not, when the shard count is not the gathered chunk count (the
        DP scope's replicas, else the world): a loader built around the DP-aware eval build would
        have its distinct rows dropped as sibling duplicates, and its padding cut off the wrong
        geometry. Per-rank, no collective.
        """
        shard = batch_sampler_shard(self.accelerator.gradient_state.active_dataloader)
        if shard is None:
            return None
        scope = self._dp_metric_gather_scope
        gathered_ranks = len(scope[0]) if scope is not None else get_global_world_size()
        if shard.num_processes != gathered_ranks:
            raise RuntimeError(
                f"The active eval loader is sharded {shard.num_processes} ways, but the evaluation gather "
                f"holds {gathered_ranks} data-parallel chunks: it was not built through the DP-aware eval "
                f"loader, so its rows cannot be gathered. Build it with get_eval_dataloader / "
                f"get_test_dataloader."
            )
        return shard

    def _active_final_round(self) -> tuple[int, int, BatchSamplerShard] | None:
        """``(split_rows, rows_per_rank, shard)`` while the active loader's padded final round runs.

        Checks the loader's geometry on every batch first (:meth:`_active_eval_shard`).
        """
        shard = self._active_eval_shard()
        state = self.accelerator.gradient_state
        if shard is None or not state.end_of_dataloader:
            return None
        loader = state.active_dataloader
        split_rows = final_round_split_rows(loader, self._dataset_drawing_batch_samplers)
        if split_rows is None:
            return None
        return split_rows, loader.total_batch_size // shard.num_processes, shard

    def _install_dp_metric_gather(self) -> None:
        """Point HF's evaluation gather at :meth:`_dp_gather_for_metrics`, DP-scoped where it must be.

        ``evaluation_loop`` gathers predictions, labels and losses over the WHOLE world, but each
        rank's eval batch is its DP REPLICA's wherever the loader is DP-sharded: TP/CP/ETP siblings
        and pipeline chain peers all return the same rows (under PP the last stage's values are
        broadcast to the whole chain, so every stage returns them). A world gather repeats every
        replica once per sibling — ``pp_size`` copies at pp>1 — so the scope keeps one chunk per DP
        rank, in DP order, before the final round's padding is cut.

        Collective: the DP-rank map is an all-gather, so every rank must reach this.
        """
        if self._needs_custom_dataloader():
            dp_rank_by_global_rank = self._data_parallel_rank_by_global_rank()
            world_size = len(dp_rank_by_global_rank)
            keep = dp_representative_ranks(dp_rank_by_global_rank)
            if len(keep) < world_size:
                self._dp_metric_gather_scope = (keep, world_size)
                logger.info(
                    "Evaluation metrics gather scoped to the %d data-parallel replicas of %d ranks "
                    "(siblings return identical rows; a world gather would repeat each replica %dx).",
                    len(keep),
                    world_size,
                    world_size // len(keep),
                )
        self.gather_function = self._dp_gather_for_metrics

    def _rearm_dp_metric_gather(self) -> None:
        """Re-arm :meth:`_dp_gather_for_metrics` for the evaluation loop about to run.

        ``Trainer.evaluation_loop`` resets ``gather_function`` to ``accelerator.gather_for_metrics``
        on its way out, so the install done once at construction covers the first loop only — every
        later evaluate()/predict() would be back on accelerate's trim, and on a world gather where
        the loader is DP-sharded. Per-rank and derived from state the install already agreed on, so
        no collective repeats here.
        """
        if not self._dp_metric_gather_suspended:
            self.gather_function = self._dp_gather_for_metrics

    @contextlib.contextmanager
    def suspended_dp_metric_gather(self):
        """Hold whatever gather is installed across the evaluation loops run inside the block.

        Saved and restored, not forced back to False: ``Trainer.evaluate`` re-enters itself once per
        split of a dict ``eval_dataset``, and an inner exit would otherwise lift an outer suspension
        that is still running.
        """
        was_suspended = self._dp_metric_gather_suspended
        self._dp_metric_gather_suspended = True
        try:
            yield
        finally:
            self._dp_metric_gather_suspended = was_suspended

    def _dp_gather_for_metrics(self, input_data, use_gather_object: bool | None = None):
        """The evaluation gather: one chunk per DP rank, then the final round's padding cut away.

        The cut keeps the split's rows by the loader's own geometry (:func:`final_round_split_rows`)
        rather than accelerate's ``remainder``, and runs after the DP scope drops the siblings'
        duplicates, since it keeps a prefix. A scalar loss cannot shed the padding inside one rank's
        batch: a half-padded rank keeps its share of entries, so its loss still averages over the
        rows it ran. ``use_gather_object`` (default ``args.eval_use_gather_object``) and an input
        that is not all tensors (a ``preprocess_logits_for_metrics`` returning objects) take the
        object gather, under the same scope and cut (:meth:`_gather_objects_for_metrics`). A
        dispatching loader (an iterable split on plain DP, which measures its own final batch) takes
        accelerate's ``gather_for_metrics`` as it is; under a DP scope it raises, its siblings having
        drawn different rows.
        """
        if use_gather_object is None:
            use_gather_object = self.args.eval_use_gather_object
        if isinstance(self.accelerator.gradient_state.active_dataloader, DataLoaderDispatcher):
            if self._dp_metric_gather_scope is not None:
                raise RuntimeError(
                    "The active eval loader dispatches rank 0's batches by global rank, but the evaluation "
                    "gather keeps one chunk per data-parallel replica: TP/CP/ETP siblings were handed "
                    "different rows. Build it with get_eval_dataloader / get_test_dataloader, which shard "
                    "an iterable split by data-parallel rank."
                )
            return self.accelerator.gather_for_metrics(input_data, use_gather_object=use_gather_object)
        final_round = self._active_final_round()
        if use_gather_object or not _all_tensors(input_data):
            return self._gather_objects_for_metrics(input_data, final_round)
        scope = self._dp_metric_gather_scope
        with dp_scoped_gather(self.accelerator, *scope) if scope is not None else contextlib.nullcontext():
            gathered = self.accelerator.gather(input_data)
        if final_round is None:
            return gathered
        split_rows, rows_per_rank, shard = final_round
        trim = partial(
            trim_to_split_rows, split_rows=split_rows, rows_per_rank=rows_per_rank, num_ranks=shard.num_processes
        )
        return recursively_apply(trim, gathered)

    def _gather_objects_for_metrics(self, input_data, final_round: tuple[int, int, BatchSamplerShard] | None):
        """accelerate's object gather — every rank's entries, in rank order, as one list — kept to one
        rank per DP replica and cut to the split's rows per rank, as the tensor gather is.

        accelerate's own keeps each sibling's copy and cuts the list to a prefix counted off the
        dataset. Without ``even_batches`` a short final batch holds real rows alone, so only entries
        every rank sends at one length (HF's repeated scalar loss) are cut; per-row entries, as long
        as each rank's batch, are kept whole. A single process returns its input unchanged, as
        accelerate's does. COLLECTIVE.
        """
        if not is_multi_rank_run():
            return input_data
        per_rank: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(per_rank, input_data)
        scope = self._dp_metric_gather_scope
        chunks = [list(per_rank[rank]) for rank in (scope[0] if scope is not None else range(len(per_rank)))]
        if final_round is not None and (final_round[2].even_batches or len({len(chunk) for chunk in chunks}) == 1):
            split_rows, rows_per_rank, _ = final_round
            chunks = [
                chunk[: split_head(len(chunk), split_rows=split_rows, rows_per_rank=rows_per_rank, rank=rank)]
                for rank, chunk in enumerate(chunks)
            ]
        return [entry for chunk in chunks for entry in chunk]

    def _data_parallel_rank_by_global_rank(self) -> list[int]:
        """DP rank of every global rank, indexed by global rank. Collective: every rank must call it."""
        dp_rank = self.get_data_parallel_rank()
        if not (dist.is_available() and dist.is_initialized()):
            return [dp_rank]
        local = torch.tensor([dp_rank], dtype=torch.long, device=current_device())
        gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local)
        return [int(t.item()) for t in gathered]
