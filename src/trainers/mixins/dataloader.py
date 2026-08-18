"""Data-parallel-aware train/eval dataloaders for TP/CP/ETP and pre-sharded datasets.

Base ``Trainer`` shards by world size, wrong when DP < world_size (TP/CP/ETP siblings need identical
slices) or the dataset is pre-sharded per DP rank. Passes parallelism-aware DP size/rank to
``accelerate.prepare_data_loader`` and equalizes pre-sharded lengths; defers to ``super()`` otherwise.
The epoch loop is intercepted here too, to stamp the epoch onto whichever loader the trainer runs.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from functools import partial
from typing import Any

import datasets
import torch
import torch.distributed as dist
from accelerate.data_loader import prepare_data_loader
from accelerate.logging import get_logger
from accelerate.utils import recursively_apply
from torch.utils.data import DataLoader
from transformers.trainer_utils import seed_worker

from src.distributed.runtime import current_device, is_multi_rank_run

logger = get_logger(__name__, log_level="info")

# "absent from the instance __dict__", distinct from a legitimately stored ``None``.
_UNSET = object()


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

    accelerate reaches the epoch-seeded sampler through fixed probe shapes — three levels for
    ``DataLoaderShard``, one for ``DataLoaderDispatcher`` — and a mid-epoch resume adds a level:
    ``skip_first_batches`` wraps the chain in a ``SkipBatchSampler``. The stock DP chain survives
    that (``prepare_data_loader`` re-plants ``.sampler`` on the ``BatchSamplerShard``, which the
    third probe still reaches), but a dispatching loader does not, nor does any chain whose sampler
    accelerate did not re-plant. The sampler's epoch then stays 0 and the resumed epoch redraws
    epoch 0's permutation. Walking ``.batch_sampler`` and ``.sampler`` finds it at whatever depth
    the wrappers leave it.
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


class DataParallelDataLoaderMixin:
    """Parallelism-aware train/eval dataloaders. Mixed into the trainer."""

    # ``(keep_ranks, world_size)`` once the eval gather is DP-scoped, None while it is the world's.
    _dp_metric_gather_scope: tuple[list[int], int] | None = None
    # Raised by ``suspended_dp_metric_gather`` for the unequal-eval-batch escape hatch, which needs
    # its IDENTITY gather to survive the loop's re-arm.
    _dp_metric_gather_suspended: bool = False

    def dp_shard_geometry(self) -> tuple[int, int]:
        """``(size, rank)`` for splitting a dataset across DP replicas.

        A pre-sharded dataset is already this rank's disjoint slice, so it reports ``(1, 0)`` —
        cutting it again silently drops ``(dp-1)/dp`` of the rows.
        """
        if self._dataset_presharded:
            return 1, 0
        return self.get_data_parallel_size(), self.get_data_parallel_rank()

    def _train_loader_batch_size(self) -> int:
        """Rows one train-loader fetch returns. GRPO overrides it — it draws a whole generation round."""
        return self._train_batch_size

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
            params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            if self.args.dataloader_num_workers > 0:
                # Seed by DP rank, not global rank: TP/CP siblings must not desync on a stochastic transform.
                params["worker_init_fn"] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.get_data_parallel_rank()
                )

        return self._prepare_dataloader(DataLoader(train_dataset, **params))

    def _cached_eval_dataloader(self, eval_dataset, build) -> DataLoader:
        """Resolve the eval split, then hand it to ``build`` — or reuse the loader already built.

        The persistent-worker cache holds the PREPARED loader (base ``Trainer`` semantics):
        rebuilding one per ``evaluate()`` leaks a worker pool. Shared by both eval paths so the two
        agree on the key, on what is cached, and on the resolution order. The key is the split name,
        so a hit or a miss is the same on every rank and ``build`` — which issues collectives —
        stays in lockstep.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Evaluation requires an eval_dataset.")

        cache_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if self.args.dataloader_persistent_workers and cache_key in self._eval_dataloaders:
            return self._eval_dataloaders[cache_key]

        if isinstance(eval_dataset, str):
            eval_dataset = self.eval_dataset[eval_dataset]
        prepared = build(eval_dataset if eval_dataset is not None else self.eval_dataset)
        if self.args.dataloader_persistent_workers:
            self._eval_dataloaders[cache_key] = prepared
        return prepared

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """Create eval dataloader with correct DP sharding for TP/CP modes."""
        if not self._needs_custom_dataloader():
            return super().get_eval_dataloader(eval_dataset=eval_dataset)
        # Whether the caller named a split, decided before the cache resolves it to a dataset.
        caller_supplied = eval_dataset is not None

        def build(dataset):
            # Same hang as train (metrics gather); collective, every rank must reach it.
            if self._dataset_presharded:
                dataset = self._equalize_presharded_length(dataset, split="eval")
                if not caller_supplied:
                    self.eval_dataset = dataset

            dataset, params = self._loader_params(dataset, self.args.eval_batch_size, "evaluation")

            if not isinstance(dataset, torch.utils.data.IterableDataset):
                params["sampler"] = self._get_eval_sampler(dataset)
                params["drop_last"] = self.args.dataloader_drop_last
                params["prefetch_factor"] = self.args.dataloader_prefetch_factor

            return self._prepare_dataloader(DataLoader(dataset, **params))

        return self._cached_eval_dataloader(eval_dataset, build)

    def _run_epoch(self, model, epoch, train_dataloader, *args, **kwargs):
        """Stamp ``epoch`` onto the loader's sampler chain, then run the base's epoch unchanged.

        Every toolkit trainer reaches this seam, including the plain FSDP2/DP path whose loader is
        the base's own (``_needs_custom_dataloader()`` is False there). The base stamps the epoch
        itself, but only through probes a resumed chain can fall out of (see
        :func:`set_sampler_epoch`); stamping here first is idempotent where those probes reach, and
        survives the base's ``skip_first_batches`` rebuild because the wrapper keeps the same
        sampler object.
        """
        set_sampler_epoch(train_dataloader, epoch)
        return super()._run_epoch(model, epoch, train_dataloader, *args, **kwargs)

    def _set_signature_columns_if_needed(self):
        """Union the collator's ``required_dataset_columns`` into HF's signature-column set.

        Column pruning keeps only ``_signature_columns``, and a collator input like packing's
        ``seq_lengths`` is not a model-forward parameter — it survives pruning only if the trainer's
        signature set happens to name it (TRL's SFT hard-codes it; the other bases do not). The
        collator declares what it needs and every trainer honors it here; the pipeline path pins the
        same attribute into its column set in ``mixins/pipeline.py``.
        """
        super()._set_signature_columns_if_needed()
        if self._signature_columns:
            for column in getattr(getattr(self, "data_collator", None), "required_dataset_columns", ()):
                if column not in self._signature_columns:
                    self._signature_columns.append(column)

    def _loader_params(self, dataset, batch_size: int, description: str) -> tuple[Any, dict]:
        """Column-pruned dataset plus the ``DataLoader`` kwargs the train and eval paths share.

        Unused columns are dropped from the dataset when the trainer prunes datasets, else from the
        collator — the base ``Trainer`` contract. Sampler, ``drop_last``, prefetch and worker seeding
        legitimately differ per split and stay with each caller.
        """
        collator = self.data_collator
        # Branch on the DATASET type like the base does: pruning reads column_names, only datasets.Dataset has it.
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
        }
        return dataset, params

    def _needs_custom_dataloader(self) -> bool:
        """True when DP < world_size (TP/CP/ETP/PP) or the dataset is pre-sharded per DP rank.

        EP alone uses the base flow (EP ⊥ DP) unless pre-sharded, where base accelerate re-sharding
        would drop most of an already-disjoint slice. PP must take the custom path: accelerate's
        default shards by GLOBAL rank, which would hand every rank of a pipeline chain a DIFFERENT
        batch — stage 0 then forwards one row set while the last stage scores another's labels,
        silently training on garbage pairs. ``get_data_parallel_rank`` is stage-local, so the custom
        path gives all chain members the same shard.
        """
        return (
            self.parallelism_config.is_tp_mode
            or self.parallelism_config.is_cp_mode
            or self.parallelism_config.is_expert_tp_mode
            or self.parallelism_config.is_pp_mode
            or self._dataset_presharded
        )

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
                "Pre-sharded eval dataset: at least one data-parallel rank holds zero eval examples "
                "(the eval/test split has fewer non-empty shards than data_parallel_size — sharding "
                "skips empty shards, so a small test split produces fewer of them). Unequal eval "
                "lengths make ranks run different eval step counts (NCCL hang at the metrics gather) "
                "or report wrong metrics. Re-preprocess the eval split with at least "
                "data_parallel_size non-empty shards, or disable evaluation."
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
        """
        if getattr(dataloader, "_is_accelerate_prepared", False):
            return dataloader

        if num_processes is not None:
            data_parallel_size = num_processes
            data_parallel_rank = process_index or 0
        else:
            # A pre-sharded dataset reports (1, 0): device placement only, do NOT re-shard.
            data_parallel_size, data_parallel_rank = self.dp_shard_geometry()

        prepared = prepare_data_loader(
            dataloader,
            self.accelerator.device,
            num_processes=data_parallel_size,
            process_index=data_parallel_rank,
            split_batches=self.accelerator.split_batches,
            put_on_device=True,
            rng_types=self.accelerator.rng_types.copy(),
            dispatch_batches=self.accelerator.dispatch_batches,
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
        """Pin a third-party accelerate sweep — its loader AND its gather — to the DP axis.

        A sweep that builds its own loader with ``accelerator.prepare`` and reassembles the per-rank
        results with ``accelerator.gather`` (TRL's ``precompute_ref_log_probs``) keys both on the
        GLOBAL rank. Under TP/CP/ETP that hands siblings different rows — they must forward identical
        rows through the TP-sharded attention and the ETP token-space all-reduce — and the world-order
        concatenation the caller indexes by dataset position then carries one duplicate per sibling.
        Routing the loader through ``_prepare_dataloader`` restores the DP contract, and keeping one
        chunk per DP rank restores the dataset order. Both are identities at ``dp_size == world_size``.
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

    def _install_dp_metric_gather(self) -> None:
        """Scope HF's evaluation gather to the DP axis wherever the loader is DP-sharded.

        ``evaluation_loop`` gathers predictions, labels and losses over the WHOLE world, but each
        rank's eval batch is its DP REPLICA's: TP/CP/ETP siblings and pipeline chain peers all
        return the same rows (under PP the last stage's values are broadcast to the whole chain, so
        every stage returns them). A world gather therefore repeats every replica once per sibling —
        ``pp_size`` copies at pp>1 — and accelerate's end-of-dataloader remainder trim then keeps a
        prefix of the DUPLICATES instead of the real tail, so a partial final batch reports a
        different sample set again. Means survive both; nothing else does.

        Collective: the DP-rank map is an all-gather, so every rank must reach this.
        """
        if not self._needs_custom_dataloader():
            return
        dp_rank_by_global_rank = self._data_parallel_rank_by_global_rank()
        world_size = len(dp_rank_by_global_rank)
        keep = dp_representative_ranks(dp_rank_by_global_rank)
        if len(keep) == world_size:
            return
        self._dp_metric_gather_scope = (keep, world_size)
        self.gather_function = self._dp_gather_for_metrics
        logger.info(
            "Evaluation metrics gather scoped to the %d data-parallel replicas of %d ranks "
            "(siblings return identical rows; a world gather would repeat each replica %dx).",
            len(keep),
            world_size,
            world_size // len(keep),
        )

    def _rearm_dp_metric_gather(self) -> None:
        """Re-arm the DP-scoped metric gather for the evaluation loop about to run.

        ``Trainer.evaluation_loop`` resets ``gather_function`` to ``accelerator.gather_for_metrics``
        on its way out, so the install done once at construction covers the first loop only — every
        later evaluate()/predict() would gather each DP replica once per TP/CP/ETP/PP sibling again.
        Per-rank and derived from state the install already agreed on, so no collective repeats here.
        """
        if self._dp_metric_gather_scope is not None and not self._dp_metric_gather_suspended:
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

    def _dp_gather_for_metrics(self, input_data, **kwargs):
        """``gather_for_metrics`` with the duplicate DP siblings dropped BEFORE its remainder trim.

        accelerate's trim is what makes the order load-bearing: it keeps the first ``remainder``
        rows of the gathered tensor, which under duplication are the wrong ones.
        """
        keep, world_size = self._dp_metric_gather_scope
        with dp_scoped_gather(self.accelerator, keep, world_size):
            return self.accelerator.gather_for_metrics(input_data, **kwargs)

    def _data_parallel_rank_by_global_rank(self) -> list[int]:
        """DP rank of every global rank, indexed by global rank. Collective: every rank must call it."""
        dp_rank = self.get_data_parallel_rank()
        if not (dist.is_available() and dist.is_initialized()):
            return [dp_rank]
        local = torch.tensor([dp_rank], dtype=torch.long, device=current_device())
        gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local)
        return [int(t.item()) for t in gathered]
