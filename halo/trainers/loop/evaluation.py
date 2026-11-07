"""The evaluation loop and its data-parallel gather. accelerate pads every rank's last eval batch with the split's
first rows; the cut counts them off the active loader's geometry, after the DP scope drops what siblings repeat."""

from __future__ import annotations

import contextlib
import logging
from functools import partial
from typing import Any, ClassVar

import torch
import torch.distributed as dist
from accelerate.data_loader import BatchSamplerShard, DataLoaderDispatcher
from accelerate.utils import recursively_apply
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler, SubsetRandomSampler
from transformers.trainer_pt_utils import LengthGroupedSampler
from trl.trainer.utils import RepeatSampler

from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.runtime.collectives import barrier
from halo.runtime.consensus import rank_consensus, rank_max
from halo.runtime.ranks import get_global_world_size, has_process_group, is_global_main_process, is_multi_rank_run
from halo.trainers.loop.dataloaders import dp_representative_ranks, dp_scoped_gather, find_batch_sampler_shard

logger = logging.getLogger(__name__)

# Samplers whose ``len`` is the number of indices they yield (TRL's ``RepeatSampler`` counts its repeats).
_COUNTED_SAMPLERS = (SequentialSampler, RandomSampler, SubsetRandomSampler, LengthGroupedSampler, RepeatSampler)


def final_round_split_rows(loader, dataset_drawing: tuple[type[BatchSampler], ...] = ()) -> int | None:
    """How many of the final-round rows, over every rank in rank order, are the split's own; ``None`` when nothing is
    cut. accelerate's ``remainder`` counts the dataset, which miscounts TRL's ``RepeatSampler`` and trims real rows
    from a one-process loader."""
    shard = find_batch_sampler_shard(loader)
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
    """``tensor`` itself when ``real_rows`` is every row: a recorded full-range slice would copy a whole-tensor
    gradient in backward."""
    return tensor if real_rows >= tensor.size(0) else tensor[:real_rows]


def split_rows_mean(values: torch.Tensor, real_rows: int) -> torch.Tensor:
    """Mean of ``values``' first ``real_rows`` rows, 0 over none; over every row ``values.mean()`` itself, so a train
    batch reduces bit for bit like a plain mean."""
    if real_rows >= values.size(0):
        return values.mean()
    head = values[:real_rows]
    return head.sum() / max(head.numel(), 1)


def _rank_split_rows(split_rows: int, rows_per_rank: int, rank: int) -> int:
    """Real rows among ``rank``'s final round; the split fills it in rank order."""
    return max(0, min(rows_per_rank, split_rows - rank * rows_per_rank))


def _split_head(entries: int, *, split_rows: int, rows_per_rank: int, rank: int) -> int:
    """``rank``'s share of ``entries`` gathered entries, rounded up: a chunk need not hold one entry per row (HF
    repeats the scalar eval loss ``eval_batch_size`` times) and a rank holding any real row keeps one."""
    return -(-_rank_split_rows(split_rows, rows_per_rank, rank) * entries // rows_per_rank)


def trim_to_split_rows(tensor: torch.Tensor, *, split_rows: int, rows_per_rank: int, num_ranks: int) -> torch.Tensor:
    """The split's share of a gathered final round: ``num_ranks`` equal chunks in rank order, each standing for
    ``rows_per_rank`` rows, the first ``split_rows`` real."""
    chunk, ragged = divmod(tensor.shape[0], num_ranks)
    if ragged:
        raise ValueError(
            f"Gathered tensor has {tensor.shape[0]} rows, not a multiple of the {num_ranks} data-parallel "
            f"ranks: it did not come from a gather of equal per-rank chunks, so its padding cannot be cut."
        )
    parts = tensor.reshape(num_ranks, chunk, *tensor.shape[1:])
    return torch.cat(
        [
            part[: _split_head(chunk, split_rows=split_rows, rows_per_rank=rows_per_rank, rank=rank)]
            for rank, part in enumerate(parts)
        ]
    )


class EvaluationMixin:
    # ``(keep_ranks, world_size)`` once the eval gather is DP-scoped, None while it is the world's.
    _dp_metric_gather_scope: tuple[list[int], int] | None = None
    # Raised for the unequal-eval-batch escape hatch, whose identity gather must survive the loop's re-arm.
    _dp_metric_gather_suspended: bool = False
    # Batch samplers whose iteration draws every row of their ``dataset`` once, so a final round is counted off it.
    DATASET_DRAWING_BATCH_SAMPLERS: ClassVar[tuple[type[BatchSampler], ...]] = ()

    def evaluate(self, *args, **kwargs):
        """COLLECTIVE. Swap the eval gather for the identity where ranks hold unequal batch counts, since the per-step
        ``gather_for_metrics`` deadlocks there; each rank scores its own shard and the logged metrics are rank 0's. A
        dict ``eval_dataset`` is measured split by split, when HF re-enters here for each."""
        eval_dataset = kwargs.get("eval_dataset", args[0] if args else None)
        if isinstance(self.eval_dataset if eval_dataset is None else eval_dataset, dict):
            return super().evaluate(*args, **kwargs)
        if (
            self._needs_custom_accelerator()
            and has_process_group()
            and self._eval_ranks_have_unequal_batches(eval_dataset)
        ):
            if getattr(self.args, "top_entropy_quantile", 1.0) < 1.0:
                # TRL's entropy threshold pads-and-gathers through this same accelerator; identity padding aborts.
                raise ValueError(
                    "Evaluation with unequal (or unmeasurable) per-rank batch counts cannot run "
                    "with entropy masking (top_entropy_quantile < 1.0): the eval escape hatch "
                    "disables cross-rank padding, which that gather needs. Provide a sized, evenly "
                    "divisible eval dataset or set top_entropy_quantile: 1.0."
                )
            metric_key_prefix = kwargs.get("metric_key_prefix", args[2] if len(args) > 2 else "eval")
            if is_global_main_process():
                logger.warning(
                    f"Evaluation batch counts differ across ranks (or cannot be measured), so the "
                    f"evaluation loop's cross-rank gather is replaced by the identity to avoid a deadlock: "
                    f"'{metric_key_prefix}_loss' and any compute_metrics output this evaluation logs are "
                    f"rank 0's own shard, not the global value. An eval dataset that splits evenly across "
                    f"the data-parallel ranks restores global metrics."
                )
            original_gather = self.gather_function
            self.gather_function = lambda x: x
            # evaluation_loop also pads logits/labels outside gather_function; identity is safe (own shard only).
            self.accelerator.pad_across_processes = lambda tensor, *pad_args, **pad_kwargs: tensor
            try:
                # Holds the identity across evaluation_loop, which otherwise re-arms the DP-scoped gather.
                with self.suspended_dp_metric_gather():
                    result = super().evaluate(*args, **kwargs)
            finally:
                self.gather_function = original_gather
                # Assigning the bound method back would pin a permanent instance attribute.
                self.accelerator.__dict__.pop("pad_across_processes", None)
            barrier()
            return result
        return super().evaluate(*args, **kwargs)

    def evaluation_loop(self, *args, **kwargs):
        """Re-arm the DP-scoped gather unless suspended, since HF hands ``gather_function`` back to accelerate after
        every loop, then re-shard a pipeline stage the forward-only drives left gathered."""
        if not self._dp_metric_gather_suspended:
            self.gather_function = self._dp_gather_for_metrics
        output = super().evaluation_loop(*args, **kwargs)
        if self._pp_runner is not None:
            reshard_fsdp2_modules(self.model)
        return output

    @contextlib.contextmanager
    def suspended_dp_metric_gather(self):
        """Hold whatever gather is installed across the block's evaluation loops."""
        was_suspended = self._dp_metric_gather_suspended
        self._dp_metric_gather_suspended = True
        try:
            yield
        finally:
            self._dp_metric_gather_suspended = was_suspended

    def eval_split_rows(self, num_rows: int) -> int:
        """How many of this rank's ``num_rows`` batch rows are the split's own, since accelerate's padding rows would
        count twice. ``num_rows`` in train mode, outside a padded final round, without ``even_batches`` and on an
        iterable split. Per-rank."""
        final_round = None if self.model.training else self._active_final_round()
        if final_round is None or not final_round[2].even_batches:
            return num_rows
        split_rows, rows_per_rank, shard = final_round
        return min(num_rows, _rank_split_rows(split_rows, rows_per_rank, shard.process_index))

    def _eval_ranks_have_unequal_batches(self, eval_dataset) -> bool:
        """COLLECTIVE. True when batch counts differ across ranks or a rank cannot measure its own (iterable dataset),
        so the gather is dropped rather than risk a deadlock."""
        try:
            local_batches = len(self.get_eval_dataloader(eval_dataset))
            measurable = True
        except TypeError:
            local_batches, measurable = 0, False

        # Agree measurability first: a rank that early-returns leaves peers at the reduce below.
        if not rank_consensus(measurable).all_ok:
            return True
        most, negated_fewest = rank_max([local_batches, -local_batches])
        return most != -negated_fewest

    def _install_dp_metric_gather(self) -> None:
        """COLLECTIVE. DP-scoped where the loader is DP-sharded, since TP/CP/ETP siblings and pipeline peers all return
        their replica's rows (PP broadcasts the last stage's)."""
        if self._needs_custom_dataloader():
            dp_rank_by_global_rank = self._data_parallel_rank_by_global_rank()
            world_size = len(dp_rank_by_global_rank)
            keep = dp_representative_ranks(dp_rank_by_global_rank)
            if len(keep) < world_size:
                self._dp_metric_gather_scope = (keep, world_size)
                if is_global_main_process():
                    logger.info(
                        "Evaluation metrics gather scoped to the %d data-parallel replicas of %d ranks "
                        "(siblings return identical rows; a world gather would repeat each replica %dx).",
                        len(keep),
                        world_size,
                        world_size // len(keep),
                    )
        self.gather_function = self._dp_gather_for_metrics

    def _dp_gather_for_metrics(self, input_data, use_gather_object: bool | None = None):
        """One chunk per DP rank, then the final round's padding cut; a half-padded rank's scalar loss keeps its share
        of entries. Non-tensor input takes the object gather; a dispatching loader takes accelerate's
        ``gather_for_metrics``, and raises under a DP scope."""
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
        if not use_gather_object:
            try:
                recursively_apply(lambda tensor: tensor, input_data, error_on_other_type=True)
            except TypeError:
                use_gather_object = True
        if use_gather_object:
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

    def _active_final_round(self) -> tuple[int, int, BatchSamplerShard] | None:
        """``(split_rows, rows_per_rank, shard)`` while the active loader's padded final round runs."""
        shard = self._active_eval_shard()
        state = self.accelerator.gradient_state
        if shard is None or not state.end_of_dataloader:
            return None
        loader = state.active_dataloader
        split_rows = final_round_split_rows(loader, self.DATASET_DRAWING_BATCH_SAMPLERS)
        if split_rows is None:
            return None
        return split_rows, loader.total_batch_size // shard.num_processes, shard

    def _active_eval_shard(self) -> BatchSamplerShard | None:
        """The active loader's shard, refused when its shard count is not the gathered chunk count: a loader built
        around the DP-aware eval build would lose its distinct rows as sibling duplicates. Per-rank, no collective."""
        shard = find_batch_sampler_shard(self.accelerator.gradient_state.active_dataloader)
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

    def _gather_objects_for_metrics(self, input_data, final_round: tuple[int, int, BatchSamplerShard] | None):
        """COLLECTIVE. accelerate's object gather kept to one rank per DP replica and cut to the split's rows. Without
        ``even_batches`` only entries every rank sends at one length (HF's repeated scalar loss) are cut."""
        if not is_multi_rank_run():
            return input_data
        per_rank: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(per_rank, input_data)
        scope = self._dp_metric_gather_scope
        chunks = [list(per_rank[rank]) for rank in (scope[0] if scope is not None else range(len(per_rank)))]
        if final_round is not None and (final_round[2].even_batches or len({len(chunk) for chunk in chunks}) == 1):
            split_rows, rows_per_rank, _ = final_round
            chunks = [
                chunk[: _split_head(len(chunk), split_rows=split_rows, rows_per_rank=rows_per_rank, rank=rank)]
                for rank, chunk in enumerate(chunks)
            ]
        return [entry for chunk in chunks for entry in chunk]
