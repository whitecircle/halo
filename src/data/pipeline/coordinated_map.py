"""Rank-coordinated dataset map, filter, whole-dataset transform and pack.

The load rank runs each operation to a deterministic cache file and the other ranks load it back, so no rank
repeats the work or keeps its own copy. Set ``HF_DATASETS_CACHE`` for a cache directory every rank shares.
"""

import contextlib
import functools
import inspect
import logging
import multiprocessing
import os
import shutil
import types
import uuid
from collections.abc import Callable
from typing import Any

import datasets
from datasets import Dataset, DatasetDict, load_from_disk
from datasets import fingerprint as hf_fingerprint
from trl.data_utils import pack_dataset as _trl_pack_dataset

from src.data.pipeline.map_cache_key import (
    map_cache_file_name,
    packed_cache_file_name,
    stamp_cache_key,
    transform_cache_name,
)
from src.data.pipeline.rejected_rows import dataset_total_size, is_valid_example, report_rejected_rows
from src.data.sources.dataset_cache import publish_cached_download
from src.env import env_int
from src.runtime.collectives import broadcast_from_rank0
from src.runtime.consensus import store_join_recorded_failure
from src.runtime.filesystem import fs_aware_makedirs, is_fs_load_rank
from src.runtime.ranks import get_global_rank, is_local_main_process

logger = logging.getLogger(__name__)


def _resolve_dataset_num_proc() -> int:
    """Default map/filter worker processes: a quarter of the cores, capped at 4 since every rank spawns them.
    ``HALO_DATASET_NUM_PROC`` pins it fleet-wide: HF keys its per-worker cache files by the count, so a node
    deriving another value would miss the writer rank's caches."""
    name = "HALO_DATASET_NUM_PROC"
    override = env_int(name, None)
    if override is None:
        # max(1, ...): datasets reads 0 as "no multiprocessing" on some paths and rejects it on others.
        return max(1, min(multiprocessing.cpu_count() // 4, 4))
    if override < 1:
        raise ValueError(
            f"{name}={override} is invalid: the fleet-wide num_proc pin must be >= 1 "
            f"(use 1 to disable dataset-map multiprocessing)."
        )
    return override


DATASET_NUM_PROC = _resolve_dataset_num_proc()


def resolve_map_num_proc(configured: int | None) -> int:
    """``dataset_num_proc`` → map/filter worker count. Unset means the toolkit default, not 1 worker."""
    return configured or DATASET_NUM_PROC


def ensure_cache_dir(writer_rank: Callable[[], bool] = is_local_main_process) -> str:
    """The datasets cache dir every rank agrees on (``HF_DATASETS_CACHE`` or HF's default), created by
    ``writer_rank`` behind a fenced write, so a read-only or full volume raises on every rank."""
    cache_dir = os.environ.get("HF_DATASETS_CACHE") or str(datasets.config.HF_DATASETS_CACHE)
    fs_aware_makedirs(cache_dir, writer_rank=writer_rank)
    return cache_dir


def reject_self_capturing_fn(func: Callable, num_proc: int | None, operation: str) -> None:
    """Refuse a bound method as a multi-worker map callable: at ``num_proc > 1`` dill pickles its whole ``self``
    graph to every worker, the model and under EP the unpicklable process groups included."""
    if num_proc is None or num_proc <= 1:
        return
    target = func.func if isinstance(func, functools.partial) else func
    if inspect.ismethod(target):
        owner = type(getattr(target, "__self__", None)).__name__
        raise TypeError(
            f"{operation} got the bound method {owner}.{target.__name__} with num_proc={num_proc}. "
            f"Every worker would pickle its `self` — including the model and, under EP, the DeepEP/"
            f"NCCL process groups. Use a module-level function and pass its state via fn_kwargs."
        )


def _cached_map_or_filter(
    operation: str,
    dataset: Dataset | DatasetDict,
    fn: Callable,
    desc: str | None,
    num_proc: int,
    op_kwargs: dict,
    cache_key_extras: dict | None = None,
) -> Dataset | DatasetDict:
    """Shared body of :func:`coordinated_map` / :func:`coordinated_filter`: ``operation`` names both
    the ``datasets`` method and the cache-key namespace, so the two seams cannot drift on which kwargs
    they refuse, which callables they reject, or what enters the cache name."""
    operation_name = f"{operation} operation ({desc})" if desc else f"{operation} operation"
    # Execution knobs coordinated_dataset_operation sets itself: the on-disk cache is the cross-rank transport.
    owned = sorted(set(op_kwargs) & {"keep_in_memory", "load_from_cache_file"})
    if owned:
        raise TypeError(
            f"{operation_name} does not accept {owned}: coordinated dataset operations own these "
            f"execution knobs, so the value would be ignored. Drop them; the worker count is the "
            f"op's own num_proc parameter."
        )
    reject_self_capturing_fn(fn, num_proc, operation_name)
    cache_file_name = map_cache_file_name(operation, fn, dataset, desc, op_kwargs, cache_key_extras=cache_key_extras)

    return coordinated_dataset_operation(
        lambda **kwargs: getattr(dataset, operation)(fn, desc=desc, **op_kwargs, **kwargs),
        dataset=dataset,
        operation_name=operation_name,
        num_proc=num_proc,
        cache_file_name=cache_file_name,
    )


def coordinated_map(
    dataset: Dataset | DatasetDict,
    map_fn: Callable,
    desc: str | None = None,
    num_proc: int = DATASET_NUM_PROC,
    cache_key_extras: dict | None = None,
    **map_kwargs,
) -> Dataset | DatasetDict:
    """COLLECTIVE: a deterministically-cached ``dataset.map`` (:func:`coordinated_dataset_operation`);
    ``cache_key_extras`` keys the output-affecting tunables the function's source and closure do not show."""
    return _cached_map_or_filter("map", dataset, map_fn, desc, num_proc, map_kwargs, cache_key_extras)


def coordinated_filter(
    dataset: Dataset | DatasetDict,
    filter_fn: Callable,
    desc: str | None = None,
    num_proc: int = DATASET_NUM_PROC,
    **filter_kwargs,
) -> Dataset | DatasetDict:
    """COLLECTIVE: a deterministically-cached ``dataset.filter`` (:func:`coordinated_dataset_operation`)."""
    return _cached_map_or_filter("filter", dataset, filter_fn, desc, num_proc, filter_kwargs)


def _run_or_record(work: Callable[[], Any], label: str) -> tuple[Any, BaseException | None]:
    """Run ``work``, returning ``(result, failure)`` with any ``BaseException`` recorded rather than raised: a
    rank-local raise would strand the peers in the join that follows, which raises it on every rank instead."""
    logger.info(f"{label}...")
    try:
        result = work()
    except BaseException as e:
        return None, e
    logger.info(f"{label} completed")
    return result, None


def _reject_operation_failure(failure: BaseException | None, operation_name: str) -> None:
    """COLLECTIVE (store join): abort every rank when the rank that ran the operation failed; it re-raises its
    own exception, whose type callers rely on, and the peers a ``RuntimeError`` carrying the cause. The store,
    not NCCL, carries the join, since the peers wait out the writer's whole map."""
    store_join_recorded_failure("dataset_op", failure, f"Dataset operation {operation_name!r}")


def run_load_rank_first(work: Callable[[], Any], operation_name: str) -> Any:
    """COLLECTIVE (two store joins): run ``work`` on the load rank (:func:`is_fs_load_rank`, per input
    filesystem) first, then on every other rank, for work whose first run fills a cache the others read back. A
    failure in either phase raises its cause on every rank. Never call it inside a main-first block, which would
    hold the peers outside both joins."""
    global_rank = get_global_rank()
    is_main = is_fs_load_rank()
    result, failure = None, None
    if is_main:
        result, failure = _run_or_record(work, f"[{operation_name}] Main (rank {global_rank})")
    _reject_operation_failure(failure, operation_name)
    if not is_main:
        result, failure = _run_or_record(work, f"[{operation_name}] Rank {global_rank} from the main rank's output")
    _reject_operation_failure(failure, operation_name)
    return result


def coordinated_dataset_operation(
    operation_fn: Callable,
    dataset: Dataset | DatasetDict,
    operation_name: str,
    num_proc: int | None,
    cache_file_name: str,
) -> Dataset | DatasetDict:
    """COLLECTIVE: run a dataset op under :func:`run_load_rank_first` into ``cache_file_name``, which the other
    ranks then load; ``operation_fn`` captures the dataset. This is the rank ordering: inside a main-first block
    ``ensure_cache_dir``'s barrier would run on the main rank alone and the store joins go off by one."""
    cache_dir = ensure_cache_dir()

    # None, not 1: HF still spawns a subprocess at num_proc=1.
    effective_num_proc = None if num_proc is not None and num_proc <= 1 else num_proc
    operation_kwargs = {"num_proc": effective_num_proc, "load_from_cache_file": True}

    if isinstance(dataset, DatasetDict):
        split_cache_names = {split: cache_file_name.replace(".arrow", f"_{split}.arrow") for split in dataset}
        operation_kwargs["cache_file_names"] = {
            split: os.path.join(cache_dir, name) for split, name in split_cache_names.items()
        }
    else:
        operation_kwargs["cache_file_name"] = os.path.join(cache_dir, cache_file_name)

    result = run_load_rank_first(functools.partial(operation_fn, **operation_kwargs), operation_name)

    # HF's _fingerprint/cache_files diverge between writer and loader ranks — stamp a rank-stable key.
    if isinstance(dataset, DatasetDict):
        for split, split_ds in result.items():
            stamp_cache_key(split_ds, split_cache_names[split])
    else:
        stamp_cache_key(result, cache_file_name)

    return result


@contextlib.contextmanager
def _uncached_maps_in(staging_dir: str):
    """HF's map cache off, with every map's output written under ``staging_dir`` (removed on exit) instead of the
    process-lifetime ``$TMPDIR/hf_datasets-*`` directory, which keeps every intermediate on the root filesystem;
    ``datasets`` has no setting for it, so its module-level holder is swapped."""
    caching = datasets.is_caching_enabled()
    held = hf_fingerprint._TEMP_DIR_FOR_TEMP_CACHE_FILES
    os.makedirs(staging_dir)
    hf_fingerprint._TEMP_DIR_FOR_TEMP_CACHE_FILES = types.SimpleNamespace(name=staging_dir)
    datasets.disable_caching()
    try:
        yield
    finally:
        if caching:
            datasets.enable_caching()
        hf_fingerprint._TEMP_DIR_FOR_TEMP_CACHE_FILES = held
        try:
            shutil.rmtree(staging_dir)
        except OSError as exc:  # never in place of the transform's own error
            logger.warning(f"Could not remove the dataset-preparation staging directory {staging_dir}: {exc}")


def coordinated_dataset_transform(
    dataset: Dataset,
    transform: Callable[[], Dataset],
    operation_name: str,
    cache_key_extras: dict,
) -> Dataset:
    """COLLECTIVE: run a whole-dataset transform that cannot take a cache file (TRL's ``_prepare_dataset``, whose
    maps key random fingerprints) once per filesystem scope, publishing it under :func:`transform_cache_name` for
    the other ranks and later runs to load. ``transform`` runs on the load rank alone, so it issues no
    collective."""
    run_token = broadcast_from_rank0(uuid.uuid4().hex)
    target = os.path.join(
        ensure_cache_dir(), transform_cache_name(operation_name, dataset, cache_key_extras, run_token)
    )

    def prepare_into(tmp_path: str) -> None:
        # Named like the publish's own temp dirs, so its stale-publish reaper collects one a crash left.
        with _uncached_maps_in(f"{target}.tmp-{uuid.uuid4().hex}"):
            transform().save_to_disk(tmp_path)

    def publish_and_load() -> Dataset:
        publish_cached_download(
            target,
            f"{target}.lock",
            operation_name,
            prepare_into,
            live_fingerprint=None,
            fresh_fingerprint=lambda: None,
        )
        return load_from_disk(target)

    result = run_load_rank_first(publish_and_load, operation_name)
    stamp_cache_key(result, os.path.basename(target))
    return result


def coordinated_map_and_filter(
    dataset: Dataset | DatasetDict,
    process_fn: Callable,
    filter_field: str = "input_ids",
    num_proc: int = DATASET_NUM_PROC,
    remove_columns: list[str] | None = None,
    desc: str | None = None,
    cache_key_extras: dict | None = None,
) -> Dataset | DatasetDict:
    """COLLECTIVE: :func:`coordinated_map`, then a filter dropping the rejection rows (:func:`is_valid_example` on
    ``filter_field``)."""
    original_size = dataset_total_size(dataset)

    processed = coordinated_map(
        dataset,
        process_fn,
        desc=desc,
        num_proc=num_proc,
        remove_columns=remove_columns,
        cache_key_extras=cache_key_extras,
    )

    filtered = coordinated_filter(
        processed,
        functools.partial(is_valid_example, filter_field=filter_field),
        desc=f"filtering rejected {filter_field}",
        num_proc=num_proc,
    )

    report_rejected_rows(original_size, dataset_total_size(filtered), f"processing (checked via {filter_field})")
    return filtered


def coordinated_pack(
    dataset: Dataset,
    seq_length: int,
    strategy: str = "bfd",
    split: str = "train",
) -> Dataset:
    """COLLECTIVE: TRL's ``pack_dataset`` under :func:`run_load_rank_first` into a deterministic cache file the
    other ranks reuse, instead of a corpus copy per rank; ``dataset`` must be on disk for the file to apply."""
    # Outside the phases: ensure_cache_dir() barriers, which the main rank alone would reach inside them.
    cache_file_name = os.path.join(ensure_cache_dir(), packed_cache_file_name(dataset, strategy, seq_length, split))
    return run_load_rank_first(
        functools.partial(
            _trl_pack_dataset,
            dataset,
            seq_length=seq_length,
            strategy=strategy,
            map_kwargs={"cache_file_name": cache_file_name, "load_from_cache_file": True},
        ),
        f"packing the {split} split",
    )
