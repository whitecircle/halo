"""Rank-coordinated dataset map, filter, whole-dataset transform and pack: the load rank runs each operation to a
deterministic cache file and the others load it back. Set ``HF_DATASETS_CACHE`` to a directory all ranks share."""

import contextlib
import functools
import inspect
import logging
import os
import shutil
import types
import uuid
from collections.abc import Callable
from typing import Any

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from datasets import Dataset, DatasetDict, load_from_disk
from datasets import fingerprint as hf_fingerprint
from multiprocess.reduction import ForkingPickler
from trl.data_utils import _pack_bfd, _pack_wrapped

from halo.data.pipeline.map_cache_key import (
    map_cache_file_name,
    packed_cache_file_name,
    stamp_cache_key,
    transform_cache_name,
)
from halo.data.pipeline.rejected_rows import dataset_total_size, is_valid_example, report_rejected_rows
from halo.data.sources.dataset_cache import publish_cached_download
from halo.env import env_int
from halo.hardware import available_cpu_count
from halo.runtime.collectives import broadcast_from_rank0
from halo.runtime.consensus import store_join_recorded_failure
from halo.runtime.filesystem import fs_aware_makedirs, is_fs_load_rank
from halo.runtime.ranks import get_global_rank, get_local_world_size, is_local_main_process

logger = logging.getLogger(__name__)

# Where :func:`scratch_dataset_maps` redirects coordinated operations, for its block.
_scratch_cache_dir: str | None = None

# The rows TRL's ``pack_dataset`` packs as one unit: its batched map leaves ``batch_size`` at the datasets default.
_PACK_BATCH_ROWS = inspect.signature(Dataset.map).parameters["batch_size"].default


def _pack_wrapped_with_document_lengths(examples: pa.Table, seq_length: int) -> pa.Table:
    """TRL's wrapped pack plus the ``seq_lengths`` its chunks lack: each row's pieces of the source documents, cut at
    document ends and at the chunk edge, summing to the row length."""
    packed = _pack_wrapped(examples, seq_length)
    document_ends = np.cumsum(pc.list_value_length(examples["input_ids"]).to_numpy(zero_copy_only=False))
    total = int(document_ends[-1]) if len(document_ends) else 0
    chunk_starts = np.arange(0, total, seq_length)
    # Unique drops the empty documents' repeated ends.
    cuts = np.unique(np.concatenate(([0], document_ends, chunk_starts, [total])))
    pieces = np.diff(cuts).astype(np.int32)
    row_of_piece = cuts[:-1] // seq_length
    offsets = np.concatenate(([0], np.cumsum(np.bincount(row_of_piece, minlength=len(chunk_starts))))).astype(np.int32)
    lengths = pa.ListArray.from_arrays(offsets, pa.array(pieces, type=pa.int32()))
    return packed.append_column("seq_lengths", lengths)


_TRL_PACKERS = {
    "bfd": functools.partial(_pack_bfd, on_seq_length_overflow="truncate"),
    "bfd_split": functools.partial(_pack_bfd, on_seq_length_overflow="split"),
    "wrapped": _pack_wrapped_with_document_lengths,
}


def dataset_num_proc_override() -> int | None:
    """``None`` while unset; below 1 raises."""
    name = "HALO_DATASET_NUM_PROC"
    override = env_int(name, None)
    if override is not None and override < 1:
        raise ValueError(
            f"{name}={override} is invalid: the num_proc override must be >= 1 "
            f"(use 1 to disable dataset-map multiprocessing)."
        )
    return override


def _resolve_dataset_num_proc() -> int:
    """Three quarters of the CPUs this process may use, at most 64. The load rank runs a coordinated operation while
    its node's other ranks wait and read its cache files back; those ranks compute only on a per-rank key (a
    pre-sharded load), sharing the count then."""
    override = dataset_num_proc_override()
    if override is not None:
        return override
    # max(1, ...): datasets reads 0 as "no multiprocessing" on some paths and rejects it on others.
    return max(1, min(available_cpu_count() * 3 // 4, 64))


DATASET_NUM_PROC = _resolve_dataset_num_proc()


def resolve_map_num_proc(configured: int | None) -> int:
    """Unset means the toolkit default, not 1 worker."""
    return configured or DATASET_NUM_PROC


def _maps_in_workers(num_proc: int | None) -> bool:
    """At 1 or below ``datasets`` gets ``None`` and maps in process."""
    return num_proc is not None and num_proc > 1


def datasets_cache_dir() -> str:
    return os.environ.get("HF_DATASETS_CACHE") or str(datasets.config.HF_DATASETS_CACHE)


def ensure_cache_dir(writer_rank: Callable[[], bool] = is_local_main_process) -> str:
    """COLLECTIVE: :func:`datasets_cache_dir` (inside :func:`scratch_dataset_maps`, its staging dir), so a read-only or
    full volume raises on every rank."""
    cache_dir = _scratch_cache_dir or datasets_cache_dir()
    fs_aware_makedirs(cache_dir, writer_rank=writer_rank)
    return cache_dir


def reject_self_capturing_fn(func: Callable, num_proc: int | None, operation: str) -> None:
    """Refuse a bound method as a multi-worker map callable: at ``num_proc > 1`` dill pickles its whole ``self``
    graph to every worker, the model and under EP the unpicklable process groups included."""
    if not _maps_in_workers(num_proc):
        return
    target = func.func if isinstance(func, functools.partial) else func
    if inspect.ismethod(target):
        owner = type(getattr(target, "__self__", None)).__name__
        raise TypeError(
            f"{operation} got the bound method {owner}.{target.__name__} with num_proc={num_proc}. "
            f"Every worker would pickle its `self` — including the model and, under EP, the DeepEP/"
            f"NCCL process groups. Use a module-level function and pass its state via fn_kwargs."
        )


def _load_map_callable(payload: bytes, cache_key: str) -> Callable:
    """The bare callable a :class:`_PickledOnceCallable` pickled; ``cache_key`` only enters the fingerprint."""
    return ForkingPickler.loads(payload)


class _PickledOnceCallable:
    """A multi-worker map callable that pickles itself once: ``datasets`` pickles the callable into each task serially,
    about a second each for a captured tokenizer. ``cache_key`` rides along so the fingerprint ``datasets`` takes of
    the bytes, which hold no helper's source, changes with the cache name."""

    def __init__(self, fn: Callable, fn_kwargs: dict | None, cache_key: str):
        self._fn = functools.partial(fn, **fn_kwargs) if fn_kwargs else fn
        self._cache_key = cache_key
        self._payload: bytes | None = None

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def __reduce__(self):
        if self._payload is None:
            # The pickler the pool's task queue uses, so a worker rebuilds what a per-task pickle would carry.
            self._payload = bytes(ForkingPickler.dumps(self._fn))
        return _load_map_callable, (self._payload, self._cache_key)


def _remove_columns_by_split(
    dataset: DatasetDict, remove_columns: str | list[str] | None
) -> dict[str, list[str]] | None:
    """Each split's own list where a split lacks a column another split holds, a source's schemas not matching; None
    where the one list fits every split. A column no split holds stays listed, for ``datasets`` to refuse the typo."""
    if not remove_columns:
        return None
    listed = [remove_columns] if isinstance(remove_columns, str) else remove_columns
    held = set().union(*(split.column_names for split in dataset.values()))
    by_split = {
        name: [column for column in listed if column in split.column_names or column not in held]
        for name, split in dataset.items()
    }
    return None if all(len(columns) == len(listed) for columns in by_split.values()) else by_split


def _map_with_columns_by_split(
    dataset: DatasetDict,
    fn: Callable,
    *,
    remove_columns: dict[str, list[str]],
    cache_file_names: dict[str, str],
    **map_kwargs,
) -> DatasetDict:
    """``DatasetDict.map`` per split, not ``Dataset.map``, whose ``on_mixed_types`` default turns a mixed-type column
    into JSON where ``DatasetDict.map`` raises."""
    return DatasetDict(
        {
            name: DatasetDict({name: split}).map(
                fn, remove_columns=remove_columns[name], cache_file_names={name: cache_file_names[name]}, **map_kwargs
            )[name]
            for name, split in dataset.items()
        }
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
    """Shared body of the map and filter seams, so they cannot drift on which kwargs they refuse, which callables they
    reject or what enters the cache name."""
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
    # Wrapped after the naming, which reads the bare callable's source and closure. A coroutine function stays bare:
    # datasets' map and filter pick their async path off the callable they are handed.
    if _maps_in_workers(num_proc) and not inspect.iscoroutinefunction(fn):
        fn = _PickledOnceCallable(fn, op_kwargs.pop("fn_kwargs", None), cache_file_name)

    run_operation = getattr(dataset, operation)
    if operation == "map" and isinstance(dataset, DatasetDict):
        by_split = _remove_columns_by_split(dataset, op_kwargs.get("remove_columns"))
        if by_split is not None:
            run_operation = functools.partial(_map_with_columns_by_split, dataset)
            op_kwargs = {**op_kwargs, "remove_columns": by_split}

    return coordinated_dataset_operation(
        lambda **kwargs: run_operation(fn, desc=desc, **op_kwargs, **kwargs),
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
    """COLLECTIVE: a deterministically-cached ``dataset.map``; ``cache_key_extras`` keys the output-affecting tunables
    the function's source and closure do not show."""
    return _cached_map_or_filter("map", dataset, map_fn, desc, num_proc, map_kwargs, cache_key_extras)


def coordinated_filter(
    dataset: Dataset | DatasetDict,
    filter_fn: Callable,
    desc: str | None = None,
    num_proc: int = DATASET_NUM_PROC,
    **filter_kwargs,
) -> Dataset | DatasetDict:
    """COLLECTIVE: a deterministically-cached ``dataset.filter``."""
    return _cached_map_or_filter("filter", dataset, filter_fn, desc, num_proc, filter_kwargs)


def _run_or_record(work: Callable[[], Any], label: str) -> tuple[Any, BaseException | None]:
    """``(result, failure)``, any ``BaseException`` recorded: a rank-local raise would strand the peers in the join
    that follows, which raises it on every rank."""
    logger.info(f"{label}...")
    try:
        result = work()
    except BaseException as e:
        return None, e
    logger.info(f"{label} completed")
    return result, None


def _reject_operation_failure(failure: BaseException | None, operation_name: str) -> None:
    """COLLECTIVE (store join): the failing rank re-raises its own exception, whose type callers rely on, the peers a
    ``RuntimeError`` with the cause. The store carries the join, since peers wait out the writer's whole map."""
    store_join_recorded_failure("dataset_op", failure, f"Dataset operation {operation_name!r}")


def run_load_rank_first(work: Callable[[], Any], operation_name: str) -> Any:
    """COLLECTIVE (two store joins): run ``work`` on the load rank first, then every other rank, for work whose first
    run fills a cache. A failure in either phase raises on every rank. Never call it inside a main-first block, which
    would hold the peers outside both joins."""
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
    """COLLECTIVE: run a dataset op under :func:`run_load_rank_first` into ``cache_file_name``, which other ranks load.
    In a main-first block ``ensure_cache_dir``'s barrier would run on the main rank alone, skewing the store joins."""
    cache_dir = ensure_cache_dir()

    # A follower computes only on a per-rank key, when all the node's followers map at once, so each takes its share.
    if _maps_in_workers(num_proc) and not is_fs_load_rank():
        num_proc = max(1, num_proc // get_local_world_size())
    # None, not 1: HF still spawns a subprocess at num_proc=1.
    effective_num_proc = num_proc if _maps_in_workers(num_proc) else None
    operation_kwargs = {"num_proc": effective_num_proc, "load_from_cache_file": True}

    if isinstance(dataset, DatasetDict):
        split_cache_names = {split: cache_file_name.replace(".arrow", f"_{split}.arrow") for split in dataset}
        operation_kwargs["cache_file_names"] = {
            split: os.path.join(cache_dir, name) for split, name in split_cache_names.items()
        }
    else:
        operation_kwargs["cache_file_name"] = os.path.join(cache_dir, cache_file_name)

    result = run_load_rank_first(functools.partial(operation_fn, **operation_kwargs), operation_name)

    # HF's _fingerprint/cache_files diverge between writer and loader ranks, so stamp a rank-stable key.
    if isinstance(dataset, DatasetDict):
        for split, split_ds in result.items():
            stamp_cache_key(split_ds, split_cache_names[split])
    else:
        stamp_cache_key(result, cache_file_name)

    return result


@contextlib.contextmanager
def scratch_dataset_maps(staging_dir: str):
    """Every map inside the block writes under ``staging_dir``, removed on exit, including maps that would otherwise
    sit in the process-lifetime ``$TMPDIR/hf_datasets-*`` (``datasets`` has no setting, so its module-level holder is
    swapped). For work no later run re-reads, done by one process: other ranks would key elsewhere."""
    global _scratch_cache_dir
    caching = datasets.is_caching_enabled()
    held_temp_dir, held_cache_dir = hf_fingerprint._TEMP_DIR_FOR_TEMP_CACHE_FILES, _scratch_cache_dir
    os.makedirs(staging_dir)
    hf_fingerprint._TEMP_DIR_FOR_TEMP_CACHE_FILES = types.SimpleNamespace(name=staging_dir)
    _scratch_cache_dir = staging_dir
    datasets.disable_caching()
    try:
        yield
    finally:
        if caching:
            datasets.enable_caching()
        hf_fingerprint._TEMP_DIR_FOR_TEMP_CACHE_FILES = held_temp_dir
        _scratch_cache_dir = held_cache_dir
        try:
            shutil.rmtree(staging_dir)
        except OSError as exc:  # never in place of the block's own error
            logger.warning(f"Could not remove the dataset-preparation staging directory {staging_dir}: {exc}")


def coordinated_dataset_transform(
    dataset: Dataset,
    transform: Callable[[], Dataset],
    operation_name: str,
    cache_key_extras: dict,
) -> Dataset:
    """COLLECTIVE: run a transform that cannot take a cache file (TRL's ``_prepare_dataset``, whose maps key random
    fingerprints) once per filesystem scope, publishing it for other ranks and later runs. ``transform`` runs on the
    load rank alone, so it issues no collective."""
    run_token = broadcast_from_rank0(uuid.uuid4().hex)
    target = os.path.join(
        ensure_cache_dir(), transform_cache_name(operation_name, dataset, cache_key_extras, run_token)
    )

    def prepare_into(tmp_path: str) -> None:
        # Named like the publish's own temp dirs, so its stale-publish reaper collects one a crash left.
        with scratch_dataset_maps(f"{target}.tmp-{uuid.uuid4().hex}"):
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
    """COLLECTIVE: :func:`coordinated_map`, then a filter dropping the rejection rows."""
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


def _pack_batches(batch_starts: pa.Table, rows: Dataset, seq_length: int, strategy: str) -> pa.Table:
    """Packed rows of each ``_PACK_BATCH_ROWS``-row batch from a ``batch_start``, packed alone as in TRL's map."""
    packer = _TRL_PACKERS[strategy]
    starts = batch_starts.column("batch_start").to_pylist()
    return pa.concat_tables([packer(rows[start : start + _PACK_BATCH_ROWS], seq_length) for start in starts])


def coordinated_pack(
    dataset: Dataset,
    seq_length: int,
    strategy: str = "bfd",
    split: str = "train",
    num_proc: int = DATASET_NUM_PROC,
) -> Dataset:
    """COLLECTIVE: TRL's ``pack_dataset`` into a deterministic cache file the other ranks reuse. Workers pack whole
    batches, so rows match TRL's single-process pack at any worker count. The packed rows carry no ``attention_mask``:
    the packing collator builds its own and drops it."""
    if strategy not in _TRL_PACKERS:
        raise ValueError(f"Invalid packing strategy {strategy!r}: TRL packs with one of {sorted(_TRL_PACKERS)}.")
    operation_name = f"packing the {split} split"
    documents = dataset.remove_columns("attention_mask") if "attention_mask" in dataset.column_names else dataset
    # Wrapped rows once packed without ``seq_lengths``: a distinct name never serves those.
    cache_strategy = "wrapped_documents" if strategy == "wrapped" else strategy
    cache_file_name = packed_cache_file_name(documents, cache_strategy, seq_length, split)
    # Workers split the map's input contiguously at any boundary, so its rows are the batches, never the documents.
    batch_starts = Dataset.from_dict({"batch_start": list(range(0, len(dataset), _PACK_BATCH_ROWS))})
    num_proc = max(1, min(num_proc, len(batch_starts)))  # no idle worker, and no datasets warning about one
    pack = functools.partial(
        _pack_batches, rows=documents.with_format("arrow"), seq_length=seq_length, strategy=strategy
    )
    if _maps_in_workers(num_proc):
        pack = _PickledOnceCallable(pack, None, cache_file_name)

    def pack_rows(**operation_kwargs) -> Dataset:
        if not len(dataset):  # inside the work, so a rank's empty shard raises on every rank
            raise ValueError(f"Cannot pack the {split} split: it has no rows.")
        return batch_starts.with_format("arrow").map(
            pack, batched=True, batch_size=1, remove_columns=["batch_start"], desc=operation_name, **operation_kwargs
        )

    packed = coordinated_dataset_operation(
        pack_rows, dataset=dataset, operation_name=operation_name, num_proc=num_proc, cache_file_name=cache_file_name
    )
    # TRL's own restore: the input's format, its format columns joined by the ones the packer added.
    source_format = documents.format
    added = [name for name in packed.column_names if name not in documents.column_names]
    return packed.with_format(
        source_format["type"],
        columns=source_format["columns"] + added,
        output_all_columns=source_format["output_all_columns"],
        **source_format["format_kwargs"],
    )
