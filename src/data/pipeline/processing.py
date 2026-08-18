"""Distributed dataset processing with multi-process coordination.

Main rank runs each map/filter to a deterministic cache file; others wait then load it — no
duplicate work / per-rank copies. Set HF_DATASETS_CACHE for a consistent cache dir. Prefer the
high-level ``coordinated_map`` / ``coordinated_filter``.
"""

import functools
import hashlib
import importlib.metadata
import inspect
import json
import logging
import multiprocessing
import os
import types
from collections.abc import Callable
from typing import Any

import datasets
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from trl.data_utils import pack_dataset as _trl_pack_dataset

from src.data.pipeline.row_processors import is_valid_example
from src.distributed.filesystem import RUN_LOG_DIR_NAME, fs_aware_main_first, store_join_recorded_failure
from src.distributed.runtime import (
    fs_aware_load_rank,
    fs_aware_makedirs,
    fs_aware_save_rank,
    get_global_rank,
    is_global_main_process,
    is_local_main_process,
)
from src.env import env_int
from src.log import warn_once

logger = logging.getLogger(__name__)

__all__ = [
    "DATASET_NUM_PROC",
    "resolve_map_num_proc",
    "get_function_identifier",
    "reject_self_capturing_fn",
    "ensure_cache_dir",
    "report_rejected_rows",
    "missing_render_column_splits",
    "require_render_column",
    "coordinated_map",
    "coordinated_filter",
    "coordinated_dataset_operation",
    "filter_by_length",
    "log_dataset_examples",
    "process_dataset_with_map_and_filter",
    "pack_dataset_coordinated",
    "dataset_total_size",
]


def _resolve_dataset_num_proc() -> int:
    """Default map/filter ``num_proc`` — worker PROCESSES, capped at 4 because in distributed
    training (ranks × num_proc) each extra process adds memory + IPC cost.

    ``HALO_DATASET_NUM_PROC`` pins it fleet-wide: HF keys its per-worker cache files by num_proc, so
    a node deriving a different CPU-based value misses the writer rank's cache set and re-maps.
    """
    name = "HALO_DATASET_NUM_PROC"
    override = env_int(name, None)
    if override is None:
        # max(1, ...): a <=7-core box floors the quarter-share to 0, which datasets reads as
        # "no multiprocessing" on some paths and rejects on others.
        return max(1, min(multiprocessing.cpu_count() // 4, 4))
    if override < 1:
        raise ValueError(
            f"{name}={override} is invalid: the fleet-wide num_proc pin must be >= 1 "
            f"(use 1 to disable dataset-map multiprocessing)."
        )
    return override


DATASET_NUM_PROC = _resolve_dataset_num_proc()

# Folded into every cache name: get_function_identifier hashes no third-party source, so without
# this a transformers/TRL bump that changes render or packing output would reuse pre-bump caches.
_RENDER_LIBRARY_VERSIONS = f"tf{importlib.metadata.version('transformers')}-trl{importlib.metadata.version('trl')}"

# A post-map filter dropping at least this share of the corpus is reported as a WARNING: legitimate
# drop rates exist, but a rate this high is usually a config bug (see :func:`report_rejected_rows`).
_HIGH_REJECTION_WARN_FRACTION = 0.5

_SAMPLE_FILE_NAMES = {"train": "train_sample.txt", "test": "eval_sample.txt", "eval": "eval_sample.txt"}

# Values a cache-key fingerprint can read directly; anything else takes the one-time skip warning.
_SCALAR_TYPES = (str, int, float, bool, type(None))

# (owner, type name) pairs already reported by :func:`_warn_unfingerprintable` — one line per shape,
# not per row.
_UNFINGERPRINTABLE_WARNED: set[tuple[str, str]] = set()

# Owned by coordinated_dataset_operation and refused from a caller's kwargs: they steer HOW an
# operation runs, never what it produces. The worker count is NOT here — ``num_proc`` is a named
# parameter of both coordinated ops, so it can never arrive through ``**kwargs`` to be refused.
_MANAGED_OPERATION_KWARGS = frozenset({"keep_in_memory", "load_from_cache_file"})


def resolve_map_num_proc(configured: int | None) -> int:
    """``dataset_num_proc`` → map/filter worker count. Unset means the toolkit default, not 1 worker."""
    return configured or DATASET_NUM_PROC


def dataset_total_size(dataset: Dataset | DatasetDict) -> int:
    """Get total number of examples across all splits."""
    if isinstance(dataset, Dataset):
        return len(dataset)
    return sum(len(ds) for ds in dataset.values())


def _referenced_local_functions(func: Callable, seen: set) -> list[Callable]:
    """Repo-local functions ``func``'s code references through its module globals.

    They shape the output like the function's own body, and a module global is invisible to source
    and closure inspection. Third-party callables are excluded: their source churns without
    changing our outputs.
    """
    code = getattr(func, "__code__", None)
    func_globals = getattr(func, "__globals__", None)
    if code is None or func_globals is None:
        return []
    referenced = []
    for name in code.co_names:
        value = func_globals.get(name)
        if (
            isinstance(value, types.FunctionType)
            and value not in seen
            and getattr(value, "__module__", "").startswith(("src.", "scripts."))
        ):
            referenced.append(value)
    return referenced


def get_function_identifier(func: Callable, _seen: set | None = None) -> str:
    """Deterministic, cache-safe identifier for a function (hashes its source/bytecode, not its
    address), folding in the source of every repo-local helper the code references — editing a
    helper invalidates the caches of its callers."""
    # partial hides its func + bound args from source inspection: unfolded, every partial shares one id.
    if isinstance(func, functools.partial):
        base = get_function_identifier(func.func, _seen)
        bound = _get_kwargs_fingerprint({f"_arg{i}": a for i, a in enumerate(func.args)} | dict(func.keywords or {}))
        return f"partial[{base}]_{bound}"

    func_name = getattr(func, "__name__", "unknown")
    seen = _seen if _seen is not None else set()
    seen.add(func)

    try:
        source = inspect.getsource(func)
        helper_ids = [get_function_identifier(helper, seen) for helper in _referenced_local_functions(func, seen)]
        code_hash = hashlib.md5("".join(source.split() + sorted(helper_ids)).encode()).hexdigest()[:12]
        return f"{func_name}_{code_hash}"
    except (OSError, TypeError):
        logger.debug("Could not get source for %s, falling back to bytecode hash", func_name)

    try:
        code = func.__code__
        code_repr = f"{code.co_argcount}_{code.co_nlocals}_{code.co_code[:32].hex()}"
        code_hash = hashlib.md5(code_repr.encode()).hexdigest()[:12]
        return f"{func_name}_{code_hash}"
    except AttributeError:
        logger.debug("Could not get bytecode for %s, using qualified name", func_name)

    module = getattr(func, "__module__", "")
    qualname = getattr(func, "__qualname__", func_name)
    return f"{module}.{qualname}".replace("<", "").replace(">", "")


def _template_sig(chat_template: Any) -> str | None:
    """Short hash of a chat template. Dict templates (multi-template processors) hash per-name."""
    if isinstance(chat_template, str):
        return hashlib.md5(chat_template.encode()).hexdigest()[:8]
    if isinstance(chat_template, dict):
        joined = "|".join(f"{name}={template}" for name, template in sorted(chat_template.items()))
        return hashlib.md5(joined.encode()).hexdigest()[:8]
    return None


def _leaf_tokenizer_identity(val: Any) -> str | None:
    """Cache-key signature for a bare tokenizer, or None if ``val`` isn't one.

    The template hash and the special-token ids are the load-bearing terms: ``--force_chat_template``
    and an in-vocab ``--eos-token``/``--bos-token``/``--pad-token`` override both change the ids a map
    bakes into every row while leaving name_or_path, vocab_size and len identical.
    """
    name_or_path = getattr(val, "name_or_path", None)
    vocab_size = getattr(val, "vocab_size", None)
    if name_or_path is None and vocab_size is None:
        return None
    try:
        length = len(val)
    except TypeError:
        length = None
    specials = ":".join(
        str(getattr(val, attr, None))
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "padding_side", "truncation_side")
    )
    return f"{type(val).__name__}:{name_or_path}:{vocab_size}:{length}:{_template_sig(getattr(val, 'chat_template', None))}:{specials}"


def _tokenizer_identity(val: Any) -> str | None:
    """Cache-key signature for a tokenizer or processor-like object, or None if it is neither.

    A processor exposes neither ``name_or_path`` nor ``vocab_size``, so its identity is its inner
    ``.tokenizer`` plus its own chat template; without the descent two checkpoints sharing a processor
    class collide on one cache file. The descent is typed on ``ProcessorMixin`` because only that
    claims completeness — any other ``.tokenizer`` holder (a collator) also carries render knobs this
    signature cannot see, so it keys by class name and its knobs belong in ``cache_key_extras``.
    """
    leaf = _leaf_tokenizer_identity(val)
    if leaf is not None:
        return leaf
    if not isinstance(val, ProcessorMixin):
        return None
    inner = getattr(val, "tokenizer", None)
    inner_sig = _leaf_tokenizer_identity(inner) if inner is not None and inner is not val else None
    template_sig = _template_sig(getattr(val, "chat_template", None))
    if inner_sig is None and template_sig is None:
        return None
    return f"{type(val).__name__}[{inner_sig}]:{template_sig}"


def _warn_unfingerprintable(owner: str, kind: str) -> None:
    """Report ONCE per (owner, type) that a cache-key input could not be fingerprinted: a value the
    key cannot see does not invalidate the cache, so editing it reuses the previous run's rows."""
    warn_once(
        logger,
        _UNFINGERPRINTABLE_WARNED,
        (owner, kind),
        "Dataset-map cache fingerprint for %s skips a value of type %s it cannot fingerprint — "
        "changes to that value will NOT invalidate the cache. If it affects the map output, thread "
        "it through cache_key_extras.",
        owner,
        kind,
    )


def _json_content_repr(val: Any) -> str | None:
    """Deterministic content repr for a JSON-serializable value, ``None`` when it is not one.

    Nested collections — a dict of flags, a list of pairs — carry the whole cache-relevant content
    and no scalar branch can read them; without this they key on their TYPE NAME, so two different
    values of one knob share a cache file and the second run loads the first's rows.
    """
    try:
        return json.dumps(val, sort_keys=True)
    except (TypeError, ValueError):
        return None


def _get_kwargs_fingerprint(kwargs: dict) -> str:
    """Fingerprint fn_kwargs so the same map fn with different kwargs (tokenizer, etc.) keys a
    different cache, else stale cache loads another model's token IDs.
    """
    if not kwargs:
        return ""
    parts = []
    for key in sorted(kwargs.keys()):
        val = kwargs[key]
        try:
            tok_sig = _tokenizer_identity(val)
            if tok_sig is not None:
                parts.append(f"{key}={tok_sig}")
            elif isinstance(val, (str, int, float, bool)):
                parts.append(f"{key}={val}")
            elif isinstance(val, (list, tuple, set, frozenset)) and all(
                isinstance(item, (str, int, float, bool, type(None))) for item in val
            ):
                # Sets are sorted first: their iteration order is hash-seeded, so the raw repr
                # would key a different cache per process for identical content.
                ordered = sorted(val, key=lambda item: (item is None, str(type(item)), str(item)))
                parts.append(f"{key}={ordered!r}")
            elif (content := _json_content_repr(val)) is not None:
                parts.append(f"{key}={content}")
            else:
                if getattr(val, "tokenizer", None) is not None:
                    # Refused by the ProcessorMixin gate above: keying by class name is the right
                    # answer, but a silent one — the render knobs it hides must ride cache_key_extras
                    # or editing them reuses the previous run's rows.
                    _warn_unfingerprintable(f"fn_kwargs[{key}]", type(val).__name__)
                parts.append(f"{key}={type(val).__name__}")
        except Exception:
            # A kwarg the key cannot read is a cache the kwarg cannot invalidate — report it.
            _warn_unfingerprintable(f"fn_kwargs[{key}]", type(kwargs[key]).__name__)
            parts.append(f"{key}=?")
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


def _scalar_collection_repr(val: Any) -> str | None:
    """Deterministic repr for a list/tuple/set of scalars, or None if ``val`` isn't one.

    Sets are sorted so the fingerprint doesn't depend on iteration order. Dicts are deliberately
    NOT fingerprinted: captured dicts are schema carriers (``none_example``), pinned as
    cache-irrelevant — they take the one-time skip warning instead.
    """
    if isinstance(val, (list, tuple)) and all(isinstance(item, _SCALAR_TYPES) for item in val):
        return repr(val)
    if isinstance(val, (set, frozenset)) and all(isinstance(item, _SCALAR_TYPES) for item in val):
        return repr(sorted(val, key=repr))
    return None


def _get_closure_fingerprint(func: Callable) -> str:
    """Fingerprint output-affecting values captured in ``func``'s closure (tokenizer, max_length,
    flags, scalar collections, helper functions).

    Free variables are invisible to :func:`get_function_identifier` and
    :func:`_get_kwargs_fingerprint`, so without this two runs differing only in tokenizer or
    max_length collide on one cache file. Any other cell type is skipped with a one-time warning;
    thread such values through ``cache_key_extras``.
    """
    closure = getattr(func, "__closure__", None)
    if not closure:
        return ""
    parts = []
    for cell in closure:
        try:
            val = cell.cell_contents
        except ValueError:
            continue  # unbound cell
        if isinstance(val, _SCALAR_TYPES):
            parts.append(repr(val))
            continue
        collection_repr = _scalar_collection_repr(val)
        if collection_repr is not None:
            parts.append(collection_repr)
            continue
        try:
            tok_sig = _tokenizer_identity(val)
        except Exception:  # a raising attribute/property must land in the skip warning, not crash cache naming
            tok_sig = None
        if tok_sig is not None:
            parts.append(tok_sig)
            continue
        if isinstance(val, (types.FunctionType, functools.partial)):
            # A captured helper shapes the output as much as the wrapper's own source — recurse so
            # editing it invalidates the cache. Plain functions only; callable tokenizers go above.
            parts.append(get_function_identifier(val))
            continue
        _warn_unfingerprintable(getattr(func, "__qualname__", repr(func)), type(val).__name__)
    if not parts:
        return ""
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


def ensure_cache_dir(writer_rank: Callable[[], bool] = is_local_main_process) -> str:
    """Return the datasets cache dir every rank agrees on (``HF_DATASETS_CACHE`` or HF's own
    default), created by ``writer_rank``.

    Local rank 0 by default (the per-node cache write); the run's one-time setup passes
    :func:`fs_aware_load_rank` so a shared input FS creates it once for the job. ``fs_aware_makedirs``
    fences the write, else a read-only or full volume raises on one rank while its peers wait out the
    barrier instead of reporting the errno.
    """
    cache_dir = os.environ.get("HF_DATASETS_CACHE") or str(datasets.config.HF_DATASETS_CACHE)
    fs_aware_makedirs(cache_dir, writer_rank=writer_rank)
    return cache_dir


def report_rejected_rows(original_size: int, kept_size: int, context: str) -> None:
    """Log how much of a corpus a post-map filter dropped — WARNING past
    :data:`_HIGH_REJECTION_WARN_FRACTION`, INFO below it.

    One implementation behind every drop-and-continue filter, because the magnitude is the signal:
    legitimately high drop rates exist, but most of a corpus vanishing is a config bug that would
    otherwise scroll by at INFO. Local main only, like every other dataset-stage log.
    """
    if not is_local_main_process():
        return
    rejected = original_size - kept_size
    fraction = rejected / original_size if original_size > 0 else 0.0
    message = f"Filtered out {rejected}/{original_size} examples ({100 * fraction:.1f}%) rejected during {context}"
    if fraction >= _HIGH_REJECTION_WARN_FRACTION:
        logger.warning(
            f"{message}. A rejection rate this high usually means a config bug — an "
            f"assistant_message_template that never matches the rendered turns, a wrong "
            f"conversation/text field, or a max_length below the typical sequence length."
        )
    else:
        logger.info(message)


def missing_render_column_splits(dataset: DatasetDict, column: str) -> list[str]:
    """Splits this loader will render that do not carry ``column``.

    Only the splits the loader goes on to render: an extra split a source happens to carry
    (e.g. "validation") is never filtered or mapped here, so its schema is not a contract.
    """
    consumed = [split for split in ("train", "test") if split in dataset]
    return sorted(split for split in consumed if column not in dataset[split].column_names)


def require_render_column(dataset: DatasetDict, path: str, knob: str, column: str) -> None:
    """Fail loud when a declared render column is absent from a raw dataset.

    A typo'd column would otherwise surface inside an HF ``map`` worker, long after a multi-node model
    load, as a ``KeyError`` naming neither the knob nor the dataset — and for an optional knob
    (``tools_field``) not at all: the rows just render without tools. One home for every consumer that
    declares a render column.
    """
    missing = missing_render_column_splits(dataset, column)
    if not missing:
        return
    consumed = [split for split in ("train", "test") if split in dataset]
    available = sorted(set().union(*(dataset[split].column_names for split in consumed)))
    raise ValueError(
        f"{knob}='{column}' names a column the dataset {path} does not carry (missing from split(s) "
        f"{missing}; available columns: {available}). Point {knob} at an existing column in the YAML."
    )


def _rank_stable_dataset_id(dataset: Dataset) -> str:
    """A content-stable, rank- and run-independent identifier for a Dataset.

    HF's ``_fingerprint`` and ``cache_files`` both diverge across ranks, so neither can key a shared
    cache. Prefer the ``_toolkit_cache_key`` stamped in :func:`coordinated_dataset_operation`, then
    ``cache_files`` for untouched datasets, then ``_fingerprint`` for in-memory ones.
    """
    stamp = getattr(dataset, "_toolkit_cache_key", None)
    if stamp:
        return stamp
    cache_files = getattr(dataset, "cache_files", None) or []
    paths = sorted(cf["filename"] for cf in cache_files if isinstance(cf, dict) and cf.get("filename"))
    if paths:
        return hashlib.md5("|".join(paths).encode()).hexdigest()[:16]
    return getattr(dataset, "_fingerprint", None) or "nofp"


def _dataset_content_fingerprint(dataset: Dataset | DatasetDict) -> str:
    """Content-sensitive fingerprint for a Dataset or DatasetDict: row count + :func:`_rank_stable_dataset_id`.

    For a ``DatasetDict``, aggregates ``(split, rows, id)`` per split — it has no fingerprint and
    ``len()`` is the split count, so per-split info is needed to avoid cross-run key collisions.
    """
    if isinstance(dataset, DatasetDict):
        parts = [
            f"{name}:{len(dataset[name])}:{_rank_stable_dataset_id(dataset[name])}" for name in sorted(dataset.keys())
        ]
        return "|".join(parts)
    return f"{len(dataset)}:{_rank_stable_dataset_id(dataset)}"


def reject_self_capturing_fn(func: Callable, num_proc: int | None, operation: str) -> None:
    """Reject a bound method as a multi-worker map/filter callable.

    At ``num_proc > 1`` dill pickles the callable by value, dragging a bound method's whole ``self``
    graph to every worker — the model, and under EP the unpicklable DeepEP/NCCL process groups.
    Single-worker maps pickle nothing, so they are left alone.
    """
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


def _build_cache_file_name(
    operation: str,
    func: Callable,
    dataset: Dataset | DatasetDict,
    desc: str | None,
    kwargs: dict,
    cache_key_extras: dict | None = None,
) -> str:
    """Build a deterministic cache file name for a dataset operation.

    Combines function identity, closure fingerprint, dataset content fingerprint and every kwarg
    fingerprint, so the key changes whenever the tokenizer, config, call shape or data does. Every
    kwarg reaching here shapes the output — the execution knobs are refused at the op and the worker
    count is a named parameter. ``cache_key_extras`` threads in tunables the closure cannot reach.
    """
    func_id = get_function_identifier(func)
    closure_fp = _get_closure_fingerprint(func)
    fn_kwargs_fp = _get_kwargs_fingerprint(kwargs.get("fn_kwargs", {}))
    op_kwargs = {k: v for k, v in kwargs.items() if k != "fn_kwargs"}
    op_kwargs_fp = _get_kwargs_fingerprint(op_kwargs)
    extras_fp = _get_kwargs_fingerprint(cache_key_extras or {})
    ds_fingerprint = _dataset_content_fingerprint(dataset)
    cache_key = (
        f"{operation}_{desc or 'default'}_{func_id}_{closure_fp}_{fn_kwargs_fp}_{op_kwargs_fp}"
        f"_{extras_fp}_{ds_fingerprint}_{_RENDER_LIBRARY_VERSIONS}"
    )
    return f"cache-{hashlib.md5(cache_key.encode()).hexdigest()}.arrow"


def _reject_managed_operation_kwargs(kwargs: dict, operation_name: str) -> None:
    """Refuse the execution knobs :func:`coordinated_dataset_operation` sets itself.

    It pins ``load_from_cache_file``/``keep_in_memory`` (the on-disk cache IS the cross-rank
    transport), so a value passed here can never take effect — accepting one would leave a call site
    reading as if it steered caching.
    """
    owned = sorted(set(kwargs) & _MANAGED_OPERATION_KWARGS)
    if owned:
        raise TypeError(
            f"{operation_name} does not accept {owned}: coordinated dataset operations own these "
            f"execution knobs, so the value would be ignored. Drop them; the worker count is the "
            f"op's own num_proc parameter."
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
    _reject_managed_operation_kwargs(op_kwargs, operation_name)
    reject_self_capturing_fn(fn, num_proc, operation_name)
    cache_file_name = _build_cache_file_name(
        operation, fn, dataset, desc, op_kwargs, cache_key_extras=cache_key_extras
    )

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
    """Coordinated, deterministically-cached ``dataset.map``.

    cache_key_extras fingerprints closure-captured tunables that affect output but aren't visible in
    the function source (e.g. tools_field, system_prompt).
    """
    return _cached_map_or_filter("map", dataset, map_fn, desc, num_proc, map_kwargs, cache_key_extras)


def coordinated_filter(
    dataset: Dataset | DatasetDict,
    filter_fn: Callable,
    desc: str | None = None,
    num_proc: int = DATASET_NUM_PROC,
    **filter_kwargs,
) -> Dataset | DatasetDict:
    """Coordinated, deterministically-cached ``dataset.filter``."""
    return _cached_map_or_filter("filter", dataset, filter_fn, desc, num_proc, filter_kwargs)


def filter_by_length(
    dataset: Dataset,
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    num_proc: int = DATASET_NUM_PROC,
) -> Dataset:
    """Drop rows whose tokenized ``text`` column exceeds ``max_length``.

    Reports through :func:`report_rejected_rows` like every other drop-and-continue filter, so a
    length budget that removes most of the corpus takes the high-rejection warning instead of
    scrolling by at INFO.
    """
    original_size = len(dataset)

    def check_length(example):
        if example.get("text") is None:
            return False
        tokenized = tokenizer(example["text"], truncation=False, padding=False)
        return len(tokenized["input_ids"]) <= max_length

    context = f"length filtering (max_length={max_length})"
    filtered_dataset = coordinated_filter(dataset, check_length, desc=context, num_proc=num_proc)
    report_rejected_rows(original_size, len(filtered_dataset), context)
    return filtered_dataset


def log_dataset_examples(
    datasets: dict[str, Dataset],
    num_examples: int = 1,
    *,
    tokenizer: PreTrainedTokenizer | None = None,
    output_dir: str | None = None,
    write_decoded_samples: bool = False,
) -> None:
    """Log example rows (global main only); optionally dump decoded samples to disk.

    With write_decoded_samples plus a tokenizer + output_dir, the first num_examples rows of each
    dataset with an ``input_ids`` column are decoded to ``<output_dir>/log/{train,eval,...}_sample.txt``.
    The file write is gated on the FS-aware save rank — the same gate ``run.log`` uses — so a
    non-shared filesystem gets the samples beside the log on EVERY node, and a shared one still has a
    single writer.
    """
    log_examples = is_global_main_process()
    write_samples = write_decoded_samples and fs_aware_save_rank()
    if not (log_examples or write_samples):
        return

    for name, dataset in datasets.items():
        if dataset is None:
            continue

        if isinstance(dataset, DatasetDict):
            if len(dataset) == 0:
                continue
            split_name = next(iter(dataset))
            split_dataset = dataset[split_name]
            header = f"Example(s) from {name} dataset (split: {split_name}):"
        else:
            split_dataset = dataset
            header = f"Example(s) from {name} dataset:"

        if len(split_dataset) == 0:
            continue

        n = min(num_examples, len(split_dataset))
        if log_examples:
            logger.info(header)
            for i in range(n):
                logger.info(split_dataset[i])

        if write_samples and tokenizer is not None and output_dir is not None:
            _write_decoded_samples(
                split_dataset, name=name, num_examples=n, tokenizer=tokenizer, output_dir=output_dir
            )


def _write_decoded_samples(
    dataset: Dataset,
    *,
    name: str,
    num_examples: int,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
) -> None:
    if "input_ids" not in dataset.column_names:
        return

    log_dir = os.path.join(output_dir, RUN_LOG_DIR_NAME)
    os.makedirs(log_dir, exist_ok=True)
    file_name = _SAMPLE_FILE_NAMES.get(name, f"{name}_sample.txt")
    out_path = os.path.join(log_dir, file_name)

    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(num_examples):
            input_ids = dataset[i]["input_ids"]
            decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
            f.write(f"=== Sample {i + 1} ({len(input_ids)} tokens) ===\n")
            f.write(decoded)
            f.write("\n\n")
    logger.info(f"Wrote {num_examples} decoded sample(s) to {out_path}")


def _run_or_record(operation_fn: Callable, operation_kwargs: dict, label: str) -> tuple[Any, BaseException | None]:
    """Run ``operation_fn``, returning ``(result, failure)`` — a failure RECORDED, never raised here.

    A rank-local raise strands the peers in the join that follows, so a plain data error reads as a
    hang; :func:`_reject_operation_failure` turns the record into a raise on every rank.
    ``BaseException``, so a ``KeyboardInterrupt`` reaches the join too — at the cost of waiting for
    the peers, bounded by ``DIST_STORE_TIMEOUT_HOURS``.
    """
    logger.info(f"{label}...")
    try:
        result = operation_fn(**operation_kwargs)
    except BaseException as e:
        return None, e
    logger.info(f"{label} completed")
    return result, None


def _reject_operation_failure(failure: BaseException | None, operation_name: str) -> None:
    """Abort EVERY rank when the rank that ran the operation failed. Collective-EQUIVALENT — all
    ranks call it, in the same order.

    The peers take a uniform ``RuntimeError`` carrying the cause; the rank that failed re-raises its
    OWN exception, keeping the type callers rely on (``tokenize_vlm_dataset``'s capability refusal is
    a ``NotImplementedError`` by contract). The join rides the c10d store rather than a collective:
    peers wait out the writer's entire map, which the NCCL watchdog would cap and kill.
    """
    store_join_recorded_failure("dataset_op", failure, f"Dataset operation {operation_name!r}")


def coordinated_dataset_operation(
    operation_fn: Callable,
    dataset: Dataset | DatasetDict,
    operation_name: str = "dataset operation",
    num_proc: int = DATASET_NUM_PROC,
    cache_file_name: str | None = None,
    **kwargs,
) -> Dataset | DatasetDict:
    """Run a dataset op with distributed coordination via the deterministic cache file.

    Scope follows the INPUT filesystem: on a shared one only the global main maps (no NFS/Lustre write
    race), on a non-shared one each node's local main does; the others wait and load the cache.
    ``operation_fn`` must capture the dataset in its closure. num_proc<=1 maps to None (HF still spawns
    subprocesses at 1).

    Both phases join through :func:`_reject_operation_failure` on the c10d store, not a barrier: the
    op runs on one rank while the rest wait, so a failure has to be carried to them, and the store
    budget lets peers wait out an hours-long fresh-cache map the NCCL watchdog would kill.

    This IS the rank ordering — never wrap a call in a main-first block
    (``PartialState().local_main_process_first()``, :func:`fs_aware_main_first`). Those hold the peers
    outside the body, so ``ensure_cache_dir``'s barrier runs on the main rank alone and the store
    joins go permanently off-by-one on the equal-entry invariant.
    """
    global_rank = get_global_rank()
    is_main = fs_aware_load_rank()

    cache_dir = ensure_cache_dir()

    cache_path = None
    cache_file_names = None

    if cache_file_name:
        cache_path = os.path.join(cache_dir, cache_file_name)

        if isinstance(dataset, DatasetDict):
            cache_file_names = {}
            for split_name in dataset:
                split_cache_name = cache_file_name.replace(".arrow", f"_{split_name}.arrow")
                cache_file_names[split_name] = os.path.join(cache_dir, split_cache_name)

    # None (not 1) fully disables multiprocessing — num_proc=1 still spawns HF subprocesses (can crash)
    effective_num_proc = None if num_proc is not None and num_proc <= 1 else num_proc

    operation_kwargs = {
        "num_proc": effective_num_proc,
        "load_from_cache_file": True,
        **kwargs,
    }

    if cache_file_name:
        if "keep_in_memory" in operation_kwargs:
            del operation_kwargs["keep_in_memory"]

        if isinstance(dataset, Dataset):
            operation_kwargs["cache_file_name"] = cache_path
        elif isinstance(dataset, DatasetDict):
            operation_kwargs["cache_file_names"] = cache_file_names

    result = None
    failure = None
    if is_main:
        result, failure = _run_or_record(
            operation_fn, operation_kwargs, f"[{operation_name}] Main (rank {global_rank}) processing dataset"
        )
    _reject_operation_failure(failure, operation_name)

    if not is_main:
        result, failure = _run_or_record(
            operation_fn, operation_kwargs, f"[{operation_name}] Rank {global_rank} loading from local cache"
        )
    _reject_operation_failure(failure, operation_name)

    # HF's _fingerprint/cache_files diverge between writer and loader ranks — stamp a rank-stable key.
    if cache_file_name:
        if isinstance(result, Dataset):
            result._toolkit_cache_key = cache_file_name
        elif isinstance(result, DatasetDict) and cache_file_names:
            for split_name, split_ds in result.items():
                if isinstance(split_ds, Dataset) and split_name in cache_file_names:
                    split_ds._toolkit_cache_key = os.path.basename(cache_file_names[split_name])

    return result


def process_dataset_with_map_and_filter(
    dataset: Dataset | DatasetDict,
    process_fn: Callable,
    filter_field: str = "input_ids",
    num_proc: int = DATASET_NUM_PROC,
    remove_columns: list[str] | None = None,
    desc: str | None = None,
    log_stats: bool = True,
    cache_key_extras: dict | None = None,
) -> Dataset | DatasetDict:
    """Coordinated map then drop rejection sentinels (see :func:`is_valid_example` on filter_field)."""
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

    if log_stats:
        report_rejected_rows(original_size, dataset_total_size(filtered), f"processing (checked via {filter_field})")

    return filtered


def pack_dataset_coordinated(
    dataset: Dataset,
    seq_length: int,
    strategy: str = "bfd",
    split: str = "train",
) -> Dataset:
    """Pack a tokenized dataset to a deterministic cache file with rank coordination.

    Wraps :func:`trl.pack_dataset`: the main rank packs inside :func:`fs_aware_main_first` while the
    others wait and reuse the cache — unguarded, ``pack_dataset`` materializes a full corpus copy per
    rank. ``dataset`` must be on-disk for the explicit ``cache_file_name`` to be picked up.
    """
    cache_key = _rank_stable_dataset_id(dataset)
    # ensure_cache_dir() barriers, so it must stay OUTSIDE the phase below: inside, only the main
    # rank would reach the barrier while its peers sit on a store key, hanging the job.
    cache_file_name = os.path.join(
        ensure_cache_dir(),
        f"packed_{strategy}_{seq_length}_{cache_key}_{split}_{_RENDER_LIBRARY_VERSIONS}.arrow",
    )
    with fs_aware_main_first("dataset_pack"):
        return _trl_pack_dataset(
            dataset,
            seq_length=seq_length,
            strategy=strategy,
            map_kwargs={
                "cache_file_name": cache_file_name,
                "load_from_cache_file": True,
            },
        )
