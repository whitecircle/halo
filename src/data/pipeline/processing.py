"""Distributed dataset processing with multi-process coordination.

Main rank runs each map/filter to a deterministic cache file; others wait then load it — no
duplicate work / per-rank copies. Set HF_DATASETS_CACHE for a consistent cache dir. Prefer the
high-level ``coordinated_map`` / ``coordinated_filter``.
"""

import contextlib
import functools
import hashlib
import importlib.metadata
import inspect
import json
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
from transformers import PreTrainedTokenizer, ProcessorMixin
from trl.data_utils import pack_dataset as _trl_pack_dataset

from src.data.pipeline.row_processors import is_valid_example
from src.data.sources.dataset_cache import publish_cached_download
from src.data.sources.paths import eval_split_name
from src.distributed.filesystem import RUN_LOG_DIR_NAME, store_join_recorded_failure
from src.distributed.runtime import (
    broadcast_from_rank0,
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
    "coordinated_dataset_transform",
    "carry_cache_key",
    "run_load_rank_first",
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

# Decoded-sample file per dataset name, where it is not ``<name>_sample.txt``: the test split is the eval set.
_SAMPLE_FILE_NAMES = {"test": "eval_sample.txt"}

# Values a cache-key fingerprint can read directly; anything else takes the one-time skip warning.
_SCALAR_TYPES = (str, int, float, bool, type(None))

# Per-call state a fast tokenizer serializes alongside its content: ``PreTrainedTokenizerFast.__call__``
# rewrites both whenever a call passes ``truncation=``/``padding=``, so a signature over the raw
# ``backend_tokenizer.to_str()`` would key on whatever the process tokenized last — and the writer
# rank, the only one that runs a map's fn at ``num_proc <= 1``, would then diverge from its peers.
_MUTABLE_BACKEND_STATE_KEYS = ("padding", "truncation")

# (owner, type name) pairs already reported by :func:`_warn_unfingerprintable` — one line per shape,
# not per row.
_UNFINGERPRINTABLE_WARNED: set[tuple[str, str]] = set()

# Tokenizer type names whose content hash raised, reported once by :func:`_tokenizer_content_sig`.
_CONTENT_SIG_FAILED_WARNED: set[str] = set()

# Owned by coordinated_dataset_operation and refused from a caller's kwargs: they steer how an
# operation runs, not what it produces. The worker count is not listed, since ``num_proc`` is a named
# parameter of both coordinated ops and cannot arrive through ``**kwargs``.
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


def _tokenizer_content_sig(val: Any) -> str | None:
    """Content hash of what a tokenizer does to text: the fast backend's serialized state (vocab,
    merges, normalizer) minus its per-call ``_MUTABLE_BACKEND_STATE_KEYS``, else the vocab table.
    ``None`` when neither is exposed, or when reading one raises — the latter warned once per type,
    since the caller's ``name_or_path`` fallback misses the cache on every resume leg and reuses it
    across a same-path content change."""
    try:
        backend = getattr(val, "backend_tokenizer", None)
        if backend is not None:
            state = json.loads(backend.to_str())
            for key in _MUTABLE_BACKEND_STATE_KEYS:
                state.pop(key, None)
            return hashlib.md5(json.dumps(state, sort_keys=True).encode()).hexdigest()[:16]
        get_vocab = getattr(val, "get_vocab", None)
        if callable(get_vocab):
            return hashlib.md5(json.dumps(sorted(get_vocab().items())).encode()).hexdigest()[:16]
    except Exception as exc:
        # Not narrowable: ``to_str`` raises a bare ``Exception`` for a custom Python component, and a
        # remote-code ``get_vocab`` raises whatever it raises (bytes keys fail ``json.dumps``).
        warn_once(
            logger,
            _CONTENT_SIG_FAILED_WARNED,
            type(val).__name__,
            "Dataset-map cache key cannot hash the content of tokenizer %s (%s: %s) and keys on its "
            "name_or_path %r instead: a resume from a checkpoint path re-runs every map, and a changed "
            "tokenizer at the same path reuses the previous run's rows.",
            type(val).__name__,
            type(exc).__name__,
            exc,
            getattr(val, "name_or_path", None),
        )
        return None
    return None


def _leaf_tokenizer_identity(val: Any) -> str | None:
    """Cache-key signature for a bare tokenizer, or None if ``val`` isn't one.

    The template hash and the special-token ids are the load-bearing terms: ``--force_chat_template``
    and an in-vocab ``--eos-token``/``--bos-token``/``--pad-token`` override both change the ids a map
    bakes into every row while leaving content, vocab_size and len identical. The source term is the
    tokenizer's CONTENT hash, not its path: a resume repoints the load at the run's own checkpoint,
    and a path term would miss the cache on every resume leg while the tokenization function is
    unchanged. ``name_or_path`` is the fallback only where no content is readable.
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
    source = _tokenizer_content_sig(val) or name_or_path
    return f"{type(val).__name__}:{source}:{vocab_size}:{length}:{_template_sig(getattr(val, 'chat_template', None))}:{specials}"


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
            elif (content := _scalar_collection_repr(val) or _json_content_repr(val)) is not None:
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


def _rendered_splits(dataset: DatasetDict) -> list[str]:
    """The splits of ``dataset`` a loader renders: ``train`` and its held-out split
    (:func:`~src.data.sources.paths.eval_split_name`) — a lone ``validation`` split is rendered as
    the test split, whether the training loader renamed it already or ``prepare_dataset`` bakes it."""
    return [split for split in ("train", eval_split_name(dataset)) if split in dataset]


def missing_render_column_splits(dataset: DatasetDict, column: str) -> list[str]:
    """Splits this loader will render that do not carry ``column``.

    Only the splits the loader goes on to render: an extra split a source happens to carry
    (a "validation" split beside a "test" one) is never filtered or mapped here, so its schema is
    not a contract.
    """
    return sorted(split for split in _rendered_splits(dataset) if column not in dataset[split].column_names)


def require_render_column(dataset: DatasetDict, path: str, knob: str, column: str) -> None:
    """Fail loud when a declared render column is absent from a raw dataset.

    A typo'd column would otherwise surface inside an HF ``map`` worker, long after a multi-node model
    load, as a ``KeyError`` naming neither the knob nor the dataset — and for an optional knob
    (``tools_field``) not at all: the rows just render without tools. One home for every consumer that
    declares a render column, whether ``knob`` is a config field or a CLI flag.
    """
    missing = missing_render_column_splits(dataset, column)
    if not missing:
        return
    available = sorted(set().union(*(dataset[split].column_names for split in _rendered_splits(dataset))))
    raise ValueError(
        f"{knob}='{column}' names a column the dataset {path} does not carry (missing from split(s) "
        f"{missing}; available columns: {available}). Point {knob} at an existing column."
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


def carry_cache_key(source: Dataset, derived: Dataset, step: str) -> Dataset:
    """Stamp ``derived`` with ``source``'s rank-stable cache key extended by ``step``, the deterministic
    transform that produced it (a column rename).

    ``datasets`` deep-copies the stamp onto a renamed dataset unchanged, so without the extension the
    renamed rows key exactly like their source: a coordinated cache (a TRL preparation that renders
    the aliased ``tools`` column) would serve a run that aliased nothing the rows of one that did, or
    the reverse. An unstamped ``source`` leaves ``derived`` as it is.
    """
    stamp = getattr(source, "_toolkit_cache_key", None)
    if stamp:
        derived._toolkit_cache_key = f"{stamp}|{step}"
    return derived


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

    It pins ``load_from_cache_file`` and keeps ``keep_in_memory`` at its default (off), since the
    on-disk cache is the cross-rank transport, so a value passed here could never take effect and
    accepting one would leave a call site reading as if it steered caching.
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


def _run_or_record(work: Callable[[], Any], label: str) -> tuple[Any, BaseException | None]:
    """Run ``work``, returning ``(result, failure)`` — a failure RECORDED, never raised here.

    A rank-local raise strands the peers in the join that follows, so a plain data error reads as a
    hang; :func:`_reject_operation_failure` turns the record into a raise on every rank.
    ``BaseException``, so a ``KeyboardInterrupt`` reaches the join too — at the cost of waiting for
    the peers, bounded by ``DIST_STORE_TIMEOUT_HOURS``.
    """
    logger.info(f"{label}...")
    try:
        result = work()
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


def run_load_rank_first(work: Callable[[], Any], operation_name: str) -> Any:
    """Run ``work`` on the load rank first, then on every other rank, joining each phase on the store.

    The ordering for read-side work whose first run fills a cache the others read back — a dataset
    map or filter, a pack, a whole-source load. The load rank follows the INPUT filesystem
    (:func:`~src.distributed.runtime.fs_aware_load_rank`): global rank 0 on a shared one, each node's
    local rank 0 on per-node storage. Both joins ride the c10d store, so the peers wait out an
    hours-long first run under ``DIST_STORE_TIMEOUT_HOURS`` instead of the NCCL watchdog, and a
    failure raises its cause on every rank: a load rank that failed is never retried by its peers,
    and a peer that failed reading the cache back is not left behind a collective.

    Collective-equivalent: every rank calls it, in the same order — and never inside a main-first
    block, which would hold the peers outside both joins.
    """
    global_rank = get_global_rank()
    is_main = fs_aware_load_rank()
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
    """Run a dataset op with distributed coordination via the deterministic cache file.

    The op runs under :func:`run_load_rank_first`: on a shared INPUT filesystem only the global main
    maps (no NFS/Lustre write race), on a non-shared one each node's local main does; the others then
    load the cache. ``operation_fn`` must capture the dataset in its closure. num_proc<=1 maps to None
    (HF still spawns subprocesses at 1).

    This IS the rank ordering — never wrap a call in a main-first block
    (``PartialState().local_main_process_first()``, :func:`fs_aware_main_first`). Those hold the peers
    outside the body, so ``ensure_cache_dir``'s barrier runs on the main rank alone and the store
    joins go permanently off-by-one on the equal-entry invariant.
    """
    cache_dir = ensure_cache_dir()

    # None (not 1) fully disables multiprocessing — num_proc=1 still spawns HF subprocesses (can crash)
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
            split_ds._toolkit_cache_key = split_cache_names[split]
    else:
        result._toolkit_cache_key = cache_file_name

    return result


@contextlib.contextmanager
def _uncached_maps_in(staging_dir: str):
    """HF's map cache off, with every map's output written under ``staging_dir``, removed on exit.

    With the cache off, ``datasets`` writes each map over an on-disk input to one process-lifetime
    ``$TMPDIR/hf_datasets-*`` directory and keeps every intermediate there until exit (KTO's
    preparation runs five maps), on the root filesystem when only ``HF_DATASETS_CACHE`` was moved to
    the large volume. ``datasets`` exposes no setting for that directory, so its module-level holder
    is swapped for the block; ``.name`` is all ``datasets`` reads of it.
    """
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
    """Run a whole-dataset transform once per filesystem scope; every rank loads its published output.

    For preparation code that cannot be handed a cache file (TRL's ``_prepare_dataset``): its maps key
    HF's own fingerprints, and a map closing over the trainer hashes to a random one, so each rank
    writes a full copy of the result on every run. Here the load rank runs ``transform`` with the map
    cache off and publishes the result atomically under a name keyed like :func:`coordinated_map`'s —
    the input's rank-stable identity, ``cache_key_extras`` and the library versions — and the other
    ranks, and later runs with the same key, load it back. The transform's intermediate maps write a
    staging directory beside the published copy, removed once it finishes. ``transform`` must issue
    no collective: it runs on the load rank alone while the peers wait on the store.

    An input without the toolkit's content-derived stamp (one built outside the loaders) keys on this
    run alone: its file paths name neither its content nor an in-memory selection, so no later run
    may trust a match.
    """
    run_token = broadcast_from_rank0(uuid.uuid4().hex)
    scope = "" if getattr(dataset, "_toolkit_cache_key", None) else f"_run{run_token}"
    key = (
        f"{operation_name}_{_dataset_content_fingerprint(dataset)}_{_get_kwargs_fingerprint(cache_key_extras)}"
        f"_{_RENDER_LIBRARY_VERSIONS}{scope}"
    )
    target = os.path.join(ensure_cache_dir(), f"transform-{hashlib.md5(key.encode()).hexdigest()}")

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
    result._toolkit_cache_key = os.path.basename(target)
    return result


def process_dataset_with_map_and_filter(
    dataset: Dataset | DatasetDict,
    process_fn: Callable,
    filter_field: str = "input_ids",
    num_proc: int = DATASET_NUM_PROC,
    remove_columns: list[str] | None = None,
    desc: str | None = None,
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

    report_rejected_rows(original_size, dataset_total_size(filtered), f"processing (checked via {filter_field})")
    return filtered


def pack_dataset_coordinated(
    dataset: Dataset,
    seq_length: int,
    strategy: str = "bfd",
    split: str = "train",
) -> Dataset:
    """Pack a tokenized dataset to a deterministic cache file with rank coordination.

    Wraps :func:`trl.pack_dataset` in :func:`run_load_rank_first`: the main rank packs while the
    others wait, then they reuse its cache — unguarded, ``pack_dataset`` materializes a full corpus
    copy per rank. ``dataset`` must be on-disk for the explicit ``cache_file_name`` to be picked up.
    """
    cache_key = _rank_stable_dataset_id(dataset)
    # ensure_cache_dir() barriers, so it must stay OUTSIDE the phases below: inside, only the main
    # rank would reach the barrier while its peers sit on a store key, hanging the job.
    cache_file_name = os.path.join(
        ensure_cache_dir(),
        f"packed_{strategy}_{seq_length}_{cache_key}_{split}_{_RENDER_LIBRARY_VERSIONS}.arrow",
    )
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
