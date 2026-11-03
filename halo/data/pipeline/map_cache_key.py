"""What keys a coordinated dataset map's cache: the callable's source and closure, its arguments, the input's
rank-stable identity and the render-library versions. The source fold follows bare-name globals only, so a helper
reached as an attribute or a method drops out of the key and editing it would serve stale rows."""

import functools
import hashlib
import importlib.metadata
import inspect
import json
import logging
import types
from collections.abc import Callable
from typing import Any

from datasets import Dataset, DatasetDict
from transformers import ProcessorMixin

from halo.log import warn_once

logger = logging.getLogger(__name__)

# The datasets attribute holding a rank-stable cache key; only its value enters a key.
CACHE_KEY_ATTR = "_halo_cache_key"

# Derived from this package's root, so a package rename keeps invalidating on helper edits.
_FIRST_PARTY_PREFIXES = (__name__.partition(".")[0] + ".", "scripts.")

# In every cache name, since no third-party source is hashed: a transformers/TRL bump re-keys every map.
_RENDER_LIBRARY_VERSIONS = f"tf{importlib.metadata.version('transformers')}-trl{importlib.metadata.version('trl')}"

# Anything else takes the one-time skip warning.
_SCALAR_TYPES = (str, int, float, bool, type(None))

# Per-call state a fast tokenizer's serialized backend carries (any call with ``truncation=``/``padding=``
# rewrites it): hashed, it would key on the last call and split the writer rank from its peers.
_MUTABLE_BACKEND_STATE_KEYS = ("padding", "truncation")

_unfingerprintable_warned: set[tuple[str, str]] = set()

_content_sig_failed_warned: set[str] = set()


def _referenced_local_functions(func: Callable, seen: set) -> list[Callable]:
    """The first-party functions ``func``'s code reads from its module globals, which source and closure
    inspection cannot see; third-party source churns without changing our outputs, so it stays out."""
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
            and getattr(value, "__module__", "").startswith(_FIRST_PARTY_PREFIXES)
        ):
            referenced.append(value)
    return referenced


def _template_sig(chat_template: Any) -> str | None:
    if isinstance(chat_template, str):
        return hashlib.md5(chat_template.encode()).hexdigest()[:8]
    if isinstance(chat_template, dict):
        joined = "|".join(f"{name}={template}" for name, template in sorted(chat_template.items()))
        return hashlib.md5(joined.encode()).hexdigest()[:8]
    return None


def _tokenizer_content_sig(val: Any) -> str | None:
    """Content hash of what a tokenizer does to text: its fast backend's state minus the per-call keys, else its
    vocab. ``None`` when neither is readable, warned once per type when reading raised: the ``name_or_path``
    fallback then misses the cache on every resume leg and reuses it across a same-path content change."""
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
            _content_sig_failed_warned,
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
    """Signature for a bare tokenizer, or None. The template hash and special-token ids catch overrides that change
    baked ids at identical content; the source term is the content hash, not a path a resume repoints."""
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
    """Signature for a tokenizer or ``ProcessorMixin``, or None. Any other ``.tokenizer`` holder (a collator) carries
    render knobs this cannot see, so it keys by class name and its knobs belong in ``cache_key_extras``."""
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
    """Warn once: a value the key cannot see does not invalidate the cache, so editing it reuses old rows."""
    warn_once(
        logger,
        _unfingerprintable_warned,
        (owner, kind),
        "Dataset-map cache fingerprint for %s skips a value of type %s it cannot fingerprint — "
        "changes to that value will NOT invalidate the cache. If it affects the map output, thread "
        "it through cache_key_extras.",
        owner,
        kind,
    )


def _scalar_collection_repr(val: Any) -> str | None:
    """For a list, tuple or (sorted) set of scalars, or None. Dicts are left out: a captured dict is a schema carrier
    (``none_example``) and takes the skip warning."""
    if isinstance(val, (list, tuple)) and all(isinstance(item, _SCALAR_TYPES) for item in val):
        return repr(val)
    if isinstance(val, (set, frozenset)) and all(isinstance(item, _SCALAR_TYPES) for item in val):
        return repr(sorted(val, key=repr))
    return None


def kwargs_fingerprint(kwargs: dict) -> str:
    """Tokenizers by identity, scalars and JSON-able collections by content, anything else by type name."""
    if not kwargs:
        return ""
    parts = []
    for key in sorted(kwargs.keys()):
        val = kwargs[key]
        try:
            tok_sig = _tokenizer_identity(val)
            if tok_sig is not None:
                parts.append(f"{key}={tok_sig}")
                continue
            if isinstance(val, (str, int, float, bool)):
                parts.append(f"{key}={val}")
                continue
            content = _scalar_collection_repr(val)
            if content is None:
                try:
                    content = json.dumps(val, sort_keys=True)
                except (TypeError, ValueError):
                    content = None
            if content is not None:
                parts.append(f"{key}={content}")
                continue
            if getattr(val, "tokenizer", None) is not None:
                # A tokenizer holder the ProcessorMixin gate refused hides render knobs; say so.
                _warn_unfingerprintable(f"fn_kwargs[{key}]", type(val).__name__)
            parts.append(f"{key}={type(val).__name__}")
        except Exception:
            _warn_unfingerprintable(f"fn_kwargs[{key}]", type(kwargs[key]).__name__)
            parts.append(f"{key}=?")
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


def function_cache_id(func: Callable, _seen: set | None = None) -> str:
    """A hash of the source (else bytecode, else qualified name) folded with the ids of the first-party helpers it
    calls, so editing a helper re-keys its callers."""
    # partial hides its func + bound args from source inspection: unfolded, every partial shares one id.
    if isinstance(func, functools.partial):
        base = function_cache_id(func.func, _seen)
        bound = kwargs_fingerprint({f"_arg{i}": a for i, a in enumerate(func.args)} | dict(func.keywords or {}))
        return f"partial[{base}]_{bound}"

    func_name = getattr(func, "__name__", "unknown")
    seen = _seen if _seen is not None else set()
    seen.add(func)

    try:
        source = inspect.getsource(func)
        helper_ids = [function_cache_id(helper, seen) for helper in _referenced_local_functions(func, seen)]
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

    # Keyed on where the callable lives, so a module move re-keys it.
    module = getattr(func, "__module__", "")
    qualname = getattr(func, "__qualname__", func_name)
    return f"{module}.{qualname}".replace("<", "").replace(">", "")


def _closure_fingerprint(func: Callable) -> str:
    """The output-affecting values ``func`` captures, which neither source nor arguments show; others warn."""
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
            parts.append(function_cache_id(val))
            continue
        _warn_unfingerprintable(getattr(func, "__qualname__", repr(func)), type(val).__name__)
    if not parts:
        return ""
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


def stamp_cache_key(dataset: Dataset, key: str) -> None:
    setattr(dataset, CACHE_KEY_ATTR, key)


def rank_stable_dataset_id(dataset: Dataset) -> str:
    """A rank- and run-independent identifier, where HF's ``_fingerprint`` and ``cache_files`` diverge across ranks:
    the stamp, else cache file paths, else in-memory ``_fingerprint``."""
    stamp = getattr(dataset, CACHE_KEY_ATTR, None)
    if stamp:
        return stamp
    cache_files = getattr(dataset, "cache_files", None) or []
    paths = sorted(cf["filename"] for cf in cache_files if isinstance(cf, dict) and cf.get("filename"))
    if paths:
        return hashlib.md5("|".join(paths).encode()).hexdigest()[:16]
    return getattr(dataset, "_fingerprint", None) or "nofp"


def carry_cache_key(source: Dataset, derived: Dataset, step: str) -> Dataset:
    """Stamp ``derived`` with ``source``'s key extended by ``step``: ``datasets`` copies the stamp unchanged, which
    would key the derived rows like their source. An unstamped ``source`` leaves ``derived`` as it is."""
    stamp = getattr(source, CACHE_KEY_ATTR, None)
    if stamp:
        stamp_cache_key(derived, f"{stamp}|{step}")
    return derived


def dataset_content_fingerprint(dataset: Dataset | DatasetDict) -> str:
    if isinstance(dataset, DatasetDict):
        parts = [
            f"{name}:{len(dataset[name])}:{rank_stable_dataset_id(dataset[name])}" for name in sorted(dataset.keys())
        ]
        return "|".join(parts)
    return f"{len(dataset)}:{rank_stable_dataset_id(dataset)}"


def map_cache_file_name(
    operation: str,
    func: Callable,
    dataset: Dataset | DatasetDict,
    desc: str | None,
    kwargs: dict,
    cache_key_extras: dict | None = None,
) -> str:
    """Function identity, closure, every kwarg, ``cache_key_extras``, the input and the library versions."""
    func_id = function_cache_id(func)
    closure_fp = _closure_fingerprint(func)
    fn_kwargs_fp = kwargs_fingerprint(kwargs.get("fn_kwargs", {}))
    op_kwargs = {k: v for k, v in kwargs.items() if k != "fn_kwargs"}
    op_kwargs_fp = kwargs_fingerprint(op_kwargs)
    extras_fp = kwargs_fingerprint(cache_key_extras or {})
    ds_fingerprint = dataset_content_fingerprint(dataset)
    cache_key = (
        f"{operation}_{desc or 'default'}_{func_id}_{closure_fp}_{fn_kwargs_fp}_{op_kwargs_fp}"
        f"_{extras_fp}_{ds_fingerprint}_{_RENDER_LIBRARY_VERSIONS}"
    )
    return f"cache-{hashlib.md5(cache_key.encode()).hexdigest()}.arrow"


def transform_cache_name(operation_name: str, dataset: Dataset, cache_key_extras: dict, run_token: str) -> str:
    """Keyed like :func:`map_cache_file_name`. An input without the toolkit's stamp keys on ``run_token`` too: its file
    paths name neither content nor an in-memory selection, so no later run may trust a match."""
    scope = "" if getattr(dataset, CACHE_KEY_ATTR, None) else f"_run{run_token}"
    key = (
        f"{operation_name}_{dataset_content_fingerprint(dataset)}_{kwargs_fingerprint(cache_key_extras)}"
        f"_{_RENDER_LIBRARY_VERSIONS}{scope}"
    )
    return f"transform-{hashlib.md5(key.encode()).hexdigest()}"


def packed_cache_file_name(dataset: Dataset, strategy: str, seq_length: int, split: str) -> str:
    """Keyed on packed columns: a pack storing another set (an ``attention_mask``) never serves these rows."""
    columns = hashlib.md5("|".join(sorted(dataset.column_names)).encode()).hexdigest()[:8]
    return (
        f"packed_{strategy}_{seq_length}_{rank_stable_dataset_id(dataset)}_{columns}_{split}"
        f"_{_RENDER_LIBRARY_VERSIONS}.arrow"
    )
