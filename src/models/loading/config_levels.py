"""Composite-config level access: read and write a field on every level that declares it.

``PreTrainedConfig`` has no ``__getattr__``, so on a composite (VLM/multimodal) config a decoder
field lives only on ``get_text_config()`` while others live only on the wrapper, and a one-level read
returns the default instead. This module provides the level enumeration, the reads and writes built
on it (plain and run-scoped), and the special-token-id snapshot/restore, so every consumer resolves
a family's levels the same way.
"""

import contextlib
import weakref
from typing import Any

_SPECIAL_TOKEN_ID_FIELDS = ("eos_token_id", "bos_token_id", "pad_token_id")

# Sentinel for a level that did not declare the field at snapshot time: composite wrappers carry no
# top-level ids, align_special_tokens plants one there, and the restore has to remove it.
_SPECIAL_TOKEN_ID_ABSENT = object()

# Pre-mutation values of run-scoped config writes, kept outside the config object so nothing carries
# them into ``to_dict()``/``config.json``. Keyed by ``id``: transformers 5 configs define ``__eq__``
# without ``__hash__``, which a WeakKeyDictionary rejects. ``_MISSING`` means the field was absent
# from the instance ``__dict__``, so serialization fell through to the class default.
_RUN_SCOPED_FIELD_ORIGINALS: dict[int, dict[str, Any]] = {}
# Config objects that take no weak reference (a ``__slots__`` stand-in), pinned so their ``id``
# cannot be recycled onto another config that would inherit their originals, which is the aliasing
# the finalizer prevents for everything else.
_RUN_SCOPED_UNFINALIZABLE: list[Any] = []
_MISSING = object()


def text_config(cfg):
    """The sub-config carrying the text-model fields; ``cfg`` itself for a plain config.

    A composite (multimodal) config keeps ``model_type``, the head counts and the expert/router
    fields on its text sub-config, and ``PreTrainedConfig`` defines no ``__getattr__``, so a
    wrapper-only read returns the default instead of raising.
    """
    getter = getattr(cfg, "get_text_config", None)
    return (getter() if callable(getter) else None) or cfg


def config_sources(cfg) -> tuple:
    """``cfg`` plus its text sub-config, deduplicated: every level a field read or write must cover,
    for the reason :func:`text_config` states. ``get_text_config()`` returns ``self`` for a plain
    config, which therefore yields one level."""
    text_cfg = text_config(cfg)
    return (cfg,) if text_cfg is cfg else (cfg, text_cfg)


def get_config_field(cfg, field: str, default=None, *, per_layer_reduce=None):
    """First non-``None`` ``field`` across :func:`config_sources`, else ``default``.

    The modeling code reads these off the text config, so a wrapper-only read yields the default and
    turns a balancing decision into a no-op.
    """
    for source in config_sources(cfg):
        try:
            value = getattr(source, field, None)
        except RuntimeError as e:
            # transformers' AmbiguousGlobalPerLayerAttributeError (a RuntimeError, so the getattr
            # default does not swallow it): the field is registered per-layer (Gemma 4's head_dim),
            # and the declared value must still win where the per-layer configs agree. Matched by
            # name to avoid importing transformers internals here.
            if type(e).__name__ != "AmbiguousGlobalPerLayerAttributeError":
                raise
            per_layer = getattr(source, "per_layer_config", None) or ()
            distinct = {getattr(layer_cfg, field, None) for layer_cfg in per_layer} - {None}
            if len(distinct) > 1:
                if per_layer_reduce is None:
                    raise ValueError(
                        f"config field {field!r} varies across layers ({sorted(distinct)}); a "
                        f"single global value does not exist for this model."
                    ) from e
                value = per_layer_reduce(distinct)
            else:
                value = distinct.pop() if distinct else None
        if value is not None:
            return value
    return default


def configs_declaring(config, field: str) -> list:
    """Every config level of ``config`` that declares ``field``: the wrapper and/or its text sub-config.

    Both levels, because different consumers read different ones: pooling keys on
    ``config.get_text_config().pad_token_id`` while the head is sized from the wrapper's
    ``num_labels``. Empty means neither level declares the field.
    """
    return [level for level in config_sources(config) if hasattr(level, field)]


def set_config_field(config, field: str, value, *, only_declared: bool) -> bool:
    """Write ``value`` to config levels of ``config``; ``False`` when the write reached none.

    ``only_declared=True`` writes just the levels that already declare ``field``, so a value nothing
    reads is never planted, and the return says whether any level declared it. ``False`` writes every
    level (always at least one, hence always ``True``), which is what a toolkit-driven flag needs,
    since HF forwards read the text config while call sites hold whichever object ``model.config``
    returned.
    """
    targets = configs_declaring(config, field) if only_declared else list(config_sources(config))
    for target in targets:
        setattr(target, field, value)
    return bool(targets)


def _config_storage_field(source, field: str) -> str:
    """The instance-``__dict__`` key a ``setattr(source, field, ...)`` lands in.

    ``PreTrainedConfig.__setattr__`` remaps aliased fields through ``attribute_map``, so the
    run-scoped registry records and restores the mapped key; otherwise the restore would target a
    slot the write never touched and leave the run-scoped value in ``config.json``.
    """
    return (getattr(source, "attribute_map", None) or {}).get(field, field)


def _run_scoped_originals(source) -> dict[str, Any]:
    key = id(source)
    originals = _RUN_SCOPED_FIELD_ORIGINALS.get(key)
    if originals is None:
        originals = _RUN_SCOPED_FIELD_ORIGINALS[key] = {}
        # The finalizer drops the entry so a recycled id() cannot inherit another config's originals.
        try:
            weakref.finalize(source, _RUN_SCOPED_FIELD_ORIGINALS.pop, key, None)
        except TypeError:
            _RUN_SCOPED_UNFINALIZABLE.append(source)
    return originals


def set_config_field_run_scoped(cfg, field: str, value) -> None:
    """An ``only_declared=False`` write for run-scoped state: mutations that configure this training
    run (zeroed ``router_aux_loss_coef``, forced ``output_router_logits``, toolkit stamps) but do not
    describe the artifact. The pre-mutation value is recorded once per config source, and
    :func:`config_export_ready` restores it around every config serialization; without that, the
    exported ``config.json`` overwrites the hub's aux coefficient and makes plain-transformers
    consumers pay the router-logit plane on every forward.
    """
    for source in config_sources(cfg):
        storage = _config_storage_field(source, field)
        _run_scoped_originals(source).setdefault(storage, source.__dict__.get(storage, _MISSING))
    set_config_field(cfg, field, value, only_declared=False)


@contextlib.contextmanager
def config_export_ready(cfg):
    """Temporarily restore every run-scoped field mutation on ``cfg`` (and its text config) so a
    serialization inside the block writes the artifact's fields; the run state is re-applied on exit.
    Instance-``__dict__`` based on both sides, mirroring how the run-scoped setter wrote."""
    if cfg is None:
        yield
        return
    reapply: list[tuple[Any, str, Any]] = []
    for source in config_sources(cfg):
        for field, original in (_RUN_SCOPED_FIELD_ORIGINALS.get(id(source)) or {}).items():
            reapply.append((source, field, source.__dict__.get(field, _MISSING)))
            if original is _MISSING:
                source.__dict__.pop(field, None)
            else:
                setattr(source, field, original)
    try:
        yield
    finally:
        for source, field, value in reversed(reapply):
            if value is _MISSING:
                source.__dict__.pop(field, None)
            else:
                setattr(source, field, value)


def snapshot_special_token_ids(config) -> list[tuple[object, str, object]]:
    """Capture every config level's special-token ids, list-valued ones copied.

    HF's ``align_special_tokens`` (run by ``Trainer.train``) overwrites ``config.eos_token_id`` with
    the tokenizer's single id, collapsing a list-valued stop set (Gemma 4's ``[1, 106]``, GLM's three
    role enders) in the config every export ships, so serving never stops on turn ends and a stage-2
    run masking against the exported config trains zero tokens.
    """
    snapshot = []
    for level in config_sources(config):
        for field in _SPECIAL_TOKEN_ID_FIELDS:
            if hasattr(level, field):
                value = getattr(level, field)
                snapshot.append((level, field, list(value) if isinstance(value, list) else value))
            else:
                snapshot.append((level, field, _SPECIAL_TOKEN_ID_ABSENT))
    return snapshot


def restore_special_token_ids(snapshot: list[tuple[object, str, object]]) -> None:
    """Re-assert the pristine special-token ids captured by :func:`snapshot_special_token_ids`.

    Run before every config write: the generation config keeps HF's unioned EOS set (that part of
    the alignment is correct and preserved), only the model config's collapse is undone.
    """
    for level, field, value in snapshot:
        if value is _SPECIAL_TOKEN_ID_ABSENT:
            if field in getattr(level, "__dict__", {}):
                delattr(level, field)
        else:
            setattr(level, field, list(value) if isinstance(value, list) else value)
