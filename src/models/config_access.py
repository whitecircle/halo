"""Composite-config level access: read and write a field on every level that declares it.

``PreTrainedConfig`` has no ``__getattr__``, so on a composite (VLM) config a decoder field lives only on
``get_text_config()`` while others live only on the wrapper, and a one-level read returns the default instead."""

import contextlib
import weakref
from typing import Any

from accelerate.logging import get_logger
from transformers.integrations.heterogeneity import AmbiguousGlobalPerLayerAttributeError

logger = get_logger(__name__)

# The config field every family spells its input/output embedding tie with.
TIE_WORD_EMBEDDINGS_FLAG = "tie_word_embeddings"
# HF's align_special_tokens collapses a list-valued stop set (Gemma 4's ``[1, 106]``) to the tokenizer's single id
# in the config every export ships, and TRL's trainers write the tokenizer's pad id there at construction.
SPECIAL_TOKEN_ID_FIELDS = ("eos_token_id", "bos_token_id", "pad_token_id")
# TRL's SFTTrainer writes these onto the text config at construction, over what the balancing mode set for the
# run and what an export carries.
ROUTER_AUX_FIELDS = ("output_router_logits", "router_aux_loss_coef")

# A field the level did not hold when captured, which the restore removes rather than writes back (a special-token
# id align_special_tokens planted, a run-scoped field the instance ``__dict__`` lacked).
_ABSENT = object()

# Pre-mutation values of run-scoped config writes, kept outside the config object so nothing carries
# them into ``to_dict()``/``config.json``. Keyed by ``id``: transformers 5 configs define ``__eq__``
# without ``__hash__``, which a WeakKeyDictionary rejects.
_run_scoped_field_originals: dict[int, dict[str, Any]] = {}
# Config objects that take no weak reference (a ``__slots__`` stand-in), pinned so their ``id``
# cannot be recycled onto another config that would inherit their originals, which is the aliasing
# the finalizer prevents for everything else.
_run_scoped_unfinalizable: list[Any] = []


def text_config(cfg):
    """The sub-config carrying the text-model fields (``model_type``, head counts, expert/router
    fields); ``cfg`` itself for a plain config."""
    getter = getattr(cfg, "get_text_config", None)
    return (getter() if callable(getter) else None) or cfg


def config_sources(cfg) -> tuple:
    """``cfg`` plus its text sub-config, deduplicated: every level a field read or write must cover."""
    text_cfg = text_config(cfg)
    return (cfg,) if text_cfg is cfg else (cfg, text_cfg)


def config_model_types(cfg) -> tuple[str, ...]:
    """Every ``model_type`` spelling across :func:`config_sources`, wrapper first, deduplicated; ``()`` for no config."""
    if cfg is None:
        return ()
    return tuple(dict.fromkeys(mt for level in config_sources(cfg) if (mt := getattr(level, "model_type", None))))


def model_type_matches(cfg, *prefixes: str) -> bool:
    """Whether the ``model_type`` of any :func:`config_sources` level starts with one of ``prefixes``."""
    return any(model_type.startswith(prefixes) for model_type in config_model_types(cfg))


def tie_flag_configs(config, prefix: str = "") -> dict[str, object]:
    """Dotted sub-config path → the config object there that declares the embedding tie.

    A composite (VLM) config declares it on its text sub-config as well as, or instead of, the top
    level, and transformers resolves the tie across every submodel, so the top level alone is not the
    whole declaration. Keyed by path rather than position so the values can be put back on a
    different config tree (``from_pretrained`` hands the model a deep copy).
    """
    holders = {prefix: config} if hasattr(config, TIE_WORD_EMBEDDINGS_FLAG) else {}
    for name in getattr(config, "sub_configs", None) or ():
        sub = getattr(config, name, None)
        if sub is not None:
            holders.update(tie_flag_configs(sub, f"{prefix}{name}."))
    return holders


def ties_word_embeddings(config) -> bool:
    """Whether any level of ``config`` declaring the flag ties the embedding/head pair."""
    return any(bool(getattr(level, TIE_WORD_EMBEDDINGS_FLAG)) for level in tie_flag_configs(config).values())


def untie_word_embeddings(config) -> None:
    """Clear the embedding tie on every level of ``config`` that declares it."""
    for level in tie_flag_configs(config).values():
        setattr(level, TIE_WORD_EMBEDDINGS_FLAG, False)


def get_config_field(cfg, field: str, default=None, *, per_layer_reduce=None):
    """First non-``None`` ``field`` across :func:`config_sources`, else ``default``."""
    for source in config_sources(cfg):
        try:
            value = getattr(source, field, None)
        except AmbiguousGlobalPerLayerAttributeError as e:
            # A RuntimeError, so the getattr default does not swallow it: the field is registered
            # per-layer (Gemma 4's head_dim), and the declared value must still win where the
            # per-layer configs agree.
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
    """Every config level of ``config`` that declares ``field``: the wrapper and/or its text sub-config
    (pooling reads the text level's ``pad_token_id``, the head is sized from the wrapper's ``num_labels``)."""
    return [level for level in config_sources(config) if hasattr(level, field)]


def set_config_field(config, field: str, value, *, only_declared: bool) -> bool:
    """Write ``value`` to config levels of ``config``; ``False`` when the write reached none.

    ``only_declared=True`` writes just the levels already declaring ``field``, so a value nothing reads
    is never planted. ``False`` writes every level, what a toolkit-driven flag needs, since HF forwards
    read the text config while call sites hold whichever object ``model.config`` returned.
    """
    targets = configs_declaring(config, field) if only_declared else list(config_sources(config))
    for target in targets:
        setattr(target, field, value)
    return bool(targets)


def set_config_field_run_scoped(cfg, field: str, value) -> None:
    """An ``only_declared=False`` write for state that configures this run but does not describe the
    artifact (zeroed ``router_aux_loss_coef``, forced ``output_router_logits``, toolkit stamps). The
    pre-mutation value is recorded once per level, for :func:`config_export_ready` to restore.
    """
    for source in config_sources(cfg):
        originals = _run_scoped_field_originals.get(id(source))
        if originals is None:
            originals = _run_scoped_field_originals[id(source)] = {}
            # The finalizer drops the entry so a recycled id() cannot inherit another config's originals.
            try:
                weakref.finalize(source, _run_scoped_field_originals.pop, id(source), None)
            except TypeError:
                _run_scoped_unfinalizable.append(source)
        # ``PreTrainedConfig.__setattr__`` remaps aliased fields through ``attribute_map``; a restore of
        # the unmapped key would leave the run-scoped value in ``config.json``.
        storage = (getattr(source, "attribute_map", None) or {}).get(field, field)
        originals.setdefault(storage, source.__dict__.get(storage, _ABSENT))
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
        for field, original in (_run_scoped_field_originals.get(id(source)) or {}).items():
            reapply.append((source, field, source.__dict__.get(field, _ABSENT)))
            if original is _ABSENT:
                source.__dict__.pop(field, None)
            else:
                setattr(source, field, original)
    try:
        yield
    finally:
        for source, field, value in reversed(reapply):
            if value is _ABSENT:
                source.__dict__.pop(field, None)
            else:
                setattr(source, field, value)


def snapshot_config_fields(config, fields: tuple[str, ...]) -> list[tuple[object, str, object]]:
    """``fields`` on every config level as ``(level, field, value)``, list values copied, absence recorded,
    for :func:`restore_config_fields`."""
    snapshot = []
    for level in config_sources(config):
        for field in fields:
            if hasattr(level, field):
                value = getattr(level, field)
                snapshot.append((level, field, list(value) if isinstance(value, list) else value))
            else:
                snapshot.append((level, field, _ABSENT))
    return snapshot


def restore_config_fields(snapshot: list[tuple[object, str, object]]) -> None:
    """Re-assert the config fields a snapshot captured, removing those a level did not hold. The special-token
    ids are restored before every config write; the generation config keeps HF's unioned EOS set."""
    for level, field, value in snapshot:
        if value is _ABSENT:
            if field in getattr(level, "__dict__", {}):
                delattr(level, field)
        else:
            setattr(level, field, list(value) if isinstance(value, list) else value)


def apply_config_overrides(model_config, overrides: dict | None) -> None:
    """Write ``model_config_overrides`` to every config level that declares each key; an unknown key raises,
    since a skipped override would train with the stock config.

    On a composite config the decoder knobs a run overrides (``output_router_logits``,
    ``router_aux_loss_coef``) exist only on ``text_config``. The run's precision is refused rather than
    written: every loader passes ``dtype=`` explicitly, which wins over the config.
    """
    if not overrides:
        return
    # Before any write, so a rejected key cannot leave the config half-overridden. Both spellings:
    # transformers 5 keeps ``torch_dtype`` as an alias of ``dtype``, so a config declares both.
    rejected = sorted(key for key in overrides if key in ("dtype", "torch_dtype"))
    if rejected:
        raise ValueError(
            f"model_config_overrides may not set {rejected}. Every loader passes the run's dtype "
            f"explicitly, resolved from the run's own precision flags, and that argument wins over "
            f"the config — so the override would be silently discarded and the run would train in "
            f"the precision it did not ask for. Set the run's dtype with `bf16: true` or "
            f"`fp16: true` in the training config (bf16 is the toolkit default), and fp32 master "
            f"weights with fp32_non_ep_params / fp32_experts."
        )
    logger.info(f"Applying model config overrides: {overrides}")
    for key, value in overrides.items():
        targets = configs_declaring(model_config, key)
        if not targets:
            raise ValueError(
                f"model_config_overrides key '{key}' does not exist on this model's config "
                f"({type(model_config).__name__}) or its text sub-config — a skipped override "
                f"would silently train with the stock value. Fix or remove the key."
            )
        for target in targets:
            setattr(target, key, value)
        logger.info(f"  Set {key} = {value} on {', '.join(type(t).__name__ for t in targets)}")
