"""Composite-config level access. ``PreTrainedConfig`` has no ``__getattr__``, so on a composite (VLM) config a decoder
field lives only on ``get_text_config()`` and a one-level read returns the default instead."""

import contextlib
import logging
import weakref
from typing import Any

from transformers.integrations.heterogeneity import AmbiguousGlobalPerLayerAttributeError

from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

TIE_WORD_EMBEDDINGS_FLAG = "tie_word_embeddings"
# HF's align_special_tokens collapses a list-valued stop set (Gemma 4's ``[1, 106]``) to the tokenizer's single id
# in the config every export ships, and TRL's trainers write the tokenizer's pad id there at construction.
SPECIAL_TOKEN_ID_FIELDS = ("eos_token_id", "bos_token_id", "pad_token_id")
# TRL's SFTTrainer writes these onto the text config at construction, over what the balancing mode set for the
# run and what an export carries.
ROUTER_AUX_FIELDS = ("output_router_logits", "router_aux_loss_coef")

# Config attributes recording this run's own decisions, which no artifact describes. ``from_pretrained`` deep-copies
# the config it is handed, so a stamp written before the load reaches the model with no recorded original to restore.
TOOLKIT_STAMP_PREFIX = "_halo_"

# A field the level did not hold when captured, which the restore removes rather than writes back.
_ABSENT = object()

# Kept outside the config so nothing carries them into ``config.json``. Keyed by ``id``, as transformers 5 configs
# define ``__eq__`` without ``__hash__``, which a WeakKeyDictionary rejects.
_run_scoped_field_originals: dict[int, dict[str, Any]] = {}
# Configs taking no weak reference, pinned so their ``id`` cannot be recycled onto another config.
_run_scoped_unfinalizable: list[Any] = []


def text_config(cfg):
    getter = getattr(cfg, "get_text_config", None)
    return (getter() if callable(getter) else None) or cfg


def config_sources(cfg) -> tuple:
    text_cfg = text_config(cfg)
    return (cfg,) if text_cfg is cfg else (cfg, text_cfg)


def config_model_types(cfg) -> tuple[str, ...]:
    if cfg is None:
        return ()
    return tuple(dict.fromkeys(mt for level in config_sources(cfg) if (mt := getattr(level, "model_type", None))))


def model_type_matches(cfg, *prefixes: str) -> bool:
    return any(model_type.startswith(prefixes) for model_type in config_model_types(cfg))


def _config_levels(config, prefix: str = "") -> dict[str, object]:
    """Dotted sub-config path → each level of ``config``'s tree, the root first."""
    levels = {prefix: config}
    for name in getattr(config, "sub_configs", None) or ():
        sub = getattr(config, name, None)
        if sub is not None:
            levels.update(_config_levels(sub, f"{prefix}{name}."))
    return levels


def built_level_path(config, built) -> str:
    """The level of ``config``'s tree a model carrying ``built`` took: a sub-config's path where its class takes only
    that level (``AutoModelForCausalLM`` hands a composite's text class the text config), else the root's. By class, as
    ``from_pretrained`` hands the model a deep copy."""
    return next((path for path, level in _config_levels(config).items() if type(level) is type(built)), "")


def tie_flag_configs(config) -> dict[str, object]:
    """Dotted sub-config path → the config declaring the embedding tie. Keyed by path so the values can be put back on
    a different config tree (``from_pretrained`` hands the model a deep copy)."""
    return {path: level for path, level in _config_levels(config).items() if hasattr(level, TIE_WORD_EMBEDDINGS_FLAG)}


def ties_word_embeddings(config) -> bool:
    return any(bool(getattr(level, TIE_WORD_EMBEDDINGS_FLAG)) for level in tie_flag_configs(config).values())


def narrow_to_text_config(config):
    """``config``'s text level for a class built from it alone. A composite states the tie at either level and a text
    class reads only its own, so a tie stated only on the root comes along: the checkpoint was saved tied either way. A
    sibling tower's flag describes that tower, and an older ``config.json`` can spell it out as a default."""
    text = text_config(config)
    if text is not config and getattr(config, TIE_WORD_EMBEDDINGS_FLAG, False):
        setattr(text, TIE_WORD_EMBEDDINGS_FLAG, True)
    return text


def untie_word_embeddings(config) -> None:
    for level in tie_flag_configs(config).values():
        setattr(level, TIE_WORD_EMBEDDINGS_FLAG, False)


def get_config_field(cfg, field: str, default=None, *, per_layer_reduce=None):
    """First non-``None`` ``field`` across :func:`config_sources`, else ``default``."""
    for source in config_sources(cfg):
        try:
            value = getattr(source, field, None)
        except AmbiguousGlobalPerLayerAttributeError as e:
            # Per-layer fields (Gemma 4's head_dim) raise a RuntimeError past the getattr default, and the declared
            # value must still win where the per-layer configs agree.
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
    return [level for level in config_sources(config) if hasattr(level, field)]


def set_config_field(config, field: str, value, *, only_declared: bool) -> bool:
    """``only_declared=True`` never plants a value nothing reads. ``False`` writes every level, as HF forwards read the
    text config while call sites hold whichever object ``model.config`` returned."""
    targets = configs_declaring(config, field) if only_declared else list(config_sources(config))
    for target in targets:
        setattr(target, field, value)
    return bool(targets)


def set_config_field_run_scoped(cfg, field: str, value) -> None:
    """For state that configures this run but does not describe the artifact. The pre-mutation value is recorded once
    per level, for :func:`config_export_ready` to restore."""
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


def _toolkit_stamps(level) -> list[str]:
    return [key for key in getattr(level, "__dict__", {}) if key.startswith(TOOLKIT_STAMP_PREFIX)]


def drop_toolkit_stamps(cfg) -> None:
    """A ``config.json`` serialized outside :func:`config_export_ready` can hold another run's stamps, which
    transformers reads back as live attributes."""
    for source in config_sources(cfg):
        dropped = _toolkit_stamps(source)
        for key in dropped:
            del source.__dict__[key]
        if dropped and is_global_main_process():
            logger.info(f"Ignoring the toolkit stamps {dropped} the checkpoint's {type(source).__name__} carries.")


@contextlib.contextmanager
def config_export_ready(cfg):
    """Restore every run-scoped field and lift out toolkit stamps, so a serialization inside the block writes the
    artifact's fields; the run state is re-applied on exit."""
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
        for field in _toolkit_stamps(source):
            reapply.append((source, field, source.__dict__.pop(field)))
    try:
        yield
    finally:
        for source, field, value in reversed(reapply):
            if value is _ABSENT:
                source.__dict__.pop(field, None)
            else:
                setattr(source, field, value)


def snapshot_config_fields(config, fields: tuple[str, ...]) -> list[tuple[object, str, object]]:
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
    """The special-token ids are restored before every config write; the generation config keeps HF's EOS union."""
    for level, field, value in snapshot:
        if value is _ABSENT:
            if field in getattr(level, "__dict__", {}):
                delattr(level, field)
        else:
            setattr(level, field, list(value) if isinstance(value, list) else value)


def apply_config_overrides(model_config, overrides: dict | None) -> None:
    """An unknown key raises, as a skipped override would train with the stock config. The run's precision is refused,
    as every loader passes ``dtype=`` explicitly, which wins over the config."""
    if not overrides:
        return
    # Before any write, so a rejected key cannot leave the config half-overridden.
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
    if is_global_main_process():
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
        if is_global_main_process():
            logger.info(f"  Set {key} = {value} on {', '.join(type(t).__name__ for t in targets)}")
