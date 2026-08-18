"""Recompute HF model buffers the meta-device build leaves unset.

Fixes buffers that ``device_map="meta"`` / ``init_empty_weights()`` leave uninitialized or stranded on
meta. Architecture-specific (RoPE, Bailing Lightning-Attention slopes, Gemma4 ``embed_scale``), not EP-specific.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import math
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from src.distributed.runtime import current_device
from src.log import warn_once
from src.models.attention_geometry import resolve_head_dim
from src.models.loading.config_levels import text_config

logger = logging.getLogger(__name__)

# ``warn_once`` scope for the unrecognized-rotary warnings, keyed by (reason, class name): an
# unhandled layout is a property of the FAMILY, and this logger reaches every rank on a model
# carrying a rotary module in every layer.
_WARNED_ROTARY: set = set()

# A fixer returns how many buffers it rebuilt, or ``None`` to hand the module to the next fixer.
_Fixer = Callable[[nn.Module, str], int | None]


def default_rope_parameters(config, device=None, **kwargs) -> tuple[torch.Tensor, float]:
    """``(inv_freq, attention_scaling)`` for plain unscaled RoPE, read ``rope_parameters``-first.

    transformers 5 moved ``rope_theta`` / ``partial_rotary_factor`` into the ``rope_parameters``
    dict; the flat attributes survive only on remote-code configs written for transformers 4.
    Reading the dict first and the attribute second is what lets one formula serve both, which is why
    the remote-code shims register this back as the registry's ``"default"`` entry. ``kwargs``
    absorbs the ``layer_type`` the registry passes to per-layer-type rotaries.
    """
    # A composite (multimodal) config keeps rope_parameters and the head fields on its text
    # sub-config; a leaf (incl. vision) config resolves to itself.
    config = text_config(config)
    rope_parameters = getattr(config, "rope_parameters", None) or {}

    def _param(name: str, fallback: float) -> float:
        value = rope_parameters.get(name, getattr(config, name, None))
        return float(fallback if value is None else value)

    head_dim = resolve_head_dim(config)
    dim = int(head_dim * _param("partial_rotary_factor", 1.0))
    base = _param("rope_theta", 10000.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, 1.0


def _register_recomputed_buffer(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    """Re-register a recomputed buffer PRESERVING its declared persistence.

    A remote-code family may declare ``inv_freq`` persistent (shipped in its checkpoints) where
    native rotaries do not, and forcing ``persistent=False`` would silently drop the key from every
    save. An unregistered buffer keeps the native non-persistent default.
    """
    persistent = name in module._buffers and name not in module._non_persistent_buffers_set
    module.register_buffer(name, tensor, persistent=persistent)


def _is_rotary_module(module: nn.Module) -> bool:
    """Whether ``module`` is a rotary-embedding layer, by class identity.

    They share no common base, so the ``*RotaryEmbedding`` class name is the remaining declaration
    (the rule :func:`~src.models.structure.is_normalization_module` applies to norms). Used only to
    decide whether an unrecognized layout deserves a warning, never to select a fix.
    """
    return "rotary" in type(module).__name__.lower()


def _materialized_device(tensor: torch.Tensor | None) -> torch.device:
    """``tensor``'s device, or this rank's when it has none or still sits on meta.

    This pass runs on models whose weights may still be meta (the lazy loaders), and a buffer built
    there turns the first forward's activations into meta tensors rather than raising.
    """
    if tensor is None or tensor.device.type == "meta":
        return current_device()
    return tensor.device


def _apply_recomputed_rope(
    module: nn.Module, prefix: str, inv_freq: torch.Tensor, scaling: float, *, mirror_twin: bool, create_scaling: bool
) -> None:
    """Install a recomputed rotary table in fp32: ``{prefix}inv_freq``, its twin, the scaling.

    The twin goes in by plain assignment, which torch routes through ``register_buffer`` with that
    buffer's own persistence — while ``register_buffer`` itself REJECTS the twin a transformers-4
    remote-code rotary keeps as a plain attribute. ``mirror_twin`` aliases the live table as those
    rotaries do; ``create_scaling`` installs one the module never declared (per-layer-type only).
    """
    name = f"{prefix}inv_freq"
    inv_freq = inv_freq.float().to(_materialized_device(getattr(module, name, None)))
    _register_recomputed_buffer(module, name, inv_freq)
    twin = f"{prefix}original_inv_freq"
    if mirror_twin:
        setattr(module, twin, inv_freq)
    elif hasattr(module, twin):
        setattr(module, twin, inv_freq.clone())
    scaling_name = f"{prefix}attention_scaling"
    if create_scaling or hasattr(module, scaling_name):
        setattr(module, scaling_name, scaling)


def _fix_layer_type_inv_freq(module: nn.Module, module_name: str, layer_type: str, rope_init_fn) -> bool:
    """Recompute one per-layer-type rotary's ``<layer_type>_inv_freq`` triple. Returns whether it did.

    A declared layer type with no buffer beside it is one this instance does not rotate (a null
    ``rope_parameters`` entry — Laguna, Gemma 4): its own init fn would raise on that entry, and
    building a table would invent state.
    """
    buf_name = f"{layer_type}_inv_freq"
    if getattr(module, buf_name, None) is None:
        return False
    # Size the tables off the layer-TYPE-resolved config view, as the module's own __init__ does:
    # the raw config raises AmbiguousGlobalPerLayerAttributeError on Gemma4's sliding leg and sizes
    # the global one off the wrong head_dim. A config without per-layer state rejects the view.
    config = module.config
    with contextlib.suppress(AttributeError, ValueError):
        config = config.per_layer_config[layer_type]
    inv_freq, scaling = rope_init_fn(config, layer_type=layer_type)
    _apply_recomputed_rope(module, f"{layer_type}_", inv_freq, scaling, mirror_twin=False, create_scaling=True)
    # DEBUG and stat-free: one line per rotary module per rank, and reading inv_freq.min()/.max()
    # would force a device sync per layer on a path that runs for every rank at load.
    logger.debug(f"Fixed {buf_name} for {module_name} (rope_type={module.rope_type.get(layer_type)})")
    return True


def _resolve_rope_init_fn(module: nn.Module, layer_type: str) -> Callable:
    """One layer type's init fn, from ``ROPE_INIT_FUNCTIONS`` unless the type asks for plain RoPE."""
    rope_type = module.rope_type.get(layer_type, "default")
    return module.compute_default_rope_parameters if rope_type == "default" else ROPE_INIT_FUNCTIONS[rope_type]


def _fix_gemma4_vision_rope(module: nn.Module, name: str) -> int | None:
    """Gemma4's vision rotary, whose ``spatial_dim = head_dim // 2`` the generic formula doubles."""
    if type(module).__name__ != "Gemma4VisionRotaryEmbedding":
        return None
    config = getattr(module, "config", None)
    if config is None:
        return 0
    inv_freq, scaling = module.compute_default_rope_parameters(config)
    _apply_recomputed_rope(module, "", inv_freq, scaling, mirror_twin=False, create_scaling=False)
    logger.debug(f"Fixed vision inv_freq for {name} (spatial_dim formula)")
    return 1


def _fix_per_layer_type_rope(module: nn.Module, name: str) -> int | None:
    """Every rotated layer type of a rotary that keys inv_freq BY layer type — claimed off the
    module's own ``rope_init_fns`` mapping or its ``layer_types`` + dict ``rope_type``, never by
    class name (Gemma4 and DeepSeek-V4 reached this shape independently, and DeepSeek-V4 repeats
    it in every compressor/indexer)."""
    if hasattr(module, "rope_init_fns"):
        init_fns = module.rope_init_fns.items()
    elif hasattr(module, "layer_types") and isinstance(getattr(module, "rope_type", None), dict):
        init_fns = ((layer_type, _resolve_rope_init_fn(module, layer_type)) for layer_type in module.layer_types)
    else:
        return None
    if getattr(module, "config", None) is None:
        return 0
    return sum(_fix_layer_type_inv_freq(module, name, layer_type, fn) for layer_type, fn in init_fns)


def _fix_single_inv_freq_rope(module: nn.Module, name: str) -> int | None:
    """The ordinary single-``inv_freq`` rotary — every layout the fixers above do not claim."""
    if getattr(module, "inv_freq", None) is None:
        # A rotary carrying neither a plain ``inv_freq`` nor a recognized per-layer-type mapping
        # keeps whatever bf16/meta garbage the load produced. Silent here means corrupted RoPE
        # positions, so name the module rather than skipping quietly.
        if _is_rotary_module(module):
            warn_once(
                logger,
                _WARNED_ROTARY,
                ("no-inv-freq", type(module).__name__),
                f"Rotary module {name} ({type(module).__name__}) exposes no 'inv_freq' and no "
                f"recognized per-layer-type buffer mapping ('rope_init_fns' / 'layer_types'), so "
                f"its frequencies were NOT recomputed in fp32. If this family stores them under "
                f"another attribute, extend fix_rotary_inv_freq — an uninitialized inv_freq "
                f"silently corrupts every position past the first few thousand tokens. Reported "
                f"once per rotary class; every layer of this family is affected.",
            )
        return None

    config = getattr(module, "config", None)
    rope_type = getattr(module, "rope_type", "default")

    if config is not None and (rope_type == "default" or rope_type in ROPE_INIT_FUNCTIONS):
        if rope_type == "default":
            # rope_parameters-aware (a manual getattr(config, "rope_theta") falls back to 10000.0 in v5).
            init_fn = getattr(module, "compute_default_rope_parameters", None) or default_rope_parameters
            inv_freq, scaling = init_fn(config)
            how = "recomputed in float32"
        else:
            inv_freq, scaling = ROPE_INIT_FUNCTIONS[rope_type](config, _materialized_device(module.inv_freq))
            how = f"recomputed via ROPE_INIT_FUNCTIONS[{rope_type}] in float32"
        _apply_recomputed_rope(module, "", inv_freq, scaling, mirror_twin=True, create_scaling=False)
        logger.debug(f"Fixed inv_freq for {name}: {how}")
        return 1

    warn_once(
        logger,
        _WARNED_ROTARY,
        ("unfixable", type(module).__name__, rope_type),
        f"Cannot fix inv_freq for {name} ({type(module).__name__}): no config or unrecognized "
        f"rope_type={rope_type}. Reported once per (class, rope_type); every layer of this "
        f"family is affected.",
    )
    return 0


def _get_alibi_slopes(n: int) -> list[float]:
    """Compute ALiBi slopes for n attention heads (Lightning Attention-2 paper)."""

    def _power_of_2_slopes(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start**i for i in range(n)]

    if math.log2(n).is_integer():
        return _power_of_2_slopes(n)
    closest_power_of_2 = 2 ** math.floor(math.log2(n))
    return (
        _power_of_2_slopes(closest_power_of_2)
        + _get_alibi_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
    )


def _fix_alibi_slope(module: nn.Module, name: str) -> int | None:
    """Bailing MoE Lightning-Attention-2 decay slopes, derived from ctor args nothing stores."""
    if not (hasattr(module, "slope") and hasattr(module, "layer_idx") and hasattr(module, "num_heads")):
        return None
    config = getattr(module, "config", None)
    if config is None:
        return 0

    num_heads = module.num_heads
    layer_idx = module.layer_idx
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if num_hidden_layers is None:
        return 0

    slopes = torch.tensor(_get_alibi_slopes(num_heads), dtype=torch.float)
    new_slope = -slopes * (1 - (layer_idx - 1) / (num_hidden_layers - 1) + 1e-5)
    # Resolve device off a real materialized tensor, not module.slope (still on meta in the lazy path).
    materialized = next(
        (t for t in itertools.chain(module.parameters(), module.buffers()) if t.device.type != "meta"),
        None,
    )
    module.register_buffer("slope", new_slope.to(_materialized_device(materialized)), persistent=False)
    logger.debug(f"Fixed slope for {name}: layer_idx={layer_idx}")
    return 1


def _fix_embed_scale(module: nn.Module, name: str) -> int | None:
    """Gemma4's ``sqrt(hidden_dim)`` scaled-word-embedding factor."""
    if type(module).__name__ != "Gemma4TextScaledWordEmbedding":
        return None
    embed_scale = torch.tensor(module.scalar_embed_scale, device=_materialized_device(module.weight))
    module.register_buffer("embed_scale", embed_scale, persistent=False)
    logger.debug(f"Fixed embed_scale for {name}: scalar={module.scalar_embed_scale:.4g}")
    return 1


# One summary line per group, over a first-match chain: a specialized rotary layout must claim its
# module before the single-``inv_freq`` fallback rebuilds a vision table with the generic formula or
# warns a per-layer-type rotary for the ``inv_freq`` it legitimately lacks. Groups stay independent —
# a module may owe fixes to more than one family.
_ROTARY_FIXERS: tuple[str, tuple[_Fixer, ...]] = (
    "rotary embedding inv_freq buffer(s) to float32",
    (_fix_gemma4_vision_rope, _fix_per_layer_type_rope, _fix_single_inv_freq_rope),
)
_NON_PERSISTENT_FIXERS: tuple[str, tuple[_Fixer, ...]] = (
    "non-persistent buffer(s)",
    (_fix_alibi_slope, _fix_embed_scale),
)


def _walk_and_fix(model: nn.Module, fixers: Sequence[tuple[str, Sequence[_Fixer]]]) -> None:
    """Offer every module to each group's chain in ONE walk of the tree, then report per group."""
    counts = [0] * len(fixers)
    for name, module in model.named_modules():
        for index, (_, chain) in enumerate(fixers):
            for fixer in chain:
                fixed = fixer(module, name)
                if fixed is not None:
                    counts[index] += fixed
                    break
    for count, (summary, _) in zip(counts, fixers, strict=True):
        if count > 0:
            logger.info(f"Fixed {count} {summary}")


def fix_rotary_inv_freq(model: nn.Module) -> None:
    """Recompute inv_freq in explicit float32 for all rotary embedding modules.

    Under bf16 ``from_pretrained`` the ``base**(...)`` runs in bf16 and produces garbage inv_freq; under
    lazy loading it's uninitialized. ``dynamic_rope_update`` doesn't fix static rope types (e.g. yarn).
    """
    _walk_and_fix(model, (_ROTARY_FIXERS,))


def fix_non_persistent_buffers(model: nn.Module) -> None:
    """Recompute non-persistent buffers ``from_pretrained`` / the lazy loaders leave unset.

    Non-persistent buffers (absent from state_dict) hold garbage after ``init_empty_weights()`` / stay on
    meta. Handles Bailing MoE Lightning-Attention-2 slopes and Gemma4 ``embed_scale``.
    """
    _walk_and_fix(model, (_NON_PERSISTENT_FIXERS,))


def _retie_shared_weights(model: nn.Module) -> None:
    """Re-tie shared weights unless the load produced two REAL, distinct tied-pair tensors.

    The re-tie exists because ``from_pretrained`` and the lazy loaders leave the shadow tied weight
    on meta until tying. But transformers 5 honours a checkpoint shipping a DISTINCT head even under
    ``tie_word_embeddings: true``, so when both sides are materialized and distinct an unconditional
    ``tie_weights()`` would silently overwrite the loaded head with the embedding.
    """
    if not hasattr(model, "tie_weights"):
        return
    get_out = getattr(model, "get_output_embeddings", None)
    get_in = getattr(model, "get_input_embeddings", None)
    out_w = getattr(get_out() if callable(get_out) else None, "weight", None)
    in_w = getattr(get_in() if callable(get_in) else None, "weight", None)
    if (
        out_w is not None
        and in_w is not None
        and out_w is not in_w
        and out_w.device.type != "meta"
        and in_w.device.type != "meta"
    ):
        return
    model.tie_weights()


def finalize_loaded_model(model: nn.Module) -> None:
    """Post-load repair EVERY load path must run once weights sit on their final device.

    transformers 5 builds on meta unconditionally and re-materializes every non-persistent buffer as
    ``torch.empty_like``; the only upstream repair is ``_init_weights``' RotaryEmbedding branch,
    which a remote-code family overriding ``_init_weights`` without ``super()`` (Bailing/Ling V3)
    never runs — and a zero inv_freq degenerates RoPE to NoPE with a plausible loss. Runs both fixer
    groups in ONE walk and re-ties shared weights (never over a loaded distinct head). A buffer no
    fix covers stays meta and is rejected at the trainer's device placement.
    """
    _walk_and_fix(model, (_ROTARY_FIXERS, _NON_PERSISTENT_FIXERS))
    _retie_shared_weights(model)
