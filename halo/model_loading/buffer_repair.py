"""Recompute the model buffers a meta-device build leaves uninitialized or stranded on meta: RoPE tables,
Bailing's Lightning-Attention slopes, the Gemma scaled embeddings' ``embed_scale``."""

from __future__ import annotations

import contextlib
import itertools
import logging
import math
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from halo.hardware import current_device
from halo.log import warn_once
from halo.models.attention.layout import resolve_head_dim
from halo.models.introspection.config_access import text_config

logger = logging.getLogger(__name__)

# Keyed by (reason, class name), since an unhandled layout is a family property and this logger reaches every rank.
_warned_unfixed: set = set()

# A fixer returns how many buffers it rebuilt, or ``None`` to hand the module to the next fixer.
_Fixer = Callable[[nn.Module, str], int | None]


def default_rope_parameters(config, device=None, **kwargs) -> tuple[torch.Tensor, float]:
    """``(inv_freq, attention_scaling)`` for plain unscaled RoPE, read ``rope_parameters``-first since the flat
    attributes survive only on transformers-4 remote-code configs. ``kwargs`` absorbs the registry's ``layer_type``."""
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


def _materialized_device(tensor: torch.Tensor | None) -> torch.device:
    """``tensor``'s device, or this rank's when absent or on meta; a buffer built on meta turns the first forward's
    activations into meta tensors rather than raising."""
    if tensor is None or tensor.device.type == "meta":
        return current_device()
    return tensor.device


def _apply_recomputed_rope(
    module: nn.Module, prefix: str, inv_freq: torch.Tensor, scaling: float, *, mirror_twin: bool, create_scaling: bool
) -> None:
    """Install a recomputed rotary table in fp32. The twin goes in by plain assignment, as ``register_buffer`` rejects
    the twin a transformers-4 rotary keeps as a plain attribute. ``mirror_twin`` aliases the live table;
    ``create_scaling`` installs a scaling the module never declared (per-layer-type only)."""
    name = f"{prefix}inv_freq"
    inv_freq = inv_freq.float().to(_materialized_device(getattr(module, name, None)))
    # Keep the declared persistence: a remote-code family may ship ``inv_freq`` in its checkpoints.
    persistent = name in module._buffers and name not in module._non_persistent_buffers_set
    module.register_buffer(name, inv_freq, persistent=persistent)
    twin = f"{prefix}original_inv_freq"
    if mirror_twin:
        setattr(module, twin, inv_freq)
    elif hasattr(module, twin):
        setattr(module, twin, inv_freq.clone())
    scaling_name = f"{prefix}attention_scaling"
    if create_scaling or hasattr(module, scaling_name):
        setattr(module, scaling_name, scaling)


def _unfixable(module: nn.Module, name: str, reason: str) -> int:
    """Claim ``module`` without rebuilding its buffers, warned once per (class, reason)."""
    warn_once(
        logger,
        _warned_unfixed,
        ("unfixable", type(module).__name__, reason),
        f"Cannot recompute the buffers of {name} ({type(module).__name__}): {reason}, so they keep "
        f"whatever the load produced — uninitialized memory unless the model's own init filled them. "
        f"Reported once per (class, reason); every layer of this family is affected.",
    )
    return 0


def _fix_layer_type_inv_freq(module: nn.Module, module_name: str, layer_type: str, rope_init_fn) -> bool:
    """Recompute one per-layer-type rotary's table; returns whether it did. A layer type with no buffer beside it does
    not rotate (a null ``rope_parameters`` entry, as in Laguna and Gemma 4), and building a table would invent
    state."""
    buf_name = f"{layer_type}_inv_freq"
    if getattr(module, buf_name, None) is None:
        return False
    # The layer-type-resolved view, as the module's own __init__ uses: the raw config raises
    # AmbiguousGlobalPerLayerAttributeError on Gemma4's sliding leg and sizes the global one off the wrong head_dim.
    config = module.config
    with contextlib.suppress(AttributeError, ValueError):
        config = config.per_layer_config[layer_type]
    inv_freq, scaling = rope_init_fn(config, layer_type=layer_type)
    _apply_recomputed_rope(module, f"{layer_type}_", inv_freq, scaling, mirror_twin=False, create_scaling=True)
    # Stat-free: reading inv_freq.min()/.max() would force a device sync per layer on every rank.
    logger.debug(f"Fixed {buf_name} for {module_name} (rope_type={module.rope_type.get(layer_type)})")
    return True


def _own_rope_init_fn(module: nn.Module, rope_type: str) -> Callable | None:
    """The rotary's own ``compute_<rope_type>_rope_parameters``, for types ``ROPE_INIT_FUNCTIONS`` lacks."""
    return getattr(module, f"compute_{rope_type}_rope_parameters", None)


def _resolve_rope_init_fn(module: nn.Module, layer_type: str) -> Callable:
    rope_type = module.rope_type.get(layer_type, "default")
    return _own_rope_init_fn(module, rope_type) or ROPE_INIT_FUNCTIONS[rope_type]


def _fix_per_layer_type_rope(module: nn.Module, name: str) -> int | None:
    """Every rotated layer type of a rotary keying inv_freq by layer type, claimed off ``rope_init_fns`` or
    ``layer_types`` plus a dict ``rope_type``."""
    if hasattr(module, "rope_init_fns"):
        init_fns = module.rope_init_fns.items()
    elif hasattr(module, "layer_types") and isinstance(getattr(module, "rope_type", None), dict):
        init_fns = ((layer_type, _resolve_rope_init_fn(module, layer_type)) for layer_type in module.layer_types)
    else:
        return None
    if getattr(module, "config", None) is None:
        return _unfixable(module, name, "the module carries no config to size its rotary tables from")
    return sum(_fix_layer_type_inv_freq(module, name, layer_type, fn) for layer_type, fn in init_fns)


def _fix_single_inv_freq_rope(module: nn.Module, name: str) -> int | None:
    """The ordinary single-``inv_freq`` rotary: every layout the fixers above do not claim."""
    if getattr(module, "inv_freq", None) is None:
        # Rotaries share no base class, so the class name is the signal; it only decides the warning.
        if "rotary" in type(module).__name__.lower():
            warn_once(
                logger,
                _warned_unfixed,
                ("no-inv-freq", type(module).__name__),
                f"Rotary module {name} ({type(module).__name__}) exposes no 'inv_freq' and no "
                f"recognized per-layer-type buffer mapping ('rope_init_fns' / 'layer_types'), so "
                f"its frequencies were NOT recomputed in fp32. If this family stores them under "
                f"another attribute, add a fixer to _ROTARY_FIXERS — an uninitialized inv_freq "
                f"silently corrupts every position past the first few thousand tokens. Reported "
                f"once per rotary class; every layer of this family is affected.",
            )
        return None

    config = getattr(module, "config", None)
    rope_type = getattr(module, "rope_type", "default")
    own_init_fn = _own_rope_init_fn(module, rope_type)

    if config is not None and (own_init_fn is not None or rope_type == "default" or rope_type in ROPE_INIT_FUNCTIONS):
        if own_init_fn is not None or rope_type == "default":
            # rope_parameters-aware (a manual getattr(config, "rope_theta") falls back to 10000.0 in v5).
            init_fn = own_init_fn or default_rope_parameters
            inv_freq, scaling = init_fn(config)
            how = f"recomputed via the module's own {rope_type} init in float32"
        else:
            inv_freq, scaling = ROPE_INIT_FUNCTIONS[rope_type](config, _materialized_device(module.inv_freq))
            how = f"recomputed via ROPE_INIT_FUNCTIONS[{rope_type}] in float32"
        _apply_recomputed_rope(module, "", inv_freq, scaling, mirror_twin=True, create_scaling=False)
        logger.debug(f"Fixed inv_freq for {name}: {how}")
        return 1

    return _unfixable(module, name, f"no config or unrecognized rope_type={rope_type}")


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
    num_hidden_layers = getattr(getattr(module, "config", None), "num_hidden_layers", None)
    if num_hidden_layers is None:
        return _unfixable(module, name, "no config num_hidden_layers to derive the decay slopes from")

    num_heads = module.num_heads
    layer_idx = module.layer_idx

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
    """A scaled embedding's ``embed_scale``, which Gemma-lineage classes derive from ``scalar_embed_scale``."""
    if not hasattr(module, "scalar_embed_scale") or "embed_scale" not in module._buffers:
        return None
    embed_scale = torch.tensor(module.scalar_embed_scale, device=_materialized_device(module.weight))
    module.register_buffer("embed_scale", embed_scale, persistent=False)
    logger.debug(f"Fixed embed_scale for {name}: scalar={module.scalar_embed_scale:.4g}")
    return 1


# First-match chains: a per-layer-type rotary must claim its module before the single-``inv_freq`` fallback warns it.
# Groups stay independent, since a module may need fixes from more than one family.
_ROTARY_FIXERS: tuple[str, tuple[_Fixer, ...]] = (
    "rotary embedding inv_freq buffer(s) to float32",
    (_fix_per_layer_type_rope, _fix_single_inv_freq_rope),
)
_NON_PERSISTENT_FIXERS: tuple[str, tuple[_Fixer, ...]] = (
    "non-persistent buffer(s)",
    (_fix_alibi_slope, _fix_embed_scale),
)


def _walk_and_fix(model: nn.Module, fixers: Sequence[tuple[str, Sequence[_Fixer]]]) -> None:
    """Offer every module to each group's chain in one walk of the tree, then report per group."""
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


def apply_buffer_repairs(model: nn.Module) -> None:
    """Recompute, in one walk, every buffer the fixer chains cover, each group reported once."""
    _walk_and_fix(model, (_ROTARY_FIXERS, _NON_PERSISTENT_FIXERS))
