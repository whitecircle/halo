"""Which checkpoint a live model's weights and config schema came from, and whether that was a given one."""

from __future__ import annotations

import os

from halo.models.introspection.unwrapping import unwrap_toolkit_wrappers

# Stamped by load_distributed_model: where the weights were read (None under init_from_scratch).
LOADED_WEIGHTS_FROM_ATTR = "_loaded_weights_from"


def config_schema_source(model) -> str | None:
    """The checkpoint this model's schema came from, or None; an ``init_from_scratch`` model took its geometry from
    ``config._name_or_path``, so that fallback is the right source there too."""
    return getattr(model, LOADED_WEIGHTS_FROM_ATTR, None) or getattr(
        getattr(model, "config", None), "_name_or_path", None
    )


def weights_read_from(model) -> str | None:
    """Where the live weights were read, or None for a from-config random init; an unstamped model falls back to
    ``config._name_or_path``, which is where any ``from_pretrained`` read them."""
    live = unwrap_toolkit_wrappers(model)
    if hasattr(live, LOADED_WEIGHTS_FROM_ATTR):
        return getattr(live, LOADED_WEIGHTS_FROM_ATTR)
    return str(getattr(getattr(live, "config", None), "_name_or_path", "")) or None


def built_from_checkpoint(live_source: str | None, checkpoint: str) -> bool:
    """``realpath`` identity; ``None`` (init_from_scratch) never counts. Node-local, so every caller joins the verdict
    across ranks, since a per-rank answer splits the world on a non-shared filesystem."""
    return live_source is not None and os.path.realpath(live_source) == os.path.realpath(checkpoint)


def describe_weights_source(live_source: str | None, subject: str = "model") -> str:
    if live_source is None:
        return f"this {subject} read no weights at all (init_from_scratch)"
    return f"this {subject} was loaded from '{live_source}'"
