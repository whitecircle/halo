"""Which checkpoint a live model's weights and config schema came from, and whether that was a given one."""

from __future__ import annotations

import os

from src.models.unwrapping import unwrap_toolkit_wrappers

# Stamped by load_distributed_model on the loaded model: where its weights were actually read (None
# under init_from_scratch). One spelling for the writer and the resume-coverage reader.
LOADED_WEIGHTS_FROM_ATTR = "_loaded_weights_from"


def config_schema_source(model) -> str | None:
    """The checkpoint this model's schema came from: a local directory or a Hub repo id, or None.

    ``load_distributed_model`` stamps :data:`LOADED_WEIGHTS_FROM_ATTR`; every other loader leaves
    transformers' ``config._name_or_path``, the same string. An ``init_from_scratch`` model took its
    geometry from that config, so the fallback is the right schema source there too.
    """
    return getattr(model, LOADED_WEIGHTS_FROM_ATTR, None) or getattr(
        getattr(model, "config", None), "_name_or_path", None
    )


def weights_read_from(model) -> str | None:
    """Where the live model's weights were read, or None for a from-config random init.

    ``load_distributed_model`` stamps ``_loaded_weights_from`` (None under ``init_from_scratch``,
    whose model carries the checkpoint's ``_name_or_path`` while holding random weights). A model
    constructed elsewhere (tests, user code) falls back to ``config._name_or_path``, which for any
    ``from_pretrained`` construction is also where the weights came from.
    """
    live = unwrap_toolkit_wrappers(model)
    if hasattr(live, LOADED_WEIGHTS_FROM_ATTR):
        return getattr(live, LOADED_WEIGHTS_FROM_ATTR)
    return str(getattr(getattr(live, "config", None), "_name_or_path", "")) or None


def built_from_checkpoint(live_source: str | None, checkpoint: str) -> bool:
    """Whether a model whose weights were read from ``live_source`` was built from ``checkpoint``.

    ``realpath`` identity, so a symlinked or relative spelling of the same directory still counts;
    ``None`` (an ``init_from_scratch`` build) never does. Resolution is node-local, so every caller
    joins the verdict across ranks rather than branching on its own — on a non-shared filesystem a
    per-rank answer splits the world.
    """
    return live_source is not None and os.path.realpath(live_source) == os.path.realpath(checkpoint)


def describe_weights_source(live_source: str | None, subject: str = "model") -> str:
    """Where ``subject``'s live weights actually came from, for the refusals that name it."""
    if live_source is None:
        return f"this {subject} read no weights at all (init_from_scratch)"
    return f"this {subject} was loaded from '{live_source}'"
