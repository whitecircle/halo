"""The post-load finalization every load path runs once weights sit on their final device."""

from collections.abc import Callable

import torch.nn as nn

from halo.kernels.fla_launch import apply_fla_launch_options
from halo.model_loading.buffer_repair import apply_buffer_repairs
from halo.model_loading.external_tables import attach_external_tables
from halo.models.module_tree import embedding_module


def _retie_shared_weights(model: nn.Module) -> None:
    """Re-tie unless the load produced two real, distinct tied tensors: loaders leave the shadow on meta until tying,
    while transformers 5 honours a distinct head even under ``tie_word_embeddings`` and ``tie_weights()`` would
    overwrite it."""
    if not hasattr(model, "tie_weights"):
        return
    out_w = getattr(embedding_module(model, "output"), "weight", None)
    in_w = getattr(embedding_module(model, "input"), "weight", None)
    if (
        out_w is not None
        and in_w is not None
        and out_w is not in_w
        and out_w.device.type != "meta"
        and in_w.device.type != "meta"
    ):
        return
    model.tie_weights()


def finalize_loaded_model(
    model: nn.Module, weights_dir: str | None = None, checkpoint_name: Callable[[str], str] | None = None
) -> None:
    """Post-load repair every load path must run. transformers 5 re-materializes non-persistent buffers as
    ``torch.empty_like``, and a remote-code ``_init_weights`` without ``super()`` never repairs them: a zero inv_freq
    degenerates RoPE to NoPE at a plausible loss. Also sets FLA launch options and attaches external tables."""
    apply_buffer_repairs(model)
    _retie_shared_weights(model)
    apply_fla_launch_options()
    attach_external_tables(model, weights_dir, checkpoint_name)
