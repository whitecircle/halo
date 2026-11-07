"""The post-load finalization every load path runs once weights sit on their final device."""

import logging
from collections.abc import Callable

import torch.nn as nn

from halo.kernels.fla_launch import apply_fla_launch_options
from halo.log import warn_once
from halo.model_loading.buffer_repair import apply_buffer_repairs
from halo.model_loading.external_tables import attach_external_tables
from halo.models.introspection.config_access import text_config
from halo.models.introspection.module_tree import backbone_with_layers, decoder_layers, embedding_module
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

_warned_idle_tails: set[str] = set()


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


def _warn_on_idle_mtp_tail(model: nn.Module) -> None:
    """A remote-code release (Ling-3.0-flash) builds its MTP tail as live decoder layers and defaults the loss weight to
    0: every forward pays for them, no gradient reaches them, and every save re-exports them untrained."""
    text = text_config(getattr(model, "config", None))
    if getattr(text, "mtp_loss_scaling_factor", None) != 0:
        return
    backbone = backbone_with_layers(model)
    layers = decoder_layers(backbone) if backbone is not None else ()
    idle = sum(p.numel() for layer in layers[text.num_hidden_layers :] for p in layer.parameters())
    if idle and is_global_main_process():
        warn_once(
            logger,
            _warned_idle_tails,
            type(model).__name__,
            f"{type(model).__name__} built its MTP tail ({idle} parameters) with mtp_loss_scaling_factor: 0: each "
            "forward runs it, nothing trains it, and every save exports it untrained. Drop it with model_init_kwargs: "
            "{num_nextn_predict_layers: 0}, or train it with a non-zero mtp_loss_scaling_factor.",
        )


def finalize_loaded_model(
    model: nn.Module, weights_dir: str | None = None, checkpoint_name: Callable[[str], str] | None = None
) -> None:
    """Post-load repair every load path must run. transformers 5 re-materializes non-persistent buffers as
    ``torch.empty_like``, and a remote-code ``_init_weights`` without ``super()`` never repairs them: a zero inv_freq
    degenerates RoPE to NoPE at a plausible loss. Also sets FLA launch options and attaches external tables."""
    apply_buffer_repairs(model)
    _retie_shared_weights(model)
    _warn_on_idle_mtp_tail(model)
    apply_fla_launch_options()
    attach_external_tables(model, weights_dir, checkpoint_name)
