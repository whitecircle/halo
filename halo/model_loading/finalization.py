"""The post-load finalization every load path runs once weights sit on their final device: the buffer repair
a meta-device build needs, the shared-weight re-tie, FLA's launch options and the external tables a host-table
build holds outside its state."""

from collections.abc import Callable

import torch.nn as nn

from halo.kernels.fla_launch import apply_fla_launch_options
from halo.model_loading.buffer_repair import apply_buffer_repairs
from halo.model_loading.external_tables import attach_external_tables
from halo.models.module_tree import embedding_module


def _retie_shared_weights(model: nn.Module) -> None:
    """Re-tie shared weights unless the load produced two real, distinct tied-pair tensors.

    The re-tie exists because ``from_pretrained`` and the lazy loaders leave the shadow tied weight
    on meta until tying. transformers 5 honours a checkpoint shipping a distinct head even under
    ``tie_word_embeddings: true``, so when both sides are materialized and distinct an unconditional
    ``tie_weights()`` would overwrite the loaded head with the embedding.
    """
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
    """Post-load repair every load path must run once weights sit on their final device.

    transformers 5 re-materializes every non-persistent buffer as ``torch.empty_like``, and a
    remote-code family overriding ``_init_weights`` without ``super()`` never repairs it: a zero
    inv_freq degenerates RoPE to NoPE at a plausible loss. A buffer no fix covers stays on meta and is
    rejected at the trainer's device placement. The FLA launch options the loaded modules need before their first
    launch are set here too (:func:`~halo.kernels.fla_launch.apply_fla_launch_options`), and the external tables a
    host-table build holds outside its state are attached
    (:func:`~halo.model_loading.external_tables.attach_external_tables`) from ``weights_dir``, the local directory a
    loader read the weights from, when it names one, by ``checkpoint_name`` where the model's own names are not the
    checkpoint's (a pipeline stage's re-based layers).
    """
    apply_buffer_repairs(model)
    _retie_shared_weights(model)
    apply_fla_launch_options()
    attach_external_tables(model, weights_dir, checkpoint_name)
