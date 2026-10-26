"""Toolkit patches for the upstream Zaya modeling, applied at model load.

All are idempotent. A renamed upstream internal fails loud rather than skipping a patch.
"""

from __future__ import annotations

import functools
import logging

import torch
import transformers.models.zaya.modeling_zaya as zaya_modeling

from src.kernels.histogram import accumulate_bincount
from src.models.attention.packed_position_ids import install_packed_position_ids_injection
from src.models.attention.selection import is_flash_attn_implementation
from src.models.gradient_checkpointing import counts_toward_expert_load
from src.models.param_roles import FP32_INTERLAYER_RESIDUAL_ATTR
from src.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


def apply_zaya_patches(attn_implementation: str | None = None) -> None:
    """Apply every Zaya patch; the flash one only when a flash implementation is selected."""
    patch_zaya_router_load_recording()
    patch_zaya_gradient_checkpointing_refusal()
    patch_zaya_fp32_interlayer_residual()
    if is_flash_attn_implementation(attn_implementation):
        patch_zaya_flash_packed_position_ids()


def patch_zaya_router_load_recording() -> None:
    """Give ``ZayaRouter`` the expert-load recording its native ``balancing_biases`` buffer implies,
    inside the gate's own forward so EP and plain FSDP share it; upstream counts no selections."""
    router_cls = zaya_modeling.ZayaRouter
    if getattr(router_cls.forward, "_halo_load_recording", False):
        return

    router_cls.expert_load_counter = None  # the slot iter_balancing_routers keys on
    # Armed by BalancingBiasCallback: the native buffer always exists, so presence cannot gate
    # recording as it does on the EP base.
    router_cls.balancing_active = False
    # The discard-slot exclusion (bias update and load metrics) keys on this.
    router_cls._has_discard_expert_slot = True
    original_forward = router_cls.forward

    @functools.wraps(original_forward)
    def forward(self, hidden_states, *args, **kwargs):
        outputs = original_forward(self, hidden_states, *args, **kwargs)
        # The EP base's one-pass-per-microbatch gate, so a frozen reference pass never counts.
        if self.balancing_active and self.training and counts_toward_expert_load():
            with torch.no_grad():
                # Discarded tokens come back as index 0 with zero probability; count them on the
                # last (discard) slot, which the update excludes, not as expert-0 load.
                indices = outputs[2].flatten().clone()
                indices[outputs[1].flatten() == 0] = self.num_router_classes - 1
                self.expert_load_counter = accumulate_bincount(
                    self.expert_load_counter, indices, self.num_router_classes
                )
        return outputs

    forward._halo_load_recording = True
    router_cls.forward = forward
    if is_global_main_process():
        logger.info("Patched ZayaRouter with expert-load recording (bias-update balancing)")


def patch_zaya_gradient_checkpointing_refusal() -> None:
    """Refuse gradient checkpointing on Zaya at the class attribute, so transformers raises its own error
    up front: the recompute through the CCA ``nn.Conv1d`` pair faults in cuDNN, and per-layer GC re-wraps
    the cross-layer EDA state."""
    pretrained_cls = zaya_modeling.ZayaPreTrainedModel
    if pretrained_cls.supports_gradient_checkpointing is False:
        return
    pretrained_cls.supports_gradient_checkpointing = False
    if is_global_main_process():
        logger.info("Patched ZayaPreTrainedModel to refuse gradient checkpointing (cuDNN CCA Conv1d fault)")


def patch_zaya_fp32_interlayer_residual() -> None:
    """Declare on the decoder-layer class that Zaya carries an fp32 residual across FSDP layers, so the
    FSDP2 wrap leaves ``cast_forward_inputs`` off (the bf16 cast breaks non-reentrant GC recompute)."""
    setattr(zaya_modeling.ZayaDecoderLayer, FP32_INTERLAYER_RESIDUAL_ATTR, True)


def patch_zaya_flash_packed_position_ids() -> None:
    """Make Zaya's flash path see ``position_ids``, so packed attention stays isolated. ``ZayaModel``
    never forwards it to its layers, so the stash rides the model forward; the CCA convolution still
    crosses document boundaries on every backend."""
    installed = install_packed_position_ids_injection(
        zaya_modeling,
        zaya_modeling.ZayaModel,
        lambda model: [layer.self_attn for layer in model.layers if hasattr(layer, "self_attn")],
    )
    if installed and is_global_main_process():
        logger.info("Patched Zaya flash attention to receive position_ids (packed-document isolation)")
