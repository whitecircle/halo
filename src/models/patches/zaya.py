"""Toolkit patches for the upstream Zaya modeling, applied at model load.

All are idempotent; a renamed upstream internal fails the import or attribute access LOUDLY, since
silently skipping one would train with no balancing, allow GC into the cuDNN CCA fault, or leak
packed attention.
"""

from __future__ import annotations

import functools

import torch
import transformers.models.zaya.modeling_zaya as zaya_modeling
from accelerate.logging import get_logger

from src.distributed.expert_parallel.gc_scope import counts_toward_expert_load
from src.kernels.histogram import accumulate_bincount
from src.models.patches.attention import install_packed_position_ids_injection

logger = get_logger(__name__)


def apply_zaya_patches(attn_implementation: str | None = None) -> None:
    """Apply every Zaya patch; the flash one only when a flash implementation is selected."""
    patch_zaya_router_load_recording()
    patch_zaya_gradient_checkpointing_refusal()
    patch_zaya_fp32_interlayer_residual()
    if attn_implementation and attn_implementation.startswith("flash_attention"):
        patch_zaya_flash_packed_position_ids()


def patch_zaya_router_load_recording() -> None:
    """Give ``ZayaRouter`` the load recording its native ``balancing_biases`` buffer implies.

    The hub modeling ships the persistent balancing buffer but nothing that counts expert
    selections, so ``RouterBiasBalancingCallback`` would compute every update from zero loads — a
    silent no-op. Declares the ``expert_load_counter`` slot ``iter_balancing_routers`` keys on and
    records inside the gate's own forward, which serves EP and plain FSDP alike.
    """
    router_cls = zaya_modeling.ZayaRouter
    if getattr(router_cls.forward, "_halo_load_recording", False):
        return

    router_cls.expert_load_counter = None  # the slot iter_balancing_routers keys on
    # Armed by RouterBiasBalancingCallback at train begin. Zaya's native balancing_biases always
    # exists, so presence cannot gate recording (as it does on the EP base): without this flag a
    # moe_balancing=none run would scatter_add_ into a counter nothing consumes, every forward.
    router_cls.balancing_active = False
    # The callback's exclusion of the discard slot from the bias update keys on this (with the
    # model-type backstop); the upstream router declares no such flag of its own.
    router_cls._has_discard_expert_slot = True
    original_forward = router_cls.forward

    @functools.wraps(original_forward)
    def forward(self, hidden_states, *args, **kwargs):
        outputs = original_forward(self, hidden_states, *args, **kwargs)
        # Same one-pass-per-microbatch gate as the EP base's _record_expert_load, so a frozen
        # reference/teacher pass never skews the balance the callback corrects.
        if self.balancing_active and self.training and counts_toward_expert_load():
            with torch.no_grad():
                # The router returned indices with its discard slot already masked to 0 and those
                # tokens' probs zeroed — counting them as expert-0 load would drive the callback to
                # starve expert 0, so route them back to the last (discard) slot, which the update
                # excludes.
                indices = outputs[2].flatten().clone()
                indices[outputs[1].flatten() == 0] = self.num_router_classes - 1
                self.expert_load_counter = accumulate_bincount(
                    self.expert_load_counter, indices, self.num_router_classes
                )
        return outputs

    forward._halo_load_recording = True
    router_cls.forward = forward
    logger.info("Patched ZayaRouter with expert-load recording (bias-update balancing)")


def patch_zaya_gradient_checkpointing_refusal() -> None:
    """Refuse gradient checkpointing on Zaya up front, at the class attribute.

    Upstream declares ``supports_gradient_checkpointing = True``, but GC backward recompute through
    the CCA's ``nn.Conv1d`` pair hits a cuDNN internal error on the training image, and per-layer GC
    re-wraps the cross-layer EDA state with a fresh grad_fn. Flipping the class attribute raises
    transformers' own clear error instead of faulting mid-backward. See ``agent-docs/models/zaya.md``.
    """
    pretrained_cls = zaya_modeling.ZayaPreTrainedModel
    if pretrained_cls.supports_gradient_checkpointing is False:
        return
    pretrained_cls.supports_gradient_checkpointing = False
    logger.info("Patched ZayaPreTrainedModel to refuse gradient checkpointing (cuDNN CCA Conv1d fault)")


def patch_zaya_fp32_interlayer_residual() -> None:
    """Declare on the decoder-layer class that Zaya carries an fp32 residual across FSDP layers.

    The FSDP2 wrap reads this to leave ``cast_forward_inputs`` off: casting the layer-boundary
    activation to bf16 makes non-reentrant GC recompute produce an fp32 tensor where the forward
    saved a bf16 view, which torch rejects. Declared by the class that owns the residual, so the wrap
    stays model-agnostic.
    """
    zaya_modeling.ZayaDecoderLayer._fp32_interlayer_residual = True


def patch_zaya_flash_packed_position_ids() -> None:
    """Make Zaya's flash path see ``position_ids``, so packed ATTENTION stays isolated.

    ``ZayaModel.forward`` declares ``position_ids`` and never forwards it into its decoder layers, so
    the varlen packed path never engages and a packed row runs as ONE dense causal sequence on FA
    (dense backends form the packed mask at the model level). The stash therefore rides the MODEL
    forward, unlike Mistral4's. Attention isolation is necessary, not sufficient: the CCA convolution
    still carries state across document boundaries on every backend (``agent-docs/models/zaya.md``).
    """
    installed = install_packed_position_ids_injection(
        zaya_modeling,
        zaya_modeling.ZayaModel,
        lambda model: [layer.self_attn for layer in model.layers if hasattr(layer, "self_attn")],
    )
    if installed:
        logger.info("Patched Zaya flash attention to receive position_ids (packed-document isolation)")
