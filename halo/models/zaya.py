"""Toolkit patches for the upstream Zaya modeling. All idempotent; a renamed upstream internal fails loud."""

from __future__ import annotations

import functools
import logging

import torch
import torch.nn.functional as F
import transformers.models.zaya.modeling_zaya as zaya_modeling
from torch import nn

from halo.kernels.histogram import accumulate_bincount
from halo.models.attention.packed_position_ids import get_stashed_position_ids, install_packed_position_ids_injection
from halo.models.gradient_checkpointing import counts_toward_expert_load
from halo.models.param_roles import FP32_INTERLAYER_RESIDUAL_ATTR
from halo.models.segment_markers import document_ids, document_starts
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

_DOCUMENT_ISOLATED_ATTR = "_halo_document_isolated"


def apply_zaya_patches() -> None:
    patch_zaya_router_load_recording()
    patch_zaya_gradient_checkpointing_refusal()
    patch_zaya_fp32_interlayer_residual()
    patch_zaya_packed_document_isolation()


def patch_zaya_router_load_recording() -> None:
    """Inside the gate's own forward, so EP and plain FSDP share it; upstream counts no selections."""
    router_cls = zaya_modeling.ZayaRouter
    if getattr(router_cls.forward, "_halo_load_recording", False):
        return

    router_cls.expert_load_counter = None  # the slot iter_balancing_routers keys on
    # Armed by BalancingBiasCallback, as the native buffer always exists and presence cannot gate recording.
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
    """The recompute through the CCA ``nn.Conv1d`` pair faults in cuDNN, and per-layer GC re-wraps the EDA state."""
    pretrained_cls = zaya_modeling.ZayaPreTrainedModel
    if pretrained_cls.supports_gradient_checkpointing is False:
        return
    pretrained_cls.supports_gradient_checkpointing = False
    if is_global_main_process():
        logger.info("Patched ZayaPreTrainedModel to refuse gradient checkpointing (cuDNN CCA Conv1d fault)")


def patch_zaya_fp32_interlayer_residual() -> None:
    """So the FSDP2 wrap leaves ``cast_forward_inputs`` off, as the bf16 cast breaks non-reentrant GC recompute."""
    setattr(zaya_modeling.ZayaDecoderLayer, FP32_INTERLAYER_RESIDUAL_ATTR, True)


def _same_document_masks(position_ids: torch.Tensor | None, seq_len: int, max_shift: int, device) -> list:
    """Per shift ``s`` in ``1..max_shift``, the mask of tokens whose token ``t - s`` shares their document."""
    seq_idx = None if position_ids is None else document_ids(position_ids)
    rows = 1 if seq_idx is None else seq_idx.shape[0]
    offsets = torch.arange(seq_len, device=device) - document_starts(seq_idx, rows, seq_len, device)
    return [offsets >= shift for shift in range(1, max_shift + 1)]


def _document_causal_cca_conv(projection, qk_states: torch.Tensor, same_document: list) -> torch.Tensor:
    """Every tap reaching before its token's document start reads zero. A single depthwise output cannot serve both
    grouped consumers at a boundary, so the pair runs as one grouped map per input shift. The depthwise bias passes
    through every grouped tap, padding included, so it folds into one constant."""
    depthwise, grouped = projection.conv_qk_depthwise, projection.conv_qk_grouped
    for conv in (depthwise, grouped):
        if not isinstance(conv, nn.Conv1d) or conv.stride != (1,) or conv.dilation != (1,) or conv.padding != (0,):
            raise TypeError(
                f"Zaya's document-aware CCA reads {type(conv).__name__} weights directly and reproduces only a "
                f"stride-1, undilated, unpadded nn.Conv1d; an adapter on the CCA convs (or a changed upstream "
                f"conv) would be bypassed. Leave conv_qk_depthwise / conv_qk_grouped out of lora_target_modules."
            )
    batch, seq_len, channels = qk_states.shape
    depth_k, grouped_k = depthwise.kernel_size[0], grouped.kernel_size[0]
    groups = grouped.groups
    width = channels // groups
    # Tap index k reads the token k - (kernel - 1) back: [G, 1, in, d] and [G, out, in, g].
    depth_weight = depthwise.weight.view(groups, 1, width, depth_k)
    grouped_weight = grouped.weight.view(groups, width, width, grouped_k)

    out = None
    for shift in range(depth_k + grouped_k - 1):
        # Every (grouped lag, depthwise lag) pair adding up to this shift.
        weight = sum(
            grouped_weight[..., grouped_k - 1 - lag] * depth_weight[..., depth_k - 1 - (shift - lag)]
            for lag in range(max(0, shift - depth_k + 1), min(shift, grouped_k - 1) + 1)
        )
        source = qk_states
        if shift:
            source = F.pad(qk_states, (0, 0, shift, 0))[:, :seq_len] * same_document[shift - 1][..., None].to(
                qk_states.dtype
            )
        term = torch.einsum("btgi,goi->btgo", source.view(batch, seq_len, groups, width), weight)
        out = term if out is None else out + term
    out = out.reshape(batch, seq_len, channels)
    if depthwise.bias is not None:
        out = out + torch.einsum("goi,gi->go", grouped_weight.sum(-1), depthwise.bias.view(groups, width)).reshape(-1)
    return out if grouped.bias is None else out + grouped.bias


def _install_document_isolated_cca(projection_cls) -> None:
    """Outside cached decoding the conv taps and delayed value reset at every packed document start."""
    original_forward = projection_cls.forward

    @functools.wraps(original_forward)
    def forward(self, hidden_states, past_key_values, conv_mask=None):
        if past_key_values is not None and past_key_values.has_previous_state(self.layer_idx):
            return original_forward(self, hidden_states, past_key_values, conv_mask)
        if conv_mask is not None:
            hidden_states = hidden_states * conv_mask[:, :, None].to(hidden_states.dtype)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        projected_queries = self.q_proj(hidden_states)
        projected_keys = self.k_proj(hidden_states)
        qk_states = torch.cat([projected_queries, projected_keys], dim=-1)

        query_residual = projected_queries.view(*hidden_shape)
        key_residual = projected_keys.view(*hidden_shape).transpose(1, 2)
        key_residual = zaya_modeling.repeat_kv(key_residual, self.num_key_value_groups).transpose(1, 2)
        query_residual = (query_residual + key_residual) * 0.5
        key_residual = query_residual.view(*input_shape, -1, self.num_key_value_groups, self.head_dim).mean(dim=-2)

        if past_key_values is not None:
            conv_state = F.pad(qk_states.transpose(1, 2), (self.conv_kernel_size, 0))[..., -self.conv_kernel_size :]
            past_key_values.update_conv_state(conv_state, self.layer_idx)

        same_document = _same_document_masks(
            get_stashed_position_ids(self), input_shape[1], self.conv_kernel_size, hidden_states.device
        )
        qk_states = _document_causal_cca_conv(self, qk_states, same_document)

        query_hidden_size = query_residual.shape[-2] * query_residual.shape[-1]
        query = qk_states[..., :query_hidden_size].view(*hidden_shape) + query_residual
        key = qk_states[..., query_hidden_size:].view(*hidden_shape) + key_residual

        value_current = self.v_proj_current(hidden_states)
        delayed_v_state = self.v_proj_delayed(hidden_states)
        start_v_state = self.v_proj_delayed(hidden_states.new_zeros(input_shape[0], 1, self.hidden_size))
        previous_v_state = torch.cat([start_v_state, delayed_v_state[:, :-1]], dim=1)
        value_delayed = torch.where(same_document[0][..., None], previous_v_state, start_v_state)

        if past_key_values is not None:
            past_key_values.update_recurrent_state(delayed_v_state[:, -1, :], self.layer_idx)

        value = torch.cat([value_current, value_delayed], dim=-1).view(*hidden_shape)
        return query, key, value

    setattr(forward, _DOCUMENT_ISOLATED_ATTR, True)
    projection_cls.forward = forward


def is_zaya_cca_document_isolated() -> bool:
    return getattr(zaya_modeling.ZayaCCAProjection.forward, _DOCUMENT_ISOLATED_ATTR, False)


def patch_zaya_packed_document_isolation() -> None:
    """``ZayaModel`` never forwards ``position_ids`` to its layers, so the model forward stashes them on each attention
    module and CCA projection, keeping a packed row's documents apart on every backend."""
    if is_zaya_cca_document_isolated():
        return
    install_packed_position_ids_injection(
        zaya_modeling,
        zaya_modeling.ZayaModel,
        lambda model: [
            module
            for layer in model.layers
            if hasattr(layer, "self_attn")
            for module in (layer.self_attn, layer.self_attn.qkv_proj)
        ],
    )
    _install_document_isolated_cca(zaya_modeling.ZayaCCAProjection)
    if is_global_main_process():
        logger.info("Patched Zaya attention and CCA projections for packed-document isolation")
