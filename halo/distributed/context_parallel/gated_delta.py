"""Ulysses head redistribution for dense Qwen3.5/3.6 GatedDeltaNet layers."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5

from halo.distributed.context_parallel.all_to_all import UlyssesAllToAll, heads_to_sequence
from halo.distributed.context_parallel.config import CPConfigError
from halo.distributed.context_parallel.sequence_layer import UlyssesSequenceLayerBase


class Qwen3_5UlyssesGatedDelta(UlyssesSequenceLayerBase):
    """Dense Qwen3.5/3.6 GatedDeltaNet on full sequences for a rank's key/value-head groups."""

    HF_MODULE_NAMES = ("Qwen3_5GatedDeltaNet",)

    def __init__(self, original_attention: nn.Module, cp_group: dist.ProcessGroup, cp_size: int):
        super().__init__()
        self.reject_unsupported_geometry(original_attention, cp_size)
        self.original_attention = original_attention
        self.cp_group = cp_group
        self.cp_size = cp_size
        self.cp_rank = dist.get_rank(cp_group)

    @staticmethod
    def reject_unsupported_geometry(module: nn.Module, cp_size: int) -> None:
        """Refuse any geometry outside dense hybrid CP2, without normalizing the module."""
        if cp_size != 2:
            raise CPConfigError(
                "Hybrid GatedDeltaNet supports context_parallel_size: 2 only. Use CP2, or train without CP."
            )
        for label, heads in (("key", module.num_k_heads), ("value", module.num_v_heads)):
            if heads < 1 or heads % cp_size:
                raise CPConfigError(
                    f"GatedDeltaNet {label} heads ({heads}) must be divisible by context_parallel_size ({cp_size})."
                )
        if module.num_v_heads % module.num_k_heads:
            raise CPConfigError(
                "GatedDeltaNet value heads must form complete groups of key heads. Use the native model configuration."
            )
        if module.conv1d.bias is not None:
            raise CPConfigError(
                "Hybrid CP requires the native bias-free GatedDeltaNet convolution. Use the unmodified model, or train without CP."
            )

    def forward(self, hidden_states, cache=None, attention_mask=None, **kwargs):
        """COLLECTIVE over the CP group: return this rank's sequence chunk, refusing caches and packed rows."""
        if (
            cache is not None
            or any(kwargs.get(key) is not None for key in ("past_key_value", "past_key_values"))
            or kwargs.get("use_cache") not in (None, False)
        ):
            raise CPConfigError("Hybrid CP has no inference-state exchange. Set use_cache=False and pass no cache.")
        if attention_mask is not None:
            raise CPConfigError(
                "Dense hybrid CP expects no layer attention_mask. Use UlyssesModel with right-padded rows."
            )
        if any(
            kwargs.get(key) is not None
            for key in ("seq_idx", "cu_seq_lens_q", "cu_seq_lens_k", "cu_seqlens", "cu_seqlens_cpu")
        ):
            raise CPConfigError("Hybrid CP does not support packed documents. Disable packing and padding_free.")
        unexpected = kwargs.keys() - {
            "past_key_value",
            "past_key_values",
            "use_cache",
            "seq_idx",
            "cu_seq_lens_q",
            "cu_seq_lens_k",
            "cu_seqlens",
            "cu_seqlens_cpu",
        }
        if unexpected:
            raise TypeError(f"Unsupported GatedDeltaNet forward arguments under hybrid CP: {sorted(unexpected)}")
        attn = self.original_attention
        batch, local_length, _ = hidden_states.shape
        q, k, v = attn.in_proj_qkv(hidden_states).split((attn.key_dim, attn.key_dim, attn.value_dim), dim=-1)
        q = q.reshape(batch, local_length, attn.num_k_heads, attn.head_k_dim)
        k = k.reshape(batch, local_length, attn.num_k_heads, attn.head_k_dim)
        v = v.reshape(batch, local_length, attn.num_v_heads, attn.head_v_dim)
        a, b = attn.in_proj_a(hidden_states), attn.in_proj_b(hidden_states)
        q, k, v, a, b = UlyssesAllToAll.apply(self.cp_group, True, q, k, v, a[..., None], b[..., None])
        mixed = torch.cat((q.flatten(-2), k.flatten(-2), v.flatten(-2)), dim=-1).transpose(1, 2)
        local_keys, local_values = attn.num_k_heads // self.cp_size, attn.num_v_heads // self.cp_size
        key_width, value_width = local_keys * attn.head_k_dim, local_values * attn.head_v_dim
        key_start, value_start = self.cp_rank * key_width, self.cp_rank * value_width
        weight = attn.conv1d.weight.squeeze(1)
        weight = torch.cat(
            (
                weight[key_start : key_start + key_width],
                weight[attn.key_dim + key_start : attn.key_dim + key_start + key_width],
                weight[2 * attn.key_dim + value_start : 2 * attn.key_dim + value_start + value_width],
            )
        )
        mixed = modeling_qwen3_5.causal_conv1d_fn(mixed, weight, None, activation=attn.activation).transpose(1, 2)
        length = local_length * self.cp_size
        q, k, v = mixed.split((key_width, key_width, value_width), dim=-1)
        repeat = local_values // local_keys
        q = q.reshape(batch, length, local_keys, attn.head_k_dim).repeat_interleave(repeat, dim=2)
        k = k.reshape(batch, length, local_keys, attn.head_k_dim).repeat_interleave(repeat, dim=2)
        v = v.reshape(batch, length, local_values, attn.head_v_dim)
        heads = slice(self.cp_rank * local_values, (self.cp_rank + 1) * local_values)
        beta = b.squeeze(-1).sigmoid()
        g = -attn.A_log[heads].float().exp() * F.softplus(a.squeeze(-1).float() + attn.dt_bias[heads])
        core, _ = modeling_qwen3_5.torch_chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        core = heads_to_sequence(core, self.cp_group)
        z = attn.in_proj_z(hidden_states).reshape(-1, attn.head_v_dim)
        output = attn.norm(core.reshape(-1, attn.head_v_dim), z)
        return attn.out_proj(output.reshape(batch, local_length, attn.value_dim))
