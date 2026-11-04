"""Ulysses attention: project, RoPE on the local chunk, all-to-all into head parallelism, flash attention over the full
sequence with native GQA, all-to-all back, output projection."""

from __future__ import annotations

import contextlib
import functools
import inspect

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from halo.distributed.context_parallel.all_to_all import heads_to_sequence, sequence_to_heads
from halo.models.attention.selection import (
    flash_attention_deterministic,
    flash_kernel,
    preferred_flash_impl,
    supports_fa4_backward,
)


def _adapt_to_fa2_contract(func):
    """``func`` behind FA2's ``flash_attn_func`` contract: a keyword it lacks (FA4 takes no ``dropout_p``) is
    dropped when neutral and raises otherwise, and a tuple return (FA4's ``(out, lse)``) yields ``out``."""
    params = set(inspect.signature(func).parameters)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for name, neutral in (("dropout_p", 0.0), ("window_size", (-1, -1))):
            if name in kwargs and name not in params:
                if kwargs[name] != neutral:
                    raise ValueError(
                        f"CP flash-attn fallback {func.__module__}.{func.__name__} does not support "
                        f"{name}={kwargs[name]!r} (only the neutral {neutral!r} can be dropped)."
                    )
                kwargs.pop(name)
        out = func(*args, **kwargs)
        return out[0] if isinstance(out, tuple) else out

    return wrapper


@functools.lru_cache(maxsize=2)
def _flash_attn_func(allow_fa4: bool = True):
    """``flash_attn_func`` of :func:`preferred_flash_impl`, else FA2. This probe, not ``_attn_implementation``,
    picks a CP run's kernel; the arch's own build comes first because both images also ship FA2."""
    preferred = preferred_flash_impl(allow_fa4=allow_fa4)
    if preferred != "flash_attention_2":
        with contextlib.suppress(ImportError):
            return _adapt_to_fa2_contract(flash_kernel(preferred, varlen=False))

    try:
        return flash_kernel("flash_attention_2", varlen=False)
    except ImportError as error:
        raise ImportError(
            f"flash-attn is required for Context Parallelism, but neither {preferred} nor flash_attention_2 imports "
            f"here. Run in the Blackwell or Hopper image, which ship flash-attn."
        ) from error


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE over the leading ``cos.shape[-1]`` channels of the last dim; the rest pass through."""
    rotary_dim = cos.shape[-1]
    if rotary_dim == x.shape[-1]:
        return (x * cos) + (rotate_half(x) * sin)
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_rot = (x_rot * cos) + (rotate_half(x_rot) * sin)
    return torch.cat([x_rot, x_pass], dim=-1)


class UlyssesAttention(nn.Module):
    """Ulysses attention over one HF attention module, read on every call since PEFT wraps its projections after this
    is built. Families override :meth:`_project_qkv`, :meth:`_apply_rotary` and :meth:`_post_attention`."""

    HF_MODULE_NAMES: tuple[str, ...] = ()

    # False declares the wrapper's geometry flash-compatible for a family whose modeling code cannot carry a flash
    # label (v4-era remote code declaring `_supports_flash_attn_2`, which transformers v5 ignores).
    REQUIRES_FLASH_ATTN_LABEL: bool = True

    OUTPUT_PROJECTION = "o_proj"

    def __init__(self, original_attention: nn.Module, cp_group: dist.ProcessGroup, cp_size: int):
        super().__init__()
        self.original_attention = original_attention
        self.cp_group = cp_group
        self.cp_size = cp_size

        self.config = original_attention.config
        self.layer_idx = original_attention.layer_idx
        self.head_dim = original_attention.head_dim
        self.scaling = self._resolve_scaling()
        self.attention_dropout = original_attention.attention_dropout
        self.is_causal = original_attention.is_causal
        self.sliding_window = getattr(original_attention, "sliding_window", None)

        self.num_q_heads = self.config.num_attention_heads
        self.num_kv_heads = self._resolve_num_kv_heads(original_attention)

        if self.num_q_heads % cp_size != 0:
            raise ValueError(f"Q heads ({self.num_q_heads}) must be divisible by context_parallel_size ({cp_size})")
        if self.num_kv_heads % cp_size != 0:
            raise ValueError(f"KV heads ({self.num_kv_heads}) must be divisible by context_parallel_size ({cp_size})")

        self.local_q_heads = self.num_q_heads // cp_size
        self.local_kv_heads = self.num_kv_heads // cp_size

        self._allow_fa4 = supports_fa4_backward(self.config)

    def _resolve_scaling(self) -> float:
        """The softmax scale; ``self.head_dim`` is already set when this runs."""
        return self.original_attention.scaling

    @staticmethod
    def _resolve_num_kv_heads(attention: nn.Module) -> int:
        """Head count of the K/V :meth:`_project_qkv` emits, read off the unpatched module so
        ``reject_unsupported_ulysses_model`` resolves it from the same place."""
        return attention.config.num_key_value_heads

    def extra_repr(self) -> str:
        return f"Q={self.num_q_heads}->{self.local_q_heads}, KV={self.num_kv_heads}->{self.local_kv_heads}"

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``[batch, seq/CP, hidden]`` in and out. The other decoder kwargs (mask, cache) are dropped: CP is
        training-only, and the flash call takes causal masking from ``causal=``."""
        batch_size, local_seq_len, _ = hidden_states.shape
        query, key, value, *extras = self._project_qkv(hidden_states, batch_size, local_seq_len)
        cos, sin = position_embeddings
        query, key = self._apply_rotary(query, key, cos.unsqueeze(2), sin.unsqueeze(2), position_ids)
        query, key, value = sequence_to_heads(query, key, value, self.cp_group)
        attn_output = heads_to_sequence(self._attend(query, key, value), self.cp_group)
        attn_output = self._post_attention(attn_output.reshape(batch_size, local_seq_len, -1), *extras)
        return getattr(self.original_attention, self.OUTPUT_PROJECTION)(attn_output), None

    def _project_qkv(
        self, hidden_states: torch.Tensor, batch_size: int, local_seq_len: int
    ) -> tuple[torch.Tensor, ...]:
        """``(query, key, value)`` in ``[B, S, H, D]``, plus any extras :meth:`_post_attention` takes (passed as
        locals, so gradient-checkpoint recompute stays re-entrant). The explicit head counts make a projection
        whose width disagrees with the config raise instead of reshaping."""
        attn = self.original_attention
        return (
            attn.q_proj(hidden_states).view(batch_size, local_seq_len, self.num_q_heads, self.head_dim),
            attn.k_proj(hidden_states).view(batch_size, local_seq_len, self.num_kv_heads, self.head_dim),
            attn.v_proj(hidden_states).view(batch_size, local_seq_len, self.num_kv_heads, self.head_dim),
        )

    def _apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RoPE on this rank's chunk in ``[B, S, H, D]``; rotate-half by default (Qwen3, Qwen3.5/3.6, Bailing)."""
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)

    def _attend(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Flash attention over the full sequence for this rank's heads, native GQA. The sliding window follows HF:
        applied only past its own length, minus 1 for the diagonal."""
        window = self.sliding_window
        return _flash_attn_func(self._allow_fa4)(
            query,
            key,
            value,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            softmax_scale=self.scaling,
            causal=self.is_causal,
            window_size=(window - 1, window - 1) if window and key.shape[1] > window else (-1, -1),
            deterministic=flash_attention_deterministic(),
        )

    def _post_attention(self, attn_output: torch.Tensor, *extras: torch.Tensor) -> torch.Tensor:
        """Family hook on the attention output just before the output projection (default: identity)."""
        return attn_output


def _deinterleave_rotary_channels(x: torch.Tensor) -> torch.Tensor:
    """Regroup the last dim's pairs ``(x0, x1), (x2, x3), ...`` as ``(x0, x2, ..., x1, x3, ...)``, so rotate-half
    then applies an interleaved rotary (transformers' ``apply_rotary_pos_emb_interleave``)."""
    return x.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2)


class MLAUlyssesAttention(UlyssesAttention):
    """DeepSeek-V3-style MLA (GLM4-MoE-Lite, Mistral4): compressed KV expanded through ``kv_b_proj``, RoPE on
    the rope half of each head, and a ``v_head_dim`` that may differ from ``qk_head_dim``."""

    def __init__(self, original_attention: nn.Module, cp_group: dist.ProcessGroup, cp_size: int):
        # MLA spells the Q/K head dim ``qk_head_dim``; back-filled only where the family carries no ``head_dim``.
        if not hasattr(original_attention, "head_dim"):
            original_attention.head_dim = original_attention.qk_head_dim
        super().__init__(original_attention, cp_group, cp_size)

        self.qk_head_dim = original_attention.qk_head_dim
        self.qk_nope_head_dim = original_attention.qk_nope_head_dim
        self.qk_rope_head_dim = original_attention.qk_rope_head_dim
        self.v_head_dim = original_attention.v_head_dim
        self.kv_lora_rank = original_attention.kv_lora_rank

        # Q compression only (KV is always compressed); transformers keeps the unused branch's attribute
        # present but None, so presence alone does not answer it.
        self.use_q_lora = getattr(original_attention, "q_a_proj", None) is not None

        # Read unconditionally, as both families' own forwards do: their configs default it to True.
        self.rope_interleave = bool(self.config.rope_interleave)

    @staticmethod
    def _resolve_num_kv_heads(attention: nn.Module) -> int:
        """Derived from ``kv_b_proj``, which both families build with one head per query head, so
        ``config.num_key_value_heads`` does not describe the tensor reshaped."""
        return attention.kv_b_proj.out_features // (attention.qk_nope_head_dim + attention.v_head_dim)

    def extra_repr(self) -> str:
        return (
            f"{super().extra_repr()}, qk_head_dim={self.qk_head_dim} (nope={self.qk_nope_head_dim}, "
            f"rope={self.qk_rope_head_dim}), v_head_dim={self.v_head_dim}, kv_lora_rank={self.kv_lora_rank}, "
            f"use_q_lora={self.use_q_lora}, rope_interleave={self.rope_interleave}"
        )

    def _project_qkv(
        self, hidden_states: torch.Tensor, batch_size: int, local_seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Q/K ``[B, S, heads, qk_head_dim]`` and V ``[B, S, kv_heads, v_head_dim]``. The trailing ``qk_rope_head_dim``
        slots of the compressed KV are the shared rotary K, broadcast to every KV head so the all-to-all's head scatter
        sees a contiguous tensor."""
        attn = self.original_attention

        if self.use_q_lora:
            q = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(hidden_states)))
        else:
            q = attn.q_proj(hidden_states)
        q = q.view(batch_size, local_seq_len, self.num_q_heads, self.qk_head_dim)

        compressed_kv = attn.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot_shared = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = attn.kv_b_proj(attn.kv_a_layernorm(k_pass))
        kv = kv.view(batch_size, local_seq_len, self.num_kv_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]

        k_rot = k_rot_shared.view(batch_size, local_seq_len, 1, self.qk_rope_head_dim)
        k_rot = k_rot.expand(batch_size, local_seq_len, self.num_kv_heads, self.qk_rope_head_dim)

        k = torch.cat([k_nope, k_rot.contiguous()], dim=-1)
        return q, k, v

    def _apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RoPE on the trailing ``qk_rope_head_dim`` channels of each head only."""

        def rope(x: torch.Tensor) -> torch.Tensor:
            x_nope, x_rope = x[..., : self.qk_nope_head_dim], x[..., self.qk_nope_head_dim :]
            if self.rope_interleave:
                x_rope = _deinterleave_rotary_channels(x_rope)
            return torch.cat([x_nope, apply_rope(x_rope, cos, sin)], dim=-1)

        return rope(q), rope(k)

    def _attend(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """V padded to ``qk_head_dim`` for the kernel, which needs one head dim across Q/K/V, then cropped."""
        pad = self.qk_head_dim - self.v_head_dim
        if pad <= 0:
            return super()._attend(query, key, value)
        return super()._attend(query, key, F.pad(value, (0, pad)))[..., : self.v_head_dim].contiguous()
