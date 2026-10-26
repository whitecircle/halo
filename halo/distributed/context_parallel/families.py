"""The per-family Ulysses attention wrappers, and the HF attention class name -> wrapper map read off their
``HF_MODULE_NAMES``."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.models.cohere2_moe.modeling_cohere2_moe import rotate_half as rotate_interleaved_pairs
from transformers.models.mistral4.modeling_mistral4 import get_llama_4_attn_scale

from halo.distributed.context_parallel.ulysses_attention import MLAUlyssesAttention, UlyssesAttention
from halo.subclass_registry import build_class_claim_map


class BailingMoeV2UlyssesAttention(UlyssesAttention):
    """Bailing MoE / Ling 2.0 (v4-era remote code): a fused ``query_key_value`` projection, optional Q/K RMSNorm,
    partial rotary, a ``dense`` output projection, an inline softmax scale and a three-value return. All three
    backend classes are claimed (same weights); the Lightning-Attention-2 sibling sharing their names is refused
    by :func:`~halo.distributed.context_parallel.model_rules.reject_unsupported_ulysses_model`."""

    HF_MODULE_NAMES = ("BailingMoeV2Attention", "BailingMoeV2FlashAttention2", "BailingMoeV2SdpaAttention")

    # The remote code declares only the v4 ``_supports_flash_attn_2``, so every Bailing run is labelled sdpa;
    # the geometry (uniform head_dim GQA, no sinks, no additive bias) is what CP's flash kernel needs.
    REQUIRES_FLASH_ATTN_LABEL = False

    OUTPUT_PROJECTION = "dense"

    def __init__(self, original_attention: nn.Module, cp_group: dist.ProcessGroup, cp_size: int):
        super().__init__(original_attention, cp_group, cp_size)
        self.use_qk_norm = bool(getattr(self.config, "use_qk_norm", False))

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, use_qk_norm={self.use_qk_norm}"

    def _resolve_scaling(self) -> float:
        return self.head_dim**-0.5

    def _project_qkv(
        self, hidden_states: torch.Tensor, batch_size: int, local_seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn = self.original_attention

        qkv = attn.query_key_value(hidden_states).view(
            batch_size, local_seq_len, self.num_q_heads + 2 * self.num_kv_heads, self.head_dim
        )
        query, key, value = qkv.split([self.num_q_heads, self.num_kv_heads, self.num_kv_heads], dim=-2)

        if self.use_qk_norm:
            query, key = attn.query_layernorm(query), attn.key_layernorm(key)
        return query, key, value

    def forward(
        self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor], **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None, object]:
        """The base forward in the v4 return contract: Bailing's decoder and MTP layers unpack
        ``(hidden_states, attn_weights, present_key_value)``. CP never writes the cache handed back."""
        attn_output, attn_weights = super().forward(hidden_states, position_embeddings, **kwargs)
        return attn_output, attn_weights, kwargs.get("past_key_value")


def _apply_interleaved_rope_fp32(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Cohere2 MoE's ``apply_rotary_pos_emb`` on one tensor: interleaved rotary computed in fp32."""
    dtype = x.dtype
    x = x.float()
    rotated = rotate_interleaved_pairs(x)
    return (x * cos + rotated * sin).to(dtype)


class Cohere2MoeUlyssesAttention(UlyssesAttention):
    """Cohere2 MoE (Command A+): interleaved fp32 RoPE on sliding-window layers (and dense-prefix
    ``force_rope`` ones) only; full-attention layers are NoPE."""

    HF_MODULE_NAMES = ("Cohere2MoeAttention",)

    def __init__(self, original_attention: nn.Module, cp_group: dist.ProcessGroup, cp_size: int):
        super().__init__(original_attention, cp_group, cp_size)
        self._use_rope = self.sliding_window is not None or getattr(original_attention, "force_rope", False)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, rope={self._use_rope}"

    def _apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._use_rope:
            return q, k
        return _apply_interleaved_rope_fp32(q, cos, sin), _apply_interleaved_rope_fp32(k, cos, sin)


class Glm4MoeLiteUlyssesAttention(MLAUlyssesAttention):
    """GLM4 MoE Lite: plain MLA."""

    HF_MODULE_NAMES = ("Glm4MoeLiteAttention",)


class GptOssUlyssesAttention(UlyssesAttention):
    """GPT-OSS: split-halves RoPE, and attention sinks the CP kernels drop, so they must be neutralized."""

    HF_MODULE_NAMES = ("GptOssAttention",)

    def __init__(self, original_attention: nn.Module, cp_group: dist.ProcessGroup, cp_size: int):
        super().__init__(original_attention, cp_group, cp_size)
        self._sinks_checked = False

    @property
    def sinks(self):
        """The wrapped module's sinks, read live: an alias registered here would take the sinks policy's rebind
        while ``original_attention`` kept, and the saves exported, the pretrained tensor."""
        return self.original_attention.sinks

    def _project_qkv(
        self, hidden_states: torch.Tensor, batch_size: int, local_seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._check_sinks_neutralized()
        return super()._project_qkv(hidden_states, batch_size, local_seq_len)

    def _check_sinks_neutralized(self) -> None:
        """Raise at the first forward unless the sinks are reset (None, or dtype-min): the CP kernels drop them, so
        live sinks would mis-normalize every softmax. Not at construction: the reset runs after the CP wrap."""
        if self._sinks_checked:
            return
        self._sinks_checked = True
        sinks = self.sinks
        if sinks is not None and not torch.all(sinks <= torch.finfo(sinks.dtype).min * 0.99):
            raise ValueError(
                "GPT-OSS under Context Parallelism requires reset sinks (reset_sinks: true): the CP "
                "attention kernels drop the sink column, so non-reset sinks give a wrong softmax "
                "normalization in every layer."
            )

    def _apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``cos``/``sin`` cover half the head dim."""
        return self._rotate_split_halves(q, cos, sin), self._rotate_split_halves(k, cos, sin)

    @staticmethod
    def _rotate_split_halves(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        first_half, second_half = x.chunk(2, dim=-1)
        first_ = first_half * cos - second_half * sin
        second_ = second_half * cos + first_half * sin
        return torch.cat((first_, second_), dim=-1)


class Mistral4UlyssesAttention(MLAUlyssesAttention):
    """Mistral4 (the text backbone of mistral3 VLMs): MLA with Q scaled by the position-dependent llama-4
    factor after RoPE, which ``Mistral4Attention.forward`` applies unconditionally."""

    HF_MODULE_NAMES = ("Mistral4Attention",)

    # ``Mistral4Config`` injects these only when ``rope_parameters`` is None, so a checkpoint shipping its own
    # dict can lack them.
    _SCALE_ROPE_KEYS = ("llama_4_scaling_beta", "original_max_position_embeddings")

    def __init__(self, original_attention: nn.Module, cp_group: dist.ProcessGroup, cp_size: int):
        super().__init__(original_attention, cp_group, cp_size)
        rope_params = getattr(self.config, "rope_parameters", None) or {}
        missing = [key for key in self._SCALE_ROPE_KEYS if rope_params.get(key) is None]
        if missing:
            raise ValueError(
                f"Mistral4 under Context Parallelism cannot resolve the llama-4 attention scale: "
                f"config.rope_parameters is missing {missing}. Add them to rope_parameters "
                f"(model_init_kwargs), or train this checkpoint without CP."
            )
        self._llama_4_scaling_beta = rope_params["llama_4_scaling_beta"]
        self._llama_4_original_max_pe = rope_params["original_max_position_embeddings"]

    def _apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MLA's RoPE, then Q scaled by the llama-4 factor of each position (upstream's ``[B, 1, S, 1]``, moved
        to ``[B, S, 1, 1]``)."""
        q, k = super()._apply_rotary(q, k, cos, sin, position_ids)
        scale = get_llama_4_attn_scale(position_ids, self._llama_4_scaling_beta, self._llama_4_original_max_pe)
        return q * scale.transpose(1, 2).to(q.dtype), k


class Qwen3MoeUlyssesAttention(UlyssesAttention):
    """Qwen3 / Qwen3-MoE / Qwen3-VL text: Q/K RMSNorm after the projection, rotate-half RoPE."""

    HF_MODULE_NAMES = ("Qwen3MoeAttention", "Qwen3Attention", "Qwen3VLTextAttention")

    def _project_qkv(
        self, hidden_states: torch.Tensor, batch_size: int, local_seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn = self.original_attention
        query_states = attn.q_norm(attn.q_proj(hidden_states).view(batch_size, local_seq_len, -1, self.head_dim))
        return query_states, *self._project_kv(hidden_states, batch_size, local_seq_len)

    def _project_kv(
        self, hidden_states: torch.Tensor, batch_size: int, local_seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn = self.original_attention
        shape = (batch_size, local_seq_len, -1, self.head_dim)
        return attn.k_norm(attn.k_proj(hidden_states).view(shape)), attn.v_proj(hidden_states).view(shape)


class Qwen3_5MoeUlyssesAttention(Qwen3MoeUlyssesAttention):
    """Qwen3.5 / Qwen3.6 full-attention layers: a double-width ``q_proj`` splits into Q and a sigmoid gate,
    computed on the local chunk and applied after the all-to-all. The family's GatedDeltaNet layers are refused by
    :func:`~halo.distributed.context_parallel.model_rules.reject_unsupported_ulysses_model`."""

    HF_MODULE_NAMES = ("Qwen3_5MoeAttention", "Qwen3_5Attention")

    def _project_qkv(
        self, hidden_states: torch.Tensor, batch_size: int, local_seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attn = self.original_attention
        q_out = attn.q_proj(hidden_states).view(batch_size, local_seq_len, -1, self.head_dim * 2)
        query_states, gate = torch.chunk(q_out, 2, dim=-1)
        gate = gate.reshape(batch_size, local_seq_len, -1)
        key_states, value_states = self._project_kv(hidden_states, batch_size, local_seq_len)
        return attn.q_norm(query_states), key_states, value_states, gate

    def _post_attention(self, attn_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return attn_output * torch.sigmoid(gate)


ULYSSES_ATTENTION_MAP = build_class_claim_map(UlyssesAttention, "HF attention class name")
