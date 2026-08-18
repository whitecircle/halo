"""Ulysses attention wrapper for GPT-OSS models.

Model-specific features handled here:

- Attention sinks added inside the eager softmax.
- Split-concat RoPE pattern (rather than the rotate-half pattern Qwen3 uses).
"""

from __future__ import annotations

import torch

from src.distributed.context_parallel.base_layer import UlyssesAttentionBase


class GptOssUlyssesAttention(UlyssesAttentionBase):
    """Ulysses-wrapped attention for GPT-OSS models."""

    HF_MODULE_NAMES = ("GptOssAttention",)

    # Set on the instance by the first forward's check; the class-level default needs no ctor.
    _sinks_checked = False

    @property
    def sinks(self):
        """Live view of the wrapped module's sinks, not a second binding.

        An ``__init__``-time alias would register the parameter on this wrapper, so the FA2 sinks
        policy's rebind would land here while ``original_attention`` kept the pretrained tensor —
        which the save paths then export under the clean HF key. Read-only; the policy rebinds
        through the wrapped module.
        """
        return self.original_attention.sinks

    def _check_sinks_neutralized(self) -> None:
        """Raise at the first forward if the sinks were not neutralized.

        The CP flash paths never pass sinks to the kernel, so frozen pretrained sinks
        (``reset_sinks: false``) would give a wrong softmax normalization in every layer; reset sinks
        (None, or a dtype-min fill contributing ``exp(min) == 0``) match eager. Checked at forward
        rather than construction, because ``_reset_gpt_oss_sinks`` runs after the CP wrap.
        """
        if self._sinks_checked:
            return
        self._sinks_checked = True
        sinks = self.sinks
        if sinks is None:
            return
        if not torch.all(sinks <= torch.finfo(sinks.dtype).min * 0.99):
            raise ValueError(
                "GPT-OSS under Context Parallelism requires reset sinks (reset_sinks: true): the CP "
                "attention kernels drop the sink column, so non-reset sinks give a wrong softmax "
                "normalization in every layer."
            )

    def _project_qkv(
        self, hidden_states: torch.Tensor, batch_size: int, local_seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The plain projection, gated on the first-forward sinks check."""
        self._check_sinks_neutralized()
        return self._project_qkv_plain(hidden_states, batch_size, local_seq_len)

    def _apply_rotary_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-copy RoPE in ``[B, S, H, D]`` layout. cos/sin pre-shaped to ``[B, S, 1, D/2]``."""
        return self._rotary_emb(q, cos, sin), self._rotary_emb(k, cos, sin)

    def _rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Split-concat rotary; layout-agnostic (operates only on the last dim)."""
        first_half, second_half = x.chunk(2, dim=-1)
        first_ = first_half * cos - second_half * sin
        second_ = second_half * cos + first_half * sin
        return torch.cat((first_, second_), dim=-1)
