"""Toolkit patches for the upstream NemotronH modeling, applied at model load.

``NemotronHBlock.forward`` hands its Mamba-2 mixer no forward kwargs, so the ``seq_idx`` a packed or
padding-free row carries never reaches the conv and scan kernels and their state crosses every
document boundary while attention stays isolated. The mixer itself threads kwargs into both kernels.
The Mamba-2 scan runs from a Hub build (:mod:`src.models.patches.kernel_dispatch`), resolved here so
a missing build fails at load.
"""

from __future__ import annotations

import functools

import transformers.models.nemotron_h.modeling_nemotron_h as nemotron_h_modeling

from src.models.patches.kernel_dispatch import resolve_hub_kernels

_PATCHED_FLAG = "_halo_mixer_segment_markers"


def apply_nemotron_h_patches() -> None:
    patch_nemotron_h_mixer_segment_markers()
    resolve_hub_kernels(nemotron_h_modeling.__name__)


def patch_nemotron_h_mixer_segment_markers() -> None:
    """Forward ``seq_idx`` from the block to its Mamba-2 mixer. Idempotent."""
    block_cls = nemotron_h_modeling.NemotronHBlock
    if getattr(block_cls.forward, _PATCHED_FLAG, False):
        return
    original_forward = block_cls.forward

    @functools.wraps(original_forward)
    def forward(self, hidden_states, past_key_values=None, attention_mask=None, position_ids=None, **kwargs):
        if self.block_type != "linear_attention" or kwargs.get("seq_idx") is None:
            return original_forward(
                self, hidden_states, past_key_values, attention_mask, position_ids=position_ids, **kwargs
            )
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        hidden_states = self.mixer(
            hidden_states, cache_params=past_key_values, attention_mask=attention_mask, seq_idx=kwargs["seq_idx"]
        )
        return residual + hidden_states

    setattr(forward, _PATCHED_FLAG, True)
    block_cls.forward = forward
