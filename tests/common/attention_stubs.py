"""A stand-in decoder attention module for kernel tests calling an attention function directly. ``sdpa_flex_sliding``
hands FlexAttention only the call a module declares, so a bare ``nn.Module`` would send every call to SDPA."""

from __future__ import annotations

import torch


def causal_attention(
    heads: int = 4,
    kv_heads: int = 2,
    head_dim: int = 64,
    scaling: float | None = 1.0,
    sliding_window: int | None = None,
) -> torch.nn.Module:
    """A causal attention module declaring heads, width, ``scaling`` and ``sliding_window`` (None on a full layer)."""
    module = torch.nn.Module()
    module.is_causal = True
    module.num_heads = heads
    module.num_key_value_groups = heads // kv_heads
    module.head_dim = head_dim
    module.scaling = scaling
    module.sliding_window = sliding_window
    return module
