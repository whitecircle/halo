"""A stand-in decoder attention module for kernel tests that call an attention function directly.

``sdpa_flex_sliding`` hands FlexAttention only the call a module declares (its heads, head width, window and
``scaling``, read off the module as off a real attention layer, the call the warm-up compiles for it), and SDPA's
KV repeat reads its GQA group count, so a bare ``nn.Module`` would send every call to SDPA and a test counting
FlexAttention calls would test nothing.
"""

from __future__ import annotations

import torch


def causal_attention(
    heads: int = 4,
    kv_heads: int = 2,
    head_dim: int = 64,
    scaling: float | None = 1.0,
    sliding_window: int | None = None,
) -> torch.nn.Module:
    """A causal attention module declaring ``heads`` query heads over ``kv_heads``, ``head_dim`` wide, at
    ``scaling``, and the ``sliding_window`` it passes (``None`` on a full layer), kept on the module as the
    sliding-window families keep it."""
    module = torch.nn.Module()
    module.is_causal = True
    module.num_heads = heads
    module.num_key_value_groups = heads // kv_heads
    module.head_dim = head_dim
    module.scaling = scaling
    module.sliding_window = sliding_window
    return module
