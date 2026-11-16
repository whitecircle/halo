"""Inkling 2's patched forwards: the router at the run dtype, the compact relative logits
``inkling2_relative_flex`` reads, and short convolutions whose taps stop at the start of each packed document."""

from __future__ import annotations

import functools
import math
from contextvars import ContextVar

import torch
import torch.nn.functional as F
import transformers.models.inkling2.modeling_inkling2 as inkling2_modeling

from halo.models.attention.inkling2_relative_flex import INKLING2_RELATIVE_FLEX
from halo.models.upstream.upstream_guard import verify_upstream_parameters

# ``Inkling2TopKRouter``'s renormalization floor.
_TOPK_WEIGHT_NORM_EPS = 1e-20
# Kept for the parity tests: this module is the only one that replaces them.
UPSTREAM_ROUTER_FORWARD = inkling2_modeling.Inkling2TopKRouter.forward
UPSTREAM_RELATIVE_LOGITS_FORWARD = inkling2_modeling.Inkling2RelativeLogits.forward
UPSTREAM_SHORT_CONVOLUTION_FORWARD = inkling2_modeling.Inkling2ShortConvolution.forward
UPSTREAM_DYNAMIC_SHORT_CONVOLUTION_FORWARD = inkling2_modeling.Inkling2DynamicShortConvolution.forward

_ATTENTION_CLASSES = (inkling2_modeling.Inkling2SlidingAttention, inkling2_modeling.Inkling2FullAttention)
_DECODER_LAYER_CLASSES = (inkling2_modeling.Inkling2SlidingDecoderLayer, inkling2_modeling.Inkling2FullDecoderLayer)

# The ``position_ids`` of the decoder layer running a full-sequence forward (no KV cache). Set per layer call, so a
# gradient-checkpoint recompute, which calls the layer again, sees it too.
_LAYER_POSITIONS: ContextVar[torch.Tensor | None] = ContextVar("inkling2_layer_positions", default=None)
# Set on the forward's ``position_ids``: per kernel size, the convolution taps that cross a document start.
_CROSSING_TAPS_ATTR = "_inkling2_crossing_taps"


def _router_forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The upstream router matmuls fp32 hidden states against a weight transformers pins to fp32; the toolkit trains
    that weight at the run dtype (FSDP2 takes one dtype per shard group), so it is upcast at use."""
    flat = hidden_states.float().reshape(-1, self.hidden_dim)
    logits = F.linear(flat, self.weight.float())
    scores = self.score_fn(logits)
    indices = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + _TOPK_WEIGHT_NORM_EPS)
    return logits, weights * self.routed_scaling_factor, indices


def _relative_logits_forward(self, relative_states, query_positions, key_positions):
    """Under ``inkling2_relative_flex``, the ``[batch, heads, q_len, rel_extent]`` logits in place of the dense
    ``[batch, heads, q_len, kv_len]`` bias the eager path adds."""
    if not getattr(self, "emits_compact_logits", False):
        return UPSTREAM_RELATIVE_LOGITS_FORWARD(self, relative_states, query_positions, key_positions)
    # relative_states: [batch, q_len, num_heads, d_rel] -> [batch, num_heads, q_len, rel_extent]
    return (relative_states @ self.proj).transpose(1, 2)


def _attention_forward(upstream):
    @functools.wraps(upstream)
    def forward(self, *args, **kwargs):
        self.rel_logits_proj.emits_compact_logits = self.config._attn_implementation == INKLING2_RELATIVE_FLEX
        return upstream(self, *args, **kwargs)

    return forward


def _decoder_layer_forward(upstream):
    @functools.wraps(upstream)
    def forward(self, *args, **kwargs):
        positions = kwargs.get("position_ids") if kwargs.get("past_key_values") is None else None
        token = _LAYER_POSITIONS.set(positions)
        try:
            return upstream(self, *args, **kwargs)
        finally:
            _LAYER_POSITIONS.reset(token)

    return forward


def crossing_taps(position_ids: torch.Tensor, kernel_size: int) -> tuple[tuple[int, torch.Tensor], ...]:
    """Per lag, the tokens of a one-row ``position_ids`` whose tap at that lag reaches into the previous document:
    ``position < lag <= token index``. Empty on a row holding one document."""
    cache = getattr(position_ids, _CROSSING_TAPS_ATTR, None)
    if cache is None:
        cache = {}
        setattr(position_ids, _CROSSING_TAPS_ATTR, cache)
    if kernel_size not in cache:
        positions = position_ids[0]
        index = torch.arange(positions.shape[0], device=positions.device)
        taps = []
        for lag in range(1, kernel_size):
            tokens = torch.nonzero((positions < lag) & (index >= lag)).squeeze(-1)
            if tokens.numel():
                taps.append((lag, tokens))
        cache[kernel_size] = tuple(taps)
    return cache[kernel_size]


def _layer_crossing_taps(hidden_states: torch.Tensor, kernel_size: int) -> tuple[tuple[int, torch.Tensor], ...]:
    positions = _LAYER_POSITIONS.get()
    if positions is None or positions.dim() != 2 or positions.shape[0] != 1:
        return ()
    taps = crossing_taps(positions, kernel_size)
    if taps and hidden_states.shape[0] != 1:
        raise ValueError(
            f"A batch of {hidden_states.shape[0]} rows shares one position_ids row that restarts; packed documents "
            "run as one [1, N] row."
        )
    return taps


def _short_convolution_forward(self, hidden_states, past_key_values=None, conv_mask=None, **kwargs):
    """On a packed row, the upstream convolution less the taps that reach into the previous document; a row holding
    one document runs the upstream forward."""
    taps = () if past_key_values is not None else _layer_crossing_taps(hidden_states, self.conv_kernel_size)
    if not taps:
        return UPSTREAM_SHORT_CONVOLUTION_FORWARD(
            self, hidden_states, past_key_values=past_key_values, conv_mask=conv_mask, **kwargs
        )
    input_dtype = hidden_states.dtype
    residual = hidden_states.float()
    context = inkling2_modeling.apply_mask_to_padding_states(residual, conv_mask).transpose(1, 2)
    weight = self.conv1d.weight.squeeze(1)
    out = inkling2_modeling.causal_conv1d_fn(context, weight, self.conv1d.bias)
    # Output t sums weight[:, K - 1 - lag] * x[t - lag]; remove the terms whose source token is another document's.
    for lag, tokens in taps:
        leaked = weight[:, self.conv_kernel_size - 1 - lag, None] * context[0][:, tokens - lag]
        out = out.index_add(-1, tokens, -leaked.to(out.dtype).unsqueeze(0))
    out = (out.transpose(1, 2) + residual) / math.sqrt(self.conv_kernel_size + 1)
    return out.to(input_dtype)


def _dynamic_short_convolution_forward(
    self, hidden_states, dynamic_input, past_key_values=None, conv_mask=None, **kwargs
):
    """On a packed row, the upstream convolution with each tap's weight zeroed where it reaches into the previous
    document; a row holding one document runs the upstream forward."""
    taps = () if past_key_values is not None else _layer_crossing_taps(hidden_states, self.conv_kernel_size)
    if not taps:
        return UPSTREAM_DYNAMIC_SHORT_CONVOLUTION_FORWARD(
            self, hidden_states, dynamic_input, past_key_values=past_key_values, conv_mask=conv_mask, **kwargs
        )
    kernel_size = self.conv_kernel_size
    input_dtype = hidden_states.dtype
    batch_size, seq_len, hidden_size = hidden_states.shape
    dynamic_weights = self.weight_proj(dynamic_input).float()
    dynamic_weights = dynamic_weights.view(batch_size, seq_len, self.num_heads, 1, kernel_size)
    weights = self.static_weight.float().T.view(self.num_heads, -1, kernel_size) + dynamic_weights
    # The tap at lag l reads the token l back, which is in this token's document only while l <= its position.
    positions = _LAYER_POSITIONS.get()[0]
    reach = positions[:, None] >= torch.arange(kernel_size, device=positions.device)
    weights = weights * reach[None, :, None, None, :]

    residual = hidden_states.float()
    context = inkling2_modeling.apply_mask_to_padding_states(residual, conv_mask).transpose(1, 2)
    context = F.pad(context, (kernel_size - 1, 0))
    windows = context.unfold(-1, kernel_size, 1).flip(-1).transpose(1, 2)
    windows = windows.reshape(batch_size, seq_len, self.num_heads, -1, kernel_size)
    out = (windows * weights).sum(-1).reshape(batch_size, seq_len, hidden_size)
    return ((out + residual) / math.sqrt(2 * kernel_size + 1)).to(input_dtype)


def apply_inkling2_patches() -> None:
    router = inkling2_modeling.Inkling2TopKRouter
    if router.forward is _router_forward:
        return
    verify_upstream_parameters(router.forward, ("self", "hidden_states"), "_router_forward")
    verify_upstream_parameters(
        UPSTREAM_RELATIVE_LOGITS_FORWARD,
        ("self", "relative_states", "query_positions", "key_positions"),
        "_relative_logits_forward",
    )
    verify_upstream_parameters(
        UPSTREAM_SHORT_CONVOLUTION_FORWARD,
        ("self", "hidden_states", "past_key_values", "conv_mask", "kwargs"),
        "_short_convolution_forward",
    )
    verify_upstream_parameters(
        UPSTREAM_DYNAMIC_SHORT_CONVOLUTION_FORWARD,
        ("self", "hidden_states", "dynamic_input", "past_key_values", "conv_mask", "kwargs"),
        "_dynamic_short_convolution_forward",
    )
    router.forward = _router_forward
    inkling2_modeling.Inkling2RelativeLogits.forward = _relative_logits_forward
    inkling2_modeling.Inkling2ShortConvolution.forward = _short_convolution_forward
    inkling2_modeling.Inkling2DynamicShortConvolution.forward = _dynamic_short_convolution_forward
    for cls in _ATTENTION_CLASSES:
        cls.forward = _attention_forward(cls.forward)
    for cls in _DECODER_LAYER_CLASSES:
        cls.forward = _decoder_layer_forward(cls.forward)
