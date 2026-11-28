"""Inkling 2's patched forwards: the router at the run dtype, the compact relative logits
``inkling2_relative_flex`` reads (with no dense relative distances behind them), short convolutions whose taps stop at
the start of each packed document, and the block attention residual as plain matmuls."""

from __future__ import annotations

import functools
import math

import torch
import torch.nn.functional as F
import transformers.models.inkling2.modeling_inkling2 as inkling2_modeling

from halo.models.attention.inkling2_relative_flex import INKLING2_RELATIVE_FLEX
from halo.models.gradient_checkpointing import patch_layer_forward
from halo.models.segment_markers import positions_restart
from halo.models.upstream.upstream_guard import takes_legacy_cache_names, verify_upstream_parameters

# ``Inkling2TopKRouter``'s renormalization floor.
_TOPK_WEIGHT_NORM_EPS = 1e-20
# Kept for the parity tests: this module is the only one that replaces them.
UPSTREAM_GET_RELATIVE_DISTANCE = inkling2_modeling.get_relative_distance
UPSTREAM_RELATIVE_POSITION_BIAS_FORWARD = inkling2_modeling.Inkling2RelativePositionBias.forward
UPSTREAM_SHORT_CONVOLUTION_FORWARD = inkling2_modeling.Inkling2ShortConvolution.forward
UPSTREAM_DYNAMIC_SHORT_CONVOLUTION_FORWARD = inkling2_modeling.Inkling2DynamicShortConvolution.forward
UPSTREAM_BLOCK_ATTENTION_RESIDUAL_FORWARD = inkling2_modeling.Inkling2BlockAttentionResidual.forward

_DECODER_LAYER_CLASSES = (inkling2_modeling.Inkling2SlidingDecoderLayer, inkling2_modeling.Inkling2FullDecoderLayer)

# Set on the forward's ``position_ids``: per kernel size, the convolution taps that cross a document start, and
# whether a row restarts.
_CROSSING_TAPS_ATTR = "_inkling2_crossing_taps"
_RESTARTS_ATTR = "_inkling2_restarts"


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


def _block_attention_residual_forward(self, block_states, partial_block_states):
    """Each position's queries attend over its block states, as two matmuls and a softmax over the blocks. transformers
    5.23 handed that to SDPA, whose memory-efficient backward held a gradient tile of a whole query block per position
    (three times this peak at Inkling-2-Small's width); 5.24 routes it through the attention interface, which for this
    family runs the eager attention, the same two matmuls."""
    stacked_states = torch.cat([block_states, partial_block_states.unsqueeze(-2)], dim=-2).float()
    stacked_states = self.norm(stacked_states)
    hidden_size = stacked_states.shape[-1]
    queries = self.query.float().view(-1, hidden_size)
    # [..., blocks, queries], normalized over the blocks
    weights = (stacked_states @ queries.T * hidden_size**-0.5).softmax(dim=-2)
    return (weights.transpose(-1, -2) @ stacked_states).unbind(-2)


def _get_relative_distance(inputs_embeds, cache, is_sliding):
    """Without a cache, every layer's distance is ``arange(q)[:, None] - arange(q)[None, :]``, which the model builds
    as two dense int64 ``[q_len, q_len]`` matrices held through the decoder loop (16 GiB at 32,768 tokens) while
    ``inkling2_relative_flex`` reads only their shape. Here it is a zero-stride view of that shape, and the eager bias
    rebuilds the values for its own layer."""
    if cache is not None:
        return UPSTREAM_GET_RELATIVE_DISTANCE(inputs_embeds, cache, is_sliding)
    q_len = inputs_embeds.shape[1]
    return torch.zeros((), dtype=torch.long, device=inputs_embeds.device).expand(q_len, q_len)


def _relative_position_bias_forward(self, hidden_states, relative_distance, position_ids):
    """Under ``inkling2_relative_flex``, the ``[batch, heads, q_len, rel_extent]`` logits in place of the dense
    ``[batch, heads, q_len, kv_len]`` bias the eager path adds."""
    # A KV-cache decode step has fewer queries than keys and runs the eager attention.
    q_len, kv_len = relative_distance.shape[-2:]
    if not getattr(self, "emits_compact_logits", False) or q_len != kv_len:
        if relative_distance.stride() == (0, 0):  # _get_relative_distance's stand-in for an uncached forward
            positions = torch.arange(q_len, device=relative_distance.device)
            relative_distance = positions[:, None] - positions[None, :]
        return UPSTREAM_RELATIVE_POSITION_BIAS_FORWARD(self, hidden_states, relative_distance, position_ids)
    relative_states = self.linear(hidden_states).view(*hidden_states.shape[:-1], self.num_heads, -1)
    relative_states = self.norm(relative_states, position_ids)
    # relative_states: [batch, q_len, num_heads, d_rel] -> [batch, num_heads, q_len, rel_extent]
    return (relative_states @ self.proj).transpose(1, 2)


def _decoder_layer_forward(upstream):
    """Sets what the layer's patched modules read for this call: whether the relative logits stay compact, and on each
    short convolution the ``position_ids`` of a full-sequence forward (None under a KV cache). Set per layer call, so
    a gradient-checkpoint recompute, which calls the layer again, sets it too."""

    @takes_legacy_cache_names
    @functools.wraps(upstream)
    def forward(self, *args, **kwargs):
        attention = self.self_attn
        attention.relative_position_bias.emits_compact_logits = (
            attention.config._attn_implementation == INKLING2_RELATIVE_FLEX
        )
        positions = kwargs.get("position_ids") if kwargs.get("cache") is None else None
        convolutions = (
            attention.k_proj.sconv,
            attention.v_proj.sconv,
            self.post_attention_convolution,
            self.post_mlp_convolution,
        )
        for convolution in convolutions:
            convolution.document_positions = positions
        try:
            return upstream(self, *args, **kwargs)
        finally:
            for convolution in convolutions:
                convolution.document_positions = None

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


def _has_restart(position_ids: torch.Tensor) -> bool:
    restarts = getattr(position_ids, _RESTARTS_ATTR, None)
    if restarts is None:
        restarts = positions_restart(position_ids)
        setattr(position_ids, _RESTARTS_ATTR, restarts)
    return restarts


def _layer_crossing_taps(convolution, hidden_states: torch.Tensor) -> tuple[tuple[int, torch.Tensor], ...]:
    positions = getattr(convolution, "document_positions", None)
    if positions is None or positions.dim() != 2:
        return ()
    if positions.shape[0] != 1:
        if _has_restart(positions):
            raise ValueError(
                f"position_ids of shape {tuple(positions.shape)} restart inside a row: the short convolutions stop "
                "their taps at document starts on one [1, N] row only; packed documents run as one [1, N] row."
            )
        return ()
    taps = crossing_taps(positions, convolution.conv_kernel_size)
    if taps and hidden_states.shape[0] != 1:
        raise ValueError(
            f"A batch of {hidden_states.shape[0]} rows shares one position_ids row that restarts; packed documents "
            "run as one [1, N] row."
        )
    return taps


@takes_legacy_cache_names
def _short_convolution_forward(
    self, hidden_states, dynamic_input=None, cache=None, conv_mask=None, cache_offset=0, **kwargs
):
    """On a packed row, the upstream convolution less the taps that reach into the previous document; a row holding
    one document runs the upstream forward."""
    taps = () if cache is not None else _layer_crossing_taps(self, hidden_states)
    if not taps:
        return UPSTREAM_SHORT_CONVOLUTION_FORWARD(
            self, hidden_states, dynamic_input, cache=cache, conv_mask=conv_mask, cache_offset=cache_offset, **kwargs
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


@takes_legacy_cache_names
def _dynamic_short_convolution_forward(
    self, hidden_states, dynamic_input, cache=None, conv_mask=None, cache_offset=0, **kwargs
):
    """On a packed row, the upstream convolution with each tap's weight zeroed where it reaches into the previous
    document; a row holding one document runs the upstream forward."""
    taps = () if cache is not None else _layer_crossing_taps(self, hidden_states)
    if not taps:
        return UPSTREAM_DYNAMIC_SHORT_CONVOLUTION_FORWARD(
            self, hidden_states, dynamic_input, cache=cache, conv_mask=conv_mask, cache_offset=cache_offset, **kwargs
        )
    kernel_size = self.conv_kernel_size
    input_dtype = hidden_states.dtype
    batch_size, seq_len, hidden_size = hidden_states.shape
    dynamic_weights = self.weight_proj(dynamic_input).float()
    dynamic_weights = dynamic_weights.view(batch_size, seq_len, self.num_heads, 1, kernel_size)
    weights = self.static_weight.float().T.view(self.num_heads, -1, kernel_size) + dynamic_weights
    # The tap at lag l reads the token l back, which is in this token's document only while l <= its position.
    positions = self.document_positions[0]
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
        UPSTREAM_GET_RELATIVE_DISTANCE, ("inputs_embeds", "cache", "is_sliding"), "_get_relative_distance"
    )
    verify_upstream_parameters(
        UPSTREAM_RELATIVE_POSITION_BIAS_FORWARD,
        ("self", "hidden_states", "relative_distance", "position_ids"),
        "_relative_position_bias_forward",
    )
    convolution_parameters = ("self", "hidden_states", "dynamic_input", "cache", "conv_mask", "cache_offset", "kwargs")
    verify_upstream_parameters(
        UPSTREAM_SHORT_CONVOLUTION_FORWARD, convolution_parameters, "_short_convolution_forward"
    )
    verify_upstream_parameters(
        UPSTREAM_DYNAMIC_SHORT_CONVOLUTION_FORWARD, convolution_parameters, "_dynamic_short_convolution_forward"
    )
    verify_upstream_parameters(
        UPSTREAM_BLOCK_ATTENTION_RESIDUAL_FORWARD,
        ("self", "block_states", "partial_block_states"),
        "_block_attention_residual_forward",
    )
    patch_layer_forward(router, _router_forward)
    inkling2_modeling.get_relative_distance = _get_relative_distance
    patch_layer_forward(inkling2_modeling.Inkling2BlockAttentionResidual, _block_attention_residual_forward)
    patch_layer_forward(inkling2_modeling.Inkling2RelativePositionBias, _relative_position_bias_forward)
    patch_layer_forward(inkling2_modeling.Inkling2ShortConvolution, _short_convolution_forward)
    patch_layer_forward(inkling2_modeling.Inkling2DynamicShortConvolution, _dynamic_short_convolution_forward)
    for cls in _DECODER_LAYER_CLASSES:
        patch_layer_forward(cls, _decoder_layer_forward(cls.forward))
