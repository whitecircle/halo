"""SDPA's mem-efficient kernel over a boolean mask, one tile of queries over the keys it selected, with a closed-form
backward, and the local ``sdpa`` override that runs GLM-5 Next's DSA layers on it: their ``[T, T]`` masks keep SDPA on that
kernel, whose own backward takes most of a long step."""

from typing import NamedTuple

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface
from transformers.models.glm5_next import modeling_glm5_next

from halo.env import env_flag
from halo.kernels.fp32_matmul import GEMM_ALIGNMENT, bmm_fp32, to_autocast_dtype, wide_dtype
from halo.kernels.masked_softmax import masked_softmax_backward
from halo.models.attention.selection import effective_attn_implementation

# fp32 scores a tile's backward holds (heads x queries x keys): 2 GiB, alongside g @ v^T and 16-bit copies of what the
# softmax backward makes of them, 6 GiB, on top of the fp32 key and value gradients.
SCORES_PER_TILE = 1 << 29

_KERNEL_DEVICES = ("cuda",)
_KERNEL_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
# The kernel takes a head width and a 16-bit bias's row stride at multiples of this: off the first SDPA leaves it, and
# its ``pad_bias`` arranges the second.
_KERNEL_ALIGNMENT = 8
# What the stock interface reads off a call beyond the arguments named here.
_SDPA_ONLY_KWARGS = ("position_bias", "cache")


class _Tile(NamedTuple):
    """``keys`` indexes the key axis of the tile's row and is all that row is ever read or written through."""

    row: int
    queries: slice
    keys: slice


def _tile_key_count(allowed: torch.Tensor) -> torch.Tensor:
    """A tile's ``[Q, K]`` mask scores through the last key any of its queries selected, read off the mask, so no shape
    of mask can lose a key, then up to a GEMM-aligned count, or the row's end, whose few products stay unaligned."""
    positions = torch.arange(1, allowed.shape[-1] + 1, device=allowed.device)
    last = (allowed.any(dim=0) * positions).amax()
    aligned = (last + GEMM_ALIGNMENT - 1) // GEMM_ALIGNMENT * GEMM_ALIGNMENT
    return aligned.clamp(GEMM_ALIGNMENT, allowed.shape[-1])


def _plan_tiles(allowed: torch.Tensor, heads: int) -> list[_Tile]:
    """Per batch row, so a row's views stay strided as they came and each row crops to its own keys. A GEMM-aligned
    number of queries, the inner dimension of ``d_k`` and ``d_v``. One host read."""
    batch, _, seq, _ = allowed.shape
    step = max(GEMM_ALIGNMENT, SCORES_PER_TILE // (heads * seq) // GEMM_ALIGNMENT * GEMM_ALIGNMENT)
    spans = [(row, slice(start, min(start + step, seq))) for row in range(batch) for start in range(0, seq, step)]
    counts = torch.stack([_tile_key_count(allowed[row, 0, queries]) for row, queries in spans]).tolist()
    return [_Tile(row, queries, slice(count)) for (row, queries), count in zip(spans, counts, strict=True)]


def _tile_bias(allowed: torch.Tensor, heads: int, dtype: torch.dtype) -> torch.Tensor:
    """The tile's ``[Q, K]`` mask as the kernel reads a bias: additive, per head, on an aligned row stride."""
    queries, keys = allowed.shape
    bias = torch.full((queries, keys + -keys % _KERNEL_ALIGNMENT), float("-inf"), dtype=dtype, device=allowed.device)
    bias[:, :keys].masked_fill_(allowed, 0.0)
    return bias[None, None, :, :keys].expand(1, heads, -1, -1)


def _tile_softmax_backward(q, k, v, upstream, out, allowed, lse, scale, needs_scores_grad, needs_value_grad):
    """Its own frame, as the fp32 products outlive the launch otherwise, beside the matmuls that read its results."""
    upstream_scores, delta = None, None
    if needs_scores_grad:
        upstream_scores = bmm_fp32(upstream, v.transpose(1, 2))
        delta = (upstream.to(lse.dtype) * out.to(lse.dtype)).sum(-1)
    return masked_softmax_backward(
        bmm_fp32(q, k.transpose(1, 2)), allowed, lse, scale, upstream_scores, delta, needs_value_grad, q.dtype
    )


def _backward_tile(saved, tile: _Tile, grad: torch.Tensor, scale: float, d_query, d_key, d_value) -> None:
    """Only the products an input requiring grad reads. ``d_key`` and ``d_value`` sum over tiles in fp32."""
    query, key, value, allowed, out, lse = saved
    row, queries, keys = tile
    q, k = query[row, :, queries], key[row, :, keys]
    upstream = grad[row, queries].transpose(0, 1)
    probs, d_scores = _tile_softmax_backward(
        q,
        k,
        value[row, :, keys],
        upstream,
        out[row, queries].transpose(0, 1),
        allowed[row, 0, queries, keys],
        lse[row, :, queries],
        scale,
        d_query is not None or d_key is not None,
        d_value is not None,
    )
    if d_query is not None:
        d_query[row, :, queries] = bmm_fp32(d_scores, k).to(query.dtype)
    if d_key is not None:
        bmm_fp32(d_scores.transpose(1, 2), q, out=d_key[row, :, keys])
    if d_value is not None:
        bmm_fp32(probs.transpose(1, 2), upstream, out=d_value[row, :, keys])


class _MaskedMatmulAttention(torch.autograd.Function):
    """``softmax(q @ k^T * scale, mask) @ v`` per tile of queries, applied outside autocast. The forward is the stock
    kernel, grad off in here, so it records no backward of its own; a query with no key gets a zero output. Saves no
    score tensor: the backward rebuilds each tile's from ``q``, ``k`` and the kernel's log-sum-exp, in fp32, with its
    products at the inputs' precision."""

    @staticmethod
    def forward(ctx, query, key, value, allowed, scale):
        batch, heads, seq, _ = query.shape
        tiles = _plan_tiles(allowed, heads)
        records_grad = any(ctx.needs_input_grad[:3])
        out = torch.empty(batch, seq, heads, value.shape[-1], dtype=value.dtype, device=value.device)
        lse = (
            torch.empty(batch, heads, seq, dtype=wide_dtype(query.dtype), device=query.device)
            if records_grad
            else None
        )
        for row, queries, keys in tiles:
            tile_out, tile_lse, *_ = torch._scaled_dot_product_efficient_attention(
                query[row : row + 1, :, queries],
                key[row : row + 1, :, keys],
                value[row : row + 1, :, keys],
                _tile_bias(allowed[row, 0, queries, keys], heads, query.dtype),
                records_grad,
                scale=scale,
            )
            out[row, queries] = tile_out[0].transpose(0, 1)
            if records_grad:
                # Padded on the query axis.
                lse[row, :, queries] = tile_lse[0, :, : queries.stop - queries.start]
        if records_grad:
            ctx.save_for_backward(query, key, value, allowed, out, lse)
            ctx.tiles, ctx.scale = tiles, scale
        return out

    @staticmethod
    def backward(ctx, grad):
        saved = ctx.saved_tensors
        query, key, value, *_ = saved
        needs_query, needs_key, needs_value = ctx.needs_input_grad[:3]
        d_query = torch.empty_like(query) if needs_query else None
        d_key = torch.zeros(key.shape, dtype=wide_dtype(key.dtype), device=key.device) if needs_key else None
        d_value = torch.zeros(value.shape, dtype=wide_dtype(value.dtype), device=value.device) if needs_value else None
        with torch.autocast(grad.device.type, enabled=False):
            for tile in ctx.tiles:
                _backward_tile(saved, tile, grad, ctx.scale, d_query, d_key, d_value)
        return (
            d_query,
            None if d_key is None else d_key.to(key.dtype),
            None if d_value is None else d_value.to(value.dtype),
            None,
            None,
        )


def masked_matmul(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, allowed: torch.Tensor, scale: float
) -> torch.Tensor:
    """``[B, H, S, D]`` inputs of one dtype and a boolean ``[B, 1, S, S]`` mask. Returns ``[B, S, H, D]``."""
    with torch.autocast(query.device.type, enabled=False):
        return _MaskedMatmulAttention.apply(query, key, value, allowed, scale)


def _kernel_serves(module, query, key, value, attention_mask, dropout, kwargs) -> bool:
    """A plain tensor by exact type, as a DTensor is a subclass and this layer imports no ``torch.distributed``."""
    return (
        attention_mask is not None
        and attention_mask.dtype == torch.bool
        and all(
            type(t) is torch.Tensor and t.device.type in _KERNEL_DEVICES and t.dtype in _KERNEL_DTYPES
            for t in (query, key, value)
        )
        and not dropout
        and not kwargs.get("output_attentions", False)
        and all(kwargs.get(name) is None for name in _SDPA_ONLY_KWARGS)
        and getattr(module, "num_key_value_groups", 1) == 1
        and query.shape == key.shape == value.shape
        and query.shape[-1] % _KERNEL_ALIGNMENT == 0
        and attention_mask.shape == (query.shape[0], 1, query.shape[-2], key.shape[-2])
    )


def masked_matmul_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Every call the kernel does not serve goes to the library's ``sdpa`` unchanged."""
    if _kernel_serves(module, query, key, value, attention_mask, dropout, kwargs):
        q, k, v = to_autocast_dtype(query, key, value)
        if q.dtype == k.dtype == v.dtype:
            scale = q.shape[-1] ** -0.5 if scaling is None else scaling
            return masked_matmul(q, k, v, attention_mask, scale), None
    return ALL_ATTENTION_FUNCTIONS["sdpa"](
        module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs
    )


_GLM5_NEXT_ATTENTION_FUNCTIONS = AttentionInterface()
_GLM5_NEXT_ATTENTION_FUNCTIONS["sdpa"] = masked_matmul_attention


def dsa_matmul_enabled() -> bool:
    return env_flag("HALO_DSA_MATMUL", True)


def install_glm5_next_masked_matmul() -> None:
    """The local override transformers documents for a modeling file, so ``config._attn_implementation`` stays ``sdpa``
    and the stock boolean mask, selection, warm-up and layout code keep reading it. Idempotent; a module already behind
    tree attention's view of the library registry stays there and runs stock SDPA."""
    if modeling_glm5_next.ALL_ATTENTION_FUNCTIONS is ALL_ATTENTION_FUNCTIONS:
        modeling_glm5_next.ALL_ATTENTION_FUNCTIONS = _GLM5_NEXT_ATTENTION_FUNCTIONS


def warm_masked_matmul_kernels(
    model: torch.nn.Module, *, dtype: torch.dtype, device: torch.device | None = None
) -> None:
    """Compile the backward's launch in each form a trained or frozen input reaches, so no rank compiles one mid-step.
    Rank-local; the caller fences it."""
    if (
        modeling_glm5_next.ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None) is not masked_matmul_attention
        or effective_attn_implementation(model.config) != "sdpa"
        or not any(isinstance(module, modeling_glm5_next.Glm5NextTextAttention) for module in model.modules())
    ):
        return
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    side = GEMM_ALIGNMENT
    scores = torch.zeros(1, side, side, device=device, dtype=torch.float32)
    allowed = torch.ones(side, side, device=device, dtype=torch.bool)
    lse = torch.zeros(1, side, device=device, dtype=torch.float32)
    masked_softmax_backward(scores, allowed, lse, 1.0, scores, lse, True, dtype)
    masked_softmax_backward(scores, allowed, lse, 1.0, scores, lse, False, dtype)
    masked_softmax_backward(scores, allowed, lse, 1.0, None, None, True, dtype)
