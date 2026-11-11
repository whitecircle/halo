"""Block-sparse causal attention in Triton, CUDA only, for a query that attends a few compression blocks of keys plus the
keys after its last complete block. One program serves one query token and K/V head, so a launch compiles once for every
row length. The backward accumulates dK and dV with fp32 atomics, which are not deterministic."""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from halo.log import warn_once

logger = logging.getLogger(__name__)

# ``tl.dot`` takes no operand under this many rows or lanes.
MIN_DOT_EXTENT = 16
# The fp32 elements of the ``[group tile, head_dim]`` accumulator one program holds in registers across its warps.
MAX_ACCUMULATOR_ELEMENTS = 16384
SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)
# The row count and the top-k width change per call; specialized, each divisibility class of them would compile a binary.
# Strides and pointers stay specialized: ``_kernel_layout`` hands every launch the same 16-aligned layout, which also
# lets the loads vectorize.
_RUNTIME_ARGUMENTS = ["n_tokens", "block_topk"]


class LaunchConfig(NamedTuple):
    keys: int
    warps: int
    stages: int


class DirectionConfigs(NamedTuple):
    forward: LaunchConfig
    backward: LaunchConfig


# Tunable per SM major; any other takes ``DEFAULT_LAUNCH_CONFIG``.
LAUNCH_CONFIGS = {10: DirectionConfigs(LaunchConfig(64, 4, 2), LaunchConfig(32, 4, 1))}
DEFAULT_LAUNCH_CONFIG = DirectionConfigs(LaunchConfig(32, 4, 1), LaunchConfig(32, 4, 1))
_NO_SOFTMAX = tl.constexpr(-1.0e20)

# (dtype, device, head_dim, group tile, ratio, block-index dtype) of every binary compiled or warmed.
_compiled_variants: set[tuple] = set()


def compiled_variants() -> frozenset[tuple]:
    return frozenset(_compiled_variants)


def launch_configs(device: torch.device) -> DirectionConfigs:
    return LAUNCH_CONFIGS.get(torch.cuda.get_device_capability(device)[0], DEFAULT_LAUNCH_CONFIG)


def group_tile(groups: int) -> int:
    """The query rows one program holds: its K/V head's group, rounded up to a ``tl.dot`` tile."""
    return max(MIN_DOT_EXTENT, 1 << (groups - 1).bit_length())


def block_sparse_unsupported(
    dtype: torch.dtype, device: torch.device, head_dim: int, heads: int, kv_heads: int, ratio: int
) -> str | None:
    """Why the kernel cannot run this shape, dtype or device, else ``None``."""
    if dtype not in SUPPORTED_DTYPES:
        return f"the block-sparse kernel takes bf16 or fp16 Q/K/V, got {dtype}"
    if head_dim < MIN_DOT_EXTENT or head_dim & (head_dim - 1):
        return f"the block-sparse kernel needs a power-of-two head_dim of at least {MIN_DOT_EXTENT}, got {head_dim}"
    if kv_heads < 1 or heads % kv_heads:
        return f"{heads} query heads do not group over {kv_heads} K/V heads"
    accumulator = group_tile(heads // kv_heads) * head_dim
    if accumulator > MAX_ACCUMULATOR_ELEMENTS:
        return (
            f"a K/V head serving {heads // kv_heads} query heads of width {head_dim} needs a {accumulator}-element "
            f"accumulator, over the kernel's {MAX_ACCUMULATOR_ELEMENTS}"
        )
    for tile in sorted({config.keys for pair in (*LAUNCH_CONFIGS.values(), DEFAULT_LAUNCH_CONFIG) for config in pair}):
        if tile % ratio or tile <= ratio:
            return f"compress_ratio {ratio} must divide the kernel's key tile {tile} and leave room for the tail"
    if device.type != "cuda" or not torch.cuda.is_available():
        return f"the block-sparse kernel needs a CUDA device, got {device}"
    shared_memory = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    for config in launch_configs(device):
        # A bound on what one block holds: the K and V tiles of every pipeline stage, the fp32 copy of the V tile the
        # forward's probability product reads, and the query tile.
        element = torch.finfo(dtype).bits // 8
        needed = (
            2 * config.keys * head_dim * element * config.stages
            + config.keys * head_dim * 4
            + group_tile(heads // kv_heads) * head_dim * element
        )
        if needed > shared_memory:
            return (
                f"head_dim {head_dim} at key tile {config.keys} needs about {needed} B of shared memory, over "
                f"{shared_memory}"
            )
    return None


@triton.jit
def _query_keys(SPANS, row, block_topk, RATIO: tl.constexpr, SLOTS: tl.constexpr):
    """A query's document start, its visible keys, how many leading block slots are selected, the key tiles they fill,
    and the keys after its last complete block."""
    first = tl.load(SPANS + row * 2).to(tl.int64)
    visible = tl.load(SPANS + row * 2 + 1).to(tl.int64)
    complete = visible // RATIO
    selected = tl.minimum(complete, block_topk)
    return first, complete, selected, (selected + SLOTS - 1) // SLOTS, visible - complete * RATIO


@triton.jit
def _tile_keys(
    BLOCKS,
    row,
    block_topk,
    first,
    complete,
    selected,
    tiles,
    tail,
    tile,
    keys,
    RATIO: tl.constexpr,
    SLOTS: tl.constexpr,
):
    """Key positions of key tile ``tile`` and which are real. Tile ``tiles`` is the tail; its slots are all unselected,
    so the block load is masked there."""
    slot = tile * SLOTS + keys // RATIO
    in_selected = slot < selected
    block = tl.load(BLOCKS + row * block_topk + slot, mask=in_selected, other=0).to(tl.int64)
    is_tail = tile == tiles
    position = tl.where(is_tail, first + complete * RATIO + keys, first + block * RATIO + keys % RATIO)
    return position, tl.where(is_tail, keys < tail, in_selected)


@triton.jit(do_not_specialize=_RUNTIME_ARGUMENTS)
def _forward_kernel(
    Q,
    K,
    V,
    O,  # noqa: E741
    LSE,
    BLOCKS,
    SPANS,
    n_tokens,
    block_topk,
    scale,
    qs0,
    qs1,
    qs2,
    ks0,
    ks1,
    ks2,
    vs0,
    vs1,
    vs2,
    os0,
    os1,
    os2,
    GROUPS: tl.constexpr,
    G_PAD: tl.constexpr,
    D: tl.constexpr,
    RATIO: tl.constexpr,
    BN: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    kv_head = tl.program_id(1).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)
    row = batch * n_tokens + token
    SLOTS: tl.constexpr = BN // RATIO
    first, complete, selected, tiles, tail = _query_keys(SPANS, row, block_topk, RATIO, SLOTS)
    group = tl.arange(0, G_PAD)
    lane = tl.arange(0, D)
    keys = tl.arange(0, BN)
    in_group = group < GROUPS
    head = kv_head * GROUPS + group
    query = tl.load(
        Q + batch * qs0 + head[:, None] * qs1 + token * qs2 + lane[None, :], mask=in_group[:, None], other=0.0
    )
    accumulator = tl.zeros((G_PAD, D), tl.float32)
    running_max = tl.full((G_PAD,), _NO_SOFTMAX, tl.float32)
    running_sum = tl.zeros((G_PAD,), tl.float32)
    for tile in range(0, tiles + 1):
        position, valid = _tile_keys(
            BLOCKS, row, block_topk, first, complete, selected, tiles, tail, tile, keys, RATIO, SLOTS
        )
        key = tl.load(
            K + batch * ks0 + kv_head * ks1 + position[None, :] * ks2 + lane[:, None], mask=valid[None, :], other=0.0
        )
        value = tl.load(
            V + batch * vs0 + kv_head * vs1 + position[:, None] * vs2 + lane[None, :], mask=valid[:, None], other=0.0
        )
        scores = tl.where(valid[None, :], tl.dot(query, key) * scale, _NO_SOFTMAX)
        new_max = tl.maximum(running_max, tl.max(scores, 1))
        decay = tl.exp(running_max - new_max)
        probs = tl.where(valid[None, :], tl.exp(scores - new_max[:, None]), 0.0)
        accumulator = accumulator * decay[:, None] + tl.dot(probs, value.to(tl.float32), input_precision="tf32x3")
        running_sum = running_sum * decay + tl.sum(probs, 1)
        running_max = new_max
    output = accumulator / tl.maximum(running_sum[:, None], 1.0e-20)
    tl.store(
        O + batch * os0 + head[:, None] * os1 + token * os2 + lane[None, :],
        output.to(O.dtype.element_ty),
        mask=in_group[:, None],
    )
    heads = tl.num_programs(1) * GROUPS
    # A query that sees no key keeps a logsumexp no score reaches, so the backward's probabilities there are zero.
    tl.store(
        LSE + (batch * heads + head) * n_tokens + token,
        tl.where(running_sum > 0, running_max + tl.log(running_sum), -_NO_SOFTMAX),
        mask=in_group,
    )


@triton.jit(do_not_specialize=_RUNTIME_ARGUMENTS)
def _backward_kernel(
    Q,
    K,
    V,
    O,  # noqa: E741
    LSE,
    DO,
    DQ,
    DK,
    DV,
    BLOCKS,
    SPANS,
    n_tokens,
    block_topk,
    scale,
    qs0,
    qs1,
    qs2,
    ks0,
    ks1,
    ks2,
    vs0,
    vs1,
    vs2,
    os0,
    os1,
    os2,
    ds0,
    ds1,
    ds2,
    GROUPS: tl.constexpr,
    G_PAD: tl.constexpr,
    D: tl.constexpr,
    RATIO: tl.constexpr,
    BN: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    kv_head = tl.program_id(1).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)
    row = batch * n_tokens + token
    SLOTS: tl.constexpr = BN // RATIO
    first, complete, selected, tiles, tail = _query_keys(SPANS, row, block_topk, RATIO, SLOTS)
    group = tl.arange(0, G_PAD)
    lane = tl.arange(0, D)
    keys = tl.arange(0, BN)
    in_group = group < GROUPS
    head = kv_head * GROUPS + group
    heads = tl.num_programs(1) * GROUPS
    query = tl.load(
        Q + batch * qs0 + head[:, None] * qs1 + token * qs2 + lane[None, :], mask=in_group[:, None], other=0.0
    )
    grad_output = tl.load(
        DO + batch * ds0 + head[:, None] * ds1 + token * ds2 + lane[None, :], mask=in_group[:, None], other=0.0
    )
    output = tl.load(
        O + batch * os0 + head[:, None] * os1 + token * os2 + lane[None, :], mask=in_group[:, None], other=0.0
    )
    delta = tl.sum(output.to(tl.float32) * grad_output.to(tl.float32), 1)
    lse = tl.load(LSE + (batch * heads + head) * n_tokens + token, mask=in_group, other=-_NO_SOFTMAX)
    grad_query = tl.zeros((G_PAD, D), tl.float32)
    kv_base = ((batch * tl.num_programs(1) + kv_head) * n_tokens) * D
    for tile in range(0, tiles + 1):
        position, valid = _tile_keys(
            BLOCKS, row, block_topk, first, complete, selected, tiles, tail, tile, keys, RATIO, SLOTS
        )
        key = tl.load(
            K + batch * ks0 + kv_head * ks1 + position[None, :] * ks2 + lane[:, None], mask=valid[None, :], other=0.0
        )
        value = tl.load(
            V + batch * vs0 + kv_head * vs1 + position[None, :] * vs2 + lane[:, None], mask=valid[None, :], other=0.0
        )
        probs = tl.where(valid[None, :] & in_group[:, None], tl.exp(tl.dot(query, key) * scale - lse[:, None]), 0.0)
        grad_probs = tl.dot(grad_output, value)
        grad_scores = probs * (grad_probs - delta[:, None]) * scale
        grad_query += tl.dot(grad_scores.to(query.dtype), tl.trans(key))
        grad_key = tl.dot(tl.trans(grad_scores).to(query.dtype), query)
        grad_value = tl.dot(tl.trans(probs).to(grad_output.dtype), grad_output)
        cells = kv_base + position[:, None] * D + lane[None, :]
        tl.atomic_add(DK + cells, grad_key, mask=valid[:, None], sem="relaxed")
        tl.atomic_add(DV + cells, grad_value, mask=valid[:, None], sem="relaxed")
    # The gradient is laid out as the output is.
    tl.store(
        DQ + batch * os0 + head[:, None] * os1 + token * os2 + lane[None, :],
        grad_query.to(DQ.dtype.element_ty),
        mask=in_group[:, None],
    )


def _require_inputs(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, blocks: torch.Tensor, spans: torch.Tensor, ratio: int
) -> None:
    batch, heads, tokens, dim = query.shape
    if key.shape != value.shape or key.ndim != 4 or key.shape[0] != batch or key.shape[2:] != (tokens, dim):
        raise ValueError(
            f"block_sparse_attention takes query [batch, heads, tokens, dim] and key/value [batch, kv_heads, tokens, "
            f"dim], got {tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}"
        )
    if not query.dtype == key.dtype == value.dtype:
        raise ValueError(
            f"block_sparse_attention takes one dtype for Q/K/V, got {query.dtype}, {key.dtype}, {value.dtype}"
        )
    if blocks.ndim != 3 or blocks.shape[:2] != (batch, tokens) or spans.shape != (batch, tokens, 2):
        raise ValueError(
            f"block_sparse_attention takes blocks [batch, tokens, block_topk] and spans [batch, tokens, 2] for "
            f"{(batch, tokens)} rows, got {tuple(blocks.shape)} and {tuple(spans.shape)}"
        )
    reason = block_sparse_unsupported(query.dtype, query.device, dim, heads, key.shape[1], ratio)
    if reason is not None:
        raise ValueError(reason)


def _kernel_layout(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor``, or a contiguous copy where its pointer or a stride is off the 16-element alignment Triton
    specializes on, which would compile a second binary."""
    aligned = tensor.stride(-1) == 1 and all(stride % MIN_DOT_EXTENT == 0 for stride in tensor.stride()[:-1])
    if aligned and tensor.data_ptr() % (MIN_DOT_EXTENT * tensor.element_size()) == 0:
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def variant_key(
    dtype: torch.dtype, device: torch.device, head_dim: int, groups: int, ratio: int, index_dtype: torch.dtype
) -> tuple:
    return dtype, device, head_dim, group_tile(groups), ratio, index_dtype


def _note_variant(variant: tuple) -> None:
    warn_once(
        logger,
        _compiled_variants,
        variant,
        "block_sparse_attention compiles variant %s, which load-time warm-up did not cover; this rank compiles it now "
        "and stalls its peers at the next collective.",
        variant,
    )


class _BlockSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, blocks, spans, ratio, scale):
        batch, heads, tokens, dim = query.shape
        kv_heads = key.shape[1]
        groups = heads // kv_heads
        query, key, value = (_kernel_layout(t) for t in (query, key, value))
        blocks, spans = blocks.contiguous(), spans.contiguous()
        # [batch, tokens, heads, dim] storage, as the layer's output reshape reads it.
        output = torch.empty(batch, tokens, heads, dim, dtype=query.dtype, device=query.device).transpose(1, 2)
        lse = torch.empty(batch, heads, tokens, dtype=torch.float32, device=query.device)
        config = launch_configs(query.device).forward
        _note_variant(variant_key(query.dtype, query.device, dim, groups, ratio, blocks.dtype))
        if tokens:
            _forward_kernel[(tokens, kv_heads, batch)](
                query, key, value, output, lse, blocks, spans, tokens, blocks.shape[-1], scale,
                *query.stride()[:3], *key.stride()[:3], *value.stride()[:3], *output.stride()[:3],
                GROUPS=groups, G_PAD=group_tile(groups), D=dim, RATIO=ratio, BN=config.keys,
                num_warps=config.warps, num_stages=config.stages,
            )  # fmt: skip
        ctx.save_for_backward(query, key, value, output, lse, blocks, spans)
        ctx.ratio, ctx.scale = ratio, scale
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        query, key, value, output, lse, blocks, spans = ctx.saved_tensors
        batch, heads, tokens, dim = query.shape
        kv_heads = key.shape[1]
        groups = heads // kv_heads
        grad_output = _kernel_layout(grad_output)
        grad_query = torch.empty(batch, tokens, heads, dim, dtype=query.dtype, device=query.device).transpose(1, 2)
        grad_key = torch.zeros(key.shape, dtype=torch.float32, device=key.device)
        grad_value = torch.zeros(value.shape, dtype=torch.float32, device=value.device)
        config = launch_configs(query.device).backward
        if tokens:
            _backward_kernel[(tokens, kv_heads, batch)](
                query, key, value, output, lse, grad_output, grad_query, grad_key, grad_value, blocks, spans,
                tokens, blocks.shape[-1], ctx.scale,
                *query.stride()[:3], *key.stride()[:3], *value.stride()[:3], *output.stride()[:3],
                *grad_output.stride()[:3],
                GROUPS=groups, G_PAD=group_tile(groups), D=dim, RATIO=ctx.ratio, BN=config.keys,
                num_warps=config.warps, num_stages=config.stages,
            )  # fmt: skip
        return grad_query, grad_key.to(key.dtype), grad_value.to(value.dtype), None, None, None, None


def block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    blocks: torch.Tensor,
    spans: torch.Tensor,
    ratio: int,
    scale: float,
) -> torch.Tensor:
    """``[batch, heads, tokens, dim]`` output for ``query`` of that shape and ``key``/``value`` of ``[batch, kv_heads,
    tokens, dim]``. Query ``t`` attends the blocks ``blocks[b, t, :min(visible // ratio, block_topk)]`` (each ``ratio``
    keys counted from the document start ``first``) and the keys after its last complete block, ``spans[b, t] =
    (first, visible)`` being its document's first key and how many keys it sees. A query with ``visible == 0`` reads
    zeros. Raises ``ValueError`` where :func:`block_sparse_unsupported` gives a reason. dK and dV are not
    deterministic."""
    _require_inputs(query, key, value, blocks, spans, ratio)
    return _BlockSparseAttention.apply(query, key, value, blocks, spans, ratio, scale)


def warm_block_sparse_kernels(
    device: torch.device, dtype: torch.dtype, heads: int, kv_heads: int, head_dim: int, ratio: int
) -> None:
    """Compile the forward and backward for this layout with each block-index dtype the selection can hold. Rank-local;
    the caller fences it."""
    reason = block_sparse_unsupported(dtype, device, head_dim, heads, kv_heads, ratio)
    if reason is not None:
        raise ValueError(reason)
    length = 2 * ratio + 1
    shape = (1, heads, length, head_dim), (1, kv_heads, length, head_dim)
    spans = torch.tensor([[[0, t + 1] for t in range(length)]], dtype=torch.int32, device=device)
    for index_dtype in (torch.int16, torch.int32):
        _compiled_variants.add(variant_key(dtype, device, head_dim, heads // kv_heads, ratio, index_dtype))
        blocks = torch.zeros(1, length, 1, dtype=index_dtype, device=device)
        query, key, value = (
            torch.zeros(s, dtype=dtype, device=device, requires_grad=True) for s in (*shape, shape[1])
        )
        block_sparse_attention(query, key, value, blocks, spans, ratio, 1.0).backward(torch.zeros_like(query))
    torch.cuda.synchronize(device)
