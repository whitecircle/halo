"""Expert-sorted routing for the grouped-GEMM MoE compute: :func:`sort_routing_by_expert` orders a dispatch's local
expert ids by expert into what the grouped GEMM and the permute read. With the dispatch's host-side receive counts on
CUDA it is a counting sort in three Triton launches; without them, or off CUDA, the torch construction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from halo.kernels.histogram import sync_free_bincount
from halo.kernels.moe_permute import build_inv_map

# Slots one program sorts: whole rows of the power-of-two padded top_k, so every slot of a row lands in one program,
# which then builds that row's ``inv_map`` too.
_BLOCK_SLOTS = 1024
_LANE_BITS = (_BLOCK_SLOTS - 1).bit_length()
# Elements of one scan tile, blocks x padded local experts.
_SCAN_TILE = 2048
# Message of the device-side check, on every expert path, that the receive counts match the received ids.
RECEIVE_COUNTS_DISAGREE = "DeepEP receive counts disagree with the received expert ids"


@triton.jit
def _block_slots(experts_ptr, n_rows, n_local, TOP_K: tl.constexpr, K_PAD: tl.constexpr, ROWS: tl.constexpr):
    # This program's ROWS rows as ROWS * K_PAD lanes in slot order; lanes past TOP_K or the last row hold no slot.
    lane = tl.arange(0, ROWS * K_PAD)
    row = tl.program_id(0).to(tl.int64) * ROWS + lane // K_PAD
    k = lane % K_PAD
    exists = (row < n_rows) & (k < TOP_K)
    slot = row * TOP_K + k
    expert = tl.load(experts_ptr + slot, mask=exists, other=-1)
    local = exists & (expert >= 0) & (expert < n_local)
    return lane, slot, exists, expert, local


# The row, block and slot counts change every dispatch; specialized, each class of them would compile its own binary,
# an unseen one between the dispatch and its combine (see ``fused_glu._glu_fwd_kernel``).
@triton.jit(do_not_specialize=["n_rows"])
def _histogram_kernel(
    experts_ptr,
    hist_ptr,
    n_rows,
    n_local,
    TOP_K: tl.constexpr,
    K_PAD: tl.constexpr,
    ROWS: tl.constexpr,
    E_PAD: tl.constexpr,
):
    # hist[b, e] = the local slots of block b routed to expert e.
    _, _, _, expert, local = _block_slots(experts_ptr, n_rows, n_local, TOP_K, K_PAD, ROWS)
    counts = tl.histogram(tl.where(local, expert, 0).to(tl.int32), E_PAD, mask=local)
    tl.store(hist_ptr + tl.program_id(0).to(tl.int64) * E_PAD + tl.arange(0, E_PAD), counts)


@triton.jit(do_not_specialize=["n_blocks", "n_valid"])
def _scan_kernel(
    hist_ptr, offs_ptr, shift_ptr, ok_ptr, n_blocks, n_local, n_valid, E_PAD: tl.constexpr, CHUNK: tl.constexpr
):
    # offs[e] = the end of expert e's group, its slots and every lower expert's summed over the blocks.
    # shift[b, e] = where block b's first expert-e slot lands, minus that slot's position in the block's sorted order.
    # ok = the local slots number n_valid, the host count that sized the outputs.
    e = tl.arange(0, E_PAD)
    totals = tl.zeros((E_PAD,), dtype=tl.int32)
    for b0 in range(0, n_blocks, CHUNK):
        b = b0 + tl.arange(0, CHUNK)
        h = tl.load(hist_ptr + b[:, None] * E_PAD + e[None, :], mask=(b < n_blocks)[:, None], other=0)
        totals += tl.sum(h, axis=0)
    end = tl.cumsum(totals, axis=0)
    tl.store(offs_ptr + e, end, mask=e < n_local)
    running = end - totals
    for b0 in range(0, n_blocks, CHUNK):
        b = b0 + tl.arange(0, CHUNK)
        tile = b[:, None] * E_PAD + e[None, :]
        in_range = (b < n_blocks)[:, None]
        h = tl.load(hist_ptr + tile, mask=in_range, other=0)
        block_start = running[None, :] + tl.cumsum(h, axis=0) - h
        first_in_block = tl.cumsum(h, axis=1) - h
        tl.store(shift_ptr + tile, block_start - first_in_block, mask=in_range)
        running += tl.sum(h, axis=0)
    tl.store(ok_ptr, (tl.sum(totals, axis=0) == n_valid).to(tl.int32))


@triton.jit(do_not_specialize=["n_rows", "n_valid"])
def _place_kernel(
    experts_ptr,
    weights_ptr,
    shift_ptr,
    dest_ptr,
    sorted_token_ptr,
    sorted_weight_ptr,
    sorted_expert_ptr,
    inv_map_ptr,
    n_rows,
    n_local,
    n_valid,
    TOP_K: tl.constexpr,
    K_PAD: tl.constexpr,
    ROWS: tl.constexpr,
    E_PAD: tl.constexpr,
    LANE_BITS: tl.constexpr,
):
    lane, slot, exists, expert, local = _block_slots(experts_ptr, n_rows, n_local, TOP_K, K_PAD, ROWS)
    pid = tl.program_id(0).to(tl.int64)
    # The block's slots by expert, in slot order within one: (expert, lane) packed in one int64 and sorted, the
    # non-local slots keyed past every expert. A sorted lane's position is its index ``lane``.
    key = tl.where(local, expert, n_local).to(tl.int64)
    packed = tl.sort((key << LANE_BITS) | lane.to(tl.int64))
    s_key = packed >> LANE_BITS
    s_lane = packed - (s_key << LANE_BITS)
    s_local = s_key < n_local
    shift = tl.load(shift_ptr + pid * E_PAD + s_key, mask=s_local, other=0).to(tl.int64)
    s_dest = tl.where(s_local, shift + lane, n_valid)
    s_row = pid * ROWS + s_lane // K_PAD
    s_slot = s_row * TOP_K + s_lane % K_PAD
    tl.store(sorted_token_ptr + s_dest, s_row, mask=s_local)
    tl.store(sorted_weight_ptr + s_dest, tl.load(weights_ptr + s_slot, mask=s_local), mask=s_local)
    tl.store(sorted_expert_ptr + s_dest, s_key, mask=s_local)
    # Back in slot order by a sort on the lane (every lane distinct), then each row's sorted positions ascending, its
    # non-local slots last as the sentinel n_valid.
    dest = tl.sort((s_lane << 32) | s_dest) - (lane.to(tl.int64) << 32)
    tl.store(dest_ptr + slot, dest.to(tl.int32), mask=exists)
    row_dest = tl.sort(tl.reshape(dest, ROWS, K_PAD), dim=1)
    tl.store(inv_map_ptr + tl.reshape(slot, ROWS, K_PAD), row_dest, mask=tl.reshape(exists, ROWS, K_PAD))


class _CountingSort(torch.autograd.Function):
    """The counting sort; only the gate weights carry a gradient, each slot's read back at its sorted position."""

    @staticmethod
    def forward(ctx, weights: torch.Tensor, experts: torch.Tensor, num_local_experts: int, counts: tuple[int, ...]):
        n_rows, top_k = experts.shape
        device = experts.device
        k_pad = triton.next_power_of_2(top_k)
        rows = _BLOCK_SLOTS // k_pad
        e_pad = triton.next_power_of_2(num_local_experts)
        n_blocks = triton.cdiv(n_rows, rows)
        n_valid = sum(counts)
        experts, weights = experts.contiguous(), weights.contiguous()
        hist = torch.empty((n_blocks, e_pad), dtype=torch.int32, device=device)
        shift = torch.empty_like(hist)
        offs = torch.empty(num_local_experts, dtype=torch.int32, device=device)
        ok = torch.empty(1, dtype=torch.int32, device=device)
        dest = torch.empty(n_rows * top_k, dtype=torch.int32, device=device)
        sorted_token_idx = torch.empty(n_valid, dtype=torch.long, device=device)
        sorted_weights = torch.empty(n_valid, dtype=weights.dtype, device=device)
        sorted_expert_ids = torch.empty(n_valid, dtype=experts.dtype, device=device)
        inv_map = torch.empty((n_rows, top_k), dtype=torch.long, device=device)
        block = {"TOP_K": top_k, "K_PAD": k_pad, "ROWS": rows, "E_PAD": e_pad}
        _histogram_kernel[(n_blocks,)](experts, hist, n_rows, num_local_experts, **block)
        _scan_kernel[(1,)](
            hist, offs, shift, ok, n_blocks, num_local_experts, n_valid, E_PAD=e_pad, CHUNK=max(1, _SCAN_TILE // e_pad)
        )
        # Device-side, before the placement: a wrong total would write past the outputs the counts sized, or leave their
        # tail unwritten.
        torch._assert_async(ok, RECEIVE_COUNTS_DISAGREE)
        _place_kernel[(n_blocks,)](
            experts,
            weights,
            shift,
            dest,
            sorted_token_idx,
            sorted_weights,
            sorted_expert_ids,
            inv_map,
            n_rows,
            num_local_experts,
            n_valid,
            LANE_BITS=_LANE_BITS,
            **block,
        )
        ctx.save_for_backward(dest)
        ctx.weights_shape = weights.shape
        ctx.mark_non_differentiable(sorted_token_idx, sorted_expert_ids, offs, inv_map)
        # The integer outputs take no gradient; materialized, each would cost a zero-fill in every backward.
        ctx.set_materialize_grads(False)
        return sorted_token_idx, sorted_weights, sorted_expert_ids, offs, inv_map

    @staticmethod
    def backward(ctx, _token_grad, sorted_weights_grad, _expert_grad, _offs_grad, _inv_map_grad):
        if sorted_weights_grad is None:
            return None, None, None, None
        (dest,) = ctx.saved_tensors
        # The zero padded on at position n_valid is every non-local slot's gradient.
        grad = F.pad(sorted_weights_grad, (0, 1)).index_select(0, dest)
        return grad.view(ctx.weights_shape), None, None, None


# Eager under torch.compile: its kernels are already fused, and traced, the host counts would become graph inputs
# that change every step.
@torch.compiler.disable
def _counting_sort(experts: torch.Tensor, weights: torch.Tensor, num_local_experts: int, counts: tuple[int, ...]):
    return _CountingSort.apply(weights, experts, num_local_experts, counts)


def _argsort_routing(
    experts: torch.Tensor, weights: torch.Tensor, num_local_experts: int, counts: tuple[int, ...] | None
):
    """The torch construction (see :func:`sort_routing_by_expert`); ``counts`` sizes the compaction, None when every id
    is local."""
    n_rows, top_k = experts.shape
    flat_experts = experts.reshape(-1)
    flat_weights = weights.reshape(-1)
    flat_token_idx = torch.arange(n_rows, device=experts.device).unsqueeze(1).expand(-1, top_k).reshape(-1)
    if counts is not None:
        num_valid = sum(counts)
        valid = (flat_experts >= 0) & (flat_experts < num_local_experts)
        # Device-side: a count that disagrees with the received ids would silently truncate or -1-pad the compaction.
        torch._assert_async(valid.sum() == num_valid, RECEIVE_COUNTS_DISAGREE)
        valid_idx = torch.nonzero_static(valid, size=num_valid).squeeze(1)
        flat_experts = flat_experts[valid_idx]
        flat_weights = flat_weights[valid_idx]
        flat_token_idx = flat_token_idx[valid_idx]
    order = torch.argsort(flat_experts, stable=True)
    sorted_expert_ids = flat_experts[order]
    sorted_token_idx = flat_token_idx[order]
    # A histogram, not unique_consecutive(return_counts=True): the latter sizes its output from the run count.
    offs = torch.cumsum(sync_free_bincount(sorted_expert_ids, num_local_experts, dtype=torch.long), 0).to(torch.int32)
    inv_map = build_inv_map(sorted_token_idx, n_rows, top_k)
    return sorted_token_idx, flat_weights[order], sorted_expert_ids, offs, inv_map


def sort_routing_by_expert(
    experts: torch.Tensor, weights: torch.Tensor, num_local_experts: int, counts: tuple[int, ...] | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(sorted_token_idx, sorted_weights, sorted_expert_ids, offs, inv_map)`` of a dispatch's ``[N, top_k]`` local
    expert ids and gate weights: each local expert's slots in slot order, expert after expert, ids outside
    ``[0, num_local_experts)`` (DeepEP's ``-1``) dropped, ``offs`` the int32 group ends and ``inv_map`` the permute's
    (:func:`build_inv_map`). ``counts[e]`` is the host-side number of slots routed to expert ``e``, None when every id
    is local; their sum sizes the outputs, and a sum that disagrees with the ids' local slots trips a device-side
    assert. Only ``sorted_weights`` carries a gradient.

    With ``counts`` on CUDA it is a counting sort: a per-block histogram, one scan over the blocks that also writes the
    group ends, and a placement that also builds ``inv_map``. Nothing is read back from the device and no chain of
    small aten ops runs between the dispatch and the first grouped GEMM. The kernels take only launch scalars from the
    host: a copy of the counts per sort would stage through a pinned host block held until the copy lands, so the
    pinned pool would grow, one page-locked allocation at a time, whenever the host ran further ahead of the device
    than before. Otherwise it is the torch construction: a ``nonzero_static`` compaction sized by the counts where they
    exist, a stable argsort, a histogram and :func:`build_inv_map`. Both return the same integers in the same order."""
    if counts is not None and len(counts) != num_local_experts:
        raise ValueError(f"{len(counts)} receive counts for {num_local_experts} local experts")
    if counts is not None and experts.is_cuda:
        return _counting_sort(experts, weights, num_local_experts, counts)
    return _argsort_routing(experts, weights, num_local_experts, counts)


def warm_route_sort_kernels(top_k: int, num_local_experts: int, device: torch.device) -> None:
    """Sort one synthetic receive on ``device`` with counts: on CUDA that compiles the counting sort's kernels for this
    ``top_k`` and expert count, and they take their row, block and slot counts unspecialized, so one call covers every
    dispatch size. The ids and weights carry the dtypes DeepEP hands back: ``topk_idx_t`` (int64) and fp32."""
    experts = torch.zeros((1, top_k), dtype=torch.long, device=device)
    weights = torch.zeros((1, top_k), dtype=torch.float32, device=device)
    sort_routing_by_expert(experts, weights, num_local_experts, (top_k,) + (0,) * (num_local_experts - 1))
