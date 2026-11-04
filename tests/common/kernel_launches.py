"""The CUDA kernels a block of aten ops would launch, recorded on any device as classes CUDA loads apart. CUDA loads a
kernel on first launch, so a warm-up has to launch every class its hot path reaches; missing ones load mid-step. Models
``indexSelectSmallIndex`` and the bf16 grouped-GEMM CUTLASS tiles, not ``gather``'s alignment-based pick."""

from __future__ import annotations

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

INDEX_SELECT_SMALL_INDEX_ROWS = 16
GROUPED_MM_SMALL_TILE_DIM = 128
_INT32_MAX = 2**31 - 1


def _fits_32bit_offsets(tensor: torch.Tensor) -> bool:
    """ATen's ``canUse32BitIndexMath``: fewer elements, and a last element at a lower offset, than ``INT32_MAX``."""
    last = sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride(), strict=True) if size)
    return tensor.numel() < _INT32_MAX and last < _INT32_MAX


def _collapsed_rank(tensor: torch.Tensor) -> int:
    """The rank ATen's ``collapseDims`` leaves for a dim-0 select: that dim, plus one per contiguous run of non-unit
    dims."""
    runs, outer_stride = 0, None
    for size, stride in zip(tensor.shape[1:], tensor.stride()[1:], strict=True):
        if size == 1:
            continue
        if outer_stride != size * stride:
            runs += 1
        outer_stride = stride
    return 1 + runs


def index_select_kernel(source: torch.Tensor, index: torch.Tensor) -> tuple | None:
    """The kernel class ``source.index_select(0, index)`` launches on CUDA; None for an empty output."""
    out = torch.empty((index.numel(), *source.shape[1:]), dtype=source.dtype, device="meta")
    if out.numel() == 0:
        return None
    if index.numel() > INDEX_SELECT_SMALL_INDEX_ROWS or not all(map(_fits_32bit_offsets, (out, source, index))):
        return ("index_select", source.dtype, index.dtype, "gather")
    rank = _collapsed_rank(source)
    specialized = rank if rank == _collapsed_rank(out) and rank <= 3 and index.is_contiguous() else -1
    return ("index_select", source.dtype, index.dtype, f"small, rank {specialized}")


def _major_order(tensor: torch.Tensor) -> str:
    """Row-major where the operand's last dim is unit-strided, as ATen reads it."""
    return "row-major" if tensor.stride(-1) == 1 else "column-major"


def grouped_mm_kernel(mat_a: torch.Tensor, mat_b: torch.Tensor) -> tuple:
    """The kernel class ``torch._grouped_mm`` launches for a 2-D ``mat_a``: the tile reads its rows per group against a
    3-D ``mat_b``, its rows whole against 2-D."""
    if mat_a.dtype != torch.bfloat16 or mat_b.dtype != torch.bfloat16:
        return ("grouped_mm", mat_a.dtype, mat_b.dtype, "fallback")
    rows = mat_a.shape[0] // mat_b.shape[0] if mat_b.dim() == 3 else mat_a.shape[0]
    small = min(rows, mat_b.shape[-1]) <= GROUPED_MM_SMALL_TILE_DIM
    layout = f"a {_major_order(mat_a)}, b {_major_order(mat_b)}"
    return ("grouped_mm", torch.bfloat16, layout, "small" if small else "large")


def grouped_mm_ragged_rows(mat_a: torch.Tensor, mat_b: torch.Tensor, offs: torch.Tensor) -> int:
    """Token rows per group: ``mat_a``'s rows over a 3-D ``mat_b``'s groups, or against 2-D (the weight gradient) its
    columns over ``offs``' groups."""
    return mat_a.shape[0] // mat_b.shape[0] if mat_b.dim() == 3 else mat_a.shape[1] // offs.numel()


class KernelLaunches(TorchDispatchMode):
    """The kernel classes of every aten op while active, backward included."""

    def __init__(self):
        super().__init__()
        self.kernels: set[tuple] = set()
        self.grouped_mm_calls: list[tuple[tuple, int]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.ops.aten.index_select.default:
            source, dim, index = args
            if dim != 0:
                raise NotImplementedError(f"index_select along dim {dim}: the class is modelled for dim 0")
            kernel = index_select_kernel(source, index)
            if kernel is not None:
                self.kernels.add(kernel)
        elif func is torch.ops.aten._grouped_mm.default:
            mat_a, mat_b = args[:2]
            offs = args[2] if len(args) > 2 else kwargs.get("offs")
            if mat_a.dim() != 2 or offs is None:
                raise NotImplementedError(
                    "a grouped GEMM without a ragged 2-D mat_a: the class is modelled for that one"
                )
            kernel = grouped_mm_kernel(mat_a, mat_b)
            self.kernels.add(kernel)
            self.grouped_mm_calls.append((kernel, grouped_mm_ragged_rows(mat_a, mat_b, offs)))
        else:
            leaves = tree_leaves((args, kwargs))
            self.kernels.add((str(func), tuple(leaf.dtype for leaf in leaves if isinstance(leaf, torch.Tensor))))
        return func(*args, **kwargs)
