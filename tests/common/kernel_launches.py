"""The CUDA kernels a block of aten ops would launch, recorded on any device as classes CUDA loads apart: an op with its
tensor operands' dtypes, ``index_select`` also by the kernel ATen picks from the index length, and ``_grouped_mm`` by
the CUTLASS kernel ATen picks from the operand layouts and the group extents.

An index of up to 16 rows, within 32-bit offsets, takes ``indexSelectSmallIndex``, specialized on the collapsed rank of
the source and the output and on a contiguous index; a longer one runs through ``gather``
(``aten/src/ATen/native/cuda/Indexing.cu``), classed here by dtype, though CUDA picks its vectorized kernel (any dtype)
or its elementwise one by alignment, which this does not model.

On SM 9.x and 10.x a bf16 grouped GEMM runs a CUTLASS kernel specialized on each operand's major order and on one of
two tiles: small where the rows per group or the output width is at most 128, large otherwise
(``dispatch_bf16_grouped_kernel_on_tile_size``, ``aten/src/ATen/native/cuda/GroupMM.cu``). The rows per group are the
ragged rows over the group count, or, for the weight gradient, ragged along the contraction, the weight's input width.
Other dtypes take ``_grouped_mm_fallback``, a matmul per group, classed by dtype alone.

CUDA loads a kernel on its first launch, so a warm-up has to launch every class its hot path reaches at any receive
size: a hot path's classes that its warm-up's lack are the kernels it would load mid-step.
"""

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
    """The rank ATen's ``collapseDims`` leaves ``tensor`` at for a select along dim 0: that dim, plus one per run of the
    other dims of size above one that is contiguous in memory."""
    runs, outer_stride = 0, None
    for size, stride in zip(tensor.shape[1:], tensor.stride()[1:], strict=True):
        if size == 1:
            continue
        if outer_stride != size * stride:
            runs += 1
        outer_stride = stride
    return 1 + runs


def index_select_kernel(source: torch.Tensor, index: torch.Tensor) -> tuple | None:
    """The kernel class ``source.index_select(0, index)`` launches on CUDA; None for an empty output, which launches
    none."""
    out = torch.empty((index.numel(), *source.shape[1:]), dtype=source.dtype, device="meta")
    if out.numel() == 0:
        return None
    if index.numel() > INDEX_SELECT_SMALL_INDEX_ROWS or not all(map(_fits_32bit_offsets, (out, source, index))):
        return ("index_select", source.dtype, index.dtype, "gather")
    rank = _collapsed_rank(source)
    specialized = rank if rank == _collapsed_rank(out) and rank <= 3 and index.is_contiguous() else -1
    return ("index_select", source.dtype, index.dtype, f"small, rank {specialized}")


def _major_order(tensor: torch.Tensor) -> str:
    """The major order ATen reads a grouped-GEMM operand in: row-major where its last dim is unit-strided."""
    return "row-major" if tensor.stride(-1) == 1 else "column-major"


def grouped_mm_kernel(mat_a: torch.Tensor, mat_b: torch.Tensor) -> tuple:
    """The kernel class ``torch._grouped_mm(mat_a, mat_b, offs)`` launches on CUDA for a 2-D ``mat_a``. Against a 3-D
    ``mat_b`` the tile reads ``mat_a``'s rows over the groups; against a 2-D one (the weight gradient, ragged along the
    contraction) its rows whole."""
    if mat_a.dtype != torch.bfloat16 or mat_b.dtype != torch.bfloat16:
        return ("grouped_mm", mat_a.dtype, mat_b.dtype, "fallback")
    rows = mat_a.shape[0] // mat_b.shape[0] if mat_b.dim() == 3 else mat_a.shape[0]
    small = min(rows, mat_b.shape[-1]) <= GROUPED_MM_SMALL_TILE_DIM
    layout = f"a {_major_order(mat_a)}, b {_major_order(mat_b)}"
    return ("grouped_mm", torch.bfloat16, layout, "small" if small else "large")


def grouped_mm_ragged_rows(mat_a: torch.Tensor, mat_b: torch.Tensor, offs: torch.Tensor) -> int:
    """The token rows per group of a grouped GEMM over a 2-D ``mat_a``: its rows over a 3-D ``mat_b``'s groups, or
    against a 2-D ``mat_b`` (the weight gradient, contracting over the tokens) its columns over ``offs``' groups."""
    return mat_a.shape[0] // mat_b.shape[0] if mat_b.dim() == 3 else mat_a.shape[1] // offs.numel()


class KernelLaunches(TorchDispatchMode):
    """The kernel classes of every aten op run while active, backward included: ``index_select`` by
    :func:`index_select_kernel` (nothing for an empty output), ``_grouped_mm`` by :func:`grouped_mm_kernel`, each
    call also listed with its :func:`grouped_mm_ragged_rows` in ``grouped_mm_calls``, any other op by its name and its
    tensor operands' dtypes."""

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
