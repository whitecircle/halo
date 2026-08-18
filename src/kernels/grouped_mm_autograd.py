"""Autograd wrapper over ``torch.nn.functional.grouped_mm``, used for the bf16 path of
:mod:`src.kernels.grouped_gemm`.

Materializes the broadcast (zero-stride) grads the native backward rejects on SM100+ Blackwell
(``"Invalid strides/sizes, got [0, 0, 0]"``) and normalizes offsets to the int32 the kernel requires.
"""

import torch
import torch.nn.functional as F
from torch.autograd import Function


def grouped_mm_grads(mat_a, mat_b, grad_output, offs, needs_a: bool, needs_b: bool):
    """Gradient pair for a grouped matmul, with the SM100+ zero-stride materialization.

    Dtype handling is left to the caller: ``grad_output`` and the operands must already share a dtype
    (mixed-dtype ``grouped_mm`` raises).
    """
    if any(s == 0 for s in grad_output.stride()):
        if grad_output.numel() == 0:
            # ``contiguous()`` is a no-op on a 0-element broadcast grad (it keeps the [0, 0] strides),
            # and the kernel validates strides against sizes even at M == 0, which happens under EP
            # when a rank's dispatch routes zero tokens.
            grad_output = torch.empty(grad_output.shape, dtype=grad_output.dtype, device=grad_output.device)
        else:
            grad_output = grad_output.contiguous()
    grad_a = F.grouped_mm(grad_output, mat_b.transpose(-2, -1), offs=offs) if needs_a else None
    grad_b = F.grouped_mm(mat_a.transpose(-2, -1), grad_output, offs=offs) if needs_b else None
    return grad_a, grad_b


class _GroupedMMFunction(Function):
    """Autograd wrapper around ``F.grouped_mm`` that fixes backward on SM100+.

    ``aten::_grouped_mm`` registers no autocast kernel, so the forward output carries ``mat_a``'s
    dtype even inside the expert compute's ``autocast`` scope — which is why ``backward`` hands
    ``grad_output`` to the GEMMs uncast.
    """

    @staticmethod
    def forward(ctx, mat_a, mat_b, offs):
        if offs is not None and offs.dtype != torch.int32:
            offs = offs.to(torch.int32)

        ctx.save_for_backward(mat_a, mat_b, offs)
        return F.grouped_mm(mat_a, mat_b, offs=offs)

    @staticmethod
    def backward(ctx, grad_output):
        mat_a, mat_b, offs = ctx.saved_tensors
        grad_mat_a, grad_mat_b = grouped_mm_grads(
            mat_a, mat_b, grad_output, offs, ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        )
        return grad_mat_a, grad_mat_b, None


def grouped_mm(mat_a, mat_b, *, offs=None):
    """Grouped matrix multiply with correct backward on SM100+ (Blackwell)."""
    return _GroupedMMFunction.apply(mat_a, mat_b, offs)
