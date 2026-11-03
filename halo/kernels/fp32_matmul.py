"""Matmuls with an fp32 (or wider) result that keep half-precision CUDA inputs on the tensor cores."""

import torch

# Dtypes whose CUDA GEMM runs on the tensor cores and can return an fp32 result.
_TENSOR_CORE_DTYPES = (torch.float16, torch.bfloat16)


def gemm_dtype(t: torch.Tensor) -> torch.dtype:
    """``t``'s half dtype on CUDA, where the tensor cores accumulate in fp32, and fp32 otherwise. Half operands are
    never upcast there, as under the pinned ``highest`` fp32 matmul precision they would run on the CUDA cores."""
    return t.dtype if t.is_cuda and t.dtype in _TENSOR_CORE_DTYPES else torch.float32


def bmm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``bmm``'s ``out_dtype`` has no derivative and no CPU kernel; a caller needing gradients wraps this itself."""
    wide = torch.promote_types(a.dtype, torch.float32)
    if gemm_dtype(a) != torch.float32:
        return torch.bmm(a, b, out_dtype=wide)
    return torch.bmm(a.to(wide), b.to(wide))


def mm_fp32(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """``a @ b`` as fp32, or ``out += a @ b`` into an fp32 ``out``. With a half-precision CUDA ``b`` both operands take
    its dtype and the tensor-core GEMM accumulates in fp32, so neither product nor sum rounds to half precision."""
    dtype = gemm_dtype(b)
    if dtype != torch.float32:
        a = a.to(dtype)
        if out is None:
            return torch.mm(a, b, out_dtype=torch.float32)
        return torch.addmm(out, a, b, out_dtype=torch.float32, out=out)
    product = a.float() @ b.float()
    return product if out is None else out.add_(product)
