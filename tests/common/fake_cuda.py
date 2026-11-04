"""A CPU tensor reporting ``is_cuda``, for tests of code branching to a CUDA-only path on that attribute alone, and
stand-ins for the ``out_dtype`` matmuls and the attention kernels that path calls, which CPU lacks."""

import torch

_KERNEL_ALIGNMENT = 8
_LSE_QUERY_PADDING = 128


class CudaLike(torch.Tensor):
    """``tensor.as_subclass(CudaLike)``; ops return the subclass, so a kernel stand-in takes
    ``as_subclass(torch.Tensor)`` back first."""

    is_cuda = True


class CudaBmm:
    """``torch.bmm`` and ``torch.baddbmm``. A patch, since a function mode does not see the backward of an autograd
    function."""

    def __init__(self, bmm):
        self._bmm = bmm
        self.calls: list[tuple[torch.dtype, torch.dtype, torch.dtype | None]] = []

    def __call__(self, a, b, *, out_dtype=None):
        self.calls.append((a.dtype, b.dtype, out_dtype))
        if out_dtype is None:
            return self._bmm(a, b)
        plain = [t.as_subclass(torch.Tensor).to(out_dtype) for t in (a, b)]
        return self._bmm(*plain).as_subclass(CudaLike)

    def baddbmm(self, accumulator, a, b, *, out_dtype, out):
        self.calls.append((a.dtype, b.dtype, out_dtype))
        plain = [t.as_subclass(torch.Tensor).to(out_dtype) for t in (a, b)]
        return out.add_(self._bmm(*plain))


def efficient_attention(
    query, key, value, attn_bias, compute_log_sumexp, dropout_p=0.0, is_causal=False, *, scale=None
) -> tuple[torch.Tensor, ...]:
    """``torch._scaled_dot_product_efficient_attention``, holding callers to what the kernel checks: a head width and a
    bias row stride at a multiple of 8, a bias over every head. A query with no key gets a zero output and a zero log-sum-exp, whose query axis pads
    with values the kernel never wrote."""
    batch, heads, queries, width = query.shape
    if width % _KERNEL_ALIGNMENT:
        raise RuntimeError(
            f"Mem efficient attention requires last dimension of inputs to be divisible by 8, got {width}"
        )
    if attn_bias.shape != (batch, heads, queries, key.shape[2]):
        raise RuntimeError(f"attn_bias: wrong shape {tuple(attn_bias.shape)}")
    if attn_bias.stride(2) % _KERNEL_ALIGNMENT:
        raise RuntimeError(f"attn_bias is not correctly aligned, stride {attn_bias.stride(2)}")
    query, key, value, attn_bias = (t.as_subclass(torch.Tensor) for t in (query, key, value, attn_bias))
    wide = torch.promote_types(query.dtype, torch.float32)
    scores = query.to(wide) @ key.to(wide).transpose(-1, -2) * scale + attn_bias.to(wide)
    lse = torch.logsumexp(scores, dim=-1).nan_to_num(neginf=0.0)
    probs = torch.exp(scores - lse[..., None]).to(value.dtype)
    out = (probs.to(wide) @ value.to(wide)).to(value.dtype)
    if not compute_log_sumexp:
        return out, torch.empty(0), torch.empty(0), torch.empty(0)
    padded = torch.full((batch, heads, queries + -queries % _LSE_QUERY_PADDING), float("nan"), dtype=wide)
    padded[..., :queries] = lse
    return out, padded, torch.empty(0), torch.empty(0)


def masked_softmax_backward(scores, allowed, lse, scale, upstream_scores, delta, writes_probs, dtype):
    """``halo.kernels.masked_softmax.masked_softmax_backward``, whose Triton launch has no CPU build, pass by pass."""
    probs = torch.exp(scores * scale - lse[..., None]).masked_fill(~allowed, 0.0)
    d_scores = None if upstream_scores is None else (probs * (upstream_scores - delta[..., None]) * scale).to(dtype)
    return probs.to(dtype) if writes_probs else None, d_scores
