"""The element-wise half of a masked softmax's backward as one launch over a tile's fp32 scores, whose separate passes
run at memory bandwidth: from the scores to the probabilities and the cotangent the matmuls then read."""

import torch
import triton
import triton.language as tl

_BLOCK_KEYS = 512


# Tile shapes and the mask's and normalizers' view offsets change on every call; specialized, each class of them would
# compile its own binary mid-step. A ``None`` output is a constant the compile drops.
@triton.jit(
    do_not_specialize=["n_queries", "n_keys", "stride_allowed"],
    do_not_specialize_on_alignment=["allowed_ptr", "lse_ptr"],
)
def _softmax_backward_kernel(
    scores_ptr,
    allowed_ptr,
    lse_ptr,
    upstream_ptr,
    delta_ptr,
    probs_ptr,
    d_scores_ptr,
    n_queries,
    n_keys,
    stride_allowed,
    scale,
    BLOCK_KEYS: tl.constexpr,
):
    # int64, as rows * n_keys can exceed 2**31.
    row = tl.program_id(0).to(tl.int64)
    keys = tl.program_id(1) * BLOCK_KEYS + tl.arange(0, BLOCK_KEYS)
    mask = keys < n_keys
    offsets = row * n_keys + keys
    allowed = tl.load(allowed_ptr + (row % n_queries) * stride_allowed + keys, mask=mask, other=0)
    scores = tl.load(scores_ptr + offsets, mask=mask)
    probs = tl.where(allowed > 0, tl.exp(scores * scale - tl.load(lse_ptr + row)), 0.0)
    if probs_ptr is not None:
        tl.store(probs_ptr + offsets, probs, mask=mask)
    if d_scores_ptr is not None:
        upstream = tl.load(upstream_ptr + offsets, mask=mask)
        tl.store(d_scores_ptr + offsets, probs * (upstream - tl.load(delta_ptr + row)) * scale, mask=mask)


def masked_softmax_backward(
    scores: torch.Tensor,
    allowed: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    upstream_scores: torch.Tensor | None,
    delta: torch.Tensor | None,
    writes_probs: bool,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """``probs = exp(scores * scale - lse)`` under the ``[Q, K]`` mask ``allowed``, and with ``upstream_scores`` the
    softmax's backward onto its scores, ``probs * (upstream_scores - delta) * scale``, both ``[H, Q, K]`` in ``dtype``.
    ``scores`` and ``upstream_scores`` are ``[H, Q, K]``, ``lse`` and ``delta`` ``[H, Q]``."""
    heads, queries, keys = scores.shape
    probs = torch.empty(scores.shape, dtype=dtype, device=scores.device) if writes_probs else None
    d_scores = None if upstream_scores is None else torch.empty(scores.shape, dtype=dtype, device=scores.device)
    allowed = allowed.contiguous() if allowed.stride(-1) != 1 else allowed
    _softmax_backward_kernel[heads * queries, triton.cdiv(keys, _BLOCK_KEYS)](
        scores.contiguous(),
        allowed.view(torch.uint8),
        lse.contiguous(),
        None if upstream_scores is None else upstream_scores.contiguous(),
        None if delta is None else delta.contiguous(),
        probs,
        d_scores,
        queries,
        keys,
        allowed.stride(0),
        scale,
        BLOCK_KEYS=_BLOCK_KEYS,
    )
    return probs, d_scores
