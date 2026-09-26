#!/usr/bin/env python
"""Fused MoE permute/unpermute kernels match an fp64 ``index_add_`` reference, forward and backward.

The kernels walk ``inv_map`` (sentinel-padded sorted positions per token) instead of padding and
materializing a ``[recv_N, top_k, H]`` gather, so the cases cover tokens routed to fewer than ``top_k``
local experts (sentinel slots), tokens routed to none, a hidden size that is not a multiple of the
column tile, and an empty dispatch.
"""

import pytest
import torch

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.kernels.moe_permute import MoEWeightedUnpermute, gather_reduce_rows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp(min=1e-12)).item()


def _routing(n_tokens: int, top_k: int, n_sorted: int, generator: torch.Generator):
    """Sorted-row → token map with at most ``top_k`` rows per token, and its ``inv_map``."""
    slots = torch.arange(n_tokens, device="cuda").repeat_interleave(top_k)
    token_idx = slots[torch.randperm(slots.numel(), generator=generator, device="cuda")[:n_sorted]]
    token_idx = token_idx.sort(stable=True).values if n_sorted == 0 else token_idx
    return token_idx, EPMoELayerBase._build_inv_map(token_idx, n_tokens, top_k)


@pytest.mark.parametrize(("dtype", "tol"), [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize(
    ("n_tokens", "top_k", "hidden", "fill"), [(517, 8, 2816, 0.5), (64, 4, 100, 1.0), (33, 8, 1025, 0.1)]
)
def test_weighted_unpermute_matches_index_add(dtype, tol, n_tokens, top_k, hidden, fill):
    generator = torch.Generator(device="cuda").manual_seed(n_tokens)
    n_sorted = int(n_tokens * top_k * fill)
    token_idx, inv_map = _routing(n_tokens, top_k, n_sorted, generator)
    expert_out = torch.randn(n_sorted, hidden, generator=generator, device="cuda", dtype=dtype).requires_grad_(True)
    weights = torch.rand(n_sorted, generator=generator, device="cuda", dtype=dtype).requires_grad_(True)
    grad = torch.randn(n_tokens, hidden, generator=generator, device="cuda", dtype=dtype)

    out = MoEWeightedUnpermute.apply(expert_out, weights, token_idx, inv_map)
    out.backward(grad)

    ref_out = expert_out.detach().double().requires_grad_(True)
    ref_w = weights.detach().double().requires_grad_(True)
    expected = torch.zeros(n_tokens, hidden, dtype=torch.float64, device="cuda")
    expected = expected.index_add(0, token_idx, ref_out * ref_w.unsqueeze(-1))
    expected.backward(grad.double())

    assert _rel(out, expected) < tol
    assert _rel(expert_out.grad, ref_out.grad) < tol
    assert _rel(weights.grad, ref_w.grad) < tol


def test_gather_reduce_sums_only_real_rows():
    """Sentinel slots contribute nothing and a token with no rows comes back zero."""
    src = torch.arange(1, 7, device="cuda", dtype=torch.float32).unsqueeze(-1).expand(6, 3).contiguous()
    inv_map = torch.tensor([[0, 3], [5, 6], [6, 6]], device="cuda")  # 6 == len(src): sentinel
    out = gather_reduce_rows(src, inv_map)
    torch.testing.assert_close(out[:, 0], torch.tensor([1.0 + 4.0, 6.0, 0.0], device="cuda"))


def test_empty_dispatch():
    inv_map = torch.zeros(0, 8, dtype=torch.long, device="cuda")
    out = MoEWeightedUnpermute.apply(
        torch.zeros(0, 16, device="cuda"),
        torch.zeros(0, device="cuda"),
        torch.zeros(0, dtype=torch.long, device="cuda"),
        inv_map,
    )
    assert out.shape == (0, 16)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
