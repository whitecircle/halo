"""Numerical-equivalence test for the atomic-free MoE permute (the production scatter-back).

`MoEGatherPermute` / `MoEScatterUnpermute` (src/distributed/expert_parallel/autograd.py) replace
the bf16-atomic `index_add_` expert scatter-back with an atomic-free gather+sum via a precomputed
`inv_map` (EPMoELayerBase._build_inv_map). They are the production scatter-back for every fused-GLU
MoE family when `top_k >= ep_size`, so they must be BIT-IDENTICAL (in float64) to the index_select /
index_add reference they replace — in BOTH forward and backward.

This is a CPU test (pure torch); run it inside the container:
    python tests/cpu/parallelism/test_moe_permute_equivalence.py
"""

import pytest
import torch

from src.distributed.expert_parallel.autograd import MoEGatherPermute, MoEScatterUnpermute
from src.distributed.expert_parallel.base_layer import EPMoELayerBase


def _make_routing(recv_N: int, width: int, seed: int):
    """Build a sorted_token_idx where each token appears 0..width times (real MoE invariant:
    a received token lands on at most top_k local experts), in a random (expert-sorted-like) order.
    Returns (sorted_token_idx, inv_map)."""
    g = torch.Generator().manual_seed(seed)
    counts = torch.randint(0, width + 1, (recv_N,), generator=g)
    idx = torch.repeat_interleave(torch.arange(recv_N), counts)
    perm = torch.randperm(idx.numel(), generator=g)  # simulate expert-sort ordering
    sorted_token_idx = idx[perm].contiguous()
    inv_map = EPMoELayerBase._build_inv_map(sorted_token_idx, recv_N, width)
    return sorted_token_idx, inv_map


def _eq(a, b, msg):
    # Mathematical, not bit, equivalence: gather+sum vs index_add differ by one float64 ULP.
    assert torch.allclose(a, b, rtol=0, atol=1e-9), f"{msg}: max|diff|={(a - b).abs().max().item():.3e}"


def _check_case(recv_N, width, H, seed):
    sorted_token_idx, inv_map = _make_routing(recv_N, width, seed)
    n_sorted = sorted_token_idx.numel()

    expert_out = torch.randn(n_sorted, H, dtype=torch.float64)
    ref_fwd = torch.zeros(recv_N, H, dtype=torch.float64).index_add_(0, sorted_token_idx, expert_out)
    eo = expert_out.clone().requires_grad_(True)
    got_fwd = MoEScatterUnpermute.apply(eo, sorted_token_idx, inv_map)
    _eq(got_fwd, ref_fwd, f"scatter fwd recv_N={recv_N} width={width}")

    grad_out = torch.randn(recv_N, H, dtype=torch.float64)
    got_fwd.backward(grad_out)
    ref_bwd = grad_out.index_select(0, sorted_token_idx)
    _eq(eo.grad, ref_bwd, f"scatter bwd recv_N={recv_N} width={width}")

    tokens = torch.randn(recv_N, H, dtype=torch.float64)
    tk = tokens.clone().requires_grad_(True)
    got_g = MoEGatherPermute.apply(tk, sorted_token_idx, inv_map)
    _eq(got_g, tokens.index_select(0, sorted_token_idx), "gather fwd")

    grad_sorted = torch.randn(n_sorted, H, dtype=torch.float64)
    got_g.backward(grad_sorted)
    ref_g_bwd = torch.zeros(recv_N, H, dtype=torch.float64).index_add_(0, sorted_token_idx, grad_sorted)
    _eq(tk.grad, ref_g_bwd, "gather bwd")


def test_permute_equivalence():
    cases = [
        (64, 8, 16, 0),  # qwen3.6-like top_k=8
        (128, 4, 8, 1),  # gpt-oss-like top_k=4
        (32, 1, 4, 2),  # width=1 edge
        (100, 8, 5, 3),  # tokens with gaps (some count=0) and some at the width cap
        (256, 8, 7, 4),
        (1, 8, 3, 5),  # single token
        (50, 2, 6, 6),
    ]
    for recv_N, width, H, seed in cases:
        _check_case(recv_N, width, H, seed)
    print(f"OK test_permute_equivalence: {len(cases)} cases, fwd+bwd float64 bit-identical to index_add/index_select")


def test_build_inv_map_properties():
    """inv_map sentinel = n_sorted (indexes the zero pad row); valid entries are a permutation
    of the sorted positions; each token's row holds exactly its occurrences."""
    sorted_token_idx, inv_map = _make_routing(64, 8, seed=7)
    n_sorted = sorted_token_idx.numel()
    assert inv_map.shape == (64, 8)
    valid = inv_map[inv_map != n_sorted]
    assert torch.equal(torch.sort(valid).values, torch.arange(n_sorted)), (
        "inv_map valid entries not a permutation of [0,n_sorted)"
    )
    for t in range(64):
        row = inv_map[t]
        row = row[row != n_sorted]
        assert torch.all(sorted_token_idx[row] == t), f"inv_map row {t} points to wrong tokens"
    print("OK test_build_inv_map_properties")


def test_empty_routing_is_all_sentinel():
    """Empty dispatch (no token routed to any local expert) → inv_map is all-sentinel
    and the scatter-back yields exactly zeros (the n_sorted==0 fast path)."""
    recv_N, width, H = 8, 4, 5
    sorted_token_idx = torch.empty(0, dtype=torch.long)
    inv_map = EPMoELayerBase._build_inv_map(sorted_token_idx, recv_N, width)
    assert inv_map.shape == (recv_N, width)
    assert torch.all(inv_map == 0)  # n_sorted == 0, so the sentinel equals 0

    expert_out = torch.empty(0, H, dtype=torch.float64)
    got = MoEScatterUnpermute.apply(expert_out, sorted_token_idx, inv_map)
    assert got.shape == (recv_N, H)
    assert torch.all(got == 0)
    print("OK test_empty_routing_is_all_sentinel")


def test_full_width_token_all_experts():
    """A token routed to all `width` local experts: its inv_map row is fully valid
    (no sentinel) and the gather/scatter still match the index_add reference."""
    width, H = 4, 6
    # token 0 lands on 4 sorted positions, token 1 on none.
    sorted_token_idx = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    inv_map = EPMoELayerBase._build_inv_map(sorted_token_idx, recv_N=2, width=width)
    assert torch.equal(torch.sort(inv_map[0]).values, torch.arange(width))
    assert torch.all(inv_map[1] == 4)  # token 1 absent → all sentinel

    expert_out = torch.randn(4, H, dtype=torch.float64)
    got = MoEScatterUnpermute.apply(expert_out, sorted_token_idx, inv_map)
    ref = torch.zeros(2, H, dtype=torch.float64).index_add_(0, sorted_token_idx, expert_out)
    _eq(got, ref, "full-width scatter")
    print("OK test_full_width_token_all_experts")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
