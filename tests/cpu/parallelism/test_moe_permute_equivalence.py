"""The atomic-free MoE permute (halo/kernels/moe_permute.py) against the index_select / weighted index_add
reference, in float64, forward and backward. CPU pins the eager forms."""

import pytest
import torch

from halo.kernels.moe_permute import MoEGatherPermute, MoEWeightedUnpermute, build_inv_map
from tests.common.ep_stubs import StubEPLayerBase


def _make_routing(recv_N: int, width: int, seed: int):
    """A received token lands on at most top_k local experts. Returns (sorted_token_idx, inv_map)."""
    g = torch.Generator().manual_seed(seed)
    counts = torch.randint(0, width + 1, (recv_N,), generator=g)
    idx = torch.repeat_interleave(torch.arange(recv_N), counts)
    perm = torch.randperm(idx.numel(), generator=g)  # Simulate expert-sort ordering.
    sorted_token_idx = idx[perm].contiguous()
    inv_map = build_inv_map(sorted_token_idx, recv_N, width)
    return sorted_token_idx, inv_map


def _weighted_unpermute_and_grads(expert_out, weights, sorted_token_idx, inv_map, grad_out, *, reference):
    eo = expert_out.clone().requires_grad_(True)
    w = weights.clone().requires_grad_(True)
    if reference:
        out = torch.zeros(inv_map.shape[0], eo.shape[1], dtype=eo.dtype).index_add(
            0, sorted_token_idx, eo * w[:, None]
        )
    else:
        out = MoEWeightedUnpermute.apply(eo, w, sorted_token_idx, inv_map)
    out.backward(grad_out)
    return out.detach(), eo.grad, w.grad


def _eq(a, b, msg):
    # Mathematical, not bit, equivalence: gather+sum vs index_add differ by one float64 ULP.
    assert torch.allclose(a, b, rtol=0, atol=1e-9), f"{msg}: max|diff|={(a - b).abs().max().item():.3e}"


def _check_case(recv_N, width, H, seed):
    sorted_token_idx, inv_map = _make_routing(recv_N, width, seed)
    n_sorted = sorted_token_idx.numel()

    expert_out = torch.randn(n_sorted, H, dtype=torch.float64)
    weights = torch.rand(n_sorted, dtype=torch.float64)
    grad_out = torch.randn(recv_N, H, dtype=torch.float64)
    args = (expert_out, weights, sorted_token_idx, inv_map, grad_out)
    got = _weighted_unpermute_and_grads(*args, reference=False)
    want = _weighted_unpermute_and_grads(*args, reference=True)
    for name, g, r in zip(("fwd", "expert grad", "weight grad"), got, want, strict=True):
        _eq(g, r, f"weighted unpermute {name} recv_N={recv_N} width={width}")

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
        (64, 8, 16, 0),
        (128, 4, 8, 1),
        (32, 1, 4, 2),
        (100, 8, 5, 3),  # Tokens with gaps (some count=0) and some at the width cap.
        (256, 8, 7, 4),
        (1, 8, 3, 5),
        (50, 2, 6, 6),
    ]
    for recv_N, width, H, seed in cases:
        _check_case(recv_N, width, H, seed)
    print(f"OK test_permute_equivalence: {len(cases)} cases, fwd+bwd match index_add/index_select in float64")


def test_build_inv_map_properties():
    """The sentinel n_sorted indexes the zero pad row."""
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
    recv_N, width, H = 8, 4, 5
    sorted_token_idx = torch.empty(0, dtype=torch.long)
    inv_map = build_inv_map(sorted_token_idx, recv_N, width)
    assert inv_map.shape == (recv_N, width)
    assert torch.all(inv_map == 0)  # n_sorted == 0, so the sentinel equals 0.

    expert_out = torch.empty(0, H, dtype=torch.float64)
    got = MoEWeightedUnpermute.apply(expert_out, torch.empty(0, dtype=torch.float64), sorted_token_idx, inv_map)
    assert got.shape == (recv_N, H)
    assert torch.all(got == 0)
    print("OK test_empty_routing_is_all_sentinel")


def _saved_and_expert_grad(expert_out, weights, sorted_token_idx, inv_map, grad_out, *, weights_need_grad):
    eo = expert_out.clone().requires_grad_(True)
    w = weights.clone().requires_grad_(weights_need_grad)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        out = MoEWeightedUnpermute.apply(eo, w, sorted_token_idx, inv_map)
    out.backward(grad_out)
    return saved, eo.grad


def test_weights_needing_no_grad_keep_no_expert_outputs():
    """Otherwise an ``[n_sorted, H]`` tensor per MoE layer stays alive until backward."""
    sorted_token_idx, inv_map = _make_routing(64, 8, seed=11)
    n_sorted, hidden = sorted_token_idx.numel(), 16
    expert_out = torch.randn(n_sorted, hidden, dtype=torch.float64)
    args = (expert_out, torch.rand(n_sorted, dtype=torch.float64), sorted_token_idx, inv_map)
    grad_out = torch.randn(64, hidden, dtype=torch.float64)

    saved_full, expert_grad_full = _saved_and_expert_grad(*args, grad_out, weights_need_grad=True)
    saved, expert_grad = _saved_and_expert_grad(*args, grad_out, weights_need_grad=False)

    assert any(t.shape == expert_out.shape for t in saved_full), "the probe would not see expert_out saved"
    assert not any(t.shape == expert_out.shape for t in saved), "expert_out kept with no weight gradient"
    assert sum(t.numel() for t in saved) == sum(t.numel() for t in saved_full) - expert_out.numel()
    assert torch.equal(expert_grad, expert_grad_full)


def test_full_width_token_all_experts():
    width, H = 4, 6
    # Token 0 lands on 4 sorted positions, token 1 on none.
    sorted_token_idx = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    inv_map = build_inv_map(sorted_token_idx, recv_N=2, width=width)
    assert torch.equal(torch.sort(inv_map[0]).values, torch.arange(width))
    assert torch.all(inv_map[1] == 4)

    expert_out = torch.randn(4, H, dtype=torch.float64)
    weights = torch.rand(4, dtype=torch.float64)
    args = (expert_out, weights, sorted_token_idx, inv_map, torch.randn(2, H, dtype=torch.float64))
    got = _weighted_unpermute_and_grads(*args, reference=False)
    want = _weighted_unpermute_and_grads(*args, reference=True)
    for name, g, r in zip(("fwd", "expert grad", "weight grad"), got, want, strict=True):
        _eq(g, r, f"full-width weighted unpermute {name}")
    print("OK test_full_width_token_all_experts")


class _GroupedPathLayer(StubEPLayerBase):
    def __init__(self, ep_size: int, experts_per_rank: int):
        super().__init__()
        self.ep_size = ep_size
        self.experts_per_rank = experts_per_rank
        self._use_grouped_mm = True


@pytest.mark.parametrize(
    ("top_k", "ep_size"), ((4, 8), (2, 2), (1, 2)), ids=("top_k_below_ep_size", "top_k_at_ep_size", "top_1")
)
def test_the_grouped_path_permutes_atomic_free_at_every_top_k_and_ep_size(monkeypatch, top_k, ep_size):
    """gpt-oss's top-4 at EP8 included; dispatch padding (``-1``) and other ranks' ids are dropped."""
    calls = []
    for cls in (MoEGatherPermute, MoEWeightedUnpermute):
        apply = cls.apply
        monkeypatch.setattr(
            cls, "apply", lambda *args, _apply=apply, _name=cls.__name__: calls.append(_name) or _apply(*args)
        )
    layer = _GroupedPathLayer(ep_size, experts_per_rank=3)
    g = torch.Generator().manual_seed(top_k * 10 + ep_size)
    recv_n, hidden = 11, 5
    tokens = torch.randn(recv_n, hidden, dtype=torch.float64, generator=g)
    experts = torch.randint(-1, 5, (recv_n, top_k), generator=g)  # -1 padding; 3..4 are not this rank's.
    weights = torch.rand(recv_n, top_k, dtype=torch.float64, generator=g)
    scale = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    local_ids = experts[(experts >= 0) & (experts < 3)]
    recv_counts = tuple(torch.bincount(local_ids, minlength=3).tolist())
    out = layer._run_grouped_mm_compute(
        tokens,
        experts,
        weights,
        torch.float64,
        lambda rows, _offs, eids: rows * scale[eids, None],
        recv_counts=recv_counts,
    )

    assert calls == ["MoEGatherPermute", "MoEWeightedUnpermute"]
    local = (experts >= 0) & (experts < 3)
    rows, slots = local.nonzero(as_tuple=True)
    eids = experts[rows, slots]
    expected = torch.zeros(recv_n, hidden, dtype=torch.float64).index_add(
        0, rows, tokens[rows] * scale[eids, None] * weights[rows, slots, None]
    )
    _eq(out, expected, f"top_k={top_k} ep_size={ep_size} scatter-back")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
