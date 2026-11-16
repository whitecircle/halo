#!/usr/bin/env python
"""Expert groups padded to whole DeepGEMM row segments (``pad_routing_groups``): each slot keeps its token, weight and
expert at its padded row, padding rows are token -1 / weight 0 / expert -1, and the permute, expert-bias gather and
weighted unpermute through the padded layout give the dense layout's outputs and gradients."""

import pytest
import torch

from halo.kernels.moe_permute import (
    MoEExpertBiasGather,
    MoEGatherPermute,
    MoEWeightedUnpermute,
    gather_padded_rows,
)
from halo.kernels.moe_route_sort import pad_routing_groups, sort_routing_by_expert

ALIGN = 128
LOCAL = 5


def _routing(counts: tuple[int, ...], top_k: int, seed: int):
    """``[T, top_k]`` local expert ids holding exactly ``counts[e]`` slots of expert ``e``, the rest DeepEP's -1."""
    g = torch.Generator().manual_seed(seed)
    slots = sum(counts)
    rows = -(-slots // top_k) + 3
    ids = torch.cat([torch.full((c,), e) for e, c in enumerate(counts)] + [torch.full((rows * top_k - slots,), -1)])
    experts = ids[torch.randperm(rows * top_k, generator=g)].view(rows, top_k)
    weights = torch.rand(rows, top_k, generator=g, dtype=torch.float32)
    return experts, weights


CASES = {
    "uneven+empty": (0, 5, 128, 129, 300),
    "already_whole": (128, 256, 0, 128, 384),
    "all_empty_but_one": (0, 0, 7, 0, 0),
}


@pytest.mark.parametrize("host_counts", [True, False], ids=["ep_counts", "ep1_readback"])
@pytest.mark.parametrize("case", CASES)
def test_each_slot_keeps_its_token_weight_and_expert_at_its_padded_row(case, host_counts):
    counts = CASES[case]
    experts, weights = _routing(counts, top_k=4, seed=0)
    routing = sort_routing_by_expert(experts, weights, LOCAL, counts)
    idx, w, eids, offs, inv = routing
    p_idx, p_w, p_eids, p_offs, p_inv = pad_routing_groups(*routing, ALIGN, counts if host_counts else None)

    segments = [-(-c // ALIGN) * ALIGN for c in counts]
    assert p_offs.tolist() == torch.tensor(segments).cumsum(0).tolist()
    assert p_idx.numel() == sum(segments) and p_offs.dtype == torch.int32
    real = inv < idx.numel()
    rows = p_inv[real]
    assert torch.equal(p_idx[rows], idx[inv[real]]), "a slot's padded row holds another token"
    assert torch.equal(p_w[rows], w[inv[real]]) and torch.equal(p_eids[rows], eids[inv[real]])
    assert (p_inv[~real] == p_idx.numel()).all(), "a dropped slot must point past the padded rows"
    pad = torch.ones(p_idx.numel(), dtype=torch.bool)
    pad[rows] = False
    assert (p_idx[pad] == -1).all() and (p_w[pad] == 0).all() and (p_eids[pad] == -1).all()
    for e, (start, end) in enumerate(zip([0, *p_offs.tolist()[:-1]], p_offs.tolist(), strict=True)):
        group = p_eids[start:end]
        assert (group[: counts[e]] == e).all() and (group[counts[e] :] == -1).all(), f"expert {e} is not in its group"


@pytest.mark.parametrize("case", CASES)
def test_the_padded_layout_permutes_and_unpermutes_as_the_dense_one(case):
    """Permute -> per-expert linear plus expert bias -> weighted unpermute: the output and the token and router-weight
    gradients equal the dense layout's bit for bit, as a padding row's zero input and zero bias leave it zero and give
    it no gradient. The bias gradient is one GEMM over every row, whose sum the zero rows reassociate: equal to fp32
    rounding."""
    counts = CASES[case]
    experts, weights = _routing(counts, top_k=4, seed=1)
    g = torch.Generator().manual_seed(2)
    tokens = torch.randn(experts.shape[0], 16, generator=g)
    proj = torch.randn(LOCAL, 16, 16, generator=g)
    bias = torch.randn(LOCAL, 16, generator=g)

    def run(padded: bool):
        t = tokens.clone().requires_grad_(True)
        w = weights.clone().requires_grad_(True)
        b = bias.clone().requires_grad_(True)
        routing = sort_routing_by_expert(experts, w, LOCAL, counts)
        if padded:
            routing = pad_routing_groups(*routing, ALIGN, counts)
        idx, sw, eids, offs, inv = routing
        x = MoEGatherPermute.apply(t, idx, inv, padded)
        groups = torch.diff(offs.long(), prepend=offs.new_zeros(1, dtype=torch.long)).tolist()
        y = torch.cat([chunk @ proj[e] for e, chunk in enumerate(x.split(groups))])
        y = y + MoEExpertBiasGather.apply(b, eids, padded)
        if padded:
            assert not y[idx < 0].any(), "a padding row must stay zero through the projection and its bias"
        out = MoEWeightedUnpermute.apply(y, sw, idx, inv)
        out.backward(torch.randn(out.shape, generator=torch.Generator().manual_seed(3)))
        return out, t.grad, w.grad, b.grad

    *dense, dense_bias = run(False)
    *padded, padded_bias = run(True)
    for name, d, p in zip(("output", "grad_tokens", "grad_weights"), dense, padded, strict=True):
        assert torch.equal(d, p), f"{name} differs between the dense and padded layouts"
    torch.testing.assert_close(padded_bias, dense_bias, rtol=1e-6, atol=1e-5)


def test_a_padding_row_takes_a_zero_bias_and_returns_it_no_gradient():
    bias = torch.randn(3, 4, requires_grad=True)
    eids = torch.tensor([0, -1, 2, -1, 1])
    out = MoEExpertBiasGather.apply(bias, eids, True)
    assert torch.equal(out[eids < 0], torch.zeros(2, 4))
    out.backward(torch.ones_like(out))
    assert torch.equal(bias.grad, torch.ones(3, 4)), "each expert's one real row is its whole gradient"


def test_a_padding_row_takes_no_gradient_from_an_unpermute_whose_first_upstream_row_is_not_finite():
    """The CPU backward reads a padding row's -1 token clamped to the first row: selected away, where a product with
    its zero mask would be NaN."""
    idx = torch.tensor([2, -1, 1, 2, -1])
    inv_map = torch.tensor([[5, 5], [2, 5], [0, 3]])
    expert_out = torch.randn(5, 4, requires_grad=True)
    weights = torch.tensor([0.5, 0.0, 0.25, 0.75, 0.0], requires_grad=True)
    upstream = torch.randn(3, 4)
    upstream[0] = float("inf")
    MoEWeightedUnpermute.apply(expert_out, weights, idx, inv_map).backward(upstream)
    assert expert_out.grad.isfinite().all() and weights.grad.isfinite().all()
    assert not expert_out.grad[idx < 0].any() and not weights.grad[idx < 0].any()
    assert torch.equal(expert_out.grad[0], 0.5 * upstream[2])


def test_the_padded_gather_gives_a_padding_row_zeros_and_copies_the_rest():
    tokens = torch.randn(3, 4)
    tokens[0, 0] = float("nan")  # the row a clamped -1 reads
    idx = torch.tensor([2, -1, 1, -1])
    out = gather_padded_rows(tokens, idx)
    assert torch.equal(out[[0, 2]], tokens[[2, 1]]) and torch.equal(out[[1, 3]], torch.zeros(2, 4))


@pytest.mark.parametrize("pad_align", [0, -128])
def test_a_non_positive_padding_is_refused(pad_align):
    experts, weights = _routing((3, 0, 5, 1, 2), top_k=2, seed=0)
    with pytest.raises(ValueError, match="pad_align must be a positive row count"):
        sort_routing_by_expert(experts, weights, LOCAL, (3, 0, 5, 1, 2), pad_align)


def test_host_counts_that_move_a_padded_group_are_refused():
    """127 and 129 slots under counts of 128 each: the total holds, and the second group would end a segment past the
    rows the counts sized."""
    experts = torch.tensor([0] * 127 + [1] * 129).view(-1, 1)
    weights = torch.rand(256, 1)
    with pytest.raises(RuntimeError, match="receive counts disagree"):
        sort_routing_by_expert(experts, weights, 2, (128, 128), ALIGN)
    sort_routing_by_expert(experts, weights, 2, (127, 129), ALIGN)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
