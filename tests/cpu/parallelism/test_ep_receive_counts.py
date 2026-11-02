#!/usr/bin/env python
"""At ``ep_size > 1`` the expert compute sizes its index reads off the dispatch's host-side receive counts, bit
for bit what the device-sized ``nonzero`` / ``torch.where`` give.

DeepEP's synced dispatch already reports, on the host, how many (token, expert) slots each local expert
received: V2 on its handle (``num_recv_tokens_per_expert_list``), V1 as its fourth dispatch output. The
backends hand them on, the dispatcher passes them through, and the expert compute reads nothing back from the
device to size its indices: the grouped sort compacts with a ``nonzero_static`` of their sum, the per-expert
loop selects each expert's slots with a ``nonzero_static`` of its own count. Asserted here:

- the sort's six outputs equal the ``nonzero`` sort's bit for bit: padded receives, an idle expert, rows
  whose every slot is local, and a receive with no local slot;
- every tiny family's expert compute at ``ep_size > 1``, grouped and per-expert loop, each alone and under
  expert TP, gives bit-identical outputs and gradients (tokens, routing weights, every expert parameter);
- a count that disagrees with the received ids raises (a device-side assert, synchronous on CPU), and a
  missing or mis-sized count raises before any index is read;
- each backend hands on the counts from where its DeepEP version reports them.

``tests/gpu/parallelism/ep/test_ep_receive_counts.py`` checks the counts against real DeepEP dispatch on
both backends.

Run: ``python tests/cpu/parallelism/test_ep_receive_counts.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import halo.distributed.expert_parallel.config as ep_config_module
from halo.distributed.expert_parallel.deepep.backends import _ElasticBackend, _LegacyBackend
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase, named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from tests.common.ep_reference import (
    GroupedSortLayer,
    grouped_sort_mismatches,
    nonzero_grouped_sort,
    receive_counts,
    received_expert_ids,
    where_expert_loop,
)
from tests.common.parallelism import single_process_ep_config
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

# Routing width of the family sweep's synthetic receive; the expert compute takes any width (Gemma 4's router
# sits outside its EP layer, which then carries no top_k of its own).
FAMILY_TOP_K = 4
FAMILY_CASES = tuple((family, expert_tp_size) for family in TINY_MOE_FAMILIES for expert_tp_size in (1, 2))
FAMILY_CASE_IDS = [f"{family}_etp{expert_tp_size}" for family, expert_tp_size in FAMILY_CASES]


def _assert_identical(got, want, label: str) -> None:
    mismatched = grouped_sort_mismatches(got, want)
    assert not mismatched, f"{label}: {mismatched} differ from the nonzero sort"


def _sort_both(experts: torch.Tensor, experts_per_rank: int, ep_size: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randn(experts.shape[0], 8, dtype=torch.float64, generator=g)
    weights = torch.rand(experts.shape, dtype=torch.float64, generator=g)
    layer = GroupedSortLayer(ep_size=ep_size, experts_per_rank=experts_per_rank)
    got = EPMoELayerBase._sort_tokens_for_grouped_mm(
        layer, tokens, experts, weights, recv_counts=receive_counts(experts, experts_per_rank)
    )
    return got, nonzero_grouped_sort(layer, tokens, experts, weights)


@pytest.mark.parametrize(
    ("top_k", "experts_per_rank", "ep_size"),
    ((4, 4, 8), (8, 32, 8), (2, 3, 2), (1, 2, 2), (6, 4, 4)),
    ids=("gpt_oss_ep8", "qwen3_6_ep8", "top2_ep2", "top1_ep2", "top_k_over_local"),
)
def test_the_counted_compaction_sorts_bit_identically(top_k, experts_per_rank, ep_size):
    experts = received_expert_ids(64, top_k, experts_per_rank, ep_size, seed=top_k * 100 + ep_size)
    # A received top-1 token's one expert is local, so only top_k > 1 carries padding.
    assert (experts < 0).any() == (top_k > 1)
    got, want = _sort_both(experts, experts_per_rank, ep_size)
    _assert_identical(got, want, f"top_k={top_k} experts_per_rank={experts_per_rank} ep_size={ep_size}")


def test_an_idle_expert_and_full_rows_sort_bit_identically():
    """Expert 1 receives nothing, the second row is local in every slot, the rest are padded."""
    experts = torch.tensor([[0, -1, 2], [2, 0, 3], [-1, 3, -1], [-1, -1, 0]])
    assert receive_counts(experts, 4)[1] == 0
    got, want = _sort_both(experts, experts_per_rank=4, ep_size=2)
    _assert_identical(got, want, "idle expert")


def test_a_receive_with_no_local_slot_sorts_to_nothing():
    experts = torch.full((3, 2), -1)
    got, want = _sort_both(experts, experts_per_rank=2, ep_size=2)
    _assert_identical(got, want, "no local slot")
    assert got[2].numel() == 0


def test_ep_size_one_needs_no_counts():
    """At ep_size 1 nothing was dispatched and every id is local: no counts, no compaction."""
    experts = torch.randint(0, 4, (16, 2))
    tokens = torch.randn(16, 8, dtype=torch.float64)
    weights = torch.rand(16, 2, dtype=torch.float64)
    layer = GroupedSortLayer(ep_size=1, experts_per_rank=4)
    got = EPMoELayerBase._sort_tokens_for_grouped_mm(layer, tokens, experts, weights, recv_counts=None)
    _assert_identical(got, nonzero_grouped_sort(layer, tokens, experts, weights), "ep_size 1")


@pytest.mark.parametrize("shift", (1, -1), ids=("one_too_many", "one_too_few"))
def test_a_count_that_disagrees_with_the_received_ids_raises(shift):
    """A wrong total would truncate the compaction or pad it with -1, which gathers the last slot."""
    experts = received_expert_ids(32, 4, 4, 8, seed=3)
    counts = list(receive_counts(experts, 4))
    counts[0] += shift
    layer = GroupedSortLayer(ep_size=8, experts_per_rank=4)
    tokens = torch.randn(experts.shape[0], 8)
    with pytest.raises(RuntimeError, match="receive counts disagree"):
        EPMoELayerBase._sort_tokens_for_grouped_mm(
            layer, tokens, experts, torch.rand(experts.shape), recv_counts=tuple(counts)
        )


@pytest.mark.parametrize("recv_counts", (None, (1, 2)), ids=("missing", "wrong_length"))
def test_missing_or_mis_sized_counts_raise_at_ep_size_above_one(recv_counts):
    """The mis-sized counts still sum to the three local slots, so only the length check refuses them."""
    layer = GroupedSortLayer(ep_size=2, experts_per_rank=4)
    experts = torch.tensor([[0, -1], [1, 2]])
    with pytest.raises(RuntimeError, match="needs the dispatch's per-local-expert receive counts"):
        EPMoELayerBase._sort_tokens_for_grouped_mm(
            layer, torch.randn(2, 8), experts, torch.rand(2, 2), recv_counts=recv_counts
        )


def _ep_layer(family: str, *, grouped: bool, expert_tp_size: int = 1) -> EPMoELayerBase:
    """``family``'s first EP layer on one rank owning every expert (CPU runs the grouped path on the
    ``grouped_mm`` fallback), then marked ``ep_size=2`` so its expert compute reads counts like a dispatching
    rank's. Under expert TP the layer holds ETP rank 0's shard; the compute itself never reaches the ETP group,
    which a stand-in therefore fills."""
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_FAMILIES[family])
    config = single_process_ep_config(
        use_grouped_gemm=grouped,
        expert_tp_size=expert_tp_size,
        stage_world_size=expert_tp_size,
        nvlink_domain_size=expert_tp_size,
    )
    if expert_tp_size > 1:
        config.expert_tp_group = object()
    with patch.object(ep_config_module, "supports_grouped_mm", lambda: True):
        apply_ep_layers(model, config)
    layer = named_ep_layers(model)[0][1]
    assert layer.expert_tp_size == expert_tp_size
    assert layer._grouped_mm_enabled() == grouped, f"{family}: not on the expert-compute path this test exercises"
    layer.ep_size = 2
    return layer


def _family_inputs(layer: EPMoELayerBase):
    """A padded receive for ``layer`` (rank 0 of two), its tokens, routing weights, upstream grad and counts."""
    dtype = next(p.dtype for _, p in layer.expert_named_params())
    # A receive of every global expert hands each local one every token, and equal counts cannot show a count
    # read off the wrong expert.
    top_k = min(FAMILY_TOP_K, 2 * layer.experts_per_rank - 1)
    experts = received_expert_ids(48, top_k, layer.experts_per_rank, 2, seed=11)
    g = torch.Generator().manual_seed(5)
    tokens = torch.randn(experts.shape[0], layer.hidden_dim, generator=g).to(dtype)
    weights = torch.rand(experts.shape, generator=g)
    grad_out = torch.randn(experts.shape[0], layer.hidden_dim, generator=g).to(dtype)
    return tokens, experts, weights, grad_out, receive_counts(experts, layer.experts_per_rank)


def _expert_step(layer: EPMoELayerBase, compute, tokens, weights, grad_out):
    """One ``compute(tokens, weights)`` and its backward: output and every gradient it reaches."""
    layer.zero_grad(set_to_none=True)
    x = tokens.clone().requires_grad_(True)
    w = weights.clone().requires_grad_(True)
    out = compute(x, w)
    out.backward(grad_out)
    grads = {name: param.grad for name, param in layer.expert_named_params()}
    return out.detach(), x.grad, w.grad, grads


def _assert_steps_identical(family: str, got, want) -> None:
    for name, g_val, w_val in zip(("output", "token grad", "weight grad"), got[:3], want[:3], strict=True):
        assert torch.equal(g_val, w_val), f"{family}: {name} differs"
    assert got[3].keys() == want[3].keys()
    for name, grad in got[3].items():
        assert grad is not None and grad.abs().sum() > 0, f"{family}: {name} got no gradient"
        assert torch.equal(grad, want[3][name]), f"{family}: {name} gradient differs"


@pytest.mark.parametrize(("family", "expert_tp_size"), FAMILY_CASES, ids=FAMILY_CASE_IDS)
def test_every_family_computes_bit_identically_from_the_counts(family, expert_tp_size, monkeypatch):
    """The grouped sort compacts off the counts, on every family's grouped path, its expert-TP shard included,
    and matches the ``nonzero`` sort bit for bit through the whole expert compute."""
    layer = _ep_layer(family, grouped=True, expert_tp_size=expert_tp_size)
    tokens, experts, weights, grad_out, counts = _family_inputs(layer)

    counted_sort = EPMoELayerBase._sort_tokens_for_grouped_mm
    sorts: list[str] = []

    def spy(self, tokens, experts, weights, *, recv_counts):
        sorts.append("counted")
        return counted_sort(self, tokens, experts, weights, recv_counts=recv_counts)

    def reference(self, tokens, experts, weights, *, recv_counts):
        sorts.append("nonzero")
        return nonzero_grouped_sort(self, tokens, experts, weights)

    def compute(x, w):
        return layer._compute_experts(x, experts, w, tokens.dtype, recv_counts=counts)

    monkeypatch.setattr(EPMoELayerBase, "_sort_tokens_for_grouped_mm", spy)
    got = _expert_step(layer, compute, tokens, weights, grad_out)
    monkeypatch.setattr(EPMoELayerBase, "_sort_tokens_for_grouped_mm", reference)
    want = _expert_step(layer, compute, tokens, weights, grad_out)

    assert sorts == ["counted", "nonzero"], f"{family}: the grouped sort did not run once per step: {sorts}"
    _assert_steps_identical(family, got, want)


@pytest.mark.parametrize(("family", "expert_tp_size"), FAMILY_CASES, ids=FAMILY_CASE_IDS)
def test_every_family_loop_computes_bit_identically_from_the_counts(family, expert_tp_size, monkeypatch):
    """The per-expert loop sizes each expert's slot read off its count, one ``nonzero_static`` per local expert,
    and matches the ``torch.where`` loop bit for bit."""
    layer = _ep_layer(family, grouped=False, expert_tp_size=expert_tp_size)
    tokens, experts, weights, grad_out, counts = _family_inputs(layer)
    assert 0 not in counts, f"{family}: an idle expert would leave its parameter gradient unchecked"
    assert len(set(counts)) > 1, f"{family}: equal counts would hide a count read off the wrong expert"

    static_sizes: list[int] = []
    nonzero_static = torch.nonzero_static

    def spy(mask, *, size, **kwargs):
        static_sizes.append(size)
        return nonzero_static(mask, size=size, **kwargs)

    monkeypatch.setattr(torch, "nonzero_static", spy)
    got = _expert_step(
        layer,
        lambda x, w: layer._compute_experts(x, experts, w, tokens.dtype, recv_counts=counts),
        tokens,
        weights,
        grad_out,
    )
    assert static_sizes == list(counts), f"{family}: the loop did not size its reads off the counts"
    want = _expert_step(
        layer, lambda x, w: where_expert_loop(layer, x, experts, w, tokens.dtype), tokens, weights, grad_out
    )
    assert static_sizes == list(counts), f"{family}: the torch.where reference read the counts"
    _assert_steps_identical(family, got, want)


def _loop_layer_and_receive():
    layer = _ep_layer("qwen3_moe", grouped=False)
    tokens, experts, weights, _, counts = _family_inputs(layer)
    return layer, tokens, experts, weights, counts


@pytest.mark.parametrize("change", ("one_too_many", "one_too_few", "same_total_reversed"))
def test_a_loop_count_that_disagrees_with_the_received_ids_raises(change):
    """A per-expert count that is wrong but keeps the total still drops one expert's slots and pads another's
    with -1, which indexes the last row."""
    layer, tokens, experts, weights, counts = _loop_layer_and_receive()
    counts = list(counts)
    if change == "same_total_reversed":
        assert counts != counts[::-1]
        counts.reverse()
    else:
        counts[0] += 1 if change == "one_too_many" else -1
    with pytest.raises(RuntimeError, match="receive counts disagree"):
        layer._compute_experts(tokens, experts, weights, tokens.dtype, recv_counts=tuple(counts))


@pytest.mark.parametrize("recv_counts", (None, (1, 2)), ids=("missing", "wrong_length"))
def test_the_loop_refuses_missing_or_mis_sized_counts(recv_counts):
    layer, tokens, experts, weights, _ = _loop_layer_and_receive()
    with pytest.raises(RuntimeError, match="needs the dispatch's per-local-expert receive counts"):
        layer._compute_experts(tokens, experts, weights, tokens.dtype, recv_counts=recv_counts)


class _ElasticHandle:
    def __init__(self, counts: list[int]):
        self.num_recv_tokens_per_expert_list = counts


def test_the_elastic_backend_hands_on_the_handles_counts():
    """V2 reports the counts on its ``EPHandle``."""
    backend = _ElasticBackend(SimpleNamespace(hidden_dim=256, num_experts=8))
    handle = _ElasticHandle([3, 0, 2, 5])
    recv = (torch.zeros(4, 256), torch.zeros(4, 2, dtype=torch.long), torch.zeros(4, 2))
    buffer = SimpleNamespace(dispatch=lambda x, **kwargs: (*recv, handle, None))
    backend._arena = SimpleNamespace(buffer=buffer, capacity=256)
    *_, got_handle, counts = backend.dispatch_forward(torch.zeros(4, 256), recv[1], recv[2])
    assert got_handle is handle
    assert counts == (3, 0, 2, 5) and isinstance(counts, tuple)


def test_the_legacy_backend_hands_on_its_fourth_dispatch_output():
    """V1 returns ``(recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event)``."""
    backend = _LegacyBackend(SimpleNamespace(hidden_dim=64, num_experts=8))
    recv = (torch.zeros(4, 64), torch.zeros(4, 2, dtype=torch.long), torch.zeros(4, 2))
    buffer = SimpleNamespace(
        get_dispatch_layout=lambda topk_idx, num_experts: (None, None, None, None, None),
        dispatch=lambda x, **kwargs: (*recv, [1, 4, 0, 2], "v1-handle", None),
    )
    backend._arena = SimpleNamespace(buffer=buffer)
    *_, handle, counts = backend.dispatch_forward(torch.zeros(4, 64), recv[1], recv[2])
    assert handle == "v1-handle"
    assert counts == (1, 4, 0, 2) and isinstance(counts, tuple)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
