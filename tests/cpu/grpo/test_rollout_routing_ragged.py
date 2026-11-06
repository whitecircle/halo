#!/usr/bin/env python
"""R3 keeps a round's engine routing ragged on the host and pads only the rows a scoring call forwards. Each forward
layout (dense per-row, padded micro-batches, packed bins) arms exactly the slice the dense, whole-round mask would hold
for its rows, and no call pads more rows than it forwards."""

import pytest
import torch

from halo.trainers.grpo.environmental.rollout_routing import (
    ROLLOUT_COVERAGE_SHAPES,
    RolloutRouting,
    build_rollout_routing,
)
from halo.trainers.grpo.environmental.routing_replay import RoutingReplayInjector
from halo.trainers.grpo.logprob_scoring import dense_row_spans
from halo.trainers.grpo.microbatch_plan import MicrobatchPlan
from tests.common.length_deal import ragged_round, scoring_host, single_process_accelerator, stacked
from tests.common.routing import BareEPLayer, distinct_expert_picks

_LAYERS, _TOP_K, _EXPERTS, _ROWS = 2, 2, 8, 10
_CPU = torch.device("cpu")
_PROMPTS = [3, 5, 1, 4, 2, 5, 3, 1, 4, 2]
_COMPLETIONS = [4, 2, 6, 3, 5, 1, 6, 2, 4, 3]
_BATCH_SIZE = 3


def _engine_mask(tokens: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return distinct_expert_picks((tokens,), _LAYERS, _TOP_K, _EXPERTS, generator)


def _turn_masks() -> list[tuple[torch.Tensor, int | None] | None]:
    """Every convention a row can fall into: full, engine omits its last token, completion-only, a drifted prompt count,
    no routing, and one that matches none."""
    turns: list[tuple[torch.Tensor, int | None] | None] = []
    for i, (p, c) in enumerate(zip(_PROMPTS, _COMPLETIONS, strict=True)):
        kind = i % 6
        if kind == 0:
            turns.append((_engine_mask(p + c, i), p))
        elif kind == 1:
            turns.append((_engine_mask(p + c - 1, i), p))
        elif kind == 2:
            turns.append((_engine_mask(c, i), None))
        elif kind == 3:
            turns.append((_engine_mask(p + c, i), p + 1))
        elif kind == 4:
            turns.append(None)
        else:
            turns.append((_engine_mask(p + c + 5, i), p))
    return turns


def _dense_reference(turns, prompt_lens, completion_lens, max_prompt, max_completion) -> torch.Tensor:
    """The whole-round ``[rows, P + C, layers, top_k]`` mask the dense path built before the routing went ragged."""
    out = torch.full((len(turns), max_prompt + max_completion, _LAYERS, _TOP_K), -1, dtype=torch.int16)
    for i, turn in enumerate(turns):
        if turn is None:
            continue
        mask, engine_p = turn
        p, c = prompt_lens[i], completion_lens[i]
        ep = engine_p if engine_p is not None else p
        tokens = mask.shape[0]
        if tokens in (ep + c, ep + c - 1):
            covered = min(tokens - ep, c)
            out[i, max_prompt : max_prompt + covered] = mask[ep : ep + covered]
            if ep == p:
                out[i, max_prompt - p : max_prompt] = mask[:p]
        elif tokens in (c, c - 1):
            out[i, max_prompt : max_prompt + tokens] = mask
    return out


@pytest.fixture(name="round_routing")
def _round_routing() -> tuple[RolloutRouting, torch.Tensor, dict[str, int]]:
    turns = _turn_masks()
    routing, stats = build_rollout_routing(
        turns, _PROMPTS, _COMPLETIONS, max(_PROMPTS), max(_COMPLETIONS), _LAYERS, _TOP_K, _EXPERTS
    )
    dense = _dense_reference(turns, _PROMPTS, _COMPLETIONS, max(_PROMPTS), max(_COMPLETIONS))
    return routing, dense, stats


def test_every_coverage_convention_is_present_and_pads_to_the_dense_rows(round_routing):
    routing, dense, stats = round_routing
    assert all(stats[shape] > 0 for shape in ROLLOUT_COVERAGE_SHAPES) and stats["prompt_len_mismatch"] > 0
    assert torch.equal(routing.pad_rows(range(_ROWS), _CPU), dense)


def test_padding_a_subset_follows_the_requested_order_and_a_negative_row_is_natural(round_routing):
    routing, dense, _ = round_routing
    rows = [7, -1, 0, 3]
    padded = routing.pad_rows(rows, _CPU)
    assert padded.shape[0] == len(rows)
    assert torch.equal(padded[0], dense[7]) and torch.equal(padded[2], dense[0]) and torch.equal(padded[3], dense[3])
    assert (padded[1] == -1).all()
    with pytest.raises(IndexError, match="out of range"):
        routing.pad_rows([_ROWS], _CPU)


def test_the_ragged_routing_holds_no_padding(round_routing):
    """Its host footprint is the covered tokens alone, below the dense round's padded rows."""
    routing, dense, _ = round_routing
    held = sum(seg.numel() for seg in routing._segments if seg is not None)
    assert held == int((dense[..., 0, 0] >= 0).sum()) * _LAYERS * _TOP_K
    assert held < dense.numel()


class _Calls:
    """What every scoring call saw armed: per call, each layer's forced indices (``None`` under a pause)."""

    def __init__(self) -> None:
        self.armed: list[list[torch.Tensor | None]] = []
        self.rows_padded: list[int] = []


def _host(calls: _Calls, monkeypatch, *, dense_spans: bool):
    injector = RoutingReplayInjector(
        [BareEPLayer(top_k=_TOP_K, num_experts=_EXPERTS) for _ in range(_LAYERS)],
        engine_layers=_LAYERS,
        layer_indices=list(range(_LAYERS)),
    )
    host = scoring_host(None, None, chunks=4, accelerator=single_process_accelerator())
    host.args.per_device_train_batch_size = _BATCH_SIZE
    host.pad_token_id = 0
    host._routing_injector, host._rollout_routing_replay, host._replay_row_spans = injector, True, dense_spans

    def score(_model, ids, mask, keep, compute_entropy=False):
        """A scorer that consumes what the call armed, as the EP layers' dispatch does."""
        tokens = sum(hi - lo for lo, hi in dense_row_spans(mask)[0]) if dense_spans else ids.numel()
        calls.armed.append([layer._forced_topk_indices for layer in injector._layers])
        for layer in injector._layers:
            if layer._forced_topk_indices is not None:
                layer._apply_routing_replay(torch.zeros(tokens, _TOP_K, dtype=torch.long))
        return torch.zeros(ids.size(0), keep), None, None

    host._get_per_token_logps_and_entropies = score
    real = RolloutRouting.pad_rows

    def counting(self, rows, device):
        calls.rows_padded.append(len(rows))
        return real(self, rows, device)

    monkeypatch.setattr(RolloutRouting, "pad_rows", counting)
    return host, injector


def _expected(dense, rows, attended, spans):
    """What the dense path arms for ``rows`` forwarded together: attended positions, trimmed to the spans."""
    masks = dense[rows].masked_fill(~attended.bool()[:, :, None, None], -1)
    if spans is None:
        return [masks[:, :, i].reshape(-1, _TOP_K).long() for i in range(_LAYERS)]
    return [torch.cat([masks[j, lo:hi, i] for j, (lo, hi) in enumerate(spans)]).long() for i in range(_LAYERS)]


def _round_tensors():
    batch = ragged_round(_PROMPTS, _COMPLETIONS, vocab=50, seed=3)
    return stacked(batch)


def _check_calls(calls, dense, mask, forwarded, groups, dense_spans):
    all_spans = dense_row_spans(mask[forwarded])[0]
    assert len(calls.armed) == len(groups)
    for armed, group in zip(calls.armed, groups, strict=True):
        rows = [forwarded[p] for p in group]
        spans = [all_spans[p] for p in group] if dense_spans else None
        want = _expected(dense, rows, mask[rows], spans)
        for i, layer_armed in enumerate(armed):
            assert torch.equal(layer_armed, want[i]), f"layer {i}, rows {rows}"


@pytest.mark.parametrize("dealt", [False, True], ids=["batch_order", "dealt"])
@pytest.mark.parametrize("dense_spans", [True, False], ids=["per_row", "padded"])
def test_each_micro_batch_arms_the_slice_the_dense_mask_holds_for_its_rows(
    round_routing, monkeypatch, dense_spans, dealt
):
    routing, dense, _ = round_routing
    ids, mask, keep = _round_tensors()
    order = torch.tensor([7, 2, 9, 0, 4, 1, 8, 3, 6, 5]) if dealt else None
    calls = _Calls()
    host, _ = _host(calls, monkeypatch, dense_spans=dense_spans)
    host._recompute_logps_and_routing_masks(ids, mask, keep, routing, "train", order)
    forwarded = list(range(_ROWS)) if order is None else order.tolist()
    groups = [range(s, min(s + _BATCH_SIZE, _ROWS)) for s in range(0, _ROWS, _BATCH_SIZE)]
    _check_calls(calls, dense, mask, forwarded, groups, dense_spans)
    # The memory bound: no call pads more than its micro-batch, and the whole round is never padded at once.
    assert max(calls.rows_padded) <= _BATCH_SIZE < _ROWS
    assert sum(calls.rows_padded) == _ROWS


def test_each_packed_bin_arms_the_slice_the_dense_mask_holds_for_its_rows(round_routing, monkeypatch):
    routing, dense, _ = round_routing
    ids, mask, keep = _round_tensors()
    plan = MicrobatchPlan(bins=((6, 0, 3), (2, 5), (), (1, 4, 9)), rows_per_bin=3, slots=4, round_rows=_ROWS)
    calls = _Calls()
    host, _ = _host(calls, monkeypatch, dense_spans=True)
    host._recompute_logps_and_routing_masks(ids, mask, keep, routing, "train", torch.tensor(plan.order), plan)
    # The empty bin's inert forward runs paused, with nothing armed.
    assert [all(a is None for a in armed) for armed in calls.armed] == [False, False, True, False]
    real_calls = [armed for armed in calls.armed if armed[0] is not None]
    calls.armed = real_calls
    groups, start = [], 0
    for size in (3, 2, 3):
        groups.append(range(start, start + size))
        start += size
    _check_calls(calls, dense, mask, plan.order, groups, True)
    assert max(calls.rows_padded) <= plan.rows_per_bin and sum(calls.rows_padded) == len(plan.order)


def test_a_scoring_call_leaves_nothing_armed_and_a_mis_tiled_call_raises(round_routing, monkeypatch):
    routing, _, _ = round_routing
    ids, mask, keep = _round_tensors()
    calls = _Calls()
    host, injector = _host(calls, monkeypatch, dense_spans=True)
    host._recompute_logps_and_routing_masks(ids, mask, keep, routing, "train", None)
    assert not injector._armed and all(layer._forced_topk_indices is None for layer in injector._layers)

    def forward_nothing(_model, ids, mask, keep, compute_entropy=False):
        return torch.zeros(ids.size(0), keep), None, None

    host._get_per_token_logps_and_entropies = forward_nothing
    with pytest.raises(RuntimeError, match="mis-consumed"):
        host._recompute_logps_and_routing_masks(ids, mask, keep, routing, "train", None)


@pytest.mark.parametrize("bad", ["repeated", "out_of_range"])
def test_a_mask_deepep_would_fault_on_is_refused_at_build(bad):
    mask = _engine_mask(8, 0)
    if bad == "repeated":
        mask[2, 1, 1] = mask[2, 1, 0]
    else:
        mask[2, 1, 1] = _EXPERTS
    with pytest.raises(ValueError, match="twice|outside"):
        build_rollout_routing([(mask, 4)], [4], [4], 4, 4, _LAYERS, _TOP_K, _EXPERTS)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
