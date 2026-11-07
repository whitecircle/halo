#!/usr/bin/env python
"""The packed scoring branch on a CPU Llama: a flattened micro-batch scores and trains as each row alone, positions
restart per row, a neighbour's tokens move nothing, and routing capture and replay tile the bins' spans."""

import pytest
import torch

from halo.models.segment_markers import SegmentMarkers
from halo.trainers.grpo.environmental.routing_replay import RoutingReplayInjector
from halo.trainers.grpo.microbatch_plan import MicrobatchPlan
from halo.trainers.grpo.packed_forward import build_packed_rows
from tests.common.length_deal import ragged_round, scoring_host, single_process_accelerator, stacked, tiny_llama
from tests.common.routing import BareEPLayer, dense_rows, distinct_expert_picks, ragged_routing

_VOCAB = 64
_TOL = {"rtol": 1e-4, "atol": 1e-5}
# Includes 1-token prompts and completions, whose spans the scorer widens by a neighbour.
_PROMPTS = [3, 1, 5, 2, 4, 1, 3]
_COMPLETIONS = [4, 2, 1, 5, 3, 1, 6]


def _round(seed: int = 3) -> tuple[torch.Tensor, torch.Tensor, int]:
    return stacked(ragged_round(_PROMPTS, _COMPLETIONS, vocab=_VOCAB, seed=seed))


def _host(packed: bool, seed: int = 0):
    host = scoring_host(tiny_llama(_VOCAB, seed), None, chunks=4, accelerator=single_process_accelerator())
    host._packs_microbatches = packed
    host.pad_token_id = 0
    return host


def _score(host, ids, mask, keep, entropy: bool = False):
    return host._get_per_token_logps_and_entropies(host.model, ids, mask, keep, compute_entropy=entropy)


def test_a_packed_call_scores_every_row_as_the_per_row_path_does():
    ids, mask, keep = _round()
    dense, dense_entropy, _ = _score(_host(False), ids, mask, keep, entropy=True)
    host, calls = _host(True), []
    backbone_forward = host.model.model.forward
    host.model.model.forward = lambda *args, **kwargs: calls.append(kwargs["input_ids"].shape) or backbone_forward(
        *args, **kwargs
    )
    packed, packed_entropy, _ = _score(host, ids, mask, keep, entropy=True)
    assert len(calls) == 1 and calls[0][0] == 1, f"the rows must reach the backbone as one flattened call, got {calls}"
    torch.testing.assert_close(packed, dense, **_TOL)
    torch.testing.assert_close(packed_entropy, dense_entropy, **_TOL)
    assert packed.shape == (len(_PROMPTS), keep) and dense.abs().sum() > 0


def test_a_packed_call_trains_as_the_per_row_path_does():
    ids, mask, keep = _round()
    grads = {}
    for packed in (False, True):
        host = _host(packed)
        logps, _, _ = _score(host, ids, mask, keep)
        (logps * mask[:, -keep:]).sum().backward()
        grads[packed] = [p.grad.clone() for p in host.model.parameters() if p.grad is not None]
    assert grads[True] and len(grads[True]) == len(grads[False])
    for got, want in zip(grads[True], grads[False], strict=True):
        torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-5)


def test_changing_one_row_moves_no_other_rows_scores():
    ids, mask, keep = _round()
    host = _host(True)
    before, _, _ = _score(host, ids, mask, keep)
    changed = ids.clone()
    victim = 2
    lo = int(mask[victim].nonzero()[0])
    changed[victim, lo:] = (changed[victim, lo:] + 7) % _VOCAB
    after, _, _ = _score(host, changed, mask, keep)
    others = [i for i in range(len(_PROMPTS)) if i != victim]
    torch.testing.assert_close(after[others], before[others], rtol=0, atol=1e-6)
    assert (after[victim] - before[victim]).abs().max() > 1e-3, "the edit must reach the row it changed"


def test_negative_control_positions_that_continue_across_rows_change_the_hidden_states():
    """A flattened row whose positions did not restart would read the later rows at the wrong RoPE offsets."""
    ids, mask, _ = _round()
    spans = [(int(row.nonzero()[0]), int(row.nonzero()[-1]) + 1) for row in mask]
    rows = build_packed_rows(ids, spans, [1] * len(spans), SegmentMarkers())
    continuing = torch.arange(rows.input_ids.size(1)).unsqueeze(0)
    backbone = _host(True).model.model
    with torch.no_grad():
        right = backbone(input_ids=rows.input_ids, position_ids=rows.position_ids, use_cache=False)
        wrong = backbone(input_ids=rows.input_ids, position_ids=continuing, use_cache=False)
    assert not torch.allclose(right.last_hidden_state[:, -3:], wrong.last_hidden_state[:, -3:], atol=1e-4)


def test_the_flattened_input_restarts_positions_and_indexes_each_rows_scored_positions():
    ids = torch.tensor([[0, 0, 11, 12, 13], [21, 22, 23, 24, 25], [0, 0, 0, 31, 32]])
    spans = [(2, 5), (0, 5), (3, 5)]
    rows = build_packed_rows(ids, spans, [2, 1, 1], SegmentMarkers(seq_idx=True, cu_seq_lens=True))
    assert rows.input_ids.tolist() == [[11, 12, 13, 21, 22, 23, 24, 25, 31, 32]]
    assert rows.position_ids.tolist() == [[0, 1, 2, 0, 1, 2, 3, 4, 0, 1]]
    # Each row scores the ``width`` positions before its last token: [0, 1], [7], [8].
    assert rows.hidden_index.tolist() == [0, 1, 6, 8]
    assert rows.kwargs["cu_seq_lens_q"].tolist() == [0, 3, 8, 10] and rows.kwargs["max_length_q"] == 5
    assert rows.kwargs["seq_idx"].tolist() == [[0, 0, 0, 1, 1, 1, 1, 1, 2, 2]]


def test_a_single_row_forwards_with_no_markers_as_the_per_row_path_does():
    rows = build_packed_rows(torch.tensor([[5, 6, 7]]), [(0, 3)], [2], SegmentMarkers(seq_idx=True, cu_seq_lens=True))
    assert rows.kwargs == {} and rows.position_ids.tolist() == [[0, 1, 2]]


def test_evaluation_mode_keeps_the_per_row_path():
    host = _host(True)
    host.model.eval()
    host._packed_logps = lambda *args, **kwargs: pytest.fail("an evaluation forward took the packed branch")
    ids, mask, keep = _round()
    with torch.no_grad():
        _score(host, ids, mask, keep)


_LAYERS, _TOP_K = 2, 2


def _routing_host(packed: bool):
    """A Llama whose embedding output stands in for the router: each token's routing names its own embedding, so it
    cannot depend on its neighbours."""
    host = _host(packed)
    injector = RoutingReplayInjector(
        [BareEPLayer(top_k=_TOP_K, num_experts=8) for _ in range(_LAYERS)],
        engine_layers=_LAYERS,
        layer_indices=list(range(_LAYERS)),
    )
    host._routing_injector, host._replay_row_spans = injector, True

    def route(_module, _inputs, embeddings):
        flat = embeddings.reshape(-1, embeddings.size(-1))
        for i, layer in enumerate(injector._layers):
            natural = torch.stack([(flat.argmax(-1) + i + k) % 8 for k in range(_TOP_K)], dim=-1)
            if layer._capture_routing:
                layer._captured_routing_chunks.append(natural.to(torch.int16))
            if layer._forced_topk_indices is not None:
                layer._apply_routing_replay(natural)

    host.model.model.embed_tokens.register_forward_hook(route)
    return host


_PLAN = MicrobatchPlan(bins=((6, 0, 3), (2, 5), (), (1, 4)), rows_per_bin=3, slots=4, round_rows=7)


def test_capture_through_packed_bins_equals_capture_through_per_row_forwards():
    """R2: the bins' forwards, an empty bin's inert one included, capture each row's routing into its own batch row."""
    ids, mask, keep = _round()
    order = torch.tensor(_PLAN.order)
    per_row = _routing_host(False)
    packed = _routing_host(True)
    with torch.no_grad():
        _, routing_per_row = per_row._recompute_logps_and_routing(ids, mask, keep, None, "train", order)
        _, routing_packed = packed._recompute_logps_and_routing(ids, mask, keep, None, "train", order, _PLAN)
    assert routing_per_row is not None and routing_packed is not None
    routing_packed, routing_per_row = dense_rows(routing_packed), dense_rows(routing_per_row)
    assert torch.equal(routing_packed, routing_per_row)
    assert routing_packed.shape[0] == len(_PROMPTS) and (routing_packed != -1).any()


def test_replay_across_packed_bins_consumes_exactly_the_armed_routing():
    """R3: one armed mask feeds the bins' forwards cursor-wise; the empty bin's inert forward consumes none of it, and
    disarm raises on a mis-consumed total."""
    ids, mask, keep = _round()
    order = torch.tensor(_PLAN.order)
    host = _routing_host(True)
    engine = ragged_routing(distinct_expert_picks((len(_PROMPTS), ids.size(1)), _LAYERS, _TOP_K, 8))
    with torch.no_grad():
        host._recompute_logps_and_routing(ids, mask, keep, engine, "train", order, _PLAN)
    assert not host._routing_injector._armed, "the recompute must disarm the mask it armed"
    assert all(layer._forced_topk_indices is None for layer in host._routing_injector._layers)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
