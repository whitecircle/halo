#!/usr/bin/env python
"""R2 keeps its captured routing ragged on the host. Whatever a forward arms from it equals what the dense
``[rows, width, layers, top_k]`` capture it replaced armed, and nothing of that size is ever allocated."""

import pytest
import torch

from halo.trainers.grpo.environmental.rollout_routing import RolloutRouting
from tests.common.routing import bare_replay_injector, distinct_expert_picks

_LAYERS, _TOP_K, _WIDTH = 3, 2, 12
# Span-trimmed rows of different lengths, one of them empty and one a single token.
_SPANS = [(3, 12), (0, 12), (7, 12), (5, 5), (2, 3), (4, 11)]
_PACKED_ORDER = [4, 1, 5, 0]
_CASES = {
    "full_rows": (None, None),
    "span_trimmed_rows": (_SPANS, None),
    "dealt_span_trimmed_rows": (_SPANS, [5, 2, 0, 4, 1, 3]),
    "packed_bin_subset": ([_SPANS[r] for r in _PACKED_ORDER], _PACKED_ORDER),
}


def _forwards(spans, order, rows):
    """``(batch row, span)`` in forward order."""
    if spans is None:
        return [(r, (0, _WIDTH)) for r in range(rows)]
    targets = range(rows) if order is None else order
    return list(zip(targets, spans, strict=True))


def _capture(spans, order, rows):
    """The new ragged capture of random forward chunks, with the chunks it was fed."""
    injector = bare_replay_injector(layers=_LAYERS, top_k=_TOP_K)
    generator = torch.Generator().manual_seed(7)
    forwards = _forwards(spans, order, rows)
    fed = [
        distinct_expert_picks((sum(hi - lo for _, (lo, hi) in forwards),), _LAYERS, _TOP_K, 16, generator)[:, i]
        for i in range(_LAYERS)
    ]
    with injector.capture(
        rows, _WIDTH, row_spans=spans, row_order=None if order is None else torch.tensor(order)
    ) as result:
        for layer, flat in zip(injector._layers, fed, strict=True):
            offset = 0
            for _, (lo, hi) in forwards:
                layer._captured_routing_chunks.append(flat[offset : offset + hi - lo])
                offset += hi - lo
    return injector, result[0], fed


def _dense_reference(fed, spans, order, rows):
    """The ``[rows, width, layers, top_k]`` mask the capture used to assemble: a ``-1`` frame scattered per layer, then
    stacked on the layer axis."""
    per_layer = []
    for flat in fed:
        if spans is None:
            per_layer.append(flat.view(rows, _WIDTH, -1))
            continue
        full = flat.new_full((rows, _WIDTH, flat.size(-1)), -1)
        offset = 0
        for r, (lo, hi) in _forwards(spans, order, rows):
            full[r, lo:hi] = flat[offset : offset + hi - lo]
            offset += hi - lo
        per_layer.append(full)
    return torch.stack(per_layer, dim=2)


def _armed(arm, spans, order, rows, group):
    """The forced selections per layer after ``arm(group_rows, attended, group_spans)`` for one forward of ``group``."""
    group_spans = [dict(_forwards(spans, order, rows))[r] for r in group]
    attended = torch.stack([(torch.arange(_WIDTH) >= lo) & (torch.arange(_WIDTH) < hi) for lo, hi in group_spans])
    injector = arm(group, attended, None if spans is None else group_spans)
    return [layer._forced_topk_indices for layer in injector._layers]


def _groups(spans, order, rows):
    """Every forwarded row alone, all of them together, and the reverse order, as scoring calls chunk them."""
    forwarded = [r for r, _ in _forwards(spans, order, rows)]
    return [[r] for r in forwarded] + [forwarded, forwarded[::-1]]


def _replays_equal(routing, injector, dense, spans, order, rows) -> bool:
    def ragged(group, attended, group_spans):
        injector.arm_rows(routing, group, attended, group_spans)
        return injector

    def reference(group, attended, group_spans):
        injector.arm(dense[group].masked_fill(~attended[:, :, None, None], -1), group_spans)
        return injector

    for group in _groups(spans, order, rows):
        want = _armed(reference, spans, order, rows, group)
        got = _armed(ragged, spans, order, rows, group)
        if not all(torch.equal(w, g) for w, g in zip(want, got, strict=True)):
            return False
    return True


@pytest.mark.parametrize("case", _CASES)
def test_the_ragged_capture_replays_what_the_dense_capture_did(case):
    spans, order = _CASES[case]
    rows = len(_SPANS) if case != "full_rows" else 4
    injector, routing, fed = _capture(spans, order, rows)
    dense = _dense_reference(fed, spans, order, rows)
    assert torch.equal(routing.pad_rows(list(range(rows)), torch.device("cpu")), dense)
    assert _replays_equal(routing, injector, dense, spans, order, rows)


def test_the_equality_check_fails_when_a_replayed_selection_is_perturbed():
    spans, order = _CASES["dealt_span_trimmed_rows"]
    injector, routing, fed = _capture(spans, order, len(_SPANS))
    dense = _dense_reference(fed, spans, order, len(_SPANS))
    routing._segments[0] = routing._segments[0].clone()
    routing._segments[0][0, 0, 0] += 1
    assert not _replays_equal(routing, injector, dense, spans, order, len(_SPANS))


def test_a_row_the_captures_never_forwarded_pads_to_natural_routing():
    spans, order = _CASES["packed_bin_subset"]
    _, routing, _ = _capture(spans, order, len(_SPANS))
    padded = routing.pad_rows([2, 3], torch.device("cpu"))
    assert (padded == -1).all()


def test_the_capture_holds_only_each_rows_tokens_and_allocates_no_round_sized_tensor(monkeypatch):
    spans, order = _CASES["span_trimmed_rows"]
    rows = len(_SPANS)
    sizes = []
    for name in ("stack", "cat", "full"):
        real = getattr(torch, name)
        monkeypatch.setattr(
            torch, name, lambda *a, _real=real, **kw: (sizes.append((out := _real(*a, **kw)).numel()), out)[1]
        )
    _, routing, _ = _capture(spans, order, rows)
    tokens = sum(hi - lo for lo, hi in spans)
    held = [s for s in routing._segments if s is not None]
    assert [s.shape for s in held] == [(hi - lo, _LAYERS, _TOP_K) for lo, hi in spans]
    assert sum(s.numel() for s in held) == tokens * _LAYERS * _TOP_K < rows * _WIDTH * _LAYERS * _TOP_K
    assert max(sizes) <= tokens * _LAYERS * _TOP_K
    assert isinstance(routing, RolloutRouting) and all(s.device.type == "cpu" for s in held)


def test_a_wrong_token_count_still_raises():
    injector = bare_replay_injector(layers=_LAYERS, top_k=_TOP_K)
    with pytest.raises(RuntimeError, match="wrong token count"), injector.capture(2, 4, row_spans=[(0, 4), (1, 3)]):
        pass


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
