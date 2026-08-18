#!/usr/bin/env python
"""Hand-sliced TP shards (GptOss sinks) must reach the engine gathered, exactly once, after the stream.

Under tensor parallelism every param the plan owns is a DTensor, which the sync materializes full one
at a time. The sinks are not: they carry no plan entry, are sliced by hand across the TP group, and
are recorded in ``model._tp_sharded_non_dtensor``. That leaves the sync two ways to be silently
wrong, both of which serve a corrupt policy with no error at sync time:

* stream them like any other param — the engine gets TP-rank 0's *slice* under the full-tensor name,
  so the served model attends with a quarter of its sinks (or the load is rejected on arrival, after
  the engine is paused and partly written);
* skip them and forget the drain — nothing is ever pushed for those slots, and the engine keeps
  generating with the pretrained sinks while the trainer moves on. Permanently off-policy, which is
  exactly the failure ``validate_weight_sync_support`` refuses the FA2 sink reset over.

So ``_send_dense_weights`` skips them by suffix in the streaming loop and re-sends them gathered from
``iter_tp_sharded_non_dtensor_full``. This pins that wiring: the sinks arrive once, at full shape,
after everything streamed. The gather itself moves its tensors to CUDA before the collective, so the
group half is the GPU tier's (``parallelism/tp``); what stands in for it here is a fake TP drain, and
the sync's own halves — the skip and the drain — are what is under test.

    python tests/cpu/grpo/test_weight_sync_tp_sink_drain.py
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.distributed.tensor_parallel.state_dict import tp_sharded_non_dtensor_suffixes
from src.trainers.grpo.rollout import weight_sync
from src.trainers.grpo.rollout.weight_sync import gather_and_send_weights

HIDDEN, NUM_HEADS, TP_SIZE = 8, 4, 2
SINK_NAME = "layers.0.self_attn.sinks"
# Makes the peer's slice distinguishable from this rank's, so a shipped local slice cannot pass.
PEER_OFFSET = 100.0


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        # This TP rank's slice of the sinks: a plain Parameter, which is what makes it invisible to
        # the DTensor gather the streaming loop runs.
        self.sinks = nn.Parameter(torch.arange(float(NUM_HEADS)))


class _TPModel(nn.Module):
    """A TP-parallelized policy as ``parallelize_attention`` leaves it: the hand-sliced params
    recorded on the model as ``(suffix, shard_dim)``."""

    def __init__(self, register_sinks: bool = True):
        super().__init__()
        layer = nn.Module()
        layer.self_attn = _Attention()
        self.layers = nn.ModuleList([layer])
        self.lm_head = nn.Linear(HIDDEN, HIDDEN, bias=False)
        if register_sinks:
            self._tp_sharded_non_dtensor = [("sinks", 0)]

    def forward(self, x):  # pragma: no cover - never called
        return x


class _RecordingSender:
    """Records every forward in order, duplicates included — sending a param twice is a failure mode
    here, so it must be observable rather than asserted away."""

    def __init__(self):
        self.sent: list[tuple[str, torch.Tensor]] = []

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        self.sent.append((name, weights.clone()))

    def reset_prefix_cache(self) -> None:  # pragma: no cover - the caller flushes, not the gather
        pass


def _fake_tp_drain(model: nn.Module):
    """Stand-in for ``iter_tp_sharded_non_dtensor_full``: the TP group's all-gather of each recorded
    hand-sliced param, with the peer rank's slice faked in. Same contract — collective, in
    ``named_parameters`` order, yielding full tensors."""
    for name, param in model.named_parameters():
        for suffix, shard_dim in getattr(model, "_tp_sharded_non_dtensor", None) or ():
            if name.endswith(suffix):
                yield name, torch.cat([param.data, param.data + PEER_OFFSET], dim=shard_dim)


def _sync(model: nn.Module, monkeypatch) -> _RecordingSender:
    monkeypatch.setattr(weight_sync, "iter_tp_sharded_non_dtensor_full", _fake_tp_drain)
    sender = _RecordingSender()
    gather_and_send_weights(model, sender)
    return sender


def test_sinks_are_forwarded_once_gathered_and_last(monkeypatch):
    model = _TPModel()
    assert tp_sharded_non_dtensor_suffixes(model) == ("sinks",), "the model no longer registers hand-sliced sinks"

    names = [name for name, _tensor in _sync(model, monkeypatch).sent]

    assert names.count(SINK_NAME) == 1, (
        f"the sinks were forwarded {names.count(SINK_NAME)} times, not once — the streaming loop and the "
        f"drain must not both send them: {names}"
    )
    assert names[-1] == SINK_NAME, f"the gathered sinks must follow every streamed param, got {names}"
    assert "lm_head.weight" in names, "anti-vacuity: the dense params must still stream"


def test_the_forwarded_sinks_are_the_full_tensor_not_this_ranks_slice(monkeypatch):
    """Shape alone would pass a same-shape slice, so the values are what is pinned."""
    model = _TPModel()
    local = model.layers[0].self_attn.sinks.detach().clone()

    sent = dict(_sync(model, monkeypatch).sent)[SINK_NAME]

    assert sent.shape == (TP_SIZE * NUM_HEADS,), f"a TP slice reached the engine under the full name: {sent.shape}"
    assert torch.equal(sent, torch.cat([local, local + PEER_OFFSET])), (
        "the forwarded sinks are not the TP-gathered tensor"
    )


def test_unregistered_sinks_still_stream(monkeypatch):
    """Anti-over-rejection: without TP nothing is hand-sliced, so the same param must stream normally
    — a skip keyed on the suffix instead of the registration would drop it from every dense run."""
    model = _TPModel(register_sinks=False)

    sent = _sync(model, monkeypatch).sent
    names = [name for name, _tensor in sent]

    assert names.count(SINK_NAME) == 1, f"the sinks never reached the engine on a non-TP run: {names}"
    assert torch.equal(dict(sent)[SINK_NAME], model.layers[0].self_attn.sinks.detach())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
