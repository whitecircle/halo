#!/usr/bin/env python
"""Hand-sliced TP shards (GptOss sinks) must reach the engine gathered, exactly once, after the stream. Streamed, the
engine gets one rank's slice under the full name; skipped with no drain, it keeps the pretrained sinks."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from halo.distributed.tensor_parallel.plain_params import hand_sliced_dims, hand_sliced_suffixes, register_hand_sliced
from halo.trainers.grpo.rollout import weight_push
from halo.trainers.grpo.rollout.weight_push import gather_and_send_weights

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
    """A policy as ``parallelize_attention`` leaves it: hand-sliced params registered with their shard dim."""

    def __init__(self, register_sinks: bool = True):
        super().__init__()
        layer = nn.Module()
        layer.self_attn = _Attention()
        self.layers = nn.ModuleList([layer])
        self.lm_head = nn.Linear(HIDDEN, HIDDEN, bias=False)
        if register_sinks:
            register_hand_sliced(self, "sinks", 0)

    def forward(self, x):  # pragma: no cover - never called
        return x


class _RecordingSender:
    """Records every forward in order, duplicates included, since sending a param twice is a failure mode here."""

    def __init__(self):
        self.sent: list[tuple[str, torch.Tensor]] = []

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        self.sent.append((name, weights.clone()))

    def finish_weight_update(self) -> None:  # pragma: no cover - the caller flushes, not the gather
        pass


def _fake_tp_drain(model: nn.Module):
    """Stand-in for ``iter_hand_sliced_full`` with the peer's slice faked in. Same contract: collective, in
    ``named_parameters`` order, yielding full tensors."""
    for name, param in model.named_parameters():
        for suffix, shard_dim in hand_sliced_dims(model).items():
            if name.endswith(suffix):
                yield name, torch.cat([param.data, param.data + PEER_OFFSET], dim=shard_dim)


def _sync(model: nn.Module, monkeypatch) -> _RecordingSender:
    monkeypatch.setattr(weight_push, "iter_hand_sliced_full", _fake_tp_drain)
    sender = _RecordingSender()
    gather_and_send_weights(model, sender)
    return sender


def test_sinks_are_forwarded_once_gathered_and_last(monkeypatch):
    model = _TPModel()
    assert hand_sliced_suffixes(model) == ("sinks",), "the model no longer registers hand-sliced sinks"

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
    """Without TP nothing is hand-sliced; a skip keyed on the suffix instead of the registration would drop the param
    from every dense run."""
    model = _TPModel(register_sinks=False)

    sent = _sync(model, monkeypatch).sent
    names = [name for name, _tensor in sent]

    assert names.count(SINK_NAME) == 1, f"the sinks never reached the engine on a non-TP run: {names}"
    assert torch.equal(dict(sent)[SINK_NAME], model.layers[0].self_attn.sinks.detach())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
