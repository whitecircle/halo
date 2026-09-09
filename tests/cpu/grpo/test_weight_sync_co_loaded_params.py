#!/usr/bin/env python
"""CPU tests for the co-load groups a weight-sync client declares.

SGLang's MLA loaders fuse ``q_a_proj`` and ``kv_a_proj_with_mqa`` from a cache local to one
``load_weights`` call, so a half arriving in a request without the other is dropped without error;
every request is one chunk, and the chunk boundary falls wherever the byte budget fills. What is
pinned: a group a boundary would split waits whole for the next chunk, on both the streamed and the
whole-payload path; a buffer holding nothing but such members sends nothing; a group the sync never
completes refuses the close rather than closing over a dropped half.

    python tests/cpu/grpo/test_weight_sync_co_loaded_params.py
"""

import pytest
import torch

import src.distributed.nccl.clients.base as base_module
from src.distributed.nccl.clients.base import chunk_by_bytes, split_co_loaded
from tests.common.weight_sync import offline_sglang_client

GROUPS = (("self_attn.q_a_proj.weight", "self_attn.kv_a_proj_with_mqa.weight"),)
# Four bf16 elements: 8 bytes each, so a budget of 16 holds two.
_BUDGET = 16


def _param() -> torch.Tensor:
    return torch.zeros(4, dtype=torch.bfloat16)


def _layer(index: int, leaf: str) -> str:
    return f"model.layers.{index}.self_attn.{leaf}.weight"


def _names(chunk) -> list[str]:
    return [name for name, _ in chunk]


def test_split_holds_only_the_members_of_an_incomplete_group():
    chunk = [
        (_layer(0, "q_a_proj"), _param()),
        (_layer(0, "q_b_proj"), _param()),
        (_layer(0, "kv_a_proj_with_mqa"), _param()),
        (_layer(1, "q_a_proj"), _param()),
    ]
    sendable, held = split_co_loaded(chunk, GROUPS)
    assert _names(sendable) == [_layer(0, "q_a_proj"), _layer(0, "q_b_proj"), _layer(0, "kv_a_proj_with_mqa")]
    assert _names(held) == [_layer(1, "q_a_proj")], "layer 1's pair is incomplete, so its half waits"


def test_a_suffix_claims_a_name_only_at_a_module_boundary():
    """``xself_attn.q_a_proj.weight`` is another module's tensor and must not join layer 0's group."""
    chunk = [("model.layers.0.xself_attn.q_a_proj.weight", _param()), (_layer(0, "q_a_proj"), _param())]
    sendable, held = split_co_loaded(chunk, GROUPS)
    assert _names(sendable) == ["model.layers.0.xself_attn.q_a_proj.weight"]
    assert _names(held) == [_layer(0, "q_a_proj")]


def test_chunk_by_bytes_moves_a_split_pair_whole_into_the_next_chunk():
    payload = [
        (_layer(0, "q_a_proj"), _param()),
        (_layer(0, "q_b_proj"), _param()),
        (_layer(0, "kv_a_proj_with_mqa"), _param()),
        (_layer(0, "o_proj"), _param()),
    ]
    plain = [_names(chunk) for chunk in chunk_by_bytes(payload, _BUDGET)]
    assert plain == [
        [_layer(0, "q_a_proj"), _layer(0, "q_b_proj")],
        [_layer(0, "kv_a_proj_with_mqa"), _layer(0, "o_proj")],
    ]

    grouped = [_names(chunk) for chunk in chunk_by_bytes(payload, _BUDGET, GROUPS)]
    assert grouped == [
        [_layer(0, "q_b_proj")],
        [_layer(0, "q_a_proj"), _layer(0, "kv_a_proj_with_mqa")],
        [_layer(0, "o_proj")],
    ], "the budget's cut fell between the halves and must move the first one across"


def test_chunk_by_bytes_refuses_a_pair_the_payload_never_completes():
    with pytest.raises(ValueError, match="never completed"):
        chunk_by_bytes([(_layer(0, "q_a_proj"), _param())], _BUDGET, GROUPS)


@pytest.fixture
def streamed(monkeypatch):
    """An SGLang client whose transport records each chunk's names; the update phases are stubbed."""
    client = offline_sglang_client()
    sent: list[list[str]] = []
    monkeypatch.setattr(base_module, "WEIGHT_SYNC_CHUNK_BYTES", _BUDGET)
    monkeypatch.setattr(client, "begin_weight_update", lambda: None)
    monkeypatch.setattr(client, "_broadcast_chunk", lambda chunk, final: sent.append(_names(chunk)))
    monkeypatch.setattr(client, "end_weight_update", lambda tail: sent.append(_names(tail)))
    return client, sent


def test_the_streamed_path_keeps_the_pair_in_one_request(streamed):
    client, sent = streamed
    for leaf in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
        client.update_named_param(_layer(0, leaf), _param())
    client.reset_prefix_cache()
    assert sent == [
        [_layer(0, "q_b_proj")],
        [_layer(0, "q_a_proj"), _layer(0, "kv_a_proj_with_mqa")],
        [_layer(0, "o_proj")],
    ]


def test_a_buffer_of_nothing_but_a_waiting_half_sends_nothing(streamed):
    client, sent = streamed
    client.update_named_param(_layer(0, "q_a_proj"), _param())
    client.flush_chunk()
    assert sent == [], "a lone half must wait for its partner, not go out as a chunk"
    assert _names(client._param_buffer) == [_layer(0, "q_a_proj")]


def test_the_close_refuses_a_pair_the_sync_never_completed(streamed):
    client, sent = streamed
    client.update_named_param(_layer(0, "q_a_proj"), _param())
    with pytest.raises(RuntimeError, match="missing their partners"):
        client.reset_prefix_cache()
    assert sent == [], "nothing may reach the engine once the close is refused"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
