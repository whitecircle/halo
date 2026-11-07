#!/usr/bin/env python
"""Candidate log-probs off SGLang's flat base64 arrays. A parser that misreads a slot, a padding or a clamp feeds the
objective a wrong tail mass without any error, so the format is pinned against hand-built replies."""

import base64

import numpy as np
import pytest

from halo.environments.candidate_logprobs import parse_sglang_candidates
from halo.environments.engine_wire import build_payload, turn_from_chat_completion
from halo.environments.rollout_config import SGLANG_BACKEND, RolloutConfig


def _sglang_meta(ids: np.ndarray, logps: np.ndarray, null_prefix: int = 0) -> dict:
    return {
        "output_top_logprobs_shape": list(ids.shape),
        "output_top_logprobs_null_prefix": null_prefix,
        "output_top_logprobs_val_flat_b64": base64.b64encode(logps.astype("<f4").tobytes()).decode(),
        "output_top_logprobs_idx_flat_b64": base64.b64encode(ids.astype("<i4").tobytes()).decode(),
        "output_top_logprobs_val_flat_b64_dtype": "float32",
        "output_top_logprobs_idx_flat_b64_dtype": "int32",
    }


def test_sglang_candidates_decode_flat_arrays_and_honour_the_null_prefix():
    ids = np.array([[4, 2, 7], [1, 3, 9]], dtype=np.int32)
    logps = np.array([[-0.5, -1.0, -2.0], [-0.1, -1.5, np.finfo(np.float32).min]], dtype=np.float32)
    parsed = parse_sglang_candidates(_sglang_meta(ids, logps, null_prefix=1), tokens=3, k=4)
    assert parsed.ids.tolist() == [[-1] * 4, [4, 2, 7, -1], [1, 3, 9, -1]], "the unfilled leading row and the padding"
    assert np.isneginf(parsed.logps[0]).all() and np.isneginf(parsed.logps[2, 2:]).all(), "float32 min is zero mass"
    assert parsed.logps[1, :3].tolist() == pytest.approx([-0.5, -1.0, -2.0])
    assert parse_sglang_candidates(_sglang_meta(ids, logps), tokens=2, k=2).ids.tolist() == [[4, 2], [1, 3]]


def test_sglang_candidates_are_none_when_the_arrays_do_not_cover_the_tokens():
    ids, logps = np.zeros((2, 3), np.int32), np.zeros((2, 3), np.float32)
    assert parse_sglang_candidates(_sglang_meta(ids, logps), tokens=3, k=3) is None
    assert parse_sglang_candidates({}, tokens=2, k=3) is None
    wrong = _sglang_meta(ids, logps) | {"output_top_logprobs_val_flat_b64_dtype": "float64"}
    assert parse_sglang_candidates(wrong, tokens=2, k=3) is None


def _chat_reply() -> dict:
    meta = {"output_token_logprobs": [[-0.1, 5, None], [-0.2, 8, None]]} | _sglang_meta(
        np.array([[5, 6], [8, 1]], np.int32), np.array([[-0.1, -2.0], [-0.2, -3.0]], np.float32)
    )
    choice = {"message": {"content": "ab"}, "meta_info": meta, "prompt_token_ids": [1]}
    return {"choices": [choice], "usage": {"completion_tokens": 2}, "prompt_token_ids": [1]}


def test_a_turn_carries_candidates_aligned_with_its_sampled_ids_only_when_asked():
    asked = RolloutConfig(backend=SGLANG_BACKEND, capture_token_ids=True, capture_candidate_logprobs=2)
    turn = turn_from_chat_completion(_chat_reply(), asked)
    assert turn.token_ids == [5, 8]
    assert turn.candidates.ids.tolist() == [[5, 6], [8, 1]]
    np.testing.assert_allclose(turn.candidates.logps, [[-0.1, -2.0], [-0.2, -3.0]], rtol=1e-6)
    unasked = turn_from_chat_completion(_chat_reply(), RolloutConfig(backend=SGLANG_BACKEND, capture_token_ids=True))
    assert unasked.candidates is None


def test_the_request_asks_for_k_flat_candidates_and_states_the_plain_sampler():
    config = RolloutConfig(
        backend=SGLANG_BACKEND, capture_token_ids=True, capture_candidate_logprobs=16, top_p=1.0, temperature=1.0
    )
    payload = build_payload([], config)
    assert payload["logprobs"] is True and payload["top_logprobs"] == 16
    assert payload["return_flat_raw_output_top_logprobs"] is True and payload["return_meta_info"] is True
    assert "return_flat_raw_top_logprobs_b64" not in payload, "SGLang's chat route defaults to the base64 arrays"
    assert (payload["top_p"], payload["top_k"], payload["min_p"], payload["repetition_penalty"]) == (
        1.0,
        -1,
        0.0,
        1.0,
    ), "the engine fills an omitted filter from the model's generation_config.json, so each goes out stated"
    plain = build_payload([], RolloutConfig(backend=SGLANG_BACKEND, capture_token_ids=True))
    assert plain["top_logprobs"] == 0 and "return_flat_raw_output_top_logprobs" not in plain


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
