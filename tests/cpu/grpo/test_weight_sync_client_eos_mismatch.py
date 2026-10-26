#!/usr/bin/env python
"""``VLLMWeightSyncClient.generate`` must take its completion IDs from the server, or fail.

The request sets ``return_token_ids``, so every choice carries the sampled ``token_ids``. Rebuilding
them from the decoded text instead can differ from the sampled stream token for
token, which silently desyncs the GRPO importance-sampling ratio — a missing field has to raise.
"""

import pytest

from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient
from tests.common.weight_sync import offline_client


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _client_answering(choices, tokenize=lambda text: {"prompt": [11, 12]}[text]) -> VLLMWeightSyncClient:
    """A real client whose server answers ``/v1/completions`` with ``choices`` and tokenizes with ``tokenize``."""

    def post_once(path, **kwargs):
        assert path == "/v1/completions"
        assert kwargs["json"]["return_token_ids"] is True, "the client must ask the server for token ids"
        return _FakeResponse({"choices": choices})

    client = offline_client(VLLMWeightSyncClient)
    client._post_once = post_once
    client._tokenize_text = tokenize
    return client


def test_generate_returns_the_server_token_ids_aligned_with_the_logprobs():
    client = _client_answering(
        [
            {
                "text": "answer",
                "token_ids": [21, 22, 99],
                "finish_reason": "stop",
                "logprobs": {"token_logprobs": [-0.1, -0.2, -0.3], "tokens": ["a", "b", "<|im_end|>"]},
            }
        ]
    )

    out = client.generate(["prompt"])

    assert out["prompt_ids"] == [[11, 12]]
    assert out["completion_ids"] == [[21, 22, 99]]
    # TRL 1.6's VLLMClient contract: (batch, len, num_logprobs) with the sampled token at index 0.
    assert out["logprobs"] == [[[-0.1], [-0.2], [-0.3]]]
    assert out["logprob_token_ids"] == [[[21], [22], [99]]]


def test_generate_never_retokenizes_the_completion_text():
    """Re-tokenizing a completion is what desyncs the IS ratio; only the prompt may be tokenized."""
    tokenized = []

    def recording_tokenize(text):
        tokenized.append(text)
        return [11, 12]

    client = _client_answering(
        [
            {
                "text": "answer",
                "token_ids": [41, 42, 43],
                "finish_reason": "stop",
                "logprobs": {"token_logprobs": [-0.1, -0.2, -0.3], "tokens": ["a", "b", "<|im_end|>"]},
            }
        ],
        tokenize=recording_tokenize,
    )

    out = client.generate(["prompt"])

    assert out["completion_ids"] == [[41, 42, 43]]
    assert tokenized == ["prompt"], f"the completion text was re-tokenized: {tokenized}"


def test_generate_keeps_one_prompt_id_per_unique_prompt():
    client = _client_answering(
        [
            {
                "text": "answer_a",
                "token_ids": [21],
                "finish_reason": "length",
                "logprobs": {"token_logprobs": [-0.1], "tokens": ["a"]},
            },
            {
                "text": "answer_b",
                "token_ids": [22],
                "finish_reason": "length",
                "logprobs": {"token_logprobs": [-0.2], "tokens": ["b"]},
            },
        ]
    )

    out = client.generate(["prompt"], n=2)

    assert out["prompt_ids"] == [[11, 12]]
    assert out["completion_ids"] == [[21], [22]]


def test_generate_raises_when_the_server_omits_token_ids():
    """A server that ignores ``return_token_ids`` must stop the rollout, not be worked around:
    ids rebuilt from the decoded text can disagree with the sampled stream and silently desync the
    GRPO importance-sampling ratio, which no later gate catches."""
    client = _client_answering(
        [
            {
                "text": "answer",
                "finish_reason": "stop",
                "logprobs": {"token_logprobs": [-0.1, -0.2], "tokens": ["a", "b"]},
            }
        ]
    )

    with pytest.raises(RuntimeError, match="token_ids"):
        client.generate(["prompt"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
