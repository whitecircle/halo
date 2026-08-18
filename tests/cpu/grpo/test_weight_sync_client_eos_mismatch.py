#!/usr/bin/env python
"""``VLLMWeightSyncClient.generate`` must take its completion IDs from the server, or fail.

The request sets ``return_token_ids``, so every choice carries the sampled ``token_ids``. Rebuilding
them from the decoded text instead (the old fallback) can differ from the sampled stream token for
token, which silently desyncs the GRPO importance-sampling ratio — a missing field has to raise.
"""

import os
import sys
import types

import pytest

from tests.common.utils import load_script_module

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _build_import_stubs() -> dict:
    """Lightweight stand-ins so ``vllm_client`` loads without real torch / src.

    Returned as a ``{name: module}`` mapping so the caller can install them into
    ``sys.modules`` and then restore the originals — leaking these globally shadows the
    real ``src`` / ``torch`` packages for every test imported afterwards.
    """
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [os.path.join(PROJECT_ROOT, "src")]
    distributed_pkg = types.ModuleType("src.distributed")
    distributed_pkg.__path__ = [os.path.join(PROJECT_ROOT, "src", "distributed")]
    nccl_pkg = types.ModuleType("src.distributed.nccl")
    nccl_pkg.__path__ = [os.path.join(PROJECT_ROOT, "src", "distributed", "nccl")]

    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = object

    class _Device:
        pass

    torch_mod.device = _Device
    nn_mod = types.ModuleType("torch.nn")
    nn_mod.Module = object
    torch_mod.nn = nn_mod
    dist_mod = types.ModuleType("torch.distributed")
    dtensor_mod = types.ModuleType("torch.distributed.tensor")
    dtensor_mod.DTensor = type("DTensor", (), {})
    dist_mod.tensor = dtensor_mod
    torch_mod.distributed = dist_mod

    nccl_communicator_mod = types.ModuleType("src.distributed.nccl.transport.pynccl")
    nccl_communicator_mod.PyNcclCommunicator = type("PyNcclCommunicator", (), {})

    packed_tensor_mod = types.ModuleType("src.distributed.nccl.transport.packed_tensor")
    packed_tensor_mod.packed_broadcast_producer = lambda *args, **kwargs: None
    packed_tensor_mod.DEFAULT_PACKED_BUFFER_SIZE_BYTES = 1024 * 1024 * 1024
    packed_tensor_mod.DEFAULT_PACKED_NUM_BUFFERS = 2

    stateless_pg_mod = types.ModuleType("src.distributed.nccl.transport.stateless_group")
    stateless_pg_mod.StatelessProcessGroup = type("StatelessProcessGroup", (), {})

    clients_pkg = types.ModuleType("src.distributed.nccl.clients")
    clients_pkg.__path__ = [os.path.join(PROJECT_ROOT, "src", "distributed", "nccl", "clients")]
    transport_pkg = types.ModuleType("src.distributed.nccl.transport")
    transport_pkg.__path__ = [os.path.join(PROJECT_ROOT, "src", "distributed", "nccl", "transport")]

    return {
        "src": src_pkg,
        "src.distributed": distributed_pkg,
        "src.distributed.nccl": nccl_pkg,
        "src.distributed.nccl.clients": clients_pkg,
        "src.distributed.nccl.transport": transport_pkg,
        "torch": torch_mod,
        "torch.nn": nn_mod,
        "torch.distributed": dist_mod,
        "torch.distributed.tensor": dtensor_mod,
        "src.distributed.nccl.transport.pynccl": nccl_communicator_mod,
        "src.distributed.nccl.transport.packed_tensor": packed_tensor_mod,
        "src.distributed.nccl.transport.stateless_group": stateless_pg_mod,
    }


def _load_client_class():
    """Load ``VLLMWeightSyncClient`` against stubbed deps, restoring ``sys.modules`` after.

    The stubs are installed only for the duration of ``exec_module`` and then removed, so
    importing this test module does not poison ``src`` / ``torch`` for the rest of the
    pytest session (full-directory collection imports every test module in one process).
    """
    stubs = _build_import_stubs()
    saved = {name: sys.modules.get(name) for name in stubs}
    # The real imports pull siblings into sys.modules under canonical names while torch is stubbed;
    # left behind, clients.base keeps a fake DTensor and later tests skip the DTensor rejection.
    preexisting = set(sys.modules)
    sys.modules.update(stubs)
    try:
        module = load_script_module("src/distributed/nccl/clients/vllm.py", "vllm_client_under_test")
        return module.VLLMWeightSyncClient
    finally:
        for name in [n for n in set(sys.modules) - preexisting if n.startswith("src.distributed.nccl")]:
            sys.modules.pop(name, None)
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


VLLMWeightSyncClient = _load_client_class()


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _make_client():
    client = object.__new__(VLLMWeightSyncClient)
    client._generation_timeout = 60.0  # set in __init__, which object.__new__ bypasses
    client.base_url = "http://stub"
    return client


def _completions_response(choices):
    def fake_post(path, **kwargs):
        assert path == "/v1/completions"
        assert kwargs["json"]["return_token_ids"] is True, "the client must ask the server for token ids"
        return _FakeResponse({"choices": choices})

    return fake_post


def test_generate_returns_the_server_token_ids_aligned_with_the_logprobs():
    client = _make_client()
    client._tokenize_text = lambda text: {"prompt": [11, 12]}[text]
    client._post_once = _completions_response(
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
    client = _make_client()
    tokenized = []

    def fake_tokenize(text):
        tokenized.append(text)
        return [11, 12]

    client._tokenize_text = fake_tokenize
    client._post_once = _completions_response(
        [
            {
                "text": "answer",
                "token_ids": [41, 42, 43],
                "finish_reason": "stop",
                "logprobs": {"token_logprobs": [-0.1, -0.2, -0.3], "tokens": ["a", "b", "<|im_end|>"]},
            }
        ]
    )

    out = client.generate(["prompt"])

    assert out["completion_ids"] == [[41, 42, 43]]
    assert tokenized == ["prompt"], f"the completion text was re-tokenized: {tokenized}"


def test_generate_keeps_one_prompt_id_per_unique_prompt():
    client = _make_client()
    client._tokenize_text = lambda text: {"prompt": [11, 12]}[text]
    client._post_once = _completions_response(
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
    client = _make_client()
    client._tokenize_text = lambda text: [11, 12]
    client._post_once = _completions_response(
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
