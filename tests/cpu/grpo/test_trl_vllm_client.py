#!/usr/bin/env python
"""``TrlVLLMClient``: the request it sends vLLM, the ids it hands TRL, and every use TRL makes of it.

The completion ids must be the server's sampled ``token_ids``: ids rebuilt from the decoded text can
differ from the sampled stream token for token, which silently desyncs the GRPO importance-sampling
ratio, so a missing field raises. Online GRPO installs the class in place of TRL's ``VLLMClient`` and
TRL keeps the instance, so a method or keyword TRL uses that the class lacks fails only mid-run, at the
first generation or weight sync. The surface tests read TRL's own source, so a TRL bump that uses
something new fails here.
"""

import ast
import inspect
import types

import pytest
import trl.generation.vllm_generation as trl_vllm_generation

from halo.trainers.grpo.online import DistributedGRPOTrainer
from halo.trainers.grpo.rollout.trl_vllm_client import TrlVLLMClient
from tests.common.weight_sync import offline_client

_TRL_TREE = ast.parse(inspect.getsource(trl_vllm_generation))
# TRL's VLLMGeneration methods online GRPO installs its own version over: their client calls never run.
_REPLACED_BY_ONLINE_GRPO = frozenset({"sync_weights"})


class _JsonReply:
    def __init__(self, payload: dict):
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def _serve(choices: list[dict], tokens: dict[str, list[int]] | None = None):
    """A real client whose server tokenizes from ``tokens`` and answers completions with ``choices``.

    Returns the client, the texts sent to ``/tokenize`` and the ``/v1/completions`` bodies, in order.
    """
    tokens = {"prompt": [11, 12]} if tokens is None else tokens
    tokenized, bodies = [], []

    def request(method, path, *, retry, json, **_):
        if path == "/tokenize":
            assert (method, retry) == ("POST", True), "tokenizing is idempotent and rides the retrying session"
            tokenized.append(json["prompt"])
            return _JsonReply({"tokens": tokens[json["prompt"]]})
        assert (method, path, retry) == ("POST", "/v1/completions", False), (
            "a generation must go out once: a retried read would re-queue it server-side"
        )
        bodies.append(json)
        return _JsonReply({"choices": choices})

    client = offline_client(TrlVLLMClient)
    client.request = request
    return client, tokenized, bodies


def test_generate_sends_one_completions_request_carrying_every_sampling_field():
    """``generation_kwargs`` override the named fields but not ``return_token_ids``, ``logprobs`` is
    floored at 1 (the route fills ``token_logprobs`` only from 1, and GRPO asks for 0), the regex goes
    out as ``structured_outputs`` (the server keeps an unknown field such as ``guided_regex`` without
    applying it), and only text prompts go through ``/tokenize``."""
    client, tokenized, bodies = _serve([], tokens={"p": [11, 12]})

    out = client.generate(
        ["p", [3, 4]],
        n=2,
        temperature=0.7,
        top_p=0.9,
        top_k=5,
        min_p=0.1,
        repetition_penalty=1.2,
        max_tokens=8,
        logprobs=0,
        structured_outputs_regex="a+",
        generation_kwargs={"seed": 3, "top_p": 0.5, "return_token_ids": False},
    )

    assert bodies == [
        {
            "prompt": ["p", [3, 4]],
            "n": 2,
            "temperature": 0.7,
            "top_p": 0.5,
            "top_k": 5,
            "min_p": 0.1,
            "repetition_penalty": 1.2,
            "max_tokens": 8,
            "logprobs": 1,
            "seed": 3,
            "return_token_ids": True,
            "structured_outputs": {"regex": "a+"},
        }
    ]
    assert tokenized == ["p"]
    assert out["prompt_ids"] == [[11, 12], [3, 4]]


def test_off_filters_are_stated_rather_than_omitted():
    """vLLM's completions route fills an omitted filter from the model's generation_config.json, so
    TRL's defaults go out at their off values, top_k 0 being vLLM's own."""
    client, _, bodies = _serve([])

    client.generate([[1, 2]], top_k=0, min_p=0.0, repetition_penalty=1.0)

    (body,) = bodies
    assert (body["top_k"], body["min_p"], body["repetition_penalty"]) == (0, 0.0, 1.0)


def test_generate_returns_the_server_token_ids_aligned_with_the_logprobs():
    client, _, _ = _serve(
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
    # TRL's VLLMClient contract: (batch, len, num_logprobs) with the sampled token at index 0.
    assert out["logprobs"] == [[[-0.1], [-0.2], [-0.3]]]
    assert out["logprob_token_ids"] == [[[21], [22], [99]]]


def test_generate_never_retokenizes_the_completion_text():
    """Re-tokenizing a completion is what desyncs the IS ratio; only the prompt may be tokenized."""
    client, tokenized, _ = _serve(
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
    client, _, _ = _serve(
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


def test_choices_are_returned_in_index_order():
    """TRL slices the flat completion list by prompt and sample, so the server's order must not leak."""
    client, _, _ = _serve(
        [
            {"index": 1, "token_ids": [22], "logprobs": {"token_logprobs": [-0.2]}},
            {"index": 0, "token_ids": [21], "logprobs": {"token_logprobs": [-0.1]}},
        ]
    )

    out = client.generate([[1, 2]], n=2)

    assert out["completion_ids"] == [[21], [22]]
    assert out["logprobs"] == [[[-0.1]], [[-0.2]]]


def test_generate_raises_when_the_server_omits_token_ids():
    """A server that ignores ``return_token_ids`` must stop the rollout, not be worked around:
    ids rebuilt from the decoded text can disagree with the sampled stream and silently desync the
    GRPO importance-sampling ratio, which no later gate catches."""
    client, _, _ = _serve(
        [
            {
                "text": "answer",
                "finish_reason": "stop",
                "logprobs": {"token_logprobs": [-0.1, -0.2], "tokens": ["a", "b"]},
            }
        ]
    )

    with pytest.raises(RuntimeError, match="token_ids") as refusal:
        client.generate(["prompt"])
    assert f"make build-vllm, {client.ENGINE_RELEASE}" in str(refusal.value), "the remedy must name the pinned image"


def test_a_prompt_with_an_image_is_refused_before_any_request():
    """The completions route is text-only: an image would be dropped and the rollout image-blind. TRL
    processes images through ``image_features`` before it generates with the features it returns."""
    client, tokenized, bodies = _serve([])

    with pytest.raises(NotImplementedError, match="text-only"):
        client.image_features([None, [object()]])
    with pytest.raises(NotImplementedError, match="text-only"):
        client.generate(["prompt", "prompt"], features=[None, {"pixel_values": object()}])

    assert tokenized == bodies == []


def test_an_image_slot_left_empty_is_text_generation():
    client, _, bodies = _serve([])

    features = client.image_features([None])
    client.generate([[1, 2]], features=features)

    assert features == [None] and len(bodies) == 1


def test_no_prompts_sends_no_request():
    client, tokenized, bodies = _serve([])

    out = client.generate([])

    assert out == {"prompt_ids": [], "completion_ids": [], "logprobs": [], "logprob_token_ids": []}
    assert tokenized == bodies == []


def _client_references() -> tuple[set[str], set[str]]:
    """What TRL reads off ``self.vllm_client`` outside the methods online GRPO replaces: every
    attribute, and the subset it calls."""
    replaced = {
        id(node)
        for function in ast.walk(_TRL_TREE)
        if isinstance(function, ast.FunctionDef) and function.name in _REPLACED_BY_ONLINE_GRPO
        for node in ast.walk(function)
    }
    live = [node for node in ast.walk(_TRL_TREE) if id(node) not in replaced]
    read = {
        node.attr for node in live if isinstance(node, ast.Attribute) and ast.unparse(node.value) == "self.vllm_client"
    }
    called = {
        node.func.attr
        for node in live
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and ast.unparse(node.func.value) == "self.vllm_client"
    }
    return read, called


def _generate_keywords() -> set[str]:
    """Every keyword TRL passes to ``self.vllm_client.generate``, a splatted dict literal's keys included."""
    keywords = set()
    for function in ast.walk(_TRL_TREE):
        if not isinstance(function, ast.FunctionDef):
            continue
        dict_literals = {
            node.targets[0].id: node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Dict)
        }
        for node in ast.walk(function):
            if not (isinstance(node, ast.Call) and ast.unparse(node.func) == "self.vllm_client.generate"):
                continue
            assert not node.args, "TRL now passes generate positional arguments, which this walk does not bind"
            for keyword in node.keywords:
                if keyword.arg is None:
                    keywords |= {key.value for key in dict_literals[ast.unparse(keyword.value)].keys}
                else:
                    keywords.add(keyword.arg)
    return keywords


def _constructor_keywords() -> list[set[str]]:
    """The keywords of each ``VLLMClient(...)`` construction in TRL's generation module."""
    return [
        {keyword.arg for keyword in node.keywords}
        for node in ast.walk(_TRL_TREE)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "VLLMClient"
    ]


def test_every_attribute_trl_uses_is_on_the_client():
    read, called = _client_references()
    assert {"generate", "init_communicator", "image_features"} <= called, (
        f"TRL's generation module no longer reads as the server-mode client contract: {sorted(called)}"
    )
    client = offline_client(TrlVLLMClient)
    missing = sorted(name for name in read if not hasattr(client, name))
    uncallable = sorted(name for name in called if not callable(getattr(client, name, None)))
    assert not missing, f"TRL reads {missing} off its VLLMClient, which TrlVLLMClient does not have"
    assert not uncallable, f"TRL calls {uncallable} on its VLLMClient, which TrlVLLMClient does not define"


def test_the_client_takes_every_keyword_trl_generates_with():
    keywords = _generate_keywords()
    assert {"prompts", "features", "n", "max_tokens", "generation_kwargs"} <= keywords, (
        f"TRL's generate call no longer reads as the server-mode contract: {sorted(keywords)}"
    )
    inspect.signature(TrlVLLMClient.generate).bind(None, **dict.fromkeys(keywords))


def test_the_client_takes_trl_s_constructor_keywords():
    constructions = _constructor_keywords()
    assert constructions, "TRL's generation module no longer constructs a VLLMClient"
    for keywords in constructions:
        offline_client(TrlVLLMClient, **dict.fromkeys(keywords - {"base_url"}))


def test_online_grpo_replaces_each_trl_method_the_surface_walk_skips():
    """The walk may skip a TRL method's client calls only while the trainer installs its own version."""
    host = object.__new__(DistributedGRPOTrainer)
    host.vllm_generation = types.SimpleNamespace()
    host.parallelism_config = types.SimpleNamespace(has_ep_group=False, is_tp_mode=False, is_expert_tp_mode=False)

    DistributedGRPOTrainer._install_weight_sync(host)

    assert vars(trl_vllm_generation.VLLMGeneration).keys() >= _REPLACED_BY_ONLINE_GRPO
    assert vars(host.vllm_generation).keys() >= _REPLACED_BY_ONLINE_GRPO


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
