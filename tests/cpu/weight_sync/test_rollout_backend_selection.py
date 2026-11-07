#!/usr/bin/env python
"""Rollout-backend selection: registry, config gating and the engine-specific request payload. SGLang ignores unknown
fields and its OpenAI layer drops the token id, so a misconfigured run would quietly train on re-tokenized text."""

import logging
import re
from typing import get_args, get_type_hints

import pytest

import halo.environments.engine_wire as engine_wire
from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.environments.engine_wire import capture_generation_tokens, turn_from_chat_completion
from halo.environments.ray_actors import EnvironmentActor
from halo.environments.rollout_config import RolloutConfig
from halo.trainers.grpo.rollout.weight_push import reject_unsyncable_model
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from halo.weight_sync.registry import resolve_weight_sync_client, rollout_backends
from tests.common.rosters import ep_layer_class_by_model_type
from tests.common.weight_sync import StockModel, offline_client


def test_registry_resolves_every_backend_and_rejects_unknown():
    assert rollout_backends() == ["sglang", "vllm"]
    assert resolve_weight_sync_client("vllm") is VLLMWeightSyncClient
    assert resolve_weight_sync_client("sglang") is SGLangWeightSyncClient
    with pytest.raises(ValueError, match="Unknown rollout backend"):
        resolve_weight_sync_client("tensorrt")


def test_a_client_defined_outside_the_package_cannot_capture_a_backend_key():
    """``__subclasses__()`` returns every class that ever subclassed the base, so without the registry's module filter
    a test double silently takes over its ``BACKEND_KEY``."""

    class Impostor(VLLMWeightSyncClient):
        BACKEND_KEY = "vllm"

    assert resolve_weight_sync_client("vllm") is VLLMWeightSyncClient
    assert rollout_backends() == ["sglang", "vllm"]


def test_sglang_clients_default_to_distinct_group_names_per_server():
    """c10d registers group names process-globally, so the default is keyed on the server endpoint. Group port 0 is one
    value for every auto-port client."""
    a = offline_client(SGLangWeightSyncClient, "http://localhost:30000", group_port=0)
    b = offline_client(SGLangWeightSyncClient, "http://localhost:30001", group_port=0)
    named = offline_client(SGLangWeightSyncClient, "http://localhost:30000", group_port=51216, group_name="custom")
    assert a.group_name != b.group_name, "two auto-port clients collided on one default group name"
    assert "localhost:30000" in a.group_name and "localhost:30001" in b.group_name
    assert named.group_name == "custom"


def test_each_engine_pins_the_families_its_loader_cannot_take():
    """Moving a family out means its loader was shown to take the online update in an end-to-end run."""
    assert set(SGLangWeightSyncClient.UNSERVABLE_MODEL_TYPES) == {
        "mistral4",
        "bailing_hybrid",
        "bailing_moe_linear",
        "zaya",
        "laguna",
        "step3p7",
        "step3p5",
        "deepseek_v4",
    }
    assert set(VLLMWeightSyncClient.UNSERVABLE_MODEL_TYPES) == {
        "zaya",
        "mistral4",
        "deepseek_v4",
        "bailing_hybrid",
        "bailing_moe_linear",
    }


def test_every_unservable_spelling_is_a_family_the_toolkit_trains():
    """An entry for a spelling no EP class claims refuses nothing, and hides a rename."""
    known = set(ep_layer_class_by_model_type())
    for client in (SGLangWeightSyncClient, VLLMWeightSyncClient):
        unknown = set(client.UNSERVABLE_MODEL_TYPES) - known
        assert not unknown, f"{client.__name__} lists model types no EP family claims: {sorted(unknown)}"


@pytest.mark.parametrize("client", [SGLangWeightSyncClient, VLLMWeightSyncClient])
def test_every_unservable_entry_refuses_its_family_with_its_loader_fact(client):
    for model_type, reason in client.UNSERVABLE_MODEL_TYPES.items():
        with pytest.raises(ValueError, match=re.escape(reason)):
            reject_unsyncable_model(StockModel(model_type), client.BACKEND_KEY)


def test_sglang_declares_the_fused_a_proj_halves_as_one_request():
    """SGLang's MLA loaders drop a half that arrives without the other, so a chunk boundary must not separate them."""
    assert SGLangWeightSyncClient.CO_LOADED_PARAM_GROUPS == (
        ("self_attn.q_a_proj.weight", "self_attn.kv_a_proj_with_mqa.weight"),
    )
    assert VLLMWeightSyncClient.CO_LOADED_PARAM_GROUPS == (), "vLLM's layerwise reload assembles a layer itself"


@pytest.mark.parametrize("backend", rollout_backends())
def test_each_client_declares_whether_the_pause_it_sends_aborts_the_turns_in_flight(backend):
    """The env-GRPO retry-budget refusal reads ``PAUSE_ABORTS_IN_FLIGHT``. SGLang's ``abort`` mode drops the requests
    in flight, vLLM's ``keep`` freezes them."""
    client = resolve_weight_sync_client(backend)
    request = client.PAUSE_REQUEST
    mode = {**request.get("json", {}), **request.get("params", {})}["mode"]
    assert client.PAUSE_ABORTS_IN_FLIGHT is (mode == "abort"), f"{client.__name__} pauses with mode={mode!r}"


def test_every_registered_backend_is_a_selectable_config_value():
    """A client under a key the config rejects is unreachable; a config value with no client raises at train time."""
    literal_values = set(AsyncRolloutConfig.__annotations__["rollout_backend"].__args__)
    assert set(rollout_backends()) == literal_values


def _sglang_config(**overrides) -> AsyncRolloutConfig:
    return AsyncRolloutConfig(**{"rollout_backend": "sglang", **overrides})


def test_sglang_supports_train_on_sampled_tokens():
    """SGLang carries the sampled ids in the meta_info it echoes per choice, so the default stands."""
    config = _sglang_config(train_on_sampled_tokens=True)
    assert config.get_rollout_config().capture_token_ids is True


def test_sglang_supports_rollout_routing_replay():
    """SGLang returns routed experts in a raw-int32 wire format the decoder handles, so R3 is not gated here."""
    config = _sglang_config(routing_replay="rollout", train_on_sampled_tokens=True)
    assert config.get_rollout_config().capture_routed_experts is True


def test_sglang_rejects_thinking_budget():
    with pytest.raises(ValueError, match="rollout_max_thinking_tokens is not supported"):
        _sglang_config(rollout_max_thinking_tokens=4096)


def test_chat_template_kwargs_refuse_the_effort_key():
    """The level is per episode and travels top-level. A nested copy would duplicate it or, on a disagreement, override
    it on SGLang and lose to it on vLLM."""
    with pytest.raises(ValueError, match="reasoning_effort"):
        AsyncRolloutConfig(rollout_chat_template_kwargs={"reasoning_effort": "low"})
    assert AsyncRolloutConfig(
        rollout_chat_template_kwargs={"preserve_thinking": True}
    ).get_rollout_config().chat_template_kwargs == {"preserve_thinking": True}


def test_sglang_accepts_the_supported_shape():
    config = _sglang_config(routing_replay="recompute", rollout_max_thinking_tokens=None)
    assert config.rollout_backend == "sglang"
    assert config.get_rollout_config().backend == "sglang"


def test_vllm_still_accepts_everything_sglang_rejects():
    config = AsyncRolloutConfig(rollout_backend="vllm", rollout_max_thinking_tokens=4096)
    assert config.rollout_backend == "vllm"


def test_rollout_backend_defaults_to_vllm():
    """Every existing config omits the field and relies on it."""
    assert AsyncRolloutConfig().rollout_backend == "vllm"
    assert AsyncRolloutConfig().get_rollout_config().backend == "vllm"


def _build_payload(backend: str, reasoning_effort: str | None = None, **rollout_kwargs) -> dict:
    cls = EnvironmentActor.__ray_metadata__.modified_class
    actor = cls.__new__(cls)
    actor.__init__(actor_id=0, env_type="native_math", env_config={"max_turns": 3})
    return actor._build_payload(
        [{"role": "user", "content": "2+2?"}],
        RolloutConfig(backend=backend, **rollout_kwargs),
        reasoning_effort,
    )


def test_the_effort_level_goes_out_top_level_and_nested_only_where_the_template_needs_it():
    """vLLM resolves a disagreement with a nested copy to the top-level value, so a copy there is only ambiguous.
    SGLang resolves it to the nested value, so there the same value rides in both places."""
    vllm_payload = _build_payload("vllm", reasoning_effort="high")
    assert vllm_payload["reasoning_effort"] == "high"
    assert "chat_template_kwargs" not in vllm_payload
    sglang_payload = _build_payload("sglang", reasoning_effort="high")
    assert sglang_payload["reasoning_effort"] == "high"
    assert sglang_payload["chat_template_kwargs"] == {"reasoning_effort": "high"}


def test_an_unset_effort_sends_no_field_at_all():
    """With no level the served model's own template default must stand."""
    assert "reasoning_effort" not in _build_payload("vllm")


def test_sglang_payload_omits_vllm_only_fields():
    """SGLang drops unknown keys silently, so sending these would be a no-op the run never reports."""
    payload = _build_payload("sglang", capture_token_ids=True, max_thinking_tokens=4096)
    assert "return_token_ids" not in payload
    assert "thinking_token_budget" not in payload


def test_vllm_payload_still_carries_them():
    payload = _build_payload("vllm", capture_token_ids=True, max_thinking_tokens=4096)
    assert payload["return_token_ids"] is True
    assert payload["thinking_token_budget"] == 4096


def test_dropped_thinking_budget_is_announced_once(caplog):
    """The cap here is the turn less its answer room (``rollout_max_answer_tokens``), since
    ``rollout_max_thinking_tokens`` is rejected for SGLang. Dropping it silently would leave a cap the config states
    and nothing enforces."""
    engine_wire._thinking_budget_unenforced_warned.discard("sglang")
    try:
        with caplog.at_level(logging.WARNING, logger=engine_wire.__name__):
            _build_payload("sglang", max_thinking_tokens=4096)
            _build_payload("sglang", max_thinking_tokens=8192)
        dropped = [r for r in caplog.records if "NOT enforced" in r.getMessage()]
        assert len(dropped) == 1, [r.getMessage() for r in dropped]
        message = dropped[0].getMessage()
        assert "4096" in message and "sglang" in message
    finally:
        engine_wire._thinking_budget_unenforced_warned.discard("sglang")


def test_vllm_never_warns_about_the_budget_it_enforces(caplog):
    engine_wire._thinking_budget_unenforced_warned.discard("vllm")
    with caplog.at_level(logging.WARNING, logger=engine_wire.__name__):
        _build_payload("vllm", max_thinking_tokens=4096)
    assert not [r for r in caplog.records if "NOT enforced" in r.getMessage()]


def test_shared_fields_reach_both_engines():
    """SGLang supports stop_token_ids and the logprobs pair; gating them would silently disable turn termination."""
    for backend in ("vllm", "sglang"):
        payload = _build_payload(backend, capture_token_ids=True, stop_token_ids=[42])
        assert payload["stop_token_ids"] == [42], backend
        assert payload["logprobs"] is True and payload["top_logprobs"] == 0, backend


def test_routed_experts_opt_in_is_sglang_only():
    """vLLM attaches the payload to every response once its server flag is set; an unknown field there would 400."""
    sglang = _build_payload("sglang", capture_routed_experts=True)
    assert sglang["return_routed_experts"] is True
    assert "routed_experts_start_len" not in sglang  # 0 default = full sequence incl. prompt
    vllm = _build_payload("vllm", capture_routed_experts=True)
    assert "return_routed_experts" not in vllm
    off = _build_payload("sglang", capture_routed_experts=False)
    assert "return_routed_experts" not in off


_FILTERS = ("top_k", "min_p", "repetition_penalty")


def _sampler_fields(body: dict) -> dict:
    return {key: body.get(key) for key in _FILTERS}


def test_the_filter_defaults_are_the_engines_off_values():
    """Both engines fill an omitted filter from generation_config.json (vLLM `--generation-config auto`, SGLang
    `--sampling-defaults model`; Qwen3.6 ships top_k 20), so these go out on every request. SGLang refuses top_k 0."""
    assert _sampler_fields(vars(RolloutConfig())) == {"top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0}


@pytest.mark.parametrize(
    "filters", [{}, {"top_k": 20, "min_p": 0.05, "repetition_penalty": 1.1}], ids=["defaults", "set"]
)
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_rollout_and_eval_requests_carry_the_configured_filters(backend, filters):
    """A filter left out renormalizes every reported log-prob over what the model's default keeps, while the trainer
    scores the full distribution."""
    config = RolloutConfig(backend=backend, **filters)
    expected = {name: getattr(config, name) for name in _FILTERS}
    assert (
        _sampler_fields(_build_payload(backend, reasoning_effort="high", capture_token_ids=True, **filters))
        == expected
    )
    assert _sampler_fields(engine_wire.generation_control_fields(config)) == expected


def test_sglang_capture_reads_ids_from_meta_info_on_the_choice():
    """Shapes from a live SGLang response: meta_info triples are [logprob, token_id, text], and the prompt ids sit on
    the choice."""
    choice = {
        # The OpenAI logprobs field reports text and drops the id, so capturing from it would yield None.
        "logprobs": {"content": [{"token": "<think>", "logprob": -0.001}, {"token": "\n", "logprob": -1e-7}]},
        "meta_info": {
            "output_token_logprobs": [[-0.001, 151667, "<think>"], [-1e-7, 198, "\n"]],
            "output_token_logprobs_length": 2,
            "completion_tokens": 2,
        },
        "prompt_token_ids": [1, 2, 3],
    }
    ids, logprobs, prompt_ids = capture_generation_tokens(choice, {}, "sglang")
    assert ids == [151667, 198]
    assert logprobs == pytest.approx([-0.001, -1e-7])
    assert prompt_ids == [1, 2, 3]


def test_sglang_capture_falls_back_when_meta_info_absent():
    """Without return_meta_info the caller re-tokenizes rather than inventing ids from the text form."""
    choice = {"logprobs": {"content": [{"token": "hi", "logprob": -0.5}]}}
    ids, logprobs, _ = capture_generation_tokens(choice, {}, "sglang")
    assert ids is None and logprobs is None


def test_vllm_capture_reads_token_id_strings_and_top_level_prompt_ids():
    choice = {"logprobs": {"content": [{"token": "token_id:42", "logprob": -0.25}]}}
    ids, logprobs, prompt_ids = capture_generation_tokens(choice, {"prompt_token_ids": [7, 8]}, "vllm")
    assert ids == [42]
    assert logprobs == pytest.approx([-0.25])
    assert prompt_ids == [7, 8]


def test_each_backend_capture_rejects_the_other_shape():
    sglang_choice = {"meta_info": {"output_token_logprobs": [[-0.1, 5, "a"]]}, "prompt_token_ids": [1]}
    assert capture_generation_tokens(sglang_choice, {}, "vllm")[0] is None
    vllm_choice = {"logprobs": {"content": [{"token": "token_id:42", "logprob": -0.25}]}}
    assert capture_generation_tokens(vllm_choice, {}, "sglang")[0] is None


def test_sglang_routing_mask_is_read_from_the_choice_meta_info():
    """With ``return_meta_info`` SGLang leaves the routing payload in the choice's ``meta_info`` and publishes no
    ``sglext`` entry; without it, response-level."""
    config = RolloutConfig(backend="sglang", capture_routed_experts=True)

    def routing_mask(choice_extras: dict, **response_extras) -> str | None:
        choice = {"message": {"content": "ok"}, "finish_reason": "stop", **choice_extras}
        return turn_from_chat_completion({"choices": [choice], **response_extras}, config).routing_mask

    with_meta = {"meta_info": {"routed_experts": "AAAA", "output_token_logprobs": [[-0.1, 5, "a"]]}}
    assert routing_mask(with_meta) == "AAAA"
    assert routing_mask({}, sglext={"routed_experts": "BBBB"}) == "BBBB"
    assert routing_mask({"routed_experts": "CCCC"}) == "CCCC"
    assert routing_mask({"meta_info": {}}) is None


def test_sglang_payload_requests_the_meta_info_capture():
    """The ids only come back when both flags are set; dropping either silently loses them."""
    payload = _build_payload("sglang", capture_token_ids=True)
    assert payload["logprobs"] is True
    assert payload["return_meta_info"] is True
    assert payload["return_prompt_token_ids"] is True
    # vLLM's spelling must not leak to SGLang, which would ignore it silently.
    assert "return_token_ids" not in payload


def test_capture_flags_absent_when_capture_disabled():
    for backend in ("vllm", "sglang"):
        payload = _build_payload(backend, capture_token_ids=False)
        for key in ("logprobs", "top_logprobs", "return_token_ids", "return_meta_info", "return_prompt_token_ids"):
            assert key not in payload, f"{backend}: {key}"


def test_rollout_config_backend_roster_matches_the_gate_and_the_readers():
    """A backend without a reader would fail on its first captured turn, and a reader without a backend is dead."""
    gate = set(get_args(get_type_hints(AsyncRolloutConfig)["rollout_backend"]))
    mirror = set(get_args(get_type_hints(RolloutConfig)["backend"]))
    assert gate and gate == mirror
    assert set(engine_wire._TOKEN_CAPTURE) == gate


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
