#!/usr/bin/env python
"""Rollout-backend selection: registry, config gating, and the engine-specific request payload.

SGLang ignores unknown request fields instead of rejecting them, and its OpenAI layer discards the
token id while converting logprobs. Both make the failure mode here SILENT — a run that looks
configured for sampled-token training quietly trains on re-tokenized text. These tests pin the loud
rejections and the payload gating that stand in the way.
"""

import logging
import sys
import types
from unittest.mock import patch

import pytest
import torch

import src.environments.engine_wire as engine_wire
from src.configs.async_training_config import AsyncTrainingConfig
from src.distributed.expert_parallel.expert_weights import ep_layer_classes
from src.distributed.expert_parallel.layers.gpt_oss import EPGptOssMoELayer
from src.distributed.expert_parallel.layers.qwen3 import EPQwen3MoELayer
from src.distributed.nccl.clients.sglang import SGLangWeightSyncClient
from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient
from src.distributed.nccl.registry import resolve_weight_sync_client, rollout_backends
from src.environments.ray_actors import RolloutConfig


def _ep_config(ep_size: int = 1, expert_tp_size: int = 1) -> types.SimpleNamespace:
    """The four fields the gate reads. Stubbed rather than a real ParallelismConfig: constructing one
    with ep_size=2 needs an NVLink domain this CPU tier does not have."""
    group = ep_size * expert_tp_size
    return types.SimpleNamespace(
        is_ep_mode=group > 1, ep_size=ep_size, expert_tp_size=expert_tp_size, ep_group_size=group
    )


def _model_holding(layer_cls: type) -> torch.nn.Module:
    """A module tree carrying one uninitialized instance of ``layer_cls``.

    ``object.__new__`` skips the family ``__init__`` (which needs a live EP process group and a model
    config) while leaving ``type(module)`` — all the gate reads — exactly right.
    """
    layer = object.__new__(layer_cls)
    torch.nn.Module.__init__(layer)
    root = torch.nn.Module()
    root.add_module("mlp", layer)
    return root


# --- Registry — derived from the client hierarchy, not a hand-maintained table ---


def test_registry_resolves_every_backend_and_rejects_unknown():
    assert rollout_backends() == ["sglang", "vllm"]
    assert resolve_weight_sync_client("vllm") is VLLMWeightSyncClient
    assert resolve_weight_sync_client("sglang") is SGLangWeightSyncClient
    with pytest.raises(ValueError, match="Unknown rollout backend"):
        resolve_weight_sync_client("tensorrt")


def test_a_client_defined_outside_the_package_cannot_capture_a_backend_key():
    """A foreign subclass must not displace a shipped client.

    ``__subclasses__()`` returns every class that ever subclassed the base, so without the registry's
    module filter a class defined anywhere — a test double, or this module tree imported a second
    time under another name — silently takes over its ``BACKEND_KEY``. That is how
    ``resolve_weight_sync_client("vllm")`` returns a stub during a full-directory run while passing
    when the file runs alone: the swap is silent, so only the identity assertion catches it.
    """

    class Impostor(VLLMWeightSyncClient):
        BACKEND_KEY = "vllm"

    assert Impostor.__module__ != VLLMWeightSyncClient.__module__
    assert resolve_weight_sync_client("vllm") is VLLMWeightSyncClient
    assert rollout_backends() == ["sglang", "vllm"]


def test_sglang_clients_default_to_distinct_group_names_per_server():
    """c10d registers group names process-globally on the trainer, so two servers' clients sharing
    one fixed name fail group formation with "group name has already been created". The default is
    keyed on the server endpoint, NOT on group_port: port 0 means "auto-pick at group formation",
    so a port-keyed default collided every auto-port client on one name. An explicit name still
    wins."""
    with patch.object(SGLangWeightSyncClient, "check_server"):
        a = SGLangWeightSyncClient(base_url="http://localhost:30000", group_port=0)
        b = SGLangWeightSyncClient(base_url="http://localhost:30001", group_port=0)
        named = SGLangWeightSyncClient(base_url="http://localhost:30000", group_port=51216, group_name="custom")
    assert a.group_name != b.group_name, "two auto-port clients collided on one default group name"
    assert "localhost:30000" in a.group_name and "localhost:30001" in b.group_name
    assert named.group_name == "custom"


def test_expert_parallelism_is_refused_for_an_engine_that_cannot_sync_under_it():
    """SGLang + expert distribution must fail at construction, not ten minutes into the run.

    The two requirements are irreconcilable in one process: the cross-container sync communicator
    needs NCCL's CUDA-IPC transports off, DeepEP needs them on for symmetric memory, and NCCL caches
    both flags process-wide on first read. Without this gate the run loads the model, collects
    rollouts, pauses the engine and only then dies mid-broadcast with the served weights half
    overwritten.
    """
    from src.trainers.grpo.rollout.weight_sync import validate_backend_parallelism

    with pytest.raises(ValueError, match="cannot be combined with expert distribution"):
        validate_backend_parallelism("sglang", _ep_config(ep_size=2), torch.nn.Linear(1, 1))

    # vLLM is validated under EP, and an undistributed config is fine on either engine.
    validate_backend_parallelism("vllm", _ep_config(ep_size=2), torch.nn.Linear(1, 1))
    validate_backend_parallelism("sglang", _ep_config(), torch.nn.Linear(1, 1))


def test_pure_etp_is_refused_and_the_message_names_the_knob_that_would_fix_it():
    """``is_ep_mode`` is ``ep_group_size > 1`` = ``ep_size * expert_tp_size``, so pure ETP
    (``ep_size=1``) trips the same gate. Telling that config to "drop to ep_size=1" names a state it
    is already in — the message has to name expert_tensor_parallel_size or it sends the user in a
    circle."""
    from src.trainers.grpo.rollout.weight_sync import validate_backend_parallelism

    with pytest.raises(ValueError) as excinfo:
        validate_backend_parallelism("sglang", _ep_config(ep_size=1, expert_tp_size=4), torch.nn.Linear(1, 1))

    message = str(excinfo.value)
    assert "expert_tensor_parallel_size=4" in message
    # The knob that is ALREADY 1 must not be presented as the remedy on its own.
    assert "drop to ep_size=1" not in message


def test_a_family_without_a_fused_gather_is_refused_before_the_engine_is_touched():
    """SGLang loads experts fused, and ``gather_fused_expert_state_dict`` is a per-family hook whose
    base default raises. Reaching the first sync before discovering that leaves the engine paused and
    partly written — exactly what this gate exists to prevent. The EP wrappers are present
    at ``ep_group_size == 1`` too (grouped-GEMM is on by default), so this is reachable without EP.

    Both sides are drawn from the REAL layer registry rather than local subclasses: a throwaway
    subclass would join ``EPMoELayerBase.__subclasses__()``, which ``ep_layer_classes()`` walks to
    build cached cross-family unions — and it would stop tracking the roster, so a family that gained
    or lost the override would not move this test.
    """
    from src.trainers.grpo.rollout.weight_sync import validate_backend_parallelism

    implementing = [c for c in ep_layer_classes() if c.implements_fused_expert_layout()]
    missing = [c for c in ep_layer_classes() if not c.implements_fused_expert_layout()]
    assert implementing, "no EP family implements the fused gather — the anti-vacuity side is vacuous"
    assert missing, "every EP family now implements the fused gather; this gate is dead code — delete it"

    with pytest.raises(ValueError, match="does not implement"):
        validate_backend_parallelism("sglang", _ep_config(), _model_holding(missing[0]))

    # Anti-vacuity: neither the missing override nor the engine alone can be what fires the raise.
    validate_backend_parallelism("sglang", _ep_config(), _model_holding(implementing[0]))
    validate_backend_parallelism("vllm", _ep_config(), _model_holding(missing[0]))


def test_an_unrecognized_expert_layout_is_refused_rather_than_guessed():
    """The knob is two-valued and each shipped client declares one of the two named spellings. An
    inequality test would route a third one into whichever branch it happens to miss, and the gather
    it picks is rejected on arrival — after the engine is paused and partly written."""
    from src.distributed.nccl.clients.base import BaseWeightSyncClient
    from src.trainers.grpo.rollout.weight_sync import wants_fused_experts

    declared = {BaseWeightSyncClient.UNFUSED_EXPERT_LAYOUT, BaseWeightSyncClient.FUSED_EXPERT_LAYOUT}
    assert {VLLMWeightSyncClient.EXPERT_LAYOUT, SGLangWeightSyncClient.EXPERT_LAYOUT} == declared

    assert wants_fused_experts(SGLangWeightSyncClient.EXPERT_LAYOUT) is True
    assert wants_fused_experts(VLLMWeightSyncClient.EXPERT_LAYOUT) is False
    with pytest.raises(ValueError, match="unknown expert layout"):
        wants_fused_experts("fused-mxfp4")


def test_gptoss_is_the_only_family_that_carries_the_fused_gather():
    """Pins the roster split to the ENGINE's truth so the docs' claim cannot go stale silently.

    SGLang 0.5.17's fused expert mapping (``make_expert_params_mapping_fused``) exists for gpt_oss
    alone; its qwen3_moe loader maps per-expert names and silently drops fused keys before its
    not-found warning. A family joining this list therefore asserts its engine-side loader consumes
    the fused pair — Qwen3-MoE re-joining without that evidence re-opens a silent expert freeze.
    """
    implementing = [c for c in ep_layer_classes() if c.implements_fused_expert_layout()]
    assert implementing == [EPGptOssMoELayer]
    assert EPQwen3MoELayer not in implementing


def test_gptoss_fused_gather_is_the_checkpoint_gather():
    """GptOss's hub checkpoint layout is already the fused interleaved one, so the fused gather must
    be a pure delegation — if the two ever diverge, a fused-layout engine and a gathered checkpoint
    would silently receive different expert tensors."""
    sentinel = {"experts.gate_up_proj": object()}
    layer = EPGptOssMoELayer.__new__(EPGptOssMoELayer)
    layer.gather_expert_state_dict = lambda device="cpu", merge_lora=False, retain=True: (
        sentinel if (device, merge_lora, retain) == ("cuda", True, True) else {}
    )
    assert layer.gather_fused_expert_state_dict("cuda", merge_lora=True) is sentinel
    assert layer.gather_fused_expert_state_dict("cuda", merge_lora=True, retain=False) == {}


def test_every_registered_backend_is_a_selectable_config_value():
    """The registry and the config's Literal must not drift apart — a client registered under a key
    the config rejects is unreachable, and a config value with no client raises only at train time."""
    literal_values = set(AsyncTrainingConfig.__annotations__["rollout_backend"].__args__)
    assert set(rollout_backends()) == literal_values


# --- Config gating — each knob SGLang cannot honor must raise, not warn ---


def _sglang_config(**overrides) -> AsyncTrainingConfig:
    return AsyncTrainingConfig(**{"rollout_backend": "sglang", **overrides})


def test_sglang_supports_train_on_sampled_tokens():
    """SGLang carries the sampled ids in the meta_info it echoes per choice, so the default stands."""
    config = _sglang_config(train_on_sampled_tokens=True)
    assert config.get_rollout_config().capture_token_ids is True


def test_sglang_supports_rollout_routing_replay():
    """SGLang publishes routed experts response-level in a raw-int32 wire format the decoder
    handles, so R3 is not capability-gated on this backend."""
    config = _sglang_config(routing_replay="rollout", train_on_sampled_tokens=True)
    assert config.get_rollout_config().capture_routed_experts is True


def test_sglang_rejects_thinking_budget():
    with pytest.raises(ValueError, match="rollout_max_thinking_tokens is not supported"):
        _sglang_config(rollout_max_thinking_tokens=4096)


def test_sglang_accepts_the_supported_shape():
    """Anti-vacuity: the rejections above must come from the specific knobs, not from
    rollout_backend='sglang' being unusable on its own."""
    config = _sglang_config(routing_replay="recompute", rollout_max_thinking_tokens=None)
    assert config.rollout_backend == "sglang"
    assert config.get_rollout_config().backend == "sglang"


def test_vllm_still_accepts_everything_sglang_rejects():
    """The gate must be engine-specific: not a defect on vLLM, which supports the budget field."""
    config = AsyncTrainingConfig(rollout_backend="vllm", rollout_max_thinking_tokens=4096)
    assert config.rollout_backend == "vllm"


def test_rollout_backend_defaults_to_vllm():
    """The default must stay vLLM — every existing config omits the field and relies on it."""
    assert AsyncTrainingConfig().rollout_backend == "vllm"
    assert AsyncTrainingConfig().get_rollout_config().backend == "vllm"


# --- Payload gating — vLLM-only fields must not be sent to SGLang ---


def _build_payload(backend: str, reasoning_effort: str | None = None, **rollout_kwargs) -> dict:
    from src.environments.ray_actors import EnvironmentActor

    cls = EnvironmentActor.__ray_metadata__.modified_class
    actor = cls.__new__(cls)
    actor.__init__(actor_id=0, env_type="native_math", env_config={"max_turns": 3})
    return actor._build_payload(
        [{"role": "user", "content": "2+2?"}],
        RolloutConfig(backend=backend, **rollout_kwargs),
        reasoning_effort,
    )


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_the_effort_level_goes_out_top_level_and_never_nested(backend):
    """One spelling, from one owner, on both engines.

    Only the top-level field derives the engines' thinking toggles: vLLM's request model sets
    ``enable_thinking`` from it and SGLang's validator sets ``thinking``/``enable_thinking``, both
    before anything reads ``chat_template_kwargs``. The nested spelling reaches the template render
    and sets neither. It is also not needed as a belt-and-braces copy — vLLM merges the nested dict
    UNDER the top-level field and drops that field when unset, so it never clobbers — while sending
    both is ambiguous: vLLM resolves a disagreement to the top-level value, SGLang to the nested one.
    """
    payload = _build_payload(backend, reasoning_effort="high")
    assert payload["reasoning_effort"] == "high"
    assert "chat_template_kwargs" not in payload


def test_an_unset_effort_sends_no_field_at_all():
    """Anti-vacuity: with no level the served model's own template default must stand."""
    assert "reasoning_effort" not in _build_payload("vllm")


def test_sglang_payload_omits_vllm_only_fields():
    """SGLang drops unknown keys silently, so sending these would be a no-op the run never reports."""
    payload = _build_payload("sglang", capture_token_ids=True, max_thinking_tokens=4096)
    assert "return_token_ids" not in payload
    assert "thinking_token_budget" not in payload


def test_vllm_payload_still_carries_them():
    """Anti-vacuity: the keys above are genuinely emitted for vLLM, so their absence above is the gate."""
    payload = _build_payload("vllm", capture_token_ids=True, max_thinking_tokens=4096)
    assert payload["return_token_ids"] is True
    assert payload["thinking_token_budget"] == 4096


def test_dropped_thinking_budget_is_announced_once(caplog):
    """A per-effort budget that reaches no engine field must say so — once, not per turn.

    ``rollout_max_thinking_tokens`` is rejected outright for SGLang, so a budget arriving at the
    payload builder came from the environment's ``reasoning_effort_profiles``. It is still scored by
    ``reasoning_compliance_weight``, so dropping it silently prices the policy against a band nothing
    enforced.
    """
    engine_wire._THINKING_BUDGET_UNENFORCED_WARNED.discard("sglang")
    try:
        with caplog.at_level(logging.WARNING, logger=engine_wire.__name__):
            _build_payload("sglang", max_thinking_tokens=4096)
            _build_payload("sglang", max_thinking_tokens=8192)
        dropped = [r for r in caplog.records if "NOT enforced" in r.getMessage()]
        assert len(dropped) == 1, [r.getMessage() for r in dropped]
        message = dropped[0].getMessage()
        assert "4096" in message and "sglang" in message
    finally:
        engine_wire._THINKING_BUDGET_UNENFORCED_WARNED.discard("sglang")


def test_vllm_never_warns_about_the_budget_it_enforces(caplog):
    """Anti-vacuity: the warning is backend-specific, not emitted whenever a budget is set."""
    engine_wire._THINKING_BUDGET_UNENFORCED_WARNED.discard("vllm")
    with caplog.at_level(logging.WARNING, logger=engine_wire.__name__):
        _build_payload("vllm", max_thinking_tokens=4096)
    assert not [r for r in caplog.records if "NOT enforced" in r.getMessage()]


def test_shared_fields_reach_both_engines():
    """Only the two vLLM-only keys are gated — SGLang supports stop_token_ids and the logprobs pair,
    so gating those too would silently disable turn termination."""
    for backend in ("vllm", "sglang"):
        payload = _build_payload(backend, capture_token_ids=True, stop_token_ids=[42])
        assert payload["stop_token_ids"] == [42], backend
        assert payload["logprobs"] is True and payload["top_logprobs"] == 0, backend


def test_routed_experts_opt_in_is_sglang_only():
    """SGLang wants the per-request `return_routed_experts` opt-in; vLLM attaches the payload to
    every response once its server flag is set, and an unknown request field there would 400."""
    sglang = _build_payload("sglang", capture_routed_experts=True)
    assert sglang["return_routed_experts"] is True
    assert "routed_experts_start_len" not in sglang  # 0 default = full sequence incl. prompt
    vllm = _build_payload("vllm", capture_routed_experts=True)
    assert "return_routed_experts" not in vllm
    off = _build_payload("sglang", capture_routed_experts=False)
    assert "return_routed_experts" not in off


# --- Sampled-token capture — the two engines report the same facts in different places ---


def test_sglang_capture_reads_ids_from_meta_info_on_the_choice():
    """Shapes taken from a live SGLang response: meta_info triples are
    [logprob, token_id, text], and the prompt ids sit on the CHOICE, not the response root."""
    from src.environments.engine_wire import capture_generation_tokens

    choice = {
        # The OpenAI logprobs field reports text and drops the id — capturing from it would yield None.
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
    """Without return_meta_info the ids are unrecoverable — return None so the caller re-tokenizes
    rather than inventing ids from the text form."""
    from src.environments.engine_wire import capture_generation_tokens

    choice = {"logprobs": {"content": [{"token": "hi", "logprob": -0.5}]}}
    ids, logprobs, _ = capture_generation_tokens(choice, {}, "sglang")
    assert ids is None and logprobs is None


def test_vllm_capture_reads_token_id_strings_and_top_level_prompt_ids():
    """Anti-vacuity: the vLLM shape is genuinely different, so a single reader could not serve both."""
    from src.environments.engine_wire import capture_generation_tokens

    choice = {"logprobs": {"content": [{"token": "token_id:42", "logprob": -0.25}]}}
    ids, logprobs, prompt_ids = capture_generation_tokens(choice, {"prompt_token_ids": [7, 8]}, "vllm")
    assert ids == [42]
    assert logprobs == pytest.approx([-0.25])
    assert prompt_ids == [7, 8]


def test_each_backend_capture_rejects_the_other_shape():
    """The readers must not silently half-succeed on the wrong engine's payload."""
    from src.environments.engine_wire import capture_generation_tokens

    sglang_choice = {"meta_info": {"output_token_logprobs": [[-0.1, 5, "a"]]}, "prompt_token_ids": [1]}
    assert capture_generation_tokens(sglang_choice, {}, "vllm")[0] is None
    vllm_choice = {"logprobs": {"content": [{"token": "token_id:42", "logprob": -0.25}]}}
    assert capture_generation_tokens(vllm_choice, {}, "sglang")[0] is None


def test_sglang_payload_requests_the_meta_info_capture():
    """The ids only come back when BOTH flags are set; dropping either silently loses them."""
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
    """``RolloutConfig.backend`` mirrors ``AsyncTrainingConfig.rollout_backend`` (the validated gate), and every
    selectable backend has a token-capture reader — the import-time guard in ``engine_wire`` depends on both."""
    from typing import get_args, get_type_hints

    from src.configs.async_training_config import AsyncTrainingConfig
    from src.configs.rollout_config import RolloutConfig
    from src.environments import engine_wire

    gate = set(get_args(get_type_hints(AsyncTrainingConfig)["rollout_backend"]))
    mirror = set(get_args(get_type_hints(RolloutConfig)["backend"]))
    assert gate and gate == mirror
    assert frozenset(gate) == engine_wire._SELECTABLE_BACKENDS
    assert not engine_wire._UNREADABLE_BACKENDS


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
