#!/usr/bin/env python
"""Construction-time gates of the environmental GRPO trainer, each refusing before the run starts: TRL knobs this
trainer never reads, a dataset missing the column it grades against, budgets that cannot bind or be neutralized."""

import ast
import inspect
import logging
import re
import textwrap
import types

import pytest
import torch
from datasets import Dataset
from transformers import CONFIG_MAPPING
from trl import GRPOConfig, GRPOTrainer

from halo.configs.grpo_environmental import AsyncRolloutConfig, ISMaskConfig
from halo.environments.reasoning_effort import has_thinking_budget
from halo.environments.registry import build_environment
from halo.environments.rollout_config import SGLANG_BACKEND, VLLM_BACKEND, resolve_reasoning_end_token_id
from halo.trainers.grpo.environmental import reasoning
from halo.trainers.grpo.environmental.environmental_rules import (
    reject_answerless_datasets,
    reject_inert_is_stages,
    reject_inert_reasoning_terms,
    reject_off_policy_mask_threshold,
    reject_straddling_train_groups,
    reject_unneutralized_forced_closes,
    reject_unread_grpo_knobs,
    reject_unsupported_routing_replay,
    reject_unverified_carried_reasoning,
    reject_vespo_loss,
)
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.objective.logratio import compute_is_ratio, zero_engine_forced_closes
from halo.trainers.training_args_overrides import keep_all_dataset_columns
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient


def _grpo_config(tmp_path, **overrides) -> GRPOConfig:
    return GRPOConfig(output_dir=str(tmp_path / "out"), bf16=False, use_cpu=True, **overrides)


# The gates below run as bound methods on a bare host, which pins their logic but not their wiring. A deleted call
# site would leave those tests green and the gate dead.
_INIT_GATES = (
    "reject_unverified_carried_reasoning",
    "reject_straddling_train_groups",
    "reject_straddling_eval_groups",
    "keep_all_dataset_columns",
    "reject_answerless_datasets",
    "reject_inert_reasoning_terms",
    "_resolve_reasoning_end_ids",
    "_resolve_forced_close_ids",
    "reject_unneutralized_forced_closes",
    "reject_inert_is_stages",
    "reject_off_policy_mask_threshold",
    "reject_vespo_loss",
    "reject_inert_std_floor",
    "resolve_rollout_stop_token_ids",
    "_arm_update_breaker",
    "reject_unread_grpo_knobs",
    "reject_prefetch_past_retry_budget",
)


def _called_names(fn: ast.FunctionDef) -> set[str]:
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                names.add(node.func.id)
    return names


_INIT_PHASES = ("__init__", "_configure_rollout_state", "_configure_importance_sampling")


def _method_source(name: str) -> ast.FunctionDef:
    method = getattr(DistributedAsyncEnvironmentalGRPOTrainer, name)
    return ast.parse(textwrap.dedent(inspect.getsource(method))).body[0]


def test_every_gate_is_still_called_from_the_trainers_init():
    init = _method_source("__init__")
    assert {"_configure_rollout_state", "_configure_importance_sampling"} <= _called_names(init)
    called = set().union(*(_called_names(_method_source(name)) for name in _INIT_PHASES))
    missing = [gate for gate in _INIT_GATES if gate not in called]
    assert not missing, f"DistributedAsyncEnvironmentalGRPOTrainer.__init__ no longer calls: {missing}"


@pytest.mark.parametrize("gate", ["reject_off_policy_mask_threshold", "reject_vespo_loss"])
def test_a_refused_knob_is_refused_before_the_balance_gate_reads_it(gate):
    """Run first, the balance gate would refuse the pair, and a user who unsets the balance meets the real refusal only
    on the next launch."""
    first_call = {}
    for node in ast.walk(_method_source("_configure_rollout_state")):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            first_call[node.func.id] = min(node.lineno, first_call.get(node.func.id, node.lineno))
    assert first_call[gate] < first_call["reject_unbalanceable_token_mass"]


def test_off_policy_mask_threshold_is_refused_with_the_working_knob_named(tmp_path):
    with pytest.raises(ValueError, match="isr_opsm_delta"):
        reject_off_policy_mask_threshold(_grpo_config(tmp_path, off_policy_mask_threshold=0.5))


def test_the_trl_default_passes(tmp_path):
    reject_off_policy_mask_threshold(_grpo_config(tmp_path))


def test_vespo_is_refused_naming_the_mechanism_and_the_per_token_losses(tmp_path):
    with pytest.raises(ValueError, match=r"loss_type='vespo'") as refused:
        reject_vespo_loss(_grpo_config(tmp_path, loss_type="vespo"))
    message = str(refused.value)
    # The drop is conditional on a zeroed ratio, which a run with no mask stage and no forced close never makes.
    assert "whenever the IS correction masks a token (isr_*) or the engine forces a reasoning close" in message, (
        message
    )
    named = re.findall(r"\b(dapo|dr_grpo|cispo|grpo)\b", re.search(r"Use a per-token loss: (.*)\.$", message).group(1))
    assert named, message
    for loss_type in named:
        reject_vespo_loss(_grpo_config(tmp_path, loss_type=loss_type))


def test_one_forced_close_drops_a_whole_turn_row_under_trls_vespo_weight():
    """The refusal's premise, on the installed TRL: VESPO's weight takes the log of every token's ratio summed over the
    row, so the zeroed ratio at one forced close silences the whole row."""
    width, close = 32, 7
    completion_ids = torch.full((1, width), 3)
    completion_ids[0, 10] = close
    sampling = torch.full((1, width), -0.5)
    sampling[0, 10] = 0.0
    mask, has_sampling = torch.ones(1, width), torch.ones(1, dtype=torch.bool)
    ratio, _, _ = compute_is_ratio(sampling, sampling, mask, has_sampling, 3.0)
    ratio, forced = zero_engine_forced_closes(ratio, sampling, mask, has_sampling, completion_ids, (close,))
    assert forced.sum() == 1 and ratio.sum() == width - 1
    weight = GRPOTrainer.get_gamma_weights(torch.ones(1, 1), torch.zeros(1, width), mask, ratio)
    assert weight.item() < 1e-12


@pytest.mark.parametrize(
    ("knob", "value", "named"),
    [
        ("top_p", 0.9, "rollout_top_p"),
        ("top_k", 20, "rollout_top_k"),
        ("min_p", 0.05, "rollout_min_p"),
        ("repetition_penalty", 1.1, "rollout_repetition_penalty"),
        ("generation_kwargs", {"temperature": 0.7}, "rollout_top_p"),
        ("vllm_importance_sampling_clip_min", 0.5, "vllm_importance_sampling_clip_max"),
        ("vllm_group_port", 52000, "rollout_group_port"),
    ],
)
def test_a_grpo_knob_this_trainer_never_reads_is_refused_when_set(tmp_path, knob, value, named):
    """The actors sample from the rollout config and the IS correction truncates per token from above, so these change
    nothing."""
    with pytest.raises(ValueError, match=rf"(?s)'{knob}'.*{named}"):
        reject_unread_grpo_knobs(_grpo_config(tmp_path, **{knob: value}))


@pytest.mark.parametrize("mode", ["token_mask", "sequence_truncate"])
def test_an_is_mode_the_correction_does_not_apply_is_refused_naming_the_one_it_does(tmp_path, mode):
    """``token_mask`` would zero a ratio above the cap that the correction clamps, ``sequence_truncate`` would take one
    ratio per sequence."""
    with pytest.raises(ValueError, match=rf"vllm_importance_sampling_mode='{mode}'") as refused:
        reject_unread_grpo_knobs(_grpo_config(tmp_path, vllm_importance_sampling_mode=mode))
    named = re.search(r"TRL's '(\w+)'", str(refused.value)).group(1)
    assert named.startswith("token_")
    reject_unread_grpo_knobs(_grpo_config(tmp_path, vllm_importance_sampling_mode=named))


def test_the_correction_is_trls_token_truncate_and_not_its_token_mask():
    """Each token's ratio is clamped from above at the clip max (``token_truncate`` at an unset lower clip)."""
    generator = torch.Generator().manual_seed(0)
    sampling = -torch.rand(2, 16, generator=generator) - 0.05
    recompute = sampling + torch.randn(2, 16, generator=generator)
    mask = torch.ones(2, 16)
    clip_max = 2.0
    ratio, _, _ = compute_is_ratio(recompute, sampling, mask, torch.ones(2, dtype=torch.bool), clip_max)
    raw = torch.exp((recompute - sampling) * mask)
    assert (raw > clip_max).any(), "the draw must cross the cap for the two modes to differ"
    assert torch.equal(ratio, torch.clamp(raw, min=None, max=clip_max))
    assert not torch.equal(ratio, raw.masked_fill(raw > clip_max, 0.0))


def test_the_trl_defaults_of_the_unread_knobs_pass_even_when_spelled_out(tmp_path):
    reject_unread_grpo_knobs(_grpo_config(tmp_path))
    reject_unread_grpo_knobs(
        _grpo_config(
            tmp_path,
            top_p=1.0,
            top_k=0,
            min_p=None,
            repetition_penalty=1.0,
            generation_kwargs=None,
            vllm_importance_sampling_mode="sequence_mask",
            vllm_importance_sampling_clip_min=None,
        )
    )


def _check_carried_reasoning(backend: str, carry_reasoning: bool) -> None:
    reject_unverified_carried_reasoning(types.SimpleNamespace(carry_reasoning=carry_reasoning), backend)


def test_carried_reasoning_on_sglang_is_refused_as_unverified():
    with pytest.raises(ValueError, match="unverified"):
        _check_carried_reasoning("sglang", carry_reasoning=True)


def test_carried_reasoning_on_vllm_and_plain_sglang_pass():
    _check_carried_reasoning("vllm", carry_reasoning=True)
    _check_carried_reasoning("sglang", carry_reasoning=False)


def test_the_wire_backend_keys_are_the_weight_sync_clients_keys():
    """The tokenize path and the carried-reasoning gate compare against the wire module's spellings."""
    assert VLLM_BACKEND == VLLMWeightSyncClient.BACKEND_KEY
    assert SGLANG_BACKEND == SGLangWeightSyncClient.BACKEND_KEY


_PROMPT = [[{"role": "user", "content": "solve it"}]]


def _dataset(**columns) -> Dataset:
    return Dataset.from_dict({"prompt": _PROMPT, **columns})


def _check_answers(env_type: str, env_kwargs: dict, train, eval_dataset=None) -> None:
    reject_answerless_datasets(build_environment(env_type, env_kwargs), train, eval_dataset)


def test_a_grading_environment_refuses_a_dataset_without_the_answer_column():
    """code_contests reads its hidden tests from ``answer``; with no column every submission grades 0."""
    with pytest.raises(ValueError, match="CodeContestsEnvironment"):
        _check_answers("code_contests", {"sandbox_backend": "local"}, _dataset())


def test_a_grading_environment_accepts_the_dataset_that_carries_it():
    _check_answers("code_contests", {"sandbox_backend": "local"}, _dataset(answer=["{}"]))


def test_the_eval_dataset_is_held_to_the_same_column():
    """An eval split without the column scores nothing on every evaluation round."""
    with pytest.raises(ValueError, match="eval_dataset"):
        _check_answers("exam_qa", {}, _dataset(answer=["A"]), eval_dataset=_dataset())


def test_a_non_grading_environment_accepts_an_answer_less_dataset():
    """native_math pays for completing the task, so prompts alone are a complete dataset for it."""
    _check_answers("native_math", {}, _dataset(), eval_dataset=_dataset())


def _level_host(budgets: dict, effort: str | None = "random", **config):
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host.async_config = AsyncRolloutConfig(**config)
    host._rollout_env = types.SimpleNamespace(
        reasoning_effort=effort, thinking_budget_for_effort=budgets.get, thinking_budget_enforced=lambda level: True
    )
    return host


def _check_reasoning_terms(host) -> None:
    reject_inert_reasoning_terms(
        host.async_config,
        host._rollout_env,
        budgeted=has_thinking_budget(host._efforts_by_level()),
        every_budget_soft=host._every_budget_soft,
    )


_EVERY_LEVEL = {"low": 8192, "medium": 12288, "high": 16384}


def test_a_drawable_levels_budget_must_sit_below_the_turn_cap():
    """Each turn of such a level is cut mid-reasoning and trains as a length-cut turn forever. Only levels the
    environment can draw are checked, since a budget elsewhere binds no episode."""
    turn = {"rollout_max_tokens": 16384}
    with pytest.raises(
        ValueError, match=r"'high' level's thinking_tokens \(16384\) must sit below rollout_max_tokens"
    ):
        _check_reasoning_terms(_level_host(_EVERY_LEVEL, **turn))
    with pytest.raises(ValueError, match="'high' level's thinking_tokens"):
        _check_reasoning_terms(_level_host(_EVERY_LEVEL, effort="high", **turn))
    _check_reasoning_terms(_level_host(_EVERY_LEVEL, effort="low", **turn))
    _check_reasoning_terms(_level_host(_EVERY_LEVEL, effort=None, **turn))
    _check_reasoning_terms(_level_host(_EVERY_LEVEL, rollout_max_tokens=16385))


def test_column_pruning_is_forced_off(tmp_path):
    """The rollout context is every non-``prompt`` column; pruning keeps only what the forward signature names."""
    args = _grpo_config(tmp_path, remove_unused_columns=True)
    keep_all_dataset_columns(args, "forcing it off")
    assert args.remove_unused_columns is False


def test_an_enforced_thinking_budget_needs_the_is_correction():
    """Forced reasoning closes are neutralized through the IS ratio; with the correction off they would train
    with the episode's advantage and teach the model to stop closing its reasoning."""
    with pytest.raises(ValueError, match="importance-sampling correction is off"):
        reject_unneutralized_forced_closes((7,), is_correction=False)
    reject_unneutralized_forced_closes((7,), is_correction=True)
    reject_unneutralized_forced_closes(None, is_correction=False)


_CLOSE_ID = 7
# gpt-oss's final-channel opener: added tokens around two plain words, the shape a harmony close takes.
_OPENER = "<|start|>assistant<|channel|>final<|message|>"
_OPENER_IDS = (70, 71, 72, 73, 74)


class _Tokenizer:
    """``knows_close`` decides whether the default ``</think>`` is an added token or splits into plain text, as on
    Gemma 4 and gpt-oss."""

    def __init__(self, knows_close: bool):
        self._encodings = {
            AsyncRolloutConfig.rollout_reasoning_end_token: [_CLOSE_ID] if knows_close else [60, 61, 62],
            _OPENER: _OPENER_IDS,
        }
        self.added_tokens_decoder = dict.fromkeys(({_CLOSE_ID} if knows_close else set()) | {70, 72, 74})

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert not add_special_tokens, "the engine encodes its reasoning end string without special tokens"
        return list(self._encodings[text])


def _close_host(budgets: dict, knows_close: bool = True, soft: tuple = (), model_config=None, **config):
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host.async_config = AsyncRolloutConfig(**config)
    host._rollout_env = types.SimpleNamespace(
        reasoning_effort="random",
        thinking_budget_for_effort=budgets.get,
        thinking_budget_enforced=lambda level: level not in soft,
    )
    host._tokenizer = _Tokenizer(knows_close)
    host.model = torch.nn.Module()
    if model_config is not None:
        host.model.config = model_config
    host._reasoning_end_ids = host._resolve_reasoning_end_ids()
    return host


def _forced_close_ids(host) -> tuple[int, ...] | None:
    return host._resolve_forced_close_ids(host._reasoning_end_ids)


def test_a_vllm_thinking_budget_resolves_the_close_the_engine_forces():
    """A level's ``thinking_tokens`` or the run's ceiling each enforce a budget; the loss finds forced closes by id."""
    assert _forced_close_ids(_close_host({"high": 16384})) == (_CLOSE_ID,)
    assert _forced_close_ids(_close_host({}, rollout_max_thinking_tokens=8192)) == (_CLOSE_ID,)


def test_a_multi_token_close_resolves_to_its_whole_sequence():
    """gpt-oss's budget forces a five-token opener; the loss needs every id of it, in the engine's order."""
    host = _close_host({"high": 16384}, knows_close=False, rollout_reasoning_end_token=_OPENER)
    assert _forced_close_ids(host) == _OPENER_IDS


def test_no_forced_close_where_no_budget_can_be_enforced():
    """SGLang enforces no thinking budget and an unbudgeted run caps nothing. A resolved marker would demand the IS
    correction of a run that needs none, and zero the ratio at a close the model itself emitted with certainty."""
    assert _forced_close_ids(_close_host({"high": 16384}, rollout_backend=SGLANG_BACKEND)) is None
    assert _forced_close_ids(_close_host({})) is None


def test_a_soft_budget_forces_a_close_only_where_a_ceiling_caps_it():
    """A soft budget forces no close of its own; its ceiling does (the turn's answer room, or the run's cap)."""
    every = {"low": 8192, "medium": 12288, "high": 16384}
    soft = ("low", "medium", "high")
    room = _close_host(every, soft=soft, rollout_max_tokens=30000, rollout_max_answer_tokens=8192)
    assert _forced_close_ids(room) == (_CLOSE_ID,)
    assert _forced_close_ids(_close_host(every, soft=soft, rollout_max_thinking_tokens=21800)) == (_CLOSE_ID,)
    assert _forced_close_ids(_close_host(every, soft=soft)) is None, "no ceiling: no cap can bind"
    assert _forced_close_ids(_close_host(every, soft=("low",))) == (_CLOSE_ID,), "an enforced level still binds"


def test_a_vllm_budget_on_a_mistral_common_served_model_is_refused():
    """vLLM encodes Mistral 4's [THINK]/[/THINK] markers as plain text while the model writes control tokens, so the
    budget it accepts never counts. The wrapper's own model_type is mistral3; the text tower's spelling must match."""
    mistral_small_4 = CONFIG_MAPPING["mistral3"](text_config=CONFIG_MAPPING["mistral4"]())
    with pytest.raises(ValueError, match=r"cannot enforce a thinking budget for model_type \['mistral4'\]"):
        _close_host({"high": 16384}, model_config=mistral_small_4)
    sglang = _close_host({"high": 16384}, model_config=mistral_small_4, rollout_backend=SGLANG_BACKEND)
    assert _forced_close_ids(sglang) is None
    assert _forced_close_ids(_close_host({}, model_config=mistral_small_4)) is None
    assert _forced_close_ids(_close_host({"high": 16384}, model_config=CONFIG_MAPPING["mistral3"]())) == (_CLOSE_ID,)


def test_a_marker_the_tokenizer_does_not_write_warns_and_trains_on_the_forced_closes(caplog):
    """Nothing else reads the marker, so a family whose reasoning ends otherwise still runs, told that
    its forced closes stay in the loss."""
    host = _close_host({"high": 16384}, knows_close=False)
    with caplog.at_level(logging.WARNING, logger="halo.trainers.grpo.environmental.reasoning"):
        assert _forced_close_ids(host) is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "stay in the policy loss" in w and AsyncRolloutConfig.rollout_reasoning_end_token in w for w in warnings
    ), warnings


def test_the_marker_warning_is_logged_once_by_the_main_process_under_the_trainers_name(caplog, monkeypatch):
    for is_main, expected in ((True, 1), (False, 0)):
        monkeypatch.setattr(reasoning, "is_global_main_process", lambda is_main=is_main: is_main)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert _forced_close_ids(_close_host({"high": 16384}, knows_close=False)) is None
        warnings = [r.name for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == ["halo.trainers.grpo.environmental.reasoning"] * expected, (is_main, warnings)


def _count_end_id(budgets: dict, knows_close: bool = True, **config) -> int | None:
    return resolve_reasoning_end_token_id(_close_host(budgets, knows_close, **config)._reasoning_end_ids)


def test_the_reasoning_count_reads_up_to_a_one_token_close_on_every_budgeted_vllm_run():
    """A turn's reasoning is counted up to the marker wherever a vLLM run's episodes carry a budget: a soft one forces
    no close, but the model still writes its own. A marker spanning several tokens (gpt-oss's opener) has no one id."""
    assert _count_end_id({"high": 16384}) == _CLOSE_ID
    assert _count_end_id({}, rollout_max_thinking_tokens=8192) == _CLOSE_ID
    assert _count_end_id({"high": 16384}, soft=("high",)) == _CLOSE_ID
    assert _count_end_id({}) is None
    assert _count_end_id({"high": 16384}, rollout_backend=SGLANG_BACKEND) is None
    assert _count_end_id({"high": 16384}, knows_close=False) is None
    assert _count_end_id({"high": 16384}, knows_close=False, rollout_reasoning_end_token=_OPENER) is None


_VETO = ISMaskConfig(veto_min=0.1)
_NO_STAGE = ISMaskConfig()


def test_the_shape_is_and_replay_gates_pass_a_supported_run():
    reject_straddling_train_groups(per_device_train_batch_size=2, steps_per_generation=2, num_generations=4)
    reject_inert_is_stages(_VETO, is_correction=True, skip_update_masked_frac=0.5)
    reject_inert_is_stages(_NO_STAGE, is_correction=False, skip_update_masked_frac=None)
    reject_unsupported_routing_replay("rollout", train_on_sampled_tokens=True, runs_logprob_recompute=False)
    reject_unsupported_routing_replay("recompute", train_on_sampled_tokens=False, runs_logprob_recompute=True)
    reject_unsupported_routing_replay("none", train_on_sampled_tokens=False, runs_logprob_recompute=False)


@pytest.mark.parametrize(
    ("refusal", "match"),
    [
        (
            lambda: reject_straddling_train_groups(
                per_device_train_batch_size=3, steps_per_generation=1, num_generations=2
            ),
            r"\(3\) must be divisible by num_generations \(2\): .* a group cannot straddle ranks",
        ),
        (
            lambda: reject_inert_is_stages(_VETO, is_correction=False, skip_update_masked_frac=None),
            r"isr_geo_band/isr_veto/isr_opsm knobs require the importance-sampling correction",
        ),
        (
            lambda: reject_inert_is_stages(_NO_STAGE, is_correction=True, skip_update_masked_frac=0.5),
            r"skip_update_masked_frac requires .* the circuit breaker would silently never fire",
        ),
        (
            lambda: reject_inert_is_stages(_NO_STAGE, is_correction=False, skip_update_masked_frac=0.5),
            r"skip_update_masked_frac requires .* the circuit breaker would silently never fire",
        ),
        (
            lambda: reject_unsupported_routing_replay(
                "rollout", train_on_sampled_tokens=False, runs_logprob_recompute=True
            ),
            r"routing_replay='rollout' requires train_on_sampled_tokens",
        ),
        (
            lambda: reject_unsupported_routing_replay(
                "recompute", train_on_sampled_tokens=True, runs_logprob_recompute=False
            ),
            r"routing_replay='recompute' captures the mask in the no-grad logprob-recompute pass",
        ),
    ],
    ids=[
        "straddling-train-group",
        "is-stage-without-correction",
        "breaker-without-stage",
        "breaker-without-correction",
        "rollout-replay-on-a-rerender",
        "recompute-replay-without-recompute",
    ],
)
def test_each_shape_is_and_replay_refusal_names_its_mechanism(refusal, match):
    with pytest.raises(ValueError, match=match):
        refusal()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
