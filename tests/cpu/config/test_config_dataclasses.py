#!/usr/bin/env python
"""SMPOConfig, EnvironmentConfig and AsyncRolloutConfig ``__post_init__`` validation and defaults."""

import contextlib
import logging
import os

import pytest

import halo.configs.grpo_environmental as grpo_environmental
from halo.configs.grpo_environmental import AsyncRolloutConfig, EnvironmentConfig
from halo.configs.grpo_offline import OfflineGRPOConfig
from halo.configs.grpo_online import OnlineGRPOScriptArguments
from halo.configs.parser import HaloArgumentParser
from halo.configs.preference import SMPOConfig
from halo.rewards.terms import EnvironmentTerm

OUTPUT_DIR = "/tmp/test_output"

# TrainingArguments' ``__post_init__`` raises when bf16 is on with no CUDA device, so configs expected to construct
# pass bf16=False. The SMPO validation tests raise before that tail.
_CPU_OK = {"output_dir": OUTPUT_DIR, "bf16": False}

_POSITIVE_ROLLOUT_KNOBS = (
    "rollout_temperature",
    "rollout_max_tokens",
    "request_timeout",
    "episode_timeout",
    "rollout_connection_timeout",
)


def test_smpo_defaults():
    cfg = SMPOConfig(**_CPU_OK)
    assert cfg.beta == 1.2
    assert cfg.target_margin == 0.35
    assert cfg.loss_type == "smooth_lower_bound"
    assert cfg.chosen_sft_ratio == 0.8
    assert cfg.use_margin_schedule is True
    assert cfg.initial_margin == 0.01
    assert cfg.lower_clip_percentile == 0.02
    assert cfg.upper_clip_percentile is None
    assert cfg.min_log_prob == -2.3
    assert cfg.max_length == 1024
    # Null shares are derived from max_length by resolve_length_budget(), so they scale with a context-resolved budget.
    assert cfg.max_prompt_length is None
    assert cfg.max_completion_length is None
    assert cfg.resolve_length_budget() == (1024, 512, 512)
    assert cfg.truncation_mode == "keep_end"
    assert cfg.disable_dropout is True
    assert cfg.padding_free is False


def test_smpo_padding_free():
    cfg = SMPOConfig(padding_free=True, **_CPU_OK)
    assert cfg.padding_free is True


def test_smpo_negative_target_margin():
    raised = False
    try:
        SMPOConfig(output_dir=OUTPUT_DIR, target_margin=-0.1)
    except ValueError as e:
        raised = True
        assert "target_margin" in str(e)
    assert raised, "Should raise ValueError for negative target_margin"


def test_smpo_margin_schedule_initial_ge_target():
    raised = False
    try:
        SMPOConfig(
            output_dir=OUTPUT_DIR,
            use_margin_schedule=True,
            initial_margin=0.5,
            target_margin=0.3,
        )
    except ValueError as e:
        raised = True
        assert "initial_margin" in str(e)
    assert raised, "Should raise ValueError when initial_margin >= target_margin"


def test_smpo_margin_schedule_equal():
    # match=: a weakened bound would otherwise reach TrainingArguments' own bf16 ValueError on the GPU-less tier.
    with pytest.raises(ValueError, match="initial_margin"):
        SMPOConfig(
            output_dir=OUTPUT_DIR,
            use_margin_schedule=True,
            initial_margin=0.35,
            target_margin=0.35,
        )


def test_smpo_margin_schedule_disabled_ok():
    cfg = SMPOConfig(use_margin_schedule=False, target_margin=0.3, **_CPU_OK)
    assert cfg.target_margin == 0.3


def test_smpo_refuses_an_initial_margin_with_no_schedule_to_read_it():
    with pytest.raises(ValueError, match="use_margin_schedule off does not support.*initial_margin"):
        SMPOConfig(use_margin_schedule=False, initial_margin=0.2, **_CPU_OK)
    assert SMPOConfig(initial_margin=0.2, **_CPU_OK).initial_margin == 0.2


def test_smpo_refuses_the_crossing_opt_in_without_a_multi_document_row():
    with pytest.raises(ValueError, match="padding_free off does not support.*allow_cross_document_linear_attention"):
        SMPOConfig(allow_cross_document_linear_attention=True, **_CPU_OK)
    assert SMPOConfig(allow_cross_document_linear_attention=True, padding_free=True, **_CPU_OK).padding_free


@pytest.mark.parametrize("knob", ["min_log_prob", "initial_min_log_prob"])
@pytest.mark.parametrize("value", [0.0, 0.5])
def test_offline_grpo_refuses_a_log_prob_floor_no_log_prob_sits_under(knob, value):
    with pytest.raises(ValueError, match=rf"{knob} must be < 0 \(log probabilities are <= 0\)"):
        OfflineGRPOConfig(**{knob: value}, **_CPU_OK)


def test_offline_grpo_refuses_a_schedule_toward_no_floor():
    with pytest.raises(ValueError, match="initial_min_log_prob=-0.5 schedules min_log_prob, which is unset"):
        OfflineGRPOConfig(min_log_prob=None, initial_min_log_prob=-0.5, **_CPU_OK)
    assert OfflineGRPOConfig(min_log_prob=None, **_CPU_OK).initial_min_log_prob is None
    assert OfflineGRPOConfig(initial_min_log_prob=-0.5, **_CPU_OK).min_log_prob == -3.0


def test_smpo_lower_clip_percentile_zero():
    raised = False
    try:
        SMPOConfig(output_dir=OUTPUT_DIR, lower_clip_percentile=0.0)
    except ValueError as e:
        raised = True
        assert "lower_clip_percentile" in str(e)
    assert raised, "Should raise ValueError for lower_clip_percentile=0"


def test_smpo_lower_clip_percentile_too_high():
    with pytest.raises(ValueError, match="lower_clip_percentile"):
        SMPOConfig(output_dir=OUTPUT_DIR, lower_clip_percentile=0.6)


def test_smpo_lower_clip_percentile_at_half():
    cfg = SMPOConfig(lower_clip_percentile=0.5, **_CPU_OK)
    assert cfg.lower_clip_percentile == 0.5


def test_smpo_upper_clip_percentile_below_half():
    raised = False
    try:
        SMPOConfig(output_dir=OUTPUT_DIR, upper_clip_percentile=0.4)
    except ValueError as e:
        raised = True
        assert "upper_clip_percentile" in str(e)
    assert raised, "Should raise ValueError for upper_clip_percentile=0.4"


def test_smpo_upper_clip_percentile_at_one():
    with pytest.raises(ValueError, match="upper_clip_percentile"):
        SMPOConfig(output_dir=OUTPUT_DIR, upper_clip_percentile=1.0)


def test_smpo_min_log_prob_positive():
    raised = False
    try:
        SMPOConfig(output_dir=OUTPUT_DIR, min_log_prob=0.5)
    except ValueError as e:
        raised = True
        assert "min_log_prob" in str(e)
    assert raised, "Should raise ValueError for positive min_log_prob"


def test_smpo_min_log_prob_zero():
    with pytest.raises(ValueError, match="min_log_prob"):
        SMPOConfig(output_dir=OUTPUT_DIR, min_log_prob=0.0)


def test_smpo_chosen_sft_ratio_out_of_range():
    for bad_val in [-0.1, 1.1]:
        raised = False
        try:
            SMPOConfig(output_dir=OUTPUT_DIR, chosen_sft_ratio=bad_val)
        except ValueError as e:
            raised = True
            assert "chosen_sft_ratio" in str(e)
        assert raised, f"Should raise ValueError for chosen_sft_ratio={bad_val}"


def test_smpo_chosen_sft_ratio_boundaries():
    cfg0 = SMPOConfig(chosen_sft_ratio=0.0, **_CPU_OK)
    assert cfg0.chosen_sft_ratio == 0.0
    cfg1 = SMPOConfig(chosen_sft_ratio=1.0, **_CPU_OK)
    assert cfg1.chosen_sft_ratio == 1.0


def test_smpo_upper_clip_percentile_valid_boundary():
    cfg_lo = SMPOConfig(upper_clip_percentile=0.5, **_CPU_OK)
    assert cfg_lo.upper_clip_percentile == 0.5
    cfg_hi = SMPOConfig(upper_clip_percentile=0.99, **_CPU_OK)
    assert cfg_hi.upper_clip_percentile == 0.99


def test_smpo_clip_percentiles_none_disables_validation():
    cfg = SMPOConfig(
        lower_clip_percentile=None,
        upper_clip_percentile=None,
        min_log_prob=None,
        **_CPU_OK,
    )
    assert cfg.lower_clip_percentile is None
    assert cfg.upper_clip_percentile is None
    assert cfg.min_log_prob is None


def test_smpo_bf16_defaults_true_when_fp16_false():
    cfg = SMPOConfig(output_dir=OUTPUT_DIR, bf16=None, fp16=True)
    assert cfg.bf16 is False

    # use_cpu=True accepts the resolved bf16=True whatever GPU is visible; a try/except on transformers' bf16 rejection
    # would pass only in a CPU-only container.
    cfg = SMPOConfig(output_dir=OUTPUT_DIR, bf16=None, fp16=False, use_cpu=True)
    assert cfg.bf16 is True


def test_smpo_invalid_loss_type_value():
    """Direct construction stores a bogus loss_type verbatim; ``HaloArgumentParser`` is the gate every entry point
    goes through."""
    cfg = SMPOConfig(loss_type="bogus", **_CPU_OK)
    assert cfg.loss_type == "bogus"

    with pytest.raises(ValueError, match="loss_type"):
        HaloArgumentParser((SMPOConfig,)).parse_dict({"loss_type": "bogus", **_CPU_OK})


def test_env_config_defaults():
    cfg = EnvironmentConfig()
    assert cfg.environment_type == "react_math"
    # The default reward is the environment's own all-or-nothing grade, at weight 1 and exponent 1.
    assert cfg.rewards == [{"source": "environment"}]
    assert cfg.reward_terms == (EnvironmentTerm(),)
    # None defers to the environment class's own default (CodeContests 15, SWE 20, ExamQA 8).
    assert cfg.max_turns is None
    assert "max_turns" not in cfg.to_env_config()
    assert cfg.environment_kwargs == {}


def test_env_config_refuses_per_env_reward_knobs():
    """A reward's magnitude is a term's ``weight``; a per-environment knob must fail at construction and not parse
    into a config that changes nothing."""
    for knob in ("success_reward", "failure_reward", "partial_reward"):
        with pytest.raises(TypeError, match=knob):
            EnvironmentConfig(**{knob: 0.5})


def test_env_config_to_env_config():
    cfg = EnvironmentConfig(
        rewards=[{"source": "environment", "weight": 2.0}],
        max_turns=20,
        environment_kwargs={"search_backend": "duckduckgo", "open_book": True},
    )
    result = cfg.to_env_config()
    assert result["reward_terms"] == [{"source": "environment", "weight": 2.0}]
    assert result["max_turns"] == 20
    assert result["search_backend"] == "duckduckgo"
    assert result["open_book"] is True


def test_env_config_to_env_config_empty_kwargs():
    cfg = EnvironmentConfig(rewards=[{"source": "environment", "weight": 1.5}], max_turns=5)
    result = cfg.to_env_config()
    # The raw term dicts travel as-is (the environment parses them); empty kwargs add nothing.
    assert result == {"reward_terms": [{"source": "environment", "weight": 1.5}], "max_turns": 5}


@pytest.mark.parametrize(("key", "owner"), [("max_turns", "max_turns"), ("reward_terms", "rewards")])
def test_env_config_kwargs_may_not_shadow_a_top_level_field(key, owner):
    """``to_env_config`` applies environment_kwargs last, so such a key would override the field past its
    validation."""
    with pytest.raises(ValueError, match=f"must not carry \\['{key}'\\].*top-level fields .*{owner}"):
        EnvironmentConfig(environment_kwargs={key: 0})
    assert key in EnvironmentConfig(max_turns=3)._core_env_config(), "the refused keys are the ones the config writes"


@pytest.mark.parametrize("value", [None, "search_backend", ["open_book"]])
def test_env_config_kwargs_must_be_a_mapping(value):
    """A bare ``environment_kwargs:`` header parses as null."""
    with pytest.raises(ValueError, match="environment_kwargs must map option names to values"):
        EnvironmentConfig(environment_kwargs=value)


def test_env_config_rejects_non_positive_max_turns():
    """``max_turns: 0`` makes the rollout loop a no-op, a silently all-zero batch; it must fail before Ray and vLLM
    are up."""
    for bad in (0, -1):
        try:
            EnvironmentConfig(max_turns=bad)
        except ValueError as e:
            assert "max_turns" in str(e)
        else:
            raise AssertionError(f"max_turns={bad} must raise, not silently produce empty episodes")
    assert EnvironmentConfig(max_turns=1).max_turns == 1
    assert EnvironmentConfig().max_turns is None


def test_env_config_max_turns_guard_survives_cli_override(tmp_path):
    config = tmp_path / "env.yaml"
    config.write_text("max_turns: 4\n")
    parser = HaloArgumentParser((EnvironmentConfig,))
    with pytest.raises(ValueError, match="max_turns"):
        parser.parse_dict(parser.load_values([str(config), "--max_turns=0"]))


def test_env_config_custom_env_type_passthrough():
    """environment_type is not validated here; the registry resolves it later."""
    cfg = EnvironmentConfig(environment_type="code_contests", environment_kwargs={"timeout_per_test": 10})
    assert cfg.environment_type == "code_contests"
    out = cfg.to_env_config()
    assert "environment_type" not in out
    assert out["timeout_per_test"] == 10


def test_env_config_spells_environment_type_as_the_registry_keys_it(tmp_path):
    """The name is lowercased once at parse, on the YAML and CLI-override paths alike, so no consumer comparing it
    refuses it."""
    assert EnvironmentConfig(environment_type="Code_Contests").environment_type == "code_contests"
    config = tmp_path / "env.yaml"
    config.write_text("environment_type: react_math\n")
    parser = HaloArgumentParser((EnvironmentConfig,))
    (cfg,) = parser.parse_dict(parser.load_values([str(config), "--environment_type=Native_Math"]))
    assert cfg.environment_type == "native_math"
    with pytest.raises(ValueError, match="^environment_type must be a registry name, got None$"):
        EnvironmentConfig(environment_type=None)


def test_rlvr_args_refuse_a_veto_judge():
    """The online arm's reward functions are independent, with no objective for a ``checks`` judge to gate."""
    judge = {"source": "judge", "name": "conduct"}
    checks = [{"name": "cheated", "description": "Hard-coded the expected output.", "veto": True}]
    with pytest.raises(ValueError, match=r"\['conduct'\] list 'checks'.*list 'requirements' instead"):
        OnlineGRPOScriptArguments(rewards=[{**judge, "checks": checks}])
    requirements = [{"name": "clear", "description": "Clear."}]
    (term,) = OnlineGRPOScriptArguments(rewards=[{**judge, "requirements": requirements}]).reward_terms
    assert not term.is_veto


def test_async_config_defaults():
    cfg = AsyncRolloutConfig()
    assert cfg.rollout_actor_budget == 256
    assert cfg.max_concurrent_rollouts is None
    assert cfg.rollout_server_url == "http://localhost:8000"
    assert cfg.rollout_temperature == 0.7
    assert cfg.rollout_top_p == 0.95
    assert cfg.rollout_max_tokens == 32768
    assert cfg.enable_prefetch is None
    assert cfg.max_retries == 3
    assert cfg.retry_base_wait == 1.0


def test_async_config_get_server_urls_single():
    cfg = AsyncRolloutConfig(rollout_server_url="http://gpu1:8000")
    urls = cfg.get_server_urls()
    assert urls == ["http://gpu1:8000"]


def test_async_config_get_server_urls_multi():
    cfg = AsyncRolloutConfig(
        rollout_server_configs=[
            {"url": "http://node1:8000", "group_port": 51216},
            {"url": "http://node2:8000", "group_port": 51217},
        ]
    )
    urls = cfg.get_server_urls()
    assert urls == ["http://node1:8000", "http://node2:8000"]


def test_async_config_get_server_urls_empty_list_falls_back():
    cfg = AsyncRolloutConfig(rollout_server_url="http://gpu1:8000", rollout_server_configs=[])
    assert cfg.get_server_urls() == ["http://gpu1:8000"]


def test_async_config_get_rollout_config_threads_fields():
    cfg = AsyncRolloutConfig(
        rollout_temperature=0.3,
        rollout_top_p=0.8,
        rollout_max_tokens=256,
        model_name="my-model",
        request_timeout=30.0,
        max_retries=5,
        retry_base_wait=2.0,
    )
    rc = cfg.get_rollout_config()
    assert rc.temperature == 0.3
    assert rc.top_p == 0.8
    assert rc.max_tokens == 256
    assert rc.model_name == "my-model"
    assert rc.request_timeout == 30.0
    assert rc.max_retries == 5
    assert rc.retry_base_wait == 2.0


@contextlib.contextmanager
def _nccl_watchdog_minutes(minutes: str | None):
    """Set/clear DIST_NCCL_TIMEOUT_MINUTES for the duration of the block, restoring the prior value."""
    prior = os.environ.get("DIST_NCCL_TIMEOUT_MINUTES")
    if minutes is None:
        os.environ.pop("DIST_NCCL_TIMEOUT_MINUTES", None)
    else:
        os.environ["DIST_NCCL_TIMEOUT_MINUTES"] = minutes
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("DIST_NCCL_TIMEOUT_MINUTES", None)
        else:
            os.environ["DIST_NCCL_TIMEOUT_MINUTES"] = prior


def test_async_config_episode_timeout_above_watchdog_raises():
    """A straggler would trip the peers' per-step collective before it is cancelled, aborting the run on an opaque
    watchdog timeout."""
    with _nccl_watchdog_minutes(None):  # default watchdog = 30 min = 1800s
        cfg = AsyncRolloutConfig(episode_timeout=2700.0)
        raised = False
        try:
            cfg.get_rollout_config()
        except ValueError as e:
            raised = True
            assert "episode_timeout" in str(e) and "watchdog" in str(e)
        assert raised, "episode_timeout above the watchdog must raise"


def test_async_config_episode_timeout_below_watchdog_ok():
    with _nccl_watchdog_minutes("60"):  # watchdog = 3600s
        cfg = AsyncRolloutConfig(episode_timeout=2700.0)
        rc = cfg.get_rollout_config()
        assert rc.episode_timeout == 2700.0


def test_async_config_episode_timeout_equal_watchdog_does_not_raise():
    """The stock default (1800s on both) is a race, not a certainty: it warns and does not raise, or every default
    env-GRPO run would break at construction."""
    with _nccl_watchdog_minutes(None):
        cfg = AsyncRolloutConfig(episode_timeout=1800.0)
        rc = cfg.get_rollout_config()
        assert rc.episode_timeout == 1800.0


@pytest.mark.parametrize("main_process", [True, False])
def test_async_config_watchdog_warnings_are_said_once_not_once_per_rank(monkeypatch, caplog, main_process):
    """Every rank builds its rollout config, and both near-watchdog warnings describe the config alone."""
    monkeypatch.setattr(grpo_environmental, "is_global_main_process", lambda: main_process)
    with _nccl_watchdog_minutes(None), caplog.at_level(logging.WARNING, logger=grpo_environmental.logger.name):
        AsyncRolloutConfig(episode_timeout=1800.0, request_timeout=1800.0).get_rollout_config()
    for warning in ("episode_timeout (1800s) is within", "Rollout retry budget"):
        assert (warning in caplog.text) is main_process, warning


@pytest.mark.parametrize("field", _POSITIVE_ROLLOUT_KNOBS)
@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_async_config_rejects_non_positive_or_non_finite_rollout_knobs(field, bad):
    """NaN passes every ordered comparison and infinity passes ``> 0``. ``rollout_temperature`` divides the chunked
    log-prob sweep, ``rollout_max_tokens`` becomes the dr_grpo loss normalizer, and a non-positive deadline
    cancels every episode on entry."""
    with pytest.raises(ValueError, match=field):
        AsyncRolloutConfig(**{field: bad})


@pytest.mark.parametrize("bad", [0, -1])
def test_async_config_rejects_a_vanishing_concurrency_cap(bad):
    """The field is read as ``value or default``, so a 0 reads as unset and the per-rank semaphore cap disappears."""
    with pytest.raises(ValueError, match="max_concurrent_rollouts"):
        AsyncRolloutConfig(max_concurrent_rollouts=bad)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, float("nan"), float("inf")])
def test_async_config_rejects_out_of_range_top_p(bad):
    """The value is forwarded verbatim, so an out-of-range one is a per-request server rejection on every episode."""
    with pytest.raises(ValueError, match="rollout_top_p"):
        AsyncRolloutConfig(rollout_top_p=bad)


@pytest.mark.parametrize(
    ("knob", "bad"),
    [
        ("rollout_top_k", 0),
        ("rollout_top_k", -2),
        ("rollout_min_p", 1.5),
        ("rollout_min_p", -0.1),
        ("rollout_min_p", float("nan")),
        ("rollout_repetition_penalty", 0.0),
        ("rollout_repetition_penalty", 2.5),
        ("rollout_repetition_penalty", float("nan")),
    ],
)
def test_async_config_rejects_a_filter_the_engines_refuse(knob, bad):
    """Each filter goes out on every request, so a value SGLang refuses (top_k 0, min_p outside [0, 1], a penalty
    outside (0, 2]) rejects every episode."""
    with pytest.raises(ValueError, match=knob):
        AsyncRolloutConfig(**{knob: bad})


@pytest.mark.parametrize("bad", [True, 20.5, 20.0, "20"])
def test_async_config_rejects_a_top_k_that_is_not_an_int(bad):
    """``true`` clears the range as 1, every rollout sampling greedily, and a fractional value fails every request."""
    with pytest.raises(ValueError, match="rollout_top_k must be an int"):
        AsyncRolloutConfig(rollout_top_k=bad)


@pytest.mark.parametrize(
    ("knob", "good"),
    [("rollout_top_k", -1), ("rollout_top_k", 1), ("rollout_min_p", 1.0), ("rollout_repetition_penalty", 2.0)],
)
def test_async_config_accepts_a_filter_at_its_range_edge(knob, good):
    assert getattr(AsyncRolloutConfig(**{knob: good}), knob) == good


@pytest.mark.parametrize("bad", [-0.5, float("nan"), float("inf")])
def test_async_config_rejects_a_non_finite_or_negative_retry_base_wait(bad):
    """The backoff grows from this base: a negative one shrinks it, NaN slips past the sign check, and an infinite
    one parks the first retry forever."""
    with pytest.raises(ValueError, match="retry_base_wait"):
        AsyncRolloutConfig(retry_base_wait=bad)
    AsyncRolloutConfig(retry_base_wait=0.0)  # no raise: retry immediately


def test_async_config_rejects_a_thinking_budget_that_eats_the_whole_turn():
    """At or above the turn cap every turn spends its whole budget on reasoning and is cut before the answer or tool
    call, which trains as a length-cut turn forever."""
    with pytest.raises(ValueError, match="rollout_max_thinking_tokens"):
        AsyncRolloutConfig(rollout_max_tokens=4096, rollout_max_thinking_tokens=4096)
    AsyncRolloutConfig(rollout_max_tokens=4096, rollout_max_thinking_tokens=4095)  # no raise


@pytest.mark.parametrize("bad", [4095, 0, -1, True, 4096.0, "4096"])
def test_async_config_rejects_an_episode_budget_below_one_turn_or_not_a_count(bad):
    """Below ``rollout_max_tokens`` the first turn could never use the per-turn cap, a bool is an int that spells a
    mistake, and a float or string reaches the engine's ``max_tokens`` as no count."""
    with pytest.raises(
        ValueError, match=r"rollout_max_episode_tokens must be an integer >= 4096 \(at least rollout_max_tokens\)"
    ):
        AsyncRolloutConfig(rollout_max_tokens=4096, rollout_max_episode_tokens=bad)


def test_async_config_mirrors_the_episode_budget_into_the_rollout_config():
    """One whole turn fits at equality, null is unbounded, and the actors read only the built ``RolloutConfig``."""
    one_turn = AsyncRolloutConfig(rollout_max_tokens=4096, rollout_max_episode_tokens=4096)
    assert one_turn.get_rollout_config().max_episode_tokens == 4096
    bounded = AsyncRolloutConfig(rollout_max_tokens=4096, rollout_max_episode_tokens=65536)
    assert bounded.get_rollout_config().max_episode_tokens == 65536
    assert AsyncRolloutConfig(rollout_max_episode_tokens=None).get_rollout_config().max_episode_tokens is None


@pytest.mark.parametrize("bad", [-1, 0, float("nan"), True, 8000.5], ids=["negative", "zero", "nan", "bool", "float"])
def test_async_config_rejects_a_thinking_budget_that_is_not_a_positive_int(bad):
    """0 would be sent as 1 while counting as no cap at all, so the forced-close gates never arm; ``null`` is the
    spelling for no run-wide cap."""
    with pytest.raises(
        ValueError, match=r"rollout_max_thinking_tokens must be an integer in \[1, \d+\) \(below rollout_max_tokens\)"
    ):
        AsyncRolloutConfig(rollout_max_thinking_tokens=bad)


def test_async_config_range_guards_survive_a_cli_override(tmp_path):
    config = tmp_path / "async.yaml"
    config.write_text("rollout_temperature: 1.0\n")
    parser = HaloArgumentParser((AsyncRolloutConfig,))
    with pytest.raises(ValueError, match="rollout_temperature"):
        parser.parse_dict(parser.load_values([str(config), "--rollout_temperature=0"]))


def test_async_config_template_variables_are_the_yamls_own_kwargs():
    """No run-wide variable is injected beside the YAML's kwargs (the per-episode level and budget travel per
    request), and the rollout config gets its own copy."""
    run_kwargs = {"preserve_thinking": True}
    config = AsyncRolloutConfig(rollout_chat_template_kwargs=run_kwargs)
    mirrored = config.get_rollout_config().chat_template_kwargs
    assert mirrored == run_kwargs and mirrored is not config.rollout_chat_template_kwargs
    assert AsyncRolloutConfig().get_rollout_config().chat_template_kwargs == {}
    with pytest.raises(ValueError, match="must not carry"):
        AsyncRolloutConfig(rollout_chat_template_kwargs={"reasoning_budget": 8192})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
