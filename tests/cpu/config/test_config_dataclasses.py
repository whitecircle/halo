#!/usr/bin/env python
"""
Tests for configuration dataclasses: SmoothMarginPOConfig, EnvironmentConfig,
and AsyncTrainingConfig __post_init__ validation and defaults.

Run: python tests/cpu/config/test_config_dataclasses.py
"""

import contextlib
import logging
import os

import pytest

import src.configs.async_training_config as async_training_config
from src.args.rlvr_online_grpo_args import RLVROnlineGRPOScriptArguments
from src.configs.async_training_config import POSITIVE_ROLLOUT_FIELDS, AsyncTrainingConfig
from src.configs.environment_config import EnvironmentConfig
from src.configs.smpo_config import SmoothMarginPOConfig
from src.rewards.terms import EnvironmentTerm
from src.training.parser import H4ArgumentParser

OUTPUT_DIR = "/tmp/test_output"

# SmoothMarginPOConfig extends transformers.TrainingArguments, whose __post_init__ (called at the
# END of the SMPO __post_init__, after all SMPO-specific field validation) raises
# "Your setup doesn't support bf16/gpu" when bf16 is on and no CUDA device is visible. The CPU test
# container has no GPU, so VALID configs (those expected to construct successfully) pass bf16=False
# to reach that tail cleanly. The SMPO validation-branch tests raise BEFORE this tail, so they leave
# bf16 at its default and never reach the GPU check.
_CPU_OK = {"output_dir": OUTPUT_DIR, "bf16": False}

# The rollout knobs AsyncTrainingConfig refuses at or below 0, spelled out so the sweep is static at
# collection; test_positive_rollout_knob_sweep_covers_the_production_tuple holds it to the source.
_POSITIVE_ROLLOUT_KNOBS = (
    "rollout_temperature",
    "rollout_max_tokens",
    "request_timeout",
    "episode_timeout",
    "rollout_connection_timeout",
)


# SmoothMarginPOConfig tests


def test_smpo_defaults():
    """Default SMPO config should be valid."""

    cfg = SmoothMarginPOConfig(**_CPU_OK)
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
    # The two shares default to null and are derived from max_length by resolve_length_budget(),
    # so they scale with a context-resolved budget instead of pinning prompts at a constant.
    assert cfg.max_prompt_length is None
    assert cfg.max_completion_length is None
    assert cfg.resolve_length_budget() == (1024, 512, 512)
    assert cfg.truncation_mode == "keep_end"
    assert cfg.disable_dropout is True
    assert cfg.padding_free is False


def test_smpo_padding_free():
    """padding_free config field should be settable."""

    cfg = SmoothMarginPOConfig(padding_free=True, **_CPU_OK)
    assert cfg.padding_free is True


def test_smpo_negative_target_margin():
    """target_margin < 0 should raise ValueError."""

    raised = False
    try:
        SmoothMarginPOConfig(output_dir=OUTPUT_DIR, target_margin=-0.1)
    except ValueError as e:
        raised = True
        assert "target_margin" in str(e)
    assert raised, "Should raise ValueError for negative target_margin"


def test_smpo_margin_schedule_initial_ge_target():
    """initial_margin >= target_margin with use_margin_schedule should raise ValueError."""

    raised = False
    try:
        SmoothMarginPOConfig(
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
    """initial_margin == target_margin with use_margin_schedule should raise ValueError."""

    # match=: without it, a weakened bound lets construction reach TrainingArguments'
    # own "no bf16 on this setup" ValueError on the GPU-less tier, which a bare except accepts.
    with pytest.raises(ValueError, match="initial_margin"):
        SmoothMarginPOConfig(
            output_dir=OUTPUT_DIR,
            use_margin_schedule=True,
            initial_margin=0.35,
            target_margin=0.35,
        )


def test_smpo_margin_schedule_disabled_ok():
    """initial_margin >= target_margin is fine when use_margin_schedule=False."""

    cfg = SmoothMarginPOConfig(
        use_margin_schedule=False,
        initial_margin=0.5,
        target_margin=0.3,
        **_CPU_OK,
    )
    assert cfg.initial_margin == 0.5


def test_smpo_lower_clip_percentile_zero():
    """lower_clip_percentile = 0 should raise ValueError (must be > 0)."""

    raised = False
    try:
        SmoothMarginPOConfig(output_dir=OUTPUT_DIR, lower_clip_percentile=0.0)
    except ValueError as e:
        raised = True
        assert "lower_clip_percentile" in str(e)
    assert raised, "Should raise ValueError for lower_clip_percentile=0"


def test_smpo_lower_clip_percentile_too_high():
    """lower_clip_percentile > 0.5 should raise ValueError."""

    with pytest.raises(ValueError, match="lower_clip_percentile"):
        SmoothMarginPOConfig(output_dir=OUTPUT_DIR, lower_clip_percentile=0.6)


def test_smpo_lower_clip_percentile_at_half():
    """lower_clip_percentile = 0.5 should be valid (boundary)."""

    cfg = SmoothMarginPOConfig(lower_clip_percentile=0.5, **_CPU_OK)
    assert cfg.lower_clip_percentile == 0.5


def test_smpo_upper_clip_percentile_below_half():
    """upper_clip_percentile < 0.5 should raise ValueError."""

    raised = False
    try:
        SmoothMarginPOConfig(output_dir=OUTPUT_DIR, upper_clip_percentile=0.4)
    except ValueError as e:
        raised = True
        assert "upper_clip_percentile" in str(e)
    assert raised, "Should raise ValueError for upper_clip_percentile=0.4"


def test_smpo_upper_clip_percentile_at_one():
    """upper_clip_percentile = 1.0 should raise ValueError (must be < 1)."""

    with pytest.raises(ValueError, match="upper_clip_percentile"):
        SmoothMarginPOConfig(output_dir=OUTPUT_DIR, upper_clip_percentile=1.0)


def test_smpo_min_log_prob_positive():
    """min_log_prob >= 0 should raise ValueError."""

    raised = False
    try:
        SmoothMarginPOConfig(output_dir=OUTPUT_DIR, min_log_prob=0.5)
    except ValueError as e:
        raised = True
        assert "min_log_prob" in str(e)
    assert raised, "Should raise ValueError for positive min_log_prob"


def test_smpo_min_log_prob_zero():
    """min_log_prob = 0 should raise ValueError."""

    with pytest.raises(ValueError, match="min_log_prob"):
        SmoothMarginPOConfig(output_dir=OUTPUT_DIR, min_log_prob=0.0)


def test_smpo_chosen_sft_ratio_out_of_range():
    """chosen_sft_ratio outside [0, 1] should raise ValueError."""

    for bad_val in [-0.1, 1.1]:
        raised = False
        try:
            SmoothMarginPOConfig(output_dir=OUTPUT_DIR, chosen_sft_ratio=bad_val)
        except ValueError as e:
            raised = True
            assert "chosen_sft_ratio" in str(e)
        assert raised, f"Should raise ValueError for chosen_sft_ratio={bad_val}"


def test_smpo_chosen_sft_ratio_boundaries():
    """chosen_sft_ratio = 0 and 1 should be valid."""

    cfg0 = SmoothMarginPOConfig(chosen_sft_ratio=0.0, **_CPU_OK)
    assert cfg0.chosen_sft_ratio == 0.0
    cfg1 = SmoothMarginPOConfig(chosen_sft_ratio=1.0, **_CPU_OK)
    assert cfg1.chosen_sft_ratio == 1.0


def test_smpo_upper_clip_percentile_valid_boundary():
    """upper_clip_percentile = 0.5 (lower boundary) and 0.99 (interior) are valid."""

    cfg_lo = SmoothMarginPOConfig(upper_clip_percentile=0.5, **_CPU_OK)
    assert cfg_lo.upper_clip_percentile == 0.5
    cfg_hi = SmoothMarginPOConfig(upper_clip_percentile=0.99, **_CPU_OK)
    assert cfg_hi.upper_clip_percentile == 0.99


def test_smpo_clip_percentiles_none_disables_validation():
    """Setting both clip percentiles to None disables the range checks (no raise)."""

    cfg = SmoothMarginPOConfig(
        lower_clip_percentile=None,
        upper_clip_percentile=None,
        min_log_prob=None,
        **_CPU_OK,
    )
    assert cfg.lower_clip_percentile is None
    assert cfg.upper_clip_percentile is None
    assert cfg.min_log_prob is None


def test_smpo_bf16_defaults_true_when_fp16_false():
    """bf16=None resolves to ``not fp16`` in __post_init__ (toolkit bf16-by-default)."""

    # fp16=True path: bf16 resolves to False.
    cfg = SmoothMarginPOConfig(output_dir=OUTPUT_DIR, bf16=None, fp16=True)
    assert cfg.bf16 is False

    # fp16=False path: bf16 resolves to True. Assert it directly rather than relying on the
    # transformers bf16/gpu validation as a proxy — that check rejects bf16 on a GPU-less host
    # but ACCEPTS it on a GPU one, so a try/except on the rejection makes the test pass only in
    # a CPU-only container and fail under `--gpus all`. use_cpu=True accepts the resolved
    # bf16=True regardless of GPU visibility, so the assertion is hardware-independent.
    cfg = SmoothMarginPOConfig(output_dir=OUTPUT_DIR, bf16=None, fp16=False, use_cpu=True)
    assert cfg.bf16 is True


def test_smpo_invalid_loss_type_value():
    """Direct construction does not range-check loss_type — the parser is where the Literal bites.

    SMPO's ``__post_init__`` stores a bogus value verbatim, so the dataclass Literal is advisory to
    a caller that builds the config by hand. Every supported entry point goes through
    ``H4ArgumentParser`` instead, and that is the gate this pins: if the parser's Literal check
    stops firing, a typo'd loss_type reaches the loss dispatch as a live config.
    """

    cfg = SmoothMarginPOConfig(loss_type="bogus", **_CPU_OK)
    assert cfg.loss_type == "bogus"

    with pytest.raises(ValueError, match="loss_type"):
        H4ArgumentParser((SmoothMarginPOConfig,)).parse_dict({"loss_type": "bogus", **_CPU_OK})


# EnvironmentConfig tests


def test_env_config_defaults():
    """EnvironmentConfig should have sensible defaults."""

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
    """A reward's magnitude is a term's ``weight``, not a per-environment field: such a knob must
    fail at construction rather than parse into a config that changes nothing about the run."""

    for knob in ("success_reward", "failure_reward", "partial_reward"):
        with pytest.raises(TypeError, match=knob):
            EnvironmentConfig(**{knob: 0.5})


def test_env_config_to_env_config():
    """to_env_config() should merge the reward terms and turn cap with environment_kwargs."""

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
    """to_env_config() with empty environment_kwargs should return the reward terms and turn cap only."""

    cfg = EnvironmentConfig(rewards=[{"source": "environment", "weight": 1.5}], max_turns=5)
    result = cfg.to_env_config()
    # The raw term dicts travel as-is (the environment parses them); empty kwargs add nothing.
    assert result == {"reward_terms": [{"source": "environment", "weight": 1.5}], "max_turns": 5}


@pytest.mark.parametrize(("key", "owner"), [("max_turns", "max_turns"), ("reward_terms", "rewards")])
def test_env_config_kwargs_may_not_shadow_a_top_level_field(key, owner):
    """``to_env_config`` applies environment_kwargs last, so a key it writes from a top-level field would
    override that field past its validation (a ``max_turns: 0`` turning every episode into a no-op)."""
    with pytest.raises(ValueError, match=f"must not carry \\['{key}'\\].*top-level fields .*{owner}"):
        EnvironmentConfig(environment_kwargs={key: 0})
    assert key in EnvironmentConfig(max_turns=3)._core_env_config(), "the refused keys are the ones the config writes"


def test_env_config_rejects_non_positive_max_turns():
    """``max_turns: 0`` makes the rollout loop a no-op, so every episode returns reward 0 with no
    error — a silently all-zero batch. It must fail at parse time, not after Ray and vLLM are up."""

    for bad in (0, -1):
        try:
            EnvironmentConfig(max_turns=bad)
        except ValueError as e:
            assert "max_turns" in str(e)
        else:
            raise AssertionError(f"max_turns={bad} must raise, not silently produce empty episodes")
    assert EnvironmentConfig(max_turns=1).max_turns == 1  # the boundary is still allowed
    assert EnvironmentConfig().max_turns is None  # null still defers to the environment class


def test_env_config_max_turns_guard_survives_cli_override():
    """CLI overrides land via setattr, so ``__post_init__`` never re-runs; the RangeValidatedConfig
    seam must re-check them."""

    cfg = EnvironmentConfig()
    cfg.max_turns = 0
    try:
        cfg.__post_override__({"max_turns"})
    except ValueError as e:
        assert "max_turns" in str(e)
    else:
        raise AssertionError("--max_turns=0 must be rejected on the override path too")


def test_env_config_custom_env_type_passthrough():
    """environment_type is not validated here; the registry resolves it later."""

    cfg = EnvironmentConfig(environment_type="code_contests", environment_kwargs={"timeout_per_test": 10})
    assert cfg.environment_type == "code_contests"
    # environment_type is NOT part of to_env_config() — only reward/turn settings + kwargs are.
    out = cfg.to_env_config()
    assert "environment_type" not in out
    assert out["timeout_per_test"] == 10


def test_env_config_spells_environment_type_as_the_registry_keys_it(tmp_path):
    """The registry lowercases the name it resolves, so a mixed-case YAML trains; the name is lowercased
    once at parse, on the YAML and the CLI-override path alike, so no consumer comparing it refuses it."""
    assert EnvironmentConfig(environment_type="Code_Contests").environment_type == "code_contests"
    config = tmp_path / "env.yaml"
    config.write_text("environment_type: react_math\n")
    (cfg,) = H4ArgumentParser((EnvironmentConfig,)).parse_yaml_and_args(
        str(config), ["--environment_type=Native_Math"]
    )
    assert cfg.environment_type == "native_math"
    with pytest.raises(ValueError, match="^environment_type must be a registry name, got None$"):
        EnvironmentConfig(environment_type=None)


# RLVROnlineGRPOScriptArguments reward terms


def test_rlvr_args_refuse_a_veto_judge():
    """A ``checks`` judge gates an environment objective; the online arm's reward functions are
    independent, with no objective to gate, so the term is refused at parse time."""
    judge = {"source": "judge", "name": "conduct"}
    checks = [{"name": "cheated", "description": "Hard-coded the expected output.", "veto": True}]
    with pytest.raises(ValueError, match=r"\['conduct'\] list 'checks'.*list 'requirements' instead"):
        RLVROnlineGRPOScriptArguments(rewards=[{**judge, "checks": checks}])
    requirements = [{"name": "clear", "description": "Clear."}]
    (term,) = RLVROnlineGRPOScriptArguments(rewards=[{**judge, "requirements": requirements}]).reward_terms
    assert not term.is_veto


# AsyncTrainingConfig tests


def test_async_config_defaults():
    """AsyncTrainingConfig should have sensible defaults."""
    cfg = AsyncTrainingConfig()
    assert cfg.num_rollout_workers == 64
    assert cfg.max_concurrent_rollouts is None
    assert cfg.rollout_server_url == "http://localhost:8000"
    assert cfg.rollout_temperature == 0.7
    assert cfg.rollout_top_p == 0.95
    assert cfg.rollout_max_tokens == 32768
    assert cfg.enable_prefetch is True
    assert cfg.max_retries == 3
    assert cfg.retry_base_wait == 1.0


def test_async_config_get_server_urls_single():
    """get_server_urls() should return single URL when no multi-server config."""
    cfg = AsyncTrainingConfig(rollout_server_url="http://gpu1:8000")
    urls = cfg.get_server_urls()
    assert urls == ["http://gpu1:8000"]


def test_async_config_get_server_urls_multi():
    """get_server_urls() should return URLs from rollout_server_configs when set."""
    cfg = AsyncTrainingConfig(
        rollout_server_configs=[
            {"url": "http://node1:8000", "group_port": 51216},
            {"url": "http://node2:8000", "group_port": 51217},
        ]
    )
    urls = cfg.get_server_urls()
    assert urls == ["http://node1:8000", "http://node2:8000"]


def test_async_config_get_server_urls_empty_list_falls_back():
    """An empty (falsy) rollout_server_configs list falls back to the single rollout_server_url."""
    cfg = AsyncTrainingConfig(rollout_server_url="http://gpu1:8000", rollout_server_configs=[])
    assert cfg.get_server_urls() == ["http://gpu1:8000"]


def test_async_config_get_rollout_config_threads_fields():
    """get_rollout_config() forwards the rollout/retry knobs into RolloutConfig."""
    cfg = AsyncTrainingConfig(
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
    """episode_timeout > the NCCL watchdog must fail fast: a straggler would trip the peers' per-step
    collective before it is cancelled, aborting the run on an opaque watchdog timeout."""
    with _nccl_watchdog_minutes(None):  # default watchdog = 30 min = 1800s
        cfg = AsyncTrainingConfig(episode_timeout=2700.0)  # 2700 > 1800
        raised = False
        try:
            cfg.get_rollout_config()
        except ValueError as e:
            raised = True
            assert "episode_timeout" in str(e) and "watchdog" in str(e)
        assert raised, "episode_timeout above the watchdog must raise"


def test_async_config_episode_timeout_below_watchdog_ok():
    """A raised watchdog admits a longer episode_timeout: 2700s < 3600s (60 min) builds cleanly and
    threads the value into RolloutConfig."""
    with _nccl_watchdog_minutes("60"):  # watchdog = 3600s
        cfg = AsyncTrainingConfig(episode_timeout=2700.0)
        rc = cfg.get_rollout_config()
        assert rc.episode_timeout == 2700.0


def test_async_config_episode_timeout_equal_watchdog_does_not_raise():
    """The stock default (episode_timeout == watchdog == 1800s) is a race, not a certainty: it warns but
    must NOT raise, or every default env-GRPO run would break at construction."""
    with _nccl_watchdog_minutes(None):
        cfg = AsyncTrainingConfig(episode_timeout=1800.0)  # == default 1800s watchdog
        rc = cfg.get_rollout_config()  # no raise
        assert rc.episode_timeout == 1800.0


@pytest.mark.parametrize("main_process", [True, False])
def test_async_config_watchdog_warnings_are_said_once_not_once_per_rank(monkeypatch, caplog, main_process):
    """Every rank builds its rollout config, and both near-watchdog warnings describe the config alone."""
    monkeypatch.setattr(async_training_config, "is_global_main_process", lambda: main_process)
    with _nccl_watchdog_minutes(None), caplog.at_level(logging.WARNING, logger=async_training_config.logger.name):
        AsyncTrainingConfig(episode_timeout=1800.0, request_timeout=1800.0).get_rollout_config()
    for warning in ("episode_timeout (1800s) is within", "Rollout retry budget"):
        assert (warning in caplog.text) is main_process, warning


def test_positive_rollout_knob_sweep_covers_the_production_tuple():
    """The sweep below spells its fields out; this fails if the guarded tuple grows or shrinks."""

    assert set(POSITIVE_ROLLOUT_FIELDS) == set(_POSITIVE_ROLLOUT_KNOBS)


@pytest.mark.parametrize("field", _POSITIVE_ROLLOUT_KNOBS)
@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_async_config_rejects_non_positive_or_non_finite_rollout_knobs(field, bad):
    """Each of these reaches a consumer with no reading for 0, a negative, NaN or infinity, far from
    the knob. NaN passes every ordered comparison and infinity passes ``> 0``, so both must be refused
    up front rather than by the sign check alone.

    ``rollout_temperature`` overwrites the trainer's own and then divides the chunked log-prob sweep
    (0 → ZeroDivisionError inside the first optimizer step); ``rollout_max_tokens`` becomes TRL's
    ``max_completion_length``, the dr_grpo loss normalizer; the three deadlines are compared against
    wall-clock, so a non-positive one cancels every episode on entry and the run halts two steps
    later reporting an empty batch instead of a bad config.
    """
    with pytest.raises(ValueError, match=field):
        AsyncTrainingConfig(**{field: bad})


@pytest.mark.parametrize("bad", [0, -1])
def test_async_config_rejects_a_vanishing_concurrency_cap(bad):
    """``max_concurrent_rollouts`` is read as ``value or default``, so a 0 reads as "unset" and the
    per-rank semaphore cap disappears — every rollout of the round hits the servers at once."""
    with pytest.raises(ValueError, match="max_concurrent_rollouts"):
        AsyncTrainingConfig(max_concurrent_rollouts=bad)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, float("nan"), float("inf")])
def test_async_config_rejects_out_of_range_top_p(bad):
    """``rollout_top_p`` is forwarded verbatim, so an out-of-range value is a per-request server
    rejection — every episode errors and the step reads as a dead environment."""
    with pytest.raises(ValueError, match="rollout_top_p"):
        AsyncTrainingConfig(rollout_top_p=bad)


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
    """Each filter goes out on every request, so a value SGLang refuses (top_k 0, min_p outside
    [0, 1], a penalty outside (0, 2]) is a per-request rejection on every episode."""
    with pytest.raises(ValueError, match=knob):
        AsyncTrainingConfig(**{knob: bad})


@pytest.mark.parametrize(
    ("knob", "good"), [("rollout_top_k", 1), ("rollout_min_p", 1.0), ("rollout_repetition_penalty", 2.0)]
)
def test_async_config_accepts_a_filter_at_its_range_edge(knob, good):
    """Anti-vacuity for the refusals above: the edge of each accepted range constructs."""
    assert getattr(AsyncTrainingConfig(**{knob: good}), knob) == good


@pytest.mark.parametrize("bad", [-0.5, float("nan"), float("inf")])
def test_async_config_rejects_a_non_finite_or_negative_retry_base_wait(bad):
    """The retry backoff grows from ``retry_base_wait``: a negative base shrinks it, NaN slips past
    the sign check, and an infinite base parks the first retry forever."""
    with pytest.raises(ValueError, match="retry_base_wait"):
        AsyncTrainingConfig(retry_base_wait=bad)
    AsyncTrainingConfig(retry_base_wait=0.0)  # no raise: retry immediately


def test_async_config_rejects_a_thinking_budget_that_eats_the_whole_turn():
    """The answer room is ``rollout_max_tokens - rollout_max_thinking_tokens``, none where they meet.

    At or above the turn cap the floor hides the mistake: every turn spends its whole budget on
    reasoning and is cut before the answer or tool call, which trains as a length-cut turn forever.
    """
    with pytest.raises(ValueError, match="rollout_max_thinking_tokens"):
        AsyncTrainingConfig(rollout_max_tokens=4096, rollout_max_thinking_tokens=4096)
    AsyncTrainingConfig(rollout_max_tokens=4096, rollout_max_thinking_tokens=4095)  # no raise


@pytest.mark.parametrize("bad", [4095, 0, -1, True, 4096.0, "4096"])
def test_async_config_rejects_an_episode_budget_below_one_turn_or_not_a_count(bad):
    """The episode budget narrows every turn's cap to what is left: below ``rollout_max_tokens`` the
    first turn could never use the per-turn cap the run states, a bool is an int that spells a mistake,
    and a float or a string would reach the engine's ``max_tokens`` as no count. Re-checked on a CLI
    override, which never re-runs ``__post_init__``."""
    with pytest.raises(ValueError, match="rollout_max_episode_tokens must be an int >= rollout_max_tokens"):
        AsyncTrainingConfig(rollout_max_tokens=4096, rollout_max_episode_tokens=bad)
    cfg = AsyncTrainingConfig(rollout_max_tokens=4096)
    cfg.rollout_max_episode_tokens = bad
    with pytest.raises(ValueError, match="rollout_max_episode_tokens must be an int >= rollout_max_tokens"):
        cfg.__post_override__({"rollout_max_episode_tokens"})


def test_async_config_mirrors_the_episode_budget_into_the_rollout_config():
    """One whole turn fits at equality, null is unbounded, and the actors read only the built
    ``RolloutConfig``, so the knob has to land on ``max_episode_tokens``."""
    one_turn = AsyncTrainingConfig(rollout_max_tokens=4096, rollout_max_episode_tokens=4096)
    assert one_turn.get_rollout_config().max_episode_tokens == 4096
    bounded = AsyncTrainingConfig(rollout_max_tokens=4096, rollout_max_episode_tokens=65536)
    assert bounded.get_rollout_config().max_episode_tokens == 65536
    assert AsyncTrainingConfig(rollout_max_episode_tokens=None).get_rollout_config().max_episode_tokens is None


@pytest.mark.parametrize("bad", [-1, 0, float("nan"), True, 8000.5], ids=["negative", "zero", "nan", "bool", "float"])
def test_async_config_rejects_a_thinking_budget_that_is_not_a_positive_int(bad):
    """A negative budget is below every turn cap, NaN passes every ordered comparison, a bool or a float
    would reach the engine as a nonsense ``thinking_token_budget``, and 0 would be sent as 1 while counting
    as no cap at all (the forced-close gates then never arm); the one range check refuses them all.
    ``null`` is the spelling for no run-wide cap."""
    with pytest.raises(ValueError, match="rollout_max_thinking_tokens must be an int in"):
        AsyncTrainingConfig(rollout_max_thinking_tokens=bad)
    cfg = AsyncTrainingConfig()
    cfg.rollout_max_thinking_tokens = bad
    with pytest.raises(ValueError, match="rollout_max_thinking_tokens must be an int in"):
        cfg.__post_override__({"rollout_max_thinking_tokens"})


def test_async_config_range_guards_survive_a_cli_override():
    """``__post_init__`` never re-runs under ``--key=value``; the guards live in ``_validate_ranges``
    so the override path re-runs them whole."""
    cfg = AsyncTrainingConfig()
    cfg.rollout_temperature = 0.0
    with pytest.raises(ValueError, match="rollout_temperature"):
        cfg.__post_override__({"rollout_temperature"})


def test_async_config_template_variables_are_the_yamls_own_kwargs():
    """Every request and every trainer-side render carries exactly the YAML's ``rollout_chat_template_kwargs``:
    no run-wide variable is injected beside them (the per-episode level and budget travel per request), and
    the rollout config gets its own copy, never the YAML's mapping."""
    run_kwargs = {"preserve_thinking": True}
    config = AsyncTrainingConfig(rollout_chat_template_kwargs=run_kwargs)
    mirrored = config.get_rollout_config().chat_template_kwargs
    assert mirrored == run_kwargs and mirrored is not config.rollout_chat_template_kwargs
    assert AsyncTrainingConfig().get_rollout_config().chat_template_kwargs == {}
    with pytest.raises(ValueError, match="must not carry"):
        AsyncTrainingConfig(rollout_chat_template_kwargs={"reasoning_budget": 8192})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
