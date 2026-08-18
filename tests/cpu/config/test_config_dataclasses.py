#!/usr/bin/env python
"""
Tests for configuration dataclasses: SmoothMarginPOConfig, EnvironmentConfig,
and AsyncTrainingConfig __post_init__ validation and defaults.

Run: python tests/cpu/config/test_config_dataclasses.py
"""

import contextlib
import os

import pytest

OUTPUT_DIR = "/tmp/test_output"

# SmoothMarginPOConfig extends transformers.TrainingArguments, whose __post_init__ (called at the
# END of the SMPO __post_init__, after all SMPO-specific field validation) raises
# "Your setup doesn't support bf16/gpu" when bf16 is on and no CUDA device is visible. The CPU test
# container has no GPU, so VALID configs (those expected to construct successfully) pass bf16=False
# to reach that tail cleanly. The SMPO validation-branch tests raise BEFORE this tail, so they leave
# bf16 at its default and never reach the GPU check.
_CPU_OK = {"output_dir": OUTPUT_DIR, "bf16": False}


# SmoothMarginPOConfig tests


def test_smpo_defaults():
    """Default SMPO config should be valid."""
    from src.configs.smpo_config import SmoothMarginPOConfig

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
    assert cfg.label_pad_token_id == -100
    assert cfg.disable_dropout is True
    assert cfg.padding_free is False


def test_smpo_padding_free():
    """padding_free config field should be settable."""
    from src.configs.smpo_config import SmoothMarginPOConfig

    cfg = SmoothMarginPOConfig(padding_free=True, **_CPU_OK)
    assert cfg.padding_free is True


def test_smpo_negative_target_margin():
    """target_margin < 0 should raise ValueError."""
    from src.configs.smpo_config import SmoothMarginPOConfig

    raised = False
    try:
        SmoothMarginPOConfig(output_dir=OUTPUT_DIR, target_margin=-0.1)
    except ValueError as e:
        raised = True
        assert "target_margin" in str(e)
    assert raised, "Should raise ValueError for negative target_margin"


def test_smpo_margin_schedule_initial_ge_target():
    """initial_margin >= target_margin with use_margin_schedule should raise ValueError."""
    from src.configs.smpo_config import SmoothMarginPOConfig

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
    from src.configs.smpo_config import SmoothMarginPOConfig

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
    from src.configs.smpo_config import SmoothMarginPOConfig

    cfg = SmoothMarginPOConfig(
        use_margin_schedule=False,
        initial_margin=0.5,
        target_margin=0.3,
        **_CPU_OK,
    )
    assert cfg.initial_margin == 0.5


def test_smpo_lower_clip_percentile_zero():
    """lower_clip_percentile = 0 should raise ValueError (must be > 0)."""
    from src.configs.smpo_config import SmoothMarginPOConfig

    raised = False
    try:
        SmoothMarginPOConfig(output_dir=OUTPUT_DIR, lower_clip_percentile=0.0)
    except ValueError as e:
        raised = True
        assert "lower_clip_percentile" in str(e)
    assert raised, "Should raise ValueError for lower_clip_percentile=0"


def test_smpo_lower_clip_percentile_too_high():
    """lower_clip_percentile > 0.5 should raise ValueError."""
    from src.configs.smpo_config import SmoothMarginPOConfig

    with pytest.raises(ValueError, match="lower_clip_percentile"):
        SmoothMarginPOConfig(output_dir=OUTPUT_DIR, lower_clip_percentile=0.6)


def test_smpo_lower_clip_percentile_at_half():
    """lower_clip_percentile = 0.5 should be valid (boundary)."""
    from src.configs.smpo_config import SmoothMarginPOConfig

    cfg = SmoothMarginPOConfig(lower_clip_percentile=0.5, **_CPU_OK)
    assert cfg.lower_clip_percentile == 0.5


def test_smpo_upper_clip_percentile_below_half():
    """upper_clip_percentile < 0.5 should raise ValueError."""
    from src.configs.smpo_config import SmoothMarginPOConfig

    raised = False
    try:
        SmoothMarginPOConfig(output_dir=OUTPUT_DIR, upper_clip_percentile=0.4)
    except ValueError as e:
        raised = True
        assert "upper_clip_percentile" in str(e)
    assert raised, "Should raise ValueError for upper_clip_percentile=0.4"


def test_smpo_upper_clip_percentile_at_one():
    """upper_clip_percentile = 1.0 should raise ValueError (must be < 1)."""
    from src.configs.smpo_config import SmoothMarginPOConfig

    with pytest.raises(ValueError, match="upper_clip_percentile"):
        SmoothMarginPOConfig(output_dir=OUTPUT_DIR, upper_clip_percentile=1.0)


def test_smpo_min_log_prob_positive():
    """min_log_prob >= 0 should raise ValueError."""
    from src.configs.smpo_config import SmoothMarginPOConfig

    raised = False
    try:
        SmoothMarginPOConfig(output_dir=OUTPUT_DIR, min_log_prob=0.5)
    except ValueError as e:
        raised = True
        assert "min_log_prob" in str(e)
    assert raised, "Should raise ValueError for positive min_log_prob"


def test_smpo_min_log_prob_zero():
    """min_log_prob = 0 should raise ValueError."""
    from src.configs.smpo_config import SmoothMarginPOConfig

    with pytest.raises(ValueError, match="min_log_prob"):
        SmoothMarginPOConfig(output_dir=OUTPUT_DIR, min_log_prob=0.0)


def test_smpo_chosen_sft_ratio_out_of_range():
    """chosen_sft_ratio outside [0, 1] should raise ValueError."""
    from src.configs.smpo_config import SmoothMarginPOConfig

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
    from src.configs.smpo_config import SmoothMarginPOConfig

    cfg0 = SmoothMarginPOConfig(chosen_sft_ratio=0.0, **_CPU_OK)
    assert cfg0.chosen_sft_ratio == 0.0
    cfg1 = SmoothMarginPOConfig(chosen_sft_ratio=1.0, **_CPU_OK)
    assert cfg1.chosen_sft_ratio == 1.0


def test_smpo_upper_clip_percentile_valid_boundary():
    """upper_clip_percentile = 0.5 (lower boundary) and 0.99 (interior) are valid."""
    from src.configs.smpo_config import SmoothMarginPOConfig

    cfg_lo = SmoothMarginPOConfig(upper_clip_percentile=0.5, **_CPU_OK)
    assert cfg_lo.upper_clip_percentile == 0.5
    cfg_hi = SmoothMarginPOConfig(upper_clip_percentile=0.99, **_CPU_OK)
    assert cfg_hi.upper_clip_percentile == 0.99


def test_smpo_clip_percentiles_none_disables_validation():
    """Setting both clip percentiles to None disables the range checks (no raise)."""
    from src.configs.smpo_config import SmoothMarginPOConfig

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
    from src.configs.smpo_config import SmoothMarginPOConfig

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
    from src.configs.smpo_config import SmoothMarginPOConfig
    from src.training.parser import H4ArgumentParser

    cfg = SmoothMarginPOConfig(loss_type="bogus", **_CPU_OK)
    assert cfg.loss_type == "bogus"

    with pytest.raises(ValueError, match="loss_type"):
        H4ArgumentParser((SmoothMarginPOConfig,)).parse_dict({"loss_type": "bogus", **_CPU_OK})


# EnvironmentConfig tests


def test_env_config_defaults():
    """EnvironmentConfig should have sensible defaults."""
    from src.configs.environment_config import EnvironmentConfig

    cfg = EnvironmentConfig()
    assert cfg.environment_type == "react_math"
    assert cfg.success_reward == 1.0
    assert cfg.failure_reward == 0.0
    # No partial_reward knob: answer grading is all-or-nothing (success_reward / failure_reward).
    assert not hasattr(cfg, "partial_reward")
    # None defers to the environment class's own default (CodeContests 15, SWE 20, ExamQA 8).
    assert cfg.max_turns is None
    assert "max_turns" not in cfg.to_env_config()
    assert cfg.environment_kwargs == {}


def test_env_config_to_env_config():
    """to_env_config() should merge core settings with environment_kwargs."""
    from src.configs.environment_config import EnvironmentConfig

    cfg = EnvironmentConfig(
        success_reward=2.0,
        max_turns=20,
        environment_kwargs={"search_backend": "duckduckgo", "open_book": True},
    )
    result = cfg.to_env_config()
    assert result["success_reward"] == 2.0
    assert result["max_turns"] == 20
    assert result["search_backend"] == "duckduckgo"
    assert result["open_book"] is True


def test_env_config_to_env_config_empty_kwargs():
    """to_env_config() with empty environment_kwargs should return core settings only."""
    from src.configs.environment_config import EnvironmentConfig

    cfg = EnvironmentConfig(success_reward=1.5, max_turns=5)
    result = cfg.to_env_config()
    # to_env_config() emits the reachable core reward/turn settings — failure_reward is wired
    # alongside success_reward — and empty kwargs add nothing.
    assert result == {"success_reward": 1.5, "failure_reward": 0.0, "max_turns": 5}


def test_env_config_kwargs_override_core_key():
    """environment_kwargs is applied via dict.update LAST, so it overrides a core key.

    This is a real footgun: putting ``max_turns`` inside environment_kwargs shadows the
    top-level ``max_turns``. Pinning the precedence (kwargs win) documents it.
    """
    from src.configs.environment_config import EnvironmentConfig

    cfg = EnvironmentConfig(max_turns=10, environment_kwargs={"max_turns": 99})
    result = cfg.to_env_config()
    assert result["max_turns"] == 99


def test_env_config_rejects_non_positive_max_turns():
    """``max_turns: 0`` makes the rollout loop a no-op, so every episode returns reward 0 with no
    error — a silently all-zero batch. It must fail at parse time, not after Ray and vLLM are up."""
    from src.configs.environment_config import EnvironmentConfig

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
    from src.configs.environment_config import EnvironmentConfig

    cfg = EnvironmentConfig()
    cfg.max_turns = 0
    try:
        cfg.__post_override__({"max_turns"})
    except ValueError as e:
        assert "max_turns" in str(e)
    else:
        raise AssertionError("--max_turns=0 must be rejected on the override path too")


def test_env_config_custom_env_type_passthrough():
    """environment_type is stored verbatim (registry resolves it later, no validation here)."""
    from src.configs.environment_config import EnvironmentConfig

    cfg = EnvironmentConfig(environment_type="code_contests", environment_kwargs={"timeout_per_test": 10})
    assert cfg.environment_type == "code_contests"
    # environment_type is NOT part of to_env_config() — only reward/turn settings + kwargs are.
    out = cfg.to_env_config()
    assert "environment_type" not in out
    assert out["timeout_per_test"] == 10


# AsyncTrainingConfig tests


def _import_async_training_config():
    """Import AsyncTrainingConfig, skipping if ray/aiohttp unavailable."""
    try:
        from src.configs.async_training_config import AsyncTrainingConfig

        return AsyncTrainingConfig
    except ImportError as e:
        raise RuntimeError(f"Cannot import AsyncTrainingConfig (missing dependency: {e})") from e


def test_async_config_defaults():
    """AsyncTrainingConfig should have sensible defaults."""
    AsyncTrainingConfig = _import_async_training_config()
    cfg = AsyncTrainingConfig()
    assert cfg.num_rollout_workers == 64
    assert cfg.max_concurrent_rollouts is None
    assert cfg.rollout_server_url == "http://localhost:8000"
    assert cfg.rollout_temperature == 0.7
    assert cfg.rollout_top_p == 0.95
    assert cfg.rollout_max_tokens == 32768
    assert cfg.enable_prefetch is True
    assert cfg.num_prefetch_batches == 1
    assert cfg.max_retries == 3
    assert cfg.retry_base_wait == 1.0


def test_async_config_get_server_urls_single():
    """get_server_urls() should return single URL when no multi-server config."""
    AsyncTrainingConfig = _import_async_training_config()
    cfg = AsyncTrainingConfig(rollout_server_url="http://gpu1:8000")
    urls = cfg.get_server_urls()
    assert urls == ["http://gpu1:8000"]


def test_async_config_get_server_urls_multi():
    """get_server_urls() should return URLs from rollout_server_configs when set."""
    AsyncTrainingConfig = _import_async_training_config()
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
    AsyncTrainingConfig = _import_async_training_config()
    cfg = AsyncTrainingConfig(rollout_server_url="http://gpu1:8000", rollout_server_configs=[])
    assert cfg.get_server_urls() == ["http://gpu1:8000"]


def test_async_config_get_rollout_config_threads_fields():
    """get_rollout_config() forwards the rollout/retry knobs into RolloutConfig."""
    AsyncTrainingConfig = _import_async_training_config()
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
    AsyncTrainingConfig = _import_async_training_config()
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
    AsyncTrainingConfig = _import_async_training_config()
    with _nccl_watchdog_minutes("60"):  # watchdog = 3600s
        cfg = AsyncTrainingConfig(episode_timeout=2700.0)
        rc = cfg.get_rollout_config()
        assert rc.episode_timeout == 2700.0


def test_async_config_episode_timeout_equal_watchdog_does_not_raise():
    """The stock default (episode_timeout == watchdog == 1800s) is a race, not a certainty: it warns but
    must NOT raise, or every default env-GRPO run would break at construction."""
    AsyncTrainingConfig = _import_async_training_config()
    with _nccl_watchdog_minutes(None):
        cfg = AsyncTrainingConfig(episode_timeout=1800.0)  # == default 1800s watchdog
        rc = cfg.get_rollout_config()  # no raise
        assert rc.episode_timeout == 1800.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
