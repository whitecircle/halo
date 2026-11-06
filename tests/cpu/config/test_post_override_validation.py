#!/usr/bin/env python
"""CLI overrides clear the same guards a YAML value does: ``HaloArgumentParser.parse_dict`` builds each dataclass
once from the YAML and the overrides, so every ``__post_init__`` guard sees the CLI value."""

import ast
import inspect
import textwrap
from typing import get_type_hints

import pytest

from halo.config_validation import ValidatedConfig
from halo.configs.classification import ClassificationConfig
from halo.configs.distillation import (
    SDPGArguments,
    SelfDistillScriptArguments,
    TeacherDistillationConfig,
    TeacherDistillScriptArguments,
)
from halo.configs.embedding import EmbeddingConfig
from halo.configs.grpo_environmental import AsyncRolloutConfig, EnvironmentConfig
from halo.configs.grpo_offline import OfflineGRPOConfig
from halo.configs.grpo_online import OnlineGRPOScriptArguments
from halo.configs.parallelism_arguments import ParallelismArguments
from halo.configs.parser import HaloArgumentParser, _literal_choices
from halo.configs.preference import SMPOConfig
from halo.entrypoints.script import init_training_script
from halo.env import DEFAULT_NCCL_TIMEOUT_MINUTES
from halo.subclass_registry import iter_subclasses
from halo.trainers.distillation.divergences import DIVERGENCES

# bf16/use_cpu keep TrainingArguments constructible on a GPU-less runner.
_CPU_TAIL = "bf16: false\nuse_cpu: true\n"


def _parse(dataclass_type, tmp_path, overrides: list[str], yaml_body: str = ""):
    """Parse a minimal YAML for ``dataclass_type``, then apply ``overrides`` the way a launch does."""
    body = yaml_body
    if hasattr(dataclass_type, "output_dir"):
        body += f"output_dir: {tmp_path / 'out'}\n" + _CPU_TAIL
    path = tmp_path / "config.yaml"
    path.write_text(body or "{}\n")
    parser = HaloArgumentParser((dataclass_type,))
    (parsed,) = parser.parse_dict(parser.load_values([str(path), *overrides]))
    return parsed


# Every guard a CLI override must meet: (config class, flag, out-of-range value).
_RANGE_VIOLATIONS = [
    (ClassificationConfig, "focal_gamma", "-1"),
    (ClassificationConfig, "label_smoothing", "1.5"),
    (ClassificationConfig, "multi_label_threshold", "0"),
    (SMPOConfig, "target_margin", "-0.1"),
    (SMPOConfig, "chosen_sft_ratio", "1.5"),
    (SMPOConfig, "min_log_prob", "0.5"),
    (SMPOConfig, "lower_clip_percentile", "0.9"),
    (EmbeddingConfig, "loss_scale", "0"),
    (OfflineGRPOConfig, "best_completion_emphasis", "0.5"),
    (OfflineGRPOConfig, "best_completion_emphasis", "atuo"),
    (AsyncRolloutConfig, "sync_weights_every_n_steps", "0"),
    (AsyncRolloutConfig, "max_retries", "-1"),
    (AsyncRolloutConfig, "scale_rewards_std_floor", "-0.05"),
    (AsyncRolloutConfig, "prefetch_depth", "0"),
    (OnlineGRPOScriptArguments, "scale_rewards_std_floor", "nan"),
    (OnlineGRPOScriptArguments, "sdpg_temperature", "0"),
    (OnlineGRPOScriptArguments, "sdpg_hint_template", "{solution}"),
    (SelfDistillScriptArguments, "sdpg_beta_warmup_steps", "-1"),
    (SelfDistillScriptArguments, "sdpg_hint_template", "{answr}"),
    (SelfDistillScriptArguments, "reference_kl_coef", "-0.1"),
    (SelfDistillScriptArguments, "reference_kl_coef", "nan"),
    (SelfDistillScriptArguments, "confidence_power", "0"),
    (SelfDistillScriptArguments, "train_on_completions_only", "false"),
]

# Presence guards. Each needs a valid YAML baseline so the raise can only come from the override.
_EMPTIED_REQUIRED_FIELDS = [
    (TeacherDistillScriptArguments, "teacher_model", "teacher_model: some-org/teacher\n", ""),
]


@pytest.mark.parametrize(
    ("config_cls", "flag", "value"),
    _RANGE_VIOLATIONS,
    ids=[f"{c.__name__}.{f}={v}" for c, f, v in _RANGE_VIOLATIONS],
)
def test_cli_override_violating_a_range_raises(config_cls, flag, value, tmp_path):
    with pytest.raises(ValueError, match=flag):
        _parse(config_cls, tmp_path, [f"--{flag}={value}"])


@pytest.mark.parametrize(
    ("config_cls", "flag", "value"),
    [
        (ClassificationConfig, "multi_label_threshold", "0.7"),
        (SMPOConfig, "chosen_sft_ratio", "0.5"),
        (OfflineGRPOConfig, "best_completion_emphasis", "1.5"),
        (OfflineGRPOConfig, "best_completion_emphasis", "auto"),
        (AsyncRolloutConfig, "max_retries", "0"),
        (AsyncRolloutConfig, "sync_weights_every_n_steps", "4"),
    ],
    ids=lambda v: str(v),
)
def test_in_range_cli_override_still_parses(config_cls, flag, value, tmp_path):
    parsed = _parse(config_cls, tmp_path, [f"--{flag}={value}"])
    assert str(getattr(parsed, flag)) == value


@pytest.mark.parametrize(
    ("config_cls", "flag", "yaml_body", "value"),
    _EMPTIED_REQUIRED_FIELDS,
    ids=[f"{c.__name__}.{f}={v!r}" for c, f, _, v in _EMPTIED_REQUIRED_FIELDS],
)
def test_cli_override_emptying_a_required_field_raises(config_cls, flag, yaml_body, value, tmp_path):
    """Distillation would reach the loader with a blank teacher id."""
    assert _parse(config_cls, tmp_path, [], yaml_body), "yaml baseline must parse, or the test is vacuous"
    with pytest.raises(ValueError, match=flag):
        _parse(config_cls, tmp_path, [f"--{flag}={value}"], yaml_body)


def test_smpo_cross_field_margin_guard_survives_override(tmp_path):
    """A cross-field guard: overriding either side must re-check."""
    with pytest.raises(ValueError, match="initial_margin"):
        _parse(SMPOConfig, tmp_path, ["--initial_margin=0.9"], "use_margin_schedule: true\n")


def test_classification_class_weight_exclusivity_survives_override(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        _parse(ClassificationConfig, tmp_path, ["--derive_class_weights=true"], "class_weights: [1.0, 2.0]\n")


def test_every_validated_config_implements_the_hook_and_keeps_one_post_init_chain():
    """A subclass implementing no ``_validate_fields`` runs no guard, a ``__post_init__`` override skipping
    ``super()`` leaves the config unguarded, and one also calling ``_validate_fields`` validates twice."""
    configs = iter_subclasses(ValidatedConfig)
    assert configs, "no config routes through ValidatedConfig — the seam is dead"
    for cls in configs:
        assert cls._validate_fields is not ValidatedConfig._validate_fields, (
            f"{cls.__name__} inherits ValidatedConfig but never implements _validate_fields"
        )
        override = cls.__dict__.get("__post_init__")
        if override is not None:
            source = inspect.getsource(override)
            assert "super().__post_init__()" in source, (
                f"{cls.__name__}.__post_init__ skips super().__post_init__(), so its fields go unguarded"
            )
            assert "_validate_fields" not in source, (
                f"{cls.__name__}.__post_init__ calls _validate_fields beside the base hook, validating twice"
            )


def test_every_validate_fields_override_opens_the_cooperative_chain():
    """An override that skips ``super()`` keeps only its own guards, so a class mixing two guarded bases silently
    drops one's checks."""
    for cls in iter_subclasses(ValidatedConfig):
        override = cls.__dict__.get("_validate_fields")
        if override is None:
            continue
        assert "super()._validate_fields()" in inspect.getsource(override), (
            f"{cls.__name__}._validate_fields does not open with super()._validate_fields(), so any "
            f"guarded base earlier in its MRO is skipped"
        )


def test_dead_knobs_are_rejected_not_ignored():
    """``num_labels`` (derived from the dataset) and ``partial_reward`` (no tier the grader pays) would be settable
    no-ops."""
    for parser, key in (
        (HaloArgumentParser((ClassificationConfig,)), "num_labels"),
        (HaloArgumentParser((EnvironmentConfig,)), "partial_reward"),
    ):
        with pytest.raises(ValueError, match=key):
            parser.parse_dict({key: 2})


def test_overwrite_output_dir_reaches_both_output_dir_guards():
    """The parser routes a YAML key only to the dataclass declaring it: CheckpointArguments alone, which
    ``init_training_script`` hands to the setup guard and the resume one."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(init_training_script)))
    handed = {
        call.func.id
        for call in ast.walk(tree)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        for keyword in call.keywords
        if keyword.arg == "overwrite_output_dir" and ast.unparse(keyword.value) == "args.overwrite_output_dir"
    }
    assert handed == {"configure_training_run", "prepare_distributed_resume"}, handed


def test_offline_grpo_max_completion_length_defaults_to_none():
    """dr_grpo's normalizer guard fires only on None; a 1024 default would silently normalize by a number nobody
    chose."""
    assert OfflineGRPOConfig.max_completion_length is None


def test_async_episode_timeout_default_clears_the_watchdog_warning():
    """The shipped default must not trip AsyncRolloutConfig's own >= 0.8 * NCCL-watchdog warning."""
    assert AsyncRolloutConfig.episode_timeout < 0.8 * DEFAULT_NCCL_TIMEOUT_MINUTES * 60


# Each distillation arm admits its own slice of the one divergence registry.
_DIVERGENCE_FIELDS = (
    (SDPGArguments, "sdpg_loss"),
    (SelfDistillScriptArguments, "reference_kl_loss"),
    (TeacherDistillationConfig, "distill_loss"),
)


def _divergence_choices(owner, field_name) -> set:
    declared = _literal_choices(get_type_hints(owner)[field_name])
    assert declared is not None, (
        f"{owner.__name__}.{field_name} is not Literal-annotated, so the parser cannot gate it"
    )
    return set(declared)


@pytest.mark.parametrize(("owner", "field_name"), _DIVERGENCE_FIELDS, ids=[name for _, name in _DIVERGENCE_FIELDS])
def test_every_admitted_divergence_resolves_in_the_registry(owner, field_name):
    """A name the parser admits but the registry lacks raises only at trainer construction."""
    unresolved = _divergence_choices(owner, field_name) - set(DIVERGENCES)
    assert not unresolved, f"{owner.__name__}.{field_name} admits {sorted(unresolved)}, which DIVERGENCES lacks"


def test_the_reference_anchor_admits_only_the_divergences_that_take_no_beta():
    """The anchor resolves its loss with no β, so a β-taking one would raise at trainer construction."""
    beta_free = {name for name, fn in DIVERGENCES.items() if "jsd_beta" not in inspect.signature(fn).parameters}
    assert _divergence_choices(SelfDistillScriptArguments, "reference_kl_loss") == beta_free & _divergence_choices(
        SDPGArguments, "sdpg_loss"
    )


def test_every_registered_divergence_is_admitted_by_some_arm():
    """A registry entry no Literal admits is dead code that a YAML can never reach."""
    admitted = set().union(*(_divergence_choices(owner, field_name) for owner, field_name in _DIVERGENCE_FIELDS))
    assert set(DIVERGENCES) == admitted


def test_unknown_pipeline_schedule_is_rejected_at_parse_time(tmp_path):
    """Declared ``str``, pipeline_schedule would skip the parser's Literal gate entirely."""
    path = tmp_path / "config.yaml"
    path.write_text("pipeline_schedule: 1f1c\n")
    with pytest.raises(ValueError, match="pipeline_schedule"):
        HaloArgumentParser((ParallelismArguments,)).parse_yaml_file(str(path))


def test_unknown_reasoning_effort_is_rejected_at_parse_time(tmp_path):
    """The RLVR path has only the parser, and most chat templates silently ignore an unknown level."""
    with pytest.raises(ValueError, match="reasoning_effort"):
        _parse(OnlineGRPOScriptArguments, tmp_path, ["--reasoning_effort=hgih"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
