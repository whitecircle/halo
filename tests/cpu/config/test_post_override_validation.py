#!/usr/bin/env python
"""CLI overrides must clear the same guards a YAML value does.

``H4ArgumentParser.parse_yaml_and_args`` applies ``--key=value`` with ``setattr``, so
``__post_init__`` never re-runs and every range guard it holds is bypassed. ``RangeValidatedConfig``
closes that hole: guards live in ``_validate_ranges``, which runs from ``__post_init__`` AND from
``__post_override__``. These tests fail if a guard is dropped, if a guarded config stops routing
through the base, or if a Literal field's allowed set drifts from the runtime table it mirrors.

Run: pytest tests/cpu/config/test_post_override_validation.py
"""

import ast
import importlib
import inspect
from dataclasses import fields
from pathlib import Path
from typing import get_type_hints

import pytest
from transformers import TrainingArguments

from src.args.distill_args import DistillScriptArguments
from src.args.distributed_args import DistributedArguments
from src.args.rlvr_online_grpo_args import RLVROnlineGRPOScriptArguments
from src.args.validation import RangeValidatedConfig
from src.configs.async_training_config import AsyncTrainingConfig
from src.configs.classification_config import ClassificationConfig
from src.configs.embedding_config import EmbeddingConfig
from src.configs.environment_config import EnvironmentConfig
from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.configs.smpo_config import SmoothMarginPOConfig
from src.distributed.parallelism_config import PP_SCHEDULES
from src.environments.base import VALID_REASONING_EFFORTS
from src.training.parser import H4ArgumentParser, _literal_choices
from src.training.script_runner import init_training_script

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# bf16/use_cpu keep TrainingArguments constructible on a GPU-less runner.
_CPU_TAIL = "bf16: false\nuse_cpu: true\n"


def _parse(dataclass_type, tmp_path, overrides: list[str], yaml_body: str = ""):
    """Parse a minimal YAML for ``dataclass_type``, then apply ``overrides`` the way a launch does."""
    body = yaml_body
    if hasattr(dataclass_type, "output_dir"):
        body += f"output_dir: {tmp_path / 'out'}\n" + _CPU_TAIL
    path = tmp_path / "config.yaml"
    path.write_text(body or "{}\n")
    parser = H4ArgumentParser((dataclass_type,))
    (parsed,) = parser.parse_yaml_and_args(str(path), overrides)
    return parsed


def _subclasses(cls) -> set[type]:
    found = set()
    for sub in cls.__subclasses__():
        found.add(sub)
        found |= _subclasses(sub)
    return found


def _parse_target_dataclasses() -> dict[str, type]:
    """Every toolkit dataclass an entry script hands to ``H4ArgumentParser``, keyed by class name.

    Read from the scripts' ASTs rather than a hand list so a new entry script — or a new config
    tuple on an existing one — is covered the moment it lands.
    """
    targets: dict[str, type] = {}
    for script in sorted((PROJECT_ROOT / "scripts" / "training").rglob("*.py")):
        tree = ast.parse(script.read_text())
        imported = {
            alias.asname or alias.name: node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src.")
            for alias in node.names
        }
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "H4ArgumentParser"):
                continue
            for argument in node.args:
                elements = argument.elts if isinstance(argument, ast.Tuple | ast.List) else [argument]
                for element in elements:
                    name = ast.unparse(element)
                    if name in imported:
                        targets[name] = getattr(importlib.import_module(imported[name]), name)
    return targets


# Every guard that must survive the setattr override path: (config class, flag, out-of-range value).
_RANGE_VIOLATIONS = [
    (ClassificationConfig, "focal_gamma", "-1"),
    (ClassificationConfig, "label_smoothing", "1.5"),
    (ClassificationConfig, "multi_label_threshold", "0"),
    (SmoothMarginPOConfig, "target_margin", "-0.1"),
    (SmoothMarginPOConfig, "chosen_sft_ratio", "1.5"),
    (SmoothMarginPOConfig, "min_log_prob", "0.5"),
    (SmoothMarginPOConfig, "lower_clip_percentile", "0.9"),
    (EmbeddingConfig, "loss_scale", "0"),
    (OfflineGRPOConfig, "best_completion_emphasis", "0.5"),
    (OfflineGRPOConfig, "best_completion_emphasis", "atuo"),
    (AsyncTrainingConfig, "sync_weights_every_n_steps", "0"),
    (AsyncTrainingConfig, "max_retries", "-1"),
    (AsyncTrainingConfig, "advantage_hard_group_threshold", "nan"),
    (AsyncTrainingConfig, "scale_rewards_std_floor", "-0.05"),
    (RLVROnlineGRPOScriptArguments, "advantage_hard_group_threshold", "nan"),
]

# Presence guards rather than ranges, and the same bypass. Each needs a VALID yaml baseline so the
# raise can only come from the override — parsing an empty config would trip the guard on its own.
_EMPTIED_REQUIRED_FIELDS = [
    (DistillScriptArguments, "teacher_model", "teacher_model: some-org/teacher\n", ""),
]


@pytest.mark.parametrize(
    ("config_cls", "flag", "value"),
    _RANGE_VIOLATIONS,
    ids=[f"{c.__name__}.{f}={v}" for c, f, v in _RANGE_VIOLATIONS],
)
def test_cli_override_violating_a_range_raises(config_cls, flag, value, tmp_path):
    """The failure mode this pins: setattr overrides skip __post_init__, so --focal_gamma=-1 trains."""
    with pytest.raises(ValueError, match=flag):
        _parse(config_cls, tmp_path, [f"--{flag}={value}"])


@pytest.mark.parametrize(
    ("config_cls", "flag", "value"),
    [
        (ClassificationConfig, "focal_gamma", "3.0"),
        (SmoothMarginPOConfig, "chosen_sft_ratio", "0.5"),
        (OfflineGRPOConfig, "best_completion_emphasis", "1.5"),
        (OfflineGRPOConfig, "best_completion_emphasis", "auto"),
        (AsyncTrainingConfig, "max_retries", "0"),
        (AsyncTrainingConfig, "sync_weights_every_n_steps", "4"),
    ],
    ids=lambda v: str(v),
)
def test_in_range_cli_override_still_parses(config_cls, flag, value, tmp_path):
    """The guards must not reject legitimate values — 0 retries and 'auto' emphasis are valid."""
    parsed = _parse(config_cls, tmp_path, [f"--{flag}={value}"])
    assert str(getattr(parsed, flag)) == value


@pytest.mark.parametrize(
    ("config_cls", "flag", "yaml_body", "value"),
    _EMPTIED_REQUIRED_FIELDS,
    ids=[f"{c.__name__}.{f}={v!r}" for c, f, _, v in _EMPTIED_REQUIRED_FIELDS],
)
def test_cli_override_emptying_a_required_field_raises(config_cls, flag, yaml_body, value, tmp_path):
    """A valid YAML plus an emptying override must not parse clean: distillation would reach the
    loader with a blank teacher id, and an environment run an empty required field."""
    assert _parse(config_cls, tmp_path, [], yaml_body), "yaml baseline must parse, or the test is vacuous"
    with pytest.raises(ValueError, match=flag):
        _parse(config_cls, tmp_path, [f"--{flag}={value}"], yaml_body)


def test_smpo_cross_field_margin_guard_survives_override(tmp_path):
    """initial_margin < target_margin is a CROSS-field guard: overriding either side must re-check."""
    with pytest.raises(ValueError, match="initial_margin"):
        _parse(SmoothMarginPOConfig, tmp_path, ["--initial_margin=0.9"], "use_margin_schedule: true\n")


def test_classification_class_weight_exclusivity_survives_override(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        _parse(ClassificationConfig, tmp_path, ["--derive_class_weights=true"], "class_weights: [1.0, 2.0]\n")


def test_every_range_validated_subclass_implements_and_calls_the_hook():
    """Derived from the class hierarchy, not a hand list: inheriting the base without implementing
    ``_validate_ranges`` would make every CLI override raise NotImplementedError, and defining
    ``__post_init__`` without calling it would leave the YAML path unguarded."""
    subclasses = _subclasses(RangeValidatedConfig)
    assert subclasses, "no config routes through RangeValidatedConfig — the seam is dead"
    for cls in subclasses:
        assert cls._validate_ranges is not RangeValidatedConfig._validate_ranges, (
            f"{cls.__name__} inherits RangeValidatedConfig but never implements _validate_ranges"
        )
        if "__post_init__" in cls.__dict__:
            assert "_validate_ranges" in inspect.getsource(cls.__post_init__), (
                f"{cls.__name__}.__post_init__ does not call _validate_ranges, so YAML values go unguarded"
            )


def test_every_validate_ranges_override_opens_the_cooperative_chain():
    """``_validate_ranges`` is cooperative (the contract is stated on ``RangeValidatedConfig``): an
    override that skips ``super()`` keeps only its own guards, so a class mixing two guarded bases
    silently drops one base's checks. ``OfflineGRPOConfig`` already mixes ``ChunkedLogprobsArguments``
    with ``RangeValidatedConfig``, which puts it one guarded base away from that."""
    for cls in _subclasses(RangeValidatedConfig):
        override = cls.__dict__.get("_validate_ranges")
        if override is None:
            continue
        assert "super()._validate_ranges()" in inspect.getsource(override), (
            f"{cls.__name__}._validate_ranges does not open with super()._validate_ranges(), so any "
            f"guarded base earlier in its MRO is skipped"
        )


def test_every_guarded_parse_target_routes_through_the_override_hook():
    """The converse of the test above.

    A config whose ``__post_init__`` raises but which does NOT inherit ``RangeValidatedConfig`` is
    guarded on the YAML path and unguarded on the CLI path: ``parse_yaml_and_args`` applies
    ``--key=value`` with ``setattr``, so the guard never re-runs and the run starts with the very
    value it exists to refuse (``DistillScriptArguments``' required teacher is one such field).
    """
    targets = _parse_target_dataclasses()
    assert targets, "no H4ArgumentParser target resolved — the AST scan is broken, not the configs"
    unguarded = [
        name
        for name, cls in targets.items()
        if "__post_init__" in cls.__dict__
        and any(isinstance(n, ast.Raise) for n in ast.walk(ast.parse(inspect.getsource(cls.__post_init__).strip())))
        and not issubclass(cls, RangeValidatedConfig)
    ]
    assert not unguarded, (
        f"{unguarded} raise from __post_init__ but do not inherit RangeValidatedConfig, so a CLI "
        f"override bypasses the guard. Move the checks into _validate_ranges() and inherit the base."
    )


def test_dead_knobs_are_rejected_not_ignored():
    """``num_labels`` (derived from the dataset) and ``partial_reward`` (no reward tier the grader
    ever pays) were settable no-ops; the parser must now refuse them."""
    for parser, key in (
        (H4ArgumentParser((ClassificationConfig,)), "num_labels"),
        (H4ArgumentParser((EnvironmentConfig,)), "partial_reward"),
    ):
        with pytest.raises(ValueError, match=key):
            parser.parse_dict({key: 2})


@pytest.mark.parametrize("field_name", ["save_max_shard_size", "overwrite_output_dir"])
def test_distributed_only_knob_is_forwarded_to_the_training_config(field_name):
    """Both knobs are declared ONLY on DistributedArguments, but every consumer reads them off the
    HF/TRL training config (``self.args`` / ``training_config``). The parser routes a YAML key only
    to the dataclass declaring it, so without ``init_training_script``'s forward the reads always
    take their ``getattr(..., default)`` fallback and the knobs are unsettable in practice."""
    declared = {f.name for f in fields(DistributedArguments)}
    assert field_name in declared
    assert field_name not in {f.name for f in fields(TrainingArguments)}, (
        f"{field_name} is now native to TrainingArguments — the forward is redundant, drop one"
    )
    forwarder = Path(inspect.getfile(init_training_script)).read_text()
    assert field_name in forwarder, f"init_training_script no longer forwards {field_name} onto training_config"


def test_offline_grpo_max_completion_length_defaults_to_none():
    """dr_grpo's normalizer guard fires only on None; a 1024 default silently normalized by a number
    nobody chose — exactly what the guard's error text warns against."""
    assert OfflineGRPOConfig.max_completion_length is None


def test_async_episode_timeout_default_clears_the_watchdog_warning():
    """The shipped default must not trip AsyncTrainingConfig's own >= 0.8 * NCCL-watchdog warning."""
    from src.env import DEFAULT_NCCL_TIMEOUT_MINUTES

    assert AsyncTrainingConfig.episode_timeout < 0.8 * DEFAULT_NCCL_TIMEOUT_MINUTES * 60


@pytest.mark.parametrize(
    ("owner", "field_name", "expected"),
    [
        (DistributedArguments, "pipeline_schedule", PP_SCHEDULES),
        (RLVROnlineGRPOScriptArguments, "reasoning_effort", (*VALID_REASONING_EFFORTS, "random", None)),
    ],
    ids=["pipeline_schedule", "reasoning_effort"],
)
def test_literal_annotation_matches_its_runtime_table(owner, field_name, expected):
    """These fields restate a tuple that lives elsewhere (PP_SCHEDULES / VALID_REASONING_EFFORTS)
    because importing it would drag torch or the environments package into the arg dataclasses.
    Pin the equality — via the parser's own choice extractor — so the restatement cannot drift."""
    declared = _literal_choices(get_type_hints(owner)[field_name])
    assert declared is not None, (
        f"{owner.__name__}.{field_name} is not Literal-annotated, so the parser cannot gate it"
    )
    assert set(declared) == set(expected)


def test_unknown_pipeline_schedule_is_rejected_at_parse_time(tmp_path):
    """Declared ``str``, pipeline_schedule skipped the parser's Literal gate entirely."""
    path = tmp_path / "config.yaml"
    path.write_text("pipeline_schedule: 1f1c\n")
    with pytest.raises(ValueError, match="pipeline_schedule"):
        H4ArgumentParser((DistributedArguments,)).parse_yaml_file(str(path))


def test_unknown_reasoning_effort_is_rejected_at_parse_time(tmp_path):
    """The env-GRPO path validates this steer in BaseEnvironment.__init__; the RLVR path did not,
    and an unknown level is silently ignored by most chat templates (no steer at all)."""
    with pytest.raises(ValueError, match="reasoning_effort"):
        _parse(RLVROnlineGRPOScriptArguments, tmp_path, ["--reasoning_effort=hgih"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
