#!/usr/bin/env python
"""Contract of the shared script-argument fields: the values every training script assumes it has.

  * ``project_name`` is written straight into ``WANDB_PROJECT`` / ``CLEARML_PROJECT`` by
    ``configure_training_run``. ``os.environ`` accepts strings only, so a YAML ``project_name:
    null`` would surface as a bare ``TypeError`` from the tracking setup, phases into the run and
    nowhere near the key that caused it. The gate is ``CommonScriptArguments._validate_fields``,
    which every script-arg class runs from ``__post_init__``, where a ``--project_name=`` override
    is held to the same rule.
  * The SFT re-declarations of the ``GenerationEvalArguments`` / collator flags are plain ``bool``:
    every consumer declares ``bool``, nothing produces ``None``, and an Optional there would admit a
    third state no branch handles.

Run: pytest tests/cpu/config/test_script_args_contract.py
"""

import os
from types import SimpleNamespace
from typing import get_type_hints

import pytest

from halo.configs.classification import ClassificationScriptArguments
from halo.configs.distillation import SelfDistillScriptArguments, TeacherDistillScriptArguments
from halo.configs.embedding import EmbeddingScriptArguments
from halo.configs.grpo_environmental import EnvironmentalGRPOScriptArguments
from halo.configs.grpo_offline import OfflineGRPOScriptArguments
from halo.configs.grpo_online import OnlineGRPOScriptArguments
from halo.configs.parser import HaloArgumentParser
from halo.configs.preference import DPOScriptArguments, KTOScriptArguments, SMPOScriptArguments
from halo.configs.reward import RewardScriptArguments
from halo.configs.script_arguments import GenerationEvalArguments
from halo.configs.sft import SFTScriptArguments
from halo.entrypoints.run_setup import _configure_tracking_env_vars

# Every script-arg class and the tracking project its runs land in. Pinned by NAME, not merely "not
# the placeholder": a class that declares no ``PROJECT_NAME`` of its own inherits a parent's project
# and silently files its runs under another method's experiments.
_PROJECT_NAMES = {
    ClassificationScriptArguments: "classification",
    DPOScriptArguments: "dpo-tuning",
    TeacherDistillScriptArguments: "llm_distillation",
    EmbeddingScriptArguments: "embedding",
    EnvironmentalGRPOScriptArguments: "environmental-grpo",
    KTOScriptArguments: "kto-tuning",
    OfflineGRPOScriptArguments: "offline-grpo-tuning",
    OnlineGRPOScriptArguments: "rlvr-online-grpo",
    RewardScriptArguments: "reward-modeling",
    SFTScriptArguments: "sft-tuning",
    SMPOScriptArguments: "smpo-tuning",
    SelfDistillScriptArguments: "self-distillation",
}

# Fields a class refuses to default; supplied so the class can be constructed at all.
_REQUIRED_KWARGS = {TeacherDistillScriptArguments: {"teacher_model": "teacher/model"}}

_SCRIPT_ARG_CLASSES = (SFTScriptArguments, SMPOScriptArguments, SelfDistillScriptArguments)


# project_name


@pytest.mark.parametrize(
    ("cls", "expected"),
    sorted(_PROJECT_NAMES.items(), key=lambda pair: pair[0].__name__),
    ids=lambda value: getattr(value, "__name__", value),
)
def test_each_script_declares_its_own_project_name(cls, expected):
    assert cls(**_REQUIRED_KWARGS.get(cls, {})).project_name == expected


@pytest.mark.parametrize("cls", _SCRIPT_ARG_CLASSES)
@pytest.mark.parametrize("empty", [None, "", "   "])
def test_null_or_blank_project_name_is_refused_at_config_time(cls, empty):
    with pytest.raises(ValueError, match="project_name"):
        cls(project_name=empty)


def test_explicit_project_name_survives():
    assert SFTScriptArguments(project_name="my-project").project_name == "my-project"


def test_yaml_null_project_name_is_refused_by_the_parser(tmp_path):
    """The path a user actually takes to None: a key written with no value."""
    path = tmp_path / "sft.yaml"
    path.write_text("project_name:\n")
    with pytest.raises(ValueError, match="project_name"):
        HaloArgumentParser((SFTScriptArguments,)).parse_yaml_file(str(path))


@pytest.mark.parametrize("cls", _SCRIPT_ARG_CLASSES)
@pytest.mark.parametrize("value", ["", "None", "null"])
def test_cli_override_cannot_null_the_project_name(cls, value, tmp_path):
    """``project_name`` is annotated ``str``, so the parser's null-spelling conversion (Optional fields
    only) leaves ``None`` as four literal characters: the run would then report itself to a tracking
    project literally named "None", or to ``WANDB_PROJECT=""`` — both silently, and both landing in
    a different place from every other run of the same experiment.
    """
    path = tmp_path / "config.yaml"
    path.write_text("{}\n")
    parser = HaloArgumentParser((cls,))
    with pytest.raises(ValueError, match="project_name"):
        parser.parse_dict(parser.load_values([str(path), f"--project_name={value}"]))


def test_a_real_cli_project_name_still_parses(tmp_path):
    """Anti-over-rejection: the override the guard exists to allow must reach the args unchanged."""
    path = tmp_path / "config.yaml"
    path.write_text("{}\n")
    parser = HaloArgumentParser((SFTScriptArguments,))
    (parsed,) = parser.parse_dict(parser.load_values([str(path), "--project_name=nonsense-42"]))
    assert parsed.project_name == "nonsense-42"


def test_tracking_setup_writes_project_name_verbatim(monkeypatch):
    """Why the gate exists: the value goes into the environment unconverted, so it must already be
    a string by the time the run reaches here."""
    monkeypatch.setenv("WANDB_PROJECT", "stale")
    monkeypatch.setenv("CLEARML_PROJECT", "stale")
    monkeypatch.setenv("WANDB_RUN_ID", "fixed-run-id")
    training_config = SimpleNamespace(run_name=None, output_dir="/tmp/test_script_args_contract")

    _configure_tracking_env_vars(SFTScriptArguments(project_name="halo-sft"), training_config, "sft")

    assert os.environ["WANDB_PROJECT"] == "halo-sft"
    assert os.environ["CLEARML_PROJECT"] == "halo-sft"


# SFT's re-declared flags


@pytest.mark.parametrize("field_name", ["generate_eval_examples", "train_on_last_assistant_only"])
def test_sft_flag_redeclarations_are_plain_bools(field_name):
    """A re-declaration exists to change the DEFAULT, not the type: the collator factory and the
    generation callback both declare ``bool``, so an Optional here is a state nobody handles."""
    assert get_type_hints(SFTScriptArguments)[field_name] is bool


def test_sft_matches_the_mixin_it_redeclares():
    assert (
        get_type_hints(SFTScriptArguments)["generate_eval_examples"]
        is get_type_hints(GenerationEvalArguments)["generate_eval_examples"]
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
