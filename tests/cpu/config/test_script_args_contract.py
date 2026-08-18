#!/usr/bin/env python
"""Contract of the shared script-argument fields: the values every training script assumes it has.

  * ``project_name`` is written straight into ``WANDB_PROJECT`` / ``CLEARML_PROJECT`` by
    ``setup_training_environment``. ``os.environ`` accepts strings only, so a YAML ``project_name:
    null`` would surface as a bare ``TypeError`` from the tracking setup, phases into the run and
    nowhere near the key that caused it. The gate is ``CommonScriptArguments._validate_ranges``,
    which every script-arg class runs from ``__post_init__`` AND — via ``RangeValidatedConfig`` —
    from ``__post_override__``, so a ``--project_name=`` override is held to the same rule.
  * The SFT re-declarations of the ``GenerationEvalArguments`` / collator flags are plain ``bool``:
    every consumer declares ``bool``, nothing produces ``None``, and an Optional there would admit a
    third state no branch handles.

Run: pytest tests/cpu/config/test_script_args_contract.py
"""

import os
from types import SimpleNamespace
from typing import get_type_hints

import pytest

from src.args.classification_args import CLFScriptArguments
from src.args.distill_args import DistillScriptArguments
from src.args.dpo_args import DPOScriptArguments
from src.args.embedding_args import EmbeddingScriptArguments
from src.args.environmental_grpo_args import EnvironmentalGRPOScriptArguments
from src.args.kto_args import KTOScriptArguments
from src.args.mixins import GenerationEvalArguments
from src.args.offline_grpo_args import OfflineGRPOScriptArguments
from src.args.reward_args import RMScriptArguments
from src.args.rlvr_online_grpo_args import RLVROnlineGRPOScriptArguments
from src.args.self_distill_args import SelfDistillationArguments
from src.args.sft_args import SFTScriptArguments
from src.args.smpo_args import SMPOScriptArguments
from src.training.environment import _setup_tracking_env_vars
from src.training.parser import H4ArgumentParser

# Every script-arg class and the tracking project its runs land in. Pinned by NAME, not merely "not
# the placeholder": a class that declares no ``__post_init__`` of its own inherits a parent's project
# and silently files its runs under another method's experiments.
_PROJECT_NAMES = {
    CLFScriptArguments: "classification",
    DPOScriptArguments: "dpo-tuning",
    DistillScriptArguments: "llm_distillation",
    EmbeddingScriptArguments: "embedding",
    EnvironmentalGRPOScriptArguments: "environmental-grpo",
    KTOScriptArguments: "kto-tuning",
    OfflineGRPOScriptArguments: "offline-grpo-tuning",
    RLVROnlineGRPOScriptArguments: "rlvr-online-grpo",
    RMScriptArguments: "reward-modeling",
    SFTScriptArguments: "sft-tuning",
    SMPOScriptArguments: "smpo-tuning",
    SelfDistillationArguments: "self-distillation",
}

# Fields a class refuses to default; supplied so the class can be constructed at all.
_REQUIRED_KWARGS = {DistillScriptArguments: {"teacher_model": "teacher/model"}}

_SCRIPT_ARG_CLASSES = (SFTScriptArguments, SMPOScriptArguments, SelfDistillationArguments)


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
        H4ArgumentParser((SFTScriptArguments,)).parse_yaml_file(str(path))


@pytest.mark.parametrize("cls", _SCRIPT_ARG_CLASSES)
@pytest.mark.parametrize("value", ["", "None", "null"])
def test_cli_override_cannot_null_the_project_name(cls, value, tmp_path):
    """``--project_name=`` lands by ``setattr``, so ``__post_init__`` never re-runs.

    ``project_name`` is annotated ``str``, so the parser's null-spelling conversion (Optional fields
    only) leaves ``None`` as four literal characters: the run then reported itself to a tracking
    project literally named "None", or to ``WANDB_PROJECT=""`` — both silently, and both landing in
    a different place from every other run of the same experiment.
    """
    path = tmp_path / "config.yaml"
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="project_name"):
        H4ArgumentParser((cls,)).parse_yaml_and_args(str(path), [f"--project_name={value}"])


def test_a_real_cli_project_name_still_parses(tmp_path):
    """Anti-over-rejection: the override the guard exists to allow must reach the args unchanged."""
    path = tmp_path / "config.yaml"
    path.write_text("{}\n")
    (parsed,) = H4ArgumentParser((SFTScriptArguments,)).parse_yaml_and_args(str(path), ["--project_name=nonsense-42"])
    assert parsed.project_name == "nonsense-42"


def test_tracking_setup_writes_project_name_verbatim(monkeypatch):
    """Why the gate exists: the value goes into the environment unconverted, so it must already be
    a string by the time the run reaches here."""
    monkeypatch.setenv("WANDB_PROJECT", "stale")
    monkeypatch.setenv("CLEARML_PROJECT", "stale")
    monkeypatch.setenv("WANDB_RUN_ID", "fixed-run-id")
    training_config = SimpleNamespace(run_name=None, output_dir="/tmp/test_script_args_contract")

    _setup_tracking_env_vars(SFTScriptArguments(project_name="halo-sft"), training_config, "sft")

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
    raise SystemExit(pytest.main([__file__, "-q"]))
