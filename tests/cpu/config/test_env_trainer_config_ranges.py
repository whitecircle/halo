#!/usr/bin/env python
"""Parse-time range checks on the training configs, ``AsyncTrainingConfig`` and ``EnvironmentConfig`` in depth.

* Every numeric knob a training script parses refuses a bool and a non-finite value, and an int knob a
  fraction: a bare range check reads ``true`` as 1 (``rollout_min_p: true`` samples every rollout
  greedily) and passes a NaN, or an infinity past an open bound, and a fractional count reaches its
  consumer as no count at all.
* ``skip_update_masked_frac`` is a fraction in (0, 1]: 0 trips the breaker on every step, above 1 never.
  Checked with the other knobs at parse time, not at trainer construction after the servers are up.
* ``max_train_row_tokens`` must exceed ``rollout_max_tokens``: a training row is prompt + completion
  and the completion alone may run to the per-turn budget, so a cap at or below it leaves out every
  turn that used its budget — a length bias against long turns, not the memory bound the knob is.
* ``rollout_max_episode_tokens`` bounds the episode's sampled tokens and must hold one whole turn,
  so the first turn can use the per-turn cap the run states.
* ``rollout_max_answer_tokens`` bounds what a turn generates past its reasoning cap: an int in
  ``[1, rollout_max_tokens)``, since at or above the turn cap it shrinks no turn and below 1 a turn the
  engine force-closed has no room to answer.

    python tests/cpu/config/test_env_trainer_config_ranges.py
"""

import ast
import importlib
import math
import types
from dataclasses import fields
from pathlib import Path
from typing import Union, get_args, get_origin, get_type_hints

import pytest
from transformers import TrainingArguments

from src.configs.async_training_config import AsyncTrainingConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Keeps a TrainingArguments subclass constructible on a GPU-less runner, so a raise can only come from
# the knob under test.
_TRAINING_ARGS_BASE = {"output_dir": "/tmp/numeric-knob-sweep", "use_cpu": True, "bf16": False}
# Fields a config refuses to construct without.
_REQUIRED = {"DistillScriptArguments": {"teacher_model": "org/teacher"}}
# A knob that is refused at its value only beside another one set.
_COMPANIONS = {
    "isr_geo_band_min": {"isr_geo_band_max": 1.5},
    "isr_geo_band_max": {"isr_geo_band_min": 0.5},
    "reasoning_price_cap": {"reasoning_price": {"low": 0.1, "medium": 0.1, "high": 0.1}},
}
_BAD_NUMBERS = {"true": True, "false": False, "nan": math.nan, "inf": math.inf}


def _parse_targets() -> list[type]:
    """Every toolkit dataclass a training entry script hands to ``H4ArgumentParser``.

    Read from the scripts' ASTs rather than a hand list, so a new entry script — or a new config tuple on
    an existing one — is swept the moment it lands.
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
                for element in argument.elts if isinstance(argument, ast.Tuple | ast.List) else [argument]:
                    name = ast.unparse(element)
                    if name in imported:
                        targets[name] = getattr(importlib.import_module(imported[name]), name)
    return list(targets.values())


def _numeric_knobs(config: type) -> list[tuple[type, str, bool]]:
    """``(declaring class, name, int-only)`` for every int- or float-annotated field of ``config`` that a
    toolkit class declares, optional or not; a bool is neither here. Upstream's fields are upstream's."""
    hints = get_type_hints(config)
    knobs = []
    for f in fields(config):
        annotation = hints[f.name]
        members = get_args(annotation) if get_origin(annotation) in (Union, types.UnionType) else (annotation,)
        if not {int, float} & set(members):
            continue
        owner = next(cls for cls in config.__mro__ if f.name in vars(cls).get("__annotations__", {}))
        if owner.__module__.startswith("src."):
            knobs.append((owner, f.name, float not in members))
    return knobs


def _sweep() -> list:
    base_of = {
        config: {
            **(_TRAINING_ARGS_BASE if issubclass(config, TrainingArguments) else {}),
            **_REQUIRED.get(config.__name__, {}),
        }
        for config in _parse_targets()
    }
    params, seen = [], set()
    for config, base in base_of.items():
        for owner, name, int_only in _numeric_knobs(config):
            if (owner, name) in seen:
                continue
            seen.add((owner, name))
            bad_values = {**_BAD_NUMBERS, **({"fraction": 2.5} if int_only else {})}
            for bad_id, bad in bad_values.items():
                kwargs = {**base, **_COMPANIONS.get(name, {}), name: bad}
                params.append(pytest.param(config, name, kwargs, id=f"{config.__name__}.{name}-{bad_id}"))
    for bad_id, bad in _BAD_NUMBERS.items():
        params.append(
            pytest.param(
                AsyncTrainingConfig,
                "early_stop_entropy_band",
                {"early_stop_entropy_band": [0.1, bad]},
                id=f"AsyncTrainingConfig.early_stop_entropy_band-{bad_id}",
            )
        )
    return params


_NUMERIC_KNOBS = _sweep()


def test_the_sweep_reaches_the_knobs_of_every_parsed_config():
    """Anti-vacuity: the script scan and the annotation scan find the knobs the sweep is for, inherited and
    mixed-in ones included."""
    knobs = {param.values[1] for param in _NUMERIC_KNOBS}
    assert {
        "rollout_min_p",
        "max_retries",
        "scale_rewards_std_floor",
        "isr_geo_band_max",
        "max_turns",
        "rollout_max_tokens",
        "sdpg_beta_base",
        "reference_kl_coef",
        "dataset_num_proc",
        "tensor_parallel_size",
        "profiler_active",
    } <= knobs


@pytest.mark.parametrize(("config", "knob", "kwargs"), _NUMERIC_KNOBS)
def test_a_numeric_knob_refuses_a_bool_or_a_non_finite_value(config, knob, kwargs):
    with pytest.raises(ValueError, match=knob):
        config(**kwargs)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, math.nan])
def test_skip_update_masked_frac_outside_the_unit_interval_is_refused_at_parse_time(bad):
    with pytest.raises(ValueError, match="skip_update_masked_frac must be"):
        AsyncTrainingConfig(skip_update_masked_frac=bad)


@pytest.mark.parametrize("ok", [0.3, 1.0, None])
def test_skip_update_masked_frac_in_range_passes(ok):
    assert AsyncTrainingConfig(skip_update_masked_frac=ok).skip_update_masked_frac == ok


@pytest.mark.parametrize("cap", [1, 999, 1000])
def test_a_row_cap_at_or_below_the_per_turn_budget_is_refused(cap):
    with pytest.raises(ValueError, match="must be an int above rollout_max_tokens"):
        AsyncTrainingConfig(rollout_max_tokens=1000, max_train_row_tokens=cap)


def test_a_row_cap_above_the_per_turn_budget_passes():
    assert AsyncTrainingConfig(rollout_max_tokens=1000, max_train_row_tokens=1001).max_train_row_tokens == 1001
    assert AsyncTrainingConfig(rollout_max_tokens=1000).max_train_row_tokens is None


def test_an_episode_budget_holds_one_whole_turn_and_is_unbounded_when_null():
    with pytest.raises(ValueError, match="rollout_max_episode_tokens must be an int >= rollout_max_tokens"):
        AsyncTrainingConfig(rollout_max_tokens=1000, rollout_max_episode_tokens=999)
    assert (
        AsyncTrainingConfig(rollout_max_tokens=1000, rollout_max_episode_tokens=1000).rollout_max_episode_tokens
        == 1000
    )
    assert AsyncTrainingConfig(rollout_max_tokens=1000).rollout_max_episode_tokens is None


@pytest.mark.parametrize("bad", [0, -1, 1000, 1001, True, 512.0, "512"])
def test_an_answer_bound_outside_one_to_the_turn_cap_or_not_a_count_is_refused(bad):
    with pytest.raises(
        ValueError, match=r"rollout_max_answer_tokens must be an int in \[1, rollout_max_tokens=1000\)"
    ):
        AsyncTrainingConfig(rollout_max_tokens=1000, rollout_max_answer_tokens=bad)


@pytest.mark.parametrize("ok", [1, 999, None])
def test_an_answer_bound_inside_the_turn_cap_passes_and_reaches_the_rollout_config(ok):
    cfg = AsyncTrainingConfig(rollout_max_tokens=1000, rollout_max_answer_tokens=ok)
    assert cfg.get_rollout_config(in_process_group=False).max_answer_tokens == ok


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
