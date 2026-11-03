#!/usr/bin/env python
"""Parse-time range checks on the training configs. A bare range check reads ``true`` as 1 and passes a NaN, and a
fractional count reaches its consumer as no count, so every numeric knob refuses all three.
``max_train_row_tokens`` must exceed ``rollout_max_tokens``, or a cap leaves out every turn that used its budget."""

import ast
import importlib
import math
import types
from dataclasses import fields
from typing import Union, get_args, get_origin, get_type_hints

import pytest
from transformers import TrainingArguments

from halo.configs.grpo_environmental import AsyncRolloutConfig
from tests.common.repo import REPO_ROOT

# Keeps a TrainingArguments subclass constructible on a GPU-less runner, so a raise can only come from
# the knob under test.
_TRAINING_ARGS_BASE = {"output_dir": "/tmp/numeric-knob-sweep", "use_cpu": True, "bf16": False}
_REQUIRED = {"TeacherDistillScriptArguments": {"teacher_model": "org/teacher"}}
# A knob that is refused at its value only beside another one set.
_COMPANIONS = {
    "isr_geo_band_min": {"isr_geo_band_max": 1.5},
    "isr_geo_band_max": {"isr_geo_band_min": 0.5},
    "reasoning_price_cap": {"reasoning_price": {"low": 0.1, "medium": 0.1, "high": 0.1}},
}
_BAD_NUMBERS = {"true": True, "false": False, "nan": math.nan, "inf": math.inf}


def _parse_targets() -> list[type]:
    """Every toolkit dataclass a training entry script hands to ``parse_script_args``, read from the scripts' ASTs
    so a new script is swept when it lands."""
    targets: dict[str, type] = {}
    for script in sorted((REPO_ROOT / "scripts" / "training").rglob("*.py")):
        tree = ast.parse(script.read_text())
        imported = {
            alias.asname or alias.name: node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("halo.")
            for alias in node.names
        }
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "parse_script_args"):
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
        if owner.__module__.startswith("halo."):
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
                AsyncRolloutConfig,
                "early_stop_entropy_band",
                {"early_stop_entropy_band": [0.1, bad]},
                id=f"AsyncRolloutConfig.early_stop_entropy_band-{bad_id}",
            )
        )
    return params


_NUMERIC_KNOBS = _sweep()


def test_the_sweep_reaches_the_knobs_of_every_parsed_config():
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


def test_a_rescore_budget_is_read_only_beside_the_engine_reference():
    """Set with the engine re-score off it would change nothing, so it is refused at parse time."""
    with pytest.raises(ValueError, match="isr_engine_rescore_per_server set with isr_engine_reference off"):
        AsyncRolloutConfig(isr_engine_rescore_per_server=64)
    assert AsyncRolloutConfig(isr_engine_reference=True, isr_engine_rescore_per_server=64)


@pytest.mark.parametrize("bad", [-1, 65536])
def test_a_weight_sync_group_port_outside_the_port_range_is_refused(bad):
    with pytest.raises(ValueError, match="rollout_group_port"):
        AsyncRolloutConfig(rollout_group_port=bad)


@pytest.mark.parametrize("ok", [0, 51216, 65535])
def test_a_weight_sync_group_port_in_range_passes(ok):
    assert AsyncRolloutConfig(rollout_group_port=ok).rollout_group_port == ok


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, math.nan])
def test_skip_update_masked_frac_outside_the_unit_interval_is_refused_at_parse_time(bad):
    with pytest.raises(ValueError, match="skip_update_masked_frac must be"):
        AsyncRolloutConfig(skip_update_masked_frac=bad)


@pytest.mark.parametrize("ok", [0.3, 1.0, None])
def test_skip_update_masked_frac_in_range_passes(ok):
    assert AsyncRolloutConfig(skip_update_masked_frac=ok).skip_update_masked_frac == ok


@pytest.mark.parametrize("cap", [1, 999, 1000])
def test_a_row_cap_at_or_below_the_per_turn_budget_is_refused(cap):
    with pytest.raises(
        ValueError, match=r"max_train_row_tokens must be an integer > 1000 \(above rollout_max_tokens\)"
    ):
        AsyncRolloutConfig(rollout_max_tokens=1000, max_train_row_tokens=cap)


def test_a_row_cap_above_the_per_turn_budget_passes():
    assert AsyncRolloutConfig(rollout_max_tokens=1000, max_train_row_tokens=1001).max_train_row_tokens == 1001
    assert AsyncRolloutConfig(rollout_max_tokens=1000).max_train_row_tokens is None


def test_an_episode_budget_holds_one_whole_turn_and_is_unbounded_when_null():
    with pytest.raises(
        ValueError, match=r"rollout_max_episode_tokens must be an integer >= 1000 \(at least rollout_max_tokens\)"
    ):
        AsyncRolloutConfig(rollout_max_tokens=1000, rollout_max_episode_tokens=999)
    assert (
        AsyncRolloutConfig(rollout_max_tokens=1000, rollout_max_episode_tokens=1000).rollout_max_episode_tokens == 1000
    )
    assert AsyncRolloutConfig(rollout_max_tokens=1000).rollout_max_episode_tokens is None


@pytest.mark.parametrize("bad", [0, -1, 1000, 1001, True, 512.0, "512"])
def test_an_answer_bound_outside_one_to_the_turn_cap_or_not_a_count_is_refused(bad):
    with pytest.raises(
        ValueError, match=r"rollout_max_answer_tokens must be an integer in \[1, 1000\) \(below rollout_max_tokens\)"
    ):
        AsyncRolloutConfig(rollout_max_tokens=1000, rollout_max_answer_tokens=bad)


@pytest.mark.parametrize("ok", [1, 999, None])
def test_an_answer_bound_inside_the_turn_cap_passes_and_reaches_the_rollout_config(ok):
    cfg = AsyncRolloutConfig(rollout_max_tokens=1000, rollout_max_answer_tokens=ok)
    assert cfg.get_rollout_config(in_process_group=False).max_answer_tokens == ok


_TWO_SERVERS = [{"url": "http://s0:8000"}, {"url": "http://s1:8000"}]


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "2"])
def test_a_prefetch_depth_below_one_or_not_a_count_is_refused(bad):
    with pytest.raises(ValueError, match=r"prefetch_depth must be an integer >= 1"):
        AsyncRolloutConfig(rollout_server_configs=_TWO_SERVERS, prefetch_depth=bad)


_UNSET_ON_ONE_SERVER = r"but enable_prefetch is unset, .* and this run has one\. Set enable_prefetch: true"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rollout_server_configs": _TWO_SERVERS, "enable_prefetch": False}, "but enable_prefetch is false"),
        ({"enable_prefetch": False}, "but enable_prefetch is false"),
        ({}, _UNSET_ON_ONE_SERVER),
        ({"rollout_server_configs": _TWO_SERVERS[:1]}, _UNSET_ON_ONE_SERVER),
    ],
    ids=["false-two-servers", "false-one-url", "unset-one-url", "unset-one-entry-list"],
)
def test_a_prefetch_depth_above_one_names_why_prefetch_is_off(kwargs, match):
    with pytest.raises(ValueError, match=match):
        AsyncRolloutConfig(prefetch_depth=2, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rollout_server_configs": _TWO_SERVERS},
        {"enable_prefetch": True},
        {"rollout_server_configs": _TWO_SERVERS[:1], "enable_prefetch": True},
    ],
    ids=["unset-two-servers", "true-one-url", "true-one-entry-list"],
)
def test_a_prefetch_depth_passes_on_every_run_that_prefetches(kwargs):
    assert AsyncRolloutConfig(prefetch_depth=3, **kwargs).prefetch_depth == 3


def test_depth_one_passes_anywhere():
    assert AsyncRolloutConfig(enable_prefetch=False).prefetch_depth == 1
    assert AsyncRolloutConfig().prefetch_depth == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
