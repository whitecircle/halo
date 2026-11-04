#!/usr/bin/env python
"""Env-GRPO checks its eval round only when the run evaluates. Refusing a ``per_device_eval_batch_size`` no round uses,
which TRL's own check accepts, would kill a recipe at startup over a dead key."""

import pytest
from trl import GRPOConfig

from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.trainers.grpo.environmental.environmental_rules import reject_straddling_eval_groups
from tests.cpu.config.test_examples_parse import (
    ENV_GRPO_EXAMPLES,
    ENV_GRPO_SCRIPT,
    EXAMPLES_ROOT,
    _bf16_capable,  # noqa: F401 — autouse fixture: validate the configs as a training host does
    parsed_field,
    parser_for,
)


def _grpo_config(tmp_path, **overrides) -> GRPOConfig:
    return GRPOConfig(output_dir=str(tmp_path / "out"), bf16=False, use_cpu=True, **overrides)


def test_env_grpo_recipes_exist():
    assert ENV_GRPO_EXAMPLES, "no env-GRPO recipe found; the sweep below would check nothing"


@pytest.mark.parametrize("config", ENV_GRPO_EXAMPLES, ids=lambda c: c.relative_to(EXAMPLES_ROOT).as_posix())
def test_every_env_grpo_recipe_passes_the_eval_round_gate(config):
    parsed = parser_for(ENV_GRPO_SCRIPT).parse_yaml_file(str(config))
    grpo_config = next(obj for obj in parsed if isinstance(obj, GRPOConfig))
    async_config = next(obj for obj in parsed if isinstance(obj, AsyncRolloutConfig))
    num_generations_eval = parsed_field(parsed, "num_generations_eval") or parsed_field(parsed, "num_generations")
    reject_straddling_eval_groups(grpo_config, async_config, num_generations_eval)


def test_a_run_that_never_evaluates_skips_the_gate(tmp_path):
    args = _grpo_config(tmp_path, eval_strategy="no", per_device_eval_batch_size=1, num_generations=8)
    reject_straddling_eval_groups(args, AsyncRolloutConfig(), num_generations_eval=8)


def test_eval_on_start_still_checks_the_round(tmp_path):
    args = _grpo_config(tmp_path, eval_strategy="no", eval_on_start=True, per_device_eval_batch_size=1)
    with pytest.raises(ValueError, match=r"per_device_eval_batch_size \(1\) must be divisible"):
        reject_straddling_eval_groups(args, AsyncRolloutConfig(), num_generations_eval=8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
