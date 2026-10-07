#!/usr/bin/env python
"""Env-GRPO checks its eval round only when the run evaluates.

``_validate_eval_round`` runs in the trainer's ``__init__`` after the model has loaded. Refusing a
``per_device_eval_batch_size`` that no eval round uses (``eval_strategy: no``), which TRL's own check
(gated on evaluation) accepts, would kill a recipe at startup over a dead key. The gate applies
whenever the loop evaluates: an eval strategy, or ``eval_on_start``. Every shipped env-GRPO
recipe is run through it with the values its own parser produces.

Run: pytest tests/cpu/grpo/test_env_grpo_eval_round_gate.py
"""

import types

import pytest
from trl import GRPOConfig

from src.configs.async_training_config import AsyncTrainingConfig
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from tests.cpu.config.test_examples_parse import (
    _EXAMPLES,
    EXAMPLES_ROOT,
    _bf16_capable,  # noqa: F401 — autouse fixture: validate the configs as a training host does
    parsed_field,
    parser_for,
    script_for,
)

_ENV_GRPO_SCRIPT = "environmental_grpo.py"
_ENV_GRPO_EXAMPLES = [config for config in _EXAMPLES if script_for(config) == _ENV_GRPO_SCRIPT]


def _host(args, async_config, num_generations_eval: int):
    """The attributes the gate reads, as the trainer holds them after TRL's ``__init__``."""
    return types.SimpleNamespace(args=args, async_config=async_config, num_generations_eval=num_generations_eval)


def _validate(host) -> None:
    DistributedAsyncEnvironmentalGRPOTrainer._validate_eval_round(host)


def _grpo_config(tmp_path, **overrides) -> GRPOConfig:
    return GRPOConfig(output_dir=str(tmp_path / "out"), bf16=False, use_cpu=True, **overrides)


def test_env_grpo_recipes_exist():
    assert _ENV_GRPO_EXAMPLES, "no env-GRPO recipe found; the sweep below would check nothing"


@pytest.mark.parametrize("config", _ENV_GRPO_EXAMPLES, ids=lambda c: c.relative_to(EXAMPLES_ROOT).as_posix())
def test_every_env_grpo_recipe_passes_the_eval_round_gate(config):
    parsed = parser_for(_ENV_GRPO_SCRIPT).parse_yaml_file(str(config))
    grpo_config = next(obj for obj in parsed if isinstance(obj, GRPOConfig))
    async_config = next(obj for obj in parsed if isinstance(obj, AsyncTrainingConfig))
    num_generations_eval = parsed_field(parsed, "num_generations_eval") or parsed_field(parsed, "num_generations")
    _validate(_host(grpo_config, async_config, num_generations_eval))


def test_a_run_that_never_evaluates_skips_the_gate(tmp_path):
    args = _grpo_config(tmp_path, eval_strategy="no", per_device_eval_batch_size=1, num_generations=8)
    _validate(_host(args, AsyncTrainingConfig(), num_generations_eval=8))


def test_eval_on_start_still_checks_the_round(tmp_path):
    args = _grpo_config(tmp_path, eval_strategy="no", eval_on_start=True, per_device_eval_batch_size=1)
    with pytest.raises(ValueError, match=r"per_device_eval_batch_size \(1\) must be divisible"):
        _validate(_host(args, AsyncTrainingConfig(), num_generations_eval=8))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
