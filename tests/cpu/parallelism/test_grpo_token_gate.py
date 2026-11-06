#!/usr/bin/env python
"""Online and environmental GRPO declare their forward budget to the EP dispatch gate at load: their configs have no
``max_length``, so without it the cross-node 8,192 tokens/rank ceiling raises at the first long row, mid-run."""

from __future__ import annotations

import types

import pytest

from halo.configs.parallelism_arguments import ParallelismArguments, parallelism_config_from_args
from halo.model_loading.parallel.load_rules import reject_unsupported_model
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.online import DistributedOnlineGRPOTrainer
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.parallelism import simulated_world

CONTEXT_WINDOW = 131072
GIN_CEILING = 8192


class _MoEConfig:
    num_experts = 128
    hidden_size = 4096
    num_experts_per_tok = 8
    max_position_embeddings = CONTEXT_WINDOW


def _training_config(*, completion=2048, train_rows=1, eval_rows=1, eval_strategy="no"):
    return types.SimpleNamespace(
        per_device_train_batch_size=train_rows,
        per_device_eval_batch_size=eval_rows,
        eval_strategy=eval_strategy,
        max_completion_length=completion,
    )


def _build(trainer_cls, training_config, *sources, cross_node=True):
    """Two 8-GPU nodes with one EP group across both, or the same job with node-local EP groups."""
    args = ParallelismArguments(
        expert_parallel_size=16 if cross_node else 8, ep_scope="global" if cross_node else "node"
    )
    with simulated_world(world_size=16, gpus_per_node=8):
        return parallelism_config_from_args(
            args, trainer_cls=trainer_cls, training_config=training_config, row_sources=sources
        )


def _gate(config) -> None:
    reject_unsupported_model(config, _MoEConfig(), sinks_policy=SinksPolicy.NEUTRALIZED, wraps_experts=False)


def _prompts(max_prompt_length):
    return types.SimpleNamespace(max_prompt_length=max_prompt_length)


def _rollout(max_train_row_tokens, rollout_max_tokens=1024):
    return types.SimpleNamespace(max_train_row_tokens=max_train_row_tokens, rollout_max_tokens=rollout_max_tokens)


def test_online_grpo_declares_prompt_plus_completion_and_the_gate_refuses_an_over_ceiling_row():
    config = _build(DistributedOnlineGRPOTrainer, _training_config(completion=8192), _prompts(4096))
    assert (config.ep_rows_per_device, config.ep_declared_max_length) == (1, 4096 + 8192)
    with pytest.raises(ValueError, match="12288 tokens/rank"):
        _gate(config)


def test_online_grpo_under_the_ceiling_passes():
    config = _build(DistributedOnlineGRPOTrainer, _training_config(completion=2048), _prompts(4096))
    assert config.ep_declared_max_length == 4096 + 2048
    _gate(config)


def test_an_unbounded_prompt_is_judged_against_the_context_window_across_nodes():
    config = _build(DistributedOnlineGRPOTrainer, _training_config(), _prompts(None))
    assert (config.ep_rows_per_device, config.ep_declared_max_length) == (1, 0)
    with pytest.raises(ValueError, match="context window"):
        _gate(config)


def test_node_local_ep_declares_nothing_for_an_open_ended_row():
    """The context window is far past the node-local wire ceiling too; the dispatcher stays the backstop there."""
    config = _build(DistributedOnlineGRPOTrainer, _training_config(), _prompts(None), cross_node=False)
    assert (config.ep_rows_per_device, config.ep_declared_max_length) == (0, 0)
    _gate(config)


def test_node_local_ep_keeps_a_declared_bound():
    config = _build(DistributedOnlineGRPOTrainer, _training_config(), _prompts(2048), cross_node=False)
    assert (config.ep_rows_per_device, config.ep_declared_max_length) == (1, 2048 + 2048)


@pytest.mark.parametrize(("cap", "refused"), [(GIN_CEILING * 2, True), (GIN_CEILING, False)])
def test_environmental_grpo_is_bounded_by_max_train_row_tokens(cap, refused):
    config = _build(DistributedAsyncEnvironmentalGRPOTrainer, _training_config(completion=256), _rollout(cap))
    assert (config.ep_rows_per_device, config.ep_declared_max_length) == (1, cap)
    if refused:
        with pytest.raises(ValueError, match="max_train_row_tokens|row-length cap"):
            _gate(config)
    else:
        _gate(config)


def test_environmental_turn_budgets_do_not_bound_a_whole_trajectory_row():
    """A short ``rollout_max_tokens`` still leaves the observation tokens of a multi-turn row unbounded."""
    config = _build(
        DistributedAsyncEnvironmentalGRPOTrainer,
        _training_config(completion=256),
        _prompts(1024),
        _rollout(None, rollout_max_tokens=512),
    )
    assert config.ep_declared_max_length == 0
    with pytest.raises(ValueError, match="context window"):
        _gate(config)


def test_the_eval_rows_set_the_budget_when_an_evaluation_runs():
    wide_eval = _training_config(completion=2048, train_rows=1, eval_rows=4, eval_strategy="steps")
    config = _build(DistributedOnlineGRPOTrainer, wide_eval, _prompts(2048))
    assert config.ep_rows_per_device == 4
    with pytest.raises(ValueError, match="16384 tokens/rank"):
        _gate(config)


def test_the_eval_rows_are_ignored_where_no_evaluation_runs():
    config = _build(DistributedOnlineGRPOTrainer, _training_config(train_rows=1, eval_rows=4), _prompts(2048))
    assert config.ep_rows_per_device == 1
    _gate(config)


def test_a_trainer_with_no_max_length_and_no_generation_shape_still_declares_nothing():
    config = _build(DistributedSFTTrainer, _training_config(), _prompts(2048))
    assert (config.ep_rows_per_device, config.ep_declared_max_length) == (0, 0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
