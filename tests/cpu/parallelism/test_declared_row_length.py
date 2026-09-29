"""The per-row token cap the EP capacity gate sizes against, read off the training config."""

import pytest

from src.args.distributed_args import DistributedArguments
from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.trainers.grpo.offline import OfflineGRPOTrainer
from src.training.parallelism_args import declared_row_length, parallelism_config_from_args
from tests.common.parallelism import simulated_world


def _offline(tmp_path, **overrides) -> OfflineGRPOConfig:
    return OfflineGRPOConfig(output_dir=str(tmp_path), report_to=[], **overrides)


def test_offline_grpo_rows_are_capped_by_prompt_plus_completion(tmp_path):
    config = _offline(tmp_path, max_prompt_length=512, max_completion_length=1024)
    assert config.max_length is None, "offline GRPO's max_length is PP-only, so the budget must come elsewhere"
    assert declared_row_length(config) == 512 + 1024


def test_an_unbounded_side_leaves_the_gate_to_the_context_window(tmp_path):
    assert declared_row_length(_offline(tmp_path, max_prompt_length=512, max_completion_length=None)) == 0
    assert declared_row_length(_offline(tmp_path, max_prompt_length=None, max_completion_length=1024)) == 0


def test_max_length_wins_where_a_config_sets_it():
    class _Config:
        max_length = 4096
        max_prompt_length = 512
        max_completion_length = 1024

    assert declared_row_length(_Config()) == 4096


@pytest.mark.parametrize("unset", [None, 0, -1])
def test_non_positive_lengths_read_as_unset(unset):
    class _Config:
        max_length = unset
        max_prompt_length = unset
        max_completion_length = unset

    assert declared_row_length(_Config()) == 0


def test_the_builder_hands_the_ep_gate_the_declared_row_length(tmp_path):
    """Offline GRPO refuses ``max_length``, so its budget reaches the EP capacity gate only through
    this wiring; dropped, the gate would judge the run against the model's whole context window."""
    config = _offline(tmp_path, max_prompt_length=512, max_completion_length=1024, per_device_train_batch_size=2)
    args = DistributedArguments(expert_parallel_size=8, ep_scope="node", fsdp_shard_ep1_experts=False)
    with simulated_world(world_size=8, gpus_per_node=8):
        built = parallelism_config_from_args(args, trainer_cls=OfflineGRPOTrainer, training_config=config)
    assert built.ep_declared_max_length == 512 + 1024
    assert built.ep_rows_per_device == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
