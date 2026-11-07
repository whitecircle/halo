"""The per-row token cap the EP capacity gate sizes against, read off the training config."""

import types

import pytest

from halo.configs.grpo_offline import OfflineGRPOConfig
from halo.configs.grpo_online import OnlineGRPOScriptArguments
from halo.configs.parallelism_arguments import ParallelismArguments, parallelism_config_from_args
from halo.models.context_window import bounded_row_length
from halo.trainers.distillation.sdpg import DistributedSDPGTrainer
from halo.trainers.grpo.offline.trainer import OfflineGRPOTrainer
from halo.trainers.grpo.online import DistributedOnlineGRPOTrainer
from tests.common.parallelism import simulated_world


def _offline(tmp_path, **overrides) -> OfflineGRPOConfig:
    return OfflineGRPOConfig(output_dir=str(tmp_path), report_to=[], **overrides)


def test_offline_grpo_rows_are_capped_by_prompt_plus_completion(tmp_path):
    config = _offline(tmp_path, max_prompt_length=512, max_completion_length=1024)
    assert config.max_length is None, "offline GRPO's max_length is PP-only, so the budget must come elsewhere"
    assert bounded_row_length(config) == 512 + 1024


def test_an_unbounded_side_leaves_the_gate_to_the_context_window(tmp_path):
    assert bounded_row_length(_offline(tmp_path, max_prompt_length=512, max_completion_length=None)) is None
    assert bounded_row_length(_offline(tmp_path, max_prompt_length=None, max_completion_length=1024)) is None


def test_max_length_wins_where_a_config_sets_it():
    class _Config:
        max_length = 4096
        max_prompt_length = 512
        max_completion_length = 1024

    assert bounded_row_length(_Config()) == 4096


@pytest.mark.parametrize("unset", [None, 0, -1])
def test_non_positive_lengths_read_as_unset(unset):
    class _Config:
        max_length = unset
        max_prompt_length = unset
        max_completion_length = unset

    assert bounded_row_length(_Config()) is None


def test_the_builder_hands_the_ep_gate_the_bounded_row_length(tmp_path):
    """Offline GRPO has no ``max_length`` key, so its budget reaches the gate only through this wiring; dropped, the
    gate judges the whole context window."""
    config = _offline(tmp_path, max_prompt_length=512, max_completion_length=1024, per_device_train_batch_size=2)
    args = ParallelismArguments(expert_parallel_size=8, ep_scope="node", fsdp_shard_ep1_experts=False)
    with simulated_world(world_size=8, gpus_per_node=8):
        built = parallelism_config_from_args(args, trainer_cls=OfflineGRPOTrainer, training_config=config)
    assert built.ep_declared_max_length == 512 + 1024
    assert built.ep_rows_per_device == 2


_HINT = "\n[Hint] answer: {answer}\n"


def test_sdpg_declares_the_hint_template_on_top_of_the_online_row():
    """The teacher row is prompt + hint + completion; the template's literal bytes bound the hint's fixed tokens."""
    config = types.SimpleNamespace(max_completion_length=1024)
    args = OnlineGRPOScriptArguments(max_prompt_length=512, sdpg_hint_template=_HINT, use_sdpg=True)
    online = DistributedOnlineGRPOTrainer.declared_row_length(config, args)
    assert online == 512 + 1024
    assert DistributedSDPGTrainer.declared_row_length(config, args) == online + len("\n[Hint] answer: \n")


def test_sdpg_with_the_term_off_declares_the_online_row():
    config = types.SimpleNamespace(max_completion_length=1024)
    args = OnlineGRPOScriptArguments(max_prompt_length=512, sdpg_beta_base=0.0, use_sdpg=True)
    assert DistributedSDPGTrainer.declared_row_length(config, args) == 512 + 1024


def test_an_unbounded_sdpg_row_stays_unbounded_and_a_missing_template_source_raises():
    config = types.SimpleNamespace(max_completion_length=None)
    args = OnlineGRPOScriptArguments(max_prompt_length=512, use_sdpg=True)
    assert DistributedSDPGTrainer.declared_row_length(config, args) is None
    bounded = types.SimpleNamespace(max_completion_length=1024)
    with pytest.raises(ValueError, match="SDPG arguments"):
        DistributedSDPGTrainer.declared_row_length(bounded, types.SimpleNamespace(max_prompt_length=512))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
