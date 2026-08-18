#!/usr/bin/env python
"""
Server-free checks for DistributedGRPOTrainer (online GRPO with vLLM).

Online GRPO refuses to construct without vLLM server-mode generation, so nothing here builds a
trainer: what is reachable without a server is the declared surface plus the reward extraction.

1. ParallelismConfig - defaults and mode-detection flags the trainer branches on
2. Trainer class attributes - the ``_supports_*`` flags and the MRO the mixin spine depends on
3. Reward extraction - ``extract_last_boxed`` over the completion shapes TRL hands a reward fn

The construction gate itself (``_require_vllm_server_mode``) is pinned on CPU by
``tests/cpu/grpo/test_grpo_dataloader_and_server_gates.py``.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/grpo/test_online_grpo_mock.py
"""

import torch

from src.environments.rewards import extract_last_boxed
from tests.common.harness import gpu_test_main, record_check
from tests.common.utils import log


def accuracy_reward(completions, answer, **kwargs):
    """Reward based on whether the completion's boxed answer matches ground truth."""
    rewards = []
    for completion, gt in zip(completions, answer, strict=False):
        content = (completion[-1]["content"] if completion else "") if isinstance(completion, list) else completion
        extracted = (extract_last_boxed(content) or "").strip()
        gt_normalized = str(gt).strip()
        rewards.append(1.0 if extracted == gt_normalized else 0.0)
    return rewards


def test_parallelism_config_defaults():
    """A bare ParallelismConfig is single-replica on every axis.

    The trainer builds one when the caller passes none, so a default that drifted above 1 would
    silently shard a run nobody asked to shard.
    """
    from src.distributed.parallelism_config import ParallelismConfig

    config = ParallelismConfig()
    assert config.ep_size == 1
    assert config.tp_size == 1
    assert config.cp_size == 1
    assert config.expert_tp_size == 1
    assert config.pp_size == 1


def test_parallelism_config_modes():
    """Verify ParallelismConfig mode detection flags."""
    from src.distributed.parallelism_config import ParallelismConfig

    config_none = ParallelismConfig()
    assert not config_none.is_ep_mode
    assert not config_none.is_cp_mode
    assert not config_none.is_tp_mode
    assert not config_none.is_ep_tp_mode

    config_ep = ParallelismConfig(ep_size=2)
    assert config_ep.is_ep_mode
    assert not config_ep.is_cp_mode
    assert not config_ep.is_tp_mode

    config_tp = ParallelismConfig(tp_size=2)
    assert config_tp.is_tp_mode
    assert not config_tp.is_ep_mode
    assert not config_tp.is_cp_mode

    config_cp = ParallelismConfig(cp_size=2)
    assert config_cp.is_cp_mode
    assert not config_cp.is_ep_mode
    assert not config_cp.is_tp_mode


def test_trainer_class_flags():
    """Verify DistributedGRPOTrainer class-level parallelism support flags."""
    from src.trainers.grpo.online import DistributedGRPOTrainer

    assert DistributedGRPOTrainer._supports_ep is True
    assert DistributedGRPOTrainer._supports_cp is False
    assert DistributedGRPOTrainer._supports_tp is True


def test_trainer_mixin_inheritance():
    """Verify DistributedGRPOTrainer inherits from correct classes."""
    from trl import GRPOTrainer

    from src.trainers.grpo.online import DistributedGRPOTrainer
    from src.trainers.mixins.base import DistributedTrainerMixin

    assert issubclass(DistributedGRPOTrainer, DistributedTrainerMixin)
    assert issubclass(DistributedGRPOTrainer, GRPOTrainer)


def test_accuracy_reward_correct():
    """Test accuracy reward with correct answer."""
    completions = ["<think>Let me calculate.</think> The answer is \\boxed{42}."]
    answer = ["42"]
    rewards = accuracy_reward(completions, answer)
    assert rewards == [1.0], f"Expected [1.0], got {rewards}"


def test_accuracy_reward_incorrect():
    """Test accuracy reward with incorrect answer."""
    completions = ["The answer is \\boxed{43}."]
    answer = ["42"]
    rewards = accuracy_reward(completions, answer)
    assert rewards == [0.0], f"Expected [0.0], got {rewards}"


def test_accuracy_reward_no_boxed():
    """Test accuracy reward when no boxed answer is present."""
    completions = ["The answer is 42."]
    answer = ["42"]
    rewards = accuracy_reward(completions, answer)
    assert rewards == [0.0], f"Expected [0.0], got {rewards}"


def test_accuracy_reward_conversation_format():
    """Test accuracy reward with conversation format (list of dicts)."""
    completions = [[{"role": "assistant", "content": "Let me think... \\boxed{100}"}]]
    answer = ["100"]
    rewards = accuracy_reward(completions, answer)
    assert rewards == [1.0], f"Expected [1.0], got {rewards}"


def test_parallelism_config_mode_string():
    """Verify mode_string property gives human-readable description."""
    from src.distributed.parallelism_config import ParallelismConfig

    config_none = ParallelismConfig()
    mode_str = config_none.mode_string or ""
    assert isinstance(mode_str, str)

    config_ep = ParallelismConfig(ep_size=2)
    mode_str_ep = config_ep.mode_string
    assert mode_str_ep is not None and len(mode_str_ep) > 0
    assert "EP" in mode_str_ep.upper() or "expert" in mode_str_ep.lower()


def test_parallelism_config_ep_not_cp():
    """Verify that EP+CP is valid but CP alone is rejected by GRPO trainer class."""
    from src.trainers.grpo.online import DistributedGRPOTrainer

    assert DistributedGRPOTrainer._supports_cp is False

    # The refusal is the trainer's, not the config's — cp_size > 1 stays constructible.
    from src.distributed.parallelism_config import ParallelismConfig

    config = ParallelismConfig(cp_size=2)
    assert config.is_cp_mode is True


def run(ctx):
    log(f"\n{'=' * 70}")
    log("  Online GRPO server-free checks (DistributedGRPOTrainer)")
    log(f"  World size: {ctx.world_size}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"{'=' * 70}")

    checks: dict[str, bool] = {}

    log("\n--- Configuration Tests ---")
    record_check(checks, "ParallelismConfig defaults", test_parallelism_config_defaults)
    record_check(checks, "ParallelismConfig modes", test_parallelism_config_modes)
    record_check(checks, "ParallelismConfig mode_string", test_parallelism_config_mode_string)

    log("\n--- Class Structure Tests ---")
    record_check(checks, "Trainer class flags", test_trainer_class_flags)
    record_check(checks, "Trainer MRO inheritance", test_trainer_mixin_inheritance)
    record_check(checks, "ParallelismConfig EP not CP", test_parallelism_config_ep_not_cp)

    log("\n--- Reward Function Tests ---")
    record_check(checks, "Accuracy reward (correct)", test_accuracy_reward_correct)
    record_check(checks, "Accuracy reward (incorrect)", test_accuracy_reward_incorrect)
    record_check(checks, "Accuracy reward (no boxed)", test_accuracy_reward_no_boxed)
    record_check(checks, "Accuracy reward (conversation format)", test_accuracy_reward_conversation_format)

    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="online_grpo_mock", partial_state=False)(run)

if __name__ == "__main__":
    main()
