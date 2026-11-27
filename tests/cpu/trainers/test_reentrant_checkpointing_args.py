#!/usr/bin/env python
"""The ``use_reentrant`` the mixin forces reaches the model's ``activation_checkpointing_enable`` where no EP layer
enables checkpointing itself: CP on a dense model and a routed MoE without EP layers. transformers' ``Trainer.train``
reads ``activation_checkpointing_kwargs``, whichever spelling the run configured."""

import pytest
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM, Trainer, TrainingArguments

from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.models import TINY_QWEN3_CONFIG, TINY_QWEN3_MOE_CONFIG

ROW = {"input_ids": [2, 3, 4], "labels": [2, 3, 4]}


class _InitStopped(Exception):
    """Carries control out of the trainer's shared init after the training-args overrides."""


class _EnableReached(Exception):
    """Carries the kwargs ``Trainer.train`` hands the model out of ``train()``."""


class _InitHost:
    def _snapshot_model_config_state(self, model_config):
        pass

    def _apply_training_args_overrides(self, training_args, model):
        DistributedTrainerMixin._apply_training_args_overrides(self, training_args, model)
        raise _InitStopped


def _dense():
    return Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG, pad_token_id=0, eos_token_id=1))


def _routed_moe():
    return Qwen3MoeForCausalLM(Qwen3MoeConfig(**TINY_QWEN3_MOE_CONFIG, pad_token_id=0, eos_token_id=1))


def _kwargs_reaching_enable(tmp_path, model, parallelism: ParallelismConfig, **checkpointing) -> dict | None:
    args = TrainingArguments(output_dir=str(tmp_path), max_steps=1, report_to="none", use_cpu=True, **checkpointing)
    with pytest.raises(_InitStopped):
        DistributedTrainerMixin._init_distributed_config(
            _InitHost(), {"parallelism_config": parallelism, "model": model, "args": args}
        )

    def record(activation_checkpointing_kwargs=None, **_):
        raise _EnableReached(activation_checkpointing_kwargs)

    model.activation_checkpointing_enable = record
    with pytest.raises(_EnableReached) as reached:
        Trainer(model=model, args=args, train_dataset=[ROW]).train()
    return reached.value.args[0]


SPELLINGS = {
    "deprecated": {"gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False}},
    "current": {"activation_checkpointing": True, "activation_checkpointing_kwargs": {"use_reentrant": False}},
    "unset_kwargs": {"gradient_checkpointing": True},
}
MODES = {
    "cp2_dense": (_dense, {"world_size": 2, "gpus_per_node": 2, "cp_size": 2}),
    "routed_moe_without_ep_layers": (_routed_moe, {"world_size": 1, "gpus_per_node": 1}),
}


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("mode", MODES)
def test_forced_reentrancy_reaches_the_model(tmp_path, mode, spelling):
    build, parallelism = MODES[mode]
    reached = _kwargs_reaching_enable(tmp_path, build(), ParallelismConfig(**parallelism), **SPELLINGS[spelling])
    assert reached == {"use_reentrant": True}


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_a_dense_fsdp_run_keeps_its_configured_reentrancy(tmp_path):
    reached = _kwargs_reaching_enable(
        tmp_path, _dense(), ParallelismConfig(world_size=2, gpus_per_node=2), **SPELLINGS["deprecated"]
    )
    assert reached == {"use_reentrant": False}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
