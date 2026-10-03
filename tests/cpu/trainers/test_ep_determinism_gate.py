#!/usr/bin/env python
"""A trainer refuses ``full_determinism`` over an EP group that spans NVLink domains.

Across NVLink domains DeepEP runs its hybrid RDMA kernels, which have no deterministic mode: they place
received tokens in atomic claim order, so every expert weight gradient is summed in a different order on
every step and two identical runs drift apart. The refusal comes from the shared
``_init_distributed_config``, beside the flex-sliding one, before ``Trainer.__init__`` turns the mode on.
A node-local EP run and a run without ``full_determinism`` construct normally. The construction is cut
short right after that setup (TRL's own ctor needs a live model and, for GRPO, a server).

    python tests/cpu/trainers/test_ep_determinism_gate.py
"""

import types

import pytest
import torch.nn as nn
from accelerate import PartialState

from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from src.trainers.grpo.online import DistributedGRPOTrainer
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.preference.dpo import DistributedDPOTrainer
from src.trainers.sft import DistributedSFTTrainer
from tests.common.parallelism import create_config

PartialState()  # the trainers' accelerate logger requires an initialized state

TRAINERS = [
    DistributedSFTTrainer,
    DistributedDPOTrainer,
    DistributedGRPOTrainer,
    DistributedAsyncEnvironmentalGRPOTrainer,
]


class _SetupDone(Exception):
    """Carries control out of the trainer ctor once the distributed setup has run."""


@pytest.fixture(autouse=True)
def stop_after_distributed_setup(monkeypatch):
    real = DistributedTrainerMixin._init_distributed_config

    def _run_then_stop(self, kwargs, *args, **extra):
        real(self, kwargs, *args, **extra)
        raise _SetupDone

    monkeypatch.setattr(DistributedTrainerMixin, "_init_distributed_config", _run_then_stop)


def _construct(trainer_cls, *, full_determinism: bool, ep_scope: str):
    training_args = types.SimpleNamespace(
        full_determinism=full_determinism,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        bf16=False,
        optim="adamw_torch",
        save_on_each_node=False,
    )
    # 16 ranks over two 8-GPU NVLink domains: ep_scope=global spans both, ep_scope=node keeps each group in one.
    config = create_config(ep_size=8, world_size=16, gpus_per_node=8, ep_scope=ep_scope)
    trainer_cls(model=nn.Linear(2, 2), args=training_args, parallelism_config=config)


@pytest.mark.parametrize("trainer_cls", TRAINERS, ids=lambda cls: cls.__name__)
def test_full_determinism_over_a_cross_domain_ep_group_is_refused(trainer_cls):
    with pytest.raises(ValueError, match="one NVLink domain"):
        _construct(trainer_cls, full_determinism=True, ep_scope="global")


@pytest.mark.parametrize(
    ("full_determinism", "ep_scope"),
    [(True, "node"), (False, "global")],
    ids=["node_local_ep", "no_full_determinism"],
)
def test_runs_deepep_can_serve_construct(full_determinism, ep_scope):
    with pytest.raises(_SetupDone):
        _construct(DistributedSFTTrainer, full_determinism=full_determinism, ep_scope=ep_scope)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
