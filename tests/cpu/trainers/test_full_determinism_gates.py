#!/usr/bin/env python
"""A trainer refuses ``full_determinism`` where the run cannot honor it, before ``Trainer.__init__`` turns
the mode on.

Both refusals come from the shared ``_init_distributed_config``, which runs before ``super().__init__``:

* Once the flex-sliding warm-up compiled its graphs without the mode. Dynamo guards on it, so the
  sliding-layer graphs ``warmup_flex_sliding_kernels`` compiled at load would recompile rank by rank at
  each rank's first long sliding call, mid-forward, with EP peers waiting in DeepEP's dispatch. A
  process that warmed nothing (a single-rank run, or ``HALO_FLEX_SLIDING=0``), one whose warm-up already
  ran deterministic, and a run without ``full_determinism`` construct normally.
* On an EP group spanning NVLink domains, whose DeepEP dispatch has no deterministic mode. The entry
  scripts refuse it before the load (``tests/cpu/parallelism/test_cross_domain_determinism_gate.py``);
  this is the backstop for a hand-built config.

The construction is cut short right after that setup (TRL's own ctor needs a live model and, for GRPO,
a server).

    python tests/cpu/trainers/test_full_determinism_gates.py
"""

import types

import pytest
import torch
import torch.nn as nn
from accelerate import PartialState
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.models.patches import flex_sliding_attention
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from src.trainers.grpo.online import DistributedGRPOTrainer
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.preference.dpo import DistributedDPOTrainer
from src.trainers.sft import DistributedSFTTrainer
from tests.common.models import TINY_GEMMA4_WIDE_HEAD_CONFIG
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


def _construct(trainer_cls, full_determinism: bool, parallelism_config: ParallelismConfig | None = None):
    training_args = types.SimpleNamespace(
        full_determinism=full_determinism,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        bf16=False,
        optim="adamw_torch",
        save_on_each_node=False,
        output_dir="unused",
    )
    trainer_cls(
        model=nn.Linear(2, 2), args=training_args, parallelism_config=parallelism_config or ParallelismConfig()
    )


@pytest.mark.parametrize("trainer_cls", TRAINERS, ids=lambda cls: cls.__name__)
def test_full_determinism_after_a_non_deterministic_warmup_is_refused(trainer_cls, monkeypatch):
    monkeypatch.setattr(flex_sliding_attention, "_WARMED_DETERMINISTIC", False)
    with pytest.raises(ValueError, match="HALO_FLEX_SLIDING=0"):
        _construct(trainer_cls, full_determinism=True)


@pytest.mark.parametrize(
    ("warmed_deterministic", "full_determinism"),
    [(None, True), (True, True), (False, False)],
    ids=["no_warmup", "warmed_deterministic", "no_full_determinism"],
)
def test_runs_that_keep_the_warmed_mode_construct(warmed_deterministic, full_determinism, monkeypatch):
    monkeypatch.setattr(flex_sliding_attention, "_WARMED_DETERMINISTIC", warmed_deterministic)
    with pytest.raises(_SetupDone):
        _construct(DistributedSFTTrainer, full_determinism=full_determinism)


def test_a_warmup_that_compiles_nothing_records_no_mode(monkeypatch):
    """A wide-head model with no sliding layer is built with the flex-sliding implementation, but its warm-up
    has no graph to compile, so it must leave nothing for ``full_determinism`` to be refused over."""
    monkeypatch.setattr(flex_sliding_attention, "_WARMED_DETERMINISTIC", None)
    config = Gemma4TextConfig(**TINY_GEMMA4_WIDE_HEAD_CONFIG, layer_types=["full_attention", "full_attention"])
    config._attn_implementation = flex_sliding_attention.register_flex_sliding_attention()
    model = Gemma4ForCausalLM(config)
    assert flex_sliding_attention.sliding_attention_calls(model) == set()  # premise: no sliding layer
    flex_sliding_attention.warmup_flex_sliding_kernels(model, dtype=torch.bfloat16)
    assert flex_sliding_attention._WARMED_DETERMINISTIC is None
    with pytest.raises(_SetupDone):
        _construct(DistributedSFTTrainer, full_determinism=True)


def test_full_determinism_over_a_hand_built_cross_domain_ep_config_is_refused(monkeypatch):
    monkeypatch.setattr(flex_sliding_attention, "_WARMED_DETERMINISTIC", None)
    # 16 ranks over two 8-GPU NVLink domains, one EP group spanning both.
    config = create_config(ep_size=8, world_size=16, gpus_per_node=8, ep_scope="global")
    with pytest.raises(ValueError, match="spanning NVLink domains"):
        _construct(DistributedSFTTrainer, full_determinism=True, parallelism_config=config)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
