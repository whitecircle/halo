#!/usr/bin/env python
"""The shared constructor bundle reaches every trainer, and its checkpoint knobs reach the savers.

``distributed_trainer_kwargs`` is the one bundle every entry script splats into its trainer. A trainer
whose explicit signature lacks one of its keys raises a ``TypeError`` at construction; one that stops
forwarding a checkpoint knob to ``_init_distributed_config``, or a spine that stops popping it, turns
``save_max_shard_size`` / ``allow_optimizer_warm_restart`` into silent no-ops, since the savers read them
only off the contexts the trainer builds; the same holds for ``max_consecutive_nonfinite_steps``, which the step
guard reads off the trainer. ``_init_distributed_config`` also owns the HF settings every distributed trainer
needs (``fsdp`` off, ``ddp_find_unused_parameters`` under EP).

Run: python tests/cpu/trainers/test_distributed_trainer_kwargs.py
"""

import inspect
from types import SimpleNamespace

import pytest
import torch.nn as nn
from transformers import TrainingArguments

from halo.entrypoints.script import distributed_trainer_kwargs
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.embedding.trainer import EmbeddingTrainer
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.parallelism import make_parallelism_config
from tests.common.rosters import distributed_trainer_classes

TRAINERS = distributed_trainer_classes()


def _bundle(parallelism_config) -> dict:
    args = SimpleNamespace(
        moe_balancing="auto",
        save_sharded_ep=False,
        save_max_shard_size="1GB",
        allow_optimizer_warm_restart=True,
        max_consecutive_nonfinite_steps=7,
    )
    return distributed_trainer_kwargs(args, parallelism_config, dataset_presharded=False)


def test_the_derived_roster_covers_every_trainer():
    """Anti-vacuity: a roster that stopped finding trainers would make the checks below assert nothing."""
    assert len(TRAINERS) >= 13, f"the roster collapsed to {sorted(cls.__name__ for cls in TRAINERS)}"


@pytest.mark.parametrize("trainer_cls", TRAINERS, ids=lambda cls: cls.__name__)
def test_the_bundle_binds_to_every_trainer_signature(trainer_cls):
    inspect.signature(trainer_cls.__init__).bind_partial(None, **_bundle(None))


def test_the_binding_check_rejects_a_signature_missing_a_key():
    """Anti-tautology: an explicit signature without ``**kwargs`` that drops a key must fail the bind."""

    def narrow(self, parallelism_config=None, moe_balancing="auto", save_sharded_ep=False, dataset_presharded=False):
        pass

    with pytest.raises(TypeError, match="save_max_shard_size"):
        inspect.signature(narrow).bind_partial(None, **_bundle(None))


def test_the_checkpoint_knobs_reach_the_checkpoint_contexts(tmp_path, monkeypatch):
    monkeypatch.setattr("halo.trainers.checkpointing.is_fs_save_rank", lambda: True)
    model = nn.Linear(2, 2)
    trainer = object.__new__(DistributedSFTTrainer)
    remaining = trainer._init_distributed_config(
        {
            "args": TrainingArguments(output_dir=str(tmp_path)),
            **_bundle(make_parallelism_config(world_size=1, gpus_per_node=1)),
        }
    )
    assert not {"save_max_shard_size", "allow_optimizer_warm_restart", "max_consecutive_nonfinite_steps"} & set(
        remaining
    ), "the trainer knobs were left in the kwargs the trainer forwards to its base"
    assert trainer._max_consecutive_nonfinite_steps == 7, "the non-finite step limit never reached the step guard"
    trainer.model, trainer.optimizer, trainer.lr_scheduler, trainer.processing_class = model, None, None, None
    trainer._top_level_model = lambda: model
    trainer._get_tp_rank = lambda: 0

    assert DistributedTrainerMixin._checkpoint_context(trainer).max_shard_size == "1GB"
    assert DistributedTrainerMixin._checkpoint_load_context(trainer).allow_optimizer_warm_restart is True


def test_the_embedding_signature_forwards_the_checkpoint_knobs(tmp_path, monkeypatch):
    """``EmbeddingTrainer`` names the knobs in its signature, so only its own forward carries them to the spine."""
    recorded = {}

    class Recorded(Exception):
        pass

    def record(self, kwargs, training_args=None, ctor_args=(), ctor_positions=None, **explicit):
        recorded.update({**explicit, **kwargs})
        raise Recorded

    monkeypatch.setattr(DistributedTrainerMixin, "_init_distributed_config", record)
    with pytest.raises(Recorded):
        EmbeddingTrainer(
            model=nn.Linear(2, 2),
            args=TrainingArguments(output_dir=str(tmp_path)),
            **_bundle(make_parallelism_config(world_size=1, gpus_per_node=1)),
        )
    assert recorded["save_max_shard_size"] == "1GB"
    assert recorded["allow_optimizer_warm_restart"] is True
    assert recorded["max_consecutive_nonfinite_steps"] == 7


@pytest.mark.parametrize("ep_size", [1, 2], ids=["dp", "ep"])
def test_the_spine_sets_the_hf_settings_every_distributed_trainer_needs(ep_size, tmp_path):
    args = TrainingArguments(output_dir=str(tmp_path))
    trainer = object.__new__(DistributedTrainerMixin)
    trainer._init_distributed_config(
        {"args": args, "parallelism_config": make_parallelism_config(world_size=2, gpus_per_node=2, ep_size=ep_size)}
    )
    assert args.fsdp == ""
    assert args.ddp_find_unused_parameters is (True if ep_size > 1 else None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
