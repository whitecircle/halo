#!/usr/bin/env python
"""The on-policy GRPO trainers refuse the TRL options they cannot honor, before TRL's constructor.

TRL 1.14 adds options the toolkit's trainers would silently drop: ``use_adaptive_entropy`` (its
coefficient lives in a sidecar a sharded resume never restores), ``entropy_coef`` on the chunked sweep
(whose entropy is detached, so the bonus has no gradient), and an ``IterableDataset`` for the
DP-sharded loader (whose prompt-group sampler needs a length). The gate runs in the construction spine
both trainers share, after the chunked-sweep switch is read and before TRL opens the rollout client.

    python tests/cpu/grpo/test_on_policy_trl_option_refusals.py
"""

from types import SimpleNamespace

import pytest
from datasets import Dataset, IterableDataset

from src.configs.grpo_environmental import AsyncTrainingConfig
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from src.trainers.grpo.mixins.on_policy_init import OnPolicyGRPOInitMixin
from src.trainers.grpo.online import DistributedGRPOTrainer
from src.trainers.mixins.base import DistributedTrainerMixin

_ROWS = [{"prompt": "p"}] * 4


def _training_args(**fields):
    defaults = {"beta": 0.0, "use_liger_kernel": False, "use_adaptive_entropy": False, "entropy_coef": 0.0}
    return SimpleNamespace(**(defaults | fields))


@pytest.fixture
def begin(monkeypatch):
    """The shared ``_begin_on_policy_init`` on a bare trainer, the distributed setup stubbed out."""
    monkeypatch.setattr(DistributedTrainerMixin, "_init_distributed_config", lambda self, kwargs, **_: kwargs)

    def run(ctor_args=(), *, config=None, chunked=False, custom_loader=False, **kwargs):
        trainer = object.__new__(DistributedGRPOTrainer)
        trainer._use_chunked_grpo_logprobs = chunked
        trainer._needs_custom_dataloader = lambda: custom_loader
        if not ctor_args:
            kwargs["args"] = config or _training_args()
        return trainer._begin_on_policy_init(ctor_args, kwargs)

    return run


def test_adaptive_entropy_is_refused(begin):
    with pytest.raises(ValueError, match="use_adaptive_entropy"):
        begin(config=_training_args(use_adaptive_entropy=True))


def test_an_entropy_bonus_is_refused_only_on_the_chunked_sweep(begin):
    with pytest.raises(ValueError, match="detached entropy"):
        begin(chunked=True, config=_training_args(entropy_coef=0.01))
    begin(chunked=False, config=_training_args(entropy_coef=0.01))


@pytest.mark.parametrize(
    "datasets",
    [
        {"train_dataset": IterableDataset.from_generator(lambda: iter(_ROWS))},
        {"eval_dataset": {"held_out": IterableDataset.from_generator(lambda: iter(_ROWS))}},
    ],
    ids=["train", "eval_split"],
)
def test_an_iterable_dataset_is_refused_on_the_dp_sharded_loader(begin, datasets):
    with pytest.raises(ValueError, match="IterableDataset"):
        begin(custom_loader=True, **datasets)
    begin(custom_loader=False, **datasets)


def test_a_positional_iterable_train_dataset_is_refused(begin):
    """``GRPOTrainer(model, reward_funcs, args, train_dataset, ...)``."""
    iterable = IterableDataset.from_generator(lambda: iter(_ROWS))
    with pytest.raises(ValueError, match="train_dataset is an IterableDataset"):
        begin((None, None, _training_args(), iterable), custom_loader=True)


def test_a_map_style_dataset_passes(begin):
    begin(custom_loader=True, train_dataset=Dataset.from_list(_ROWS), eval_dataset=Dataset.from_list(_ROWS))


class _GateReached(Exception):
    pass


@pytest.mark.parametrize(
    ("trainer_cls", "kwargs"),
    [
        (DistributedGRPOTrainer, {"use_chunked_grpo_logprobs": True}),
        (
            DistributedAsyncEnvironmentalGRPOTrainer,
            {"async_config": AsyncTrainingConfig(use_chunked_grpo_logprobs=True)},
        ),
    ],
    ids=lambda v: getattr(v, "__name__", ""),
)
def test_each_constructor_reads_the_chunked_switch_before_the_gate(monkeypatch, trainer_cls, kwargs):
    seen = []

    def begin_on_policy_init(self, args, kwargs):
        seen.append(getattr(self, "_use_chunked_grpo_logprobs", None))
        raise _GateReached

    monkeypatch.setattr(OnPolicyGRPOInitMixin, "_begin_on_policy_init", begin_on_policy_init)
    with pytest.raises(_GateReached):
        trainer_cls(**kwargs)
    assert seen == [True]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
