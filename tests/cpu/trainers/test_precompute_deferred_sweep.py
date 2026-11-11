#!/usr/bin/env python
"""TRL's constructor-time reference sweeps run after the toolkit wraps the model, not inside TRL's ``__init__``.
Swept unwrapped, an fp32-master model reaches the EP combine as fp32."""

import pytest
from accelerate import PartialState
from datasets import Dataset, concatenate_datasets
from trl import DPOTrainer, KTOTrainer

from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.preference.trl_preference import TRLPreferenceMixin
from tests.common.preference_precompute import REFERENCE_COLUMNS, TRAINERS, precompute_trainer, token_rows

PartialState()

TRL_BASES = {"dpo": DPOTrainer, "kto": KTOTrainer}
TRAIN_BATCH = 2
EVAL_BATCH = 3


@pytest.fixture(params=sorted(TRAINERS))
def kind(request):
    return request.param


@pytest.fixture
def construct(kind, monkeypatch):
    """Builds the real trainer's ``__init__`` around a TRL ``__init__`` that makes the constructor's sweep calls,
    returning the order of events."""
    trainer_cls, _ = TRAINERS[kind]
    events: list = []

    def trl_init(self, *args, precompute, train_dataset, eval_dataset, **kwargs):
        events.append("trl_init")
        self.train_dataset, self.eval_dataset = train_dataset, eval_dataset
        if not precompute:
            return
        self.train_dataset = self._precompute_ref_logps(self.train_dataset, "train", TRAIN_BATCH)
        if isinstance(self.eval_dataset, dict):
            self.eval_dataset = {
                name: self._precompute_ref_logps(dataset, name, EVAL_BATCH)
                for name, dataset in self.eval_dataset.items()
            }
        elif self.eval_dataset is not None:
            self.eval_dataset = self._precompute_ref_logps(self.eval_dataset, "eval", EVAL_BATCH)

    sweep = TRLPreferenceMixin._sweep_reference_logps

    def record_sweep(self, dataset, name, batch_size, needed):
        events.append(("sweep", name, batch_size))
        return sweep(self, dataset, name, batch_size, needed)

    monkeypatch.setattr(TRL_BASES[kind], "__init__", trl_init)
    monkeypatch.setattr(DistributedTrainerMixin, "_init_distributed_config", lambda self, kwargs, **_: kwargs)
    monkeypatch.setattr(trainer_cls, "_validate_reference_model", lambda self, ref_model: None)
    monkeypatch.setattr(trainer_cls, "_validate_held_reference_model", lambda self: None)
    monkeypatch.setattr(trainer_cls, "_configure_distributed_modes", lambda self: events.append("wrapped"))
    monkeypatch.setattr(trainer_cls, "_sweep_reference_logps", record_sweep)

    def build(*, precompute=True, eval_dataset=None, presharded=False, train=None):
        trainer = precompute_trainer(kind)
        trainer._dataset_presharded = presharded
        trainer_cls.__init__(
            trainer,
            model=None,
            args=None,
            precompute=precompute,
            train_dataset=token_rows(kind) if train is None else train,
            eval_dataset=eval_dataset,
        )
        return trainer, events

    return build


def test_splits_sweep_after_the_wrap_in_trls_order(kind, construct):
    trainer, events = construct(eval_dataset={"alpha": token_rows(kind), "beta": token_rows(kind)})

    assert events == [
        "trl_init",
        "wrapped",
        ("sweep", "train", TRAIN_BATCH),
        ("sweep", "alpha", EVAL_BATCH),
        ("sweep", "beta", EVAL_BATCH),
    ]
    for dataset in (trainer.train_dataset, *trainer.eval_dataset.values()):
        assert set(REFERENCE_COLUMNS[kind]) <= set(dataset.column_names)


def test_a_single_eval_dataset_sweeps_as_eval(kind, construct):
    trainer, events = construct(eval_dataset=token_rows(kind))

    assert events == ["trl_init", "wrapped", ("sweep", "train", TRAIN_BATCH), ("sweep", "eval", EVAL_BATCH)]
    assert set(REFERENCE_COLUMNS[kind]) <= set(trainer.eval_dataset.column_names)


def test_without_an_eval_dataset_only_train_sweeps(kind, construct):
    _, events = construct()

    assert events == ["trl_init", "wrapped", ("sweep", "train", TRAIN_BATCH)]


def test_precompute_off_sweeps_nothing(kind, construct):
    trainer, events = construct(precompute=False, eval_dataset={"alpha": token_rows(kind)})

    assert events == ["trl_init", "wrapped"]
    assert not set(REFERENCE_COLUMNS[kind]) & set(trainer.train_dataset.column_names)
    assert not set(REFERENCE_COLUMNS[kind]) & set(trainer.eval_dataset["alpha"].column_names)


def test_a_later_sweep_runs_at_once(kind, construct):
    """TRL's ``evaluate()`` sweeps a dataset it is handed after construction; that one is not deferred."""
    trainer, events = construct()
    events.clear()

    swept = trainer._precompute_ref_logps(token_rows(kind), "late", EVAL_BATCH)

    assert events == [("sweep", "late", EVAL_BATCH)]
    assert set(REFERENCE_COLUMNS[kind]) <= set(swept.column_names)


def test_an_eval_split_named_train_is_refused(kind, construct):
    with pytest.raises(ValueError, match="share the name 'train'"):
        construct(eval_dataset={"train": token_rows(kind)})


@pytest.mark.parametrize("carrier", ["train", "eval"])
def test_only_the_splits_without_columns_sweep(kind, construct, carrier):
    carried = {name: [-1.0] * len(token_rows(kind)) for name in REFERENCE_COLUMNS[kind]}
    with_columns = concatenate_datasets([token_rows(kind), Dataset.from_dict(carried)], axis=1)
    train, eval_dataset = (with_columns, token_rows(kind)) if carrier == "train" else (None, with_columns)

    _, events = construct(train=train, eval_dataset=eval_dataset)

    swept = "eval" if carrier == "train" else "train"
    batch = EVAL_BATCH if swept == "eval" else TRAIN_BATCH
    assert events == ["trl_init", "wrapped", ("sweep", swept, batch)]


def test_splits_trl_recorded_out_of_the_held_order_are_refused(kind, construct):
    trainer, _ = construct(precompute=False, eval_dataset=token_rows(kind))
    trainer._deferred_reference_splits = [("eval", EVAL_BATCH), ("train", TRAIN_BATCH)]

    with pytest.raises(RuntimeError, match="Re-audit TRLPreferenceMixin"):
        trainer._sweep_deferred_reference_splits()


def test_a_presharded_dataset_is_refused_before_the_wrap(kind, construct, monkeypatch):
    seen: list = []
    monkeypatch.setattr(TRAINERS[kind][0], "_configure_distributed_modes", lambda self: seen.append("wrapped"))

    with pytest.raises(ValueError, match="pre-sharded"):
        construct(presharded=True)

    assert seen == []


def test_presharded_columns_the_dataset_carries_are_accepted(kind, construct):
    carried = {name: [-1.0] * len(token_rows(kind)) for name in REFERENCE_COLUMNS[kind]}
    dataset = concatenate_datasets([token_rows(kind), Dataset.from_dict(carried)], axis=1)

    _, events = construct(presharded=True, train=dataset)

    assert events == ["trl_init", "wrapped"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
