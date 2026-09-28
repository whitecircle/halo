#!/usr/bin/env python
"""The helpers the replayed-resume GPU bodies grade with fail where their callers rely on them.

The merged-resume, precompute-resume and embedding-resume bodies compare a resumed run against the
uninterrupted one through these helpers and run only on the GPU tiers, so a helper that stopped failing
would pass every row there unnoticed. These pins drive each with stand-in inputs and assert the verdict
flips on the input that should flip it.

Run: python tests/cpu/conventions/test_resume_replay_checks.py
"""

import math
import random
from types import SimpleNamespace

import pytest
import torch

import src.optimizers.adamw_bf16 as adamw_bf16
from src.optimizers.adamw_bf16 import SR_SEED
from tests.common.checkpoint_io import LOADING_INFO_KINDS, ReplayRestorePoint, loading_problems
from tests.common.utils import relative_l2, resumed_loss_deltas, snapshot_trainable

SAVE_STEP = 2
TOTAL_STEPS = 4
UNINTERRUPTED = [3.0, 2.8, 2.6, 2.5]


def _tensors(*values) -> dict[str, torch.Tensor]:
    return {f"t{i}": torch.tensor(value) for i, value in enumerate(values)}


def test_relative_l2_is_zero_on_equal_tensors_and_scales_with_the_miss():
    reference = _tensors([3.0, 4.0])
    assert relative_l2(reference, reference) == 0.0
    assert relative_l2(_tensors([3.0, 4.5]), reference) == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("actual", "reference"),
    [
        ({}, {}),  # nothing compared
        (_tensors([1.0]), _tensors([0.0])),  # an all-zero reference has no scale
        (_tensors([1.0]), {"other": torch.tensor([1.0])}),  # a key dropped on one side
        (_tensors([1.0], [2.0]), _tensors([1.0])),  # an extra key on one side
    ],
)
def test_relative_l2_is_infinite_on_a_comparison_of_nothing(actual, reference):
    assert relative_l2(actual, reference) == math.inf


def test_relative_l2_propagates_nan():
    assert math.isnan(relative_l2(_tensors([math.nan]), _tensors([1.0])))


def test_resumed_loss_deltas_compare_the_steps_after_the_save():
    # A resumed run's log history opens with the steps its checkpoint carried.
    resumed = [*UNINTERRUPTED[:SAVE_STEP], 2.6, 2.75]
    deltas = resumed_loss_deltas(UNINTERRUPTED, resumed, save_step=SAVE_STEP, total_steps=TOTAL_STEPS)
    assert deltas == pytest.approx([0.0, 0.25])


@pytest.mark.parametrize(
    ("uninterrupted", "resumed"),
    [
        (UNINTERRUPTED, [2.6]),  # the resumed run stopped a step early
        (UNINTERRUPTED[:-1], UNINTERRUPTED),  # the uninterrupted run did
    ],
)
def test_resumed_loss_deltas_refuse_a_missing_step(uninterrupted, resumed):
    assert resumed_loss_deltas(uninterrupted, resumed, save_step=SAVE_STEP, total_steps=TOTAL_STEPS) is None


def test_resumed_loss_deltas_read_a_non_finite_loss_as_an_infinite_miss():
    resumed = [*UNINTERRUPTED[:SAVE_STEP], math.nan, 2.5]
    assert resumed_loss_deltas(UNINTERRUPTED, resumed, save_step=SAVE_STEP, total_steps=TOTAL_STEPS)[0] == math.inf


def test_loading_problems_names_only_the_non_empty_kinds():
    info = {kind: [] for kind in LOADING_INFO_KINDS}
    assert loading_problems(info) == {}
    info["unexpected_keys"] = ["model.extra.weight"]
    assert loading_problems(info) == {"unexpected_keys": ["model.extra.weight"]}


def test_loading_problems_raise_on_a_kind_the_info_no_longer_reports():
    info = {kind: [] for kind in LOADING_INFO_KINDS if kind != "mismatched_keys"}
    with pytest.raises(KeyError):
        loading_problems(info)


def test_snapshot_trainable_copies_the_trainable_parameters_only():
    model = torch.nn.Linear(2, 2)
    model.bias.requires_grad_(False)
    snapshot = snapshot_trainable(model)
    assert list(snapshot) == ["weight"]
    with torch.no_grad():
        model.weight.add_(1.0)
    assert not torch.equal(snapshot["weight"], model.weight), "a snapshot must not alias the live parameter"


class _Probe(ReplayRestorePoint):
    """Records where the SR stream stood when the snapshot's entries were taken."""

    def extra(self) -> dict:
        return {"sr_state": adamw_bf16._SR_RNG.getstate()}


def _trainer() -> SimpleNamespace:
    return SimpleNamespace(model=torch.nn.Linear(2, 2), lr_scheduler=SimpleNamespace(last_epoch=SAVE_STEP))


def test_the_replay_restore_point_rewinds_the_sr_stream_after_its_first_capture():
    adamw_bf16._SR_RNG.random()  # a run has advanced the stream past a fresh process's start
    advanced = adamw_bf16._SR_RNG.getstate()
    probe = _Probe("save", _trainer(), capture_optimizer=False)
    probe.on_save(None, SimpleNamespace(global_step=SAVE_STEP), None)
    assert probe.captured["sr_state"] == advanced, "the capture must read the state before the rewind"
    fresh = random.Random(SR_SEED)
    assert adamw_bf16._SR_RNG.getrandbits(64) == fresh.getrandbits(64), "not rewound to a fresh process's stream"

    # A later save (the final checkpoint a max_steps run writes) neither re-captures nor rewinds again.
    probe.on_save(None, SimpleNamespace(global_step=TOTAL_STEPS), None)
    assert adamw_bf16._SR_RNG.getrandbits(64) == fresh.getrandbits(64)
    assert probe.captured["global_step"] == SAVE_STEP


def test_the_replay_restore_point_fires_on_its_own_event_only():
    probe = _Probe("train_begin", _trainer(), capture_optimizer=False)
    probe.on_save(None, SimpleNamespace(global_step=SAVE_STEP), None)
    assert probe.captured is None
    probe.on_train_begin(None, SimpleNamespace(global_step=SAVE_STEP), None)
    assert probe.captured["global_step"] == SAVE_STEP


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
