#!/usr/bin/env python
"""A forward-only pipeline drive must not run the schedule's trailing gradient reduction.

Both single-stage schedules end ``_step_microbatches`` with ``stage.perform_reduce_grad``, which
``eval``'s backward suppression does not reach. On an FSDP2 stage that call reshards the stage and
finds no gradients to reduce, so each forward-only batch would pay a full parameter all-gather per
stage to unshard it again.

The counter below stands in for that reshard. These tests fail if it runs on any forward-only path,
and also if the suppression reaches into training or outlives its scope: a leak would silently
disable the real gradient reduction.

Run: pytest tests/cpu/parallelism/test_pp_eval_reshard.py
"""

import pytest
import torch
from torch.distributed.pipelining import PipelineStage

from halo.distributed.pipeline_parallel.runner import _REQUIRED_STAGE_INTERNALS
from tests.common.pp_fakes import fake_runner

ROWS = 4
COLS = 6


def _batch():
    input_ids = torch.ones(ROWS, COLS, dtype=torch.long)
    labels = torch.ones(ROWS, COLS, dtype=torch.long)
    return input_ids, labels


def test_a_training_step_still_reduces_gradients():
    """The suppression is scoped to forward-only drives.

    Skipping reduce-grad on a training step would leave FSDP2's gradients unreduced across the DP
    mesh, so every replica would step on its own shard's gradient alone.
    """
    rt = fake_runner()

    rt.step(*_batch())

    assert rt.stage.reshards == 1, "a training step must still run the schedule's gradient reduction"


@pytest.mark.parametrize("batches", [1, 5])
def test_eval_loss_never_reshards_the_stage(batches):
    """N eval batches cost zero reshards, not N.

    A forward-only drive produces no gradients, so reduce-grad has nothing to reduce and its only
    effect would be the reshard, paid once per eval batch.
    """
    rt = fake_runner()
    input_ids, labels = _batch()

    with torch.no_grad():
        for _ in range(batches):
            rt.eval_loss(input_ids, labels)

    assert rt.schedule.eval_calls == batches, "the eval drives did not reach the schedule"
    assert rt.stage.reshards == 0, f"{rt.stage.reshards} reshards across {batches} eval batches; expected none"


def test_forward_only_never_reshards_the_stage():
    """``forward_only`` drives the same schedule and must be suppressed on the same grounds."""
    rt = fake_runner()
    input_ids, _ = _batch()

    with torch.no_grad():
        rt.forward_only(input_ids)

    assert rt.schedule.eval_calls == 1
    assert rt.stage.reshards == 0


def test_the_dict_target_eval_path_is_suppressed_too():
    """``extra_targets`` on the last stage takes its own branch, straight into ``_step_microbatches``.

    That branch bypasses ``schedule.eval``, so a suppression wired only around the ``schedule.eval``
    call would leave every preference and GRPO eval resharding per batch.
    """
    rt = fake_runner()
    input_ids, labels = _batch()

    with torch.no_grad():
        rt.eval_loss(input_ids, labels, extra_targets={"margin": torch.zeros(ROWS)})

    assert rt.schedule.eval_calls == 0, "the dict-target branch should not route through schedule.eval"
    assert rt.stage.reshards == 0


def test_training_resumes_reducing_after_an_eval():
    """The suppression does not leak: a step after an eval reduces again.

    A shadow left installed on the stage would disable gradient reduction for the rest of the run
    while losses stay finite and the model still moves, on unreduced gradients.
    """
    rt = fake_runner()
    input_ids, labels = _batch()

    with torch.no_grad():
        rt.eval_loss(input_ids, labels)
        rt.forward_only(input_ids)
    rt.step(input_ids, labels)

    assert "perform_reduce_grad" not in rt.stage.__dict__, "the stage was left with the suppression installed"
    assert rt.stage.reshards == 1, "the training step after an eval did not reduce gradients"


def test_a_raising_drive_still_restores_the_stage():
    """A failure inside the schedule must not strand the stage with reduction disabled."""
    rt = fake_runner()

    def exploding(*args, **kwargs):
        raise RuntimeError("collective failed")

    rt.schedule.eval = exploding
    with torch.no_grad(), pytest.raises(RuntimeError, match="collective failed"):
        rt.eval_loss(*_batch())

    assert "perform_reduce_grad" not in rt.stage.__dict__


def test_the_shadowed_torch_member_is_declared_in_the_contract():
    """The runner shadows a private torch member, so a rename must fail loud at construction.

    ``_assert_pipelining_contract`` probes every name in ``_REQUIRED_STAGE_INTERNALS``. If
    ``perform_reduce_grad`` were missing from that tuple, a torch version that renamed it would leave
    the shadow on a dead attribute and the reshard would return unnoticed.
    """
    assert "perform_reduce_grad" in _REQUIRED_STAGE_INTERNALS
    assert hasattr(PipelineStage, "perform_reduce_grad"), "torch renamed the member the runner suppresses"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
