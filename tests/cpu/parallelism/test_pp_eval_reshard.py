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

from src.distributed.pipeline_parallel.runtime import _REQUIRED_STAGE_INTERNALS, PipelineRuntime

N_MICROBATCHES = 2
ROWS = 4
COLS = 6


class _ReshardCountingStage:
    """Stand-in for ``PipelineStage`` wrapping an FSDP2 module, counting reduce-grad's reshard.

    ``perform_reduce_grad`` is defined on the class because the runtime suppresses it by shadowing the
    attribute in the instance dict: an instance attribute would be overwritten rather than shadowed,
    and a failed restore would go unnoticed.
    """

    def __init__(self):
        self.reshards = 0
        self.has_backward = True
        self.cleared = 0

    def clear_runtime_states(self):
        self.cleared += 1

    def perform_reduce_grad(self, grad_scale_factor):
        del grad_scale_factor
        self.reshards += 1


class _FakeSchedule:
    """Mirrors the real schedules' structure: ``step`` and ``eval`` both end in reduce-grad.

    In torch both funnel through ``_step_microbatches``, whose last statement is
    ``self._stage.perform_reduce_grad(...)``, and so do the fake's.
    """

    def __init__(self):
        self._stage = _ReshardCountingStage()
        self._n_microbatches = N_MICROBATCHES
        self._has_backward = True
        self._loss_fn = lambda output, target: output
        self._stage_forward_initialized = True
        self._stage_backward_initialized = True
        self.eval_calls = 0
        self.step_calls = 0

    def _split_inputs(self, args, kwargs):
        arg_mbs = [
            tuple(torch.tensor_split(a, self._n_microbatches)[i] for a in args) for i in range(self._n_microbatches)
        ]
        kwarg_mbs = [
            {k: torch.tensor_split(v, self._n_microbatches)[i] for k, v in (kwargs or {}).items()}
            for i in range(self._n_microbatches)
        ]
        return arg_mbs, kwarg_mbs

    def _step_microbatches(self, arg_mbs, kwarg_mbs, target_mbs, losses, return_outputs=False):
        del arg_mbs, kwarg_mbs, target_mbs, return_outputs
        losses.append(torch.tensor(1.0))
        self._stage.perform_reduce_grad(1)

    def step(self, *args, target=None, losses=None, return_outputs=False, **kwargs):
        del args, target, return_outputs, kwargs
        self.step_calls += 1
        self._step_microbatches(None, None, None, losses)

    def eval(self, *args, target=None, losses=None, return_outputs=False, **kwargs):
        del args, target, return_outputs, kwargs
        self.eval_calls += 1
        self._step_microbatches(None, None, None, losses if losses is not None else [])


def _runtime() -> PipelineRuntime:
    """A runtime around the fake schedule; ``__init__`` needs dist + FSDP, the drive logic does not."""
    rt = object.__new__(PipelineRuntime)
    rt.schedule = _FakeSchedule()
    rt.stage = rt.schedule._stage
    rt.stage_module = None
    rt.is_first = True
    rt.is_last = True
    rt.paired_examples = False
    rt.n_microbatches = N_MICROBATCHES
    rt._fused_head_loss = False
    rt._num_items_in_batch = 1
    return rt


def _batch():
    input_ids = torch.ones(ROWS, COLS, dtype=torch.long)
    labels = torch.ones(ROWS, COLS, dtype=torch.long)
    return input_ids, labels


def test_a_training_step_still_reduces_gradients():
    """The suppression is scoped to forward-only drives.

    Skipping reduce-grad on a training step would leave FSDP2's gradients unreduced across the DP
    mesh, so every replica would step on its own shard's gradient alone.
    """
    rt = _runtime()

    rt.step(*_batch())

    assert rt.stage.reshards == 1, "a training step must still run the schedule's gradient reduction"


@pytest.mark.parametrize("batches", [1, 5])
def test_eval_loss_never_reshards_the_stage(batches):
    """N eval batches cost zero reshards, not N.

    A forward-only drive produces no gradients, so reduce-grad has nothing to reduce and its only
    effect would be the reshard, paid once per eval batch.
    """
    rt = _runtime()
    input_ids, labels = _batch()

    with torch.no_grad():
        for _ in range(batches):
            rt.eval_loss(input_ids, labels)

    assert rt.schedule.eval_calls == batches, "the eval drives did not reach the schedule"
    assert rt.stage.reshards == 0, f"{rt.stage.reshards} reshards across {batches} eval batches; expected none"


def test_forward_only_never_reshards_the_stage():
    """``forward_only`` drives the same schedule and must be suppressed on the same grounds."""
    rt = _runtime()
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
    rt = _runtime()
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
    rt = _runtime()
    input_ids, labels = _batch()

    with torch.no_grad():
        rt.eval_loss(input_ids, labels)
        rt.forward_only(input_ids)
    rt.step(input_ids, labels)

    assert "perform_reduce_grad" not in rt.stage.__dict__, "the stage was left with the suppression installed"
    assert rt.stage.reshards == 1, "the training step after an eval did not reduce gradients"


def test_a_raising_drive_still_restores_the_stage():
    """A failure inside the schedule must not strand the stage with reduction disabled."""
    rt = _runtime()

    def exploding(*args, **kwargs):
        raise RuntimeError("collective failed")

    rt.schedule.eval = exploding
    with torch.no_grad(), pytest.raises(RuntimeError, match="collective failed"):
        rt.eval_loss(*_batch())

    assert "perform_reduce_grad" not in rt.stage.__dict__


def test_the_shadowed_torch_member_is_declared_in_the_contract():
    """The runtime shadows a private torch member, so a rename must fail loud at construction.

    ``_assert_pipelining_schedule_contract`` probes every name in ``_REQUIRED_STAGE_INTERNALS``. If
    ``perform_reduce_grad`` were missing from that tuple, a torch version that renamed it would leave
    the shadow on a dead attribute and the reshard would return unnoticed.
    """
    assert "perform_reduce_grad" in _REQUIRED_STAGE_INTERNALS
    assert hasattr(PipelineStage, "perform_reduce_grad"), "torch renamed the member the runtime suppresses"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
