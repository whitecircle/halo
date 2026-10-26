#!/usr/bin/env python
"""The pipeline runtime's dict targets: per-example side tensors split in lockstep with the rows.

``_step_with_dict_targets`` must split every extra target with the same ``tensor_split`` torch applies
to the rows. A drift pairs advantages or reference log-probs with the wrong examples while the loss
stays finite. These tests drive the method against a fake schedule and fail on any lockstep,
housekeeping or validation break.

Run: python tests/cpu/parallelism/test_pp_dict_targets.py
"""

import pytest
import torch

from src.distributed.pipeline_parallel.losses import PPLossAdapter
from src.distributed.pipeline_parallel.runtime import PipelineRuntime


class _FakeStage:
    def __init__(self):
        self.has_backward = None
        self.cleared = False

    def clear_runtime_states(self):
        self.cleared = True


class _FakeSchedule:
    """Records exactly what the real torch schedule would receive."""

    def __init__(self, n_microbatches: int, has_backward: bool = True):
        self._n_microbatches = n_microbatches
        self._has_backward = has_backward
        self._stage = _FakeStage()
        self.captured = None

    def _split_inputs(self, args, kwargs):
        # Torch splits args/kwargs with tensor_split too; the fake mirrors the row split.
        arg_mbs = [
            tuple(torch.tensor_split(a, self._n_microbatches)[i] for a in args) for i in range(self._n_microbatches)
        ]
        kwarg_mbs = [
            {k: torch.tensor_split(v, self._n_microbatches)[i] for k, v in (kwargs or {}).items()}
            for i in range(self._n_microbatches)
        ]
        return arg_mbs, kwarg_mbs

    def _step_microbatches(self, arg_mbs, kwarg_mbs, target_mbs, losses, return_outputs):
        assert self._stage.cleared, "clear_runtime_states must run before the microbatches"
        self.captured = (arg_mbs, kwarg_mbs, target_mbs, return_outputs)


def _runtime_with(schedule) -> PipelineRuntime:
    rt = object.__new__(PipelineRuntime)  # bypass __init__ (needs dist + FSDP); method under test is pure
    rt.schedule = schedule
    return rt


@pytest.mark.parametrize("rows,n_mb", [(8, 4), (7, 3), (5, 2)])
def test_extras_split_in_lockstep_with_labels(rows, n_mb):
    labels = torch.arange(rows).unsqueeze(1).expand(rows, 6).contiguous()
    advantages = torch.arange(rows, dtype=torch.float32) * 10
    ref_logps = torch.arange(rows, dtype=torch.float32) * -1
    schedule = _FakeSchedule(n_mb)
    rt = _runtime_with(schedule)

    rt._step_with_dict_targets((), {}, labels, {"advantages": advantages, "ref_logps": ref_logps}, [], backward=True)

    _, _, target_mbs, return_outputs = schedule.captured
    assert return_outputs is False
    assert len(target_mbs) == n_mb
    expected_labels = torch.tensor_split(labels, n_mb)
    for i, target in enumerate(target_mbs):
        assert set(target) == {"labels", "advantages", "ref_logps"}
        assert torch.equal(target["labels"], expected_labels[i])
        # Row r must carry advantage 10r and ref -r — any mispairing breaks this.
        row_ids = target["labels"][:, 0]
        assert torch.equal(target["advantages"], row_ids.float() * 10)
        assert torch.equal(target["ref_logps"], row_ids.float() * -1)


def test_housekeeping_matches_torch_step_preamble():
    schedule = _FakeSchedule(2, has_backward=True)
    rt = _runtime_with(schedule)
    rt._step_with_dict_targets((), {}, torch.zeros(4, 3, dtype=torch.long), {"a": torch.zeros(4)}, [], backward=True)
    assert schedule._stage.has_backward is True
    assert schedule._stage.cleared


def test_eval_suppresses_backward_and_restores_the_schedule_flag():
    """backward=False must reach the stage, or eval runs a backward into the next step's gradients;
    the schedule's own flag must come back so the following step still trains."""
    schedule = _FakeSchedule(2, has_backward=True)
    rt = _runtime_with(schedule)
    with torch.no_grad():
        rt._step_with_dict_targets(
            (), {}, torch.zeros(4, 3, dtype=torch.long), {"a": torch.zeros(4)}, [], backward=False
        )
    assert schedule._stage.has_backward is False
    assert schedule._has_backward is True


def test_backward_under_no_grad_raises():
    schedule = _FakeSchedule(2, has_backward=True)
    rt = _runtime_with(schedule)
    with torch.no_grad(), pytest.raises(RuntimeError, match="eval_loss"):
        rt._step_with_dict_targets(
            (), {}, torch.zeros(4, 3, dtype=torch.long), {"a": torch.zeros(4)}, [], backward=True
        )


def test_extra_target_dim0_mismatch_raises():
    rt = object.__new__(PipelineRuntime)
    with pytest.raises(ValueError, match="leading dim"):
        rt._validate_extra_targets(torch.zeros(4, 3), {"advantages": torch.zeros(5)})


def test_adapter_defaults_are_inert():
    adapter = PPLossAdapter(token_loss_fn=lambda logits, target: logits.sum())
    assert not adapter.paired_examples
    assert adapter.batch_transform is None and adapter.normalizer is None
    assert adapter.extra_target_keys == () and adapter.pad_spec is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
