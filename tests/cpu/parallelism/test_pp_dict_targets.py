#!/usr/bin/env python
"""The pipeline runner's dict targets: per-example side tensors split in lockstep with the rows.

``_step_with_dict_targets`` must split every extra target with the same ``tensor_split`` torch applies
to the rows. A drift pairs advantages or reference log-probs with the wrong examples while the loss
stays finite. These tests drive the method against a fake schedule and fail on any lockstep,
housekeeping or validation break.

Run: python tests/cpu/parallelism/test_pp_dict_targets.py
"""

import pytest
import torch

from tests.common.pp_fakes import FakeSchedule, fake_runner


@pytest.mark.parametrize("rows,n_mb", [(8, 4), (7, 3), (5, 2)])
def test_extras_split_in_lockstep_with_labels(rows, n_mb):
    labels = torch.arange(rows).unsqueeze(1).expand(rows, 6).contiguous()
    advantages = torch.arange(rows, dtype=torch.float32) * 10
    ref_logps = torch.arange(rows, dtype=torch.float32) * -1
    schedule = FakeSchedule(n_mb)
    rt = fake_runner(schedule)

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
    schedule = FakeSchedule(2, has_backward=True)
    rt = fake_runner(schedule)
    rt._step_with_dict_targets((), {}, torch.zeros(4, 3, dtype=torch.long), {"a": torch.zeros(4)}, [], backward=True)
    assert schedule._stage.has_backward is True
    assert schedule.cleared_before_microbatches == 1, "clear_runtime_states must run before the microbatches"


def test_eval_suppresses_backward_and_restores_the_schedule_flag():
    """backward=False must reach the stage, or eval runs a backward into the next step's gradients;
    the schedule's own flag must come back so the following step still trains."""
    schedule = FakeSchedule(2, has_backward=True)
    rt = fake_runner(schedule)
    with torch.no_grad():
        rt._step_with_dict_targets(
            (), {}, torch.zeros(4, 3, dtype=torch.long), {"a": torch.zeros(4)}, [], backward=False
        )
    assert schedule._stage.has_backward is False
    assert schedule._has_backward is True


def test_backward_under_no_grad_raises():
    schedule = FakeSchedule(2, has_backward=True)
    rt = fake_runner(schedule)
    with torch.no_grad(), pytest.raises(RuntimeError, match="eval_loss"):
        rt._step_with_dict_targets(
            (), {}, torch.zeros(4, 3, dtype=torch.long), {"a": torch.zeros(4)}, [], backward=True
        )


def test_extra_target_dim0_mismatch_raises():
    rt = fake_runner()
    with pytest.raises(ValueError, match="leading dim"):
        rt.step(
            torch.zeros(4, 3, dtype=torch.long),
            torch.zeros(4, 3, dtype=torch.long),
            extra_targets={"advantages": torch.zeros(5)},
        )


def test_paired_step_refuses_a_microbatch_that_splits_a_pair():
    rt = fake_runner(FakeSchedule(2))
    rt.rows_per_example = 2
    with pytest.raises(ValueError, match="chosen/rejected pair"):
        rt.step(torch.zeros(6, 3, dtype=torch.long), torch.zeros(6, 3, dtype=torch.long))
    assert rt.step(torch.zeros(4, 3, dtype=torch.long), torch.zeros(4, 3, dtype=torch.long)) is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
