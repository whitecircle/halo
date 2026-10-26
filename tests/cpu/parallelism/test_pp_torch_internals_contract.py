#!/usr/bin/env python
"""The pipeline runtime against real ``torch.distributed.pipelining``, on one CPU rank.

The runtime steers private FSDP2 state, schedule and stage members, probes them at construction, and
calls the schedule's private split, init and microbatch-step paths directly. These tests build the
real engine over a ``fully_shard`` stage root on a single-rank gloo group, as the trainer wraps it:
construction must fail naming any member torch lacks, and a one-stage chain's step, loss-only eval and
forward-only pass must match the unsplit model. A torch upgrade that renames a member or changes a
private signature fails here, before a PP GPU run.

Run: python tests/cpu/parallelism/test_pp_torch_internals_contract.py
"""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.pipelining import Schedule1F1B, ScheduleGPipe

from src.data.spans import LABEL_IGNORE_INDEX
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.pipeline_parallel import runtime
from src.distributed.pipeline_parallel.losses import causal_lm_token_loss, loss_token_count

VOCAB = 16
HIDDEN = 8
ROWS = 6
MICROBATCHES = 3
ABSENT_MEMBER = "_member_no_torch_defines"
SCHEDULES = sorted(runtime._SCHEDULES)


class _TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.head = nn.Linear(HIDDEN, VOCAB)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(input_ids))


@pytest.fixture
def single_rank_gloo(tmp_path):
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.fixture
def models(single_rank_gloo):
    """``(stage_root, reference)``: the stage root under ``fully_shard`` and an unsharded copy of it."""
    torch.manual_seed(0)
    stage_root = _TinyCausalLM()
    reference = copy.deepcopy(stage_root)
    fully_shard(stage_root, mesh=init_device_mesh("cpu", (1,)), reshard_after_forward=False)
    return stage_root, reference


def _runtime(stage_root: nn.Module, schedule: str, **kwargs) -> runtime.PipelineRuntime:
    """The engine for a one-stage chain.

    A ``ParallelismConfig`` refuses a pipeline schedule at ``pp_size=1``, so a namespace carries the
    five fields the runtime reads.
    """
    config = SimpleNamespace(pp_rank=0, pp_size=1, pp_schedule=schedule, is_first_pp_stage=True, is_last_pp_stage=True)
    return runtime.PipelineRuntime(stage_root, config, dist.group.WORLD, torch.device("cpu"), MICROBATCHES, **kwargs)


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Rows with unequal target counts, so normalizing per microbatch would change the loss."""
    input_ids = torch.randint(VOCAB, (ROWS, 7), generator=torch.Generator().manual_seed(1))
    labels = input_ids.clone()
    for row in range(ROWS):
        labels[row, : row + 1] = LABEL_IGNORE_INDEX
    return input_ids, labels


def _reference_loss_and_grads(reference: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor):
    loss = causal_lm_token_loss(reference(input_ids), labels) / loss_token_count(labels)
    loss.backward()
    return loss.detach(), {name: param.grad for name, param in reference.named_parameters()}


def _assert_grads_match(stage_root: nn.Module, expected_grads: dict[str, torch.Tensor]) -> None:
    # An eval drive leaves the unsharded params registered; the reduced gradients sit on the shards.
    reshard_fsdp2_modules(stage_root)
    for name, param in stage_root.named_parameters():
        assert param.grad is not None, f"{name} received no reduced gradient"
        torch.testing.assert_close(param.grad.full_tensor(), expected_grads[name], msg=name)


@pytest.mark.parametrize("schedule,schedule_cls", [("1f1b", Schedule1F1B), ("gpipe", ScheduleGPipe)])
def test_the_runtime_builds_the_configured_schedule_without_grad_scaling(schedule, schedule_cls, models):
    engine = _runtime(models[0], schedule)

    assert type(engine.schedule) is schedule_cls
    # The loss already divides by the step's token count; torch's scaling would divide again.
    assert engine.schedule.scale_grads is False


@pytest.mark.parametrize(
    "required", ["_REQUIRED_FSDP_INTERNALS", "_REQUIRED_SCHEDULE_INTERNALS", "_REQUIRED_STAGE_INTERNALS"]
)
def test_construction_fails_naming_a_member_torch_lacks(required, models, monkeypatch):
    monkeypatch.setattr(runtime, required, (*getattr(runtime, required), ABSENT_MEMBER))

    with pytest.raises(RuntimeError, match=ABSENT_MEMBER):
        _runtime(models[0], "1f1b")


def test_construction_refuses_a_stage_root_outside_fully_shard(single_rank_gloo):
    with pytest.raises(RuntimeError, match="not an FSDPModule"):
        _runtime(_TinyCausalLM(), "1f1b")


@pytest.mark.parametrize("eval_first", [False, True], ids=["step_first", "eval_first"])
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_a_one_stage_chain_matches_the_unsplit_model(schedule, eval_first, models):
    """Step, loss-only eval and forward-only agree with the unsplit model in either order.

    Eval first initializes the stage from a drive without backward and must leave the following step
    training; step first must leave the eval drives scoring the same model.
    """
    stage_root, reference = models
    input_ids, labels = _batch()
    expected_loss, expected_grads = _reference_loss_and_grads(reference, input_ids, labels)
    engine = _runtime(stage_root, schedule)

    def assert_eval_drives_match():
        torch.testing.assert_close(engine.eval_loss(input_ids, labels), expected_loss)
        torch.testing.assert_close(engine.forward_only(input_ids), reference(input_ids).detach())

    if eval_first:
        assert_eval_drives_match()
    torch.testing.assert_close(engine.step(input_ids, labels).detach(), expected_loss)
    if not eval_first:
        assert_eval_drives_match()

    _assert_grads_match(stage_root, expected_grads)


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_dict_targets_reach_the_loss_with_their_own_rows(schedule, models):
    """Per-example side tensors go through the schedule's own microbatch step beside their rows."""
    stage_root, reference = models
    input_ids, labels = _batch()
    expected_loss, expected_grads = _reference_loss_and_grads(reference, input_ids, labels)
    targets_seen = []

    def token_loss_fn(logits, target):
        targets_seen.append(target)
        return causal_lm_token_loss(logits, target["labels"])

    engine = _runtime(stage_root, schedule, token_loss_fn=token_loss_fn)
    extra_targets = {"row": torch.arange(ROWS)}

    torch.testing.assert_close(engine.eval_loss(input_ids, labels, extra_targets=extra_targets), expected_loss)
    torch.testing.assert_close(engine.step(input_ids, labels, extra_targets=extra_targets).detach(), expected_loss)

    microbatch_rows = [split.tolist() for split in torch.tensor_split(torch.arange(ROWS), MICROBATCHES)]
    assert [target["row"].tolist() for target in targets_seen] == microbatch_rows * 2, "eval, then step"
    for target in targets_seen:
        assert torch.equal(target["labels"], labels[target["row"]])
    _assert_grads_match(stage_root, expected_grads)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
