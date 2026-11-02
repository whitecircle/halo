#!/usr/bin/env python
"""The FSDP2 internals ``complete_fsdp2_post_backward`` drives, against the installed torch, on one CPU rank.

The completion reads private FSDP2 state (``FSDPModule._get_fsdp_state()._fsdp_param_group``, each group's
window flags, sharded and training state and communication context, each ``FSDPParam``'s unsharded
gradient and accumulator) and calls ``post_backward``, ``finalize_backward`` and ``use_training_state``.
These tests pin every member it reads and the defect it repairs: a ``fully_shard`` block checkpointed
reentrant around its ``__call__`` and run three times before one backward is left gathered with an
unreduced gradient once the root's final callback has run, and the completion turns that into the
gradient of the same blocks run without checkpointing. A torch upgrade that renames a member fails the
first test; one that fixes the defect upstream fails the second, the cue to retire the completion. A gradient
FSDP2 parked in its accumulator (a no-reduce microstep under a reduce dtype other than the parameters') counts as
unreduced too: the guard refuses it and the completion reduces it.

Run: python tests/cpu/parallelism/test_fsdp2_post_backward_internals.py
"""

import copy
import inspect

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPCommContext, FSDPParamGroup
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

from halo.distributed import fsdp

HIDDEN = 8
BLOCKS = 2
ROWS = 3


class _Block(nn.Module):
    """A residual MLP checkpointed reentrant around its own ``__call__``, FSDP2's hooks inside the checkpoint
    (the placement of transformers' ``GradientCheckpointingLayer``)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(HIDDEN, 2 * HIDDEN)
        self.fc2 = nn.Linear(2 * HIDDEN, HIDDEN)
        self.checkpointed = False

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(x)))

    def __call__(self, x):
        if self.checkpointed and self.training:
            return checkpoint(super().__call__, x, use_reentrant=True)
        return super().__call__(x)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(HIDDEN, HIDDEN)
        self.blocks = nn.ModuleList(_Block() for _ in range(BLOCKS))

    def forward(self, rows):
        loss = 0.0
        for x in rows:
            h = self.embed(x)
            for block in self.blocks:
                h = block(h)
            loss = loss + h.pow(2).sum()
        return loss


@pytest.fixture
def single_rank_gloo(tmp_path):
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


def _rows() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return [torch.randn(2 + i, HIDDEN, generator=generator) for i in range(ROWS)]


def _models(mesh) -> tuple[nn.Module, nn.Module]:
    """The reference without checkpointing, and the checkpointed per-block FSDP2 model, from one init."""
    torch.manual_seed(0)
    reference = _Model()
    model = copy.deepcopy(reference)
    for block in model.blocks:
        block.checkpointed = True
        fully_shard(block, mesh=mesh, reshard_after_forward=False)
    fully_shard(model, mesh=mesh, reshard_after_forward=False)
    return reference, model


def _sharded_after_backward(mesh) -> tuple[nn.Module, nn.Module, list[FSDPParamGroup]]:
    """Both models after one backward over every row."""
    reference, model = _models(mesh)
    reference(_rows()).backward()
    model(_rows()).backward()
    return reference, model, fsdp.fsdp2_param_groups(model)


def _block_groups(groups: list[FSDPParamGroup]) -> list[FSDPParamGroup]:
    return [group for group in groups if "blocks." in repr(group)]


def test_the_members_the_completion_reads_exist(single_rank_gloo):
    """Every FSDP2 member the completion and its guard touch, on the objects a real wrap builds."""
    _reference, _model, groups = _sharded_after_backward(single_rank_gloo)
    assert {state.name for state in TrainingState} >= {"IDLE", "PRE_BACKWARD", "POST_BACKWARD"}
    assert list(inspect.signature(FSDPParamGroup.post_backward).parameters) == ["self", "unused"]
    assert list(inspect.signature(FSDPParamGroup.finalize_backward).parameters) == ["self"]
    assert list(inspect.signature(FSDPParamGroup.use_training_state).parameters) == ["self", "training_state"]
    for group in groups:
        assert isinstance(group, FSDPParamGroup)
        assert isinstance(group.reduce_grads, bool) and isinstance(group.reshard_after_backward, bool)
        assert isinstance(group.is_unsharded, bool) and isinstance(group.is_sharded, bool)
        assert isinstance(group._training_state, TrainingState)
        assert isinstance(group.comm_ctx, FSDPCommContext)
        assert hasattr(group.comm_ctx, "reduce_scatter_state") and hasattr(group.comm_ctx, "device_handle")
        for fsdp_param in group.fsdp_params:
            assert hasattr(fsdp_param, "_unsharded_param")
            assert fsdp_param.unsharded_param is fsdp_param._unsharded_param
            assert hasattr(fsdp_param, "unsharded_accumulated_grad")


def test_a_reentrant_recompute_still_leaves_the_post_backward_undone(single_rank_gloo):
    """The defect, as this torch has it: each block group ends the backward idle, gathered and holding an
    unreduced gradient, which the guard refuses."""
    _reference, _model, groups = _sharded_after_backward(single_rank_gloo)
    blocks = _block_groups(groups)
    assert len(blocks) == BLOCKS
    for group in blocks:
        assert group._training_state == TrainingState.IDLE
        assert group.is_unsharded
        assert any(fsdp_param.unsharded_param.grad is not None for fsdp_param in group.fsdp_params)
    with pytest.raises(RuntimeError, match="post-backward left undone"):
        fsdp.reject_unfinished_fsdp2_backward(groups)


def test_the_completion_reduces_every_segment_and_reshards(single_rank_gloo):
    """After the completion every group is sharded with nothing left unreduced, and each sharded gradient is
    the reference's."""
    reference, model, groups = _sharded_after_backward(single_rank_gloo)
    assert fsdp.complete_fsdp2_post_backward(groups) == BLOCKS
    fsdp.reject_unfinished_fsdp2_backward(groups)
    assert all(group.is_sharded for group in groups)
    for (name, param), ref in zip(model.named_parameters(), reference.parameters(), strict=True):
        assert isinstance(param, DTensor), name
        torch.testing.assert_close(param.grad.full_tensor(), ref.grad, msg=name)
    assert fsdp.complete_fsdp2_post_backward(groups) == 0


def test_the_completion_refuses_a_backward_in_flight(single_rank_gloo):
    """Called while the backward still runs, its groups mid-pass, the completion raises rather than reduce
    gradients the backward is still writing."""
    _reference, model = _models(single_rank_gloo)
    groups = fsdp.fsdp2_param_groups(model)

    def complete_mid_backward(_module, _args, output):
        output.register_hook(lambda grad: fsdp.complete_fsdp2_post_backward(groups))

    model.embed.register_forward_hook(complete_mid_backward)
    with pytest.raises(RuntimeError, match="mid-pass"):
        model(_rows()).backward()


def test_an_accumulated_gradient_with_the_reduce_armed_is_reduced(single_rank_gloo):
    """fp32 parameters reducing in fp64: a no-reduce microstep leaves every group's gradient in FSDP2's accumulator,
    off the parameter. With the reduce re-armed and no backward to follow, the guard refuses that state, and the
    completion reduces it to the reference's gradient."""
    torch.manual_seed(0)
    reference = _Model()
    model = copy.deepcopy(reference)
    policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float64)
    for block in model.blocks:
        fully_shard(block, mesh=single_rank_gloo, reshard_after_forward=False, mp_policy=policy)
    fully_shard(model, mesh=single_rank_gloo, reshard_after_forward=False, mp_policy=policy)
    model.set_requires_gradient_sync(False)
    reference(_rows()).backward()
    model(_rows()).backward()
    groups = fsdp.fsdp2_param_groups(model)
    parked = [fsdp_param for group in groups for fsdp_param in group.fsdp_params]
    assert all(p.unsharded_accumulated_grad is not None and p.unsharded_param.grad is None for p in parked)
    assert not any(group.is_unsharded for group in groups), "premise: only the accumulator holds the gradient"

    model.set_requires_gradient_sync(True)
    with pytest.raises(RuntimeError, match=rf"{len(groups)} group\(s\) still hold an unreduced gradient"):
        fsdp.reject_unfinished_fsdp2_backward(groups)
    assert fsdp.complete_fsdp2_post_backward(groups) == len(groups)
    fsdp.reject_unfinished_fsdp2_backward(groups)
    for (name, param), ref in zip(model.named_parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(param.grad.full_tensor(), ref.grad, msg=name)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
