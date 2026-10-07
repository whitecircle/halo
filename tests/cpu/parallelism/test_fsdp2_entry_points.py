#!/usr/bin/env python
"""The two FSDP2 entry points must wrap a model identically — one body, one mesh — and the EP+TP
wrap must go through the TP one.

``setup_fsdp2_for_dp`` (plain/EP/CP data parallelism) and ``setup_fsdp2_for_tp`` (the DP dimension of
a TP run) differ only in how the DP mesh is obtained. A copy of the wrap per path would let a policy
or reshard fix land on one path and leave the other behind — silently, because both still report
success and every failure mode of a wrong FSDP2 policy (a cast that should not happen, a reduce
dtype, a layer that never became its own shard group) shows up as drift, not as an error.

The TP path also takes the mesh the loader already attached: rebuilding one there would leave the
model's params sharded on the first mesh while FSDP2 reduces their grads over the second. EP+TP
takes the same path with its EP modules as ignored params, so it inherits the cast policy and the
single-device check rather than wrapping beside them.

    python tests/cpu/parallelism/test_fsdp2_entry_points.py
"""

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

import src.distributed.fsdp as fsdp
from src.distributed.mesh import MeshDim
from src.trainers.mixins.grad_sync import GradientSyncMixin

_ARGS = SimpleNamespace(bf16=True, fp16=False, fp32_grad_reduce=False)
_FP32_ARGS = SimpleNamespace(bf16=False, fp16=False, fp32_grad_reduce=False)


class _FakeMesh:
    """The wrap reads a mesh's size, its dim names, and (TP) its ``dp`` slice — nothing else."""

    def __init__(self, size: int, dim_names: tuple[str, ...] = ()):
        self._size = size
        self.mesh_dim_names = dim_names

    def size(self) -> int:
        return self._size

    def __getitem__(self, dim: str) -> "_FakeMesh":
        assert dim in self.mesh_dim_names, f"sliced {dim!r} out of a mesh with {self.mesh_dim_names}"
        return _FakeMesh(self._size, ())


class _CausalLmLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        self.lm_head = nn.Linear(8, 4)


@contextlib.contextmanager
def _fake_world(world_size: int = 8):
    """A live group as far as the wrap can tell, logging as rank 0 (both paths log there)."""
    with (
        patch(f"{fsdp.__name__}.dist.is_initialized", return_value=True),
        patch(f"{fsdp.__name__}.dist.get_world_size", return_value=world_size),
        patch.object(fsdp, "is_global_main_process", return_value=True),
    ):
        yield


def _record_wrap(entry, **kwargs) -> list[tuple[str, int, torch.dtype | None, bool]]:
    """Run one entry point and return, per ``fully_shard`` call, what it wrapped and how.

    Modules are recorded by NAME so runs over two model instances stay comparable.
    """
    model = _CausalLmLike()
    names = {id(module): name or "<root>" for name, module in model.named_modules()}
    calls: list[tuple[str, int, torch.dtype | None, bool]] = []

    def fake_fully_shard(module, **shard_kwargs):
        policy = shard_kwargs["mp_policy"]
        calls.append(
            (
                names[id(module)],
                shard_kwargs["mesh"].size(),
                None if policy is None else policy.param_dtype,
                bool(shard_kwargs["reshard_after_forward"]),
            )
        )

    with (
        _fake_world(),
        patch.object(fsdp, "fully_shard", side_effect=fake_fully_shard),
        patch.object(fsdp, "create_dp_mesh", return_value=_FakeMesh(4)),
        patch.object(fsdp, "create_dp_tp_mesh", return_value=_FakeMesh(4, (MeshDim.DP, MeshDim.TP))),
    ):
        entry(model, **kwargs)
    return calls


def test_the_tp_entry_point_takes_its_dp_width_from_the_caller_not_the_world():
    """``data_parallel_size`` is ``stage_world_size // max(tp, cp, expert_tp)``; ``world // tp_size``
    reproduces it only while PP+TP, TP+CP and TP+ETP are all rejected. Deriving it here would build
    a DP mesh spanning ranks that hold a DIFFERENT batch the day one of those is allowed — grads
    reduced across unrelated data, with nothing to see but drift."""
    model = _CausalLmLike()
    model._device_mesh = _FakeMesh(4, (MeshDim.DP, MeshDim.TP))

    # world 8, tp 2 -> the world-derived answer is 4; a cp2/pp2 composition makes the true DP 2.
    with _fake_world(world_size=8), patch.object(fsdp, "_apply_fsdp2") as applied:
        wrapped, _ = fsdp.setup_fsdp2_for_tp(model, tp_size=2, dp_size=2, args=_ARGS)
    assert wrapped and "2×2" in applied.call_args.kwargs["topology"], (
        f"the wrap reported {applied.call_args.kwargs['topology']}, not the caller's dp_size"
    )

    # And dp_size==1 skips FSDP even though world // tp_size would say 4.
    with _fake_world(world_size=8), patch.object(fsdp, "_apply_fsdp2") as never:
        assert fsdp.setup_fsdp2_for_tp(model, tp_size=2, dp_size=1, args=_ARGS) == (False, None)
    never.assert_not_called()


def test_both_entry_points_wrap_the_same_modules_the_same_way():
    """The shared body, pinned end to end: same groups, same mesh, same policy, same reshard mode."""
    dp_calls = _record_wrap(fsdp.setup_fsdp2_for_dp, dp_size=4, args=_ARGS)
    tp_calls = _record_wrap(fsdp.setup_fsdp2_for_tp, tp_size=2, dp_size=4, args=_ARGS)

    assert dp_calls, "the DP entry point wrapped nothing"
    assert dp_calls == tp_calls, f"the two entry points diverged:\n  dp={dp_calls}\n  tp={tp_calls}"
    assert [name for name, *_ in dp_calls] == ["model.layers.0", "model.layers.1", "model", "<root>"], (
        f"per-layer groups plus backbone plus root is the wrap both paths owe: {dp_calls}"
    )
    assert {dtype for _, _, dtype, _ in dp_calls} == {torch.bfloat16}


def test_both_entry_points_carry_the_reshard_and_precision_knobs_into_the_wrap():
    """Anti-vacuity for the comparison above: the recorded fields must track the arguments."""
    for entry, axis in (
        (fsdp.setup_fsdp2_for_dp, {"dp_size": 4}),
        (fsdp.setup_fsdp2_for_tp, {"tp_size": 2, "dp_size": 4}),
    ):
        full_shard = _record_wrap(entry, args=_ARGS, reshard_after_forward=True, **axis)
        assert all(reshard for *_, reshard in full_shard), f"reshard_after_forward never reached fully_shard: {axis}"
        fp32 = _record_wrap(entry, args=_FP32_ARGS, **axis)
        assert {dtype for _, _, dtype, _ in fp32} == {torch.float32}, f"the args never reached the policy: {axis}"


def test_the_tp_entry_point_reuses_the_mesh_the_loader_attached():
    """A second mesh would shard the params on one group and reduce their grads over another."""
    model = _CausalLmLike()
    model._device_mesh = _FakeMesh(4, (MeshDim.DP, MeshDim.TP))

    with (
        _fake_world(),
        patch.object(fsdp, "fully_shard"),
        patch.object(fsdp, "create_dp_tp_mesh", side_effect=AssertionError("built a second mesh")),
    ):
        wrapped, mesh = fsdp.setup_fsdp2_for_tp(model, tp_size=2, dp_size=4, args=_ARGS)

    assert wrapped and mesh is model._device_mesh


def test_the_tp_entry_point_refuses_a_mesh_with_no_dp_dimension():
    """Wrapping over a TP-only mesh trains with no DP gradient sync, and reports success."""
    model = _CausalLmLike()
    model._device_mesh = _FakeMesh(4, (MeshDim.TP,))

    with (
        _fake_world(),
        patch.object(fsdp, "fully_shard"),
        pytest.raises(RuntimeError, match="requires a device mesh with a 'dp' dimension"),
    ):
        fsdp.setup_fsdp2_for_tp(model, tp_size=2, dp_size=4, args=_ARGS)


class _Fp32ResidualLayer(nn.Linear):
    """A layer declaring an fp32 activation across layer boundaries (Zaya's residual)."""

    _fp32_interlayer_residual = True


class _EPTPHost(GradientSyncMixin):
    """What the EP+TP wrap reads: the config, the args, the model and the FSDP exclusions."""

    def __init__(self, model, excluded):
        self.model = model
        self.args = _ARGS
        self.parallelism_config = SimpleNamespace(
            tp_size=2, data_parallel_size=4, fp32_non_ep_params=False, fsdp_reshard_after_forward=False
        )
        self._excluded = excluded

    def _fsdp_exclusions(self):
        return SimpleNamespace(params=self._excluded)


def _ep_tp_model() -> _CausalLmLike:
    model = _CausalLmLike()
    model.model.layers[0] = _Fp32ResidualLayer(8, 8)
    model._device_mesh = _FakeMesh(4, (MeshDim.DP, MeshDim.TP))
    return model


def test_the_ep_tp_wrap_runs_the_shared_body_with_its_ep_modules_left_out():
    """EP+TP wraps over the loader's mesh like TP does, plus the EP modules as ignored params; the
    shared body is what reads the model's fp32-residual declaration into the cast policy."""
    model = _ep_tp_model()
    expert = model.model.layers[1].weight
    calls = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs["mp_policy"], kwargs.get("ignored_params")))

    with (
        _fake_world(),
        patch.object(fsdp, "fully_shard", side_effect=fake_fully_shard),
        patch.object(fsdp, "create_dp_tp_mesh", side_effect=AssertionError("built a second mesh")),
    ):
        _EPTPHost(model, [expert])._setup_ep_tp_gradient_sync()

    assert calls, "the EP+TP wrap sharded nothing"
    assert {policy.cast_forward_inputs for _, policy, _ in calls} == {False}, (
        "the fp32 inter-layer residual must keep its dtype across the layer boundary"
    )
    layer_ignored = next(ignored for module, _, ignored in calls if module is model.model.layers[1])
    assert layer_ignored is not None and expert in layer_ignored, "the EP module was sharded with the layer"


def test_the_ep_tp_wrap_refuses_a_model_spread_over_two_devices():
    """The single-device check is rank-local and must run before the wrap's collective mesh work."""
    model = _ep_tp_model()
    model.lm_head.to("meta")

    with _fake_world(), patch.object(fsdp, "fully_shard"), pytest.raises(RuntimeError, match="single-device"):
        _EPTPHost(model, [])._setup_ep_tp_gradient_sync()


def test_neither_entry_point_wraps_a_single_rank_dp_group():
    """One DP rank has nothing to synchronize, and both paths must skip rather than shard."""
    with _fake_world(world_size=2), patch.object(fsdp, "fully_shard", side_effect=AssertionError("wrapped")):
        assert fsdp.setup_fsdp2_for_dp(_CausalLmLike(), dp_size=1, args=_ARGS) is False
        assert fsdp.setup_fsdp2_for_tp(_CausalLmLike(), tp_size=2, dp_size=1, args=_ARGS) == (False, None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
