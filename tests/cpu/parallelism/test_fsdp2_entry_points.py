#!/usr/bin/env python
"""The two FSDP2 entry points wrap a model identically, and EP+TP wraps through the TP one.
Every failure of a wrong FSDP2 policy shows up as drift, so a per-path copy would hide a fix landing on one path."""

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

import halo.distributed.fsdp as fsdp
import halo.trainers.parallelism.mode_setup as mode_setup
from halo.distributed.mesh import MeshDim
from halo.trainers.parallelism.mode_setup import ModeSetupMixin

_BF16 = {"compute_dtype": torch.bfloat16, "fp32_reduce": False}
_FP32 = {"compute_dtype": torch.float32, "fp32_reduce": False}


class _FakeMesh:
    """The wrap reads a mesh's size, dim names and (TP) its ``dp`` slice."""

    def __init__(self, size: int, dim_names: tuple[str, ...] = ()):
        self._size = size
        self.mesh_dim_names = dim_names

    def size(self) -> int:
        return self._size

    def __getitem__(self, dim: str) -> "_FakeMesh":
        assert dim in self.mesh_dim_names, f"sliced {dim!r} out of a mesh with {self.mesh_dim_names}"
        return _FakeMesh(self._size, ())


_DP_TP_MESH = _FakeMesh(4, (MeshDim.DP, MeshDim.TP))


class _CausalLmLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        self.lm_head = nn.Linear(8, 4)


@contextlib.contextmanager
def _fake_world(world_size: int = 8):
    """Logs as rank 0, where both paths log."""
    with (
        patch(f"{fsdp.__name__}.dist.is_initialized", return_value=True),
        patch(f"{fsdp.__name__}.dist.get_world_size", return_value=world_size),
        patch.object(fsdp, "is_global_main_process", return_value=True),
    ):
        yield


def _record_wrap(entry, *args, **kwargs) -> list[tuple[str, int, torch.dtype | None, bool]]:
    """Modules recorded by name (a unit's joined by ``+``) so runs over two instances stay comparable."""
    model = _CausalLmLike()
    names = {id(module): name or "<root>" for name, module in model.named_modules()}
    calls: list[tuple[str, int, torch.dtype | None, bool]] = []

    def fake_fully_shard(module, **shard_kwargs):
        policy = shard_kwargs["mp_policy"]
        calls.append(
            (
                "+".join(names[id(each)] for each in module) if isinstance(module, list) else names[id(module)],
                shard_kwargs["mesh"].size(),
                None if policy is None else policy.param_dtype,
                bool(shard_kwargs["reshard_after_forward"]),
            )
        )

    with (
        _fake_world(),
        patch.object(fsdp, "fully_shard", side_effect=fake_fully_shard),
        patch.object(fsdp, "_install_unit_hooks"),
        patch.object(fsdp, "build_dp_mesh", return_value=_FakeMesh(4)),
    ):
        entry(model, *args, **kwargs)
    return calls


def test_the_tp_entry_point_takes_its_dp_width_from_the_caller_not_the_world():
    """``world // tp_size`` matches only while PP+TP, TP+CP and TP+ETP are rejected; afterwards grads reduce across
    unrelated data."""
    model = _CausalLmLike()

    # World 8, tp 2 derives 4; a cp2/pp2 composition makes the true DP 2.
    with _fake_world(world_size=8), patch.object(fsdp, "_apply_fsdp2") as applied:
        wrapped = fsdp.apply_fsdp2_for_tp(model, _DP_TP_MESH, dp_size=2, **_BF16)
    assert wrapped and "DP×TP 2×" in applied.call_args.kwargs["topology"], (
        f"the wrap reported {applied.call_args.kwargs['topology']}, not the caller's dp_size"
    )

    with _fake_world(world_size=8), patch.object(fsdp, "_apply_fsdp2") as never:
        assert fsdp.apply_fsdp2_for_tp(model, _DP_TP_MESH, dp_size=1, **_BF16) is False
    never.assert_not_called()


def test_both_entry_points_wrap_the_same_modules_the_same_way():
    dp_calls = _record_wrap(fsdp.apply_fsdp2_for_dp, dp_size=4, **_BF16)
    tp_calls = _record_wrap(fsdp.apply_fsdp2_for_tp, _DP_TP_MESH, dp_size=4, **_BF16)

    assert dp_calls, "the DP entry point wrapped nothing"
    assert dp_calls == tp_calls, f"the two entry points diverged:\n  dp={dp_calls}\n  tp={tp_calls}"
    assert [name for name, *_ in dp_calls] == ["model.layers.0+model.layers.1", "model", "<root>"], (
        f"one two-layer unit plus backbone plus root is the wrap both paths owe: {dp_calls}"
    )
    assert {dtype for _, _, dtype, _ in dp_calls} == {torch.bfloat16}


def test_both_entry_points_carry_the_reshard_and_precision_knobs_into_the_wrap():
    """Anti-vacuity: the recorded fields must track the arguments."""
    for entry, args in ((fsdp.apply_fsdp2_for_dp, ()), (fsdp.apply_fsdp2_for_tp, (_DP_TP_MESH,))):
        full_shard = _record_wrap(entry, *args, dp_size=4, **_BF16, reshard_after_forward=True)
        assert all(reshard for *_, reshard in full_shard), f"reshard_after_forward never reached fully_shard: {entry}"
        assert [name for name, *_ in full_shard] == ["model.layers.0", "model.layers.1", "model", "<root>"], (
            f"ZeRO-3 keeps one decoder layer per group: {entry} {full_shard}"
        )
        fp32 = _record_wrap(entry, *args, dp_size=4, **_FP32)
        assert {dtype for _, _, dtype, _ in fp32} == {torch.float32}, (
            f"the compute dtype never reached the policy: {entry}"
        )


class _PureTPHost(ModeSetupMixin):
    """The state the mode ladder's TP branch reads."""

    def __init__(self, model, *, dp_size: int):
        self.model = model
        self.args = SimpleNamespace(bf16=True)
        self.parallelism_config = SimpleNamespace(
            is_pp_mode=False,
            has_ep_group=False,
            is_cp_mode=False,
            is_tp_mode=True,
            tp_size=2,
            data_parallel_size=dp_size,
            fsdp_reduces_in_fp32=True,
            fsdp_reshard_after_forward=False,
        )

    def _top_level_model(self):
        return self.model

    def _install_tp_clip(self) -> None:
        pass


def test_a_tp_run_wraps_over_the_mesh_the_loader_attached():
    """A second mesh would shard the params on one group and reduce their grads over another."""
    model = _CausalLmLike()
    model._device_mesh = _DP_TP_MESH
    host = _PureTPHost(model, dp_size=4)

    with patch.object(mode_setup, "apply_fsdp2_for_tp", return_value=True) as wrap:
        host._wrap_for_parallel_mode()

    assert wrap.call_args.args == (model, _DP_TP_MESH)
    assert wrap.call_args.kwargs["fp32_reduce"] is True, "the wrap dropped fsdp_reduces_in_fp32"
    assert host._device_mesh is _DP_TP_MESH
    assert host._fsdp_wrapped is True


@pytest.mark.parametrize("dp_size", [1, 4], ids=["pure_tp", "tp_dp"])
def test_a_tp_run_refuses_a_model_that_carries_no_loader_mesh(dp_size):
    """Nothing was sharded on any TP group: a mesh built here would train full replicas with the TP reduces silent."""
    with (
        _fake_world(),
        patch.object(fsdp, "fully_shard", side_effect=AssertionError("wrapped")),
        pytest.raises(RuntimeError, match="no device mesh reached the trainer"),
    ):
        _PureTPHost(_CausalLmLike(), dp_size=dp_size)._wrap_for_parallel_mode()


def test_the_tp_entry_point_refuses_a_mesh_with_no_dp_dimension():
    """It would train with no DP gradient sync and report success."""
    with (
        _fake_world(),
        patch.object(fsdp, "fully_shard"),
        pytest.raises(RuntimeError, match="requires a device mesh with a 'dp' dimension"),
    ):
        fsdp.apply_fsdp2_for_tp(_CausalLmLike(), _FakeMesh(4, (MeshDim.TP,)), dp_size=4, **_BF16)


class _Fp32ResidualLayer(nn.Linear):
    """Declares an fp32 activation across layer boundaries (Zaya's residual)."""

    _fp32_interlayer_residual = True


class _EPTPHost(ModeSetupMixin):
    """The config, the args, the model and the FSDP exclusions."""

    def __init__(self, model, excluded):
        self.model = model
        self._device_mesh = model._device_mesh
        self.args = SimpleNamespace(bf16=True)
        self.parallelism_config = SimpleNamespace(
            tp_size=2,
            data_parallel_size=4,
            fsdp_reduces_in_fp32=False,
            fsdp_reshard_after_forward=False,
        )
        self._excluded = excluded

    def _fsdp_exclusions(self):
        return SimpleNamespace(params=self._excluded)


def _ep_tp_model() -> _CausalLmLike:
    model = _CausalLmLike()
    model.model.layers[0] = _Fp32ResidualLayer(8, 8)
    model._device_mesh = _DP_TP_MESH
    return model


def test_the_ep_tp_wrap_runs_the_shared_body_with_its_ep_modules_left_out():
    """The shared body reads the model's fp32-residual declaration into the cast policy."""
    model = _ep_tp_model()
    expert = model.model.layers[1].weight
    calls = []

    def fake_fully_shard(module, **kwargs):
        modules = module if isinstance(module, list) else [module]
        calls.append((modules, kwargs["mp_policy"], kwargs.get("ignored_params")))

    with (
        _fake_world(),
        patch.object(fsdp, "fully_shard", side_effect=fake_fully_shard),
        patch.object(fsdp, "_install_unit_hooks"),
    ):
        _EPTPHost(model, [expert])._apply_ep_tp_fsdp2()

    assert calls, "the EP+TP wrap sharded nothing"
    assert {policy.cast_forward_inputs for _, policy, _ in calls} == {False}, (
        "the fp32 inter-layer residual must keep its dtype across the layer boundary"
    )
    layer_ignored = next(
        ignored for modules, _, ignored in calls if any(module is model.model.layers[1] for module in modules)
    )
    assert layer_ignored is not None and expert in layer_ignored, "the EP module was sharded with the layer"


def test_the_ep_tp_wrap_refuses_a_model_spread_over_two_devices():
    """The check is rank-local and must run before the wrap's collective mesh work."""
    model = _ep_tp_model()
    model.lm_head.to("meta")

    with _fake_world(), patch.object(fsdp, "fully_shard"), pytest.raises(RuntimeError, match="single-device"):
        _EPTPHost(model, [])._apply_ep_tp_fsdp2()


def test_neither_entry_point_wraps_a_single_rank_dp_group():
    with _fake_world(world_size=2), patch.object(fsdp, "fully_shard", side_effect=AssertionError("wrapped")):
        assert fsdp.apply_fsdp2_for_dp(_CausalLmLike(), dp_size=1, **_BF16) is False
        assert fsdp.apply_fsdp2_for_tp(_CausalLmLike(), _DP_TP_MESH, dp_size=1, **_BF16) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
