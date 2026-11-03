#!/usr/bin/env python
"""A hand-built collective tensor belongs on the backend's device: gloo moves host memory, so a CUDA tensor is a
staging copy at best."""

import contextlib
from unittest.mock import patch

import pytest
import torch

from halo.runtime.collectives import collective_device
from halo.runtime.consensus import agree_across_ranks, rank_consensus
from halo.runtime.load_coordination import _log_global_load_duration_seconds

_MOD = "halo.runtime.collectives"


@contextlib.contextmanager
def _world(backend: str):
    with (
        patch(f"{_MOD}.dist.is_available", return_value=True),
        patch(f"{_MOD}.dist.is_initialized", return_value=True),
        patch(f"{_MOD}.dist.get_world_size", return_value=2),
        patch(f"{_MOD}.dist.get_rank", return_value=0),
        patch(f"{_MOD}.dist.get_backend", return_value=backend),
    ):
        yield


def test_a_gloo_group_collects_on_cpu():
    with _world("gloo"):
        assert collective_device().type == "cpu", "a gloo collective must not be handed a CUDA tensor"


def test_an_nccl_group_collects_on_the_cuda_device_when_there_is_one():
    """The NCCL path keeps its device-side tensor."""
    with _world("nccl"):
        assert collective_device().type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_no_group_at_all_collects_on_cpu():
    with patch(f"{_MOD}.dist.is_initialized", return_value=False):
        assert collective_device().type == "cpu"


def test_a_gloo_subgroup_under_an_nccl_world_collects_on_cpu():
    """The device follows the group the collective runs on."""
    gloo_subgroup = object()
    with (
        _world("nccl"),
        patch(f"{_MOD}.dist.get_backend", side_effect=lambda group=None: "gloo" if group is gloo_subgroup else "nccl"),
        patch(f"{_MOD}.torch.cuda.is_available", return_value=True),
        patch(f"{_MOD}.current_device", return_value=torch.device("cuda", 0)),
    ):
        assert collective_device().type == "cuda"
        assert collective_device(gloo_subgroup).type == "cpu"


def test_a_group_agreement_places_its_tensor_for_that_group():
    asked: list[object] = []
    subgroup = object()

    def device_for(group=None):
        asked.append(group)
        return torch.device("cpu")

    with (
        _world("gloo"),
        patch("halo.runtime.consensus.collective_device", side_effect=device_for),
        patch(f"{_MOD}.dist.all_reduce"),
    ):
        agree_across_ranks("value", group=subgroup)

    assert asked == [subgroup]


def test_the_collective_call_sites_place_their_tensors_there():
    devices: list[torch.device] = []

    with (
        _world("gloo"),
        patch(f"{_MOD}.dist.all_reduce", side_effect=lambda tensor, **kw: devices.append(tensor.device)),
    ):
        rank_consensus(True)
        _log_global_load_duration_seconds(tag="probe", method="test", t_start_wall=0.0, t_end_wall=1.0)

    assert devices, "no collective ran"
    assert all(device.type == "cpu" for device in devices), f"CUDA tensors on a gloo group: {devices}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
