#!/usr/bin/env python
"""A hand-built collective tensor belongs on the BACKEND's device, not on this rank's compute device.

``current_device()`` answers ``cuda:N`` whenever CUDA exists, which is right for model and metric
tensors and wrong for a collective on a **gloo** group: gloo moves host memory, so a CUDA tensor is a
staging copy at best and unsupported for the op at worst — and every CPU-test world on a GPU box is
exactly that shape. ``collective_device()`` is the seam the collectives use instead.

    python tests/cpu/parallelism/test_collective_device.py
"""

import contextlib
from unittest.mock import patch

import pytest
import torch

from src.distributed.runtime import collective_device, log_global_load_duration_seconds, rank_consensus

_MOD = "src.distributed.runtime"


@contextlib.contextmanager
def _world(backend: str):
    """A live 2-rank group on ``backend``, as far as the helpers can tell."""
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
    """Anti-over-correction: the NCCL path must keep its device-side tensor."""
    with _world("nccl"):
        assert collective_device().type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_no_group_at_all_collects_on_cpu():
    """Nothing initialized: there is no backend to ask, and no collective to run."""
    with patch(f"{_MOD}.dist.is_initialized", return_value=False):
        assert collective_device().type == "cpu"


def test_the_collective_call_sites_place_their_tensors_there():
    """The seam is worth nothing if a call site still builds its tensor on the compute device."""
    devices: list[torch.device] = []

    with (
        _world("gloo"),
        patch(f"{_MOD}.dist.all_reduce", side_effect=lambda tensor, **kw: devices.append(tensor.device)),
    ):
        rank_consensus(True)
        log_global_load_duration_seconds(tag="probe", method="test", t_start_wall=0.0, t_end_wall=1.0)

    assert devices, "no collective ran"
    assert all(device.type == "cpu" for device in devices), f"CUDA tensors on a gloo group: {devices}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
