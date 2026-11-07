#!/usr/bin/env python
"""The sync's expert gather delivers each layer to the forwarding rank alone, on real gloo groups: the bytes equal the
all-gather's (zeros keep their sign), the ranks outside the forwarder's EP group join no collective, and issuing a layer
ahead leaves the order of collectives on every rank as it was."""

from __future__ import annotations

import datetime
import json
import os
import types

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from halo.distributed.expert_parallel.moe_layer import gather as gather_module
from halo.distributed.expert_parallel.moe_layer.gather import EPGatherMixin
from halo.models.moe.expert_layout import ExpertLayout, ExpertLayoutKind
from halo.models.moe.family_specs import MoEFamilySpec
from halo.runtime import collectives
from halo.runtime.consensus import DeferredRankFailure
from halo.trainers.grpo.rollout.weight_push import _send_ep_expert_weights
from tests.common.gloo import new_groups, run_gloo_ranks

PG_TIMEOUT = datetime.timedelta(seconds=90)
LAYERS = 3
# Below one tensor's bytes, so a gather takes several rounds and a window edge falls inside a rank's span.
WINDOW_BYTES = 40
_SPEC = MoEFamilySpec(model_types=("gather_to_rank",), expert_layout=ExpertLayout(ExpertLayoutKind.FUSED))


def _fill(rank: int, layer: int, shape: tuple[int, ...]) -> torch.Tensor:
    """Distinct per rank and layer, with a negative zero and a NaN payload a float sum would change."""
    values = (
        torch.arange(int(torch.Size(shape).numel()), dtype=torch.float32).reshape(shape) + 100 * rank + 1000 * layer
    )
    values.view(-1)[0] = -0.0
    values.view(-1)[1] = float("nan")
    return values.to(torch.bfloat16)


class _Layer(EPGatherMixin, nn.Module):
    has_expert_lora = False

    def __init__(self, rank: int, index: int, process_group, ep_group, ep_size: int):
        nn.Module.__init__(self)
        self.FAMILY = _SPEC
        self.ep_config = types.SimpleNamespace(
            process_group=process_group, dispatch_ep_group=ep_group, ep_size=ep_size
        )
        self.expert_tp_group, self.expert_tp_size = None, 1
        self.expert_lora_scaling, self._expert_lora_attrs = 1.0, frozenset()
        self.gate_up_proj = nn.Parameter(_fill(rank, index, (2, 3, 5)), requires_grad=False)
        self.down_proj = nn.Parameter(_fill(rank, index, (2, 5, 3)), requires_grad=False)

    def _base_expert_named_params(self):
        return [("gate_up_proj", self.gate_up_proj), ("down_proj", self.down_proj)]

    expert_named_params = _base_expert_named_params


class _Forwarder:
    def __init__(self):
        self.sent: dict[str, torch.Tensor] = {}

    def send(self, name: str, tensor: torch.Tensor, *, bounded: bool = True) -> None:
        self.sent[name] = tensor.clone()

    def flush(self) -> None:
        pass


def _worker(rank: int, out: str, group_lists: list[list[int]], ahead: int) -> None:
    os.environ["HALO_WEIGHT_SYNC_GATHER_AHEAD"] = str(ahead)
    collectives._GATHER_WINDOW_BYTES = WINDOW_BYTES
    real_hub = gather_module.to_hub_expert_tensors
    gather_module.to_hub_expert_tensors = lambda spec, tensors, *, device: real_hub(spec, tensors, device="cpu")
    trace: list[str] = []
    real_reduce, real_all_gather = dist.reduce, dist.all_gather_into_tensor

    def traced_reduce(tensor, dst, group=None, **kwargs):
        trace.append(f"reduce:{tensor.numel()}")
        return real_reduce(tensor, dst=dst, group=group, **kwargs)

    def traced_all_gather(output, tensor, group=None, **kwargs):
        trace.append(f"all_gather:{tensor.numel()}")
        return real_all_gather(output, tensor, group=group, **kwargs)

    dist.reduce, dist.all_gather_into_tensor = traced_reduce, traced_all_gather
    group = new_groups(rank, group_lists, PG_TIMEOUT)
    ep_size = len(next(ranks for ranks in group_lists if rank in ranks))
    layers = {f"layers.{i}": _Layer(rank, i, group, group, ep_size) for i in range(LAYERS)}
    forwarder = _Forwarder() if rank == 0 else None
    guard = DeferredRankFailure("weight-sync push")
    _send_ep_expert_weights(layers, forwarder, guard)
    guard.reject()
    delivered_trace = list(trace)
    # The old path on the same layers, for the byte comparison; every rank of a group enters it.
    reference = {}
    if rank == 0:
        reference = {
            f"layers.{i}.{name}": tensor
            for i, layer in enumerate(layers.values())
            for name, tensor in layer.gather_expert_state_dict(device="cpu").items()
        }
    else:
        for layer in layers.values():
            layer.gather_expert_state_dict(device="cpu", retain=False)
    if rank == 0:
        assert set(forwarder.sent) == set(reference)
        for name, tensor in reference.items():
            assert forwarder.sent[name].view(torch.int16).equal(tensor.view(torch.int16)), name
    with open(os.path.join(out, f"trace_{rank}.json"), "w") as fh:
        json.dump(delivered_trace, fh)


def _traces(tmp_path, world: int, group_lists: list[list[int]], ahead: int) -> list[list[str]]:
    run_gloo_ranks(_worker, world, str(tmp_path), group_lists, ahead, pg_timeout=PG_TIMEOUT)
    return [json.loads((tmp_path / f"trace_{rank}.json").read_text()) for rank in range(world)]


@pytest.mark.parametrize("ahead", [0, 1])
def test_the_forwarder_receives_the_all_gather_bytes_and_every_peer_runs_the_same_collectives(tmp_path, ahead):
    traces = _traces(tmp_path, 4, [[0, 1, 2, 3]], ahead)

    assert traces[0] == traces[1] == traces[2] == traces[3]
    assert traces[0], "no collective ran"
    assert not any(entry.startswith("all_gather") for entry in traces[0]), "the sync gathered to every rank"


def test_issuing_a_layer_ahead_keeps_the_collective_order(tmp_path):
    in_turn = _traces(tmp_path / "in_turn", 2, [[0, 1]], 0)
    ahead = _traces(tmp_path / "ahead", 2, [[0, 1]], 1)

    assert in_turn == ahead


def test_ranks_outside_the_forwarders_group_join_no_collective(tmp_path):
    traces = _traces(tmp_path, 4, [[0, 1], [2, 3]], 1)

    assert traces[0] == traces[1] and traces[0]
    assert traces[2] == [] and traces[3] == []


@pytest.fixture(autouse=True)
def _tmp_dirs(tmp_path):
    for name in ("in_turn", "ahead"):
        (tmp_path / name).mkdir(exist_ok=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
