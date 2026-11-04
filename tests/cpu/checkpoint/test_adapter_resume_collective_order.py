#!/usr/bin/env python
"""A PEFT adapter resume is collective: same file, same key order, on every rank. ``distribute_tensor`` per key is a
mesh collective, so differing order deadlocks naming it rather than the adapter."""

from __future__ import annotations

import json
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

import halo.checkpoint.parallel.adapters as peft_mod
import halo.checkpoint.parallel.reading as coordination
from halo.checkpoint.adapters import adapter_weight_paths
from halo.checkpoint.format import ADAPTER_WEIGHT_NAMES
from halo.checkpoint.parallel.adapters import _load_peft_adapter_state, restore_adapters
from halo.checkpoint.parallel.reading import consensus_read
from halo.runtime import dtensor
from halo.runtime.consensus import RankConsensus
from tests.common.gloo import run_gloo_ranks

# Deliberately unsorted: a dict preserves the adapter file's order.
UNSORTED_KEYS = (
    "base_model.model.layers.2.self_attn.q_proj.lora_A.weight",
    "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
    "base_model.model.layers.1.self_attn.q_proj.lora_A.weight",
)


@pytest.fixture
def single_rank_gloo(tmp_path):
    """The DTensor branch keys on ``isinstance(param.data, DTensor)``, which no stub satisfies."""
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        yield
    finally:
        dist.destroy_process_group()


class _LoraModel(nn.Module):
    def __init__(self, mesh):
        super().__init__()
        self.params = nn.ParameterDict(
            {
                key.replace(".", "|"): nn.Parameter(distribute_tensor(torch.zeros(2, 2), mesh, [Shard(0)]))
                for key in UNSORTED_KEYS
            }
        )

    def named_parameters(self, *_args, **_kwargs):
        for key, param in self.params.items():
            yield key.replace("|", "."), param


def test_dtensor_adapter_tensors_are_distributed_in_sorted_key_order(single_rank_gloo, monkeypatch):
    """Each tensor's value is its sorted position, so the recorded sequence names the order ``distribute_tensor``
    ran in."""
    mesh = init_device_mesh("cpu", (1,))
    model = _LoraModel(mesh)
    order = {key: index for index, key in enumerate(sorted(UNSORTED_KEYS))}
    state = {key: torch.full((2, 2), float(order[key])) for key in UNSORTED_KEYS}

    issued: list[float] = []
    real_distribute = dtensor.distribute_tensor
    monkeypatch.setattr(
        dtensor,
        "distribute_tensor",
        lambda tensor, device_mesh, placements, **kwargs: (
            issued.append(tensor.flatten()[0].item()),
            real_distribute(tensor, device_mesh, placements, **kwargs),
        )[1],
    )

    _load_peft_adapter_state(dict(model.named_parameters()), state)
    assert issued == sorted(issued), (
        f"distribute_tensor ran in adapter-file order {issued}, not sorted key order: two ranks "
        f"reading differently-ordered adapter files would deadlock in the mesh collective."
    )


def _run_consensus_read(tmp_path, monkeypatch, *, local: dict[str, bool], elsewhere: dict[str, bool]):
    """``rank_consensus`` answers from ``elsewhere``: the real all-reduce needs peers."""
    for name, present in local.items():
        if present:
            (tmp_path / name).write_text(name)
    verdicts = iter(
        [
            RankConsensus(local[name] and elsewhere[name], local[name] or elsewhere[name])
            for name in ADAPTER_WEIGHT_NAMES
        ]
    )
    # Presence probes are scripted in candidate order; the read-success join takes this rank's own verdict.
    monkeypatch.setattr(
        coordination, "rank_consensus", lambda local_ok: next(verdicts, RankConsensus(local_ok, local_ok))
    )
    return consensus_read(
        adapter_weight_paths(str(tmp_path)),
        os.path.basename,
        what="Adapter checkpoint",
        checkpoint=str(tmp_path),
    )


def test_the_chosen_adapter_file_is_the_one_every_rank_has(tmp_path, monkeypatch):
    """The first local hit gives two ranks different files, and the mesh collectives two orders."""
    safetensors, adapter_bin = ADAPTER_WEIGHT_NAMES
    _value, path = _run_consensus_read(
        tmp_path,
        monkeypatch,
        local={safetensors: True, adapter_bin: True},
        elsewhere={safetensors: False, adapter_bin: True},
    )
    assert os.path.basename(path) == adapter_bin


def test_disjoint_adapter_files_across_ranks_raise(tmp_path, monkeypatch):
    """A torn or partial save, not a file to pick between."""
    safetensors, adapter_bin = ADAPTER_WEIGHT_NAMES
    with pytest.raises(RuntimeError, match="torn/partial"):
        _run_consensus_read(
            tmp_path,
            monkeypatch,
            local={safetensors: True, adapter_bin: False},
            elsewhere={safetensors: False, adapter_bin: True},
        )


class _LoraTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)


def _restore_under_another_scaling_on_rank1(rank: int, adapter_dir: str, outcome_dir: str) -> None:
    """Rank 1 runs at twice the saved LoRA alpha."""
    reads = []
    real_read = peft_mod.read_adapter_file
    peft_mod.read_adapter_file = lambda path: reads.append(path) or real_read(path)
    model = get_peft_model(_LoraTarget(), LoraConfig(r=2, lora_alpha=4 * (rank + 1), target_modules=["fc"]))
    try:
        restore_adapters(adapter_dir, model, is_cp_mode=False)
        outcome = "restored"
    except ValueError as e:
        outcome = str(e)
    with open(os.path.join(outcome_dir, f"rank{rank}.json"), "w") as fh:
        json.dump({"outcome": outcome, "reads": len(reads)}, fh)


def test_a_scaling_mismatch_on_one_rank_is_refused_on_every_rank_under_the_reads_label(tmp_path):
    """The read and its scaling refusal are one collective step."""
    adapter_dir, outcome_dir = tmp_path / "adapter", tmp_path / "outcomes"
    outcome_dir.mkdir()
    get_peft_model(_LoraTarget(), LoraConfig(r=2, lora_alpha=4, target_modules=["fc"])).save_pretrained(adapter_dir)

    run_gloo_ranks(_restore_under_another_scaling_on_rank1, 2, str(adapter_dir), str(outcome_dir))

    outcomes = [json.loads((outcome_dir / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert outcomes[0]["outcome"] == outcomes[1]["outcome"], outcomes
    assert outcomes[0]["outcome"].startswith("Adapter checkpoint failed on 1 of 2 rank(s) [1]"), outcomes[0]
    assert "another LoRA scaling" in outcomes[0]["outcome"]
    assert [outcome["reads"] for outcome in outcomes] == [1, 1], "the adapter file must be read once per rank"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
