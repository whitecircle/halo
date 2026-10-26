#!/usr/bin/env python
"""A PEFT adapter resume is collective: same file, same key order, on every rank.

``_load_peft_adapter_state`` calls ``distribute_tensor`` per key once the adapter params are FSDP2
DTensors — a mesh collective, so the ranks must issue them in the SAME order or they deadlock naming
``distribute_tensor`` rather than the adapter. A dict's iteration order is its file's, and on a
non-shared filesystem the ranks need not even read the same file: ``consensus_read`` accepts an
ordered list of candidates (safetensors, then PEFT's ``.bin`` fallback), which is only safe while the
CHOICE is a world fact rather than each rank's first local hit.

Both halves are pinned here: the load order, and the pick.

    python tests/cpu/checkpoint/test_adapter_resume_collective_order.py
"""

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

import src.checkpoint.parallel.adapters as peft_mod
import src.checkpoint.parallel.reading as coordination
from src.checkpoint.adapters import adapter_weight_paths
from src.checkpoint.format import ADAPTER_WEIGHT_NAMES
from src.checkpoint.parallel.adapters import _load_peft_adapter_state, restore_adapters
from src.checkpoint.parallel.reading import consensus_read
from src.runtime import dtensor
from src.runtime.consensus import RankConsensus
from tests.common.gloo import run_gloo_ranks

# Deliberately NOT sorted: a dict preserves insertion order, which is the adapter file's order.
UNSORTED_KEYS = (
    "base_model.model.layers.2.self_attn.q_proj.lora_A.weight",
    "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
    "base_model.model.layers.1.self_attn.q_proj.lora_A.weight",
)


@pytest.fixture
def single_rank_gloo(tmp_path):
    """A 1-rank gloo group: ``distribute_tensor`` needs a real mesh, and the DTensor branch under
    test is chosen by ``isinstance(param.data, DTensor)``, which no stub satisfies."""
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        yield
    finally:
        dist.destroy_process_group()


class _LoraModel(nn.Module):
    """A model whose LoRA params are DTensors, keyed exactly as the adapter file spells them."""

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
    """The mesh collectives must be issued in a rank-independent order.

    Each saved tensor carries its sorted position as its value, so the recorded sequence names the
    order ``distribute_tensor`` ran in — file order would replay the dict, which differs per rank the
    moment two ranks read different adapter files (or PEFT writes them in another order).
    """
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

    assert _load_peft_adapter_state(model, state) == []
    assert issued == sorted(issued), (
        f"distribute_tensor ran in adapter-file order {issued}, not sorted key order: two ranks "
        f"reading differently-ordered adapter files would deadlock in the mesh collective."
    )


def _run_consensus_read(tmp_path, monkeypatch, *, local: dict[str, bool], elsewhere: dict[str, bool]):
    """Drive ``consensus_read`` over the adapter candidates with a scripted peer view.

    ``rank_consensus`` is replaced by what the world WOULD answer given ``elsewhere``: the real
    all-reduce needs peers, and the decision under test is what this rank does with its verdicts.
    """
    for name, present in local.items():
        if present:
            (tmp_path / name).write_text(name)
    verdicts = iter(
        [
            RankConsensus(local[name] and elsewhere[name], local[name] or elsewhere[name])
            for name in ADAPTER_WEIGHT_NAMES
        ]
    )
    # The presence probes are scripted in candidate order; the read-success join that follows them
    # takes this rank's own verdict (every rank read what it picked).
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
    """This rank holds both; a peer holds only the ``.bin``. Picking the first LOCAL hit gives the two
    ranks different files — same tensors, different key order — and the DTensor load above then
    issues its mesh collectives in two different orders."""
    safetensors, adapter_bin = ADAPTER_WEIGHT_NAMES
    _value, path = _run_consensus_read(
        tmp_path,
        monkeypatch,
        local={safetensors: True, adapter_bin: True},
        elsewhere={safetensors: False, adapter_bin: True},
    )
    assert os.path.basename(path) == adapter_bin


def test_disjoint_adapter_files_across_ranks_raise(tmp_path, monkeypatch):
    """No candidate is on every rank: a torn/partial save, not a file to pick between."""
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
    """Rank 1 runs at twice the LoRA alpha the adapter was saved with; every rank records what it raised
    and how often it read the adapter file."""
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
    """The read and its scaling refusal are one collective step: the rank whose live scaling differs
    makes every rank raise the same text, labelled by the read it refused, after one adapter read."""
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
