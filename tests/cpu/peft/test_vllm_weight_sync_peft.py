#!/usr/bin/env python
"""PEFT weight sync pushes the folded adapter and leaves the trainer's own weights untouched.
An in-place merge with PEFT's bf16 unmerge misses ``w`` by a rounding step, so the base would walk over the syncs."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, init_device_mesh

from halo.trainers.grpo.rollout.weight_push import sync_weights_to_client
from tests.common.adapters import randomize_adapters
from tests.common.gloo import run_gloo_ranks
from tests.common.weight_sync import (
    RecordingSender,
    as_pushed,
    folded_in_place,
    local_parameters,
    lora_bases_and_merges,
    merged_by_peft,
    moved_parameters,
)

PUSHES = 6
_DIM = 64
FSDP_WORLD_SIZE = 2


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(_DIM, _DIM, bias=False)
        self.o_proj = nn.Linear(_DIM, _DIM, bias=False)

    def forward(self, x):
        return self.o_proj(self.q_proj(x))


class _Policy(nn.Module):
    """LoRA-targetable projections in per-layer blocks (the FSDP2 units), and a head no adapter wraps."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(_Block() for _ in range(2))
        self.lm_head = nn.Linear(_DIM, _DIM, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


def _bf16_lora_policy(use_dora: bool = False) -> nn.Module:
    """The same seeded bf16 policy on every rank; ``lora_B`` starts at zero, so it is drawn for a ~10% delta."""
    torch.manual_seed(0)
    model = get_peft_model(
        _Policy().to(torch.bfloat16),
        LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "o_proj"], use_dora=use_dora),
    )
    randomize_adapters(model, std=0.05, lora_b_only=True)
    return model


def _push(model: nn.Module, forwarding: bool = True) -> dict[str, torch.Tensor]:
    """One sync through the trainers' per-sync entry; what the forwarding rank handed its client."""
    sender = RecordingSender(keep_values=True) if forwarding else None
    sync_weights_to_client(model, sender, is_main=forwarding, is_tp_main=True)
    return {param.name: param.value for param in sender.params} if sender else {}


def test_repeated_pushes_leave_the_frozen_base_bit_identical():
    model = _bf16_lora_policy()
    before = local_parameters(model)
    bases, merges = lora_bases_and_merges(model)
    expected = as_pushed(model, merges)

    pushes = [_push(model) for _ in range(PUSHES)]

    moved = moved_parameters(before, local_parameters(model))
    assert not moved, f"{len(moved)} parameters moved across {PUSHES} pushes, e.g. {moved[:3]}"
    assert all(not torch.equal(merges[key], bases[key]) for key in bases), "premise: every fold moves its weight"
    wrong = sorted({key for push in pushes for key in expected if not torch.equal(push[key], expected[key])})
    assert len(expected) == len(merges) and not wrong, f"pushed weights are not base + delta: {wrong}"

    for _ in range(PUSHES):
        with folded_in_place(model):
            _push(model)
    assert moved_parameters(before, local_parameters(model)), (
        "premise: a bf16 unmerge alone moves the base of this model"
    )


def _fsdp2_push_worker(rank: int, use_dora: bool) -> None:
    """One rank of a real 2-way ``fully_shard``; the fold and gather take the sharded path.
    Every collective runs before the first assertion, so a failing rank never strands its peer inside one."""
    model = _bf16_lora_policy(use_dora)
    bases, merges = as_pushed(model, local_parameters(model)), as_pushed(model, merged_by_peft(model))
    mesh = init_device_mesh("cpu", (FSDP_WORLD_SIZE,))
    for block in model.base_model.model.layers:
        fully_shard(block, mesh=mesh, reshard_after_forward=False)
    fully_shard(model, mesh=mesh, reshard_after_forward=False)
    lora_params = {
        name: param for name, param in model.named_parameters() if ".base_layer." in name or ".lora_" in name
    }
    sharded = bool(lora_params) and all(isinstance(param.data, DTensor) for param in lora_params.values())
    before = local_parameters(model)

    pushes = []
    for _ in range(PUSHES):
        # A forward leaves the unsharded params registered, as each training step before a sync does.
        model(torch.ones(2, _DIM, dtype=torch.bfloat16))
        pushes.append(_push(model, forwarding=rank == 0))
    after = local_parameters(model)

    for _ in range(PUSHES):
        with folded_in_place(model):
            _push(model, forwarding=rank == 0)
    drifted = moved_parameters(before, local_parameters(model))

    assert sharded, "premise: the LoRA'd weights and adapters are FSDP2 DTensor shards"
    moved = moved_parameters(before, after)
    assert not moved, f"rank {rank}: {len(moved)} local shards moved across {PUSHES} pushes, e.g. {moved[:3]}"
    if rank == 0:
        assert len(pushes) == PUSHES and all(push.keys() == merges.keys() for push in pushes)
        wrong = sorted({key for push in pushes for key in merges if not torch.equal(push[key], merges[key])})
        assert not wrong, f"pushed weights are not PEFT's merge of the unsharded model: {wrong}"
        assert any(not torch.equal(merges[key], bases[key]) for key in bases), "premise: the merge moves weights"
    # PEFT's in-place DoRA merge replaces ``.data``, which does not reach an FSDP2 shard; only plain LoRA moves here.
    assert use_dora or drifted, f"premise: rank {rank}'s shards move under a bf16 unmerge alone"


@pytest.mark.parametrize("use_dora", [False, True], ids=["lora", "dora"])
def test_fsdp2_sharded_pushes_leave_the_frozen_base_bit_identical(use_dora):
    run_gloo_ranks(_fsdp2_push_worker, FSDP_WORLD_SIZE, use_dora)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
