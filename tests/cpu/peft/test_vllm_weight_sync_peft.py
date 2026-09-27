#!/usr/bin/env python
"""PEFT weight sync must push the *folded* adapter, and leave the trainer's own weights untouched.

``gather_and_send_weights`` folds the LoRA delta into each base weight out of place and forwards the
result under the base-model name; the vendored client snapshots each ``update_named_param`` payload
and flushes the snapshots later via ``reset_prefix_cache``. A recording client that stages exactly as
the real one does pins that the flushed weight is ``W + B @ A``, not the base ``W``. CPU: neither
property depends on the device or the real NCCL transport.

The frozen base must come out of every sync as it went in. A sync that merged in place and relied on
PEFT's bf16 unmerge would not: ``(w + d) - d`` misses ``w`` by a rounding step wherever the two
roundings do not cancel, and the sync repeats every few steps for the whole run, so the base would
walk away from the one the run loaded and each push would serve a different base. Repeated pushes are
pinned on plain tensors and on a real 2-rank ``fully_shard`` over gloo, where the fold and the gather
run on DTensor shards: base and adapters bit-identical after every push, every push bit-identical to
the others and to ``w + delta`` as PEFT's merge adds it. Each also pushes inside an in-place PEFT
merge/unmerge (``folded_in_place``) and requires the base to move, so a model on which the roundings
happen to cancel cannot pass vacuously.

Run: ``python tests/cpu/peft/test_vllm_weight_sync_peft.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, init_device_mesh

from src.distributed.nccl.clients.base import snapshot_param
from src.trainers.grpo.rollout.weight_sync import gather_and_send_weights, sync_weights_to_client
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


class _TinyModel(nn.Module):
    """Minimal module with a single LoRA-targetable linear (no download)."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):  # pragma: no cover - never called
        return self.proj(x)


class _RecordingClient:
    """Stand-in for the vendored NCCL client's snapshot-then-late-flush semantics."""

    def __init__(self):
        self._buffer: list[tuple[str, torch.Tensor]] = []
        self.flushed: dict[str, torch.Tensor] = {}

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        # The client's own staging: a snapshot taken when the tensor is forwarded, never a reference.
        self._buffer.append((name, snapshot_param(weights, None)))

    def reset_prefix_cache(self) -> None:
        # The broadcast happens here, after gather_and_send_weights has returned.
        self.flushed = {name: tensor.detach().clone() for name, tensor in self._buffer}
        self._buffer = []


def _build_lora_model() -> nn.Module:
    torch.manual_seed(0)
    base = _TinyModel()
    cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["proj"], task_type=None)
    model = get_peft_model(base, cfg)
    # PEFT zero-inits lora_B, which would make merged == base and hide the bug. Give the adapter a
    # non-trivial delta so W + B@A differs from W.
    for name, param in model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                param.copy_(torch.randn_like(param))
    return model


def test_peft_sync_broadcasts_merged_not_base():
    model = _build_lora_model()
    merged = as_pushed(model, merged_by_peft(model))
    assert "proj.weight" in merged

    client = _RecordingClient()
    peft = gather_and_send_weights(model, client)
    # The caller flushes after the gather returns, as the real sync does.
    client.reset_prefix_cache()

    assert peft is True
    assert "proj.weight" in client.flushed, "base weight was never forwarded"
    # No adapter-only params should leak to vLLM.
    assert not any("lora_" in k for k in client.flushed)

    base_weight = model.base_model.model.proj.base_layer.weight.detach()
    sent = client.flushed["proj.weight"]

    # The flushed weight is the merged adapter.
    assert torch.allclose(sent, merged["proj.weight"], atol=1e-6), (
        "vLLM received weights that do not match the merged adapter (W + B@A)"
    )
    # And it must NOT be the reverted base (the adapter delta is non-trivial by construction).
    assert not torch.allclose(sent, base_weight, atol=1e-6), (
        "vLLM received the base weights — the adapter was never folded in"
    )


# --- repeated pushes leave the run's own weights untouched ---------------------------------------


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
    """The same seeded bf16 policy on every rank, its adapters at a trained size: ``lora_B`` starts at
    zero, which folds nothing, so it is drawn to make the delta ~10% of the base weights."""
    torch.manual_seed(0)
    model = get_peft_model(
        _Policy().to(torch.bfloat16),
        LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "o_proj"], use_dora=use_dora),
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".lora_B." in name:
                param.normal_(std=0.05)
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
    """One rank of a real 2-way ``fully_shard``: the LoRA'd weights and their adapters are DTensor
    shards, so the fold and the gather take the sharded path, and every push must equal PEFT's merge
    of the unsharded model. Every collective runs before the first assertion, so a failing rank never
    strands its peer inside one."""
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
    # PEFT's in-place DoRA merge replaces ``.data``, which does not reach an FSDP2 shard, so only plain
    # LoRA gives this control something to move.
    assert use_dora or drifted, f"premise: rank {rank}'s shards move under a bf16 unmerge alone"


@pytest.mark.parametrize("use_dora", [False, True], ids=["lora", "dora"])
def test_fsdp2_sharded_pushes_leave_the_frozen_base_bit_identical(use_dora):
    run_gloo_ranks(_fsdp2_push_worker, FSDP_WORLD_SIZE, use_dora)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
