#!/usr/bin/env python
"""The expert gather's merge_lora refusal must reject a real fold under expert TP. ``_send_ep_expert_weights`` asks
every layer for ``merge_lora=True``, so a guard on the flag alone rejects every sync at ``expert_tp_size > 1``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from halo.distributed.expert_parallel.moe_layer import gather as expert_gather
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from halo.models.moe.family_specs import MoEFamilySpec
from halo.trainers.grpo.rollout.weight_push import gather_and_send_weights

LAYER_PATH = "mlp"
# HIDDEN != 2 * INTERMEDIATE on purpose: at equal dims a stray transpose still satisfies every shape.
NUM_EXPERTS, HIDDEN, INTERMEDIATE = 2, 8, 6
# The grouped GEMM's 16-byte stride contract at the bf16 the adapters run in: a rank that is a multiple of 8.
LORA_R, LORA_ALPHA = 8, 16


class _EPLayerStub(EPMoELayerBase):
    """An EP MoE layer running the real base gather and guard on one CPU process. Stubbed, off the path under test:
    ``all_gather_cat`` (pure transport) and the gather's destination device (the sync hardcodes ``"cuda"``)."""

    FAMILY = MoEFamilySpec(model_types=())

    def __init__(self, expert_tp_size: int = 1, *, expert_lora: bool = False):
        nn.Module.__init__(self)
        self.expert_tp_size = expert_tp_size
        self.expert_tp_rank = 0
        self.expert_tp_group = "etp-group"  # never used: all_gather_cat is stubbed
        self._expert_lora_attrs = frozenset()
        self.expert_lora_scaling = 1.0
        self.expert_lora_dropout = nn.Identity()

        shard = INTERMEDIATE // expert_tp_size
        if expert_tp_size > 1:
            self.gate_proj = nn.Parameter(torch.randn(NUM_EXPERTS, HIDDEN, shard))
            self.up_proj = nn.Parameter(torch.randn(NUM_EXPERTS, HIDDEN, shard))
            self.down_proj = nn.Parameter(torch.randn(NUM_EXPERTS, shard, HIDDEN))
        else:
            self.gate_up_proj = nn.Parameter(torch.randn(NUM_EXPERTS, HIDDEN, 2 * INTERMEDIATE))
            self.down_proj = nn.Parameter(torch.randn(NUM_EXPERTS, INTERMEDIATE, HIDDEN))
        self.gate = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False)

        spec = ExpertLoraSpec(r=LORA_R, alpha=LORA_ALPHA) if expert_lora else None
        self.ep_config = SimpleNamespace(ep_size=1, process_group=None, dispatch_ep_group=None, expert_lora=spec)
        self._init_expert_lora()
        # _init_expert_lora zero-inits B (delta 0), which would make merged == base and hide a dropped fold.
        for attr in self._expert_lora_attrs:
            with torch.no_grad():
                getattr(self, f"{attr}_lora_B").copy_(torch.randn_like(getattr(self, f"{attr}_lora_B")))

    def forward(self, hidden_states, **kwargs):  # pragma: no cover - never called
        return hidden_states

    def gather_expert_state_dict(self, *, device="cpu", merge_lora: bool = False, retain: bool = True) -> dict:
        """The real gather, pinned to CPU."""
        return super().gather_expert_state_dict(device="cpu", merge_lora=merge_lora, retain=retain)


@pytest.fixture(autouse=True)
def single_process_all_gather(monkeypatch):
    def all_gather_cat(tensor, dim, group, world_size):
        return tensor if world_size <= 1 else torch.cat([tensor] * world_size, dim=dim)

    monkeypatch.setattr(expert_gather, "all_gather_cat", all_gather_cat)


class _Model(nn.Module):
    """One EP layer at ``mlp`` plus a dense param, so the non-expert sync path runs too."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.embed = nn.Embedding(4, HIDDEN)
        self.mlp = layer

    def forward(self, x):  # pragma: no cover - never called
        return x


class _RecordingSender:
    def __init__(self):
        self.sent: dict[str, torch.Tensor] = {}

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        self.sent[name] = weights

    def finish_weight_update(self) -> None:  # pragma: no cover - the caller flushes, not the gather
        pass


def _sync(layer: nn.Module) -> _RecordingSender:
    sender = _RecordingSender()
    gather_and_send_weights(_Model(layer), sender)
    return sender


def test_full_finetune_syncs_under_expert_tp():
    layer = _EPLayerStub(expert_tp_size=2, expert_lora=False)
    assert not layer.has_expert_lora

    sender = _sync(layer)

    assert f"{LAYER_PATH}.experts.gate_up_proj" in sender.sent, "expert weights never reached the client"
    assert f"{LAYER_PATH}.experts.down_proj" in sender.sent

    # Values too: a swapped [up; gate] order or a shipped local slice keeps the shape.
    tp = layer.expert_tp_size
    expected_gate_up = torch.cat([layer.gate_proj.detach()] * tp + [layer.up_proj.detach()] * tp, dim=2)
    expected_down = torch.cat([layer.down_proj.detach()] * tp, dim=1)
    assert torch.equal(sender.sent[f"{LAYER_PATH}.experts.gate_up_proj"], expected_gate_up.transpose(1, 2))
    assert torch.equal(sender.sent[f"{LAYER_PATH}.experts.down_proj"], expected_down.transpose(1, 2))

    assert f"{LAYER_PATH}.gate.weight" in sender.sent
    assert "embed.weight" in sender.sent


def test_expert_lora_under_expert_tp_still_raises():
    """With adapters the fold is ambiguous under expert TP, so the sync must fail loud instead of serving frozen base
    experts, on every rank through its rank-uniform verdict, the original type and text carried in the message."""
    layer = _EPLayerStub(expert_tp_size=2, expert_lora=True)
    assert layer.has_expert_lora, "the stub built no adapters — the test would be vacuous"

    with pytest.raises(RuntimeError, match="NotImplementedError: .*merge_lora"):
        _sync(layer)


def test_expert_lora_without_expert_tp_still_folds():
    """The fold the guard protects must survive: at expert_tp_size == 1 vLLM gets base + scaling*(A@B)."""
    layer = _EPLayerStub(expert_tp_size=1, expert_lora=True)
    assert "gate_up_proj" in layer._expert_lora_attrs

    sender = _sync(layer)
    sent = sender.sent[f"{LAYER_PATH}.experts.gate_up_proj"]

    base = layer.gate_up_proj.detach()
    delta = layer.expert_lora_scaling * torch.bmm(
        layer.gate_up_proj_lora_A.detach(), layer.gate_up_proj_lora_B.detach()
    )
    assert torch.allclose(sent, (base + delta).transpose(1, 2), atol=1e-5), (
        "the expert-LoRA delta was not folded into the weights forwarded to vLLM"
    )
    assert not torch.allclose(sent, base.transpose(1, 2), atol=1e-5), (
        "vLLM received the FROZEN base experts — the adapter delta is non-trivial by construction"
    )


@pytest.mark.parametrize("expert_tp_size", [1, 2])
@pytest.mark.parametrize("expert_lora", [False, True])
def test_guard_raises_only_on_a_real_fold_under_expert_tp(expert_tp_size, expert_lora):
    layer = _EPLayerStub(expert_tp_size=expert_tp_size, expert_lora=expert_lora)

    layer.gather_expert_state_dict(merge_lora=False)
    if expert_lora and expert_tp_size > 1:
        with pytest.raises(NotImplementedError):
            layer.gather_expert_state_dict(merge_lora=True)
    else:
        layer.gather_expert_state_dict(merge_lora=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
