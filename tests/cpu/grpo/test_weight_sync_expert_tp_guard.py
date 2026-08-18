#!/usr/bin/env python
"""``_reject_merge_lora_under_expert_tp`` must reject a real FOLD under expert TP, not the request.

``_send_ep_expert_weights`` asks every EP layer to gather with ``merge_lora=True`` unconditionally —
it cannot know which layers carry native grouped adapters, and ``_materialize_expert_weight`` no-ops
the fold on an adapter-less weight. A guard keyed on the flag alone therefore rejects every
vLLM-syncing GRPO run at ``expert_tp_size > 1``, including a plain full fine-tune with no PEFT
anywhere. The predicate is :attr:`EPMoELayerBase.has_expert_lora` — what ``_init_expert_lora`` built.

All three cells are pinned, because a "fix" that deleted the guard would satisfy only the first:

* no adapters + ``expert_tp_size > 1`` → the sync completes and forwards reassembled expert tensors;
* real adapters + ``expert_tp_size > 1`` → ``NotImplementedError``. Defense in depth: ``EPConfig``
  already refuses this pairing at construction, so a production run cannot reach it — but the
  sharded layout has no fold seam, and a gather that folded nothing would ship the FROZEN base
  experts to vLLM and run the generator off-policy with no error;
* real adapters + ``expert_tp_size == 1`` → the fold still happens (value-checked), so the guard
  cannot be neutered into suppressing the merge everywhere.

Run: ``python tests/cpu/grpo/test_weight_sync_expert_tp_guard.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.config import ExpertLoraSpec
from src.trainers.grpo.rollout.weight_sync import gather_and_send_weights

LAYER_PATH = "mlp"
# HIDDEN != 2 * INTERMEDIATE on purpose: at equal dims a stray transpose still satisfies every shape.
NUM_EXPERTS, HIDDEN, INTERMEDIATE = 2, 8, 6
LORA_R, LORA_ALPHA = 2, 4


class _EPLayerStub(EPMoELayerBase):
    """An EP MoE layer running the REAL base gather (and therefore the real guard) on one CPU process.

    Stores the same two layouts the production sharding produces — fused ``gate_up_proj`` at
    ``expert_tp_size == 1``, split ``gate_proj``/``up_proj``/``down_proj`` under expert TP — and builds
    its adapters through the production ``_init_expert_lora``. Two things are stubbed, neither on the
    path under test: ``_all_gather_cat`` (pure transport) and the gather's destination device (the sync
    hardcodes ``"cuda"``, which no CPU tier has). The guard, the fold and the layout math stay real.
    """

    def __init__(self, expert_tp_size: int = 1, *, expert_lora: bool = False):
        nn.Module.__init__(self)
        self.expert_tp_size = expert_tp_size
        self.expert_tp_rank = 0
        self.expert_tp_group = "etp-group"  # never used: _all_gather_cat is stubbed below
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

    def gather_expert_state_dict(self, device: str = "cpu", merge_lora: bool = False, retain: bool = True) -> dict:
        """The real base gather, pinned to CPU — the sync asks for ``"cuda"``."""
        return super().gather_expert_state_dict("cpu", merge_lora=merge_lora, retain=retain)

    @classmethod
    def merge_shards_to_hf(cls, prefix: str, params: dict) -> dict:  # pragma: no cover - paired override
        """Unused here; ``__init_subclass__`` requires it alongside the gather override above."""
        return super().merge_shards_to_hf(prefix, params)

    @staticmethod
    def _all_gather_cat(tensor, dim, group, world_size):
        """Single-process stand-in for the EP / expert-TP all-gather: same shapes, no transport."""
        if world_size <= 1:
            return tensor
        return torch.cat([tensor] * world_size, dim=dim)


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

    def reset_prefix_cache(self) -> None:  # pragma: no cover - the caller flushes, not the gather
        pass


def _sync(layer: nn.Module) -> _RecordingSender:
    sender = _RecordingSender()
    gather_and_send_weights(_Model(layer), sender)
    return sender


def test_full_finetune_syncs_under_expert_tp():
    """The P0: no PEFT anywhere + expert_tp_size > 1 must reach vLLM, not raise at the first sync."""
    layer = _EPLayerStub(expert_tp_size=2, expert_lora=False)
    assert not layer.has_expert_lora

    sender = _sync(layer)

    assert f"{LAYER_PATH}.experts.gate_up_proj" in sender.sent, "expert weights never reached the client"
    assert f"{LAYER_PATH}.experts.down_proj" in sender.sent

    # Values, not just shapes: a swapped [up; gate] order or a shipped local slice keeps the shape.
    tp = layer.expert_tp_size
    expected_gate_up = torch.cat([layer.gate_proj.detach()] * tp + [layer.up_proj.detach()] * tp, dim=2)
    expected_down = torch.cat([layer.down_proj.detach()] * tp, dim=1)
    assert torch.equal(sender.sent[f"{LAYER_PATH}.experts.gate_up_proj"], expected_gate_up.transpose(1, 2))
    assert torch.equal(sender.sent[f"{LAYER_PATH}.experts.down_proj"], expected_down.transpose(1, 2))

    assert f"{LAYER_PATH}.gate.weight" in sender.sent
    assert "embed.weight" in sender.sent


def test_expert_lora_under_expert_tp_still_raises():
    """Anti-over-rejection control: with adapters actually present the fold IS ambiguous under expert
    TP, so the sync must still fail loud rather than serve the frozen base experts.

    Through the sync's rank-uniform verdict, which is what the retaining gather runs under: the
    refusal reaches every rank with its original type and text carried in the message.
    """
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
    """The guard's predicate, directly: merge_lora alone is not enough — an adapter must exist."""
    layer = _EPLayerStub(expert_tp_size=expert_tp_size, expert_lora=expert_lora)

    layer._reject_merge_lora_under_expert_tp(False)  # never raises without a fold request
    if expert_lora and expert_tp_size > 1:
        with pytest.raises(NotImplementedError):
            layer._reject_merge_lora_under_expert_tp(True)
    else:
        layer._reject_merge_lora_under_expert_tp(True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
