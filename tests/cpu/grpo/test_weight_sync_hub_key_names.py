#!/usr/bin/env python
"""The vLLM weight sync must forward the HUB key spelling, exactly like a gathered checkpoint.

vLLM loads by hub name and **silently skips** an unknown one, so a family whose live module tree is
spelled differently from its hub checkpoint (Laguna: ``mlp.gate.e_score_correction_bias`` and
``mlp.shared_experts.*``) loses precisely those tensors from every sync — the served policy keeps its
initial router correction bias and shared expert while the trainer moves on, with no error anywhere.

``gather_ep_layer_weights`` already applies the family's ``_EXPORT_KEY_RENAMES``; the sync path builds
its keys itself, so the pin here is EQUIVALENCE — the sync's key set must equal what a gathered
checkpoint of the same module would contain (plus the non-EP params, which no rename touches). That is
what keeps the two paths from drifting.

Run: ``python tests/cpu/grpo/test_weight_sync_hub_key_names.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.expert_weights import gather_ep_layer_weights
from src.distributed.expert_parallel.layers.laguna import EPLagunaMoELayer
from src.trainers.grpo.rollout.weight_sync import gather_and_send_weights

LAYER_PATH = "mlp"
NUM_EXPERTS, HIDDEN, INTERMEDIATE = 2, 6, 4


class _LagunaEPLayerStub(EPMoELayerBase):
    """A gatherable stand-in for the EP-wrapped Laguna MoE block.

    Built on the base (``EPLagunaMoELayer.__init__`` needs live EP process groups) but carrying the
    family's own rename declaration and the module names the real wrapper holds: the router at
    ``gate`` with a frozen ``e_score_correction_bias`` Parameter, and the shared expert at
    ``shared_experts``. ``gather_expert_state_dict`` ignores the requested device — this CPU test
    asserts key spelling, and the real per-family gathers are covered by the checkpoint tests.
    """

    _EXPORT_KEY_RENAMES = EPLagunaMoELayer._EXPORT_KEY_RENAMES
    _PER_EXPERT_UNFUSED_KEYS = None  # this stub does its own (trivial) gather

    def __init__(self):
        nn.Module.__init__(self)
        self._expert_lora_attrs = frozenset()
        self.gate_up_proj = nn.Parameter(torch.randn(NUM_EXPERTS, HIDDEN, 2 * INTERMEDIATE))
        self.down_proj = nn.Parameter(torch.randn(NUM_EXPERTS, INTERMEDIATE, HIDDEN))
        self.gate = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False)
        self.gate.e_score_correction_bias = nn.Parameter(torch.randn(NUM_EXPERTS), requires_grad=False)
        self.shared_experts = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, hidden_states, **kwargs):  # pragma: no cover - never called
        return hidden_states

    def expert_named_params(self):
        return [("gate_up_proj", self.gate_up_proj), ("down_proj", self.down_proj)]

    def gather_expert_state_dict(self, device: str = "cpu", merge_lora: bool = False, retain: bool = True) -> dict:
        if not retain:
            return {}
        return {
            "experts.gate_up_proj": self.gate_up_proj.detach(),
            "experts.down_proj": self.down_proj.detach(),
        }

    @classmethod
    def merge_shards_to_hf(cls, prefix: str, params: dict) -> dict:  # pragma: no cover - paired override
        return {}


class _PlainEPLayerStub(_LagunaEPLayerStub):
    """The same layer for a family whose hub and module spellings agree (every family but Laguna)."""

    _EXPORT_KEY_RENAMES = ()


class _Model(nn.Module):
    """One EP layer at ``mlp`` plus a dense param, so the non-EP names are exercised too."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.embed = nn.Embedding(4, HIDDEN)
        self.mlp = layer

    def forward(self, x):  # pragma: no cover - never called
        return x


class _RecordingSender:
    """Records what the sync forwards (the real client buffers by name for a later broadcast)."""

    def __init__(self):
        self.sent: dict[str, torch.Tensor] = {}

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        assert name not in self.sent, f"{name} forwarded twice"
        self.sent[name] = weights

    def reset_prefix_cache(self) -> None:  # pragma: no cover - the caller flushes, not the gather
        pass


def _sync(layer: nn.Module) -> tuple[_RecordingSender, _Model]:
    model = _Model(layer)
    sender = _RecordingSender()
    assert gather_and_send_weights(model, sender) is False
    return sender, model


def test_sync_keys_equal_the_gathered_checkpoint_keys():
    """The equivalence pin: what vLLM receives is what a gathered checkpoint would carry."""
    layer = _LagunaEPLayerStub()
    sender, _model = _sync(layer)

    gathered = set(gather_ep_layer_weights(LAYER_PATH, layer, retain=True))
    # The sync also forwards params outside the EP layer; no rename applies to those.
    assert set(sender.sent) == gathered | {"embed.weight"}, (
        f"sync keys drifted from the gather: only-sync={sorted(set(sender.sent) - gathered - {'embed.weight'})}, "
        f"only-gather={sorted(gathered - set(sender.sent))}"
    )


def test_renamed_modules_are_forwarded_under_the_hub_spelling():
    """The two Laguna renames, named explicitly: these are the tensors vLLM was dropping."""
    sender, _model = _sync(_LagunaEPLayerStub())

    assert f"{LAYER_PATH}.experts.e_score_correction_bias" in sender.sent
    assert f"{LAYER_PATH}.shared_expert.weight" in sender.sent
    assert f"{LAYER_PATH}.gate.e_score_correction_bias" not in sender.sent
    assert f"{LAYER_PATH}.shared_experts.weight" not in sender.sent
    # A rename must not be a blanket rewrite: the router weight and the experts keep their names.
    assert f"{LAYER_PATH}.gate.weight" in sender.sent
    assert f"{LAYER_PATH}.experts.gate_up_proj" in sender.sent


def test_renaming_moves_the_key_not_the_payload():
    """A rewrite that shuffled payloads between keys would serve a router bias as a shared expert."""
    layer = _LagunaEPLayerStub()
    sender, _model = _sync(layer)

    assert torch.equal(
        sender.sent[f"{LAYER_PATH}.experts.e_score_correction_bias"], layer.gate.e_score_correction_bias
    )
    assert torch.equal(sender.sent[f"{LAYER_PATH}.shared_expert.weight"], layer.shared_experts.weight)
    assert torch.equal(sender.sent[f"{LAYER_PATH}.gate.weight"], layer.gate.weight)


def test_a_family_without_renames_is_untouched():
    """Anti-over-rejection: every other family's forwarded names must be byte-identical to before."""
    layer = _PlainEPLayerStub()
    sender, _model = _sync(layer)

    assert set(sender.sent) == set(gather_ep_layer_weights(LAYER_PATH, layer, retain=True)) | {"embed.weight"}
    assert f"{LAYER_PATH}.gate.e_score_correction_bias" in sender.sent
    assert f"{LAYER_PATH}.shared_experts.weight" in sender.sent


def test_dense_only_model_forwards_live_names():
    """A model with no EP layer at all is unaffected by the rewrite."""
    model = _Model(nn.Linear(HIDDEN, HIDDEN, bias=False))
    sender = _RecordingSender()
    gather_and_send_weights(model, sender)

    assert set(sender.sent) == {"embed.weight", f"{LAYER_PATH}.weight"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
