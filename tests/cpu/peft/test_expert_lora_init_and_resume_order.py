#!/usr/bin/env python
"""Native expert-LoRA init scale and the rank-consistency of the adapter resume loop.

Two invariants that fail silently:

* ``load_expert_lora_state_dict`` re-shards DTensor adapters with ``distribute_tensor`` — a
  collective. Its loop therefore has to visit the adapters in an order that is a deterministic
  function of the NAMES, not of the container holding them: ``_expert_lora_attrs`` is a frozenset,
  and frozenset iteration order over strings varies per process (``PYTHONHASHSEED`` is unpinned under
  torchrun), so ranks would scatter in different orders.
* ``lora_A`` must be drawn on PEFT's scale. PEFT initializes a 2-D ``[r, K]`` weight, whose
  ``kaiming_uniform_(a=sqrt(5))`` bound reduces to ``1/sqrt(K)``; handing torch the 3-D ``[E, K, r]``
  tensor folds ``r`` into ``fan_in`` and shrinks the bound by ``sqrt(r)``, so expert and attention
  adapters in one run would warm up on different scales.

Run: pytest tests/cpu/peft/test_expert_lora_init_and_resume_order.py
"""

import math

import pytest
import torch
import torch.nn as nn

from src.distributed.expert_parallel.config import ExpertLoraSpec
from tests.common.ep_stubs import StubEPLayerBase

NUM_LOCAL_EXPERTS = 2
HIDDEN = 64
INTERMEDIATE = 32
RANK = 8


class _StubEPLayer(StubEPLayerBase):
    """EP layer carrying two adapted expert weights, without live EP groups."""

    def __init__(self, attrs=("down_proj", "gate_up_proj")):
        super().__init__()
        self.expert_tp_size = 1
        self.expert_start, self.expert_end = 0, NUM_LOCAL_EXPERTS
        self._expert_lora_attrs = frozenset(attrs)
        for attr in attrs:
            setattr(self, f"{attr}_lora_A", nn.Parameter(torch.zeros(NUM_LOCAL_EXPERTS, HIDDEN, RANK)))
            setattr(self, f"{attr}_lora_B", nn.Parameter(torch.zeros(NUM_LOCAL_EXPERTS, RANK, INTERMEDIATE)))


class _OrderRecordingState(dict):
    """A layer_state that records the order the load loop reads its keys in."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.access_order: list[str] = []

    def __getitem__(self, key):
        self.access_order.append(key)
        return super().__getitem__(key)


def _saved_state(attrs) -> _OrderRecordingState:
    state = _OrderRecordingState()
    for attr in attrs:
        state[f"experts.{attr}.lora_A"] = torch.randn(NUM_LOCAL_EXPERTS, HIDDEN, RANK)
        state[f"experts.{attr}.lora_B"] = torch.randn(NUM_LOCAL_EXPERTS, RANK, INTERMEDIATE)
    state.access_order.clear()
    return state


def test_resume_visits_adapters_in_name_sorted_order():
    """Independent of how the attr container itself iterates — that is the whole point."""
    attrs = ("down_proj", "gate_up_proj")
    layer = _StubEPLayer(attrs)
    # Adversarial container order: a loop over the attribute directly would visit gate_up_proj first.
    layer._expert_lora_attrs = ["gate_up_proj", "down_proj"]
    state = _saved_state(attrs)

    layer.load_expert_lora_state_dict(state)

    assert state.access_order == [
        "experts.down_proj.lora_A",
        "experts.down_proj.lora_B",
        "experts.gate_up_proj.lora_A",
        "experts.gate_up_proj.lora_B",
    ]


def test_resume_order_matches_the_save_order():
    """``gather_expert_lora_state_dict`` writes sorted; the load must walk that same order.

    The container is skewed the other way for the same reason as above — left as a real frozenset,
    an implementation that iterates the container instead of sorting would agree with the save order
    or not depending on PYTHONHASHSEED."""
    attrs = ("down_proj", "gate_up_proj")
    layer = _StubEPLayer(attrs)
    layer._expert_lora_attrs = sorted(attrs, reverse=True)
    state = _saved_state(attrs)

    layer.load_expert_lora_state_dict(state)

    saved_order = [f"experts.{attr}.lora_{w}" for attr in sorted(attrs) for w in ("A", "B")]
    assert state.access_order == saved_order


def test_resume_copies_the_saved_values():
    """Ordering aside, the load must still land the right tensor on the right adapter."""
    attrs = ("down_proj", "gate_up_proj")
    layer = _StubEPLayer(attrs)
    state = _saved_state(attrs)

    layer.load_expert_lora_state_dict(state)

    for attr in attrs:
        for which in ("A", "B"):
            expected = state[f"experts.{attr}.lora_{which}"]
            assert torch.equal(getattr(layer, f"{attr}_lora_{which}").data, expected)


def test_expert_lora_a_init_matches_peft_scale():
    """The grouped ``lora_A`` bound must be PEFT's ``1/sqrt(K)``, not the 3-D ``1/sqrt(K*r)``."""
    layer = _StubEPLayer(("gate_up_proj",))
    layer.ep_config = type("_Cfg", (), {"expert_lora": ExpertLoraSpec(r=RANK, alpha=2 * RANK)})()
    layer.gate_up_proj = nn.Parameter(torch.zeros(NUM_LOCAL_EXPERTS, HIDDEN, INTERMEDIATE))
    layer._expert_lora_attrs = frozenset()

    layer._init_expert_lora()

    lora_a = layer.gate_up_proj_lora_A
    peft_bound = 1.0 / math.sqrt(HIDDEN)
    assert layer._expert_lora_attrs == frozenset({"gate_up_proj"})
    assert lora_a.abs().max().item() <= peft_bound
    # The 3-D fan_in would cap the draw at bound/sqrt(r); require the real spread.
    assert lora_a.abs().max().item() > peft_bound / math.sqrt(RANK)
    assert torch.count_nonzero(layer.gate_up_proj_lora_B) == 0, "B must stay zero (delta 0 at init)"
    assert not layer.gate_up_proj.requires_grad, "the base expert must be frozen"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
