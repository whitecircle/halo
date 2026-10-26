#!/usr/bin/env python
"""CPU tests for the per-expert SAVE-split ↔ LOAD-fuse round trip.

Two halves of one contract, declared in two places and never checked against each other until here:

* SAVE — the EP gather splits the fused ``experts.gate_up_proj`` ``[E, 2M, H]`` into a
  ``FUSED_SPLIT_PER_EXPERT`` family's ``per_expert_keys`` hub names.
* LOAD — the lazy loader fuses those names back via ``per_expert_fusion_map()``, resolving the
  gate/up halves by POSITION.

If the two ever disagree, a checkpoint reloads with gate and up swapped: identical shapes, no
missing keys, no error — a silently corrupted MoE. ``test_gate_up_swap_is_detected`` proves the
assertion is sharp enough to see that.

The fusion map is keyed by projection NAME across the WHOLE roster, so a new family reusing a name
at a different position would silently swap the other family's halves — whichever spec comes last
wins. That conflict must RAISE.

    python tests/cpu/checkpoint/test_per_expert_fusion_roundtrip.py
"""

from __future__ import annotations

import pytest
import torch

from halo.distributed.expert_parallel.families.roster import ep_layer_classes
from halo.models.moe import family_specs
from halo.models.moe.expert_layout import ExpertLayout, ExpertLayoutKind
from halo.models.moe.family_specs import MoEFamilySpec, per_expert_fusion_map
from tests.common.ep_stubs import bare_ep_layer

E, H, M = 3, 8, 16


def _reference():
    """Per-expert gate/up/down and the fused form the transformers conversion produces:
    stack experts on dim 0, concatenate ``[gate; up]`` on dim 1 (``F.linear`` convention)."""
    torch.manual_seed(0)
    gate = torch.randn(E, M, H)
    up = torch.randn(E, M, H)
    down = torch.randn(E, H, M)
    fused = {
        "experts.gate_up_proj": torch.cat([gate, up], dim=1),  # [E, 2M, H]
        "experts.down_proj": down,  # [E, H, M]
    }
    return gate, up, down, fused


def _split_via_family(cls, fused):
    """The SAVE side: the family's real gather on one process, its experts in the runtime matmul convention."""
    layer = bare_ep_layer(
        cls,
        gate_up_proj=fused["experts.gate_up_proj"].transpose(1, 2).clone(),
        down_proj=fused["experts.down_proj"].transpose(1, 2).clone(),
    )
    return layer.gather_expert_state_dict()


def _fuse_via_loader(cls, split):
    """The LOAD side: re-fuse the per-expert tensors exactly as ``ExpertFuser`` does — halves ordered
    by ``per_expert_fusion_map()`` position, concatenated on dim 0, stacked on dim 0."""
    fusion_map = per_expert_fusion_map()
    gate_up, down = [], []
    for i in range(E):
        by_group: dict[str, dict[int, torch.Tensor]] = {}
        for name in cls.FAMILY.expert_layout.per_expert_keys:
            key = f"experts.{i}.{name}.weight"
            assert key in split, f"{cls.__name__}: save split produced no '{key}' (got {sorted(split)[:4]})"
            group, position = fusion_map[f"{name}.weight"]
            by_group.setdefault(group, {})[position] = split[key]
        gate_up.append(torch.cat([by_group["gate_up"][0], by_group["gate_up"][1]], dim=0))
        down.append(by_group["down"][0])
    return torch.stack(gate_up, dim=0), torch.stack(down, dim=0)


def _unfusing_families():
    return [
        cls
        for cls in ep_layer_classes()
        if "FAMILY" in vars(cls) and cls.FAMILY.expert_layout.kind is ExpertLayoutKind.FUSED_SPLIT_PER_EXPERT
    ]


def test_at_least_one_family_declares_a_per_expert_layout():
    """Anti-vacuity: the round-trip test below would pass trivially on an empty roster."""
    assert _unfusing_families(), "no EP family declares a FUSED_SPLIT_PER_EXPERT layout — round trip untested"


@pytest.mark.parametrize("cls", _unfusing_families(), ids=lambda c: c.__name__)
def test_save_split_then_load_fuse_is_identity(cls):
    """Whole roster: split on save, fuse on load, get the original fused tensors back bitwise."""
    _, _, _, fused = _reference()
    gate_up, down = _fuse_via_loader(cls, _split_via_family(cls, fused))
    assert torch.equal(gate_up, fused["experts.gate_up_proj"]), f"{cls.__name__}: gate_up round trip corrupted"
    assert torch.equal(down, fused["experts.down_proj"]), f"{cls.__name__}: down round trip corrupted"


@pytest.mark.parametrize("cls", _unfusing_families(), ids=lambda c: c.__name__)
def test_gate_up_swap_is_detected(cls):
    """Sharpness check: the round-trip assertion must FAIL when the halves are swapped.

    Without this, a fusion map that mapped both halves to the same position (or the split emitting
    them reversed) could still satisfy the identity above by symmetry.
    """
    _, _, _, fused = _reference()
    split = _split_via_family(cls, fused)
    gate_name, up_name, _ = cls.FAMILY.expert_layout.per_expert_keys
    for i in range(E):
        gate_key, up_key = f"experts.{i}.{gate_name}.weight", f"experts.{i}.{up_name}.weight"
        split[gate_key], split[up_key] = split[up_key], split[gate_key]
    gate_up, _ = _fuse_via_loader(cls, split)
    assert not torch.equal(gate_up, fused["experts.gate_up_proj"]), (
        f"{cls.__name__}: swapping gate and up produced the SAME fused tensor — the round-trip "
        f"assertion cannot see a half swap"
    )


def test_roster_has_no_conflicting_fusion_claims():
    """The live roster must build a fusion map at all (a conflict raises)."""
    assert per_expert_fusion_map(), "roster produced an empty per-expert fusion map"


def test_conflicting_family_claim_raises(monkeypatch):
    """A family reusing a projection name at a DIFFERENT position must raise, not last-write-win.

    Built with plain assignment the map would let the conflicting family overwrite the earlier
    claim, and the loser's experts fuse with gate/up swapped — no error, right shapes.
    """
    gate_name, up_name, down_name = _unfusing_families()[0].FAMILY.expert_layout.per_expert_keys
    # Swapping gate and up claims the same two names at the opposite positions.
    conflicting = MoEFamilySpec(
        model_types=("conflicting_moe",),
        expert_layout=ExpertLayout(ExpertLayoutKind.FUSED_SPLIT_PER_EXPERT, (up_name, gate_name, down_name)),
    )
    monkeypatch.setattr(family_specs, "MOE_FAMILY_SPECS", (*family_specs.MOE_FAMILY_SPECS, conflicting))
    per_expert_fusion_map.cache_clear()
    try:
        with pytest.raises(ValueError, match="claimed as"):
            per_expert_fusion_map()
    finally:
        monkeypatch.undo()
        per_expert_fusion_map.cache_clear()

    # The roster must be healthy again for every later test/consumer.
    assert per_expert_fusion_map()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
