"""CPU tests for the ``_PER_EXPERT_UNFUSED_KEYS``-driven per-expert gather split.

GLM4-Lite and LFM2 store experts fused (``gate_up_proj`` = the transformers conversion's
``[gate; up]`` output-dim concatenation) but their hub checkpoints and vLLM loaders use per-expert
names — GLM4: ``experts.{i}.{gate,up,down}_proj.weight``; LFM2: ``experts.{i}.w{1,3,2}.weight``.
Each family declares only ``_PER_EXPERT_UNFUSED_KEYS``; the base ``gather_expert_state_dict``
splits the fused gather automatically. The split must keep the conversion's order (gate first) —
a swapped split has identical shapes and would corrupt the served policy silently, so these tests
assert exact tensor identity against a reference fusion built the way
``MergeModulelist(dim=0) + Concatenate(dim=1)`` builds it. Families that leave the attribute unset
must get the fused dict back unchanged.

    python tests/cpu/checkpoint/test_ep_gather_unfuse_glm4_lfm2.py
"""

from unittest.mock import patch

import pytest
import torch

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.layers.glm4 import EPGlm4MoELayer
from src.distributed.expert_parallel.layers.lfm2 import EPLfm2MoELayer
from src.distributed.expert_parallel.layers.qwen3_5 import EPQwen3_5MoELayer

E, H, M = 4, 8, 16


def _fused_reference():
    """Per-expert gate/up/down plus their fused form exactly as the transformers conversion fuses
    them: stack experts on dim 0, concatenate [gate; up] on dim 1 (F.linear convention)."""
    gate = torch.randn(E, M, H)
    up = torch.randn(E, M, H)
    down = torch.randn(E, H, M)
    fused = {
        "experts.gate_up_proj": torch.cat([gate, up], dim=1),  # [E, 2M, H]
        "experts.down_proj": down,  # [E, H, M]
    }
    return gate, up, down, fused


def _gather_with_stubbed_base(layer_cls, fused):
    """Invoke the family's gather with the fused gather stubbed (no GPU/EP state needed)."""
    instance = object.__new__(layer_cls)
    with patch.object(EPMoELayerBase, "_gather_fused_expert_state_dict", return_value=dict(fused)) as base:
        state = layer_cls.gather_expert_state_dict(instance, device="cpu", merge_lora=True)
    # merge_lora must be forwarded to the fused gather (the vLLM-sync fold).
    assert base.call_args.kwargs.get("merge_lora") is True
    return state


def test_glm4_unfuses_to_per_expert_hub_names():
    gate, up, down, fused = _fused_reference()
    state = _gather_with_stubbed_base(EPGlm4MoELayer, fused)
    assert set(state) == {
        f"experts.{i}.{proj}.weight" for i in range(E) for proj in ("gate_proj", "up_proj", "down_proj")
    }
    for i in range(E):
        assert torch.equal(state[f"experts.{i}.gate_proj.weight"], gate[i])
        assert torch.equal(state[f"experts.{i}.up_proj.weight"], up[i])
        assert torch.equal(state[f"experts.{i}.down_proj.weight"], down[i])


def test_lfm2_unfuses_to_per_expert_w_names():
    gate, up, down, fused = _fused_reference()
    state = _gather_with_stubbed_base(EPLfm2MoELayer, fused)
    assert set(state) == {f"experts.{i}.w{n}.weight" for i in range(E) for n in (1, 2, 3)}
    for i in range(E):
        assert torch.equal(state[f"experts.{i}.w1.weight"], gate[i])  # w1 = gate
        assert torch.equal(state[f"experts.{i}.w3.weight"], up[i])  # w3 = up
        assert torch.equal(state[f"experts.{i}.w2.weight"], down[i])  # w2 = down


def test_fused_family_passes_through_unchanged():
    """A family without ``_PER_EXPERT_UNFUSED_KEYS`` (fused hub layout) gets the fused gather
    back untouched — guards against the base split misfiring on fused families."""
    _, _, _, fused = _fused_reference()
    assert EPQwen3_5MoELayer._PER_EXPERT_UNFUSED_KEYS is None
    state = _gather_with_stubbed_base(EPQwen3_5MoELayer, fused)
    assert set(state) == set(fused)
    for key in fused:
        assert torch.equal(state[key], fused[key])


def test_split_order_detects_swap():
    """The gate/up halves are NOT interchangeable: a fused tensor built up-first must not
    reproduce the reference gate — guards the silent-swap failure mode."""
    gate, up, down, fused = _fused_reference()
    swapped = {
        "experts.gate_up_proj": torch.cat([up, gate], dim=1),
        "experts.down_proj": down,
    }
    state = _gather_with_stubbed_base(EPGlm4MoELayer, swapped)
    assert not torch.equal(state["experts.0.gate_proj.weight"], gate[0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
