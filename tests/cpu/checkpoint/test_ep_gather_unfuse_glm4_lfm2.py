"""The ``FUSED_SPLIT_PER_EXPERT`` layout the EP gather writes for GLM4-Lite, GLM-5 Next and LFM2. A swapped split has identical
shapes and corrupts the served policy, so tensors match a reference fusion."""

import pytest
import torch

from halo.distributed.expert_parallel.families.glm4 import EPGlm4MoELayer
from halo.distributed.expert_parallel.families.glm5_next import EPGlm5NextMoELayer
from halo.distributed.expert_parallel.families.lfm2 import EPLfm2MoELayer
from halo.distributed.expert_parallel.families.qwen3_5 import EPQwen3_5MoELayer
from tests.common.ep_stubs import bare_ep_layer

E, H, M = 4, 8, 16


def _fused_reference():
    """Stack experts on dim 0, concatenate [gate; up] on dim 1 (F.linear convention)."""
    gate = torch.randn(E, M, H)
    up = torch.randn(E, M, H)
    down = torch.randn(E, H, M)
    return gate, up, down, torch.cat([gate, up], dim=1)


def _gather(layer_cls, gate_up_flinear, down_flinear):
    """Its experts stored in the runtime's matmul convention."""
    layer = bare_ep_layer(
        layer_cls, gate_up_proj=gate_up_flinear.transpose(1, 2), down_proj=down_flinear.transpose(1, 2)
    )
    return layer.gather_expert_state_dict(merge_lora=True)


def test_glm4_unfuses_to_per_expert_hub_names():
    gate, up, down, gate_up = _fused_reference()
    state = _gather(EPGlm4MoELayer, gate_up, down)
    assert list(state) == [
        f"experts.{i}.{proj}.weight" for i in range(E) for proj in ("gate_proj", "up_proj", "down_proj")
    ]
    for i in range(E):
        assert torch.equal(state[f"experts.{i}.gate_proj.weight"], gate[i])
        assert torch.equal(state[f"experts.{i}.up_proj.weight"], up[i])
        assert torch.equal(state[f"experts.{i}.down_proj.weight"], down[i])


def test_glm5_next_unfuses_to_per_expert_hub_names():
    gate, up, down, gate_up = _fused_reference()
    state = _gather(EPGlm5NextMoELayer, gate_up, down)
    assert list(state) == [
        f"experts.{i}.{proj}.weight" for i in range(E) for proj in ("gate_proj", "up_proj", "down_proj")
    ]
    for i in range(E):
        assert torch.equal(state[f"experts.{i}.gate_proj.weight"], gate[i])
        assert torch.equal(state[f"experts.{i}.up_proj.weight"], up[i])
        assert torch.equal(state[f"experts.{i}.down_proj.weight"], down[i])


def test_lfm2_unfuses_to_per_expert_w_names():
    gate, up, down, gate_up = _fused_reference()
    state = _gather(EPLfm2MoELayer, gate_up, down)
    assert set(state) == {f"experts.{i}.w{n}.weight" for i in range(E) for n in (1, 2, 3)}
    for i in range(E):
        assert torch.equal(state[f"experts.{i}.w1.weight"], gate[i])  # w1 = gate
        assert torch.equal(state[f"experts.{i}.w3.weight"], up[i])  # w3 = up
        assert torch.equal(state[f"experts.{i}.w2.weight"], down[i])  # w2 = down


def test_fused_family_passes_through_unchanged():
    """In ``F.linear`` convention, unsplit."""
    _, _, down, gate_up = _fused_reference()
    state = _gather(EPQwen3_5MoELayer, gate_up, down)
    assert set(state) == {"experts.gate_up_proj", "experts.down_proj"}
    assert torch.equal(state["experts.gate_up_proj"], gate_up)
    assert torch.equal(state["experts.down_proj"], down)


def test_split_order_detects_swap():
    """A fused tensor built up-first must not reproduce the reference gate."""
    gate, up, down, _ = _fused_reference()
    state = _gather(EPGlm4MoELayer, torch.cat([up, gate], dim=1), down)
    assert not torch.equal(state["experts.0.gate_proj.weight"], gate[0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
