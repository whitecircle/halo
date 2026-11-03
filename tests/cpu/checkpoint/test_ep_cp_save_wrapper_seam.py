#!/usr/bin/env python
"""EP+CP checkpoint save operates on the inner model. ``UlyssesModel`` overrides ``state_dict()`` but not
``named_modules()``, so a walk through it misclassifies every expert param."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open

from halo.checkpoint.parallel.ep import reject_unmergeable_ep_sharded_save, save_ep_gathered_checkpoint
from halo.distributed.context_parallel.ulysses_attention import UlyssesAttention
from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from halo.models.unwrapping import unwrap_toolkit_wrappers
from tests.common.checkpoint_io import save_context
from tests.common.ep_stubs import StubEPLayerBase
from tests.common.parallelism import make_parallelism_config

NUM_EXPERTS = 4
NUM_LOCAL_EXPERTS = 2  # Simulates an ep2 rank: local params hold E/ep experts, the gather returns E.
HIDDEN = 8
INTER = 6


class _StubConfig(SimpleNamespace):
    def save_pretrained(self, output_dir):
        pass


class _StubEPLayer(StubEPLayerBase):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(NUM_LOCAL_EXPERTS, HIDDEN, 2 * INTER))
        self.down_proj = nn.Parameter(torch.randn(NUM_LOCAL_EXPERTS, INTER, HIDDEN))
        self.router = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False)
        self.ep_config = SimpleNamespace(ep_size=2, ep_group_size=2, expert_tp_size=1, num_ep_groups=1)

    def expert_named_params(self):
        return [("gate_up_proj", self.gate_up_proj), ("down_proj", self.down_proj)]

    def gather_expert_state_dict(self, device: str = "cpu", merge_lora: bool = False, retain: bool = True) -> dict:
        if not retain:
            return {}
        return {
            "experts.gate_up_proj": torch.zeros(NUM_EXPERTS, 2 * INTER, HIDDEN),
            "experts.down_proj": torch.zeros(NUM_EXPERTS, HIDDEN, INTER),
        }


class _StubUlyssesAttention(UlyssesAttention):
    """Skips base init; only the isinstance matters here."""

    def __init__(self):
        nn.Module.__init__(self)
        self.original_attention = nn.Linear(HIDDEN, HIDDEN, bias=False)


def _build_inner(cp_patched_attention: bool = False) -> nn.Module:
    """The HF-shaped tree an EP+CP trainer holds."""
    inner = nn.Module()
    backbone = nn.Module()
    layer = nn.Module()
    layer.self_attn = _StubUlyssesAttention() if cp_patched_attention else nn.Linear(HIDDEN, HIDDEN, bias=False)
    layer.mlp = _StubEPLayer()
    backbone.layers = nn.ModuleList([layer])
    backbone.register_buffer("scalar", torch.tensor([2.0]), persistent=True)
    inner.model = backbone
    inner.config = _StubConfig(model_type="gpt_oss", auto_map=None, tie_word_embeddings=False)
    return inner


def _wrap(inner: nn.Module) -> UlyssesModel:
    """Without attention patching, which needs groups."""
    wrapper = UlyssesModel.__new__(UlyssesModel)
    nn.Module.__init__(wrapper)
    wrapper.model = inner
    return wrapper


def test_unwrap_model_peels_declared_wrappers():
    inner = _build_inner()
    wrapper = _wrap(inner)
    assert unwrap_toolkit_wrappers(wrapper) is inner
    assert unwrap_toolkit_wrappers(inner) is inner


def test_gathered_ep_save_unwraps_cp_wrapper(tmp_path):
    inner = _build_inner()
    wrapper = _wrap(inner)

    save_ep_gathered_checkpoint(
        save_context(wrapper, make_parallelism_config(world_size=2, gpus_per_node=2, cp_size=2)), str(tmp_path)
    )

    ckpt = tmp_path / "model.safetensors"
    assert ckpt.is_file(), "gathered EP save wrote no model.safetensors"
    with safe_open(str(ckpt), framework="pt") as f:
        saved = {k: f.get_tensor(k) for k in list(f.keys())}

    doubled = [k for k in saved if k.startswith("model.model.")]
    assert not doubled, f"wrapper-prefixed keys in checkpoint: {doubled}"

    expected = {
        "model.layers.0.self_attn.weight",
        "model.layers.0.mlp.router.weight",
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
        "model.scalar",
    }
    assert set(saved) == expected, f"saved keys {sorted(saved)} != expected {sorted(expected)}"

    # Not this rank's local [E/ep, ...] shard.
    assert saved["model.layers.0.mlp.experts.gate_up_proj"].shape[0] == NUM_EXPERTS
    assert saved["model.layers.0.mlp.experts.down_proj"].shape[0] == NUM_EXPERTS


def test_sharded_ep_save_rejected_under_cp():
    """The sharded path has no CP key remap, so shards would carry raw '.original_attention.' keys."""
    with pytest.raises(ValueError, match="Context Parallelism"):
        reject_unmergeable_ep_sharded_save(_build_inner(cp_patched_attention=True), world_size=2)
    # The trainer validates through the CP wrapper before any save.
    with pytest.raises(ValueError, match="Context Parallelism"):
        reject_unmergeable_ep_sharded_save(_wrap(_build_inner(cp_patched_attention=True)), world_size=2)


def test_sharded_ep_save_allowed_without_cp():
    reject_unmergeable_ep_sharded_save(_build_inner(cp_patched_attention=False), world_size=2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
