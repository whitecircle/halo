#!/usr/bin/env python
"""EP+CP checkpoint save must operate on the inner model, never the CP wrapper.

``UlyssesCPModelWrapper`` overrides ``named_parameters()``/``state_dict()`` to yield inner-model
names but not ``named_modules()``. A save that walks the wrapper therefore sees EP-layer module
paths prefixed with the wrapper's ``model.`` level while parameter names stay clean, so the
expert classification in ``_save_ep_gathered`` never matches: every expert param takes the non-EP
branch (the rank-local shard is written under the canonical key), the gathered full experts land
under doubled bogus keys, and persistent buffers save under wrapper-prefixed keys that never
reload. ``save_ep_model`` must unwrap the CP wrapper; ``validate_ep_sharded_save`` must reject
sharded EP saves on a CP-patched model (the sharded path has no CP key remap).

Run: ``python tests/cpu/checkpoint/test_ep_cp_save_wrapper_seam.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from accelerate import PartialState
from safetensors import safe_open

from src.distributed.context_parallel.base_layer import UlyssesAttentionBase
from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper
from src.distributed.expert_parallel.saving import save_ep_model, validate_ep_sharded_save
from src.models.structure import unwrap_model
from tests.common.ep_stubs import StubEPLayerBase

PartialState()  # save_ep_model's accelerate logger needs accelerate state initialized

NUM_EXPERTS = 4
NUM_LOCAL_EXPERTS = 2  # simulates an ep2 rank: local params hold E/ep experts, the gather returns E
HIDDEN = 8
INTER = 6


class _StubConfig(SimpleNamespace):
    def save_pretrained(self, output_dir):
        pass


class _StubEPLayer(StubEPLayerBase):
    """Minimal EP layer: local expert shards as params, full-expert gather, no collectives."""

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

    @classmethod
    def merge_shards_to_hf(cls, prefix: str, params: dict) -> dict:  # pragma: no cover - paired override
        return {}


class _StubUlyssesAttention(UlyssesAttentionBase):
    """CP-patched attention marker (skips base init — only the isinstance matters here)."""

    def __init__(self):
        nn.Module.__init__(self)
        self.original_attention = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def _project_qkv(self, hidden_states, batch_size, local_seq_len):  # pragma: no cover
        raise NotImplementedError

    def _apply_rotary_pos_emb(self, q, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


def _build_inner(cp_patched_attention: bool = False) -> nn.Module:
    """`inner.model.layers.0.{self_attn,mlp}` — the HF-shaped tree an EP+CP trainer holds."""
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


def _wrap(inner: nn.Module) -> UlyssesCPModelWrapper:
    """A real UlyssesCPModelWrapper around ``inner`` without attention patching (needs no groups)."""
    wrapper = UlyssesCPModelWrapper.__new__(UlyssesCPModelWrapper)
    nn.Module.__init__(wrapper)
    wrapper.model = inner
    return wrapper


def test_unwrap_model_peels_declared_wrappers():
    """The unwrap protocol is class-declared: the CP wrapper's ``_toolkit_inner_model_attr``
    descends to the inner model; an undeclared module passes through unchanged."""
    inner = _build_inner()
    wrapper = _wrap(inner)
    assert unwrap_model(wrapper) is inner
    assert unwrap_model(inner) is inner


def test_gathered_ep_save_unwraps_cp_wrapper(tmp_path):
    """Saving through the CP wrapper must produce exactly the inner model's canonical keys with
    FULL expert tensors — no wrapper-prefixed keys, no rank-local expert shards."""
    inner = _build_inner()
    wrapper = _wrap(inner)

    save_ep_model(wrapper, str(tmp_path), sharded=False, cp_key_remap=True)

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

    # Experts must be the FULL gathered [E, ...] tensors, not this rank's local [E/ep, ...] shard.
    assert saved["model.layers.0.mlp.experts.gate_up_proj"].shape[0] == NUM_EXPERTS
    assert saved["model.layers.0.mlp.experts.down_proj"].shape[0] == NUM_EXPERTS


def test_sharded_ep_save_rejected_under_cp():
    """A CP-patched model must be rejected for sharded EP saves — the sharded path has no CP key
    remap, so the shards would carry raw '.original_attention.' keys the merge cannot fix."""
    with pytest.raises(ValueError, match="Context Parallelism"):
        validate_ep_sharded_save(_build_inner(cp_patched_attention=True), world_size=2)
    # Same check through the CP wrapper (the trainer validates before any save).
    with pytest.raises(ValueError, match="Context Parallelism"):
        validate_ep_sharded_save(_wrap(_build_inner(cp_patched_attention=True)), world_size=2)


def test_sharded_ep_save_allowed_without_cp():
    """Control: the identical topology without CP patching passes the CP gate (and the rest of the
    sharded-save validation for a supported family / single spanning EP group)."""
    validate_ep_sharded_save(_build_inner(cp_patched_attention=False), world_size=2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
