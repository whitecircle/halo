#!/usr/bin/env python
"""Policies the model-patch layer DERIVES rather than lists, and the one it deliberately does not.

Each of these decides something silent when it is wrong — a NaN-producing attention backend, a
recompute dtype mismatch, a norm quantized to bf16, a RoPE table left as load garbage — so the pins
here are on the derivation itself, not on a call reaching some branch:

* ``model_fa4_backward_nan_prone`` stays a per-family verdict, and the config numbers show why a
  ``head_dim == 256`` derivation cannot replace it;
* the FSDP2 ``cast_forward_inputs`` policy reads a CLASS attribute the Zaya load patch stamps,
  with no model_type list in the wrap;
* the PEFT bf16 cast classifies norms by class, not by a substring of the module path;
* the per-layer-type rotary recompute (one helper, two layouts) rebuilds exactly the buffers that
  exist and invents none.

    python tests/cpu/models/test_model_patch_derivations.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import transformers

from src.distributed.fsdp import _should_cast_forward_inputs
from src.distributed.loading.peft_setup import _peft_module_casting_to_bf16
from src.models.patches.attention import model_fa4_backward_nan_prone
from src.models.patches.buffer_fixes import fix_rotary_inv_freq


def _config(model_type: str):
    return transformers.CONFIG_MAPPING[model_type]()


@pytest.mark.parametrize("model_type", ["qwen3_next", "qwen3_5_moe", "glm4_moe_lite"])
def test_the_fa4_nan_families_are_all_refused(model_type):
    """Every family whose FA4 backward emits NaN gradients must be refused the backend.

    Missing one is silent: the run trains, the loss goes to NaN some steps later, and the cause is a
    kernel choice nothing logged.
    """
    assert model_fa4_backward_nan_prone(_config(model_type)) is True


@pytest.mark.parametrize("model_type", ["qwen3", "llama", "gpt_oss"])
def test_clean_families_keep_fa4(model_type):
    """The refusal must stay narrow — it costs 2-3x on the attention kernel for every model it hits."""
    assert model_fa4_backward_nan_prone(_config(model_type)) is False


def test_the_nan_trigger_is_not_derivable_from_the_declared_head_dim():
    """Pins WHY the predicate is a family verdict instead of ``head_dim == 256 and partial rotary``.

    The two families spell the same physical shape in vocabularies that do not overlap: Qwen3-Next
    declares it directly, GLM-4 MoE Lite is MLA and declares a 64-wide ``head_dim`` with the 256
    split across ``qk_nope_head_dim``/``qk_rope_head_dim`` and no partial-rotary field at all. A
    numeric derivation would therefore read GLM-4 MoE Lite as clean and hand it back to FA4.
    """
    qwen = _config("qwen3_next")
    assert qwen.head_dim == 256 and qwen.rope_parameters["partial_rotary_factor"] < 1.0

    glm = _config("glm4_moe_lite")
    assert glm.head_dim != 256, "GLM-4 MoE Lite now declares head_dim=256 — revisit the derivation"
    assert glm.qk_nope_head_dim + glm.qk_rope_head_dim == 256
    assert "partial_rotary_factor" not in glm.rope_parameters


class _ResidualLayer(nn.Module):
    """Stand-in for a decoder layer whose class declares the fp32 inter-layer residual."""

    _fp32_interlayer_residual = True


def test_the_fp32_residual_policy_is_read_off_the_declaring_class():
    """``cast_forward_inputs`` must be decided by the module tree, not by a model_type list.

    Casting the layer-boundary activation to bf16 makes non-reentrant GC recompute hand torch an
    fp32 tensor where the forward saved a bf16 view, which it rejects mid-backward.
    """
    plain = nn.Sequential(nn.Linear(2, 2))
    assert _should_cast_forward_inputs(plain) is True

    carrier = nn.Sequential(nn.Linear(2, 2), _ResidualLayer())
    assert _should_cast_forward_inputs(carrier) is False


def test_the_zaya_load_patch_declares_the_fp32_residual_on_its_decoder_layer():
    """The other half of that derivation: the hub-native Zaya modeling gets the attribute at load.

    Without the stamp the wrap would cast Zaya's residual and GC would fault in recompute.
    """
    from accelerate import PartialState
    from transformers.models.zaya import modeling_zaya

    from src.models.patches.zaya import patch_zaya_fp32_interlayer_residual

    PartialState()  # the zaya patch module logs through accelerate's logger
    patch_zaya_fp32_interlayer_residual()
    assert modeling_zaya.ZayaDecoderLayer._fp32_interlayer_residual is True


class _FooRMSNorm(nn.Module):
    """A norm by CLASS, at a module path with no "norm" in it."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))


class _Gate(nn.Module):
    """Not a norm, at a module path that does contain "norm"."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))


def test_the_peft_bf16_cast_classifies_norms_by_class_not_by_path():
    """The fp32 exemption belongs to norm MODULES, and a dotted path is not what makes one.

    Both directions corrupt silently: a family whose norms sit at a path without "norm" trains them
    in bf16, and an ordinary projection under a "norm"-named parent is upcast to fp32 and runs a
    dtype-mismatched matmul against bf16 activations.
    """
    model = nn.Module()
    model.scaler = _FooRMSNorm()
    model.norm_gate = _Gate()

    _peft_module_casting_to_bf16(model)

    assert model.scaler.weight.dtype == torch.float32, "a norm class outside a 'norm' path was not kept in fp32"
    assert model.norm_gate.weight.dtype == torch.bfloat16, "a non-norm module under a 'norm' path was upcast"


class _PerLayerTypeRotary(nn.Module):
    """Rotary exposing the ``rope_init_fns`` mapping layout (Gemma 4 / DeepSeek-V4 shape)."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.rope_type = {"full_attention": "default", "sliding_attention": "default"}
        # Only ONE of the two declared layer types ships a buffer, and it comes back from the load
        # in bf16 — the corruption this pass exists to repair.
        self.register_buffer("full_attention_inv_freq", torch.zeros(4, dtype=torch.bfloat16), persistent=False)
        self.rope_init_fns = {
            "full_attention": self._init_fn,
            "sliding_attention": self._init_fn,
        }

    @staticmethod
    def _init_fn(config, device=None, layer_type=None, **kwargs):
        return torch.full((4,), 3.0, device=device), 2.0


def test_the_per_layer_type_rotary_recompute_rebuilds_only_the_buffers_that_exist():
    """The shared helper must recompute each declared layer type's table in fp32 — and no other.

    Inventing a buffer for a layer type this instance does not use would put a table into
    ``state_dict``-adjacent state that the modeling never reads; leaving a real one at its loaded
    bf16 collapses adjacent RoPE positions past a few thousand tokens, silently.
    """
    model = nn.Module()
    model.rotary = _PerLayerTypeRotary()

    fix_rotary_inv_freq(model)

    assert model.rotary.full_attention_inv_freq.dtype == torch.float32
    assert torch.allclose(model.rotary.full_attention_inv_freq, torch.full((4,), 3.0))
    assert model.rotary.full_attention_attention_scaling == 2.0
    assert not hasattr(model.rotary, "sliding_attention_inv_freq"), (
        "a layer type with no buffer had one invented for it"
    )


class _LayerTypesRotary(nn.Module):
    """Rotary exposing the ``layer_types`` + ``rope_type``-dict layout (Laguna / DeepSeek-V4 shape)."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            rope_parameters={"full_attention": {"rope_theta": 3.0}, "sliding_attention": None}
        )
        self.layer_types = ["full_attention", "sliding_attention"]
        # A null rope_parameters entry means the type is not rotated: the family registers neither a
        # buffer nor a rope_type for it, so only the rotated type ships one (in load-corrupted bf16).
        self.rope_type = {"full_attention": "default"}
        self.register_buffer("full_attention_inv_freq", torch.zeros(4, dtype=torch.bfloat16), persistent=False)

    def compute_default_rope_parameters(self, config, device=None, layer_type=None, **kwargs):
        # Indexes the entry the way every family's own init fn does — a null one raises here.
        return torch.full((4,), config.rope_parameters[layer_type]["rope_theta"], device=device), 2.0


def test_the_layer_types_rotary_recompute_skips_a_type_the_family_does_not_rotate():
    """The second layout obeys the same rule, and it is the layout where breaking it is fatal.

    A layer type whose rope entry is null has no buffer AND no rope_type, so recomputing it both
    invents state and calls the family's init fn on the null entry — turning a supported config
    (Laguna with a non-rotated layer type) into a load-time crash instead of a repaired table.
    """
    model = nn.Module()
    model.rotary = _LayerTypesRotary()

    fix_rotary_inv_freq(model)

    assert model.rotary.full_attention_inv_freq.dtype == torch.float32
    assert torch.allclose(model.rotary.full_attention_inv_freq, torch.full((4,), 3.0))
    assert model.rotary.full_attention_attention_scaling == 2.0
    assert not hasattr(model.rotary, "sliding_attention_inv_freq"), (
        "a layer type the family does not rotate had a table invented for it"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
