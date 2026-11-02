#!/usr/bin/env python
"""Policies the model-patch layer DERIVES rather than lists, and the one it deliberately does not.

Each of these decides something silent or late when it is wrong — an attention backend a run cannot train on, a
recompute dtype mismatch, a norm quantized to bf16, a RoPE table left as load garbage — so the pins
here are on the derivation itself, not on a call reaching some branch:

* FA4's head_dim-256 refusal reads the query/key width, MLA's split included (transformers pads the value
  head up to it), and holds only where transformers asks for a deterministic backward;
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
from transformers.integrations import flash_attention
from transformers.models.zaya import modeling_zaya

from halo.distributed.fsdp import _should_cast_forward_inputs
from halo.model_loading.buffer_repair import finalize_loaded_model
from halo.model_loading.parallel.peft_setup import _peft_module_casting_to_bf16
from halo.models.attention.layout import resolve_qk_v_head_dims
from halo.models.attention.selection import supports_fa4_backward
from halo.models.zaya import patch_zaya_fp32_interlayer_residual


def _config(model_type: str):
    return transformers.CONFIG_MAPPING[model_type]()


# The families FA4 trains on its dedicated head_dim-256 kernel.
_FA4_HEAD_DIM_256_MODEL_TYPES = ("qwen3_next", "qwen3_5", "qwen3_5_moe")
# The head_dim-256 families that kernel does not train: GLM-4.7-Flash's latent attention, Gemma 3's windows and
# Gemma 2's softcap.
_FA4_REFUSED_HEAD_DIM_256_MODEL_TYPES = ("glm4_moe_lite", "gemma3_text", "gemma2")


@pytest.mark.parametrize("model_type", _FA4_HEAD_DIM_256_MODEL_TYPES + _FA4_REFUSED_HEAD_DIM_256_MODEL_TYPES)
def test_a_deterministic_run_refuses_fa4_at_head_dim_256(model_type, monkeypatch):
    """The dedicated kernel's backward asserts ``not deterministic``, which transformers passes under
    ``FLASH_ATTENTION_DETERMINISTIC``: such a run would stop at its first backward."""
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    assert supports_fa4_backward(_config(model_type)) is False


@pytest.mark.parametrize("model_type", _FA4_HEAD_DIM_256_MODEL_TYPES)
def test_head_dim_256_keeps_fa4_outside_a_deterministic_run(model_type, monkeypatch):
    """With q/k/v made contiguous for that kernel, nothing else refuses it."""
    monkeypatch.delenv("FLASH_ATTENTION_DETERMINISTIC", raising=False)
    assert supports_fa4_backward(_config(model_type)) is True


@pytest.mark.parametrize("model_type", _FA4_REFUSED_HEAD_DIM_256_MODEL_TYPES)
def test_fa4_does_not_train_a_refused_head_dim_256_attention_in_any_run(model_type, monkeypatch):
    """The dedicated kernel's refusals hold outside a deterministic run too, so Context Parallelism, whose Ulysses
    kernel reads this verdict, never runs FA4 on them."""
    monkeypatch.delenv("FLASH_ATTENTION_DETERMINISTIC", raising=False)
    assert supports_fa4_backward(_config(model_type)) is False


@pytest.mark.parametrize("model_type", ["qwen3", "llama", "gpt_oss", "mistral4"])
def test_the_deterministic_refusal_stays_off_other_head_widths(model_type, monkeypatch):
    """Every other width runs FA4's general kernel, which has a deterministic backward."""
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    assert supports_fa4_backward(_config(model_type)) is True


def test_transformers_pads_the_value_head_to_the_query_key_width(monkeypatch):
    """Premise of the query/key-only verdict: transformers' flash path hands the kernel a value as wide as the
    query (MLA's narrower value padded) and crops the output back."""
    widths = []

    def kernel(query, key, value, *args, **kwargs):
        widths.append((query.shape[-1], key.shape[-1], value.shape[-1]))
        return torch.zeros_like(value)

    monkeypatch.setattr(flash_attention, "_flash_attention_forward", kernel)
    module = SimpleNamespace(config=SimpleNamespace(_attn_implementation="flash_attention_4"), is_causal=True)
    query, key = (torch.zeros(1, 2, 3, 256, dtype=torch.bfloat16) for _ in range(2))
    value = torch.zeros(1, 2, 3, 128, dtype=torch.bfloat16)
    out, _ = flash_attention.flash_attention_forward(module, query, key, value, None)
    assert widths == [(256, 256, 256)]
    assert out.shape[-1] == 128


class _NarrowValueConfig(transformers.PretrainedConfig):
    """A 256-wide query/key head over a 128-wide value, with no latent (so no latent-attention refusal)."""

    model_type = "halo_test_narrow_value"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size, self.num_attention_heads, self.head_dim, self.v_head_dim = 2048, 16, 256, 128


@pytest.mark.parametrize(("deterministic", "supported"), [("1", False), ("0", True)])
def test_the_deterministic_refusal_reads_the_query_key_width_alone(deterministic, supported, monkeypatch):
    """A 256-wide q/k over a 128-wide v reaches the dedicated kernel with its value padded to 256, so a
    deterministic run refuses FA4 there too, and an ordinary one keeps it."""
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", deterministic)
    config = _NarrowValueConfig()
    assert resolve_qk_v_head_dims(config) == (256, 128)
    assert supports_fa4_backward(config) is supported


@pytest.mark.parametrize(
    ("model_type", "declared_head_dim", "widths"),
    [("glm4_moe_lite", 64, (256, 256)), ("mistral4", 128, (128, 128)), ("qwen3_next", 256, (256, 256))],
)
def test_the_head_widths_count_the_mla_split(model_type, declared_head_dim, widths):
    """MLA declares a ``head_dim`` that is only its RoPE slice (GLM-4.7-Flash: 64) and the query/key width as
    ``qk_nope_head_dim + qk_rope_head_dim``, so reading ``head_dim`` alone would hand it back FA4's deterministic
    assert."""
    config = _config(model_type)
    assert config.head_dim == declared_head_dim
    assert resolve_qk_v_head_dims(config) == widths


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

    finalize_loaded_model(model)

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

    finalize_loaded_model(model)

    assert model.rotary.full_attention_inv_freq.dtype == torch.float32
    assert torch.allclose(model.rotary.full_attention_inv_freq, torch.full((4,), 3.0))
    assert model.rotary.full_attention_attention_scaling == 2.0
    assert not hasattr(model.rotary, "sliding_attention_inv_freq"), (
        "a layer type the family does not rotate had a table invented for it"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
