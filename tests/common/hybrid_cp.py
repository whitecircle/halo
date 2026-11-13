"""Tiny native dense hybrid fixtures for CP2 layer and trainer qualification."""

import math
from dataclasses import dataclass

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from tests.common.numerics import cos_sim

GDN_OUTPUT_REL_L2 = 1e-2
GDN_GRAD_REL_L2 = 2e-2
GDN_GRAD_COSINE_MIN = 0.999
GDN_ZERO_ABS_L2 = 1e-12


@dataclass(frozen=True)
class HybridSGDStep:
    delta: torch.Tensor
    ideal: torch.Tensor
    rounding: torch.Tensor


def hybrid_sgd_step(before, gradient, after, learning_rate):
    """Require exact native-device/dtype SGD, then retain host doubles for the cross-arm rounding proof."""
    if (
        before.shape != gradient.shape
        or before.shape != after.shape
        or before.dtype != gradient.dtype
        or before.dtype != after.dtype
        or before.device != gradient.device
        or before.device != after.device
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
        or not bool(before.isfinite().all() & gradient.isfinite().all() & after.isfinite().all())
    ):
        raise AssertionError("SGD requires finite, same-shaped native-device/dtype snapshots and a positive rate")
    expected = before.clone().add_(gradient, alpha=-learning_rate)
    torch.testing.assert_close(after, expected, rtol=0, atol=0)
    ideal = -learning_rate * gradient.double()
    delta = after.double() - before.double()
    rounding = expected.double() - before.double() - ideal
    return HybridSGDStep(delta.cpu(), ideal.cpu(), rounding.cpu())


def hybrid_sgd_agreement(actual, expected):
    """Bound raw update disagreement only by ideal-gradient disagreement and predicted storage rounding."""
    if actual is None or any(
        got.shape != want.shape or not bool(got.isfinite().all() & want.isfinite().all())
        for got, want in (
            (actual.delta, expected.delta),
            (actual.ideal, expected.ideal),
            (actual.rounding, expected.rounding),
        )
    ):
        return False, {"finite_and_same_shape": False}
    gradient_difference = float((actual.ideal - expected.ideal).norm())
    actual_rounding, reference_rounding = float(actual.rounding.norm()), float(expected.rounding.norm())
    budget = gradient_difference + actual_rounding + reference_rounding
    absolute = float((actual.delta - expected.delta).norm())
    _, scores = hybrid_tensor_agreement(actual.delta, expected.delta, gradient=True)
    return absolute <= budget, scores | {
        "ideal_gradient_difference": gradient_difference,
        "actual_storage_rounding": actual_rounding,
        "reference_storage_rounding": reference_rounding,
        "derived_rounding_budget": budget,
    }


def hybrid_text_config(**overrides):
    values = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 8,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 4,
        "layer_types": ["linear_attention", "full_attention"],
        "max_position_embeddings": 128,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        "use_cache": False,
        "tie_word_embeddings": False,
    }
    return Qwen3_5TextConfig(**(values | overrides))


def build_hybrid_model(**overrides):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(114)
        return Qwen3_5ForCausalLM(hybrid_text_config(**overrides))


def hybrid_hidden_batch(batch, length, hidden_size, *, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(114_002)
    hidden = torch.randn(batch, length, hidden_size, generator=generator, device=device) / 3
    lengths = length - 1 - torch.arange(batch, device=device) % 3
    mask = torch.arange(length, device=device)[None, :] < lengths[:, None]
    targets = torch.linspace(-0.5, 0.7, hidden.numel(), device=device).reshape_as(hidden)
    return hidden, mask, targets


def build_gpu_hybrid_model(**overrides):
    """Build a native-kernel-sized hybrid whose distinct decay and bias values expose swapped head shards."""
    model = build_hybrid_model(
        hidden_size=128,
        intermediate_size=256,
        head_dim=32,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        **overrides,
    )
    with torch.no_grad():
        for layer in model.model.layers:
            if hasattr(layer, "linear_attn"):
                mixer = layer.linear_attn
                mixer.A_log.copy_(torch.linspace(-1.0, 0.2, mixer.num_v_heads))
                mixer.dt_bias.copy_(torch.linspace(-0.3, 0.5, mixer.num_v_heads))
    return model


def hybrid_token_batch(batch, vocabulary, *, boundary_only=False, length=130):
    """Return seeded CP2 rows with boundary-only or empty-first-shard supervision; reject unsupported lengths."""
    if length % 2 or length == 16386:
        raise ValueError("Use an even hybrid CP2 fixture length other than the excluded 16,386 vendor row")
    generator = torch.Generator().manual_seed(114)
    ids = torch.randint(3, vocabulary, (batch, length), generator=generator)
    mask = torch.arange(length)[None, :] < (length - 1 - 2 * torch.arange(batch))[:, None]
    labels = ids.masked_fill(~mask, -100)
    if boundary_only:
        labels.fill_(-100)
        labels[:, length // 2] = ids[:, length // 2]
    else:
        labels[:, : length // 2 + 1] = -100
    return {"input_ids": ids, "attention_mask": mask.long(), "labels": labels}


def hybrid_tensor_agreement(actual, expected, *, gradient=False):
    """Strict native-block bars, including absolute L2 for expected zero; no negligible-gradient exemption."""
    if (
        actual is None
        or actual.shape != expected.shape
        or not bool(torch.isfinite(actual).all() & torch.isfinite(expected).all())
    ):
        return False, {"finite_and_same_shape": False}
    actual, expected = actual.double(), expected.double()
    reference_norm, actual_norm = float(expected.norm()), float(actual.norm())
    absolute = float((actual - expected).norm())
    scores = {"reference_norm": reference_norm, "actual_norm": actual_norm, "absolute_l2": absolute}
    if reference_norm == 0:
        return absolute <= GDN_ZERO_ABS_L2, scores | {"expected_zero": True}
    relative = absolute / reference_norm
    cosine = cos_sim(actual, expected, label="hybrid native agreement") if actual_norm else 0.0
    scores |= {"relative_l2": relative, "cosine": cosine}
    passed = math.isfinite(relative) and relative <= (GDN_GRAD_REL_L2 if gradient else GDN_OUTPUT_REL_L2)
    return passed and (not gradient or cosine >= GDN_GRAD_COSINE_MIN), scores
