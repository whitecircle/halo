"""Tiny native dense hybrid fixtures for CP2 layer and trainer qualification."""

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig


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
