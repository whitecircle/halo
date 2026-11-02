#!/usr/bin/env python
"""Ulysses CP's model gate on a wrapped attention class: a flash attention label, a resolvable head count on each
side, and head counts ``cp_size`` divides. Each refusal fires on its own condition, named by its mechanism, and
the supported model passes.

Run: python tests/cpu/parallelism/test_cp_model_rules.py
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from halo.distributed.context_parallel.config import CPConfigError
from halo.distributed.context_parallel.model_rules import (
    SUPPORTED_ATTN_IMPLEMENTATIONS,
    reject_unsupported_ulysses_model,
)

Q_HEADS, KV_HEADS = 4, 2


def _qwen3(attn_label: str = SUPPORTED_ATTN_IMPLEMENTATIONS[0]) -> Qwen3ForCausalLM:
    """A tiny Qwen3 built eager, with ``attn_label`` as the backend the gate reads: transformers refuses to build
    a flash tree on a CPU host, and no attention kernel is called here."""
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=Q_HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=8,
        max_position_embeddings=32,
        attn_implementation="eager",
    )
    model = Qwen3ForCausalLM(config)
    model.config._attn_implementation = attn_label
    return model


def _without_q_heads() -> Qwen3ForCausalLM:
    """A model config that names no Q heads (the strict config class refuses ``None``, so a stand-in)."""
    model = _qwen3()
    model.config = SimpleNamespace(model_type="qwen3", _attn_implementation=SUPPORTED_ATTN_IMPLEMENTATIONS[0])
    return model


def _without_kv_heads() -> Qwen3ForCausalLM:
    """The attention module's config no longer names its KV heads, which the wrapper reads them off."""
    model = _qwen3()
    model.model.layers[0].self_attn.config = SimpleNamespace()
    return model


@pytest.mark.parametrize("label", SUPPORTED_ATTN_IMPLEMENTATIONS)
def test_every_flash_label_with_dividing_head_counts_passes(label):
    reject_unsupported_ulysses_model(_qwen3(label), cp_size=KV_HEADS)


@pytest.mark.parametrize(
    ("model", "cp_size", "match"),
    [
        (lambda: _qwen3("eager"), 2, r"Ulysses CP requires Flash Attention. Got attn_implementation='eager'"),
        (lambda: _qwen3("sdpa"), 2, r"Ulysses CP requires Flash Attention. Got attn_implementation='sdpa'"),
        # A hub kernel label names a kernel the CP attention never loads: it calls the flash_attn build.
        (
            lambda: _qwen3("kernels-community/flash-attn2"),
            2,
            r"Ulysses CP requires Flash Attention. Got attn_implementation='kernels-community/flash-attn2'",
        ),
        (_without_q_heads, 2, r"Model config must have num_attention_heads for Ulysses CP"),
        (_without_kv_heads, 2, r"Cannot resolve the KV head count for Qwen3Attention under Ulysses CP"),
        (_qwen3, 3, r"^Q heads \(4\) must be divisible by context_parallel_size \(3\)"),
        (_qwen3, 4, r"^KV heads \(2\) must be divisible by context_parallel_size \(4\)"),
    ],
    ids=[
        "eager",
        "sdpa",
        "hub-kernel-label",
        "no-q-heads",
        "no-kv-heads",
        "q-heads-indivisible",
        "kv-heads-indivisible",
    ],
)
def test_each_refusal_names_its_mechanism(model, cp_size, match):
    with pytest.raises(CPConfigError, match=match):
        reject_unsupported_ulysses_model(model(), cp_size)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
