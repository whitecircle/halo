#!/usr/bin/env python
"""Mistral4, the text backbone of Mistral3 VLMs, is claimed by the EP, CP and TP registries.

Run: python tests/cpu/parallelism/test_mistral4_registries.py
"""

import pytest

from src.distributed.context_parallel.layers.mistral4 import Mistral4UlyssesAttention
from src.distributed.context_parallel.layers.registry import WRAPPER_CLASS_MAP
from src.distributed.expert_parallel.layers.mistral4 import EPMistral4MoELayer
from src.distributed.expert_parallel.patching import MOE_LAYER_MAP
from src.distributed.tensor_parallel.module_types import TP_SHARDABLE_ATTENTION_CLASSES


def test_ep_wraps_the_mistral4_moe_block():
    assert MOE_LAYER_MAP["Mistral4MoE"] is EPMistral4MoELayer


def test_cp_wraps_mistral4_attention():
    assert WRAPPER_CLASS_MAP["Mistral4Attention"] is Mistral4UlyssesAttention


def test_tp_shards_mistral4_attention():
    assert "Mistral4Attention" in TP_SHARDABLE_ATTENTION_CLASSES


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
