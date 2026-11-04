#!/usr/bin/env python
"""Mistral4, the text backbone of Mistral3 VLMs, is claimed by the EP, CP and TP registries."""

import pytest

from halo.distributed.context_parallel.families import ULYSSES_ATTENTION_MAP, Mistral4UlyssesAttention
from halo.distributed.expert_parallel.families.mistral4 import EPMistral4MoELayer
from halo.distributed.expert_parallel.families.roster import MOE_LAYER_MAP
from halo.distributed.tensor_parallel.parallelize_attention import TP_SHARDABLE_ATTENTION_CLASSES


def test_ep_wraps_the_mistral4_moe_block():
    assert MOE_LAYER_MAP["Mistral4MoE"] is EPMistral4MoELayer


def test_cp_wraps_mistral4_attention():
    assert ULYSSES_ATTENTION_MAP["Mistral4Attention"] is Mistral4UlyssesAttention


def test_tp_shards_mistral4_attention():
    assert "Mistral4Attention" in TP_SHARDABLE_ATTENTION_CLASSES


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
