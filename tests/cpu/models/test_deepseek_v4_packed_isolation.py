#!/usr/bin/env python
"""DeepSeek-V4 packed-document isolation on the dense (eager) path, the family's only backend.
The masked layers need ``use_cache=False``; the KV compressors cross documents by construction."""

import pytest
import torch
from transformers.models.deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

from tests.common.models import TINY_DSV4_CONFIG

SEED = 1234

# Expert-batch composition noise (doc A's tokens change each expert's grouped-GEMM batch, which
# reorders fp32 reductions for doc B's tokens). Leaks read ~1e-1, four orders above.
NOISE_BOUND = 1e-5


def _drift(layer_types: list[str], use_cache: bool, lens: tuple[int, int] = (6, 6)) -> float:
    """Doc B's max logit drift when doc A's content changes, which only state crossing the boundary explains."""
    torch.manual_seed(SEED)
    config = DeepseekV4Config(**{**TINY_DSV4_CONFIG, "layer_types": layer_types})
    model = DeepseekV4ForCausalLM(config).eval()
    model.config._attn_implementation = "eager"

    g = torch.Generator().manual_seed(7)
    input_ids = torch.randint(4, 100, (1, sum(lens)), generator=g)
    position_ids = torch.cat([torch.arange(n) for n in lens]).unsqueeze(0)
    variant = input_ids.clone()
    variant[0, : lens[0]] = (variant[0, : lens[0]] + 17) % 100

    with torch.no_grad():
        base = model(input_ids=input_ids, position_ids=position_ids, use_cache=use_cache).logits
        swapped = model(input_ids=variant, position_ids=position_ids, use_cache=use_cache).logits
    return (base[0, lens[0] :] - swapped[0, lens[0] :]).abs().max().item()


def test_masked_attention_isolates_on_the_training_path():
    """Every trainer forward carries ``use_cache=False``, which isolates masked attention since transformers 5.14."""
    drift = _drift(["sliding_attention"] * 3, use_cache=False)
    assert drift < NOISE_BOUND, (
        f"packed documents attend across each other ({drift:.2e}) with use_cache=False — the "
        f"transformers' conditional-cache fix regressed, and every packed DSv4 run trains on "
        f"contaminated context"
    )


def test_cache_still_suppresses_the_packed_mask():
    """A live cache disables the packed mask in ``masking_utils``, so isolation rests on ``use_cache=False``.
    If this nears zero, transformers builds packed masks alongside a cache and the test above is moot."""
    assert _drift(["sliding_attention"] * 3, use_cache=True) > 1e-3


def test_compressed_attention_crosses_documents_by_construction():
    """One CSA layer leaks under a correct packed mask, since its KV windows are cut at fixed row indices.
    If this nears zero, re-examine the packing refusal and the isolation matrix in agent-docs/data/collators.md."""
    layer_types = ["sliding_attention", "compressed_sparse_attention", "sliding_attention"]
    assert _drift(layer_types, use_cache=False) > 1e-3


def test_heavily_compressed_attention_crosses_documents_by_construction():
    """Documents must outlast the compress rate (16), or the compressor has no windows and the probe reads isolated."""
    layer_types = ["sliding_attention", "heavily_compressed_attention", "sliding_attention"]
    assert _drift(layer_types, use_cache=False, lens=(160, 160)) > 1e-3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
