#!/usr/bin/env python
"""DeepSeek-V4 packed-document isolation on the dense (eager) path — the family's only backend.

Two independent mechanisms, pinned separately:

1. The packed mask. ``masking_utils`` builds the packed mask only when ``past_key_values is None``,
   so an unconditional ``DynamicCache`` in ``DeepseekV4Model.forward`` runs every packed row as ONE
   dense causal sequence. transformers >= 5.14 makes that cache conditional on ``use_cache``, which the
   training path always runs with ``False`` (``TrainingArguments.use_cache`` defaults False and
   ``Trainer.__init__`` writes it into ``model.config``; TRL's SFT ``compute_loss`` also forces the
   kwarg).
2. The KV compressor. ``compressed_sparse_attention`` layers pool KV spans across the whole row
   with no document awareness, so they cross packed boundaries BY CONSTRUCTION even under a
   correct packed mask — the same accepted class as the linear-attention/conv mixers
   (see ``agent-docs/data/collators.md``).

    python tests/cpu/models/test_deepseek_v4_packed_isolation.py
"""

import pytest
import torch
from transformers.models.deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

from tests.common.models import TINY_DSV4_CONFIG

SEED = 1234

# Expert-batch composition noise (doc A's tokens change each expert's grouped-GEMM batch, which
# reorders fp32 reductions for doc B's tokens). Leaks read ~1e-1, five orders above.
NOISE_BOUND = 1e-5


def _drift(layer_types: list[str], use_cache: bool, lens: tuple[int, int] = (6, 6)) -> float:
    """Doc B's max logit drift when doc A's content changes (doc A precedes doc B, so causality
    alone never explains a drift — only state crossing the boundary does)."""
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
    """use_cache=False — the invariant every trainer forward carries — must isolate the masked
    attention layers (the conditional-cache fix, transformers >= 5.14)."""
    drift = _drift(["sliding_attention"] * 3, use_cache=False)
    assert drift < NOISE_BOUND, (
        f"packed documents attend across each other ({drift:.2e}) with use_cache=False — the "
        f"transformers' conditional-cache fix regressed, and every packed DSv4 run trains on "
        f"contaminated context"
    )


def test_cache_still_suppresses_the_packed_mask():
    """The mechanism pin: a live cache disables packed-mask synthesis, so isolation rests entirely
    on use_cache=False reaching the forward. If this ever nears zero, transformers builds packed
    masks alongside a cache and the invariant above stops being load-bearing — re-examine both."""
    assert _drift(["sliding_attention"] * 3, use_cache=True) > 1e-3


def test_compressed_attention_crosses_documents_by_construction():
    """The compressor pin: one ``compressed_sparse_attention`` layer leaks regardless of the packed
    mask. If this ever nears zero, upstream made the compressor document-aware — update the
    per-family isolation matrix in agent-docs/data/collators.md."""
    layer_types = ["sliding_attention", "compressed_sparse_attention", "sliding_attention"]
    assert _drift(layer_types, use_cache=False) > 1e-3


def test_heavily_compressed_attention_crosses_documents_by_construction():
    """Same pin for HCA — the config default alternates CSA and HCA, so ~half a real stack is HCA.
    Documents must be longer than the compress rate (16) or the compressor degenerates to zero
    windows and the probe reads vacuously isolated."""
    layer_types = ["sliding_attention", "heavily_compressed_attention", "sliding_attention"]
    assert _drift(layer_types, use_cache=False, lens=(160, 160)) > 1e-3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
