#!/usr/bin/env python
"""Packed-document isolation on Mistral4. A flash backend isolates a packed row's documents only when
``Mistral4Attention`` hands ``position_ids`` to the attention interface; flash kernels cannot run on CPU, so a spy
checks the interface's inputs."""

import pytest
import torch
import transformers.models.mistral4.modeling_mistral4 as m4
from transformers import Mistral4Config
from transformers.models.mistral4.modeling_mistral4 import ALL_ATTENTION_FUNCTIONS, Mistral4ForCausalLM

SEED = 1234


def _tiny_model() -> Mistral4ForCausalLM:
    torch.manual_seed(SEED)
    config = Mistral4Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        kv_lora_rank=32,
        q_lora_rank=None,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=512,
    )
    return Mistral4ForCausalLM(config).eval()


def _packed_inputs(doc_lens: tuple[int, ...], vocab: int = 128, seed: int = 7):
    """One flattened row with per-document position_ids, the collator's shape."""
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(4, vocab, (1, sum(doc_lens)), generator=g)
    position_ids = torch.cat([torch.arange(n) for n in doc_lens]).unsqueeze(0)
    return input_ids, position_ids


class _SpyInterface:
    """Records the ``position_ids`` each call saw, then answers with the module's own eager."""

    def __init__(self):
        self.position_ids: list[torch.Tensor | None] = []

    def __call__(self, module, query, key, value, attention_mask, **kwargs):
        self.position_ids.append(kwargs.pop("position_ids", None))
        return m4.eager_attention_forward(module, query, key, value, attention_mask, **kwargs)


def test_flash_interface_receives_position_ids():
    model = _tiny_model()
    input_ids, position_ids = _packed_inputs((5, 4, 3))

    spy = _SpyInterface()
    ALL_ATTENTION_FUNCTIONS.register("flash_spy", spy)
    model.config._attn_implementation = "flash_spy"
    model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
    received = spy.position_ids
    assert len(received) == model.config.num_hidden_layers and all(
        seen is not None and torch.equal(seen, position_ids) for seen in received
    ), (
        "Mistral4Attention dropped position_ids before the flash interface — packed documents attend across each "
        "other on every shipped mistral4 recipe (FA2 + packing)"
    )


def test_dense_path_isolates_documents():
    """Doc A precedes doc B, so causality already hides B from A; a broken packed mask lets B attend back into A.
    Pins the masking_utils packed-mask route Mistral4 relies on for dense backends."""
    model = _tiny_model()
    lens = (6, 5)
    input_ids, position_ids = _packed_inputs(lens)
    variant = input_ids.clone()
    variant[0, : lens[0]] = (variant[0, : lens[0]] + 17) % 128

    model.config._attn_implementation = "sdpa"
    with torch.no_grad():
        base = model(input_ids=input_ids, position_ids=position_ids, use_cache=False).logits
        swapped = model(input_ids=variant, position_ids=position_ids, use_cache=False).logits

    # Doc A's tokens change each expert's grouped-GEMM batch composition, which reorders fp32 reductions for doc B
    # (~1e-7). A mask leak reads ~1e-1.
    drift = (base[0, lens[0] :] - swapped[0, lens[0] :]).abs().max().item()
    assert drift < 1e-6, f"doc B's logits moved {drift:.2e} when doc A changed — dense packed mask broken"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
