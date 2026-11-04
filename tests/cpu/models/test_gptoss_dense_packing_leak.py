#!/usr/bin/env python
"""gpt-oss packed documents leak on dense attention backends: pin the defect and the gate.
``GptOssModel.forward`` builds causal masks without ``position_ids``, so the collator factory refuses dense packing."""

from unittest.mock import MagicMock

import pytest
import torch
from transformers import GptOssConfig, Mistral4Config
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

from halo.data.collators.factory import select_data_collator
from halo.data.collators.packing import DataCollatorWithPacking
from halo.models.segment_markers import has_dense_packing_leak

SEED = 1234


def _tiny_config() -> GptOssConfig:
    return GptOssConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        sliding_window=8,
        max_position_embeddings=256,
    )


def _mock_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    tok.padding_side = "right"
    return tok


@pytest.mark.parametrize("attn_impl", ["sdpa", "eager"])
def test_dense_leak_defect_pin(attn_impl):
    """Doc A precedes doc B, so the leak is doc B attending back into doc A.
    At drift 0.0 transformers fixed the mask plumbing: clear ``GPT_OSS.dense_packing_leaks`` and retire this test."""
    torch.manual_seed(SEED)
    model = GptOssForCausalLM(_tiny_config()).eval()
    # Post-construction write: _sdpa_can_dispatch refuses gpt-oss at _from_config, and this probes the same mask math.
    model.config._attn_implementation = attn_impl

    lens = (6, 6)
    g = torch.Generator().manual_seed(7)
    input_ids = torch.randint(4, 128, (1, sum(lens)), generator=g)
    position_ids = torch.cat([torch.arange(n) for n in lens]).unsqueeze(0)
    variant = input_ids.clone()
    variant[0, : lens[0]] = (variant[0, : lens[0]] + 17) % 128

    with torch.no_grad():
        base = model(input_ids=input_ids, position_ids=position_ids, use_cache=False).logits
        swapped = model(input_ids=variant, position_ids=position_ids, use_cache=False).logits

    drift = (base[0, lens[0] :] - swapped[0, lens[0] :]).abs().max().item()
    assert drift > 1e-3, (
        f"gpt-oss {attn_impl} isolated packed documents — transformers fixed the mask plumbing "
        "upstream; clear GPT_OSS.dense_packing_leaks and retire this defect pin"
    )


@pytest.mark.parametrize("attn_impl", ["sdpa", "eager"])
def test_factory_refuses_dense_packing_for_gpt_oss(attn_impl):
    config = _tiny_config()
    config._attn_implementation = attn_impl
    with pytest.raises(ValueError, match="refused"):
        select_data_collator(_mock_tokenizer(), train_on_completions_only=False, packing=True, model_config=config)


def test_factory_refuses_the_composite_wrapper_too():
    """The family spec is read off the text sub-config, so a VLM wrapper's own model_type cannot slip the gate."""

    class _CompositeConfig:
        model_type = "gpt_oss_mm"
        _attn_implementation = "sdpa"

        def __init__(self):
            self.text_config = _tiny_config()

        def get_text_config(self):
            return self.text_config

    with pytest.raises(ValueError, match="refused"):
        select_data_collator(
            _mock_tokenizer(), train_on_completions_only=False, packing=True, model_config=_CompositeConfig()
        )


def test_factory_allows_varlen_packing_for_gpt_oss():
    config = _tiny_config()
    config._attn_implementation = "flash_attention_2"
    collator = select_data_collator(
        _mock_tokenizer(), train_on_completions_only=False, packing=True, model_config=config
    )
    assert isinstance(collator, DataCollatorWithPacking)


def test_factory_only_warns_for_isolating_families_on_dense():
    """mistral4's dense mask path isolates (drift 0.0, see test_mistral4_packed_isolation), so it only warns."""
    config = Mistral4Config(num_hidden_layers=1)
    config._attn_implementation = "sdpa"
    assert not has_dense_packing_leak(config)
    collator = select_data_collator(
        _mock_tokenizer(), train_on_completions_only=False, packing=True, model_config=config
    )
    assert isinstance(collator, DataCollatorWithPacking)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
