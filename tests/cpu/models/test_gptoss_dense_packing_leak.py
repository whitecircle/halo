#!/usr/bin/env python
"""gpt-oss packed documents LEAK on dense attention backends — pin the defect and the gate.

``GptOssModel.forward`` builds its causal masks from a ``mask_kwargs`` dict that omits
``position_ids``, so ``masking_utils`` never sees the packed boundaries and a packed row runs as
ONE dense causal sequence on eager/SDPA/flex. The collator
factory therefore REFUSES packing + non-varlen for this family (``DENSE_PACKING_LEAK_MODEL_TYPES``)
— the varlen kernels are its production path and do isolate (they read cu_seqlens from
position_ids delivered at the attention interface).

The defect-pin half asserts the leak EXISTS: when a transformers upgrade fixes gpt-oss upstream,
it fails first — the signal to drop the family from the table and retire the gate.

    python tests/cpu/models/test_gptoss_dense_packing_leak.py
"""

from unittest.mock import MagicMock

import pytest
import torch
from accelerate import PartialState
from transformers import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

from src.data.collators.factory import DENSE_PACKING_LEAK_MODEL_TYPES, select_data_collator
from src.data.collators.packing import DataCollatorWithPacking

PartialState()  # the factory logs through accelerate's logger, which needs the state initialized


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
    """Doc B's logits MUST move when doc A changes on a dense backend — the leak this gate exists
    for.

    Direction matters: doc A precedes doc B, so causality already hides doc B from doc A; the leak
    is doc B attending BACK into doc A. If this drift ever reads 0.0, transformers fixed gpt-oss's
    mask plumbing upstream — remove "gpt_oss" from DENSE_PACKING_LEAK_MODEL_TYPES and retire this
    test."""
    torch.manual_seed(SEED)
    model = GptOssForCausalLM(_tiny_config()).eval()
    # Post-construction write: 5.14.1's _sdpa_can_dispatch refuses gpt-oss at _from_config, but the
    # mask math this probes is the same one a future dispatchable build would run.
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
        "upstream; remove 'gpt_oss' from DENSE_PACKING_LEAK_MODEL_TYPES and retire this defect pin"
    )


@pytest.mark.parametrize("attn_impl", ["sdpa", "eager"])
def test_factory_refuses_dense_packing_for_gpt_oss(attn_impl):
    config = _tiny_config()
    config._attn_implementation = attn_impl
    with pytest.raises(ValueError, match="refused"):
        select_data_collator(_mock_tokenizer(), packing=True, model_config=config)


def test_factory_refuses_the_composite_wrapper_too():
    """A VLM wrapper around a leaking family must not slip the gate on its wrapper-level
    model_type — ``model_type_matches`` reads the text sub-config."""

    class _CompositeConfig:
        model_type = "gpt_oss_mm"
        _attn_implementation = "sdpa"

        def __init__(self):
            self.text_config = _tiny_config()

        def get_text_config(self):
            return self.text_config

    with pytest.raises(ValueError, match="refused"):
        select_data_collator(_mock_tokenizer(), packing=True, model_config=_CompositeConfig())


def test_factory_allows_varlen_packing_for_gpt_oss():
    config = _tiny_config()
    config._attn_implementation = "flash_attention_2"
    collator = select_data_collator(_mock_tokenizer(), packing=True, model_config=config)
    assert isinstance(collator, DataCollatorWithPacking)


def test_factory_only_warns_for_isolating_families_on_dense():
    """A family whose dense mask path DOES isolate (mistral4, drift 0.0 — see
    test_mistral4_packed_isolation) keeps the warn-not-raise behavior."""
    from transformers import Mistral4Config

    config = Mistral4Config(num_hidden_layers=1)
    config._attn_implementation = "sdpa"
    assert config.model_type not in DENSE_PACKING_LEAK_MODEL_TYPES
    collator = select_data_collator(_mock_tokenizer(), packing=True, model_config=config)
    assert isinstance(collator, DataCollatorWithPacking)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
