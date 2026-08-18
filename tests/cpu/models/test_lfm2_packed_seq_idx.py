#!/usr/bin/env python
"""LFM2 packed documents cross the ShortConv unless ``seq_idx`` is supplied.

LFM2's conv mixer reads per-token segment ids from forward kwargs (``seq_idx``); both its
causal_conv1d fast path and the per-segment fallback honor them, and nothing model-side derives
them. The packing/padding-free collators emit the key when the factory sees an LFM2 config —
without it the conv carries state across document boundaries on EVERY attention backend, with
attention itself correctly isolated.

    python tests/cpu/models/test_lfm2_packed_seq_idx.py
"""

# The device-aware kernel-dispatch shim installs on `src` import and must land BEFORE the lfm2
# modeling module below, which binds transformers' hub-kernel fallback factory at import. Ordered
# after it (where isort puts first-party), this file's CPU forward reaches the CUDA-only
# causal_conv1d kernel and dies with `Expected x.is_cuda()` on a standalone run.
import src.models.patches.kernel_dispatch  # noqa: F401  # isort: skip

from unittest.mock import MagicMock

import pytest
import torch
from accelerate import PartialState
from transformers.models.lfm2_moe import Lfm2MoeConfig
from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeForCausalLM
from transformers.models.qwen3 import Qwen3Config

from src.data.collators.factory import select_data_collator

PartialState()  # the factory logs through accelerate's logger, which needs the state initialized


SEED = 1234


def _tiny_model() -> Lfm2MoeForCausalLM:
    torch.manual_seed(SEED)
    config = Lfm2MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_dense_layers=0,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=256,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
    )
    model = Lfm2MoeForCausalLM(config).eval()
    model.config._attn_implementation = "sdpa"
    return model


def _mock_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    tok.padding_side = "right"

    def _pad(features, **kwargs):
        max_len = max(len(f["input_ids"]) for f in features)
        out = {"input_ids": [], "attention_mask": []}
        for f in features:
            ids = list(f["input_ids"])
            pad = max_len - len(ids)
            out["input_ids"].append(ids + [tok.pad_token_id] * pad)
            out["attention_mask"].append([1] * len(ids) + [0] * pad)
        return {key: torch.tensor(value) for key, value in out.items()}

    tok.pad.side_effect = _pad
    return tok


def _doc_b_drift(model, seq_idx: torch.Tensor | None) -> float:
    """Doc B's max logit drift when doc A's content changes (doc A precedes doc B, so causality
    alone never explains a drift — only conv/attention state crossing the boundary does)."""
    lens = (6, 6)
    g = torch.Generator().manual_seed(7)
    input_ids = torch.randint(4, 128, (1, sum(lens)), generator=g)
    position_ids = torch.cat([torch.arange(n) for n in lens]).unsqueeze(0)
    variant = input_ids.clone()
    variant[0, : lens[0]] = (variant[0, : lens[0]] + 17) % 128

    kwargs = {} if seq_idx is None else {"seq_idx": seq_idx}
    with torch.no_grad():
        base = model(input_ids=input_ids, position_ids=position_ids, use_cache=False, **kwargs).logits
        swapped = model(input_ids=variant, position_ids=position_ids, use_cache=False, **kwargs).logits
    return (base[0, lens[0] :] - swapped[0, lens[0] :]).abs().max().item()


def test_seq_idx_isolates_the_conv():
    """Without seq_idx the conv leaks (the defect pin — if this half fails, transformers made LFM2
    derive segments itself and the collator emission can retire); with it, isolation is exact."""
    model = _tiny_model()
    position_ids = torch.cat([torch.arange(6), torch.arange(6)]).unsqueeze(0)
    seq_idx = ((position_ids == 0).cumsum(dim=1) - 1).to(torch.int32)

    assert _doc_b_drift(model, None) > 0.0, (
        "LFM2 conv isolated packed documents without seq_idx — transformers now derives segments "
        "model-side; retire the collator's seq_idx emission for the family"
    )
    assert _doc_b_drift(model, seq_idx) == 0.0, "seq_idx did not isolate the conv"


def test_factory_emits_seq_idx_for_lfm2_only():
    lfm2_config = _tiny_model().config
    collator = select_data_collator(_mock_tokenizer(), packing=True, model_config=lfm2_config)
    rows = [{"input_ids": [5, 6, 7, 8], "attention_mask": [1, 1, 1, 1], "seq_lengths": [2, 2]}]
    batch = collator.torch_call(rows)
    assert batch["seq_idx"].tolist() == [[0, 0, 1, 1]]
    assert batch["seq_idx"].dtype == torch.int32

    # Through the factory, not a hand-built collator: the negative half must exercise the family
    # gate itself, or widening emit_seq_idx to every family would pass unnoticed.
    qwen3_config = Qwen3Config(num_hidden_layers=1)
    qwen3_config._attn_implementation = "flash_attention_2"
    other = select_data_collator(_mock_tokenizer(), packing=True, model_config=qwen3_config)
    assert "seq_idx" not in other.torch_call(rows)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
