#!/usr/bin/env python
"""LFM2's conv mixer reads ``seq_idx`` from forward kwargs and nothing model-side derives it, so packed documents cross
the ShortConv on every attention backend unless the collators emit it."""

from unittest.mock import MagicMock

import pytest
import torch
from transformers.models.lfm2_moe import Lfm2MoeConfig
from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeForCausalLM
from transformers.models.qwen3 import Qwen3Config

from halo.data.collators.factory import select_data_collator

SEED = 1234


def _tiny_model(*, sparse_ffn: bool = True) -> Lfm2MoeForCausalLM:
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
    if not sparse_ffn:
        config.num_dense_layers = config.num_hidden_layers
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


def _doc_b_drift(
    model,
    seq_idx: torch.Tensor | None,
    *,
    conv_only: bool = False,
    token_shift: int = 17,
    perturb_doc_b: bool = False,
) -> float:
    """Doc B's max activation drift when doc A changes, or doc B with ``perturb_doc_b`` (the positive control)."""
    lens = (6, 6)
    g = torch.Generator().manual_seed(7)
    input_ids = torch.randint(4, 128, (1, sum(lens)), generator=g)
    position_ids = torch.cat([torch.arange(n) for n in lens]).unsqueeze(0)
    variant = input_ids.clone()
    changed = slice(lens[0], None) if perturb_doc_b else slice(None, lens[0])
    variant[0, changed] = (variant[0, changed] + token_shift) % 128

    kwargs = {} if seq_idx is None else {"seq_idx": seq_idx}
    conv_outputs = []
    handle = None
    if conv_only:
        handle = model.model.layers[0].conv.register_forward_hook(
            lambda _module, _inputs, output: conv_outputs.append(output.detach().clone())
        )
    try:
        with torch.no_grad():
            base = model(input_ids=input_ids, position_ids=position_ids, use_cache=False, **kwargs).logits
            swapped = model(input_ids=variant, position_ids=position_ids, use_cache=False, **kwargs).logits
    finally:
        if handle is not None:
            handle.remove()
    if conv_only:
        assert len(conv_outputs) == 2
        base, swapped = conv_outputs
    return (base[0, lens[0] :] - swapped[0, lens[0] :]).abs().max().item()


@pytest.mark.parametrize("sparse_ffn", [False, True], ids=["dense-logits", "moe-conv"])
@pytest.mark.parametrize("token_shift", [17, 31])
def test_seq_idx_isolates_the_conv(sparse_ffn, token_shift):
    # A's routing changes expert GEMM batch sizes, so MoE logits can differ by rounding alone; check its conv exactly.
    model = _tiny_model(sparse_ffn=sparse_ffn)
    position_ids = torch.cat([torch.arange(6), torch.arange(6)]).unsqueeze(0)
    seq_idx = ((position_ids == 0).cumsum(dim=1) - 1).to(torch.int32)

    check = {"conv_only": sparse_ffn, "token_shift": token_shift}
    assert _doc_b_drift(model, None, **check) > 0.0, (
        "LFM2 conv isolated packed documents without seq_idx — transformers now derives segments "
        "model-side; retire the collator's seq_idx emission for the family"
    )
    assert _doc_b_drift(model, seq_idx, **check) == 0.0, "seq_idx did not isolate the conv"
    assert _doc_b_drift(model, seq_idx, perturb_doc_b=True, **check) > 0.0


def test_factory_emits_seq_idx_for_lfm2_only():
    lfm2_config = _tiny_model().config
    collator = select_data_collator(
        _mock_tokenizer(), train_on_completions_only=False, packing=True, model_config=lfm2_config
    )
    rows = [{"input_ids": [5, 6, 7, 8], "attention_mask": [1, 1, 1, 1], "seq_lengths": [2, 2]}]
    batch = collator.torch_call(rows)
    assert batch["seq_idx"].tolist() == [[0, 0, 1, 1]]
    assert batch["seq_idx"].dtype == torch.int32

    # Through the factory, so widening segment_markers_for's seq_idx to every family would fail here.
    qwen3_config = Qwen3Config(num_hidden_layers=1)
    qwen3_config._attn_implementation = "flash_attention_2"
    other = select_data_collator(
        _mock_tokenizer(), train_on_completions_only=False, packing=True, model_config=qwen3_config
    )
    assert "seq_idx" not in other.torch_call(rows)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
