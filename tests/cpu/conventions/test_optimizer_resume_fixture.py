#!/usr/bin/env python
"""The tied-Qwen3 checkpoint the optimizer-resume GPU suite trains has a vocabulary native TP can shard."""

import pytest
import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM, Qwen3MoeForCausalLM

from tests.common.tiny_models import build_tied_qwen3_checkpoint, word_level_tokenizer

# 65 tokens with <unk> and <eos>: odd, so a TP2 embedding cannot split it until padded to the next multiple of 128.
ODD_VOCAB_WORDS = tuple(f"token{i}" for i in range(63))
PADDED_VOCAB = 128


@pytest.mark.parametrize(("moe", "model_class"), [(False, Qwen3ForCausalLM), (True, Qwen3MoeForCausalLM)])
def test_the_checkpoint_pads_an_odd_tokenizer_vocab_and_keeps_the_tie(tmp_path, moe, model_class):
    tokenizer = word_level_tokenizer(ODD_VOCAB_WORDS)
    assert len(tokenizer) % 2 == 1, "the fixture must catch an unshardable TP2 vocabulary"

    build_tied_qwen3_checkpoint(str(tmp_path), tokenizer, moe=moe, seed=0)

    restored = model_class.from_pretrained(tmp_path, dtype=torch.bfloat16, attn_implementation="eager")
    vocab = (
        restored.config.vocab_size,
        restored.get_input_embeddings().num_embeddings,
        restored.get_output_embeddings().out_features,
    )
    assert vocab == (PADDED_VOCAB,) * 3
    assert restored.get_input_embeddings().weight is restored.get_output_embeddings().weight
    assert len(AutoTokenizer.from_pretrained(tmp_path)) == len(tokenizer)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
