#!/usr/bin/env python
"""The optimizer-resume GPU fixture writes a tied vocabulary that native TP can shard."""

import pytest

from tests.common.utils import probe_findings


def test_optimizer_resume_checkpoint_pads_an_odd_tokenizer_vocab():
    script = """
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models
from transformers import AutoTokenizer, PreTrainedTokenizerFast, Qwen3ForCausalLM, Qwen3MoeForCausalLM

from tests.common.tiny_models import VOCAB_PAD_MULTIPLE
from tests.common.utils import load_script_module

suite = load_script_module("tests/gpu/parallelism/ep/test_ep_optimizer_resume.py")
vocab = {"<unk>": 0, "<eos>": 1, **{f"token{i}": i + 2 for i in range(63)}}
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="<unk>")),
    unk_token="<unk>", eos_token="<eos>", pad_token="<eos>",
)
assert len(tokenizer) % 2 == 1, "the fixture must catch an unshardable TP2 vocabulary"
expected_vocab = -(-len(tokenizer) // VOCAB_PAD_MULTIPLE) * VOCAB_PAD_MULTIPLE
findings = []
with tempfile.TemporaryDirectory() as root:
    for mode, model_class in (("tp", Qwen3ForCausalLM), ("ep", Qwen3MoeForCausalLM)):
        checkpoint = str(Path(root, mode))
        with patch.object(suite.AutoTokenizer, "from_pretrained", return_value=tokenizer):
            suite._build_tiny_checkpoint(mode, checkpoint)
        restored = model_class.from_pretrained(checkpoint, dtype=torch.bfloat16, attn_implementation="eager")
        actual = (restored.config.vocab_size, restored.get_input_embeddings().num_embeddings,
                  restored.get_output_embeddings().out_features)
        if actual != (expected_vocab,) * 3:
            findings.append(f"{mode}: config/embedding/head vocab {actual}, expected {expected_vocab}")
        assert restored.get_input_embeddings().weight is restored.get_output_embeddings().weight
        assert len(AutoTokenizer.from_pretrained(checkpoint)) == len(tokenizer)
print("VOCAB_MISMATCHES:" + "|".join(findings))
"""
    assert not probe_findings(script, "VOCAB_MISMATCHES:")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
