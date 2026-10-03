"""Offline, deterministic tokenizer and grouped rows for offline-GRPO trainer GPU tests."""

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3MoeConfig, Qwen3MoeForCausalLM

from tests.common.models import TINY_QWEN3_MOE_CONFIG

OFFLINE_WORDS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet")
OFFLINE_VOCAB = {
    word: index
    for index, word in enumerate(
        ("<pad>", "<eos>", "<unk>", "question", "answer", "yes", "no", "maybe", *OFFLINE_WORDS)
    )
}
OFFLINE_QWEN3_MOE_CONFIG = TINY_QWEN3_MOE_CONFIG | {
    "vocab_size": len(OFFLINE_VOCAB),
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 32,
    "pad_token_id": OFFLINE_VOCAB["<pad>"],
    "eos_token_id": OFFLINE_VOCAB["<eos>"],
}


def make_offline_tokenizer():
    """A local-only tokenizer whose IDs and EOS behavior match the tiny model checkpoint."""
    tokenizer = Tokenizer(WordLevel(OFFLINE_VOCAB, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        model_max_length=128,
    )


def save_offline_moe_base(path, seed):
    """A shared local-only tiny MoE checkpoint for full-FT and native expert-LoRA GPU lifecycles."""
    torch.manual_seed(seed)
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**OFFLINE_QWEN3_MOE_CONFIG))
    model.to(torch.bfloat16).save_pretrained(path)
    make_offline_tokenizer().save_pretrained(path)


def offline_grpo_dataset(groups, offset=0):
    """Two unequal-length, oppositely rewarded completions for each distinct prompt group."""
    records = []
    for index in range(groups):
        word = OFFLINE_WORDS[(index + offset) % len(OFFLINE_WORDS)]
        other = OFFLINE_WORDS[(index + offset + 1) % len(OFFLINE_WORDS)]
        records.append(
            {
                "prompt": f"question {word} answer",
                "completions": [f"yes {word}", f"no {other} maybe"],
                "rewards": [1.0, -1.0],
            }
        )
    return Dataset.from_list(records)
