"""Whitespace stand-ins for a VLM tokenizer and processor; ``__call__`` is the seam subclasses override per test."""

from __future__ import annotations

import torch


class FakeVLMTokenizer:
    """Minimal tokenizer stand-in: pad/eos ids, vocab lookup for image-token masking, vocab size."""

    pad_token_id = 0
    eos_token_id = 1

    def get_vocab(self) -> dict[str, int]:
        return {"<|image_pad|>": 99}

    def __len__(self) -> int:
        return 100  # The metadata stamp records the vocab size.


class FakeVLMProcessorBase:
    """A whitespace processor honouring the HF processor call contract."""

    # Recorded by the bake like a real processor's template.
    chat_template = "{% for m in messages %}{{ m.role }} : {{ m.content }} {% endfor %}"

    def __init__(self, tokenizer=None):
        self.tokenizer = FakeVLMTokenizer() if tokenizer is None else tokenizer

    def apply_chat_template(self, history, tokenize=False, add_generation_prompt=False, **kwargs) -> str:
        parts = []
        for msg in history:
            content = msg["content"]
            if isinstance(content, list):
                # The Arrow round-trip pads every item with the union of keys (text=None on images).
                content = " ".join(item.get("text") or "" for item in content)
            parts.append(f"{msg['role']} : {content}")
        return " ".join(parts)

    def encode_batch(self, text, max_length: int | None = None) -> dict[str, torch.Tensor]:
        """Whitespace ids + mask, right-padded to the batch's widest row, truncated when asked."""
        rows = [[2 + (j % 97) for j, _ in enumerate(t.split())] for t in text]
        if max_length is not None:
            rows = [row[:max_length] for row in rows]
        width = max(len(row) for row in rows)
        input_ids = torch.full((len(rows), width), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, row in enumerate(rows):
            input_ids[i, : len(row)] = torch.tensor(row)
            attention_mask[i, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def __call__(self, text, return_tensors="pt", images=None, videos=None, add_special_tokens=True, **kwargs):
        return self.encode_batch(text)
