"""Build the shared benchmark dataset: UltraChat-200k train_sft conversations concatenated in order until
the Gemma 4 chat template renders >= 2300 tokens, BENCH_ROWS rows (512 = the protocol rows). Deterministic (no
shuffling), so any larger BENCH_ROWS starts with the same 512 rows."""

import json
import os

from datasets import load_dataset
from transformers import AutoTokenizer

BENCH_ROOT = os.environ["BENCH_ROOT"]  # see paths.env
ROWS = int(os.environ.get("BENCH_ROWS", "512"))
SUFFIX = os.environ.get("BENCH_ROWS_SUFFIX", "")

tok = AutoTokenizer.from_pretrained("google/gemma-4-26B-A4B-it")
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
out, cur = [], []
for ex in ds:
    cur = cur + ex["messages"]
    ids = tok.apply_chat_template(cur, tokenize=True)
    ids = ids["input_ids"] if not isinstance(ids, list) else ids
    if len(ids) >= 2300:
        out.append({"messages": cur})
        cur = []
        if len(out) == ROWS:
            break
with open(f"{BENCH_ROOT}/data/ultrachat_gemma4_2k{SUFFIX}.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(len(out))
