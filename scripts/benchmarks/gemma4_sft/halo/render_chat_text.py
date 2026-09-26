#!/usr/bin/env python
"""Render the protocol {"messages": [...]} rows to {"text": ...} with the Gemma 4 chat template
(tokenizer's bundled template, no generation prompt). Halo's chat path drops rows longer than
max_length instead of truncating; the text path + bfd packing truncates each row to max_length."""

import json
import sys

from transformers import AutoTokenizer

src, dst, model = sys.argv[1], sys.argv[2], sys.argv[3]
tok = AutoTokenizer.from_pretrained(model)
n, lens = 0, []
with open(src) as f, open(dst, "w") as g:
    for line in f:
        row = json.loads(line)
        text = tok.apply_chat_template(row["messages"], tokenize=False)
        lens.append(len(tok(text, add_special_tokens=False)["input_ids"]))
        g.write(json.dumps({"text": text}) + "\n")
        n += 1
print(f"rendered {n} rows; min tokens {min(lens)}, max {max(lens)}, rows<2048: {sum(k < 2048 for k in lens)}")
