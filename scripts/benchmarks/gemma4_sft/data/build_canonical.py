"""The protocol data: one tokenization, one label mask, one row order for every framework.

messages.jsonl       the protocol rows (BENCH_ROWS, 512 by default; files carry the BENCH_ROWS_SUFFIX) with Gemma 4's empty thought channel written at the start of
                        every assistant message. transformers' Gemma 4 template strips it from history turns,
                        so it does not reach the canonical tokens below.
canonical_tokens.jsonl per row: input_ids (2,048) and labels (assistant spans through <turn|>
                        inclusive; -100 elsewhere), from apply_chat_template(tokenize=False) then
                        tokenizer(text, add_special_tokens=False), truncated to 2,048.
Order                   no shuffling; at optimizer step k (1-based) rank r trains row 2*(k-1)+r (the
                        DistributedSampler(shuffle=False) assignment with 2 ranks, micro-batch 1, GA 1).
"""

import hashlib
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

BENCH_ROOT = os.environ["BENCH_ROOT"]  # see paths.env
SUFFIX = os.environ.get("BENCH_ROWS_SUFFIX", "")

THOUGHT, MODEL_TURN, END_TURN = "<|channel>thought\n<channel|>", "<|turn>model\n", "<turn|>"
tok = AutoTokenizer.from_pretrained("google/gemma-4-26B-A4B-it")
rows_in = [
    json.loads(line) for line in Path(f"{BENCH_ROOT}/data/ultrachat_gemma4_2k{SUFFIX}.jsonl").read_text().splitlines()
]
with (
    open(f"{BENCH_ROOT}/data/messages{SUFFIX}.jsonl", "w") as fm,
    open(f"{BENCH_ROOT}/data/canonical_tokens{SUFFIX}.jsonl", "w") as ft,
):
    fracs = []
    for i, row in enumerate(rows_in):
        msgs = [dict(m, content=THOUGHT + m["content"]) if m["role"] == "assistant" else m for m in row["messages"]]
        fm.write(json.dumps({"messages": msgs}) + "\n")
        text = tok.apply_chat_template(msgs, tokenize=False)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"][:2048], enc["offset_mapping"][:2048]
        spans, pos = [], 0
        while (s := text.find(MODEL_TURN, pos)) >= 0:
            a = s + len(MODEL_TURN)
            e = text.find(END_TURN, a)
            e = len(text) if e < 0 else e + len(END_TURN)
            spans.append((a, e))
            pos = e
        labels = [t if any(a <= o0 < e for a, e in spans) else -100 for t, (o0, _) in zip(ids, offs, strict=False)]
        assert len(ids) == 2048
        fracs.append(sum(t != -100 for t in labels) / 2048)
        ft.write(json.dumps({"row": i, "input_ids": ids, "labels": labels}) + "\n")
print("rows", len(rows_in), "label fraction mean", round(sum(fracs) / len(fracs), 4), "min", round(min(fracs), 3))
for f in (f"messages{SUFFIX}.jsonl", f"canonical_tokens{SUFFIX}.jsonl"):
    h = hashlib.sha256(Path(f"{BENCH_ROOT}/data/{f}").read_bytes()).hexdigest()
    Path(f"{BENCH_ROOT}/data/{f}.sha256").write_text(f"{h}  {f}\n")
    print(f, h)
