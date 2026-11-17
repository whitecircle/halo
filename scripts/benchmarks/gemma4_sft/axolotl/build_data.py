"""Axolotl's copy of the protocol rows: the canonical tokens with an all-ones attention mask, in the
field order Axolotl's pre-tokenized json loader reads. No `type:` in the configs, so Axolotl uses the tokens as is.

Reads $BENCH_CANON, writes $BENCH_ROOT/axolotl/data_axolotl$BENCH_SEQ_SUFFIX.jsonl and checks it against the
recorded sha256 of its sequence length (see paths.env).
"""

import hashlib
import json
import os
from pathlib import Path

BENCH_ROOT = os.environ["BENCH_ROOT"]
SHA256 = {
    2048: "9835ddc6d50db117c76892164961d8a821738fce3a1316629774264cf147ee99",
    16384: "e06a75eab8e0d665e0700bd55fe636a0615cb4845c6e66d3638e6f2eff2daba5",
}[int(os.environ["BENCH_SEQ"])]
src = os.environ["BENCH_CANON"]
dst = os.path.join(BENCH_ROOT, f"axolotl/data_axolotl{os.environ['BENCH_SEQ_SUFFIX']}.jsonl")
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(src) as fin, open(dst, "w") as fout:
    for line in fin:
        row = json.loads(line)
        ids = row["input_ids"]
        fout.write(json.dumps({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": row["labels"]}) + "\n")
digest = hashlib.sha256(Path(dst).read_bytes()).hexdigest()
print(dst, digest, "OK" if digest == SHA256 else f"MISMATCH (expected {SHA256})")
if digest != SHA256:
    raise SystemExit(1)
