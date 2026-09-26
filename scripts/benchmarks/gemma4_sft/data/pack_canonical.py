"""Long-sequence protocol rows: consecutive canonical rows (BENCH_ROWS of them) joined into rows of BENCH_SEQ tokens.

canonical_tokens_<SEQ>.jsonl  row i = canonical rows k*i .. k*i+k-1 concatenated (k = SEQ / 2048): input_ids and
                              labels joined as they are, so each row is one plain causal sequence of SEQ tokens
                              (every framework attends across the joined conversations the same way).
messages_<SEQ>.jsonl          the matching message lists joined, for frameworks that key their input on messages.
"""

import hashlib
import json
import os
from pathlib import Path

BENCH_ROOT = os.environ["BENCH_ROOT"]  # see paths.env
SEQ = int(os.environ["BENCH_SEQ"])
SUFFIX = os.environ.get("BENCH_ROWS_SUFFIX", "")
BASE = 2048
if SEQ % BASE:
    raise SystemExit(f"BENCH_SEQ must be a multiple of {BASE}, got {SEQ}")
k = SEQ // BASE
data = Path(f"{BENCH_ROOT}/data")
canon = [json.loads(line) for line in (data / f"canonical_tokens{SUFFIX}.jsonl").read_text().splitlines()]
msgs = [json.loads(line)["messages"] for line in (data / f"messages{SUFFIX}.jsonl").read_text().splitlines()]
n = len(canon) // k
tok_lines, msg_lines = [], []
for i in range(n):
    part = canon[k * i : k * (i + 1)]
    ids = [t for r in part for t in r["input_ids"]]
    labels = [t for r in part for t in r["labels"]]
    tok_lines.append(json.dumps({"row": i, "input_ids": ids, "labels": labels}))
    msg_lines.append(json.dumps({"messages": [m for j in range(k * i, k * (i + 1)) for m in msgs[j]]}))
for name, lines in ((f"canonical_tokens_{SEQ}.jsonl", tok_lines), (f"messages_{SEQ}.jsonl", msg_lines)):
    path = data / name
    path.write_text("\n".join(lines) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (data / f"{name}.sha256").write_text(f"{digest}  {name}\n")
    print(name, n, "rows of", SEQ, "tokens", digest)
