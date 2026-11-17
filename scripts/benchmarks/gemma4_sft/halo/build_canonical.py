"""Halo pre-sharded datasets from the protocol canonical tokens ($BENCH_CANON, BENCH_SEQ tokens per row): shard r
holds canonical rows r, r+2, ... so that with sequential sampling rank r trains row 2*(k-1)+r at step k. Two label
variants: canonical (assistant-only) and all-token (labels = input_ids), same rows and order. The last 6 rows form
the (unused) test split."""

import json
import os
import shutil
from pathlib import Path

from datasets import Dataset

BENCH_ROOT = os.environ["BENCH_ROOT"]  # see paths.env
SEQ = int(os.environ["BENCH_SEQ"])
SUFFIX = "" if SEQ == 2048 else f"_{SEQ}"

BASE = Path(f"{BENCH_ROOT}/halo/data/ultrachat_gemma4_2k_halo2048")
rows = [json.loads(line) for line in Path(os.environ["BENCH_CANON"]).read_text().splitlines()]
n_train = len(rows) - 6
for name, all_tokens in (("canonical", False), ("alltokens", True)):
    out = Path(f"{BENCH_ROOT}/halo/data/{name}{SUFFIX}")
    if out.exists():
        shutil.rmtree(out)
    recs = [
        {
            "input_ids": r["input_ids"],
            "attention_mask": [1] * SEQ,
            "labels": r["input_ids"] if all_tokens else r["labels"],
            "seq_lengths": [SEQ],
            "row": r["row"],
        }
        for r in rows
    ]
    for split, data in (("train", recs[:n_train]), ("test", recs[n_train:])):
        shards = [data[0::2], data[1::2]]
        for i, part in enumerate(shards):
            Dataset.from_list(part).save_to_disk(str(out / split / f"shard_{i:04d}"))
        idx = json.loads((BASE / split / "shard_index.json").read_text())
        for i, sh in enumerate(idx["shards"]):
            sh["num_examples"] = len(shards[i])
        idx["total_examples"] = len(data)
        (out / split / "shard_index.json").write_text(json.dumps(idx, indent=2))
    meta = json.loads((BASE / "metadata.json").read_text())
    meta.update(
        train_on_completions_only=not all_tokens, max_length=SEQ, total_train_examples=n_train, total_test_examples=6
    )
    meta["config"].update(train_on_completions_only=not all_tokens, max_length=SEQ)
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(name, [r["row"] for r in recs[0:6:2]], [r["row"] for r in recs[1:6:2]])
