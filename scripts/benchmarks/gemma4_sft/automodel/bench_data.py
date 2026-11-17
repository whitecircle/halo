"""the protocol data for the AutoModel VLM recipe: canonical pre-tokenized rows fed directly.

Canonical rows are HF-style (labels aligned with input_ids; the loss shifts). AutoModel's recipe expects
pre-shifted labels (its own collate does labels[:, 1:] / input_ids[:, :-1]), so we feed
input_ids = canonical input_ids (BENCH_SEQ tokens) and labels = canonical labels[1:] + [-100],
i.e. exactly the next-token targets HF's shifted CE would use on the same inputs.
No tokenizer, no chat template, no thought-prefix injection is applied.
"""

import json
import os

import torch


def make_canonical_dataset(path_or_dataset: str, split: str = "train", **kwargs):
    rows = []
    with open(path_or_dataset) as f:
        for line in f:
            r = json.loads(line)
            rows.append({"row": r["row"], "input_ids": r["input_ids"], "labels": r["labels"]})
    rows.sort(key=lambda r: r["row"])
    assert [r["row"] for r in rows] == list(range(len(rows)))
    return rows


def canonical_collate_fn(examples, processor=None, **kwargs):
    ids = torch.tensor([e["input_ids"] for e in examples], dtype=torch.long)
    lab = torch.tensor([e["labels"][1:] + [-100] for e in examples], dtype=torch.long)
    log = os.environ.get("BENCH_ORDER_LOG")
    if log:
        with open(f"{log}.rank{os.environ.get('RANK', '0')}", "a") as f:
            f.write(json.dumps([e["row"] for e in examples]) + "\n")
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": lab}
