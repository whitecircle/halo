"""Unsloth vs HF on canonical rows 0,1: forward loss in different call modes (1 GPU)."""

import json
import os
import sys
from pathlib import Path

mode = sys.argv[1]
if mode == "unsloth":
    import unsloth  # noqa
    from unsloth import FastModel
import torch  # noqa: E402  (unsloth must be imported before torch)

BENCH_ROOT = os.environ["BENCH_ROOT"]  # see paths.env
path = os.environ["BENCH_MODEL"]
if mode == "unsloth":
    model, _ = FastModel.from_pretrained(
        model_name=path,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=True,
        device_map={"": 0},
        attn_implementation="sdpa",
        use_gradient_checkpointing=False,
    )
else:
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        path, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa"
    )
rows = [json.loads(line) for line in Path(f"{BENCH_ROOT}/data/canonical_tokens.jsonl").read_text().splitlines()][:2]


def run(tag, train, grad=False, **extra):
    model.train(train)
    tot = cnt = 0
    with torch.set_grad_enabled(grad):
        for r in rows:
            ids = torch.tensor([r["input_ids"]], device="cuda")
            lab = torch.tensor([r["labels"]], device="cuda")
            kw = {}
            if "pos" in extra:
                kw["position_ids"] = torch.arange(2048, device="cuda")[None]
            if "am" in extra:
                kw["attention_mask"] = torch.ones_like(ids)
            if "mm" in extra:
                kw["mm_token_type_ids"] = torch.zeros_like(ids)
            out = model(input_ids=ids, labels=lab, **kw)
            out.loss.detach()
            n = int((lab[:, 1:] != -100).sum())
            tot += out.loss.float().item() * n
            cnt += n
    print(mode, tag, "token-weighted", round(tot / cnt, 4), flush=True)


if mode == "hf":
    run("eval", False)
    run("train+am+mm", True, am=1, mm=1)
    run("train+am+mm grad", True, grad=True, am=1, mm=1)
else:
    run("eval", False)
    run("train+pos grad", True, grad=True, pos=1)
    run("train grad", True, grad=True)
    run("train+am grad", True, grad=True, am=1)
