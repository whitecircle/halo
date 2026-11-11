"""Unsloth full fine-tuning for the protocol (PROTOCOL.txt): Gemma 4 26B-A4B, bf16 weights, canonical tokens/labels fed directly
(pre-tokenized dataset: input_ids + labels, so TRL skips tokenization and its collator uses the given labels),
sequential order (train_sampling_strategy="sequential"; with DDP rank r gets row 2*(k-1)+r at step k).

Launch (DDP): torchrun --nproc_per_node=2 train_unsloth.py
Env: BENCH_OPTIM (adamw_torch_fused | adamw_8bit), BENCH_GC (unsloth|true|false), BENCH_OUT (json path), BENCH_MODEL.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_callback
import torch
import unsloth  # noqa: F401  (must be imported before transformers/trl)
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastModel

BENCH_ROOT = os.environ["BENCH_ROOT"]  # see paths.env
BENCH_TMP = os.environ.get("BENCH_TMP", os.path.join(os.environ["BENCH_ROOT"], "tmp"))  # see paths.env

bench_callback.install()

MODEL = os.environ.get("BENCH_MODEL", "google/gemma-4-26B-A4B-it")
CANON = os.environ.get("BENCH_CANON", f"{BENCH_ROOT}/data/canonical_tokens.jsonl")
SEQ = int(os.environ.get("BENCH_SEQ", "2048"))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
gc = {"unsloth": "unsloth", "true": True, "false": False}[os.environ.get("BENCH_GC", "false")]

model, processor = FastModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=SEQ,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=True,
    device_map={"": local_rank},
    use_gradient_checkpointing=gc,
    attn_implementation=os.environ.get("BENCH_ATTN", "sdpa"),
)
tokenizer = getattr(processor, "tokenizer", processor)

rows = [json.loads(line) for line in Path(CANON).read_text().splitlines()]
ds = Dataset.from_dict({"input_ids": [r["input_ids"] for r in rows], "labels": [r["labels"] for r in rows]})

args = SFTConfig(
    output_dir=f"{BENCH_TMP}/unsloth_bench_out",
    max_length=SEQ,
    packing=False,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    max_steps=int(os.environ.get("BENCH_STEPS", "25")),
    learning_rate=1e-5,
    lr_scheduler_type="constant",
    warmup_steps=0,
    max_grad_norm=1.0,
    optim=os.environ.get("BENCH_OPTIM", "adamw_torch_fused"),
    weight_decay=0.0,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    bf16=True,
    logging_steps=1,
    save_strategy="no",
    eval_strategy="no",
    report_to="none",
    seed=42,
    train_sampling_strategy="sequential",
    dataloader_num_workers=2,
    # vision-tower params are trainable but unused on text-only data -> DDP needs unused-param detection
    ddp_find_unused_parameters=True,
    dataset_num_proc=8,
    # TRL's SFTConfig defaults gradient_checkpointing=True; follow BENCH_GC.
    gradient_checkpointing=gc is not False,
)

trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=ds, args=args)
trainer.train()
