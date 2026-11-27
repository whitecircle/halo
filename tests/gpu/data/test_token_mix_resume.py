#!/usr/bin/env python
"""DistributedSFTTrainer on a token mix over 4-byte datatrove folders, on two GPUs: every step reads its slice of the
mix (split by DP rank, identical on TP siblings), the checkpoint's trainer state carries the mix's fingerprint, and a
resume from mid-run reads the rows and reaches the losses of the uninterrupted run."""

import argparse
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer
from trl import SFTConfig

from halo.callbacks.token_mix_checkpoint import TokenMixCheckpointCallback, find_saved_token_mix
from halo.checkpoint.parallel.resume_source import resolve_resume_weights_source
from halo.data.collators.factory import select_data_collator
from halo.data.sources.token_mix import TokenSource, build_token_mix_across_ranks
from halo.distributed.parallelism_config import ParallelismConfig
from halo.entrypoints.data_setup import disable_trl_dataset_prep
from halo.model_loading.parallel.loader import load_distributed_model
from halo.runtime.collectives import gather_on_rank0
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.datatrove_shards import write_token_folder
from tests.common.distributed import shared_output_dir
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log
from tests.common.tolerances import TOL

MODEL = QWEN3_0_6B
SEED = 42
MAX_LENGTH = 512
BATCH = 2
GRAD_ACCUM = 2
MAX_STEPS = 6
SAVE_STEP = 3
MODES = {"fsdp": {}, "tp": {"tp_size": 2}}
# Each source's name, file count and weight.
SOURCES = (("web", 3, 3.0), ("code", 1, 1.0))


def _digest(ids: list[int]) -> str:
    return hashlib.blake2b(str(ids).encode(), digest_size=8).hexdigest()


def _write_sources(root: str, vocab_size: int) -> None:
    """Folders of 4-byte ids, documents of random lengths, some longer than a window."""
    rng = np.random.default_rng(SEED)
    for name, files, _ in SOURCES:
        documents = {}
        for index in range(files):
            tokens = 40 * MAX_LENGTH + int(rng.integers(0, MAX_LENGTH))
            cuts = np.unique(np.concatenate(([0, tokens], rng.integers(1, tokens, size=tokens // 300))))
            documents[f"{index:05d}.ds"] = [
                rng.integers(0, vocab_size, size=end - begin).tolist()
                for begin, end in zip(cuts[:-1], cuts[1:], strict=True)
            ]
        write_token_folder(Path(root) / name, documents, token_bytes=4, tokenizer=MODEL)


def _load_model(parallelism_config: ParallelismConfig, weights: str):
    model, _ = load_distributed_model(
        model_name_or_path=weights,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    return model


def _row_digests(trainer) -> dict[int, list[str]]:
    """A digest of every row this rank's micro-batches carried, by the optimizer step's zero-based index."""
    rows: dict[int, list[str]] = {}
    compute_loss = trainer.compute_loss

    def recording(model, inputs, *args, **kwargs):
        flat = inputs["input_ids"][0].tolist()
        rows.setdefault(trainer.state.global_step, []).extend(
            _digest(flat[start : start + MAX_LENGTH]) for start in range(0, len(flat), MAX_LENGTH)
        )
        return compute_loss(model, inputs, *args, **kwargs)

    trainer.compute_loss = recording
    return rows


def _run_phase(mode, tokenizer, mix, output_dir, checkpoint=None):
    parallelism_config = ParallelismConfig(**MODES[mode])
    weights = resolve_resume_weights_source(checkpoint, SimpleNamespace(model_name_or_path=MODEL), parallelism_config)
    args = SFTConfig(
        output_dir=output_dir,
        max_steps=MAX_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEP,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-5,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        report_to="none",
        max_length=MAX_LENGTH,
        dataloader_num_workers=0,
        train_sampling_strategy="sequential",
        packing=True,
        seed=SEED,
    )
    disable_trl_dataset_prep(args)
    model = _load_model(parallelism_config, weights)
    collator = select_data_collator(
        tokenizer,
        train_on_completions_only=False,
        packing=True,
        model_config=model.config,
        per_device_train_batch_size=BATCH,
        emit_varlen_one_document=True,
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=args,
        train_dataset=mix,
        processing_class=tokenizer,
        data_collator=collator,
        parallelism_config=parallelism_config,
        callbacks=[TokenMixCheckpointCallback(mix.fingerprint())],
    )
    rows = _row_digests(trainer)
    trainer.train(resume_from_checkpoint=checkpoint)
    losses = {entry["step"]: entry["loss"] for entry in trainer.state.log_history if "loss" in entry}
    trainer.cleanup_ep()
    del trainer, model
    cleanup_memory()
    return rows, losses


def run_mode(ctx, mode: str) -> dict[str, bool]:
    log(f"\n  Token mix resume: mode {mode}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    root = os.path.join(shared_output_dir(ctx), mode)
    if ctx.rank == 0:
        _write_sources(os.path.join(root, "sources"), tokenizer.vocab_size)
    ctx.barrier()
    sources = [TokenSource(name, os.path.join(root, "sources", name), weight) for name, _, weight in SOURCES]
    data_parallel_size = ParallelismConfig(**MODES[mode]).data_parallel_size
    global_batch = BATCH * GRAD_ACCUM * data_parallel_size
    mix = build_token_mix_across_ranks(
        sources, max_length=MAX_LENGTH, num_steps=MAX_STEPS, rows_per_step=global_batch, seed=SEED
    )
    uninterrupted_rows, uninterrupted_losses = _run_phase(mode, tokenizer, mix, os.path.join(root, "uninterrupted"))
    checkpoint = os.path.join(root, "uninterrupted", f"checkpoint-{SAVE_STEP}")
    resumed_rows, resumed_losses = _run_phase(mode, tokenizer, mix, os.path.join(root, "resumed"), checkpoint)
    gathered = gather_on_rank0((uninterrupted_rows, resumed_rows))
    checks = {
        "resumed_rows_match": all(
            resumed_rows[step] == uninterrupted_rows[step] for step in range(SAVE_STEP, MAX_STEPS)
        ),
        "resumed_losses_match": all(
            abs(resumed_losses[step] - uninterrupted_losses[step]) <= TOL.replayed_resume_loss_abs
            for step in range(SAVE_STEP + 1, MAX_STEPS + 1)
        ),
    }
    if ctx.rank == 0:
        expected = [
            sorted(_digest(mix[index]["input_ids"]) for index in range(step * global_batch, (step + 1) * global_batch))
            for step in range(MAX_STEPS)
        ]
        per_rank = [rows for rows, _ in gathered]
        if data_parallel_size == ctx.world_size:
            held = [sorted(row for rows in per_rank for row in rows[step]) for step in range(MAX_STEPS)]
        else:
            held = [sorted(per_rank[0][step]) for step in range(MAX_STEPS)]
            checks["siblings_read_the_same_rows"] = all(rows == per_rank[0] for rows in per_rank)
        checks["each_step_reads_its_slice"] = held == expected
        checks["checkpoint_holds_the_fingerprint"] = find_saved_token_mix(checkpoint) == mix.fingerprint()
    log(f"  {mode}: {checks}")
    return {f"{mode}_{name}": ok for name, ok in ctx.broadcast_checks(checks).items()}


def run(ctx) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="fsdp", choices=list(MODES))
    return {"checks": run_mode(ctx, parser.parse_args().mode)}


main = gpu_test_main(exact_world_size=2, prefix="token_mix_resume")(run)

if __name__ == "__main__":
    main()
