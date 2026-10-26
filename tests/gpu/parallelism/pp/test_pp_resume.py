#!/usr/bin/env python
"""PP checkpoint resume through the real DistributedSFTTrainer (pp2 × dp2, tiny dense Qwen3).

Scope under test (identical-topology resume): run A trains 8 steps straight; run B trains 4, saves
at step 4, and a new trainer resumes from that checkpoint and trains 4 more. The seams that must
hold, each with a check that fails when it breaks:

  * the global→local weight remap (post-load stage weights bitwise equal the checkpoint tensors —
    a shifted remap or dropped key resumes from base weights)
  * the scripts' resume construction: ``load_distributed_model`` pointed at the checkpoint (the PP
    stage-loader dispatch, its rank-0 lazy-load verdict and joined gates) builds a stage bitwise
    equal to the checkpoint
  * per-rank optimizer-shard restore (live moments bitwise equal the shard content)
  * training continuity (B's resumed steps 5-8 reproduce A's losses within rel 1e-3 — catches a
    silently warm-restarted optimizer, a re-warmed LR schedule, or desynced data order)
  * the negative gates: a shifted pp_stage_partition raises, a deleted optimizer shard raises
    (torn), a fingerprint pp_size mismatch raises — never a silent warm restart under PP.

Topology: 4 GPUs as 2 simulated nodes of 2 (NVLink domain = 2), pipeline crossing the boundary.

Run: torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_resume.py
"""

import os
import shutil
import sys

import torch
import torch.distributed as dist
from datasets import Dataset
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict
from transformers import AutoTokenizer, default_data_collator
from trl import SFTConfig

from src.checkpoint.format import StreamingCheckpointReader
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import resolve_param_tensor
from src.models.unwrapping import unwrap_toolkit_wrappers
from src.trainers.sft import DistributedSFTTrainer
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_CONFIG
from tests.common.pp_equivalence import build_dense_model
from tests.common.utils import log

PP_SIZE = 2
SEQ_LEN = 64
N_STEPS = 8
SAVE_STEP = 4


def build_dataset(vocab_size: int, n_rows: int = 16, seed: int = 7) -> Dataset:
    g = torch.Generator().manual_seed(seed)
    rows = torch.randint(4, vocab_size, (n_rows, SEQ_LEN), generator=g)
    labels = rows.clone()
    for i in range(labels.size(0)):
        labels[i, : 4 + (i % 11)] = -100  # ragged -> DP replicas hold unequal loss-token counts
    return Dataset.from_dict(
        {
            "input_ids": rows.tolist(),
            "labels": labels.tolist(),
            "attention_mask": torch.ones_like(rows).tolist(),
        }
    )


def build_args(output_dir: str, max_steps: int, save_steps: int | None = None) -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,  # becomes the pipeline's microbatch count
        learning_rate=5e-3,
        lr_scheduler_type="constant",
        max_length=SEQ_LEN,
        logging_steps=1,
        save_strategy="steps" if save_steps else "no",
        save_steps=save_steps or 500,
        report_to=[],
        logging_nan_inf_filter=False,
        seed=42,
        bf16=False,
        gradient_checkpointing=False,
    )


def _expect_raise(fn, *args) -> bool:
    """True when fn raised on every rank (the gates must be rank-uniform, no silent survivor)."""
    raised = False
    try:
        fn(*args)
    except (RuntimeError, ValueError) as exc:
        log(f"[gate] raised as expected: {str(exc)[:140]}")
        raised = True
    flag = torch.tensor([1.0 if raised else 0.0], device=torch.cuda.current_device())
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _tamper_checkpoint(source: str, dest: str, mutate) -> None:
    """Rank 0 copies the checkpoint and applies ``mutate(meta_or_dir)``; all ranks wait."""
    if dist.get_rank() == 0:
        shutil.copytree(source, dest, dirs_exist_ok=True)
        mutate(dest)
    dist.barrier()


@gpu_test_main(exact_world_size=4, prefix="pp_resume")
def run(ctx):
    checks, metrics = {}, {}
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)

    # The PP checkpoint is a multi-writer single-directory format: all ranks must share one root.
    dirs: list = [None] * ctx.world_size
    dist.all_gather_object(dirs, ctx.output_dir)
    root = dirs[0]

    def make_trainer(subdir: str, max_steps: int, save_steps: int | None = None, model=None) -> DistributedSFTTrainer:
        return DistributedSFTTrainer(
            model=build_dense_model() if model is None else model,
            args=build_args(os.path.join(root, subdir), max_steps, save_steps),
            train_dataset=build_dataset(TINY_QWEN3_CONFIG["vocab_size"]),
            processing_class=tokenizer,
            data_collator=default_data_collator,
            parallelism_config=config,
        )

    # ── run A: 8 straight steps (the reference loss curve) ──────────────────────────────────────
    trainer_a = make_trainer("runA", N_STEPS)
    trainer_a.train()
    losses_a = [e["loss"] for e in trainer_a.state.log_history if "loss" in e]
    checks["run_a_completed"] = trainer_a.state.global_step == N_STEPS and len(losses_a) == N_STEPS

    # ── run B: 4 steps, checkpoint at step 4 ────────────────────────────────────────────────────
    trainer_b = make_trainer("runB", SAVE_STEP, save_steps=SAVE_STEP)
    trainer_b.train()
    ckpt = os.path.join(root, "runB", f"checkpoint-{SAVE_STEP}")
    dist.barrier()
    rank = dist.get_rank()
    checks["checkpoint_written"] = (
        os.path.isfile(os.path.join(ckpt, "model.safetensors.index.json"))
        and os.path.isfile(os.path.join(ckpt, f"optimizer_shard_{rank:05d}.pt"))
        and os.path.isfile(os.path.join(ckpt, "optimizer_meta.pt"))
    )
    # The recorded partition must be the live per-stage split (never a formula recompute) — each
    # rank checks its own stage's entry; the MIN all-reduce across ranks covers the whole partition.
    meta = torch.load(os.path.join(ckpt, "optimizer_meta.pt"), map_location="cpu", weights_only=False)
    saved_partition = [tuple(entry) for entry in meta.get("pp_stage_partition", [])]
    stage_b = unwrap_toolkit_wrappers(trainer_b.model)
    partition_ok = len(saved_partition) == PP_SIZE and saved_partition[config.pp_rank] == stage_b.layer_range
    flag = torch.tensor([1.0 if partition_ok else 0.0], device=torch.cuda.current_device())
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    checks["meta_carries_live_partition"] = bool(flag.item())

    # ── bitwise restore checks on a fresh trainer ───────────────────────────────────────────────
    trainer_c = make_trainer("runC", N_STEPS)
    trainer_c._load_from_checkpoint(ckpt)
    stage = unwrap_toolkit_wrappers(trainer_c.model)
    local_state = stage.state_dict()
    with StreamingCheckpointReader(ckpt, [stage.global_parameter_name(k) for k in local_state]) as reader:
        saved = {key: reader.get(key) for key in reader.available}
    weights_ok, n_compared = True, 0
    init_state = build_dense_model().state_dict()
    moved = 0
    for local, value in sorted(local_state.items()):
        full = resolve_param_tensor(value).cpu()  # collective inside the stage mesh — sorted order
        reference = saved[stage.global_parameter_name(local)]
        weights_ok &= torch.equal(full, reference)
        n_compared += 1
        global_name = stage.global_parameter_name(local)
        if global_name in init_state and not torch.equal(full, init_state[global_name]):
            moved += 1
    checks["post_load_weights_bitwise_equal_checkpoint"] = weights_ok and n_compared == len(local_state)
    # Guard against a vacuous pass (checkpoint == init would hide a no-op load).
    checks["checkpoint_weights_differ_from_init"] = moved > n_compared * 0.9
    log(f"[rank {ctx.rank}] weights compared={n_compared} moved_from_init={moved}")

    # ── the scripts' resume construction: the stage is loaded from the checkpoint ──────────────
    resumed_model, _ = load_distributed_model(
        ckpt, config, dtype=torch.float32, attn_implementation="sdpa", preserve_checkpoint_precision=True
    )
    stage_e = unwrap_toolkit_wrappers(make_trainer("runE", N_STEPS, model=resumed_model).model)
    built_ok, n_built = True, 0
    for local, value in sorted(stage_e.state_dict().items()):
        full = resolve_param_tensor(value).cpu()  # collective inside the stage mesh — sorted order
        reference = saved.get(stage_e.global_parameter_name(local))
        built_ok &= reference is not None and torch.equal(full, reference)
        n_built += 1
    checks["production_stage_load_reads_the_checkpoint_bitwise"] = built_ok and n_built == len(local_state)

    trainer_c.create_optimizer_and_scheduler(num_training_steps=N_STEPS)
    trainer_c._load_optimizer_and_scheduler(ckpt)
    shard = torch.load(os.path.join(ckpt, f"optimizer_shard_{rank:05d}.pt"), map_location="cpu", weights_only=False)
    live_osd = get_optimizer_state_dict(
        stage, trainer_c.optimizer, options=StateDictOptions(full_state_dict=False, cpu_offload=True)
    )

    def _local(t):
        return t.to_local() if isinstance(t, torch.distributed.tensor.DTensor) else t

    optim_ok = bool(shard["state"])
    for fqn, saved_state in shard["state"].items():
        live_state = live_osd["state"].get(fqn)
        if live_state is None:
            log(f"[optim] live state missing for {fqn}")
            optim_ok = False
            continue
        for key, saved_value in saved_state.items():
            if not torch.is_tensor(saved_value):
                continue
            live_value = live_state.get(key)
            if live_value is None or not torch.equal(_local(saved_value).cpu(), _local(live_value).cpu()):
                log(f"[optim] mismatch at {fqn}.{key}")
                optim_ok = False
    checks["optimizer_moments_bitwise_equal_shard"] = optim_ok

    # ── run D: resume from the checkpoint, steps 5-8 must reproduce run A ───────────────────────
    losses_b = [e["loss"] for e in trainer_b.state.log_history if "loss" in e]
    trainer_d = make_trainer("runD", N_STEPS)
    trainer_d.train(resume_from_checkpoint=ckpt)
    # Resume restores log_history from trainer_state.json, so trainer_d carries run B's steps 1-4
    # followed by the freshly trained steps 5-8.
    losses_d = [e["loss"] for e in trainer_d.state.log_history if "loss" in e]
    checks["resume_trained_remaining_steps"] = trainer_d.state.global_step == N_STEPS and len(losses_d) == N_STEPS
    checks["resume_restored_trainer_state_history"] = losses_d[:SAVE_STEP] == losses_b
    resumed = losses_d[SAVE_STEP:]
    resumed_ok = len(resumed) == N_STEPS - SAVE_STEP
    for step, (a, d) in enumerate(zip(losses_a[SAVE_STEP:], resumed, strict=False), start=SAVE_STEP + 1):
        rel = abs(a - d) / max(abs(a), 1e-9)
        if rel > 1e-3:
            log(f"[resume] step {step}: A={a:.6f} D={d:.6f} rel={rel:.3e}")
            resumed_ok = False
    checks["resumed_losses_match_straight_run"] = resumed_ok
    metrics["losses_a_tail"] = ", ".join(f"{v:.5f}" for v in losses_a[SAVE_STEP:])
    metrics["losses_d_resumed"] = ", ".join(f"{v:.5f}" for v in resumed)

    # ── negative gates (each on a tampered copy; must raise on every rank) ──────────────────────
    def shift_partition(dest):
        meta_path = os.path.join(dest, "optimizer_meta.pt")
        tampered = torch.load(meta_path, map_location="cpu", weights_only=False)
        tampered["pp_stage_partition"] = [[lo + 1, hi + 1] for lo, hi in tampered["pp_stage_partition"]]
        torch.save(tampered, meta_path)

    def drop_shard(dest):
        os.remove(os.path.join(dest, f"optimizer_shard_{ctx.world_size - 1:05d}.pt"))

    def wrong_pp_size(dest):
        meta_path = os.path.join(dest, "optimizer_meta.pt")
        tampered = torch.load(meta_path, map_location="cpu", weights_only=False)
        tampered["fingerprint"]["pp_size"] = 4
        torch.save(tampered, meta_path)

    for name, mutate in (
        ("shifted_partition_raises", shift_partition),
        ("deleted_shard_raises", drop_shard),
        ("fingerprint_pp_size_mismatch_raises", wrong_pp_size),
    ):
        tampered_dir = os.path.join(root, f"tampered_{name}")
        _tamper_checkpoint(ckpt, tampered_dir, mutate)
        checks[name] = _expect_raise(trainer_c._load_optimizer_and_scheduler, tampered_dir)

    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
