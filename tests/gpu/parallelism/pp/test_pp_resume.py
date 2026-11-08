#!/usr/bin/env python
"""PP checkpoint resume through the real DistributedSFTTrainer (pp2 x dp2): bitwise restore and a replay of the
unbroken run.

A shifted global-to-local remap resumes from base weights. A shifted partition, a deleted optimizer shard and a
fingerprint mismatch must raise, never warm-restart silently.
"""

import os
import shutil
import sys

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, default_data_collator
from trl import SFTConfig

from halo.checkpoint.reading import StreamingCheckpointReader
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.introspection.unwrapping import unwrap_toolkit_wrappers
from halo.runtime.dtensor import full_cpu_tensor
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.distributed import shared_output_dir, world_all
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_CONFIG
from tests.common.pp_equivalence import build_dense_model
from tests.common.pp_trainers import build_pp_args, build_sft_rows, restored_moments_match_shard
from tests.common.rank_log import log

PP_SIZE = 2
SEQ_LEN = 64
N_STEPS = 8
SAVE_STEP = 4


def _expect_raise(fn, *args) -> bool:
    """True when fn raised on every rank: the gates must be rank-uniform, no silent survivor."""
    raised = False
    try:
        fn(*args)
    except (RuntimeError, ValueError) as exc:
        log(f"[gate] raised as expected: {str(exc)[:140]}")
        raised = True
    return world_all(raised)


@gpu_test_main(exact_world_size=4, prefix="pp_resume")
def run(ctx):
    checks, metrics = {}, {}
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)

    # Multi-writer single-directory format: all ranks share one root.
    root = shared_output_dir(ctx)

    def build_trainer(subdir: str, max_steps: int, save_steps: int | None = None, model=None) -> DistributedSFTTrainer:
        args = build_pp_args(
            SFTConfig,
            os.path.join(root, subdir),
            max_steps=max_steps,
            per_device_train_batch_size=4,
            max_length=SEQ_LEN,
            save_every=save_steps,
            gradient_checkpointing=False,
        )
        return DistributedSFTTrainer(
            model=build_dense_model() if model is None else model,
            args=args,
            train_dataset=build_sft_rows(TINY_QWEN3_CONFIG["vocab_size"]),
            processing_class=tokenizer,
            data_collator=default_data_collator,
            parallelism_config=config,
        )

    trainer_a = build_trainer("runA", N_STEPS)
    trainer_a.train()
    losses_a = [e["loss"] for e in trainer_a.state.log_history if "loss" in e]
    checks["run_a_completed"] = trainer_a.state.global_step == N_STEPS and len(losses_a) == N_STEPS

    trainer_b = build_trainer("runB", SAVE_STEP, save_steps=SAVE_STEP)
    trainer_b.train()
    ckpt = os.path.join(root, "runB", f"checkpoint-{SAVE_STEP}")
    dist.barrier()
    rank = dist.get_rank()
    checks["checkpoint_written"] = (
        os.path.isfile(os.path.join(ckpt, "model.safetensors.index.json"))
        and os.path.isfile(os.path.join(ckpt, f"optimizer_shard_{rank:05d}.pt"))
        and os.path.isfile(os.path.join(ckpt, "optimizer_meta.pt"))
    )
    # The recorded partition must be the live split, never a formula recompute; the MIN all-reduce covers the whole
    # partition.
    meta = torch.load(os.path.join(ckpt, "optimizer_meta.pt"), map_location="cpu", weights_only=False)
    saved_partition = [tuple(entry) for entry in meta.get("pp_stage_partition", [])]
    stage_b = unwrap_toolkit_wrappers(trainer_b.model)
    checks["meta_carries_live_partition"] = world_all(
        len(saved_partition) == PP_SIZE and saved_partition[config.pp_rank] == stage_b.layer_range
    )

    trainer_c = build_trainer("runC", N_STEPS)
    trainer_c._load_from_checkpoint(ckpt)
    stage = unwrap_toolkit_wrappers(trainer_c.model)
    local_state = stage.state_dict()
    with StreamingCheckpointReader(ckpt, [stage.global_parameter_name(k) for k in local_state]) as reader:
        saved = {key: reader.get(key) for key in reader.available}
    weights_ok, n_compared = True, 0
    init_state = build_dense_model().state_dict()
    moved = 0
    for local, value in sorted(local_state.items()):
        full = full_cpu_tensor(value).cpu()  # COLLECTIVE inside the stage mesh: sorted order.
        reference = saved[stage.global_parameter_name(local)]
        weights_ok &= torch.equal(full, reference)
        n_compared += 1
        global_name = stage.global_parameter_name(local)
        if global_name in init_state and not torch.equal(full, init_state[global_name]):
            moved += 1
    checks["post_load_weights_bitwise_equal_checkpoint"] = weights_ok and n_compared == len(local_state)
    # Checkpoint == init would hide a no-op load.
    checks["checkpoint_weights_differ_from_init"] = moved > n_compared * 0.9
    log(f"[rank {ctx.rank}] weights compared={n_compared} moved_from_init={moved}")

    resumed_model, _ = load_distributed_model(
        ckpt,
        config,
        dtype=torch.float32,
        attn_implementation="sdpa",
        preserve_checkpoint_precision=True,
        trust_remote_code=True,
    )
    stage_e = unwrap_toolkit_wrappers(build_trainer("runE", N_STEPS, model=resumed_model).model)
    built_ok, n_built = True, 0
    for local, value in sorted(stage_e.state_dict().items()):
        full = full_cpu_tensor(value).cpu()  # COLLECTIVE inside the stage mesh: sorted order.
        reference = saved.get(stage_e.global_parameter_name(local))
        built_ok &= reference is not None and torch.equal(full, reference)
        n_built += 1
    checks["production_stage_load_reads_the_checkpoint_bitwise"] = built_ok and n_built == len(local_state)

    trainer_c.create_optimizer_and_scheduler(num_training_steps=N_STEPS)
    trainer_c._load_optimizer_and_scheduler(ckpt)
    checks["optimizer_moments_bitwise_equal_shard"], _ = restored_moments_match_shard(
        stage, trainer_c.optimizer, ckpt, rank
    )

    losses_b = [e["loss"] for e in trainer_b.state.log_history if "loss" in e]
    trainer_d = build_trainer("runD", N_STEPS)
    trainer_d.train(resume_from_checkpoint=ckpt)
    # Resume restores log_history, so trainer_d carries run B's steps 1-4 then the fresh steps 5-8.
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

    # Each on a tampered copy; must raise on every rank.
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
        # Rank 0 tampers with a copy; every rank waits for it.
        tampered_dir = os.path.join(root, f"tampered_{name}")
        if rank == 0:
            shutil.copytree(ckpt, tampered_dir, dirs_exist_ok=True)
            mutate(tampered_dir)
        dist.barrier()
        checks[name] = _expect_raise(trainer_c._load_optimizer_and_scheduler, tampered_dir)

    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
