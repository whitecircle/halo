#!/usr/bin/env python
"""Packed PP microbatches on flash kernels: the stage seam's varlen set reproduces the per-layer derivation bit for
bit, without its host syncs.

The reference arm withholds the seam and runs twice, since a reference that doesn't repeat bit for bit can't hold the
seam to bit equality. ``--attn flash_attention_4`` skips off Blackwell.
"""

import contextlib
import os
import sys
import warnings
from collections.abc import Iterator

import torch
from datasets import Dataset
from trl import SFTConfig

from halo.data.collators.packing import DataCollatorWithPacking
from halo.distributed.parallelism_config import ParallelismConfig
from halo.models.attention.gpt_oss_sinks import SinksPolicy, apply_sinks_policy
from halo.trainers.sft import DistributedSFTTrainer
from tests.common import pp_equivalence as ppeq
from tests.common.harness import gpu_test_main, skip_unless_blackwell
from tests.common.packed_flash_varlen import (
    PACKING_FAMILIES,
    SEED,
    AttentionHostReads,
    install_loaded_flash,
    use_flash,
)
from tests.common.pp_packed_flash_varlen import (
    MAX_LENGTH,
    PP_SIZE,
    ScheduleArm,
    ScheduleCase,
    arm_differences,
    dp_rows,
    run_schedule_arm,
    seam_findings,
    seam_withheld,
)
from tests.common.pp_trainers import build_pp_args, enable_gradient_checkpointing_as_train_does, tiny_vocab_tokenizer
from tests.common.rank_log import log
from tests.common.tiny_models import tiny_family_model

# transformers forwards it as the kernel's ``deterministic`` flag, so the FA2 backward accumulates in a fixed order.
os.environ.setdefault("FLASH_ATTENTION_DETERMINISTIC", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SCHEDULE = "1f1b"
MICROBATCHES = 2
GC = False
FUSED = False
ATTN = "flash_attention_2"
TRAINER = False
EP_SIZE = 1
SYNC_WARNING = "synchronizing CUDA operation"


@contextlib.contextmanager
def counted_syncs(counts: list[int]) -> Iterator[None]:
    """Append to ``counts`` the device syncs the block makes, from torch's sync debug warnings."""
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            yield
        finally:
            torch.cuda.set_sync_debug_mode(previous)
    counts.append(sum(SYNC_WARNING in str(warning.message) for warning in caught))


def _runner_arm(ctx, config, groups, reads, attn_implementation: str) -> tuple[ScheduleArm, int]:
    case = ScheduleCase(schedule=SCHEDULE, n_microbatches=MICROBATCHES, gc=GC, fused=FUSED)
    syncs: list[int] = []
    arm = run_schedule_arm(
        case,
        config,
        groups,
        ctx.device,
        reads,
        attn_implementation=attn_implementation,
        dtype=torch.bfloat16,
        reference_kernels=False,
        step_scope=lambda: counted_syncs(syncs),
    )
    return arm, syncs[0]


def _trainer_arm(ctx, config, reads, attn_implementation: str) -> tuple[ScheduleArm, int]:
    torch.manual_seed(SEED)
    family = "gpt_oss" if EP_SIZE > 1 else "dense:qwen3"
    model = ppeq.build_patched_moe_model(config) if EP_SIZE > 1 else tiny_family_model(PACKING_FAMILIES[family])
    use_flash(model.config, attn_implementation)
    apply_sinks_policy(model, model.config, policy=SinksPolicy.NEUTRALIZED, attn_implementation=attn_implementation)
    tokenizer = tiny_vocab_tokenizer()
    case = ScheduleCase(family=family, n_microbatches=MICROBATCHES, gc=GC)
    rows = dp_rows(case, config.get_data_parallel_rank())
    dataset = Dataset.from_list([{**row, "labels": row["input_ids"]} for row in rows])
    args = build_pp_args(
        SFTConfig,
        f"{ctx.output_dir}/arm",
        max_steps=1,
        per_device_train_batch_size=MICROBATCHES,
        gradient_accumulation_steps=MICROBATCHES,
        max_length=MAX_LENGTH,
        bf16=True,
        gradient_checkpointing=GC,
    )
    trainer = DistributedSFTTrainer(
        model=model.to(torch.bfloat16),
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPacking(tokenizer=tokenizer),
        parallelism_config=config,
    )
    try:
        enable_gradient_checkpointing_as_train_does(trainer)
        checkpointed = ppeq.count_checkpoint_calls(trainer.model) if GC else [0]
        batch = trainer.data_collator([dict(row) for row in rows])
        counts = install_loaded_flash(reads, attn_implementation)
        reads.reads.clear()
        syncs: list[int] = []
        with counted_syncs(syncs), reads:
            loss = trainer.training_step(trainer.model, batch)
        # Runs the EP replica sweep and reads the norm; max_norm leaves every gradient as is.
        trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), 1e6)
        grads = ppeq.gather_stage_grads(trainer.model, to_cpu=True)
    finally:
        trainer.cleanup_ep()
    arm = ScheduleArm(
        loss=loss.detach().cpu(),
        grads=grads,
        eval_loss=None,
        outputs=None,
        counts=dict(counts),
        reads=list(reads.reads),
        checkpointed=checkpointed[0],
    )
    return arm, syncs[0]


@gpu_test_main(exact_world_size=4, prefix="pp_packed_flash_varlen")
def run(ctx):
    torch.use_deterministic_algorithms(True, warn_only=True)
    attn_implementation = ATTN
    config = ParallelismConfig(
        world_size=ctx.world_size,
        gpus_per_node=2,
        nvlink_domain_size=2,
        pp_size=PP_SIZE,
        pp_schedule=SCHEDULE,
        ep_size=EP_SIZE,
    )
    reads = AttentionHostReads()
    reads.attending(attn_implementation)
    if TRAINER:

        def arm():
            return _trainer_arm(ctx, config, reads, attn_implementation)
    else:
        groups = ppeq.build_stage_groups(config)

        def arm():
            return _runner_arm(ctx, config, groups, reads, attn_implementation)

    with seam_withheld():
        reference, reference_syncs = arm()
        repeat, _ = arm()
    seam, seam_syncs = arm()

    repeat_gaps = arm_differences(repeat, reference)
    seam_gaps = arm_differences(seam, reference)
    findings = seam_findings(seam, reference)
    for name, gap in sorted(seam_gaps.items())[:10]:
        log(f"[rank {ctx.rank}] seam vs reference: {name} differs by {gap:.3e}")
    for finding in findings:
        log(f"[rank {ctx.rank}] {finding}")
    log(
        f"[rank {ctx.rank}] {attn_implementation}: syncs per step {reference_syncs} -> {seam_syncs}; "
        f"derivations {reference.counts} -> {seam.counts}; attention host reads {len(reference.reads)} -> "
        f"{len(seam.reads)}"
    )
    checks = {
        "reference_repeats_bit_for_bit": not repeat_gaps,
        "seam_matches_reference_bit_for_bit": not seam_gaps,
        "seam_leaves_nothing_derived": not findings,
        "sync_count_observed": reference_syncs > 0,
        "seam_removes_syncs": seam_syncs < reference_syncs,
    }
    if GC:
        checks["checkpointing_engaged"] = seam.checkpointed > 0
    metrics = {
        "attn_implementation": attn_implementation,
        "reference_syncs": reference_syncs,
        "seam_syncs": seam_syncs,
        "seam_max_gap": max(seam_gaps.values(), default=0.0),
        "repeat_max_gap": max(repeat_gaps.values(), default=0.0),
        "reference_derivations": reference.counts["_prepare_from_posids"],
    }
    return {"checks": checks, "metrics": metrics}


def _pop_flag(name: str) -> bool:
    if name in sys.argv:
        sys.argv.remove(name)
        return True
    return False


def _pop_value(name: str) -> str | None:
    if name not in sys.argv:
        return None
    i = sys.argv.index(name)
    value = sys.argv[i + 1]
    del sys.argv[i : i + 2]
    return value


if __name__ == "__main__":
    GC = _pop_flag("--gc")
    FUSED = _pop_flag("--fused")
    TRAINER = _pop_flag("--trainer")
    SCHEDULE = _pop_value("--schedule") or SCHEDULE
    MICROBATCHES = int(_pop_value("--microbatches") or MICROBATCHES)
    ATTN = _pop_value("--attn") or ATTN
    if ATTN == "flash_attention_4":
        skip_unless_blackwell("test_pp_packed_flash_varlen.py --attn flash_attention_4")
    EP_SIZE = int(_pop_value("--ep") or EP_SIZE)
    sys.exit(run())
