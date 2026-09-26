#!/usr/bin/env python
"""DPO / KTO ``precompute_ref_log_probs`` resumed under Expert Parallelism (ep2, tiny Qwen3-MoE).

TRL computes the reference log-probs inside ``__init__`` over ``self.ref_model or self.model``, and
an EP resume builds the policy from the checkpoint before the trainer exists (Path B). A sweep there
scores the trained weights, so the reference equals the policy and every log-ratio is zero. Each
checkpoint therefore carries the untrained columns (``reference_logps.pt``), and a trainer handed the
resume checkpoint at construction attaches them instead of sweeping.

One checkpoint, four phases:

  1. **Continuous** — ``TOTAL_STEPS`` from the base, checkpointing at ``SAVE_AT_STEP``.
  2. **Resume** the production way (policy built from the checkpoint; ``resume_checkpoint`` and
     ``policy_from_checkpoint=True``): the reference columns equal phase 1's bit for bit, and the
     first resumed step's loss equals the continuous run's at that step to ``LOSS_ATOL``.
  3. **Control** — the same resume with the trainer not told: its sweep scores the trained policy,
     and its columns and first-step loss must miss phase 1's by far more than those tolerances. That
     is what gives the phase-2 comparisons their teeth.
  4. **Refusal** — the checkpoint copied without the sidecar and resumed the production way raises
     on every rank instead of sweeping.

Run: torchrun --nproc_per_node=2 tests/gpu/parallelism/ep/test_ep_preference_precompute_resume.py \
         --trainer dpo
"""

import argparse
import math
import os
import shutil

import torch
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM
from trl import DPOConfig, KTOConfig

from src.checkpoint.format import REFERENCE_LOGPS_FILE
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.preference.dpo import DistributedDPOTrainer
from src.trainers.preference.kto import DistributedKTOTrainer
from tests.common.distributed import shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_MOE_CONFIG
from tests.common.utils import cleanup_memory, log

parser = argparse.ArgumentParser()
parser.add_argument("--trainer", choices=("dpo", "kto"), default="dpo")
ARGS, _ = parser.parse_known_args()

TRAINERS = {"dpo": (DistributedDPOTrainer, DPOConfig), "kto": (DistributedKTOTrainer, KTOConfig)}

EP_SIZE = 2
SEED = 42
N_ROWS = 16
BATCH_SIZE = 2
TOTAL_STEPS = 4
SAVE_AT_STEP = 2
MAX_LENGTH = 96
# Large enough that two steps move the policy's log-probs well off the reference, so a resume that
# re-derived the reference from the trained weights lands visibly elsewhere.
LEARNING_RATE = 5e-4
# The resumed first step sees the continuous run's weights (bf16, saved and reloaded exactly), batch
# and reference columns in the same process: measured |delta| 0.0 for DPO and KTO on B300.
LOSS_ATOL = 1e-3
# The control must miss by at least this much, or the comparisons above prove nothing. Measured on
# B300: first-step loss off by 0.30 (DPO, at ln 2) and 0.021 (KTO, at 0.5); reference columns off
# by 16 (DPO) and 13 (KTO) nats.
CONTROL_MIN_LOSS_DELTA = 10 * LOSS_ATOL
CONTROL_MIN_LOGP_DELTA = 1.0
# The real Qwen tokenizer tokenizes the text rows, so the tiny model must span its full id range.
VOCAB_SIZE = 151936


def _build_tiny_checkpoint(target_dir: str) -> None:
    """Rank 0: the seeded tiny MoE plus the Qwen tokenizer, saved as a loadable HF checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    torch.manual_seed(SEED)
    config = Qwen3MoeConfig(**{**TINY_QWEN3_MOE_CONFIG, "vocab_size": VOCAB_SIZE}, pad_token_id=0, eos_token_id=1)
    Qwen3MoeForCausalLM(config).to(torch.bfloat16).save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)


def _build_rows(kind: str) -> dict:
    """Text rows with ragged completions, so rows (and a swapped reference) score differently."""
    prompts = [f"Question {i}: what is {i} plus {i}?" for i in range(N_ROWS)]
    answers = [" The answer is " + " ".join(str(2 * i + j) for j in range(1 + i % 5)) + "." for i in range(N_ROWS)]
    if kind == "dpo":
        rejected = [
            " Perhaps " + " ".join(str(3 * i + j) for j in range(1 + (i + 3) % 5)) + " ok." for i in range(N_ROWS)
        ]
        return {"prompt": prompts, "chosen": answers, "rejected": rejected}
    return {"prompt": prompts, "completion": answers, "label": [i % 2 == 0 for i in range(N_ROWS)]}


def _training_args(kind: str, output_dir: str, *, save_at: int | None):
    return TRAINERS[kind][1](
        output_dir=output_dir,
        max_steps=TOTAL_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",
        logging_steps=1,
        logging_nan_inf_filter=False,
        save_strategy="steps" if save_at else "no",
        save_steps=save_at or 0,
        report_to=[],
        seed=SEED,
        bf16=True,
        gradient_checkpointing=False,
        precompute_ref_log_probs=True,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=True,  # EP leaves inactive experts gradient-free
        fsdp="",  # the mixin owns FSDP wrapping
        max_length=MAX_LENGTH,
    )


def _trainer(kind, weights_dir, dataset_dir, output_dir, pc, *, save_at=None, **resume_context):
    """The trainer over a policy loaded from ``weights_dir`` the way the entry scripts load it."""
    model, tokenizer = load_distributed_model(
        model_name_or_path=weights_dir,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        use_liger_kernel=False,
    )
    return TRAINERS[kind][0](
        model=model,
        args=_training_args(kind, output_dir, save_at=save_at),
        train_dataset=load_from_disk(dataset_dir),
        processing_class=tokenizer,
        parallelism_config=pc,
        **resume_context,
    )


def _reference_columns(trainer) -> dict[str, torch.Tensor]:
    return {
        column: torch.tensor(list(trainer.train_dataset[column]), dtype=torch.float32)
        for column in trainer._required_ref_logps_columns()
    }


def _losses_by_step(trainer) -> dict[int, float]:
    return {entry["step"]: entry["loss"] for entry in trainer.state.log_history if "loss" in entry}


def _max_delta(actual: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> float:
    return max(float((actual[column] - reference[column]).abs().max()) for column in reference)


def _finish(trainer) -> None:
    """Release a phase's DeepEP buffers before the next phase builds its own."""
    trainer.cleanup_ep()
    cleanup_memory()


@gpu_test_main(exact_world_size=2, prefix=f"ep_pref_precompute_resume_{ARGS.trainer}")
def run(ctx):
    kind = ARGS.trainer
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}

    shared = shared_scratch_dir(f"ep_pref_precompute_resume_{kind}")
    tiny_dir = os.path.join(shared, "tiny_model")
    dataset_dir = os.path.join(shared, "dataset")
    train_out = os.path.join(shared, "train_out")
    ckpt_dir = os.path.join(train_out, f"checkpoint-{SAVE_AT_STEP}")
    bare_ckpt_dir = os.path.join(shared, "checkpoint_without_sidecar")
    if ctx.rank == 0:
        # A standalone rerun gets the same MASTER_PORT-keyed dir; phase 4 copies into a fixed path.
        shutil.rmtree(shared, ignore_errors=True)
        ctx.on_teardown(lambda: shutil.rmtree(shared, ignore_errors=True))
        _build_tiny_checkpoint(tiny_dir)
        # Built once for every rank and phase, so each phase tokenizes the same rows.
        Dataset.from_dict(_build_rows(kind)).save_to_disk(dataset_dir)
    ctx.barrier()
    pc = ParallelismConfig(ep_size=EP_SIZE)
    first_resumed_step = SAVE_AT_STEP + 1

    log(f"\n--- Phase 1 ({kind}): continuous {TOTAL_STEPS} steps, checkpoint at {SAVE_AT_STEP} ---")
    trainer = _trainer(kind, tiny_dir, dataset_dir, train_out, pc, save_at=SAVE_AT_STEP)
    ctx.on_teardown(trainer.cleanup_ep)
    base_columns = _reference_columns(trainer)
    trainer.train()
    continuous = _losses_by_step(trainer)
    _finish(trainer)
    ctx.barrier()
    checks["continuous_ran_all_steps"] = sorted(continuous) == list(range(1, TOTAL_STEPS + 1))
    checks["checkpoint_carries_the_sidecar"] = os.path.isfile(os.path.join(ckpt_dir, REFERENCE_LOGPS_FILE))
    if not (checks["continuous_ran_all_steps"] and checks["checkpoint_carries_the_sidecar"]):
        return {"checks": checks, "metrics": metrics}
    metrics["continuous_first_resumed_step_loss"] = continuous[first_resumed_step]

    log(f"\n--- Phase 2 ({kind}): resume from checkpoint-{SAVE_AT_STEP}, trainer told ---")
    trainer = _trainer(
        kind,
        ckpt_dir,
        dataset_dir,
        os.path.join(shared, "resume_out"),
        pc,
        resume_checkpoint=ckpt_dir,
        policy_from_checkpoint=True,
    )
    ctx.on_teardown(trainer.cleanup_ep)
    resumed_columns = _reference_columns(trainer)
    trainer.train(resume_from_checkpoint=ckpt_dir)
    resumed = _losses_by_step(trainer)
    _finish(trainer)
    ctx.barrier()
    checks["resumed_reference_columns_equal_the_base_run"] = all(
        torch.equal(resumed_columns[column], base_columns[column]) for column in base_columns
    )
    resumed_delta = abs(resumed.get(first_resumed_step, math.inf) - continuous[first_resumed_step])
    metrics["resumed_first_step_loss"] = resumed.get(first_resumed_step, math.nan)
    metrics["resumed_first_step_loss_delta"] = resumed_delta
    checks["resumed_first_step_loss_matches_continuous"] = resumed_delta < LOSS_ATOL

    log(f"\n--- Phase 3 ({kind}): the same resume, trainer NOT told (control) ---")
    trainer = _trainer(kind, ckpt_dir, dataset_dir, os.path.join(shared, "control_out"), pc)
    ctx.on_teardown(trainer.cleanup_ep)
    control_columns = _reference_columns(trainer)
    trainer.train(resume_from_checkpoint=ckpt_dir)
    control = _losses_by_step(trainer)
    _finish(trainer)
    ctx.barrier()
    metrics["control_reference_max_logp_delta"] = _max_delta(control_columns, base_columns)
    metrics["control_first_step_loss"] = control.get(first_resumed_step, math.nan)
    control_delta = abs(control.get(first_resumed_step, math.inf) - continuous[first_resumed_step])
    metrics["control_first_step_loss_delta"] = control_delta
    checks["control_reference_misses_the_base_run"] = (
        metrics["control_reference_max_logp_delta"] > CONTROL_MIN_LOGP_DELTA
    )
    checks["control_first_step_loss_misses_continuous"] = math.isfinite(control_delta) and (
        control_delta > CONTROL_MIN_LOSS_DELTA
    )

    log(f"\n--- Phase 4 ({kind}): resume from a copy without {REFERENCE_LOGPS_FILE} ---")
    if ctx.rank == 0:
        shutil.copytree(ckpt_dir, bare_ckpt_dir, ignore=shutil.ignore_patterns(REFERENCE_LOGPS_FILE))
    ctx.barrier()
    try:
        _trainer(
            kind,
            bare_ckpt_dir,
            dataset_dir,
            os.path.join(shared, "refusal_out"),
            pc,
            resume_checkpoint=bare_ckpt_dir,
            policy_from_checkpoint=True,
        )
        checks["resume_without_the_sidecar_refuses"] = False
    except RuntimeError as exc:
        checks["resume_without_the_sidecar_refuses"] = "TRAINED weights as the reference" in str(exc)
    cleanup_memory()

    log(
        f"continuous step-{first_resumed_step} loss {continuous[first_resumed_step]:.6f}; resumed "
        f"{metrics['resumed_first_step_loss']:.6f} (|delta| {resumed_delta:.2e}); control "
        f"{metrics['control_first_step_loss']:.6f} (|delta| {control_delta:.2e}); control reference "
        f"max |delta logp| {metrics['control_reference_max_logp_delta']:.3f}"
    )
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
