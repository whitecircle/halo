#!/usr/bin/env python
"""Test: a ``merge_expert_lora_on_save`` checkpoint serves AND resumes exactly (tiny MoE, EP=2).

A merge-on-save checkpoint holds merged bf16 weights for serving. Resuming from those weights builds
the policy on a fold that lost part of the delta to bf16 rounding, re-creates the adapters fresh
(``lora_B = 0``) and restores the adapter optimizer moments onto them: the run silently leaves its
trajectory. So the checkpoint also carries the unmerged adapters (``resume_adapter/``) and a marker,
and resume builds the policy from the BASE and restores the adapters onto it. On a tiny random-init
MoE (``--family``) with native expert LoRA alone or mixed with attention PEFT (``--adapters``):

  1. Uninterrupted run: ``TOTAL_STEPS`` steps, merged checkpoint at ``SAVE_AT_STEP``. The save
     leaves every parameter bit-identical (the attention merge is undone exactly, not by a bf16
     subtraction), and the live adapters are gathered right after it (``on_save``).
  2. The checkpoint serves: stock ``from_pretrained`` loads it with no missing, unexpected or
     mis-shaped keys, its expert (and attention) weights moved off the base (it is the merge, not
     the base), and it carries the resume adapter and its marker with no root ``adapter_config.json``.
  3. Resume from it through the production resolver: the policy source is the BASE; after the
     restore (``on_train_begin``) every adapter is BIT-EQUAL to the one the uninterrupted run held at
     the save; the first resumed step's loss equals the uninterrupted run's within ``FIRST_LOSS_TOL``
     (its forward reads only restored state), the later steps within ``LOSS_TOL`` (the
     stochastic-rounding stream restarts on resume), and the final adapters sit within
     ``FINAL_ADAPTER_RTOL`` of the uninterrupted run's.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_merged_save_resume.py --family qwen3_moe --adapters mixed
"""

import argparse
import math
import os
import random
from types import SimpleNamespace

import torch
import torch.distributed as dist
from accelerate.utils import extract_model_from_parallel
from torch.distributed.tensor import DTensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GptOssConfig,
    GptOssForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
    TrainerCallback,
)
from trl import SFTConfig

import src.optimizers.adamw_bf16 as adamw_bf16_mod
from src.checkpoint.format import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_SAFETENSORS_FILE,
    load_full_state_dict,
    resume_adapter_dir,
)
from src.distributed.expert_parallel.expert_weights import gather_ep_lora_adapters
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.parallelism_config import ParallelismConfig
from src.models.structure import unwrap_model
from src.trainers.sft import DistributedSFTTrainer
from src.training.environment import resolve_resume_weights_source
from tests.common.datasets import create_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B, TINY_GPTOSS_CONFIG, TINY_QWEN3_MOE_CONFIG
from tests.common.peft_helpers import load_peft_model
from tests.common.utils import cleanup_memory, log, step_losses

# Two expert layouts the resume adapter must round-trip: per-expert unfused (Qwen3-MoE) and
# interleaved fused gate_up with attention sinks (GptOss).
FAMILIES = {
    "qwen3_moe": (Qwen3MoeConfig, Qwen3MoeForCausalLM, TINY_QWEN3_MOE_CONFIG),
    "gpt_oss": (GptOssConfig, GptOssForCausalLM, TINY_GPTOSS_CONFIG),
}
# The peft_helpers mode each adapter shape loads through.
ADAPTER_MODES = {"expert": "expert_lora", "mixed": "mixed"}

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=sorted(FAMILIES), default="qwen3_moe")
parser.add_argument("--adapters", choices=sorted(ADAPTER_MODES), default="mixed")
ARGS, _ = parser.parse_known_args()

EP_SIZE = 2
SEED = 42
TOTAL_STEPS = 6
SAVE_AT_STEP = 3
BATCH_SIZE = 2
MAX_SEQ_LENGTH = 256
# High enough that the adapters carry most of the run's movement within a few steps, as a LoRA
# learning rate does.
LEARNING_RATE = 2e-3
# The first resumed step's forward reads only restored state, so it reproduces the uninterrupted
# loss (measured 0.0 on every row); a frozen base one bf16 rounding step off shifts it by ~4e-4.
FIRST_LOSS_TOL = 1e-5
# Later steps also carry the stochastic-rounding stream, which restarts on resume: measured up to
# 2.2e-3, against >=5e-2 when the adapters resume fresh.
LOSS_TOL = 5e-3
# Measured <=9e-3 (the same SR noise), against ~1.0 for fresh adapters.
FINAL_ADAPTER_RTOL = 2e-2


def _build_tiny_checkpoint(family: str, target_dir: str) -> None:
    """Rank 0: the family's seeded tiny MoE at the Qwen tokenizer's vocab, saved with that tokenizer."""
    config_cls, model_cls, tiny_config = FAMILIES[family]
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = config_cls(
        **{**tiny_config, "vocab_size": len(tokenizer)},
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    torch.manual_seed(SEED)
    model_cls(config).to(torch.bfloat16).save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)


def _materialize(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor.full_tensor() if isinstance(tensor, DTensor) else tensor).detach().cpu().clone()


def _adapter_snapshot(model) -> dict[str, torch.Tensor]:
    """Every adapter tensor, whole and on the host: the grouped expert adapters gathered across the EP
    group, and the attention PEFT adapters (``.lora_`` params) un-sharded from FSDP2. Collective."""
    unwrapped = extract_model_from_parallel(model, recursive=True)
    snapshot = {key: value.clone() for key, value in gather_ep_lora_adapters(unwrap_model(unwrapped)).items()}
    for name, param in unwrapped.named_parameters():
        if ".lora_" in name:
            snapshot[name] = _materialize(param.data)
    return snapshot


def _local_parameters(model) -> dict[str, torch.Tensor]:
    """This rank's copy of every parameter, adapters and frozen base alike. Rank-local: a DTensor
    contributes its local shard, resharded first as the save itself does."""
    reshard_fsdp2_modules(model)
    return {
        name: (param.data.to_local() if isinstance(param.data, DTensor) else param.data).detach().clone()
        for name, param in extract_model_from_parallel(model, recursive=True).named_parameters()
    }


class _SaveCapture(TrainerCallback):
    """Around the checkpoint save at ``SAVE_AT_STEP``: every local parameter just before it
    (``on_step_end``) and just after it (``on_save``), plus the full adapters after it. Every rank
    runs the callback, so the adapter gathers stay collective."""

    def __init__(self, trainer_ref: dict):
        self.trainer_ref = trainer_ref

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == SAVE_AT_STEP:
            self.trainer_ref["before_save"] = _local_parameters(self.trainer_ref["trainer"].model)
        return control

    def on_save(self, args, state, control, **kwargs):
        if state.global_step == SAVE_AT_STEP:
            model = self.trainer_ref["trainer"].model
            self.trainer_ref["after_save"] = _local_parameters(model)
            self.trainer_ref["adapters"] = _adapter_snapshot(model)
        return control


class _RestoreCapture(TrainerCallback):
    """The adapters a resume restored, before its first step (``on_train_begin``). Collective."""

    def __init__(self, trainer_ref: dict):
        self.trainer_ref = trainer_ref

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer_ref["adapters"] = _adapter_snapshot(self.trainer_ref["trainer"].model)
        return control


def _sft_config(output_dir: str, *, save: bool) -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir,
        max_steps=TOTAL_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",
        bf16=True,
        logging_steps=1,
        save_strategy="steps" if save else "no",
        save_steps=SAVE_AT_STEP,
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        use_liger_kernel=False,
        seed=SEED,
    )


def _make_trainer(ctx, model_source: str, output_dir: str, *, save: bool):
    """The production load (``split_expert_lora_targets`` → ``load_distributed_model`` →
    ``setup_peft_model``) and trainer, with the stochastic-rounding stream reset so both phases start
    from the same one."""
    adamw_bf16_mod._SR_RNG = random.Random(0xB165EED)
    parallelism_config = ParallelismConfig(ep_size=EP_SIZE, merge_expert_lora_on_save=True)
    model, tokenizer, peft_config = load_peft_model(
        ADAPTER_MODES[ARGS.adapters],
        parallelism_config,
        model_name=model_source,
        attn_implementation="eager",
        use_liger_kernel=False,
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=_sft_config(output_dir, save=save),
        train_dataset=create_sft_dataset(64, tokenizer, seed=SEED),
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
        peft_config=peft_config,
    )
    ctx.on_teardown(trainer.cleanup_ep)
    return trainer, parallelism_config


def _all_ranks_true(local: bool, device) -> bool:
    flag = torch.tensor([1 if local else 0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _serving_checks(checkpoint: str, base_dir: str) -> dict[str, bool]:
    """The merged checkpoint as a serving engine sees it: stock ``from_pretrained``, full coverage,
    the MERGED weights, and the resume state kept out of the root. Rank-local reads only."""
    checks = {}
    model, info = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16, output_loading_info=True)
    problems = {
        kind: info.get(kind) for kind in ("missing_keys", "unexpected_keys", "mismatched_keys") if info.get(kind)
    }
    if problems:
        log(f"  from_pretrained loading info: {problems}")
    checks["merged_checkpoint_loads_with_stock_from_pretrained"] = not problems
    checks["from_pretrained_built_the_model_not_an_adapter"] = (
        type(model).__name__ == FAMILIES[ARGS.family][1].__name__
    )
    del model

    merged, base = load_full_state_dict(checkpoint), load_full_state_dict(base_dir)
    moved = sorted(key for key in set(merged) & set(base) if not torch.equal(merged[key], base[key]))
    checks["merged_experts_carry_the_delta"] = any(".experts." in key for key in moved)
    if ARGS.adapters == "mixed":
        checks["merged_attention_carries_the_delta"] = any(".self_attn." in key for key in moved)
    log(f"  {len(moved)} of {len(base)} base tensors moved by the merge (e.g. {moved[:2]})")

    adapter_dir = resume_adapter_dir(checkpoint)
    checks["checkpoint_is_marked_for_adapter_resume"] = adapter_dir is not None
    checks["resume_adapter_written"] = adapter_dir is not None and all(
        os.path.isfile(os.path.join(adapter_dir, name)) for name in (ADAPTER_SAFETENSORS_FILE, ADAPTER_CONFIG_FILE)
    )
    checks["no_adapter_config_at_the_root"] = not os.path.exists(os.path.join(checkpoint, ADAPTER_CONFIG_FILE))
    return checks


def _relative_l2(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    num = sum(float((a[k].float() - b[k].float()).pow(2).sum()) for k in a)
    den = sum(float(a[k].float().pow(2).sum()) for k in a)
    return math.sqrt(num / den) if den else math.inf


@gpu_test_main(exact_world_size=EP_SIZE, prefix=f"lora_merged_resume_{ARGS.family}_{ARGS.adapters}")
def run(ctx):
    log(
        f"\n{'=' * 70}\n  merge_expert_lora_on_save resume: {ARGS.family}, {ARGS.adapters} adapters, ep{EP_SIZE}\n{'=' * 70}"
    )
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    shared = [ctx.output_dir]
    dist.broadcast_object_list(shared, src=0)
    base_dir = os.path.join(shared[0], "tiny_base")
    train_out = os.path.join(shared[0], "train_out")
    checkpoint = os.path.join(train_out, f"checkpoint-{SAVE_AT_STEP}")
    if ctx.rank == 0:
        _build_tiny_checkpoint(ARGS.family, base_dir)
    ctx.barrier()

    log(f"\n[1/3] Uninterrupted {TOTAL_STEPS}-step run, merged checkpoint at step {SAVE_AT_STEP}...")
    trainer, _ = _make_trainer(ctx, base_dir, train_out, save=True)
    at_save: dict = {"trainer": trainer}
    trainer.add_callback(_SaveCapture(at_save))
    trainer.train()
    uninterrupted = step_losses(trainer)
    final_uninterrupted = _adapter_snapshot(trainer.model)
    checks["uninterrupted_ran_all_steps"] = len(uninterrupted) == TOTAL_STEPS
    before, after = at_save.get("before_save", {}), at_save.get("after_save", {})
    moved = sorted(name for name in before if name not in after or not torch.equal(before[name], after[name]))
    # The run a checkpoint is resumed from must be the run that wrote it: a save that nudges the
    # frozen base (a bf16 merge -> unmerge that does not reverse) leaves every resume off by that step.
    checks["save_left_every_parameter_bit_identical"] = _all_ranks_true(bool(before) and not moved, ctx.device)
    log(f"  {len(before) - len(moved)}/{len(before)} local parameters bit-identical across the save {moved[:2]}")
    del trainer
    cleanup_memory()
    ctx.barrier()

    log("\n[2/3] The merged checkpoint as a server loads it...")
    serving = _serving_checks(checkpoint, base_dir)
    checks.update({name: _all_ranks_true(ok, ctx.device) for name, ok in serving.items()})
    ctx.barrier()

    log(f"\n[3/3] Resuming from {checkpoint}...")
    parallelism_config = ParallelismConfig(ep_size=EP_SIZE, merge_expert_lora_on_save=True)
    source = resolve_resume_weights_source(
        checkpoint, SimpleNamespace(model_name_or_path=base_dir), parallelism_config
    )
    checks["policy_source_is_the_base"] = source == base_dir
    log(f"  policy weights source: {source}")
    trainer, _ = _make_trainer(ctx, source, train_out, save=False)
    restored: dict = {"trainer": trainer}
    trainer.add_callback(_RestoreCapture(restored))
    trainer.train(resume_from_checkpoint=checkpoint)
    resumed = step_losses(trainer)
    final_resumed = _adapter_snapshot(trainer.model)
    del trainer
    cleanup_memory()

    saved_adapters, restored_adapters = at_save.get("adapters", {}), restored.get("adapters", {})
    unequal = sorted(
        key
        for key in saved_adapters
        if key not in restored_adapters or not torch.equal(saved_adapters[key], restored_adapters[key])
    )
    checks["adapters_bit_equal_after_restore"] = bool(saved_adapters) and not unequal
    log(f"  {len(saved_adapters) - len(unequal)}/{len(saved_adapters)} adapters bit-equal after restore {unequal[:2]}")

    tail, reference = resumed[-(TOTAL_STEPS - SAVE_AT_STEP) :], uninterrupted[SAVE_AT_STEP:]
    checks["resumed_ran_remaining_steps"] = len(tail) == len(reference) == TOTAL_STEPS - SAVE_AT_STEP
    if checks["resumed_ran_remaining_steps"]:
        deltas = [
            abs(a - b) if math.isfinite(a) and math.isfinite(b) else math.inf
            for a, b in zip(tail, reference, strict=True)
        ]
        metrics["first_resumed_loss_delta"] = deltas[0]
        metrics["resumed_loss_max_delta"] = max(deltas)
        checks["first_resumed_loss_matches"] = deltas[0] < FIRST_LOSS_TOL
        checks["resumed_losses_track_uninterrupted"] = max(deltas) < LOSS_TOL
        log(
            f"  uninterrupted {[f'{x:.5f}' for x in reference]}  resumed {[f'{x:.5f}' for x in tail]}  "
            f"deltas {[f'{d:.2e}' for d in deltas]}"
        )
    drift = _relative_l2(final_uninterrupted, final_resumed) if final_uninterrupted else math.inf
    metrics["final_adapter_relative_l2"] = drift
    checks["final_adapters_match_uninterrupted"] = drift < FINAL_ADAPTER_RTOL
    log(f"  final adapters, resumed vs uninterrupted: relative L2 {drift:.3e} (tol {FINAL_ADAPTER_RTOL})")
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
