"""The ``merge_expert_lora_on_save`` serve-and-resume body the merged-resume GPU suites run.

A merge-on-save checkpoint holds merged bf16 weights for serving. Resuming from those weights builds
the policy on a fold that lost part of the delta to bf16 rounding, re-creates the adapters fresh
(``lora_B = 0``) and restores the adapter optimizer moments onto them: the run silently leaves its
trajectory. So the checkpoint also carries the unmerged adapters (``resume_adapter/``) and a marker,
and resume builds the policy from the BASE and restores the adapters onto it. On a family's tiny
random-init MoE (:data:`~tests.common.tiny_models.TINY_MOE_FAMILIES`), with native expert LoRA alone
or mixed with attention PEFT, on two ranks at ``ep_size`` 2 (plain per-rank experts, FSDP-ignored)
or 1 (experts FSDP-sharded as DTensors, DP=2), with ``cp_size`` 2 under EP+CP:

  1. Uninterrupted run: ``TOTAL_STEPS`` steps, merged checkpoint at ``SAVE_AT_STEP``. The save
     leaves every parameter bit-identical (the attention merge is undone exactly, not by a bf16
     subtraction), and the live adapters are gathered right after it (``on_save``).
  2. The checkpoint serves: stock ``from_pretrained`` loads it with no missing, unexpected or
     mis-shaped keys and no adapter, its expert (and attention) weights moved off the base (it is the
     merge, not the base), and it carries the resume adapter and its marker with no root
     ``adapter_config.json``.
  3. Resume from it through the production resolver: the policy source is the BASE; after the
     restore (``on_train_begin``) every adapter is BIT-EQUAL to the one the uninterrupted run held at
     the save; every resumed step's loss matches the uninterrupted run's within ``LOSS_TOL`` and the
     final adapters sit within ``FINAL_ADAPTER_RTOL`` of its. A resumed process restarts the
     stochastic-rounding stream of the bf16 optimizer (``_SR_RNG``), which the uninterrupted run
     would otherwise have advanced past the save, so both restart it at the step after the save: the
     comparison is then of the restored state alone.
  4. A kill between the base save and the resume adapter leaves the merged weights without their
     marker. Resumed through the production resolver, that checkpoint builds the policy from its own
     merged weights, and the resume must refuse on every rank rather than restart the adapters from
     init under their restored optimizer moments.

Under EP+CP a family Ulysses cannot run (``TinyFamily.ulysses_cp``) must instead be refused at load,
on every rank.
"""

import argparse
import math
import os
import random
import shutil
from collections.abc import Iterable
from types import SimpleNamespace

import torch
import torch.distributed as dist
from accelerate.utils import extract_model_from_parallel
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer, TrainerCallback
from trl import SFTConfig

import src.optimizers.adamw_bf16 as adamw_bf16_mod
from src.checkpoint.format import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_SAFETENSORS_FILE,
    RESUME_ADAPTER_DIR,
    RESUME_ADAPTER_MARKER_FILE,
    resume_adapter_dir,
)
from src.distributed.context_parallel.validation import UlyssesConfigError
from src.distributed.expert_parallel.expert_weights import gather_ep_lora_adapters
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.parallelism_config import ParallelismConfig
from src.models.structure import unwrap_model
from src.trainers.sft import DistributedSFTTrainer
from src.training.environment import resolve_resume_weights_source
from tests.common.datasets import create_sft_dataset
from tests.common.models import QWEN3_0_6B
from tests.common.peft_helpers import attention_target_modules, load_peft_model, mixed_targets
from tests.common.tiny_models import TINY_MOE_FAMILIES, TinyFamily, build_tiny_family_checkpoint
from tests.common.utils import cleanup_memory, log, step_losses

# The peft_helpers mode each adapter shape loads through.
ADAPTER_MODES = {"expert": "expert_lora", "mixed": "mixed"}
# (ep_size, cp_size) on two ranks: the layouts merge-on-save runs under.
LAYOUTS = ((2, 1), (1, 1), (2, 2))
WORLD_SIZE = 2
SEED = 42
TOTAL_STEPS = 6
SAVE_AT_STEP = 3
BATCH_SIZE = 2
MAX_SEQ_LENGTH = 256
# High enough that the adapters carry most of the run's movement within a few steps, as a LoRA
# learning rate does.
LEARNING_RATE = 2e-3
# Seed both runs restart the bf16 optimizer's stochastic-rounding stream from.
SR_SEED = 0xB165EED
# Measured on B300 over the 60 training rows (every family, adapter shape and layout): every resumed
# step's loss equals the uninterrupted run's exactly (|delta| 0.0), and the final adapters sit within
# 4.6e-6 relative L2 (mixed rows; 0.0 on most). Adapters resumed fresh over the merged weights, the
# failure this pins, miss the first step by >=1.2e-4, later steps by >=5e-2 and the adapters by ~1.0.
LOSS_TOL = 1e-4
FINAL_ADAPTER_RTOL = 1e-4


def merged_resume_parser(families: Iterable[str]) -> argparse.ArgumentParser:
    """The CLI a merged-resume suite takes, over the ``families`` it runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(families), required=True)
    parser.add_argument("--adapters", choices=sorted(ADAPTER_MODES), default="mixed")
    parser.add_argument("--ep-size", type=int, choices=sorted({ep for ep, _ in LAYOUTS}), default=2)
    parser.add_argument("--cp-size", type=int, choices=sorted({cp for _, cp in LAYOUTS}), default=1)
    return parser


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


def _restart_sr_stream() -> None:
    """Restart the bf16 optimizer's stochastic-rounding stream, as a fresh process does."""
    adamw_bf16_mod._SR_RNG = random.Random(SR_SEED)


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
            _restart_sr_stream()
        return control


class _RestoreCapture(TrainerCallback):
    """The adapters a resume restored, before its first step (``on_train_begin``). Collective."""

    def __init__(self, trainer_ref: dict):
        self.trainer_ref = trainer_ref

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer_ref["adapters"] = _adapter_snapshot(self.trainer_ref["trainer"].model)
        # After the optimizer-state load, whose zero-LR materialization step draws from the stream.
        _restart_sr_stream()
        return control


def _sft_config(output_dir: str, *, save: bool) -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir,
        max_steps=TOTAL_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",
        bf16=True,
        # Off for every family: Zaya refuses it, and none of what this body checks depends on it.
        gradient_checkpointing=False,
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


def _all_ranks_true(local: bool, device) -> bool:
    flag = torch.tensor([1 if local else 0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _serving_checks(family: TinyFamily, checkpoint: str, base_dir: str, attention_targets: set[str]) -> dict:
    """The merged checkpoint as a serving engine sees it: stock ``from_pretrained``, full coverage, no
    adapter, the MERGED weights, and the resume state kept out of the root. Rank-local reads only."""
    checks = {}
    load = {"dtype": torch.bfloat16, "trust_remote_code": family.trust_remote_code}
    model, info = family.load_class.from_pretrained(checkpoint, output_loading_info=True, **load)
    problems = {
        kind: info.get(kind) for kind in ("missing_keys", "unexpected_keys", "mismatched_keys") if info.get(kind)
    }
    if problems:
        log(f"  from_pretrained loading info: {problems}")
    checks["merged_checkpoint_loads_with_stock_from_pretrained"] = not problems
    checks["from_pretrained_loaded_no_adapter"] = not getattr(model, "_hf_peft_config_loaded", False)
    # Compared as loaded, in the module namespace: a family's on-disk expert spelling differs between
    # the gathered save and save_pretrained (fused vs per-expert for Cohere2 MoE).
    merged = model.state_dict()
    base = family.load_class.from_pretrained(base_dir, **load).state_dict()
    del model
    moved = sorted(key for key in base if key in merged and not torch.equal(merged[key], base[key]))
    checks["merged_experts_carry_the_delta"] = any("expert" in key for key in moved)
    if attention_targets:
        checks["merged_attention_carries_the_delta"] = any(
            key.rsplit(".", 2)[-2] in attention_targets for key in moved if key.endswith(".weight")
        )
    log(f"  {len(moved)} of {len(base)} base tensors moved by the merge (e.g. {moved[:2]})")

    adapter_dir = resume_adapter_dir(checkpoint)
    checks["checkpoint_is_marked_for_adapter_resume"] = adapter_dir is not None
    checks["resume_adapter_written"] = adapter_dir is not None and all(
        os.path.isfile(os.path.join(adapter_dir, name)) for name in (ADAPTER_SAFETENSORS_FILE, ADAPTER_CONFIG_FILE)
    )
    checks["no_adapter_config_at_the_root"] = not os.path.exists(os.path.join(checkpoint, ADAPTER_CONFIG_FILE))
    return checks


def _attention_lora_b_moved(adapters: dict[str, torch.Tensor], attention_targets: set[str]) -> bool:
    """Whether any attention ``lora_B`` left its zero init."""
    return any(
        bool(value.any())
        for key, value in adapters.items()
        if ".lora_B." in key and key.split(".lora_B.")[0].rsplit(".", 1)[-1] in attention_targets
    )


def _relative_l2(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    num = sum(float((a[k].float() - b[k].float()).pow(2).sum()) for k in a)
    den = sum(float(a[k].float().pow(2).sum()) for k in a)
    return math.sqrt(num / den) if den else math.inf


def run_merged_resume(ctx, *, family: str, adapters: str, ep_size: int, cp_size: int) -> dict:
    """The four phases above for one family, adapter shape and layout; returns the harness result."""
    tiny = TINY_MOE_FAMILIES[family]
    log(
        f"\n{'=' * 70}\n  merge_expert_lora_on_save resume: {family}, {adapters} adapters, "
        f"ep{ep_size} cp{cp_size} on {WORLD_SIZE} ranks\n{'=' * 70}"
    )
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    shared = [ctx.output_dir]
    dist.broadcast_object_list(shared, src=0)
    base_dir = os.path.join(shared[0], "tiny_base")
    train_out = os.path.join(shared[0], "train_out")
    checkpoint = os.path.join(train_out, f"checkpoint-{SAVE_AT_STEP}")
    if ctx.rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        build_tiny_family_checkpoint(tiny, base_dir, tokenizer, SEED)
    ctx.barrier()
    # Read once, off the base: a merged checkpoint's index spells its attention in the family's hub
    # namespace, which need not match the module names PEFT targets.
    targets = (
        mixed_targets(list(tiny.attention_targets or attention_target_modules(base_dir)))
        if adapters == "mixed"
        else None
    )

    def make_trainer(model_source: str, output_dir: str, *, save: bool):
        """The production load (``split_expert_lora_targets`` → ``load_distributed_model`` →
        ``setup_peft_model``) and trainer, with the stochastic-rounding stream restarted so every phase
        starts from the same one. Ulysses CP runs flash attention (auto-selected, or through its own
        probe under the family's ``cp_attn_implementation``); the rest stay on eager."""
        _restart_sr_stream()
        parallelism_config = ParallelismConfig(ep_size=ep_size, cp_size=cp_size, merge_expert_lora_on_save=True)
        model, tokenizer, peft_config = load_peft_model(
            ADAPTER_MODES[adapters],
            parallelism_config,
            model_name=model_source,
            attn_implementation="eager" if cp_size == 1 else tiny.cp_attn_implementation,
            use_liger_kernel=False,
            lora_target_modules=targets,
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
        return trainer, parallelism_config, set(peft_config.target_modules) if peft_config else set()

    if cp_size > 1 and not tiny.ulysses_cp:
        log("\n[1/1] Ulysses CP cannot run this family: the load must refuse it...")
        try:
            make_trainer(base_dir, train_out, save=True)
            refusal = None
        except UlyssesConfigError as exc:
            refusal = str(exc)
        log(f"  refusal: {(refusal or 'none')[:160]}")
        checks["ulysses_cp_refuses_the_family_on_every_rank"] = _all_ranks_true(refusal is not None, ctx.device)
        return {"checks": checks, "metrics": metrics}

    log(f"\n[1/4] Uninterrupted {TOTAL_STEPS}-step run, merged checkpoint at step {SAVE_AT_STEP}...")
    trainer, _, attention_targets = make_trainer(base_dir, train_out, save=True)
    at_save: dict = {"trainer": trainer}
    trainer.add_callback(_SaveCapture(at_save))
    trainer.train()
    uninterrupted = step_losses(trainer)
    final_uninterrupted = _adapter_snapshot(trainer.model)
    checks["uninterrupted_ran_all_steps"] = len(uninterrupted) == TOTAL_STEPS
    if attention_targets:
        # A target with a dead gradient never leaves lora_B = 0, so its merge has nothing to carry.
        checks["attention_adapters_trained"] = _all_ranks_true(
            _attention_lora_b_moved(at_save.get("adapters", {}), attention_targets), ctx.device
        )
    before, after = at_save.get("before_save", {}), at_save.get("after_save", {})
    moved = sorted(name for name in before if name not in after or not torch.equal(before[name], after[name]))
    # The run a checkpoint is resumed from must be the run that wrote it: a save that nudges the
    # frozen base (a bf16 merge -> unmerge that does not reverse) leaves every resume off by that step.
    checks["save_left_every_parameter_bit_identical"] = _all_ranks_true(bool(before) and not moved, ctx.device)
    log(f"  {len(before) - len(moved)}/{len(before)} local parameters bit-identical across the save {moved[:2]}")
    del trainer
    cleanup_memory()
    ctx.barrier()

    log("\n[2/4] The merged checkpoint as a server loads it...")
    serving = _serving_checks(tiny, checkpoint, base_dir, attention_targets)
    checks.update({name: _all_ranks_true(ok, ctx.device) for name, ok in serving.items()})
    ctx.barrier()

    log(f"\n[3/4] Resuming from {checkpoint}...")
    parallelism_config = ParallelismConfig(ep_size=ep_size, cp_size=cp_size, merge_expert_lora_on_save=True)
    source = resolve_resume_weights_source(
        checkpoint, SimpleNamespace(model_name_or_path=base_dir), parallelism_config
    )
    checks["policy_source_is_the_base"] = source == base_dir
    log(f"  policy weights source: {source}")
    trainer, _, _ = make_trainer(source, train_out, save=False)
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
        checks["resumed_losses_match_uninterrupted"] = max(deltas) < LOSS_TOL
        log(
            f"  uninterrupted {[f'{x:.5f}' for x in reference]}  resumed {[f'{x:.5f}' for x in tail]}  "
            f"deltas {[f'{d:.2e}' for d in deltas]}"
        )
    drift = _relative_l2(final_uninterrupted, final_resumed) if final_uninterrupted else math.inf
    metrics["final_adapter_relative_l2"] = drift
    checks["final_adapters_match_uninterrupted"] = drift < FINAL_ADAPTER_RTOL
    log(f"  final adapters, resumed vs uninterrupted: relative L2 {drift:.3e} (tol {FINAL_ADAPTER_RTOL})")

    log("\n[4/4] Resuming from the same checkpoint without its resume adapter (a kill before it)...")
    torn = os.path.join(shared[0], f"torn-checkpoint-{SAVE_AT_STEP}")
    if ctx.rank == 0:
        shutil.copytree(
            checkpoint, torn, ignore=shutil.ignore_patterns(RESUME_ADAPTER_DIR, RESUME_ADAPTER_MARKER_FILE)
        )
    ctx.barrier()
    source = resolve_resume_weights_source(torn, SimpleNamespace(model_name_or_path=base_dir), parallelism_config)
    checks["torn_checkpoint_builds_from_its_merged_weights"] = source == torn
    trainer, _, _ = make_trainer(source, os.path.join(shared[0], "torn_out"), save=False)
    try:
        trainer.train(resume_from_checkpoint=torn)
        refusal = None
    except ValueError as exc:
        refusal = str(exc)
    log(f"  resume refusal: {(refusal or 'none')[:160]}")
    checks["torn_checkpoint_refuses_on_every_rank"] = _all_ranks_true(
        refusal is not None and "without its resume adapter" in refusal, ctx.device
    )
    del trainer
    cleanup_memory()
    return {"checks": checks, "metrics": metrics}
