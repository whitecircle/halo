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
     leaves every parameter bit-identical (it folds each delta into the tensor it writes, out of
     place), and the live adapters are gathered right after it (``on_save``).
  2. The checkpoint serves: stock ``from_pretrained`` loads it with no missing, unexpected or
     mis-shaped keys and no adapter, its expert (and attention) weights moved off the base (it is the
     merge, not the base), and it carries the resume adapter and its marker with no root
     ``adapter_config.json``.
  3. Resume from it through the production resolver: the policy source is the BASE; after the
     restore (``on_train_begin``) every adapter is BIT-EQUAL to the one the uninterrupted run held at
     the save; every resumed step's loss matches the uninterrupted run's within
     ``TOL.replayed_resume_loss_abs`` and the final adapters sit within
     ``TOL.replayed_resume_weight_rtol`` of its. Both runs rewind the bf16 optimizer's
     stochastic-rounding stream at their restore point
     (:class:`~tests.common.checkpoint_io.ReplayRestorePoint`), so the comparison is of the restored
     state alone. DeepEP's default dispatch hands out receive slots with atomics, so the order an
     expert's tokens arrive in, and with it the rounding of each expert adapter gradient summed over
     them, changes from run to run; the body builds every DeepEP buffer in deterministic mode, or two
     identical runs could part by that rounding alone.
  4. A kill between the base save and the resume adapter leaves the merged weights without their
     marker. Resumed through the production resolver, that checkpoint builds the policy from its own
     merged weights, and the resume must refuse on every rank rather than restart the adapters from
     init under their restored optimizer moments.

Under EP+CP a family Ulysses cannot run (``TinyFamily.ulysses_cp``) must instead be refused at load,
on every rank.
"""

import argparse
import functools
import os
import shutil
from collections.abc import Iterable
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer
from trl import SFTConfig

from src.checkpoint.format import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_SAFETENSORS_FILE,
    RESUME_ADAPTER_DIR,
    RESUME_ADAPTER_MARKER_FILE,
    resume_adapter_dir,
)
from src.distributed.context_parallel.validation import UlyssesConfigError
from src.distributed.expert_parallel.extension import deep_ep
from src.distributed.parallelism_config import ParallelismConfig
from src.optimizers.adamw_bf16 import reset_sr_stream
from src.trainers.sft import DistributedSFTTrainer
from src.training.environment import resolve_resume_weights_source
from tests.common.checkpoint_io import ReplayRestorePoint, loading_problems
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import shared_output_dir, world_all
from tests.common.models import QWEN3_0_6B
from tests.common.peft_helpers import (
    attention_target_modules,
    load_peft_model,
    mixed_targets,
    snapshot_adapters,
    unwrap,
)
from tests.common.tiny_models import TINY_MOE_FAMILIES, TinyFamily, shared_tiny_family_checkpoint
from tests.common.tolerances import TOL
from tests.common.utils import (
    finish_phase,
    log,
    relative_l2,
    resumed_loss_deltas,
    snapshot_trainable,
    step_losses,
)
from tests.common.weight_sync import local_parameters, moved_parameters

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


def merged_resume_parser(families: Iterable[str]) -> argparse.ArgumentParser:
    """The CLI a merged-resume suite takes, over the ``families`` it runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(families), required=True)
    parser.add_argument("--adapters", choices=sorted(ADAPTER_MODES), default="mixed")
    parser.add_argument("--ep-size", type=int, choices=sorted({ep for ep, _ in LAYOUTS}), default=2)
    parser.add_argument("--cp-size", type=int, choices=sorted({cp for _, cp in LAYOUTS}), default=1)
    return parser


def _adapter_snapshot(model) -> dict[str, torch.Tensor]:
    """Every adapter tensor, whole and on the host: the grouped expert adapters gathered across the EP
    group, and the attention PEFT adapters (``.lora_`` params) un-sharded from FSDP2. Collective."""
    unwrapped = unwrap(model)
    peft_adapters = {name: value for name, value in snapshot_trainable(unwrapped).items() if ".lora_" in name}
    return {**snapshot_adapters(unwrapped, expert_lora=True), **peft_adapters}


def _parallelism_config(ep_size: int, cp_size: int) -> ParallelismConfig:
    """A fresh config per phase: ``create_ep_config`` caches the ``EPConfig`` it builds on the config,
    so a phase reusing another's would build its model against that phase's expert groups."""
    return ParallelismConfig(ep_size=ep_size, cp_size=cp_size, merge_expert_lora_on_save=True)


def _pin_deterministic_dispatch() -> None:
    """Build every DeepEP ``ElasticBuffer`` of this process in deterministic mode, which places each
    received token by source rank and token index rather than by atomic claim order."""
    buffer_cls = deep_ep().ElasticBuffer
    buffer_cls.__init__ = functools.partialmethod(buffer_cls.__init__, deterministic=True)


class _RestorePoint(ReplayRestorePoint):
    """The replay restore point with every adapter whole (:func:`_adapter_snapshot`) and, at the save,
    this rank's parameters just after it (``parameters``) and just before it (:attr:`before_save`, the
    last ``on_step_end`` ahead of the first save). Every rank runs callbacks, so the gathers stay
    collective."""

    def __init__(self, event: str, trainer):
        super().__init__(event, trainer, capture_optimizer=False)
        self.before_save: dict[str, torch.Tensor] = {}

    def extra(self) -> dict:
        extra = {"adapters": _adapter_snapshot(self.trainer.model)}
        if self.event == "save":
            extra["parameters"] = local_parameters(unwrap(self.trainer.model))
        return extra

    def on_step_end(self, args, state, control, **kwargs):
        if self.event == "save" and self.captured is None:
            self.before_save = local_parameters(unwrap(self.trainer.model))


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


def _serving_checks(family: TinyFamily, checkpoint: str, base_dir: str, attention_targets: set[str]) -> dict:
    """The merged checkpoint as a serving engine sees it: stock ``from_pretrained``, full coverage, no
    adapter, the MERGED weights, and the resume state kept out of the root. Rank-local reads only."""
    checks = {}
    load = {"dtype": torch.bfloat16, "trust_remote_code": family.trust_remote_code}
    model, info = family.load_class.from_pretrained(checkpoint, output_loading_info=True, **load)
    problems = loading_problems(info)
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


def run_merged_resume(ctx, *, family: str, adapters: str, ep_size: int, cp_size: int) -> dict:
    """The four phases above for one family, adapter shape and layout; returns the harness result."""
    tiny = TINY_MOE_FAMILIES[family]
    if ep_size > 1:
        _pin_deterministic_dispatch()
    log(
        f"\n{'=' * 70}\n  merge_expert_lora_on_save resume: {family}, {adapters} adapters, "
        f"ep{ep_size} cp{cp_size} on {WORLD_SIZE} ranks\n{'=' * 70}"
    )
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    output_dir = shared_output_dir(ctx)
    train_out = os.path.join(output_dir, "train_out")
    checkpoint = os.path.join(train_out, f"checkpoint-{SAVE_AT_STEP}")
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    base_dir = shared_tiny_family_checkpoint(ctx, tiny, f"merged_resume_{family}_base", tokenizer, SEED)
    # Read once, off the base: a merged checkpoint's index spells its attention in the family's hub
    # namespace, which need not match the module names PEFT targets.
    targets = (
        mixed_targets(list(tiny.attention_targets or attention_target_modules(base_dir)))
        if adapters == "mixed"
        else None
    )

    def make_trainer(model_source: str, phase_dir: str, *, save: bool):
        """The production load (``split_expert_lora_targets`` → ``load_distributed_model`` →
        ``setup_peft_model``) and trainer, with the stochastic-rounding stream rewound so every phase
        starts from the same one, and the attention modules PEFT adapts. Ulysses CP runs flash
        attention (auto-selected, or through its own probe under the family's
        ``cp_attn_implementation``); the rest stay on eager."""
        reset_sr_stream()
        parallelism_config = _parallelism_config(ep_size, cp_size)
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
            args=_sft_config(phase_dir, save=save),
            train_dataset=create_sft_dataset(64, tokenizer, seed=SEED),
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
            peft_config=peft_config,
        )
        ctx.on_teardown(trainer.cleanup_ep)
        return trainer, set(peft_config.target_modules) if peft_config else set()

    if cp_size > 1 and not tiny.ulysses_cp:
        log("\n[1/1] Ulysses CP cannot run this family: the load must refuse it...")
        try:
            make_trainer(base_dir, train_out, save=True)
            refusal = None
        except UlyssesConfigError as exc:
            refusal = str(exc)
        log(f"  refusal: {(refusal or 'none')[:160]}")
        checks["ulysses_cp_refuses_the_family_on_every_rank"] = world_all(refusal is not None, ctx.device)
        return {"checks": checks, "metrics": metrics}

    log(f"\n[1/4] Uninterrupted {TOTAL_STEPS}-step run, merged checkpoint at step {SAVE_AT_STEP}...")
    trainer, attention_targets = make_trainer(base_dir, train_out, save=True)
    at_save = _RestorePoint("save", trainer)
    trainer.add_callback(at_save)
    trainer.train()
    uninterrupted = step_losses(trainer)
    final_uninterrupted = _adapter_snapshot(trainer.model)
    saved = at_save.captured or {}
    checks["uninterrupted_ran_all_steps"] = len(uninterrupted) == TOTAL_STEPS
    if attention_targets:
        # A target with a dead gradient never leaves lora_B = 0, so its merge has nothing to carry.
        checks["attention_adapters_trained"] = world_all(
            _attention_lora_b_moved(saved.get("adapters", {}), attention_targets), ctx.device
        )
    before = at_save.before_save
    moved = sorted(moved_parameters(before, saved.get("parameters", {})))
    # The run a checkpoint is resumed from must be the run that wrote it: a save that nudges the
    # frozen base (a bf16 merge -> unmerge that does not reverse) leaves every resume off by that step.
    checks["save_left_every_parameter_bit_identical"] = world_all(bool(before) and not moved, ctx.device)
    log(f"  {len(before) - len(moved)}/{len(before)} local parameters bit-identical across the save {moved[:2]}")
    finish_phase(trainer)

    log("\n[2/4] The merged checkpoint as a server loads it...")
    serving = _serving_checks(tiny, checkpoint, base_dir, attention_targets)
    checks.update({name: world_all(ok, ctx.device) for name, ok in serving.items()})
    ctx.barrier()

    log(f"\n[3/4] Resuming from {checkpoint}...")
    base_source = SimpleNamespace(model_name_or_path=base_dir)
    source = resolve_resume_weights_source(checkpoint, base_source, _parallelism_config(ep_size, cp_size))
    checks["policy_source_is_the_base"] = source == base_dir
    log(f"  policy weights source: {source}")
    trainer, _ = make_trainer(source, train_out, save=False)
    restored = _RestorePoint("train_begin", trainer)
    trainer.add_callback(restored)
    trainer.train(resume_from_checkpoint=checkpoint)
    resumed = step_losses(trainer)
    final_resumed = _adapter_snapshot(trainer.model)
    finish_phase(trainer)

    saved_adapters, restored_adapters = saved.get("adapters", {}), (restored.captured or {}).get("adapters", {})
    unequal = sorted(
        key
        for key in saved_adapters
        if key not in restored_adapters or not torch.equal(saved_adapters[key], restored_adapters[key])
    )
    checks["adapters_bit_equal_after_restore"] = bool(saved_adapters) and not unequal
    log(f"  {len(saved_adapters) - len(unequal)}/{len(saved_adapters)} adapters bit-equal after restore {unequal[:2]}")

    deltas = resumed_loss_deltas(uninterrupted, resumed, save_step=SAVE_AT_STEP, total_steps=TOTAL_STEPS)
    checks["resumed_ran_remaining_steps"] = deltas is not None
    if deltas is not None:
        metrics["first_resumed_loss_delta"] = deltas[0]
        metrics["resumed_loss_max_delta"] = max(deltas)
        checks["resumed_losses_match_uninterrupted"] = max(deltas) < TOL.replayed_resume_loss_abs
    drift = relative_l2(final_resumed, final_uninterrupted)
    metrics["final_adapter_relative_l2"] = drift
    checks["final_adapters_match_uninterrupted"] = drift < TOL.replayed_resume_weight_rtol
    log(f"  final adapters, resumed vs uninterrupted: relative L2 {drift:.3e} (tol {TOL.replayed_resume_weight_rtol})")

    log("\n[4/4] Resuming from the same checkpoint without its resume adapter (a kill before it)...")
    torn = os.path.join(output_dir, f"torn-checkpoint-{SAVE_AT_STEP}")
    if ctx.rank == 0:
        shutil.copytree(
            checkpoint, torn, ignore=shutil.ignore_patterns(RESUME_ADAPTER_DIR, RESUME_ADAPTER_MARKER_FILE)
        )
    ctx.barrier()
    source = resolve_resume_weights_source(torn, base_source, _parallelism_config(ep_size, cp_size))
    checks["torn_checkpoint_builds_from_its_merged_weights"] = source == torn
    trainer, _ = make_trainer(source, os.path.join(output_dir, "torn_out"), save=False)
    try:
        trainer.train(resume_from_checkpoint=torn)
        refusal = None
    except ValueError as exc:
        refusal = str(exc)
    log(f"  resume refusal: {(refusal or 'none')[:160]}")
    checks["torn_checkpoint_refuses_on_every_rank"] = world_all(
        refusal is not None and "without its resume adapter" in refusal, ctx.device
    )
    finish_phase(trainer)
    return {"checks": checks, "metrics": metrics}
