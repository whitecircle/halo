#!/usr/bin/env python
"""Adapter checkpoint resume on the Offline GRPO trainer, across the adapter x parallelism matrix.

``expert_lora`` adds ep2 resumed at ep1: adapters are saved gathered, so the restore stays exact while the optimizer
shards warm-restart loudly. Under EP every DeepEP buffer is deterministic, without which receive order changes how an
adapter's gradient is summed.
"""

import argparse
import math
import os
import shutil

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoTokenizer, TrainerCallback

import halo.checkpoint.parallel.optimizer as optimizer_store_mod
from halo.checkpoint.format import ADAPTER_SAFETENSORS_FILE
from halo.configs.grpo_offline import OfflineGRPOConfig
from halo.distributed.expert_parallel.moe_layer.expert_lora import disable_expert_adapters
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.distributed.parallelism_config import ParallelismConfig
from halo.models.unwrapping import find_peft_model
from halo.trainers.grpo.offline.trainer import OfflineGRPOTrainer
from tests.common.adapters import (
    is_expert_lora_active,
    is_lora_b_key,
    load_peft_model,
    model_name_for,
    parallelism_config_for,
    snapshot_adapters,
)
from tests.common.datasets import create_offline_grpo_dataset
from tests.common.distributed import ensure_model_downloaded, pin_deterministic_ep_dispatch, shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.numerics import max_or_nan
from tests.common.optimizer_state import local_optimizer_state, optimizer_state_matches
from tests.common.rank_log import log, logged_messages
from tests.common.tolerances import TOL
from tests.common.training_checks import step_losses

MODES = ("lora", "lora_ep", "expert_lora", "lora_etp")

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=MODES, default="lora")
ARGS, _ = parser.parse_known_args()

SAVE_AT_STEP = 2
TOTAL_STEPS = 4
LEARNING_RATE = 1e-3  # Large, so the steps before the save move the adapters well clear of every tolerance below.
MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 512
NUM_TRAIN_SAMPLES = 32
BASE_PROBE_TENSORS = 8
BASE_PROBE_ELEMENTS = 4096
SEED = 42
# One whole completion group per step: a step of only negative-advantage rows has no gradient (every token pins to the
# ``min_log_prob`` floor). Accumulated rather than batched: a 4-row gpt-oss forward routes past the first MoE layer's
# cached DeepEP capacity and is refused.
NUM_COMPLETIONS = 4
# Live at every step on the PEFT rows, so the resumed loss depends on the base-vs-adapter gap the restore reproduces.
# An expert-only row builds no PeftModel, and a live beta there loads a second dense 20B reference, so it stays at 0.0.
KL_BETA = 0.02

# Half the tokens are scored, mirroring the prompt/completion split of the trainer's reference pass.
PROBE_TEXT = (
    "User: What is 17 plus 25?\nAssistant: The answer is 42. A resumed adapter must score this "
    "batch exactly as the checkpoint's adapter did, and score it as the frozen base once disabled."
)
PROBE_MAX_TOKENS = 64

# Measured on all four rows: |reference - base| = 0.0, so the bound is kernel headroom. The trained adapters move the
# same log-probs by 0.94 (dense) to 7.5 (ETP) nats; the separation guard rejects adapters that do nothing.
REF_LOGP_TOL = 1e-4
ADAPTER_EFFECT_MIN = 0.25


def _config(
    output_dir: str, max_steps: int, save_at: int | None, pc: ParallelismConfig, mode: str
) -> OfflineGRPOConfig:
    return OfflineGRPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=NUM_COMPLETIONS,
        learning_rate=LEARNING_RATE,
        # Phase 1 stops at 3 and phase 2 at 4, so a decaying schedule would give one global step two LRs.
        lr_scheduler_type="constant",
        kl_beta=0.0 if mode == "expert_lora" else KL_BETA,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="steps" if save_at else "no",
        save_steps=save_at or 0,
        save_only_model=False,
        report_to="none",
        logging_nan_inf_filter=False,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        seed=SEED,
    )


def _make_trainer(mode: str, pc: ParallelismConfig, tokenizer, dataset, config) -> OfflineGRPOTrainer:
    # gpt-oss needs flex_attention once the experts are distributed; dense qwen and the ep1 leg, whose sinks the
    # default reset neutralizes, take the auto default.
    model, _, peft_config = load_peft_model(
        mode, pc, attn_implementation="flex_attention" if pc.has_ep_group else None
    )
    return OfflineGRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        parallelism_config=pc,
    )


def _probe_batch(tokenizer, device):
    ids = tokenizer(PROBE_TEXT, return_tensors="pt", truncation=True, max_length=PROBE_MAX_TOKENS)["input_ids"]
    ids = ids.to(device)
    return ids, torch.ones_like(ids), ids.size(1) // 2


def _adapters_disabled(model):
    """A PEFT-wrapped policy uses ``disable_adapter()``; an expert-only run has no PeftModel and reverts on the EP
    side."""
    peft_model = find_peft_model(model)
    return peft_model.disable_adapter() if isinstance(peft_model, PeftModel) else disable_expert_adapters(model)


def _probe_logps(trainer, probe, *, adapters_disabled: bool) -> torch.Tensor:
    """``adapters_disabled`` only after the last step: ``disable_adapter()`` restores ``requires_grad`` onto the
    unsharded copies and leaves the sharded adapters frozen (``agent-docs/optimization/peft.md``)."""
    ids, mask, keep = probe
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    was_training = trainer.model.training
    trainer.model.eval()
    try:
        with torch.no_grad():
            if adapters_disabled:
                with _adapters_disabled(unwrapped):
                    logps = trainer._get_per_token_logps(trainer.model, ids, mask, keep)
            else:
                logps = trainer._get_per_token_logps(trainer.model, ids, mask, keep)
    finally:
        reshard_fsdp2_modules(trainer.model)
        if was_training:
            trainer.model.train()
    return logps.detach().float().cpu()


def _base_weight_probe(model) -> dict[str, torch.Tensor]:
    """A strided sample of frozen base tensors, local to the rank: an all-gather would average away a rank-local
    divergence. The expert-LoRA restore's ``copy_`` targets sit deep in the stack, past the embedding and layer 0 a
    prefix would sample."""
    frozen = [
        (name, param) for name, param in model.named_parameters() if "lora_" not in name and not param.requires_grad
    ]
    if not frozen:
        return {}
    probe = {}
    for name, param in frozen[:: max(1, len(frozen) // BASE_PROBE_TENSORS)][:BASE_PROBE_TENSORS]:
        local = param.data.to_local() if hasattr(param.data, "to_local") else param.data
        flat = local.detach().flatten()
        probe[name] = flat[:: max(1, flat.numel() // BASE_PROBE_ELEMENTS)][:BASE_PROBE_ELEMENTS].clone().cpu()
    return probe


def _base_weights_unchanged(before: dict, after: dict) -> tuple[bool, str]:
    if not before or set(before) != set(after):
        return False, f"base probe key sets differ ({len(before)} vs {len(after)})"
    changed = sorted(k for k in before if not torch.equal(before[k], after[k]))
    if changed:
        return False, f"{len(changed)}/{len(before)} base tensors changed across the restore, e.g. {changed[:3]}"
    return True, f"{len(before)} base tensors unchanged"


def _optimizer_state_is_warm_restarted(snapshot: dict) -> tuple[bool, str]:
    """No trained optimizer state survived. ``optimizer.state`` is a defaultdict and torch's zero-LR init step counts,
    so an empty check won't do: every moment must be zero and no ``step`` reach the checkpoint's."""
    carried = [
        f"{fqn}.{key}"
        for fqn, entry in snapshot["state"].items()
        for key, value in entry.items()
        if (bool(value.abs().sum() > 0) if torch.is_tensor(value) else (key == "step" and value >= SAVE_AT_STEP))
    ]
    if carried:
        return False, f"{len(carried)} optimizer entries carried across the topology change, e.g. {carried[:3]}"
    return True, f"{len(snapshot['state'])} optimizer entries reinitialized"


def _adapter_file(directory: str) -> dict[str, torch.Tensor]:
    return load_file(os.path.join(directory, ADAPTER_SAFETENSORS_FILE))


def _adapter_files_identical(reference: str, candidate: str) -> tuple[bool, str]:
    ref, cand = _adapter_file(reference), _adapter_file(candidate)
    if set(ref) != set(cand):
        missing, extra = sorted(set(ref) - set(cand)), sorted(set(cand) - set(ref))
        return False, f"key sets differ (missing {missing[:3]}, extra {extra[:3]})"
    differing = sorted(k for k in ref if not torch.equal(ref[k], cand[k]))
    if differing:
        worst = max(differing, key=lambda k: (ref[k].float() - cand[k].float()).abs().max().item())
        delta = (ref[worst].float() - cand[worst].float()).abs().max().item()
        return False, f"{len(differing)}/{len(ref)} tensors differ, worst {worst} by {delta:.3e}"
    return True, f"{len(ref)} adapter tensors bit-identical"


def _checkpoint_carries_trained_adapters(checkpoint: str) -> tuple[bool, str]:
    """The file every restore is compared against holds trained adapters. ``lora_B`` is zero-init, so an all-zero B
    half would pass every "restored == file" check even for a resume that restored nothing."""
    saved = _adapter_file(checkpoint)
    b_tensors = {k: v for k, v in saved.items() if is_lora_b_key(k)}
    if not b_tensors:
        return False, f"no lora_B tensors in the checkpoint (keys: {sorted(saved)[:3]})"
    moved = [k for k, v in b_tensors.items() if v.abs().sum().item() > 0]
    if not moved:
        return False, f"all {len(b_tensors)} saved lora_B tensors are zero — the checkpoint is untrained"
    return True, f"{len(moved)}/{len(b_tensors)} saved lora_B tensors are non-zero"


def _fresh_adapters_are_zero_init(model, *, expert_lora: bool) -> tuple[bool, str]:
    snapshot = snapshot_adapters(model, expert_lora=expert_lora)
    b_tensors = {k: v for k, v in snapshot.items() if is_lora_b_key(k)}
    if not b_tensors:
        return False, f"no lora_B adapters on the fresh model (keys: {sorted(snapshot)[:3]})"
    non_zero = sorted(k for k, v in b_tensors.items() if v.abs().sum().item() > 0)
    if non_zero:
        return False, f"{len(non_zero)} lora_B adapters are non-zero before resume, e.g. {non_zero[:3]}"
    return True, f"{len(b_tensors)} lora_B adapters zero-init"


class _Phase1Capture(TrainerCallback):
    """The frozen-base reading before step 1 and the state at the save, taken inside ``train()`` to match the resumed
    run's."""

    def __init__(self, holder: dict, probe):
        self.holder = holder
        self.probe = probe

    def on_train_begin(self, args, state, control, **kwargs):
        trainer = self.holder["trainer"]
        self.holder["base_logps"] = _probe_logps(trainer, self.probe, adapters_disabled=False)
        self.holder["base_probe"] = _base_weight_probe(trainer.accelerator.unwrap_model(trainer.model))
        return control

    def on_save(self, args, state, control, **kwargs):
        # HF writes a second checkpoint when training stops; the resume reads this step's shards.
        if state.global_step != SAVE_AT_STEP:
            return control
        trainer = self.holder["trainer"]
        self.holder["saved"] = {
            "step": int(state.global_step),
            "optimizer": local_optimizer_state(trainer.model, trainer.optimizer),
        }
        return control


class _ResumeCapture(TrainerCallback):
    """Restored state at on_train_begin. Adapters are re-saved through ``save_model``, the writer that produced the
    checkpoint, so no test-side copy of the key transform can drift. ``restore_config_fields`` re-asserts its config
    snapshot permanently, inert here since the masks come from the collator."""

    def __init__(self, holder: dict, verify_dir: str):
        self.holder = holder
        self.verify_dir = verify_dir

    def on_train_begin(self, args, state, control, **kwargs):
        trainer = self.holder["trainer"]
        trainer.save_model(self.verify_dir)
        self.holder["restored"] = {
            "optimizer": local_optimizer_state(trainer.model, trainer.optimizer),
            "sched_last_epoch": int(trainer.lr_scheduler.last_epoch),
            "base_probe": _base_weight_probe(trainer.accelerator.unwrap_model(trainer.model)),
        }
        return control


def _release(live: dict) -> None:
    """DeepEP buffers first, then the callback cycle that would keep a 20B model resident. The caller drops its own
    binding."""
    trainer = live.pop("trainer", None)
    if trainer is not None:
        trainer.cleanup_ep()
    cleanup_memory()


@gpu_test_main(exact_world_size=2, prefix=f"lora_offgrpo_resume_{ARGS.mode}")
def run(ctx) -> dict:
    mode = ARGS.mode
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}

    log(f"\n{'=' * 70}\n  Offline GRPO adapter RESUME — mode={mode}, world={ctx.world_size}\n{'=' * 70}")

    layout = parallelism_config_for(mode, ctx.world_size)
    if layout.ep_size > 1:
        pin_deterministic_ep_dispatch()
    # The save rank writes the checkpoint and every rank resumes from it, so the output dir is one shared path.
    workspace = shared_scratch_dir(f"lora_offgrpo_resume_{mode}")
    if ctx.rank == 0:
        ctx.on_teardown(lambda: shutil.rmtree(workspace, ignore_errors=True))
    train_out = os.path.join(workspace, "train_out")
    checkpoint = os.path.join(train_out, f"checkpoint-{SAVE_AT_STEP}")

    model_name = model_name_for(mode, layout)
    ensure_model_downloaded(model_name, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = create_offline_grpo_dataset(tokenizer, NUM_TRAIN_SAMPLES, seed=SEED, num_completions=NUM_COMPLETIONS)
    probe = _probe_batch(tokenizer, ctx.device)

    # So a body that raises still releases its DeepEP buffers.
    live: dict = {}
    ctx.on_teardown(lambda: _release(live))

    log(f"\n--- Phase 1 ({mode}): train {TOTAL_STEPS} steps, save at {SAVE_AT_STEP} ---")
    pc = parallelism_config_for(mode, ctx.world_size)
    trainer = _make_trainer(mode, pc, tokenizer, dataset, _config(train_out, TOTAL_STEPS, SAVE_AT_STEP, pc, mode))
    live["trainer"] = trainer
    expert_lora = is_expert_lora_active(trainer.accelerator.unwrap_model(trainer.model))
    checks["adapter_kind_as_declared"] = expert_lora == (mode == "expert_lora")

    # At zero-init lora_B the adapted model is the frozen base, so the reading needs no disable_adapter() before a
    # step.
    ok, detail = _fresh_adapters_are_zero_init(
        trainer.accelerator.unwrap_model(trainer.model), expert_lora=expert_lora
    )
    log(f"phase-1 adapters: {detail}")
    checks["phase1_adapters_zero_init"] = ok

    trainer.add_callback(_Phase1Capture(live, probe))
    trainer.train()
    phase1_losses = step_losses(trainer)
    log(f"phase-1 losses: {[f'{loss:.5f}' for loss in phase1_losses]}")
    checks["phase1_ran_all_steps"] = len(phase1_losses) == TOTAL_STEPS and all(
        math.isfinite(loss) for loss in phase1_losses
    )
    base_logps = live.get("base_logps")
    base_probe = live.get("base_probe")
    saved_state = live.get("saved")
    checks["save_capture_fired"] = saved_state is not None and saved_state["step"] == SAVE_AT_STEP
    checks["base_reference_captured"] = base_logps is not None and bool(base_probe)
    _release(live)
    del trainer  # _release cannot unbind this scope's name, and the next phase loads a second model.

    if ctx.rank == 0:
        ok, detail = _checkpoint_carries_trained_adapters(checkpoint)
        log(f"checkpoint adapters: {detail}")
        checks["checkpoint_carries_trained_adapters"] = ok
    ctx.barrier()

    log(f"\n--- Phase 2 ({mode}): resume checkpoint-{SAVE_AT_STEP} to step {TOTAL_STEPS} ---")
    verify_dir = os.path.join(workspace, "restored_adapters")
    pc = parallelism_config_for(mode, ctx.world_size)
    live.clear()
    trainer = _make_trainer(mode, pc, tokenizer, dataset, _config(train_out, TOTAL_STEPS, None, pc, mode))
    live["trainer"] = trainer

    ok, detail = _fresh_adapters_are_zero_init(
        trainer.accelerator.unwrap_model(trainer.model), expert_lora=expert_lora
    )
    log(f"pre-resume adapters: {detail}")
    checks["fresh_adapters_zero_init"] = ok

    trainer.add_callback(_ResumeCapture(live, verify_dir))
    trainer.train(resume_from_checkpoint=checkpoint)
    resumed_losses = step_losses(trainer)
    log(f"resumed losses: {[f'{loss:.5f}' for loss in resumed_losses]}")

    checks["resumed_reached_final_step"] = trainer.state.global_step == TOTAL_STEPS
    # log_history opens with the checkpoint's entries for steps 1-2, then appends steps 3-4.
    checks["resumed_losses_finite"] = len(resumed_losses) == TOTAL_STEPS and all(
        math.isfinite(loss) for loss in resumed_losses
    )
    if checks["phase1_ran_all_steps"] and checks["resumed_losses_finite"]:
        checks["trainer_state_history_restored"] = resumed_losses[:SAVE_AT_STEP] == phase1_losses[:SAVE_AT_STEP]
        phase1_tail, resumed_tail = phase1_losses[SAVE_AT_STEP:], resumed_losses[SAVE_AT_STEP:]
        deltas = [abs(a - b) for a, b in zip(phase1_tail, resumed_tail, strict=True)]
        metrics["restored_step_loss_delta"] = deltas[0]
        metrics["resume_loss_max_delta"] = max_or_nan(deltas)
        log(
            f"steps {SAVE_AT_STEP + 1}-{TOTAL_STEPS}: phase1={[f'{loss:.5f}' for loss in phase1_tail]} "
            f"resumed={[f'{loss:.5f}' for loss in resumed_tail]} |deltas|={[f'{d:.3e}' for d in deltas]}"
        )
        checks["restored_weights_reproduce_the_loss"] = deltas[0] < TOL.replayed_resume_loss_abs
        checks["resumed_loss_tracks_phase1"] = max_or_nan(deltas) < TOL.replayed_resume_loss_abs

    restored = live.get("restored")
    checks["resume_capture_fired"] = restored is not None
    if restored is not None:
        checks["scheduler_restored"] = restored["sched_last_epoch"] == SAVE_AT_STEP
        ok, detail = _base_weights_unchanged(base_probe or {}, restored["base_probe"])
        log(f"base weights: {detail}")
        checks["base_weights_untouched"] = ok
        if saved_state is not None:
            equal, why = optimizer_state_matches(saved_state["optimizer"], restored["optimizer"])
            if not equal:
                log(f"OPTIMIZER STATE MISMATCH after restore: {why}")
            checks["optimizer_state_restored_exactly"] = equal

    if base_logps is not None:
        ref_logps = _probe_logps(trainer, probe, adapters_disabled=True)
        policy_logps = _probe_logps(trainer, probe, adapters_disabled=False)
        ref_delta = (ref_logps - base_logps).abs().max().item()
        policy_delta = (policy_logps - base_logps).abs().max().item()
        metrics["reference_vs_base_max_delta"] = ref_delta
        metrics["policy_vs_base_max_delta"] = policy_delta
        log(f"probe log-probs: |reference - base|={ref_delta:.3e}, |policy - base|={policy_delta:.5f}")
        checks["reference_logps_are_the_frozen_base"] = ref_delta < REF_LOGP_TOL
        checks["adapters_move_the_policy"] = policy_delta > ADAPTER_EFFECT_MIN
    _release(live)
    del trainer

    if ctx.rank == 0:
        ok, detail = _adapter_files_identical(checkpoint, verify_dir)
        log(f"restored adapters vs checkpoint file: {detail}")
        checks["restored_adapters_match_checkpoint_file"] = ok
    ctx.barrier()

    if mode == "expert_lora":
        log(f"\n--- Phase 3 ({mode}): resume the ep_size=2 checkpoint at ep_size=1 ---")
        ep1_dir = os.path.join(workspace, "restored_adapters_ep1")
        pc = ParallelismConfig(ep_size=1)
        live.clear()
        with logged_messages(optimizer_store_mod.logger) as store_log:
            trainer = _make_trainer(mode, pc, tokenizer, dataset, _config(train_out, TOTAL_STEPS, None, pc, mode))
            live["trainer"] = trainer
            trainer.add_callback(_ResumeCapture(live, ep1_dir))
            trainer.train(resume_from_checkpoint=checkpoint)
        ep1_losses = step_losses(trainer)
        log(f"ep1 resumed losses: {[f'{loss:.5f}' for loss in ep1_losses]}")
        checks["ep1_run_proceeded"] = (
            trainer.state.global_step == TOTAL_STEPS
            and len(ep1_losses) == TOTAL_STEPS
            and all(math.isfinite(loss) for loss in ep1_losses)
        )
        restored = live.get("restored")
        checks["ep1_resume_capture_fired"] = restored is not None
        if restored is not None:
            checks["ep1_scheduler_restored"] = restored["sched_last_epoch"] == SAVE_AT_STEP
            # Per-rank: the warm restart must have happened on every rank, not only the one that logs.
            ok, detail = _optimizer_state_is_warm_restarted(restored["optimizer"])
            log(f"ep1 optimizer: {detail}")
            checks["ep1_optimizer_warm_restarted"] = ok
        # The warm restart logs on the main process only, so the other rank asserts the run continued.
        if ctx.rank == 0:
            warned = any("fingerprint mismatch" in msg and "ep_size" in msg for msg in store_log)
            if not warned:
                log(f"captured optimizer-store messages: {store_log}")
            checks["ep1_optimizer_warm_restart_warned"] = warned
        _release(live)
        del trainer

        if ctx.rank == 0:
            ok, detail = _adapter_files_identical(checkpoint, ep1_dir)
            log(f"ep1-restored adapters vs checkpoint file: {detail}")
            checks["ep1_adapters_match_checkpoint_file"] = ok
        ctx.barrier()

    return {"checks": ctx.broadcast_checks(checks), "metrics": metrics}


if __name__ == "__main__":
    run()
