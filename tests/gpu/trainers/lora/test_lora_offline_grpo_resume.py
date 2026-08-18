#!/usr/bin/env python
"""Adapter checkpoint RESUME on the Offline GRPO trainer, across the adapter x parallelism matrix.

``test_lora_offline_grpo.py`` proves an adapter run TRAINS and round-trips a save; this file proves it
RESUMES. Phase 1 trains 4 steps saving at step 2; phase 2 builds a FRESH trainer from the base model
(an adapter checkpoint carries no base weights, so the loader restores only the adapters) and resumes
that checkpoint to step 4, so phase 1 is also the continuous reference its tail is compared against.
Every check fails when a piece of the resume breaks:

  - the restored adapters equal the checkpoint FILE bit for bit — re-saved through the same
    ``save_model`` that wrote the checkpoint, so the two files differ only if the restore did;
  - the frozen base the adapters ride on is byte-identical to the pre-training one;
  - this rank's optimizer state at ``on_train_begin`` is bit-equal to the phase-1 step-2 snapshot,
    and the LR scheduler is back at step 2;
  - the resumed run's steps 3-4 track phase 1's, checked at the two precisions they hold: step 3 is a
    forward on the restored WEIGHTS and reproduces phase 1's loss exactly, while step 4 follows an
    update driven by the restored MOMENTS and carries the resume's stochastic-rounding noise;
  - the resumed adapters, DISABLED, score a fixed batch as the frozen base scored it before any
    training — the reference log-probs offline GRPO computes at ``kl_beta > 0``. Anti-vacuous by the
    companion check that the same adapters ENABLED move those log-probs by orders more. The base
    reading is taken with the adapters live but still zero-init, where ``lora_B @ lora_A`` is exactly
    zero and the adapted model IS the base (asserted, not assumed).

Modes / parallelism (``--mode``):
    lora        — attention PEFT LoRA, dense Qwen3-0.6B, FSDP2
    lora_ep     — attention PEFT LoRA, GptOss-20B MoE, ep_size=2
    expert_lora — native grouped LoRA on the MoE expert FFNs, GptOss-20B, ep_size=2
    lora_etp    — attention PEFT LoRA, GptOss-20B MoE, ep_size=1 + expert_tp_size=2 (pure ETP)

``expert_lora`` adds a third phase: the ep_size=2 checkpoint resumed at ep_size=1. The adapters are
saved gathered to the full expert count, so the ep1 restore must still be exact even though the
optimizer shards cannot map onto the new topology and warm-restart loudly.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_offline_grpo_resume.py --mode lora
    ... --mode lora_ep | --mode expert_lora | --mode lora_etp
"""

import argparse
import logging
import math
import os
import shutil

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoTokenizer, TrainerCallback

import src.distributed.checkpoint.optimizer as optimizer_store_mod
from src.checkpoint.format import ADAPTER_SAFETENSORS_FILE
from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.distributed.checkpoint.peft import find_peft_model
from src.distributed.expert_parallel.base_layer import disable_expert_adapters
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.grpo.offline import OfflineGRPOTrainer
from tests.common.datasets import create_offline_grpo_dataset
from tests.common.distributed import ensure_model_downloaded, shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.peft_helpers import (
    is_expert_lora_active,
    is_lora_b_key,
    load_peft_model,
    model_name_for,
    parallelism_config_for,
    snapshot_adapters,
)
from tests.common.utils import (
    cleanup_memory,
    local_optimizer_state,
    log,
    optimizer_state_matches,
    step_losses,
)

MODES = ("lora", "lora_ep", "expert_lora", "lora_etp")

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=MODES, default="lora")
ARGS, _ = parser.parse_known_args()

SAVE_AT_STEP = 2
TOTAL_STEPS = 4
LEARNING_RATE = 1e-3  # large, so three steps move the adapters well clear of every tolerance below
MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 512
NUM_TRAIN_SAMPLES = 32
BASE_PROBE_TENSORS = 8
BASE_PROBE_ELEMENTS = 4096
SEED = 42
# One whole completion group per optimizer step: a step of only negative-advantage rows yields NO
# gradient (every token pins to the flat ``min_log_prob`` floor), while a full group always carries
# unclamped positive-advantage rows. Accumulated rather than batched — a 4-row gpt-oss forward
# routes past the first MoE layer's cached DeepEP capacity and is refused.
NUM_COMPLETIONS = 4
# Live at every step on the PEFT-wrapped rows, so the resumed loss depends on the base-vs-adapter gap
# the restore is supposed to reproduce. The expert-only row builds no PeftModel: a live beta there
# loads a second DENSE 20B reference instead of reverting adapters, so it stays at the 0.0 default
# and its reference invariant is probed directly (``_probe_logps``) like every other row's.
KL_BETA = 0.02

# Fixed-batch probe of the policy function. Half the tokens are scored, mirroring the
# prompt/completion split the trainer's own reference pass runs on.
PROBE_TEXT = (
    "User: What is 17 plus 25?\nAssistant: The answer is 42. A resumed adapter must score this "
    "batch exactly as the checkpoint's adapter did, and score it as the frozen base once disabled."
)
PROBE_MAX_TOKENS = 64

# Measured, all four rows: |reference - base| = 0.0 exactly (disabling the adapters
# restores the base computation itself, not an approximation of it), and the trained adapters move
# the same log-probs by 0.94 (dense) to 7.5 (ETP) nats. The tolerance leaves room for the bf16
# expert combine DeepEP reorders per launch; the separation guard is what keeps the equality from
# passing on a model whose adapters do nothing.
REF_LOGP_TOL = 0.05
ADAPTER_EFFECT_MIN = 0.25
# Step 3 replays phase 1's forward on the restored weights: measured delta 0.0 on all four rows, the
# band covering only the logged rounding and the bf16 combine reorder. Step 4 additionally carries
# the stochastic-rounding stream restarting on resume — measured up to 0.067 (lora_etp) across
# repeats — so its band is the loose one; the bit-exact optimizer comparison pins the moments.
RESTORED_STEP_LOSS_TOL = 2e-3
LOSS_TOL = 0.15


def _config(
    output_dir: str, max_steps: int, save_at: int | None, pc: ParallelismConfig, mode: str
) -> OfflineGRPOConfig:
    return OfflineGRPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=NUM_COMPLETIONS,
        learning_rate=LEARNING_RATE,
        # Step-count-invariant: phase 1 stops at 3 and phase 2 at 4, so a decaying schedule would
        # give the same global step a different LR in each and break the loss comparison.
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
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        seed=SEED,
        fsdp="",
        ddp_find_unused_parameters=pc.is_ep_mode,
    )


def _make_trainer(mode: str, pc: ParallelismConfig, tokenizer, dataset, config) -> OfflineGRPOTrainer:
    # gpt-oss needs flex_attention once the experts are distributed; dense qwen and the ep1 leg,
    # whose sinks the default reset neutralizes, take the auto default.
    model, _, peft_config = load_peft_model(mode, pc, attn_implementation="flex_attention" if pc.is_ep_mode else None)
    return OfflineGRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        parallelism_config=pc,
    )


def _probe_batch(tokenizer, device):
    """(input_ids, attention_mask, logits_to_keep) — identical tokens in every phase."""
    ids = tokenizer(PROBE_TEXT, return_tensors="pt", truncation=True, max_length=PROBE_MAX_TOKENS)["input_ids"]
    ids = ids.to(device)
    return ids, torch.ones_like(ids), ids.size(1) // 2


def _adapters_disabled(model):
    """The production reference context for this model's adapter kind.

    A PEFT-wrapped policy uses ``disable_adapter()`` — already EP-aware where native expert adapters
    ride along, patched by ``make_disable_adapter_ep_aware``. An expert-only run builds no PeftModel,
    so its adapters are reverted by the EP-side context directly.
    """
    peft_model = find_peft_model(model)
    return peft_model.disable_adapter() if isinstance(peft_model, PeftModel) else disable_expert_adapters(model)


def _probe_logps(trainer, probe, *, adapters_disabled: bool) -> torch.Tensor:
    """Per-token log-probs of the fixed probe, through the trainer's own reference kernel.

    ``reshard_fsdp2_modules`` afterwards for the reason its own docstring gives. ``adapters_disabled``
    is used only AFTER the last training step: opening ``disable_adapter()`` here would restore
    ``requires_grad`` onto the unsharded copies this forward creates and leave the sharded adapters
    frozen (``agent-docs/optimization/peft.md``), which is why the frozen-base reading below is taken from
    the zero-init adapters rather than by disabling them.
    """
    ids, mask, keep = probe
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    was_training = trainer.model.training
    trainer.model.eval()
    try:
        with torch.no_grad():
            if adapters_disabled:
                with _adapters_disabled(unwrapped):
                    logps, _ = trainer._get_per_token_logps(trainer.model, ids, mask, keep)
            else:
                logps, _ = trainer._get_per_token_logps(trainer.model, ids, mask, keep)
    finally:
        reshard_fsdp2_modules(trainer.model)
        if was_training:
            trainer.model.train()
    return logps.detach().float().cpu()


def _base_weight_probe(model) -> dict[str, torch.Tensor]:
    """A strided sample of frozen base tensors, for the "the restore touched only adapters" comparison.

    Local shards, no collective: a rank-local divergence is the failure this looks for, and averaging
    it across an all-gather would hide exactly that. Strided on both axes because a prefix would
    certify the wrong thing — the first frozen tensors of every family here are the embedding and
    layer 0's attention, while the restore that CAN corrupt base weights is the expert-LoRA one, whose
    ``copy_`` targets are resolved per expert projection deep in the stack. Within a tensor the sample
    is strided too: the corruption is a whole-tensor ``copy_`` into the wrong parameter, so a spread
    of elements catches it without holding gigabytes of a 20B model's experts across two phases.
    """
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
    """No TRAINED optimizer state survived: every entry is back at its freshly-initialized value.

    Not "the state is empty": ``optimizer.state`` is a defaultdict and ``get_optimizer_state_dict``
    materializes empty entries through a zero-grad init step, so the discriminating reading is that
    every moment is zero and no ``step`` counter reached the checkpoint's.
    """
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
    """Rank 0: two adapter saves hold the same tensors, key for key, bit for bit."""
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
    """Rank 0 anti-vacuity: the file every restore is compared against must hold TRAINED adapters.

    ``lora_B`` is zero-init, so an all-zero B half would mean phase 1 wrote its adapters before
    training them, and every "restored == file" verdict would then also hold for a resume that
    restored nothing at all.
    """
    saved = _adapter_file(checkpoint)
    b_tensors = {k: v for k, v in saved.items() if is_lora_b_key(k)}
    if not b_tensors:
        return False, f"no lora_B tensors in the checkpoint (keys: {sorted(saved)[:3]})"
    moved = [k for k, v in b_tensors.items() if v.abs().sum().item() > 0]
    if not moved:
        return False, f"all {len(b_tensors)} saved lora_B tensors are zero — the checkpoint is untrained"
    return True, f"{len(moved)}/{len(b_tensors)} saved lora_B tensors are non-zero"


def _fresh_adapters_are_zero_init(model, *, expert_lora: bool) -> tuple[bool, str]:
    """Every ``lora_B`` of a freshly built model is zero — the state a resume has to overwrite."""
    snapshot = snapshot_adapters(model, expert_lora=expert_lora)
    b_tensors = {k: v for k, v in snapshot.items() if is_lora_b_key(k)}
    if not b_tensors:
        return False, f"no lora_B adapters on the fresh model (keys: {sorted(snapshot)[:3]})"
    non_zero = sorted(k for k, v in b_tensors.items() if v.abs().sum().item() > 0)
    if non_zero:
        return False, f"{len(non_zero)} lora_B adapters are non-zero before resume, e.g. {non_zero[:3]}"
    return True, f"{len(b_tensors)} lora_B adapters zero-init"


class _Phase1Capture(TrainerCallback):
    """Phase 1: the frozen-base reading before step 1, and the state the shards hold at the save.

    Both are taken from inside ``train()`` so they are comparable with the resumed run's — the model
    is wrapped and the parameters sharded exactly as they will be at ``on_train_begin`` of phase 2.
    """

    def __init__(self, holder: dict, probe):
        self.holder = holder
        self.probe = probe

    def on_train_begin(self, args, state, control, **kwargs):
        trainer = self.holder["trainer"]
        self.holder["base_logps"] = _probe_logps(trainer, self.probe, adapters_disabled=False)
        self.holder["base_probe"] = _base_weight_probe(trainer.accelerator.unwrap_model(trainer.model))
        return control

    def on_save(self, args, state, control, **kwargs):
        # HF writes a second checkpoint when training stops, so pin the snapshot to the step whose
        # shards the resume actually reads — otherwise the comparison is against the later state.
        if state.global_step != SAVE_AT_STEP:
            return control
        trainer = self.holder["trainer"]
        self.holder["saved"] = {
            "step": int(state.global_step),
            "optimizer": local_optimizer_state(trainer.model, trainer.optimizer),
        }
        return control


class _ResumeCapture(TrainerCallback):
    """Snapshot restored state at on_train_begin: after the resume hooks, before the first step.

    The adapters are captured by re-saving them through ``save_model`` — the very writer that
    produced the checkpoint — so the two files are comparable bit for bit and no test-side
    re-implementation of the adapter key transform can drift away from the real one. Two effects the
    save carries into the run: ``restore_special_token_ids`` re-asserts its config snapshot
    permanently, and on a warm-restarted optimizer the state read below materializes empty state with
    a zero-grad step. Both are inert here — the offline-GRPO masks come from the collator, and no
    check downstream of a warm restart compares losses.
    """

    def __init__(self, holder: dict, verify_dir: str):
        self.holder = holder
        self.verify_dir = verify_dir

    def on_train_begin(self, args, state, control, **kwargs):
        trainer = self.holder["trainer"]
        trainer.save_model(self.verify_dir)
        # After the save: get_optimizer_state_dict materializes empty state with a zero-grad init
        # step, which must not land between the restore and the adapter snapshot above.
        self.holder["restored"] = {
            "optimizer": local_optimizer_state(trainer.model, trainer.optimizer),
            "sched_last_epoch": int(trainer.lr_scheduler.last_epoch),
            "base_probe": _base_weight_probe(trainer.accelerator.unwrap_model(trainer.model)),
        }
        return control


class _WarningTap(logging.Handler):
    """Collect the optimizer store's warm-restart warnings for the topology-mismatch phase."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _release(live: dict) -> None:
    """End a phase: DeepEP buffers first, then the callback cycle holding the model.

    ``live`` is the same dict the phase's callbacks hold, so popping it here breaks the cycle that
    would otherwise keep a 20B model resident through the next phase. The caller must drop its own
    binding as well — a name in another scope is not reachable from here.
    """
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

    # The trainer writes the checkpoint on the save rank and every rank resumes from it, so the
    # output dir must be one shared path rather than a per-rank temp dir.
    workspace = shared_scratch_dir(f"lora_offgrpo_resume_{mode}")
    if ctx.rank == 0:
        ctx.on_teardown(lambda: shutil.rmtree(workspace, ignore_errors=True))
    train_out = os.path.join(workspace, "train_out")
    checkpoint = os.path.join(train_out, f"checkpoint-{SAVE_AT_STEP}")

    model_name = model_name_for(mode, parallelism_config_for(mode, ctx.world_size))
    ensure_model_downloaded(model_name, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = create_offline_grpo_dataset(tokenizer, NUM_TRAIN_SAMPLES, seed=SEED, num_completions=NUM_COMPLETIONS)
    probe = _probe_batch(tokenizer, ctx.device)

    # Every phase registers its trainer here, so a body that raises still releases its DeepEP buffers.
    live: dict = {}
    ctx.on_teardown(lambda: _release(live))

    # Phase 1: train 3 steps, checkpoint at step 2

    log(f"\n--- Phase 1 ({mode}): train {TOTAL_STEPS} steps, save at {SAVE_AT_STEP} ---")
    pc = parallelism_config_for(mode, ctx.world_size)
    trainer = _make_trainer(mode, pc, tokenizer, dataset, _config(train_out, TOTAL_STEPS, SAVE_AT_STEP, pc, mode))
    live["trainer"] = trainer
    expert_lora = is_expert_lora_active(trainer.accelerator.unwrap_model(trainer.model))
    checks["adapter_kind_as_declared"] = expert_lora == (mode == "expert_lora")

    # Premise for the base reading the callback takes below: at zero-init lora_B the adapted model
    # IS the frozen base, so the reading needs no disable_adapter() before a training step.
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
    del trainer  # _release cannot unbind this scope's name, and the next phase loads a second model

    if ctx.rank == 0:
        ok, detail = _checkpoint_carries_trained_adapters(checkpoint)
        log(f"checkpoint adapters: {detail}")
        checks["checkpoint_carries_trained_adapters"] = ok
    ctx.barrier()

    # Phase 2: a fresh trainer resumes the checkpoint to step 4

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
    # The resumed run's log_history opens with the checkpoint's own entries (steps 1-2, restored from
    # trainer_state.json) and appends steps 3-4, so it holds TOTAL_STEPS rows.
    checks["resumed_losses_finite"] = len(resumed_losses) == TOTAL_STEPS and all(
        math.isfinite(loss) for loss in resumed_losses
    )
    if checks["phase1_ran_all_steps"] and checks["resumed_losses_finite"]:
        checks["trainer_state_history_restored"] = resumed_losses[:SAVE_AT_STEP] == phase1_losses[:SAVE_AT_STEP]
        phase1_tail, resumed_tail = phase1_losses[SAVE_AT_STEP:], resumed_losses[SAVE_AT_STEP:]
        deltas = [abs(a - b) for a, b in zip(phase1_tail, resumed_tail, strict=True)]
        metrics["restored_step_loss_delta"] = deltas[0]
        metrics["resume_loss_max_delta"] = max(deltas)
        log(
            f"steps {SAVE_AT_STEP + 1}-{TOTAL_STEPS}: phase1={[f'{loss:.5f}' for loss in phase1_tail]} "
            f"resumed={[f'{loss:.5f}' for loss in resumed_tail]} |deltas|={[f'{d:.5f}' for d in deltas]}"
        )
        checks["restored_weights_reproduce_the_loss"] = deltas[0] < RESTORED_STEP_LOSS_TOL
        checks["resumed_loss_tracks_phase1"] = max(deltas) < LOSS_TOL

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
        log(f"probe log-probs: |reference - base|={ref_delta:.5f}, |policy - base|={policy_delta:.5f}")
        checks["reference_logps_are_the_frozen_base"] = ref_delta < REF_LOGP_TOL
        checks["adapters_move_the_policy"] = policy_delta > ADAPTER_EFFECT_MIN
    _release(live)
    del trainer

    if ctx.rank == 0:
        ok, detail = _adapter_files_identical(checkpoint, verify_dir)
        log(f"restored adapters vs checkpoint file: {detail}")
        checks["restored_adapters_match_checkpoint_file"] = ok
    ctx.barrier()

    # Phase 3 (expert_lora): the ep_size=2 checkpoint resumed at ep_size=1

    if mode == "expert_lora":
        log(f"\n--- Phase 3 ({mode}): resume the ep_size=2 checkpoint at ep_size=1 ---")
        ep1_dir = os.path.join(workspace, "restored_adapters_ep1")
        pc = ParallelismConfig(ep_size=1)
        tap = _WarningTap()
        optimizer_store_mod.logger.addHandler(tap)
        live.clear()
        try:
            trainer = _make_trainer(mode, pc, tokenizer, dataset, _config(train_out, TOTAL_STEPS, None, pc, mode))
            live["trainer"] = trainer
            trainer.add_callback(_ResumeCapture(live, ep1_dir))
            trainer.train(resume_from_checkpoint=checkpoint)
        finally:
            optimizer_store_mod.logger.removeHandler(tap)
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
            # Per-rank, unlike the warning below: the warm restart must have happened on every rank,
            # not merely been announced on the one that logs.
            ok, detail = _optimizer_state_is_warm_restarted(restored["optimizer"])
            log(f"ep1 optimizer: {detail}")
            checks["ep1_optimizer_warm_restarted"] = ok
        # The warm restart is what the fingerprint promises when the topology changes; it logs on the
        # main process only, so the other rank asserts the run continued rather than the message.
        if ctx.rank == 0:
            warned = any("fingerprint mismatch" in msg and "ep_size" in msg for msg in tap.messages)
            if not warned:
                log(f"captured optimizer-store warnings: {tap.messages}")
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
