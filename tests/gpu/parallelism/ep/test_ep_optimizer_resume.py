#!/usr/bin/env python
"""Optimizer-state continuity on resume for EP and CP runs (2 GPUs).

The checkpoint loader restores per-rank optimizer shards for EP/CP whenever the topology
fingerprint in ``optimizer_meta.pt`` matches; a mismatch warm-restarts loudly. This test pins
the whole contract on a tiny random-init model through the real trainer save/resume flow:

``--mode ep`` (tiny Qwen3 MoE, ep_size=2):
  1. Continuous reference: train 6 steps uninterrupted, record per-step losses.
  2. Train 3 steps, checkpoint at step 3; snapshot this rank's optimizer state (the same
     ``get_optimizer_state_dict`` view the shard files hold) and verify the checkpoint carries
     per-rank shards + a fingerprint meta + real model weights (the source layout, trained values).
  3. Resume to step 6: at ``on_train_begin`` (post-restore, pre-step) the optimizer state must
     equal the snapshot EXACTLY (bf16 moments bit-identical; SR only perturbs later steps), the
     LR scheduler must be at step 3, and the resumed loss trajectory (steps 4-6) must track the
     continuous run closely (SR noise makes bit-exactness across the boundary unlikely).
  4. Mismatch path: resume the ep_size=2 checkpoint at ep_size=1 → the fingerprint warm-restart
     warning fires naming ``ep_size``, the optimizer starts empty, training proceeds.

``--mode ep1`` (tiny Qwen3 MoE, ep_size=1): phases 1-3 on the DEFAULT MoE shape at ep_size=1 — EP
wrappers with no expert distribution, so ``fsdp_shard_ep1_experts`` leaves the experts to FSDP2 and
their moments are DTensor shards rather than the ``ep`` row's plain FSDP-ignored per-rank tensors.
Different shard layout and a different save gather (``full_tensor()``, not an EP all-gather), same
contract. No mismatch phase: save and resume run the same topology by construction.

``--mode cp`` (tiny dense Qwen3, cp_size=2): phases 1-3 (no mismatch phase).

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_ep_optimizer_resume.py --mode ep
"""

import argparse
import logging
import os
import random

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.trainer_callback import TrainerCallback
from trl import SFTConfig

import src.distributed.checkpoint.optimizer as optimizer_store_mod
import src.optimizers.adamw_bf16 as adamw_bf16_mod
from src.checkpoint.format import load_full_state_dict
from src.distributed.checkpoint.fingerprint import OptimizerStateFingerprint
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.utils import cleanup_memory, local_optimizer_state, log, optimizer_state_matches, step_losses

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["ep", "ep1", "cp"], default="ep")
ARGS, _ = parser.parse_known_args()

# Modes whose tiny model is MoE — the only ones with expert weights to gather on save.
MOE_MODES = ("ep", "ep1")

SEED = 42
TOTAL_STEPS = 6
SAVE_AT_STEP = 3
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 256
# Resumed steps 4-6 vs the continuous run: identical batches and restored moments, differing only
# by stochastic-rounding noise (the SR stream restarts on resume) — a warm restart instead shifts
# the trajectory by the full Adam-moment reset, far above this tolerance.
LOSS_TOL = 0.05

_TINY_COMMON = {
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "max_position_embeddings": 4096,
    "tie_word_embeddings": True,
}


def _build_tiny_checkpoint(mode: str, target_dir: str) -> None:
    """Rank 0: random-init tiny model + Qwen tokenizer saved as a loadable HF checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B, trust_remote_code=True)
    torch.manual_seed(SEED)
    if mode in MOE_MODES:
        config = Qwen3MoeConfig(
            **_TINY_COMMON,
            vocab_size=len(tokenizer),
            moe_intermediate_size=128,
            num_experts=8,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            mlp_only_layers=[],
        )
        model = Qwen3MoeForCausalLM(config)
    else:
        config = Qwen3Config(**_TINY_COMMON, vocab_size=len(tokenizer))
        model = Qwen3ForCausalLM(config)
    model.to(torch.bfloat16).save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)


def _parallelism_config(mode: str, world_size: int) -> ParallelismConfig:
    if mode == "ep":
        return ParallelismConfig(ep_size=world_size)
    if mode == "ep1":
        # Defaults are the point: use_grouped_gemm applies the EP wrappers with no expert
        # distribution, and fsdp_shard_ep1_experts then hands the replicated experts to FSDP2.
        return ParallelismConfig(ep_size=1)
    return ParallelismConfig(cp_size=world_size)


def _sft_config(output_dir: str, max_steps: int, save_at: int | None) -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",  # step-count-invariant: phases with different max_steps stay comparable
        bf16=True,
        logging_steps=1,
        save_strategy="steps" if save_at else "no",
        save_steps=save_at or 0,
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        seed=SEED,
    )


def _make_trainer(model_path, pc, tokenizer, train_dataset, config):
    # Reset the rank-synchronized SR RNG so every phase consumes the same stochastic-rounding
    # stream from step 1 — the continuous and to-be-resumed runs then produce bit-comparable
    # optimizer states at the save step.
    adamw_bf16_mod._SR_RNG = random.Random(0xB165EED)
    # CP rejects sdpa (Ulysses needs a Flash kernel), so let CP auto-detect (FA4 on Blackwell) and
    # keep the cheaper sdpa path for the EP mode, which does not constrain the kernel.
    model, _ = load_distributed_model(
        model_name_or_path=model_path,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=None if pc.is_cp_mode else "sdpa",
        use_liger_kernel=False,
    )
    return DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        parallelism_config=pc,
    )


def _snapshot_values(snapshot: dict) -> list[tuple[str, object]]:
    """Every state value in a snapshot, flattened across params — for whole-snapshot scans."""
    return [(f"{fqn}.{key}", value) for fqn, entry in snapshot["state"].items() for key, value in entry.items()]


class _ResumeCapture(TrainerCallback):
    """Snapshot restored state at on_train_begin: after the resume hooks, before the first step."""

    def __init__(self, trainer_ref: dict):
        self.trainer_ref = trainer_ref

    def on_train_begin(self, args, state, control, **kwargs):
        trainer = self.trainer_ref["trainer"]
        self.trainer_ref["capture"] = {
            "snapshot": local_optimizer_state(trainer.model, trainer.optimizer),
            "optimizer_state_len": len(trainer.optimizer.state),
            "sched_last_epoch": int(trainer.lr_scheduler.last_epoch),
        }
        return control


def _all_ranks_true(local: bool, device) -> bool:
    t = torch.tensor([1 if local else 0], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())


def _verify_checkpoint_files(ckpt_dir: str, world_size: int, pc: ParallelismConfig, optimizer) -> tuple[bool, str]:
    """Rank 0: per-rank shards present, no stale optimizer.pt, meta fingerprint matches the run."""
    files = set(os.listdir(ckpt_dir))
    problems = []
    for r in range(world_size):
        if f"optimizer_shard_{r:05d}.pt" not in files:
            problems.append(f"missing optimizer_shard_{r:05d}.pt")
    if "optimizer.pt" in files:
        problems.append("stale single-rank optimizer.pt present")
    if "optimizer_meta.pt" not in files:
        problems.append("missing optimizer_meta.pt")
    else:
        meta = torch.load(os.path.join(ckpt_dir, "optimizer_meta.pt"), map_location="cpu", weights_only=False)
        saved_fp = OptimizerStateFingerprint.from_dict(meta.get("fingerprint"))
        if saved_fp is None:
            problems.append("optimizer_meta.pt carries no fingerprint")
        else:
            expected = OptimizerStateFingerprint.capture(pc, optimizer, world_size)
            mismatched = expected.mismatches(saved_fp)
            if mismatched:
                problems.append(f"fingerprint mismatch vs live run: {mismatched}")
    if "scheduler.pt" not in files:
        problems.append("missing scheduler.pt")
    return not problems, "; ".join(problems) or "ok"


def _verify_checkpoint_weights(ckpt_dir: str, source_dir: str, mode: str) -> tuple[bool, str]:
    """Rank 0: the saved weights are the whole model, at full shapes, carrying the TRAINED values.

    The source checkpoint every phase loads from is the reference layout, so a key it declares that
    the save did not write back — or wrote at a sharded shape — is a dropped/half-gathered tensor.
    That is the failure this shape invites: at ``ep_size=1`` the experts are FSDP2 DTensors, so the
    save must ``full_tensor()`` them rather than write a rank's slice (or nothing at all), and a
    reload would then silently re-init them.

    Extra keys are reported, not failed: the discriminating half is that every declared key came
    back whole, and the one extra-key spelling that IS corruption (a doubled ``experts.experts.``
    prefix from a mis-prefixed gather) is failed explicitly.
    """
    saved = load_full_state_dict(ckpt_dir)
    source = load_full_state_dict(source_dir)
    if saved is None or source is None:
        return False, f"no readable checkpoint weights (saved={saved is not None}, source={source is not None})"

    problems = []
    missing = sorted(set(source) - set(saved))
    if missing:
        problems.append(f"{len(missing)} key(s) missing from the save, e.g. {missing[:3]}")
    mis_shaped = sorted(k for k in set(source) & set(saved) if saved[k].shape != source[k].shape)
    if mis_shaped:
        problems.append(f"{len(mis_shaped)} key(s) saved at the wrong shape, e.g. {mis_shaped[:3]}")
    doubled = sorted(k for k in saved if ".experts.experts." in k)
    if doubled:
        problems.append(f"{len(doubled)} doubled-prefix expert key(s), e.g. {doubled[:3]}")
    non_finite = sorted(k for k, v in saved.items() if v.is_floating_point() and not torch.isfinite(v).all())
    if non_finite:
        problems.append(f"{len(non_finite)} key(s) hold non-finite values, e.g. {non_finite[:3]}")

    # Anti-vacuity: the save must hold the TRAINED weights. Identical-to-source everywhere means the
    # writer emitted the initial values (or the reload source, not the live model). Per-key rather
    # than all-keys: with top-2-of-8 routing over 3 steps an individual expert can legitimately be
    # untouched, so the premise is that at least one expert tensor moved.
    changed = [k for k in set(source) & set(saved) if not torch.equal(saved[k].float(), source[k].float())]
    if not changed:
        problems.append("every saved tensor equals the untrained source — the save wrote initial weights")
    elif mode in MOE_MODES and not any(".experts." in k for k in changed):
        problems.append("no expert tensor moved off the untrained source — the experts were not gathered live")

    extra = sorted(set(saved) - set(source))
    detail = f"{len(saved)} keys, {len(changed)} changed vs source"
    if extra:
        detail += f", {len(extra)} extra (not failed): {extra[:3]}"
    return not problems, "; ".join(problems) or detail


@gpu_test_main(min_world_size=2, prefix=f"ep_optim_resume_{ARGS.mode}")
def run(ctx):
    mode = ARGS.mode
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    device = ctx.device

    # Shared workspace: rank 0's output dir (single-node shared FS).
    dirs = [ctx.output_dir]
    dist.broadcast_object_list(dirs, src=0)
    shared_dir = dirs[0]
    tiny_dir = os.path.join(shared_dir, "tiny_model")
    train_out = os.path.join(shared_dir, "train_out")
    ckpt_dir = os.path.join(train_out, f"checkpoint-{SAVE_AT_STEP}")

    if ctx.rank == 0:
        _build_tiny_checkpoint(mode, tiny_dir)
    ctx.barrier()

    pc = _parallelism_config(mode, ctx.world_size)
    tokenizer = AutoTokenizer.from_pretrained(tiny_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = create_sft_dataset(64, tokenizer, seed=SEED)

    # ── Phase 1: continuous 6-step reference ─────────────────────────────────
    log(f"\n--- Phase 1 ({mode}): continuous {TOTAL_STEPS}-step reference ---")
    trainer = _make_trainer(tiny_dir, pc, tokenizer, train_dataset, _sft_config(train_out, TOTAL_STEPS, None))
    ctx.on_teardown(trainer.cleanup_ep)
    trainer.train()
    continuous_losses = step_losses(trainer)
    log(f"continuous losses: {[f'{loss:.4f}' for loss in continuous_losses]}")
    checks["continuous_ran_all_steps"] = len(continuous_losses) == TOTAL_STEPS
    del trainer
    cleanup_memory()
    ctx.barrier()

    # ── Phase 2: train to the save point, snapshot the state the shards hold ─
    log(f"\n--- Phase 2 ({mode}): train {SAVE_AT_STEP} steps + checkpoint ---")
    trainer = _make_trainer(tiny_dir, pc, tokenizer, train_dataset, _sft_config(train_out, SAVE_AT_STEP, SAVE_AT_STEP))
    ctx.on_teardown(trainer.cleanup_ep)
    trainer.train()
    saved_losses = step_losses(trainer)
    snapshot_ref = local_optimizer_state(trainer.model, trainer.optimizer)
    checks["saved_state_nonempty"] = any(
        torch.is_tensor(v) and v.dtype.is_floating_point and (v != 0).any() for _, v in _snapshot_values(snapshot_ref)
    )
    ctx.barrier()
    if ctx.rank == 0:
        files_ok, detail = _verify_checkpoint_files(ckpt_dir, ctx.world_size, pc, trainer.optimizer)
        log(f"checkpoint files: {detail}")
        weights_ok, weights_detail = _verify_checkpoint_weights(ckpt_dir, tiny_dir, mode)
        log(f"checkpoint weights: {weights_detail}")
    else:
        files_ok = weights_ok = True
    checks["checkpoint_files_ok"] = _all_ranks_true(files_ok, device)
    checks["checkpoint_weights_ok"] = _all_ranks_true(weights_ok, device)
    del trainer
    cleanup_memory()
    ctx.barrier()

    # ── Phase 3: resume → optimizer state restored EXACTLY, losses track the reference ─
    log(f"\n--- Phase 3 ({mode}): resume from checkpoint-{SAVE_AT_STEP} ---")
    trainer = _make_trainer(ckpt_dir, pc, tokenizer, train_dataset, _sft_config(train_out, TOTAL_STEPS, None))
    ctx.on_teardown(trainer.cleanup_ep)
    trainer_ref: dict = {"trainer": trainer, "capture": None}
    trainer.add_callback(_ResumeCapture(trainer_ref))
    trainer.train(resume_from_checkpoint=ckpt_dir)
    resumed_losses = step_losses(trainer)
    capture = trainer_ref["capture"]
    checks["resume_capture_fired"] = capture is not None
    if capture is not None:
        equal, why = optimizer_state_matches(snapshot_ref, capture["snapshot"])
        if not equal:
            log(f"OPTIMIZER STATE MISMATCH after restore: {why}")
        checks["optimizer_state_restored_exactly"] = _all_ranks_true(equal, device)
        checks["scheduler_restored"] = capture["sched_last_epoch"] == SAVE_AT_STEP
    resumed_tail = resumed_losses[-(TOTAL_STEPS - SAVE_AT_STEP) :]
    continuous_tail = continuous_losses[SAVE_AT_STEP:]
    checks["resumed_ran_remaining_steps"] = len(resumed_tail) == TOTAL_STEPS - SAVE_AT_STEP
    if checks["resumed_ran_remaining_steps"] and checks["continuous_ran_all_steps"]:
        max_delta = max(abs(a - b) for a, b in zip(continuous_tail, resumed_tail, strict=True))
        metrics["resume_loss_max_delta"] = max_delta
        log(
            f"continuous tail: {[f'{loss:.4f}' for loss in continuous_tail]}  "
            f"resumed tail: {[f'{loss:.4f}' for loss in resumed_tail]}  max |delta| = {max_delta:.5f}"
        )
        checks["resumed_losses_track_continuous"] = max_delta < LOSS_TOL
    log(f"phase-2 losses: {[f'{loss:.4f}' for loss in saved_losses]}")
    del trainer
    cleanup_memory()
    ctx.barrier()

    # ── Phase 4 (ep only): topology mismatch → loud warm restart, run proceeds ─
    if mode == "ep":
        log("\n--- Phase 4 (ep): resume the ep_size=2 checkpoint at ep_size=1 (mismatch) ---")
        warnings: list[str] = []

        class _Grab(logging.Handler):
            def emit(self, record):
                warnings.append(record.getMessage())

        handler = _Grab(level=logging.WARNING)
        optimizer_store_mod.logger.addHandler(handler)
        try:
            pc_ep1 = ParallelismConfig(ep_size=1, use_grouped_gemm=True)
            trainer = _make_trainer(
                ckpt_dir, pc_ep1, tokenizer, train_dataset, _sft_config(train_out, SAVE_AT_STEP + 1, None)
            )
            ctx.on_teardown(trainer.cleanup_ep)
            trainer_ref = {"trainer": trainer, "capture": None}
            trainer.add_callback(_ResumeCapture(trainer_ref))
            trainer.train(resume_from_checkpoint=ckpt_dir)
        finally:
            optimizer_store_mod.logger.removeHandler(handler)
        capture = trainer_ref["capture"]
        checks["mismatch_capture_fired"] = capture is not None
        if capture is not None:
            # Warm restart: no TRAINED moments may carry over — every state entry must still be at
            # its freshly-initialized value. Asserting "state is empty" would be wrong on two counts:
            # ``optimizer.state`` is a defaultdict (a bare read inserts an empty entry) and
            # ``get_optimizer_state_dict`` itself materializes empty state via a zero-grad init step.
            # That init step is also why ``step`` is excluded from the zero test and asserted against
            # the saved step instead; a wrongly loaded shard shows non-zero moments AND the saved step.
            carried = _snapshot_values(capture["snapshot"])
            carried_moments = sorted(
                key for key, value in carried if torch.is_tensor(value) and bool(value.abs().sum() > 0)
            )
            carried_steps = sorted(
                f"{key}={value}"
                for key, value in carried
                if key.endswith(".step") and not torch.is_tensor(value) and value >= SAVE_AT_STEP
            )
            if (carried_moments or carried_steps) and ctx.rank == 0:
                log(f"mismatch resume carried: moments={carried_moments[:5]} steps={carried_steps[:5]}")
            checks["mismatch_warm_restarted"] = _all_ranks_true(not carried_moments and not carried_steps, device)
        mismatch_warned = any("fingerprint mismatch" in w and "ep_size" in w for w in warnings)
        if ctx.rank == 0 and not mismatch_warned:
            log(f"captured warnings: {warnings}")
        # The warning logs on the main process only; every rank must have taken the warm restart.
        checks["mismatch_warning_fired"] = _all_ranks_true(mismatch_warned or ctx.rank != 0, device)
        mismatch_losses = step_losses(trainer)
        checks["mismatch_run_proceeded"] = trainer.state.global_step == SAVE_AT_STEP + 1 and all(
            torch.isfinite(torch.tensor(mismatch_losses)).tolist()
        )
        del trainer
        cleanup_memory()

    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
