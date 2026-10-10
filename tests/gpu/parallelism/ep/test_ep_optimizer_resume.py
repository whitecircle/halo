#!/usr/bin/env python
"""Optimizer-state continuity and checkpoint master precision under EP, CP, TP and FSDP2 (2 GPUs).

The checkpoint loader restores per-rank optimizer shards for every sharded run (FSDP2, EP, CP, TP)
whenever the topology fingerprint in ``optimizer_meta.pt`` matches; a mismatch warm-restarts loudly. This test pins
the whole contract on a tiny random-init model through the real trainer save/resume flow:

``--mode ep`` (tiny Qwen3 MoE, ep_size=2):
  1. Continuous reference: train 6 steps uninterrupted, record per-step losses.
  2. Train 3 steps, checkpoint at step 3; snapshot this rank's optimizer state (the same
     ``get_optimizer_state_dict`` view the shard files hold) and verify the checkpoint carries
     per-rank shards + a fingerprint meta + real model weights (the source layout, trained values).
  3. Resume to step 6: at ``on_train_begin`` (post-restore, pre-step) the optimizer state and every
     trainable weight must equal the save-time snapshot EXACTLY and the LR scheduler must be at
     step 3; the resumed steps 4-6 must then reproduce the continuous run, losses and final weights
     bit for bit. Nothing is reset between the phases: AdamWBF16 keys its stochastic rounding by the
     parameter's step and name, so a resumed step rounds as the uninterrupted one did, and every
     DeepEP buffer is built in deterministic mode
     (:func:`~tests.common.distributed.pin_deterministic_ep_dispatch`), without which the atomic
     receive order alone changes how each expert weight gradient is summed.
  4. Mismatch path: resume the ep_size=2 checkpoint at ep_size=1 → the fingerprint warm-restart
     warning fires naming ``ep_size``, the optimizer starts empty, training proceeds.

``--mode ep1`` (tiny Qwen3 MoE, ep_size=1): phases 1-3 on the DEFAULT MoE shape at ep_size=1 — EP
wrappers with no expert distribution, so ``fsdp_shard_ep1_experts`` leaves the experts to FSDP2 and
their moments are DTensor shards rather than the ``ep`` row's plain FSDP-ignored per-rank tensors.
Different shard layout and a different save gather (``full_tensor()``, not an EP all-gather), same
contract. No mismatch phase: save and resume run the same topology by construction.

``--mode ep_cp`` overlays EP and CP on the same two ranks. ``--mode cp`` uses dense Qwen3 with
cp_size=2, ``--mode tp`` uses dense Qwen3 with native TP2, and ``--mode fsdp`` uses FSDP2 alone.
Each runs phases 1-3.

``--fp32-masters`` enables router, expert and non-expert masters together. The ``ep1`` row needs
``--unsharded-ep1-experts``: managed EP1 experts cannot combine with ``fp32_non_ep_params``.
``--eager-loading`` selects the non-lazy EP loader. Each master row checks live dtypes and that
trained values outside BF16's representable set survive resume bit for bit.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_ep_optimizer_resume.py --mode ep
"""

import argparse
import logging
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from trl import SFTConfig

import src.distributed.checkpoint.optimizer as optimizer_store_mod
from src.checkpoint.format import load_full_state_dict
from src.distributed.checkpoint.fingerprint import OptimizerStateFingerprint
from src.distributed.expert_parallel.base_layer import find_ep_layers
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import pin_deterministic_ep_dispatch, world_all
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.tiny_models import build_tied_qwen3_checkpoint
from tests.common.tolerances import TOL
from tests.common.utils import (
    cleanup_memory,
    local_optimizer_state,
    log,
    max_or_nan,
    optimizer_state_matches,
    relative_l2,
    snapshot_trainable,
    step_losses,
)

# Modes whose tiny model is MoE — the only ones with expert weights to gather on save.
MOE_MODES = ("ep", "ep1", "ep_cp")
# The parameter groups each mode's model holds, every one of which keeps an fp32 master under
# --fp32-masters (router, expert and non-expert flags together). An ep1 row takes it only with
# --unsharded-ep1-experts: FSDP-managed ep1 experts and fp32_non_ep_params refuse each other.
FP32_MASTER_GROUPS = {
    "ep": {"expert", "router", "non_expert"},
    "ep1": {"expert", "router", "non_expert"},
    "ep_cp": {"expert", "router", "non_expert"},
    "cp": {"non_expert"},
    "tp": {"non_expert"},
    "fsdp": {"non_expert"},
}

SEED = 42
TOTAL_STEPS = 6
SAVE_AT_STEP = 3
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 256
# Resumed steps 4-6 vs the continuous run: identical batches, restored weights and moments, and the
# same rounding noise, so every row measured bit-exact (|delta| 0.0); a warm restart shifts the
# trajectory by the full Adam-moment reset, and a replayed rounding stream by ~1e-3.
LOSS_TOL = TOL.replayed_resume_loss_abs


def optimizer_resume_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["ep", "ep1", "ep_cp", "cp", "tp", "fsdp"], default="ep")
    parser.add_argument("--fp32-masters", action="store_true", help="enable all three configured master flags")
    parser.add_argument("--eager-loading", action="store_true", help="load EP weights through from_pretrained")
    parser.add_argument(
        "--unsharded-ep1-experts", action="store_true", help="keep EP1 experts outside FSDP2's shard groups"
    )
    return parser


def _parallelism_config(
    mode: str,
    world_size: int,
    *,
    fp32_masters: bool = False,
    eager_loading: bool = False,
    unsharded_ep1_experts: bool = False,
) -> ParallelismConfig:
    if unsharded_ep1_experts and mode != "ep1":
        raise ValueError("--unsharded-ep1-experts applies only to --mode ep1")
    return ParallelismConfig(
        ep_size=world_size if mode in ("ep", "ep_cp") else 1,
        cp_size=world_size if mode in ("cp", "ep_cp") else 1,
        tp_size=world_size if mode == "tp" else 1,
        ep_fp32_router=fp32_masters,
        ep_fp32_experts=fp32_masters,
        fp32_non_ep_params=fp32_masters,
        ep_lazy_loading=not eager_loading,
        fsdp_shard_ep1_experts=not unsharded_ep1_experts,
    )


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
        logging_nan_inf_filter=False,
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        seed=SEED,
    )


def _make_trainer(model_path, pc, tokenizer, train_dataset, config, *, preserve_checkpoint_precision=False):
    # CP rejects sdpa (Ulysses needs a Flash kernel), so let CP auto-detect (FA4 on Blackwell) and
    # keep the cheaper sdpa path for the other modes, which do not constrain the kernel.
    model, _ = load_distributed_model(
        model_name_or_path=model_path,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=None if pc.is_cp_mode else "sdpa",
        use_liger_kernel=False,
        preserve_checkpoint_precision=preserve_checkpoint_precision,
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
            "weights": _trainable_weights(trainer.model),
        }
        return control


def _trainable_weights(model) -> dict[str, torch.Tensor]:
    """Every trainable parameter, whole and at its live dtype. Collective under FSDP2."""
    reshard_fsdp2_modules(model)
    return snapshot_trainable(model)


def _configured_master_checks(model, mode: str) -> dict[str, bool]:
    """Live parameter identities separate routers, experts and the remaining parameters; each group
    :data:`FP32_MASTER_GROUPS` names for ``mode`` must hold fp32 masters carrying more than bf16 precision."""
    reshard_fsdp2_modules(model)
    layers = list(find_ep_layers(model))
    experts = {id(parameter) for _, layer in layers for _, parameter in layer.expert_named_params()}
    routers = {
        id(parameter)
        for _, layer in layers
        if (router := layer._live_router_module()) is not None
        for parameter in router.parameters()
    }
    groups = {"expert": {}, "router": {}, "non_expert": {}}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = "expert" if id(parameter) in experts else "router" if id(parameter) in routers else "non_expert"
            groups[group][name] = parameter
    expected = FP32_MASTER_GROUPS[mode]
    checks = {
        "master_parameter_groups_present": {group for group, parameters in groups.items() if parameters} == expected
    }
    for group in expected:
        parameters = groups[group].values()
        checks[f"{group}_configured_master_dtype"] = all(parameter.dtype == torch.float32 for parameter in parameters)
        # A BF16-exact source would pass a loader that discards its stored FP32 mantissa.
        checks[f"{group}_masters_exceed_bf16_precision"] = any(
            not torch.equal(local, local.to(torch.bfloat16).float())
            for parameter in parameters
            for local in [parameter.detach().to_local() if isinstance(parameter, DTensor) else parameter.detach()]
        )
    return checks


def _unequal(expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]) -> list[str]:
    """Keys whose tensors differ in value or dtype, or are missing from ``actual``."""
    return sorted(
        key
        for key, value in expected.items()
        if key not in actual or actual[key].dtype != value.dtype or not torch.equal(actual[key], value)
    )


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


@gpu_test_main(min_world_size=2, prefix="ep_optim_resume")
def run(ctx):
    args = optimizer_resume_parser().parse_args()
    mode = args.mode
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
        release_tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B, trust_remote_code=True)
        build_tied_qwen3_checkpoint(tiny_dir, release_tokenizer, moe=mode in MOE_MODES, seed=SEED)
    ctx.barrier()

    pc = _parallelism_config(
        mode,
        ctx.world_size,
        fp32_masters=args.fp32_masters,
        eager_loading=args.eager_loading,
        unsharded_ep1_experts=args.unsharded_ep1_experts,
    )
    if pc.ep_size > 1:
        pin_deterministic_ep_dispatch()
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
    continuous_final = _trainable_weights(trainer.model)
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
    saved_weights = _trainable_weights(trainer.model)
    if args.fp32_masters:
        checks |= {
            f"saved_{name}": world_all(ok, device)
            for name, ok in _configured_master_checks(trainer.model, mode).items()
        }
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
    checks["checkpoint_files_ok"] = world_all(files_ok, device)
    checks["checkpoint_weights_ok"] = world_all(weights_ok, device)
    del trainer
    cleanup_memory()
    ctx.barrier()

    # ── Phase 3: resume → optimizer state restored EXACTLY, losses track the reference ─
    log(f"\n--- Phase 3 ({mode}): resume from checkpoint-{SAVE_AT_STEP} ---")
    trainer = _make_trainer(
        ckpt_dir,
        pc,
        tokenizer,
        train_dataset,
        _sft_config(train_out, TOTAL_STEPS, None),
        preserve_checkpoint_precision=True,
    )
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
        checks["optimizer_state_restored_exactly"] = world_all(equal, device)
        checks["scheduler_restored"] = capture["sched_last_epoch"] == SAVE_AT_STEP
        unrestored = _unequal(saved_weights, capture["weights"])
        if unrestored:
            log(f"{len(unrestored)} trainable weights differ after the restore, e.g. {unrestored[:3]}")
        checks["weights_restored_exactly"] = world_all(bool(saved_weights) and not unrestored, device)
    resumed_tail = resumed_losses[-(TOTAL_STEPS - SAVE_AT_STEP) :]
    continuous_tail = continuous_losses[SAVE_AT_STEP:]
    checks["resumed_ran_remaining_steps"] = len(resumed_tail) == TOTAL_STEPS - SAVE_AT_STEP
    if checks["resumed_ran_remaining_steps"] and checks["continuous_ran_all_steps"]:
        max_delta = max_or_nan(abs(a - b) for a, b in zip(continuous_tail, resumed_tail, strict=True))
        metrics["resume_loss_max_delta"] = max_delta
        log(
            f"continuous tail: {[f'{loss:.4f}' for loss in continuous_tail]}  "
            f"resumed tail: {[f'{loss:.4f}' for loss in resumed_tail]}  max |delta| = {max_delta:.5f}"
        )
        checks["resumed_losses_match_continuous"] = max_delta < LOSS_TOL
    resumed_final = _trainable_weights(trainer.model)
    drifted = _unequal(continuous_final, resumed_final)
    metrics["final_weights_relative_l2"] = relative_l2(resumed_final, continuous_final)
    log(
        f"final weights: {len(continuous_final) - len(drifted)}/{len(continuous_final)} bit-identical to the "
        f"continuous run, relative L2 {metrics['final_weights_relative_l2']:.3e}"
    )
    checks["final_weights_match_continuous"] = world_all(bool(continuous_final) and not drifted, device)
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
                ckpt_dir,
                pc_ep1,
                tokenizer,
                train_dataset,
                _sft_config(train_out, SAVE_AT_STEP + 1, None),
                preserve_checkpoint_precision=True,
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
            checks["mismatch_warm_restarted"] = world_all(not carried_moments and not carried_steps, device)
        mismatch_warned = any("fingerprint mismatch" in w and "ep_size" in w for w in warnings)
        if ctx.rank == 0 and not mismatch_warned:
            log(f"captured warnings: {warnings}")
        # The warning logs on the main process only; every rank must have taken the warm restart.
        checks["mismatch_warning_fired"] = world_all(mismatch_warned or ctx.rank != 0, device)
        mismatch_losses = step_losses(trainer)
        checks["mismatch_run_proceeded"] = trainer.state.global_step == SAVE_AT_STEP + 1 and all(
            torch.isfinite(torch.tensor(mismatch_losses)).tolist()
        )
        del trainer
        cleanup_memory()

    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
