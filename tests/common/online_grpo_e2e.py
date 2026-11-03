"""Shared body of the online-GRPO and SDPG e2e tests against a live vLLM server, one row per (trainer, mode, resume).
Assertions are on the served policy: a sync that gathered the wrong shard still returns a finite loss from
``train()``."""

import math
import os
from dataclasses import dataclass
from types import SimpleNamespace
from urllib.parse import urlparse

import torch
from transformers import AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from trl import GRPOConfig

from halo.checkpoint.parallel.resume_source import resolve_resume_weights_source
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.model_loading.parallel.frozen_models import load_frozen_auxiliary_model
from halo.trainers.distillation.sdpg import DistributedSDPGTrainer
from halo.trainers.grpo.online import DistributedOnlineGRPOTrainer
from tests.common.adapters import assert_only_adapters_trainable, snapshot_adapters, unwrap
from tests.common.checkpoint_io import RestorePointSnapshot
from tests.common.distributed import ensure_model_downloaded, shared_output_dir
from tests.common.ep_reference import ep_layers
from tests.common.memory import cleanup_memory
from tests.common.on_policy_e2e import (
    RESUME_MAX_STEPS,
    RESUME_SAVE_STEP,
    PolicyLayout,
    adapter_file_agreement,
    expert_lora_under_etp_refusal,
    frozen_base_weights,
    load_policy,
    local_view,
    logged_lrs,
    perturbation_round,
    probe_top_logprobs,
    qa_dataset,
    record_adapter_training,
    record_dp_layout,
    record_optimizer_restore,
    record_parallelism_engaged,
    record_resume_continuity,
    record_scheduler_restore,
    record_served_baseline,
    record_step_losses,
    served_policy_moved,
    sink_round,
)
from tests.common.rank_log import log
from tests.common.weight_sync import moved_parameters

# Short because every step is a full engine round-trip, and neither a landed sync nor a resume push needs convergence.
MAX_STEPS = 3
NUM_TRAIN_SAMPLES = 16
# Large, since the sync has to move the served weights measurably within a few steps.
FULL_LEARNING_RATE = 1e-4
LORA_LEARNING_RATE = 1e-3
MAX_COMPLETION_LENGTH = 64
NUM_GENERATIONS = 2
PER_DEVICE_BATCH_SIZE = 2


@dataclass(frozen=True)
class _Mode:
    """One row's shape: its parallel layout, world size and adapter (``None``, ``"lora"`` or ``"expert_lora"``).
    ``beta`` is the KL coefficient; non-zero on a full fine-tune, the row holds a frozen reference.
    ``checkpoint_optimizer`` is off for a MoE full fine-tune, where AdamWBF16 moments cost over a hundred GB of disk
    and the same in host RAM, for what the adapter rows and ``test_ep_optimizer_resume.py`` already cover."""

    family: str
    layout: PolicyLayout = PolicyLayout()
    world_size: int = 2
    adapter: str | None = None
    refuses_lora_under_tp: bool = False
    rejects_expert_lora: bool = False
    beta: float = 0.0
    checkpoint_optimizer: bool = True

    @property
    def expert_lora(self) -> bool:
        return self.adapter == "expert_lora"


MODES = {
    "full_ep2": _Mode("moe", PolicyLayout(ep_size=2), checkpoint_optimizer=False),
    "lora_ep2": _Mode("moe", PolicyLayout(ep_size=2), adapter="lora"),
    "expert_lora_ep2": _Mode("moe", PolicyLayout(ep_size=2), adapter="expert_lora"),
    "lora_etp2": _Mode("moe", PolicyLayout(expert_tp_size=2), adapter="lora", rejects_expert_lora=True),
    # ep1: the experts in the FSDP2 shard as DTensors, or kept whole and FSDP-ignored.
    "full_ep1": _Mode("moe", checkpoint_optimizer=False),
    "full_ep1_unsharded": _Mode("moe", PolicyLayout(shard_ep1_experts=False), checkpoint_optimizer=False),
    "full_ep2_tp2": _Mode("moe", PolicyLayout(ep_size=2, tp_size=2), checkpoint_optimizer=False),
    "full_ep2_dp2": _Mode("moe", PolicyLayout(ep_size=2), world_size=4, checkpoint_optimizer=False),
    "full_ep2x2dom": _Mode("moe", PolicyLayout(ep_size=2, gpus_per_node=2), world_size=4, checkpoint_optimizer=False),
    "full_ep2_etp2": _Mode("moe", PolicyLayout(ep_size=2, expert_tp_size=2), world_size=4, checkpoint_optimizer=False),
    "full_tp2": _Mode("dense", PolicyLayout(tp_size=2)),
    "lora_fsdp": _Mode("dense", adapter="lora"),
    "lora_tp2_rejected": _Mode("dense", PolicyLayout(tp_size=2), adapter="lora", refuses_lora_under_tp=True),
    "full_fsdp_kl": _Mode("dense", beta=0.01),
    "full_fsdp_zero3": _Mode("dense", PolicyLayout(reshard_after_forward=True)),
    "full_tp2_dp2": _Mode("dense", PolicyLayout(tp_size=2), world_size=4),
    # Two simulated NVLink domains of two, sharded within each and replicated across them.
    "full_hsdp": _Mode("dense", PolicyLayout(use_hsdp=True, gpus_per_node=2), world_size=4),
    "full_world1": _Mode("world1", world_size=1),
    "lora_world1": _Mode("world1", world_size=1, adapter="lora"),
}


def modes_for(family: str) -> tuple[str, ...]:
    return tuple(name for name, spec in MODES.items() if spec.family == family)


def reward_len(completions, **kwargs):
    """A reward that varies with the completion: a tied group has zero advantage, so adapter-movement and OPD checks
    measure nothing."""
    return [-abs(len(str(completion)) - 32) / 32.0 for completion in completions]


def _grpo_config(spec: _Mode, *, output_dir, server_url, group_port, max_steps, save_steps) -> GRPOConfig:
    parsed = urlparse(server_url)
    return GRPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LORA_LEARNING_RATE if spec.adapter else FULL_LEARNING_RATE,
        lr_scheduler_type="linear",
        warmup_steps=1,
        bf16=True,
        logging_steps=1,
        save_strategy="steps" if save_steps else "no",
        save_steps=save_steps or 0,
        save_only_model=not spec.checkpoint_optimizer,
        report_to="none",
        logging_nan_inf_filter=False,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=parsed.hostname or "localhost",
        vllm_server_port=parsed.port or 8000,
        vllm_server_timeout=600.0,
        vllm_group_port=group_port,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        beta=spec.beta,
        generation_kwargs={"temperature": 0.7},
        remove_unused_columns=False,
    )


def _build_trainer(
    *,
    trainer_kind: str,
    spec: _Mode,
    model_name: str,
    tokenizer,
    server_url: str,
    group_port: int,
    output_dir: str,
    max_steps: int,
    save_steps: int,
    weights_source: str | None = None,
):
    """Construct the row's trainer; returns ``(trainer, parallelism_config)``. ``weights_source`` is a hub id or a
    resume checkpoint."""
    parallelism_config = spec.layout.parallelism_config()
    model, peft_config = load_policy(weights_source or model_name, parallelism_config, spec.adapter)
    kwargs = {
        "model": model,
        "reward_funcs": [reward_len],
        "args": _grpo_config(
            spec,
            output_dir=output_dir,
            server_url=server_url,
            group_port=group_port,
            max_steps=max_steps,
            save_steps=save_steps,
        ),
        "train_dataset": qa_dataset(NUM_TRAIN_SAMPLES),
        "processing_class": tokenizer,
        "parallelism_config": parallelism_config,
        "peft_config": peft_config,
    }
    if spec.beta and not spec.adapter:
        kwargs["ref_model"] = load_frozen_auxiliary_model(
            model_name, dtype=torch.bfloat16, trust_remote_code=True, reset_sinks=False, download_tag=None
        )
    if trainer_kind == "sdpg":
        trainer = DistributedSDPGTrainer(
            **kwargs,
            sdpg_answer_field="answer",
            opd_positive_advantage_only=True,
            sdpg_loss="reverse_kl",
            sdpg_beta_base=1.0,
        )
    else:
        trainer = DistributedOnlineGRPOTrainer(**kwargs)
    return trainer, parallelism_config


def _release(trainer) -> None:
    """Free a trainer the way its dying process would. The vLLM communicator closes explicitly, since phase 2 rebinds
    its group port and engine."""
    trainer.cleanup_ep()
    if trainer.accelerator.is_main_process:
        trainer.vllm_generation.vllm_client.close_communicator()
    cleanup_memory()


class _CheckpointOnce(TrainerCallback):
    """Write only the checkpoint the resume targets; HF's default flow would save a second copy when training stops.
    Register after its callback."""

    def __init__(self, step: int):
        self._step = step

    def on_step_end(self, args, state, control, **kwargs):
        control.should_save = state.global_step == self._step
        return control


def _weight_witnesses(model, spec: _Mode) -> dict[str, torch.Tensor]:
    """One trained tensor per stream a full fine-tune's checkpoint must carry back; rank-local, and empty for an
    adapter row."""
    if spec.adapter:
        return {}
    unwrapped = unwrap(model)
    witnesses = {}
    for name, param in unwrapped.named_parameters():
        if "layers.0" in name and name.endswith(".weight") and param.dtype.is_floating_point:
            witnesses[f"dense:{name}"] = local_view(param.data)
            break
    for layer in ep_layers(unwrapped):
        for name, param in layer.expert_named_params():
            if param.dtype.is_floating_point:
                witnesses[f"expert:{name}"] = local_view(param.data)
                break
        break
    return witnesses


class _RestorePoint(RestorePointSnapshot):
    """The shared restore point plus the trained weights; read after ``train()`` they already carry the resumed step's
    update."""

    def __init__(self, event: str, trainer, spec: _Mode, *, capture_optimizer: bool, expert_lora: bool | None = None):
        super().__init__(event, trainer, capture_optimizer=capture_optimizer, expert_lora=expert_lora)
        self._spec = spec

    def extra(self) -> dict:
        return {"weights": _weight_witnesses(self.trainer.model, self._spec)}


def _perturb_and_sync(ctx, trainer, spec: _Mode, checks: dict, *, check_base_untouched: bool = True) -> None:
    perturbation_round(
        ctx,
        trainer.model,
        adapter=spec.adapter,
        push=trainer._distributed_sync_weights,
        checks=checks,
        check_base_untouched=check_base_untouched,
    )


def _sync_the_sinks(ctx, trainer, server_url: str, model_name: str, checks: dict) -> None:
    """The sink stream's own round; inert on a sink-less family."""
    sink_round(
        ctx,
        trainer.model,
        server_url=server_url,
        model_name=model_name,
        push=trainer._distributed_sync_weights,
        checks=checks,
    )


def _sdpg_checks(trainer, checks: dict) -> None:
    """The OPD term fired and contributed: ``opd_loss`` is logged whenever ``sdpg_beta_base != 0``, so only a non-zero
    value proves it."""
    opd = [float(entry["opd_loss"]) for entry in trainer.state.log_history if "opd_beta" in entry]
    checks["sdpg_opd_term_fired"] = bool(opd)
    checks["sdpg_opd_loss_finite"] = bool(opd) and all(math.isfinite(v) for v in opd)
    checks["sdpg_opd_term_contributed"] = any(v != 0.0 for v in opd)
    log(f"  OPD over {len(opd)} step(s): opd_loss={opd}")


def _kl_checks(trainer, checks: dict) -> None:
    """The KL term ran against the row's reference, held in the run dtype, and logged a finite KL on every step."""
    reference = trainer.accelerator.unwrap_model(trainer.ref_model) if trainer.ref_model is not None else None
    checks["kl_reference_held_in_the_run_dtype"] = reference is not None and {
        p.dtype for p in reference.parameters()
    } == {torch.bfloat16}
    kl = [float(entry["kl"]) for entry in trainer.state.log_history if "kl" in entry]
    checks["kl_logged_finite"] = bool(kl) and all(math.isfinite(v) for v in kl)
    checks["kl_engaged"] = any(v > 0 for v in kl)
    log(f"  KL over {len(kl)} step(s): {kl}")


def _lora_under_tp_refused(**build_kwargs) -> dict[str, bool]:
    """The TP + LoRA refusal on a real construction: ``reject_lora_under_tp`` runs after the model wrappers and client
    are built. The raise leaves no handle to the client, so the ``atexit`` hook ``init_communicator`` registers closes
    it."""
    try:
        _build_trainer(**build_kwargs)
    except ValueError as error:
        message = str(error)
        log(f"  refusal: {message.splitlines()[0]}")
        return {
            "lora_under_tp_refused": True,
            "refusal_names_tensor_parallelism": "Tensor Parallelism" in message
            and "tensor_parallel_size > 1" in message,
        }
    log("  a TP + LoRA trainer CONSTRUCTED: the rank-inconsistent adapter would train unnoticed")
    return {"lora_under_tp_refused": False, "refusal_names_tensor_parallelism": False}


def _run_resume(ctx, *, trainer_kind, spec: _Mode, model_name, tokenizer, server_url, group_port, checks) -> dict:
    """A resumed run's first rollout comes from the checkpoint's weights, the invariant TRL's ``_last_loaded_step``
    sentinel carries. Phase 1 moves the engine off the checkpoint's policy; phase 2 has one step left, so the
    sentinel's push is all that can reach the engine."""
    output_dir = shared_output_dir(ctx)
    checkpoint = os.path.join(output_dir, f"checkpoint-{RESUME_SAVE_STEP}")

    trainer, parallelism_config = _build_trainer(
        trainer_kind=trainer_kind,
        spec=spec,
        model_name=model_name,
        tokenizer=tokenizer,
        server_url=server_url,
        group_port=group_port,
        output_dir=output_dir,
        max_steps=RESUME_MAX_STEPS,
        save_steps=RESUME_SAVE_STEP,
    )
    live = {"trainer": trainer}
    ctx.on_teardown(lambda: live["trainer"] and _release(live["trainer"]))
    trainer.add_callback(_CheckpointOnce(RESUME_SAVE_STEP))
    saved = _RestorePoint("save", trainer, spec, capture_optimizer=spec.checkpoint_optimizer)
    trainer.add_callback(saved)
    record_parallelism_engaged(ctx, trainer, spec.layout, checks)

    trainer.train()
    record_step_losses(trainer, checks)
    reshard_fsdp2_modules(unwrap(trainer.model))
    record_dp_layout(ctx, trainer, spec.layout, checks)
    phase1_lrs = logged_lrs(trainer)
    checks["checkpoint_captured_at_the_save_step"] = (
        saved.captured is not None and saved.captured["global_step"] == RESUME_SAVE_STEP
    )
    ctx.barrier()

    # The last step's pre-generation sync pushed the checkpoint's policy, so no extra merge stands between this and
    # phase 2's load.
    checkpoint_policy = {}
    if ctx.rank == 0:
        checkpoint_policy = probe_top_logprobs(server_url, model_name)
        checks["checkpoint_policy_reproducible"] = checkpoint_policy == probe_top_logprobs(server_url, model_name)
        checks["checkpoint_written"] = os.path.isdir(checkpoint)
        log(f"  checkpoint policy: { {k: round(v, 4) for k, v in checkpoint_policy.items()} }")
    ctx.barrier()

    # The fold is judged by the resumed engine landing back on the checkpoint's policy.
    _perturb_and_sync(ctx, trainer, spec, checks, check_base_untouched=False)
    ctx.barrier()
    if ctx.rank == 0:
        moved_off = probe_top_logprobs(server_url, model_name)
        checks["engine_moved_off_the_checkpoint_policy"] = served_policy_moved(moved_off, checkpoint_policy)
        log(f"  engine moved off the checkpoint policy: {checks['engine_moved_off_the_checkpoint_policy']}")
    ctx.barrier()

    # The dense delta above does not cover the sink stream, and the resume has to bring both back.
    _sync_the_sinks(ctx, trainer, server_url, model_name, checks)

    _release(trainer)
    live["trainer"] = None
    # Holding the trainer would keep phase 1's model on the GPUs next to phase 2's.
    saved.trainer = None
    del trainer
    cleanup_memory()
    ctx.barrier()

    # A fresh model and trainer put ``_last_loaded_step`` back at ``-1``. Under EP/CP a full checkpoint's weights load
    # only at construction; an adapter-only one keeps the base.
    parallelism_config = spec.layout.parallelism_config()
    weights_source = resolve_resume_weights_source(
        checkpoint, SimpleNamespace(model_name_or_path=model_name), parallelism_config
    )
    checks["resume_weights_source_resolved"] = weights_source == (model_name if spec.adapter else checkpoint)
    log(f"  resume: policy weights from {weights_source}")

    # Same group port: phase 1's close freed it on both ends.
    resumed, _ = _build_trainer(
        trainer_kind=trainer_kind,
        spec=spec,
        model_name=model_name,
        tokenizer=tokenizer,
        server_url=server_url,
        group_port=group_port,
        output_dir=output_dir,
        max_steps=RESUME_MAX_STEPS,
        save_steps=0,
        weights_source=weights_source,
    )
    live["trainer"] = resumed
    restored = _RestorePoint(
        "train_begin",
        resumed,
        spec,
        capture_optimizer=spec.checkpoint_optimizer,
        expert_lora=spec.expert_lora if spec.adapter else None,
    )
    resumed.add_callback(restored)

    resumed.train(resume_from_checkpoint=checkpoint)
    reshard_fsdp2_modules(unwrap(resumed.model))
    ctx.barrier()

    if ctx.rank == 0:
        after = probe_top_logprobs(server_url, model_name)
        # Exact on every row: weights or adapters restore bit-equal, and phase 1's syncs left the base as loaded.
        checks["resume_returned_the_engine_to_the_checkpoint_policy"] = after == checkpoint_policy
        if not checks["resume_returned_the_engine_to_the_checkpoint_policy"]:
            log("  the resumed run's first rollout did NOT come from the checkpoint's weights")
        log(f"  post-resume: { {k: round(v, 4) for k, v in after.items()} }")

    capture = restored.captured
    checks["resume_ran_the_remaining_step"] = resumed.state.global_step == RESUME_MAX_STEPS
    record_resume_continuity(checks, restored=capture, phase1_lrs=phase1_lrs, phase2_lrs=logged_lrs(resumed))

    if spec.checkpoint_optimizer:
        record_scheduler_restore(checks, capture)
        record_optimizer_restore(checks, saved.captured, capture)

    # Train-begin captures, since one resumed step has already moved them. A save that never fired leaves no witnesses.
    witnesses = saved.captured["weights"] if saved.captured else {}
    if witnesses:
        # A full fine-tune's checkpoint carries the weights, so an inexact restore shows here before the engine sees
        # it.
        mismatched = [
            k for k in witnesses if not torch.equal(witnesses[k], capture["weights"].get(k, torch.tensor([])))
        ]
        checks["resume_restored_the_trained_weights"] = not mismatched
        log(f"  weight witnesses restored exactly: {len(witnesses) - len(mismatched)}/{len(witnesses)} {mismatched}")
    if spec.adapter and ctx.rank == 0:
        matching, compared, nonzero_b = adapter_file_agreement(capture["adapters"], checkpoint)
        checks["resume_adapters_match_the_checkpoint_file"] = compared > 0 and matching == compared
        checks["resume_adapters_are_nonzero"] = nonzero_b > 0
        log(f"  restored adapters vs {checkpoint}: {matching}/{compared} exact, {nonzero_b} non-zero B")
    ctx.barrier()

    if trainer_kind == "sdpg":
        _sdpg_checks(resumed, checks)
    return {
        "checks": ctx.broadcast_checks(checks),
        "metrics": {"resumed_global_step": float(resumed.state.global_step)},
    }


def run_online_grpo_e2e(
    ctx, *, trainer_kind: str, mode: str, resume: bool, server_url: str, group_port: int, model_name: str
) -> dict:
    spec = MODES[mode]
    checks: dict[str, bool] = {}
    log(f"\n  row: trainer={trainer_kind} mode={mode} resume={resume} model={model_name} server={server_url}")

    if spec.rejects_expert_lora:
        refusal = expert_lora_under_etp_refusal(spec.layout.expert_tp_size)
        checks["expert_lora_under_etp_refused"] = bool(refusal)
        checks["etp_refusal_names_expert_tensor_parallel_size"] = "expert_tensor_parallel_size > 1" in refusal

    if ctx.rank == 0:
        record_served_baseline(server_url, model_name, checks)
    ctx.barrier()

    ensure_model_downloaded(model_name, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    build_kwargs = {
        "trainer_kind": trainer_kind,
        "spec": spec,
        "model_name": model_name,
        "tokenizer": tokenizer,
        "server_url": server_url,
        "group_port": group_port,
        "output_dir": ctx.output_dir,
    }

    if spec.refuses_lora_under_tp:
        checks.update(_lora_under_tp_refused(**build_kwargs, max_steps=1, save_steps=0))
        return {"checks": ctx.broadcast_checks(checks), "metrics": {}}

    if resume:
        return _run_resume(
            ctx,
            trainer_kind=trainer_kind,
            spec=spec,
            model_name=model_name,
            tokenizer=tokenizer,
            server_url=server_url,
            group_port=group_port,
            checks=checks,
        )

    trainer, _ = _build_trainer(**build_kwargs, max_steps=MAX_STEPS, save_steps=0)
    ctx.on_teardown(lambda: _release(trainer))
    record_parallelism_engaged(ctx, trainer, spec.layout, checks)

    adapters_before, base_before = {}, {}
    if spec.adapter:
        adapters_before = snapshot_adapters(unwrap(trainer.model), expert_lora=spec.expert_lora)
        base_before = frozen_base_weights(unwrap(trainer.model))
        ok, detail = assert_only_adapters_trainable(unwrap(trainer.model))
        checks["only_adapters_trainable"] = ok
        log(f"  frozen base: {detail}")

    trainer.train()
    log("  training completed")
    losses, trained = record_step_losses(trainer, checks)
    reshard_fsdp2_modules(unwrap(trainer.model))

    if spec.adapter:
        record_adapter_training(trainer, checks, before=adapters_before, expert_lora=spec.expert_lora, trained=trained)
        # Every step's sync folded the adapters into copies it pushed; the frozen base must come out as loaded.
        moved = moved_parameters(base_before, frozen_base_weights(unwrap(trainer.model)))
        checks["training_syncs_left_the_base_weights_alone"] = bool(base_before) and not moved
        log(f"  frozen base weights the training syncs moved: {len(moved)}/{len(base_before)} {moved[:3]}")
    record_dp_layout(ctx, trainer, spec.layout, checks)

    ctx.barrier()
    before = probe_top_logprobs(server_url, model_name) if ctx.rank == 0 else {}
    if ctx.rank == 0:
        checks["served_logprobs_finite"] = all(math.isfinite(v) for v in before.values())
    _perturb_and_sync(ctx, trainer, spec, checks)
    ctx.barrier()

    if ctx.rank == 0:
        after = probe_top_logprobs(server_url, model_name)
        checks["forced_sync_moved_the_served_policy"] = served_policy_moved(after, before)
        if not checks["forced_sync_moved_the_served_policy"]:
            log("  IDENTICAL or non-finite logprobs after the perturbation: the weight sync did not land")
        # A failed update leaves the engine partially written, so check that it still answers.
        checks["server_usable_after_sync"] = bool(probe_top_logprobs(server_url, model_name))
    ctx.barrier()

    _sync_the_sinks(ctx, trainer, server_url, model_name, checks)

    if trainer_kind == "sdpg":
        _sdpg_checks(trainer, checks)
    if spec.beta:
        _kl_checks(trainer, checks)

    return {
        "checks": ctx.broadcast_checks(checks),
        "metrics": {"steps_logged": float(len(losses)), "training_had_a_gradient": float(trained)},
    }
