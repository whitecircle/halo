"""Shared body for the online-GRPO and SDPG end-to-end tests against a live vLLM server.

``test_online_grpo_vllm_moe_e2e.py`` (Qwen3-30B-A3B) and ``test_online_grpo_vllm_dense_e2e.py``
(Qwen3-0.6B) are thin wrappers over :func:`run_online_grpo_e2e`. Both trainers and every
adapter x parallelism shape assert the same things about the same server, so the assertions live in
one place; the scripts stay separate because the manifest attaches the family markers and the
model-size timeout per file, and because one URL cannot serve both checkpoints.

The assertions are on the served policy rather than the trainer's tensors: a sync that gathered the
wrong shard, or an adapter fold that never ran, still returns a finite loss from ``train()``. The
shape both on-policy families share (the probe, the perturbation round, the restore-point snapshot,
the parallelism-engaged verdict, the resume comparisons) is :mod:`tests.common.on_policy_e2e`; what
lives here is the axis x adapter table, the vLLM-server ``GRPOConfig``, the two trainer classes, the
TP + LoRA refusal and SDPG's OPD term.

``adapter`` puts the row on a fold no full fine-tune needs: attention PEFT reaches the engine through
the sync's ``merge_adapter``, native grouped expert LoRA through ``merge_lora=True`` in the expert
gather.

``resume`` covers the invariant TRL's ``_last_loaded_step`` sentinel carries: a resumed run generates
its first rollout from the checkpoint's weights rather than from whatever the engine still held.
Phase 1 trains to ``RESUME_MAX_STEPS`` with a checkpoint at ``RESUME_SAVE_STEP``, and the
perturbation round then moves the engine off that policy. Phase 2 is a fresh model and trainer
resuming with one step left, so the sentinel's push is the only thing that can reach the engine
before the resumed rollout. Both phases run in one process against one server on the same
weight-transfer group port, as described in :mod:`tests.common.on_policy_e2e`.
"""

import math
import os
from dataclasses import dataclass
from types import SimpleNamespace
from urllib.parse import urlparse

import torch
from transformers import AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from trl import GRPOConfig

from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.distillation.sdpg import DistributedSDPGTrainer
from src.trainers.grpo.online import DistributedGRPOTrainer
from src.training.environment import resolve_resume_weights_source
from tests.common.distributed import ensure_model_downloaded
from tests.common.ep_reference import ep_layers
from tests.common.on_policy_e2e import (
    RESUME_MAX_STEPS,
    RESUME_SAVE_STEP,
    RestorePointSnapshot,
    adapter_file_agreement,
    expert_lora_under_etp_refusal,
    fresh_parallelism_config,
    load_policy,
    local_view,
    logged_lrs,
    parallelism_engaged,
    perturbation_round,
    probe_top_logprobs,
    qa_dataset,
    record_adapter_training,
    record_optimizer_restore,
    record_resume_continuity,
    record_scheduler_restore,
    record_served_baseline,
    record_step_losses,
    shared_output_dir,
    sink_round,
)
from tests.common.peft_helpers import assert_only_adapters_trainable, snapshot_adapters, unwrap
from tests.common.utils import cleanup_memory, log

# Deliberately short: every step is a full engine round-trip, and the properties under test are that
# the sync landed and that the resume pushed, neither of which needs a converged policy.
MAX_STEPS = 3
NUM_TRAIN_SAMPLES = 16
# Deliberately large: the sync has to move the served weights measurably within a few steps.
FULL_LEARNING_RATE = 1e-4
LORA_LEARNING_RATE = 1e-3
MAX_COMPLETION_LENGTH = 64
NUM_GENERATIONS = 2
PER_DEVICE_BATCH_SIZE = 2
# How far two served logprobs may sit apart and still count as the same policy, on an adapter row.
# Phase 1's last push merges into a base that has been through the earlier steps' merge/unmerge
# round-trips while phase 2 merges into a freshly loaded one, and each round-trip rounds the base
# twice in bf16, so the two pushes differ by that rounding even when the adapters are bit-identical. A
# full fine-tune restores the base itself and is compared exactly. The perturbation round moves these
# logprobs by whole nats, well above this.
SERVED_LOGPROB_TOL = 1e-2


@dataclass(frozen=True)
class _Mode:
    """One row's shape: the axes it engages and which adapter (if any) it trains.

    ``adapter`` is the ``peft_helpers`` mode name: ``None`` (full fine-tune), ``"lora"`` (stock PEFT
    handed to the trainer as ``peft_config``) or ``"expert_lora"`` (native grouped expert adapters
    built inside the EP layers). ``refuses_lora_under_tp`` marks the row that asserts a
    construction-time refusal instead of training; ``rejects_expert_lora`` marks the ETP row, which
    asserts the config-time expert-LoRA refusal before it loads anything. ``family`` is the script a
    mode belongs to, so neither script repeats the list.

    ``checkpoint_optimizer`` carries the optimizer and scheduler into the resume checkpoint, which
    makes the continuity assertions possible. Off for a full fine-tune of the MoE policy, where that
    state is 4 B/param of AdamWBF16 moments (over a hundred GB of disk, and the same again in host RAM
    to compare) for a property the adapter rows and
    ``tests/gpu/parallelism/ep/test_ep_optimizer_resume.py`` already cover at EP scale.
    """

    family: str
    ep_size: int = 1
    expert_tp_size: int = 1
    tp_size: int = 1
    adapter: str | None = None
    refuses_lora_under_tp: bool = False
    rejects_expert_lora: bool = False
    checkpoint_optimizer: bool = True

    @property
    def expert_lora(self) -> bool:
        """Whether this row's adapters are the native grouped expert ones, whose gather is collective."""
        return self.adapter == "expert_lora"


MODES = {
    "full_ep2": _Mode("moe", ep_size=2, checkpoint_optimizer=False),
    "lora_ep2": _Mode("moe", ep_size=2, adapter="lora"),
    "expert_lora_ep2": _Mode("moe", ep_size=2, adapter="expert_lora"),
    "lora_etp2": _Mode("moe", expert_tp_size=2, adapter="lora", rejects_expert_lora=True),
    "full_tp2": _Mode("dense", tp_size=2),
    "lora_fsdp": _Mode("dense", adapter="lora"),
    "lora_tp2_rejected": _Mode("dense", tp_size=2, adapter="lora", refuses_lora_under_tp=True),
}


def modes_for(family: str) -> tuple[str, ...]:
    """The ``--mode`` choices of one script, derived from the table rather than repeated in it."""
    return tuple(name for name, spec in MODES.items() if spec.family == family)


def reward_len(completions, **kwargs):
    """A reward that varies with the completion, so every GRPO group keeps a live advantage spread.

    A tied group has zero advantage and therefore a zero gradient: nothing moves, and both the
    adapter-movement and the SDPG OPD assertions would then measure nothing.
    """
    return [-abs(len(str(completion)) - 32) / 32.0 for completion in completions]


def _config_for(spec: _Mode) -> ParallelismConfig:
    return fresh_parallelism_config(spec.ep_size, spec.tp_size, spec.expert_tp_size)


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
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=parsed.hostname or "localhost",
        vllm_server_port=parsed.port or 8000,
        vllm_server_timeout=600.0,
        vllm_group_port=group_port,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        beta=0.0,
        generation_kwargs={"temperature": 0.7},
        fsdp="",
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
    """Construct the row's trainer. Returns ``(trainer, parallelism_config)``.

    ``weights_source`` is where the weights come from: the hub id normally, or the checkpoint
    directory on a resume whose loader expects the model to be constructed from it.
    """
    parallelism_config = _config_for(spec)
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
    if trainer_kind == "sdpg":
        trainer = DistributedSDPGTrainer(
            **kwargs, sdpg_answer_field="answer", sdpg_loss="reverse_kl", sdpg_beta_base=1.0
        )
    else:
        trainer = DistributedGRPOTrainer(**kwargs)
    return trainer, parallelism_config


def _release(trainer) -> None:
    """Free a trainer the way its dying process would, so the next one can build on the same GPUs.

    The vLLM communicator is closed explicitly rather than left to its ``atexit`` hook: it is what
    frees the trainer-side group port and drops the server's weight-transfer engine, and phase 2
    rebinds both.
    """
    if hasattr(trainer, "cleanup_ep"):
        trainer.cleanup_ep()
    if trainer.accelerator.is_main_process:
        trainer.vllm_generation.vllm_client.close_communicator()
    cleanup_memory()


class _CheckpointOnce(TrainerCallback):
    """Write only the checkpoint the resume targets.

    Under a step-based save strategy HF's default flow saves again when training stops
    (``DefaultFlowCallback.on_step_end``), which at MoE scale is a second copy of the model and its
    optimizer state that nothing in this row reads. Registered after the default callback, so this
    verdict is the one that survives.
    """

    def __init__(self, step: int):
        self._step = step

    def on_step_end(self, args, state, control, **kwargs):
        control.should_save = state.global_step == self._step
        return control


def _weight_witnesses(model, spec: _Mode) -> dict[str, torch.Tensor]:
    """One trained tensor per stream a full fine-tune's checkpoint must carry back.

    Rank-local views, no collective: the expert shards a rank holds are the same in both phases (same
    ``ep_size``), so a rank-local comparison is exact and runs no collective. Empty for an adapter
    row, whose checkpoint holds adapters and whose base is compared separately.
    """
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
    """The shared restore point plus the trained weights a full fine-tune's checkpoint must bring back.

    Read after ``train()`` the witnesses already carry the resumed step's update, and a comparison
    against the checkpoint would fail on a correct restore, hence the callback.
    """

    def __init__(self, event: str, trainer, spec: _Mode, *, capture_optimizer: bool, expert_lora: bool | None = None):
        super().__init__(event, trainer, capture_optimizer=capture_optimizer, expert_lora=expert_lora)
        self._spec = spec

    def extra(self) -> dict:
        return {"weights": _weight_witnesses(self.trainer.model, self._spec)}


def _served_policy_unchanged(after: dict, before: dict, *, exact: bool) -> bool:
    """Whether two served-policy probes describe the same policy.

    Exact for a full fine-tune, whose checkpoint restores the base itself; within
    ``SERVED_LOGPROB_TOL`` for an adapter row, where the two pushes merge into bases a bf16
    merge/unmerge round-trip apart.
    """
    if exact:
        return after == before
    return after.keys() == before.keys() and all(abs(after[k] - before[k]) <= SERVED_LOGPROB_TOL for k in before)


def _perturb_and_sync(ctx, trainer, spec: _Mode, checks: dict, *, check_base_untouched: bool = True) -> None:
    """This row's :func:`perturbation_round`, pushed through the trainer's own weight sync."""
    perturbation_round(
        ctx,
        trainer.model,
        adapter=spec.adapter,
        push=trainer._distributed_sync_weights,
        checks=checks,
        expert_stream=spec.family == "moe",
        check_base_untouched=check_base_untouched,
    )


def _sync_the_sinks(ctx, trainer, server_url: str, model_name: str, checks: dict) -> None:
    """The sink stream's own round, for a policy that has one. Inert on a sink-less family."""
    sink_round(
        ctx,
        trainer.model,
        server_url=server_url,
        model_name=model_name,
        push=trainer._distributed_sync_weights,
        checks=checks,
    )


def _sdpg_checks(trainer, checks: dict) -> None:
    """The OPD term fired AND contributed.

    ``opd_loss``/``opd_beta`` are logged on every step whenever ``sdpg_beta_base != 0``, so their
    presence only shows an SDPG step ran. A regression (the fused-Liger loss bypass, a gate that masks
    every token) would remove the non-zero ``opd_loss``, which needs the privileged-teacher forward
    and at least one positive-advantage rollout, both guaranteed here by a graded reward that keeps
    the group's advantage spread alive.
    """
    opd = [float(entry["opd_loss"]) for entry in trainer.state.log_history if "opd_beta" in entry]
    checks["sdpg_opd_term_fired"] = bool(opd)
    checks["sdpg_opd_loss_finite"] = bool(opd) and all(math.isfinite(v) for v in opd)
    checks["sdpg_opd_term_contributed"] = any(v != 0.0 for v in opd)
    log(f"  OPD over {len(opd)} step(s): opd_loss={opd}")


def _lora_under_tp_refused(**build_kwargs) -> dict[str, bool]:
    """The TP + LoRA refusal, asserted on a real construction rather than on a config object.

    ``_validate_lora_tp_compatibility`` runs inside ``_setup_distributed_modes``, after the trainer's
    own ``__init__`` has built the model wrappers and TRL's weight-transfer client, so a config-level
    assertion never reaches it. The client that construction left behind is closed by the ``atexit``
    hook ``init_communicator`` registers (``src/distributed/nccl/clients/vllm.py``), since the raise
    leaves no handle to close it through.
    """
    try:
        _build_trainer(**build_kwargs)
    except ValueError as error:
        message = str(error)
        log(f"  refusal: {message.splitlines()[0]}")
        return {
            "lora_under_tp_refused": True,
            "refusal_names_tensor_parallelism": "Tensor Parallelism" in message and "tp_size > 1" in message,
        }
    log("  a TP + LoRA trainer CONSTRUCTED: the rank-inconsistent adapter would train unnoticed")
    return {"lora_under_tp_refused": False, "refusal_names_tensor_parallelism": False}


def _run_resume(ctx, *, trainer_kind, spec: _Mode, model_name, tokenizer, server_url, group_port, checks) -> dict:
    """Phase 1 (train, checkpoint, move the engine off it) then phase 2 (resume, prove the push)."""
    output_dir = shared_output_dir(ctx)
    checkpoint = os.path.join(output_dir, f"checkpoint-{RESUME_SAVE_STEP}")
    exact = spec.adapter is None

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
    checks["parallelism_engaged"] = parallelism_engaged(trainer, spec.ep_size, spec.tp_size, spec.expert_tp_size)

    trainer.train()
    record_step_losses(trainer, checks)
    reshard_fsdp2_modules(unwrap(trainer.model))
    phase1_lrs = logged_lrs(trainer)
    checks["checkpoint_captured_at_the_save_step"] = (
        saved.captured is not None and saved.captured["global_step"] == RESUME_SAVE_STEP
    )
    ctx.barrier()

    # The last step's pre-generation sync pushed the checkpoint's own policy, so the engine already
    # holds it and no extra merge stands between this probe and what phase 2 loads.
    checkpoint_policy = {}
    if ctx.rank == 0:
        checkpoint_policy = probe_top_logprobs(server_url, model_name)
        checks["checkpoint_policy_reproducible"] = checkpoint_policy == probe_top_logprobs(server_url, model_name)
        checks["checkpoint_written"] = os.path.isdir(checkpoint)
        log(f"  checkpoint policy: { {k: round(v, 4) for k, v in checkpoint_policy.items()} }")
    ctx.barrier()

    # The base-weight witness belongs to the single-phase rows: here the merge/unmerge round-trip is
    # judged by the resumed engine landing back on the checkpoint's policy.
    _perturb_and_sync(ctx, trainer, spec, checks, check_base_untouched=False)
    ctx.barrier()
    if ctx.rank == 0:
        moved_off = probe_top_logprobs(server_url, model_name)
        checks["engine_moved_off_the_checkpoint_policy"] = not _served_policy_unchanged(
            moved_off, checkpoint_policy, exact=exact
        )
        log(f"  engine moved off the checkpoint policy: {checks['engine_moved_off_the_checkpoint_policy']}")
    ctx.barrier()

    # Off it in the sink stream too, which the dense delta above does not cover: the resume has to
    # bring both back.
    _sync_the_sinks(ctx, trainer, server_url, model_name, checks)

    _release(trainer)
    live["trainer"] = None
    # The snapshot outlives its trainer, but only as values: holding the trainer would keep phase 1's
    # model on the GPUs next to phase 2's.
    saved.trainer = None
    del trainer
    cleanup_memory()
    ctx.barrier()

    # Phase 2: a fresh model and trainer, as a restarted job builds them, which puts TRL's
    # ``_last_loaded_step`` back at its ``-1`` sentinel. The production rule rather than a test-local
    # one: under the EP/CP wrappers a full checkpoint's weights load only at construction, while an
    # adapter-only checkpoint keeps the base and has its adapters restored.
    parallelism_config = _config_for(spec)
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
        checks["resume_returned_the_engine_to_the_checkpoint_policy"] = _served_policy_unchanged(
            after, checkpoint_policy, exact=exact
        )
        if not checks["resume_returned_the_engine_to_the_checkpoint_policy"]:
            log("  the resumed run's first rollout did NOT come from the checkpoint's weights")
        log(f"  post-resume: { {k: round(v, 4) for k, v in after.items()} }")

    capture = restored.captured
    checks["resume_ran_the_remaining_step"] = resumed.state.global_step == RESUME_MAX_STEPS
    record_resume_continuity(checks, restored=capture, phase1_lrs=phase1_lrs, phase2_lrs=logged_lrs(resumed))

    if spec.checkpoint_optimizer:
        record_scheduler_restore(checks, capture)
        record_optimizer_restore(checks, saved.captured, capture)

    # Train-begin captures rather than post-train reads: one resumed step has already moved them. A
    # save that never fired leaves no witnesses, which the check above reports.
    witnesses = saved.captured["weights"] if saved.captured else {}
    if witnesses:
        # Only a full fine-tune has these: its checkpoint carries the weights themselves, so an
        # inexact restore is visible here before the engine ever sees it.
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
    """Drive one (trainer × mode × resume) row against the live server at ``server_url``."""
    spec = MODES[mode]
    checks: dict[str, bool] = {}
    log(f"\n  row: trainer={trainer_kind} mode={mode} resume={resume} model={model_name} server={server_url}")

    if spec.rejects_expert_lora:
        refusal = expert_lora_under_etp_refusal(spec.expert_tp_size)
        checks["expert_lora_under_etp_refused"] = bool(refusal)
        checks["etp_refusal_names_expert_tp_size"] = "expert_tp_size > 1" in refusal

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
    checks["parallelism_engaged"] = parallelism_engaged(trainer, spec.ep_size, spec.tp_size, spec.expert_tp_size)
    log(f"  parallelism engaged: {checks['parallelism_engaged']} ({mode})")

    adapters_before = {}
    if spec.adapter:
        adapters_before = snapshot_adapters(unwrap(trainer.model), expert_lora=spec.expert_lora)
        ok, detail = assert_only_adapters_trainable(unwrap(trainer.model))
        checks["only_adapters_trainable"] = ok
        log(f"  frozen base: {detail}")

    trainer.train()
    log("  training completed")
    losses, trained = record_step_losses(trainer, checks)
    reshard_fsdp2_modules(unwrap(trainer.model))

    if spec.adapter:
        record_adapter_training(trainer, checks, before=adapters_before, expert_lora=spec.expert_lora, trained=trained)

    ctx.barrier()
    before = probe_top_logprobs(server_url, model_name) if ctx.rank == 0 else {}
    if ctx.rank == 0:
        checks["served_logprobs_finite"] = all(math.isfinite(v) for v in before.values())
    _perturb_and_sync(ctx, trainer, spec, checks)
    ctx.barrier()

    if ctx.rank == 0:
        after = probe_top_logprobs(server_url, model_name)
        checks["forced_sync_moved_the_served_policy"] = after != before
        if after == before:
            log("  IDENTICAL logprobs after the perturbation: the weight sync did not land")
        # A failed update leaves the engine partially written, so check that it still answers.
        checks["server_usable_after_sync"] = bool(probe_top_logprobs(server_url, model_name))
    ctx.barrier()

    _sync_the_sinks(ctx, trainer, server_url, model_name, checks)

    if trainer_kind == "sdpg":
        _sdpg_checks(trainer, checks)

    return {"checks": ctx.broadcast_checks(checks), "metrics": {"steps_logged": float(len(losses))}}
