"""Shared body for the per-engine Environmental-GRPO end-to-end tests.

``test_env_grpo_vllm_e2e.py`` and ``test_env_grpo_sglang_e2e.py`` are thin wrappers over
:func:`run_env_grpo_e2e`: the engines differ only in which server the rollouts and the weight-sync
group talk to, so the assertions live in one place. They stay separate scripts because the manifest
attaches ``vllm_server`` / ``sglang_server`` markers per file, and a single file carrying both would
require both servers for either selection.

The assertions are on the served policy rather than the trainer's tensors: a run whose weight sync
never landed still returns from ``train()``. The trainer-agnostic half (the served-policy probe, the
perturbation round, the restore-point snapshot and the resume comparisons) lives in
:mod:`tests.common.on_policy_e2e`, which the online-GRPO body drives too; what is here is what
Environmental GRPO alone needs (Ray rollout actors, multi-turn trajectories, per-turn token capture).

The model is a MoE in both modes, so each engine's rows are an A/B across the ep split:
``ep_size=1`` shards experts as DTensor under FSDP2 (``fsdp_shard_ep1_experts``), while
``ep_size>1`` leaves them FSDP-ignored plain tensors. Those are different gather paths into the same
engine loader, and only the served policy distinguishes them.

``peft`` puts the run on the adapter path (``"lora"`` attention PEFT, ``"expert_lora"`` native
grouped expert adapters), where an adapter reaches the engine only through the fold the sync performs
(``merge_adapter`` for attention, ``merge_lora=True`` in the expert gather).

``resume`` covers the train-begin force sync. Phase 1 trains to ``RESUME_MAX_STEPS`` with a
checkpoint at ``RESUME_SAVE_STEP``, and the perturbation round then moves the engine off that policy.
Phase 2 is a fresh model and trainer resuming with one step left, so the train-begin push is the only
thing that can reach the engine before the resumed rollout. Where the resumed policy lands is exact
for a full fine-tune and a nearer-of-the-two verdict for an adapter row, whose two pushes merge into
bases one bf16 merge round-trip apart.
"""

import math
import os
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer
from trl import GRPOConfig

from src.configs.async_training_config import AsyncTrainingConfig
from src.configs.environment_config import EnvironmentConfig
from src.distributed.fsdp import reshard_fsdp2_modules
from src.env import env_int, env_str
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from src.training.environment import resolve_resume_weights_source
from tests.common.distributed import ensure_model_downloaded
from tests.common.models import QWEN3_30B_A3B
from tests.common.on_policy_e2e import (
    RESUME_MAX_STEPS,
    RESUME_SAVE_STEP,
    RestorePointSnapshot,
    adapter_file_agreement,
    expert_lora_under_etp_refusal,
    fresh_parallelism_config,
    load_policy,
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
    served_policy_delta,
    shared_output_dir,
    sink_round,
)
from tests.common.peft_helpers import assert_only_adapters_trainable, snapshot_adapters, unwrap
from tests.common.thinking_budget import (
    ReasoningTurnRecorder,
    record_budget_enforcement,
    record_thinking_budget_checks,
)
from tests.common.utils import cleanup_memory, log

# Default for the vLLM half (its callers take it via run_env_grpo_e2e's model_name default); the
# SGLang caller passes gpt-oss explicitly, so HALO_TEST_ENV_GRPO_MODEL does not reach it.
MODEL_NAME = env_str("HALO_TEST_ENV_GRPO_MODEL", QWEN3_30B_A3B)
MAX_STEPS = env_int("HALO_TEST_ENV_GRPO_MAX_STEPS", 2)
# Deliberately large: the sync has to move the served weights measurably within MAX_STEPS.
LEARNING_RATE = 1e-4


def _make_trainer(
    *, model, tokenizer, dataset, parallelism_config, peft_config, grpo_config, async_config, trainer_cls
):
    return trainer_cls(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        async_config=async_config,
        parallelism_config=parallelism_config,
        peft_config=peft_config,
        environment_config=EnvironmentConfig(environment_type="native_math", environment_kwargs={"max_turns": 2}),
    )


def _grpo_config(*, output_dir: str, max_steps: int, group_port: int, save: bool, peft: str | None) -> GRPOConfig:
    return GRPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        bf16=True,
        logging_steps=1,
        save_strategy="steps" if save else "no",
        save_steps=RESUME_SAVE_STEP,
        # A full fine-tune's optimizer shards are 6 B/param, 3x the checkpoint's own weights at this
        # size. The adapter rows carry the optimizer-continuity assertion; here the EP saver still
        # writes scheduler.pt under save_only_model, which is what this row's continuity check reads.
        save_only_model=save and peft is None,
        # The invariant under test is the train-begin push, not the dataloader's replay: re-walking
        # the consumed batches would re-run generation for steps already in the checkpoint.
        ignore_data_skip=True,
        report_to="none",
        num_generations=2,
        max_completion_length=128,
        beta=0.0,
        generation_kwargs={"temperature": 0.7},
        fsdp="",
        vllm_group_port=group_port,
        remove_unused_columns=False,
    )


def _async_config(
    *, backend: str, server_url: str, model_name: str, thinking_budget: int | None, routing_replay: str
) -> AsyncTrainingConfig:
    return AsyncTrainingConfig(
        rollout_backend=backend,
        rollout_server_url=server_url,
        rollout_connection_timeout=180.0,
        num_rollout_workers=2,
        max_concurrent_rollouts=2,
        rollout_max_tokens=128,
        rollout_max_thinking_tokens=thinking_budget,
        routing_replay=routing_replay,
        enable_prefetch=False,
        sync_weights_every_n_steps=1,
        model_name=model_name,
        # The engine-specific capture wiring exists for this: the trainer trains the ids the engine
        # sampled, not a re-tokenization of the text it rendered.
        train_on_sampled_tokens=True,
    )


def record_routing_replay(trainer, checks: dict[str, bool]) -> None:
    """The engine's own routed-expert ids reached the loss, and lined up with the sampled tokens.

    Reaching here already covers the capture wire: a batch in which no rollout returned
    ``routed_experts`` raises rather than falling back. What it does not cover is placement: a mask
    the assembler cannot line up against the turn's tokens lands in ``unresolved``, and that turn
    replays nothing while the run continues.
    """
    coverage = [
        {key: float(value) for key, value in entry.items() if key.startswith("routing/rollout_")}
        for entry in trainer.state.log_history
        if any(key.startswith("routing/rollout_") for key in entry)
    ]
    checks["routing_replay_masks_assembled"] = bool(coverage)
    checks["routing_replay_placed_every_mask"] = bool(coverage) and all(
        step.get("routing/rollout_unresolved_frac", 0.0) == 0.0 for step in coverage
    )
    log(f"  routing replay coverage per step: {coverage}")


def run_env_grpo_e2e(
    ctx,
    *,
    backend: str,
    server_url: str,
    group_port: int,
    ep_size: int,
    tp_size: int = 1,
    expert_tp_size: int = 1,
    peft: str | None = None,
    resume: bool = False,
    model_name: str = MODEL_NAME,
    thinking_budget: int | None = None,
    routing_replay: str = "none",
) -> dict:
    """Train Environmental GRPO against a live ``backend`` server and assert the sync landed.

    ``model_name`` is the checkpoint both sides use: the trainer loads it and the server must already
    serve it. It is a parameter because the two backends do not accept the same families: SGLang loads
    MoE experts in the checkpoint-fused layout that only the GptOss layer gathers, so every other MoE
    family is refused at construction and a shared default could only assert that refusal.

    ``peft`` selects the adapter path (``"lora"`` / ``"expert_lora"``) and ``resume`` adds the
    second, resumed phase; both are described in the module docstring. ``thinking_budget`` sets
    ``rollout_max_thinking_tokens`` and grades what the engine reasoned against it
    (:mod:`tests.common.thinking_budget`).
    """
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    pre_perturb: dict[str, float] = {}
    expert_lora = peft == "expert_lora"
    max_steps = RESUME_MAX_STEPS if resume else MAX_STEPS

    # ── 0. expert TP refuses the native grouped expert adapters, at config time ───────────────
    # Before the load, which is the point of the refusal: it fires ahead of a multi-hundred-GB
    # checkpoint download rather than at group construction.
    if expert_tp_size > 1:
        refusal = expert_lora_under_etp_refusal(expert_tp_size)
        checks["expert_lora_under_expert_tp_refused"] = bool(refusal)
        checks["expert_lora_etp_refusal_names_expert_tp"] = "expert_tp_size > 1" in refusal
        log(f"  expert-LoRA under expert_tp_size={expert_tp_size} refusal: {refusal or '<NOT RAISED>'}")

    # ── 1. the server is up, serving the checkpoint we are about to train, and reproducible ───
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if ctx.rank == 0:
        log(f"  backend: {backend}")
        record_served_baseline(server_url, model_name, checks)
        if thinking_budget is not None:
            # Before the load: without a reasoning parser the server 400s every request carrying the
            # field, and the run would train on all-masked batches with a moving loss.
            record_budget_enforcement(server_url, model_name, thinking_budget, tokenizer, checks)
    ctx.barrier()

    # ── 2. train ──────────────────────────────────────────────────────────────────────────────
    ensure_model_downloaded(model_name, ctx.rank)
    output_dir = shared_output_dir(ctx) if resume else ctx.output_dir
    dataset = qa_dataset(8)
    async_config = _async_config(
        backend=backend,
        server_url=server_url,
        model_name=model_name,
        thinking_budget=thinking_budget,
        routing_replay=routing_replay,
    )
    grpo_config = _grpo_config(
        output_dir=output_dir, max_steps=max_steps, group_port=group_port, save=resume, peft=peft
    )

    parallelism_config = fresh_parallelism_config(ep_size, tp_size, expert_tp_size)
    model, peft_config = load_policy(model_name, parallelism_config, peft)

    trainer_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "dataset": dataset,
        "parallelism_config": parallelism_config,
        "peft_config": peft_config,
        "grpo_config": grpo_config,
        "async_config": async_config,
        "trainer_cls": ReasoningTurnRecorder
        if thinking_budget is not None
        else DistributedAsyncEnvironmentalGRPOTrainer,
    }

    # Adapters under TP are refused where the trainer realizes its parallel modes, so the refusal
    # needs a loaded, PEFT-wrapped model and nothing more. The row ends here: there is no trainer.
    if peft is not None and tp_size > 1:
        refusal = ""
        try:
            _make_trainer(**trainer_kwargs)
        except ValueError as e:
            refusal = str(e)
        checks["adapters_under_tp_refused"] = bool(refusal)
        # Matched by name rather than by exception type: every other construction-time refusal on
        # this path (weight sync, environment spec, batch shape) also raises ValueError.
        checks["tp_refusal_names_tensor_parallelism"] = "Tensor Parallelism (tp_size > 1)" in refusal
        log(f"  adapters under tp_size={tp_size} refusal: {refusal.splitlines()[0] if refusal else '<NOT RAISED>'}")
        return {"checks": ctx.broadcast_checks(checks), "metrics": metrics}

    # One finalizer over a mutable slot rather than a bound method per phase: a registered
    # ``trainer.cleanup_ep`` pins its trainer (and its model's GPU memory) for the whole run, which a
    # resume row cannot afford, since it builds a second one.
    live: dict[str, object] = {"trainer": None}
    ctx.on_teardown(lambda: live["trainer"].cleanup_ep() if live["trainer"] is not None else None)

    trainer = _make_trainer(**trainer_kwargs)
    live["trainer"] = trainer
    saved_state = RestorePointSnapshot("save", trainer, capture_optimizer=peft is not None)
    if resume:
        trainer.add_callback(saved_state)

    checks["parallelism_engaged"] = parallelism_engaged(trainer, ep_size, tp_size, expert_tp_size)
    log(
        f"  ep_size={ep_size} tp_size={tp_size} expert_tp_size={expert_tp_size} peft={peft} "
        f"engaged={checks['parallelism_engaged']} (ep_config={getattr(trainer._ep_config, 'ep_size', None)})"
    )

    adapters_before: dict[str, torch.Tensor] = {}
    if peft is not None:
        # Inspected on the trainer's model: attention LoRA is applied by TRL's get_peft_model, so the
        # adapters only exist after construction.
        ok, detail = assert_only_adapters_trainable(unwrap(trainer.model))
        checks["only_adapters_trainable"] = ok
        log(f"  [trainable] {detail}")
        adapters_before = snapshot_adapters(unwrap(trainer.model), expert_lora=expert_lora)

    trainer.train()
    log("  training completed")
    # Everything below reads or writes parameters by identity, and the run leaves the FSDP2 modules
    # holding their transient unsharded params, which the optimizer never stepped and the next unshard
    # discards. Reading those reports zero movement for a run that trained, and perturbing them writes
    # into a buffer the trainer's own weights never see. Same call the optimizer build and the save
    # path make.
    reshard_fsdp2_modules(unwrap(trainer.model))

    losses, trained = record_step_losses(trainer, checks)
    metrics["steps_logged"] = float(len(losses))
    metrics["training_had_a_gradient"] = float(trained)

    # ── 3. the engine's own sampled ids reached the loss ──────────────────────────────────────
    # The trainer falls back to re-tokenization when a rollout returns no ids, and that fallback warns
    # rather than raising, so on a backend whose capture spelling is wrong training succeeds while
    # optimizing tokens the engine never sampled.
    checks["trained_on_engine_sampled_ids"] = trainer._train_on_sampled_tokens and not trainer._warned_capture_missing

    if peft is not None:
        record_adapter_training(trainer, checks, before=adapters_before, expert_lora=expert_lora, trained=trained)

    # ── 4. a sync through the trainer's own client must reach the served policy ───────────────
    ctx.barrier()
    if ctx.rank == 0:
        pre_perturb = probe_top_logprobs(server_url, model_name)
        checks["served_logprobs_finite"] = all(math.isfinite(v) for v in pre_perturb.values())
        log(f"  post-training: { {k: round(v, 4) for k, v in pre_perturb.items()} }")
    what = perturbation_round(
        ctx,
        trainer.model,
        adapter=peft,
        push=lambda: trainer._sync_weights_to_engine(force=True),
        checks=checks,
    )
    ctx.barrier()

    if ctx.rank == 0:
        after = probe_top_logprobs(server_url, model_name)
        checks["forced_sync_moved_the_served_policy"] = after != pre_perturb
        if after == pre_perturb:
            log(f"  IDENTICAL logprobs after a {what} perturbation: the weight sync did not land")
        log(f"  post-forced-sync: { {k: round(v, 4) for k, v in after.items()} }")
        # A failed update leaves the engine partially written, and both engines' docs say to discard
        # such a server rather than keep serving from it, so check that it still answers.
        checks["server_usable_after_sync"] = bool(probe_top_logprobs(server_url, model_name))
    ctx.barrier()

    # ── 5. GptOss attention sinks, in a round of their own ────────────────────────────────────
    # Inert for a sink-less family (the vLLM half's default Qwen3 MoE); see :func:`sink_round`.
    sink_round(
        ctx,
        trainer.model,
        server_url=server_url,
        model_name=model_name,
        push=lambda: trainer._sync_weights_to_engine(force=True),
        checks=checks,
    )

    if thinking_budget is not None:
        record_thinking_budget_checks(trainer, checks, thinking_budget)
    if routing_replay == "rollout":
        record_routing_replay(trainer, checks)

    if not resume:
        return {"checks": ctx.broadcast_checks(checks), "metrics": metrics}

    # ── 6. a resumed run's FIRST rollout must come from the checkpoint's weights ──────────────
    # ``pre_perturb`` is that policy as the engine served it: with sync_weights_every_n_steps=1 the
    # last push of phase 1 ran at the top of the step after the checkpoint, so when train() returned
    # the engine held the checkpoint's weights. Everything since moved it off them.
    checkpoint = os.path.join(output_dir, f"checkpoint-{RESUME_SAVE_STEP}")
    checks["resume_checkpoint_written"] = os.path.isdir(checkpoint)
    phase1_lrs = logged_lrs(trainer)
    moved_off: dict[str, float] = {}
    if ctx.rank == 0:
        # Negative control: an engine still sitting on the checkpoint's policy would satisfy the
        # comparison below with nothing pushed. It is also the state a resume that skipped its push
        # would leave behind, bit for bit.
        moved_off = probe_top_logprobs(server_url, model_name)
        checks["engine_moved_off_the_checkpoint_policy"] = moved_off != pre_perturb
    ctx.barrier()

    # A fresh model and trainer, as a resumed job starts. The client is closed first: its group port
    # is held for the process lifetime, and phase 2 rebuilds a client on the same one.
    trainer._cleanup_async_components()
    trainer.cleanup_ep()
    # The snapshot outlives its trainer, but only as values: holding the trainer would keep phase 1's
    # model on the GPU next to phase 2's.
    saved_state.trainer = None
    live["trainer"] = None
    del trainer, model
    cleanup_memory()
    ctx.barrier()

    parallelism_config = fresh_parallelism_config(ep_size, tp_size, expert_tp_size)
    # The production rule rather than a test-local one: under the EP/CP wrappers a full checkpoint's
    # weights load only at construction (the trainer's loader refuses to reload them), while an
    # adapter-only checkpoint keeps the base and has its adapters restored by the loader.
    weights_source = resolve_resume_weights_source(
        checkpoint, SimpleNamespace(model_name_or_path=model_name), parallelism_config
    )
    checks["resume_weights_source_resolved"] = weights_source == (model_name if peft is not None else checkpoint)
    log(f"  resume: policy weights from {weights_source}")

    model, peft_config = load_policy(weights_source, parallelism_config, peft)
    trainer = _make_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        parallelism_config=parallelism_config,
        peft_config=peft_config,
        # The resumed phase writes nothing: it is judged on the served policy and on what it
        # restored, and a second full-model checkpoint would only cost scratch space.
        grpo_config=_grpo_config(
            output_dir=output_dir, max_steps=max_steps, group_port=group_port, save=False, peft=peft
        ),
        async_config=async_config,
        trainer_cls=DistributedAsyncEnvironmentalGRPOTrainer,
    )
    live["trainer"] = trainer
    restored_state = RestorePointSnapshot(
        "train_begin", trainer, capture_optimizer=peft is not None, expert_lora=expert_lora if peft else None
    )
    trainer.add_callback(restored_state)

    trainer.train(resume_from_checkpoint=checkpoint)
    log("  resumed run completed")
    reshard_fsdp2_modules(unwrap(trainer.model))

    # Exactly one step ran, so the train-begin push is the only thing that reached the engine: the
    # per-step sync declines a step whose number the train-begin push already stamped.
    checks["resume_ran_the_remaining_step"] = trainer.state.global_step == RESUME_MAX_STEPS
    if ctx.rank == 0:
        after_resume = probe_top_logprobs(server_url, model_name)
        # A skipped push leaves the engine bit-identical to what phase 1 left there (the server is
        # idle and the probe is greedy), so this half is exact on every row.
        checks["resume_left_the_stale_engine_state"] = after_resume != moved_off
        to_checkpoint = served_policy_delta(after_resume, pre_perturb)
        if peft is None:
            # A full fine-tune's checkpoint restores the weights themselves: both pushes send the
            # same tensors, so the two probes must agree exactly.
            checks["resumed_rollouts_served_the_checkpoint_policy"] = after_resume == pre_perturb
        else:
            # An adapter run's two pushes merge into different bases: PEFT's
            # merge_adapter/unmerge_adapter round-trip is not bf16-reversible, so phase 1 pushed from
            # a base that had round-tripped and phase 2 from a freshly loaded one. The verdict is
            # therefore which state the engine is nearer, since a push that never happened sits at
            # zero from the stale one. What the push carried is pinned by
            # resume_adapters_match_the_checkpoint_file below.
            checks["resumed_rollouts_served_the_checkpoint_policy"] = to_checkpoint < served_policy_delta(
                after_resume, moved_off
            )
        if not checks["resumed_rollouts_served_the_checkpoint_policy"]:
            log("  the resumed run's engine is NOT on the checkpoint's policy: the train-begin sync did not land")
        log(
            f"  post-resume: { {k: round(v, 4) for k, v in after_resume.items()} } (delta to checkpoint {to_checkpoint:.4f}, to stale {served_policy_delta(after_resume, moved_off):.4f})"
        )

    capture = restored_state.captured
    record_resume_continuity(checks, restored=capture, phase1_lrs=phase1_lrs, phase2_lrs=logged_lrs(trainer))
    record_scheduler_restore(checks, capture)

    if peft is not None:
        record_optimizer_restore(checks, saved_state.captured, capture)
        # Captured at train-begin, not here: the resumed step moves the adapters off the values the
        # restore put back, and the checkpoint file is what the restore is judged against.
        if ctx.rank == 0:
            matching, compared, nonzero_b = adapter_file_agreement(capture["adapters"], checkpoint)
            checks["resume_adapters_match_the_checkpoint_file"] = compared > 0 and matching == compared
            # Same reason as ``adapters_moved_...``: with no gradient anywhere in phase 1 the
            # checkpoint holds the zero-init B, so "restored non-zero" would assert nothing.
            checks["resume_restored_a_trained_adapter"] = nonzero_b > 0 or not trained
            log(f"  restored adapters vs {checkpoint}: {matching}/{compared} exact, {nonzero_b} non-zero B")
    metrics["resume_steps"] = float(RESUME_MAX_STEPS - RESUME_SAVE_STEP)
    ctx.barrier()

    return {"checks": ctx.broadcast_checks(checks), "metrics": metrics}
