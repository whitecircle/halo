"""Shared body of the Environmental-GRPO end-to-end tests, which the vLLM, SGLang and single-process wrappers drive
through :func:`run_env_grpo_e2e`, asserting on the served policy (:mod:`tests.common.on_policy_e2e`). Here is what
Environmental GRPO alone needs (Ray rollout actors, multi-turn trajectories, per-turn token capture), the multi-rank
rows' task (:class:`DrawNumberEnvironment`) and the flip-rate bar each R3 row measures (:func:`routing_probe_round`).
"""

import contextlib
import math
import os
import random
import re
from functools import partial
from types import SimpleNamespace

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig

from halo.checkpoint.parallel.resume_source import resolve_resume_weights_source
from halo.configs.grpo_environmental import AsyncRolloutConfig, EnvironmentConfig
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.env import env_int, env_str
from halo.environments.episode_scoring import EpisodeGrade
from halo.environments.messages import Trajectory
from halo.environments.protocols.native_tool_use import NativeToolUseEnvironment
from halo.environments.tools.definitions import NativeTool, NativeToolRegistry
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.environmental.trajectory_tokenize import SAMPLED_IDS_MISSING_WARNING
from halo.trainers.grpo.objective.logratio import LOGRATIO_MEAN_KEY
from tests.common.adapters import assert_only_adapters_trainable, snapshot_adapters, unwrap
from tests.common.checkpoint_io import RestorePointSnapshot
from tests.common.distributed import ensure_model_downloaded, shared_output_dir
from tests.common.ep_reference import ep_layers
from tests.common.harness import record_check
from tests.common.memory import cleanup_memory
from tests.common.models import QWEN3_30B_A3B
from tests.common.numerics import max_or_nan, min_or_nan
from tests.common.on_policy_e2e import (
    RESUME_MAX_STEPS,
    RESUME_SAVE_STEP,
    PolicyLayout,
    adapter_file_agreement,
    expert_lora_under_etp_refusal,
    expert_round,
    load_policy,
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
    served_policy_delta,
    served_policy_moved,
    sink_round,
)
from tests.common.rank_log import log
from tests.common.thinking_budget import (
    ReasoningTurnRecorder,
    record_budget_acceptance,
    record_budget_enforcement,
    record_thinking_budget_checks,
)

# Default for the vLLM legs; the SGLang and 4-GPU wrappers own their own knob (their servers and
# default families differ).
MODEL_NAME = env_str("HALO_TEST_ENV_GRPO_MODEL", QWEN3_30B_A3B)
MAX_STEPS = env_int("HALO_TEST_ENV_GRPO_MAX_STEPS", 2)
# Per-family overrides for a pass on a family the defaults do not fit: an attention backend the
# family's code lacks (Bailing's remote code has no FA4), or projection names the checkpoint index
# cannot derive (``query_key_value``). Unset keeps auto-selection and the derived targets.
ATTN_IMPL = env_str("HALO_TEST_ENV_GRPO_ATTN_IMPL") or None
LORA_TARGETS = [name for name in env_str("HALO_TEST_ENV_GRPO_LORA_TARGETS", "").split(",") if name] or None
# Deliberately large: the sync has to move the served weights measurably within MAX_STEPS.
LEARNING_RATE = 1e-4
# Steps a prefetch row runs past its prefetch_depth cold rounds, each popping the round submitted that many steps
# earlier.
PREFETCH_POPPED_STEPS = 2
MAX_TURNS = 2
# The task every policy trains on: one call of the draw tool, then the number it returned as the answer.
DRAW_TOOL = "draw_number"
DRAW_RANGE = 10**6
DRAW_PROMPT = f"Call the {DRAW_TOOL} tool once, then reply with the number it returned and nothing else."
# Openings of different lengths the rows take in turn, so their prompts route apart and the length deal reorders rows
# that differ.
DRAW_PREAMBLES = (
    "",
    "The river bends twice before it reaches the old mill. ",
    "Farmers along the river measure the season by the height of the reeds, and the miller keeps a ledger of every "
    "flood. ",
    "The miller's grandfather built the wheel from oak felled on the north ridge, hauled it down in winter on "
    "sledges, and set it turning the spring the stone bridge was finished. ",
)
# gpt-oss's tool-call terminator, which harmony-disabled serving does not stop on (it is no eos); a tokenizer without
# it ends its turns at eos alone.
HARMONY_CALL_TOKEN = "<|call|>"
# The flip-rate bar's two ends, measured on this one row: the policy's routing against itself with every MoE layer
# input moved by 2**-7 of its magnitude (one to two bf16 ulps, what an independent bf16 forward disagrees by), and
# against its own routing one token over (a misplaced mask).
ROUTING_PROBE_TEXT = (
    "The river bends twice before it reaches the old mill, where the water slows and turns green. Farmers along "
    "its banks measure the season by the height of the reeds, and the miller keeps a ledger of every flood since "
    "his grandfather built the wheel."
)
ROUTING_FLOOR_PERTURBATION = 2**-7
ROUTING_PROBE_SEED = 0
# How far apart the probe's two ends must sit for the bar between them to stand clear of both.
ROUTING_PROBE_SEPARATION = 4.0


def _draw_number(**_arguments) -> str:
    """A fresh draw from ``[1, DRAW_RANGE)`` per call, unseeded: the two samples of one prompt run as separate
    episodes, on any rollout worker, and must not draw alike."""
    return str(random.SystemRandom().randrange(1, DRAW_RANGE))


class DrawNumberEnvironment(NativeToolUseEnvironment):
    """One tool, which returns a fresh random number, and the answer graded on reporting it: ``number / DRAW_RANGE``
    when the final answer carries the episode's draw, 0 otherwise.

    Two samples of one prompt draw apart, so a group in which both report spreads however peaked the policy is. A
    grade the policy's own text decides ties whenever its samples coincide, which a confident instruct model's do on
    a short answer even at temperature 1: its groups then carry no advantage and the step no gradient.
    """

    def __init__(self, **kwargs):
        registry = NativeToolRegistry().register(
            NativeTool(name=DRAW_TOOL, description="Draw a random whole number.", handler=_draw_number)
        )
        super().__init__(tool_registry=registry, tool_budgets={DRAW_TOOL: 1}, **kwargs)

    def _grade_episode(self, trajectory: Trajectory, context: dict | None = None) -> EpisodeGrade:
        drawn = next(
            (m.content.strip() for m in trajectory.messages if m.role == "tool" and m.name == DRAW_TOOL), None
        )
        answer = self._final_answer(trajectory)
        if drawn is None or not drawn.isdigit() or answer is None:
            return EpisodeGrade(0.0)
        reported = re.findall(r"\d+", answer.replace(",", ""))
        return EpisodeGrade(int(drawn) / DRAW_RANGE if drawn in reported else 0.0)


def draw_number_dataset(num_samples: int) -> Dataset:
    """``num_samples`` rows of :data:`DRAW_PROMPT`, each opening with the next of :data:`DRAW_PREAMBLES`."""
    return Dataset.from_dict(
        {"prompt": [DRAW_PREAMBLES[i % len(DRAW_PREAMBLES)] + DRAW_PROMPT for i in range(num_samples)]}
    )


def _environment_and_dataset(environment_type: str | None) -> tuple[dict, Dataset]:
    """The trainer's environment keywords and the rows it trains on: :class:`DrawNumberEnvironment` over
    :func:`draw_number_dataset`, or the registry's ``environment_type`` over the arithmetic rows it grades
    against their answers."""
    if environment_type is None:
        environment = {"environment_cls": DrawNumberEnvironment, "environment_kwargs": {"max_turns": MAX_TURNS}}
        return environment, draw_number_dataset(8)
    environment = {"environment_config": EnvironmentConfig(environment_type=environment_type, max_turns=MAX_TURNS)}
    return environment, qa_dataset(8)


def _make_trainer(
    *,
    model,
    tokenizer,
    dataset,
    environment,
    parallelism_config,
    peft_config,
    grpo_config,
    async_config,
    trainer_cls,
):
    return trainer_cls(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        async_config=async_config,
        parallelism_config=parallelism_config,
        peft_config=peft_config,
        **environment,
    )


def _grpo_config(*, output_dir: str, max_steps: int, save: bool, peft: str | None, chunked: bool) -> GRPOConfig:
    return GRPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        # The same two rows a step either way; a chunked row forwards them one per micro-batch, two micro-batches
        # a round, the shape the length deal reorders.
        per_device_train_batch_size=1 if chunked else 2,
        gradient_accumulation_steps=2 if chunked else 1,
        learning_rate=LEARNING_RATE,
        bf16=True,
        logging_steps=1,
        save_strategy="steps" if save else "no",
        save_steps=RESUME_SAVE_STEP,
        # A full fine-tune's optimizer shards are 4 B/param of AdamWBF16 moments, 2x the checkpoint's
        # own weights at this size. The adapter rows carry the optimizer-continuity assertion; here the
        # EP saver still writes scheduler.pt under save_only_model, which this row's continuity check reads.
        save_only_model=save and peft is None,
        # The invariant under test is the train-begin push, not the dataloader's replay: re-walking
        # the consumed batches would re-run generation for steps already in the checkpoint.
        ignore_data_skip=True,
        report_to="none",
        logging_nan_inf_filter=False,
        num_generations=2,
        max_completion_length=128,
        beta=0.0,
        remove_unused_columns=False,
    )


def _async_config(
    *,
    backend: str,
    server_url: str,
    pool_urls: tuple[str, ...],
    group_port: int,
    model_name: str,
    thinking_budget: int | None,
    routing_replay: str,
    prefetch_depth: int | None,
    chunked: bool,
    length_ordered: bool | None,
    stop_tokens: list[str],
) -> AsyncRolloutConfig:
    return AsyncRolloutConfig(
        rollout_backend=backend,
        rollout_server_url=server_url,
        # Each pool server's group port is the base rollout_group_port plus its index.
        rollout_server_configs=[{"url": url} for url in pool_urls] or None,
        rollout_group_port=group_port,
        rollout_connection_timeout=180.0,
        num_rollout_workers=2,
        max_concurrent_rollouts=2,
        rollout_max_tokens=128,
        rollout_stop_tokens=stop_tokens,
        rollout_max_thinking_tokens=thinking_budget,
        routing_replay=routing_replay,
        enable_prefetch=prefetch_depth is not None,
        prefetch_depth=prefetch_depth or 1,
        use_chunked_grpo_logprobs=chunked,
        length_ordered_microbatches=length_ordered,
        sync_weights_every_n_steps=1,
        model_name=model_name,
        # The engine-specific capture wiring exists for this: the trainer trains the ids the engine
        # sampled, not a re-tokenization of the text it rendered.
        train_on_sampled_tokens=True,
    )


def routing_probe_ids(tokenizer, device: torch.device) -> torch.Tensor:
    """``[1, seq]`` token ids of :data:`ROUTING_PROBE_TEXT`, the same on every rank."""
    return tokenizer(ROUTING_PROBE_TEXT, add_special_tokens=False, return_tensors="pt").input_ids.to(device)


def capture_routing(trainer, ids: torch.Tensor) -> torch.Tensor:
    """COLLECTIVE. The policy's own top-k selections on ``ids``, ``[rows, seq, EP layers, top_k]``, through the
    trainer's routing-replay injector; the model is left resharded."""
    with trainer._routing_injector.capture(ids.size(0), ids.size(1)) as captured, torch.no_grad():
        trainer.model(input_ids=ids)
    reshard_fsdp2_modules(unwrap(trainer.model))
    return captured[0]


@contextlib.contextmanager
def perturbed_moe_inputs(model, relative: float):
    """Every EP MoE layer's input moved by ``relative`` of each element's own magnitude, the element's sign drawn
    from a generator seeded per layer, so every rank and every call moves it alike."""

    def hook_for(index: int):
        def hook(_module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            generator = torch.Generator(device=hidden.device).manual_seed(ROUTING_PROBE_SEED + index)
            signs = torch.randint(0, 2, hidden.shape, generator=generator, device=hidden.device).mul_(2).sub_(1)
            moved = (hidden.float() * (1 + relative * signs)).to(hidden.dtype)
            if args:
                return (moved, *args[1:]), kwargs
            return args, {**kwargs, "hidden_states": moved}

        return hook

    handles = [
        layer.register_forward_pre_hook(hook_for(index), with_kwargs=True)
        for index, layer in enumerate(ep_layers(model))
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def replayed_flip_rate(trainer, ids: torch.Tensor, mask: torch.Tensor, *, perturbation: float = 0.0) -> float:
    """COLLECTIVE. The world share of ``mask``'s forced selections a forward of ``ids`` picks differently, counted
    by the injector the training steps count with (``routing/replay_flip_rate``), under
    :func:`perturbed_moe_inputs` when ``perturbation`` is set; the model is left resharded."""
    injector = trainer._routing_injector
    injector.arm(mask)
    moved = perturbed_moe_inputs(trainer.model, perturbation) if perturbation else contextlib.nullcontext()
    try:
        with torch.no_grad(), moved:
            trainer.model(input_ids=ids)
    finally:
        injector.disarm()
    reshard_fsdp2_modules(unwrap(trainer.model))
    rate = injector.flip_rate()
    return math.nan if rate is None else rate


def misplaced_mask(mask: torch.Tensor) -> torch.Tensor:
    """``mask`` one token over: each position gets its predecessor's selection, the first replays nothing."""
    shifted = torch.full_like(mask, -1)
    shifted[:, 1:] = mask[:, :-1]
    return shifted


def routing_probe_round(trainer, ids: torch.Tensor, before_training: torch.Tensor) -> dict[str, float]:
    """COLLECTIVE. The flip rates the run's bar is measured from, all on ``ids``: the trained policy against its own
    routing with every MoE input moved by :data:`ROUTING_FLOOR_PERTURBATION` (``floor``) and one token over
    (``misplaced``); and, recorded, against the routing it gave before training (``stale``: what the trained steps
    moved, the mismatch a round sampled before an update replays)."""
    routing = capture_routing(trainer, ids)
    probe = {
        "floor": replayed_flip_rate(trainer, ids, routing, perturbation=ROUTING_FLOOR_PERTURBATION),
        "misplaced": replayed_flip_rate(trainer, ids, misplaced_mask(routing)),
        "stale": replayed_flip_rate(trainer, ids, before_training),
    }
    log(f"  routing probe on {ids.size(1)} tokens: {probe}")
    return probe


def routing_flip_bar(floor: float, misplaced: float) -> float:
    """Halfway between the noise floor and a misplaced mask's rate, in log space: as far from each in ratio."""
    return math.sqrt(floor * misplaced)


def record_routing_replay(
    trainer,
    checks: dict[str, bool],
    metrics: dict[str, float],
    *,
    probe: dict[str, float],
    acquired: list[bool],
) -> None:
    """The engine's own routed-expert ids reached the loss, lined up with the sampled tokens, and the trainer's
    router agrees with them as closely as bf16 noise allows.

    Reaching here already covers the capture wire: a batch in which no rollout returned ``routed_experts``
    raises rather than falling back. What it does not cover is placement: a mask the assembler cannot line up
    against the turn's tokens lands in ``unresolved``, and that turn replays nothing while the run continues.
    Each step's ``routing/replay_flip_rate`` is held to a bar measured on this policy (:func:`routing_flip_bar`
    over :func:`routing_probe_round`), since the near-tie density that sets the noise floor differs per family
    (top-8 of 128 flips several times more often than top-4 of 32) and no fixed rate fits. The bar catches a mask
    shifted along its row and a server routing with materially different router weights (half a weight std of
    noise on every router flips 2.8-3.2x the bar on both default checkpoints, a tenth of it 0.7-1.3x); a whole row's
    mask swapped with a sibling's reads near the floor, since the long chat-template prompt every row shares routes
    alike.

    The bar holds on every step whose round was sampled by the weights it trains (``acquired`` False, and every
    step of a run that does not prefetch); a popped round was sampled before the updates since its submission, so
    its flip rate is recorded beside ``stale``, not bounded.
    """
    history = trainer.state.log_history
    coverage = [
        {key: float(value) for key, value in entry.items() if key.startswith("routing/rollout_")}
        for entry in history
        if any(key.startswith("routing/rollout_") for key in entry)
    ]
    checks["routing_replay_masks_assembled"] = bool(coverage)
    checks["routing_replay_placed_every_mask"] = bool(coverage) and all(
        step["routing/rollout_unresolved_frac"] == 0.0 for step in coverage
    )
    flips = {
        int(entry["step"]): float(entry["routing/replay_flip_rate"])
        for entry in history
        if "routing/replay_flip_rate" in entry
    }
    popped = {step for step in flips if step <= len(acquired) and acquired[step - 1]}
    fresh = {step: flip for step, flip in flips.items() if step not in popped}
    bar = routing_flip_bar(probe["floor"], probe["misplaced"])
    checks["routing_probe_floor_flips_a_selection"] = probe["floor"] > 0
    checks["routing_probe_separates_noise_from_a_misplaced_mask"] = (
        probe["misplaced"] >= ROUTING_PROBE_SEPARATION * probe["floor"]
    )
    checks["routing_replay_flip_rate_within_the_measured_bar"] = 1 in fresh and all(
        flip <= bar for flip in fresh.values()
    )
    metrics |= {
        "routing_replay_flip_rate_max": max(fresh.values(), default=math.nan),
        "routing_replay_popped_flip_rate_max": max((flips[step] for step in popped), default=math.nan),
        "routing_flip_bar": bar,
        **{f"routing_probe_{name}": rate for name, rate in probe.items()},
    }
    log(
        f"  routing replay coverage per step: {coverage}; flip rate per step: {flips} (popped {sorted(popped)}); "
        f"bar {bar:.4f} from floor {probe['floor']:.4f} and misplaced {probe['misplaced']:.4f}"
    )


def record_prefetch_acquisitions(trainer) -> list[bool]:
    """Record, per training step, whether its acquire popped a prefetched round (``True``) or left the step to
    collect its own prompts (``False``: a cold round, or a popped failure)."""
    acquired: list[bool] = []
    acquire = trainer._acquire_prefetched_round

    def recording():
        results = acquire()
        acquired.append(results is not None)
        return results

    trainer._acquire_prefetched_round = recording
    return acquired


def record_prefetch(trainer, checks: dict[str, bool], metrics: dict[str, float], acquired: list[bool], depth: int):
    """The run prefetched at ``depth``: exactly its first ``depth`` steps collected their own prompts, every later
    step popped the round submitted ``depth`` steps earlier, and the IS log-ratio was scored, finite, on every step,
    the stale rounds included.

    A liveness check on the log-ratio, not a bound: at this row's ``LEARNING_RATE`` no fixed limit separates a stale
    round from a wrong one, so the band is the landing A/B's to enforce."""
    history = trainer.state.log_history
    hits = max((entry.get("async/prefetch_hits", 0) for entry in history), default=0)
    logratios = [float(entry[LOGRATIO_MEAN_KEY]) for entry in history if LOGRATIO_MEAN_KEY in entry]
    checks["prefetch_ran_at_depth"] = trainer._prefetch_enabled and trainer.async_config.prefetch_depth == depth
    checks["prefetch_opened_with_depth_cold_rounds_then_popped_every_step"] = (
        acquired == [False] * depth + [True] * PREFETCH_POPPED_STEPS
    )
    checks["prefetch_skipped_no_submission"] = not any("async/prefetch_input_skips" in entry for entry in history)
    checks["logratio_scored_every_step"] = len(logratios) == len(acquired) and all(map(math.isfinite, logratios))
    # A hit (the round ready before its step asks) depends on round versus update time: measured, not checked.
    metrics["prefetch_hits"] = float(hits)
    metrics["logratio_mean_max_abs"] = max((abs(v) for v in logratios), default=math.nan)
    log(f"  prefetch depth {depth}: popped per step {acquired}, {hits} hits; {LOGRATIO_MEAN_KEY}: {logratios}")


def record_grad_norms(trainer, checks: dict[str, bool], metrics: dict[str, float]) -> None:
    """Every step logged a finite gradient norm. On a chunked row each row is its own forward, so one backward
    recomputes every reentrant-checkpointed layer once per row; a recompute FSDP2 failed to reduce leaves a
    norm missing its contributions or a guard raising mid-step. Zero is a tied group's legitimate norm, so the
    spread across steps is recorded rather than bounded."""
    norms = [float(entry["grad_norm"]) for entry in trainer.state.log_history if "grad_norm" in entry]
    checks["grad_norms_logged_finite"] = bool(norms) and all(map(math.isfinite, norms))
    metrics["grad_norm_max"] = max_or_nan(norms, default=math.nan)
    metrics["grad_norm_min"] = min_or_nan(norms, default=math.nan)
    log(f"  grad norms per step: {norms}")


def record_first_step_gradient(trainer, checks: dict[str, bool], metrics: dict[str, float]) -> None:
    """Step 1 trained on a gradient: a group's rewards spread, and the step's gradient norm is finite and positive.

    Step 1's rollouts are the checkpoint's own on every row (a prefetching row's first round is cold), so only the
    task decides whether they tie; a tied step reduces nothing through the expert-parallel or tensor-parallel
    backward, and every later check would pass over an update that never ran.
    """
    first = next((entry for entry in trainer.state.log_history if "grad_norm" in entry), {})
    spread = float(first.get("reward/within_group_std", math.nan))
    norm = float(first.get("grad_norm", math.nan))
    checks["step_1_rewards_spread_within_a_group"] = spread > 0
    checks["step_1_gradient_norm_positive"] = math.isfinite(norm) and norm > 0
    metrics["step_1_reward_within_group_std"] = spread
    metrics["step_1_grad_norm"] = norm
    log(f"  step 1: reward within-group std {spread}, grad norm {norm}")


def run_env_grpo_e2e(
    ctx,
    *,
    backend: str,
    server_url: str,
    group_port: int,
    layout: PolicyLayout,
    peft: str | None = None,
    resume: bool = False,
    model_name: str = MODEL_NAME,
    thinking_budget: int | None = None,
    routing_replay: str = "none",
    environment_type: str | None = None,
    pool_urls: tuple[str, ...] = (),
    prefetch_depth: int | None = None,
    chunked: bool = False,
    length_ordered: bool | None = None,
) -> dict:
    """Train Environmental GRPO against a live ``backend`` server and assert the sync landed.

    ``model_name`` is the checkpoint both sides use, which the server must already serve. ``layout`` is the
    policy's parallel shape: its ``ep_size`` 1 and 2 are an A/B across the expert gather (DTensor experts in the
    FSDP2 shard vs FSDP-ignored plain tensors), and its data-parallel half (FULL_SHARD, whole ep1 experts, HSDP,
    several EP groups, fp32 masters) is read off the trained policy by :func:`record_dp_layout`.
    ``peft`` puts the run on an adapter fold; ``resume`` adds a fresh phase with one step left, whose
    train-begin push must land the engine on the checkpoint's policy; ``thinking_budget`` sets and grades
    ``rollout_max_thinking_tokens``. ``pool_urls`` (``server_url`` among them) pushes through a WeightSyncPool
    instead of one client, and every one of those servers must serve the moved policy. ``prefetch_depth`` turns
    ``enable_prefetch`` on at that depth, however many servers serve; unset leaves it off. ``chunked`` scores with
    chunked log-probs, one row per micro-batch, so each row is its own forward and the length deal reorders the
    round; ``length_ordered`` sets ``length_ordered_microbatches`` (unset: on wherever it changes only the order).
    The policy trains on :class:`DrawNumberEnvironment`, whose rewards spread on every policy, so step 1 must
    carry a gradient (:func:`record_first_step_gradient`); ``environment_type`` trains a registry environment on
    arithmetic rows graded by their answers instead (the one-process rows' ``react_math``), where a confident
    policy ties its groups and step 1's gradient goes unchecked.
    """
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    pre_perturb: dict[str, float] = {}
    expert_lora = peft == "expert_lora"
    max_steps = RESUME_MAX_STEPS if resume else MAX_STEPS
    if prefetch_depth is not None:
        if resume:
            raise ValueError("a prefetch row sets its own step count, which a resume row's save step does not fit")
        max_steps = prefetch_depth + PREFETCH_POPPED_STEPS

    # ── 0. expert TP refuses the native grouped expert adapters, at config time ───────────────
    # Before the load, which is the point of the refusal: it fires ahead of a multi-hundred-GB
    # checkpoint download rather than at group construction.
    if layout.expert_tp_size > 1:
        refusal = expert_lora_under_etp_refusal(layout.expert_tp_size)
        checks["expert_lora_under_expert_tp_refused"] = bool(refusal)
        checks["expert_lora_etp_refusal_names_expert_tp"] = "expert_tensor_parallel_size > 1" in refusal
        log(f"  expert-LoRA under expert_tp_size={layout.expert_tp_size} refusal: {refusal or '<NOT RAISED>'}")

    # ── 1. the server is up, serving the checkpoint we are about to train, and reproducible ───
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if ctx.rank == 0:
        log(f"  backend: {backend}")
        record_served_baseline(server_url, model_name, checks)
        if thinking_budget is not None:
            record_budget_acceptance(server_url, model_name, thinking_budget, checks)
    ctx.barrier()

    # ── 2. train ──────────────────────────────────────────────────────────────────────────────
    ensure_model_downloaded(model_name, ctx.rank)
    output_dir = shared_output_dir(ctx) if resume else ctx.output_dir
    environment, dataset = _environment_and_dataset(environment_type)
    async_config = _async_config(
        backend=backend,
        server_url=server_url,
        pool_urls=pool_urls,
        group_port=group_port,
        model_name=model_name,
        thinking_budget=thinking_budget,
        routing_replay=routing_replay,
        prefetch_depth=prefetch_depth,
        chunked=chunked,
        length_ordered=length_ordered,
        stop_tokens=[HARMONY_CALL_TOKEN] if HARMONY_CALL_TOKEN in tokenizer.get_vocab() else [],
    )
    grpo_config = _grpo_config(output_dir=output_dir, max_steps=max_steps, save=resume, peft=peft, chunked=chunked)

    parallelism_config = layout.parallelism_config()
    model, peft_config = load_policy(
        model_name, parallelism_config, peft, attn_implementation=ATTN_IMPL, lora_target_modules=LORA_TARGETS
    )

    trainer_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "dataset": dataset,
        "environment": environment,
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
    if peft is not None and layout.tp_size > 1:
        refusal = ""
        try:
            _make_trainer(**trainer_kwargs)
        except ValueError as e:
            refusal = str(e)
        checks["adapters_under_tp_refused"] = bool(refusal)
        # Matched by name rather than by exception type: every other construction-time refusal on
        # this path (weight sync, environment spec, batch shape) also raises ValueError.
        checks["tp_refusal_names_tensor_parallelism"] = "Tensor Parallelism (tensor_parallel_size > 1)" in refusal
        log(
            f"  adapters under tp_size={layout.tp_size} refusal: "
            f"{refusal.splitlines()[0] if refusal else '<NOT RAISED>'}"
        )
        return {"checks": ctx.broadcast_checks(checks), "metrics": metrics}

    # One finalizer over a mutable slot rather than a bound method per phase: a registered
    # ``trainer.cleanup_ep`` pins its trainer (and its model's GPU memory) for the whole run, which a
    # resume row cannot afford, since it builds a second one. The engine client goes first: the
    # forced pushes below re-form the weight-sync group outside ``train()``, and a group still
    # registered when the harness destroys every process group is finalized against an engine that
    # never enters destroy — under NCCL's cuMem transports that finalize waits forever.
    live: dict[str, object] = {"trainer": None}

    def _release() -> None:
        trainer = live["trainer"]
        if trainer is not None:
            trainer._cleanup_async_components()
            trainer.cleanup_ep()

    ctx.on_teardown(_release)

    trainer = _make_trainer(**trainer_kwargs)
    live["trainer"] = trainer
    if thinking_budget is not None and ctx.rank == 0 and checks["server_accepts_the_thinking_budget"]:
        # Recorded, not raised: rank 0 runs it alone while its peers wait in their first collective.
        enforcement = partial(record_budget_enforcement, server_url, model_name, thinking_budget, tokenizer, checks)
        trainer.budget_probe = partial(record_check, checks, "thinking_budget_probe_ran", enforcement)
    saved_state = RestorePointSnapshot("save", trainer, capture_optimizer=peft is not None)
    if resume:
        trainer.add_callback(saved_state)

    record_parallelism_engaged(ctx, trainer, layout, checks)
    log(f"  {layout} peft={peft} (ep_config={getattr(trainer._ep_config, 'ep_size', None)})")

    adapters_before: dict[str, torch.Tensor] = {}
    if peft is not None:
        # Inspected on the trainer's model: attention LoRA is applied by TRL's get_peft_model, so the
        # adapters only exist after construction.
        ok, detail = assert_only_adapters_trainable(unwrap(trainer.model))
        checks["only_adapters_trainable"] = ok
        log(f"  [trainable] {detail}")
        adapters_before = snapshot_adapters(unwrap(trainer.model), expert_lora=expert_lora)

    if chunked:
        # The deal resolves at construction; a chunked row runs the shape it reorders, so only false turns it off.
        checks["rows_dealt_by_length_as_set"] = trainer._orders_rows_by_length is (length_ordered is not False)
    acquired = record_prefetch_acquisitions(trainer) if prefetch_depth is not None else []
    if routing_replay == "rollout":
        # The probe row as the checkpoint routes it; the trained policy disagreeing with it is a real weight mismatch.
        probe_ids = routing_probe_ids(tokenizer, ctx.device)
        routing_before_training = capture_routing(trainer, probe_ids)
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
    if prefetch_depth is not None:
        record_prefetch(trainer, checks, metrics, acquired, prefetch_depth)
    if environment_type is None:
        record_first_step_gradient(trainer, checks, metrics)
    if chunked:
        record_grad_norms(trainer, checks, metrics)
    record_dp_layout(ctx, trainer, layout, checks)
    # Before the perturbation rounds below move the weights off the trained policy the probe measures.
    probe = routing_probe_round(trainer, probe_ids, routing_before_training) if routing_replay == "rollout" else {}

    # ── 3. the engine's own sampled ids reached the loss ──────────────────────────────────────
    # The trainer falls back to re-tokenization when a rollout returns no ids, and that fallback warns
    # rather than raising, so on a backend whose capture spelling is wrong training succeeds while
    # optimizing tokens the engine never sampled.
    checks["trained_on_engine_sampled_ids"] = (
        trainer.async_config.train_on_sampled_tokens and SAMPLED_IDS_MISSING_WARNING not in trainer._warned_once
    )

    if peft is not None:
        record_adapter_training(trainer, checks, before=adapters_before, expert_lora=expert_lora, trained=trained)

    # ── 4. a sync through the trainer's own client must reach the served policy ───────────────
    # On every server: a pool push whose chunks stop at the first leaves the rest serving a stale policy
    # with no error at all.
    servers = pool_urls or (server_url,)
    served: dict[str, dict[str, float]] = {}
    ctx.barrier()
    if ctx.rank == 0:
        served = {url: probe_top_logprobs(url, model_name) for url in servers}
        pre_perturb = served[server_url]
        checks["served_logprobs_finite"] = all(math.isfinite(v) for probe in served.values() for v in probe.values())
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
        after = {url: probe_top_logprobs(url, model_name) for url in servers}
        unmoved = [url for url in servers if not served_policy_moved(after[url], served[url])]
        checks["forced_sync_moved_the_served_policy"] = not unmoved
        if unmoved:
            log(f"  IDENTICAL or non-finite logprobs on {unmoved} after a {what} perturbation: the sync did not land")
        log(f"  post-forced-sync: { {k: round(v, 4) for k, v in after[server_url].items()} }")
        # A failed update leaves the engine partially written, and both engines' docs say to discard
        # such a server rather than keep serving from it, so check that it still answers.
        checks["server_usable_after_sync"] = bool(probe_top_logprobs(server_url, model_name))
        if pool_urls:
            clients = len(getattr(trainer._weight_sync_client, "_clients", ()))
            checks["pool_drove_one_client_per_server"] = trainer._multi_server_mode and clients == len(pool_urls)
    ctx.barrier()

    # ── 5. the expert stream alone must move the served policy ───────────────────────────────
    # Inert for a dense policy; see :func:`expert_round` for why the mixed round above cannot show it.
    expert_round(
        ctx,
        trainer.model,
        server_url=server_url,
        model_name=model_name,
        push=lambda: trainer._sync_weights_to_engine(force=True),
        checks=checks,
    )
    # ── 6. GptOss attention sinks, in a round of their own ────────────────────────────────────
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
        record_routing_replay(trainer, checks, metrics, probe=probe, acquired=acquired)

    if not resume:
        return {"checks": ctx.broadcast_checks(checks), "metrics": metrics}

    # ── 7. a resumed run's FIRST rollout must come from the checkpoint's weights ──────────────
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
        checks["engine_moved_off_the_checkpoint_policy"] = served_policy_moved(moved_off, pre_perturb)
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

    parallelism_config = layout.parallelism_config()
    # The production rule rather than a test-local one: under the EP/CP wrappers a full checkpoint's
    # weights load only at construction (the trainer's loader refuses to reload them), while an
    # adapter-only checkpoint keeps the base and has its adapters restored by the loader.
    weights_source = resolve_resume_weights_source(
        checkpoint, SimpleNamespace(model_name_or_path=model_name), parallelism_config
    )
    checks["resume_weights_source_resolved"] = weights_source == (model_name if peft is not None else checkpoint)
    log(f"  resume: policy weights from {weights_source}")

    model, peft_config = load_policy(
        weights_source, parallelism_config, peft, attn_implementation=ATTN_IMPL, lora_target_modules=LORA_TARGETS
    )
    trainer = _make_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        environment=environment,
        parallelism_config=parallelism_config,
        peft_config=peft_config,
        # The resumed phase writes nothing: it is judged on the served policy and on what it
        # restored, and a second full-model checkpoint would only cost scratch space.
        grpo_config=_grpo_config(output_dir=output_dir, max_steps=max_steps, save=False, peft=peft, chunked=chunked),
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
        checks["resume_left_the_stale_engine_state"] = served_policy_moved(after_resume, moved_off)
        to_checkpoint = served_policy_delta(after_resume, pre_perturb)
        # Both pushes send the same tensors, so the two probes must agree exactly on every row.
        checks["resumed_rollouts_served_the_checkpoint_policy"] = after_resume == pre_perturb
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
