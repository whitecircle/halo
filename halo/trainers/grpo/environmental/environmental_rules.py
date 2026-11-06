"""Construction refusals of the environmental GRPO trainer: configs the run would honor silently wrong."""

import dataclasses
import logging
import math
from collections.abc import Mapping

from halo.configs.grpo_environmental import ReasoningCostArguments
from halo.environments.episode_keys import ANSWER_KEY
from halo.environments.reasoning_effort import REASONING_BUDGET_KEY, REASONING_PRICE_KEY
from halo.environments.rollout_config import (
    IDENTITY_SAMPLER,
    REASONING_END_TOKEN_EXAMPLES,
    SGLANG_BACKEND,
    VLLM_BACKEND,
)
from halo.models.moe.family_specs import find_moe_family_spec_for_config
from halo.runtime.ranks import is_global_main_process
from halo.trainers.training_args_overrides import uses_evaluation

logger = logging.getLogger(__name__)

# TRL sampling knobs no sampler reads here: the environment actors generate from RolloutConfig. ``temperature``
# is reconciled the other way round instead: the trainer scores at the sampling temperature.
_ROLLOUT_OWNED_SAMPLING_KNOBS = ("top_p", "top_k", "min_p", "repetition_penalty", "generation_kwargs")
# The TRL vllm_importance_sampling_mode that names the environmental correction (each token's ratio truncated
# from above); it and TRL's default pass.
_ENV_IS_MODE = "token_truncate"


def reject_unverified_carried_reasoning(env, rollout_backend: str) -> None:
    """SGLang's handling of an assistant message's ``reasoning_content`` is unverified: a dropped thought would train
    rows on context the engine never saw."""
    if env.carry_reasoning and rollout_backend == SGLANG_BACKEND:
        raise ValueError(
            "carry_reasoning: true with rollout_backend: 'sglang' is not supported: whether SGLang's chat "
            "schema forwards an assistant message's reasoning_content to the chat template is unverified, "
            "so the carried thought could be dropped without a trace. Serve rollouts with vLLM, or turn "
            "carry_reasoning off."
        )


def reject_straddling_train_groups(
    per_device_train_batch_size: int, steps_per_generation: int, num_generations: int
) -> None:
    """Advantages are grouped per rank here, while TRL only validates the global ``generation_batch_size %
    num_generations``."""
    per_rank_rows = per_device_train_batch_size * steps_per_generation
    if per_rank_rows % num_generations != 0:
        raise ValueError(
            f"per_device_train_batch_size * steps_per_generation ({per_rank_rows}) must be divisible "
            f"by num_generations ({num_generations}): environmental GRPO groups advantages "
            f"per rank, so a group cannot straddle ranks."
        )


def reject_straddling_eval_groups(args, async_config, num_generations_eval: int) -> None:
    """A per-rank eval round must not split a generation group (TRL validates only the global eval batch), and
    ``eval_rollout_batch_size`` under ``dataloader_drop_last`` would drop the whole eval."""
    if not uses_evaluation(args):
        return
    rows = async_config.eval_rollout_batch_size
    if rows is None:
        per_rank = args.per_device_eval_batch_size
        if per_rank % num_generations_eval != 0:
            raise ValueError(
                f"per_device_eval_batch_size ({per_rank}) must be divisible by num_generations_eval "
                f"({num_generations_eval}): environmental GRPO groups advantages per rank, so an "
                f"eval group cannot straddle rank batches (or set eval_rollout_batch_size to a multiple)."
            )
        return
    if rows % num_generations_eval != 0:
        raise ValueError(
            f"eval_rollout_batch_size ({rows}) must be divisible by num_generations_eval "
            f"({num_generations_eval}): environmental GRPO groups advantages per rank."
        )
    if args.dataloader_drop_last:
        # Accelerate's shard yields a round only once every rank holds a full batch; a round wider
        # than the split's share leaves fewer batches than ranks and an eval that scores nothing.
        raise ValueError("eval_rollout_batch_size needs dataloader_drop_last=false: a dropped tail is the whole eval.")


def reject_answerless_datasets(env, train_dataset, eval_dataset) -> None:
    """Such a run would score one constant, every group at zero advantage. A dataset declaring no columns is warned."""
    if not env.requires_answer:
        return
    datasets = {}
    if train_dataset is not None:
        datasets["train_dataset"] = train_dataset
    if isinstance(eval_dataset, dict):
        datasets.update({f"eval_dataset[{name!r}]": split for name, split in eval_dataset.items()})
    elif eval_dataset is not None:
        datasets["eval_dataset"] = eval_dataset
    for name, dataset in datasets.items():
        columns = getattr(dataset, "column_names", None)
        if columns is None:
            if is_global_main_process():
                logger.warning(
                    f"{name} declares no columns, so the {ANSWER_KEY!r} column "
                    f"{type(env).__name__} grades against cannot be checked here."
                )
        elif ANSWER_KEY not in columns:
            raise ValueError(
                f"{type(env).__name__} grades each episode against the dataset's "
                f"{ANSWER_KEY!r} column (requires_answer), but {name} carries {sorted(columns)}. Point "
                f"answer_field at the column holding the expected answer — it is carried into the "
                f"row as {ANSWER_KEY!r} — or train this data on an environment that grades without one."
            )


def reject_inert_reasoning_terms(
    async_config, env, level_terms: Mapping[str | None, tuple[int | None, float]]
) -> None:
    """Refuse reasoning options that would charge nothing or charge every episode: a profile key, cap or failed-group
    switch that no drawn level reads, a floor without a budget or a priced success, a profile key among the template
    variables the model reads. Each error names its case."""
    price_on = any(price > 0 for _, price in level_terms.values())
    floor_on = async_config.reasoning_floor > 0
    if env.reasoning_effort is None:
        set_keys = sorted(
            {key for profile in env.reasoning_effort_profiles.values() for key in profile}
            & {REASONING_BUDGET_KEY, REASONING_PRICE_KEY}
        )
        if set_keys:
            raise ValueError(
                f"reasoning_effort_profiles set {set_keys} but the environment's reasoning_effort is unset, so no "
                "episode carries a level to read them at."
            )
    if not price_on and async_config.reasoning_price_cap != ReasoningCostArguments.reasoning_price_cap:
        raise ValueError(
            f"reasoning_price_cap set while no level the environment draws sets {REASONING_PRICE_KEY} in its effort "
            "profile: nothing reads it. Remove it, or price a level."
        )
    if not (price_on or floor_on) and async_config.reasoning_terms_in_failed_groups:
        raise ValueError(
            "reasoning_terms_in_failed_groups set while no drawn level prices reasoning and reasoning_floor is off: it "
            "chooses where the reasoning terms charge. Turn one on, or remove it."
        )
    if floor_on and env.reasoning_effort is None:
        raise ValueError(
            f"reasoning_floor charges a failed episode its shortfall under its level's {REASONING_BUDGET_KEY}, but "
            "the environment's reasoning_effort is unset, so no episode carries a level."
        )
    if floor_on:
        unbudgeted = sorted(level for level, (budget, _) in level_terms.items() if not budget)
        if unbudgeted:
            raise ValueError(
                f"reasoning_floor charges a failed episode its shortfall under its level's {REASONING_BUDGET_KEY}, but "
                f"the drawn level(s) {unbudgeted} set none above 0 in reasoning_effort_profiles."
            )
    unread = sorted(
        level for level, (budget, price) in level_terms.items() if budget is not None and price == 0 and not floor_on
    )
    if unread:
        raise ValueError(
            f"the drawn level(s) {unread} set {REASONING_BUDGET_KEY} with no {REASONING_PRICE_KEY} while "
            "reasoning_floor is off: nothing reads it. Price the level, turn the floor on, or remove it."
        )
    misplaced = sorted({REASONING_BUDGET_KEY, REASONING_PRICE_KEY} & set(async_config.rollout_chat_template_kwargs))
    if misplaced:
        raise ValueError(
            f"rollout_chat_template_kwargs carries {misplaced}: they are effort-profile keys "
            "(environment_kwargs.reasoning_effort_profiles), and a template variable is text the model may read."
        )
    objective = env.objective_term if price_on or floor_on else None
    unpriced_success = objective is None or objective.weight <= 0
    if floor_on and unpriced_success:
        raise ValueError(
            "reasoning_floor charges an episode that earned no objective, but the reward prices no success above zero "
            "(no environment term, or one of weight <= 0), so it would charge every episode: give the environment "
            "term a positive weight, or turn the floor off."
        )
    if price_on and not async_config.reasoning_terms_in_failed_groups and unpriced_success:
        raise ValueError(
            "the reasoning terms are waived in a training group no episode of which earned the objective, but the "
            "reward prices no success above zero (no environment term, or one of weight <= 0), so every group would "
            "read as failed and the terms would charge no training episode: give the environment term a positive "
            "weight, or set reasoning_terms_in_failed_groups: true."
        )


def reject_off_policy_mask_threshold(args) -> None:
    """It masks on ``sampling_per_token_logps``, a key this trainer's batch never carries, so it thresholds a KL of
    0."""
    if args.off_policy_mask_threshold is not None:
        raise ValueError(
            f"off_policy_mask_threshold={args.off_policy_mask_threshold} is a silent no-op in environmental "
            "GRPO: TRL's mask reads sampling_per_token_logps, which this trainer's batch never carries, so it "
            "thresholds a KL of exactly 0. Use isr_opsm_delta (DeepSeek-V3.2 off-policy sequence masking "
            "against the engine's sampling log-probs) instead."
        )


def reject_vespo_loss(args) -> None:
    """Its weight sums each row's log IS ratios, so one token this trainer zeroes silently drops the whole row. TRL's
    own check never runs here: ``use_vllm`` is still off when its constructor reads it."""
    if args.loss_type == "vespo":
        raise ValueError(
            "loss_type='vespo' is not supported by environmental GRPO: VESPO weighs each turn row by the sum of its "
            "per-token log IS ratios instead of multiplying each token by its own, so whenever the IS correction "
            "masks a token (isr_*) or the engine forces a reasoning close, this trainer sets that token's ratio to 0 "
            "and the one token silently drops its whole row (weight ~1e-15). Use a per-token loss: dapo (the "
            "default), dr_grpo, cispo or grpo."
        )


def reject_unneutralized_forced_closes(forced_close_ids: tuple[int, ...] | None, is_correction: bool) -> None:
    """Forced closes would train with the episode's advantage and teach the model to stop closing its reasoning."""
    if forced_close_ids is not None and not is_correction:
        raise ValueError(
            "a vLLM reasoning cap can be enforced but the importance-sampling correction is off "
            "(train_on_sampled_tokens + vllm_importance_sampling_correction): the reasoning closes the "
            "engine forces at the cap would train with the episode's advantage."
        )


def reject_unresolved_forced_closes(
    capped: bool, reasoning_end_ids: tuple[int, ...] | None, backend: str, marker: str
) -> None:
    """vLLM forces its parser's end string at a thinking cap. A marker that resolves no ids cannot find those closes, so
    they would train with the episode's advantage, with one warning at startup."""
    if capped and reasoning_end_ids is None and backend == VLLM_BACKEND:
        raise ValueError(
            f"rollout_reasoning_end_token {marker!r} is not a marker this tokenizer writes, and a vLLM thinking cap can "
            f"bind: the closes the engine forces would stay in the policy loss and teach the model to stop closing "
            f"its reasoning. Name the end string the server's reasoning parser forces ({REASONING_END_TOKEN_EXAMPLES}), "
            f"or run with no cap: unset rollout_max_thinking_tokens and rollout_max_answer_tokens."
        )


def reject_open_ended_tool_turns(env, model, tokenizer, stop_token_ids: list[int] | None) -> None:
    """A family whose tool-call terminator is no eos runs a turn past its call unless a request stops on it: the model
    writes the tool's result itself and plays the whole episode in one turn, with no call ever executed."""
    if not env.get_tools_schema() and not env.TOOLS_ARRIVE_ON_RESET:
        return
    spec = find_moe_family_spec_for_config(getattr(model, "config", None))
    terminator = spec.tool_call_terminator if spec is not None else None
    if terminator is None:
        return
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    eos = [eos] if isinstance(eos, int) else list(eos or ())
    if tokenizer.convert_tokens_to_ids(terminator) not in {*eos, *(stop_token_ids or ())}:
        raise ValueError(
            f"{spec.model_types[0]} ends a tool call on {terminator!r}, which is no eos of this checkpoint and not in "
            f"rollout_stop_tokens, so a turn would not stop at its call: the model writes the tool's result itself and "
            f'plays the episode in one turn. Set rollout_stop_tokens: ["{terminator}"].'
        )


def reject_inert_isr_stages(isr_mask_config, is_correction: bool, skip_update_masked_frac: float | None) -> None:
    if isr_mask_config.any_stage_active and not is_correction:
        raise ValueError(
            "isr_geo_band/isr_veto/isr_opsm knobs require the importance-sampling "
            "correction (train_on_sampled_tokens + vllm_importance_sampling_correction) — "
            "without it they would silently do nothing."
        )
    if skip_update_masked_frac is not None and not (is_correction and isr_mask_config.any_stage_active):
        raise ValueError(
            "skip_update_masked_frac requires the importance-sampling correction AND at "
            "least one IS mask stage (isr_geo_band/isr_veto/isr_opsm) — without them no "
            "trajectory is ever masked and the circuit breaker would silently never fire."
        )


def reject_unsupported_engine_reference(client_cls, async_config, is_correction: bool) -> None:
    """``isr_engine_reference`` needs the IS correction's sampling log-probs, an engine that re-scores a given
    sequence, and the identity sampler, the only one whose recorded log-probs are the prefill's distribution."""
    if not is_correction:
        raise ValueError(
            "isr_engine_reference requires the importance-sampling correction "
            "(train_on_sampled_tokens + vllm_importance_sampling_correction): the engine re-score "
            "is compared against the sampling log-probs that correction captures."
        )
    if not client_cls.SUPPORTS_ENGINE_RESCORE:
        raise ValueError(
            f"isr_engine_reference is not available on {client_cls.BACKEND_NAME}: its client declares "
            "no route that returns per-token log-probs of a given sequence under the current weights."
        )
    sources = async_config.rollout_field_sources()
    identity = {sources[name]: value for name, value in IDENTITY_SAMPLER.items()}
    moved = {name: current for name, value in identity.items() if (current := getattr(async_config, name)) != value}
    if moved:
        raise ValueError(
            "isr_engine_reference requires the identity sampler "
            f"({', '.join(f'{name} {value}' for name, value in identity.items())}), got {moved}: the engine "
            "echoes the raw distribution's log-probs while the sampling log-probs are the processed "
            "ones, so only at the identity sampler do the two share a reference."
        )


def reject_prefetch_past_retry_budget(client_cls, async_config, args) -> None:
    """On an engine whose pause aborts turns in flight, a turn aborted ``max_retries + 1`` times fails its episode, and
    a round stays in flight for ``prefetch_depth`` rounds of ``steps_per_generation × num_iterations``
    micro-batches."""
    if not client_cls.PAUSE_ABORTS_IN_FLIGHT or not async_config.prefetch_active():
        return
    depth, every = async_config.prefetch_depth, async_config.sync_weights_every_n_steps
    round_micro_batches = args.steps_per_generation * args.num_iterations
    syncs = math.ceil(depth * round_micro_batches / (args.gradient_accumulation_steps * every))
    # An eval pushes off the cadence; at every step's push it takes that step's place and adds none.
    if every > 1 and args.eval_strategy not in ("no", None):
        syncs += 1
    if syncs > async_config.max_retries:
        raise ValueError(
            f"Every {client_cls.BACKEND_NAME} weight sync aborts the turns in flight, and at prefetch_depth={depth} a "
            f"prefetched round stays in flight across up to {syncs} syncs (rounds of steps_per_generation × "
            f"num_iterations = {round_micro_batches} micro-batches, gradient_accumulation_steps="
            f"{args.gradient_accumulation_steps}, sync_weights_every_n_steps={every}, and one for an eval between "
            f"cadence pushes): a long turn aborted at each of "
            f"them outruns its max_retries={async_config.max_retries} re-issues and fails its episode. Shorten the "
            "span (it grows with prefetch_depth, steps_per_generation and num_iterations, falls as "
            "sync_weights_every_n_steps rises, and is never below 1), raise max_retries to "
            f"{syncs} with (1 + max_retries) × request_timeout still within episode_timeout, or turn prefetch off "
            "(enable_prefetch: false at prefetch_depth: 1)."
        )


def reject_unread_grpo_knobs(args) -> None:
    """A GRPOConfig knob this trainer does not honor, set away from its default: a sampling knob, TRL's weight-sync
    group port, the IS lower clip, an IS mode other than :data:`_ENV_IS_MODE`."""
    defaults = {f.name: f.default for f in dataclasses.fields(type(args))}

    def changed(names: tuple[str, ...]) -> list[str]:
        return [name for name in names if name in defaults and getattr(args, name) != defaults[name]]

    sampling = changed(_ROLLOUT_OWNED_SAMPLING_KNOBS)
    if sampling:
        raise ValueError(
            f"GRPOConfig {sampling} reach no sampler under environmental GRPO: the environment actors "
            "generate from the rollout config. Remove them and set the rollout_* equivalents instead "
            "(rollout_top_p, rollout_top_k, rollout_min_p, rollout_repetition_penalty)."
        )
    mode = args.vllm_importance_sampling_mode
    if mode not in (defaults["vllm_importance_sampling_mode"], _ENV_IS_MODE):
        raise ValueError(
            f"GRPOConfig vllm_importance_sampling_mode={mode!r} is not the correction environmental GRPO "
            f"applies: it truncates each token's ratio from above at vllm_importance_sampling_clip_max, TRL's "
            f"{_ENV_IS_MODE!r}. Set that mode or leave the default; its masks are the isr_* knobs."
        )
    if changed(("vllm_group_port",)):
        raise ValueError(
            "GRPOConfig 'vllm_group_port' is not read by environmental GRPO, whose weight-sync group port is "
            "AsyncRolloutConfig's rollout_group_port on either backend. Set rollout_group_port instead."
        )
    if changed(("vllm_importance_sampling_clip_min",)):
        raise ValueError(
            "GRPOConfig 'vllm_importance_sampling_clip_min' has no effect under environmental GRPO: its "
            "importance-sampling correction truncates each token's ratio from above only, at "
            "vllm_importance_sampling_clip_max. Remove it; its masks are the isr_* knobs."
        )


def reject_unsupported_routing_replay(
    mode: str, *, train_on_sampled_tokens: bool, runs_logprob_recompute: bool
) -> None:
    """Refuse ``rollout`` replay without the sampled ids its masks align with, and ``recompute`` in a config
    whose recompute pass never runs to capture one."""
    if mode == "rollout":
        if not train_on_sampled_tokens:
            raise ValueError(
                "routing_replay='rollout' requires train_on_sampled_tokens: the engine's masks "
                "align with the sampled token ids, not a chat-template re-render."
            )
    elif mode == "recompute" and not runs_logprob_recompute:
        raise ValueError(
            "routing_replay='recompute' captures the mask in the no-grad logprob-recompute pass, "
            "which this config never runs (num_iterations=1, aligned accumulation, IS correction "
            "off, beta=0). Enable the IS correction (or beta/num_iterations>1) or set "
            "routing_replay='none'."
        )
