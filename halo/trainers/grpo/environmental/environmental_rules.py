"""Construction refusals of the environmental GRPO trainer: configs the run would honor silently wrong."""

import dataclasses
import logging
import math

from halo.data.collators.collator_rules import reject_multi_document_rows
from halo.distributed.expert_parallel.deepep.capacity import (
    DEEPEP_INDEX_LIMIT,
    EP_CAPACITY_ALIGN,
    GIN_MAX_TOKENS_PER_RANK,
    ep_dispatch_capacity,
    padded_wire_hidden,
    reject_oversized_dispatch,
)
from halo.environments.episode_keys import ANSWER_KEY
from halo.environments.rollout_config import (
    IDENTITY_SAMPLER,
    REASONING_END_TOKEN_EXAMPLES,
    SGLANG_BACKEND,
    VALID_REASONING_EFFORTS,
    VLLM_BACKEND,
)
from halo.models.attention.gpt_oss_sinks import has_live_attention_sinks
from halo.models.attention.layout import has_mrope_rotary
from halo.models.attention.selection import is_flash_attn_implementation
from halo.models.config_access import text_config
from halo.models.moe.config_fields import resolve_dispatch_width, resolve_router_topk
from halo.models.moe.family_specs import attention_reads_row_positions, find_moe_family_spec_for_config
from halo.runtime.ranks import is_global_main_process
from halo.trainers.grpo.environmental.row_order import MICROBATCH_INVARIANT_LOSS_TYPES
from halo.trainers.training_args_overrides import uses_evaluation

logger = logging.getLogger(__name__)

# TRL sampling knobs no sampler reads here: the environment actors generate from RolloutConfig. ``temperature``
# is reconciled the other way round instead: the trainer scores at the sampling temperature.
_ROLLOUT_OWNED_SAMPLING_KNOBS = ("top_p", "top_k", "min_p", "repetition_penalty", "generation_kwargs")
# The TRL vllm_importance_sampling_mode that names the environmental correction (each token's ratio truncated
# from above); it and TRL's default pass.
_ENV_IS_MODE = "token_truncate"
# Whether packed micro-batches run on M-RoPE decoders (Qwen3.5/3.6), whose flattened rows faulted the flash
# varlen kernels under ``padding_free``. Set by the verdict of tests/gpu/kernels/test_packed_rl_forward_mrope.py; False
# refuses the knob on them and leaves them one backbone call per row.
PACKED_MICROBATCH_ADMITS_MROPE = True


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


def reject_inert_reasoning_terms(async_config, env, *, budgeted: bool, every_budget_soft: bool) -> None:
    """A price with no level to read charges nothing, a floor where no episode carries a thinking cap never fires, and
    a price past a budget prices nothing unless every drawable budget is soft. A term waived in groups that earned no
    objective needs a reward that prices success above zero, or it charges no training episode."""
    terms_on = async_config.reasoning_price is not None or async_config.reasoning_floor > 0
    objective = env.objective_term if terms_on else None
    if terms_on and not async_config.reasoning_terms_in_failed_groups and (objective is None or objective.weight <= 0):
        raise ValueError(
            "reasoning_price and reasoning_floor are waived in a training group no episode of which earned the "
            "objective, but the reward prices no success above zero (no environment term, or one of weight <= 0), "
            "so every group would read as failed and the terms would charge no training episode: give the "
            "environment term a positive weight, or set reasoning_terms_in_failed_groups: true."
        )
    if async_config.reasoning_price is not None:
        if env.reasoning_effort is None:
            raise ValueError(
                "reasoning_price is set but the environment's reasoning_effort is unset, so no episode carries a "
                "level to be priced at."
            )
        if set(async_config.reasoning_price) != set(VALID_REASONING_EFFORTS):
            levels, admitted = set(async_config.reasoning_price), set(VALID_REASONING_EFFORTS)
            raise ValueError(
                f"reasoning_price must map exactly the effort levels {sorted(admitted)}; "
                f"unknown {sorted(levels - admitted)}, missing {sorted(admitted - levels)}"
            )
    if async_config.reasoning_floor > 0 and not budgeted:
        raise ValueError(
            "reasoning_floor prices an episode against its per-turn thinking cap, but no episode of this run "
            "carries one: no drawable effort level sets thinking_tokens and rollout_max_thinking_tokens is unset."
        )
    if async_config.reasoning_price_past_budget and not every_budget_soft:
        raise ValueError(
            "reasoning_price_past_budget prices reasoning past a soft per-turn budget, but a drawable effort level's "
            "budget is enforced or unset, so its turns would go unpriced: set thinking_tokens and "
            "enforce_thinking_cap: false on every level the environment draws, or price every reasoning token."
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
            "a vLLM thinking budget can be enforced but the importance-sampling correction is off "
            "(train_on_sampled_tokens + vllm_importance_sampling_correction): the reasoning closes the "
            "engine forces at the budget would train with the episode's advantage."
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
            f"or run with no cap: unset rollout_max_thinking_tokens and rollout_max_answer_tokens, and every level's "
            f"enforced thinking_tokens."
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


def reject_unpackable_options(args, async_config, *, misaligned_accumulation: bool) -> None:
    """Refuse the options under which packed micro-batches would change the objective or leave their rows unscored by
    the packed branch: every one is named with its remedy, none falls back to the per-row path."""
    knob = "packed_microbatch_tokens"
    if not async_config.use_chunked_grpo_logprobs:
        raise ValueError(
            f"{knob} needs use_chunked_grpo_logprobs: true: a packed micro-batch would otherwise materialize a "
            f"[tokens, vocab] logits plane, and TRL's full-logits scorer does not take packed rows."
        )
    if args.per_device_train_batch_size != 1:
        raise ValueError(
            f"{knob} needs per_device_train_batch_size: 1, got {args.per_device_train_batch_size}: the plan, not a "
            f"batch size, picks the rows of a micro-batch. Raise gradient_accumulation_steps or {knob} instead."
        )
    if args.loss_type not in MICROBATCH_INVARIANT_LOSS_TYPES:
        raise ValueError(
            f"{knob} is refused with loss_type={args.loss_type!r}: bnpo normalizes each micro-batch by its own token "
            f"count, so the objective would depend on which rows share a bin, and vespo is refused by this trainer. "
            f"Use one of {list(MICROBATCH_INVARIANT_LOSS_TYPES)} (grpo or dapo)."
        )
    if args.top_entropy_quantile < 1.0:
        raise ValueError(
            f"{knob} is refused with top_entropy_quantile={args.top_entropy_quantile}: the entropy threshold is taken "
            f"over each micro-batch's tokens, which a packed bin would widen from one row to several. Set it to 1.0."
        )
    if async_config.length_ordered_microbatches:
        raise ValueError(
            f"{knob} replaces the length-ordered deal (length_ordered_microbatches: true): the plan already orders "
            f"the round's rows. Unset length_ordered_microbatches."
        )
    if misaligned_accumulation:
        raise ValueError(
            f"{knob} maps a generation round onto the optimizer step's window, but gradient_accumulation_steps "
            f"({args.gradient_accumulation_steps}) is not a multiple of steps_per_generation × num_iterations "
            f"({args.steps_per_generation} × {args.num_iterations}). Align them."
        )
    if uses_evaluation(args) and args.per_device_eval_batch_size != 1:
        raise ValueError(
            f"{knob} keeps evaluation rounds on the per-row path, which needs per_device_eval_batch_size: 1, got "
            f"{args.per_device_eval_batch_size}: a padded multi-row evaluation forward would run left-padded RoPE "
            f"positions."
        )


def reject_unpackable_model(model, attn_impl: str | None) -> None:
    """Refuse a model or backend whose packed row would cross documents silently, change the objective, or lose
    attention sinks. No opt-in: an RL step that mixes documents corrupts the policy gradient without a trace."""
    config = model.config
    knob = "packed_microbatch_tokens"
    if has_mrope_rotary(config) and not PACKED_MICROBATCH_ADMITS_MROPE:
        raise ValueError(
            f"{knob} is refused for model_type={getattr(config, 'model_type', None)!r}: a flattened M-RoPE row faults "
            f"the flash varlen kernels on this build (the packed-row gate, tests/gpu/kernels/"
            f"test_packed_rl_forward_mrope.py). Unset {knob}: these rows keep one backbone call each."
        )
    if attention_reads_row_positions(config):
        raise ValueError(
            f"{knob} is refused for model_type={getattr(config, 'model_type', None)!r}: its attention reads a token's "
            f"row index, which a flattened row changes. Unset {knob}."
        )
    if not is_flash_attn_implementation(attn_impl):
        raise ValueError(
            f"{knob} needs a varlen flash attention backend, but this model resolved to attn_implementation="
            f"{attn_impl!r}: a dense kernel builds an [N, N] mask over the flattened micro-batch and, for several "
            f"families, attends across its rows. Use flash_attention_2/3/4 or unset {knob}."
        )
    reject_multi_document_rows(
        config,
        "packing",
        attn_impl=attn_impl,
        allow_cross_document_linear_attention=False,
        keeps_packed_rows=False,
        varlen_rows=True,
    )
    if has_live_attention_sinks(model) and attn_impl != "flash_attention_4":
        raise ValueError(
            f"{knob} with live attention sinks needs flash_attention_4 (its varlen entry carries them), but the "
            f"model resolved to {attn_impl!r}, which drops them. Unset full_determinism, or set reset_sinks: true, or "
            f"unset {knob}."
        )


def _dispatch_shape(parallelism_config, model_config) -> tuple[int, int] | None:
    """``(num_topk, padded wire hidden)`` of the run's EP dispatch, ``None`` without one."""
    if parallelism_config.ep_size <= 1:
        return None
    text = text_config(model_config)
    width, num_topk = resolve_dispatch_width(text), resolve_router_topk(text)
    return (num_topk, padded_wire_hidden(width)) if width and num_topk else None


def resolve_dispatch_ceiling(parallelism_config, model_config) -> int | None:
    """Tokens one EP dispatch can carry per rank on this run (the 32-bit wire index, and the proxy-GIN ceiling on a
    group spanning NVLink domains), rounded down to the arena's block; ``None`` where there is no dispatch."""
    shape = _dispatch_shape(parallelism_config, model_config)
    if shape is None:
        return None
    limits = [(DEEPEP_INDEX_LIMIT - 1) // (shape[0] * shape[1])]
    if parallelism_config.requires_rdma and GIN_MAX_TOKENS_PER_RANK:
        limits.append(GIN_MAX_TOKENS_PER_RANK)
    return min(limits) // EP_CAPACITY_ALIGN * EP_CAPACITY_ALIGN


def reject_undispatchable_budget(parallelism_config, model_config, budget: int) -> None:
    """Refuse a ``packed_microbatch_tokens`` whose arena the first dispatch would refuse, before the first rollout."""
    shape = _dispatch_shape(parallelism_config, model_config)
    if shape is None:
        return
    try:
        reject_oversized_dispatch(
            ep_dispatch_capacity(budget),
            num_topk=shape[0],
            padded_hidden=shape[1],
            spans_domains=parallelism_config.requires_rdma,
        )
    except ValueError as exc:
        raise ValueError(f"packed_microbatch_tokens={budget} cannot be dispatched: lower it. {exc}") from exc
