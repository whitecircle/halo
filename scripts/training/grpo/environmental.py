#!/usr/bin/env python
"""Environmental GRPO — multi-turn RL with async Ray rollouts against a separate rollout server.

The single environmental-GRPO training entry point. Resolves the environment from the registry via
``environment_type`` in the YAML config; ``get_registered_environments()`` is the roster (register
your own with ``register_environment``). Runs on ``DistributedAsyncEnvironmentalGRPOTrainer`` with
Expert / Tensor / Expert-Tensor Parallelism; launch with ``torchrun``. Context Parallelism is not
supported (``logits_to_keep`` and global log-probability sums are incompatible with sequence
splitting), and neither is Pipeline Parallelism.

The rollout server runs in its own container, never in the training process:
``docker compose -f docker-compose.vllm.yml up vllm-server``, or ``docker-compose.sglang.yml`` for
``rollout_backend: sglang`` (the families its loaders take an update for, expert distribution
included — agent-docs/infrastructure/rollout-servers.md). Each example config's header
lists the server flags that config needs: the ``--moe-backend triton`` gate, the tool-call and
reasoning parsers, and the GPU split between trainer and server.

Usage (each config's header carries its own exact launch line):
    # Plain data-parallel, 4 train GPUs
    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \\
        scripts/training/grpo/environmental.py \\
        examples/grpo/environmental/gptoss/vllm/gptoss-20b-code-contests-full-ep1.yaml

    # Expert parallelism (MoE), 4 train GPUs, one dispatch group
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \\
        scripts/training/grpo/environmental.py \\
        examples/grpo/environmental/qwen3_5/vllm/qwen3.6-35b-a3b-react-math-full-ep4.yaml

    # SGLang rollouts
    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \\
        scripts/training/grpo/environmental.py \\
        examples/grpo/environmental/gptoss/sglang/gptoss-20b-code-contests-full-ep1.yaml

See agent-docs/training-methods/async-grpo/README.md for the objective, the environment roster and
the rollout-server contract.
"""

import logging

from trl import GRPOConfig

from halo.config_validation import reject_unsupported_args
from halo.configs.grpo_environmental import AsyncRolloutConfig, EnvironmentalGRPOScriptArguments, EnvironmentConfig
from halo.configs.parallelism_arguments import ParallelismArguments
from halo.configs.script_arguments import DEFAULT_ANSWER_FIELD, ModelArguments
from halo.data.chat_render import render_generation_prompt
from halo.data.conversation import as_conversation
from halo.data.pipeline.coordinated_map import coordinated_map_and_filter
from halo.data.pipeline.rejected_rows import blank_conversation
from halo.data.render_columns import missing_render_column_splits, reject_missing_render_column
from halo.data.vlm_detection import reject_image_columns
from halo.entrypoints.data_setup import load_script_datasets, log_script_dataset_examples
from halo.entrypoints.model_setup import (
    apply_tokenizer_window,
    grpo_row_attn_implementation,
    load_export_processor,
    load_model_and_tokenizer,
    log_model_info,
)
from halo.entrypoints.script import (
    configure_balancing_and_build_callbacks,
    distributed_trainer_kwargs,
    init_training_script,
    parse_script_args,
    run_trainer,
    verify_backend_on_rank0,
    with_distributed_teardown,
)
from halo.environments.episode_keys import ANSWER_KEY
from halo.environments.reasoning_effort import bind_episode_effort, efforts_by_level, has_thinking_budget
from halo.environments.registry import build_environment
from halo.environments.rollout_config import VLLM_BACKEND
from halo.model_loading.parallel.frozen_models import (
    load_reference_model_for_on_policy_grpo,
    reject_unfit_on_policy_reference,
)
from halo.model_loading.parallel.model_source import probe_checkpoint
from halo.model_loading.parallel.peft_setup import configure_adapter_training
from halo.models.context_window import resolve_model_context_window
from halo.models.moe.config_fields import has_routed_experts
from halo.runtime.collectives import barrier
from halo.runtime.ranks import is_global_main_process
from halo.trainers.grpo.environmental.environmental_rules import reject_prefetch_past_retry_budget
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.environmental.trajectory_tokenize import rollout_template_kwargs
from halo.trainers.grpo.rollout.engine_preflight import (
    ServerRequirements,
    verify_context_window,
    verify_engine_prompt_logprobs,
    verify_group_servers,
    verify_sampler_logprob_reference,
    verify_server_chat_template,
    verify_server_flags,
    verify_turn_deadline,
)
from halo.trainers.grpo.rollout.weight_push import reject_unsyncable_config
from halo.weight_sync.registry import resolve_weight_sync_client

logger = logging.getLogger(__name__)


def measure_env_prompt_overhead(environment, tokenizer, template_kwargs: dict) -> int:
    """Token count of the environment's preamble (system prompt + tool schema), which the startup context-window check
    must include or it validates a prompt smaller than any the model sees. Rendered under ``template_kwargs``, as the
    preamble depends on them. A template that cannot render it raises, since rollouts render the same preamble. An MCP
    environment's tools appear only at its first reset, so the overhead under-counts, with a warning."""
    messages = ([{"role": "system", "content": environment.system_prompt}] if environment.system_prompt else []) + [
        {"role": "user", "content": ""}
    ]
    tools = environment.get_tools_schema()
    if environment.TOOLS_ARRIVE_ON_RESET and not tools and is_global_main_process():
        logger.warning(
            "An MCP server's tools arrive when its first reset connects, so the context-window check omits "
            "their schema from the prompt overhead."
        )
    rendered = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True, **template_kwargs
    )
    overhead = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
    if is_global_main_process():
        logger.info(f"Environment prompt overhead (system prompt + tool schema): {overhead} tokens")
    return overhead


def probe_template_kwargs(async_config: AsyncRolloutConfig, environment) -> dict:
    """The kwargs a rollout request carries: the run's template variables plus the level and thinking budget an episode
    binds under the run's bounds (a ``'random'`` setting draws one level), as an episode's requests state them."""
    effort = bind_episode_effort(
        None,
        environment,
        max_tokens=async_config.rollout_max_tokens,
        max_thinking_tokens=async_config.rollout_max_thinking_tokens,
        max_episode_tokens=async_config.rollout_max_episode_tokens,
        max_answer_tokens=async_config.rollout_max_answer_tokens,
    )
    return rollout_template_kwargs(
        async_config.rollout_chat_template_kwargs, effort.level, effort.thinking_budget, effort.budget_soft
    )


def server_requirements(async_config: AsyncRolloutConfig, environment, model) -> ServerRequirements:
    """What this run reads off its servers' launch flags."""
    efforts = efforts_by_level(
        environment,
        max_tokens=async_config.rollout_max_tokens,
        max_thinking_tokens=async_config.rollout_max_thinking_tokens,
        max_answer_tokens=async_config.rollout_max_answer_tokens,
        max_episode_tokens=async_config.rollout_max_episode_tokens,
    )
    budgeted = async_config.rollout_backend == VLLM_BACKEND and has_thinking_budget(efforts)
    consumers = (
        ("reasoning_price", async_config.reasoning_price is not None),
        ("reasoning_floor", async_config.reasoning_floor > 0),
        ("carry_reasoning", environment.carry_reasoning),
        ("a thinking budget", budgeted),
    )
    return ServerRequirements(
        is_moe=has_routed_experts(model.config),
        sends_tools=bool(environment.get_tools_schema()) or environment.TOOLS_ARRIVE_ON_RESET,
        captures_routing=async_config.routing_replay == "rollout",
        reasoning_consumers=tuple(name for name, on in consumers if on),
        sampled_token_ids=async_config.train_on_sampled_tokens,
    )


def verify_served_template(async_config: AsyncRolloutConfig, environment, tokenizer) -> None:
    """COLLECTIVE. Reads the two contracts a run can tell apart: two drawn levels, or a fixed level against none."""
    levels = list(
        efforts_by_level(
            environment,
            max_tokens=async_config.rollout_max_tokens,
            max_thinking_tokens=async_config.rollout_max_thinking_tokens,
            max_answer_tokens=async_config.rollout_max_answer_tokens,
            max_episode_tokens=async_config.rollout_max_episode_tokens,
        ).values()
    )[:2]
    efforts = levels if len(levels) == 2 or levels[0].level is None else [levels[0], None]
    system = [{"role": "system", "content": environment.system_prompt}] if environment.system_prompt else []
    messages = [*system, {"role": "user", "content": "Probe."}]

    def trainer_prompt_ids(effort) -> list[int]:
        kwargs = rollout_template_kwargs(
            async_config.rollout_chat_template_kwargs,
            effort.level if effort else None,
            effort.thinking_budget if effort else None,
            effort.budget_soft if effort else False,
        )
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    verify_server_chat_template(
        async_config.get_server_urls(),
        trainer_prompt_ids,
        messages,
        efforts,
        async_config.get_rollout_config(),
        connection_timeout=async_config.rollout_connection_timeout,
    )


def worst_case_generation(async_config: AsyncRolloutConfig, max_turns: int) -> int:
    """``max_turns`` turns of ``rollout_max_tokens``, or the episode output budget where it binds first."""
    generated = max_turns * async_config.rollout_max_tokens
    if async_config.rollout_max_episode_tokens is not None:
        generated = min(generated, async_config.rollout_max_episode_tokens)
    return generated


def rollout_start_log(env_settings: dict, async_config: AsyncRolloutConfig) -> list[str]:
    """The one place environment settings and rollout setup are logged, prefetch as resolved against servers."""
    if async_config.rollout_server_configs:
        servers = [
            f"rollout_server_{i}: {cfg['url']} (port {cfg.get('group_port', 'auto')})"
            for i, cfg in enumerate(async_config.rollout_server_configs)
        ]
    else:
        servers = [f"rollout_server: {async_config.rollout_server_url}"]
    return [
        *(f"{key}: {value}" for key, value in env_settings.items()),
        f"rollout_backend: {async_config.rollout_backend}",
        f"rollout_actor_budget: {async_config.rollout_actor_budget}",
        f"max_concurrent: {async_config.max_concurrent_rollouts or 'auto'}",
        *servers,
        f"weight_sync: every {async_config.sync_weights_every_n_steps} steps",
        f"prefetch: {f'depth {async_config.prefetch_depth}' if async_config.prefetch_active() else 'disabled'}",
        f"model: {async_config.model_name}",
    ]


def reject_missing_carried_columns(args: EnvironmentalGRPOScriptArguments, ds) -> None:
    """A renamed answer_field resolving to no column is a typo, yielding a full RL run with no ground truth. The default
    spelling is not checked: the trainer gates on the environment's requires_answer. prompt_field is the loader's
    conversation_field, already guarded."""
    if args.answer_field:
        if args.answer_field != DEFAULT_ANSWER_FIELD:
            reject_missing_render_column(ds, str(args.dataset), "answer_field", args.answer_field)
        elif missing_render_column_splits(ds, args.answer_field) and is_global_main_process():
            logger.warning(
                f"No '{args.answer_field}' column in {args.dataset}: rows carry no answer. An "
                f"environment that grades against one refuses this dataset at trainer construction; "
                f"one that grades against an answer only where a row has one pays every episode for "
                f"completing the task instead."
            )
    for column in args.context_fields or []:
        if column:
            reject_missing_render_column(ds, str(args.dataset), "context_fields", column)


def process_dataset(args: EnvironmentalGRPOScriptArguments, tokenizer, ds):
    def carried_columns(row):
        """Typed values, on rejected rows too, keeping every writer batch's Arrow schema identical."""
        result = {}
        if args.answer_field and args.answer_field in row:
            result[ANSWER_KEY] = row[args.answer_field]
        for field in args.context_fields or []:
            if field in row:
                result[field] = row[field]
        return result

    def rejected_row(row, messages=None):
        return {"prompt": blank_conversation(messages), **carried_columns(row)}

    def process_for_grpo(row):
        messages = as_conversation(row.get(args.prompt_field))
        if not isinstance(messages, list):
            return rejected_row(row)

        # The rendered text is only the length probe; the environment replays the message list.
        if render_generation_prompt(tokenizer, messages, max_prompt_length=args.max_prompt_length) is None:
            return rejected_row(row, messages)

        return {"prompt": messages, **carried_columns(row)}

    original_columns = list(ds["train"].column_names)
    keep_columns = {"prompt", ANSWER_KEY} | set(args.context_fields or [])
    columns_to_remove = [col for col in original_columns if col not in keep_columns]

    processed_ds = coordinated_map_and_filter(
        ds,
        process_for_grpo,
        filter_field="prompt",
        remove_columns=columns_to_remove,
        desc="Processing dataset for Distributed Environmental GRPO",
        cache_key_extras={
            "prompt_field": args.prompt_field,
            "max_prompt_length": args.max_prompt_length,
            "answer_field": args.answer_field,
            "context_fields": args.context_fields,
        },
    )

    return processed_ds


def main():
    args, env_config, async_config, grpo_config, model_config, dist_args = parse_script_args(
        EnvironmentalGRPOScriptArguments,
        EnvironmentConfig,
        AsyncRolloutConfig,
        GRPOConfig,
        ModelArguments,
        ParallelismArguments,
    )

    # The environment's registry supplies the schema, so a dataset tool column never reaches the template. The weight
    # sync forwards names verbatim, and the text-only sibling's model.layers.* would miss the multimodal server's
    # model.language_model.layers.* on first sync.
    reject_unsupported_args("Environmental GRPO", tools_field=args.tools_field, text_only_model=args.text_only_model)
    # The script pins max_completion_length to rollout_max_tokens below, so any other value is discarded.
    if grpo_config.max_completion_length not in (GRPOConfig.max_completion_length, async_config.rollout_max_tokens):
        raise ValueError(
            f"max_completion_length={grpo_config.max_completion_length} is not a knob under Environmental "
            f"GRPO: each turn generates up to rollout_max_tokens ({async_config.rollout_max_tokens}), which "
            f"the script also writes into max_completion_length. Set rollout_max_tokens instead and remove "
            f"max_completion_length."
        )
    # The trainer refuses the same config at construction; checked here, before the model load.
    reject_prefetch_past_retry_budget(
        resolve_weight_sync_client(async_config.rollout_backend), async_config, grpo_config
    )

    # A throwaway env for the startup checks, built before any load so an unknown environment_type or a refused
    # environment_kwargs fails here.
    env_settings = env_config.to_env_config()
    probe_env = build_environment(env_config.environment_type, env_settings)

    context = init_training_script(
        args,
        grpo_config,
        model_config,
        dist_args,
        script_prefix=f"env-grpo-{env_config.environment_type}",
        trainer_cls=DistributedAsyncEnvironmentalGRPOTrainer,
        row_sources=(async_config,),
    )
    parallelism_config = context.parallelism_config

    # The family's verdict, before the datasets, the load and the server preflights it would pay for first.
    checkpoint_config, _ = probe_checkpoint(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
    )
    reject_unsyncable_config(checkpoint_config, async_config.rollout_backend)

    # The trainer gets dataset_presharded so it does not re-shard a per-rank load.
    ds, dataset_presharded = load_script_datasets(
        args,
        parallelism_config,
        conversation_field=args.prompt_field,
        conversation_knob="prompt_field",
    )
    # process_dataset prunes to prompt/answer/context_fields, so an image column would vanish and train on text alone.
    reject_image_columns(ds, "Environmental GRPO")
    reject_missing_carried_columns(args, ds)

    # Trajectories are collated into padded batches.
    requested_attn = grpo_row_attn_implementation(
        model_config,
        grpo_config,
        packed_microbatches=async_config.packed_microbatch_tokens is not None,
        sinks_reset=args.reset_sinks,
        checkpoint_config=checkpoint_config,
    )
    reject_unfit_on_policy_reference(model_config, grpo_config, parallelism_config)
    model, tokenizer = load_model_and_tokenizer(context, grpo_config, model_config, args, attn_default=requested_attn)

    # A trajectory is bounded by the context window, not max_completion_length: TRL reads it as the dr_grpo
    # normalization constant only.
    grpo_config.max_completion_length = async_config.rollout_max_tokens

    # The trainer raises on a trajectory past the window rather than truncating.
    tokenizer = apply_tokenizer_window(args, model, tokenizer, resolve_model_context_window(model, tokenizer))

    peft_config = configure_adapter_training(args, model, model_config, "CAUSAL_LM")
    log_model_info(model, tokenizer)

    processed_ds = process_dataset(args, tokenizer, ds)
    train_dataset = processed_ds["train"]
    eval_dataset = processed_ds.get("test")

    log_script_dataset_examples({"train": train_dataset, "test": eval_dataset}, tokenizer, args, grpo_config)

    if async_config.model_name is None:
        async_config.model_name = model_config.model_name_or_path

    # All ranks raise together. A single turn (prompt budget, including the env's preamble, plus per-turn generation)
    # has to fit; the multi-turn worst case is advisory.
    max_turns = probe_env.max_turns
    template_kwargs = probe_template_kwargs(async_config, probe_env)
    prompt_budget = (args.max_prompt_length or 0) + measure_env_prompt_overhead(probe_env, tokenizer, template_kwargs)
    full_trajectory_tokens = prompt_budget + worst_case_generation(async_config, max_turns)
    verify_context_window(
        async_config.get_server_urls(),
        single_turn_tokens=prompt_budget + async_config.rollout_max_tokens,
        full_trajectory_tokens=full_trajectory_tokens,
        backend=async_config.rollout_backend,
        connection_timeout=async_config.rollout_connection_timeout,
    )
    verify_turn_deadline(
        async_config.get_server_urls(),
        async_config.rollout_max_tokens,
        async_config.request_timeout,
        backend=async_config.rollout_backend,
        connection_timeout=async_config.rollout_connection_timeout,
    )
    verify_server_flags(
        async_config.get_server_urls(),
        server_requirements(async_config, probe_env, model),
        backend=async_config.rollout_backend,
        connection_timeout=async_config.rollout_connection_timeout,
    )
    if async_config.rollout_sync_topology == "single_group":
        verify_group_servers(
            async_config.get_server_urls(),
            backend=async_config.rollout_backend,
            connection_timeout=async_config.rollout_connection_timeout,
        )
    verify_served_template(async_config, probe_env, tokenizer)
    # The IS ratio divides by the engine's logprobs: they must be the sampling distribution's.
    verify_sampler_logprob_reference(
        async_config.get_server_urls(),
        temperature=async_config.rollout_temperature,
        top_p=async_config.rollout_top_p,
        top_k=async_config.rollout_top_k,
        min_p=async_config.rollout_min_p,
        repetition_penalty=async_config.rollout_repetition_penalty,
        sequence_ratio_active=async_config.build_isr_mask_config().sums_sequence_logratio,
        backend=async_config.rollout_backend,
        connection_timeout=async_config.rollout_connection_timeout,
    )
    # The engine re-score reads each train row off the prompt log-probs: they must not depend on the batch.
    if async_config.isr_engine_reference:
        verify_engine_prompt_logprobs(
            async_config.get_server_urls(),
            tokenizer,
            backend=async_config.rollout_backend,
            connection_timeout=async_config.rollout_connection_timeout,
        )

    # A bad judge URL/key/model would mark every episode invalid and train on zero advantage.
    verify_backend_on_rank0(
        probe_env.verify_backend, f"Environment backend of environment_type={env_config.environment_type!r}"
    )
    # The KL reference TRL would otherwise build; None where the run holds none. After the server preflights, so a
    # misconfigured server fails before a second model load.
    ref_model = load_reference_model_for_on_policy_grpo(
        args,
        model_config,
        grpo_config,
        parallelism_config,
        tokenizer,
        peft_config=peft_config,
        reset_sinks=args.reset_sinks,
        attn_default=requested_attn,
    )

    grpo_config.reward_weights = [1.0]

    barrier()

    callbacks = configure_balancing_and_build_callbacks(
        args,
        grpo_config,
        model,
        context,
        # A trajectory accumulates every turn, so MFU would be reported against a length batches routinely exceed.
        max_seq_len=full_trajectory_tokens,
    )
    export_processor = load_export_processor(model_config, args, tokenizer)
    trainer = DistributedAsyncEnvironmentalGRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        ref_model=ref_model,
        callbacks=callbacks,
        async_config=async_config,
        environment_config=env_config,
        save_completions=args.save_completions,
        **distributed_trainer_kwargs(args, parallelism_config, dataset_presharded=dataset_presharded),
    )
    trainer.export_processing_class = export_processor
    run_trainer(
        trainer,
        context,
        method_name=f"Environmental GRPO ({env_config.environment_type})",
        extra_start_log=rollout_start_log(env_settings, async_config),
    )


if __name__ == "__main__":
    with_distributed_teardown(main)()
