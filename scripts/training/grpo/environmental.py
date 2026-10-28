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
    default_padded_attn_implementation,
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
from halo.environments.reasoning_effort import bind_episode_effort
from halo.environments.registry import build_environment
from halo.model_loading.parallel.frozen_models import load_reference_model_for_on_policy_grpo
from halo.model_loading.parallel.peft_setup import configure_adapter_training
from halo.models.context_window import resolve_model_context_window
from halo.runtime.collectives import barrier
from halo.runtime.ranks import is_global_main_process
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.environmental.trajectory_tokenize import rollout_template_kwargs
from halo.trainers.grpo.rollout.engine_preflight import (
    verify_context_window,
    verify_engine_prompt_logprobs,
    verify_sampler_logprob_reference,
)

logger = logging.getLogger(__name__)


def measure_env_prompt_overhead(environment, tokenizer, template_kwargs: dict) -> int:
    """Token count of the environment's conversation preamble (its system prompt + tool schema).

    The dataset prompt filter measures only the templated user prompt, while the rollout prompt is
    built by the environment: its own system prompt plus the tool schema the chat template renders
    into the context. The startup context-window check must include that overhead or it validates a
    prompt smaller than any the model will ever see. Rendered under ``template_kwargs`` — the
    rollout's chat-template variables and effort steer (:func:`rollout_template_kwargs`) — since the
    template's preamble depends on them. A template that cannot render the preamble raises: the rollouts
    render the same preamble, so no margin assumed in its place would hold. An MCP environment learns its
    tools only when its first reset connects to the server, and ``environment`` here is never reset, so its
    schema is absent and the overhead under-counts by it, which is warned.
    """
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
    """The chat-template kwargs a rollout request of this run carries, for the prompt-overhead probe:
    the run's template variables plus the level and thinking budget an episode binds through
    :func:`bind_episode_effort` under the run's own bounds (a ``'random'`` env setting draws one level, as an
    episode does), so the probe renders the budget an episode's requests state, flagged where it is soft."""
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


def worst_case_generation(async_config: AsyncRolloutConfig, max_turns: int) -> int:
    """The most an episode can generate, for the context-window check: ``max_turns`` turns of
    ``rollout_max_tokens``, or the episode output budget where it binds first."""
    generated = max_turns * async_config.rollout_max_tokens
    if async_config.rollout_max_episode_tokens is not None:
        generated = min(generated, async_config.rollout_max_episode_tokens)
    return generated


def rollout_start_log(env_settings: dict, async_config: AsyncRolloutConfig) -> list[str]:
    """The run's environment settings and rollout setup, as lines of the start log — the one place they
    are logged, prefetch as the trainer resolves it against the server count."""
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
        f"rollout_workers: {async_config.num_rollout_workers}",
        f"max_concurrent: {async_config.max_concurrent_rollouts or 'auto'}",
        *servers,
        f"weight_sync: every {async_config.sync_weights_every_n_steps} steps",
        f"prefetch: {'enabled' if async_config.prefetch_active() else 'disabled'}",
        f"model: {async_config.model_name}",
    ]


def process_dataset(args: EnvironmentalGRPOScriptArguments, tokenizer, ds):
    """Process a loaded dataset for Environmental GRPO."""

    def carried_columns(row):
        """Answer/context columns forwarded with the row's real (typed) values — on rejected rows
        too, keeping every writer batch's Arrow schema identical."""
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
    # A renamed answer_field that resolves to no column is a typo, and a typo yields answer-less rows:
    # a full RL run with no ground truth. The default spelling is not checked — whether an answer is
    # needed is the environment's requires_answer declaration, which the trainer gates the dataset on.
    # prompt_field is not re-checked here; it is this script's conversation_field, so the loader
    # already ran the same guard over it.
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

    # The environment's tool registry supplies the schema sent to the rollout server (get_tools_schema),
    # so a dataset tool column would never reach the rollout template. The weight sync forwards trainer
    # parameter names verbatim, and the text-only CausalLM sibling spells its decoder model.layers.*
    # where the multimodal checkpoint the server loads spells model.language_model.layers.*, so every
    # dense tensor would miss its slot on the first sync.
    reject_unsupported_args("Environmental GRPO", tools_field=args.tools_field, text_only_model=args.text_only_model)
    # The script pins max_completion_length to rollout_max_tokens below, so any other value is discarded.
    if grpo_config.max_completion_length not in (GRPOConfig.max_completion_length, async_config.rollout_max_tokens):
        raise ValueError(
            f"max_completion_length={grpo_config.max_completion_length} is not a knob under Environmental "
            f"GRPO: each turn generates up to rollout_max_tokens ({async_config.rollout_max_tokens}), which "
            f"the script also writes into max_completion_length. Set rollout_max_tokens instead and remove "
            f"max_completion_length."
        )

    # A throwaway env for the startup checks below (max_turns=None takes the class default), built
    # before any load so an unknown environment_type, or an environment_kwargs the environment
    # refuses, fails here.
    env_settings = env_config.to_env_config()
    probe_env = build_environment(env_config.environment_type, env_settings)

    context = init_training_script(
        args,
        grpo_config,
        model_config,
        dist_args,
        script_prefix=f"env-grpo-{env_config.environment_type}",
        trainer_cls=DistributedAsyncEnvironmentalGRPOTrainer,
    )
    parallelism_config = context.parallelism_config

    # Trajectories are collated into padded batches.
    requested_attn = default_padded_attn_implementation(model_config, sinks_reset=args.reset_sinks)
    model, tokenizer = load_model_and_tokenizer(context, grpo_config, model_config, args, attn_default=requested_attn)

    # The per-turn generation budget is rollout_max_tokens; a trajectory accumulates across turns and
    # is bounded by the context window (the trainer raises on overflow rather than truncating), so
    # max_completion_length is not a length knob here: TRL reads it as the dr_grpo normalization
    # constant only.
    grpo_config.max_completion_length = async_config.rollout_max_tokens

    # The trainer raises on a trajectory past the window rather than truncating.
    tokenizer = apply_tokenizer_window(args, model, tokenizer, resolve_model_context_window(model, tokenizer))

    peft_config = configure_adapter_training(args, model, model_config, "CAUSAL_LM")
    log_model_info(model, tokenizer)

    # Pre-sharded datasets are split per DP rank at load; the trainer gets dataset_presharded so it
    # does not re-shard (no-op for the usual raw prompt/answer dataset).
    ds, dataset_presharded = load_script_datasets(
        args,
        parallelism_config,
        conversation_field=args.prompt_field,
        conversation_knob="prompt_field",
    )
    # process_dataset keeps only prompt/answer/context_fields, so an image column would be pruned and
    # the run would train on the rows' text alone.
    reject_image_columns(ds, "Environmental GRPO")
    processed_ds = process_dataset(args, tokenizer, ds)
    train_dataset = processed_ds["train"]
    eval_dataset = processed_ds.get("test")

    log_script_dataset_examples({"train": train_dataset, "test": eval_dataset}, tokenizer, args, grpo_config)

    if async_config.model_name is None:
        async_config.model_name = model_config.model_name_or_path

    # All ranks raise together. A single turn (prompt budget plus per-turn generation) has to fit the
    # context; the multi-turn worst case is advisory. The prompt budget includes the environment's own
    # preamble (system prompt + tool schema), measured off the probe env.
    max_turns = probe_env.max_turns
    template_kwargs = probe_template_kwargs(async_config, probe_env)
    prompt_budget = (args.max_prompt_length or 0) + measure_env_prompt_overhead(probe_env, tokenizer, template_kwargs)
    full_trajectory_tokens = prompt_budget + worst_case_generation(async_config, max_turns)
    verify_context_window(
        async_config.get_server_urls(),
        single_turn_tokens=prompt_budget + async_config.rollout_max_tokens,
        full_trajectory_tokens=full_trajectory_tokens,
        backend=async_config.rollout_backend,
    )
    # The IS ratio divides by the engine's logprobs: they must be the sampling distribution's.
    verify_sampler_logprob_reference(
        async_config.get_server_urls(),
        temperature=async_config.rollout_temperature,
        top_p=async_config.rollout_top_p,
        top_k=async_config.rollout_top_k,
        min_p=async_config.rollout_min_p,
        repetition_penalty=async_config.rollout_repetition_penalty,
        sequence_ratio_active=async_config.build_is_mask_config().sums_sequence_logratio,
        backend=async_config.rollout_backend,
    )
    # The engine re-score reads each train row off the prompt log-probs: they must not depend on the batch.
    if async_config.isr_engine_reference:
        verify_engine_prompt_logprobs(async_config.get_server_urls(), tokenizer, backend=async_config.rollout_backend)

    # Environments that score through an external backend (an LLM judge) probe it here: a bad
    # URL/key/model would otherwise mark every episode invalid and the job would train on zero
    # advantage.
    verify_backend_on_rank0(
        probe_env.verify_backend, f"Environment backend of environment_type={env_config.environment_type!r}"
    )
    # The KL reference TRL would otherwise build itself; None where the run holds none. Loaded after the
    # server preflights, so a misconfigured server fails before a second model load.
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
        # A trajectory accumulates every turn, so the declared prompt + per-turn budgets are not its
        # bound — throughput/MFU would be reported against a length the batches routinely exceed.
        max_seq_len=full_trajectory_tokens,
    )
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
    run_trainer(
        trainer,
        context,
        method_name=f"Environmental GRPO ({env_config.environment_type})",
        extra_start_log=rollout_start_log(env_settings, async_config),
    )


if __name__ == "__main__":
    with_distributed_teardown(main)()
