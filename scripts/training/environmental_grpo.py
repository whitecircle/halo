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
``rollout_backend: sglang`` (gpt-oss only, no expert distribution). Each example config's header
lists the server flags that config needs: the ``--moe-backend triton`` gate, the tool-call and
reasoning parsers, and the GPU split between trainer and server.

Usage (each config's header carries its own exact launch line):
    # Plain data-parallel, 4 train GPUs
    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \\
        scripts/training/environmental_grpo.py \\
        examples/grpo/environmental/gptoss/vllm/gptoss-20b-code-contests-full-ep1.yaml

    # Expert parallelism (MoE), 4 train GPUs, one dispatch group
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \\
        scripts/training/environmental_grpo.py \\
        examples/grpo/environmental/qwen3_5/vllm/qwen3.6-35b-a3b-react-math-full-ep4.yaml

    # SGLang rollouts (gpt-oss, no expert distribution)
    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \\
        scripts/training/environmental_grpo.py \\
        examples/grpo/environmental/gptoss/sglang/gptoss-20b-code-contests-full-ep1.yaml

See agent-docs/training-methods/grpo/environmental-grpo.md for the objective, the environment roster and
the rollout-server contract.
"""

from accelerate.logging import get_logger
from trl import GRPOConfig, ModelConfig

from src.args.distributed_args import DistributedArguments
from src.args.environmental_grpo_args import EnvironmentalGRPOScriptArguments
from src.configs.async_training_config import AsyncTrainingConfig
from src.configs.environment_config import EnvironmentConfig
from src.data.pipeline.processing import process_dataset_with_map_and_filter, require_render_column
from src.data.pipeline.rendered import render_generation_prompt
from src.data.sources.loading import reject_image_columns
from src.distributed.loading.peft_setup import setup_peft_model
from src.distributed.runtime import barrier, broadcast_from_rank0, is_global_main_process
from src.distributed.tensor_parallel.state_dict import input_embeddings_tp_sharded
from src.environments.registry import create_environment, get_registered_environments
from src.models.loading.model_preparation import log_model_info
from src.models.loading.tokenizer_setup import get_model_context_window, setup_model_and_tokenizer
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from src.trainers.grpo.rollout.weight_sync_clients import verify_context_window_synced
from src.training.environment import run_training
from src.training.parser import H4ArgumentParser
from src.training.script_runner import (
    apply_distributed_trainer_config,
    build_training_callbacks,
    distributed_trainer_kwargs,
    init_training_script,
    load_script_datasets,
    load_script_model,
    log_script_dataset_examples,
    padded_workload_attn_implementation,
    reject_unsupported_args,
    run_trainer,
)

logger = get_logger(__name__, log_level="INFO")

# Fallback token allowance when the environment preamble cannot be rendered.
FALLBACK_ENV_PROMPT_OVERHEAD = 2048


def measure_env_prompt_overhead(environment, tokenizer) -> int:
    """Token count of the environment's conversation preamble (its system prompt + tool schema).

    The dataset prompt filter measures only the templated user prompt, while the rollout prompt is
    built by the environment: its own system prompt plus the tool schema the chat template renders
    into the context. The startup context-window check includes that overhead, or it validates a
    prompt smaller than any the model will see. Falls back to a conservative margin when the template
    cannot render the preamble (logged with the assumption).
    """
    messages = ([{"role": "system", "content": environment.system_prompt}] if environment.system_prompt else []) + [
        {"role": "user", "content": ""}
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tools=environment.get_tools_schema(), tokenize=False, add_generation_prompt=True
        )
        overhead = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        logger.info(f"Environment prompt overhead (system prompt + tool schema): {overhead} tokens (measured)")
        return overhead
    except Exception as e:
        logger.warning(
            f"Could not render the environment prompt preamble ({e}); assuming "
            f"{FALLBACK_ENV_PROMPT_OVERHEAD} tokens of overhead for the context-window check."
        )
        return FALLBACK_ENV_PROMPT_OVERHEAD


def process_dataset(args: EnvironmentalGRPOScriptArguments, tokenizer, ds):
    """Process a loaded dataset for Environmental GRPO."""

    def carried_columns(row):
        """Answer/context columns forwarded with the row's real (typed) values — on rejected rows
        too, keeping every writer batch's Arrow schema identical."""
        result = {}
        if args.answer_field and args.answer_field in row:
            result["answer"] = row[args.answer_field]
        for field in args.context_fields or []:
            if field in row:
                result[field] = row[field]
        return result

    def rejected_row(row, messages=None):
        """Type-stable rejection sentinel: a blank message list, never None — an all-rejected first
        writer batch of None rows makes Arrow infer a null prompt column and crash casting later
        real batches. is_valid_example drops all-blank-content conversations downstream."""
        blank = [{**m, "content": ""} for m in messages] if messages else [{"role": "user", "content": ""}]
        return {"prompt": blank, **carried_columns(row)}

    def process_for_grpo(row):
        prompt_field = args.prompt_field
        prompt = row.get(prompt_field)

        if prompt is None:
            return rejected_row(row)

        if isinstance(prompt, list):
            messages = prompt
        elif isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            return rejected_row(row)

        # The rendered text is only the length probe; the environment replays the message list.
        if render_generation_prompt(tokenizer, messages, max_prompt_length=args.max_prompt_length) is None:
            return rejected_row(row, messages)

        return {"prompt": messages, **carried_columns(row)}

    original_columns = list(ds["train"].column_names)
    # A mistyped answer_field yields answer-less rows: an RL run with no ground truth and all-zero
    # rewards. prompt_field is not re-checked here; it is this script's conversation_field, so the
    # loader already ran the same guard over it.
    for knob, column in [
        ("answer_field", args.answer_field),
        *[("context_fields", field) for field in args.context_fields or []],
    ]:
        if column:
            require_render_column(ds, str(args.dataset), knob, column)
    keep_columns = {"prompt", "answer"} | set(args.context_fields or [])
    columns_to_remove = [col for col in original_columns if col not in keep_columns]

    processed_ds = process_dataset_with_map_and_filter(
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
    parser = H4ArgumentParser(
        (
            EnvironmentalGRPOScriptArguments,
            EnvironmentConfig,
            AsyncTrainingConfig,
            GRPOConfig,
            ModelConfig,
            DistributedArguments,
        )
    )
    args, env_config, async_config, grpo_config, model_config, dist_args = parser.parse()

    available = get_registered_environments()
    if env_config.environment_type not in available:
        raise ValueError(
            f"Unknown environment_type: '{env_config.environment_type}'. Available: {available}. "
            f"For custom environments, register the class with register_environment() or pass "
            f"environment_cls to the trainer."
        )

    # Tools follow the same rule: the environment's tool registry supplies the schema sent to vLLM
    # (get_tools_schema), so a dataset tool column would never reach the rollout template. The weight
    # sync forwards trainer parameter names verbatim, and the text-only CausalLM sibling spells its
    # decoder model.layers.* where the multimodal checkpoint the server loads spells
    # model.language_model.layers.*, so every dense tensor would miss its slot on the first sync.
    reject_unsupported_args(
        "Environmental GRPO", tools_field=args.tools_field, text_only_model=dist_args.text_only_model
    )

    runtime = init_training_script(
        args,
        grpo_config,
        model_config,
        dist_args,
        script_prefix=f"env-grpo-{env_config.environment_type}",
        supports_cp=False,
        supports_pp=False,
    )
    parallelism_config = runtime.parallelism_config

    # Trajectories are collated into padded batches.
    requested_attn = padded_workload_attn_implementation(model_config, sinks_reset=dist_args.reset_sinks)
    model, tokenizer = load_script_model(
        runtime, grpo_config, model_config, dist_args, attn_implementation=requested_attn
    )

    # The per-turn generation budget is rollout_max_tokens; a trajectory accumulates across turns and
    # is bounded by the context window (the trainer raises on overflow rather than truncating), so
    # max_completion_length is not a length knob here: TRL reads it as the dr_grpo normalization
    # constant only.
    grpo_config.max_completion_length = async_config.rollout_max_tokens

    # tokenizer.model_max_length is the model's context window, the same limit vLLM enforces during
    # rollout; the trainer raises on a trajectory that exceeds it rather than truncating.
    tokenizer = setup_model_and_tokenizer(
        args,
        model,
        tokenizer,
        get_model_context_window(model, tokenizer),
        embeddings_sharded=input_embeddings_tp_sharded,
    )

    peft_config = setup_peft_model(args, model, model_config, "CAUSAL_LM")
    log_model_info(model, tokenizer)

    # Pre-sharded datasets are split per DP rank at load; the trainer gets dataset_presharded so it
    # does not re-shard (no-op for the usual raw prompt/answer dataset).
    ds, dataset_presharded = load_script_datasets(
        args,
        parallelism_config,
        conversation_field=args.prompt_field,
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
    # preamble (system prompt + tool schema), measured off a throwaway env whose max_turns=None takes
    # the env class default.
    probe_env = create_environment(env_config.environment_type, env_config.to_env_config())
    max_turns = probe_env.max_turns
    prompt_budget = (args.max_prompt_length or 0) + measure_env_prompt_overhead(probe_env, tokenizer)
    verify_context_window_synced(
        async_config.get_server_urls(),
        single_turn_tokens=prompt_budget + async_config.rollout_max_tokens,
        full_trajectory_tokens=prompt_budget + max_turns * async_config.rollout_max_tokens,
        backend=async_config.rollout_backend,
    )

    # Environments that score through an external backend (an LLM judge) probe it here: a bad
    # URL/key/model would otherwise mark every episode invalid and the job would train on zero
    # advantage. Rank 0 probes and broadcasts so all ranks raise together.
    probe_error: str | None = None
    if is_global_main_process():
        try:
            probe_env.verify_backend()
        except Exception as e:
            probe_error = (
                f"Environment backend probe failed for environment_type={env_config.environment_type!r}: {e}. "
                f"Fix the environment config before launching."
            )
    probe_error = broadcast_from_rank0(probe_error)
    if probe_error is not None:
        raise RuntimeError(probe_error)

    grpo_config.reward_weights = [1.0]
    apply_distributed_trainer_config(grpo_config, parallelism_config)

    barrier()

    # Parallelism sizes are logged by run_trainer's canonical start log.
    if is_global_main_process():
        sep = "=" * 60
        logger.info(sep)
        logger.info(f"Distributed Environmental GRPO — {env_config.environment_type} (mode: {runtime.mode_suffix})")
        logger.info(sep)

        env_cfg = env_config.to_env_config()
        for k, v in env_cfg.items():
            logger.info(f"  {k}: {v}")

        logger.info(f"  rollout_workers: {async_config.num_rollout_workers}")
        logger.info(f"  max_concurrent: {async_config.max_concurrent_rollouts or 'auto'}")
        if async_config.rollout_server_configs:
            for i, cfg in enumerate(async_config.rollout_server_configs):
                logger.info(f"  rollout_server_{i}: {cfg['url']} (port {cfg.get('group_port', 'auto')})")
        else:
            logger.info(f"  rollout_server: {async_config.rollout_server_url}")
        logger.info(f"  weight_sync: every {async_config.sync_weights_every_n_steps} steps")
        logger.info(f"  prefetch: {'enabled' if async_config.enable_prefetch else 'disabled'}")
        logger.info(f"  model: {async_config.model_name}")
        logger.info(sep)

    callbacks = build_training_callbacks(
        args,
        grpo_config,
        model,
        parallelism_config,
        policy_gradient_loss=True,
        syncs_to_external_generator=True,
        # A trajectory accumulates every turn, so throughput/MFU is reported against the full
        # multi-turn length rather than the declared prompt and per-turn budgets.
        max_seq_len=prompt_budget + max_turns * async_config.rollout_max_tokens,
    )
    trainer = DistributedAsyncEnvironmentalGRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
        async_config=async_config,
        environment_config=env_config,
        save_completions=args.save_completions,
        **distributed_trainer_kwargs(args, dist_args, parallelism_config, dataset_presharded=dataset_presharded),
    )
    run_trainer(trainer, runtime, method_name=f"Environmental GRPO ({env_config.environment_type})")


if __name__ == "__main__":
    run_training(main)()
