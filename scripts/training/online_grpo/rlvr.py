#!/usr/bin/env python
"""RLVR (Reinforcement Learning with Verifiable Rewards) online GRPO training.

Online GRPO against a vLLM server, scored by the ``rewards`` terms of the config: the strict
``\\boxed{}`` accuracy grader, the regex format grader, a generative judge, a served reward model.

Supported Parallelism Modes: EP, TP, ETP (CP is not supported by ``DistributedGRPOTrainer``; the
rollout server takes its own GPUs, so size the launch to the remaining ones).

Usage:
    torchrun --nproc_per_node=8 scripts/training/online_grpo/rlvr.py \\
        examples/grpo/online/rlvr-online-grpo-template.yaml --expert_parallel_size=8
"""

import asyncio

from trl import GRPOConfig, ModelConfig

from src.args.distributed_args import DistributedArguments
from src.args.mixins import RLRRArguments
from src.args.rlvr_online_grpo_args import RLVROnlineGRPOScriptArguments
from src.data.pipeline.conversation import chat_template_kwargs, fold_system_into_conversation
from src.data.pipeline.processing import process_dataset_with_map_and_filter, require_render_column
from src.data.pipeline.rendered import render_generation_prompt
from src.data.sources.loading import reject_image_columns
from src.distributed.loading.peft_setup import setup_peft_model
from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient
from src.distributed.runtime import barrier, broadcast_from_rank0, is_global_main_process
from src.environments.base import resolve_reasoning_effort
from src.models.loading.model_preparation import log_model_info
from src.rewards.functions import ScorerRewardFunction, reward_functions
from src.rewards.verifiable import RLVR_GRADERS
from src.trainers.distillation.sdpg import DistributedSDPGTrainer
from src.trainers.grpo.online import DistributedGRPOTrainer
from src.trainers.grpo.rollout.weight_sync_clients import (
    verify_context_window_synced,
    verify_sampler_logprob_reference_synced,
)
from src.training.environment import run_training
from src.training.parser import H4ArgumentParser
from src.training.script_runner import (
    apply_distributed_trainer_config,
    apply_prompt_completion_window,
    build_training_callbacks,
    distributed_trainer_kwargs,
    init_training_script,
    load_script_datasets,
    load_script_model,
    log_script_dataset_examples,
    padded_workload_attn_implementation,
    reject_non_default_args,
    reject_unsupported_args,
    run_trainer,
)


def main():
    parser = H4ArgumentParser((RLVROnlineGRPOScriptArguments, GRPOConfig, ModelConfig, DistributedArguments))
    args, grpo_config, model_config, dist_args = parser.parse()

    # The weight sync forwards trainer parameter names verbatim, and the text-only CausalLM sibling
    # spells its decoder model.layers.* where the multimodal checkpoint the server loads spells
    # model.language_model.layers.*, so every dense tensor would miss its slot on the first sync.
    reject_unsupported_args("RLVR Online GRPO", text_only_model=dist_args.text_only_model)
    # Each block's tunables reach the trainer only through its gate, so a value set beside a closed
    # gate is inert (its defaults are truthy, hence the default-comparing form).
    if not args.use_rlrr:
        reject_non_default_args("RLVR Online GRPO with use_rlrr off", args, *RLRRArguments.TUNABLES)
    if not args.use_sdpg:
        reject_non_default_args("RLVR Online GRPO with use_sdpg off", args, *args.SDPG_TUNABLES)

    runtime = init_training_script(
        args,
        grpo_config,
        model_config,
        dist_args,
        script_prefix="rlvr-online-grpo",
        supports_cp=False,
        supports_pp=False,
    )
    parallelism_config = runtime.parallelism_config

    # Prompts and completions are collated into padded batches.
    requested_attn = padded_workload_attn_implementation(model_config, sinks_reset=dist_args.reset_sinks)
    model, tokenizer = load_script_model(
        runtime, grpo_config, model_config, dist_args, attn_implementation=requested_attn
    )

    # max_completion_length is the generation budget here (TRL passes it to vLLM), so it is an explicit
    # hyperparameter; max_prompt_length is a dataset filter (rows over it are dropped, not truncated).
    tokenizer, _ = apply_prompt_completion_window(
        args,
        model,
        tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=grpo_config.max_completion_length,
        completion_budget_required=True,
    )

    peft_config = setup_peft_model(args, model, model_config, "CAUSAL_LM")
    log_model_info(model, tokenizer)

    def process_for_rlvr(row):
        """Process a dataset row for RLVR training.

        Extracts the prompt and ground truth answer from the dataset row.
        Supports both conversational (list of messages) and string prompts.
        """
        prompt_data = row[args.prompt_field]
        answer_data = row.get(args.answer_field)

        if isinstance(prompt_data, str):
            prompt_data = [{"role": "user", "content": prompt_data}]
        elif not isinstance(prompt_data, list):
            raise ValueError(f"Invalid prompt format: {type(prompt_data)}")
        # The system prompt leads only a conversation that does not already open with one; an
        # unconditional insert stacks two.
        messages = fold_system_into_conversation(
            list(prompt_data), args.system_prompt, model_supports_system_role=True
        )

        template_kwargs = chat_template_kwargs(row, interleaved_thinking=False, tools_field=args.tools_field)
        # Reasoning-effort steer ("random" → a level sampled per prompt). Single-turn, so per-prompt is
        # the per-episode analogue of the env-GRPO rollout (src/environments/base.py).
        effort = resolve_reasoning_effort(args.reasoning_effort)
        if effort is not None:
            template_kwargs["reasoning_effort"] = effort

        formatted_prompt = render_generation_prompt(
            tokenizer, messages, max_prompt_length=args.max_prompt_length, **template_kwargs
        )
        # Over-budget prompts are dropped with a blank-string sentinel, not None: an all-rejected first writer
        # batch makes Arrow infer a null column and crash casting later real string batches.
        if formatted_prompt is None:
            return {"prompt": "", "answer": ""}

        return {
            "prompt": formatted_prompt,
            "answer": str(answer_data) if answer_data is not None else "",
        }

    # Pre-sharded datasets are split per DP rank at load; the trainer gets dataset_presharded so it
    # does not re-shard (no-op for the usual raw prompt/answer dataset).
    ds, dataset_presharded = load_script_datasets(
        args,
        parallelism_config,
        conversation_field=None,  # RLVR uses prompt_field/answer_field, not conversation
    )
    reject_image_columns(ds, "RLVR Online GRPO")

    original_columns = list(ds["train"].column_names)
    # A mistyped prompt/answer field yields empty answers and so all-zero verifiable rewards. Both are
    # checked here because this path declares no conversation_field for the loader to validate.
    for knob, column in [("prompt_field", args.prompt_field), ("answer_field", args.answer_field)]:
        if column:
            require_render_column(ds, str(args.dataset), knob, column)
    columns_to_remove = [col for col in original_columns if col not in ["prompt", "answer"]]

    processed_ds = process_dataset_with_map_and_filter(
        ds,
        process_for_rlvr,
        filter_field="prompt",
        remove_columns=columns_to_remove,
        desc="Processing dataset for RLVR Online GRPO",
        cache_key_extras={
            "system_prompt": args.system_prompt,
            "tools_field": args.tools_field,
            "prompt_field": args.prompt_field,
            "answer_field": args.answer_field,
            "max_prompt_length": args.max_prompt_length,
            # Steers the prompt at map time, and the closure holding it is a dataclass the cache
            # fingerprint skips, so a changed effort would otherwise reuse the old rendering.
            "reasoning_effort": args.reasoning_effort,
        },
    )

    train_dataset = processed_ds["train"]
    eval_dataset = processed_ds.get("test", None)

    log_script_dataset_examples({"train": train_dataset, "test": eval_dataset}, tokenizer, args, grpo_config)

    # One TRL reward function per configured term, weighted by the term; process_for_rlvr renders the
    # ground truth into the "answer" column, which is what a judge term reads as the reference.
    reward_funcs, reward_weights = reward_functions(args.reward_terms, RLVR_GRADERS, reference_column="answer")
    # A judge or reward-model term is probed once before the trainer exists: a bad URL, key or model
    # would otherwise score every row None and train on nothing. Rank 0 probes, all ranks raise together.
    probe_error: str | None = None
    if is_global_main_process():
        for function in reward_funcs:
            if isinstance(function, ScorerRewardFunction):
                try:
                    asyncio.run(function.scorer.verify())
                except Exception as e:
                    probe_error = f"reward term {function.term.name!r} probe failed: {e}"
                    break
    probe_error = broadcast_from_rank0(probe_error)
    if probe_error is not None:
        raise RuntimeError(probe_error)

    # Same base-URL precedence as TRL's generation client: vllm_server_base_url wins over host:port,
    # so the probe hits the server the trainer will actually generate against.
    vllm_base_url = (
        grpo_config.vllm_server_base_url or f"http://{grpo_config.vllm_server_host}:{grpo_config.vllm_server_port}"
    )
    verify_context_window_synced(
        [vllm_base_url],
        single_turn_tokens=(args.max_prompt_length or 0) + grpo_config.max_completion_length,
        backend=VLLMWeightSyncClient.BACKEND_KEY,
    )
    # TRL's IS correction divides by the engine's logprobs too: they must carry the sampling temperature,
    # and a sequence-level IS mode sums the per-token log-ratios, which a nucleus-renormalized reference
    # drives toward a zero sequence weight.
    verify_sampler_logprob_reference_synced(
        [vllm_base_url],
        temperature=grpo_config.temperature,
        top_p=grpo_config.top_p,
        sequence_ratio_active=DistributedGRPOTrainer.sequence_level_importance_sampling(grpo_config),
        backend=VLLMWeightSyncClient.BACKEND_KEY,
    )

    grpo_config.reward_weights = reward_weights
    apply_distributed_trainer_config(grpo_config, parallelism_config)

    barrier()

    # --use_sdpg swaps in the SDPG trainer (GRPO loss plus privileged-teacher reverse-KL OPD on
    # positive-advantage rollouts). It reuses the same rollout and verifier machinery, so only the
    # class and the OPD kwargs differ.
    trainer_cls = DistributedSDPGTrainer if args.use_sdpg else DistributedGRPOTrainer
    callbacks = build_training_callbacks(
        args,
        grpo_config,
        model,
        parallelism_config,
        policy_gradient_loss=True,
        syncs_to_external_generator=True,
    )
    trainer = trainer_cls(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
        **distributed_trainer_kwargs(args, dist_args, parallelism_config, dataset_presharded=dataset_presharded),
        rlrr_config=args.build_rlrr_config(),
        advantage_shaping=args.build_advantage_shaping(),
        drop_degenerate_groups=args.drop_degenerate_groups,
        scale_rewards_std_floor=args.scale_rewards_std_floor,
        # Chunked log-probs (avoids full [B,T,vocab] logits on long completions)
        use_chunked_grpo_logprobs=args.use_chunked_grpo_logprobs,
        # Persist completions parquet decoupled from console log_completions
        save_completions=args.save_completions,
        **args.build_sdpg_kwargs(),
    )
    run_trainer(
        trainer,
        runtime,
        method_name="RLVR Online GRPO",
        extra_start_log=[
            f"Reward functions: {[f.__name__ for f in reward_funcs]}",
            f"Reward weights: {reward_weights}",
        ],
    )


if __name__ == "__main__":
    run_training(main)()
