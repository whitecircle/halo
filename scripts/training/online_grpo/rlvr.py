#!/usr/bin/env python
"""RLVR (Reinforcement Learning with Verifiable Rewards) online GRPO training.

Online GRPO against a vLLM server, scored by the ``rewards`` terms of the config: the strict
``\\boxed{}`` accuracy grader, the regex format grader, a generative judge, a served reward model.

Supported Parallelism Modes: EP, TP, ETP (CP is not supported by ``DistributedGRPOTrainer``; the
rollout server takes its own GPUs, so size the launch to the remaining ones).

Usage:
    torchrun --nproc_per_node=<GPUs left after the server> scripts/training/online_grpo/rlvr.py \\
        examples/grpo/online/rlvr-online-grpo-template.yaml
"""

from trl import GRPOConfig, ModelConfig

from src.args.distributed_args import DistributedArguments
from src.args.mixins import RLRRArguments
from src.args.rlvr_online_grpo_args import RLVROnlineGRPOScriptArguments
from src.data.pipeline.conversation import as_conversation, chat_template_kwargs, fold_system_into_conversation
from src.data.pipeline.processing import process_dataset_with_map_and_filter, require_render_column
from src.data.pipeline.rendered import render_generation_prompt
from src.data.pipeline.row_processors import blank_conversation
from src.data.sources.loading import reject_image_columns
from src.distributed.loading.frozen_models import load_reference_model_for_on_policy_grpo
from src.distributed.loading.peft_setup import setup_peft_model
from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient
from src.distributed.runtime import barrier
from src.environments.base import resolve_reasoning_effort
from src.models.loading.model_preparation import log_model_info
from src.rewards.functions import ScorerRewardFunction, reward_functions
from src.rewards.graders.verifiable import RLVR_GRADERS
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
    verify_backend_on_rank0,
)


def build_rlvr_row_fn(args: RLVROnlineGRPOScriptArguments, tokenizer):
    """Row transform for the RLVR map: the rendered generation prompt, the conversation it was rendered
    from and the ground-truth answer. Accepts conversational (list of messages) and string prompts."""

    def process_for_rlvr(row):
        prompt_data = as_conversation(row[args.prompt_field])
        answer_data = row.get(args.answer_field)

        if not isinstance(prompt_data, list):
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
        # Over-budget prompts are dropped by their blank prompt; every column keeps its real type, since
        # an all-rejected first writer batch otherwise fixes a null column the real batches cannot cast to.
        if formatted_prompt is None:
            return {"prompt": "", "conversation": blank_conversation(messages), "answer": ""}

        # The scored terms read the conversation itself, not the rendered template the engine takes.
        return {
            "prompt": formatted_prompt,
            "conversation": messages,
            "answer": str(answer_data) if answer_data is not None else "",
        }

    return process_for_rlvr


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

    # --use_sdpg swaps in the SDPG trainer (GRPO loss plus privileged-teacher reverse-KL OPD on
    # positive-advantage rollouts). It reuses the same rollout and verifier machinery, so only the
    # class and the OPD kwargs differ.
    trainer_cls = DistributedSDPGTrainer if args.use_sdpg else DistributedGRPOTrainer
    runtime = init_training_script(
        args,
        grpo_config,
        model_config,
        dist_args,
        script_prefix="rlvr-online-grpo",
        trainer_cls=trainer_cls,
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

    # Pre-sharded datasets are split per DP rank at load; the trainer gets dataset_presharded so it
    # does not re-shard (no-op for the usual raw prompt/answer dataset). The prompt column is the
    # loader's render column: it must exist, and rows with an empty prompt are dropped.
    ds, dataset_presharded = load_script_datasets(
        args,
        parallelism_config,
        conversation_field=args.prompt_field,
        conversation_knob="prompt_field",
    )
    reject_image_columns(ds, "RLVR Online GRPO")

    original_columns = list(ds["train"].column_names)
    # Read with .get, so a mistyped answer_field would yield empty answers and all-zero verifiable rewards.
    if args.answer_field:
        require_render_column(ds, str(args.dataset), "answer_field", args.answer_field)
    columns_to_remove = [col for col in original_columns if col not in ["prompt", "conversation", "answer"]]

    processed_ds = process_dataset_with_map_and_filter(
        ds,
        build_rlvr_row_fn(args, tokenizer),
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

    # One TRL reward function per configured term, weighted by the term; process_for_rlvr keeps the
    # conversation a scorer reads and renders the ground truth into "answer", a judge's reference.
    reward_funcs, reward_weights = reward_functions(
        args.reward_terms, RLVR_GRADERS, prompt_column="conversation", reference_column="answer"
    )
    # A judge or reward-model term is probed before the trainer exists: a bad URL, key or model would
    # otherwise score every row None and train on nothing.
    for function in reward_funcs:
        if isinstance(function, ScorerRewardFunction):
            verify_backend_on_rank0(function.scorer.verify, f"reward term {function.term.name!r}")

    # TRL's generation client's base-URL precedence (a set vllm_server_base_url wins over host:port),
    # so the probe hits the server the trainer will actually generate against.
    vllm_base_url = (
        grpo_config.vllm_server_base_url
        if grpo_config.vllm_server_base_url is not None
        else f"http://{grpo_config.vllm_server_host}:{grpo_config.vllm_server_port}"
    )
    verify_context_window_synced(
        [vllm_base_url],
        single_turn_tokens=(args.max_prompt_length or 0) + grpo_config.max_completion_length,
        backend=VLLMWeightSyncClient.BACKEND_KEY,
    )
    # TRL's IS correction divides by the engine's logprobs too: they must carry the sampling temperature,
    # and a sequence-level IS mode sums the per-token log-ratios, which a reference renormalized over the
    # sampler's cut drives toward a zero sequence weight. TRL's unset min_p (None) samples with no cut.
    verify_sampler_logprob_reference_synced(
        [vllm_base_url],
        temperature=grpo_config.temperature,
        top_p=grpo_config.top_p,
        top_k=grpo_config.top_k,
        min_p=0.0 if grpo_config.min_p is None else grpo_config.min_p,
        repetition_penalty=grpo_config.repetition_penalty,
        sequence_ratio_active=DistributedGRPOTrainer.sums_sequence_logratio(grpo_config),
        backend=VLLMWeightSyncClient.BACKEND_KEY,
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
        reset_sinks=dist_args.reset_sinks,
        attn_default=requested_attn,
    )

    grpo_config.reward_weights = reward_weights
    apply_distributed_trainer_config(grpo_config, parallelism_config)

    barrier()

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
        ref_model=ref_model,
        callbacks=callbacks,
        **distributed_trainer_kwargs(args, dist_args, parallelism_config, dataset_presharded=dataset_presharded),
        rlrr_config=args.build_rlrr_config(),
        drop_degenerate_groups=args.drop_degenerate_groups,
        scale_rewards_std_floor=args.scale_rewards_std_floor,
        balance_token_mass=args.balance_token_mass,
        early_stop=args.build_early_stop(),
        use_chunked_grpo_logprobs=args.use_chunked_grpo_logprobs,
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
