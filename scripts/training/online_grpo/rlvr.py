#!/usr/bin/env python
"""RLVR (Reinforcement Learning with Verifiable Rewards) online GRPO training.

Online GRPO against a vLLM server, scored by rule-based reward functions instead of a reward model:
``accuracy_reward`` (strict `\\boxed{}` exact-match) and the optional ``format_reward`` (regex).

Supported Parallelism Modes: EP, TP, ETP (CP is not supported by ``DistributedGRPOTrainer``; the
rollout server takes its own GPUs, so size the launch to the remaining ones).

Usage:
    torchrun --nproc_per_node=8 scripts/training/online_grpo/rlvr.py \\
        examples/grpo/online/rlvr-online-grpo-template.yaml --expert_parallel_size=8
"""

import re

from trl import GRPOConfig, ModelConfig

from src.args.distributed_args import DistributedArguments
from src.args.rlvr_online_grpo_args import RLVROnlineGRPOScriptArguments
from src.data.pipeline.conversation import maybe_parse_json
from src.data.pipeline.processing import process_dataset_with_map_and_filter, require_render_column
from src.data.pipeline.rendered import render_generation_prompt
from src.data.sources.loading import reject_image_columns
from src.distributed.loading.peft_setup import setup_peft_model
from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient
from src.distributed.runtime import barrier
from src.environments.base import resolve_reasoning_effort
from src.environments.rewards import extract_last_boxed
from src.models.loading.model_preparation import log_model_info
from src.trainers.distillation.sdpg import DistributedSDPGTrainer
from src.trainers.grpo.online import DistributedGRPOTrainer
from src.trainers.grpo.rollout.weight_sync_clients import verify_context_window_synced
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
    reject_unsupported_args,
    run_trainer,
)


def _completion_content(completion) -> str:
    """Last assistant message content for a conversational completion, or the string itself."""
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


def accuracy_reward(completions, answer, **kwargs):
    """Reward based on whether the completion's \\boxed{} answer matches ground truth.

    Strict boxed exact-match, not the environments' validated-answer chain (``compute_answer_reward``,
    ``DEFAULT_METHODS = exact + numeric``). The two are not interchangeable: this one strips a GSM8K
    ``####`` rationale prefix and ``,``/``$`` from both sides and then requires string equality, while
    the environment chain normalizes case, re-extracts a box from the ground truth as well, and
    accepts a numeric match within ``rtol=0.01``. Swapping in the chain would re-grade every shipped
    RLVR recipe (``0.5`` vs ``.5``, ``7 \\boxed{42}`` vs ``42``), so the difference is pinned by
    ``tests/cpu/grpo/test_rlvr_accuracy_reward.py``.

    Args:
        completions: List of completion strings.
        answer: List of ground truth answer strings.

    Returns:
        List of float rewards (1.0 or 0.0).
    """
    # strict: a short answer column would otherwise truncate and return fewer rewards than
    # completions, grading a different set of rows than was generated.
    rewards = []
    for completion, gt in zip(completions, answer, strict=True):
        content = _completion_content(completion)

        extracted = (extract_last_boxed(content) or "").strip()

        # Normalize ground truth. GSM8K stores the rationale plus "#### <final>";
        # keep only the final answer so the boxed comparison is meaningful.
        gt_normalized = str(gt)
        if "####" in gt_normalized:
            gt_normalized = gt_normalized.rsplit("####", 1)[-1]
        gt_normalized = gt_normalized.strip().replace(",", "").replace("$", "")
        extracted = extracted.replace(",", "").replace("$", "")
        # An empty extraction is never correct, even against a blank ground truth: a missing answer
        # column would otherwise pay full reward for every box-less completion.
        rewards.append(1.0 if extracted and extracted == gt_normalized else 0.0)

    return rewards


def format_reward(completions, pattern, **kwargs):
    """Reward based on whether the completion matches a regex format pattern.

    Args:
        completions: List of completion strings.
        pattern: Compiled regex pattern (the caller compiles the configured pattern once).

    Returns:
        List of float rewards (1.0 or 0.0).
    """
    rewards = []
    for completion in completions:
        content = _completion_content(completion)
        rewards.append(1.0 if pattern.search(content) else 0.0)

    return rewards


def main():
    parser = H4ArgumentParser((RLVROnlineGRPOScriptArguments, GRPOConfig, ModelConfig, DistributedArguments))
    args, grpo_config, model_config, dist_args = parser.parse()

    # The weight sync forwards trainer parameter names verbatim, and the text-only CausalLM sibling
    # spells its decoder model.layers.* where the multimodal checkpoint the server loads spells
    # model.language_model.layers.*, so every dense tensor would miss its slot on the first sync.
    reject_unsupported_args("RLVR Online GRPO", text_only_model=dist_args.text_only_model)

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

        if isinstance(prompt_data, list):
            # Conversational rows: inject the system prompt only when the conversation does not already
            # open with one (matches environmental_grpo); an unconditional insert stacks two.
            messages = list(prompt_data)
            if args.system_prompt and not (messages and messages[0].get("role") == "system"):
                messages.insert(0, {"role": "system", "content": args.system_prompt})
        elif isinstance(prompt_data, str):
            messages = []
            if args.system_prompt:
                messages.append({"role": "system", "content": args.system_prompt})
            messages.append({"role": "user", "content": prompt_data})
        else:
            raise ValueError(f"Invalid prompt format: {type(prompt_data)}")

        template_kwargs = {}
        if args.tools_field and args.tools_field in row:
            tools = maybe_parse_json(row[args.tools_field])
            if tools is not None:
                template_kwargs["tools"] = tools
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

    reward_funcs = []
    reward_weights = []

    if args.use_accuracy_reward:
        reward_funcs.append(accuracy_reward)
        reward_weights.append(args.accuracy_reward_weight)

    if args.use_format_reward:
        compiled_pattern = re.compile(args.format_pattern, re.DOTALL)

        def _format_reward(completions, **kwargs):
            return format_reward(completions, pattern=compiled_pattern, **kwargs)

        _format_reward.__name__ = "format_reward"
        reward_funcs.append(_format_reward)
        reward_weights.append(args.format_reward_weight)

    if not reward_funcs:
        raise ValueError(
            "At least one reward function must be enabled. Set use_accuracy_reward=True or use_format_reward=True."
        )

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
