# Async GRPO with Environments

Multi-turn RL: the model converses with an environment — tools, code, repositories, search — until a final reward, then trains with GRPO. Use it for tool use, agentic training and multi-step reasoning with environment feedback. For pre-collected rewards use [Offline GRPO](../offline-grpo.md); for single-turn verifiable rewards, [Online GRPO](../online-grpo.md); for pairwise preferences, [SMPO](../../preference/smpo.md) or [DPO](../../preference/dpo.md).

The episode reward is the environment's grade priced by the `rewards:` terms, plus the environment's own shaping ([Reward Terms](../rewards.md#environment-arm)). Pick an environment with `environment_type` or register your own ([Environments](../environments/README.md)). Trainer `DistributedAsyncEnvironmentalGRPOTrainer`, script `scripts/training/environmental_grpo.py`. A [rollout server](../../../infrastructure/rollout-servers.md) must be serving before you launch; [Ray](../../../infrastructure/ray.md) configures itself on one node.

## How a step works

1. The trainer pushes its weights to the engines over NCCL, then hands one prompt per batch row to `RolloutManager`, which dispatches round-robin across Ray actors and server URLs.
2. Each actor holds its own environment instance: POST `/v1/chat/completions`, parse the reply and its `tool_calls`, run the step, repeat until done or `max_turns`.
3. The environment grades each trajectory; the trainer builds training rows from the engine's sampled tokens and computes the GRPO loss.

![One async GRPO step: an NCCL weight push to every rollout engine, then a rank's round of 24 rows (3 prompts × 8 generations) handed to RolloutManager and dispatched round-robin over Ray environment actors and the vLLM or SGLang servers by POST /v1/chat/completions; each row runs a multi-turn episode of generate → parse tool_calls → execute → observe until a final answer, max_turns or spent recoveries, after which the environment grades the trajectory, every assistant turn becomes a training row of the engine's sampled ids, the group's 8 rows share one advantage, and the GRPO loss drives the optimizer step](../../../assets/diagrams/batch_rollout_pipeline.png)

## Configuration surface

One YAML, six dataclasses.

| Knob family | Class |
|---|---|
| `environment_type`, `rewards`, `max_turns`, `environment_kwargs` | `EnvironmentConfig` ([fields](../../../reference/configuration-reference.md#environmentconfig)) |
| Ray, servers, sync cadence, prefetch, rollout sampling, trust region | `AsyncTrainingConfig` ([fields](../../../reference/configuration-reference.md#asynctrainingconfig)) |
| Dataset columns, prompt filter, `save_completions` | `EnvironmentalGRPOScriptArguments` |
| Optimizer, batch, `beta`, `num_generations`, `vllm_group_port` | TRL `GRPOConfig` |
| Model id, LoRA | `ModelConfig` |
| Parallel sizes, `reset_sinks` | `DistributedArguments` |

## Dataset

`prompt` is the row; `answer` is carried whenever the dataset has it. The environment decides whether it is required: `code_contests` / `codeforces` read their hidden-test payload from it, and `exam_qa`, `qa_search` and the `react_*` presets grade the final answer against it, so a dataset without the column is refused at trainer construction, as is one for [`swe`](../environments/swe-environment.md#reward) unless judge-only. `native_*` and `mcp` need none.

Rename columns with `prompt_field` / `answer_field`, forward extras with `context_fields`. A `context_field`, or an `answer_field` renamed away from `answer`, must name a real column: an unknown one raises at startup rather than yielding answer-less rows.

The trainer forces `remove_unused_columns: false` at construction (warning when a config sets it true): the rollout context **is** the row minus `prompt`, and column pruning would strip `answer` and every `context_fields` column.

A `prompt` given as a message list reduces to its **last `user` turn** — the environment is handed the task as text and builds the conversation itself. A conversation with no `user` turn fails the batch on every rank.

The environment owns the system turn and the tool schema: this surface has no `system_prompt` field, and `tools_field` is rejected. Where an environment accepts an override it is `environment_kwargs.system_prompt` — the ReAct ones hardcode theirs and drop the key.

## Quickstart

Start a server on a GPU the trainer will not use, wait for `/health`, launch:

```bash
VLLM_MODEL=Qwen/Qwen3-4B-Instruct-2507 VLLM_CUDA_DEVICES=7 \
    docker compose -f docker-compose.vllm.yml up -d vllm-server

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \
    scripts/training/environmental_grpo.py \
    examples/grpo/environmental/environmental-grpo-template.yaml
```

`halo launch environmental-grpo <config> --nproc 7` builds the same line; any field is overridable on the command line (`--learning_rate=1e-6`).

Checkpoints land in `<output_dir>/checkpoint-<step>/`, rollouts in `<output_dir>/completions/`.

## From Python

```python
trainer = DistributedAsyncEnvironmentalGRPOTrainer(
    model=model,
    args=grpo_config,                                # TRL GRPOConfig
    train_dataset=train_dataset,
    processing_class=tokenizer,
    environment_config=EnvironmentConfig(environment_type="react_math"),
    async_config=AsyncTrainingConfig(rollout_server_url="http://localhost:8000"),
    peft_config=lora_config,                         # optional
)
```

For an unregistered `BaseEnvironment` subclass pass `environment_cls=MyEnvironment` with `environment_kwargs={...}` instead. One of the two is required.

## Testing a setup

Run the environment on a few rows first. `--training_config` samples under the training YAML's own contract: environment, template variables, stop tokens, thinking budget, sampling.

```bash
python scripts/environments/inference/run_env.py \
    --training_config examples/grpo/environmental/environmental-grpo-template.yaml \
    --dataset <hub-id> --split test --num_examples 20 \
    --base_url http://localhost:8000/v1 --model <served-model-id>
```

`pytest tests/cpu/environments tests/cpu/grpo -m cpu` covers the construction gates and row accounting. `make test-gpu-vllm` / `make test-gpu-sglang` run the end-to-end suites (`tests/gpu/trainers/grpo/test_env_grpo_vllm_e2e.py`, `test_env_grpo_sglang_e2e.py`) against a live server.

## Pages

- [Servers and Launch](setup.md) — GPU split, backends, weight sync, multi-node
- [Rollout Configuration](rollouts.md) — turn budgets, tool calls, reasoning, sampled tokens
- [Objective and Stability](objective.md) — importance sampling, trust region, advantages, KL
- [Memory and Throughput](performance.md) — the logits wall, chunked log-probs, sizing
- [Metrics and Troubleshooting](monitoring.md) — what to watch, what failures mean
- [Environments](../environments/README.md) — the registered tasks, tools and rewards
