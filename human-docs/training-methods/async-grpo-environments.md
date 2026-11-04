# Async GRPO with Environments

This is the multi-turn arm of GRPO. The model works inside an environment (calling tools, running code,
searching, editing a workspace) until the episode ends. The environment grades the whole trajectory, and
GRPO trains on the group of episodes that shared a prompt.

Pick it when the reward comes from what the model *did*. Single-turn verifiable rewards are cheaper on
[Online GRPO (RLVR)](online-grpo.md), and pre-scored data belongs on [Offline GRPO](offline-grpo.md).

Rollouts run on a **separate vLLM or SGLang container**, driven by **Ray actors** that hold the
environments and their tools. Ray configures itself on one node, so the server is the only thing you start
by hand ([Rollout Servers](../rollout-servers.md)).

![One async GRPO step: a weight push to every engine, episodes run over Ray actors and the servers until done or
max_turns, the environment grades each one, every assistant turn becomes a training row sharing its episode's
advantage, and the loss drives the optimizer step](../../agent-docs/assets/diagrams/batch_rollout_pipeline.png)

## The environments

`environment_type` picks one: `react_math` and `react_search` (plain-text actions), `native_math`,
`native_coding` and `native_combined` (native tool calls), `qa_search`, `exam_qa`, `swe` (an
edit-run-test loop over a workspace), `code_contests`, `codeforces`, or `mcp` (whatever tools an MCP
server advertises). The
[registry](../../agent-docs/training-methods/async-grpo/environments/README.md#registry) ↗ lists each
one's tools and turn budget. An environment that grades against `answer` refuses a dataset without that
column at startup
([`requires_answer`](../../agent-docs/training-methods/async-grpo/environments/README.md#common-knobs) ↗).

The Ray actor hosts need what their environment needs: a sandbox and a large `TMPDIR` for the code
environments, outbound network and usually a search API key for the search ones, and the MCP server's
launcher on `PATH` for `mcp`. Your own environment class registers alongside these
([Custom Environments](../../agent-docs/training-methods/async-grpo/environments/custom.md) ↗).

> [!WARNING]
> The default `local` sandbox does not confine the program it runs. Policy-written code can read the
> trainer's environment, `--env-file` secrets included. For RL on untrusted code, set
> `sandbox_backend: bubblewrap` (or `remote`) under `environment_kwargs`. `bubblewrap` needs extra
> container rights
> ([Sandboxes](../../agent-docs/training-methods/async-grpo/environments/sandbox.md#choosing-a-backend) ↗).

## Data

`prompt` is the task, and `answer` the ground truth where the environment grades against one. Rename
either with `prompt_field` / `answer_field`, and pass extra columns through `context_fields`.

```jsonl
{"prompt": "Solve: ...", "answer": "42"}
```

A `prompt` given as a message list is reduced to its **last user turn**. The environment owns the system
prompt and the tool schema, so this method has no `system_prompt` field.

For code contests, `halo run prepare-code-dataset` builds the pool
([Building a pool](../../agent-docs/training-methods/async-grpo/environments/tasks/code-contests/datasets.md#building-a-pool) ↗).

## Rewards

The episode reward is the environment's grade priced by an `environment` term, plus the environment's own
shaping, plus any external term you add:

```yaml
rewards:
  - source: environment    # the env's grade in [0, 1]: a solved problem, an answer match, a passing test
```

`judge` and `reward_model` terms take the same options as on the [online arm](online-grpo.md#rewards).
Only this arm also accepts a **veto judge**, a `judge` that lists `checks`: a fired veto check strips the
episode of its credit
([Veto judge](../../agent-docs/training-methods/grpo/rewards/judge.md#veto-judge) ↗).

Every code-contests recipe audits the whole episode this way, vetoing hacks and tool misuse. These recipes
set `on_error: invalid`, so a judge outage trains nothing, and a second step in a row with no valid
episode stops the run
([Audit judge](../../agent-docs/training-methods/async-grpo/environments/tasks/code-contests/reward.md#audit-judge) ↗).

Each component logs as `reward/<name>` (`reward/objective` for the environment term)
([Reward Terms](../../agent-docs/training-methods/grpo/rewards/README.md) ↗).

## Config

From `examples/grpo/environmental/qwen3_5/vllm/qwen3.6-35b-a3b-react-math-full-ep4.yaml`: four trainer
GPUs, and one server on the other four at the default `rollout_server_url`, `http://localhost:8000`.

```yaml
model_name_or_path: Qwen/Qwen3.6-35B-A3B
expert_parallel_size: 4
environment_type: react_math
max_turns: 10
num_rollout_workers: 16            # Ray environment actors per training rank
rollout_max_tokens: 8192           # budget for ONE turn, not the trajectory
enable_prefetch: false             # one server; true overlaps rollout with training
num_generations: 8                 # episodes per prompt
beta: 0.0                          # no KL anchor, so no second model in memory
scale_rewards: batch
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-06
```

`examples/grpo/environmental/environmental-grpo-template.yaml` is the annotated starting point. GPT-OSS,
Qwen3.5/3.6 and Gemma 4 ship recipes under `examples/grpo/environmental/<family>/{vllm,sglang}/`; the
SGLang ones are `ep1` only.

- **Sampling** goes through the `rollout_*` fields. TRL's `top_p`, `top_k`, `min_p`,
  `repetition_penalty` and `generation_kwargs` reach no sampler here, and setting one raises at startup.
- **`beta`** above 0 on a full fine-tune or native expert-only LoRA loads a frozen reference: under EP,
  ETP or TP a whole unsharded copy on every rank. A PEFT policy is its own reference with the adapter
  off.

Three decisions matter more than the rest:

- **Turn budget.** `rollout_max_tokens` caps one turn, `max_turns` the number of turns, and
  `rollout_max_episode_tokens` what one episode samples in all (off by default; the code-contests recipes
  set it). A row longer than the context window fails the step. If `episode/turns` sits at the cap, raise
  `max_turns`; if it sits far below, lower it
  ([Trajectory length](../../agent-docs/training-methods/async-grpo/episodes.md#trajectory-length) ↗).
- **Reasoning effort.** `environment_kwargs.reasoning_effort` (`low` / `medium` / `high` / `random`) sets
  how much the model should think. The model sees the level only through a chat template that renders it:
  pin one with `chat_template:` plus `force_chat_template: true`, and serve the same file
  ([Reasoning Budgets](../../agent-docs/training-methods/async-grpo/reasoning.md) ↗).
- **Tool budgets.** `tool_success_reward`, `tool_error_penalty` and `tool_reward_cap` price tool calls. Keep
  them small beside the objective, or calling tools pays better than finishing the task.

A turn the engine cut at its cap, an empty turn, or one whose every tool call was invalid earns no reward.
It trains only as a penalty, when its episode scored below the group's mean
([Untrainable turns](../../agent-docs/training-methods/async-grpo/objective.md#untrainable-turns) ↗).

### Thinking budgets

`reasoning_effort_profiles` gives each level a per-turn `thinking_tokens` budget, which the engine enforces
by default. `enforce_thinking_cap: false` makes it a soft target stated in the template, closed only at the
run's ceiling. Engine caps bind on vLLM only; `rollout_backend: sglang` refuses
`rollout_max_thinking_tokens` and `carry_reasoning`
([Enforced and soft budgets](../../agent-docs/training-methods/async-grpo/reasoning.md#enforced-and-soft-budgets) ↗).

`reasoning_price` and `reasoning_floor` charge long and short reasoning in the trainer. Neither is charged in
a group where no episode earned the objective, unless `reasoning_terms_in_failed_groups: true`
([Reasoning length reward](../../agent-docs/training-methods/async-grpo/reasoning.md#reasoning-length-reward) ↗).

### One server or several

![One rollout server: trainer ranks on GPUs 0-6, the engine on GPU 7 in the trainer's NCCL group, the push pausing it, prefetch off by default, step time = sync + round + update](../../agent-docs/assets/diagrams/environmental_grpo_single_server.png)

With one server the step is serial by default: sync, generate, train. `enable_prefetch: true` overlaps the
next round with the update, training each round one step stale (the importance ratio corrects it).

![Two rollout servers: six trainer ranks, engines on GPUs 6 and 7, one NCCL group per server on ports 51216 and 51217, prefetch one round deep, step time = sync + max(round, update)](../../agent-docs/assets/diagrams/environmental_grpo_multi_server.png)

With two or more, list them in `rollout_server_configs`, each with its own `group_port`. Prefetch is
then on by default. `prefetch_depth` (default 1) collects that many rounds ahead, at that many steps of
staleness. On SGLang every sync aborts the turns in flight, so `max_retries` must cover the syncs a
prefetched round spans; the trainer refuses a smaller value.

## Run

```bash
make build-vllm   # or set VLLM_IMAGE to the public image
export HALO_SCRATCH=/path/to/large/volume   # or set it in .env; compose refuses to start without it
VLLM_MODEL=Qwen/Qwen3.6-35B-A3B VLLM_CUDA_DEVICES=4,5,6,7 VLLM_TP=4 VLLM_REASONING_PARSER=qwen3 \
VLLM_TOOL_CALLING_FLAGS= \
    docker compose -f docker-compose.vllm.yml up -d vllm-server
CUDA_VISIBLE_DEVICES=0,1,2,3 DIST_NCCL_TIMEOUT_MINUTES=60 halo launch grpo/environmental \
    examples/grpo/environmental/qwen3_5/vllm/qwen3.6-35b-a3b-react-math-full-ep4.yaml -n 4
```

A native-tool environment needs the server started with the family's
[tool-call parser](../rollout-servers.md#tool-parsers). ReAct environments serve without one
(`VLLM_TOOL_CALLING_FLAGS=`, as above).

Before a long run, put the config through a few rows, which exercises the parser, the template and the
sandbox or judge backend in a minute:

```bash
halo run eval-env --training_config <config>.yaml --dataset <hub-id-or-path> --split train \
    --answer_field <column> --num_examples 20 --base_url http://localhost:8000/v1 --model <served-id>
```

`--training_config` carries the environment and rollout settings, not the dataset fields, so match
`--split` (default `test`) and `--answer_field` to the config's dataset. Evaluation during training runs
the same loop: set `eval_strategy`, and `num_generations_eval: 1` for a fast pass@1 monitor
([Evaluation](../../agent-docs/training-methods/async-grpo/evaluation.md#during-training) ↗).

## What to watch

- `reward/objective` says whether the task is being learned. A rising total reward with a flat objective
  is shaping, not progress. Code contests also log `outcome/solve_rate`.
- `episode/turns` and `episode/truncation_rate` say whether episodes finish.
- `reward/within_group_std` near zero means a group's episodes all scored the same, so its prompt teaches
  nothing. `drop_degenerate_groups` (on by default) drops such groups, and
  `sampling/degenerate_group_frac` counts them.
- `sampling/logratio_mean` drifting steadily negative means the weight sync is broken and the policy
  trains on stale rollouts. If `advantage/net_token_mass` also stays negative and `entropy` climbs, a
  KL-free run is drifting instead: turn on `balance_token_mass` and the early stop.
- `episode/reward_scored` below 1 means a judge or reward model is failing; `<source>/<name>/scored`
  names which one. Under `on_error: invalid` those episodes also leave training, and
  `sampling/invalid_episode_frac` rises.

Rollouts land in `<output_dir>/completions/` as parquet, one directory per logged step holding a part
per rank. Every metric, including the prefetch hit rate:
[Monitoring](../../agent-docs/training-methods/async-grpo/monitoring.md) ↗. Symptoms and fixes:
[Troubleshooting](../../agent-docs/training-methods/async-grpo/troubleshooting.md) ↗.

## Sizing a run

Measure the engine's decode speed with the run's own prompts at the concurrency the run will use; nothing
in the config predicts it. Then set:

- `request_timeout`: at least twice one turn's budget at that speed.
- `episode_timeout`: the time for a full episode plus tool time, below the NCCL watchdog
  (`DIST_NCCL_TIMEOUT_MINUTES`), or startup raises.
- Serving GPUs: start from one per trainer GPU, one engine per GPU while the model and its KV cache fit.

Concurrency formula and worked numbers:
[Memory and Throughput](../../agent-docs/training-methods/async-grpo/performance.md#sizing-a-run) ↗.

## Go deeper

- [Async GRPO with Environments](../../agent-docs/training-methods/async-grpo/README.md) ↗ ·
  [Environments](../../agent-docs/training-methods/async-grpo/environments/README.md) ↗: servers, objective,
  metrics and every knob.
- [Code Contests](../../agent-docs/training-methods/async-grpo/environments/tasks/code-contests/README.md) ↗ ·
  [SWE](../../agent-docs/training-methods/async-grpo/environments/tasks/swe.md) ↗: the two deepest
  environments.
- [Rollout Servers](../rollout-servers.md) · [Online GRPO (RLVR)](online-grpo.md) ·
  [Training Methods](README.md)
