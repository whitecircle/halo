# Async GRPO with Environments

This is the multi-turn arm of GRPO: the model talks to an environment — calling tools, running code, searching,
editing a workspace — until the episode ends, the environment grades the whole trajectory, and GRPO trains on the
group of episodes that shared a prompt. Pick it when the reward comes from what the model *did*. Single-turn
verifiable rewards are cheaper on [Online GRPO (RLVR)](online-grpo.md), pre-scored data on
[Offline GRPO](offline-grpo.md).

Rollouts run on a **separate vLLM or SGLang container**, driven by **Ray actors** that hold the environments and
their tools. Ray configures itself on one node, so the only thing to start by hand is the server:
[Rollout Servers](../rollout-servers.md).

![One async GRPO step: a weight push to every engine, then a rank's round of rows dispatched round-robin over Ray
environment actors and the servers; each row runs generate, parse tool calls, execute, observe until done or
max_turns, the environment grades it, every assistant turn becomes a training row, the group shares one advantage,
and the loss drives the optimizer step](../../agent-docs/assets/diagrams/batch_rollout_pipeline.png)

Each step pushes weights, collects a round of episodes, turns the sampled tokens into training rows and steps.

## The environments

`environment_type` picks one; the last column says whether it grades against your dataset's `answer` column.

| `environment_type` | The task | Needs `answer` |
| --- | --- | --- |
| `react_math`, `react_search` | ReAct: the model writes `Thought:` / `Action:` as plain text; calculator and Python, or web search | yes |
| `native_math`, `native_coding`, `native_combined` | the same tools as native function calls; grades the row's `answer` when there is one, otherwise finishing the episode | no |
| `qa_search` | factual question answering with a web-search tool | yes |
| `exam_qa` | multiple-choice and open exams, closed-book unless `open_book: true` | yes |
| `swe` | edit-run-test loop over a workspace that survives across turns | no |
| `code_contests`, `codeforces` | write a program, try it in a scratchpad, submit it against hidden tests | yes |
| `mcp` | whatever tools an MCP server advertises | no |

Where the answer is required, a dataset without that column is refused at startup. Actor hosts need what their
environment needs: a sandbox and a roomy `TMPDIR` for the code ones, outbound network and usually a search API key
for the search ones, the MCP server's launcher on `PATH`. Your own class registers alongside them
([Custom Environments](../../agent-docs/training-methods/grpo/environments/custom-environments.md) ↗).

## Data

`prompt` is the task, `answer` the ground truth where the environment grades against it. Rename either with
`prompt_field` / `answer_field`, and pass extras an environment needs through `context_fields`.

```jsonl
{"prompt": "Solve: ...", "answer": "42"}
```

A `prompt` given as a message list is reduced to its **last user turn**: the environment owns the system prompt and
tool schema and builds the conversation itself, so this surface has no `system_prompt`.

## Rewards

The episode reward is the environment's grade priced by a term, plus its own shaping, plus any external term you add:

```yaml
rewards:
  - source: environment    # the env's grade in [0, 1]: pass fraction, answer match, adherence
    exponent: 2.0          # convex partial credit — half-right earns a quarter
```

The other two sources, `judge` and `reward_model`, are configured exactly as on the
[online arm](online-grpo.md#rewards). Every term logs under its own name (`reward/objective` for the environment
term), and the components sum exactly to the reward, so a rising reward always traces to the objective or to shaping
([Reward Terms](../../agent-docs/training-methods/grpo/rewards.md) ↗).

## Config

From `examples/grpo/environmental/qwen3_5/vllm/qwen3.6-35b-a3b-react-math-full-ep4.yaml` — four trainer GPUs, one
server on the other four. `examples/grpo/environmental/environmental-grpo-template.yaml` is the annotated starting
point; gpt-oss, Qwen3.5/3.6 and Gemma 4 ship recipes under
`examples/grpo/environmental/<family>/{vllm,sglang}/`, the SGLang ones in `ep1` form only.

```yaml
model_name_or_path: Qwen/Qwen3.6-35B-A3B
expert_parallel_size: 4
environment_type: react_math
max_turns: 10
num_rollout_workers: 16            # Ray environment actors per training rank
rollout_server_url: "http://localhost:8000"
vllm_group_port: 51216             # weight-sync port, bound on the trainer host
rollout_max_tokens: 8192           # budget for ONE turn, not the trajectory
enable_prefetch: false             # needs two or more servers
num_generations: 8                 # episodes per prompt
beta: 0.0                          # no KL anchor, so no second model in memory
scale_rewards: batch
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-06
```

Three decisions matter more than the rest.

- **Turn budget.** `rollout_max_tokens` caps one turn, `max_turns` the turns. The trajectory accumulates across turns
  and is never truncated — the context window bounds it, and a row past that fails the step. Watch `episode/turns`:
  pinned at the cap, raise it; far below, lower it, since turns are sequential and set step time.
- **Reasoning effort.** `environment_kwargs.reasoning_effort` (`low` / `medium` / `high` / `random`) steers how much
  the model thinks; `reasoning_effort_profiles` attaches per-level token and interaction caps, as the code-contests
  recipes do (`{high: {thinking_tokens: 16384, max_submissions: 3, max_test_calls: 6}}`). The engine-side cap
  `rollout_max_thinking_tokens` is vLLM-only. The level only reaches the policy if the chat template renders it:
  `jinja-templates/qwen3/qwen3.6-reasoning-effort.jinja` and `jinja-templates/gemma4/gemma4-reasoning-effort.jinja`
  state the level and its per-turn budget in the system block. Pin one with `force_chat_template: true` and serve the
  same file ([Chat template](../../agent-docs/training-methods/grpo/async-grpo/rollouts.md#chat-template)).
- **Tool budgets.** An environment pays `tool_success_reward` per successful call, charges `tool_error_penalty` per
  failure, and caps what successful calls earn across the episode — not the episode reward — at `tool_reward_cap`
  (default `tool_success_reward × max_turns`). Keep them small beside the objective, or tool-calling beats finishing.

### One server or several

![One rollout server: trainer ranks on GPUs 0-6, the engine on GPU 7 in the trainer's NCCL group, the push pausing it, prefetch off, step time = sync + round + update](../../agent-docs/assets/diagrams/environmental_grpo_single_server.png)

With one server the step is serial — sync, generate, train — the push pauses the only engine, so prefetch is off.

![Two rollout servers: six trainer ranks, engines on GPUs 6 and 7, one NCCL group per server on ports 51216 and 51217, prefetch one round deep, step time = sync + max(round, update)](../../agent-docs/assets/diagrams/environmental_grpo_multi_server.png)

With two or more — `rollout_server_configs`, one entry per server, each with its own `group_port` —
`enable_prefetch: true` overlaps the next round's generation with the current update, which is the only way to stop
paying for generation and training one after the other.

## Run

```bash
VLLM_MODEL=Qwen/Qwen3.6-35B-A3B VLLM_CUDA_DEVICES=4,5,6,7 \
    docker compose -f docker-compose.vllm.yml up -d vllm-server
CUDA_VISIBLE_DEVICES=0,1,2,3 DIST_NCCL_TIMEOUT_MINUTES=60 halo launch environmental-grpo \
    examples/grpo/environmental/qwen3_5/vllm/qwen3.6-35b-a3b-react-math-full-ep4.yaml -n 4
```

A native-tool environment needs the server started with the tool-call parser for the model family: without one vLLM
rejects every rollout, and with the wrong one the calls come back as text and every episode scores zero. ReAct
environments need no parser. Before a long run, put the config through a
few rows with `halo run run-env --training_config <config>.yaml --dataset <hub-id-or-path> --num_examples 20
--base_url http://localhost:8000/v1 --model <served-id>`: that exercises the parser, the template, the sandbox or
judge backend and the dataset columns in a minute.

Evaluation runs the same loop: set `eval_strategy`, plus `num_generations_eval: 1` for a fast pass@1 monitor.
`eval_rollout_batch_size` widens an eval round (rows per rank) so the servers do not idle — a multiple of
`num_generations_eval`, at most `max_concurrent_rollouts`, with `dataloader_drop_last: false`.

## What to watch

`reward/objective` says whether the task is being learned — a rising total reward with a flat objective is shaping,
not progress (code contests add `outcome/solve_rate` beside it). `episode/turns` and `episode/truncation_rate` say
whether episodes finish, and `sampling/logratio_mean` drifting steadily negative means the weight sync is broken and
the policy is training on stale rollouts. With several servers, `async/prefetch_hit_rate` above roughly 0.8 means
generation overlaps training.

`reward/within_group_std` near zero is the quiet failure: every episode in a group scored the same, so the advantages
are zero and that prompt teaches nothing. Rollouts land in `<output_dir>/completions/` as parquet.

## Sizing a run

Rollouts in flight per server are at most `data_parallel_size × per_device_train_batch_size × steps_per_generation ÷
num_servers`, capped per rank by `max_concurrent_rollouts`. Measure the engine's decode speed at about that
concurrency with the run's own prompts — nothing in the config predicts it — then set `request_timeout` to at least
twice one turn's budget at that speed and `episode_timeout` to `max_turns` turns plus tool time. Start from one
serving GPU per trainer GPU, one engine per serving GPU while the model and its KV cache fit.

## Go deeper

- [Async GRPO with Environments](../../agent-docs/training-methods/grpo/async-grpo/README.md) ↗ ·
  [Environments](../../agent-docs/training-methods/grpo/environments/README.md) ↗ — servers, objective, metrics, knobs.
- [Code Contests](../../agent-docs/training-methods/grpo/environments/code-contests.md) ↗ · [SWE](../../agent-docs/training-methods/grpo/environments/swe-environment.md) ↗ — the two deepest environments.
- [Rollout Servers](../rollout-servers.md) · [Online GRPO (RLVR)](online-grpo.md) · [Training Methods](README.md)
