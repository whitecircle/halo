# Online GRPO (RLVR)

Online GRPO is on-policy RL with verifiable rewards. The model generates several completions per prompt,
a rule (or a judge, or a reward model) scores each one, and the group's spread becomes the advantage.
It covers one prompt, one answer, one turn: math, structured output and anything a grader can check.

Multi-turn tool use belongs on [Async GRPO with Environments](async-grpo-environments.md), and pre-scored
data on [Offline GRPO](offline-grpo.md). [Choosing a Method](../choosing-a-method.md) compares them.

![Online GRPO as one cycle: prompts repeated num_generations times, completions and sampling log-probs from
the vLLM server, reward terms, group advantages, the loss step, and the new weights sent back over
NCCL](../../agent-docs/assets/diagrams/online_grpo_pipeline.png)

Generation runs on a **separate vLLM container** that receives the updated weights over NCCL before each
round. Start it first, on GPUs no trainer rank uses ([Rollout Servers](../rollout-servers.md)).

## Data

Two columns: the prompt, and the ground truth the grader checks against.

```jsonl
{"prompt": "What is 2 + 3? Put your answer in \\boxed{}.", "answer": "5"}
{"prompt": [{"role": "user", "content": "What is 2 + 3?"}], "answer": "5"}
```

`prompt` is a string or a message list. Point `prompt_field` and `answer_field` at the real column names;
a typo raises at load instead of scoring every row zero. A prompt longer than `max_prompt_length` is
dropped, not truncated. The shipped recipes use `openai/gsm8k`, `trl-lib/DeepMath-103K` and
`open-r1/DAPO-Math-17k-Processed`.

## Rewards

The reward is a list of terms. Each term scores a completion in `[0, 1]` and prices it as
`weight × score ^ exponent`; the group normalizes the sum. A negative `weight` makes a term a penalty.
Each term logs as `rewards/<name>/mean`.

| `source` | Scores 1.0 when | What you set |
| --- | --- | --- |
| `accuracy` | the last `\boxed{...}` equals `answer` | nothing |
| `format` | the completion matches `pattern` | `pattern`, default `<think>…</think>\s*<answer>…</answer>` |
| `judge` | a generative judge grades it against your rubric | `name`, `requirements` (name, description, weight); optionally `model`, `base_url`, `scale` |
| `reward_model` | a served Bradley-Terry or classification model scores it | `name`, `url`, `model` |

```yaml
rewards:
  - source: accuracy
  - source: format
    weight: 0.5
  - source: judge
    name: quality
    weight: 0.3
    requirements:
      - {name: correctness, description: "The final answer is correct and fully justified."}
      - {name: clarity, description: "The solution is easy to follow.", weight: 0.5}
```

- A judge defaults to an OpenRouter model and reads `OPENROUTER_API_KEY`.
- A reward model is served by vLLM (`--runner pooling`) or SGLang (`--is-embedding`) on its own port.
- Terms are parsed at config time, and every judge and reward model is probed with a sample request at
  launch. A bad term, URL, key or model name fails the launch, not every step.
- A veto judge (one that lists `checks` instead of `requirements`) is refused here. It gates an
  environment's grade, which only [Async GRPO](async-grpo-environments.md#rewards) has.

All options: [Reward Terms](../../agent-docs/training-methods/grpo/rewards.md) ↗.

## Config

From `examples/grpo/online/rlvr-online-grpo-template.yaml`. The EP version of the same keys is
`examples/grpo/online/qwen3_5/online-grpo-qwen3.6-35b-a3b-dapo-math.yaml`.

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
dataset: trl-lib/DeepMath-103K
answer_field: solution

rewards:
  - source: accuracy

beta: 0.0                     # no KL anchor, so no second model in memory
num_generations: 8            # completions per prompt: the group size
loss_type: grpo
epsilon: 0.15
scale_rewards: group

max_prompt_length: 1024       # over-long rows are dropped
max_completion_length: 2048   # the generation budget, required

use_vllm: true                # server mode is mandatory
vllm_mode: server
vllm_server_host: 0.0.0.0     # the address the TRAINER DIALS, not a bind address
vllm_server_port: 8000

learning_rate: 5.0e-06
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
```

- `num_generations` trades unique prompts for samples per prompt. Below about 4, a group's rewards are
  often all equal, and an all-equal group gives no gradient.
- `beta` above 0 adds a KL anchor. On a full fine-tune or native expert-only LoRA, the script loads a
  frozen reference the way the policy loads (run dtype, revision, attention): a full extra copy on every
  rank, unsharded under EP, ETP or TP. A PEFT policy is its own reference with the adapter off. Every
  shipped recipe keeps `beta: 0`, and at their `num_iterations: 1` the `epsilon` clip is inert too.
- `scale_rewards: group` divides by each group's own spread. Keep it for a single binary reward. Use
  `batch` when the reward is a graded fraction whose group spread is mostly noise, or `none` to leave
  it unscaled.
- `max_completion_length` is the budget handed to vLLM and must be a positive integer. At startup the
  server's context window is checked against `max_prompt_length + max_completion_length`.
- Leave `top_p` at 1, `top_k` and `min_p` off, and `repetition_penalty` at 1. Under the default
  importance-sampling mode, startup refuses a repetition penalty, and a server whose logprobs are
  renormalized over a sampling cut.
- The learning rate sits near the SFT floor: `1e-6` to `5e-6` in the recipes.

Completions per optimizer step are `per_device_train_batch_size × gradient_accumulation_steps ×
data_parallel_size`; divide by `num_generations` for unique prompts.

## Run

Split the GPUs: vLLM on some, the trainer on the rest.

```bash
# server first, on GPU 7, serving the checkpoint the config trains
VLLM_MODEL=Qwen/Qwen3-4B-Instruct-2507 docker compose -f docker-compose.vllm.yml up -d vllm-server

# trainer on GPUs 0-6
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 halo launch rlvr \
    examples/grpo/online/rlvr-online-grpo-template.yaml -n 7
```

Wait for `/health` before launching. For an MoE model, set `expert_parallel_size` to the number of
trainer GPUs so they form one expert-dispatch group. That usually means a power-of-two split, such as
four trainer GPUs with `--expert_parallel_size=4` and four for the server. Context and pipeline
parallelism are rejected.

Smoke-test first with `examples/grpo/online/qwen3/online-grpo-qwen3-4b-smoke.yaml`. It runs three steps
and saves nothing, and still exercises rendering, rewards, the loss and one weight sync.

### Weight sync

After each optimizer step the trainer pushes the whole model to the server over NCCL, before the next
round of generation. Construction refuses a shape the sync cannot carry:

- a QLoRA base, or an adapter the sync cannot fold;
- GPT-OSS sinks removed by `reset_sinks`, or trained with `train_sinks: true`;
- a model family the pinned engine cannot take an online update for;
- an MoE run without EP wrappers (`expert_parallel_size: 1` with `use_grouped_gemm: false`).

Serve MoE models with `--moe-backend triton`. Otherwise the engine repacks the expert weights the sync
writes, and the served policy stops matching ([Rollout Servers](../rollout-servers.md)).

### LoRA

Add `use_peft: true` and the `lora_*` fields. LoRA runs under data, expert and expert-tensor
parallelism. Adapters on the expert projections are rejected once `expert_tensor_parallel_size > 1`, and
every adapter is rejected under tensor parallelism. The sync folds the adapter into each base weight as it sends it, so
the server serves plain weights.

## What to watch

- `rewards/accuracy/mean` is progress. `frac_reward_zero_std` is the share of groups that gave no signal.
- `sampling/importance_sampling_ratio/mean` near 1 means trainer and engine agree on the policy. A drift
  away from 1 means they disagree about the template, the sinks or dropout. It is the most useful alarm
  on this path.
- `judge/<name>/scored` or `reward_model/<name>/scored` is the share of completions a scorer reached a
  verdict on. Below 1, the scorer is failing.

Every logged step's rollouts land in `<output_dir>/completions/` as parquet. Read them when the rewards
are all zero.

| Symptom | Cause | Fix |
| --- | --- | --- |
| Every reward is 0 | The policy never emits `\boxed{}` | Ask for it in `system_prompt`, raise `max_completion_length`, read the parquet |
| Importance-sampling ratio far from 1 | Trainer and server disagree | Serve the same checkpoint and template; `reset_sinks: false` for GPT-OSS |
| Hang at sync or generation | Server down, or sharing a GPU with a trainer rank | `curl .../health`, check both `CUDA_VISIBLE_DEVICES` |

## Go deeper

- [Online GRPO (RLVR)](../../agent-docs/training-methods/grpo/online-grpo.md) ↗: objective knobs and
  batch geometry.
- [Reward Terms](../../agent-docs/training-methods/grpo/rewards.md) ↗: judge and reward-model options.
- [Rollout Servers](../rollout-servers.md) · [Offline GRPO](offline-grpo.md) ·
  [Async GRPO with Environments](async-grpo-environments.md) · [Training Methods](README.md)
