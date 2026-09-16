# Online GRPO (RLVR)

Online GRPO is on-policy reinforcement learning with verifiable rewards: the model generates several completions per
prompt, each is scored by a rule (or a judge, or a reward model), and the group's spread becomes the advantage. One
prompt, one answer, one turn — for math, structured output and anything you can check with a grader. Multi-turn tool
use belongs to [Async GRPO with Environments](async-grpo-environments.md); pre-scored data to
[Offline GRPO](offline-grpo.md). [Choosing a Method](../choosing-a-method.md) compares them.

![Online GRPO as one cycle: each prompt is repeated num_generations times and rendered by the trainer's template,
the vLLM server returns completions with their sampling log-probs, the reward terms score them, the group normalizes
them into advantages, the loss steps the model, and the new weights go back over
NCCL](../../agent-docs/assets/diagrams/online_grpo_pipeline.png)

Generation runs on a **separate vLLM container** that receives the updated weights over NCCL before each round. The
trainer never imports vLLM, and the server must own GPUs no trainer rank uses — a rank cannot broadcast to itself.
Start it first: [Rollout Servers](../rollout-servers.md).

## Data

Two columns: the prompt, and the ground truth the grader checks against.

```jsonl
{"prompt": "What is 2 + 3? Put your answer in \\boxed{}.", "answer": "5"}
{"prompt": [{"role": "user", "content": "What is 2 + 3?"}], "answer": "5"}
```

`prompt` is a string or a message list; `answer` is whatever your reward terms compare to. Point `prompt_field` and
`answer_field` at the real column names — a typo raises at load instead of quietly scoring every row zero. A prompt
longer than `max_prompt_length` is dropped, not truncated. The shipped recipes pull `openai/gsm8k`,
`trl-lib/DeepMath-103K` and `open-r1/DAPO-Math-17k-Processed`.

## Rewards

The reward is a list of terms. Each names a source of a score in `[0, 1]` and prices it as `weight × score ^ exponent`;
the sum is what the group normalizes. Terms are parsed at config time, so a bad one fails before any server is touched,
and each logs separately as `rewards/<name>/mean`.

| `source` | Scores 1.0 when | What you set |
| --- | --- | --- |
| `accuracy` | the last `\boxed{...}` equals `answer` | nothing |
| `format` | the completion matches `pattern` | `pattern`, default `<think>…</think>\s*<answer>…</answer>` |
| `judge` | a generative judge grades the response against your rubric | `requirements` (name + description + weight), optionally `model`, `base_url`, `scale` |
| `reward_model` | a served Bradley-Terry or classification model scores it | `url`, `model` |

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

A judge defaults to an OpenRouter model and reads `OPENROUTER_API_KEY`; a reward model is served by vLLM
(`--runner pooling`) or SGLang (`--is-embedding`) on its own port. Both are probed once at launch with a sample
request, so a bad URL, key or model name fails the launch rather than every step. A
negative `weight` makes a term a penalty; the full option list is in
[Reward Terms](../../agent-docs/training-methods/grpo/rewards.md) ↗.

## Config

From `examples/grpo/online/rlvr-online-grpo-template.yaml`; the same keys with EP appear in
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

- `num_generations` trades unique prompts per step for samples per prompt. Below about 4 a group's rewards are too
  often all equal, and an all-equal group produces no gradient.
- `beta` above `0` makes TRL build its own reference model — a full extra copy per rank. Every shipped recipe keeps
  `beta: 0` and relies on the clip instead.
- `scale_rewards: group` divides by the group's own spread. Keep it for a single binary reward, switch to `batch`
  when the reward is a graded fraction whose group spread is mostly shaping noise, or `none` to leave it unscaled.
- `max_completion_length` is the budget handed to vLLM, so it must be a positive integer; the server's context window
  is checked against `max_prompt_length + max_completion_length` at startup.
- The learning rate sits near the SFT floor — `1e-6` to `5e-6` in the recipes.

Completions per optimizer step are `per_device_train_batch_size × gradient_accumulation_steps ×
data_parallel_size`; divide by `num_generations` for unique prompts.

## Run

Split the GPUs: vLLM on some, the trainer on the rest.

```bash
# server first, on GPU 7 — the same checkpoint the config trains
VLLM_MODEL=Qwen/Qwen3-4B-Instruct-2507 docker compose -f docker-compose.vllm.yml up -d vllm-server

# trainer on GPUs 0-6
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 halo launch rlvr \
    examples/grpo/online/rlvr-online-grpo-template.yaml -n 7
```

Wait for `/health` before launching. For an MoE model, set `expert_parallel_size` to the number of trainer GPUs so
they form one expert-dispatch group — usually a power-of-two split, such as four trainer GPUs with
`--expert_parallel_size=4` and four for the server. Context and pipeline parallelism are rejected here.
Smoke-test first: `examples/grpo/online/qwen3/online-grpo-qwen3-4b-smoke.yaml` runs three steps and saves nothing,
which still exercises rendering, rewards, the loss and one weight sync.

### Weight sync

After an optimizer step the trainer pushes the whole model to the server over NCCL, before the next round of
generation, so rollouts always come from the current policy. Whether a given shape can be synced at all is decided
at construction: the trainer refuses a QLoRA base, a GPT-OSS whose attention sinks were reset away, and an MoE
family the pinned engine cannot take an update for. Serve MoE models with `--moe-backend triton`, or the engine
repacks the expert weights the sync writes and the policy stops matching ([Rollout Servers](../rollout-servers.md)).

### LoRA

Add `use_peft: true` and the `lora_*` fields. LoRA runs under data parallelism and expert parallelism; adapters on
the expert projections are rejected once `expert_tensor_parallel_size > 1`, and any adapter under tensor parallelism.
The weight sync merges the adapter into the base before broadcasting, so the server serves plain weights.

## What to watch

Read `rewards/accuracy/mean` for progress and `frac_reward_zero_std` for how many groups produced no signal.
`sampling/importance_sampling_ratio/mean` near 1 means trainer and engine agree on the policy; a drift away from 1
means they disagree about the template, the sinks or dropout, and is the most useful alarm on this path. Every logged
step's rollouts land in `<output_dir>/completions/` as parquet, which is where to look when rewards are all zero.

| Symptom | Cause | Fix |
| --- | --- | --- |
| Every reward is 0 | The policy never emits `\boxed{}` | Ask for it in `system_prompt`, raise `max_completion_length`, read the parquet |
| Importance-sampling ratio far from 1 | Trainer and server disagree | Serve the same checkpoint and template; `reset_sinks: false` for GPT-OSS |
| Hang at sync or generation | Server down, or sharing a GPU with a trainer rank | `curl .../health`, check both `CUDA_VISIBLE_DEVICES` |

## Go deeper

- [Online GRPO (RLVR)](../../agent-docs/training-methods/grpo/online-grpo.md) ↗ — objective knobs, batch geometry.
- [Reward Terms](../../agent-docs/training-methods/grpo/rewards.md) ↗ — judge and reward-model options in full.
- [Rollout Servers](../rollout-servers.md) · [Offline GRPO](offline-grpo.md) ·
  [Async GRPO with Environments](async-grpo-environments.md) · [Training Methods](README.md)
