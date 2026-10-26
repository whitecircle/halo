# Online GRPO (RLVR)

Online GRPO is on-policy RL with verifiable rewards. The model generates several completions per prompt,
a rule (or a judge, or a reward model) scores each one, and the group's spread becomes the advantage.
It covers one prompt, one answer, one turn: math, structured output and anything a grader can check.
Multi-turn tool use belongs on [Async GRPO with Environments](async-grpo-environments.md), and
pre-scored data on [Offline GRPO](offline-grpo.md).

Generation runs on a **separate vLLM container** that receives the updated weights over NCCL after each
optimizer step. Start it first, on GPUs no trainer rank uses ([Rollout Servers](../rollout-servers.md)).

## Data

Two columns: the prompt, and the ground truth the grader checks against.

```jsonl
{"prompt": "What is 2 + 3? Put your answer in \\boxed{}.", "answer": "5"}
{"prompt": [{"role": "user", "content": "What is 2 + 3?"}], "answer": "5"}
```

Point `prompt_field` and `answer_field` at the real column names; a typo raises at load. A prompt
longer than `max_prompt_length` is dropped, not truncated.

## Rewards

The reward is a list of terms, each scoring a completion in `[0, 1]` and logged as
`rewards/<name>/mean`. The default is one `accuracy` term: 1.0 when the last `\boxed{...}` equals
`answer`. `format`, `judge` and `reward_model` terms add a regex check, a generative judge graded
against your rubric, and a served scorer:

```yaml
rewards:
  - source: accuracy
  - source: format
    weight: 0.5
```

Every judge and reward model is probed at launch, so a bad URL, key or model name fails the launch, not
every step ([Reward Terms](../../agent-docs/training-methods/grpo/rewards/README.md) ↗).

## Run

```bash
# server first, on GPU 7, serving the checkpoint the config trains
make build-vllm   # or set VLLM_IMAGE to the public image
VLLM_MODEL=Qwen/Qwen3-4B-Instruct-2507 docker compose -f docker-compose.vllm.yml up -d vllm-server

# trainer on GPUs 0-6
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 halo launch rlvr \
    examples/grpo/online/rlvr-online-grpo-template.yaml -n 7
```

Wait for `/health` before launching. For an MoE model, set `expert_parallel_size` to the number of
trainer GPUs, which usually means a power-of-two split such as four and four. Smoke-test first with
`examples/grpo/online/qwen3/online-grpo-qwen3-4b-smoke.yaml`: three steps, one weight sync, nothing
saved.

Set `max_completion_length` (the generation budget) explicitly, keep `num_generations` at 4 or more so
groups carry a signal, and leave the sampling cuts (`top_p`, `top_k`, `min_p`, `repetition_penalty`)
off. Serve MoE models with `--moe-backend triton`, or the engine repacks the synced expert weights and
the served policy stops matching.

## What to watch

- `rewards/accuracy/mean` is progress. `frac_reward_zero_std` is the share of groups that gave no signal.
- `sampling/importance_sampling_ratio/mean` near 1 means trainer and engine agree on the policy. A drift
  away from 1 means they disagree about the template, the sinks or dropout.
- Every logged step's rollouts land in `<output_dir>/completions/` as parquet. Read them when the
  rewards are all zero: usually the policy never emits `\boxed{}`.

Every knob, the weight-sync refusals, LoRA and batch geometry:
[Online GRPO reference](../../agent-docs/training-methods/grpo/online.md) ↗.
