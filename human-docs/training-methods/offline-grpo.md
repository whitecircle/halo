# Offline GRPO

Offline GRPO trains on completions generated and scored somewhere else: several completions per prompt,
each with a reward, and no generation during the run. It is the cheapest GRPO to operate (no inference
server, no Ray, a dataset you can re-run), so it fits when scoring is expensive, already done, or must
stay reproducible. To generate as the model learns, use [Online GRPO (RLVR)](online-grpo.md) or
[Async GRPO with Environments](async-grpo-environments.md).

## Data

Three columns, and **a group is one row**: put every completion of a prompt in the same row. Two rows
that share a prompt are two groups, normalized separately (a singleton normalizes to advantage 0 and
trains nothing).

```jsonl
{"prompt": [{"role": "user", "content": "What is 2+2?"}], "completions": [[{"role": "assistant", "content": "4"}], [{"role": "assistant", "content": "5"}]], "rewards": [1.0, -0.5]}
```

`completions` holds one message list per completion (usually 4–16) and `rewards` one float per
completion, in the same order. The method is text-only. No dataset yet?
`halo run inference/reward-model/rejection-sample --output_format offline_grpo` generates candidates
against a served model, scores them with a reward model, and writes this shape.

## Run

```bash
halo launch grpo/offline examples/grpo/offline/qwen3_5/offline-grpo-qwen3.6-35b-a3b-gsm8k.yaml -n 8
```

Nothing else has to run, so the families online RL refuses for weight sync still train here (on
Qwen3.8-Flash-Next this path is CPU-tested only). `examples/grpo/offline/` also holds the GPT-OSS,
Gemma 4 and Nemotron versions. Test a config first with `--max_steps=10 --save_strategy=no` on a slice of the data.

Context and pipeline parallelism take full fine-tuning only, and pipeline parallelism needs
`max_prompt_length` and `max_completion_length` set.

- `advantage_method` turns the group's rewards into advantages; the default `quantile_norm` ranks them,
  which handles discrete or outlier-heavy rewards.
- `max_completion_length` is a truncation cap, not a generation budget. A completion cut at the cap
  trains without a stop token, so set it above what your data contains.
- `kl_beta` above 0 adds a KL anchor to the starting policy, scored once at run start for a full
  fine-tune.
- The learning rate sits near `5e-6`: GRPO refines an already-tuned policy.

## What to watch

- `positive/logps_mean` should hold steady or rise, and `negative/logps_mean` fall.
- `positive/pg_objective_mean` falling through training means the policy drifted from the distribution
  the data was collected under: set `kl_beta: 0.05` or lower the learning rate.
- A loss that goes NaN comes from low-probability tokens on negative-advantage rows: tighten
  `initial_min_log_prob` and set `max_grad_norm: 1.0`.

Every knob, the advantage formulas and the reference-model rules:
[Offline GRPO reference](../../agent-docs/training-methods/grpo/offline.md) ↗.
