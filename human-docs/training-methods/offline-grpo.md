# Offline GRPO

Offline GRPO trains on completions that were generated and scored somewhere else: several completions per prompt,
each with a reward, and no generation during the run. It is the cheapest GRPO to operate — no inference server, no
Ray, and a dataset you can re-run — so it fits when scoring is expensive, already done, or must stay reproducible.
If you want the model to generate as it learns, use [Online GRPO (RLVR)](online-grpo.md) or
[Async GRPO with Environments](async-grpo-environments.md). [Choosing a Method](../choosing-a-method.md) compares them.

![Offline GRPO in two bands: at tokenization each dataset row of prompt, completions and rewards becomes group
advantages and then one training row per completion; each step draws a micro-batch, computes the per-token loss and
normalizes it, every row weighted by 1/group_size](../../agent-docs/assets/diagrams/offline_grpo_pipeline.png)

Rewards become advantages once, at tokenization. Each training step then draws a micro-batch of completions that
already carry their advantage and their group's weight, so nothing is generated or scored at train time.

## Data

Three columns. **A group is one row**: put every completion of a prompt in the same row, because two rows sharing a
prompt are two groups and are normalized separately.

| Column | Type | Contents |
| --- | --- | --- |
| `prompt` | `list[dict]` | the conversation as `{"role", "content"}` messages |
| `completions` | `list[list[dict]]` | one message list per completion, usually 4–16 |
| `rewards` | `list[float]` | one reward per completion, in the same order |

```jsonl
{"prompt": [{"role": "user", "content": "What is 2+2?"}], "completions": [[{"role": "assistant", "content": "4"}], [{"role": "assistant", "content": "5"}]], "rewards": [1.0, -0.5]}
```

A mismatch between `completions` and `rewards` raises with the row index, and a non-finite reward raises with the
group. If you don't have such a dataset yet, `halo run rm-rejection-sampling --output_format offline_grpo`
generates candidates against a served model, scores them with a reward model and writes exactly this shape.

## Config

From `examples/grpo/offline/qwen3_5/offline-grpo-qwen3.6-35b-a3b-gsm8k.yaml` — Qwen3.6-35B-A3B on eight GPUs with
expert parallelism. `examples/grpo/offline/gptoss/` and `examples/grpo/offline/gemma4/` are the same recipe for
those families.

```yaml
model_name_or_path: Qwen/Qwen3.6-35B-A3B
dataset:
- path/to/gsm8k-qwen3.6-35b-a3b-offline-grpo
expert_parallel_size: 8

advantage_method: quantile_norm    # rewards -> advantages inside each group
loss_type: bnpo                    # how the per-token loss is normalized
kl_beta: 0.0                       # no reference model, no KL penalty
initial_min_log_prob: -0.5         # log-prob floor, scheduled to -3.0

max_prompt_length: 512             # left-truncated
max_completion_length: 6144        # cut from the end
learning_rate: 5.0e-06
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
```

- `advantage_method` — `quantile_norm` (the default) ranks the group's rewards, which handles discrete or
  outlier-heavy reward sets; `z_norm`, `minmax`, `quantile_uniform` and `robust` are the alternatives.
- `loss_type` — `bnpo` averages over the micro-batch's tokens, `grpo` averages per sequence first (use it when
  completion lengths vary a lot), `dr_grpo` divides by a constant that removes length bias.
- `kl_beta` — above `0` the run holds a reference model and penalizes drift from it: a full extra copy per rank on a
  full fine-tune, the base with the adapters off under PEFT. Leave it at `0` unless rewards fall through training.
- `initial_min_log_prob` / `min_log_prob` — a floor on low-probability tokens of negative-advantage rows. It is what
  keeps the loss finite when the policy is pushed away from something it already thinks is unlikely.
- `max_completion_length` is a truncation cap, not a generation budget: a completion cut at the cap is trained
  without a stop token, so set it above what your data actually contains.

`max_completion_length` defaults to `null` and keeps every token; `max_prompt_length` defaults to
`512`, so set it to `null` when you want prompts untouched. Add `use_chunked_grpo_logprobs: true` on a large-vocabulary
model with long completions; it computes the same log-probs without materializing full logits.

## Run

```bash
halo launch offline-grpo examples/grpo/offline/qwen3_5/offline-grpo-qwen3.6-35b-a3b-gsm8k.yaml -n 8
```

Nothing else has to be running — no vLLM, no Ray. Expert, tensor and expert-tensor parallelism all work; context
parallelism is refused when the trainer is built, because the forward is trimmed to the completion
tokens.

Test the setup first with `--max_steps=10 --save_strategy=no` on a slice of the data: that exercises tokenization,
advantage normalization and the loss in minutes, and a bad `loss_type`, `advantage_method` or reward column fails at
construction rather than an hour in.

## What to watch

The trainer logs per-sample diagnostics split by advantage sign. Three of them tell you whether it is working:
`positive/logps_mean` (stable or rising), `negative/logps_mean` (falling) and `positive/rewards_mean`.

| Symptom | Cause | Fix |
| --- | --- | --- |
| Loss goes NaN or explodes | Low-probability tokens on negative-advantage rows | `initial_min_log_prob: -0.5` for a tighter floor early, `max_grad_norm: 1.0` |
| Rewards fall through training | The policy drifted away from the distribution the data was collected under | `kl_beta: 0.05`, lower the learning rate |
| OOM | Full-vocabulary logits over long completions | `use_chunked_grpo_logprobs: true`, lower the batch or `max_completion_length` |

Two mistakes are common. The first is splitting one prompt's completions across rows, which silently turns every
group into a singleton — and a singleton group, like an exactly-tied one, normalizes to advantage 0 and contributes
no gradient. Set `drop_degenerate_groups: true` to have them dropped at tokenization instead. The second is treating
the learning rate like SFT: GRPO refines an already-tuned policy, so the recipes sit at `5e-6`.

## Go deeper

- [Offline GRPO](../../agent-docs/training-methods/grpo/offline-grpo.md) ↗ — every knob, the advantage formulas, the
  reference-model rules.
- [GRPO overview](../../agent-docs/training-methods/grpo/README.md) ↗ — the objective the three variants share.
- [Online GRPO (RLVR)](online-grpo.md) · [Async GRPO with Environments](async-grpo-environments.md) ·
  [Training Methods](README.md)
