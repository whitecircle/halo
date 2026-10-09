# Offline GRPO

Offline GRPO trains on completions generated and scored somewhere else: several completions per prompt,
each with a reward, and no generation during the run. It is the cheapest GRPO to operate (no inference
server, no Ray, a dataset you can re-run), so it fits when scoring is expensive, already done, or must
stay reproducible.

If you want the model to generate as it learns, use [Online GRPO (RLVR)](online-grpo.md) or
[Async GRPO with Environments](async-grpo-environments.md). [Choosing a Method](../choosing-a-method.md)
compares them.

![Offline GRPO in two bands: at tokenization each dataset row of prompt, completions and rewards becomes group
advantages and then one training row per completion; each step draws a micro-batch, computes the per-token loss and
normalizes it, every row weighted by 1/group_size](../../agent-docs/assets/diagrams/offline_grpo_pipeline.png)

Rewards become advantages once, at tokenization. Each training step then draws a micro-batch of
completions that already carry their advantage and their group's weight.

## Data

Three columns. **A group is one row**: put every completion of a prompt in the same row. Two rows that
share a prompt are two groups and are normalized separately.

| Column | Type | Contents |
| --- | --- | --- |
| `prompt` | `list[dict]` | the conversation as `{"role", "content"}` messages |
| `completions` | `list[list[dict]]` | one message list per completion, usually 4–16 |
| `rewards` | `list[float]` | one reward per completion, in the same order |

```jsonl
{"prompt": [{"role": "user", "content": "What is 2+2?"}], "completions": [[{"role": "assistant", "content": "4"}], [{"role": "assistant", "content": "5"}]], "rewards": [1.0, -0.5]}
```

- A count mismatch between `completions` and `rewards` raises with the row index. A non-finite reward
  raises with the group.
- The method is decoder-only and text-only. An encoder-decoder model is refused at construction, and
  the script refuses image columns and image parts in the messages.
- No dataset yet? `halo run rm-rejection-sampling --output_format offline_grpo` generates candidates
  against a served model, scores them with a reward model, and writes exactly this shape.

## Config

From `examples/grpo/offline/qwen3_5/offline-grpo-qwen3.6-35b-a3b-gsm8k.yaml`: Qwen3.6-35B-A3B on eight
GPUs with expert parallelism. `examples/grpo/offline/gptoss/` and `examples/grpo/offline/gemma4/` hold
the same recipe for those families.

```yaml
model_name_or_path: Qwen/Qwen3.6-35B-A3B
dataset:
- path/to/gsm8k_<model>_offline_grpo.jsonl
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

- **`advantage_method`:** `quantile_norm` (the default) ranks the group's rewards, which handles discrete
  or outlier-heavy reward sets. The alternatives are `z_norm`, `minmax`, `quantile_uniform` and `robust`.
- **`loss_type`:** `bnpo` averages over the micro-batch's tokens. `grpo` averages per sequence first; use
  it when completion lengths vary a lot. `dr_grpo` normalizes by a constant built from
  `max_completion_length`, which removes length bias, so it needs that cap set.
- **`kl_beta`:** above 0, a full fine-tune scores its starting policy once and saves those reference
  scores with each checkpoint, so a resume keeps the original anchor. That needs a finite, unsharded
  dataset, but no second model copy. A PEFT policy scores its reference live with the adapters off, and
  native expert LoRA loads a frozen copy of the base
  ([Reference rules](../../agent-docs/training-methods/grpo/offline-grpo.md#reference-model) ↗).
- **`initial_min_log_prob` / `min_log_prob`:** a floor on low-probability tokens of negative-advantage
  rows. It keeps the loss finite when the policy is pushed away from something it already finds
  unlikely.
- **`max_completion_length`** is a truncation cap, not a generation budget. A completion cut at the cap
  trains without a stop token, so set it above what your data contains. It defaults to `null`, which
  keeps every token.
- **`max_prompt_length`** defaults to `512`. Set it to `null` to leave prompts untouched.

On a large-vocabulary model with long completions, add `use_chunked_grpo_logprobs: true`. It computes the
same log-probs in fp32 without materializing the full logits. On a bf16 or fp16 full fine-tune with
`kl_beta > 0` outside CP, keep this setting fixed across a resume: the saved reference scores record
their precision, and a resume that toggles it is refused.

## Run

```bash
halo launch offline-grpo examples/grpo/offline/qwen3_5/offline-grpo-qwen3.6-35b-a3b-gsm8k.yaml -n 8
```

Nothing else has to run: no vLLM, no Ray. Expert, tensor and expert-tensor parallelism all work. CP and
EP+CP support full fine-tuning on
[CP-capable models](../../agent-docs/parallelism/context-parallelism.md#supported-model-architectures) ↗.
They reject adapters, and they train right-padded whole rows with chunked log-probs.

Test the setup first with `--max_steps=10 --save_strategy=no` on a slice of the data. That exercises
tokenization, advantage normalization and the loss in minutes. A bad `loss_type`, `advantage_method` or
reward column fails at construction, not an hour in.

## What to watch

The trainer logs per-sample diagnostics split by advantage sign. Three of them tell you whether it is
working:

- `positive/logps_mean` should hold steady or rise.
- `negative/logps_mean` should fall.
- `positive/pg_objective_mean` is the advantage-weighted policy term (π·A under the default
  `prob_weighted`), not the dataset reward.

| Symptom | Cause | Fix |
| --- | --- | --- |
| Loss goes NaN or explodes | Low-probability tokens on negative-advantage rows | `initial_min_log_prob: -0.5` for a tighter floor early, `max_grad_norm: 1.0` |
| `positive/pg_objective_mean` falls through training | The policy drifted away from the distribution the data was collected under | `kl_beta: 0.05`, lower the learning rate |
| OOM | Full-vocabulary logits over long completions | `use_chunked_grpo_logprobs: true`, lower the batch or `max_completion_length` |

Two mistakes are common:

- **Splitting one prompt's completions across rows.** Every group silently becomes a singleton. A
  singleton group, like an exactly tied one, normalizes to advantage 0 and gives no gradient. Set
  `drop_degenerate_groups: true` to drop such groups at tokenization instead.
- **Treating the learning rate like SFT.** GRPO refines an already-tuned policy, so the recipes sit at
  `5e-6`.

## Go deeper

- [Offline GRPO](../../agent-docs/training-methods/grpo/offline-grpo.md) ↗: every knob, the advantage
  formulas, the reference-model rules.
- [GRPO overview](../../agent-docs/training-methods/grpo/README.md) ↗: the objective the three variants
  share.
- [Online GRPO (RLVR)](online-grpo.md) · [Async GRPO with Environments](async-grpo-environments.md) ·
  [Training Methods](README.md)
