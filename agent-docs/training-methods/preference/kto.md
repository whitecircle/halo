# KTO

Kahneman-Tversky Optimization trains on **unpaired** binary feedback: each row is one completion labeled desirable or not. Use it for thumbs-up/down labels rather than preference pairs; for chosen/rejected pairs use [DPO](dpo.md) or [SMPO](smpo.md), for several scored completions per prompt [Offline GRPO](../grpo/offline-grpo.md).

Trainer `DistributedKTOTrainer`, script `scripts/training/preference/kto.py` (text or VLM). EP, TP and ETP apply; CP does not — TRL's loss path is not CP-aware. It declares `_supports_pp`, but pipeline parallelism is [not yet available in this release](../../parallelism/pipeline-parallelism.md). TRL's fused KTO-Liger loss is disabled at construction (it is broken in TRL 1.6); Liger kernels still apply at the model level.

## Dataset

```jsonl
{"prompt": [{"role": "user", "content": "What is 2+2?"}], "completion": [{"role": "assistant", "content": "4"}], "label": true}
```

`prompt` / `completion` may be plain strings or message lists — TRL templates them itself. `completion_field` / `label_field` rename those columns, and a name the dataset lacks raises before training. `tools_field` and `log_decoded_samples` are refused: TRL templates the raw columns without `tools=`, and the rows reach it untokenized.

## Configuration

```yaml
model_name_or_path: Qwen/Qwen3.5-9B
dataset: trl-lib/kto-mix-14k
beta: 0.1
desirable_weight: 1.0
undesirable_weight: 1.0
loss_type: kto

per_device_train_batch_size: 2     # > 1 is required by the `kto` loss
gradient_accumulation_steps: 4
learning_rate: 5.0e-07
max_length: 4096
gradient_checkpointing: true
output_dir: checkpoints/kto-qwen3.5-9b
```

| Knob | Default | Effect |
|---|---|---|
| `beta` | `0.1` | Deviation allowed from the reference |
| `desirable_weight` / `undesirable_weight` | `1.0` / `1.0` | Re-weight the two classes when the label counts are skewed |
| `loss_type` | `kto` | Or `apo_zero_unpaired`, which carries no KL term |
| `max_length` | `1024` | Truncates prompt + completion, keeping the start |

The default `kto` loss builds each row's KL completion from its neighbors in a fixed-order batch, so TRL raises at `per_device_train_batch_size: 1` and at any `train_sampling_strategy` other than the default `sequential`. Only `apo_zero_unpaired` is exempt. KTO has no prompt cap, so an over-long prompt eats its own completion.

The reference model follows DPO's rules — PEFT, precompute under EP/TP/PP, or a frozen copy on every other shape ([DPO — Reference model](dpo.md#reference-model)).

Pipeline parallelism is [not yet available in this release](../../parallelism/pipeline-parallelism.md); the shipped PP gates already pin its contract for this trainer — `apo_zero_unpaired` only (the default `kto` loss needs a world-global KL baseline no microbatch can compute), `precompute_ref_log_probs: true` with `ref_logps` already a column of the train dataset and of any eval dataset passed, and no live `ref_model`, PEFT or `compute_metrics`.

## Launch

```bash
torchrun --nproc_per_node=8 scripts/training/preference/kto.py \
    examples/preference/qwen3_5/kto-qwen3.5-9b-kto-mix-14k.yaml
```

`halo launch kto <config> --nproc 8` builds the same line. That recipe is dense, so it runs plain FSDP2 data parallel; add `--expert_parallel_size=8` on an MoE checkpoint. Like the other padded-batch scripts, `kto.py` defaults `attn_implementation` to `sdpa` under `reset_sinks: true` ([Flash Attention](../../optimization/flash-attention.md#model-specific-handling)).

## Vision-language

Add an `images`/`image` column — or name one with `images_field` — to train a VLM checkpoint on `{prompt, completion, label, images}` rows. Text-only data on a multimodal checkpoint stays on the text path.

A dataset that embeds `{"type": "image"}` parts in its messages while shipping no image column is refused: TRL would template them as text, expanding each part into pixel-less placeholder tokens. Move the images into an `images` column, or drop the parts.

Vision KTO is unpaired-only — a `chosen`/`rejected` column alongside images raises, so unpair the dataset first. `precompute_ref_log_probs` is rejected on vision rows, so under EP PEFT is the only reference, and under PP a vision run has no supported shape.

## Testing a setup

```bash
torchrun --nproc_per_node=2 scripts/training/preference/kto.py <config> \
    --max_steps=5 --save_strategy=no --report_to=none
```

Covering tests: `pytest tests/cpu/trainers tests/cpu/config -m cpu`, plus `tests/gpu/trainers/preference/test_kto.py` and `test_kto_fsdp_multi_gpu.py`.

## What to watch

| Signal | Reading |
|---|---|
| `rewards/chosen`, `rewards/rejected` | Desirable rows should rise, undesirable fall |
| `rewards/margins` | Their difference; the separation KTO is buying |
| `kl` | World-gathered mean, clamped at ≥ 0; constant `0.0` under `apo_zero_unpaired`. A runaway climb means `beta` is too low for the LR |

Failure signatures:

- "Actual (not effective) batch size must be > 1" — raise `per_device_train_batch_size` and halve `gradient_accumulation_steps`.
- One class dominating the gradient — set `desirable_weight` / `undesirable_weight` to the inverse label ratio.
