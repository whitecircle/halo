# Writing a config

A run is one flat YAML file: the HuggingFace `TrainingArguments` fields you
already know, plus Halo's own.

## Start from the nearest example

Don't write a config from scratch. `examples/` is organized by method, then by
model family (`examples/sft/qwen3/`, `examples/preference/qwen3_5/`). The closest
file is usually two edits from what you want:

```bash
ls examples/sft/                          # families with an SFT recipe
cp examples/sft/qwen3/qwen3-4b-ultrachat.yaml my-run.yaml
halo launch sft my-run.yaml -n 8
```

Keep the comments when you copy. They record the family's constraints: why
`sdpa` and not Flash Attention, why `packing` and not `padding_free`, which
router-balancing mode exports.

## The keys that matter

| Block | Keys |
| --- | --- |
| Model | `model_name_or_path`, `model_revision`, `attn_implementation`, `trust_remote_code`, `max_concurrent_loading` |
| Data | `dataset`, `dataset_ratio`, `test_size`, `max_length`, `packing`, `train_on_completions_only` |
| Training | `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `num_train_epochs`, `lr_scheduler_type`, `gradient_checkpointing` |
| Precision and optimizer | `bf16`, `optim`, `bf16_optimizer`, `fp32_grad_reduce`, `fp32_non_ep_params` |
| Parallelism | `expert_parallel_size`, `context_parallel_size`, `tensor_parallel_size`, `expert_tensor_parallel_size`, `ep_scope` |
| PEFT | `use_peft`, `lora_r`, `lora_alpha`, `lora_target_modules` |
| Checkpoint and eval | `output_dir`, `save_strategy`, `save_steps`, `save_only_model`, `eval_strategy`, `eval_steps` |
| Logging | `report_to`, `project_name`, `run_name`, `logging_steps` |
| Online RL (both trainers) | `num_generations`, `rewards` |
| Online GRPO | `vllm_server_host`, `vllm_server_port` |
| Async GRPO with environments | `rollout_server_url`, `environment_type`, `num_rollout_workers` |

Notes on the ones that cause trouble:

- **`attn_implementation`**: most examples pin it; keep the pin when you copy.
  Left unset, SFT and self-distillation pick the backend for your GPU and model
  (FA4 on Blackwell, FA3 on Hopper). Preference, reward, classification, teacher
  distillation and GRPO default to SDPA. Embedding takes the GPU pick under EP or
  TP and the transformers default otherwise. Per-family exceptions are in the
  [Supported matrix](supported-matrix.md#attention-backends).
- **`model_revision`** pins a Hub commit.
- **`max_concurrent_loading`** caps how many ranks per node load weights at once.
  Unset, it is half the node's GPUs, at most 4. Set `1` on a host short on CPU
  RAM.
- **`vllm_*` fields** parse on both online RL trainers, which share TRL's GRPO
  config. The async rollout block belongs to async GRPO only:
  `rollout_server_url` in an online GRPO config fails to parse.

## Overriding on the command line

Override fields at launch to sweep without copying files:

```bash
halo launch sft my-run.yaml -n 8 --learning_rate=1e-5 --max_length=32000
```

- Flags the launcher doesn't own go to the trainer, parallelism flags included
  (`--expert_parallel_size=8`), so one config can serve several shapes.
- Write each override as `--key=value`, booleans included
  (`--gradient_checkpointing=true`).
- A plain list field takes a comma-separated value.
- A field that holds a dict, a list of dicts or lists, or either a string or a
  list raises on the command line: `dataset`, `rewards`,
  `gradient_checkpointing_kwargs`, `report_to` (except `--report_to=none`). Set
  it in the YAML.

## Defaults Halo changes

The parser applies three defaults that differ from upstream, each only when your
YAML and command line leave the field unset:

- `bf16: true`. It also switches to the memory-saving `AdamWBF16` optimizer
  wherever `optim` is a stock AdamW, except under replicated DDP. Setting
  `fp16: true` turns it off; the two can't both be set.
- `use_liger_kernel: true`.
- `logging_nan_inf_filter: false`. The upstream filter logs the running average
  in place of a NaN, which hides the step where a run diverged.

`strftime` codes in `output_dir` expand at launch (`runs/sft-%m%d-%H%M`). No
other field expands them.

## What a launch refuses

These fail at startup, not deep into training:

- An unknown key. The message names it. This includes ecosystem spellings such
  as TRL's `max_seq_length`; the field is `max_length`.
- YAML 1.1 booleans on a boolean field (`packing: no`, `bf16: off`). YAML 1.2
  reads them as truthy strings, which would invert what you wrote. Use `true` or
  `false`.
- A value outside a field's allowed choices (`advantage_method: banana`).
- An invalid parallelism shape. `ParallelismConfig` rejects it after parsing and
  before any weights load ([Parallelism](parallelism.md)).

## Sequence length per method

`max_length` is the one sequence-length setting; `null` means the model's context
window. `max_prompt_length` and `max_completion_length` mean different things
per trainer:

| Method | What the length settings do |
| --- | --- |
| SFT, DPO, KTO, distillation | `max_length` only. DPO's `generation_max_prompt_length` (default 512) bounds eval-time samples, not training |
| SMPO | prompt and completion budgets come out of `max_length`; an unset prompt budget takes half |
| Offline GRPO | independent truncation caps; set both, and their sum becomes the tokenizer's `model_max_length` |
| Online GRPO | `max_prompt_length` filters the dataset (over-long rows are dropped, not truncated); `max_completion_length` is the generation budget |
| Async GRPO with environments | `max_prompt_length` is the same filter. The per-turn generation budget is `rollout_max_tokens`; a `max_completion_length` other than TRL's default (256) must equal it. `rollout_max_episode_tokens` (default `null`) caps a whole episode and must be at least `rollout_max_tokens` |

Online GRPO and async GRPO have no `max_length` field, so the key fails to parse
there. Offline GRPO parses it, then refuses it when the trainer is built.

## A complete config

`examples/sft/qwen3/qwen3-4b-ultrachat.yaml` is a full fine-tune of a 4B dense
model on UltraChat that runs on 1 to 8 GPUs:

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
attn_implementation: flash_attention_2

dataset:
- HuggingFaceH4/ultrachat_200k@train_sft
conversation_field: messages
assistant_message_template: "<|im_start|>assistant\n"
test_size: 0.01

per_device_train_batch_size: 2
per_device_eval_batch_size: 2
num_train_epochs: 1.0
gradient_accumulation_steps: 8
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
optim: adamw_torch_fused
learning_rate: 2.0e-05
max_grad_norm: 1.0
lr_scheduler_type: cosine
warmup_steps: 32
seed: 42

max_length: 4096
packing: true

output_dir: checkpoints/sft-qwen3-4b-ultrachat
save_strategy: steps
save_steps: 500
eval_strategy: steps
eval_steps: 500
save_total_limit: 3
save_only_model: true

logging_steps: 1
logging_first_step: true
report_to: wandb

dataloader_num_workers: 4
remove_unused_columns: true
use_peft: false
```

Batch size is per device. The global batch is
`per_device_train_batch_size × gradient_accumulation_steps × data_parallel_size`,
so the same file trains a different global batch on 1 GPU and on 8.

`gradient_checkpointing: true` adds about 20–30% to step time and saves a lot of
activation memory. It is usually the first thing to turn on when a run doesn't
fit.

Each [training method](training-methods/README.md) page lists its own keys. Every
field, default and `HALO_*` knob is in the
[Configuration Reference](../agent-docs/reference/configuration-reference.md) ↗.
