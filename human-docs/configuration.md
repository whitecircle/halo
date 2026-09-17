# Writing a Config

A run is one flat YAML file: the HuggingFace `TrainingArguments` fields you
already know, plus Halo's own. What you read is what the trainer gets.

## Start from the nearest example

Don't write one from scratch. `examples/` is organised by method, then by model
family (`examples/sft/qwen3/`, `examples/preference/qwen3_5/`), and the closest
file is usually within two edits of what you want:

```bash
ls examples/sft/                          # families with an SFT recipe
cp examples/sft/qwen3/qwen3-4b-ultrachat.yaml my-run.yaml
halo launch sft my-run.yaml -n 8
```

The examples carry comments explaining every non-obvious choice for that family —
why `sdpa` and not Flash Attention, why `packing` and not `padding_free`, which
router-balancing mode exports. Keep them when you copy; they are the family's
constraints, not decoration.

## The blocks a config has

| Block | The keys that matter |
| --- | --- |
| Model | `model_name_or_path`, `model_revision`, `attn_implementation`, `trust_remote_code`, `max_concurrent_loading` |
| Data | `dataset`, `dataset_ratio`, `test_size`, `max_length`, `packing`, `train_on_completions_only` |
| Training | `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `num_train_epochs`, `lr_scheduler_type`, `gradient_checkpointing` |
| Precision / optimizer | `bf16`, `optim`, `bf16_optimizer`, `fp32_grad_reduce`, `fp32_non_ep_params` |
| Parallelism | `expert_parallel_size`, `context_parallel_size`, `tensor_parallel_size`, `expert_tensor_parallel_size`, `ep_scope` |
| PEFT | `use_peft`, `lora_r`, `lora_alpha`, `lora_target_modules` |
| Checkpoint / eval | `output_dir`, `save_strategy`, `save_steps`, `save_only_model`, `eval_strategy`, `eval_steps` |
| Logging | `report_to`, `project_name`, `run_name`, `logging_steps` |
| RL / rollout | both: `num_generations`, `vllm_server_host`, `vllm_server_port` · async GRPO with environments only: `rollout_server_url`, `environment_type`, `rewards`, `num_rollout_workers` |

A few notes on the ones that bite:

- `attn_implementation` — leave it unset and Halo picks the backend for your GPU
  and model (FA4 on Blackwell, FA3 on Hopper, per-family fallbacks where a kernel
  is known-broken). Most examples pin one anyway; copy the pin with the config.
- `model_revision` pins a Hub commit, and `max_concurrent_loading` caps how many
  ranks per node load weights at once — unset it resolves to half the node's GPUs
  capped at 4, and `1` rescues a CPU-RAM-tight host.
- The two RL trainers share TRL's GRPO config, so the `vllm_*` fields parse on
  both. The async-rollout block is async GRPO's alone: `rollout_server_url` in an
  online-GRPO YAML fails to parse.

## Overriding on the command line

Most fields can be overridden at launch, which is how you sweep without copying files:

```bash
halo launch sft my-run.yaml -n 8 --learning_rate=1e-5 --max_length=32000
```

Every flag the launcher does not own goes to the trainer untouched, parallelism flags included
(`--expert_parallel_size=8`), so one config can serve several shapes. Write each
override as `--key=value`; a field holding a dict or a list of dicts (`rewards`,
`gradient_checkpointing_kwargs`) raises and has to be set in the YAML.

## What the parser does before your config reaches the trainer

Three defaults differ from upstream, each applied only if your YAML and CLI leave
the field alone:

- `bf16: true` — which also brings in the memory-lean `AdamWBF16` optimizer
  wherever `optim` is left at a stock AdamW (replicated DDP is the exception). An
  explicit `fp16: true` suppresses it; the two cannot both be set.
- `use_liger_kernel: true`.
- `logging_nan_inf_filter: false` — upstream's filter logs the running average in
  a NaN's place, hiding the step a run diverged on.

And four things a launch refuses up front rather than deep inside training — the
first three in the parser:

- An unknown or retired key, named in the message. No spelling is migrated — TRL's
  old `max_seq_length` (now `max_length`) raises like any other.
- YAML 1.1 booleans on a boolean field (`packing: no`, `bf16: off`) — YAML 1.2
  reads those as truthy strings, inverting what you wrote. Use `true` / `false`.
- A value outside a field's declared choices (`advantage_method: banana`).
- An invalid parallelism shape, rejected by `ParallelismConfig` after the parser
  runs and before any weights load ([Parallelism](parallelism.md)).

`output_dir` is the one field where `strftime` codes expand (`runs/sft-%m%d-%H%M`); other `%` prose survives.

## Sequence length, per method

`max_length` is the single sequence-length knob; `null` resolves to the model's
context window. `max_prompt_length` and `max_completion_length` mean different
things per trainer:

| Method | What the two knobs do |
| --- | --- |
| SFT, DPO, KTO, distillation | `max_length` only; DPO's `generation_max_prompt_length` (default 512) bounds eval-time samples, not training |
| SMPO | shares carved out of `max_length` — an unset prompt takes half, the completion the rest |
| Offline GRPO | independent truncation caps; set both and their sum becomes the tokenizer's `model_max_length` |
| Online GRPO, async GRPO with environments | `max_prompt_length` is a dataset *filter* (over-long rows are dropped, not truncated); `max_completion_length` is the generation budget |

The two online trainers declare no `max_length` at all, so the key fails to parse
there; offline GRPO's parses and is then refused at trainer construction.

## A complete config

`examples/sft/qwen3/qwen3-4b-ultrachat.yaml` — a full fine-tune of a 4B dense
model on UltraChat, runnable on 1–8 GPUs:

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

Batch size is per device: the effective batch is
`per_device_train_batch_size × gradient_accumulation_steps × data_parallel_size`,
so the same file trains a different global batch on 1 GPU and on 8.
`gradient_checkpointing: true` costs 20–30% throughput for a large
activation-memory saving — usually the first lever when a run doesn't fit.

Per-method keys live on each [training method](training-methods/README.md) page.
The exhaustive field list, every default and every `HALO_*` knob:
[Configuration Reference](../agent-docs/reference/configuration-reference.md) ↗.
