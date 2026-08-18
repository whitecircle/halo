# Configuration

Training is YAML-driven. A config is a flat file of trainer fields: the
HuggingFace `TrainingArguments` you already know, plus Halo's additions. Any
field can be overridden on the command line:

```bash
halo launch sft my-config.yaml -- --learning_rate=1e-5 --max_length=32000
```

The fastest way to write a config is to copy the closest one in `examples/`
and edit it. Field-by-field reference:
[Configuration Guide](../agent-docs/getting-started/configuration.md) ↗
and the full
[Configuration Reference](../agent-docs/reference/configuration-reference.md) ↗.

## The blocks every config has

| Block | Common fields |
| --- | --- |
| Model | `model_name_or_path`, `model_revision`, `attn_implementation`, `trust_remote_code`, `use_liger_kernel` |
| Data | `dataset`, `max_length`, `packing`, `train_on_completions_only`, `dataset_num_proc`, `tokenizer_backend` |
| Training | `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `num_train_epochs`, `gradient_checkpointing` |
| Precision / optimizer | `bf16`, `optim`, `bf16_optimizer`, `fp32_grad_reduce` |
| Parallelism | `expert_parallel_size`, `context_parallel_size`, `tensor_parallel_size`, `expert_tensor_parallel_size` |
| PEFT | `use_peft`, `lora_r`, `lora_alpha`, `lora_target_modules` |
| Checkpoint / eval | `output_dir`, `save_strategy`, `save_steps`, `eval_strategy`, `eval_steps` |
| Logging | `report_to`, `project_name`, `run_name`, `logging_steps` |
| RL / rollout (online GRPO) | `vllm_server_host`, `vllm_server_port`, `num_generations` |
| RL / rollout (environmental GRPO) | `rollout_server_url`, `environment_type`, `num_generations` |

The two RL trainers parse different config classes, so their server fields are
not interchangeable: `rollout_server_url` and `environment_type` exist only for
environmental GRPO, and an online-GRPO YAML carrying them fails to parse. The
online fields keep the `vllm_` prefix because TRL owns them.

Each method's page in the
[reference](../agent-docs/training-methods/sft.md) ↗
expands its own block into exact fields.

## Loading the model

`model_name_or_path` takes a Hub ID or a local directory. Useful companions:

- `model_revision` — pin a specific Hub commit when you need reproducibility;
  the shipped Laguna configs pin one because their remote code moves.
- `attn_implementation` — leave unset. Halo auto-selects the best backend for
  your GPU and model (FA4 on Blackwell, FA3 on Hopper, with per-model
  fallbacks where a kernel is known-broken). Only set it to force something.
- `max_concurrent_loading` — how many ranks per node load weights at once.
  Unset, it resolves to half the node's GPUs, capped at 4. Lower it to 1 on a
  CPU-RAM-tight host; `0` lifts the throttle entirely.

## Defaults the parser applies

Three Halo defaults differ from upstream. Each applies only when your YAML and
CLI leave the field alone:

- `bf16: true`, which also brings in the memory-lean `AdamWBF16` optimizer on
  FSDP/EP/TP/CP runs. An explicit `fp16: true` suppresses it; the two cannot
  both be set.
- `use_liger_kernel: true`.
- `logging_nan_inf_filter: false`. Upstream's filter reads the loss back to the
  host on every micro-batch and logs the running average in a NaN's place. A
  diverged run should be visible in the loss curve at the step it diverged.

The parser expands `strftime` codes in `output_dir` (`runs/sft-%m%d-%H%M`), and
only there. It migrates no spelling: every retired key raises and names the
field, TRL's old `max_seq_length` (now `max_length`) included, so an
out-of-date YAML fails loudly instead of being quietly renamed.

It also refuses two things outright. YAML 1.1 boolean spellings (`packing: no`,
`bf16: off`) raise on boolean fields, because YAML 1.2 parses them as truthy
strings and would silently invert what you wrote — use unquoted `true`/`false`.
And a value outside a field's declared choices (`advantage_method: banana`)
fails at parse time instead of deep inside training.

## Sequence length

`max_length` is the single sequence-length knob, and `null` resolves to the
model's context window at startup. The GRPO trainers are the exception. Online
and environmental GRPO declare no `max_length` at all, so the key fails to parse
(`Some keys are not used by the HfArgumentParser`); offline GRPO declares it for
pipeline-parallel runs only, so the key parses and is then rejected at trainer
construction in this release.

`max_prompt_length` and `max_completion_length` mean different things per
method:

| Method | What the two knobs do |
| --- | --- |
| SMPO | shares carved out of `max_length` — an unset prompt takes half, the completion takes the rest |
| Offline GRPO | independent truncation caps; when both are set, their sum becomes the tokenizer's `model_max_length` |
| Online / environmental GRPO | `max_prompt_length` is a dataset *filter* (over-long rows are dropped, not truncated); `max_completion_length` is the generation budget |
| DPO | no training-side prompt cap; `generation_max_prompt_length` (default 512) bounds only the eval-time generation samples |

## Sanity notes

- Batch size is per device: effective batch =
  `per_device_train_batch_size × gradient_accumulation_steps × data_parallel_size`.
- `gradient_checkpointing: true` trades ~19% throughput for a large activation
  memory saving — usually the first lever when you're out of memory.
- Invalid parallelism combinations are rejected at startup, before any GPU time
  is spent; see [Parallelism](parallelism.md).
