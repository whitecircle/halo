# Training Methods

Each method is one script under `scripts/training/` and one name you pass to `halo launch`. Every
method reads a YAML config, takes a HuggingFace checkpoint and a dataset, and writes a HuggingFace
checkpoint. If you are still deciding, [Choosing a Method](../choosing-a-method.md) maps the data you
have to the trainer that reads it.

| Method | Trains on | `halo launch` | Page |
| --- | --- | --- | --- |
| SFT | conversations, text or image | `sft` | [Supervised fine-tuning](sft.md) |
| Pre-training | raw text, or a random-init model | `sft` | [Supervised fine-tuning](sft.md#pre-training-and-from-scratch) |
| SMPO | `prompt` / `chosen` / `rejected`, no reference model | `smpo` | [Preference](preference.md) |
| DPO | `prompt` / `chosen` / `rejected`, against a reference | `dpo` | [Preference](preference.md) |
| KTO | `prompt` / `completion` / boolean `label` | `kto` | [Preference](preference.md) |
| Reward modeling | preference pairs, output is a scalar scorer | `rewards` | [Reward & classification](reward-and-classification.md) |
| Classification | text plus a single- or multi-label `label` | `classification` | [Reward & classification](reward-and-classification.md) |
| Offline GRPO | prompts with completions you already scored | `offline-grpo` | [Offline GRPO](offline-grpo.md) |
| Online GRPO (RLVR) | prompts with a verifiable answer, generated live | `rlvr` | [Online GRPO](online-grpo.md) |
| Async GRPO with Environments | multi-turn, tool-using trajectories | `environmental-grpo` | [Async GRPO](async-grpo-environments.md) |
| Teacher distillation | conversations, plus a second frozen model | `teacher-distill` | [Distillation](distillation.md) |
| Self-distillation | conversations with gold answers, one model | `self-distill` | [Distillation](distillation.md) |
| Online SDPG | prompts with an answer, generated live | `rlvr --use_sdpg=true` | [Distillation](distillation.md#online-sdpg) |
| Embedding | text pairs, triplets or scored pairs | `embedding` | [Embedding](embedding.md) |

## What every method shares

- **One config format.** A config is a flat YAML file of trainer fields, and any field can be
  overridden on the command line after the config path. Every method gets the same toolkit defaults
  (`bf16: true`, `use_liger_kernel: true`, `logging_nan_inf_filter: false`), and an unknown key
  raises ([Configuration](../configuration.md)).
- **One data stack.** Datasets come from the Hub, a local path or `s3://` through the same `dataset:`
  field, and a list mixes them ([Datasets](../data.md)). The columns each method reads differ.
- **One parallelism stack.** Expert, tensor and expert-tensor parallelism work on every trainer.
  Context parallelism covers SFT, SMPO and offline GRPO (full fine-tuning only for offline GRPO).
  Pipeline parallelism is not available in this release ([Parallelism](../parallelism.md)).
- **HuggingFace in, HuggingFace out.** Checkpoints load with `from_pretrained` and upload to the Hub
  as written. Some MoE families need a conversion before vLLM or SGLang can serve them. A LoRA run
  writes an adapter you merge with one command; expert LoRA under EP folds into the weights at save
  time instead (`merge_expert_lora_on_save: true`). See [Checkpoints & Export](../checkpoints.md).

Every hyperparameter and refusal per method is in the
[training-methods reference](../../agent-docs/training-methods/README.md) ↗.
