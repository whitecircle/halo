# Training Methods

Each method is one script under `scripts/training/` and one name you pass to `halo launch`. All of
them read a YAML config, take a HuggingFace checkpoint and a dataset, and write a HuggingFace
checkpoint back out. If you are still deciding, [Choosing a Method](../choosing-a-method.md) maps the
data you already have to the trainer that reads it.

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

One config format. A config is a flat YAML file of trainer fields, and any field can be overridden
on the command line after the config path. The parser applies the same three toolkit defaults everywhere
(`bf16: true`, `use_liger_kernel: true`, `logging_nan_inf_filter: false`) and raises on a key it does
not know rather than ignoring it — see [Configuration](../configuration.md).

One data and collator stack. Datasets come from the Hub, a local path or `s3://` in the same
`dataset:` field, mix by list, and are rendered with the model's chat template before tokenization
([Datasets](../data.md)). Columns differ per method; everything above them does not.

One parallelism stack. Expert, tensor and expert-tensor parallelism work on every trainer; context
parallelism is enabled on SFT and SMPO only, and pipeline parallelism is not available in this
release. The axis sizes are config fields or CLI flags, identical across methods
([Parallelism](../parallelism.md)).

HuggingFace in, HuggingFace out. Checkpoints are written in standard form, so `from_pretrained`,
vLLM and the Hub take them as they are; LoRA runs write an adapter you can merge with one command
([Checkpoints & Export](../checkpoints.md)).

The exhaustive per-method reference, with every hyperparameter and every refusal, is the
[training-methods section](../../agent-docs/training-methods/README.md) ↗ of `agent-docs`.
