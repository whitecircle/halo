# Supervised Fine-Tuning

SFT is cross-entropy on example conversations: the model learns a format, a behavior or a domain by
imitating the assistant turns you show it. It is where most projects start, and the step before
[preference tuning](preference.md) or RL. The same script handles vision-language data and raw text
too ([Choosing a Method](../choosing-a-method.md) if you are unsure this is the one).

## Data

One record is one conversation, under the column named by `conversation_field` (default `prompt`):

```jsonl
{"prompt": [{"role": "user", "content": "Explain gravity."}, {"role": "assistant", "content": "Gravity is ..."}]}
```

A row longer than `max_length` is **dropped, not truncated** — a conversation cut mid-turn teaches
the model to stop mid-turn, and an emptied split raises. Sources, mixing and offline preparation:
[Datasets](../data.md).

## Config

From `examples/sft/qwen3/qwen3-4b-ultrachat.yaml`, the fields that decide the run:

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
dataset:
- HuggingFaceH4/ultrachat_200k@train_sft
conversation_field: messages
assistant_message_template: "<|im_start|>assistant\n"
test_size: 0.01
max_length: 4096
packing: true
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2.0e-05
lr_scheduler_type: cosine
warmup_steps: 32
gradient_checkpointing: true
output_dir: checkpoints/sft-qwen3-4b-ultrachat
```

`test_size` carves an eval split out of the data. `warmup_steps` is an exact step count at 1 or
above and a fraction of the run below it (`0.03`); there is no `warmup_ratio`. The effective batch is
`per_device_train_batch_size × gradient_accumulation_steps × data_parallel_size`, and full fine-tunes
land at 64–128. A large model wants a learning rate near `2.5e-6`–`5e-6`; small dense models tolerate
more, which is why the 4B example uses `2e-5`. Too high a rate erases pretrained capability without
showing up in the loss curve.

## Three data decisions

**Completion-only masking** (`train_on_completions_only`, on by default) trains on assistant turns
only and masks the prompt. It needs `assistant_message_template`; the startup probe raises only when
*no* message shape renders that marker, so one only a reasoning turn renders masks every row.

**Packing** (`packing: true`) concatenates short rows into fixed `max_length` blocks so no GPU time
goes into padding. It requires an explicit `max_length`, which becomes both the pack size and the
memory bound, and it is the default choice for text SFT.

**Padding-free** (`padding_free: true`) flattens a batch into one variable-length sequence instead.
It needs a varlen Flash Attention kernel and cannot be combined with packing. Context parallelism
refuses both, and so does a vision run — images cannot be packed.

## Chat templates

Every row is rendered with `tokenizer.apply_chat_template` before tokenization, so **train under
the template you will serve under**. A mismatch costs quality and shows up nowhere in the loss.

By default the model's own template is used. `chat_template` overrides it with a path to a `.jinja`
file or an inline template string, and `force_chat_template: true` is required when the tokenizer
already carries one — without it your template is dropped without a word. Bundled templates live in
`jinja-templates/<family>/`; pin a verbatim upstream one (`*-native.jinja`, and `gpt-oss-harmony.jinja`
for GPT-OSS) when the render must match a rollout server byte for byte.

`assistant_message_template` is separate and must byte-match what that template renders for an
assistant turn — a marker that does not match masks every row and the loss goes flat.

## LoRA and QLoRA

LoRA is a config switch, not a different script. `examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml`
adds:

```yaml
use_peft: true
lora_r: 64
lora_alpha: 128
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
lora_task_type: CAUSAL_LM
learning_rate: 1.0e-04     # LoRA wants 5-10x the full fine-tuning rate
```

The `-qlora` counterpart adds `load_in_4bit: true`, `bnb_4bit_quant_type: nf4` and
`use_bnb_nested_quant: true`, which peaks at 7.9 GB for a 4B model at batch 1 and 4k, so it fits a 24 GB GPU (batch 4 packs four rows into one 16k row and peaks at 33 GB). On MoE models `all-linear` misses
the fused expert tensors, so name the expert projections explicitly; LoRA is rejected under tensor
parallelism ([PEFT](../../agent-docs/optimization/peft.md) ↗).

## Vision-language

Nothing to switch on. The model class follows the checkpoint — a multimodal config loads through
`AutoModelForImageTextToText` and its processor, with the same EP/TP wrapping a text model gets —
while the data path follows the run: the VLM pipeline engages only when the dataset declares images,
so text-only rows on a multimodal checkpoint train as a plain text run.

Images ride inside message content, or in a column named by `images_field`, which is what hub
datasets like Docmatix and FineVision do:

```yaml
dataset:
- HuggingFaceM4/Docmatix:zero-shot-exp
conversation_field: texts
images_field: images
max_length: 28672
```

That is `examples/sft/qwen3_5/qwen3.5-9b-vl-docvqa.yaml`. An image run refuses packing and
padding-free at startup and context parallelism at the first forward, and the collator raises rather
than truncating a batch over `max_length` — cutting image tokens would desync them from the pixels.

## Run

```bash
# single GPU, LoRA
halo launch sft examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml

# 8 GPUs, full fine-tune (FSDP2 data parallel)
halo launch sft examples/sft/qwen3/qwen3-4b-ultrachat.yaml -n 8

# 8 GPUs, MoE with expert parallelism pinned in the config
halo launch sft examples/sft/qwen3_5/qwen3.5-35b-a3b-ultrachat-ep.yaml -n 8
```

Override any field after the config (`--learning_rate=1e-5`). Before a long run, smoke the config on
two GPUs with `--max_steps=5 --save_strategy=no --report_to=none`: that catches template, length
and parallelism mistakes in a minute.

## Pre-training and from-scratch

Continued pre-training on raw text is the same trainer. Prepare the corpus once with
`halo run prepare-dataset --mode text --text-field text --pack-sequences ...` — it tokenizes
documents with no chat template and appends an EOS per document — then point `dataset` at the
result. The config must state `train_on_completions_only: false` and the exact `max_length` the
preparation used, or startup raises.

`init_from_scratch: true` builds random weights from the model's config instead of reading a
checkpoint. It runs on dense FSDP2 data parallelism only — EP, TP, CP and ETP raise, since
distributed random-init of sharded parameters is not implemented. For a large or MoE model,
materialize the random-init checkpoint once outside the job (`from_config` plus `save_pretrained`)
and point `model_name_or_path` at it.

## What to watch

The loss should fall smoothly in the first hundred steps; a flat loss almost always means the
assistant marker never matched and every token is masked. Set `log_decoded_samples: true` once to
read exactly what the model is shown, and check the log for how many rows the length filter dropped
(it warns once that passes half the split) before blaming the model.

## Go deeper

- [Preference tuning](preference.md) · [Distillation](distillation.md) · [Checkpoints & Export](../checkpoints.md)
- [SFT reference](../../agent-docs/training-methods/sft.md) ↗ ·
  [Pre-training](../../agent-docs/training-methods/pretraining.md) ↗ · [Collators](../../agent-docs/data/collators.md) ↗
