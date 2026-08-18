# Supervised Fine-Tuning (SFT)

Cross-entropy on conversation data — the step before preference optimization (DPO, SMPO, GRPO). Trainer `DistributedSFTTrainer`, script `scripts/training/sft.py`. SFT supports every available parallelism axis — EP, CP, TP, ETP.

## Dataset format

Conversations under `conversation_field` (default `prompt`):

```jsonl
{"prompt": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Explain gravity."}, {"role": "assistant", "content": "Gravity is a fundamental force..."}]}
```

Sources are S3 (`s3://bucket/path`), the HuggingFace Hub, or a local path. `dataset_ratio` is a per-source keep fraction in [0, 1] — it downsamples only, never upweights. A scalar broadcasts across a list of sources; a list maps 1:1 and must match its length.

## Quick start

```bash
torchrun --nproc_per_node=8 scripts/training/sft.py \
    examples/sft/gptoss/gptoss-20b-multinode-ep.yaml --expert_parallel_size=8
```

Minimal config:

```yaml
model_name_or_path: Qwen/Qwen3-4B
dataset: path/to/dataset
conversation_field: prompt
train_on_completions_only: true
assistant_message_template: "<|im_start|>assistant\n"
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 4.0e-06
num_train_epochs: 1
max_length: 16000
gradient_checkpointing: true
output_dir: checkpoints/sft-qwen3
report_to: wandb
```

Per-family configs live under `examples/sft/`. `accelerate launch --config_file launcher-configs/accelerate/fsdp2_gradop_config.yaml` remains supported for plain data-parallel runs.

## Key features

**Train only on completions** — `train_on_completions_only: true` (default) masks prompt tokens so loss lands only on assistant turns. `assistant_message_template` must byte-match the rendered assistant-turn prefix (`<|im_start|>assistant\n` for Qwen/ChatML, `<|start_header_id|>assistant<|end_header_id|>\n\n` for Llama-3); it has no default, and a missing or non-rendering marker raises at startup rather than mis-masking. `train_on_last_assistant_only: true` narrows loss to the final assistant turn and requires `train_on_completions_only: true`. TRL's `completion_only_loss` / `assistant_only_loss` are rejected at startup — they act inside the TRL dataset prep and default collator this script replaces, so they would mask nothing.

**Packing** — `packing: true` concatenates short sequences into fixed `max_length` blocks (`packing_strategy`: `bfd` default, `bfd_split`, `wrapped`). Mutually exclusive with `padding_free`, and requires an explicit `max_length` — it cannot fall back to the model context window. Pre-packed datasets are detected and skip re-packing.

**Padding-free** — `padding_free: true` flattens the batch into one varlen sequence. It needs a varlen Flash Attention kernel (FA2/FA3/FA4) and raises on any other implementation: only those consume the `cu_seq_lens` this collator exists to emit, so elsewhere it buys nothing and still pays a dense mask over the flattened batch's whole token count — use `packing` there. Also rejected under Context Parallelism, and under Pipeline Parallelism (the flattened width varies every step while a pipeline's P2P buffers would freeze on the first).

**PEFT / LoRA** — `use_peft: true` with `lora_r`, `lora_alpha`, `lora_target_modules`. LoRA needs ~10× the full-FT LR (`1e-4` in `examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml`). See [PEFT — Hyperparameters](../optimization/peft.md#hyperparameters).

**Quantization-aware training** — `lowp_precision: fp8|fp4|mxfp4` runs the matmuls in a block-scaled low-precision format over bf16 master weights (parameters and checkpoint stay bf16). SFT is the only method that accepts it: train to confirm the model converges in the target format, then export with `scripts/after_training/quantize_to_lowp.py`. See [Low-Precision MoE Kernels](../optimization/low-precision-moe-kernels.md).

## Learning rate and global batch size

```text
effective_batch = per_device_train_batch_size × gradient_accumulation_steps × data_parallel_size
```

EP is orthogonal to DP, so `data_parallel_size = world_size` under pure EP; TP, CP, ETP and PP reduce it (see [Parallelism](../parallelism/index.md)). `gptoss-20b-multinode-ep.yaml` runs batch 1 × accumulation 4 × DP=16 (2 nodes × 8 GPUs, EP orthogonal to DP) → effective batch 64.

Production full-FT configs land at effective batch **64–128**. Raise `gradient_accumulation_steps` (costs step latency, not memory) when HBM is tight, `per_device_train_batch_size` for throughput.

A learning rate that is too high erases pretrained capability without showing up in the training loss.

| Band | Learning rate | Anchor |
|---|---|---|
| Conservative floor | `3e-7` – `1e-6` | large MoE — down to `0.5e-6` at 100B+ |
| Default (full FT) | `2e-6` – `5e-6` | two-stage SFT: `5e-6` then `2.5e-6` |
| Aggressive | `1e-5` – `1.5e-5` | short runs or smaller models |

A safe default for a new full-FT run is **`2e-6`**; above `~1e-5` risks base-capability regression. Pair with `lr_scheduler_type: cosine` and a warmup of ~3–5% of the run. `warmup_ratio` is not a `TrainingArguments` field: `warmup_steps` carries both spellings — an integer is an exact step count, a float in [0, 1) a ratio of the total (`warmup_steps: 0.03`).

## Chat templates

Every conversation is rendered with `tokenizer.apply_chat_template` before tokenizing. Train under the template the model is served with — a mismatch degrades quality silently with no signal in the loss.

- `chat_template` takes a `.jinja`/`.jinja2`/`.j2` path or a raw string, and falls back to the tokenizer's built-in template. `force_chat_template: true` replaces a template the tokenizer already ships. Register new control tokens with `added_special_tokens` so they tokenize atomically.
- A modified template is a format the base model has never produced; the further it departs from the native instruct template, the more SFT data it takes.
- For GLM-family templates that must byte-match rollouts keeping `<think>…</think>` in history, set `interleaved_thinking: true` (text-only — the VLM path raises).

## Vision-language models

The same script handles both modalities, but two verdicts decide it, not one — plus one override.

The **model class follows the checkpoint**: a multimodal config loads through `AutoModelForImageTextToText` + processor, still via `load_distributed_model`, so a MoE VLM gets the same EP/TP/CP wrapping as the text path. The **data path follows the run** (`is_vlm_run`, `src/data/vlm.py`): it is the VLM path only when the checkpoint is multimodal **and** the run declares image data. A natively-multimodal checkpoint (Gemma 4, Qwen3.5/3.6, Inkling) trained on text-only rows is a text run — packing, padding-free and `train_on_last_assistant_only` stay available on it.

`text_only_model: true` overrides the first verdict: the multimodal checkpoint loads through its text-only CausalLM sibling, dropping the vision tower, and the run takes the text path whatever the data says (image columns are refused). Every VLM limit below is then moot — including the `init_from_scratch` refusal, which lives inside the branch the flag skips.

Images ride embedded in message content, or in a separate column named by `images_field` — the pipeline injects that column into the first user turn, so hub datasets that keep images outside the conversation (FineVision, the_cauldron, Docmatix) train with just field mappings. Either shape declares the run VLM, as does an `images` / `image` / `pixel_values` column ([declaration rules](../data/dataset-formats.md#sft-vlm)):

```yaml
dataset:
- HuggingFaceM4/FineVision:olmOCR-mix-0225-books
conversation_field: texts
images_field: images
```

See `examples/sft/qwen3_5/qwen3.5-9b-vl-ocr-olmocr.yaml` and `qwen3.5-9b-vl-docvqa.yaml`.

VLM limits are all fail-loud, and all bind the image-declaring **run** rather than the multimodal checkpoint:

- `packing` / `padding_free` — images cannot be packed.
- `train_on_last_assistant_only` — all assistant turns train.
- `interleaved_thinking` — no supported VLM template renders `clear_thinking`.
- `generate_eval_examples` is skipped: the generation callback needs tokenized `input_ids`.
- CP is text-only — a batch carrying `pixel_values` raises, since patch features do not slice by token chunk.
- `init_from_scratch` is the exception: it is refused on the **checkpoint**, at the model load.

The collator never truncates. A batch whose vision plus text tokens exceed `max_length` raises, because cutting expanded image-placeholder tokens would desync them from `pixel_values` — budget `max_length` with headroom for image tokens. Raw VLM data is mapped to `history`/`images` rows under a pinned Arrow schema and pre-filtered on rendered text length; those mapped columns are not forward kwargs, so the script forces `remove_unused_columns=False`.

## Parallelism

Pass the axis sizes as CLI flags on the `torchrun` line: `--expert_parallel_size` (MoE), `--context_parallel_size` (long sequences), `--tensor_parallel_size` (large dense), `--expert_tensor_parallel_size` (expert-FFN sharding). EP+CP, EP+TP and EP+ETP compose; every other pair is rejected at config time, and `--pipeline_parallel_size > 1` is [not yet available in this release](../parallelism/pipeline-parallelism.md).

- Pure ETP is `--expert_parallel_size=1 --expert_tensor_parallel_size=N`; attention TP and expert TP are mutually exclusive.
- TP+CP, ETP+CP and EP+TP+ETP are not supported.
- LoRA is rejected under TP and EP+TP.
- CP is not compatible with `padding_free`.
- Under CP, a batch whose length is not a multiple of `cp_size` is right-padded in `compute_loss`; a tokenizer with no `pad_token_id` raises there instead of padding with vocabulary token 0.
- Under CP the trainer computes `mean_token_accuracy`, `entropy`, `aux_loss` and `num_attended_tokens_seen` itself, on the local chunk. Each is a sum, so the micro-batches accumulate into one fixed-width on-device row that is reduced **once per log** rather than five to six times per micro-batch. `num_attended_tokens_seen` is unchanged by the batching; the two ratios become token-weighted over the log window, which equals the per-micro-batch average whenever the micro-batches carry equal token counts.

See [Trainer Compatibility](../reference/trainer-architecture.md#trainer-compatibility) for the trainer matrix, [Supported Models](../models/index.md#compatibility-matrix) for model × mode coverage, and [Pipeline Parallelism](../parallelism/pipeline-parallelism.md).

## Pre-processed datasets

Tokenize, pack, and shard offline to drop tokenization from the training job and load per-rank shards:

```bash
python scripts/before_training/prepare_dataset.py \
    --input "s3://bucket/raw/dataset" \
    --output "s3://bucket/preprocessed/dataset" \
    --model-name "Qwen/Qwen3-8B" \
    --num-shards 64 --pack-sequences \
    --assistant-message-template $'<|im_start|>assistant\n'
```

The script detects a pre-processed dataset from its `metadata.json` and skips tokenization; `ShardedDatasetLoader` assigns shards by DP rank.

It also holds the artifact to the run's config: a `max_length` mismatch in either direction, a `train_on_completions_only` mismatch, and any chat-render knob the run states differently from the recorded one all raise. `packing: true` against an unpacked artifact only warns — preprocessed rows are never packed at runtime. See [SFT Dataset Pre-Processing](../data/dataset-preparation.md).

## Configuration reference

`SFTScriptArguments`, `SFTConfig`, and the parallelism flags are tabulated in [Configuration Reference](../reference/configuration-reference.md#sftscriptarguments). Three toolkit defaults differ from upstream: `use_liger_kernel` and `bf16` are `True`, `logging_nan_inf_filter` is `False`. `attn_implementation` (a `ModelConfig` field, default `None`) auto-selects FA4 on Blackwell, FA3 on Hopper, else FA2 — see [Flash Attention](../optimization/flash-attention.md).

Gathered saves in HF-standard layout are the default; per-rank `save_sharded_ep` checkpoints must be reassembled with `merge_ep_shards.py`. See [Checkpoints & Resume](../reference/checkpoints.md).

## Related pages

- [Preference Optimization (SMPO, DPO)](preference/index.md) — next step after SFT
- [Pre-training](pretraining.md) · [SFT Dataset Pre-Processing](../data/dataset-preparation.md) · [Collators](../data/collators.md)
- [Padding-Free Collator](../optimization/padding-free-collator.md)
- [Checkpoints & Resume](../reference/checkpoints.md) · [Configuration Reference](../reference/configuration-reference.md)
