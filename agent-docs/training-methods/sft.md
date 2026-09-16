# Supervised Fine-Tuning (SFT)

Cross-entropy on conversation data — the step before preference optimization ([DPO/SMPO](preference/README.md)) or RL ([GRPO](grpo/README.md)). Trainer `DistributedSFTTrainer`, script `scripts/training/sft.py`. It takes text and vision-language checkpoints and is the only method that accepts `lowp_precision` QAT. Raw-text corpora and random-init runs use the same trainer — see [Pre-training](pretraining.md).

## Dataset

Conversations under `conversation_field` (default `prompt`):

```jsonl
{"prompt": [{"role": "user", "content": "Explain gravity."}, {"role": "assistant", "content": "Gravity is..."}]}
```

Sources are `s3://`, the HuggingFace Hub, or a local path. `dataset_ratio` is a per-source keep fraction in [0, 1]: a scalar broadcasts across a list of sources, a list maps 1:1. Rows over `max_length` are dropped, not truncated (a conversation cut mid-turn is corrupt), and an emptied train split raises.

Every row is rendered with `tokenizer.apply_chat_template`, so train under the template the model is served with — a mismatch degrades quality with no signal in the loss. The template knobs are in [Chat templates](../data/dataset-formats.md#chat-templates). `interleaved_thinking: true` keeps `<think>…</think>` in history for GLM-family templates that must byte-match rollouts (text-only; the VLM path raises).

## Configuration

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
dataset:
- HuggingFaceH4/ultrachat_200k@train_sft
conversation_field: messages
test_size: 0.01
train_on_completions_only: true
assistant_message_template: "<|im_start|>assistant\n"
max_length: 4096
packing: true
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 3.5e-06
lr_scheduler_type: cosine
warmup_steps: 32
gradient_checkpointing: true
output_dir: checkpoints/sft-qwen3-4b
```

| Knob | Default | Effect |
|---|---|---|
| `train_on_completions_only` | `true` | Mask prompt tokens; loss on assistant turns only |
| `assistant_message_template` | unset | The rendered assistant-turn prefix; required by the flag above and checked against the chat template at startup |
| `train_on_last_assistant_only` | `false` | Loss on the final assistant turn only; needs completions-only |
| `packing` / `packing_strategy` | `false` / `bfd` | Pack short rows into fixed `max_length` blocks |
| `padding_free` | `false` | Flatten the batch into one varlen sequence |
| `max_length` | `1024` | `null` or non-positive → the model context window |
| `use_peft` | `false` | LoRA; 5–10× the full-FT rate ([PEFT](../optimization/peft.md#hyperparameters)) |
| `lowp_precision` | `bf16` | `fp8`/`fp4`/`mxfp4` matmuls over bf16 masters ([QAT](../optimization/low-precision-moe-kernels.md)) |

Refused at startup:

- `packing` together with `padding_free`, and `packing` without an explicit `max_length` (the pack size bounds memory).
- `padding_free` on a non-varlen attention implementation, under CP, or under PP — the flattened width changes every step while the P2P buffers freeze on the first. Use `packing`, except under CP, which refuses both.
- TRL's `completion_only_loss` / `assistant_only_loss`: they act inside the dataset prep and collator this script replaces.

Per-family configs live under `examples/sft/`; full field list in [Configuration Reference](../reference/configuration-reference.md#sftscriptarguments). The parser turns `use_liger_kernel` and `bf16` on and `logging_nan_inf_filter` off; `attn_implementation` auto-selects FA4 on Blackwell, FA3 on Hopper, else FA2 ([Flash Attention](../optimization/flash-attention.md)).

## Launch

```bash
# EP/CP/TP/ETP — axis sizes are CLI flags
torchrun --nproc_per_node=8 scripts/training/sft.py \
    examples/sft/gptoss/gptoss-20b-multinode-ep.yaml --expert_parallel_size=8

# the same through the CLI
halo launch sft examples/sft/qwen3/qwen3-4b-ultrachat.yaml --nproc 8

# single GPU / LoRA
python scripts/training/sft.py examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml
```

Any YAML field overrides on the command line (`--learning_rate=1e-5`); `accelerate launch` with `accelerate/fsdp2_gradop_config.yaml` stays supported for plain data-parallel. Saves are gathered HF-standard checkpoints by default; per-rank `save_sharded_ep` ones need `scripts/after_training/merge_ep_shards.py` before resume or serving, and that merge drops optimizer state ([Checkpoints](../reference/checkpoints.md)).

## Learning rate and global batch size

```text
effective_batch = per_device_train_batch_size × gradient_accumulation_steps × data_parallel_size
```

EP is orthogonal to DP, so `data_parallel_size = world_size` under pure EP; TP, CP, ETP and PP reduce it ([Parallelism](../parallelism/README.md)). `gptoss-20b-multinode-ep.yaml` runs batch 1 × accumulation 4 × DP=16 (2 nodes × 8 GPUs, EP orthogonal to DP) → effective batch 64; production full-FT configs land at **64–128**. Raise `gradient_accumulation_steps` (costs step latency, not memory) when HBM is tight, `per_device_train_batch_size` for throughput.

Too high a learning rate erases pretrained capability without showing up in the training loss.

| Band | Learning rate | Anchor |
|---|---|---|
| Conservative floor | `0.5e-6` – `1.5e-6` | 100B+ MoE; also stage 2 of a two-stage run |
| Default (full FT) | `2.5e-6` – `5e-6` | stage-1 recipes use `3.5e-6` or `5e-6` |
| Aggressive | `8e-6` – `2e-5` | short runs or small dense models |

Pair with `lr_scheduler_type: cosine` and a warmup of ~3–5% of the run. There is no `warmup_ratio`: `warmup_steps` ≥ 1 is an exact step count, below 1 a fraction of the total (`warmup_steps: 0.03`).

## Vision-language models

Two verdicts decide a vision-language run. The **model class follows the checkpoint**: a multimodal config loads through `AutoModelForImageTextToText` + processor, still via `load_distributed_model`, so a MoE VLM gets the same EP/TP/CP wrapping. `text_only_model: true` overrides that — the checkpoint loads through its text-only CausalLM sibling, the vision tower is dropped, and image columns are refused.

The **data path follows the run** (`is_vlm_run`, `src/data/vlm.py`): the VLM path only when the checkpoint is multimodal **and** the run declares image data. A natively-multimodal checkpoint (Gemma 4, Qwen3.5/3.6, Inkling) on text-only rows is a text run, so packing, padding-free and `train_on_last_assistant_only` stay available.

Images ride embedded in message content, or in a column named by `images_field`, pairing with any image placeholders, else the first user turn — so hub datasets that keep images outside the conversation (FineVision, the_cauldron, Docmatix) need only field mappings. Either shape declares the run VLM, as does an `images` / `image` / `pixel_values` column ([rules](../data/dataset-formats.md#sft-vlm)):

```yaml
dataset:
- HuggingFaceM4/FineVision:olmOCR-mix-0225-books
conversation_field: texts
images_field: images
```

Shipped configs: `examples/sft/qwen3_5/qwen3.5-9b-vl-ocr-olmocr.yaml`, `qwen3.5-9b-vl-docvqa.yaml`.

VLM limits are fail-loud and bind the image-declaring **run**, not the multimodal checkpoint: `packing` / `padding_free` (images cannot be packed), `train_on_last_assistant_only` (all assistant turns train), `interleaved_thinking` (no VLM template renders `clear_thinking`), CP (patch features do not slice by token chunk), PP (no stage holds the vision tower), and `generate_eval_examples` (skipped). `init_from_scratch` is the exception, refused on the **checkpoint** at the model load.

The VLM collator never truncates: a batch whose vision plus text tokens exceed `max_length` raises rather than desync placeholders from `pixel_values`.

## Parallelism

Axis sizes are CLI flags: `--expert_parallel_size`, `--context_parallel_size`, `--tensor_parallel_size`, `--expert_tensor_parallel_size`. EP+CP, EP+TP and EP+ETP compose; TP+CP, ETP+CP and EP+TP+ETP are rejected at config time, as is every other unlisted combination, and `--pipeline_parallel_size > 1` is [not yet available in this release](../parallelism/pipeline-parallelism.md). Pure ETP is `--expert_parallel_size=1 --expert_tensor_parallel_size=N` (attention TP and expert TP are mutually exclusive), and LoRA is rejected under TP and EP+TP.

Under CP a batch whose length is not a multiple of `cp_size` is right-padded in `compute_loss`, and a tokenizer with no `pad_token_id` raises there rather than padding with vocabulary token 0. The loss normalizes over the CP group's tokens; metrics come off the local chunk, reduced once per log ([details](../reference/trainer-architecture.md#cp-loss-and-metrics-in-sft)).

See [Trainer Compatibility](../reference/trainer-architecture.md#trainer-compatibility) and [Supported Models](../models/README.md#compatibility-matrix).

## Pre-processed datasets

`scripts/before_training/prepare_dataset.py` tokenizes, packs and shards offline:

```bash
python scripts/before_training/prepare_dataset.py \
    --input "s3://bucket/raw/dataset" --output "s3://bucket/preprocessed/dataset" \
    --model-name "Qwen/Qwen3-8B" --max-length 4096 --test-size 0.01 \
    --num-shards 64 --pack-sequences \
    --assistant-message-template $'<|im_start|>assistant\n'
```

The training script detects the artifact from its `metadata.json`, skips tokenization, and holds the run to what was baked: `max_length` and `train_on_completions_only` must agree or startup raises, as must any render knob the YAML states. `ShardedDatasetLoader` then assigns shards by DP rank ([Pre-Processing](../data/dataset-preparation.md)).

## Testing a setup

Smoke the config first: cut `max_length`, set `max_steps: 5`, launch on 2 GPUs. `examples/sft/deepseek_v4/v4-tiny-random-smoke-ep.yaml` is a tiny-model EP smoke needing no production checkpoint.

```bash
pytest tests/cpu/config tests/cpu/data -m cpu    # config gates, collators, render knobs
torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_ep.py
```

`tests/gpu/trainers/sft/` holds suites per mode (dense, EP, EP+CP, EP+TP, FSDP2 resume, VLM, sinks) and per model.

## Related pages

- [Preference Optimization (SMPO, DPO)](preference/README.md) · [Pre-training](pretraining.md) · [Pipeline Parallelism](../parallelism/pipeline-parallelism.md)
- [Collators](../data/collators.md) · [Padding-Free Collator](../optimization/padding-free-collator.md) · [Checkpoints & Resume](../reference/checkpoints.md)
