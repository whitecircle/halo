# Supervised Fine-Tuning (SFT)

Cross-entropy on conversation data — the step before preference optimization ([DPO/SMPO](preference/README.md)) or RL ([GRPO](grpo/README.md)). Trainer `DistributedSFTTrainer`, script `scripts/training/sft.py`. It takes text and vision-language checkpoints and is the only method that accepts `lowp_precision` QAT. Raw-text corpora and random-init runs use the same trainer — see [Pre-training](pretraining.md).

## Dataset

Conversations under `conversation_field` (default `prompt`):

```jsonl
{"prompt": [{"role": "user", "content": "Explain gravity."}, {"role": "assistant", "content": "Gravity is..."}]}
```

Sources are `s3://`, the HuggingFace Hub, or a local path. `dataset_ratio` is a per-source keep fraction in (0, 1]: a scalar broadcasts across a list of sources, a list maps 1:1. Rows over `max_length` are dropped, not truncated (a conversation cut mid-turn is corrupt), and an emptied train split raises.

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

`train_on_completions_only` masks prompt tokens, so the loss covers assistant turns only; it needs `assistant_message_template`, the rendered assistant-turn prefix, which startup checks against the chat template. `train_on_last_assistant_only` narrows the loss to the final assistant turn. `packing` packs short rows into fixed `max_length` blocks, [`tree_packing`](../optimization/tree-packing.md) stores a prefix several rows share once, and `padding_free` flattens the batch into one varlen sequence. LoRA (`use_peft`) takes a higher learning rate than a full fine-tune: the shipped SFT LoRA recipes run `1e-4` ([PEFT](../optimization/peft.md#hyperparameters)). `lowp_precision` runs fp8/fp4 matmuls over bf16 masters ([QAT](../optimization/low-precision-moe-kernels.md)).

Refused at startup:

- `packing` together with `padding_free`, `packing` without an explicit `max_length` (the pack size bounds memory), and `eval_packing: true` without `packing`.
- Multi-document rows a family's forward cannot keep apart ([Document isolation](../data/collators.md#document-isolation-under-packing)): `packing` on a dense backend for GPT-OSS, Bailing/Ling and GLM-5 Next, `padding_free` on an M-RoPE decoder (Qwen3.5/3.6, Qwen-VL), either on DeepSeek-V4's compressed KV or on GatedDeltaNet layers without their segment-aware kernels, and either on linear-attention layers or short convolutions (Inkling) no segment marker reaches unless `allow_cross_document_linear_attention: true`.
- `padding_free` on a non-varlen attention implementation, under CP (which refuses `packing` too) or under PP (use `packing`; [PP packing](../parallelism/pipeline-parallelism.md#packing)).
- A pinned `flash_attention_4` the load keeps for a softcapped model (Gemma 2 27B; at head_dim 256 the resolver moves the pin to FA2) on packed, padding-free or padded rows (more than one per batch, training or evaluation), before the load: FA4's varlen kernels are not validated with a softcap.
- TRL's `completion_only_loss` / `assistant_only_loss`, and a non-default `dataset_text_field` / `dataset_kwargs`: they act inside the dataset prep and collator this script replaces. The script renders `conversation_field` itself; tokenize a raw-text column offline with `prepare_dataset.py --mode text`.
- TRL's `chat_template_path`: the trainer applies it after the script rendered every row, so the checkpoint would export a template the model never trained on. Set `chat_template` instead, with `force_chat_template: true` over a template the checkpoint ships.

With `use_liger_kernel: false`, TRL 1.14 resolves an unset `loss_type` to `chunked_nll`, which projects `lm_head` outside the model's forward and so skips the family's head transform and the router aux loss; the trainer replaces any `chunked_nll`, set or resolved, with `nll` and warns.

Per-family configs live under `examples/sft/`; every field's type and default is in [SFTConfig](../reference/configuration-reference.md#trl-trainer-configs) and [SFTScriptArguments](../reference/configuration-reference.md#sftscriptarguments). The parser turns `use_liger_kernel`, `bf16` and (with pinned batches) `accelerator_config.non_blocking` on and `logging_nan_inf_filter` off; `attn_implementation` auto-selects FA4 on Blackwell, FA3 on Hopper, else FA2. On Blackwell an unpinned run loads FA2 instead for packed rows on the head_dim-256 GatedDeltaNet families (Qwen3.5/3.6, Qwen3-Next) and padding-free ones on Qwen3-Next, and for a softcapped model whose rows reach FA4's varlen call ([Flash Attention](../optimization/flash-attention.md#model-specific-handling)); a pre-processed corpus counts as it was baked.

## Launch

```bash
# EP/CP/TP/ETP — axis sizes are CLI flags
torchrun --nproc_per_node=8 scripts/training/sft.py \
    examples/sft/qwen3_5/qwen3.5-35b-a3b-ultrachat-ep.yaml --expert_parallel_size=8

# the same through the CLI
halo launch sft examples/sft/qwen3/qwen3-4b-ultrachat.yaml --nproc 8

# single GPU / LoRA
python scripts/training/sft.py examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml
```

Scalar YAML fields override on the command line (`--learning_rate=1e-5`; container fields such as `dataset` stay in the YAML, [rules](../reference/configuration-reference.md#cli-overrides)); `accelerate launch` with `launcher-configs/accelerate/fsdp2_gradop_config.yaml` stays supported for plain data-parallel. Saves are gathered HF-standard checkpoints by default; per-rank `save_sharded_ep` ones need `scripts/after_training/merge_ep_shards.py` before resume or serving; the merged directory resumes exactly on the save's topology ([Checkpoints](../reference/checkpoints.md#expert-parallelism-ep-eptp-epcp)).

## Learning rate and global batch size

```text
effective_batch = per_device_train_batch_size × gradient_accumulation_steps × data_parallel_size
```

EP is orthogonal to DP, so `data_parallel_size = world_size` under pure EP; PP, TP, CP and ETP reduce it ([Stacking the dimensions](../parallelism/README.md#stacking-the-dimensions)). `gptoss-20b-multinode-ep.yaml` runs batch 1 × accumulation 4 × DP=16 (2 nodes × 8 GPUs, EP orthogonal to DP) → effective batch 64; production full-FT configs land at **64–128**. Raise `gradient_accumulation_steps` (costs step latency, not memory) when HBM is tight, `per_device_train_batch_size` for throughput.

Too high a learning rate erases pretrained capability without showing up in the training loss.

| Band | Learning rate | Anchor |
|---|---|---|
| Conservative floor | `0.5e-6` – `1.5e-6` | stage 2 of a two-stage run |
| Default (full FT) | `2.5e-6` – `5e-6` | most shipped full-FT recipes use `5e-6` |
| Aggressive | `8e-6` – `2e-5` | short runs or small dense models |

Pair with `lr_scheduler_type: cosine` and a warmup of ~3–5% of the run. There is no `warmup_ratio`: `warmup_steps` ≥ 1 is an exact step count, below 1 a fraction of the total (`warmup_steps: 0.03`).

## Vision-language models

Two verdicts decide a vision-language run. The **model class follows the checkpoint**: a multimodal config loads through `AutoModelForImageTextToText`, still via `load_distributed_model`, so a MoE VLM gets the same EP/TP/CP wrapping. `text_only_model: true` overrides that — the checkpoint loads through its text-only CausalLM sibling, the vision tower is dropped, and image columns are refused.

The **data path follows the run** (`is_vlm_run_across_ranks`, `halo/data/vlm_detection.py`): the VLM path only when the checkpoint is multimodal **and** the run declares image data. A natively-multimodal checkpoint (Gemma 4, Qwen3.5/3.6, Inkling) on text-only rows is a text run, so packing, padding-free and `train_on_last_assistant_only` stay available.

The processing class is the checkpoint's processor wherever it ships a processor config (`processor_config.json`, `preprocessor_config.json` or `video_preprocessor_config.json`), text runs included: the export saves it, and vLLM builds the multimodal class's processor from it at startup. GRPO keeps the tokenizer as its processing class, since TRL reads a processor as a vision run, and its exports carry the processor beside it (`load_export_processor`). A multimodal checkpoint shipping none ([Step-3.7 Flash](../models/step3p7.md#model-loading)) trains text runs through its tokenizer and refuses an image run at the processor load (`halo/entrypoints/model_setup.py`).

Images ride embedded in message content, or in a column named by `images_field`, pairing with any image placeholders, else the first user turn — so hub datasets that keep images outside the conversation (FineVision, the_cauldron, Docmatix) need only field mappings. Either shape declares the run VLM, as does an `images` / `image` / `pixel_values` column ([rules](../data/dataset-formats.md#sft-vlm)):

```yaml
dataset:
- HuggingFaceM4/FineVision:olmOCR-mix-0225-books
conversation_field: texts
images_field: images
```

Shipped configs: `examples/sft/qwen3_5/qwen3.5-9b-vl-ocr-olmocr.yaml`, `qwen3.5-9b-vl-docvqa.yaml`.

Three options apply to every script with a vision path. `freeze_vision_tower: true` freezes the vision and audio towers and their projectors (`media_module_names`, `halo/models/module_tree.py`), training the text tower on image data; adapters freeze the base already and refuse it, as does a checkpoint with no tower. A trained tower takes batches mixing image and text rows under `torchrun`; [Shard groups](../parallelism/data-parallelism.md#shard-groups) names the two setups that refuse a batch with no image. `min_pixels` and `max_pixels` bound the pixels per image, and so the tokens each one costs, on a processor that resizes by pixel count (the Qwen family; others refuse them, and a pre-processed dataset keeps what it baked). They land on the processor, so the export carries them.

VLM limits are fail-loud and bind the image-declaring **run**, not the multimodal checkpoint: `packing` / `padding_free` (images cannot be packed), `train_on_last_assistant_only` (all assistant turns train), `interleaved_thinking` (no VLM template renders `clear_thinking`), CP (patch features do not slice by token chunk), and `generate_eval_examples` (skipped). `init_from_scratch` is the exception, refused on the **checkpoint** at the model load.

The VLM collator never truncates: a batch whose vision plus text tokens exceed `max_length` raises rather than desync placeholders from `pixel_values`. With `train_on_completions_only` it refuses, where it is built, a marker the processor's chat template never renders and a template whose assistant turns end in no turn terminator; set `eos_token` to the token such a template closes a turn with ([Collators → Constraints](../data/collators.md#constraints)).

**PaliGemma / PaliGemma 2** are prefix LMs: image and prompt tokens attend bidirectionally, the trained tokens causally, a split the model reads from `token_type_ids` (0 prefix, 1 suffix). Its processor marks no suffix in a rendered conversation, so the VLM collators write `token_type_ids` themselves (`prefix_lm_token_type_ids`): 0 up to the first scored label, 1 from there on, which puts every trained token in the causal suffix (with `train_on_completions_only: false` the suffix opens at the first scored prompt token, so the image tokens after it attend causally). The SDPG teacher branch gets its own. The attention side, no flash kernel and a causal decoder wherever no mask overrides it, is in [Flash Attention → Model-specific handling](../optimization/flash-attention.md#model-specific-handling).

The VLM render applies a chat template, so a PaliGemma checkpoint whose processor carries none needs `chat_template` set. PaliGemma's processor puts `<bos>` after each row's last image and appends `\n` to each row. Every row of an image run needs an image: in a row without one the processor splices `<bos>` into the text, so the collator refuses it. A chat template that renders `<bos>` is refused at the collator's construction, since every row would hold two: remove the `{{ bos_token }}` a Gemma template opens with. The appended newline lands after the conversation and never trains; on a prompt the processor serves it lands after the generation prompt, where no training row has one. A Gemma template closes turns with `<end_of_turn>`, not PaliGemma's `<eos>`, so completion-only training under one needs `eos_token: <end_of_turn>`. A requested Liger `fused_linear_cross_entropy` is forced off ([Liger Kernels](../optimization/liger-kernels.md#fused-loss-under-a-multimodal-wrapper)). DPO, KTO and SMPO refuse PaliGemma on image data ([Modality support](../reference/trainer-architecture.md#modality-support)).

## Parallelism

Axis sizes are CLI flags: `--expert_parallel_size`, `--context_parallel_size`, `--tensor_parallel_size`, `--expert_tensor_parallel_size`, `--pipeline_parallel_size`. EP+CP, EP+TP, EP+ETP, PP+EP and PP+ETP compose; TP+CP, ETP+CP and EP+TP+ETP are rejected at config time, as is every other unlisted combination ([Pipeline Parallelism](../parallelism/pipeline-parallelism.md) composes with the expert axes only). Pure ETP is `--expert_parallel_size=1 --expert_tensor_parallel_size=N` (attention TP and expert TP are mutually exclusive), and LoRA is rejected under TP and EP+TP. Plain FSDP2 and DDP load the whole model onto every GPU before sharding, so a dense model must fit on one GPU; TP is the route past that ([Data Parallelism](../parallelism/data-parallelism.md#fsdp2-strategy-by-mode)).

Under CP a batch whose length is not a multiple of `cp_size` is right-padded in `compute_loss`, and a tokenizer with no `pad_token_id` raises there rather than padding with vocabulary token 0. The loss normalizes over the CP group's tokens; metrics come off the local chunk. With `fused_linear_cross_entropy` on, the CP wrapper scores each chunk without a logits plane ([CP fused head loss](../parallelism/context-parallelism.md#fused-head-loss)). In every mode the metrics are reduced once per log, not per micro-batch ([details](../reference/trainer-architecture.md#sft-loss-and-metrics)).

See [Trainer Compatibility](../reference/trainer-architecture.md#trainer-compatibility) and [Supported Models](../models/README.md#compatibility-matrix).

## MTP heads

`mtp_loss_weight` trains the multi-token-prediction head a family's release ships, beside the next-token loss or over a frozen trunk ([MTP heads](../optimization/mtp-heads.md)). It runs under FSDP2, EP and ETP; every other axis refuses it before the load.

## Pre-processed datasets

`scripts/before_training/prepare_dataset.py` tokenizes, packs and shards offline:

```bash
python scripts/before_training/prepare_dataset.py \
    --input "s3://bucket/raw/dataset" --output "s3://bucket/preprocessed/dataset" \
    --model_name_or_path "Qwen/Qwen3-8B" --max_length 4096 --test_size 0.01 \
    --num_shards 64 --pack_sequences \
    --assistant_message_template $'<|im_start|>assistant\n'
```

The training script detects the artifact from its `metadata.json`, skips tokenization, and holds the run to what was baked: `max_length` and `train_on_completions_only` must agree or startup raises, as must any render knob the YAML states and the chat template the checkpoint would export. `ShardedDatasetLoader` then assigns shards by DP rank ([Pre-Processing](../data/dataset-preparation.md)).

## Testing a setup

Smoke the config first: cut `max_length`, set `max_steps: 5`, launch on 2 GPUs. `examples/sft/deepseek_v4/v4-tiny-random-smoke-ep.yaml` is a tiny-model EP smoke needing no production checkpoint; materialize its tiny checkpoint first with `build_tiny_family_checkpoint` (`tests/common/tiny_models.py`, steps in the recipe header).

```bash
pytest tests/cpu/config tests/cpu/data -m cpu    # config gates, collators, render knobs
torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_gptoss_modes.py --mode ep
```

`tests/gpu/trainers/sft/` holds suites per model with one `--mode` per parallel shape (FSDP2, EP, CP, TP, EP+CP, EP+TP, pure ETP), plus resume, VLM and sinks; the MoE and dense mode suites share one body, `tests/common/sft_modes.py`.

## Related pages

- [Preference Optimization (SMPO, DPO)](preference/README.md) · [Pre-training](pretraining.md) · [Pipeline Parallelism](../parallelism/pipeline-parallelism.md)
- [Collators](../data/collators.md) · [Padding-Free Collator](../optimization/padding-free-collator.md) · [Checkpoints & Resume](../reference/checkpoints.md)
