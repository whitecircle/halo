# Reward Modeling

A reward model predicts a scalar human-preference score for a (prompt, completion) pair, trained on pairwise data with the Bradley-Terry ranking loss. Use it for rejection sampling, RLHF reward signals, or dataset scoring.

| Aspect | Value |
|--------|-------|
| Data format | Pairwise preferences (prompt, chosen, rejected) — same as DPO |
| Model | `AutoModelForSequenceClassification`, `num_labels=1` |
| Trainer | `DistributedRewardTrainer` |
| Script | `scripts/training/preference/rewards.py` |
| Modality | Text, or vision-language on the families that ship a score head ([below](#vision-language)) |
| Parallelism | EP, TP, ETP, EP+TP; no CP. Declares `_supports_pp` — [PP](../../parallelism/pipeline-parallelism.md) is not yet available in this release |

## Dataset format

Same pairwise format as DPO/SMPO — `prompt` / `chosen` / `rejected`, all `list[dict]` — and implicit-prompt datasets (no `prompt` column, shared turns inside chosen/rejected, e.g. `Skywork/Skywork-Reward-Preference-80K-v0.2`) work as-is. See [Dataset Formats](../../data/dataset-formats.md#preference-dposmpo). The script passes the raw columns through: TRL's `RewardTrainer` chat-templates and tokenizes chosen/rejected itself and filters rows over `max_length`.

## Quick start

```bash
# Standard (FSDP via accelerate)
accelerate launch --config_file launcher-configs/accelerate/fsdp2_gradop_config.yaml \
    scripts/training/preference/rewards.py \
    examples/reward/qwen3_5/rm-qwen3.5-9b-skywork-pref80k.yaml

# MoE with expert parallelism
torchrun --nproc_per_node=8 \
    scripts/training/preference/rewards.py \
    examples/reward/gptoss/rm-gptoss-20b-skywork-pref80k-ep.yaml \
    --expert_parallel_size=8
```

Minimal config:

```yaml
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
dataset: path/to/preference_dataset
test_size: 0.05
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
num_train_epochs: 1
max_length: 2048
gradient_checkpointing: true
output_dir: checkpoints/reward-model
report_to: wandb
```

## Parallelism

EP, TP, and ETP are supported. CP is not — reward scoring pools the full sequence into one representation, which CP splits across ranks. The trainer declares `_supports_pp`, but pipeline parallelism is [not yet available in this release](../../parallelism/pipeline-parallelism.md); its shipped seams here move the pooling into a last-stage loss adapter and gate `config.pad_token_id` (set, and equal to the tokenizer's) at construction. Full matrix: [Trainer Compatibility](../../reference/trainer-architecture.md#trainer-compatibility).

Where the tokenizer pads with eos (DeepSeek-V4), the id is still recorded and an INFO line names the cost: `config.pad_token_id` binds `nn.Embedding(padding_idx=<eos id>)` on the next load, masking that row's input-embedding gradient. Withholding it would not avert this — `Trainer.train` runs transformers' `align_special_tokens`, which records the id with no pad-vs-eos test. Give such a base a pad token distinct from eos (`pad_token:` / `added_special_tokens:`) if the EOS embedding must keep training on the input side.

## Vision-language

Image-carrying preference pairs train on the families that have BOTH a vision tower and a sequence-classification head. transformers 5.16 ships one for `gemma3`, `qwen3_5` (dense), `t5gemma2` and `modernvbert`; the toolkit registers two more in `src/models/seq_cls_heads.py` — Gemma 4 and MoE Qwen3.5/3.6, each in both spellings a checkpoint can carry, composite and text tower (`gemma4`/`gemma4_text`, `qwen3_5_moe`/`qwen3_5_moe_text`). The module comes out once transformers ships native heads.

The roster is read off the live registry, so a family added upstream needs no code change here; a multimodal checkpoint outside it (Qwen3-VL, LLaVA, …) is refused before the model load, with the supported list in the message. Nothing in the objective is modality-specific: `GenericForSequenceClassification` forwards `pixel_values` into the composite backbone, and the pooling rule keys on the pad id, so expanded image placeholders are ordinary content tokens to it.

**The data path follows the RUN, not the checkpoint.** A text recipe on a multimodal checkpoint is a text run and takes TRL's own tokenize map unchanged. A run declares image data by carrying an `images`/`image` column or by naming one with `images_field`; that verdict (`is_vlm_run`) selects the vision path:

```yaml
model_name_or_path: google/gemma-4-26B-A4B-it
dataset: path/to/vision_preference_dataset
images_field: images        # optional — an images/image column declares the run by itself
max_length: 4096
```

The images are merged into the **prompt** conversation, which both sides of a pair share: they fill unset `{"type": "image"}` placeholders in order when the messages carry them, and otherwise lead the first user turn. Images inside a `chosen`/`rejected` completion are refused.

On this path an `AutoProcessor` is the trainer's `processing_class`, so every checkpoint exports `processor_config.json` beside the weights. The dataset map (`src/data/pipeline/preferences.py`) renders both sides whole and keeps the images as PIL objects, so pixels never enter the Arrow cache; `src/data/collators/vlm_preference.py` then expands the placeholders per batch, emitting TRL's `[chosen ⧺ rejected]` layout with the vision tensors concatenated row-major over it (each half carries the batch's images, since both sides render the same prompt).

Constraints, all enforced rather than documented:

- `max_length` stays a **filter**, applied to the rendered text. Vision tokens are counted only at collation, where a batch over budget **raises**: cutting expanded image placeholders desyncs them from `pixel_values`/`image_grid_thw`, and a runtime batch cannot drop rows without desyncing DP ranks. Lower the image resolution or raise `max_length`.
- CP is unsupported for reward modeling in any modality (the score head pools the whole sequence).
- The shipped PP gates refuse an image-carrying **run**, not the checkpoint: the stage split keeps only the text backbone and the head, so the vision tower would be in no stage and pixel inputs would be pruned unseen. A text-only run of the same multimodal checkpoint passes that gate — the PP save seam holds the untouched vision tensors on the save rank and re-emits them under the wrapper layout, so an export would still load as the composite class.
- A mistyped `images_field` raises at the dataset load.

## Configuration

`RMScriptArguments` adds `images_field` (see [Vision-language](#vision-language)) and the default project name; the data args (`dataset`, `dataset_ratio`, `test_size`) come from `CommonScriptArguments`, and the parallelism flags from `DistributedArguments`. See [Configuration Reference](../../reference/configuration-reference.md).

TRL `RewardConfig` defaults: `max_length` (`1024`), `gradient_checkpointing` (`True`), `center_rewards_coefficient` (`None`), `dataset_num_proc` (`None`). `max_length` is a **filter**, not a truncation — TRL drops pairs whose chosen or rejected exceeds it. Set it to `null` to resolve it to the model's context window.

The script defaults `attn_implementation` to `sdpa` when the YAML sets none: batches are right-padded, and the auto-selected FA4 would run them through its slow varlen path. The default applies under `reset_sinks: true` only — with live sinks the script requests nothing, because the resolver accepts only a sink-carrying backend there.

`tools_field` names the column holding each row's tool schemas (a list of dicts or its JSON string). The script aliases it onto `tools`, the one column TRL's `RewardTrainer` passes to the chat template, and refuses a name the dataset does not carry — or any value at all on a [vision run](#vision-language), whose pair render templates without `tools=`.

## PEFT / LoRA

```yaml
use_peft: true
lora_r: 16
lora_alpha: 32
lora_target_modules: [q_proj, k_proj, v_proj, o_proj]
lora_task_type: SEQ_CLS         # required for classification/reward models
lora_modules_to_save: [score]   # train + save the freshly-initialized score head
```

LoRA is rejected under TP, EP+TP and PP.

## Use cases

**Rejection sampling.** `scripts/inference/reward_model/rm_rejection_sampling.py` generates `--n_hypos` (default 5) responses per prompt through an OpenAI-compatible backend (vLLM or OpenRouter), scores each with the local reward model, and writes the best/worst pair. `--output_format preference` (default) emits DPO/SMPO `chosen`/`rejected`; `--output_format offline_grpo` emits `completions` + `rewards`. `--prompts_source` is a JSONL file with a `prompt` field of message dicts.

```bash
python scripts/inference/reward_model/rm_rejection_sampling.py \
    --model_name my-served-model \
    --openai_base_url http://localhost:8000/v1 \
    --prompts_source path/to/prompts.jsonl \
    --rm_model_path path/to/reward-model \
    --output_folder path/to/output
```

Both scripts write `<prompts stem>_<model>_<format>.jsonl` into `--output_folder` — `rs` /
`offline_grpo` for rejection sampling, `rm_scoring` for scoring — and resume from the ids already in
that file. The generating model is in the name because resume keys on those ids: a shared file would
have a second `--model_name` read the first model's rows as its own and generate nothing.

**Dataset scoring.** `scripts/inference/reward_model/rm_scoring.py` generates one response per prompt, scores it, and prints summary statistics (same flags).

Both load the reward model in `--rm_dtype` (default `bfloat16`): reward logits are unbounded, and fp16's narrow range saturates on exactly the out-of-distribution completions the score has to separate.

**RLHF reward signal.** Serve the reward model as the reward function for online GRPO or other RL methods.
