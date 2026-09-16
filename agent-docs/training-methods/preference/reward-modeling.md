# Reward Modeling

A reward model scores a (prompt, completion) pair with one scalar, fit to pairwise preferences under the Bradley-Terry loss. Use it for rejection sampling or as an RL reward signal; to train a policy on the same pairs use [DPO](dpo.md) or [SMPO](smpo.md).

Trainer `DistributedRewardTrainer`, script `scripts/training/preference/rewards.py`, model class `AutoModelForSequenceClassification` with `num_labels=1`. EP, TP, ETP and EP+TP apply; CP does not — the score head pools the whole sequence ([matrix](../../reference/trainer-architecture.md#trainer-compatibility)). It declares `_supports_pp`, but pipeline parallelism is [not yet available in this release](../../parallelism/pipeline-parallelism.md).

## Dataset

The pairwise `prompt` / `chosen` / `rejected` format DPO and SMPO use, plus implicit-prompt datasets carrying no `prompt` column, with the shared turns repeated inside both sides (`Skywork/Skywork-Reward-Preference-80K-v0.2`). See [Dataset Formats](../../data/dataset-formats.md#reward-modeling).

TRL's `RewardTrainer` chat-templates and tokenizes the raw columns itself, concatenating `prompt + chosen` / `prompt + rejected` with no hub-shape normalization: a `prompt` column whose completions repeat those turns renders the prompt twice, silently. DPO and SMPO strip that prefix, and so does the reward VLM path.

An optional `margin` column widens the target gap per row. It must be in both splits or in neither — under PP a margin on one side alone raises.

## Configuration

```yaml
model_name_or_path: Qwen/Qwen3.5-9B
dataset: Skywork/Skywork-Reward-Preference-80K-v0.2
test_size: 0.05
attn_implementation: sdpa          # the script's own default under reset_sinks: true

max_length: 4096
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-05             # the shipped recipes' value
num_train_epochs: 1
gradient_checkpointing: true
output_dir: checkpoints/rm-qwen3.5-9b-skywork-pref80k
```

| Knob | Default | Effect |
|---|---|---|
| `max_length` | `1024` | A **filter**: TRL drops pairs whose chosen or rejected side exceeds it |
| `center_rewards_coefficient` | `None` | Penalizes `(chosen + rejected)²`, pulling scores toward zero mean |
| `tools_field` | `None` | Column of per-row tool schemas; aliased onto `tools` and rendered on conversational rows only |
| `images_field` | `None` | Image column, see [Vision-language](#vision-language) |

`max_length: null` resolves to the model's context window. `learning_rate` defaults to TRL's `1e-4`, high for a full fine-tune. A `chat_template_kwargs` column is splatted into the template too.

The script defaults `attn_implementation` to `sdpa` for its right-padded batches when the YAML pins none, and requests nothing under `reset_sinks: false`, where only a sink-carrying backend is accepted ([padded workloads](../../optimization/flash-attention.md#model-specific-handling)). `text_only_model` and `log_decoded_samples` are refused, not silently ignored.

Where the tokenizer pads with eos (DeepSeek-V4), an INFO line names the cost: the recorded `config.pad_token_id` binds `nn.Embedding(padding_idx=<eos id>)` on the next load, masking that row's input-embedding gradient. Give such a base its own `pad_token:` to keep the EOS embedding training on the input side.

### Pipeline parallelism

Pipeline parallelism is [not yet available in this release](../../parallelism/pipeline-parallelism.md); its shipped seams here move the pooling into the last-stage loss, which picks each row's rightmost non-pad token. That needs `config.pad_token_id` set and equal to the tokenizer's, and construction raises on either mismatch: a disagreement pools every row at the last filler token — a finite loss that learns nothing.

## Launch

```bash
# Dense, plain FSDP2 data parallel
torchrun --nproc_per_node=8 scripts/training/preference/rewards.py \
    examples/reward/qwen3_5/rm-qwen3.5-9b-skywork-pref80k.yaml

# MoE, expert parallel
torchrun --nproc_per_node=8 scripts/training/preference/rewards.py \
    examples/reward/gptoss/rm-gptoss-20b-skywork-pref80k-ep.yaml --expert_parallel_size=8
```

`halo launch preference/rewards <config> --nproc 8` builds the same line.

## PEFT / LoRA

```yaml
use_peft: true
lora_r: 16
lora_alpha: 32
lora_target_modules: [q_proj, k_proj, v_proj, o_proj]
lora_task_type: SEQ_CLS         # required for a pooled score head
lora_modules_to_save: [score]
```

LoRA is rejected under TP, EP+TP and PP ([PEFT](../../optimization/peft.md#parallelism-compatibility)).

## Vision-language

Image-carrying pairs train on the families with both a vision tower and a sequence-classification head. transformers 5.16 ships one for `gemma3`, `qwen3_5` (dense), `t5gemma2` and `modernvbert`; the toolkit registers Gemma 4 and MoE Qwen3.5/3.6 in `src/models/seq_cls_heads.py`. A checkpoint outside that live registry is refused before the model load, naming the supported list.

The data path follows the **run**, not the checkpoint: a vision run is one whose dataset carries an `images`/`image` column, or has `images_field` name one. A text recipe on a multimodal checkpoint stays on TRL's tokenize map.

```yaml
model_name_or_path: google/gemma-4-26B-A4B-it
dataset: path/to/vision_preference_dataset
images_field: images
max_length: 4096
```

Images merge into the **prompt** conversation, which both sides of a pair share: they fill unset `{"type": "image"}` placeholders in order, and otherwise lead the first user turn. Images inside a `chosen`/`rejected` completion are refused, as is `tools_field`. An `AutoProcessor` is the `processing_class` here, so every checkpoint exports `processor_config.json`.

Two limits:

- `max_length` filters on rendered **text** only. Vision tokens are counted at collation, where a batch over budget raises. Lower the image resolution or raise `max_length`.
- PP refuses an image-feeding run: no stage holds the vision tower. A text-only run of the same checkpoint is admitted, and its untouched vision tensors ride every checkpoint.

## Testing a setup

```bash
torchrun --nproc_per_node=2 scripts/training/preference/rewards.py <config> \
    --max_steps=5 --save_strategy=no --report_to=none
```

Covering tests: `pytest tests/cpu/trainers -m cpu`, `tests/gpu/trainers/other/test_reward.py` and `test_reward_vlm_e2e.py`.

Once trained, score fresh generations or build preference pairs with `scripts/inference/reward_model/rm_scoring.py` and `rm_rejection_sampling.py` ([Scripts](../../reference/scripts-reference.md)). Both generate against a served endpoint from a `--prompts_source` prompt file before scoring; neither scores an existing (prompt, completion) corpus. Keep their `--rm_dtype` at `bfloat16`: reward logits are unbounded, and fp16 saturates on the completions the score must separate.

## What to watch

| Signal | Reading |
|---|---|
| `accuracy` | Share of pairs scoring chosen above rejected (chance is 0.5) |
| `margin` | Mean chosen − rejected score; the separation the model is buying |
| `min_reward` / `mean_reward` / `max_reward` | Score range; `center_rewards_coefficient` is what pulls `mean_reward` toward zero |

Failure signatures:

- Rows vanishing at prep — `max_length` filters rather than truncates, and the text path drops them silently (only the VLM path reports a count). Compare the split sizes, then raise it or shorten the pairs.
- A `config.pad_token_id` raise under PP — give the tokenizer a pad token, or sync the two ids.
- Accuracy pinned near 0.5 under LoRA — the score head is freshly initialized, so it must be in `lora_modules_to_save`.
