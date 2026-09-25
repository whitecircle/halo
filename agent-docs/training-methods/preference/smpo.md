# SMPO: Smooth Margin Preference Optimization

SMPO trains on pairwise preferences with no reference model: a scheduled-margin loss plus a built-in SFT anchor, so one model sits in memory instead of two. Its default `smooth_lower_bound` margin term is exactly zero once the mean-token log-prob gap clears the current margin — where [DPO](dpo.md)'s sigmoid keeps pushing, the source of its log-prob collapse — while the SFT anchors keep training both completions.

Trainer `SmoothMarginPOTrainer`, config `SmoothMarginPOConfig` (`src/configs/smpo_config.py`), script `scripts/training/preference/smpo.py`. `ParallelismConfig` accepts EP, CP, TP, pure ETP, EP+CP, EP+TP and EP+ETP for it, and rejects everything else outside its allowlist. It declares `_supports_pp`, but pipeline parallelism is [not yet available in this release](../../parallelism/pipeline-parallelism.md).

```text
logits  = mean_token log p(chosen) − mean_token log p(rejected) − margin(step)
L_total = loss_fn(beta · logits) + chosen_sft_ratio · CE(chosen) + (1 − chosen_sft_ratio) · CE(rejected)
```

![SMPO on one preference pair: one forward over the 2N concatenated rows gives per-token log-probs; the percentile clip (rejected tail at the 2% token quantile, optional chosen-side cap, min_log_prob floor) trims the margin path only, whose per-sequence mean feeds the smooth_lower_bound term relu(−β·z)² against the scheduled margin, while the SFT anchors take the pre-clip NLL of both sides; the total is the mean margin loss plus chosen_sft_ratio-weighted cross-entropy, with no reference model](../../assets/diagrams/smpo_pipeline.png)

Under CP the per-sequence sums and counts all-reduce across the group before the mean.

## Configuration

```yaml
model_name_or_path: unsloth/gpt-oss-20b-BF16
attn_implementation: flash_attention_2   # FA2 takes no sink argument; keep reset_sinks on
dataset: allenai/llama-3.1-tulu-3-8b-preference-mixture
expert_parallel_size: 8

beta: 1.0
target_margin: 0.4
initial_margin: 0.2
chosen_sft_ratio: 0.75
loss_type: smooth_lower_bound

per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-06             # the shipped recipes' value; the config default is 1e-6
max_length: 4096
max_prompt_length: 2048
gradient_checkpointing: true
output_dir: checkpoints/smpo-gptoss-20b-tulu3-prefmix-ep
```

| Knob | Default | Effect |
|---|---|---|
| `beta` | `1.2` | Scales the margin logit before the loss |
| `target_margin` | `0.35` | Log-ratio gap a pair must reach; must be `>= 0` |
| `use_margin_schedule` / `initial_margin` | `True` / `0.01` | Margin ramps linearly to `target_margin` over the run; `initial_margin` must be smaller |
| `chosen_sft_ratio` | `0.8` | Split of the SFT anchor between chosen and rejected cross-entropy |
| `lower_clip_percentile` / `upper_clip_percentile` | `0.02` / `None` | Clip the bottom percentile of rejected log-probs, the top percentile of chosen — the margin term only |
| `min_log_prob` | `-2.3` | Absolute floor for rejected tokens, applied after the percentile clip |
| `padding_free` | `False` | Flattens the batch into one varlen sequence |

| `loss_type` | `loss_fn(z)` | Behavior |
|---|---|---|
| `smooth_lower_bound` | `max(0, -z)²` | Squared hinge: smooth gradient, zero past the margin |
| `hinge` | `max(0, -z)` | Zero past the margin, discontinuous gradient |
| `sigmoid` | `-log_sigmoid(z)` | Never reaches zero (DPO-like) |
| `ipo` | `z²` | Targets the margin exactly, penalizing overshoot |

`max_length` is the total budget and defaults to `1024`; `null` or a non-positive value resolves it to the context window. An unset `max_prompt_length` takes half of it and `max_completion_length` the remainder. Shares summing past `max_length` are rejected at construction — the two truncate independently. Prompts cut per `truncation_mode` (`keep_end`), completions from the end, keeping the terminal EOS.

`padding_free` needs a varlen Flash Attention kernel and raises on anything else, the `sdpa` the script defaults to under `reset_sinks: true` included. It is also incompatible with CP, VLM runs and PP — the shipped PP gates additionally reject PEFT, a non-null clip percentile and a `label_pad_token_id` other than `-100`.

Attention isolates each row of the flattened batch through `position_ids`; the LFM-2 and GatedDeltaNet (Qwen3.5/3.6, Qwen3-Next) conv / linear-attention mixers get the segment markers the SFT collators emit, and a GatedDeltaNet model is refused without the `causal_conv1d` / `fla` kernels that read them. Zaya's CCA has no per-document boundary parameter and carries state from each row into the next, across the chosen / rejected split too, so keep `padding_free` off for Zaya. See [Document isolation under packing](../../data/collators.md#document-isolation-under-packing).

## Launch

```bash
torchrun --nproc_per_node=8 scripts/training/preference/smpo.py \
    examples/preference/gptoss/smpo-gptoss-20b-tulu3-prefmix-ep.yaml \
    --expert_parallel_size=8
```

`halo launch smpo <config> --nproc 8 --expert_parallel_size=8` builds the same line; anything after the config is forwarded to the script. From Python, pass a `SmoothMarginPOConfig` as `args`, the tokenizer or processor as `processing_class`, the `ParallelismConfig` the model was loaded with, and LoRA as `peft_config`. `is_vlm` picks the data path, defaulting to the `processing_class` modality.

## Vision-language

The model class follows the **checkpoint**: a multimodal config loads through `AutoModelForImageTextToText` plus its processor, so the run's checkpoints carry a `processor_config.json` (unless `text_only_model` forced the text class). The data path follows the **run** — VLM only when the dataset declares images in an `images`/`image` column. Text pairs on a multimodal checkpoint are a text run, and keep CP and `padding_free`.

A dataset carrying images inside its messages alone reads as text, and the renderer refuses it rather than training on pixel-less placeholders.

A VLM run hands its rows to the trainer untouched: it chat-templates them and selects `DataCollatorForVLMSMPO`, which processes images at collation, keeping pixels out of the Arrow cache. Completions must be text-only, and prompts are never truncated — one expanding past `max_prompt_length` raises, since cutting expanded placeholders desyncs them from `pixel_values`.

## Testing a setup

```bash
torchrun --nproc_per_node=2 scripts/training/preference/smpo.py <config> \
    --max_steps=5 --save_strategy=no --report_to=none
```

Covering tests: `pytest tests/cpu/trainers -m cpu`, `tests/gpu/trainers/preference/test_smpo_*.py` (FSDP2, EP, CP, TP, VLM, padding-free and its per-document isolation, text-on-VLM, TP resume), plus `tests/gpu/parallelism/cp/test_cp_smpo_logprobs.py`.

## What to watch

| Signal | Reading |
|---|---|
| `rewards/margins` | Mean `beta ×` (chosen − rejected) log-prob gap; rises toward `beta ×` the current scheduled margin |
| `rewards/accuracies` | Share of pairs scoring chosen above rejected (chance is 0.5) |
| `logps/chosen` / `logps/rejected` | Post-clip per-sequence means: chosen rises and the gap approaches the scheduled margin; the rejected CE anchor holds its side up |
| `sft_loss/chosen` | The anchor; a steady climb means the preference term overpowers it |

Failure signatures:

- Loss NaN or exploding — keep `min_log_prob: -2.3` and `lower_clip_percentile: 0.02` to floor outlier tokens.
- Outputs degrade while margins rise — raise `chosen_sft_ratio`, or lower the LR or `beta`.
- Margin stalls — `target_margin` too high or LR too low; start from an easier `initial_margin`.
