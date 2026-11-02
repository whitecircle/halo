# SMPO: Smooth Margin Preference Optimization

SMPO trains on pairwise preferences with no reference model: a scheduled-margin loss plus a built-in SFT anchor, so one model sits in memory instead of two. Its default `smooth_lower_bound` margin term is exactly zero once the mean-token log-prob gap clears the current margin — where [DPO](dpo.md)'s sigmoid keeps pushing, the source of its log-prob collapse — while the SFT anchors keep training both completions.

Trainer `SMPOTrainer`, config `SMPOConfig` (`halo/configs/preference.py`), script `scripts/training/preference/smpo.py`. The objective — margin losses keyed by `loss_type`, the log-prob clip, the CP-global reductions and the metric folds — is `SMPOObjective` (`halo/trainers/preference/smpo_objective.py`); under PP, `SMPOPipelineLoss` (`smpo_pipeline_loss.py`) carries it over interleaved pairs. `ParallelismConfig` accepts EP, CP, TP, pure ETP, PP, EP+CP, EP+TP, EP+ETP, PP+EP and PP+ETP for it, and rejects everything else outside its allowlist.

```text
logits  = mean_token log p(chosen) − mean_token log p(rejected) − margin(step)
L_total = loss_fn(beta · logits) + chosen_sft_ratio · CE(chosen) + (1 − chosen_sft_ratio) · CE(rejected)
```

![SMPO on one preference pair: one forward over the 2N concatenated rows gives per-token log-probs; the percentile clip (rejected tail at the 2% token quantile, optional chosen-side cap, min_log_prob floor) trims the margin path only, whose per-sequence mean feeds the smooth_lower_bound term relu(−β·z)² against the scheduled margin, while the SFT anchors take the pre-clip NLL of both sides; the total is the mean margin loss plus chosen_sft_ratio-weighted cross-entropy, with no reference model](../../assets/diagrams/smpo_pipeline.png)

Every path — padded and `padding_free` — takes its per-token log-probs from one chunked fp32 log-softmax (`next_token_logprobs`, `halo/kernels/logprobs.py`), so a bf16 model's tokens are scored at fp32 precision.

The concat (`concatenate_pairs` in `halo/data/collators/smpo.py`) rolls every row left by its leading-pad count, with no host sync, on every path: the collator's leading prompt pads trail the completion, so each row's real tokens hold the positions and indices they hold unpadded. A padded row then scores as it does alone even where attention reads absolute positions (Mistral4's llama-4 query scale) or pools KV over windows cut at fixed indices (DeepSeek-V4's compressors), and the CP attention, which ignores the mask, never reaches a pad from a real token, so a batch of any size trains the same objective as without CP. Under CP the per-sequence sums and counts all-reduce across the group before the mean. The sum reduce is autograd-aware: its backward sums the gradient over the group, which cancels FSDP2's `1/cp_size` average, so the loss takes no `cp_size` factor and logs the unsplit-sequence value.

## Configuration

```yaml
model_name_or_path: unsloth/gpt-oss-20b-BF16
attn_implementation: flash_attention_4   # the padded scripts' unset pick on Blackwell; sinks stay reset
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
gradient_checkpointing: false      # a 2 × 4,096 pair holds ~33 GB of activations
output_dir: checkpoints/smpo-gptoss-20b-tulu3-prefmix-ep
```

`beta` scales the margin logit before the loss, and `target_margin` is the log-ratio gap a pair must reach, ramped linearly from `initial_margin` over the run under `use_margin_schedule`. `chosen_sft_ratio` splits the SFT anchor between chosen and rejected cross-entropy. The margin term alone clips the bottom `lower_clip_percentile` of rejected log-probs and the top `upper_clip_percentile` of chosen, then floors rejected tokens at `min_log_prob`. Types, defaults and bounds: [SMPOConfig](../../reference/configuration-reference.md#smpoconfig).

| `loss_type` | `loss_fn(z)` | Behavior |
|---|---|---|
| `smooth_lower_bound` | `max(0, -z)²` | Squared hinge: smooth gradient, zero past the margin |
| `hinge` | `max(0, -z)` | Zero past the margin, discontinuous gradient |
| `sigmoid` | `-log_sigmoid(z)` | Never reaches zero (DPO-like) |
| `ipo` | `z²` | Targets the margin exactly, penalizing overshoot |

`max_length` is the total budget and defaults to `1024`; `null` or a non-positive value resolves it to the context window. An unset `max_prompt_length` takes half of it and `max_completion_length` the remainder. Shares summing past `max_length` are rejected at construction — the two truncate independently. Prompts cut per `truncation_mode` (`keep_end`), completions from the end; a completion cut by `max_completion_length` carries no EOS.

`padding_free` needs a varlen Flash Attention kernel and raises on anything else, including the `sdpa` the script defaults to off Blackwell (and for a remote-code checkpoint) under `reset_sinks: true` outside CP. It is also incompatible with CP, PP and VLM runs, and refused on an M-RoPE decoder (Qwen3.5/3.6, text towers included) on every backend, as for SFT (`reject_mrope_padding_free`, [Collators](../../data/collators.md)).

Under PP the percentile clips are refused, since a quantile over the whole batch half changes with the microbatch split: set `lower_clip_percentile: null` (its default is `0.02`) and leave `upper_clip_percentile` unset; `min_log_prob` stays. VLM runs and PEFT are refused too ([Rejected combinations](../../parallelism/pipeline-parallelism.md#rejected-combinations)).

Attention isolates each row of the flattened batch through `position_ids`, and so does Zaya's CCA; the LFM-2 and GatedDeltaNet (Qwen3-Next) conv / linear-attention mixers get the segment markers the SFT collators emit, and a GatedDeltaNet model is refused without the `causal_conv1d` / `fla` kernels that read them. Ring's Lightning Attention-2 layers read no marker and would carry state from each row into the next, across the chosen / rejected split too, so the trainer refuses `padding_free` on such layers unless `allow_cross_document_linear_attention: true`, as the SFT collators do. See [Document isolation under packing](../../data/collators.md#document-isolation-under-packing).

## Launch

```bash
torchrun --nproc_per_node=8 scripts/training/preference/smpo.py \
    examples/preference/gptoss/smpo-gptoss-20b-tulu3-prefmix-ep.yaml \
    --expert_parallel_size=8
```

`halo launch smpo <config> --nproc 8 --expert_parallel_size=8` builds the same line; anything after the config is forwarded to the script. From Python, pass a `SMPOConfig` as `args`, the tokenizer or processor as `processing_class`, the `ParallelismConfig` the model was loaded with, and LoRA as `peft_config`. `is_vlm` picks the data path, defaulting to the `processing_class` modality.

## Vision-language

The model class follows the **checkpoint**: a multimodal config loads through `AutoModelForImageTextToText` plus its processor wherever the checkpoint ships one ([SFT — VLMs](../sft.md#vision-language-models)), so the run's checkpoints carry that `processor_config.json` (unless `text_only_model` forced the text class). The data path follows the **run** — VLM only when the dataset declares images in an `images`/`image` column. Text pairs on a multimodal checkpoint are a text run, and keep CP and `padding_free`.

A dataset carrying images inside its messages alone reads as text, and the renderer refuses it rather than training on pixel-less placeholders. A prefix-LM checkpoint (PaliGemma) is refused on image rows before the model load ([Modality support](../../reference/trainer-architecture.md#modality-support)); train it on images with SFT or distillation.

A VLM run hands its rows to the trainer untouched: it chat-templates them and selects `DataCollatorForVLMSMPO`, which processes images at collation, keeping pixels out of the Arrow cache. The row map pins its Arrow schema, so text-only and image rows mix in one dataset, and a dataset counts as already prepared only with every column the map writes, `prompt_overlap` included; one missing a column that no longer carries the raw `chosen`/`rejected` to re-tokenize is refused with the missing columns named. Completions split off the prompt jointly, as on the text path; when a token merges across the boundary, the row's `prompt_overlap` counts the prompt tokens the completions start with, and the collator ends the prompt before them. Completions must be text-only, and prompts are never truncated — one expanding past `max_prompt_length` raises, since cutting expanded placeholders desyncs them from `pixel_values`.

## Testing a setup

```bash
torchrun --nproc_per_node=2 scripts/training/preference/smpo.py <config> \
    --max_steps=5 --save_strategy=no --report_to=none
```

Covering tests: `pytest tests/cpu/trainers -m cpu`, `tests/gpu/trainers/preference/test_smpo_*.py` (FSDP2, EP, CP, TP, VLM, padding-free and its per-document isolation, text-on-VLM, TP resume), plus `tests/gpu/parallelism/cp/test_cp_smpo_logprobs.py` and, under pipeline parallelism, `tests/gpu/parallelism/pp/test_pp_smpo.py` and `test_pp_trainer_checkpoints.py --trainer smpo` (save, exact resume, export).

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
