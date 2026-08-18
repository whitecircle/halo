# Classification

Sequence classification — single-label (multi-class) and multi-label — on `AutoModelForSequenceClassification`. Trainer `ClassificationTrainer` (`src/trainers/reward/classification.py`), script `scripts/training/classification.py`. Text-only.

## Dataset format

A `prompt` field in conversation format plus a `label` field: a string for single-label, a list for multi-label.

```jsonl
{"prompt": [{"role": "user", "content": "I love this product!"}], "label": "positive"}
{"prompt": [{"role": "user", "content": "A hilarious romantic comedy."}], "label": ["comedy", "romance"]}
```

For a raw text/label dataset with no pre-built conversation — e.g. `yaful/MAGE` in the shipped config — set `text_field` to the text column and the script wraps each row as a single user turn. A dataset carrying neither a `prompt` column nor the named `text_field` is rejected right after the dataset load, before the model is loaded, instead of failing as a per-row `KeyError` inside the tokenization map.

The script chat-templates each row, tokenizes it through `tokenize_rendered` (the shared [special-token ownership](../data/dataset-formats.md#special-token-ownership) seam), and **truncates** to `max_length` — unlike the SFT path, which drops over-length rows. A classification label describes the whole document, so a shortened document keeps a valid label while dropping it would skew the class balance. A dataset handed to `ClassificationTrainer` directly (no `input_ids` column) is instead tokenized untruncated and its over-length rows dropped, with the dropped fraction logged.

At startup the script discovers the label set from the training split, stringifies and sorts it for deterministic IDs, and builds `label_to_id`. Single vs multi-label is detected from whether `label` is a list, and sets `model.config.problem_type`. Labels present only in validation/test are added with a warning. `label2id` / `id2label` are written to the model config and `num_labels` is derived from the data; with a pre-sharded dataset the label set is unioned across ranks so every rank builds an identically shaped head.

`-1` is dropped from the label set (with a warning) but rows carrying it keep the raw value, and **no loss path ignores it** — the single-label objectives use torch's `ignore_index=-100`, so a `-1` label is an out-of-range class index. Filter those rows out of the dataset.

## Quick start

```bash
torchrun --nproc_per_node=8 scripts/training/classification.py \
    examples/classification/gptoss/clf-gptoss-20b-mage-ep.yaml --expert_parallel_size=8
```

Minimal config:

```yaml
model_name_or_path: Qwen/Qwen3-4B
dataset: path/to/classification_dataset
test_size: 0.1
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2.0e-05
num_train_epochs: 3
max_length: 1024
gradient_checkpointing: true
output_dir: checkpoints/classifier
report_to: wandb
```

`examples/classification/qwen3_5/clf-qwen3.5-9b-mage.yaml` is the dense counterpart. Batches are right-padded, so the script defaults `attn_implementation` to SDPA rather than the auto-selected FA4 varlen path — but only under the default `reset_sinks: true`. With `reset_sinks: false` it requests nothing and lets the resolver auto-select, which there accepts only a sink-carrying backend.

## Loss

Set via `loss_type` (`cross_entropy` default).

| `loss_type` | Notes |
|---|---|
| `cross_entropy` | Standard classification loss. |
| `focal` | Down-weights easy examples for class imbalance via `focal_gamma` (default 2.0). Works for both single-label (softmax) and multi-label (sigmoid). `focal_alpha` balances positives against negatives (`alpha_t = alpha*y + (1-alpha)*(1-y)`) and applies to the multi-label head only — it is rejected on a single-label head, where per-class balancing is `class_weights` / `derive_class_weights`. |
| `label_smoothing_ce` | CE with smoothing epsilon `label_smoothing` (typical 0.05–0.1). Single-label only — raises on a multi-label dataset. |

HuggingFace's own `label_smoothing_factor` is rejected at construction: the trainer overrides `compute_loss` wholesale and never reaches HF's `label_smoother`, so the knob would smooth nothing. Use `loss_type: label_smoothing_ce` with `label_smoothing`.

## Class weighting

- `derive_class_weights: true` — balanced weights `n_samples / (n_present_classes * count_c)` from label counts all-reduced across every rank, so pre-sharded and TP/ETP runs weight identically (classes absent from the global train set keep weight 1.0). Single-label only; raises on multi-label. The counts are scanned once at trainer construction, so a `-1` label raises there.
- `class_weights: [1.0, 2.5, 1.0]` — manual weights ordered by label ID. On a multi-label dataset it becomes the BCE `pos_weight`.

The two are mutually exclusive; setting both raises, on the CLI as well as in YAML. Either one switches the trainer off the model's built-in head loss onto its own weighted objective.

That objective evaluates in **fp32**: the head's logits are upcast at the loss and the weight vector keeps the fp32 it was derived in, never rounded to the logits' dtype. A weighted CE is `w_y · (logit_y − logsumexp(logits))`, a cancelling difference bf16 resolves ~0.5% off its exact value — and `derive_class_weights` spans orders of magnitude, so rounding the vector is itself a reweighting. Pooled logits are `[B, num_labels]`, so the upcast costs nothing.

## Metrics

Selected automatically by problem type: accuracy (exact-match for multi-label), precision/recall/F1 (binary averaging for 2 labels, weighted for multi-class, macro for multi-label), and MCC (`compute_mcc`, default on, single-label only). Optional: `compute_per_class_metrics` (single-label only), `compute_auc_roc` (binary positive class; multiclass one-vs-rest macro; multi-label macro), and `multi_label_threshold` (default 0.5) for the sigmoid decision.

`auc_roc` is the one metric an eval slice can leave **undefined** (a class with no samples there), and it is then omitted rather than reported as NaN. So `metric_for_best_model: auc_roc` raises at construction when `compute_auc_roc` is off, and raises again at the eval that cannot define it — rank checkpoints on `accuracy` / `f1` / `mcc` instead.

## PEFT / LoRA

```yaml
use_peft: true
lora_r: 16
lora_alpha: 32
lora_target_modules: [q_proj, k_proj, v_proj, o_proj]
lora_task_type: SEQ_CLS
```

LoRA is rejected under TP, EP+TP and PP.

## Parallelism

EP, TP, pure ETP, EP+TP and EP+ETP are supported. **CP is not supported**: the classification head needs the full pooled representation, which CP's per-rank sequence split cannot provide without a full gather. The trainer declares `_supports_pp` — pipeline parallelism itself is [not yet available in this release](../parallelism/pipeline-parallelism.md); its shipped seams here are a `PPLossAdapter.predictions_fn` (the pooled `[B, num_labels]` would cross the chain instead of the full `[B, S, num_labels]` plane) plus construction gates on `config.pad_token_id` (pooling takes the rightmost non-pad position; the script records it from the tokenizer unless that tokenizer has no pad token of its own and borrows eos) and `num_labels >= 2` (transformers treats a one-logit head as regression, which the PP loss adapter does not wire). See [Trainer Compatibility](../reference/trainer-architecture.md#trainer-compatibility).

## Vision-language

Not supported. The script has no image path at all: it loads through the text-only `load_script_model`, never builds a processor, and never probes `is_vlm_run` — so even a family with a multimodal score head trains on the rendered text alone.

Head coverage is the second limit: transformers 5.16 ships a VLM sequence-classification head for only a few families (Gemma3, dense Qwen3.5, T5Gemma2, ModernVBert), and the toolkit registers two more, validated on text inputs — the [reward-modeling roster](preference/reward-modeling.md#vision-language) owns the list.

## Configuration reference

`ClassificationConfig` and the parallelism flags are tabulated in [Configuration Reference](../reference/configuration-reference.md#classificationconfig). `CLFScriptArguments` adds only `text_field` (default `None`) on top of `CommonScriptArguments`.

## Related pages

- [Reward Modeling](preference/reward-modeling.md) — same architecture for scalar reward prediction
- [SFT](sft.md) — generative alternative
- [Expert Parallelism](../parallelism/expert-parallelism.md) · [Tensor Parallelism](../parallelism/tensor-parallelism.md)
- [Scripts Reference](../reference/scripts-reference.md)
