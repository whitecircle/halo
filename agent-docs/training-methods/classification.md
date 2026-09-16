# Classification

Sequence classification — single-label (multi-class) and multi-label — on `AutoModelForSequenceClassification`. Trainer `ClassificationTrainer` (`src/trainers/reward/classification.py`), script `scripts/training/classification.py`. Text-only. For a scalar preference score on the same architecture use [Reward Modeling](preference/reward-modeling.md).

EP, TP, ETP, EP+TP and EP+ETP run, as does plain FSDP2 data parallelism. CP does not — the head pools the whole sequence ([matrix](../reference/trainer-architecture.md#trainer-compatibility)). The trainer declares `_supports_pp`, but pipeline parallelism is [not yet available in this release](../parallelism/pipeline-parallelism.md).

## Dataset

A `prompt` conversation plus a `label` column: a string for single-label, a list for multi-label. `text_field` names a raw text column instead, wrapped as one user turn. A dataset with neither is rejected right after the dataset load. Row shapes: [Dataset Formats](../data/dataset-formats.md#classification).

Rows are chat-templated and **truncated** to `max_length` — a label describes the whole document, so a shortened document keeps a valid label. A dataset handed to `ClassificationTrainer` directly (no `input_ids` column) is instead tokenized untruncated and its over-length rows dropped, with the fraction logged.

The label set comes from the training split: stringified, sorted for deterministic ids, and unioned across ranks on a pre-sharded dataset so every rank builds an identically shaped head. `num_labels`, `label2id` and `id2label` follow from it, never from YAML; single- vs multi-label follows whether `label` is a list. Labels seen only in validation or test are added with a warning.

`ds["test"]` is the eval split — a `validation` split contributes labels only.

`-1` is dropped from the label set. A multi-label row treats it as absence, but a single-label row keeps the raw value and reaches the loss as an out-of-range class index — the ignore sentinel is `-100`. Filter those rows out.

## Configuration

```yaml
model_name_or_path: Qwen/Qwen3.5-9B
dataset: yaful/MAGE
text_field: text
test_size: 0.1
attn_implementation: sdpa
max_length: 2048
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 2.0e-05
num_train_epochs: 1
gradient_checkpointing: true
output_dir: checkpoints/clf-qwen3.5-9b-mage
```

| Knob | Default | Effect |
|---|---|---|
| `max_length` | `1024` | Truncation length; `null` or non-positive → the context window |
| `loss_type` | `cross_entropy` | Also `focal` (`focal_gamma` `2.0`; `focal_alpha` on multi-label heads only) and `label_smoothing_ce` (single-label only) |
| `class_weights` | `None` | Per-class weights by label id; `pos_weight` on the multi-label BCE |
| `derive_class_weights` | `false` | Balanced weights from world-reduced label counts; single-label only |
| `multi_label_threshold` | `0.5` | Sigmoid decision threshold |
| `compute_mcc` / `compute_auc_roc` / `compute_per_class_metrics` | `true` / `false` / `false` | Extra eval metrics; MCC and per-class are single-label only |

Full field list: [ClassificationConfig](../reference/configuration-reference.md#classificationconfig).

The two weighting knobs are mutually exclusive — setting both raises, on the CLI as well as in YAML. Either switches the trainer off the model's built-in head loss onto its own weighted objective, evaluated in fp32. HF's own `label_smoothing_factor` is rejected at construction: the trainer builds its own loss and never reaches HF's `label_smoother`.

The script defaults `attn_implementation` to `sdpa` when the YAML pins none, because batches are right-padded and the auto-selected FA4 would take its slow varlen path. Under `reset_sinks: false` it requests nothing and lets the resolver auto-select, which there accepts only a sink-carrying backend.

## Launch

```bash
# MoE, expert parallel
torchrun --nproc_per_node=8 scripts/training/classification.py \
    examples/classification/gptoss/clf-gptoss-20b-mage-ep.yaml --expert_parallel_size=8

# dense, plain FSDP2 data parallel
halo launch classification examples/classification/qwen3_5/clf-qwen3.5-9b-mage.yaml --nproc 8
```

LoRA takes `lora_task_type: SEQ_CLS` and is rejected under TP, EP+TP and PP ([PEFT](../optimization/peft.md#parallelism-compatibility)).

### Pipeline parallelism

Pipeline parallelism is [not yet available in this release](../parallelism/pipeline-parallelism.md); the shipped seams here would serve `compute_metrics`, since the trainer declares `PPLossAdapter.predictions_fn` and the pooled `[B, num_labels]` would cross the chain rather than the full plane. Two construction gates ship with them and raise — a `config.pad_token_id` unset or disagreeing with the tokenizer's (pooling takes the rightmost non-pad position), and a single-label head with `num_labels < 2`, which transformers treats as regression.

## Vision-language

Not supported. The script loads through the text-only `load_script_model`, builds no processor, and refuses `text_only_model` as well as any image column.

A multimodal checkpoint is also checked for a sequence-classification head before the load and refused without one. transformers 5.16 ships that head for a few families and the toolkit registers two more ([roster](preference/reward-modeling.md#vision-language)).

## Testing a setup

```bash
torchrun --nproc_per_node=2 scripts/training/classification.py <config> \
    --max_steps=5 --save_strategy=no --report_to=none
```

Covering tests: `pytest tests/cpu/trainers tests/cpu/config -m cpu` and `tests/gpu/trainers/other/test_classification.py`.

## What to watch

| Signal | Reading |
|---|---|
| `accuracy` | Single-label; multi-label reports `exact_match_accuracy` instead |
| `f1` | Binary averaging at 2 labels, weighted for multi-class, macro for multi-label |
| `mcc` | Stays honest under class imbalance |

Failure signatures:

- `metric_for_best_model: auc_roc` raises — either `compute_auc_roc` is off, or an eval slice holds a class with no samples and the metric is undefined there. Rank on `accuracy`, `f1` or `mcc`.
- `derive_class_weights` raises at construction: a `-1` label reached the count scan. Filter it out.
- Documents silently shortened: `max_length` truncates rather than drops. Raise it when the label depends on the tail.

## Related pages

- [Reward Modeling](preference/reward-modeling.md) · [SFT](sft.md)
- [Expert Parallelism](../parallelism/expert-parallelism.md) · [Pipeline Parallelism](../parallelism/pipeline-parallelism.md)
- [ClassificationConfig Reference](../reference/configuration-reference.md#classificationconfig) · [Scripts Reference](../reference/scripts-reference.md)
