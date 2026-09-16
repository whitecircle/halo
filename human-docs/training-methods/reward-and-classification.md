# Reward Modeling and Classification

Both methods put a small scoring head on a language model and train it with
`AutoModelForSequenceClassification`. A reward model emits one unbounded scalar per
(prompt, completion) and is fit to preference pairs; a classifier emits a label per document. Use a
reward model to rank or filter generations, a classifier to tag them.

## Reward modeling

The Bradley-Terry objective says the chosen answer should score above the rejected one. The output is
a scorer you consume later — for rejection sampling, for filtering a dataset, or as the reward in
GRPO. To move the policy itself on the same pairs, use [preference tuning](preference.md) instead.

### Data

The same `prompt` / `chosen` / `rejected` pairs DPO and SMPO read, plus implicit-prompt datasets that
carry no `prompt` column and repeat the shared turns inside both sides (that is the shape of
`Skywork/Skywork-Reward-Preference-80K-v0.2`). An optional `margin` column widens the target gap per
row.

> [!WARNING]
> On the text path the rows are templated by TRL itself, with no hub-shape normalization. A dataset
> that has both a `prompt` column and completions repeating those same turns renders the prompt
> twice, silently. Check one rendered row before committing to a long run.

### Config

From `examples/reward/qwen3_5/rm-qwen3.5-9b-skywork-pref80k.yaml`:

```yaml
model_name_or_path: Qwen/Qwen3.5-9B
dataset: Skywork/Skywork-Reward-Preference-80K-v0.2
attn_implementation: sdpa    # reward batches are right-padded
test_size: 0.05
max_length: 4096
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-05
num_train_epochs: 1
output_dir: checkpoints/rm-qwen3.5-9b-skywork-pref80k
```

`max_length` here is a **filter**, not a truncation: pairs longer than it are dropped from the
dataset. If your split shrinks unexpectedly, that is why. `center_rewards_coefficient` pulls the
score distribution toward zero mean, which makes scores comparable across runs.

LoRA needs `lora_task_type: SEQ_CLS`, which is what keeps the freshly initialized `score` head
trainable. Leave it out and Halo only warns, while the head never trains and accuracy sits at 0.5.

### Run

```bash
halo launch rewards examples/reward/qwen3_5/rm-qwen3.5-9b-skywork-pref80k.yaml -n 8
```

`examples/reward/gptoss/rm-gptoss-20b-skywork-pref80k-ep.yaml` is the MoE version, with expert
parallelism pinned in the config. Once trained, `halo run rm-scoring` and
`halo run rm-rejection-sampling` generate against a served endpoint and score the results.

### What to watch

`accuracy` is the share of pairs ranked correctly (0.5 is chance) and `margin` the mean score gap.
Watch `min_reward` / `max_reward` for a distribution drifting far from zero.

## Classification

Single-label (multi-class) and multi-label sequence classification, for safety filters, topic
routing, or detectors. Text only.

### Data

A `prompt` conversation or a raw text column named by `text_field`, plus a `label` column — one value per
row for single-label (strings and integers both work; ids come from the string form), a list for
multi-label. The label set is derived from the **training**
split, sorted for stable ids, so `num_labels`, `label2id` and `id2label` are never written by hand.
Labels that appear only in validation are added with a warning.

Rows are truncated to `max_length` rather than dropped here, because a label describes a whole
document and a shortened document still carries it. A `-1` label is dropped from the label set; on a
single-label row it reaches the loss as an out-of-range class, so filter those rows out.

### Config

From `examples/classification/qwen3_5/clf-qwen3.5-9b-mage.yaml`:

```yaml
model_name_or_path: Qwen/Qwen3.5-9B
dataset: yaful/MAGE
text_field: text
attn_implementation: sdpa    # classification batches are right-padded
test_size: 0.1
max_length: 2048
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 2.0e-05
output_dir: checkpoints/clf-qwen3.5-9b-mage
```

For imbalanced data, either set `class_weights` per label id or turn on `derive_class_weights` to
compute balanced weights from the observed counts — the two are mutually exclusive and setting both
raises. `loss_type` also takes `focal` and `label_smoothing_ce`, and `multi_label_threshold` moves
the sigmoid decision point on multi-label heads.

### Run

```bash
halo launch classification examples/classification/qwen3_5/clf-qwen3.5-9b-mage.yaml -n 8
```

The MoE recipe is `examples/classification/gptoss/clf-gptoss-20b-mage-ep.yaml`. LoRA takes
`lora_task_type: SEQ_CLS` here too.

### What to watch

`accuracy` (or `exact_match_accuracy` on multi-label), `f1`, and `mcc`, which stays honest under
class imbalance — rank checkpoints on one of those. `metric_for_best_model: auc_roc` raises unless
`compute_auc_roc` is on and every class appears in the eval slice.

## Heads and modality

Both methods need the model family to have a sequence-classification head. transformers ships one
for most text families and Halo registers Gemma 4 and MoE Qwen3.5/3.6 on top. A multimodal
checkpoint without one is refused before the model loads, naming the families that work; a text
family without one fails inside the `Auto*` load instead. Reward modeling takes
images on those families (`images_field`, images merge into the shared prompt); classification is
text-only and refuses any image column.

## Go deeper

- [Preference tuning](preference.md) · [Offline GRPO](offline-grpo.md) · [Datasets](../data.md)
- [Reward modeling](../../agent-docs/training-methods/preference/reward-modeling.md) ↗ ·
  [Classification](../../agent-docs/training-methods/classification.md) ↗
