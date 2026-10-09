# Reward Modeling and Classification

Both methods put a small scoring head on a language model and train it with
`AutoModelForSequenceClassification`. A reward model emits one unbounded scalar per (prompt, completion)
and is fit to preference pairs. A classifier emits a label per document. Use a reward model to rank or
filter generations, and a classifier to tag them.

## Reward modeling

The Bradley-Terry objective trains the chosen answer to score above the rejected one. The output is a
scorer you use later: for rejection sampling (building preference or offline GRPO data from fresh
generations), or as the reward in GRPO. To move the policy itself on the same pairs, use
[preference tuning](preference.md) instead.

### Data

The same `prompt` / `chosen` / `rejected` pairs DPO and SMPO read. Implicit-prompt datasets work too:
they carry no `prompt` column and repeat the shared turns inside both sides, as
`Skywork/Skywork-Reward-Preference-80K-v0.2` does. An optional `margin` column widens the target gap per
row.

> [!WARNING]
> On the text path TRL templates the rows itself, with no normalization of the hub shape. A dataset
> with a `prompt` column whose completions repeat those same turns renders the prompt twice, silently.
> Check one rendered row before a long run.

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

- `max_length` is a **filter**, not a truncation: pairs longer than it are dropped from the dataset. If
  your split shrinks unexpectedly, that is why.
- `center_rewards_coefficient` penalizes `(chosen + rejected)²`, which pulls the score distribution
  toward zero mean.
- LoRA needs `lora_task_type: SEQ_CLS`, which keeps the freshly initialized `score` head trainable.
  Without it Halo only warns, the head stays frozen, and accuracy barely moves. Adding
  `lora_modules_to_save: [score]` also works.

### Run

```bash
halo launch rewards examples/reward/qwen3_5/rm-qwen3.5-9b-skywork-pref80k.yaml -n 8
```

The MoE versions under `examples/reward/gptoss/` and `examples/reward/gemma4/` pin expert parallelism in
the config. Once the model is trained, `halo run rm-scoring` and `halo run rm-rejection-sampling`
generate against a served endpoint and score the results.

### What to watch

`accuracy` is the share of pairs ranked correctly (0.5 is chance), and `margin` the mean score gap.
Watch `min_reward` / `max_reward` for a distribution drifting far from zero.

## Classification

Single-label (multi-class) and multi-label sequence classification, for safety filters, topic routing
or detectors. Text only.

### Data

Each row has a `prompt` conversation or a raw text column named by `text_field`, plus a `label` column:

- **Single-label:** one value per row. Strings and integers both work; ids come from the string form.
- **Multi-label:** a list per row.

The label set is derived from the **training** split and sorted for stable ids, so you never write
`num_labels`, `label2id` or `id2label` by hand. Labels that appear only in validation or test are added
with a warning.

Rows are truncated to `max_length`, not dropped: a label describes a whole document, and a shortened
document still carries it.

A `-1` label marks an unlabeled row. Multi-label rows read it as absence. A single-label split that
carries one is refused before the model loads, if the run reads that split (train, and the eval split
when evaluating). Filter those rows out of train. For an unlabeled eval split (a GLUE-style `test`), use
`dataset: <id>@train` with `test_size`, or `eval_strategy: no`.

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

- **Imbalanced data:** set `class_weights` per label id (the BCE `pos_weight` on multi-label heads). On
  single-label data you can instead turn on `derive_class_weights` to compute balanced weights from the
  observed counts. Setting both raises.
- `loss_type` also takes `focal` and, on single-label heads, `label_smoothing_ce`.
- `multi_label_threshold` moves the sigmoid decision point on multi-label heads.
- A single-label head needs at least two classes. A one-logit head is refused, since regression is not
  supported.

### Run

```bash
halo launch classification examples/classification/qwen3_5/clf-qwen3.5-9b-mage.yaml -n 8
```

The MoE recipe is `examples/classification/gptoss/clf-gptoss-20b-mage-ep.yaml`. LoRA takes
`lora_task_type: SEQ_CLS` here too.

### What to watch

Rank checkpoints on `accuracy` (`exact_match_accuracy` on multi-label), `f1`, or, on single-label runs,
`mcc`, which stays honest under class imbalance. `metric_for_best_model: auc_roc` raises unless
`compute_auc_roc` is on and every class appears in the eval slice.

## Heads and modality

Both methods need a sequence-classification head for the model family. transformers ships one for most
dense text families, but among the supported MoE families only for GPT-OSS, Qwen3 MoE and Mistral 4.
Halo adds Gemma 4 and MoE Qwen3.5/3.6; the other supported MoE families have none.

- A multimodal checkpoint without a head is refused before the model loads, with a list of the families
  that work. A text family without one fails inside the `Auto*` load.
- Reward modeling takes images on the supported families (`images_field`; images merge into the shared
  prompt).
- Classification is text-only. It refuses an `images`, `image` or `pixel_values` column and image parts
  in the prompt.

## Go deeper

- [Preference tuning](preference.md) · [Offline GRPO](offline-grpo.md) · [Datasets](../data.md)
- [Reward modeling](../../agent-docs/training-methods/preference/reward-modeling.md) ↗ ·
  [Classification](../../agent-docs/training-methods/classification.md) ↗
