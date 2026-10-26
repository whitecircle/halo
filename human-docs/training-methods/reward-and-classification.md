# Reward Modeling and Classification

Both methods put a small scoring head on a language model and train it with
`AutoModelForSequenceClassification`. A reward model emits one unbounded scalar per (prompt, completion)
and is fit to preference pairs; use it to rank or filter generations. A classifier emits a label per
document; use it to tag them. Both need a sequence-classification head for the model family, which
transformers ships for most dense text families but, among the supported MoE families, only for GPT-OSS,
Qwen3 MoE and Mistral 4; Halo adds Gemma 4 and MoE Qwen3.5/3.6.

## Reward modeling

The Bradley-Terry objective trains the chosen answer to score above the rejected one. The output is a
scorer for rejection sampling or a GRPO reward. To move the policy itself on the same pairs, use
[preference tuning](preference.md).

**Data.** The `prompt` / `chosen` / `rejected` pairs DPO and SMPO read, or implicit-prompt pairs that
repeat the shared turns inside both sides (`Skywork/Skywork-Reward-Preference-80K-v0.2`). TRL templates
the rows itself: a `prompt` column whose completions repeat those turns renders the prompt twice,
silently, so check one rendered row before a long run.

```bash
halo launch reward/reward-model examples/reward/qwen3_5/rm-qwen3.5-9b-skywork-pref80k.yaml -n 8
```

`max_length` is a filter here: pairs longer than it are dropped. LoRA needs `lora_task_type: SEQ_CLS`,
which keeps the new `score` head trainable. Once trained, `halo run inference/reward-model/score` and
`halo run inference/reward-model/rejection-sample` generate against a served endpoint and score the
results.

**What to watch.** `accuracy` is the share of pairs ranked correctly (0.5 is chance), and `margin` the
mean score gap. Watch `min_reward` / `max_reward` for a distribution drifting far from zero.

## Classification

Single-label and multi-label sequence classification, for safety filters, topic routing or detectors.
Text only.

**Data.** A `prompt` conversation or a raw text column named by `text_field`, plus `label`: one value per
row (string or integer) for single-label, a list for multi-label. The label set comes from the training
split, so you never write `num_labels` or `label2id`. Rows are truncated to `max_length`, not dropped. A
`-1` label marks an unlabeled row, which a single-label split the run reads refuses.

```bash
halo launch classification examples/classification/qwen3_5/clf-qwen3.5-9b-mage.yaml -n 8
```

For imbalanced data set `class_weights`, or `derive_class_weights` on single-label data.

**What to watch.** Rank checkpoints on `accuracy` (`exact_match_accuracy` on multi-label), `f1`, or, on
single-label runs, `mcc`, which stays honest under class imbalance.

Every knob, the vision-language reward path and the head roster:
[Reward modeling](../../agent-docs/training-methods/preference/reward-modeling.md) ↗ ·
[Classification](../../agent-docs/training-methods/classification.md) ↗.
