# Preference Tuning

Three methods learn from human or model judgments of which answer is better. SMPO and DPO read
`chosen`/`rejected` pairs; KTO reads unpaired thumbs-up/down labels. All three move the policy itself.
If you want a scorer to reuse later, train a [reward model](reward-and-classification.md) on the same
pairs instead.

## Which one

- **SMPO** holds no reference model, so one model sits in memory instead of two. Its margin loss reaches
  exactly zero once a pair is far enough apart, and a built-in SFT anchor keeps generation quality from
  drifting. Start here unless you have a reason not to.
- **DPO** keeps the policy close to a frozen reference through a KL term. Pick it when staying close to
  a specific checkpoint is the point, and budget for the second model.
- **KTO** takes one completion per row with a boolean label. Pick it when you never collected pairs.

## Data

SMPO and DPO read the same three columns, each a message list. `prompt` is the conversation up to the
point where the two answers diverge:

```jsonl
{"prompt": [{"role": "user", "content": "Capital of France?"}], "chosen": [{"role": "assistant", "content": "Paris."}], "rejected": [{"role": "assistant", "content": "It is in Europe."}]}
```

KTO reads one completion and a label, renamed with `completion_field` and `label_field`:

```jsonl
{"prompt": [{"role": "user", "content": "What is 2+2?"}], "completion": [{"role": "assistant", "content": "4"}], "label": true}
```

Only SMPO splits `max_length` into a prompt and a completion share. DPO and KTO truncate prompt plus
completion together, so filter over-long prompts out of the dataset
([Sequence length](../../agent-docs/reference/configuration-reference.md#sequence-length-caps-vs-generation-budgets) ↗).

## Run

```bash
halo launch smpo examples/preference/qwen3_5/smpo-qwen3.5-9b-tulu3-prefmix.yaml -n 8
halo launch dpo  examples/preference/qwen3_5/dpo-qwen3.5-9b-tulu3-prefmix.yaml -n 8
halo launch kto  examples/preference/qwen3_5/kto-qwen3.5-9b-kto-mix-14k.yaml -n 8
```

These recipes are dense; the MoE ones under `examples/preference/gptoss/`,
`examples/preference/gemma4/` and `examples/preference/nemotron_h/` run expert parallelism. SMPO alone also takes context parallelism. All
three take pipeline parallelism, where DPO and KTO need precomputed reference log-probs
([Supported matrix](../supported-matrix.md#training-methods)).

- **SMPO**: `target_margin` is the log-probability gap a pair should reach, and `chosen_sft_ratio` splits
  the anchor between the chosen and rejected sides.
- **DPO's reference is where the memory goes.** With `use_peft: true` it is the base model with the
  adapter switched off. Otherwise a second full model stays resident, unsharded under expert or tensor
  parallelism; there `precompute_ref_log_probs: true` scores the reference once before step 1 instead.
  KTO follows the same rules.
- **KTO**'s `kto` loss builds each row's KL term from its batch neighbors, so it needs
  `per_device_train_batch_size` of 2 or more. When the labels are skewed, set `desirable_weight` and
  `undesirable_weight` to the inverse label ratio.

## What to watch

- `rewards/accuracies` (DPO and SMPO) is the share of pairs scored in the right order. 0.5 is chance,
  and it should climb early. `rewards/margins` is the separation being bought.
- `logps/chosen` and `logps/rejected` falling together means the model makes both answers less likely,
  which degrades generation. Lower the learning rate or anchor harder: raise `chosen_sft_ratio` on SMPO,
  use `loss_type: [sigmoid, sft]` on DPO.
- A NaN loss on SMPO usually means the log-prob clips (`min_log_prob`, `lower_clip_percentile`) were
  turned off.

Every knob and loss type, and resuming a precompute run:
[SMPO](../../agent-docs/training-methods/preference/smpo.md) ↗ ·
[DPO](../../agent-docs/training-methods/preference/dpo.md) ↗ ·
[KTO](../../agent-docs/training-methods/preference/kto.md) ↗.
