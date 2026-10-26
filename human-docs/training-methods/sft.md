# Supervised Fine-Tuning

SFT trains with cross-entropy on example conversations: the model learns a format, a behavior or a
domain by imitating the assistant turns you show it. The same method takes vision-language data, raw
text for continued pre-training, and LoRA or QLoRA as a config switch
([Choosing a Method](../choosing-a-method.md) if you are unsure this is the right one).

## Data

One record is one conversation, under the column named by `conversation_field` (default `prompt`):

```jsonl
{"prompt": [{"role": "user", "content": "Explain gravity."}, {"role": "assistant", "content": "Gravity is ..."}]}
```

A row longer than `max_length` is **dropped, not truncated**. Images ride inside message content or in
an `images_field` column; a multimodal checkpoint meeting image data makes the run vision-language.
Sources and mixing are in [Datasets](../data.md).

## Run

```bash
halo launch sft examples/sft/qwen3/qwen3-4b-ultrachat.yaml -n 8
```

Its `-lora` and `-qlora` siblings run on one GPU, and `examples/sft/<family>/` holds the MoE and vision
recipes. Before a long run, smoke the config on two GPUs with
`--max_steps=5 --save_strategy=no --report_to=none`.

Three settings decide most runs:

- **`assistant_message_template`** must byte-match what the chat template renders at the start of an
  assistant turn: completion-only masking (on by default) finds the turns with it. Train under the
  template you will serve with; a mismatch costs quality and never shows in the loss.
- **`packing: true`** fills `max_length` blocks with short rows and needs an explicit `max_length`. Rows
  that share a long prefix can pack as trees instead
  ([Tree Packing](../../agent-docs/optimization/tree-packing.md) ↗).
- **The learning rate.** `2.5e-6`–`5e-6` for a full fine-tune of a large model, 5–10× that for LoRA. Too
  high a rate erases pretrained capability without showing in the loss.

Continued pre-training tokenizes the corpus once (`halo run prepare-dataset --mode text ...`) and
trains with `train_on_completions_only: false`; `init_from_scratch: true` starts from random weights,
on data parallelism only ([Pre-training](../../agent-docs/training-methods/pretraining.md) ↗).

## What to watch

The loss should fall smoothly in the first hundred steps. A flat loss almost always means the assistant
marker misses most rows, so their tokens are masked; each such row logs a "Could not find response key"
warning. Set `log_decoded_samples: true` once to read exactly what the model is shown, and check how
many rows the length filter dropped: the log warns when it drops half the dataset or more.

Every knob, the vision-language rules and the parallelism limits:
[SFT reference](../../agent-docs/training-methods/sft.md) ↗.
