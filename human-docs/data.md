# Datasets

A dataset is a list of records; conversations are lists of
`{"role": "system"|"user"|"assistant", "content": "..."}` messages. Each
method reads a fixed set of columns:

| Method | Required columns | Types |
| --- | --- | --- |
| SFT (incl. VLM) | `prompt` | `List[Dict]` |
| DPO / SMPO | `chosen`, `rejected` (+ optional `prompt`) | all `List[Dict]` |
| Reward | `chosen`, `rejected` (+ optional `prompt`) | all `List[Dict]` |
| KTO | `prompt`, `completion`, `label` | `List[Dict]`, `List[Dict]`/`str`, `bool` |
| Offline GRPO | `prompt`, `completions`, `rewards` | `List[Dict]`, `List[List[Dict]]`, `List[float]` |
| Classification | `prompt` or `text_field`, `label` | `List[Dict]`/`str`, `str`/`List[str]` |
| Online GRPO / async GRPO with environments | `prompt`, `answer` | `str`/`List[Dict]`, `str` |

Some column names are configurable — `conversation_field` for SFT, `text_field`
for classification, `completion_field`/`label_field` for KTO,
`prompt_field`/`answer_field` for the two online RL paths. Preference and
offline-GRPO column names are fixed; a preference pair with no `prompt` column
has the turns the two sides share extracted as one. One SFT record:

```jsonl
{"prompt": [{"role": "user", "content": "What is machine learning?"}, {"role": "assistant", "content": "Machine learning is ..."}]}
```

## Where data can come from

`dataset` accepts a HuggingFace Hub ID (`repo@split`, `repo:config`), a local
path (JSON/JSONL, Parquet, Arrow, or a `save_to_disk` directory), or an S3 path
(`s3://my-bucket/key`, with your own AWS credentials). Pass a list to mix sources; `dataset_ratio` is the fraction of
each source kept, not a mixing weight:

```yaml
dataset:
  - HuggingFaceH4/ultrachat_200k@train_sft
  - s3://my-bucket/chat/v1/train
dataset_ratio: [1.0, 0.5]
```

Two things to watch when you mix sources. Only the columns **common to every
source** survive (bar the ones your config names, which are null-filled where a
source lacks them), and columns whose types disagree are dropped — that's usually
why a field you need has vanished. And a list is loaded **fully replicated on
every rank**, which silently turns off the probes that detect a pre-processed or
sharded dataset. Point `dataset` at a single path when you want either.

## Large corpora: preprocess once

Tokenizing and packing a big corpus on the fly wastes startup time on every
run. Do it once, offline:

```bash
halo run prepare-dataset -- --help
```

This tokenizes, optionally packs, and shards the corpus, and can push the result
to the Hub or S3. Five flags decide whether the output is usable:

- `--mode chat` (the default) reads the conversation field through the chat
  template; `--mode text` tokenizes `--text-field` raw for (continued)
  pre-training. Either way the shard count must be at least your data-parallel
  size.
- `--pack-sequences` turns packing on; without it the rows are tokenized and
  sharded one document each.
- `--packing-strategy` then decides what happens to a document longer than
  `--max-length`. The default `bfd` packs best-fit-decreasing and **discards**
  the overflow; `bfd_split` carries the overflow into later examples and is the
  lossless choice for pre-training; `wrapped` concatenates and chunks across
  document boundaries, keeping every token but losing the boundaries the
  collator needs to reset attention.
- `--max-length` is stamped into the dataset metadata and must **equal** the
  training config's `max_length`. A mismatch either way raises at load, since
  rows are baked at prep time and never re-truncated.
- `--test-size`. Without it a single-split input is written train-only: a
  sharded output is then rejected at startup naming the missing split, and an
  unsharded one trains with a placeholder test split warned in from the first
  100 train rows.

`--tokenizer-backend gigatoken` swaps the HF tokenizer for a Rust bulk encoder,
roughly 6× faster on UltraChat 200K with the Qwen3-0.6B tokenizer. It verifies
its IDs against the HF tokenizer at startup and raises on any divergence. It
ships in the training image; outside it, `uv pip install 'halo[gigatoken]'`.

Training configs carry the matching field `tokenizer_backend` (`hf` by default,
`gigatoken` to opt in). Embedding training rejects anything but `hf` —
SentenceTransformers owns its own tokenization there.

## Data for the RL environments

The code-contests environment reads a prepared pool rather than a plain dataset.
`halo run compact-code-tests` caps a test corpus to one suite per problem, then
`halo run prepare-code-dataset` builds the pool and, with `--push_bands`, one
config per rating band — the `<repo>:<band>` the shipped examples name in
`dataset:`. How the pool is consumed during training is on
[Async GRPO with Environments](training-methods/async-grpo-environments.md); the
build flags are in
[Code Contests](../agent-docs/training-methods/grpo/environments/code-contests.md) ↗.

Full schema (including multimodal content), collators, and the S3 utilities:
[Dataset Formats](../agent-docs/data/dataset-formats.md) ↗ ·
[Pre-Processing](../agent-docs/data/dataset-preparation.md) ↗ ·
[S3 Utilities](../agent-docs/data/s3-utilities.md) ↗.
