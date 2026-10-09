# Datasets

A dataset is a list of records. Conversation columns hold lists of
`{"role": "system"|"user"|"assistant", "content": "..."}` messages. One SFT
record:

```jsonl
{"prompt": [{"role": "user", "content": "What is machine learning?"}, {"role": "assistant", "content": "Machine learning is ..."}]}
```

## Columns per method

| Method | Required columns | Types |
| --- | --- | --- |
| SFT (incl. VLM) | `prompt` | `List[Dict]` |
| DPO / SMPO | `chosen`, `rejected` (+ optional `prompt`) | `List[Dict]` (`prompt` may be `str`) |
| Reward | `chosen`, `rejected` (+ optional `prompt`) | `List[Dict]` (`prompt` may be `str`) |
| KTO | `prompt`, `completion`, `label` | `List[Dict]`, `List[Dict]`/`str`, `bool` |
| Offline GRPO | `prompt`, `completions`, `rewards` | `List[Dict]`, `List[List[Dict]]`, `List[float]` |
| Classification | `prompt` or `text_field`, `label` | `List[Dict]`/`str`, a label or a list of labels (strings or ints) |
| Online GRPO | `prompt`, `answer` | `str`/`List[Dict]`, `str` |
| Async GRPO with environments | `prompt` (+ `answer` where the environment grades one) | `str`/`List[Dict]`, `str` |

These column names can be renamed in the config:

- SFT: `conversation_field`
- Classification: `text_field`
- KTO: `completion_field`, `label_field`
- Online and async GRPO: `prompt_field`, `answer_field`

The `prompt`/`chosen`/`rejected` and offline GRPO column names are fixed. A
preference pair without a `prompt` column gets the turns both sides share
extracted as the prompt.

## Where data can come from

`dataset` takes one of:

- a HuggingFace Hub ID (`repo@split`, `repo:config`);
- a local path: JSON/JSONL, Parquet, Arrow, CSV, or a `save_to_disk` directory;
- an S3 path (`s3://my-bucket/key`), with your own AWS credentials.

Pass a list to mix sources. `dataset_ratio` is the fraction of each source to
keep, not a mixing weight:

```yaml
dataset:
  - HuggingFaceH4/ultrachat_200k@train_sft
  - s3://my-bucket/chat/v1/train
dataset_ratio: [1.0, 0.5]
```

Mixing sources has side effects:

- Only columns common to every source survive. Every source must carry the
  conversation column your config names, or the load raises naming the source.
  The tools column it names is null-filled where a source lacks it.
- Columns whose types differ between sources are dropped. This is the usual
  reason a field goes missing.
- A list loads fully on every rank. A sharded entry loads whole on every rank,
  and a pre-processed dataset is detected only at a single path. Point `dataset`
  at a single path to use either.

## Preprocess large corpora once

Tokenizing and packing a large corpus on the fly costs startup time on every
run. Do it once, offline:

```bash
halo run prepare-dataset -- --help
```

The tool tokenizes, optionally packs, and shards the corpus, and can push the
result to the Hub or S3. These flags decide whether the output is usable:

- **`--mode`.** `chat` (the default) renders the conversation field through the
  chat template. `text` tokenizes `--text-field` as raw text, for pretraining.
- **`--num-shards`.** The default `1` writes one unsharded dataset that trains at
  any data-parallel size. Above 1, the shard count must be at least your
  data-parallel size.
- **`--pack-sequences`.** Turns packing on. Without it each row holds one
  document.
- **`--packing-strategy`.** How packing fills rows:
  - `bfd` (default): best-fit-decreasing; a document longer than
    `--max-length` loses its overflow.
  - `bfd_split`: the same packing, with the overflow carried into later
    examples. Use it for lossless pretraining.
  - `wrapped`: concatenates and chunks across documents. It keeps every token
    but loses the document boundaries the collator uses to reset attention.

  In chat mode a conversation longer than `--max-length` is dropped before
  packing, whatever the strategy.
- **`--max-length`.** Stored in the dataset metadata. It must equal the training
  config's `max_length`, or the load raises, since rows are never re-truncated.
- **`--test-size`.** Without it, a single-split input is written train-only. The
  run then warns and uses the first 100 train rows as a placeholder test split,
  and `--num-shards` above 1 is refused. The output keeps only `train` and
  `test`: with no `test` split, the input's `validation` split becomes `test`,
  and any other split is dropped with a warning.

`--tokenizer-backend gigatoken` replaces the HF tokenizer with a Rust bulk
encoder, about 6× faster on UltraChat 200K with the Qwen3-0.6B tokenizer. It
checks its IDs against the HF tokenizer at startup and raises on any mismatch.
It ships in the training image; elsewhere, install it from a checkout with
`uv pip install -e '.[gigatoken]'`.

Training configs take the same choice as `tokenizer_backend` (`hf` by default,
`gigatoken` to opt in). Embedding training accepts only `hf`, because
SentenceTransformers does its own tokenization.

## Data for the RL environments

The code-contests environment reads a prepared pool, not a plain dataset:

1. `halo run compact-code-tests` compacts a test corpus into one row per
   problem, capping the tests in each suite.
2. `halo run prepare-code-dataset` builds the pool. With `--push_bands` it also
   pushes one config per rating band: the `<repo>:<band>` the shipped examples
   name in `dataset:`.

Pool format, build flags and bands:
[Code Contests](../agent-docs/training-methods/grpo/environments/code-contests.md#dataset) ↗.

Full schema (multimodal content included), collators and the S3 utilities:
[Dataset Formats](../agent-docs/data/dataset-formats.md) ↗ ·
[Pre-Processing](../agent-docs/data/dataset-preparation.md) ↗ ·
[S3 Utilities](../agent-docs/data/s3-utilities.md) ↗.
