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

A bare `org/name` is read as a Hub ID, so write a local directory as
`./data/corpus` or an absolute path; an `org/name` that also names a local
directory is refused as ambiguous
([Dataset path formats](../agent-docs/data/s3-utilities.md#dataset-path-formats) ↗).

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

The tool tokenizes, optionally packs (`--pack_sequences`), and shards the
corpus, and can push the result to the Hub or S3. Three rules decide whether the
output trains:

- `--max_length` is stored with the data and must equal the training config's
  `max_length`, or the load raises, since rows are never re-truncated.
- `--num_shards` above 1 must be at least your data-parallel size. The default
  `1` trains at any size.
- `--mode text` tokenizes raw text for pre-training; the default `chat` renders
  the conversation through the chat template.

`--tokenizer_backend gigatoken` swaps in a faster Rust encoder whose IDs are
checked against the HF tokenizer at startup; training configs take the same
choice through `tokenizer_backend`. Every flag and the packing strategies:
[Pre-Processing](../agent-docs/data/dataset-preparation.md#parameters) ↗.

## Data for the RL environments

The code-contests environment reads a prepared pool, not a plain dataset:

1. `halo run compact-code-tests` compacts a test corpus into one row per
   problem, capping the tests in each suite.
2. `halo run prepare-code-dataset` builds the pool. With `--push_bands` it also
   pushes one config per rating band: the `<repo>:<band>` the shipped examples
   name in `dataset:`.

Pool format, build flags and bands:
[Code Contests datasets](../agent-docs/training-methods/async-grpo/environments/tasks/code-contests/datasets.md) ↗.

Full schema (multimodal content included), collators and the S3 utilities:
[Dataset Formats](../agent-docs/data/dataset-formats.md) ↗ ·
[S3 Utilities](../agent-docs/data/s3-utilities.md) ↗.
