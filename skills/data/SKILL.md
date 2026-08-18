---
name: data
description: >-
  Get a dataset into the right shape for a Halo training method, and
  pre-process it (tokenize / pack / shard) when worth it. USER-INVOKED ONLY. Use
  when the user asks "what format does SFT/DPO/GRPO/reward/embedding expect",
  "prepare/tokenize/pack/shard my dataset", "load from S3/HF Hub", "which
  collator", "padding-free / completions-only", or hits a data-loading error
  (wrong schema, embedding OOB at step 0, presharded re-shard, num_shards < DP).
disable-model-invocation: true
allowed-tools:
  - Read
  - Bash
  - Grep
  - Glob
---

# data — dataset formats, prep, loading & collators

Pick the format for the method, decide whether to pre-process offline, and let the right collator get
selected. Per-method schema detail, the `prepare_dataset.py` flags, and the footguns are in
**`reference.md`** — read it before acting. Authoritative docs: `agent-docs/data/dataset-formats.md`,
`agent-docs/data/dataset-preparation.md`, `agent-docs/data/collators.md`, `agent-docs/data/s3-utilities.md`,
`agent-docs/data/filesystem-handling.md`.

## Format per method (messages are ChatML `{"role","content"}`)

| Method | Required columns | Shapes |
|---|---|---|
| SFT | `prompt` | `List[Dict]` (field name = `conversation_field`) |
| DPO / SMPO | `prompt`, `chosen`, `rejected` | all `List[Dict]` |
| Reward | `chosen`, `rejected` (+ optional `prompt`) | all `List[Dict]`; implicit-prompt sets carry no `prompt` |
| Offline GRPO | `prompt`, `completions`, `rewards` | `List[Dict]`, `List[List[Dict]]`, `List[float]` (`len` match) |
| Environmental GRPO | `prompt`, `answer` | both `str` (NOT ChatML) |
| Classification | `prompt`, `label` | `List[Dict]`, `str` / `List[str]` (multi-label; labels sorted, `-1` filtered) |
| Distillation | conversation field | `List[Dict]` (default field `messages`) |
| Embedding | text columns (+ optional `label`/`score`) | collator auto-detects pairs / triplets / scored / labeled |

The default `conversation_field` is **method-dependent** — `"prompt"` (sft.py and
`prepare_dataset.py --conversation-field`), `"messages"` (distillation). Don't assume one value;
check the script/config. Every config declares `max_length`
natively; the YAML parser migrates no spelling, so TRL's retired `max_seq_length` raises the
unknown-key error.

## Offline pre-processing (`scripts/before_training/prepare_dataset.py`) — SFT only

Tokenize → optional pack → optional shard, written to `s3://` / `hf://org/name` / local. Worth it for
large SFT corpora (pay tokenization+packing once; multi-node ranks load disjoint shards). The output
carries `metadata.json` (`preprocessed: true`); a training config just sets `dataset:` to that path —
**no flag needed**, `load_datasets_auto()` auto-detects and skips tokenization. Sharded output also
carries `shard_index.json`; the trainer computes `dataset_presharded` so the DataLoader does **not**
re-shard. Key flags (read the argparse for the rest): `--mode {chat,text}`, `--pack-sequences`
`--packing-strategy {bfd,bfd_split,wrapped}` (text only), `--num-shards N` (**must be ≥ data_parallel_size**),
`--vlm`. Non-SFT methods are not supported by this script. Full flag list + the two distinct mechanisms
(preprocessed vs presharded) in `reference.md`.

## Loading & collators (selected for you)

- Sources (`src/data/sources/loading.py`): `s3://bucket/key`, `org/name` (HF Hub, `:config`
  suffix), or a local `load_from_disk` path. Bare strings are local/HF — **not** auto-prefixed to any
  bucket.
- Multi-node: `ShardedDatasetLoader` gives each DP rank its shards; `fs_aware_main_first(tag)` /
  `DIST_INPUT_SHARED_FILESYSTEM` (falling back to the `DIST_SHARED_FILESYSTEM` umbrella, default
  `"1"`) coordinate shared-NFS vs per-node-local downloads.
- Collator routing (`src/data/collators/factory.py`, `select_data_collator`): picks
  completions-only / packing / padding-free (`cu_seq_lens`) from your YAML flags. Mutually-exclusive
  guards raise: packing+padding_free, CP+padding_free; `train_on_completions_only` needs
  `assistant_message_template`. Selection matrix in `reference.md`.

## Sources of truth

The `agent-docs/data/` pages document the intended behavior — check them first. The code is the **ultimate**
authority: when a doc, this skill, `reference.md`, or memory disagrees — or you are unsure — read the
actual file (`src/data/sources/loading.py`, `src/data/pipeline/processing.py`,
`src/data/sources/sharded_dataset.py`, `src/data/collators/factory.py`, `src/data/collators/`,
`scripts/before_training/prepare_dataset.py`) before you assert or act. (`CLAUDE.md`: docs-first, but the
code wins.)
