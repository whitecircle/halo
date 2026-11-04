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
selected. Every schema, flag and rule below lives on its owner page; read it there before acting, and
use placeholder URIs (`s3://<bucket>/<key>`) — never hardcode an internal bucket or dataset key.

## 1. Shape the dataset for the method

Required columns per method: [Dataset Formats](../../agent-docs/data/dataset-formats.md#required-columns-by-method)
(it routes Async GRPO with Environments and distillation to their method pages). Embedding reads its
columns positionally ([Embedding](../../agent-docs/training-methods/embedding.md#dataset-formats)).
The column-name fields (`conversation_field`, `prompt_field`, `answer_field`, `text_field`) default
differently per method: check the method's row in the
[configuration reference](../../agent-docs/reference/configuration-reference.md), never assume one.

## 2. Decide on offline pre-processing (SFT only)

`scripts/before_training/prepare_dataset.py` tokenizes, optionally packs and optionally shards once,
so large SFT corpora skip that work at every launch and multi-node ranks load disjoint shards. Pay
it when the corpus is large or the run is multi-node; otherwise train from the raw dataset. Flags,
output layout and the preprocessed-vs-presharded distinction:
[Dataset Pre-Processing](../../agent-docs/data/dataset-preparation.md); the tool's CLI row:
[Scripts Reference](../../agent-docs/reference/scripts-reference.md#preparation-scripts); the shard-count rule
(`--num_shards` at least the data-parallel size) and how ranks split shards:
[Data Loading](../../agent-docs/parallelism/data-loading.md#pre-processed-sharded-datasets).

## 3. Load it and let the collator follow

- Sources (`s3://`, Hub `org/name[:config][@split]`, local) and S3 caching:
  [S3 Utilities](../../agent-docs/data/s3-utilities.md#dataset-path-formats); a Hub-shaped path that also
  names a local directory is refused, so write `./org/name` for the directory. Shared vs per-node
  filesystems: [Filesystem Handling](../../agent-docs/data/filesystem-handling.md).
- The collator is picked from the YAML flags (`packing`, `padding_free`, `train_on_completions_only`,
  CP); the selection table and the combinations that raise are in
  [Collators](../../agent-docs/data/collators.md). SFT rows that share a long prefix can store it once
  with `tree_packing` ([Tree Packing](../../agent-docs/optimization/tree-packing.md)).
- A data-loading error (wrong schema, embedding OOB at step 0, a re-shard, `num_shards` below DP):
  [Troubleshooting](../../agent-docs/reference/troubleshooting.md).

## Sources of truth

The pages above document the intended behavior; the code wins when they disagree. Find the module
behind a behavior in the [module map](../../agent-docs/reference/architecture.md) and read it before
you assert or act.
