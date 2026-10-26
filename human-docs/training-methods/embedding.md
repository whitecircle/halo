# Embedding Models

Fine-tune a text encoder for retrieval, semantic similarity, deduplication or clustering, using
[sentence-transformers](https://sbert.net/) losses under Halo's parallelism. The output loads with
`SentenceTransformer(path)`.

## Data

The collator reads columns **by position**, not by name: the label is the first of `label`, `labels`,
`score`, `scores` that exists, and every other column is one text input. The table's shape picks the
loss: anchor-positive pairs train `mnrl` (the default) or `cached_mnrl`, a third negative column adds
`triplet`, a float score trains `cosent` or `angle`, and a 0/1 label the contrastive losses
([shapes and losses](../../agent-docs/training-methods/embedding.md#dataset-formats) ↗). Embedding
training is text-only.

## Run

```bash
halo launch embedding examples/embedding/qwen3/embedding-qwen3-4b-nq.yaml -n 8
```

Recipes for Qwen3-Embedding, Qwen3.5, GPT-OSS and Gemma 4 ship under `examples/embedding/`.

- `mnrl` takes its negatives from the other rows of the same micro-batch, so
  `per_device_train_batch_size` is the dominant hyperparameter; gradient accumulation and extra GPUs do
  not enlarge the pool. Switch to `cached_mnrl` when a larger batch runs out of memory.
- `pooling_mode` must match the backbone: `lasttoken` for decoder-based embedders like
  Qwen3-Embedding, `mean` for most encoders.
- `batch_sampler: no_duplicates` keeps two copies of one text out of a batch, where they would become
  each other's false negatives. Tensor and expert-tensor parallelism take only the plain
  `batch_sampler`.

sentence-transformers owns tokenization and the pipeline around the backbone, so the chat-template and
special-token knobs, the freeze patterns, `tools_field`, `log_decoded_samples`, `text_only_model` and
`tokenizer_backend: gigatoken` raise at startup. Context and pipeline parallelism are refused, and LoRA
runs on plain data parallelism only.

## What to watch

The `embed/*` metrics read the sentence embeddings of the loss forward, on logging steps and on every
eval batch.

- `embed/cos_sim` should rise, and `embed/neg_cos_sim` fall on a dataset with a third text column.
  `embed/recall@1` and `embed/mrr` are the ranking view of the same thing.
- Collapse shows up two ways: `embed/std` trending to zero, or a high `embed/cos_sim` with poor
  `embed/recall@1`.
- `embed/mrr` saturating at 1.0 within a few steps usually means the batch is too small to be a real
  ranking task.

Every knob, the save and resume rules and the limits per parallelism mode:
[Embedding reference](../../agent-docs/training-methods/embedding.md) ↗.
