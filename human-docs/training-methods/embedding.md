# Embedding Models

Fine-tune a text encoder for retrieval, semantic similarity, deduplication or clustering, using
[sentence-transformers](https://sbert.net/) losses under Halo's parallelism. The output loads with
`SentenceTransformer(path)` — the pipeline config (`modules.json`, the per-module directories) is
written alongside the weights.

This is the one method that is not a causal-LM objective, so a few shared config knobs do not apply
(see [What this path does not take](#what-this-path-does-not-take)).

## Data shapes

The collator reads columns **positionally**, not by name: the label is the first of `label`, `labels`,
`score`, `scores` that exists (in that priority), and every other column, in dataset order, is one text input. So the
shape of your table picks the losses available to you.

| Text columns | Label | What you are training | Losses |
| --- | --- | --- | --- |
| 2 (anchor, positive) | none | retrieval, semantic search | `mnrl`, `cached_mnrl` |
| 3 (+ negative) | none | retrieval with hard negatives | `triplet`, `mnrl` |
| 2 | float score | similarity regression | `cosent`, `angle`, `cosine_similarity` |
| 2 | 0/1 label | duplicate detection | `online_contrastive`, `contrastive` |
| 1 | class id | class-structured embeddings | `batch_hard_triplet`, `batch_all_triplet` (with `batch_sampler: group_by_label`) |

## Choosing a loss

`mnrl` (multiple-negatives ranking, the default) is InfoNCE over in-batch negatives: every other
positive in the same micro-batch is a negative for this anchor. That makes
`per_device_train_batch_size` the dominant hyperparameter, and neither gradient accumulation nor
extra GPUs enlarge the pool: the negatives come from the micro-batch on this device. When a larger batch OOMs,
switch to `cached_mnrl`, which caches embeddings across sub-forwards of `cached_mnrl_mini_batch_size` rows and gets
the same pool for less memory.

Use `cosent` or `angle` when your labels are graded similarity scores, `triplet` when you mined hard
negatives, and the contrastive pair for duplicate/not-duplicate data.

`matryoshka_dimensions: [256, 128, 64, 32]` wraps any of them so the embedding stays usable when
truncated to a shorter prefix — useful when you want one model to serve both a cheap first-pass index
and a precise rerank. Passing `matryoshka_weights` without the dimensions raises rather than being
quietly dropped.

## Config

From `examples/embedding/qwen3/embedding-qwen3-4b-nq.yaml`:

```yaml
model_name_or_path: Qwen/Qwen3-Embedding-4B
dataset: sentence-transformers/natural-questions
test_size: 0.01
loss_type: mnrl
pooling_mode: lasttoken
max_length: 512
batch_sampler: no_duplicates
per_device_train_batch_size: 64
dataloader_drop_last: true
learning_rate: 2.0e-05
output_dir: checkpoints/embedding-qwen3-4b-nq
```

`pooling_mode` decides how token states become one vector, and it has to match the backbone:
decoder-based embedders like Qwen3-Embedding want `lasttoken`, most encoder checkpoints want `mean`.
On a plain load, a value differing from the checkpoint's own pooling is applied with a warning, not a
stop, so set it deliberately; under EP/TP the pooling module is built from the config outright.
`batch_sampler: no_duplicates` keeps two copies of the same text out of one batch, where they would
become each other's false negatives. `normalize_embeddings` defaults to on; turning it off against a
checkpoint whose pipeline ends in a `Normalize` module raises rather than silently changing what its
similarity thresholds mean.

`max_length` defaults to 512. Long-context backbones make it tempting to raise, but embedding batches
want to be large — keep it at the length you actually embed.

## Run

```bash
# single GPU or FSDP2 data parallel
halo launch embedding examples/embedding/qwen3/embedding-qwen3-4b-nq.yaml

# MoE backbone with expert parallelism pinned in the config
halo launch embedding examples/embedding/gptoss/embedding-gptoss-20b-gooaq-ep.yaml -n 8
```

Recipes for Qwen3-Embedding, Qwen3.5, GPT-OSS and Gemma 4 ship under `examples/embedding/`. Expert,
tensor and expert-tensor parallelism all work; context parallelism does not, because pooling needs
the whole sequence on one rank. LoRA (`use_peft: true`) is supported on the plain data-parallel path
only and rejected under EP, ETP and TP.

## What this path does not take

sentence-transformers owns tokenization here, so `tokenizer_backend` must stay at its default `hf` —
a `gigatoken` value is refused, not ignored. The same goes for the chat-template and special-token
knobs (`chat_template`, `force_chat_template`, `pad_token`, `bos_token`, `eos_token`,
`added_special_tokens`), `freeze_layers_patterns` / `unfreeze_layers_patterns`, `tools_field` and
`log_decoded_samples`:
this path has no rendering or freeze stage to honor them, so it raises at startup instead of
accepting a flag that would do nothing. Image columns (`images`, `image`, `pixel_values`) are refused
too — embedding training is text-only.

## What to watch

Metrics come from a small no-grad encoding pass on logging steps. `embed/cos_sim` should rise (and
`embed/neg_cos_sim` fall, on a dataset with a third text column); `embed/recall@1` and `embed/mrr` are the ranking
view of the same thing.
Collapse is the failure mode to watch for, and it shows up two ways: `embed/std` trending to zero, or
a high `embed/cos_sim` with poor `embed/recall@1` — the model is mapping everything to nearly the
same vector. `embed/mrr` saturating at 1.0 within a few steps usually means the batch is too small to
be a real ranking task.

## Go deeper

- [Datasets](../data.md) · [Checkpoints & Export](../checkpoints.md) · [Parallelism](../parallelism.md)
- [Embedding reference](../../agent-docs/training-methods/embedding.md) ↗ ·
  [PEFT](../../agent-docs/optimization/peft.md) ↗
