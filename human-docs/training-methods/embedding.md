# Embedding Models

Fine-tune a text encoder for retrieval, semantic similarity, deduplication or clustering, using
[sentence-transformers](https://sbert.net/) losses under Halo's parallelism. The output loads with
`SentenceTransformer(path)`: the pipeline config (`modules.json` and the per-module directories) is
written alongside the weights.

sentence-transformers owns tokenization and the pipeline around the backbone here, so some shared config
knobs do not apply ([What this path does not take](#what-this-path-does-not-take)).

## Data shapes

The collator reads columns **by position**, not by name. The label is the first of `label`, `labels`,
`score`, `scores` that exists, in that order. Every other column, in dataset order, is one text input.
The shape of your table therefore decides which losses you can use.

| Text columns | Label | What you are training | Losses |
| --- | --- | --- | --- |
| 2 (anchor, positive) | none | retrieval, semantic search | `mnrl`, `cached_mnrl` |
| 3 (+ negative) | none | retrieval with hard negatives | `triplet`, `mnrl` |
| 2 | float score | similarity regression | `cosent`, `angle`, `cosine_similarity` |
| 2 | 0/1 label | duplicate detection | `online_contrastive`, `contrastive` |
| 1 | class id | class-structured embeddings | `batch_hard_triplet`, `batch_all_triplet` (with `batch_sampler: group_by_label`) |

## Choosing a loss

`mnrl` (multiple-negatives ranking, the default) is InfoNCE over in-batch negatives: every other positive
in the same micro-batch is a negative for this anchor. That makes `per_device_train_batch_size` the
dominant hyperparameter. Gradient accumulation and extra GPUs do not enlarge the pool, because the
negatives come from the micro-batch on one device. When a larger batch runs out of memory, switch to
`cached_mnrl`. It caches embeddings across sub-forwards of `cached_mnrl_mini_batch_size` rows and gets
the same pool for less memory.

Use `cosent` or `angle` for graded similarity scores, `triplet` when you mined hard negatives, and the
contrastive pair for duplicate/not-duplicate data.

`matryoshka_dimensions: [256, 128, 64, 32]` wraps any of them so the embedding stays usable when cut to a
shorter prefix. One model can then serve both a cheap first-pass index and a precise rerank. Passing
`matryoshka_weights` without the dimensions raises.

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

- `pooling_mode` decides how token states become one vector, and it must match the backbone.
  Decoder-based embedders like Qwen3-Embedding want `lasttoken`; most encoder checkpoints want `mean`.
  On a plain load, a value that differs from the checkpoint's own pooling applies with a warning, so set
  it deliberately. Under EP/TP the pooling module is built from the config outright.
- `batch_sampler: no_duplicates` keeps two copies of the same text out of one batch, where they would
  become each other's false negatives.
- `normalize_embeddings` defaults to on. Turning it off for a checkpoint whose pipeline ends in a
  `Normalize` module raises, since it would change what the model's similarity thresholds mean.
- `max_length` defaults to 512. Embedding batches want to be large, so keep it at the length you
  actually embed, even on a long-context backbone.

## Run

```bash
# single GPU; add -n 8 for FSDP2 data parallel
halo launch embedding examples/embedding/qwen3/embedding-qwen3-4b-nq.yaml

# MoE backbone with expert parallelism pinned in the config
halo launch embedding examples/embedding/gptoss/embedding-gptoss-20b-gooaq-ep.yaml -n 8
```

Recipes for Qwen3-Embedding, Qwen3.5, GPT-OSS and Gemma 4 ship under `examples/embedding/`. Expert,
tensor and expert-tensor parallelism all work. Context parallelism does not, because pooling needs the
whole sequence on one rank.

Limits by mode:

- **TP, ETP or a pre-sharded dataset** batch through Halo's own loader, which builds plain batches. They
  refuse any other `batch_sampler` (`no_duplicates`, `no_duplicates_hashed`, `group_by_label`), so set
  `batch_sampler: batch_sampler` (`--batch_sampler=batch_sampler` on the command line).
- **Multi-GPU data parallelism and pure EP** over a map-style dataset that is not pre-sharded need
  `dataloader_drop_last: true`, which sentence-transformers sets on a multi-process launch. A `false`
  that reaches the trainer is refused at startup, since a kept remainder would give some ranks one more
  step than the others and hang the run.
- **A pipeline with weights after the backbone** that train or that FSDP2 would shard (a `Dense` head)
  runs on one GPU or under DDP (`accelerate launch`) only. FSDP2, TP and EP refuse it at startup.
- **LoRA** (`use_peft: true`) runs on the plain data-parallel path only; EP, ETP and TP reject it. Its
  targets may include the input embedding (`embed_tokens`), and DoRA applies. Saves fold the adapters
  into the weights, so the output loads as a plain `SentenceTransformer`. Training checkpoints also keep
  the unfolded adapters, which `resume_from_checkpoint` restores onto the base.

## What this path does not take

Because sentence-transformers owns tokenization, `tokenizer_backend` must stay at its default `hf`; a
`gigatoken` value raises. These knobs also raise at startup, since this path has no rendering or freeze
stage to honor them:

- the chat-template and special-token knobs: `chat_template`, `force_chat_template`, `pad_token`,
  `bos_token`, `eos_token`, `added_special_tokens`;
- `freeze_layers_patterns` / `unfreeze_layers_patterns`;
- `tools_field`, `log_decoded_samples` and `text_only_model`.

Image columns (`images`, `image`, `pixel_values`) are refused too: embedding training is text-only.

## What to watch

Metrics come from a small no-grad encoding pass on logging steps.

- `embed/cos_sim` should rise, and `embed/neg_cos_sim` fall on a dataset with a third text column.
  `embed/recall@1` and `embed/mrr` are the ranking view of the same thing.
- Collapse shows up two ways: `embed/std` trending to zero, or a high `embed/cos_sim` with poor
  `embed/recall@1`. The model is mapping everything to nearly the same vector.
- `embed/mrr` saturating at 1.0 within a few steps usually means the batch is too small to be a real
  ranking task.

## Go deeper

- [Datasets](../data.md) · [Checkpoints & Export](../checkpoints.md) · [Parallelism](../parallelism.md)
- [Embedding reference](../../agent-docs/training-methods/embedding.md) ↗ ·
  [PEFT](../../agent-docs/optimization/peft.md) ↗
