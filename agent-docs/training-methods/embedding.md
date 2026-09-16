# Embedding Training

Fine-tune embedding models with [sentence-transformers](https://sbert.net/) losses under distributed parallelism. `EmbeddingTrainer` (`src/trainers/embedding/trainer.py`) layers the ST losses onto `DistributedTrainerMixin`; `EmbeddingConfig` (`src/configs/embedding_config.py`) extends `SentenceTransformerTrainingArguments`.

EP, TP, EP+TP, pure ETP and EP+ETP run, as do DDP and FSDP2. CP and PP raise at config time; CP because pooling needs the whole sequence. Text only: image columns are refused at dataset load.

## Dataset formats

`SentenceTransformerDataCollator` reads the shape **positionally**: the first column named `label`, `labels`, `score` or `scores` is the label, every remaining column in dataset order is one text input. Names carry no meaning.

| Text columns | Label | Use case | Losses |
|---|---|---|---|
| 2 (`anchor`, `positive`) | — | Semantic search, retrieval | `mnrl`, `cached_mnrl` |
| 3 (+ `negative`) | — | Hard-negative training | `triplet`, `mnrl` |
| 2 | float `score` | Semantic textual similarity | `cosent`, `angle`, `cosine_similarity` |
| 2 | `label` 0/1 | Duplicate detection | `online_contrastive`, `contrastive` |
| 1 | class int | Classification-based embedding | `batch_hard_triplet`, `batch_all_triplet` |

## Loss functions

`loss_type` selects one of ten ST losses: `mnrl` (the default), `cached_mnrl`, `cosent`, `angle`, `cosine_similarity`, `triplet`, `contrastive`, `online_contrastive`, `batch_all_triplet`, `batch_hard_triplet`. `loss_scale` reaches only `mnrl`, `cached_mnrl`, `cosent` and `angle`; it is inert for the rest.

`mnrl` is InfoNCE: each anchor's negatives are the other `batch_size - 1` positives in the same micro-batch, so batch size is the dominant hyperparameter. `gradient_accumulation_steps` does **not** grow the pool, and neither do extra GPUs — the loss is built without ST's `gather_across_devices`. Raise `per_device_train_batch_size`, or switch to `cached_mnrl` when that OOMs: it caches embeddings across sub-forwards of `cached_mnrl_mini_batch_size` rows.

Matryoshka wraps any loss for truncatable embeddings:

```yaml
loss_type: mnrl
matryoshka_dimensions: [256, 128, 64, 32]
# matryoshka_weights: [1.0, 1.0, 1.0, 1.0]   # optional; same length as the dimensions
```

`matryoshka_weights` without `matryoshka_dimensions` raises — the weights reach only `MatryoshkaLoss`, so the run would train the plain loss with them dropped. A length mismatch raises too.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `loss_type` | `mnrl` | Loss function |
| `loss_scale` | `20.0` | Scale / inverse temperature |
| `cached_mnrl_mini_batch_size` | `32` | Rows per gradient-cached sub-forward (`cached_mnrl`) |
| `matryoshka_dimensions` | `None` | Matryoshka truncation dims |
| `matryoshka_weights` | `None` | Per-dimension weights; same length as the dimensions |
| `pooling_mode` | `mean` | `mean`, `cls`, `max`, `lasttoken`, `weightedmean`, `mean_sqrt_len_tokens` |
| `normalize_embeddings` | `true` | L2-normalize output embeddings |
| `max_length` | `512` | Truncation length; `null` or non-positive → the backbone's context window |
| `disable_dropout` | `false` | Disable dropout while training |
| `batch_sampler` | `batch_sampler` | `no_duplicates` / `no_duplicates_hashed` (MNRL — avoids in-batch false negatives), `group_by_label` (batch-triplet losses) |

`pooling_mode`, `normalize_embeddings` and `max_length` describe the pipeline both loading paths train: the EP/TP path builds the `SentenceTransformer` modules from them, the standard path aligns the checkpoint's `modules.json` to them.

A `pooling_mode` differing from the checkpoint's is applied and logged — set it deliberately. `normalize_embeddings: false` against a checkpoint ending in `Normalize` raises rather than silently redefining its similarity scale; a pipeline with no `Pooling` module raises too.

## Launch

```bash
# single GPU or FSDP2 data parallel
python scripts/training/embedding.py examples/embedding/qwen3/embedding-qwen3-4b-nq.yaml

# EP (MoE, 8 GPUs)
torchrun --nproc_per_node=8 scripts/training/embedding.py \
    examples/embedding/gptoss/embedding-gptoss-20b-gooaq-ep.yaml
```

`halo launch embedding <config> --nproc 8` builds the same line. Recipes for Qwen3-Embedding, Qwen3.5, GPT-OSS and Gemma 4 ship under `examples/embedding/`; all four run `loss_type: mnrl`, `pooling_mode: lasttoken`, `max_length: 512`, `batch_sampler: no_duplicates`.

Qwen3-Embedding-4B is a decoder-based embedder: use `pooling_mode: lasttoken`, not `mean`. It is long-context, so keep `max_length` small unless you embed long documents.

Under EP or TP the backbone loads through `PreloadedTransformer` (`src/trainers/embedding/sentence_transformers_compat.py`): the built-in ST `Transformer` reloads from a path and would discard the EP/TP patches. An MoE backbone under neither EP nor ETP takes the plain ST loader, with no EP wrappers.

## PEFT / LoRA

`use_peft: true` injects LoRA into the backbone in place (`inject_adapter_in_model`), on the plain data-parallel / FSDP2 path only — EP, ETP and TP are rejected at construction ([PEFT](../optimization/peft.md#embedding-models)). Quantization, `lora_modules_to_save` and `train_sinks: true` raise at startup; a DoRA merge raises at first save.

## Saving

Saves route through the shared `save_checkpoint` ladder ([Checkpoints](../reference/checkpoints.md)); all ranks run the gather collective and only the writer retains the state dict, with in-place LoRA folded in (`<m>.weight = base + scaling · B @ A`) first. The ST pipeline config (`modules.json`, `sentence_bert_config.json`, the per-module directories) is written alongside, so the output loads with `SentenceTransformer(path)`.

## What to watch

Metrics come from a separate `torch.no_grad()` encoding pass on logging steps, capped at 256 samples per padded text group; evaluation runs it every batch under an `eval_` prefix (`eval_embed/norm`, …).

| Metric | When | Reading |
|---|---|---|
| `embed/norm`, `embed/std` | always | Mean L2 norm (~1.0 when normalized) and per-dim std; std → 0 is collapse |
| `embed/cos_sim`, `embed/mrr`, `embed/recall@{1,3,10}` | ≥ 2 text groups | Anchor-positive similarity and in-batch ranking; should rise |
| `embed/neg_cos_sim`, `embed/triplet_margin` | ≥ 3 text groups | Negative similarity should fall |

High `embed/cos_sim` with low `embed/recall@1` is also collapse; `embed/mrr` saturating at 1.0 early means the batch is too small.

## Testing a setup

```bash
pytest tests/cpu/trainers tests/cpu/config/test_embedding_pipeline_alignment.py -m cpu
torchrun --nproc_per_node=2 tests/gpu/trainers/other/test_embedding.py
```

The GPU suite trains MNRL and CoSENT plus LoRA and round-trips the gathered save and the LoRA merge on `sentence-transformers/paraphrase-MiniLM-L3-v2`.

## Related pages

- [Expert Parallelism](../parallelism/expert-parallelism.md) · [Tensor Parallelism](../parallelism/tensor-parallelism.md)
- [Dataset Formats](../data/dataset-formats.md) · [PEFT](../optimization/peft.md#embedding-models) · [Scripts Reference](../reference/scripts-reference.md)
