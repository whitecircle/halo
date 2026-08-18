# Embedding Training

Fine-tune embedding models with [sentence-transformers](https://sbert.net/) losses under distributed parallelism. `EmbeddingTrainer` (`src/trainers/embedding/trainer.py`) layers the ST losses onto `DistributedTrainerMixin`; `EmbeddingConfig` (`src/configs/embedding_config.py`) extends `SentenceTransformerTrainingArguments`, satisfying both the mixin and the ST trainer.

## Quick start

```bash
# Single GPU or FSDP data-parallel
python scripts/training/embedding.py examples/embedding/qwen3/embedding-qwen3-4b-nq.yaml

# EP (MoE, 8 GPUs)
torchrun --nproc_per_node=8 scripts/training/embedding.py \
    examples/embedding/gptoss/embedding-gptoss-20b-gooaq-ep.yaml --expert_parallel_size=8
```

Configs ship for Qwen3-Embedding, Qwen3.5, GPT-OSS, and Gemma4 under `examples/embedding/`.

## Dataset formats

`SentenceTransformerDataCollator` detects the format by column names: columns named `label`, `labels`, `score`, or `scores` are labels, everything else is a text input. See [Dataset Formats](../data/dataset-formats.md).

| Format | Columns | Use case | Compatible losses |
|--------|---------|----------|-------------------|
| Positive pairs | 2 text (e.g. `anchor`, `positive`) | Semantic search, retrieval | `mnrl`, `cached_mnrl` |
| Triplets | 3 text (`anchor`, `positive`, `negative`) | Hard-negative training | `triplet`, `mnrl` |
| Scored pairs | 2 text + `score` (float) | Semantic textual similarity | `cosent`, `angle`, `cosine_similarity` |
| Binary pairs | 2 text + `label` (0 or 1) | Duplicate detection | `contrastive`, `online_contrastive` |
| Labeled texts | 1 text + `label` (class int) | Classification-based embedding | `batch_all_triplet`, `batch_hard_triplet` |

```json
{"anchor": "What is the capital of France?", "positive": "Paris is the capital of France."}
```

## Loss functions

`loss_type` selects one of ten ST losses: `mnrl` (MultipleNegativesRankingLoss, the default), `cached_mnrl`, `cosent`, `angle`, `cosine_similarity`, `triplet`, `contrastive`, `online_contrastive`, `batch_all_triplet`, `batch_hard_triplet`. `loss_scale` (default 20.0) is the scale / inverse temperature for `mnrl`, `cached_mnrl`, `cosent`, and `angle`.

Choosing: retrieval and search → `mnrl`; STS → `cosent` or `angle`; offline-mined hard negatives → `triplet`; duplicate detection → `online_contrastive` (auto-selects hard examples, beats plain `contrastive`); classification-based → `batch_hard_triplet` with `batch_sampler: group_by_label`, which the batch-triplet losses require.

### Batch size for contrastive losses

`mnrl` is InfoNCE: each anchor's negatives are the other `batch_size - 1` positives in the same micro-batch, so batch size is the dominant hyperparameter. `gradient_accumulation_steps` does **not** grow the negative pool, and neither do extra GPUs — the loss is built without ST's `gather_across_devices`, so the pool stays per-device.

To scale negatives, raise `per_device_train_batch_size`, or switch to `cached_mnrl` when `mnrl` OOMs — it caches embeddings across sub-forwards of `cached_mnrl_mini_batch_size` rows (default 32), decoupling the negative pool from activation memory without changing the loss. Rough targets: 256–512 for semantic search, 128 for duplicate detection, 64 for STS.

### Matryoshka embeddings

Wrap any loss for truncatable embeddings:

```yaml
loss_type: mnrl
matryoshka_dimensions: [256, 128, 64, 32]
# matryoshka_weights: [1.0, 1.0, 1.0, 1.0]  # optional; must match the dimensions length
```

`matryoshka_weights` without `matryoshka_dimensions` raises at config time: the weights reach only
`MatryoshkaLoss`, which is built from the dimensions, so the run would train the plain loss with the
weights dropped. A length mismatch between the two raises as well.

## Training metrics

On logging steps the trainer runs a separate `torch.no_grad()` encoding pass (compatible with the cached losses), capped at 256 samples per text group, and logs alongside `loss` / `learning_rate` / `grad_norm`. Evaluation runs the same pass on every batch and reports the batch means under the `eval_` prefix.

Always emitted: `embed/norm` (mean L2 norm of anchor embeddings, ~1.0 when normalized) and `embed/std` (mean per-dim std). Two or more text groups add `embed/cos_sim` (anchor-positive, should rise), `embed/mrr` and `embed/recall@{1,3,10}` (the k must fit the candidate count). Three groups add `embed/neg_cos_sim` (should fall) and `embed/triplet_margin`. In-batch ranking treats `positive[i]` as the correct candidate for `anchor[i]`, mirroring MNRL.

Reading them: `embed/std` → 0 is representational collapse (lower the LR or regularize); high `embed/cos_sim` with low `embed/recall@1` is also collapse; `embed/mrr` saturating at 1.0 early means the batch is too small.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loss_type` | `mnrl` | Loss function |
| `loss_scale` | `20.0` | Scale / inverse temperature |
| `cached_mnrl_mini_batch_size` | `32` | Rows per gradient-cached sub-forward (`cached_mnrl` only) |
| `matryoshka_dimensions` | `None` | Matryoshka truncation dims |
| `matryoshka_weights` | `None` | Per-dimension weights; requires `matryoshka_dimensions` of the same length |
| `pooling_mode` | `mean` | `mean`, `cls`, `max`, `lasttoken`, `weightedmean`, `mean_sqrt_len_tokens` |
| `normalize_embeddings` | `true` | L2-normalize output embeddings |
| `max_length` | `512` | Truncation length; `null` → the backbone's context window |
| `disable_dropout` | `false` | Disable dropout during training |
| `batch_sampler` | `batch_sampler` | `batch_sampler` (random), `no_duplicates` / `no_duplicates_hashed` (MNRL — avoids in-batch duplicate false negatives), `group_by_label` (batch-triplet losses) |

`pooling_mode`, `normalize_embeddings` and `max_length` describe the pipeline the run trains on both loading paths: the EP/TP path builds the `SentenceTransformer` modules from them, and the standard data-parallel path (which loads the checkpoint's own `modules.json`) is aligned to them after loading, with `max_length` installed as ST's `max_seq_length` over whatever the checkpoint ships.

A `pooling_mode` differing from the checkpoint's is applied and logged, so set it to the checkpoint's own strategy unless you intend to change it. `normalize_embeddings: false` against a checkpoint whose pipeline carries a `Normalize` module raises rather than silently redefining its similarity scale.

## PEFT / LoRA

`use_peft: true` injects LoRA into the backbone in place (`inject_adapter_in_model`), so `lora_task_type` is not consulted on this path. It runs on the **plain data-parallel / FSDP2 path only**: the trainer rejects it under any EP or ETP mode (the EP save path gathers the expert layout with no adapter-merge step, so the checkpoint would reload adapter keys as random base weights), and the mixin's LoRA gate rejects it under TP.

Three more raises, each on something that would otherwise be silent: quantization (the ST loader never applies a `quantization_config`, so `--load_in_4bit` / `--load_in_8bit` would be ignored); merging a DoRA adapter on save (`NotImplementedError` — its magnitude reparam is not a linear merge); and `lora_modules_to_save` (adapters are injected in place, so the trainable copies are re-frozen by the adapter-only freeze and the wrapper renames the base tensor to `<mod>.original_module.*`, leaving the saved module without the plain `<mod>.weight` it reloads from). See [PEFT (LoRA)](../optimization/peft.md#embedding-models).

## Parallelism

EP, TP, EP+TP, pure ETP (`ep_size=1`), and EP+ETP are supported. **CP is not** (pooling needs the full sequence), and neither is PP. See [Trainer Compatibility](../reference/trainer-architecture.md#trainer-compatibility).

In EP/TP mode the backbone loads via `load_script_model(model_class=AutoModel)` wrapped in `PreloadedTransformer` (`src/trainers/embedding/sentence_transformers_compat.py`), which implements the ST `Transformer` interface — the built-in ST `Transformer` reloads from a string path and would discard the EP/TP patches. The resulting `PreloadedTransformer → Pooling → Normalize` pipeline works with every ST loss.

### Saving

Under EP, TP, or mixin-managed FSDP2 the trainer saves through the shared `save_checkpoint` ladder ([Checkpoints](../reference/checkpoints.md)), with the checkpoint context re-pointed from the `SentenceTransformer` `nn.Sequential` at the `auto_model` backbone. Embedding exports therefore get the same treatment as every other trainer: the configured `save_max_shard_size`, the module-tree-derived save-dtype cast (`save_dtype_caster`), hub expert layouts for MoE backbones, and the `.bin` fallback when safetensors sharding fails.

All ranks run the gather collective; only the writer rank retains the gathered state dict. The ST pipeline config (`modules.json` and the per-module configs) is written alongside it, so the output loads with `SentenceTransformer(path)`.

In-place-injected LoRA is the one case that bypasses the ladder: the model is not a `PeftModel`, so the adapters are folded into the gathered state dict (`<m>.weight = base + scaling · B @ A`) before it goes to the same shared writer.

### Vision-language

Not supported. Standard sentence-transformers mode is the only feasible path; EP/TP would need a multimodal `PreloadedTransformer`, which is not implemented.

## Decoder-based embedding models

Qwen3-Embedding-4B is a decoder-based embedding model on the Qwen3 architecture (`examples/embedding/qwen3/embedding-qwen3-4b-nq.yaml`). It differs from encoder models (BERT, BGE) in one load-bearing way: use `pooling_mode: lasttoken`, not `mean`. It is also instruction-aware (task prefixes via ST `prompts`) and long-context, so keep `max_length` small unless you really embed long documents.

## Tests

```bash
torchrun --nproc_per_node=2 tests/gpu/trainers/other/test_embedding.py
```

Covers import/alias, CP rejection, MNRL on positive pairs, CoSENT on scored pairs, LoRA training, the FSDP2 gathered-save roundtrip, and the LoRA merge-on-save roundtrip, on `sentence-transformers/paraphrase-MiniLM-L3-v2`.

## Related pages

- [Expert Parallelism](../parallelism/expert-parallelism.md) · [Tensor Parallelism](../parallelism/tensor-parallelism.md)
- [Dataset Formats](../data/dataset-formats.md) · [Scripts Reference](../reference/scripts-reference.md)
