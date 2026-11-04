# Embedding Training

Fine-tune embedding models with [sentence-transformers](https://sbert.net/) losses under distributed parallelism. `EmbeddingTrainer` (`halo/trainers/embedding/trainer.py`) layers the ST losses onto `DistributedTrainerMixin`; `EmbeddingConfig` (`halo/configs/embedding.py`) extends `SentenceTransformerTrainingArguments`.

EP, TP, EP+TP, pure ETP and EP+ETP run, as do DDP and FSDP2. CP and PP raise at config time; CP because pooling needs the whole sequence. Text only: image columns are refused at dataset load.

## Dataset formats

`SentenceTransformerDataCollator` reads the shape **positionally**: the label is the first of `label`, `labels`, `score`, `scores` the dataset has, in that priority order; every remaining column in dataset order is one text input. Names carry no meaning.

| Text columns | Label | Use case | Losses |
|---|---|---|---|
| 2 (`anchor`, `positive`) | — | Semantic search, retrieval | `mnrl`, `cached_mnrl` |
| 3 (+ `negative`) | — | Hard-negative training | `triplet`, `mnrl` |
| 2 | float `score` | Semantic textual similarity | `cosent`, `angle`, `cosine_similarity` |
| 2 | `label` 0/1 | Duplicate detection | `online_contrastive`, `contrastive` |
| 1 | class int | Classification-based embedding | `batch_hard_triplet`, `batch_all_triplet` |

## Loss functions

`loss_type` selects one of ten ST losses: `mnrl` (the default), `cached_mnrl`, `cosent`, `angle`, `cosine_similarity`, `triplet`, `contrastive`, `online_contrastive`, `batch_all_triplet`, `batch_hard_triplet`. `loss_scale` reaches only `mnrl`, `cached_mnrl`, `cosent` and `angle`, and the rest refuse it.

`mnrl` is InfoNCE: each anchor's negatives are the other `batch_size - 1` positives in the same micro-batch, so batch size is the dominant hyperparameter. `gradient_accumulation_steps` does **not** grow the pool, and neither do extra GPUs — the loss is built without ST's `gather_across_devices`. Raise `per_device_train_batch_size`, or switch to `cached_mnrl` when that OOMs: it caches embeddings across sub-forwards of `cached_mnrl_mini_batch_size` rows.

Matryoshka wraps any loss for truncatable embeddings:

```yaml
loss_type: mnrl
matryoshka_dimensions: [256, 128, 64, 32]
# matryoshka_weights: [1.0, 1.0, 1.0, 1.0]   # optional; same length as the dimensions
```

`matryoshka_weights` without `matryoshka_dimensions` raises — the weights reach only `MatryoshkaLoss`, so the run would train the plain loss with them dropped. A length mismatch raises too.

## Configuration

Types and defaults: [EmbeddingConfig](../reference/configuration-reference.md#embeddingconfig).

`max_length` is installed as the pipeline's `max_seq_length` and passed by sentence-transformers' collator as every text column's; `null` or non-positive resolves to the backbone's context window in the entry script (a hand-built `EmbeddingTrainer` refuses a non-positive one).

`batch_sampler: no_duplicates` or `no_duplicates_hashed` avoids in-batch false negatives under MNRL, and `group_by_label` serves the batch-triplet losses. All three are refused under TP, ETP or a pre-sharded dataset: those runs batch through the toolkit's DP-sharded loader, which builds plain batches.

sentence-transformers sets `dataloader_drop_last` to `true`, with a warning, on any multi-process launch, a YAML `false` included. A CLI `--dataloader_drop_last=false` survives that and is refused on a multi-process plain DP or pure EP run over a map-style dataset that is not pre-sharded, right after the datasets load: the sentence-transformers loader shards its batch sampler with `even_batches` off, so a kept remainder gives some ranks one step more and the run hangs in it. The eval loader then drops its last incomplete round too. On a multi-process plain DP or pure EP run, an iterable dataset that is not pre-sharded runs only with `accelerator_config.dispatch_batches: false` and either `split_batches: true` or at most one file shard per process, else it is refused: accelerate's default dispatches an iterable by concatenating collated batches, which the collator's string modality fields cannot take, and more shards than processes are split by shard, unevenly whatever `drop_last` says.

`pooling_mode`, `normalize_embeddings` and `max_length` describe the pipeline both loading paths train: each aligns the checkpoint's own pipeline to them (the parallel loader's path rebuilds its `Pooling` and `Normalize` first), and over a bare backbone the parallel loader's path builds those modules from them.

A `pooling_mode` differing from the checkpoint's is applied and logged — set it deliberately. `normalize_embeddings: false` against a checkpoint ending in `Normalize` raises rather than silently redefining its similarity scale; a pipeline with no `Pooling` module raises too.

## Launch

```bash
# single GPU
python scripts/training/embedding.py examples/embedding/qwen3/embedding-qwen3-4b-nq.yaml

# FSDP2 data parallel (8 GPUs)
torchrun --nproc_per_node=8 scripts/training/embedding.py examples/embedding/qwen3/embedding-qwen3-4b-nq.yaml

# EP (MoE, 8 GPUs)
torchrun --nproc_per_node=8 scripts/training/embedding.py \
    examples/embedding/gptoss/embedding-gptoss-20b-gooaq-ep.yaml

# TP (2 GPUs): the DP-sharded loader takes only the plain batch sampler
torchrun --nproc_per_node=2 scripts/training/embedding.py \
    examples/embedding/qwen3/embedding-qwen3-4b-nq.yaml \
    --tensor_parallel_size=2 --batch_sampler=batch_sampler
```

`halo launch embedding <config> --nproc 8` builds the same line. Recipes for Qwen3-Embedding, Qwen3.5, GPT-OSS and Gemma 4 ship under `examples/embedding/`; all four run `loss_type: mnrl`, `pooling_mode: lasttoken`, `max_length: 512`, `batch_sampler: no_duplicates` (override it to `batch_sampler` under TP or ETP).

Qwen3-Embedding-4B is a decoder-based embedder: use `pooling_mode: lasttoken`, not `mean`. It is long-context, so keep `max_length` small unless you embed long documents.

Under EP or TP, and on a MoE whose family has an EP layer class wherever `use_grouped_gemm` (the default) asks for the grouped-GEMM EP layers at `expert_parallel_size: 1`, the backbone loads through the parallel loader, as every other trainer's does, into `PreloadedTransformer` (`halo/trainers/embedding/preloaded_transformer.py`): the built-in ST `Transformer` reloads from a path and would discard the EP/TP patches, and sentence-transformers' own load builds the stock experts. That pipeline rebuilds the rest of a sentence-transformers checkpoint from its files before any weight loads: each `Pooling` from its saved config, a `Normalize`, the backbone module's `do_lower_case`, and the `prompts`, `default_prompt_name` and `similarity_fn_name`, aligned to the config as sentence-transformers' own load is (the configured pooling mode wins with a warning, dropping the checkpoint's `Normalize` is refused). It refuses a checkpoint whose pipeline holds any other module (a `Dense` head, say) rather than train and export without it; a bare backbone takes the configured pooling and normalization. Under `accelerate launch` its EP layers are refused, as for every trainer (set `use_grouped_gemm: false`). The load takes the loader's attention pick, except for a softcapped backbone (Gemma 2) at more than one row per batch, training or evaluation, which takes FA2 on Blackwell: sentence-transformers pads each text column's batch, and a padded batch reaches FA4's varlen kernels, which a softcap stays off. A pinned `flash_attention_4` the load keeps there is refused before any weight loads ([Flash Attention](../optimization/flash-attention.md#model-specific-handling)). Every other backbone (dense, a MoE no EP layer class wraps such as Mixtral, or any MoE at `use_grouped_gemm: false`) takes the plain ST loader, with its stock experts; it applies the family patches the parallel loader does, the backend-independent ones (Zaya's checkpointing refusal) even when no backend is pinned. `EmbeddingTrainer` calls `patch_sentence_transformer_gradient_checkpointing()` from the same module at construction: sentence-transformers' own `gradient_checkpointing_enable` lacks the `every_n_layers` and `offload` keywords transformers' trainer passes, so every run with gradient checkpointing on would fail at that call. Sentence-transformers' own load builds no EP layer, so `fp32_router` and `fp32_experts` are refused there before any weight loads.

## PEFT / LoRA

`use_peft: true` injects LoRA in place (`inject_adapter_in_model`) into the transformer backbone only (`model[0].auto_model`) and freezes every other parameter; `lora_task_type` is never read. It runs on one GPU, DDP or FSDP2 data parallelism. EP, pure ETP included, is refused at trainer construction: the save folds the adapters over the backbone's own parameter walk, which under EP holds each rank's expert shards under their local names, so the checkpoint would load nowhere. So are the grouped-GEMM EP layers at `expert_parallel_size: 1`, whose walk holds the layers' own expert layout: LoRA trains such a MoE at `use_grouped_gemm: false`, on its stock experts. The script refuses both before the model loads. TP is refused by the trainer mixin's LoRA gate ([PEFT — Parallelism compatibility](../optimization/peft.md#parallelism-compatibility)). Targets may include the input embedding (`embed_tokens`, BERT's `word_embeddings`), and `use_dora` applies. Quantization, `lora_modules_to_save` and `train_sinks: true` raise at startup; a PEFT tuner layer outside the backbone, or an adapter the save cannot fold (an `nn.MultiheadAttention` LoRA, trainable tokens), raises at trainer construction.

Injected LoRA is recognized by its PEFT tuner layers, not by parameter name, and the fold drops only the adapter tensors those layers own. A remote-code backbone whose own parameters are named `lora_*` (jina-embeddings-v3's LoRA parametrizations) is therefore not taken for injected LoRA, and its saves keep those parameters.

## Saving

Saves route through the shared `save_checkpoint` ladder ([Checkpoints](../reference/checkpoints.md)), except under injected LoRA: that branch gathers the backbone itself, folding the adapters into each tensor as it is gathered with the weight sync's out-of-place fold ([PEFT](../optimization/peft.md#online-rl--rollout-server-weight-sync)), and streams the base model's names through the same gathered writer, so a failed write raises on every rank. Either way all ranks run the gather collective and only the writer keeps the tensors. The ST pipeline config (`modules.json`, `sentence_bert_config.json`, the per-module directories) is written alongside, so the output loads with `SentenceTransformer(path)`.

The fold serves but cannot resume, so every LoRA training checkpoint also carries the unfolded trainable tensors in `resume_adapter/` (live dtype, top-level parameter names) with the run's `LoraConfig` as `adapter_config.json`, and the root marker `resume_adapter.json`; on a non-shared filesystem each node's save rank writes its own copy. Resume builds from `model_name_or_path` (the base), the script re-injects the adapters, and `EmbeddingTrainer._load_from_checkpoint` restores them bit-exact before the optimizer state (`halo/trainers/embedding/injected_lora.py`); the folded weights are never read. It refuses a model loaded from the checkpoint itself, a marked checkpoint missing its adapter file, an adapter file for other target modules or rank, a changed or unrecorded scaling (`lora_alpha`, `use_rslora`, `use_dora`, `alpha_pattern`: names and shapes would match, and the delta would silently rescale), a LoRA run resuming an unmarked checkpoint, and a full fine-tune resuming a marked one. The final `save_model()` export carries no resume state ([Merge-on-save checkpoints](../reference/checkpoints.md#merge-on-save-checkpoints)).

A full fine-tune resumes through `load_resume_weights`, which the trainer's `_weights_load_context` points at the backbone the saves write (sentence-transformers' own reload, `_load_weights_through_base`, where it declines); the optimizer state stays keyed by the `SentenceTransformer`. Under FSDP2 and pure TP it reads the checkpoint on a best-model load and on a resume whose model was built from the base (FSDP2 at `use_grouped_gemm: false`).

Under FSDP2 (torchrun), TP, EP or the grouped-GEMM EP layers the trainer refuses at construction a parameter outside the backbone (a `Dense` head, say) that is trainable, or frozen but FSDP2-sharded: the distributed save writes those modules rank-locally and the resume reloads only the backbone. A frozen module FSDP2 leaves unsharded (dtype-excluded) passes. One GPU and DDP (`accelerate launch`) without EP layers are unaffected, and the error points there. One GPU with EP layers saves through the ladder's EP saver, the only writer of their experts in the hub layout, with the pipeline config beside it.

## What to watch

Metrics read the sentence embeddings the loss forward already computed (no extra forward), over the first 256 rows of each text group: on logging steps, and on every eval batch under an `eval_` prefix (`eval_embed/norm`, …).

The eval values are this rank's per-batch means averaged over its batches, logged from rank 0, and not reduced over ranks. Where the eval loader pads its final round with repeats of the split's first rows (TP or ETP over more than one data-parallel replica with a CLI `--dataloader_drop_last=false`; a pre-sharded split is cut to the shortest rank's length and padded by nothing), those rows enter that round's means and its in-batch ranking candidates; plain DP and pure EP, which require `dataloader_drop_last: true`, drop the incomplete round instead.

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

The GPU suite trains MNRL and CoSENT plus LoRA and round-trips the gathered save and the LoRA merge on `sentence-transformers/paraphrase-MiniLM-L3-v2`. `tests/gpu/trainers/lora/test_embedding_lora_resume.py` resumes a run from its mid checkpoint against the uninterrupted one per backbone family (BERT, Qwen3, Qwen3.5, Gemma 4, GPT-OSS) and run shape (one GPU, FSDP2, accelerate DDP, pre-sharded data): LoRA on the attention projections, on the input embedding beside them or alone (`--lora`), where TP and EP must refuse the adapters, and a full fine-tune (`--lora off`), which also runs under TP (Qwen3, Qwen3.5, GPT-OSS), EP and the ep1 EP layers (Gemma 4, GPT-OSS; on one GPU and under gradient checkpointing for GPT-OSS) and checks the best-model load wherever the model has no EP layers. Every export must serve the configured pooling and `Normalize` through stock `SentenceTransformer`. LoRA trains the MoE families on their stock experts; the ep1 EP layers must refuse it. `tests/cpu/models/test_embedding_source_pipeline.py` holds the parallel loader's pipeline to the checkpoint's, and `tests/gpu/trainers/other/test_entry_scripts.py --method embedding --family gpt_oss` runs the script itself on a MoE.

## Related pages

- [Expert Parallelism](../parallelism/expert-parallelism.md) · [Tensor Parallelism](../parallelism/tensor-parallelism.md)
- [Dataset Formats](../data/dataset-formats.md) · [PEFT](../optimization/peft.md) · [Scripts Reference](../reference/scripts-reference.md)
