# Pre-training (from scratch & continued)

Three pieces differ from [SFT](sft.md): random-init model construction, raw-text data prep, and step-based training with a pre-training LR schedule. Everything else — trainer, launcher, checkpoints — is the SFT path.

`init_from_scratch` runs only on the plain data-parallel / FSDP2 path. The gate is on the parallelism mode, not the architecture: an MoE at `ep_size=1` under FSDP2 random-inits fine, while EP/TP/CP/ETP/PP raise `NotImplementedError` in `load_distributed_model` — distributed random-init of sharded experts is not implemented, and PP has nothing to be stage-aware about without a checkpoint. It is also incompatible with quantization and with VLM models.

Continued pre-training — load a checkpoint, train on raw text — works in every mode through the normal `from_pretrained` path.

## 1. Random initialization (`init_from_scratch`)

```yaml
model_name_or_path: Qwen/Qwen3-4B   # architecture/config + tokenizer only
init_from_scratch: true             # random weights — no checkpoint read
```

```bash
torchrun --nproc_per_node=8 scripts/training/sft.py my-pretrain.yaml --init_from_scratch=true
```

- The model is built with the resolved auto class's `from_config(config)`, so weights come from `_init_weights`.
- `set_seed(seed)` runs before construction, making init identical on every rank so FSDP2 shards and DP replicas agree.
- GptOss sink-reset is skipped (there are no pretrained sinks).

`model_init_kwargs` overrides config fields, so the architecture can be resized before init. Each key is written to every config level that declares it, so a decoder field of a composite config (Qwen3.5/3.6, Gemma 4, VLM wrappers) reaches the nested `text_config` where the model reads it. A key no level declares raises rather than silently training the stock value:

```yaml
model_init_kwargs:
  num_hidden_layers: 24
  hidden_size: 2048
  num_attention_heads: 16
```

### Large or MoE models — init once, then train from the checkpoint

For an MoE model or any run that needs parallelism, materialize the random-init checkpoint once outside the distributed job:

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B")
config.num_hidden_layers = 32   # optional resize before init
model = AutoModelForCausalLM.from_config(config)
model.save_pretrained("/mnt/init/qwen3-30b-a3b-fresh", safe_serialization=True)
AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B").save_pretrained("/mnt/init/qwen3-30b-a3b-fresh")
```

Point `model_name_or_path` at that directory and train with any parallelism. Keep the init seed fixed for reproducibility. The init checkpoint is also a concrete artifact to profile and tune (depth/width/heads/expert count) with the [profiling benchmarks](../optimization/throughput-benchmarks.md#running-benchmarks) before committing to the long run.

## 2. Raw-text data preparation (`--mode text`)

Pre-training needs raw-text tokenization with document packing and no chat template:

```bash
python scripts/before_training/prepare_dataset.py \
    --input  "s3://my-bucket/raw/my_corpus" \
    --output "s3://my-bucket/preprocessed/my_corpus_packed" \
    --model-name "Qwen/Qwen3-4B" \
    --mode text --text-field text \
    --max-length 8192 \
    --pack-sequences --packing-strategy bfd_split \
    --num-shards 256
```

- `--mode text` tokenizes `--text-field` directly and appends an EOS to every document before packing; `--no-append-eos` disables that when the corpus already delimits documents.
- Whole, untruncated documents flow into the packer (text mode disables tokenizer truncation whenever `--pack-sequences` is on), so the strategy decides what happens past `--max-length`. Use **`bfd_split`** for pre-training: it carries each document's overflow into later packs. The `bfd` default truncates instead and discards every token past `--max-length` of a document; `wrapped` concatenates and cuts mid-document, emitting no document boundaries.
- `--num-shards N` (use `N = k × world_size`) produces per-rank shards. `ShardedDatasetLoader` assigns them by **DP rank**, not global rank — see [Distributed Data Loading](../parallelism/data-loading.md#pre-processed-sharded-datasets).

The SFT trainer auto-detects the packed, sharded dataset. A text-mode artifact bakes its labels with `train_on_completions_only: false`, and the training YAML must state the same — the script's own default is `true`, and the mismatch raises at startup rather than training the opposite of the baked labels.

## 3. Step-based training & LR schedule

Drive the run with `max_steps`, not `num_train_epochs`:

```yaml
max_steps: 200000
lr_scheduler_type: cosine
warmup_steps: 2000
learning_rate: 3.0e-4
save_steps: 1000
max_grad_norm: 1.0
train_on_completions_only: false   # must match the text-mode artifact's baked labels
```

- `max_grad_norm` clips on the global gradient norm reduced across all DP/EP/TP shards ([EP-aware gradient clipping](../reference/trainer-architecture.md#ep-aware-gradient-clipping)).
- Under `bf16` (the toolkit default) on FSDP2, AdamWBF16 with stochastic rounding is the optimizer; it is not auto-enabled under accelerate-managed DDP (outside the validated SR matrix — set `bf16_optimizer: true` to force it) — [BF16 Optimizer](../optimization/bf16-optimizer.md).
- Checkpoints write as HF-standard safetensors, sharded at `save_max_shard_size` (default `5GB`). Exact optimizer resume requires the same world size — [Checkpoints & Resume](../reference/checkpoints.md).
- For MoE pre-training, router load balancing matters from step 0: [`RouterBiasBalancingCallback`](callbacks.md#routerbiasbalancingcallback) implements DeepSeek-V3's aux-loss-free bias update.

## Working with large corpora

There is no streaming path. `load_dataset_from_source` loads with HuggingFace `datasets` at `keep_in_memory=False`, so the corpus is memory-mapped from Arrow — disk is the limit, not RAM, but the whole dataset must be materialized on disk. A single job over a multi-terabyte corpus is impractical; process in chunks:

1. **Split at the source.** Corpora like C4 ship as many files (`c4-train.00000-of-01024.json.gz`). Group them into batches.
2. **Tokenize each batch independently** to its own S3 prefix with the command above, varying only `--input`/`--output`. These jobs are embarrassingly parallel and CPU-bound — run them as separate CPU jobs ([Infrastructure](../infrastructure/index.md)).
3. **Train over the batches as a mix.** Pass the batch prefixes as a multi-source `dataset` list with `dataset_ratio` keep-fractions. Each batch is already packed and sharded.

Keep `--max-length` and `--model-name` identical across batches so shards stay interchangeable, and point `TMPDIR`/`HF_DATASETS_CACHE` at the large volume so the Arrow cache does not fill the root FS.

Reproduce a staged curriculum (general → knowledge → long-context) as sequential phases, each a config that resumes the prior checkpoint with a different `dataset_ratio` and `max_length`. For semantic deduplication before tokenization, `scripts/inference/generation/dataset_deduplication.py` runs FAISS cosine-similarity dedup over a text field.

## Multi-node at scale

Mechanics are shared with every method; the pre-training-specific points:

- **Sharded loading per DP rank.** With `--num-shards = k × world_size` each rank reads only its slice. EP is orthogonal to DP, so EP ranks still split data ([Data Parallel size](../parallelism/data-loading.md#data-parallel-size)).
- **Filesystem coordination.** Set `DIST_SHARED_FILESYSTEM=0` for per-node local storage (default `1` = shared NFS/Lustre); `fs_aware_main_first()` adapts the read side (downloads, caching) and `fs_aware_save_rank()` the write side ([Filesystem Handling](../data/filesystem-handling.md)).
- **Launch.** Standard `torchrun` rendezvous; NCCL timeouts and node-local group math in [Multi-Node Training](../parallelism/multi-node.md).

## Optimizations

The levers are the same as SFT. Grouped GEMM (auto on SM90+) is the dominant MoE speedup; FusedLinearCE avoids the `batch × seq × vocab` logits tensor and is the largest win at long context. See [Throughput Benchmarks](../optimization/throughput-benchmarks.md#maximizing-throughput-sequence-batch), [Grouped GEMM](../optimization/grouped-gemm.md), and [Liger Kernels](../optimization/liger-kernels.md).

## Checkpoints and evaluation

Checkpoint often to a durable path (`save_steps`, `save_total_limit`, `output_dir` on the large volume) and evaluate in a separate job so the long run never blocks on eval. A checkpoint directory is self-contained — weights, `config.json`, tokenizer — and loads with `from_pretrained` or vLLM without conversion. The per-rank `save_sharded_ep` layout is not eval-ready and must be reassembled first. `eval_strategy: steps` gives in-the-loop validation loss; reserve benchmark suites for the offline path.

## See also

- [Capabilities & limitations at scale](../reference/scale-and-limitations.md)
- [SFT](sft.md) · [SFT Pre-Processing](../data/dataset-preparation.md) · [Distributed Data Loading](../parallelism/data-loading.md)
- [Multi-Node Training](../parallelism/multi-node.md) · [Checkpoints & Resume](../reference/checkpoints.md)
