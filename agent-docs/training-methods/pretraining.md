# Pre-training (from scratch & continued)

Three pieces differ from [SFT](sft.md): random-init model construction, raw-text data preparation, and step-based training. Everything else — trainer, launcher, checkpoints — is the SFT path, and `scripts/training/sft.py` is the entry script.

`init_from_scratch` is accepted only by that script; every other script raises rather than silently fine-tuning pretrained weights. Continued pre-training — load a checkpoint, train on raw text — works in every parallelism mode; CP additionally refuses packed data.

## Random initialization

```yaml
model_name_or_path: Qwen/Qwen3-4B   # architecture/config + tokenizer only
init_from_scratch: true             # random weights — no checkpoint read
```

```bash
torchrun --nproc_per_node=8 scripts/training/sft.py my-pretrain.yaml --init_from_scratch=true
```

The model is built with the resolved auto class's `from_config(config)`, so weights come from `_init_weights`. The run's `seed` is set first, making init identical on every rank so FSDP2 shards and DP replicas agree.

Three refusals:

- **EP, TP, CP, ETP or PP** raise `NotImplementedError`: distributed random-init of sharded parameters is not implemented, and PP has nothing to be stage-aware about without a checkpoint. The gate reads `ep_group_size > 1`, so an MoE without EP or ETP random-inits fine.
- **Quantization** raises — QLoRA loads pretrained weights.
- **A multimodal checkpoint** raises at the VLM load; `text_only_model: true` routes it to the text-only sibling, which does random-init.

A from-scratch GptOss has no pretrained sinks to neutralize, so they stay live and only a sink-carrying attention backend is accepted — FA4, `flex_attention` or `eager`, never FA2 or SDPA.

`model_init_kwargs` overrides config fields, so the architecture can be resized before init. Each key is written to every config level that declares it, so a decoder field of a composite config (Qwen3.5/3.6, Gemma 4) reaches the nested `text_config` the model reads. A key no level declares raises; `dtype` is refused.

```yaml
model_init_kwargs:
  num_hidden_layers: 24
  hidden_size: 2048
  num_attention_heads: 16
```

### Large or MoE models

Materialize the random-init checkpoint once outside the distributed job, then train with any parallelism:

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B")
config.num_hidden_layers = 32   # optional resize before init
AutoModelForCausalLM.from_config(config).save_pretrained("/mnt/init/fresh", safe_serialization=True)
AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B").save_pretrained("/mnt/init/fresh")
```

Point `model_name_or_path` at that directory, drop `init_from_scratch`, and keep the init seed fixed. Profile that artifact before committing to the long run ([benchmarks](../optimization/throughput-benchmarks.md#running-benchmarks)).

## Raw-text data preparation

Raw text is tokenized with document packing and no chat template:

```bash
python scripts/before_training/prepare_dataset.py \
    --input  "s3://my-bucket/raw/my_corpus" \
    --output "s3://my-bucket/preprocessed/my_corpus_packed" \
    --model-name "Qwen/Qwen3-4B" \
    --mode text --text-field text \
    --max-length 8192 --test-size 0.005 \
    --pack-sequences --packing-strategy bfd_split \
    --num-shards 256
```

`--mode text` tokenizes `--text-field` and appends an EOS per document (`--no-append-eos` when the corpus already delimits them). Whole documents reach the packer, so the strategy decides the overflow: **`bfd_split`** carries it into later packs, `bfd` discards every token past `--max-length`, and `wrapped` concatenates across boundaries, emitting none at all. Flags: [Pre-Processing](../data/dataset-preparation.md#parameters).

`--num-shards` must be at least the data-parallel size, or the train split raises; a non-multiple truncates every rank to the smallest shard. `ShardedDatasetLoader` assigns them by **DP rank**, not global rank ([Data Loading](../parallelism/data-loading.md#pre-processed-sharded-datasets)).

The SFT trainer auto-detects the artifact and holds the run to it. Text mode bakes `train_on_completions_only: false` while the training-side default is `true`, so the YAML must state it or startup raises. `max_length` must equal the prep `--max-length` exactly; both directions raise.

## Step-based training

```yaml
max_steps: 200000
lr_scheduler_type: cosine
warmup_steps: 2000
learning_rate: 3.0e-4
save_steps: 1000
max_grad_norm: 1.0
train_on_completions_only: false   # matches the text-mode artifact
```

`warmup_steps` of 1 or more is an exact step count, below 1 a fraction of the total; there is no `warmup_ratio`. `max_grad_norm` clips on the global gradient norm reduced across all DP/EP/TP shards ([clipping](../reference/trainer-architecture.md#ep-aware-gradient-clipping)).

Under `bf16` (the toolkit default) with a stock AdamW, AdamWBF16 with stochastic rounding is the optimizer unless accelerate manages DDP ([BF16 Optimizer](../optimization/bf16-optimizer.md)). Checkpoints write as HF-standard safetensors sharded at `save_max_shard_size` (default `5GB`); exact optimizer resume needs the same world size ([Checkpoints](../reference/checkpoints.md)).

For MoE pre-training, router balancing matters from step 0: [`RouterBiasBalancingCallback`](callbacks.md#routerbiasbalancingcallback) implements DeepSeek-V3's aux-loss-free bias update.

## Large corpora

There is no streaming path: `datasets` loads at `keep_in_memory=False`, so a corpus is memory-mapped from Arrow — disk bounds it, not RAM, but the whole dataset must land on disk. Split a multi-terabyte corpus at the source, tokenize each batch to its own prefix with the command above, and train the phases in sequence, each resuming the previous checkpoint.

Keep `--max-length` and `--model-name` identical across batches so shards stay interchangeable, and point `TMPDIR` / `HF_DATASETS_CACHE` at the large volume. A `dataset:` **list** is not the way to mix prepared prefixes: only a single string path is probed for the pre-processed stamp, and list entries load whole, with no per-rank sharding. `dataset_ratio` (a keep fraction, broadcast or one per entry) applies to raw sources, single or mixed.

Set `DIST_SHARED_FILESYSTEM=0` for per-node local storage (default `1` = shared NFS/Lustre) ([Filesystem](../data/filesystem-handling.md)).

## Related pages

- [SFT](sft.md) · [Pre-Processing](../data/dataset-preparation.md) · [Distributed Data Loading](../parallelism/data-loading.md)
- [Multi-Node Training](../parallelism/multi-node.md) · [Checkpoints & Resume](../reference/checkpoints.md) · [Scale & Limits](../reference/scale-and-limitations.md)
