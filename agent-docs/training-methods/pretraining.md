# Pre-training (from scratch & continued)

Three pieces differ from [SFT](sft.md): random-init model construction, raw-text data (prepared by Halo, or tokenized by datatrove), and step-based training. Everything else — trainer, launcher, checkpoints — is the SFT path, and `scripts/training/sft.py` is the entry script.

`init_from_scratch` is accepted only by that script; every other script raises rather than silently fine-tuning pretrained weights. Continued pre-training — load a checkpoint, train on raw text — works in every parallelism mode; CP additionally refuses packed data.

## Random initialization

```yaml
model_name_or_path: Qwen/Qwen3-4B   # architecture/config + tokenizer only
init_from_scratch: true             # random weights — no checkpoint read
```

```bash
torchrun --nproc_per_node=8 scripts/training/sft.py my-pretrain.yaml --init_from_scratch=true
```

The model is built from the config on the meta device and drawn on CPU by the family's own `_init_weights`, one module at a time, under a seed hashed from the run's `seed` and the module's name; a fused expert bank is drawn one expert at a time, with the expert's index in the key (`halo/model_loading/parallel/keyed_init.py`). The weights depend only on the seed and the config: every rank, every supported layout and every GPU type starts from the same ones, and the run's global random state is left untouched. They match `from_config`'s distribution, not its values, so a seed draws different weights than it did before this scheme. A parameter the family's `_init_weights` leaves undrawn takes what `from_config` gives it, under the same key: its module's own `reset_parameters` (a remote-code router) or else transformers' default `_init_weights` (a remote-code norm whose constructor writes ones). A tensor none of them draws is refused by name.

Data parallelism (DDP, FSDP2, `ep1`), EP, CP, EP+CP, attention-only TP on a MoE and EP+TP are supported: each holds the whole model on a rank, so every rank draws it on CPU, a node's ranks at a time, and the loader shards it as it shards a checkpoint. Under EP that is the host memory `ep_lazy_loading: false` costs, a full model per rank. The copies several ranks hold of one parameter come from one rank as well, since DDP broadcasts rank 0's weights and the FSDP2 wrap gives every HSDP replica the first replica's shards and every FSDP-ignored parameter DP rank 0's copy ([Data Parallelism → HSDP](../parallelism/data-parallelism.md#hsdp-hybrid-sharded-data-parallel)).

A resumed run restores its checkpoint. Under EP, CP and TP the policy loads from the checkpoint and draws nothing. Under data parallelism the run first draws the keyed weights, and the trainer restores the checkpoint over them. A GptOss run's sinks keep the from-scratch handling they started with, so `reset_sinks` never neutralizes the trained ones.

Four refusals:

- **PP, ETP or dense TP** raise `NotImplementedError`: those loaders read only their own part of a checkpoint and have no random-init route yet.
- **Quantization** raises — QLoRA loads pretrained weights.
- **A multimodal checkpoint** raises at the VLM load; `text_only_model: true` routes it to the text-only sibling, which does random-init.
- **A model with external tables** raises `ValueError` (`reject_random_init_tables`): Qwen3.8-Flash-Next's n-gram table is frozen and comes only from the checkpoint, so random weights would train against a pretrained table.

A from-scratch GptOss has no pretrained sinks to neutralize, so they stay live and only a sink-carrying attention backend is accepted — FA4, `flex_attention` or `eager`, never FA2 or SDPA.

`model_init_kwargs` overrides config fields, so the architecture can be resized before init. Each key is written to every config level that declares it, so a decoder field of a composite config (Qwen3.5/3.6, Gemma 4) reaches the nested `text_config` the model reads. A key no level declares raises; `dtype` is refused.

```yaml
model_init_kwargs:
  num_hidden_layers: 24
  hidden_size: 2048
  num_attention_heads: 16
```

### Large models, PP, ETP or dense TP

Under PP, ETP or dense TP, or where a whole model per rank does not fit in host memory, materialize the random-init checkpoint once outside the distributed job, then train from it. The host memory is saved only by a loader that reads just its rank's part of the checkpoint: lazy EP, PP and dense TP. Data parallelism, CP and attention-only TP on a MoE hold a whole model per rank either way:

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
    --model_name_or_path "Qwen/Qwen3-4B" \
    --mode text --text_field text \
    --max_length 8192 --test_size 0.005 \
    --pack_sequences --packing_strategy bfd_split \
    --num_shards 256
```

`--mode text` tokenizes `--text_field` and appends an EOS per document (`--append_eos false` when the corpus already delimits them). Whole documents reach the packer, so the strategy decides the overflow: **`bfd_split`** carries it into later packs, where each fragment restarts at position 0, sees none of its earlier half and loses its first token as a target; `bfd` discards every token past `--max_length`; and `wrapped` chunks the token stream across document boundaries, and a completion span the wrap cuts is not trained on either side. Flags: [Pre-Processing](../data/dataset-preparation.md#parameters).

`--num_shards` must be at least the data-parallel size, or the train split raises. Every rank is truncated to the smallest rank's row count (a warning names the drop), so a shard count that is not a multiple of the data-parallel size drops about a shard from each fuller rank per pass. `ShardedDatasetLoader` assigns them by **DP rank**, not global rank ([Data Loading](../parallelism/data-loading.md#pre-processed-sharded-datasets)).

The SFT trainer auto-detects the artifact and holds the run to it. Text mode bakes `train_on_completions_only: false` while the training-side default is `true`, so the YAML must state it or startup raises. `max_length` must equal the prep `--max_length` exactly; both directions raise.

## Training from datatrove token folders

`token_sources` trains on folders that datatrove's `DocumentTokenizer` wrote, mixed by weight, with no dataset preparation. Halo reads the file layout ([Dataset Formats](../data/dataset-formats.md#datatrove-token-folders)) and never imports datatrove, which is not in the image: tokenize in an environment of your own.

```python
# pip install datatrove, outside the Halo image
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer

LocalPipelineExecutor(
    pipeline=[
        ParquetReader("hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT"),
        DocumentTokenizer(
            output_folder="/data/tok/fineweb-edu",
            tokenizer_name_or_path="Qwen/Qwen3-4B",
            eos_token="<|endoftext|>",
            seed=0,
        ),
    ],
    tasks=16,
).run()
```

- `eos_token` ends every document with that token. Without it, a tokenizer whose template adds nothing (Qwen3's) writes documents back to back. Halo takes document boundaries from each file's `.ds.index` either way, never from a separator.
- `seed` fixes the order in which `DocumentTokenizer` shuffles each task's documents. Unseeded, a re-run writes another order: a resume onto a whole re-run is refused as a changed mix, but a re-run of some tasks can keep every window the fingerprint samples and pass.
- `DocumentTokenizerMerger` (one task) merges the per-task files into fewer, larger ones. Either output trains.

```yaml
model_name_or_path: Qwen/Qwen3-4B
token_sources:
  - {name: web, path: s3://my-bucket/tok/fineweb-edu, weight: 7}
  - {name: code, path: /mnt/data/tok/code, weight: 3}
eval_token_sources:
  - {name: web-val, path: s3://my-bucket/tok/fineweb-edu-val}
max_length: 4096
max_steps: 50000
packing: true                         # each window is a packed row
train_sampling_strategy: sequential   # the mix is shuffled already
train_on_completions_only: false
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
```

**Rows.** Each folder's non-empty `.ds` files, in name order, are cut into windows of `max_length` tokens. A window never crosses a file, and each file's tail shorter than a window is dropped. A window is a packed row, as the `wrapped` packing writes one: positions restart at every document end the index records, and each document's first label is masked. A document cut at a window edge thus loses one target there. Every other token trains, the document separator included. The row carries labels equal to its ids, so the collator never masks an id for matching the pad id; Qwen3, Qwen3.5 and gpt-oss pad with `<|endoftext|>`.

**Mix.** Weights are relative, set on every source or on none, which gives each source its share of the windows. They become whole-sample quotas over a pattern of 4,096 samples: every 4,096 consecutive samples hold each quota exactly, and a shorter run, such as one step's rows, stays within about two samples of its share. Each source's windows are drawn in a keyed permutation, so a source weighted past its size starts another pass, in a new order. A source whose quota rounds to zero is refused. At startup each source logs its share, its windows, the share of its tokens they hold and its passes over the run. Nothing the mix holds grows with the token count (`halo/data/sources/token_mix.py`).

**Order.** Sample `i` is a pure function of `i`, the sources, `max_length` and the seed (`data_seed`, else `seed`). A step reads `G` rows: `per_device_train_batch_size × gradient_accumulation_steps × data-parallel size`. Under pipeline parallelism the accumulation is the microbatch count, which splits the per-device batch, so `G` leaves it out. The run holds `max_steps × G` samples, and the sequential sampler gives step `t` samples `[t·G, (t+1)·G)`, split across data-parallel ranks. Tensor- and pipeline-parallel siblings read the same rows ([Data Loading](../parallelism/data-loading.md)).

**Resume.** A resume skips the `t` trained steps as `t·G` samples and reads the rows of the uninterrupted run, so it must keep `G`. Another data-parallel size reads the same rows when the per-device batch or the accumulation makes up for it; exact optimizer resume still needs the same world size.

**Fingerprint.** The mix's fingerprint holds the order version, `max_length`, the seed, `G` as `rows_per_step`, and per source its quota, token width, files and a digest of 16 evenly spaced windows. Each checkpoint carries it in `trainer_state.json`, under `stateful_callbacks` as `TokenMixCheckpointCallback`. The save publishes that file last, so every checkpoint a resume can pick holds the fingerprint, wherever the run stopped. A resume whose live run differs in any field raises before the model load and names each change. Paths and `max_steps` stay out of it, so moved folders or a longer run resume.

**Several ranks and nodes.** At startup the ranks compare their fingerprints of `token_sources`, then of `eval_token_sources`, in one all-reduce each: file names, token counts and the 16-window digest, so a copy that differs only outside those windows passes. When one rank sees other files (a folder still being written, a stale NFS view, a node-local copy that differs), every rank raises, naming the ranks and nodes that differ. On per-node storage a local path must hold the same files on every node.

**S3 sources.** An `s3://` path downloads whole into the [S3 dataset cache](../data/s3-utilities.md) once per input-filesystem scope; nothing streams from S3. The other ranks wait for their scope's download in one store join, bounded by `DIST_STORE_TIMEOUT_HOURS` (4 by default), and a download cut short starts from zero at the next launch, so stage a larger prefix with `aws s3 sync` and give its local path. A prefix that changed after it was cached is refused, not downloaded again: a run reading the copy and the checkpoints trained on it need its files as they are. Remove the cached folder to fetch the new state, upload the new state under a new prefix, or point the source at the cached folder to keep training on it. A failed fetch names the source and its prefix.

**Reading and open files.** A rank reads a window with a positional read of its `.ds` file and the window's document ends from a memory map of the file's index, and keeps both open for the next row, up to 512 files per folder at two descriptors each. A folder of more than 512 files can reopen a file for a row, which is a metadata request on a network filesystem; merge such a folder with `DocumentTokenizerMerger`. At startup each rank raises its soft open-file limit toward the hard one, to hold every descriptor the folders can keep plus 512 for the rest of the run. When the hard limit is lower, the run is refused at startup: raise it (`docker run --ulimit nofile=N:N`), or merge the files. A folder must not change while a run reads it: a file cut short raises at its next row past the cut, and a failed read of an index on a network filesystem ends the rank with a bus error.

**Held-out folders.** Each `eval_token_sources` folder is read whole in file order as its own eval split, reported as `eval_<name>_loss`, and takes no weight. The other eval metrics, such as `eval_mean_token_accuracy`, carry no folder name, so every folder logs them under one key. An evaluation's last round is uneven: padded with the folder's first windows under data parallelism, weighed as a full batch on one rank, dropped under `dataloader_drop_last`. A folder whose windows fill whole rounds of `per_device_eval_batch_size × data-parallel size` scores their exact token mean. Evaluation (`eval_strategy`, `eval_on_start` or the MTP acceptance gate) runs on these folders only and needs at least one; the gate takes exactly one, since it sets the last reading of its one key against the first.

**Refused**, each naming its remedy:

- `max_steps` unset, a sampler other than `sequential`, `packing` off, a `packing_strategy` or `eval_packing` off its default, `padding_free`;
- `ignore_data_skip: true`, which would train the resumed steps' rows again, and `auto_find_batch_size: true`, since a halved batch reads fewer than `G` rows a step and the resume skip miscounts;
- `dispatch_batches` or `split_batches` on under `accelerator_config`, which changes the rows a step reads or the resume skips, and `even_batches` off, which hangs the evaluation of a held-out folder that does not split evenly over the ranks;
- `dataloader_in_order: false` with loader workers, since a batch returned out of turn trains at another step than the resume skips;
- evaluation without `eval_token_sources`, the MTP acceptance gate with more than one, and a `metric_for_best_model` other than a held-out folder's `eval_<name>_loss`: a token run logs no `eval_loss`, which `load_best_model_at_end` reads unless told otherwise;
- `train_on_completions_only: true`, and `dataset`, `dataset_ratio`, `test_size` or a rendering field (`conversation_field`, `images_field`, `system_prompt`, `tools_field`, `reasoning_effort_field`, `tree_packing`, …) beside `token_sources`;
- context parallelism, which refuses packed rows, before any folder is read;
- a `.ds` without its `.ds.index` or `.ds.metadata`, a metadata line naming `Unknown Tokenizer` (written by `DocumentTokenizerMerger` over inputs without metadata, with a guessed token width), a folder mixing token widths, trained or held-out folders of two token widths (datatrove writes 2-byte ids up to a vocabulary of 65,536 and 4-byte ids past it, so two widths are two tokenizers), a file whose index ends elsewhere than its tokens (a partial write or copy), and a folder with no window of `max_length`;
- folders whose open files the hard open-file limit cannot hold;
- a cached `s3://` folder whose prefix has changed;
- a resume onto a checkpoint whose `trainer_state.json` carries no fingerprint, or whose mix or `rows_per_step` differs from the live run's;
- a run without `token_sources` resuming a token run's checkpoint, whose trainer state would carry the fingerprint into every checkpoint that run saves;
- a sampled window holding an id at or past the model's vocabulary. Metadata that names more than one tokenizer across the folders, or one other than the run's, only warns and lists each folder's names, since a local path spells the same tokenizer differently.

Not supported: tokenizing inside Halo, `.ds.loss` masks, packing whole documents without cutting them, changing the mix on resume, per-source training metrics, and Megatron `.bin`/`.idx` folders.

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

`warmup_steps` reads as on [SFT](sft.md#learning-rate-and-global-batch-size). `max_grad_norm` clips on the global gradient norm reduced across all DP/EP/TP shards ([clipping](../reference/trainer-architecture.md#ep-aware-gradient-clipping)).

Under `bf16` (the toolkit default) with a stock AdamW, AdamWBF16 with stochastic rounding is the optimizer unless accelerate manages DDP ([BF16 Optimizer](../optimization/bf16-optimizer.md)). Checkpoints write as HF-standard safetensors sharded at `save_max_shard_size` (default `5GB`); exact optimizer resume needs the same world size ([Checkpoints](../reference/checkpoints.md)).

For MoE pre-training, router balancing matters from step 0: [`BalancingBiasCallback`](callbacks.md#balancingbiascallback) implements DeepSeek-V3's aux-loss-free bias update.

## Large corpora

There is no streaming path: `datasets` loads at `keep_in_memory=False`, so a corpus is memory-mapped from Arrow — disk bounds it, not RAM, but the whole dataset must land on disk. Split a multi-terabyte corpus at the source, tokenize each batch to its own prefix with `prepare_dataset.py`, and train the phases in sequence. A `token_sources` run refuses `ignore_data_skip` and a mix that changed on resume, so its next phase is a new run from the last checkpoint's weights.

Every phase, the first included, runs a schedule that does not read `max_steps`, such as `constant_with_warmup` with `warmup_steps` as a step count. The scheduler is rebuilt over each phase's `max_steps`, so a decaying one ends a phase near zero and jumps back up in the next. Each phase after the first resumes the previous phase's last checkpoint with two settings, because a resume counts the steps already trained against the new prefix:

- `ignore_data_skip: true`, or the resume skips the new prefix's first batches as if the earlier steps had read them.
- `max_steps` set to the steps trained so far plus this phase's, or the earlier steps count toward this phase's length.

A resume inside a phase starts a new pass over its prefix, since `ignore_data_skip` skips no batch, so raise that phase's `max_steps` by the steps it had trained.

Keep `--max_length` and `--model_name_or_path` identical across batches so shards stay interchangeable, and point `TMPDIR` / `HF_DATASETS_CACHE` at the large volume. A `dataset:` **list** is not the way to mix prepared prefixes: only a single string path is probed for the pre-processed stamp, and list entries load whole, with no per-rank sharding. A weighted mix of tokenized sources is what [`token_sources`](#training-from-datatrove-token-folders) trains. `dataset_ratio` (a keep fraction, broadcast or one per entry) applies to raw sources, single or mixed.

Set `DIST_SHARED_FILESYSTEM=0` for per-node local storage (default `1` = shared NFS/Lustre) ([Filesystem](../data/filesystem-handling.md)).

## Related pages

- [SFT](sft.md) · [Pre-Processing](../data/dataset-preparation.md) · [Distributed Data Loading](../parallelism/data-loading.md)
- [Multi-Node Training](../parallelism/multi-node.md) · [Checkpoints & Resume](../reference/checkpoints.md) · [Scale & Limits](../reference/scale-and-limitations.md)
