# SFT Dataset Pre-Processing

`scripts/before_training/prepare_dataset.py` converts raw conversation data into tokenized, optionally packed and sharded datasets ready for training. The SFT script `scripts/training/sft.py` (text or VLM, auto-detected) detects a pre-processed dataset via its `metadata.json` and skips tokenization, cutting startup latency on large runs.

Only SFT (text and VLM) supports offline pre-processing. DPO, SMPO, GRPO, and Classification use
different data formats and tokenize during training. The script is **single-process** — run it with
plain `python`, not `torchrun`; `--num_proc` sets HF `map` workers, not ranks. Its flags are the
`PreprocessingConfig` fields (`src/data/preprocessed/metadata.py`) plus the tool's own
`PrepareDatasetArguments`, spelled like the training YAML keys; a YAML file of them works as the first
argument, with `--key=value` overrides after it.

Pipeline: load (S3 / HuggingFace Hub / local) → tokenize (tokenizer or VLM processor) → pack (text
SFT only, optional) → shard (Megatron-LM style, optional) → save (S3 / Hub / local).

> [!WARNING]
> **`--max_length` must equal the training config's `max_length` exactly**
>
> Both directions raise at startup. Preprocessed rows are not re-truncated at runtime, so a larger
> baked length would silently feed over-length batches, and a smaller one loses tokens the config
> promised. Re-preprocess, or change `max_length` to match.

The knobs baked into the rows are recorded in `metadata.json` and re-checked against the training
config at startup. A disagreement raises when the training YAML **states** the knob — a runtime value
cannot re-render already-baked rows. A knob left at its own default only warns: that config claims
nothing about the artifact, and the prepared value is what trained.

Two groups are compared. The chat-render knobs are `--conversation_field`, `--system_prompt`,
`--model_supports_system_role false`, `--tools_field`, `--interleaved_thinking`, `--images_field`, and, under
completion masking, `--assistant_message_template`. The tokenizer mutations — `--pad_token`,
`--eos_token`, `--bos_token`, `--chat_template` — change the ids themselves: an EOS override moves
the completion-mask boundaries, and a template override re-renders every turn. `--chat_template` is
recorded as resolved template *text*, so a path and the text it names compare equal.

The checked set is every `PreprocessingConfig` field outside `_UNCOMPARED_FIELDS`, so a new knob is
compared by default, and a knob the metadata does not record warns instead. `--mode text`
artifacts render no template and skip the render check entirely. A training config setting
`packing: true` against an unpacked artifact warns — preprocessed rows are never packed at runtime.
A `model_name_or_path` other than the prep `--model_name_or_path` only warns, so train with the tokenizer
the rows were baked with.

`--mode` is validated on the config, not only by the CLI's choices: an unvalidated typo like `txt`
falls through every `mode == "text"` branch and chat-templates a raw pretraining corpus. A knob belonging
to the other mode set to a non-default value (`--system_prompt` with `--mode text`, `--text_field` with
`--mode chat`) is rejected rather than silently ignored; `_KNOB_MODES` names the mode that reads each
mode-specific knob. `PreprocessingConfig` also refuses `--pack_sequences` with `--is_vlm` and an unknown
`--packing_strategy`, so every refusal lands before the input is fetched.

Over-length rows are dropped, not truncated in `chat` and `--is_vlm` mode. `--mode text` instead
truncates each document to `--max_length` (reserving the last slot for the appended EOS), unless
`--pack_sequences` is set.

If dropping empties a split the script raises and names the usual cause (a wrong
`--conversation_field`, or an `--assistant_message_template` that does not match the rendered
assistant turn) rather than `--max_length`. The same two causes are caught again after the label
bake: a probe of the first and last 64 rows raises when every label is `-100`, rather than shipping a
dataset that trains zero tokens.

`--tokenizer_backend gigatoken` (optional extra `halo[gigatoken]`) encodes text→ids with the [gigatoken](https://github.com/marcelroed/gigatoken) Rust encoder — ~6× faster tokenization on UltraChat 200K with Qwen3-0.6B — while chat templating and decoding stay on the model's tokenizer (`src/model_loading/tokenizer_backend.py`). Token IDs are verified against the HF tokenizer at startup; any divergence raises. VLM mode swaps the processor's inner tokenizer the same way, but image processing dominates there, so the win is small.

The same two values are also a **training** config field, `tokenizer_backend` (default `hf`). It resolves in `configure_run_tokenizer`, so every training method that tokenizes rows honors it with no preparation pass; embedding training rejects a non-`hf` value, since SentenceTransformer owns its own tokenization.

With `--train_on_completions_only` the completion-only `labels` are baked into the saved dataset under **the span policy the artifact's own runtime collator applies** (`mask_batch_to_completion_spans`): marker tokens trained, each turn's terminator bounded by the next marker start.

That is `COLLATOR_SPAN_POLICY` (no end-of-sequence fallback), and `PACKED_SPAN_POLICY` under `--pack_sequences`, where a turn with no terminator anywhere ends at the sequence end. Preprocessed and runtime training therefore see identical loss masks, image tokens included (masked after the span refill on both paths).

Baked labels are authoritative at training time. The SFT script disables runtime completion re-masking for preprocessed data: a re-mask would overwrite them, e.g. re-masking a packed chunk whose response marker landed in the previous chunk drops its trained tokens.

So `train_on_completions_only` in the training config cannot change them; it is validated instead, and a value disagreeing with the baked labels raises at startup. The collator only pads the stored labels (VLM uses `PreprocessedVLMDataCollator`).

Offline-packed data (`--pack_sequences`) still needs per-document position IDs at collation, so the SFT script selects the packing collator from `metadata.packed` — the collator choice follows the data, not the config flag; TRL-side packing stays off either way.

## Usage

```bash
python scripts/before_training/prepare_dataset.py \
    --input "s3://bucket/raw/my_dataset" \
    --output "s3://bucket/preprocessed/my_dataset" \
    --model_name_or_path "Qwen/Qwen3-8B" \
    --max_length 8192 \
    --test_size 0.01 \
    --num_shards 64 \
    --pack_sequences \
    --conversation_field prompt \
    --assistant_message_template $'<|im_start|>assistant\n'
```

`--conversation_field` and `--train_on_completions_only` default to the **training-side** values
(`prompt`, and masking on), so a dataset prepared with defaults is one the training defaults accept.
Masking on means `--assistant_message_template` is required — pass the marker your model's chat
template renders, or `--train_on_completions_only false` to bake full-sequence labels.

`--input` accepts an S3 URI, a Hub ID (`org/name`, with optional `:config` and `@split` suffixes,
e.g. `HuggingFaceH4/ultrachat_200k@train_sft`), or a local path. `--output` accepts an S3 URI,
`hf://org/name`, or a local path (add `--private` for a private Hub dataset). Only the explicit
`hf://` prefix uploads to the Hub — unlike `--input`, a bare `org/name` output is a **local**
directory, so a relative path such as `preprocessed/my_dataset` is never published.

The output holds two splits, `train` and `test`. A multi-split input must carry `train`; with no
`test` split its `validation` split is baked as `test`. Any other split (`validation` next to
`test`, an `unsupervised` split) is left out, with a warning that names it. To prepare one of those,
point `--input` at it with an `@split` suffix.

**VLM mode** (`--is_vlm`, e.g. Qwen2.5-VL / Qwen3-VL; packing not supported) stores `input_ids` with
vision placeholders expanded, `attention_mask`, `labels` (image tokens masked to -100),
`pixel_values` (float16 bytes), `pixel_values_shape`, and `image_grid_thw`.

Two combinations raise and point at the runtime VLM path: `--is_vlm` with `--tools_field` or
`--interleaved_thinking` (the offline render would diverge from the runtime one), and a processor
emitting a vision key outside this fixed schema on an image row (`mm_token_type_ids` and the like —
the model needs it at forward).

VLM sequences are never truncated, since cutting expanded placeholders while `pixel_values` keeps
every patch desyncs text from vision. Over-length rows are dropped here instead — the runtime VLM
collator raises on one, because a live batch cannot drop rows without desyncing DP ranks.
`--min_pixels`/`--max_pixels` are part of the map cache key, so re-running at a new resolution does
not reuse stale pixels.

Images may live inside the conversation's content parts or in a **separate column**, the shape hub
VLM datasets ship (FineVision / the_cauldron / Docmatix). `--images_field <column>` merges that
column into the messages exactly as the runtime VLM path does, so both routes bake the same rows.

The merge fills the conversation's image placeholders in order, else prepends the images to the first
user turn. It requires `--is_vlm`; the text tokenization merges nothing.

An image column (`images` / `image` / `pixel_values`) that **no** `--images_field` names is refused.
Every source column is dropped at tokenization, so it would otherwise be discarded in silence and the
artifact would carry `is_vlm: true` over rows holding no pixels, which training reads as multimodal.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | *required* | Input: S3 URI, Hub ID, or local path |
| `--output` | *required* | Output: S3 URI, `hf://org/name`, or local path |
| `--model_name_or_path` | *required* | Model name/path for the tokenizer; completion-only masking also reads its `config.json` for the turn-terminator ids, so it must be the model checkpoint, not a bare tokenizer |
| `--max_length` | `8192` | Maximum sequence length |
| `--trust_remote_code` / `--no-trust_remote_code` | `True` | Execute the checkpoint's own tokenizer/processor and config code when loading them — on, the default every tool reading a local artifact shares ([Scripts](../reference/scripts-reference.md#input-guards)) |
| `--mode` | `chat` | `chat` = apply chat template to `--conversation_field` (SFT); `text` = raw-text causal-LM for (continued) pre-training (tokenize `--text_field`, append EOS per document). See [Pre-training](../training-methods/pretraining.md). |
| `--text_field` | `text` | (mode=text) column holding raw text |
| `--append_eos` | `True` | (mode=text) the per-document EOS that preserves document boundaries; `false` skips it |
| `--conversation_field` | `prompt` | Conversation field name (mode=chat). Matches the training-side default |
| `--system_prompt` | `None` | System prompt to prepend |
| `--tools_field` | `None` | Dataset field holding a tool list (list of dicts or JSON string), forwarded into `apply_chat_template` — must match the `tools_field` used at training time. Rejected with `--is_vlm` |
| `--interleaved_thinking` | off | Keep thinking blocks in non-final assistant turns (`clear_thinking=False`), matching the live path's `interleaved_thinking`. Rejected with `--is_vlm` |
| `--model_supports_system_role` | `True` | `false` merges the system prompt into the first user message (models without a system role) |
| `--train_on_completions_only` | `True` in `--mode chat`, `False` in `--mode text` | Mask user turns; train only on assistant responses. The chat default matches the training side and requires `--assistant_message_template`; asking for masking under `--mode text` is rejected (raw documents have no assistant turns) |
| `--assistant_message_template` | `None` | Template marking start of assistant response |
| `--pack_sequences` | `False` | Pack multiple sequences (text SFT only; refused for a model with compressed-KV layers, such as DeepSeek-V4) |
| `--packing_strategy` | `bfd` | TRL strategy: `bfd` (best-fit-decreasing, drops overflow), `bfd_split` (carries overflow into later examples) or `wrapped` (concatenate-and-chunk — emits no document boundaries, so a packed row attends across itself) |
| `--is_vlm` | `False` | VLM mode (stores `pixel_values`, `image_grid_thw`) |
| `--images_field` | `None` | (`--is_vlm`) column holding the row's image(s) for datasets that keep them outside the conversation; merged into the messages like the runtime path. An image column named by nothing is refused |
| `--min_pixels` / `--max_pixels` | `None` | VLM image pixel bounds |
| `--num_shards` | `1` | `1` writes an unsharded dataset that trains at any data-parallel size. Above `1` see [Sharded loading](#sharded-loading) |
| `--num_proc` | `HALO_DATASET_NUM_PROC` | Processes for dataset map. Unset, it resolves per host to `max(1, min(cpu_count // 4, 4))`; set `HALO_DATASET_NUM_PROC` to pin one value across nodes, since HF keys its map cache on `num_proc` |
| `--tokenizer_backend` | `hf` | `hf` = the model's tokenizer; `gigatoken` = Rust bulk encoder (see above) |
| `--test_size` | `None` | Test split fraction (e.g. `0.01`), cut from `train`. Refused when the input already carries a `test`/`validation` split — point `--input` at the train split alone. Without it (and without a `test` or `validation` split in the input) the output is train-only: an unsharded one trains with the first 100 train rows warned in as a placeholder test split, and `--num_shards` above `1` is refused before tokenization, since training rejects a sharded dataset with no test split |
| `--pad_token` / `--eos_token` / `--bos_token` | `None` | Override pad / EOS / BOS token. Recorded in `metadata.json` and re-checked at training time |
| `--chat_template` | `None` | Override chat template (Jinja2 string, or a path to a `.jinja`/`.jinja2`/`.j2` file). The resolved text is recorded and re-checked |
| `--overwrite` | `False` | Replace an existing output. Only the local writer keeps the previous dataset intact through a failed publish (see below) |
| `--dry_run` | `False` | Print the planned output paths and exit before the tokenizer or input dataset load — it validates nothing about the data |
| `--private` | `False` | Make Hub dataset private |
| `--hf_token` | `None` | Hub token for the `hf://` upload; the input and tokenizer loads use the ambient credentials (`HF_TOKEN` or the stored login) |
| `--verbose` | `False` | Log at DEBUG |

## Output structure

`--num_shards 1` (the default) writes a `DatasetDict.save_to_disk` directory with no shard index;
every rank loads it whole and the DataLoader splits the rows by data-parallel rank:

```text
s3://bucket/preprocessed/dataset/
├── metadata.json              # Preprocessing config and stats
├── dataset_dict.json
├── train/                     # Arrow files + dataset_info.json + state.json
└── test/                      # only when the input has a test/validation split or --test_size is set
```

`--num_shards > 1` writes per-split shards and a shard index:

```text
s3://bucket/preprocessed/dataset/
├── metadata.json              # Preprocessing config and stats
├── train/
│   ├── shard_index.json       # Shard mapping for distributed loading
│   ├── shard_0000/            # Arrow dataset shards
│   └── ...
└── test/
    ├── shard_index.json
    └── shard_0000/
```

`metadata.json` records the preprocessing config and stats; the authoritative field list is the
`PreprocessedDatasetMetadata` dataclass in `src/data/preprocessed/metadata.py`.

Its `version` stamp is compared on load: a stamp this build does not read raises
`IncompatiblePreprocessedDataset` naming the version, rather than being swallowed into a silent
fall-back to the raw (re-tokenizing) path. Unknown fields at a stamp this build *does* read raise the
same way (a diverged build). A `metadata.json` carrying no `preprocessed` key is somebody else's
file; the dataset is treated as raw with a warning.

One rank per input-filesystem scope runs the detection probe, so its answer is the scope's. Only a
confirmed absence reads as raw: no file, an S3 404, a Hub repo without it (offline, a cache without
it). A stamp that does not parse or a Hub it cannot reach raises on every rank instead. The one
tolerated failure is an S3 outage with no local mirror of the stamp, read as raw with a warning: an
absence is never mirrored, and a warm-cache raw dataset loads without S3.

Each split's `shard_index.json` (`ShardIndex`, `src/data/preprocessed/shard_index.py`) carries the same
`version` stamp and is held to it the same way: a stamp this build does not read, or a field it does
not know, raises the same `IncompatiblePreprocessedDataset` (defined in that leaf, so a loader can
refuse without importing the writer).

The index decides which shards each rank loads, so a diverged one read with this build's field
meanings hands ranks the wrong rows.

An index listing zero shards for `train` raises too, instead of handing every rank an empty split;
the same on a non-train split warns, since a tiny eval split that does not reach every rank is
legitimate.

`metadata.json` is also the **completion marker** — `is_preprocessed_dataset` keys on it — so it is
written last and a publish never leaves it beside an incomplete payload. A local `--overwrite` copies
into a sibling temp directory and swaps, leaving the previous dataset intact if the copy fails.

Hub and S3 `--overwrite` both delete the marker first and re-upload it after the payload, so an
interrupted upload reads as *raw* (a loud failure) rather than as a silently truncated preprocessed
dataset. The S3 writer never clears the prefix up front: it overwrites objects in place and sweeps
the ones the new tree does not carry only after the upload completed.

Training does not read a Hub-published dataset in place: `load_dataset` on a saved tree returns its
sidecar files as rows, so a preprocessed dataset named by its Hub id (`org/name`) is refused at startup.
Download it first (`hf download org/name --repo-type dataset --local-dir <dir>`) and point `dataset:`
at the directory.

## Sharded loading

Point the training config at the dataset (`dataset: "s3://my-bucket/preprocessed/my_dataset"`); each
data-parallel rank then loads only its own shards. How ranks split them, the
`num_shards >= data_parallel_size` requirement and what a starved rank does:
[Distributed Data Loading](../parallelism/data-loading.md#pre-processed-sharded-datasets).

Size `--num_shards` to the largest DP degree you train at; a multiple of it balances evenly. Example:
64 GPUs across 8 nodes (DP 64, no CP/TP) → `--num_shards 64`.

> [!WARNING]
> **Verify the effective shard count**
>
> `prepare_dataset` drops empty shards when a split has fewer examples than `--num_shards`, so the
> effective count can be smaller than requested — typically on the test split. It **warns at the end
> of preparation** naming the resulting ceiling ("this dataset can only train at
> `data_parallel_size <= N`"); only `shard_index.json` records the real per-split count
> (`num_shards=len(shards)`), while `metadata.json` keeps the requested value.
