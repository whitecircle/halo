# data — reference

Per-method schema detail, `prepare_dataset.py` flags, loading internals, collator matrix, and footguns.
Grounded in the code — verify against the cited files before acting (they are the source of truth).
Use placeholder URIs (`s3://<bucket>/<key>`); never hardcode an internal bucket or dataset key.

## Dataset schemas (detail)

Authoritative: `agent-docs/data/dataset-formats.md`. Messages are OpenAI ChatML
`{"role": "system|user|assistant", "content": str}`. VLM content may be
`[{"type":"image","image":...},{"type":"text","text":...}]`.

- **SFT / SFT-VLM** — `prompt: List[Dict]` (field = `conversation_field`). VLM runs through the same
  `scripts/training/sft.py`: images ride embedded in message content or in an `images_field` column
  (`agent-docs/training-methods/sft.md#vision-language-models`).
- **DPO / SMPO** — `prompt`, `chosen`, `rejected`, all `List[Dict]`.
- **Offline GRPO** — `prompt: List[Dict]`, `completions: List[List[Dict]]`, `rewards: List[float]`;
  `len(rewards) == len(completions)`, variable group size.
- **Environmental GRPO** — `prompt: str` (task), `answer: str` (expected). NOT ChatML.
- **Reward** — `chosen`, `rejected` (`List[Dict]`), optional `prompt` (implicit-prompt sets like
  Skywork-Reward keep the shared turns inside chosen/rejected) and `images` → TRL's Bradley-Terry
  columns `chosen_ids` / `rejected_ids` (+ optional `margin`).
- **Classification** — `prompt: List[Dict]`, `label: str | List[str]` (multi-label); labels sorted
  alphabetically, `-1` reserved/filtered.
- **Distillation** — standard SFT conversation field (default `messages`); over-length rows are
  dropped at `max_length` (`agent-docs/training-methods/distillation/teacher-distillation.md`).
- **Embedding** — column-detected: pairs (`anchor`,`positive`), triplets (+`negative`), scored pairs
  (`sentence1`,`sentence2`,`score`), binary pairs (+`label` 0/1), labeled texts (`text`,`label`)
  (`agent-docs/training-methods/embedding.md`).

## `prepare_dataset.py` (SFT-only offline prep)

Pipeline: load → tokenize (chat-template or raw text) → optional pack → optional shard → save
(`s3://` / `hf://org/name` / local). Other methods unsupported (`src/data/sources/loading.py`).

Flags (`scripts/before_training/prepare_dataset.py`): `--input/-i`, `--output/-o`, `--model-name/-m`,
`--max-length` (8192); `--mode {chat,text}` (`chat`=SFT chat template, `text`=raw causal-LM for
continued pretraining), `--text-field`, `--no-append-eos`; `--conversation-field` (default
`prompt`), `--system-prompt`, `--no-system-role`, `--tools-field`, `--interleaved-thinking` (keep
thinking blocks in non-final assistant turns; rejected with `--vlm`); `--train-on-completions-only`
(`BooleanOptionalAction`; unset = on for `chat`, off for `text`), `--assistant-message-template`;
`--pack-sequences`, `--packing-strategy {bfd,bfd_split,wrapped}` (text SFT only;
`bfd`=best-fit-decreasing, keeps document boundaries but **discards** the overflow past
`--max-length`; `bfd_split` splits the overflow into later examples — the lossless choice for
pre-training; `wrapped`=concatenate-and-chunk; VLM+packing rejected); `--vlm`, `--min-pixels`,
`--max-pixels`;
`--num-shards` (1=none, Megatron-style); `--num-proc` (default `None` → `HALO_DATASET_NUM_PROC`,
else `max(1, min(cpu_count//4, 4))`); `--tokenizer-backend {hf,gigatoken}`, `--test-size`; the tokenizer
overrides recorded in `metadata.json` and re-checked at training time — `--bos-token`, `--eos-token`,
`--pad-token`, `--chat-template` (Jinja2 string or a `.jinja`/`.jinja2`/`.j2` path); `--private`,
`--hf-token`, `--overwrite`, `--dry-run`, `--verbose/-v`.

**Two distinct mechanisms** (don't conflate):
- **preprocessed** — output `metadata.json` has `preprocessed: true`; `load_datasets_auto()`
  (`src/data/sources/loading.py`) auto-detects via `is_preprocessed_dataset()` and **skips
  tokenization**. No config flag needed — just point `dataset:` at the output.
- **presharded** — sharded output has `shard_index.json`; the trainer computes
  `dataset_presharded = is_presharded_dataset_load(...)` and passes it so the DataLoader does **not**
  re-shard already-split data. `--num-shards` **must be ≥ data_parallel_size**: a short train split
  raises at load (`src/data/sources/sharded_dataset.py`), any other split warns — without the check
  a rank past the last shard would train on zero examples.

`patch_vocab.py` is model-prep, not dataset-prep (adds tokens, resizes embeddings, optional
`--reset_sinks`). Pass the multi-token patterns to promote to single tokens via `--patterns` (inline
JSON or a JSON file path).

## Loading internals (`src/data/sources/loading.py`)

- `load_dataset_from_source(path)` dispatches `s3` / `hf_hub` / `local` via `parse_dataset_source()`.
- `load_datasets(...)` handles a list + `dataset_ratio`, filters empty conversations, normalizes schema,
  and stamps deterministic `_fingerprint` / `_toolkit_cache_key` for cross-rank cache stability.
- `load_datasets_auto(...)` is the SFT entry; returns `(DatasetDict, is_preprocessed)`.
- `ShardedDatasetLoader` (`src/data/sources/sharded_dataset.py`): rank-specific shards; S3 shards
  cached per-shard under a FileLock keyed by `md5("<bucket>/<key>/<shard.path>")`.
- `is_sharded_dataset_coordinated()` all-reduces the per-rank probe (MAX) so a transient per-rank S3
  flake can't split ranks onto sharded-vs-full paths (→ divergent sizes / NCCL hang).
- `fs_aware_main_first(tag)` (`src/distributed/filesystem.py`) / `is_input_shared_filesystem()`
  (`src/distributed/runtime.py`): shared input FS
  (`DIST_INPUT_SHARED_FILESYSTEM`, falling back to the `DIST_SHARED_FILESYSTEM="1"` umbrella) → only global rank 0 downloads/processes; non-shared (`"0"`)
  → each node's local rank 0 acts, nodes in parallel. Waiters sit on a c10d store key bounded by
  `DIST_STORE_TIMEOUT_HOURS` (default 4 h), never inside a collective — the body is unbounded
  single-rank work and must itself issue no collective.

## Collator routing (`src/data/collators/factory.py` → `src/data/collators/`)

`select_data_collator(...)` guards (raise): packing+padding_free; CP+padding_free; CP+packing (the
CP attention path has no per-document boundaries, so packed documents would attend across each
other); `padding_free` on a non-varlen `attn_implementation`, and `packing` on a dense-mask backend
for a family whose forward drops `position_ids` (gpt-oss); packing/padding-free for the
GatedDeltaNet families without `causal_conv1d` + `fla>=0.2.2`, and GDN packing under PP;
`train_on_completions_only` needs `assistant_message_template` (and it must occur in the rendered
chat template); `train_on_last_assistant_only` needs `train_on_completions_only`.

| Condition | Collator |
|---|---|
| CP + completions | `DataCollatorForCompletionOnlyLM` (pad_to_multiple_of=cp_size) |
| CP | `DataCollatorForCausalLMWithPadding` (pad_to_multiple_of=cp_size) |
| padding_free + completions | `DataCollatorWithFlatteningAndCompletionMask` (FA2, flash-attn kwargs) |
| padding_free | `DataCollatorWithFlattening` (FA2, `cu_seq_lens`) |
| packing + completions | `DataCollatorForCompletionOnlyLMWithPacking` |
| packing | `DataCollatorWithPacking` |
| completions only | `DataCollatorForCompletionOnlyLM` |
| else | `None` (default TRL) |

`bfd` packing emits `seq_lengths` per packed doc; collators reset `position_ids` at each boundary and
build flash-attn `cu_seq_lens`. `wrapped` has no `seq_lengths` (cross-document attention — avoid with
FlashAttention). VLM: packing/padding-free both unsupported. TRL caveat: set `padding_free=False` +
`dataset_kwargs={"skip_prepare_dataset": True}` so TRL doesn't reject the custom collators.

## Footguns (source-cited)

- **Cache tokenizer-collision** (`src/data/pipeline/processing.py`, `_get_closure_fingerprint` /
  `get_function_identifier`): `.map` processors capture tokenizer + `max_length` as closure free vars or
  as `functools.partial` bound args, both invisible to the source hash. Without folding them into the
  cache-file name, two runs sharing processor source but differing in tokenizer collide on one cache file
  → wrong token IDs → embedding OOB crash at step 0. The closure fingerprint covers closures;
  `get_function_identifier` covers partials (the wrapped func id + bound args/keywords) — the eval path's
  `partial(prepare_generative_row, tokenizer=…, max_length=…)` evaded the closure check otherwise.
- **Rank-unstable HF fingerprints** — HF `_fingerprint`/`cache_files` diverge across ranks → each rank
  writes its own ~45 GB packed copy. The deterministic `_toolkit_cache_key` stamp is what keeps the
  key rank-stable; `pack_dataset_coordinated` packs once on the main rank under `fs_aware_main_first`.
- **Presharded re-shard trap** — a sharded dataset already gives each DP rank a disjoint slice; if
  `dataset_presharded` isn't passed, the DataLoader re-shards and drops ~(N-1)/N of each slice.
- **Offline completion masking is baked at preprocess time** (`src/data/pipeline/preprocessing.py`):
  `--train-on-completions-only` writes labels via `mask_batch_to_completion_spans` (`src/data/spans.py`
  — the one implementation the live collators and the runtime label builder also call; VLM ignores image
  tokens on top, masked after the span refill), since the preprocessed-dataset collators only pad
  the stored labels and cannot re-derive masking. It fails fast before tokenization when the flag is set
  without `--assistant-message-template`, or with `--mode text` (raw documents have no assistant turns).
- **S3 pre-cache** (`src/data/sources/s3_client.py`): `S3Client.load_dataset` caches under
  `<cache>/md5("<bucket>/<key>")/dataset` with a FileLock + `.download_complete` marker; a complete cache
  lets a relaunch survive an S3/SSO blip with no network. The cache dir is `$HALO_DATA_ROOT/s3_datasets`
  (default `~/.cache/halo/s3_datasets`; `/mnt/s3_datasets` with `HALO_DATA_ROOT=/mnt`), or an explicit
  `HALO_S3_DATASET_CACHE_DIR` override.
