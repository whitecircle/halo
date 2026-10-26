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
- **DPO / SMPO** — `chosen`, `rejected` (`List[Dict]`) and an optional `prompt` (`List[Dict]`, or a
  `str` that becomes a user turn). A missing prompt is the shared leading turns of chosen/rejected,
  and chosen/rejected that repeat the prompt keep only the continuation (`normalize_preference_row`,
  `src/data/pipeline/preferences.py`). Vision preference rows skip that normalization (the trainers render them
  themselves), so they arrive as three `List[Dict]` columns.
- **Offline GRPO** — `prompt: List[Dict]`, `completions: List[List[Dict]]`, `rewards: List[float]`;
  `len(rewards) == len(completions)`, variable group size. A full-fine-tuning KL run scores its own
  reference and needs a finite, unsharded dataset: pre-sharded KL inputs and a supplied grouped
  `ref_per_token_logps` column are refused (`agent-docs/training-methods/grpo/offline.md#reference-model`).
- **Async GRPO with environments** — `prompt`: the task as `str` **or** a ChatML list, which is
  reduced to its last `user` turn (a conversation with no user turn fails the batch on every rank);
  the environment builds the conversation itself. `answer: Any` (expected) is required only where
  the environment declares `requires_answer` — `code_contests` / `codeforces` take a hidden-test
  payload, `exam_qa` / `qa_search` / `react_*` the expected answer — refused at trainer construction
  when missing, over the train split and every eval split. `swe` needs it unless judge-only
  ([rule](../../agent-docs/training-methods/async-grpo/environments/tasks/swe.md#reward)); `native_*`
  and `mcp` need none. A ReAct preset given an `answer_validator` clears the requirement, and an
  explicit `environment_kwargs.requires_answer` overrides it, except that `swe` refuses `false`
  without a `test_function` while the reward has an `environment` term. Rename the columns with
  `prompt_field` / `answer_field` (a renamed answer still lands in the row as `answer`); an unknown
  name raises at startup.
- **Reward** — `chosen`, `rejected` (`List[Dict]`), optional `prompt` (implicit-prompt sets like
  Skywork-Reward keep the shared turns inside chosen/rejected) and `images` → TRL's Bradley-Terry
  columns `chosen_ids` / `rejected_ids` (+ optional `margin`).
- **KTO** — `prompt` and `completion` (`List[Dict]`), `label` (`bool`).
- **Classification** — `prompt: List[Dict]` or a raw-text column named by `text_field`, plus
  `label: str | List[str]` (multi-label); labels stringified and sorted. `-1` marks an unlabeled row:
  dropped from a multi-label set, refused in a single-label split the run reads
  (`agent-docs/training-methods/classification.md#dataset`).
- **Distillation** — standard SFT conversation field (default `messages`); over-length rows are
  dropped at `max_length` (`agent-docs/training-methods/distillation/teacher-distillation.md`).
- **Embedding** — positional, names carry no meaning: the label is the first of `label`, `labels`,
  `score`, `scores` present, every other column (in dataset order) a text input — pairs, triplets,
  scored or 0/1-labeled pairs, or labeled single texts (`agent-docs/training-methods/embedding.md#dataset-formats`).

## `prepare_dataset.py` (SFT-only offline prep)

Pipeline: load → tokenize (chat-template or raw text) → optional pack → optional shard → save
(`s3://` / `hf://org/name` / local). Other methods unsupported (`src/data/sources/loading.py`).

Flags (`scripts/before_training/prepare_dataset.py`) are the `PreprocessingConfig` fields plus the tool's
`PrepareDatasetArguments`, spelled like YAML keys (a YAML file of them works too): `--input`, `--output`,
`--model_name_or_path`, `--max_length` (8192); `--mode {chat,text}` (`chat`=SFT chat template, `text`=raw causal-LM for
continued pretraining), `--text_field`, `--append_eos` (`false` skips the EOS); `--conversation_field` (default
`prompt`), `--system_prompt`, `--model_supports_system_role` (`false` folds the system turn), `--tools_field`, `--interleaved_thinking` (keep
thinking blocks in non-final assistant turns; rejected with `--is_vlm`); `--train_on_completions_only`
(unset = on for `chat`, off for `text`; `false` bakes full-sequence labels), `--assistant_message_template`;
`--pack_sequences`, `--packing_strategy {bfd,bfd_split,wrapped}` (text SFT only, not VLM;
`bfd`=best-fit-decreasing, keeps document boundaries but **discards** the overflow past
`--max_length`; `bfd_split` splits the overflow into later examples — the lossless choice for
pre-training; `wrapped`=concatenate-and-chunk; VLM+packing rejected); `--is_vlm`, `--images_field`
(VLM only), `--min_pixels`, `--max_pixels`;
`--num_shards` (1=none, Megatron-style); `--num_proc` (default `None` → `HALO_DATASET_NUM_PROC`,
else `max(1, min(cpu_count//4, 4))`); `--tokenizer_backend {hf,gigatoken}`, `--test_size`; the tokenizer
overrides recorded in `metadata.json` and re-checked at training time — `--bos_token`, `--eos_token`,
`--pad_token`, `--chat_template` (Jinja2 string or a `.jinja`/`.jinja2`/`.j2` path); `--private`,
`--hf_token`, `--overwrite`, `--dry_run`, `--verbose`, `--trust_remote_code` (default **on** here).
`--tools_field` and `--interleaved_thinking` are both rejected with `--is_vlm`.

**Two distinct mechanisms** (don't conflate):

- **preprocessed** — output `metadata.json` has `preprocessed: true`; `load_datasets_auto()`
  (`src/data/sources/loading.py`) auto-detects via `is_preprocessed_dataset()`
  (`src/data/preprocessed/metadata.py`) and **skips tokenization**. No config flag needed — point `dataset:` at the output.
- **presharded** — sharded output has `shard_index.json`; the trainer computes
  `dataset_presharded = is_presharded_dataset_load(...)` and passes it so the DataLoader does **not**
  re-shard already-split data. `--num_shards` above 1 **must be ≥ data_parallel_size**: a short train split
  raises at load (`src/data/sources/sharded_dataset.py`), any other split warns — without the check
  a rank past the last shard would train on zero examples.

`patch_vocab.py` is model-prep, not dataset-prep (adds tokens, resizes embeddings, optional
`--reset_sinks`). Pass the multi-token patterns to promote to single tokens via `--patterns` (inline
JSON or a JSON file path).

## Loading internals (`src/data/sources/loading.py`)

- `load_dataset_from_source(path)` dispatches `s3` / `hf_hub` / `local` via `parse_dataset_source()`.
- `load_datasets(...)` handles a list + `dataset_ratio` (the corpus, `src/data/sources/corpus.py`: ratios,
  per-entry subsets, schema normalization), filters empty conversations, agrees the rows across ranks
  (`src/data/sources/replica_identity.py`), and stamps deterministic `_fingerprint` / `stamp_cache_key` keys for cross-rank cache stability — the
  stamp folds in the DP identity on the sharded single-path branch, so equal-length shards do not
  collide.
- `load_datasets_auto(...)` is the SFT entry; returns `(DatasetDict, is_preprocessed)`.
- `ShardedDatasetLoader` (`src/data/sources/sharded_dataset.py`): rank-specific shards; S3 shards
  cached per-shard under a FileLock keyed by `md5("<bucket>/<key>/<shard.path>")`.
- `is_sharded_dataset_across_ranks()` all-reduces the per-rank probe (MAX) so a transient per-rank S3
  flake can't split ranks onto sharded-vs-full paths (→ divergent sizes / NCCL hang).
- `fs_aware_main_first(tag)` / `is_input_shared_filesystem()`
  (`src/runtime/filesystem.py`): shared input FS
  (`DIST_INPUT_SHARED_FILESYSTEM`, falling back to the `DIST_SHARED_FILESYSTEM="1"` umbrella) →
  global rank 0 runs the body first, then the peers run it against the cache it filled; non-shared
  (`"0"`) → each node's local rank 0 leads its node, nodes in parallel. Waiters sit on a c10d store key bounded by
  `DIST_STORE_TIMEOUT_HOURS` (default 4 h), never inside a collective — the body is unbounded
  single-rank work and must itself issue no collective.

## Collator routing (`src/data/collators/factory.py` → `src/data/collators/`)

`select_data_collator(...)` guards (raise): packing+padding_free; CP+padding_free; CP+packing (the
CP attention path has no per-document boundaries, so packed documents would attend across each
other); `padding_free` on a non-varlen `attn_implementation`, and `packing` on a dense-mask backend
for a family whose forward drops `position_ids` (gpt-oss); packing/padding-free for the
GatedDeltaNet families without `causal_conv1d` + `fla>=0.2.2`, and on a compressed-KV model
(DeepSeek-V4's CSA/HCA layers, `reject_compressed_kv_rows`);
`train_on_completions_only` needs `assistant_message_template` (and it must occur in the rendered
chat template); `train_on_last_assistant_only` needs `train_on_completions_only`.

| Condition | Collator |
|---|---|
| CP + completions | `DataCollatorForCompletionOnlyLM` (pad_to_multiple_of=cp_size) |
| CP | `DataCollatorForCausalLMWithPadding` (pad_to_multiple_of=cp_size) |
| padding_free + completions | `DataCollatorWithFlatteningAndCompletionMask` (varlen FA2/FA3/FA4, flash-attn kwargs) |
| padding_free | `DataCollatorWithFlattening` (varlen FA2/FA3/FA4, `cu_seq_lens`) |
| packing + completions | `DataCollatorForCompletionOnlyLMWithPacking` |
| packing | `DataCollatorWithPacking` |
| completions only | `DataCollatorForCompletionOnlyLM` |
| else | `None` (default TRL) |

Not every packing anomaly raises: `packing` with `per_device_train_batch_size > 1` only **warns**
that rows are flattened, as does packing on a dense-mask backend for a family that keeps
`position_ids`.

`bfd` packing emits `seq_lengths` per packed doc; collators reset `position_ids` at each boundary and
build flash-attn `cu_seq_lens`. `wrapped` has no `seq_lengths` (cross-document attention — avoid with
FlashAttention). VLM: packing/padding-free both unsupported. `packing` / `padding_free` in YAML
choose the collator; after building it the scripts call `disable_trl_dataset_prep`
(`src/entrypoints/data_setup.py`), which clears both on the TRL config and sets
`dataset_kwargs: {skip_prepare_dataset: true}` so TRL keeps the script's collator. A custom script
must call it. A non-default `dataset_kwargs` (or `dataset_text_field`) is refused at startup
(`reject_trl_dataset_prep_args`).

## Footguns (source-cited)

- **Cache tokenizer-collision** (`src/data/pipeline/map_cache_key.py`, `_closure_fingerprint` /
  `function_cache_id`): `.map` processors capture tokenizer + `max_length` as closure free vars or
  as `functools.partial` bound args, both invisible to the source hash. Without folding them into the
  cache-file name, two runs sharing processor source but differing in tokenizer collide on one cache file
  → wrong token IDs → embedding OOB crash at step 0. The closure fingerprint covers closures;
  `function_cache_id` covers partials (the wrapped func id + bound args/keywords), such as the
  eval path's `partial(prepare_generative_row, tokenizer=…, max_length=…)`.
  The tokenizer term is keyed on **content**, not on `name_or_path`: the fast backend's serialized
  state (else the sorted vocab), plus the chat-template hash and the special-token ids and sides. A
  path term would miss the cache on every resume leg, since a resume repoints the load at the run's
  own checkpoint while the tokenization is unchanged — and it would also miss a `force_chat_template`
  or an in-vocab eos/bos/pad override, which change baked ids at identical vocab. The key also folds
  in dataset content and the transformers/TRL versions. A value it cannot fingerprint takes the
  one-time `Dataset-map cache fingerprint … skips a value of type …` warning — thread that value
  through `cache_key_extras`.
- **Rank-unstable HF fingerprints** — HF `_fingerprint`/`cache_files` diverge across ranks → each rank
  writes its own packed copy. The deterministic `stamp_cache_key` stamp is what keeps the
  key rank-stable; `coordinated_pack` packs once per filesystem scope under `run_load_rank_first`, store-joined.
- **Presharded re-shard trap** — a sharded dataset already gives each DP rank a disjoint slice; if
  `dataset_presharded` isn't passed, the DataLoader re-shards and drops ~(N-1)/N of each slice.
- **Offline completion masking is baked at preprocess time** (`src/data/preprocessed/tokenization.py`):
  `--train_on_completions_only` writes labels via `mask_batch_to_completion_spans`
  (`src/data/completion_spans.py`; every live collator labels its spans through the same
  `build_span_labels` rule; VLM ignores image tokens on top, masked after the span refill), since the preprocessed-dataset collators only pad
  the stored labels and cannot re-derive masking. The **chat template decides what the mask can
  reach**: gpt-oss's bundled harmony template closes a non-final assistant turn with `<|end|>`, which
  is not a completion-span terminator, so only the last turn of a multi-turn row would train — the
  SFT recipes pin `chat_template: jinja-templates/gpt-oss/gpt-oss-multiturn.jinja` with
  `force_chat_template: true` and `assistant_message_template: <|start|>assistant<|channel|>final<|message|>`
  instead. It fails fast before tokenization when the flag is set
  without `--assistant_message_template`, or set **explicitly** with `--mode text` (raw documents have
  no assistant turns; text mode otherwise defaults the flag off).
- **S3 pre-cache** (`src/data/sources/s3_dataset_store.py`; keys, markers and the cache root in
  `src/data/sources/dataset_cache.py`): `load_dataset(client, key)` caches under
  `<cache>/md5("<bucket>/<key>")/dataset` with a FileLock + `.download_complete` marker; a complete cache
  lets a relaunch survive an S3/SSO blip with no network, but is still validated against the object's
  live ETag and re-downloaded when an in-place re-preprocess moved it. The cache dir is `$HALO_DATA_ROOT/s3_datasets`
  (default `~/.cache/halo/s3_datasets`; `/mnt/s3_datasets` with `HALO_DATA_ROOT=/mnt`), or an explicit
  `HALO_S3_DATASET_CACHE_DIR` override.
