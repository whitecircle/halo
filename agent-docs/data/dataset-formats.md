# Dataset Formats for Training Scripts

Required columns per method. For Environmental GRPO see [Environmental GRPO](../training-methods/grpo/environmental-grpo.md); for [Distillation](../training-methods/distillation/index.md), teacher and self-distillation use the SFT format (self-distillation adds a privileged-answer field) and online SDPG uses the GRPO prompt/answer format.

Sources are S3, HuggingFace Hub (including the `:config` and `@split` suffixes), or local paths — see
[S3 Utilities](s3-utilities.md) for resolution rules and caching.

## Splits and combining

A dataset must have a `train` split; loading raises otherwise (with an `@split` suggestion). With no
`test` split the toolkit splits off `test_size` from train, or falls back to the first 100 samples.
Setting `test_size` on an **already-split** dataset re-splits it — train and test are concatenated
and re-divided, destroying the curated split. Pin the split you want with `@split` to keep
`test_size` splitting only that one (`openai/gsm8k:main@train`). A sharded or pre-processed dataset
carries the split decided at preparation time; `test_size` there is ignored with a warning.

Combine multiple datasets by passing lists; `dataset_ratio` is the per-dataset fraction kept:

```yaml
dataset:
  - "s3://my-bucket/safety/v1/train"
  - "HuggingFaceH4/ultrachat_200k"
dataset_ratio:
  - 1.0
  - 0.5
```

The pipeline loads each source, filters rows whose conversation field is `None` or `[]`,
ratio-subsets, normalizes the schema, and concatenates. A source that ships no `test` split
contributes training rows only — the corpus test split comes from the sources that ship one, and only
a corpus where none does falls back to a placeholder cut from its own train rows (set `test_size` to
carve a held-out split from every source instead). `_normalize_dataset_schema`
(`src/data/sources/loading.py`) keeps the intersection over train and test together: a column missing
from any dataset, or whose feature type differs across them, is dropped, with no fixed allowlist.

The **declared** render columns are the exception. `conversation_field` and `tools_field` are pinned
through the concatenation, an entry that lacks one getting a null-filled column of the carrying
entry's type, so a mixed corpus keeps them. A declared column the intersection loses anyway (a type
mismatch across the entries, which no fill can bridge) raises rather than rendering the whole corpus
without it.

**Which rows a ratio keeps** is drawn without replacement from `numpy.random.default_rng(seed)`
(PCG64) — seed `42`, `+1` for the `test` split and stepped per list entry, so every rank selects the
same rows with no exchange. Entries are subset independently and concatenated, with no cross-entry
dedup: the same source listed twice is drawn twice, and every row both draws keep lands twice.

The dataset fingerprint records the seed and the resulting sizes, never the row set, so a change of
RNG API, algorithm or per-split seed would swap the training corpus under an unchanged fingerprint.
`tests/cpu/data/test_dataset_ratio_seed.py` pins the exact selection to keep that a deliberate edit.

!!! warning "A list disables sharded and pre-processed loading"
    Every entry of a `dataset:` list is loaded **fully replicated** on every rank — passing DP
    rank/size would double-shard a sharded entry — and the pre-processed-dataset probe only runs for
    a single string path. Point `dataset:` at one path to use a pre-processed or sharded dataset.

## Required columns by method

| Method | Script | Columns | Types | Config field |
|--------|--------|---------|-------|--------------|
| SFT | `scripts/training/sft.py` | `prompt` | `List[Dict]` | `conversation_field` (default `"prompt"`) |
| SFT-VLM | `scripts/training/sft.py` | `prompt` | `List[Dict]` (multimodal) | `conversation_field` |
| DPO | `scripts/training/preference/dpo.py` | `prompt`, `chosen`, `rejected` | all `List[Dict]` | hardcoded |
| SMPO | `scripts/training/preference/smpo.py` | `prompt`, `chosen`, `rejected` | all `List[Dict]` | hardcoded |
| KTO | `scripts/training/preference/kto.py` | `prompt`, `completion`, `label` | `List[Dict]`/`str`, bool | `completion_field`, `label_field` |
| Offline GRPO | `scripts/training/offline_grpo.py` | `prompt`, `completions`, `rewards` | `List[Dict]`, `List[List[Dict]]`, `List[float]` | hardcoded |
| Classification | `scripts/training/classification.py` | `prompt` or `text_field`, `label` | `List[Dict]` or `str`, `str`/`List[str]` | `text_field` (label hardcoded) |
| Reward Model | `scripts/training/preference/rewards.py` | `chosen`, `rejected` (+ optional `prompt`, `images`) | all `List[Dict]` | `images_field` |

Every message is `{"role": "system"|"user"|"assistant", "content": str}` (or multimodal content for
VLM, below).

`conversation_field` must name a column the dataset actually carries: the loader checks the
`train`/`test` schemas at load and raises, naming the available columns. Without it a typo silently
no-ops the empty-conversation filter and surfaces much later as a `KeyError` inside the tokenizer map.

The check follows the *script*, not the YAML. SFT, prompt-tuning, distillation and both
prompt-rendering GRPO scripts (offline, environmental) declare a conversation column — each with its
own default, `prompt` for SFT and `messages` for distillation — so a dataset without it raises
whether or not the YAML names one; scripts that render no conversation (preference, reward,
classification, online GRPO, embedding) declare none and skip the check.

`tools_field` is checked the same way for a single dataset. Across a `dataset:` list it raises only
when **no** source carries the column and warns per source otherwise — a tool-use corpus concatenated
with plain chat is a legitimate shape, and only the rows of the sources without the column render
toolless. Pre-processed datasets skip both checks; their rows are already tokenized.

## SFT

```jsonl
{"prompt": [{"role": "user", "content": "What is machine learning?"}, {"role": "assistant", "content": "Machine learning is a subset of AI..."}]}
```

Multi-turn: include all turns in the single `prompt` list.

```yaml
conversation_field: "prompt"             # conversation column (default "prompt")
system_prompt: null                      # optional system prompt to prepend
model_supports_system_role: true         # if false, merge system into first user message
train_on_completions_only: true          # train only on assistant responses
train_on_last_assistant_only: false      # train only on the LAST assistant message
assistant_message_template: "<|start_header_id|>assistant<|end_header_id|>\n\n"  # completion masking
```

Pipeline: load → filter empties → apply chat template → tokenize (sequences over `max_length` are dropped) → optional packing. Datasets load with `keep_in_memory=False`, so Arrow files are memory-mapped, not copied into RAM.

Offline pre-processing (tokenization, packing, sharding) is optional, see [SFT Dataset Pre-Processing](dataset-preparation.md).

## SFT-VLM

Messages carry text and images:

```jsonl
{"prompt": [{"role": "user", "content": [{"type": "image", "image": "<image_data>"}, {"type": "text", "text": "What's in this image?"}]}, {"role": "assistant", "content": "A sunset over the ocean."}]}
```

The `image` field accepts: base64 data URI (`data:image/png;base64,...`), file path, PIL Image, or raw bytes. Text-only messages keep the plain `"content": str` form. A conversation stored as a JSON string parses the same as on the text path.

Hub datasets that store images in a separate column instead train via `images_field` (the column is injected into the first user turn), and the paired `{user, assistant}` turn shape used by the HuggingFaceM4 `texts` column is normalized to role/content messages — see [SFT — Vision-language models](../training-methods/sft.md#vision-language-models).

A run takes the VLM data path only when **both** halves agree (`is_vlm_run`, `src/data/vlm.py`): the checkpoint is multimodal, and the run declares image data. The model class follows the checkpoint alone, so a natively-multimodal checkpoint carrying text-only rows loads its multimodal class but trains through the text pipeline, packing included.

A run declares images three ways, any one of which is enough: `images_field` names the column; the dataset carries an `images`, `image` or `pixel_values` column; or the `conversation_field` column embeds `{"type": "image"}` content parts. The preference scripts have no `conversation_field`, so DPO declares by column only.

The embedded-parts declaration is read from the column's **Arrow schema**: the parts struct is typed over the whole split, so an image payload field — or a `datasets.Image` feature under any name — declares the run no matter which rows carry images. A conversation column stored as JSON strings has no struct to read and falls back to a probe of each split's first row.

The verdict is **agreed across ranks** (all-reduce MAX — any image-declaring rank wins), because on a pre-sharded corpus every rank probes a disjoint slice: its own first row, and the Arrow schema `datasets` inferred for its own shard set. A split verdict would pair one arm's barrier against the other's store wait, so the probe spends one collective per call whatever the answer.

The checkpoint half is `is_vlm_model`: a `model_type` registered under `AutoModelForImageTextToText` **whose mapped class does not end in `ForCausalLM`** (mistral4 and fuyu are registered there but decode text), or a config carrying a `vision_config`. This covers natively-multimodal models (Qwen3.5/3.6, Gemma3/4, Mistral3) with no name list. A config transformers **knows** decides both ways: a `model_type` present in `CONFIG_MAPPING` that matches neither test is text-only, and the name is never consulted.

The name-substring heuristic (`-vl`, `vl-`, `vl2`, `llava`, `vision`, `pixtral`, `molmo`, `idefics`, `paligemma`, `cogvlm`, `minicpm-v`, `qwen3.5-`, `qwen3.6-`) runs only when the config cannot load (offline, or a raw weights dir without `config.json`) or its `model_type` is unregistered (remote code). The hints match mid-word — a path containing `revision` reads as a VLM — so pass an explicit config where the path is uncontrolled.

For a JSON-string conversation column, the first-row fallback cannot see a mixed dataset whose probed rows are text-only: that run resolves to the text path and then **fails loud at the first image row**.

Every text renderer — SFT, preference prep (DPO/SMPO), Bradley-Terry reward prep, the GRPO generation-prompt renderer, generation eval, offline GRPO's templater and classification — refuses a conversation carrying an image part (`reject_image_content`) rather than expanding it into placeholder tokens with no pixels behind them, which would train vision placeholders as text in silence. The escape is an image column on the dataset, or offline `--vlm` preprocessing.

Limitations: packing and padding-free training are **NOT supported** on the VLM path (images cannot be packed). Offline pre-processing uses `--vlm`, see [SFT pre-processing](dataset-preparation.md).

## Preference (DPO/SMPO)

DPO and SMPO share the format below; full history goes in `prompt`.

```jsonl
{"prompt": [{"role": "user", "content": "Capital of France?"}], "chosen": [{"role": "assistant", "content": "Paris, the largest city in France..."}], "rejected": [{"role": "assistant", "content": "France capital is Paris I think."}]}
```

Hub shape variants normalize to this contract automatically (`normalize_preference_row`, `src/data/pipeline/preferences.py`): a plain-string `prompt` becomes a user turn; chosen/rejected that repeat the prompt turns (ultrafeedback_binarized, the Tulu-3 mixtures) have that prefix stripped; a missing `prompt` column (Skywork-Reward style) is extracted as the shared leading span of chosen/rejected.

Completions are rendered as `template(prompt + completion)` minus the rendered-prompt prefix, so strict chat templates (Qwen3.5) work and `prompt + chosen` always reconstructs the full conversation exactly.

- **DPO** requires a reference model (PEFT adapters act as the implicit reference; under EP use PEFT or `precompute_ref_log_probs`, under TP only `precompute_ref_log_probs` since PEFT is rejected there — the reference is not parallelized). Supports EP and TP; **CP not supported** (`concatenated_forward` needs full sequences).
- **SMPO** is reference-model-free. Supports EP and TP for both modalities, CP for text only. VLM mode (any multimodal model — the trainer normalizes rows itself, so hub-shape variants work) takes the same rows plus an optional `images`/`image` column; `DataCollatorForVLMSMPO` processes images at collation, and padding-free is text-only. See [SMPO — Vision-language](../training-methods/preference/smpo.md#vision-language).

**Vision DPO/KTO.** An `images`/`image` column routes to TRL's vision collators, and the rows must ALREADY be contract-shaped (prompt = message list, chosen/rejected = continuation-only) — the hub-shape normalization above runs only on the text path. Modality routing keys on the **dataset**, so a natively-multimodal model with text-only preference data trains through the normal text pipeline.

TRL rejects `precompute_ref_log_probs` for vision datasets, so vision DPO under EP uses standard-PEFT adapters: an **expert-only** EP adapter requires precomputed ref logps and is therefore text-only, while a mixed attention+expert adapter keeps the implicit reference. Under TP, where PEFT and an explicit reference are both rejected, vision DPO has no supported shape.

## Offline GRPO

```jsonl
{"prompt": [{"role": "user", "content": "Write a haiku about programming."}], "completions": [[{"role": "assistant", "content": "Code flows like water..."}], [{"role": "assistant", "content": "print hello world"}]], "rewards": [0.95, 0.23]}
```

- Variable group sizes allowed (1 to 1000+ completions per prompt).
- Provide one reward per completion. A length mismatch raises `ValueError` naming the offending row, rather than silently truncating to the shorter list.
- Each group's rewards become group-relative advantages via `advantage_method` (default `quantile_norm`).

Config in `OfflineGRPOConfig` (`src/configs/offline_grpo_config.py`); the values below are the defaults:

```yaml
max_prompt_length: 512
max_completion_length: null      # null = no cap
kl_beta: 0.0                     # KL penalty coefficient
best_completion_emphasis: 0.0    # extra weight for best completion
advantage_method: quantile_norm  # z_norm | minmax | quantile_norm | quantile_uniform | robust
loss_type: bnpo                  # grpo | bnpo | dr_grpo
```

**CP not supported** — uses the `logits_to_keep` optimization, incompatible with sequence splitting.

## Classification

Single-label uses a string; multi-label uses a list. The row carries a `prompt` conversation, or a raw text column named by `text_field` (e.g. imdb's `text`), which is wrapped as one user turn:

```jsonl
{"prompt": [{"role": "user", "content": "Best purchase ever!"}], "label": "positive"}
{"text": "A romantic comedy about time travel.", "label": ["comedy", "romance", "sci-fi"]}
```

Labels are collected from the training data, stringified, and sorted alphabetically; the label set fixes the head's `num_labels` and shape (the config field is derived, never read from YAML). Labels seen only in validation/test are added with a warning. `-1` is dropped from the label set — it is the cross-entropy ignore sentinel on a row, not a class; the rows themselves are kept.

```yaml
dataset: "path/to/classification/dataset"
text_field: text   # only when rows have no 'prompt' conversation
max_length: 2048
```

**CP not supported** — needs pooled representations from the complete sequence.

## Reward modeling

Same format as preference data. There is **no** pre-tokenization pass: the rows go straight to
`DistributedRewardTrainer` (over TRL's `RewardTrainer`), which chat-templates and tokenizes
`chosen`/`rejected` itself and trains the Bradley-Terry loss. Implicit-prompt datasets (shared
leading turns inside `chosen`/`rejected`, e.g. Skywork-Reward) therefore work with no `prompt`
column. Rows over `max_length` are filtered by the trainer, not truncated. `tools_field` is aliased onto the `tools` column the trainer renders.

**CP not supported** — pooled representations are incompatible with sequence splitting.

`dataset_num_proc` (parallel map processes, default `None`) is a field on the non-SFT method configs
(SMPO, Classification, Distillation, Offline GRPO); SFT inherits it from TRL's `SFTConfig`. Unset
means the toolkit default (`DATASET_NUM_PROC`), not one worker, and the count is the same
under every parallelism mode.

Trainer-side map/filter callables must be **module-level functions** taking their state through
`fn_kwargs` — never bound methods. At `num_proc > 1` `datasets` ships the callable to worker
processes through dill, which pickles a bound method's whole `self`: the model, and under EP the
DeepEP/NCCL process groups (unpicklable — the map dies mid-pass). `reject_self_capturing_fn`
enforces this at `coordinated_map` / `coordinated_filter`. Passing tunables through `fn_kwargs` also
puts them in the cache key, which values read off `self` were invisible to.

Those two helpers carry their own rank ordering and must never be nested in a main-first block — see
[Filesystem Handling](filesystem-handling.md#coordination-primitives).

## Chat templates

```yaml
chat_template: "jinja_templates/gpt-oss-instruct.jinja"   # file path or inline Jinja string
force_chat_template: true                                 # override model's own template

pad_token: "<|endoftext|>"
eos_token: "<|im_end|>"
bos_token: null
added_special_tokens:
  - "<special1>"

model_supports_system_role: false   # merge system message into first user message
```

### Special-token ownership

Rendered chat-template text tokenizes through `tokenize_rendered`
(`src/data/pipeline/rendered.py`), which probes each tokenizer once — tokenizing `""` with
`add_special_tokens=True` — to learn what its post-processor actually adds. `bos_token` alone is
unreliable: gpt-oss and Bailing declare a nominal BOS their post-processor never emits, and Zaya
appends a trailing `<|im_end|>` instead of prepending BOS.

- A rendered leading BOS is stripped only when the post-processor re-adds it, so specials appear
  exactly once and no BOS is forced onto a tokenizer that never emits one (gemma-4's
  template-emitted BOS survives).
- Tokenizer-appended trailing specials stay on training rows, unless the render already ends with
  the same terminator sequence (Zaya) — then the duplicate is stripped.
- They are always stripped from generation prompts (`for_generation=True`), so a prompt never ends
  with a turn terminator.

This seam covers SFT (runtime and offline), classification, the prompts-reward preprocess,
teacher/self-distillation, and generation-eval prompts. The SMPO and offline-GRPO prompt paths use
the same probe to prepend BOS only when the post-processor owns it.

`render_generation_prompt` — the prompt stage the online/environmental GRPO scripts share — applies
the same rule to the rendered TEXT it returns, and measures `max_prompt_length` with the generation
tokenization the row will really carry.
