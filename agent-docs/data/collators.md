# Collators

Collators turn tokenized samples into batches: padding, sequence packing, completion-only label
masking, and per-sequence position IDs. They live in `src/data/collators/`. `factory.py` selects
among the SFT collators of `completions_only.py` and `packing.py`, which share the span resolver in
`src/data/spans.py`. The per-method leaves — `smpo.py`, `vlm.py`, `self_distill.py`, `classification.py`,
`offline_grpo.py`, `vlm_preference.py` — are built by their own trainers and scripts,
and `fixed_shape.py` wraps any of them for a consumer whose buffer shapes freeze on the first batch.

| Collator | Packing | Padding-Free | Completion Masking | Position IDs |
|----------|---------|--------------|-------------------|--------------|
| `DataCollatorForCompletionOnlyLM` | No | No | Yes | No |
| `DataCollatorWithPacking` | Yes | No | No | reset per sequence |
| `DataCollatorForCompletionOnlyLMWithPacking` | Yes | No | Yes | reset per sequence |
| `DataCollatorWithFlattening` | No | Yes | No | reset per sequence |
| `DataCollatorWithFlatteningAndCompletionMask` | No | Yes | Yes | reset per sequence |
| `DataCollatorForCausalLMWithPadding` | No | No | No | No |

The last one is the CP-mode padded route; it preserves precomputed `labels` (the parent
`DataCollatorForLanguageModeling` rebuilds them from `input_ids` and crashes on ragged ones).

You rarely construct these directly — `select_data_collator()` picks one from your YAML config.

## Collator classes

Constructor notes that matter when you build one directly:

- `response_prompt_template` (`str` or `List[int]`, the assistant-header template) is required for
  `DataCollatorForCompletionOnlyLM` and `DataCollatorForCompletionOnlyLMWithPacking`. It is optional
  for `DataCollatorWithFlatteningAndCompletionMask`, where `None` skips completion masking. An
  **empty** template raises — it would match every position and train on all tokens.
- `tokenizer` is required for every collator except `DataCollatorWithFlattening`, which defaults it
  to `None` and never uses it.
- `pad_to_multiple_of` is set to `cp_size` under CP; the masked-label fill value is `-100`
  throughout.

**`DataCollatorForCompletionOnlyLM`** — masks loss on everything except assistant completion tokens (user/system tokens get `ignore_index`). Extends `DataCollatorForLanguageModeling`. If `response_prompt_template` is not found in a sequence, that sequence is dropped from the loss.

**`DataCollatorWithPacking`** — packs multiple sequences per sample. Position IDs reset at each boundary; labels at `position_ids == 0` are set to `-100` so the model is not penalized for predicting the next document's first token across a boundary.

It drops the dense `attention_mask` so Flash Attention builds a per-document block-diagonal `cu_seqlens` from the resetting position IDs, and flattens the packed mini-batch to a single `[1, total_tokens]` row of **real tokens only** — transformers' packed-sequence detection only engages at batch size 1, so a `[B>1, L]` batch would silently fall back to dense-causal attention.

Inter-row padding is dropped in the flatten: a pad carries position 0, so every kept pad would be its own varlen segment, and the FA4 backward pays a fixed per-segment cost (a mostly-empty partial pack co-batched with a full row measured 275× — 6.9 s vs 25 ms per layer). Where pads must survive (pipeline parallelism's fixed shapes, `pad_to_multiple_of`), the tail's position IDs are a ramp restarting every `PAD_TAIL_SEGMENT_CHUNK` (256) tokens, keeping it a handful of no-op segments.

Whether the packed documents actually stay isolated is **per family**, not universal — see [Document isolation under packing](#document-isolation-under-packing). Where the attention path is isolated, the remaining difference between backends is cost: the non-varlen path builds a dense `[L, L]` mask over the *flattened* row, so `L` is the whole batch's token count (up to `per_device_train_batch_size × max_length`) and the mask exhausts memory at long context. `select_data_collator` **warns** there, telling you to keep `per_device_train_batch_size` at 1.

It detects packed data via the `seq_lengths` field that TRL's `pack_dataset()` emits for the `bfd` and
`bfd_split` strategies. The `wrapped` strategy emits no `seq_lengths` column at all, so each row
collates as **one** document and anything concatenated inside it attends across itself; the collator
warns once per instance when it sees such rows. Online packing (`packing: true` at training time) and
the offline `prepare_dataset.py` pipeline both accept all three strategies
(see [Pre-Processing](dataset-preparation.md#parameters)).

**`DataCollatorForCompletionOnlyLMWithPacking`** — packing plus completion masking; masks each packed
sub-sequence independently, falls back to standard completion masking for non-packed input.

**`DataCollatorWithFlattening`** — padding-free. Flattens the batch into one `[1, total_tokens]`
sequence and emits Flash Attention varlen kwargs. Output keys: `input_ids`, `labels`, `position_ids`
(reset per sequence), `cu_seq_lens_q`/`cu_seq_lens_k`, `max_length_q`/`max_length_k`. The first token
of every sub-sequence is label-masked (the causal shift cannot predict it from the previous
document).

**`DataCollatorWithFlatteningAndCompletionMask`** — padding-free plus completion masking; the lowest-overhead collator for variable-length SFT trained on assistant responses only.

## Factory: `select_data_collator()`

`src/data/collators/factory.py` picks the collator from config flags and validates compatibility:

```python
from src.data.collators.factory import select_data_collator

collator = select_data_collator(
    tokenizer=tokenizer,
    padding_free=True,
    train_on_completions_only=True,
    assistant_message_template="<|im_start|>assistant\n",
)
```

Flags: `padding_free=False`, `packing=False`, `train_on_completions_only=False`,
`assistant_message_template=None`, `pad_to_multiple_of=None`, `use_context_parallel=False`,
`train_on_last_assistant_only=False`, `model_config=None`, `per_device_train_batch_size=1`,
`keeps_packed_rows=False`. The last two exist for pipeline parallelism (a seam — PP
itself is [not yet available](../parallelism/pipeline-parallelism.md)): a batch above 1 warns that
packed/flattened rows merge documents, and `keeps_packed_rows` (set when `pp_size > 1`) suppresses
that warning because a pipeline splits the packed rows back into microbatches.

**Pass `model_config`.** It is what feeds `resolve_eos_token_ids(tokenizer, model_config)`; without
it the collator falls back to the tokenizer's EOS alone and mis-masks any family whose turn
terminators live in `config.eos_token_id` (GLM-4). The SFT script passes it.
`train_on_last_assistant_only=True` requires `train_on_completions_only=True` and raises otherwise.

| Configuration | Collator |
|---------------|----------|
| CP + packing | **rejected** — the Ulysses CP attention path has no per-document boundaries, so packed documents would attend across each other |
| CP + completion masking | `DataCollatorForCompletionOnlyLM` (CP mode) |
| CP | `DataCollatorForCausalLMWithPadding` (a label-preserving `DataCollatorForLanguageModeling` subclass in `packing.py`) |
| padding_free + completion masking | `DataCollatorWithFlatteningAndCompletionMask` |
| padding_free | `DataCollatorWithFlattening` |
| packing + completion masking | `DataCollatorForCompletionOnlyLMWithPacking` |
| packing | `DataCollatorWithPacking` |
| completion masking only | `DataCollatorForCompletionOnlyLM` |
| none | `None` (default TRL behavior) |

YAML fields (`packing`, `packing_strategy`, `padding_free`, `eval_packing` come from TRL's `SFTConfig`; `train_on_completions_only` and `assistant_message_template` from `ConversationRenderArguments` in `src/args/mixins.py`; `train_on_last_assistant_only` from `src/args/sft_args.py`):

```yaml
padding_free: false
packing: true
train_on_completions_only: true
assistant_message_template: "<|start_header_id|>assistant<|end_header_id|>\n\n"
train_on_last_assistant_only: false
```

The SFT script defaults `train_on_completions_only: true`; `assistant_message_template` has no default and must be the tokenizer's exact assistant-header encoding (the ChatML form above for Qwen; see [Constraints](#constraints)) — a missing or non-rendering marker raises at startup.

For a pre-processed dataset the SFT script derives the flags from the data, not the config: `packing` follows `metadata.packed` (offline-packed chunks still need the packing collator's per-document position IDs), and completion masking is disabled at runtime because the baked `labels` are authoritative — see [SFT Dataset Pre-Processing](dataset-preparation.md).

## Document isolation under packing

Packing only pays off if document B's logits do not move when document A's content changes. That
drift is measured per family (0 = isolated) and the verdict is **not** universal: the collator emits
correct `position_ids` everywhere, but a family's forward has to carry them into its mask, and a
non-attention mixer carries state along the sequence regardless of any mask.

| Family | Attention layers | Other layers |
|---|---|---|
| Qwen3 (dense + MoE), GLM-4 MoE Lite, Laguna, Gemma 4 | isolated on their supported backends | — |
| Inkling | isolated | four depthwise causal convs per layer **cross by construction** — the modeling reads `seq_idx` but its call sites never forward kwargs, so no emission can reach them |
| Mistral4 | isolated — flash needs `patch_mistral4_flash_packed_position_ids` (upstream drops `position_ids` before the attention interface); dense clean to grouped-GEMM reduction noise | — |
| GPT-OSS | isolated on flash only; eager/SDPA/flex **leak** — the model's mask kwargs omit `position_ids`, so the packed row runs as one dense causal sequence | — |
| DeepSeek-V4 (eager-only) | isolated on the training path (`use_cache=False`; a live cache suppresses the packed mask) | the CSA/HCA compressors pool KV across the whole row — **cross by construction** |
| Qwen3.5 / 3.6, Qwen3-Next | isolated | GatedDeltaNet reads `seq_idx` (conv) + `cu_seq_lens_q` (delta rule): the collators emit both for the family, so its boundaries reach the kernels — but only the `fla` / `causal-conv1d` fast paths consume them (both installed in the images). The torch fallbacks take neither, so a multi-document row would cross in the conv *and* the scan — the factory refuses `packing` / `padding_free` unless transformers' own fast-path predicates hold |
| Bailing / Ling | remote code; see [Bailing/Ling](../models/bailing.md) | KDA linear attention — **crosses**. The KDA op accepts a `cu_seqlens` kwarg, but the model forward never threads kwargs down to it, so there is no reachable boundary parameter — the same class of crossing as Inkling's convs |
| GLM-5 Next | both layer types receive the 2D padding mask `create_recurrent_attention_mask` builds, so no packed-boundary parameter reaches the DSA attention either | KDA linear attention — **crosses**, the same unreachable-boundary class as Bailing/Ling |
| LFM-2 | isolated | ShortConv is isolated exactly: the collators emit `seq_idx` for this family and both conv paths honor it |
| Zaya | isolated — flash needs `patch_zaya_flash_packed_position_ids` | CCA convolution + delayed-value recurrence — **cross by construction**, amplifying with depth |
| Cohere2 MoE | isolated on every backend, no patch needed — the forward feeds `position_ids` into mask construction and through layer kwargs to the attention interface | — |
| Step-3.7 Flash | isolated on SDPA and eager, no patch needed — the forward feeds `position_ids` into both mask constructions (full and sliding); bit-exact through dense layers, MoE layers add expert-summation reduction noise (~1e-7 fp32). Training path only: a live cache suppresses the packed mask (`use_cache=False`, as DeepSeek-V4) | — |

The GPT-OSS leak is the one case the toolkit refuses outright: `select_data_collator` raises for
`DENSE_PACKING_LEAK_MODEL_TYPES` (`src/data/collators/factory.py`) when packing is asked for on a
non-varlen backend. Pin `flash_attention_2` or turn packing off. The refusal reads the text
sub-config too, so a composite (VLM) wrapper around a leaking family is covered.

The GatedDeltaNet families (`qwen3_5*`, `qwen3_next*`) carry two more refusals, both about markers
that would be emitted but not read:

- **Missing kernels.** transformers selects its segment-aware linear-attention kernels at
  modeling-import time via `is_causal_conv1d_available()` / `is_flash_linear_attention_available()`
  — which require the package installed, `fla >= 0.2.2`, **and** a CUDA-capable torch. The torch
  fallbacks it takes otherwise drop `seq_idx` and `cu_seq_lens_q`, so conv and recurrent state cross
  document boundaries while attention stays isolated — invisible in the loss. `packing` and
  `padding_free` are both refused unless those same predicates hold, so the refusal cannot disagree
  with the kernels actually selected (the production images satisfy them).
- **Pipeline parallelism** ([not yet available](../parallelism/pipeline-parallelism.md)) — its
  collator seam keeps the packed rows instead of flattening them, and the delta rule's varlen
  `cu_seq_lens` have no per-row convention, so the conv would isolate while the scan crossed
  documents.

"Crosses by construction" applies where the mixer has no per-document boundary parameter at
all (Zaya's CCA, DeepSeek-V4's compressors, Inkling's convs). Pack such a family only where a
small amount of cross-document mixing is acceptable, and prefer padded batches when it is not.

Under distillation the teacher and the frozen reference forward with `use_cache=False` explicitly, so
they score packed rows under the same masks the student does.

## Constraints

- **`packing` and `padding_free` are mutually exclusive.** Packing pads concatenated sequences to
  `max_length`; padding-free flattens to one variable-length tensor. Pick packing for short
  sequences, padding-free for long/variable ones.
- **Both want a varlen Flash Attention backend.** The gate is the model's **resolved**
  `_attn_implementation` against `VARLEN_ATTN_IMPLEMENTATIONS` (`flash_attention_2/3/4`,
  `src/models/patches/attention.py`) — not a model-family list. Off it, `padding_free`
  **raises** (its `cu_seq_lens` kwargs have no consumer) and `packing` **warns** (it pays a dense
  `[L, L]` mask per layer) — except for a family whose dense path leaks documents, where packing
  raises too ([Document isolation](#document-isolation-under-packing)). DeepSeek-V4 (eager-only:
  head_dim 512 exceeds FA's 256 cap) and Gemma 4 land on the warning — use padded batches.
  See [Padding-Free Collator](../optimization/padding-free-collator.md).
- **Padding-free is incompatible with Context Parallelism**, and **packing is rejected under CP**.
  Under CP use the padded collator, which sets `pad_to_multiple_of=cp_size`.
- **Pipeline parallelism** is [not yet available in this release](../parallelism/pipeline-parallelism.md);
  its shipped collator gates already reject padding-free (a flattened single row cannot split into
  microbatches) and pin packing at one packed row per microbatch.
- **Completion masking requires `assistant_message_template`** matching the tokenizer's exact
  assistant-header encoding; unmatched sequences are dropped from the loss. A marker the chat
  template never renders is refused by `select_data_collator` — it would mask every label and the run
  would train zero tokens at loss ≈ 0. The probe renders both a plain and a `thinking` assistant turn,
  so a marker gpt-oss harmony emits only under thinking passes it and is caught per batch instead.

## EOS, pad, and the response template

**The turn terminator is the resolved terminator set, not just `tokenizer.eos_token_id`.** A turn
ends at the first token in the union of the model config's `eos_token_id` (scalar or list;
`text_config.eos_token_id` for VLM/composite configs) and the tokenizer's EOS
(`resolve_eos_token_ids`, `src/data/spans.py`). This is required for templates that delimit
turns with role markers instead of a per-turn `<|endoftext|>` — GLM-4 ends an assistant turn with
`<|user|>` or `<|observation|>` and lists both in `config.eos_token_id`.

A distinct pad token is deliberately **not** a terminator: on a right-padded row whose turn lacks a
real terminator, the first trailing pad would close the span and train the model to predict pad.
Such a turn falls to the warn-and-mask path instead.

**`assistant_message_template` should be the assistant *role* token alone** when the model prefixes a
thinking marker (GLM's `<|assistant|></think>` vs `<|assistant|><think>…`, Qwen3's `<think>` blocks).
The full template `"<|assistant|></think>"` matches only the non-thinking render and silently drops
every thinking turn from the loss; the role token marks every assistant turn and yields the identical
span for non-thinking turns.

The completion-only collators locate each turn at its role token and unmask through to that turn's
first terminator, bounded by the next response start — so a truncated turn yields an empty span
rather than borrowing a later turn's terminator. The packed and padding-free routes add a global
end-of-sequence fallback, where the last position substitutes only when the sequence has **no**
terminator at all. The padded route has no fallback: on a padded row the last position is pad, so a
terminator-less turn is masked out entirely.

**One span resolver, three named policies.** Every completion mask locates its spans with
`resolve_completion_spans` in `src/data/spans.py`, under one of three declared policies:
`COLLATOR_SPAN_POLICY` (the padded route, described above), `PACKED_SPAN_POLICY` (the same plus the
global end-of-sequence fallback the packed and padding-free routes take) and
`SELF_DISTILL_SPAN_POLICY` (the span starts after the marker and runs to the first terminator
anywhere after it, falling back per span to the row end).

All three enter it through one preamble, `resolve_spans_or_warn`: it applies the policy and — when
the marker never matched, or every span closed at `-1` — warns that the sequence trains zero tokens
and returns the no-span verdict its caller turns into a fully-masked row. Only the refill afterwards
differs (the container, whether it copies from `input_ids` or `labels`, and the packed routes' EOS
rescue).

The padded collator, the offline label bake (`prepare_dataset.py`) and the self-distill / VLM label
builder (`build_completion_only_labels`) share one batch masker on top of it,
`mask_batch_to_completion_spans`. It additionally takes an optional `attention_mask` confining the
span search to a row's real tokens, and `extra_ignore_token_ids` (image tokens on the VLM path)
masked **after** the span refill — the refill copies from `input_ids`, so masking them first hands
every extra id inside a completion span back as a trainable target.

The packed and padding-free collators mask one sequence at a time instead, copying spans out of
`labels` so a document boundary keeps its own masking. The bake takes its policy from the artifact it
writes — `PACKED_SPAN_POLICY` when that artifact is packed — so a preprocessed dataset and the same
YAML on a raw dataset train the same tokens for a turn whose terminator is missing.

**When `pad_token_id == eos_token_id`** (GLM-4 uses `<|endoftext|>` for both) completion masking is
unaffected: the completion-only collators detect turn boundaries in `input_ids`, not `labels`. The
full-label collators (`DataCollatorWithPacking`, `DataCollatorForCausalLMWithPadding`) restore the
real EOS from `input_ids` at every real position after the parent collator's value-based masking
(`restore_eos_labels_when_pad_equals_eos`); offline-precomputed `labels` pass through untouched —
their baked `-100` spans are authoritative.
