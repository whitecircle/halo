# Collators

Collators turn tokenized samples into batches: padding, sequence packing, completion-only label
masking, and per-sequence position IDs. They live in `halo/data/collators/`. `factory.py` selects
among the SFT collators of `padded.py` and `packing.py` after `collator_rules.py` refuses the option
combinations none can serve; all of them share the span resolver and label rule in
`halo/data/completion_spans.py` and the turn terminators in `halo/data/turn_markers.py`.

The per-method leaves (`smpo.py`, `vlm.py`, `self_distill.py`, `classification.py`,
`offline_grpo.py`, `vlm_preference.py`) are built by their own trainers and scripts.
`fixed_shape.py` wraps any of them for a consumer whose buffer shapes freeze on the first batch.

| Collator | Packing | Padding-Free | Completion Masking | Position IDs |
|----------|---------|--------------|-------------------|--------------|
| `DataCollatorForCompletionOnlyLM` | No | No | Yes | No |
| `DataCollatorWithPacking` | Yes | No | No | reset per sequence |
| `DataCollatorForCompletionOnlyLMWithPacking` | Yes | No | Yes | reset per sequence |
| `DataCollatorWithFlattening` | No | Yes | No | reset per sequence |
| `DataCollatorWithFlatteningAndCompletionMask` | No | Yes | Yes | reset per sequence |
| `DataCollatorForCausalLMWithPadding` | No | No | No | No |

The last one is the SFT CP-mode padded route; it preserves precomputed `labels` (the parent
`DataCollatorForLanguageModeling` rebuilds them from `input_ids` and crashes on ragged ones).

You rarely construct these directly — `select_data_collator()` picks one from your YAML config.

## Collator classes

Constructor notes that matter when you build one directly:

- `response_prompt_template` (`str` or `List[int]`, the assistant-header template) is required for
  `DataCollatorForCompletionOnlyLM`, `DataCollatorForCompletionOnlyLMWithPacking` and
  `DataCollatorWithFlatteningAndCompletionMask`; the flattening one refuses `None` at construction.
  An **empty** template raises — it would match every position and train on all tokens.
- `tokenizer` is required for every collator except `DataCollatorWithFlattening`, which defaults it
  to `None` and never uses it.
- `pad_to_multiple_of` is set to `cp_size` under CP; the masked-label fill value is `-100`
  throughout.

**`DataCollatorForCompletionOnlyLM`** — masks loss on everything except assistant completion tokens (user/system tokens get `ignore_index`). Extends `DataCollatorForLanguageModeling`. If `response_prompt_template` is not found in a sequence, that sequence is dropped from the loss.

**`DataCollatorWithPacking`** — packs multiple sequences per sample. Position IDs reset at each boundary; labels at `position_ids == 0` are set to `-100` so the model is not penalized for predicting the next document's first token across a boundary.

It drops the dense `attention_mask` so Flash Attention runs its varlen kernel over per-document `cu_seq_lens`, handed over by the collator (below) or derived from the resetting position IDs, and flattens the packed mini-batch to a single `[1, total_tokens]` row of **real tokens only** — transformers' packed-sequence detection only engages at batch size 1, so a `[B>1, L]` batch would silently fall back to dense-causal attention.

On a flash run (the resolved backend `select_data_collator` reads) it also hands attention that varlen set on every multi-document row: `cu_seq_lens_q`/`cu_seq_lens_k` and host-int `max_length_q`/`max_length_k`, the values transformers would derive from the position IDs. No attention layer then re-derives them — a packed-row check, a `nonzero` and a max length read back, each a device sync, per layer — and a gradient-checkpointing recompute, reentrant or not, re-runs attention on the same set. A one-document row carries none and keeps transformers' dense kernel. PP's kept rows carry none either: the schedule splits only batch-dim tensors, so each pipeline stage builds the set per microbatch instead ([PP packing](../parallelism/pipeline-parallelism.md#packing)).

Inter-row padding is dropped in the flatten. A pad carries position 0, so every kept pad would be its own varlen segment, and the FA4 backward pays a fixed per-segment cost.

Where pads must survive (a fixed-shape batch), the tail's position IDs are a ramp restarting every `PAD_TAIL_SEGMENT_CHUNK` (256) tokens, keeping it a handful of no-op segments.

Whether the packed documents actually stay isolated is **per family**, not universal; see [Document isolation under packing](#document-isolation-under-packing).

Where the attention path is isolated, the remaining difference between backends is cost. The non-varlen path builds a dense `[L, L]` mask over the *flattened* row, so `L` is the whole batch's token count (up to `per_device_train_batch_size × max_length`) and the mask exhausts memory at long context. `select_data_collator` **warns** there, telling you to keep `per_device_train_batch_size` at 1.

It detects packed data via the `seq_lengths` field that TRL's `pack_dataset()` emits for the `bfd` and
`bfd_split` strategies. The `wrapped` strategy emits no `seq_lengths` column at all, so each row
collates as **one** document and anything concatenated inside it attends across itself; the collator
warns once per instance when it sees such rows.

Online packing (`packing: true` at training time) and the offline `prepare_dataset.py` pipeline both
accept all three strategies (see [Pre-Processing](dataset-preparation.md#parameters)).

**`DataCollatorForCompletionOnlyLMWithPacking`** — packing plus completion masking; masks each packed
sub-sequence independently, falls back to standard completion masking for non-packed input.

**`DataCollatorWithTreePacking`** — packed rows of prefix-sharing trees (`tree_packing: true`). Built by
the SFT script, not the factory; see [Tree Packing](../optimization/tree-packing.md).

**`DataCollatorWithFlattening`** — padding-free. Flattens the batch into one `[1, total_tokens]`
sequence and emits Flash Attention varlen kwargs. Output keys: `input_ids`, `labels`, `position_ids`
(reset per sequence), `cu_seq_lens_q`/`cu_seq_lens_k`, `max_length_q`/`max_length_k`. The first token
of every sub-sequence is label-masked (the causal shift cannot predict it from the previous
document).

**`DataCollatorWithFlatteningAndCompletionMask`** — padding-free plus completion masking; the lowest-overhead collator for variable-length SFT trained on assistant responses only.

### Offline GRPO

`OfflineGRPODataCollatorWithPadding` and, under CP, `OfflineGRPOCPDataCollatorWithPadding` are built
by the trainer, not this factory: [Offline GRPO](../training-methods/grpo/offline.md).

## Factory: `select_data_collator()`

`halo/data/collators/factory.py` picks the collator from config flags and validates compatibility:

```python
from halo.data.collators.factory import select_data_collator

collator = select_data_collator(
    tokenizer=tokenizer,
    padding_free=True,
    train_on_completions_only=True,
    assistant_message_template="<|im_start|>assistant\n",
)
```

Flags, keyword-only: `train_on_completions_only` (required: the run's own setting), `padding_free=False`,
`packing=False`, `assistant_message_template=None`, `pad_to_multiple_of=None`, `use_context_parallel=False`,
`train_on_last_assistant_only=False`, `model_config=None`, `per_device_train_batch_size=1`,
`keeps_packed_rows=False`. With `packing` above batch 1 the collator merges the mini-batch's packed
rows into one row (document isolation and throughput unchanged), and the factory warns: raise
`max_length` instead when a longer pack is the intent. `keeps_packed_rows` suppresses that warning;
only the PP seam sets it.

**Pass `model_config`.** It is what feeds `resolve_eos_token_ids(tokenizer, model_config)`; without
it the collator falls back to the tokenizer's EOS alone and mis-masks any family whose turn
terminators live in `config.eos_token_id` (GLM-4). The SFT script passes it.
`train_on_last_assistant_only=True` requires `train_on_completions_only=True` and raises otherwise.

| Configuration | Collator |
|---------------|----------|
| CP + packing | **rejected** — the Ulysses CP attention path has no per-document boundaries, so packed documents would attend across each other |
| PP + padding_free | **rejected** — the flattened width varies per step while the pipeline's P2P buffers are frozen after the first |
| padding_free on an M-RoPE decoder (Qwen3.5/3.6 text towers included, the Qwen-VL text models) | **rejected** on every backend (`reject_mrope_padding_free`), as is a padding-free collator handed straight to the SFT trainer (`reject_mrope_padding_free_collator`) — the flattened M-RoPE row faults the flash varlen kernels; `packing` serves the same rows |
| PP + packing on a model whose conv / linear-attention layers read segment markers (GatedDeltaNet, LFM-2 ShortConv) or no boundary at all (KDA, Lightning Attention-2) | **rejected** — a stage's forward passes those layers no segment marker, so they cross the kept rows' documents ([PP packing](../parallelism/pipeline-parallelism.md#packing)) |
| CP + completion masking | `DataCollatorForCompletionOnlyLM` (CP mode) |
| CP | `DataCollatorForCausalLMWithPadding` (a label-preserving `DataCollatorForLanguageModeling` subclass in `padded.py`) |
| padding_free + completion masking | `DataCollatorWithFlatteningAndCompletionMask` |
| padding_free | `DataCollatorWithFlattening` |
| packing + completion masking | `DataCollatorForCompletionOnlyLMWithPacking` |
| packing | `DataCollatorWithPacking` |
| completion masking only | `DataCollatorForCompletionOnlyLM` |
| none | `None` (default TRL behavior) |

YAML fields (`packing`, `packing_strategy`, `padding_free`, `eval_packing` come from TRL's `SFTConfig`; `train_on_completions_only` and `assistant_message_template` from `ConversationRenderArguments`, `train_on_last_assistant_only` from `SFTScriptArguments`, both in `halo/configs/sft.py`):

```yaml
padding_free: false
packing: true
train_on_completions_only: true
assistant_message_template: "<|start_header_id|>assistant<|end_header_id|>\n\n"
train_on_last_assistant_only: false
```

The SFT script defaults `train_on_completions_only: true`; `assistant_message_template` has no default and must be the tokenizer's exact assistant-header encoding (the ChatML form above for Qwen; see [Constraints](#constraints)) — a missing or non-rendering marker raises at startup.

For a pre-processed dataset the SFT script derives the flags from the data, not the config: `packing` follows `metadata.packed` (offline-packed chunks still need the packing collator's per-document position IDs), and completion masking is disabled at runtime because the baked `labels` are authoritative — see [SFT Dataset Pre-Processing](dataset-preparation.md).

After building its collator a script calls `disable_trl_dataset_prep` (`halo/entrypoints/data_setup.py`): it sets `dataset_kwargs: {skip_prepare_dataset: true}` and clears `packing` and `padding_free` on the TRL config, since TRL reads both again at collator selection and would swap its own collator in over rows it never tokenized. A custom script must call it too; a user `dataset_kwargs` or `dataset_text_field` is refused at startup (`reject_trl_dataset_prep_args`).

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
| DeepSeek-V4 (eager-only) | isolated on the training path (`use_cache=False`; a live cache suppresses the packed mask) | the CSA/HCA compressors cut KV windows at row indices and judge their causality by position, so every document after the first attends the row's first windows — `packing` and `padding_free` are **refused** |
| Qwen3.5 / 3.6, Qwen3-Next | isolated | GatedDeltaNet reads `seq_idx` (conv) + `cu_seq_lens_q` (delta rule): the collators and SMPO's padding-free forward emit both for the family, so its boundaries reach the kernels — but only the `fla` / `causal-conv1d` fast paths consume them (both installed in the images). The torch fallbacks take neither, so a multi-document row would cross in the conv *and* the scan — the factory refuses `packing` / `padding_free`, and SMPO its `padding_free`, unless transformers' own fast-path predicates hold |
| Bailing / Ling | Ling 2.0, Ring: isolated on flash only (`patch_v4_flash_attention_template`); eager/SDPA **leak** — the remote file builds its mask from `attention_mask` alone ([Bailing/Ling](../models/bailing.md#attention-and-packing)). Ling 3.0: flash refused, MLA on SDPA leaks the same way, so Ling 3.0 cannot pack | Ling 3.0 KDA and Ring's Lightning-Attention-2 linear layers — **cross**. The KDA op accepts a `cu_seqlens` kwarg, but the model forward never threads kwargs down to it, so there is no reachable boundary parameter — the same class of crossing as Inkling's convs |
| GLM-5 Next | both layer types receive the 2D padding mask `create_recurrent_attention_mask` builds, so no packed-boundary parameter reaches the DSA attention either — flat packing **crosses**, so it is refused; [tree packing](../optimization/tree-packing.md) isolates it (the DSA forward reads the plan's path links) | KDA linear attention — **crosses** under flat packing, the same unreachable-boundary class as Bailing/Ling; isolated under tree packing (conv and scan segmented by the plan) |
| LFM-2 | isolated | ShortConv is isolated exactly: the collators and SMPO's padding-free forward emit `seq_idx` for this family and both conv paths honor it |
| Zaya | isolated — flash needs `patch_zaya_packed_document_isolation` | CCA convolution + delayed-value recurrence — isolated exactly by the same patch, which restarts both at every document start ([Zaya](../models/zaya.md#document-isolation-under-packing)) |
| Cohere2 MoE | isolated on every backend, no patch needed — the forward feeds `position_ids` into mask construction and through layer kwargs to the attention interface | — |
| Step-3.7 Flash | isolated on SDPA and eager, no patch needed — the forward feeds `position_ids` into both mask constructions (full and sliding); bit-exact through dense layers, MoE layers add expert-summation reduction noise (~1e-7 fp32). Training path only: a live cache suppresses the packed mask (`use_cache=False`, as DeepSeek-V4) | — |

These families are refused outright in `select_data_collator`. The checks read the text sub-config
too, so a composite (VLM) wrapper is covered:

- **GPT-OSS**, **Bailing / Ling** (Ling 2.0, Ring, Ling 3.0) and **GLM-5 Next**, `dense_packing_leaks` on
  their `MoEFamilySpec`, read by `has_dense_packing_leak` in `halo/models/segment_markers.py`: packing on a
  non-varlen backend raises (`reject_dense_attention_rows`, which the SFT script also runs before the model
  load, against the backend the loader will resolve). Use a flash backend or turn packing off; Ling 3.0 and
  GLM-5 Next have no flash backend.
- **DeepSeek-V4**, or any model whose `layer_types` declare a compressed-KV layer (the
  `"compressed"` kind in `halo/models/attention/layout.py`: `compressed_sparse_attention`,
  `heavily_compressed_attention`): `packing` and `padding_free` raise on every backend
  (`reject_compressed_kv_rows`). Positions restart per document but the compressor windows are cut
  at row indices, so a later document reads another document's compressed KV as its own context.
  Train with padded batches. The SFT script runs the same check before the model load,
  `prepare_dataset.py` before an offline `--pack_sequences` pass, and the SFT trainer for a
  packing / padding-free collator handed to it directly (`reject_compressed_kv_collator`, keyed on
  the same two collator roots as the CP gate), as does SMPO for its `padding_free` forward.

Which families get which segment markers, and the kernel refusal below, live in
`halo/models/segment_markers.py`, shared by the collators and SMPO's padding-free forward. LFM-2's
ShortConv is matched by family; GatedDeltaNet by the layers a config declares, under any `model_type`:
`linear_attention` in `layer_types` on a config carrying GatedDeltaNet's `linear_num_key_heads` /
`linear_num_value_heads` split. Kimi Delta Attention spells its layers `linear_attention` too but declares a
single `linear_num_heads`, and gets no markers.

A model with GatedDeltaNet layers carries one more refusal, about markers that
would be emitted but not read. transformers selects their segment-aware linear-attention kernels at
modeling-import time via `is_causal_conv1d_available()` / `is_flash_linear_attention_available()`,
which require the package installed, `fla >= 0.2.2`, **and** a CUDA-capable torch.

The torch fallbacks it takes otherwise drop `seq_idx` and `cu_seq_lens_q`, so conv and recurrent
state cross document boundaries while attention stays isolated, invisible in the loss. SFT's
`packing` and `padding_free` and SMPO's `padding_free` are refused unless those same predicates
hold, so the refusal cannot disagree with the kernels actually selected (the production images
satisfy them).

"Crosses by construction" applies where the mixer has no per-document boundary parameter at
all (Inkling's convs). Pack such a family only where a small amount of cross-document mixing is
acceptable, and prefer padded batches when it is not.

A linear-attention layer no segment marker reaches is refused unless the run opts in:
`has_unmarked_linear_attention` (`halo/models/segment_markers.py`) reads hybrid layer groups
(`layer_group_size` above 1: Ring's Lightning Attention-2, Ling 3.0's KDA) or `linear_attention` layer
types (GLM-5 Next's KDA) outside the families that read `cu_seq_lens`. SFT's `packing` / `padding_free`
(before the model load and again at collator selection) and SMPO's `padding_free` raise on every backend
through `reject_unmarked_linear_attention_rows`, flash included, where the softmax layers isolate and the
loss shows nothing. So does the SFT trainer for a packing / padding-free collator handed to it directly
(`reject_unmarked_linear_attention_collator`), under the opt-in the factory records on the collators it
builds; a tree collator is left to its plan. `allow_cross_document_linear_attention: true` (`SFTScriptArguments`, `SMPOConfig`)
trains such rows anyway, with a warning: those layers' recurrent state then carries from each document of
a row into the next. It admits no GLM-5 Next row: the family runs only dense backends, where its DSA
layers read only the padding mask (table above), so the SFT script (before the model load) and the
collator factory refuse its flat packing before the opt-in is read, and its `padding_free` needs a varlen
kernel it does not have. Tree packing skips this check and refuses the opt-in: its plan segments the
linear-attention layers of every family it admits ([Tree Packing](../optimization/tree-packing.md)).

Under distillation the teacher and the frozen reference forward with `use_cache=False` explicitly, so
they score packed rows under the same masks the student does.

## Constraints

- **`packing` and `padding_free` are mutually exclusive.** Packing pads concatenated sequences to
  `max_length`; padding-free flattens to one variable-length tensor. Pick packing for short
  sequences, padding-free for long/variable ones.

- **Both want a varlen Flash Attention backend.** The gate is the model's **resolved**
  `_attn_implementation` against `VARLEN_ATTN_IMPLEMENTATIONS` (`flash_attention_2/3/4`,
  `halo/models/attention/selection.py`), not a model-family list.

    Off it, `padding_free` **raises** (its `cu_seq_lens` kwargs have no consumer) and `packing`
    **warns** (it pays a dense `[L, L]` mask per layer). A family whose dense path leaks documents
    makes packing raise too ([Document isolation](#document-isolation-under-packing)), and
    DeepSeek-V4 (eager-only: head_dim 512 exceeds FA's 256 cap) refuses both. Gemma 4 lands on the
    warning; its shipped config packs at batch 1. See [Padding-Free Collator](../optimization/padding-free-collator.md).

- **A softcapped model (Gemma 2) does not train on FA4's varlen call**, whose backward takes no softcap. Packed,
  padding-free and padded rows (more than one per batch: transformers unpads any batch holding a padding token)
  all reach it (`rows_reach_varlen_kernel`). Left unset, `attn_implementation` takes FA2 there; a pinned
  `flash_attention_4` the load keeps (Gemma 2 27B; the resolver moves a head_dim-256 pin to FA2) is refused before
  the load and again on the resolved backend by `select_data_collator` (`reject_fa4_varlen_training`). See [Flash Attention](../optimization/flash-attention.md#model-specific-handling).

- **Padding-free is incompatible with Context Parallelism**, and **packing is rejected under CP**.
  Under CP use the padded collator, which sets `pad_to_multiple_of=cp_size`.

- **Completion masking requires `assistant_message_template`** matching the tokenizer's exact
  assistant-header encoding; unmatched sequences are dropped from the loss.

    A marker the chat template never renders is refused by `select_data_collator` and the
    self-distillation collator: it would mask every label and the run would train zero tokens at
    loss ≈ 0. The probe renders both a plain and a `thinking` assistant turn, so a marker gpt-oss
    harmony emits only under thinking passes it and is caught per batch instead.

## EOS, pad, and the response template

**The turn terminator is the resolved terminator set, not just `tokenizer.eos_token_id`.** A turn
ends at the first token in the union of the model config's `eos_token_id` (scalar or list;
`text_config.eos_token_id` for VLM/composite configs) and the tokenizer's EOS
(`resolve_eos_token_ids`, `halo/data/turn_markers.py`). This is required for templates that delimit
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
first terminator, bounded by the next response start. A truncated turn therefore yields an empty span
rather than borrowing a later turn's terminator.

The packed and padding-free routes add a global end-of-sequence fallback, where the last position
substitutes only when the sequence has **no** terminator at all. The padded route has no fallback: on
a padded row the last position is pad, so a terminator-less turn is masked out entirely.

**One span resolver, three named policies.** Every completion mask locates its spans with
`resolve_completion_spans` in `halo/data/completion_spans.py`, under one of three declared policies:
`COLLATOR_SPAN_POLICY` (the padded route, described above), `PACKED_SPAN_POLICY` (the same plus the
global end-of-sequence fallback the packed and padding-free routes take) and
`SELF_DISTILL_SPAN_POLICY` (the span starts after the marker and runs to the first terminator
anywhere after it, falling back per span to the row end).

All three enter it through one preamble, `resolve_spans_or_warn`: it applies the policy and — when
the marker never matched, or every span closed at `-1` — warns that the sequence trains zero tokens
and returns the no-span verdict, a fully-masked row. One label rule then refills the spans,
`build_span_labels`: it copies each span from `input_ids` or from `labels` already masked, and
optionally restores each span's closing token from `input_ids` (the packed route's EOS rescue).

The padded collator, the offline label bake (`prepare_dataset.py`) and the self-distill / VLM label
builder (`build_completion_only_labels`) share one batch masker on top of it,
`mask_batch_to_completion_spans`. It additionally takes an optional `attention_mask` confining the
span search to a row's real tokens.

It also takes `extra_ignore_token_ids` (image tokens on the VLM path), masked **after** the span
refill: the refill copies from `input_ids`, so masking them first hands every extra id inside a
completion span back as a trainable target.

The packed and padding-free collators mask one sequence at a time instead, copying spans out of
`labels` so a document boundary keeps its own masking. The bake takes its policy from the artifact it
writes — `PACKED_SPAN_POLICY` when that artifact is packed — so a preprocessed dataset and the same
YAML on a raw dataset train the same tokens for a turn whose terminator is missing.

**When `pad_token_id == eos_token_id`** (GLM-4 uses `<|endoftext|>` for both) completion masking is
unaffected: the completion-only collators detect turn boundaries in `input_ids`, not `labels`. The
full-label collators (`DataCollatorWithPacking`, `DataCollatorForCausalLMWithPadding`) restore the
real EOS from `input_ids` at every real position after the parent collator's value-based masking
(`collate_preserving_precomputed_labels`); offline-precomputed `labels` pass through untouched —
their baked `-100` spans are authoritative.
