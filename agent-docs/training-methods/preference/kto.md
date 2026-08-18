# KTO

Kahneman-Tversky Optimization trains on **unpaired** binary feedback: each row is one completion labeled desirable or not — no chosen/rejected pairing. Use it when you have single completions with thumbs-up/down labels rather than preference pairs.

| Aspect | Value |
|--------|-------|
| Trainer | `DistributedKTOTrainer` (`src/trainers/preference/kto.py`) |
| Script | `scripts/training/preference/kto.py` (text or VLM) |
| Config | `examples/preference/qwen3_5/kto-qwen3.5-9b-kto-mix-14k.yaml` |
| Parallelism | EP, TP, ETP, EP+TP; no CP. Declares `_supports_pp` — [PP](../../parallelism/pipeline-parallelism.md) is not yet available in this release |
| Reference model | Required (or use PEFT / `precompute_ref_log_probs`) |

`DistributedKTOTrainer` is TRL's `KTOTrainer` plus EP/TP via `DistributedTrainerMixin`. CP is unsupported — like DPO, KTO needs full-sequence log-prob pooling and a KL reference term, incompatible with sequence splitting. KTO runs with Liger kernels applied at the model level (via `load_distributed_model`); TRL's fused KTO-Liger loss path is disabled at construction, since it hard-requires a reference model and is broken in TRL 1.6.

The full-finetune reference model loads through the same path as DPO's — same `model_revision`, validated attention implementation, and GptOss sink handling as the policy; under EP/TP full fine-tuning requires `precompute_ref_log_probs: true`. `beta != 0` without PEFT on a policy carrying live attention sinks is refused on every parallelism mode, single-GPU included (see [DPO — Reference model](dpo.md#reference-model)).

Pipeline parallelism is [not yet available in this release](../../parallelism/pipeline-parallelism.md); the shipped PP gates already pin its contract for this trainer — `loss_type: apo_zero_unpaired` only (the default `kto` loss needs a world-global detached KL baseline no microbatch can compute), precompute-only with the `ref_logps` column present, and no live `ref_model`, PEFT, `activation_offloading`, or `compute_metrics`.

## Dataset format

Unpaired triples — `prompt`, `completion`, and a boolean `label` (desirable):

```jsonl
{"prompt": [{"role": "user", "content": "What is 2+2?"}], "completion": [{"role": "assistant", "content": "4"}], "label": true}
{"prompt": [{"role": "user", "content": "What is 2+2?"}], "completion": [{"role": "assistant", "content": "5"}], "label": false}
```

`prompt`/`completion` may be plain strings or `list[dict]` messages (auto-templated). The `completion`/`label` field names are configurable via `completion_field`/`label_field`. Different shape? Pairwise chosen/rejected → [DPO](dpo.md)/[SMPO](smpo.md); multi-completion with rewards → [Offline GRPO](../grpo/offline-grpo.md), which subsumes KTO for groups of 2+.

### Vision-language

Point `kto.py` at a VLM model to run KTO on unpaired image+text rows — `{prompt, completion, label, images}`. The model class follows the **checkpoint** (`load_model_for_training` auto-detects it); for a multimodal one the processor is the `processing_class`, which marks TRL's `KTOTrainer` as VLM-aware.

The data path follows the **dataset**, and TRL reads it off the row's **columns**: it switches to `DataCollatorForVisionUnpairedPreference` when the dataset carries an `image`/`images` column, while text-only data on a multimodal checkpoint stays on the text pipeline (the run logs `modality: text`). `images_field` names the column holding a row's images under any other spelling; the script renames it to `images` before handing the dataset over, since TRL matches that name in both its probe and its collator. A named column the dataset does not carry is refused at the dataset load.

Because the probe is column-only, a dataset that instead **embeds** `{"type": "image"}` parts in its `prompt`/`completion` messages while shipping no image column is refused by the script — TRL would template those rows on the text path, expanding each part into vision placeholder tokens with no pixels behind them. Move the images into an `images` column (one list per row, leaving unfilled `{"type": "image"}` placeholders in the messages, which TRL fills at collation), or drop the image parts and train the text.

Vision KTO constraints, all upstream in TRL's `KTOTrainer`:

- **Unpaired only** — a `chosen`/`rejected` column alongside images raises (unpairing a large image dataset row-by-row is too expensive); `unpair_preference_dataset` it first.
- **No `precompute_ref_log_probs`** — vision rows are tokenized and processed on the fly in the collator, never upfront. Under EP/TP that leaves PEFT (`ref_model=None`) as the only reference for a vision run, since an explicit reference model is refused there (it is never parallelized).

Same reference-model implications as [vision DPO](dpo.md#vision-language).

## Quick start

```bash
accelerate launch scripts/training/preference/kto.py \
    examples/preference/qwen3_5/kto-qwen3.5-9b-kto-mix-14k.yaml
```

The shipped config is dense Qwen3.5-9B, so it runs plain data-parallel. EP needs `torchrun` and an MoE checkpoint — add `--expert_parallel_size=8` to a KTO config that points at one.

Key hyperparameters (TRL `KTOConfig`): `beta` (`0.1`), `desirable_weight`/`undesirable_weight` (`1.0` each, for class imbalance), `max_length` (`1024`). KTO has no `max_prompt_length`: TRL truncates the assembled prompt + completion to `max_length` keeping the start, so an over-long prompt eats its own completion.

`tools_field` and `log_decoded_samples` are refused up front: TRL templates and tokenizes the raw prompt/completion columns itself, passing no `tools=`, and the rows the script hands it carry no `input_ids` for the sample writer to decode.

## Attention

Like the other padded-batch scripts, `kto.py` defaults `attn_implementation` to `sdpa` under `reset_sinks: true` — see [Flash Attention](../../optimization/flash-attention.md#model-specific-handling).
