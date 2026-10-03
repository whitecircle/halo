# Direct Preference Optimization (DPO)

DPO fits the policy to pairwise preferences against a frozen reference model, with no reward model. Use it on `prompt` / `chosen` / `rejected` rows when a reference fits the budget; for the same data without one use [SMPO](smpo.md), for unpaired thumbs-up/down rows [KTO](kto.md).

Trainer `DistributedDPOTrainer`, script `scripts/training/preference/dpo.py` (text or VLM). EP, TP and ETP apply; CP does not — TRL's loss path is not CP-aware ([matrix](../../reference/trainer-architecture.md#trainer-compatibility)). It declares `_supports_pp`, but pipeline parallelism is [not yet available in this release](../../parallelism/pipeline-parallelism.md).

## Configuration

```yaml
model_name_or_path: Qwen/Qwen3.5-9B
dataset: allenai/llama-3.1-tulu-3-8b-preference-mixture
test_size: 0.005
beta: 0.1
loss_type: sigmoid                 # normalizes to ["sigmoid"]
max_length: 4096
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-07             # the shipped recipes' value
gradient_checkpointing: true       # bf16 and Liger are toolkit defaults
use_peft: false                    # true makes the adapter-free base the reference
output_dir: checkpoints/dpo-qwen3.5-9b
```

| Knob | Default | Effect |
|---|---|---|
| `beta` | `0.1` | Scales the implicit-reward gap inside the loss; `0` flattens the objective |
| `max_length` | `1024` | Truncates prompt + completion, keeping the start |
| `generation_max_prompt_length` | `512` | Prompt cap for the eval generation split only |

`loss_type` takes 15 TRL values — `sigmoid`, `hinge`, `ipo`, `sft`, `exo_pair`, `nca_pair`, `robust`, `bco_pair`, `sppo_hard`, `aot`, `aot_unpaired`, `discopop`, `apo_zero`, `apo_down`, `sigmoid_norm` — and a list combines several under `loss_weights` (unset: `1.0` each), so `[sigmoid, sft]` adds an SFT anchor to the preference term.

No training-side prompt cap exists, so an over-long prompt eats its own completion: filter those rows in the dataset ([sequence length](../../reference/configuration-reference.md#sequence-length-caps-vs-generation-budgets)).

## Log-prob precision

The trainer sums sequence log-probs in fp32. TRL's own `selective_log_softmax` keeps the per-token log-probs of bf16 logits in bf16, and sums them in bf16. At `|logp|` in [8192, 16384) that grid is 64 nats, so a long completion's margin carries tens of nats of rounding.

`FP32LogprobsMixin` (`src/trainers/preference/logprobs.py`) swaps that function out while TRL's loss and reference pass run. The replacement is a chunked fp32 log-softmax (`src/kernels/logprobs.py`) that never builds a full fp32 `[B, T, V]` plane. TRL sums its output in fp32, so these are fp32:

- every loss term built on the sequence log-probs, and the `ld_alpha` split;
- the log-prob term of the WPO weights;
- the `logps/*` and `rewards/*` metrics.

Terms TRL takes from the logits directly keep TRL's precision: the `sft` loss type's cross-entropy, WPO's normalizer, and the `entropy`, `logits/*` and `mean_token_accuracy` metrics. Reference columns the dataset supplies keep the precision they were computed in.

Nearly every run takes the fp32 path. TRL's fused Liger DPO loss runs only when Liger's fused linear cross-entropy was applied at load (the [per-model default](../../optimization/liger-kernels.md#configuration) for Zaya, DeepSeek-V4 and GLM-4.7-Flash, or `fused_linear_cross_entropy: true`), on an unsharded policy against a frozen reference copy. FSDP2, EP and TP switch it off, and TRL refuses it alongside precompute or PEFT.

## Reference model

Three shapes, decided by `load_reference_model_for_preference` (`src/distributed/loading/frozen_models.py`):

- **PEFT** (`use_peft: true`) — no second model: the script passes a LoRA config, so the reference is the base with the adapter disabled. A trainer built by hand around an already-wrapped `PeftModel` gets a frozen `ref` adapter copy instead.
- **EP / TP / PP with `precompute_ref_log_probs: true`** — no reference is loaded either. Log-probs come from the untrained policy before step 1, and a resume restores them from the checkpoint ([below](#resuming-a-precompute-run)).
- **A frozen copy** — every other shape, precompute on plain data parallelism included. It mirrors the policy load (same revision, attention validator, sink policy) and stays resident for the run.

Under EP, TP or PP a frozen copy is rejected outright — the reference is never parallelized — so full fine-tuning there needs precompute. TP rejects PEFT too, leaving precompute as its only shape. Expert-only native EP LoRA needs it as well: with no `PeftModel` nothing switches the adapters off. A mixed attention + expert adapter keeps the implicit reference, since `disable_adapter()` drops the expert adapters too.

A policy carrying live attention sinks (`reset_sinks: false`) is refused whenever a reference model reaches the trainer, single GPU included. Only PEFT, or EP/TP/PP with precompute, leaves none.

### Resuming a precompute run

The sweep runs inside the trainer's `__init__`, over the policy when no separate reference exists, and every rank attaches the gathered columns in memory, so per-node storage needs no shared cache file. A [Path-B resume](../../reference/checkpoints.md#resuming-training) builds the policy from the checkpoint before that, so a sweep there would score the trained weights and zero every log-ratio. Every checkpoint of a sweeping run therefore carries the columns in `reference_logps.pt`: one entry per split (`train`, then each eval dataset by name) with its row count, a digest of each token-id column the reference read, and the settings that shape the values (`max_length` and the [log-prob precision](#log-prob-precision), plus `truncation_mode` and `ld_alpha` for DPO). The script hands the trainer the resume checkpoint, and the precompute attaches the saved columns instead of sweeping (`src/trainers/preference/precompute.py`).

- When the policy was built from the checkpoint's weights and no separate reference exists, a split with no saved entry raises, and so does one whose row count, token ids, settings or columns (a KL column the run's KTO loss needs) differ from the run's. Any other resume (an adapter or merge-on-save checkpoint builds the policy from the base; a frozen reference copy) sweeps untrained weights for such a split and proceeds.
- A split whose settings record another log-prob precision, or none, is such a mismatch: its values carry that precision's rounding. The refusal names the one-step regeneration below.
- Splits the resumed run does not precompute (an eval dataset switched off) ride unchanged into its checkpoints, for a later resume that uses them again.
- To give a checkpoint without the file one, run the same config for one step from the base into a scratch directory outside the run's `output_dir`, whose rotation could otherwise delete the checkpoint (`--output_dir=<scratch> --max_steps=1 --save_strategy=steps --save_steps=1 --save_only_model=true --resume_from_checkpoint=null`; the refusal names a scratch path), and copy its `checkpoint-1/reference_logps.pt` into the checkpoint, on every node when checkpoints are node-local; the resume validates it like its own. A checkpoint whose save stopped before the file can take the previous checkpoint's copy, which holds the same values.
- Columns the dataset already carries are read from it and not persisted.

Pipeline parallelism is [not yet available in this release](../../parallelism/pipeline-parallelism.md); the shipped PP gates (`src/trainers/mixins/pp_gates.py`) already pin its contract for this trainer — `precompute_ref_log_probs: true` with the `ref_chosen_logps` / `ref_rejected_logps` columns already in the train dataset, and in the eval dataset whenever one is passed (`test_size` alone creates one, whatever `eval_strategy` says); `loss_type` ∈ {`sigmoid`, `hinge`, `ipo`} with `f_divergence_type: reverse_kl`; and no `use_weighting`, `ld_alpha` or `compute_metrics`.

## Launch

```bash
# Plain FSDP2 data parallel
torchrun --nproc_per_node=8 scripts/training/preference/dpo.py \
    examples/preference/qwen3_5/dpo-qwen3.5-9b-tulu3-prefmix.yaml

# MoE, expert parallel (that recipe is LoRA, so the adapters are the reference)
torchrun --nproc_per_node=8 scripts/training/preference/dpo.py \
    examples/preference/gptoss/dpo-gptoss-20b-tulu3-prefmix-ep.yaml --expert_parallel_size=8
```

`halo launch dpo <config> --nproc 8` builds the same line; any field is overridable (`--beta=0.05`). `accelerate launch` serves plain DP only — EP/CP/TP/PP reject it.

## Vision-language

`dpo.py` trains a VLM checkpoint on image+text pairs. The model class follows the checkpoint, the data path follows the **dataset**: text-only pairs on a natively multimodal model take the text pipeline, hub-shape normalization included.

An `images`/`image` column puts TRL in vision mode with `DataCollatorForVisionPreference`. Those rows must already be contract-shaped — prompt a message list, chosen/rejected continuation-only — since TRL normalizes no hub shapes. `tools_field` is refused: the vision render passes no `tools=`.

Set `images_field: <column>` when the dataset stores images under another name — it is renamed to `images` before the dispatch, the spelling TRL probes for. A name the splits do not carry raises at dataset preparation, after the policy and the reference have loaded.

TRL rejects `precompute_ref_log_probs` on vision datasets, so under EP the vision reference is standard PEFT adapters. Under TP, where PEFT and an explicit reference are both rejected, vision DPO has no supported shape.

## Testing a setup

```bash
torchrun --nproc_per_node=2 scripts/training/preference/dpo.py <config> \
    --max_steps=5 --save_strategy=no --report_to=none
```

Covering tests:

- `pytest tests/cpu/trainers -m cpu`
- `tests/gpu/trainers/preference/test_dpo.py`, `test_dpo_vlm.py` and `test_pref_ep_expert_lora_reference.py`
- precompute resume, over `tests/common/preference_precompute_e2e.py`: `tests/gpu/parallelism/ep/test_ep_preference_precompute_resume.py` (Qwen3-MoE and the dense model) and `tests/gpu/trainers/preference/test_preference_precompute_resume_families.py` (every other EP family)

## What to watch

| Signal | Reading |
|---|---|
| `rewards/accuracies` | Share of pairs scoring chosen above rejected (chance is 0.5) |
| `rewards/margins` | `beta ×` the log-ratio gap; rises as pairs separate |
| `logps/chosen`, `logps/rejected` | Both falling together is log-prob collapse: lower the LR, raise `beta`, or add `sft` |

Failure signatures:

- A reference-model raise on a live-sinks policy — `use_peft: true`, or precompute under EP/TP/PP. `beta: 0` does not help: it changes the loss, not whether a reference loads.
- "Cannot hold a separate dense reference" under EP/TP/PP — set `precompute_ref_log_probs: true` or `--use_peft`.
- "Cannot resume precompute_ref_log_probs" — the checkpoint's `reference_logps.pt` is missing or lacks that split. Recover it with the one-step run [above](#resuming-a-precompute-run), or put the reference columns (computed on the base model) in the dataset.
- "does not belong to this '<split>' dataset" on resume — the resumed data or reference settings differ from the saving run's (dataset, split, chat template, tokenizer, `max_length`, `truncation_mode`, `ld_alpha`). Resume with the saving run's data and settings, or regenerate the file with the one-step run [above](#resuming-a-precompute-run).
- "Regenerate the '<split>' reference log-probs for this run" on resume — the saved split was summed at another [log-prob precision](#log-prob-precision), which no setting selects. Run the one-step regeneration the message names.
