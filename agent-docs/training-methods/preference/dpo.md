# Direct Preference Optimization (DPO)

DPO trains on pairwise preference data by deriving the optimal policy directly from the KL-constrained RLHF objective, against a frozen reference model and without a separate reward model.

| Aspect | Value |
|--------|-------|
| Data format | Pairwise preferences (prompt, chosen, rejected), all `list[dict]` |
| Reference model | Required (frozen copy, PEFT base without adapter, or precomputed log probs) |
| Trainer | `DistributedDPOTrainer` |
| Script | `scripts/training/preference/dpo.py` (text or VLM) |
| Parallelism | EP, TP, ETP, EP+TP; no CP. Declares `_supports_pp` — [PP](../../parallelism/pipeline-parallelism.md) is not yet available in this release |

## Dataset format

Same pairwise format as [SMPO](smpo.md) — `prompt`, `chosen`, `rejected`, each a `list[dict]` message list. See [Dataset Formats](../../data/dataset-formats.md).

### Vision-language

Point `dpo.py` at a VLM model to run DPO on image+text pairs. The script auto-detects the modality; routing then keys on the **dataset**. Text-only preference data on a natively-multimodal model (Qwen3.5/3.6, Gemma4) trains through the normal text pipeline, hub-shape normalization included. An `images`/`image` column puts TRL 1.6's `DPOTrainer` in vision mode with its `DataCollatorForVisionPreference`, and those rows must already be contract-shaped (prompt = message list, chosen/rejected = continuation-only), because TRL applies no hub-shape normalization.

A dataset storing its images under another column name declares them with `images_field`, which is renamed to `images` before the dispatch: TRL probes for the `image`/`images` spelling and prunes every other column, so an un-aliased column would train as text on a run the toolkit already calls multimodal. The column must exist in the loaded splits — a mistyped name raises at the dataset load rather than training text on a run the toolkit calls multimodal. Unlike the [SFT field](../sft.md#vision-language-models), it is not injected into a conversation turn; TRL's vision route reads the column itself.

```yaml
images_field: image_bytes   # renamed to `images` for TRL's vision route
```

TRL rejects `precompute_ref_log_probs` for vision datasets, so the vision reference under EP is standard-PEFT adapters (native EP expert-LoRA needs precomputed ref logps and stays text-only). Under TP, where PEFT and an explicit reference are both rejected, vision DPO has no supported shape.

## Quick start

```bash
# Standard (FSDP via accelerate)
accelerate launch --config_file launcher-configs/accelerate/fsdp2_gradop_config.yaml \
    scripts/training/preference/dpo.py \
    examples/preference/qwen3_5/dpo-qwen3.5-9b-tulu3-prefmix.yaml

# Expert parallelism (MoE)
torchrun --nproc_per_node=8 scripts/training/preference/dpo.py \
    examples/preference/gptoss/dpo-gptoss-20b-tulu3-prefmix-ep.yaml --expert_parallel_size=8
```

Minimal config:

```yaml
model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
dataset: path/to/preference_dataset
test_size: 0.05

beta: 0.1
loss_type: sigmoid
max_length: 2048

per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 5.0e-7
num_train_epochs: 1
gradient_checkpointing: true   # bf16 enabled by default

# PEFT makes the reference model the adapter-free base (no second copy)
use_peft: true
lora_r: 64
lora_alpha: 128
lora_target_modules: [q_proj, k_proj, v_proj, o_proj]

output_dir: checkpoints/dpo-llama
report_to: wandb
```

`unfreeze_layers_patterns` and `freeze_layers_patterns` (freeze applied after unfreeze) take lists of layer-name patterns.

## Loss variants

`loss_type` is a `list[str]`; a YAML string normalizes to a single-element list, and multiple losses combine via `loss_weights`. Valid values: `sigmoid` (default), `hinge` (SLiC-HF), `ipo`, `sft` (NLL on chosen), `exo_pair`, `nca_pair`, `robust`, `bco_pair`, `sppo_hard`, `aot`, `aot_unpaired`, `discopop`, `apo_zero`, `apo_down`, `sigmoid_norm`.

RPO adds an NLL term on chosen by including `sft`; its `loss_weights` entry is the RPO alpha coefficient:

```yaml
loss_type: [sigmoid, sft]
loss_weights: [1.0, 1.0]
```

## Reference model

Three shapes, two of which avoid a second full model copy:

- **PEFT/LoRA** (preferred) — `use_peft: true`; the reference is the base model without the adapter.
- **Precompute** — `precompute_ref_log_probs: true` computes reference log probs once, then drops the reference model.
- **Accept the overhead** — load a full frozen copy.

The full frozen copy (`load_reference_model_for_preference` in `src/distributed/loading/frozen_models.py`) mirrors the policy load: same `model_revision` (an unpinned reference would silently load hub `main` and shift every logratio), the same attention-implementation validator, and the same GptOss sink reset/freeze driven by `reset_sinks`.

TRL builds a reference of its own whenever `beta != 0` and the model is not PEFT-wrapped, and that one mirrors nothing — the scripts clear `model_init_kwargs` after loading the policy, so it loads fp32, from hub `main`, with the config-default attention.

On a policy carrying **live attention sinks** (`reset_sinks: false`) that implicit reference is refused at construction on **every** parallelism mode, plain FSDP2 DP and single-GPU included: the policy is restricted to sink-carrying attention and the reference is not, so the pair score identical tokens differently and the KL is biased on every token. Use `beta: 0`, `use_peft: true`, or precomputed reference log probs. Without live sinks the same implicit reference only warns, and only under EP, where the un-sharded fp32 dense replica costs the most.

Under EP/TP the reference is never parallelized, so a full frozen copy is not an option: an explicit `ref_model` is rejected at construction, and full fine-tuning requires `precompute_ref_log_probs: true` — the script raises without it. Reference log probs are computed from the still-untrained policy before the first step; a resume re-derives them from the trained weights. Under EP, PEFT also works, and native EP expert-LoRA requires `precompute_ref_log_probs` too (grouped expert adapters cannot be toggled off). Under TP, PEFT is rejected — adapters are not in the TP DTensor graph — leaving `precompute_ref_log_probs` as the only shape.

Pipeline parallelism is [not yet available in this release](../../parallelism/pipeline-parallelism.md); the shipped PP gates (`src/trainers/mixins/pp_gates.py`) already pin its contract for this trainer — precompute-only with the `ref_chosen_logps` / `ref_rejected_logps` columns in the dataset, `loss_type` ∈ {`sigmoid`, `hinge`, `ipo`}, `f_divergence_type: reverse_kl`, and no PEFT / `use_weighting` / `ld_alpha` / `activation_offloading` / `compute_metrics`.

## Parallelism

Attention TP and ETP are mutually exclusive. Flags: `expert_parallel_size`, `tensor_parallel_size`, `expert_tensor_parallel_size` (all default `1`), `ep_scope` (default `auto`). Full matrix: [Trainer Compatibility](../../reference/trainer-architecture.md#trainer-compatibility).

## Configuration

`DPOScriptArguments` adds `generate_eval_examples` (`True`), `num_eval_examples` (`50`), `generation_max_prompt_length` (`512`), and `images_field` (`None`, [vision](#vision-language)). Dataset fields (`dataset`, `dataset_ratio`, `test_size`, the freeze patterns) come from `CommonScriptArguments`.

DPO has no training-side prompt cap — TRL 1.6's `DPOConfig` has no `max_prompt_length`. It truncates the concatenated prompt + completion to `max_length` `keep_start`, so an over-long prompt eats its own completion; filter over-long prompts in the dataset. `generation_max_prompt_length` bounds only the eval-time generation dataset ([Sequence length](../../reference/configuration-reference.md#sequence-length-caps-vs-generation-budgets)).

Key TRL `DPOConfig` defaults: `beta` `0.1`, `loss_type` `["sigmoid"]`, `max_length` `1024`, `precompute_ref_log_probs` `False`, `label_smoothing` `0.0`. Full list: [Configuration Reference](../../reference/configuration-reference.md).
