# Checkpoints: Save, Resume, Export, Upload

A run writes standard HuggingFace checkpoints to `output_dir` on the cadence set
by `save_strategy` / `save_steps`. Every mode (FSDP2, EP, TP, CP and their
combinations) writes a gathered checkpoint that `from_pretrained` loads. The
exceptions are the opt-in sharded EP save and LoRA adapters.

FSDP2, CP and TP saves keep the expert layout the run loaded. An EP save writes
the family's own export layout: per-expert where the serving engines need it,
fused for Qwen3.5/3.6, DeepSeek-V4, Cohere2 MoE and GLM-5 Next, and the hub
layout for Step-3.7 Flash
([details](../agent-docs/reference/checkpoints.md#serving-on-vllm--sglang) ↗).

| Training mode | Output | Load directly? |
| --- | --- | --- |
| FSDP2, CP, TP | gathered HF checkpoint | yes |
| EP (default) | gathered HF checkpoint, experts at global indices | yes |
| EP with `save_sharded_ep: true` | one expert shard per rank | merge first |
| LoRA / QLoRA | adapter only (`adapter_model.safetensors`) | serve as an adapter, or merge into the base |

EP expert LoRA cannot be merged afterwards: train it with
`merge_expert_lora_on_save: true` to save the merged model.

## Sharded EP saves

`save_sharded_ep: true` skips the gather, which makes checkpointing a very large
model faster. Run `halo run merge-ep-shards` before the checkpoint can be used.
It needs one EP group spanning all ranks, no CP, ETP or expert LoRA, a shared
filesystem across nodes, and a supported family
([full list](../agent-docs/reference/checkpoints.md#expert-parallelism-ep-eptp-epcp) ↗).
These are checked at startup.

## Resume

Set `resume_from_checkpoint` to a checkpoint directory, or to `true` to pick the
newest complete one in `output_dir`, and launch the same training script again.

A torchrun run with sharded parameters (FSDP2, EP, TP, CP) saves per-rank
optimizer shards, the LR schedule and the step. Single-GPU, QLoRA and
`accelerate` runs keep HF's own `optimizer.pt`.

- **Exact resume** restores the optimizer state too. It needs the same topology
  fingerprint: world size, every parallel size, the layout knobs and the
  optimizer class
  ([full list](../agent-docs/reference/checkpoints.md#warm-restart-vs-exact-resume-torchrun) ↗).
- **Warm restart** happens when any of those differ, or the run saved with
  `save_only_model: true`. Weights and the LR schedule are restored; optimizer
  moments start fresh.
- **A failed optimizer restore** (a truncated shard, a CUDA OOM, the optimizer
  half of an interrupted save) stops every rank with an error naming the cause.
  Fix it, or set `allow_optimizer_warm_restart: true` to accept fresh optimizer
  moments.

The weights come back one of three ways:

- **Built from the checkpoint.** At the default `use_grouped_gemm: true` (dense
  models included), and always under EP, CP or TP, the training script builds
  the model directly from the checkpoint directory. This is why the resume must
  go through the training script, not a hand-built model.
- **Loaded into a built model.** With `use_grouped_gemm: false` and no EP, CP or
  TP (plain FSDP2, single process, DDP), the checkpoint loads into a model built
  from the base. Its keys convert the way `from_pretrained` converts them, so a
  checkpoint in the hub's per-expert layout lands in the fused experts. A
  best-model reload under FSDP2 or TP takes the same path. `accelerate` FSDP
  uses the base Trainer's loader.
- **Adapters onto the base.** LoRA runs build the base model and restore the
  adapters onto it. `merge_expert_lora_on_save` and embedding LoRA checkpoints
  keep the unmerged adapters (`resume_adapter/`) for this.

MoE router-balancing state is saved and restored automatically.

### Interrupted saves

A save is safe to interrupt. `trainer_state.json` is written last, so a step
directory without it never completed.

- `resume_from_checkpoint: true` skips an incomplete directory and moves it into
  `output_dir/_incomplete_checkpoints/`.
- Naming an incomplete directory explicitly raises, and so does an `output_dir`
  whose step directories are all incomplete.
- An `output_dir` with no step directory starts fresh, with a warning.
- `save_total_limit` never deletes the previous checkpoint before its successor
  is complete.

### Best model at end

`load_best_model_at_end` is refused at startup where the end-of-run reload
cannot work: a full fine-tune under CP, a MoE model wrapped for expert compute
(including plain FSDP2 at the default `use_grouped_gemm: true`), and TP with
more than one data-parallel replica. Export the best checkpoint yourself
instead. Dense pure TP is allowed, and adapter-only runs are exempt unless
`merge_expert_lora_on_save` merges them into a full checkpoint.

## Post-processing tools

All run as `halo run <tool> <flags>`; `halo run <tool> -- --help` lists the
flags.

| Tool | Use it to |
| --- | --- |
| `merge-ep-shards` | make a sharded EP save loadable |
| `merge-peft-adapters` | merge a LoRA adapter into its base as one checkpoint |
| `merge-models` | combine models in weight space (linear, SLERP, task arithmetic, TIES) |
| `convert-to-bf16` | cast an fp32 or mixed checkpoint to bf16 for serving |
| `quantize-to-lowp` | write block-scaled mxfp8 / mxfp4 / nvfp4 weights; serving engines do not load its manifest as-is |
| `unfuse-moe-experts` | rewrite fused expert weights to the family's per-expert layout, for a serving loader that reads only that layout |
| `reset-sinks` | disable the attention sinks in a GPT-OSS checkpoint |
| `reattach-vision-tower` | restore the vision tower to a `text_only_model` Qwen3.5/3.6 export, the layout vLLM loads |

- **Write to a new directory.** The tools refuse to run in place, since writing
  over the source deletes shards they do not overwrite. Only `reset-sinks` takes
  an explicit `--in_place` (there is no undo).
- **Merge sharded EP saves first.** Every other tool rejects the per-rank layout.
- **Resume state.** Conversions of a run's own weights (`convert-to-bf16`,
  `unfuse-moe-experts`, …) keep its resume files except the optimizer shards;
  strip those before uploading. Exports that make a new base (`merge-peft-adapters`,
  `convert-to-bf16 --peft --merge_adapter`, `merge-models`, `patch-vocab`) drop
  them, `trainer_state.json` included.
- **Text-only merges.** Merging an adapter trained with `text_only_model: true`
  writes a text-only model with the run's tokenizer and none of the base's
  processor files. For Qwen3.5/3.6, vLLM loads it only after
  `reattach-vision-tower`; SGLang loads it as-is.

## Serving the result

A gathered checkpoint loads with `from_pretrained`. vLLM and SGLang serve it
only where the pinned engine reads that family's layout:

- **No serving path** on either engine: Mistral4, Ling 3.0, Ring, Inkling,
  GLM-5 Next, DeepSeek-V4 and Zaya.
- **Every other family** serves its normal save as is: each saver writes the
  layout that family's engine loader reads.
- **Fused experts on a per-expert-only loader** can be dropped without an error.
  This happens only with a checkpoint that still holds fused experts (a save
  whose layout revert warned, or a fused checkpoint from elsewhere); rewrite it
  with `unfuse-moe-experts` first
  ([per-engine loaders](../agent-docs/reference/checkpoints.md#serving-on-vllm--sglang) ↗).
- **Sharded EP saves** need `merge-ep-shards` first.
- **LoRA runs** serve as base plus adapter, or merged.

Serving during training, for the RL methods, is
[Rollout Servers](rollout-servers.md).

## Uploading to the HuggingFace Hub

Upload a final model yourself. A gathered checkpoint is a plain HF model
directory, so the standard Hub CLI works:

```bash
hf auth login          # once, or set HF_TOKEN
hf upload my-org/my-model checkpoints/sft-qwen3-4b-ultrachat/checkpoint-1000
```

Upload the checkpoint directory (or a converted copy), not the whole
`output_dir`, and leave out its training state, or it goes public with the
weights:

- `trainer_state.json` and `scheduler.pt`;
- `optimizer*` and `rng_state*`, unless the run set `save_only_model: true`;
- `router_balancing_biases.pt` (bias balancing);
- `reference_logps.pt` (DPO/KTO `precompute_ref_log_probs`, or offline GRPO full
  fine-tuning at `kl_beta > 0`);
- `prefetch_pending-*.pt` (async GRPO with prefetch): drawn but untrained
  prompts, answers included;
- `resume_adapter/` and `resume_adapter.json` (`merge_expert_lora_on_save` or
  embedding LoRA).

For a LoRA run, upload the adapter directory, or merge it with
`merge-peft-adapters` for a standalone model.

The inherited `push_to_hub` / `hub_strategy` upload is not tested end to end.
It pushes `output_dir` from global rank 0 once each checkpoint is complete,
minus the `checkpoint-*` and `_`-prefixed entries: the model files,
`log/run.log` and, for online or async GRPO, the `completions/` prompts. `hub_strategy: checkpoint` or `all_checkpoints` also
uploads training state; `end` uploads nothing
([details](../agent-docs/reference/checkpoints.md#interrupted-saves) ↗).

### Model card

Every checkpoint Halo writes, full model, adapter or tool output, holds a
`README.md` model card tagged `halo`, so the upload lists under that Hub tag.
Before you publish, fill in its body and replace a local-path `base_model` with
the Hub id. A tool whose source card has malformed metadata copies that card
untagged and warns; an unmerged `convert-to-bf16 --peft` raises instead.

Shard layouts, merge flags and the full resume mechanics:
[Checkpoints](../agent-docs/reference/checkpoints.md) ↗ ·
[Model Merging](../agent-docs/reference/model-merging.md) ↗.
