---
name: checkpoints
description: >-
  Turn a finished Halo training run into a usable artifact, and resume
  correctly. USER-INVOKED ONLY. Use when the user asks to "merge EP shards",
  "load my EP/MoE checkpoint into vLLM", "merge a LoRA/QLoRA adapter", "convert
  to bf16", "quantize to fp8/fp4", "reset GptOss sinks", "unfuse MoE experts",
  "merge several checkpoints into one",
  or "resume from a checkpoint" — anything about checkpoint format, the
  scripts/after_training/ tools, or resume semantics (EP/TP/CP/FSDP/LoRA).
disable-model-invocation: true
allowed-tools:
  - Read
  - Bash
  - Grep
  - Glob
---

# checkpoints — post-training & resume

A **gathered** save (the default for every parallelism mode — EP/ETP/EP+TP/EP+CP/TP/CP/FSDP2) is
already a single `from_pretrained`-loadable HF checkpoint. PP has its own saver
(`save_pp_checkpoint`, selected first): one or more safetensors parts per stage (split at
`save_max_shard_size`) under **global** parameter names plus a merged index — also directly
`from_pretrained`-loadable, and deliberately carrying no `metadata.format` marker. **No PP
checkpoint exists in this release**: PP is not yet available
(`agent-docs/parallelism/pipeline-parallelism.md`) and `pipeline_parallel_size > 1` is rejected at config
time, so the PP save/load path is a shipped contract, not a layout a user can produce.
The `scripts/after_training/` merge tools are
only for the opt-in **per-rank sharded EP save** (`save_sharded_ep`) and for **LoRA** adapters. Per-script detail, the format matrix, and the footguns are in **`reference.md`** — read it
before acting. Authoritative doc: `agent-docs/reference/checkpoints.md`.

## Two things people get wrong (read first)

- **Resume: set `resume_from_checkpoint: true` (or a `checkpoint-N` path) and re-run the same
  command; leave `model_name_or_path` at the base.** The scripts repoint the policy's weights at the
  checkpoint themselves (`resolve_resume_weights_source`, `src/training/environment.py`) under
  EP/ETP/EP+TP/EP+CP/CP/TP and, at the default `use_grouped_gemm: true`, every run, dense included
  (an adapter-only checkpoint, and a merged one — `merge_expert_lora_on_save` or embedding LoRA —
  through its `resume_adapter.json` marker, keeps the base and restores the adapter). The base keeps feeding the
  DPO/KTO/SDPG reference and the dataset-compat check, so repointing `model_name_or_path` yourself
  moves the reference onto trained weights. The model loads the
  trained weights at construction and the loader skips the re-read; it restores `trainer_state.json`,
  the LR scheduler from `scheduler.pt`, LoRA adapters, the router-balancing
  biases, the frozen reference scores (`reference_logps.pt`: DPO/KTO precompute,
  offline GRPO KL), and the **optimizer state from the per-rank shards when the topology
  fingerprint matches** (`OptimizerStateFingerprint`); a mismatch warm-restarts instead, and a
  matched restore that fails on any rank raises on every rank unless
  `allow_optimizer_warm_restart: true`. Only a
  `use_grouped_gemm: false` run with no EP/ETP/CP/TP reloads weights (FSDP2 via
  `load_full_state_dict`; DDP and accelerate-managed FSDP via the base Trainer loader).
  A model built from anything but the checkpoint (a custom
  script) makes the loader **refuse the resume** under EP/CP and TP+DP rather than continue on the base
  weights (`loader.py`); only an adapter-only checkpoint, which ships no base weights to check
  against, still resumes quietly. A merge-on-save checkpoint inverts the check: it refuses a model
  built from its own merged weights (the delta would apply twice), and an unmarked merged checkpoint
  under an adapter run raises. `load_best_model_at_end` is refused for a full fine-tune under CP, on
  a MoE carrying EP or grouped-GEMM wrappers (`ep_size: 1` included), and under TP+DP — export the
  best checkpoint directly. Owner:
  `agent-docs/reference/checkpoints.md#resuming-training`.
- **A per-rank sharded checkpoint is NOT directly loadable.** `load_full_state_dict` refuses
  `metadata.format` `ep_sharded` and raises pointing at the merge tool. Run the merge first.

## "I trained with X → single HF checkpoint" decision table

| Trained with | Gathered save (default) | If you used the sharded option | To get one HF checkpoint |
|---|---|---|---|
| Dense FSDP2 / DDP | loadable | — | use directly |
| LoRA / QLoRA (any mode) | adapter-only dir | — | **`merge_peft_adapters.py`** (`--task classification` for reward/clf); it refuses a native EP expert-LoRA adapter — train with `merge_expert_lora_on_save: true` instead |
| Embedding LoRA (in-place injected) | loadable, adapters folded in | — | use directly (`SentenceTransformer(path)`) |
| TP-only | loadable | — (no per-rank TP save) | use directly |
| EP — every family (each layer class in `src/distributed/expert_parallel/layers/` declares its own `HF_MODEL_TYPES`; read them there, and `supported_ep_merge_model_types()` for the resolved set) | loadable | `save_sharded_ep` → `ep_sharded` | **`merge_ep_shards.py`** |
| CP / EP+CP | loadable | **rejected** — `save_sharded_ep` raises under Ulysses attention | use gathered directly |
| ETP (`expert_tp_size > 1`), **any** multi-EP-group topology (`ep_group_size != world_size` — plain `ep2` on 8, or EP+TP with `ep_size < world_size`), native expert LoRA, `merge_expert_lora_on_save`, a run with no EP MoE layers (dense, or MoE at `use_grouped_gemm: false`), a load that converted names the merge cannot respell (Step-3.7 Flash, a vendor namespace, a SigLIP tower), multi-node non-shared output FS | loadable | **rejected at construction** (`validate_ep_sharded_save`, re-checked at save) | use gathered directly |
| PP *(not producible — PP unavailable)* | per-stage parts + merged index, loadable | — | use directly |

Extra normalizers: **`convert_to_bf16.py`** (fp32→bf16, keeps norms fp32), **`quantize_to_lowp.py`**
(mxfp8/mxfp4/nvfp4 export — an artifact tool, not a speedup), **`reset_sinks.py`** (neutralize GptOss
attention sinks for FA2 inference), **`unfuse_moe_experts.py`** (fused → per-expert keys for engines
that need them), **`merge_models.py`** (linear/slerp/task-arithmetic/TIES across several checkpoints),
**`reattach_vision_tower.py`** (rebuild the VLM wrapper layout around a `text_only_model` export, so
the pinned serving images can load it).

## Training sidecars — carry them or serve a different model

A run's output dir holds two files the weights cannot: **`router_balancing_biases.pt`** (the trained
router-balancing biases under `moe_balancing: bias_update` / `bias_update_transient`; the **only**
copy for families with no native slot) and **`training_provenance.json`** (the GptOss sink policy —
an adapter trained under `reset_sinks` must be merged onto neutralized sinks).
`apply_training_sidecars(model, source_dir)` (`src/checkpoint/tool_io.py`) is the one seam
that re-applies both; `merge_peft_adapters.py` and `convert_to_bf16.py` call it and report what it
did. Assembling an export any other way serves the **base** routing — near-tied top-k picks flip vs
training, with no error anywhere. Detail in `reference.md`.

## The scripts (`scripts/after_training/`)

All run inside the Docker image (tools on `PATH`, no prefix). `merge_ep_shards` /
`unfuse_moe_experts` / `merge_models` are pure safetensors rewriters (CPU, one tensor at a time);
`quantize_to_lowp` streams safetensors too, one input shard at a time (quantizing on the GPU when
one is visible); `merge_peft_adapters` / `convert_to_bf16` load a model. CLI flags,
family-support limits, and the resume/format internals are in **`reference.md`** — read it, don't
guess the flags.

## Verify the result

After a merge, confirm the output loads and has no leftover shard keys:

```bash
python -c "from transformers import AutoModelForCausalLM as M; m=M.from_pretrained('<out>', trust_remote_code=True); print('loaded', sum(p.numel() for p in m.parameters()))"
# MoE for vLLM: grep the index for the per-expert keys that family's server loader reads
python -c "import json; ix=json.load(open('<out>/model.safetensors.index.json')); print(ix.get('metadata',{}).get('format'), [k for k in ix['weight_map'] if '.shard_' in k][:5] or 'no shard keys')"
```

## Sources of truth

`agent-docs/reference/checkpoints.md` is the documented behavior — check it first. The code is the **ultimate**
authority: when a doc, this skill, `reference.md`, or memory disagrees with the code — or you are unsure —
read the actual file before you assert or act. Save/load orchestration lives in
`src/distributed/checkpoint/` — `save.py` (the mode ladder + per-mode save,
incl. PP), `write.py` (the shared streaming writer every gathered save funnels through),
`loader.py` (every resume path), `fingerprint.py` (the optimizer-shard topology gate),
`optimizer.py` (the per-rank optimizer shards + LR scheduler), `peft.py` (`restore_adapters`),
`context.py` (the trainer-state snapshot, mode flags included, the savers and loader read),
`coordination.py` (the rank consensus shared by the resume paths).
Per-mode mechanics:
`src/distributed/checkpoint/ep_save.py`, `src/distributed/expert_parallel/expert_weights.py`,
`src/distributed/checkpoint/tp_save.py`, `src/checkpoint/` (`format.py` layout + dtype, `config_export.py` the
config contract, `adapters.py` the saved-PEFT shape and the shared `merge_adapter_into_base`,
`tool_io.py` the tool-side directory I/O, `atomic.py` the staged, fsynced checkpoint-file publication,
`shard_writer.py` the incremental parts writer,
`fp8_dequant.py` the streaming fp8 → bf16, `model_card.py` the `halo` Hub tag on the model card of
every checkpoint Halo writes, tool conversions included), and the
`scripts/after_training/` script you are about to run.
(`CLAUDE.md`: docs-first, but the code wins.)
