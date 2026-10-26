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

A **gathered** save (the default for every parallelism mode) is already a single
`from_pretrained`-loadable HF checkpoint, and so is a PP save on a shared output filesystem. The
`scripts/after_training/` merge tools are only for the opt-in **per-rank sharded EP save**
(`save_sharded_ep`), per-node PP directories and **LoRA** adapters. Owner pages:
`agent-docs/reference/checkpoints.md` (layouts, saving, resume) and
`agent-docs/reference/scripts-reference.md#post-training-scripts` (every tool's flags and input guards).
Read the script's `--help` before running it — don't guess the flags.

## Two things people get wrong (read first)

- **Resume: set `resume_from_checkpoint: true` (or a `checkpoint-N` path), re-run the same command, and
  leave `model_name_or_path` at the base.** The scripts repoint the policy's weights at the checkpoint
  themselves; the base keeps feeding the DPO/KTO/SDPG reference and the dataset-compat check, so
  repointing it yourself moves the reference onto trained weights. Exact resume needs the saved topology
  fingerprint; a mismatch warm-restarts, and under PP it raises. Per-mode table and every restore verdict:
  `agent-docs/reference/checkpoints.md#resume-by-parallelism-mode` and
  `#warm-restart-vs-exact-resume-torchrun`.
- **A per-rank sharded checkpoint is NOT directly loadable.** Every loader refuses `metadata.format`
  `ep_sharded` and names the merge tool. Run the merge first. The merged directory (and a
  `merge_pp_stages.py` output) carries every rank's optimizer shard, so it resumes exactly on the
  topology that wrote it.

## "I trained with X → single HF checkpoint" decision table

| Trained with | Gathered save (default) | If you used the sharded option | To get one HF checkpoint |
|---|---|---|---|
| Dense FSDP2 / DDP | loadable | — | use directly |
| LoRA / QLoRA | adapter-only dir | — | **`merge_peft_adapters.py`** (`--task classification` for reward/clf); it refuses a native EP expert-LoRA adapter — train with `merge_expert_lora_on_save: true` instead |
| Embedding LoRA (in-place injected) | loadable, adapters folded in | — | use directly (`SentenceTransformer(path)`) |
| TP-only | loadable | — (no per-rank TP save) | use directly |
| EP — every family (each family's `MoEFamilySpec.model_types` in `halo/models/moe/family_specs.py`; read them there, and `moe_model_types()` for the resolved set) | loadable | `save_sharded_ep` → `ep_sharded` | **`merge_ep_shards.py`** |
| CP / EP+CP | loadable | **rejected** — `save_sharded_ep` raises under Ulysses attention | use gathered directly |
| ETP (`expert_tp_size > 1`), **any** multi-EP-group topology (`ep_group_size != world_size` — plain `ep2` on 8, or EP+TP with `ep_size < world_size`), native expert LoRA, `merge_expert_lora_on_save`, a run with no EP MoE layers (dense, or MoE at `use_grouped_gemm: false`), a load that converted names the merge cannot respell (Step-3.7 Flash, a vendor namespace, a SigLIP tower), PP, multi-node non-shared output FS | loadable | **rejected at construction** (`reject_unmergeable_ep_sharded_save`, re-checked at save; PP by the pipeline gate) | use gathered directly |
| PP | per-stage parts + merged index under the hub's names, loadable and servable as the EP-gathered save | per-node output FS: each node holds its own stages' parts | use directly; per-node directories **`merge_pp_stages.py`** first |

Other tools: `convert_to_bf16.py`, `quantize_to_lowp.py` (an artifact tool, not a speedup),
`reset_sinks.py`, `unfuse_moe_experts.py` (only for a per-expert-only serving loader),
`merge_models.py`, `reattach_vision_tower.py` — what each does and refuses:
`agent-docs/reference/scripts-reference.md#post-training-scripts`.

**Training sidecars:** export through `merge_peft_adapters.py` or `convert_to_bf16.py`, which re-apply
`router_balancing_biases.pt` and `training_provenance.json` (`apply_training_sidecars`); any other
assembly serves the base routing and sinks with no error
(`agent-docs/reference/checkpoints.md#troubleshooting`).

## Verify the result

After a merge, confirm the output loads and has no leftover shard keys:

```bash
python -c "from transformers import AutoModelForCausalLM as M; m=M.from_pretrained('<out>', trust_remote_code=True); print('loaded', sum(p.numel() for p in m.parameters()))"
# MoE for vLLM: grep the index for the per-expert keys that family's server loader reads
python -c "import json; ix=json.load(open('<out>/model.safetensors.index.json')); print(ix.get('metadata',{}).get('format'), [k for k in ix['weight_map'] if '.shard_' in k][:5] or 'no shard keys')"
```

## Sources of truth

`agent-docs/reference/checkpoints.md` is the documented behavior — check it first; the module map is
`agent-docs/reference/architecture.md`. The code is the **ultimate** authority: when a doc, this skill, or
memory disagrees with the code — or you are unsure — read `halo/checkpoint/` and the
`scripts/after_training/` script you are about to run before you assert or act. (`CLAUDE.md`:
docs-first, but the code wins.)
