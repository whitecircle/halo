---
name: parallelism
description: >-
  Recommend a VALID ParallelismConfig and REJECT unsupported EP/CP/TP/ETP/PP combos
  BEFORE a GPU run. Auto-fire whenever the user: asks how to configure or pick a
  parallelism mode (expert/context/tensor/expert-tensor/pipeline parallelism,
  EP/CP/TP/ETP/PP, DDP/FSDP/HSDP); sets or asks about --expert_parallel_size /
  --context_parallel_size / --tensor_parallel_size / --expert_tensor_parallel_size /
  --pipeline_parallel_size / --pipeline_split / --pipeline_schedule / ep_scope / ep_size /
  cp_size / tp_size / expert_tp_size / pp_size / nvlink_domain_size; asks "which parallelism
  for this MoE / dense model / seq len / 8-GPU node / multi-node" or how to compute
  data_parallel_size; or hits a parallelism error/hang (DeepEP combine deadlock,
  NCCL hang, "must divide", "node-local", "not supported", ep4 crash ~30s,
  ParallelismConfig raise). Wrong advice wastes multi-GPU runs — this is the gate.
allowed-tools:
  - Read
  - Grep
  - Glob
---

# Parallelism configuration gate

Recommend a **valid** `ParallelismConfig` (`src/distributed/parallelism_config.py`)
and **reject** unsupported combos before the user burns GPU hours. Be exact — the
full reference matrix with per-row source citations is in **[matrix.md](matrix.md)**.
Read it before giving a verdict on any non-trivial combo.

## PP is not available in this release
The verdict on any `pipeline_parallel_size > 1` request is **not runnable — shard with EP/TP/CP**.
The schedule engine is not shipped: `parallelism_config_from_args` (`src/training/parallelism_args.py`)
rejects `pipeline_parallel_size > 1` at config time, before any rank math, and `PipelineRuntime`
raises `NotImplementedError` on construction. What ships is the seam — the config surface, the rank
math, the validators, the trainer gates, the stage/loss/checkpoint contracts — so every PP row below
describes a validator or contract, never a launchable topology
(`agent-docs/parallelism/pipeline-parallelism.md`).

## Source of truth
- `src/distributed/parallelism_config.py` — the dataclass + `__post_init__` →
  `_validate()` sub-validators. Every config-time `raise` lives here.
- `src/trainers/mixins/ep_introspection.py` `_setup_ep_gradient_checkpointing` —
  a **runtime** re-check of the multi-group >2-rank EP guard for hand-built configs
  that skipped validation (config-time raise is `_validate_single_domain_multigroup_ep`).
- `CLAUDE.md` "## Parallelism" table + DeepEP / trainer-support notes.
- User docs to cross-link: `agent-docs/parallelism/{expert,context,tensor,expert-tensor}-parallelism.md`,
  `agent-docs/parallelism/{data-parallelism,multi-node,data-loading}.md`.

## Key facts (from the code)
- **The allowlist decides.** `SUPPORTED_AXIS_SETS` is checked first, before any rank
  math: plain DP, each axis alone (**EP, ETP, TP, CP, PP**), and **EP+TP, EP+CP,
  EP+ETP, PP+EP, PP+ETP**. Anything else is rejected with the mechanism from
  `AXIS_SET_MECHANISMS`, or a generic "no validated composition" when unlisted. The three
  PP entries are admitted by the allowlist but unreachable — the release gate above rejects
  `pp_size > 1` first.
- `ep_group_size = ep_size * expert_tp_size` (the full EP process group).
- `stage_world_size = world_size // pp_size`, then
  `data_parallel_size = stage_world_size // max(tp_size, cp_size, expert_tp_size)`;
  if that max is 1, `dp_size = stage_world_size`. **EP alone never reduces DP** (EP is
  orthogonal to DP). Only TP, CP, ETP, and PP reduce distinct-batch count.
- Locality unit is `nvlink_domain_size` (auto = `gpus_per_node`; set it only on
  NVL72/MNNVL). "Node-local" below means "within one NVLink domain".
- `ep_scope` defaults to `"auto"` in both the dataclass and on the CLI/YAML
  (`src/args/distributed_args.py`). `"auto"` resolves to `node` if
  `ep_group_size <= nvlink_domain_size`, else `global`.
- PP flags parse but nothing runs: `--pipeline_parallel_size`, `--pipeline_microbatches`,
  `--pipeline_schedule` (`1f1b` | `gpipe`), `--pipeline_split`.

## How to pick a mode
1. **Dense (non-MoE) model?** EP/ETP do nothing — use **TP** (memory/large hidden)
   or **CP** (long seq), else plain FSDP2 DDP (`dp_size = world_size`).
2. **MoE model?** Start with **EP** (orthogonal to DP, keeps full DP). For finer
   expert sharding add **TP** (EP+TP) or **ETP** (EP+ETP) — never both. For long
   sequences on MoE use **EP+CP** (node-local EP only).
3. **Long sequence, OOM on activations?** add **CP** (Ulysses, node-local). CP
   reduces DP by `cp_size`.
4. **Weights/attention too big for one GPU?** **TP** (node-local DTensor). TP
   reduces DP by `tp_size`.
5. **Model too deep for one rank, across nodes?** **PP** is the axis for it — the
   outermost one, the only one meant to cross NVLink domains — but it is **not
   available in this release** (above). Shard the depth with **EP** (MoE) or **TP**,
   widen the job, or cut `max_length`; never advise a `pipeline_parallel_size > 1` run.
6. Compute `dp_size` with the formula above and sanity-check it's >= 1 and the
   global batch is what you intend.

## Hard node-local / divisibility constraints
- TP, CP, ETP must be **NVLink-local**: each must divide `nvlink_domain_size` and
  be `<= nvlink_domain_size`.
- `ep_group_size` must divide its scope and not exceed it: `nvlink_domain_size` for
  `node`; for `global`, `stage_world_size` (`world_size // pp_size`), which it must
  also tile as equal contiguous per-domain blocks.
- EP+TP: `ep_size` must be a multiple of `tp_size`. Node-local EP+TP uses
  `ep_size == tp_size`; cross-node EP uses `k*tp_size`.
- `world_size % gpus_per_node == 0` and `world_size % nvlink_domain_size == 0`.

## REJECT THESE (verdict = do not run)
- **TP + CP** (`tp_size>1 and cp_size>1`) — DTensor mesh conflicts with CP groups.
- **TP + ETP**, with or without EP (`tp_size>1 and expert_tp_size>1`) — attention-TP and
  expert-TP are mutually exclusive; use EP+TP **or** pure/EP+ETP.
- **ETP + CP** (`expert_tp_size>1 and cp_size>1`) — ETP sub-EP dispatch groups break
  CP sequence reconstruction.
- **Multi-group >2-rank EP on one NVLink domain** — the predicate is
  `num_nvlink_domains == 1 and ep_size > 2 and nvlink_domain_size > ep_group_size`
  (`is_racy_single_domain_multigroup_ep`): DeepEP combine barriers race FSDP2's DP-wide NCCL.
  **Rejected at config time** (`_validate_single_domain_multigroup_ep`); the trainer re-checks
  hand-built configs. Use `ep_size=2`, `ep_size = nvlink_domain_size` (the **domain**, not the
  node — they differ on NVL72), or `ep4 + etp2` — the supported 4-way expert split across 8 GPUs,
  since `ep_group_size = ep_size * expert_tp_size` fills the domain. `ep4 + tp2` lands on the same
  rejection: attention TP leaves `ep_group_size` untouched. The measured evidence and the two
  buffer failure modes live once, in **[matrix.md](matrix.md)** → row *Multi-group >2-rank EP on
  one NVLink domain*; every other skill links there instead of restating them.
- **EP+CP with `ep_scope="global"`** (cross-NVLink-domain EP) — incompatible with CP.
- **Node-local EP+CP where `ep_group_size != nvlink_domain_size`** — EP must be
  NVLink-orthogonal to CP.
- **Any node-local size exceeding / not dividing the NVLink domain** (TP, CP, ETP,
  or `ep_group_size`) — "must divide" / "cannot exceed domain" raises.
- **TP with indivisible attention heads** (`num_attention_heads % tp_size != 0`, or
  non-MLA GQA `num_key_value_heads % tp_size != 0`) — a split inside a head corrupts
  attention; `parallelize_attention.py` raises.
- **HSDP with TP / ETP / EP** — `_validate_hsdp` rejects all three; **PP+HSDP** is
  refused by `_validate_pipeline_parallel`. HSDP wraps the standard DP path only:
  **pure DP or CP**. (EP already shards over the EP group; PP cannot restrict a 2-D mesh
  to a stage's rank block.)
- **Every PP pairing outside the expert axes** — `PP+TP`, `PP+CP`, `PP+EP+TP`, `PP+EP+ETP`
  (`PP+EP` and pure `PP+ETP` are supported, both together are not). Plus
  these PP-specific config-time raises: `pp_split` length mismatch or an entry < 1 (the
  sum-vs-layer-count check runs later, at model split); a stage that is
  not a whole NVLink domain; a 1-rank stage; `fsdp_shard_ep1_experts=False`; `use_hsdp=True`;
  `fsdp_reshard_after_forward=True`; `fsdp_reshard_after_backward=False`; `lowp_precision != "bf16"`;
  expert LoRA; and a PP-only knob (`pp_split`/`pp_microbatches`/`pp_schedule`) set at `pp_size=1`.
  At trainer construction PP also rejects `save_sharded_ep`, PEFT/LoRA, a live
  `ref_model`, `activation_offloading`, reentrant gradient checkpointing, `torch_compile`, a
  missing `max_length`, an image-bearing VLM run, and `per_device_eval_batch_size !=
  per_device_train_batch_size`. Full list: `matrix.md`.
- **`fsdp_shard_ep1_experts=False` with TP or CP**, and **`fsdp_reshard_after_forward=True`
  wherever an expert-distribution group exists** — the gate is `is_ep_mode`
  (`ep_group_size > 1`), so pure ETP (`ep_size=1`, `expert_tp_size>1`) is rejected alongside EP —
  **or with TP at `dp_size>1`**. `_validate_fsdp_settings` raises.
- **LoRA/PEFT with TP** (`tp_size > 1`) — adapters are plain tensors outside the TP graph, so the
  replicated matrix diverges per rank and the sharded one is corrupted by the TP grad sync.
  `_validate_lora_tp_compatibility` (`src/trainers/mixins/validation.py`) raises at trainer
  construction, **native EP expert LoRA included** (its own message: the expert adapters live on
  the EP-distributed weights, which every TP gate skips by param identity). CP and pure ETP leave
  attention unsharded, so LoRA is fine there.
- **CP with a trainer that doesn't support it** (DPO, KTO, GRPO online/offline/env/SDPG,
  reward, classification, distillation, embedding) — see matrix.md trainer table.
- **PP with a trainer that doesn't support it** (online/env GRPO, SDPG, both distillation
  trainers, embedding) — `_supports_pp`. Moot while the release gate above rejects `pp_size > 1`
  first, but it is the gate a PP-capable trainer will be declared against.

When rejecting, cite the exact reason and offer the nearest valid alternative
(e.g. "ep_size=4 on 8 GPUs deadlocks → use ep_size=2, ep_size=8 (the whole domain), or
ep4+etp2; EP+TP does not help — attention TP leaves `ep_group_size` at 4").

## Sources of truth
`matrix.md` + `agent-docs/parallelism/` document the supported/rejected combos. The code is the **ultimate**
authority: `src/distributed/parallelism_config.py` (the validators) and
`src/trainers/mixins/ep_introspection.py` (the runtime EP-GC setup guard) decide what actually raises or
deadlocks — when a doc, this skill, or memory disagrees, or you are unsure, read those files before
advising. (`CLAUDE.md`: docs-first, the code wins.) Related skill: `data` (sharded dataset ↔
`data_parallel_size`).
