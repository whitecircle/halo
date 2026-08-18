# Parallelism reference matrix

Every row traces to a `raise` in `src/distributed/parallelism_config.py`
(`_validate_*` sub-validators), the runtime guard in `src/trainers/mixins/ep_introspection.py`
`_setup_ep_gradient_checkpointing`, or a `CLAUDE.md` "## Parallelism" note. The owning
validator is named inline; line numbers drift, so grep the method name. `domain` =
`nvlink_domain_size` (auto = `gpus_per_node`; only differs on NVL72/MNNVL).
`ep_group_size = ep_size * expert_tp_size`.

**PP is not available in this release.** The seams ship — the config surface, the rank math, every
validator below, the trainer gates — but not the schedule engine:
`parallelism_config_from_args` (`src/training/parallelism_args.py`) rejects
`pipeline_parallel_size > 1` before any rank math, and `PipelineRuntime` raises on construction.
Read every PP row as the shape a validator enforces, never as a launchable topology
(`agent-docs/parallelism/pipeline-parallelism.md`).

## data_parallel_size formula

```
stage_world_size = world_size // pp_size        # PP is the outermost split
dp_divisor       = max(tp_size, cp_size, expert_tp_size)
data_parallel_size = stage_world_size // dp_divisor   if dp_divisor > 1
                   = stage_world_size                 otherwise   # EP-only / DDP
```
EP is **always orthogonal to DP** — `ep_size` is not in the divisor. PP is, via
`stage_world_size`: a whole pipeline chain consumes one batch.
Source: `parallelism_config.py` `__post_init__`.

## SUPPORTED modes

| Mode | Trigger | dp_size | Key constraints | Source |
|------|---------|---------|-----------------|--------|
| **Standard DDP / FSDP2** | all sizes = 1 | `world_size` | — | `__post_init__` |
| **Grouped-GEMM only** | `ep_size=1`, `use_grouped_gemm=True`, MoE | `world_size` | applies EP wrappers, no inter-rank comm (DeepEP no-op); needs torchrun | `needs_ep_wrappers`, CLAUDE.md grouped-GEMM note |
| **EP only** | `ep_size>1` | `world_size` (EP orthogonal) | `ep_group_size` must divide & not exceed scope (domain for `node`, world for `global`). Single-domain pure EP: use **ep_size=2 or ep_size=nvlink_domain_size** only | `_validate_ep_group`; `_validate_single_domain_multigroup_ep` |
| **Multi-group EP across nodes** | `num_ep_groups>1` AND `num_nodes>1`, `expert_tp_size==1` (node-local EP across domains, OR a cross-node EP group replicated within a larger cluster) | `world_size` (EP orthogonal) | **Supported via deferred cross-replica sync** (`EPConfig.is_deferred_dp`): FSDP shards non-expert params over the EP group; cross-replica DP average deferred to a post-backward sweep so nothing races the DeepEP combine | `config.py` `is_deferred_dp`; `mixin._sync_deferred_expert_grads`; `ep_introspection.py` |
| **TP only** | `tp_size>1` | `world_size / tp_size` | `tp_size` divides world; node-local DTensor; attention heads must divide `tp_size` (`parallelize_attention.py`) | `_validate_tp` |
| **CP only** | `cp_size>1` | `world_size / cp_size` | `cp_size <= domain` and divides domain (Ulysses node-local) | `_validate_cp_locality` |
| **Pure ETP** (`ep_size=1`) | `expert_tp_size>1`, `ep_size=1` | `world_size / expert_tp_size` | MoE-only; ETP divides domain; node-local; experts replicated, FFN sharded. The verified ETP shape | `_validate_expert_tp`; CLAUDE.md table |
| **EP + CP** | `ep_size>1`, `cp_size>1` | `world_size / cp_size` (only CP reduces DP) | EP must be **node-local** (`ep_scope="node"`) AND `ep_group_size == domain`. `ep_scope="global"` rejected | `_validate_ep_cp` |
| **EP + TP** | `ep_size>1`, `tp_size>1` | `world_size / tp_size` (only TP reduces DP) | `ep_size` multiple of `tp_size`; TP node-local & divides domain. On **>1 NVLink domain** the EP group must be a **single** one (`ep_size == world`, `ep_scope="global"`) — multi-domain multi-group EP+TP is rejected (cross-replica average incompatible with the `(dp, tp)` FSDP mesh). Within **one** domain multiple EP groups are legal: `ep2+tp2` on 8 GPUs = 4 EP groups, DP 4 | `_validate_tp` |
| **EP + ETP** (`ep_size>1` AND `expert_tp_size>1`) | both >1 | `world_size / expert_tp_size` | **Experimental, supported.** `expert_tp_size` divides the domain and the ETP groups stay node-local — exactly one per domain under `ep_scope="global"`; EP itself may be node-local or cross-node, so `ep_group_size` may exceed the domain. Expert-TP reduce runs in token space (outside DeepEP dispatch→combine), so no combine deadlock. The single-domain multi-group guard exempts only `ep_group_size == domain` (or `ep_size <= 2`); sub-domain multi-group EP+ETP with `ep_size > 2` is rejected at config time and re-checked at trainer setup | `_validate_expert_tp`; trainer guard in `ep_introspection.py`; CLAUDE.md EP+ETP note |
| **PP**, **PP + EP**, **PP + ETP** — *in the allowlist, not runnable* | `pp_size>1` | `stage_world_size / max(1, expert_tp_size)` | **Rejected by the release gate before any of this is reached.** PP is outermost: the world splits into `pp_size` contiguous blocks of whole NVLink domains and every inner axis runs unchanged inside one block. Composes with the expert axes only — EP, **or** pure ETP (`ep_size==1`), never both, and never TP/CP. All EP/ETP gradient sync runs in one deferred post-backward sweep; the ETP divisor drops the `expert_tp_size` factor (partners share a batch) | `_validate_pipeline_parallel`; `mixin._sync_deferred_expert_grads` |

Notes:
- For pure EP, prefer `ep_scope="auto"` so the scope resolves correctly; `"auto"` is
  the default in both the dataclass and `DistributedArguments`. Cross-node (`global`)
  EP needs `ep_group_size` to tile `stage_world_size` (`world_size // pp_size`) as
  equal contiguous per-domain blocks — divisibility alone is not enough.
- "Node-local" = within one NVLink domain. On a standard cluster that is one OS node
  (`domain == gpus_per_node`); on NVL72 it is the rack (set `nvlink_domain_size`).

## UNSUPPORTED combos (REJECT)

| Combo | Condition | Reason | Source |
|-------|-----------|--------|--------|
| **Any `pipeline_parallel_size > 1`** | `pp_size>1` | the schedule engine is not shipped in this release; rejected at the single production entry point, ahead of the axis-set check and every row below | `parallelism_config_from_args` (`src/training/parallelism_args.py`, raises); `PipelineRuntime.__init__` |
| **Anything outside `SUPPORTED_AXIS_SETS`** | active axis set not one of: `()`, `ep`, `etp`, `tp`, `cp`, `pp`, `ep+tp`, `ep+cp`, `ep+etp`, `pp+ep`, `pp+etp` | checked **first**, before any rank math. The message comes from `AXIS_SET_MECHANISMS` when the set has an entry, else "no validated composition — no equivalence gate has ever compared its gradients against an unsplit reference" | `_validate_capability_matrix` (raises) |
| **TP + CP** | `tp_size>1 and cp_size>1` | both groups are contiguous rank blocks, so a rank's TP partners ARE its CP partners; Ulysses also redistributes over heads TP already split, and `data_parallel_size` counts only one of the two axes | `_validate_capability_matrix` + `AXIS_SET_MECHANISMS[{tp,cp}]` |
| **TP + ETP** | `tp_size>1 and expert_tp_size>1`, `ep_size==1` | attention TP and expert TP would shard the same ranks along two axes | `_validate_capability_matrix` + `AXIS_SET_MECHANISMS[{tp,etp}]` |
| **EP + TP + ETP** | `ep_size>1 and tp_size>1 and expert_tp_size>1` | attention-TP and expert-TP mutually exclusive; use EP+TP **or** (pure/EP+)ETP | `_validate_capability_matrix` |
| **ETP + CP** | `expert_tp_size>1 and cp_size>1` | expert-TP partners must see the SAME tokens (`ReduceFromExpertTP` sums in token space) but CP hands each rank a different chunk | `_validate_capability_matrix` + `AXIS_SET_MECHANISMS[{etp,cp}]` |
| **PP + TP** | `pp_size>1 and tp_size>1` | needs ≥2 nodes to launch and was never exercised on real multi-node hardware; transformers' `replicated_with_grad_allreduce` hook re-reduces accumulated history when the schedule disables it on non-final microbatches | `_validate_capability_matrix` + `AXIS_SET_MECHANISMS[{pp,tp}]` |
| **PP + CP** | `pp_size>1 and cp_size>1` | the pipeline loss normalizer is stage-wide and carries no cancelling `× cp_size`, so every gradient comes out `cp_size` too small with **no error raised** | `_validate_capability_matrix` + `AXIS_SET_MECHANISMS[{pp,cp}]` |
| **PP + EP + TP** | all three >1 | not in the allowlist | `_validate_capability_matrix` |
| **PP + EP + ETP** | all three >1 | `ReduceFromExpertTP.backward` is model math, so it cannot be deferred like the DP sweep — at `ep_size>1` its strided all-reduce interleaves with the DeepEP combine the same backward is inside. `ep_size==1` (PP+ETP) is supported: no combine to interleave with | `_validate_capability_matrix` + `AXIS_SET_MECHANISMS[{pp,ep,etp}]` |
| **PP config-time raises** | `pp_split` len ≠ `pp_size` (its **sum** ≠ layer count raises at model-split time — `split.py` — since config never sees the model); a stage that is not a whole NVLink domain; 1-rank stages; `fsdp_shard_ep1_experts=False`; `use_hsdp=True`; `fsdp_reshard_after_forward=True`; `lowp_precision != "bf16"`; expert LoRA; a PP-only knob (`pp_split`/`pp_microbatches`/`pp_schedule`) set at `pp_size=1`, where nothing reads it | each raises with its own mechanism | `_validate_pipeline_parallel` (`fsdp_reshard_after_backward=False` under PP: `_validate_fsdp_settings`) |
| **PP trainer-construction raises** | PEFT/LoRA; a live `ref_model`; missing precomputed ref-logprob columns; `activation_offloading`; reentrant gradient checkpointing; `torch_compile`; positional `model`/`args` | a stage cannot satisfy full-model module names, and TRL's `training_step` wrap never engages under the PP schedule | `src/trainers/mixins/pp_gates.py` (PEFT, offloading) + `src/trainers/mixins/pipeline.py` (reentrant GC, torch_compile, VLM, collator contracts) |
| **Multi-group >2-rank EP on one NVLink domain** | `num_nvlink_domains == 1 and ep_size > 2 and nvlink_domain_size > ep_group_size` (e.g. ep4 on an 8-GPU domain) | concurrent DeepEP intra-node combine barriers race FSDP2's DP-wide NCCL, GC on or off; CDMC=1 doesn't fix it. Reproduced on an 8-GPU NVLink domain against a clean `ep8` control: the `legacy` (V1) buffer deadlocks around step 2, the `elastic` default faults with `Invalid access of peer GPU memory over nvlink`. **Rejected at config time**; the trainer re-checks hand-built configs. `ep4 + tp2` lands here too (TP leaves `ep_group_size` untouched). `ep4 + etp2` **passes** the predicate — `ep_group_size = 4*2 = 8` fills the domain — and is GPU-validated on an 8-GPU domain; it is the way to get 4-way expert sharding across 8 GPUs. Use `ep_size=2`, `ep_size = nvlink_domain_size`, or `ep4 + etp2`. (Multi-DOMAIN multi-group EP is supported — see the deferred-sync row above.) | `is_racy_single_domain_multigroup_ep`; `_validate_single_domain_multigroup_ep` (raises); `ep_introspection.py` re-check |
| **EP+CP, cross-domain EP** | `ep_group_size>1 and cp_size>1 and ep_scope=="global"` | cross-NVLink-domain EP interleaves members across domains → not orthogonal to node-local CP groups (NCCL hang / corrupt routing) | `_validate_ep_cp` (raises) |
| **EP+CP, EP not full domain** | `ep_scope=="node" and ep_group_size != domain` (with cp>1) | node-local EP+CP requires `ep_group_size == nvlink_domain_size` | `_validate_ep_cp` (raises) |
| **EP+TP+CP** | implied by the TP+CP exclusion | not in the allowlist | `_validate_capability_matrix` |
| **node-local size > domain** | `cp_size`, `tp_size`, `expert_tp_size`, or `ep_group_size` `> nvlink_domain_size` (node scope) | must stay on NVLink — "cannot exceed the NVLink domain" | `_validate_cp_locality`, `_validate_tp`, `_validate_expert_tp`, `_validate_ep_group` |
| **size doesn't divide scope** | any of `tp/cp/expert_tp/ep_group_size` not dividing domain (or world for global EP) | "must divide" — orphaned ranks / no valid groups | `_validate_tp`, `_validate_cp_locality`, `_validate_expert_tp`, `_validate_ep_group` |
| **EP+TP, ep_size not multiple of tp_size** | `ep_size>1 and ep_size % tp_size != 0` | each EP group must span whole TP groups (else replicas share a DP group → double-counted grads / misrouted tokens) | `_validate_tp` (raises) |
| **TP with indivisible heads** | `num_attention_heads % tp_size != 0`, or GQA `num_key_value_heads % tp_size != 0` (non-MLA) | `ColwiseParallel` splits Q/K/V evenly with no head awareness → a split inside a head silently corrupts attention | `parallelize_attention.py` (raises; MLA-aware) |
| **cross-node ETP off NVLink** | `ep_scope="global"` and `expert_tp_size != EP members per domain`, or `ep_size != domains spanned` | exactly one ETP group per NVLink domain is required, so the ETP all-reduce stays on NVLink | `_validate_expert_tp` (raises) |
| **HSDP with TP / ETP / EP / PP** | `use_hsdp=True and (tp_size>1 or expert_tp_size>1 or ep_size>1 or pp_size>1)` | TP / ETP build their own (dp, tp) mesh; EP already shards over the EP group and HSDP would race the combine; a stage-sized 2-D mesh cannot be built from a rank block. HSDP wraps the standard DP path only — **pure DP or CP** | `_validate_hsdp`, `_validate_pipeline_parallel` (raise) |
| **FSDP flags vs axes** | `fsdp_shard_ep1_experts=False` with TP or CP; `fsdp_reshard_after_forward=True` wherever `is_ep_mode` (`ep_group_size > 1` — EP **and** pure ETP at `ep_size=1`), or with TP at `dp_size>1`; `fsdp_reshard_after_backward=False` with TP or PP (wired through the plain-DP/CP/EP torchrun path only), or alongside `fsdp_reshard_after_forward=True` | each raises with its own mechanism | `_validate_fsdp_settings` (raises) |
| **LoRA/PEFT with TP** | `tp_size>1` with any adapter — PEFT-wrapped, injected in place, **or** native EP expert LoRA | adapters are plain tensors outside the TP graph: the replicated matrix diverges per rank (per-rank init, never broadcast), the sharded one is corrupted by the TP replicated-grad sync. Expert LoRA gets its own message — it lives on the EP-distributed weights every TP gate skips by param identity | `_validate_lora_tp_compatibility` (`src/trainers/mixins/validation.py`, raises at trainer construction) |
| **world not divisible by node/domain** | `world_size % gpus_per_node != 0` or `% nvlink_domain_size != 0` | floor division truncates `num_nodes` / orphans trailing ranks | `_validate_node_topology`, `__post_init__` |
| **bad sizes / enums** | any `*_size < 1`; `ep_scope` not in {auto,node,global}; `nvlink_domain_size % gpus_per_node != 0`; bad `lowp_precision` | fail-fast sanity guards | `__post_init__`, `_validate_lowp` |

## Trainer × mode support

From `CLAUDE.md` "Distributed Trainers" table. All extend `DistributedTrainerMixin`.

| Trainer | EP | CP | TP | ETP | PP |
|---------|:--:|:--:|:--:|:--:|:--:|
| `DistributedSFTTrainer` (SFT) | Yes | Yes | Yes | Yes | Yes |
| `SmoothMarginPOTrainer` (SMPO) | Yes | Yes | Yes | Yes | Yes |
| `DistributedDPOTrainer` (DPO) | Yes | No | Yes | Yes | Yes (precompute-only; `sigmoid`/`hinge`/`ipo`) |
| `DistributedKTOTrainer` (KTO) | Yes | No | Yes | Yes | Yes (`apo_zero_unpaired`, precompute-only) |
| `OfflineGRPOTrainer` | Yes | No | Yes | Yes | Yes (any `kl_beta`; a construction-time sweep scores the KL reference) |
| `DistributedGRPOTrainer` (online) | Yes | No | Yes | Yes | No |
| `DistributedSDPGTrainer` (online SDPG, `DistributedGRPOTrainer` subclass) | Yes | No | Yes | Yes | No |
| `DistributedAsyncEnvironmentalGRPOTrainer` | Yes | No | Yes | Yes | No |
| `DistributedRewardTrainer` | Yes | No | Yes | Yes | Yes |
| `ClassificationTrainer` | Yes | No | Yes | Yes | Yes |
| `DistributedDistillationTrainer` | Yes | No | Yes | Yes | No |
| `DistributedSelfDistillationTrainer` | Yes | No | Yes | Yes | No |
| `EmbeddingTrainer` | Yes | No | Yes | Yes | No |

The PP column is each trainer's declared `_supports_pp`, inert while the release gate rejects
`pp_size > 1` first. Only **SFT and SMPO support CP**. The authoritative gate is the per-class `_supports_cp` /
`_supports_pp` attribute (`src/trainers/mixins/base.py`, checked in
`src/trainers/mixins/validation.py`), so a hand-built config is rejected too. Each non-CP training
script additionally calls `parallelism_config_from_args(..., supports_cp=False)`
(`src/training/parallelism_args.py`), which rejects a CLI-requested `context_parallel_size > 1` with a clear
error — passing CP there is a config error, not a silent no-op. There is no `_supports_etp`: ETP
folds into `ep_group_size` and is gated by `_supports_ep`.

### CP incompatibility list (CLAUDE.md)
CP is incompatible with trainers/paths that use:
- `logits_to_keep`
- global log-probability sums
- full-sequence pooling
- dual models (reference + policy)

That is why DPO/GRPO/reward/classification/distillation/embedding reject CP.

## Worked dp_size examples (8 GPUs, 1 node, domain=8)

| Config | ep_group_size | dp_divisor | dp_size | Valid? |
|--------|:--:|:--:|:--:|--------|
| DDP | 1 | 1 | 8 | yes |
| ep_size=8 | 8 | 1 | 8 | yes (single intra-node group) |
| ep_size=2 | 2 | 1 | 8 | yes (2-rank groups) |
| ep_size=4 | 4 | 1 | 8 | **NO** — multi-group >2-rank pure EP deadlock (use 2 or 8) |
| tp_size=8 | 1 | 8 | 1 | yes |
| cp_size=2 | 1 | 2 | 4 | yes |
| expert_tp_size=2 (ep_size=1) | 2 | 2 | 4 | yes (pure ETP) |
| ep_size=8 + cp_size=8 | 8 | 8 | 1 | yes — canonical EP+CP (`cp_size == ep_group_size == domain`) |
| ep_size=8 + cp_size=2 | 8 | 2 | 4 | yes — `cp_size` only has to divide the domain |
| ep_size=4 + cp_size=2 | 4 | 2 | 4 | **NO** — node-local EP+CP requires `ep_group_size == domain`; this is the shape that raises on 8 GPUs, not EP+CP itself |
| ep_size=8 + tp_size=8 | 8 | 8 | 1 | yes (ep_size multiple of tp_size, node-local) |
| ep_size=2 + expert_tp_size=2 | 4 | 2 | 4 | yes (EP+ETP, node-local) |
| tp_size=4 + cp_size=2 | — | — | — | **NO** — TP+CP rejected |
| tp_size=2 + expert_tp_size=2 | — | — | — | **NO** — TP+ETP rejected (`ep_size=1`, so the axis set is `{tp, etp}`) |
| expert_tp_size=2 + cp_size=2 | 2 | 2 | — | **NO** — ETP+CP rejected |

For EP+CP specifically: `_validate_ep_cp` requires `ep_scope="node"` and
`ep_group_size == nvlink_domain_size` — that is the whole gate. On an 8-GPU node it pins
`ep_size=8`, and `cp_size` only has to divide the domain, so `ep8+cp2` (DP 4), `ep8+cp4` (DP 2) and
`ep8+cp8` (DP 1) all run. `cp_size == ep_group_size == domain` is the fully orthogonal *convention*,
not a rule. EP+CP does **not** need 16 GPUs.
