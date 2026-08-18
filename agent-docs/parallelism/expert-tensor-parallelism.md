# Expert Tensor Parallelism (ETP)

`expert_tp_size` shards expert FFN weights only (`gate_proj`, `up_proj`, `down_proj`) across
NVLink-local groups, cutting per-rank expert memory without touching attention. Two shapes:

- **Pure ETP** (`ep_size=1, expert_tp_size>1`): each expert's FFN is sharded `expert_tp_size`-way;
  experts are *replicated*, not distributed.
- **EP+ETP** (`ep_size>1` AND `expert_tp_size>1`, experimental): experts distributed `ep_size`-way
  **and** each expert's FFN sharded `expert_tp_size`-way (`ep_group_size = ep_size · expert_tp_size`).
  The EP group splits into `expert_tp_size` DeepEP dispatch groups coupled by a strided expert-TP
  all-reduce.

Use ETP when expert weights don't fit per rank after EP distribution and attention fits on one GPU.
Per-rank expert memory drops ~`expert_tp_size`×; ranks in the same ETP group share a batch, reducing
DP. `expert_tp_size` is mutually exclusive with attention TP (`tp_size`) and with CP (`cp_size`):
ETP partners hold shards of one expert and must see the **same** tokens, since `ReduceFromExpertTP`
sums their outputs element-wise in token space, while CP hands each rank a different sequence chunk.

| Term | Meaning |
|------|---------|
| `ep_size` | How many distinct expert subsets |
| `expert_tp_size` | How many ranks shard the FFN weights of each expert subset |
| `ep_group_size` | Full EP process group = `ep_size * expert_tp_size` (auto-computed) |

## Forward path

```text
tokens, routing weights → SumGradAcrossGroup (identity fwd, all-reduce bwd)
       → dispatch (sub-EP via DeepEP)
       → expert_compute (TP-sharded weights → partial output)
       → combine (sub-EP via DeepEP)
       → ReduceFromExpertTP (all-reduce SUM fwd, identity bwd) → output
```

The expert-TP all-reduce sits at the layer boundary (token space), **outside** the DeepEP
dispatch→combine span (`EPMoELayerBase._dispatch_compute_combine`). This is exact because the
combine is linear in the expert outputs, so summing per-shard partials in token space equals summing
them in recv space (`combine(Σ) = Σ combine`). It **must** stay outside: an all-reduce between
dispatch and combine couples the dispatch groups inside DeepEP's intranode combine barrier, and under
FSDP2 multi-stream drift they form a circular wait and the barrier times out.

The Megatron-LM scatter-gather pattern, a matched autograd pair — the scatter half is the toolkit-wide
`SumGradAcrossGroup` (`src/distributed/grad_reduce.py`, shared with TP), the gather half
`ReduceFromExpertTP` (`src/distributed/expert_parallel/autograd.py`):

- **`SumGradAcrossGroup`** — identity forward; `all_reduce(SUM)` backward over `expert_tp_group`. It is
  applied to **both** the layer input tokens **and** the routing weights: the gate multiply happens
  on this rank's *partial* expert output, so `d(loss)/d(weights)` here is a partial whose true value
  is the sum over the group. Without the second application the router — and every upstream tensor
  reached through it — trains on a gradient scaled by `1/expert_tp_size`, silently.
- **`ReduceFromExpertTP`** — `all_reduce(SUM)` forward on the combined output to sum the partial
  expert outputs; identity backward, correct because `SumGradAcrossGroup` owns the input-gradient
  reduction and each shard's weight gradient is independently correct.

Gradient divisors (`create_expert_grad_hook` / `create_router_grad_hook`, `src/distributed/expert_parallel/grad_sync.py`):

| Parameter | Divisor | Rationale |
|---|---|---|
| Expert weights | `world_size / expert_tp_size` | Each shard sees that many distinct batches |
| Router weights | `world_size` | Duplicates within expert TP cancel |
| Non-expert params | `world_size` | FSDP2 `fully_shard`, duplicates cancel |

## Quick start

```bash
# Pure ETP on 8 GPUs: experts replicated, expert FFN sharded 8-way (ep_group_size=8).
# CLI flags override any expert_parallel_size set in the YAML.
torchrun --nproc_per_node=8 scripts/training/sft.py \
    examples/sft/gptoss/gptoss-20b-multinode-ep.yaml \
    --expert_parallel_size=1 \
    --expert_tensor_parallel_size=8

# 2-GPU correctness test (expert_tp_size=2, ep_size=1)
torchrun --nproc_per_node=2 \
    tests/gpu/parallelism/combined/test_ep_etp_correctness.py
```

## Validation rules

Enforced in `src/distributed/parallelism_config.py` before model loading:

- `expert_tp_size` must divide the NVLink domain (= `gpus_per_node` on a standard node, the rack on
  NVL72).
- `ep_group_size` must divide the NVLink domain (`ep_scope=node`). Under `ep_scope=global` it must
  tile the stage world (`world_size / pp_size`) as equal contiguous per-domain blocks — divisibility
  alone is not sufficient ([Multi-Node](multi-node.md#node-local-vs-cross-node-ep)).
- `tp_size > 1` with `expert_tp_size > 1` is rejected. Use EP+TP for attention memory or EP+ETP for
  expert memory, never both.
- Cross-node EP (`ep_scope=global`) + ETP is supported as **one ETP group per NVLink domain** only:
  `expert_tp_size == EP members per domain` and `ep_size == domains spanned`. A finer split that
  would straddle a domain boundary is rejected — the ETP all-reduce must stay on NVLink.
- Multi-domain multi-group EP+ETP is rejected — the in-backward cross-replica expert grad sync races
  the DeepEP combine across domains. This rule, not the one above, is what refuses `ep2+etp4` and
  `ep4+etp2` on 2×8 (both leave `ep_group_size=8` under a 16-rank world), at either scope;
  `ep2+etp8` is the working shape.
- The racy-EP gate (`is_racy_single_domain_multigroup_ep`) fires on a **single** NVLink domain with
  `ep_size > 2` **and** `ep_group_size < nvlink_domain_size`, so an ETP shape that leaves
  `ep_group_size` short of the domain is rejected like bare `ep4` — `ep4+etp2` on a 16-GPU NVL
  domain, for instance.

!!! note "`ep4+etp2` on 8 GPUs is validated"
    It is the only 4-way expert split on a single 8-GPU node, and it clears the racy-EP gate that
    refuses bare `ep4` because ETP raises `ep_group_size` to the full domain —
    [Expert Parallelism](expert-parallelism.md#single-domain-multi-group-ep-races-and-hangs). For a
    4-way *expert-FFN* split without that topology, `ep2+etp4` reaches the same 8-way total split
    through 2-rank dispatch groups, at DP 2 instead of DP 4; it is what the 8-GPU Mistral4 matrix
    (`tests/gpu/manifest.py`) runs.

ETP reduces data parallelism — it counts toward `max(tp_size, cp_size, expert_tp_size)` in the
`dp_size` formula owned by [Distributed Data Loading](data-loading.md#data-parallel-size). On 8 GPUs
(`ep_group_size = 8` throughout): EP-only `ep8` → DP 8; `ep4+etp2` → DP 4; `ep2+etp4` → DP 2; pure
ETP `etp8` → DP 1.

## Process groups (EP+ETP combo)

`EPConfig._create_expert_tp_groups` builds one of two layouts by EP scope, both keeping the ETP
all-reduce on NVLink.

**Node-local EP (`ep_scope=node`)** — the whole EP group is in one domain, so the DeepEP dispatch
takes the contiguous part: `dispatch_ep_rank = ep_rank % ep_size`, `expert_tp_rank = ep_rank //
ep_size`. Dispatch groups are contiguous `ep_size`-rank chunks (DeepEP requires contiguous P2P
buffers); ETP groups are strided, one rank per chunk. On 8 GPUs with `ep_size=4, expert_tp_size=2`:

```text
EP group:           [0,1,2,3,4,5,6,7]
Sub-EP (dispatch):  [0,1,2,3]  [4,5,6,7]          ← DeepEP, size = ep_size = 4
ETP groups:         [0,4] [1,5] [2,6] [3,7]       ← NCCL all-reduce of FFN shards
```

**Cross-node EP (`ep_scope=global`)** — the layout mirrors: each domain's contiguous block **is** one
ETP group, and the dispatch group strides across domains over the internode Gin path.
`expert_tp_rank = ep_rank % expert_tp_size`, `dispatch_ep_rank = ep_rank // expert_tp_size`. On 16
GPUs (2×8) with `ep_size=2, expert_tp_size=8`:

```text
EP group:           [0..15]   (domain0 [0-7], domain1 [8-15])
ETP groups:         [0-7]  [8-15]                 ← node-local NCCL all-reduce
Sub-EP (dispatch):  [0,8] [1,9] ... [7,15]        ← cross-node DeepEP, size = ep_size = 2
```

This is the memory-lean cross-node MoE shape: per-rank expert memory matches a much larger pure-EP
group. On a 2-node 8×B300 cluster (EFA), gpt-oss-20b at `ep_size=2, expert_tp_size=8` holds ~18 GB
per rank — the footprint of EP=16 — while the 2-rank cross-node dispatch runs it at roughly 2× the
EP=16 throughput. Validate convergence on your own config before a long run.

## Weight sharding

ETP shards the intermediate dimension `M` of expert FFN weights; the recipe is keyed on the model's
expert layout (`src/distributed/expert_parallel/base_layer.py` + `layers/`).

- **GptOss (fused interleaved GLU)** — `gate_up_proj [E, H, 2M]` is `[g0,u0,g1,u1,…]`. De-interleave
  (`gate = [...,::2]`, `up = [...,1::2]`), shard each along dim 2 to `[E, H, M/tp]`; shard
  `down_proj [E, M, H]` dim 1.
- **Fused-GLU contiguous halves (every family whose block stores a fused `gate_up_proj`, through the
  base `_init_fused_glu_params`)** — `gate_up_proj [E, H, 2M]` is `[gate(M) | up(M)]`. Slicing the fused tensor along
  dim 2 would mismatch the gate/up pairing, so ETP splits the halves first and stores separate
  `gate_proj`/`up_proj`/`down_proj` params.
- **Qwen3 / Bailing (separate GLU projections)** — the base `_store_separate_glu_params` shards `M`
  and stores matmul-convention `gate_proj`/`up_proj [E, H, M/tp]`, `down_proj [E, M/tp, H]` for
  [Grouped GEMM](../optimization/grouped-gemm.md).

## Checkpoints

Save formats, resume paths, and merge scripts: [Checkpoints](../reference/checkpoints.md). The
ETP-unique step is that a gathered save all-gathers the TP shards within `expert_tp_group` and
re-assembles full expert weights, keyed on layout: GptOss re-interleaves; contiguous-halves families
concatenate along dim 2 and transpose back from matmul convention (the base
`gather_expert_state_dict`); GLM4/LFM2 declare `_PER_EXPERT_UNFUSED_KEYS` and the base splits to
per-expert names; Qwen3/Bailing write per-expert `experts.{i}.{gate,up,down}_proj.weight`. The
result is a standard HuggingFace checkpoint; ETP sharding is re-applied automatically at model
patching when `expert_tp_size > 1`. Per-rank sharded EP save is rejected — see below.

## Limitations

**Trainers.** There is no `_supports_etp`: ETP folds into `ep_group_size = ep_size ×
expert_tp_size`, so it is gated by `_supports_ep`, which every trainer declares `True`. Matrix:
[Trainer Compatibility](../reference/trainer-architecture.md#trainer-compatibility).

**Models.** Every EP-capable MoE family
([roster](expert-parallelism.md#supported-models)). ETP shards expert FFN
weights only — the router stays replicated in every family, whether it sits outside the EP wrapper
and is FSDP-managed (Gemma 4) or inside it and is synced by the EP router hook (everyone else,
including Zaya). A dense model raises: pure ETP has `ep_group_size > 1`, so patching zero MoE layers
is an error. `expert_tp_size` must divide each expert's intermediate size, else the top units of every expert
would be dropped silently. `ParallelismConfig.validate_against_model_config` checks it off
`config.json` at the top of the model load, through `resolve_expert_ffn_shard_width` — the one home
for "what does ETP actually shard", so the config-time gate and the layer that does the split cannot
disagree on which config spelling carries the per-expert width.

**Axis combinations.** ETP composes with EP. Pure ETP and EP+ETP are supported shapes; TP+ETP,
ETP+CP and EP+TP+ETP are refused by the [allowlist](index.md#supported-combinations), and PP shapes
are [not yet available in this release](pipeline-parallelism.md).

**Knobs.** ETP inherits every [EP knob rule](expert-parallelism.md#limitations) (`is_ep_mode` is
`ep_group_size > 1`, which pure ETP satisfies). On top of those:

| Knob | Under ETP | Gate |
|---|---|---|
| expert LoRA | rejected at config time — the replicated adapter half would receive partial gradients and drift across ranks. `EPConfig` repeats the identical raise at group construction, and the vLLM sync's `merge_lora=True` gather guards it once more, unreachably | `ParallelismConfig` |
| `save_sharded_ep` | rejected — the merge script cannot reconstruct TP-sharded expert weights | `validate_ep_sharded_save` |
| `use_hsdp` | rejected — ETP builds its own `(dp, tp)` mesh | `_validate_hsdp` |
| `use_peft` / attention LoRA | supported — pure ETP leaves attention unsharded, so the adapter is a genuine replica | — |
| `use_grouped_gemm` on GptOss | silently falls back to the per-expert loop — once TP-sharded, the interleaved `gate_up_proj` cannot be de-interleaved | `EPGptOssMoELayer._grouped_mm_enabled` |

The same-token invariant ETP depends on — partners must hold identical batches, since
`ReduceFromExpertTP` sums element-wise in token space — is maintained by the rank layout, the
DP-rank-sharded dataloader, and the ETP-leader rollout broadcast. **It is never asserted at
runtime**: a custom data path that bypasses those would sum unrelated tokens with no error.

## Troubleshooting

- **"Node-local EP group size (N) cannot exceed the NVLink domain (M)"** — `ep_size * expert_tp_size`
  must fit one domain under `ep_scope=node`. Reduce either, or use `ep_scope=global`.
- **"Cross-node EP+ETP supports one ETP group per NVLink domain only"** — set `expert_tp_size` to the
  EP members per domain and `ep_size` to the domain count (both quoted in the message), or use
  `ep_scope=node`.
- **"Sharded EP save (save_sharded_ep=True) is not supported with expert_tp_size=N"** — use the
  default gathered save.
- **OOM** — raise `expert_tp_size`, or switch to EP+TP if attention is also a bottleneck (the two
  are mutually exclusive).
