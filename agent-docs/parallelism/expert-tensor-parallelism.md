# Expert Tensor Parallelism (ETP)

`expert_tp_size` shards expert FFN weights only (`gate_proj`, `up_proj`, `down_proj`) across
NVLink-local groups, cutting per-rank expert memory without touching attention. Two shapes:

- **Pure ETP** (`ep_size=1, expert_tp_size>1`): each expert's FFN is sharded `expert_tp_size`-way;
  experts are *replicated*, not distributed.
- **EP+ETP** (`ep_size>1` AND `expert_tp_size>1`, experimental): experts distributed `ep_size`-way
  **and** each expert's FFN sharded `expert_tp_size`-way (`ep_group_size = ep_size · expert_tp_size`).
  The EP group splits into `expert_tp_size` DeepEP dispatch groups coupled by a strided expert-TP
  all-reduce.

Pure ETP also runs inside each pipeline stage ([PP + ETP](pipeline-parallelism.md#pp--etp-moe)); EP+ETP
does not compose with PP.

Use ETP when expert weights don't fit per rank after EP distribution and attention fits on one GPU.
Per-rank expert memory drops ~`expert_tp_size`×; ranks in the same ETP group share a batch, reducing
DP.

`expert_tp_size` excludes attention TP and CP
([why](README.md#supported-combinations)).

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
them in recv space (`combine(Σ) = Σ combine`).

It **must** stay outside: an all-reduce between dispatch and combine couples the dispatch groups
inside DeepEP's intranode combine barrier, and under FSDP2 multi-stream drift they form a circular
wait and the barrier times out.

The Megatron-LM scatter-gather pattern, a matched autograd pair — the scatter half is the toolkit-wide
`SumGradAcrossGroup` (`halo/distributed/grad_reduce.py`, shared with TP), the gather half
`ReduceFromExpertTP` (`halo/distributed/expert_parallel/moe_layer/layer.py`):

- **`SumGradAcrossGroup`** — identity forward; `all_reduce(SUM)` backward over `expert_tp_group`. It is
  applied to **both** the layer input tokens **and** the routing weights.

    The gate multiply happens on this rank's *partial* expert output, so `d(loss)/d(weights)` here
    is a partial whose true value is the sum over the group. Without the second application the
    router, and every upstream tensor reached through it, trains on a gradient scaled by
    `1/expert_tp_size`, silently.

- **`ReduceFromExpertTP`** — `all_reduce(SUM)` forward on the combined output to sum the partial
  expert outputs; identity backward, correct because `SumGradAcrossGroup` owns the input-gradient
  reduction and each shard's weight gradient is independently correct.

Gradient divisors (the trainer's expert sweep, `_sync_deferred_expert_grads` in `halo/trainers/parallelism/grad_sync.py`, and `build_router_grad_hook` in `halo/distributed/expert_parallel/moe_layer/grad_sync.py`):

| Parameter | Divisor | Rationale |
|---|---|---|
| Expert weights | `world_size / expert_tp_size` | Each shard sees that many distinct batches |
| Router weights | `world_size` | Duplicates within expert TP cancel |
| Non-expert params | `world_size` | FSDP2 `fully_shard`, duplicates cancel |

Under PP every `world_size` here is the stage's rank count.

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
    tests/gpu/parallelism/ep/test_ep_modes_vs_reference.py --model gpt_oss_20b --mode etp
```

## Validation rules

Enforced in `halo/distributed/parallelism_rules.py` before model loading:

- `expert_tp_size` must divide the NVLink domain (= `gpus_per_node` on a standard node, the rack on
  NVL72).
- `ep_group_size` must divide the NVLink domain (`ep_scope=node`). Under `ep_scope=global` it must
  tile the world as equal contiguous per-domain blocks — divisibility
  alone is not sufficient ([Multi-Node](multi-node.md#node-local-vs-cross-node-ep)).
- Cross-node EP (`ep_scope=global`) + ETP is supported as **one ETP group per NVLink domain** only:
  `expert_tp_size == EP members per domain` and `ep_size == domains spanned`. A finer split that
  would straddle a domain boundary is rejected — the ETP all-reduce must stay on NVLink.
- Multi-domain multi-group EP+ETP is rejected — expert-TP keeps the deferred cross-replica DP path
  (`is_deferred_dp`) off, so FSDP2's DP-wide reduce-scatter would race the narrower DeepEP combine
  across domains.

    This rule, not the one above, is what refuses `ep2+etp4` and `ep4+etp2` on 2×8 (both leave
    `ep_group_size=8` under a 16-rank world), at either scope; `ep2+etp8` is the working shape.

- On a single NVLink domain the EP rule keys on `ep_group_size`: `ep4+etp2` on 8 GPUs fills the
  domain and passes, while an ETP shape leaving `ep_group_size` short of the domain is refused like
  bare `ep4` ([single-domain rule](expert-parallelism.md#single-domain-multi-group-ep-races-and-hangs)).
  The GPU test matrix runs `ep2+etp4` instead (the 8-GPU Mistral4 and Cohere2 MoE matrices,
  `tests/gpu/manifest.py`): the same 8-way split through 2-rank dispatch groups, at DP 2 instead of 4.

ETP reduces data parallelism — within a stage it counts toward the max of TP, CP and ETP
([Stacking the dimensions](README.md#stacking-the-dimensions)). On 8 GPUs
(`ep_group_size = 8` throughout): EP-only `ep8` → DP 8; `ep4+etp2` → DP 4; `ep2+etp4` → DP 2; pure
ETP `etp8` → DP 1.

## Process groups (EP+ETP combo)

`EPConfig._build_expert_tp_groups` builds one of two layouts by EP scope, both keeping the ETP
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
group. On two 8-GPU B300 nodes over RDMA, gpt-oss-20b at `ep_size=2, expert_tp_size=8` holds ~18 GB
per rank — the footprint of EP=16 — while the 2-rank cross-node dispatch runs it at roughly 2× the
EP=16 throughput. Validate convergence on your own config before a long run.

## Weight sharding

ETP shards the intermediate dimension `M` of expert FFN weights; the recipe is keyed on the model's
expert layout (`halo/distributed/expert_parallel/moe_layer/compute.py` + `families/`).

- **GptOss (fused interleaved GLU)** — `gate_up_proj [E, H, 2M]` is `[g0,u0,g1,u1,…]`. De-interleave
  (`gate = [...,::2]`, `up = [...,1::2]`), shard each along dim 2 to `[E, H, M/tp]`; shard
  `down_proj [E, M, H]` dim 1.
- **Fused-GLU contiguous halves (every family whose block stores a fused `gate_up_proj`, through the
  base `_init_fused_glu_params`)** — `gate_up_proj [E, H, 2M]` is `[gate(M) | up(M)]`.

    Slicing the fused tensor along dim 2 would mismatch the gate/up pairing, so ETP splits the halves
    first and stores separate `gate_proj`/`up_proj`/`down_proj` params.

- **Qwen3 / Bailing (separate GLU projections)** — the base `_store_separate_glu_params` shards `M`
  and stores matmul-convention `gate_proj`/`up_proj [E, H, M/tp]`, `down_proj [E, M/tp, H]` for
  [Grouped GEMM](../optimization/grouped-gemm.md).

## Checkpoints

Save formats, resume paths, and merge scripts: [Checkpoints](../reference/checkpoints.md). The
ETP-unique step is that a gathered save all-gathers each expert attribute's TP shards within
`expert_tp_group` (`EPGatherMixin.EXPERT_TP_SHARD_DIMS`) before the dispatch-EP gather, giving the full
runtime tensors; the family's `ExpertLayout` then writes them as it would without ETP:

- GptOss re-interleaves.
- Contiguous-halves families concatenate gate and up along dim 2 and transpose back from matmul
  convention.
- GLM4/Laguna/LFM2 (`FUSED_SPLIT_PER_EXPERT`) split that to per-expert names.
- Qwen3/Bailing (`SEPARATE_GLU`) write per-expert `experts.{i}.{gate,up,down}_proj.weight`.

The result is a standard HuggingFace checkpoint; ETP sharding is re-applied automatically at model
patching when `expert_tp_size > 1`. Per-rank sharded EP save is rejected — see below.

## Limitations

**Trainers.** No trainer refuses ETP or EP. Matrix:
[Trainer Compatibility](../reference/trainer-architecture.md#trainer-compatibility).

**Models.** Every EP-capable MoE family
([roster](expert-parallelism.md#supported-models)). ETP shards expert FFN
weights only: the router stays replicated in every family, whether it sits outside the EP layer
and is FSDP-managed (Gemma 4) or inside it and is synced by the EP router hook (everyone else,
including Zaya). A dense model raises: pure ETP has `ep_group_size > 1`, so patching zero MoE layers
is an error.

`expert_tp_size` must divide each expert's intermediate size, else the top units of every expert
would be dropped silently. `reject_unsupported_model` checks it off
`config.json` at the top of the model load, through `resolve_expert_ffn_width`.

That helper is the one home for "what does ETP actually shard", so the config-time gate and the
layer that does the split cannot disagree on which config spelling carries the per-expert width.

**Axis combinations.** Pure ETP, EP+ETP, and pure ETP inside PP stages
([allowlist](README.md#supported-combinations)).

**Knobs.** ETP inherits every [EP knob rule](expert-parallelism.md#limitations) (`has_ep_group` is
`ep_group_size > 1`, which pure ETP satisfies). On top of those:

| Knob | Under ETP | Gate |
|---|---|---|
| expert LoRA | rejected at config time — the replicated adapter half would receive partial gradients and drift across ranks. Every `merge_lora=True` gather (weight sync, merge-on-save) guards it once more, unreachably | `ParallelismConfig` |
| `save_sharded_ep` | rejected — the merge script cannot reconstruct TP-sharded expert weights | `reject_unmergeable_ep_sharded_save` |
| `use_hsdp` | rejected — ETP builds its own `(dp, tp)` mesh | `_reject_hsdp` |
| `use_peft` / attention LoRA | supported — pure ETP leaves attention unsharded, so the adapter is a genuine replica | — |
| online GRPO `rollout_func` | rejected, as under TP — it replaces the generation that hands every ETP partner its leader's completions, so partners would feed different tokens into the expert-TP all-reduce | `reject_unshared_rollout_func` |
| `use_grouped_gemm` on GptOss | runs the per-expert loop, reported in the layer's init summary (`grouped_mm=False`) — ETP stores the de-interleaved gate/up pair under the plain names the loop reads, not the `*_gmm` pair the grouped path reads | `EPGptOssMoELayer._grouped_mm_enabled` |

The same-token invariant ETP depends on — partners must hold identical batches, since
`ReduceFromExpertTP` sums element-wise in token space — is maintained by the rank layout, the
DP-rank-sharded dataloader, and the ETP-leader rollout broadcast. **It is never asserted at
runtime**: a custom data path that bypasses those would sum unrelated tokens with no error.

## Troubleshooting

- **"Node-local EP group size (N) cannot exceed the NVLink domain (M)"** — `ep_size * expert_tp_size`
  must fit one domain under `ep_scope=node`. Reduce either, or use `ep_scope=global`.
- **"Cross-node EP+ETP supports one ETP group per NVLink domain only"** — set `expert_tensor_parallel_size`
  to the EP members per domain and `expert_parallel_size` to the domain count (both quoted in the
  message), or use `ep_scope=node`.
- **"Sharded EP save (save_sharded_ep=True) is not supported with expert_tensor_parallel_size=N"** — use the
  default gathered save.
- **OOM** — raise `expert_tp_size`, or switch to EP+TP if attention is also a bottleneck (the two
  are mutually exclusive).
