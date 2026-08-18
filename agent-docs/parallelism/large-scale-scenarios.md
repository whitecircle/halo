# Large-Scale Scenarios — Qwen3.5-397B-A17B

Per-rank memory arithmetic, reachable parallelism layouts and launch recipes for
**`Qwen/Qwen3.5-397B-A17B`** on **2, 3, 4 and 8 nodes of 8 × B300 (288 GB)**. Every constraint here
is enforced in code and named by its symbol; the memory columns are arithmetic from the shape below,
so treat them as a plan, not a measurement. Mechanism pages:
[Multi-Node](multi-node.md), [Expert Parallelism](expert-parallelism.md),
[Pipeline Parallelism](pipeline-parallelism.md), [Launch Recipes](launch-recipes.md),
[Qwen3.5](../models/qwen3_5.md),
[GPU Training Theory](../reference/gpu-training-theory.md#why-n-gpus-is-not-n-throughput).

Two facts set every cell and pull against each other. `nvlink_domain_size` decides which axis may
cross the slow fabric. And the routed experts are 386.5 B of the 396.8 B total, only ever sharded by
`X = ep_size × expert_tp_size`, needing `X ≥ 16` at 8 B/param to fit at all. On an 8-GPU domain
`X ≥ 16` forces the EP group across the domain boundary. [Pipeline parallelism](pipeline-parallelism.md)
— the planned axis that would keep EP on NVLink by shortening each rank's layer range — is **not yet
available in this release**; its rows below are forward-looking arithmetic.

## First question: how wide is the NVLink domain?

`nvlink_domain_size` (auto = `gpus_per_node`; set `NVLINK_DOMAIN_SIZE` on NVL72) is the locality unit
every validator divides by, and it decides which axis is allowed onto RDMA. The dispatcher keys the
decision off topology alone — `is_inter_node = num_nodes > 1 and not node_local`
(`src/distributed/expert_parallel/dispatcher.py`), which sets `EP_DISABLE_GIN=0` — Gin on. `num_nodes`
counts NVLink **domains**, not OS nodes, so a group inside one NVL72 rack stays on NVLink even
though it spans ~18 hosts.

What each axis puts on that fabric per microbatch, at this model's 60 MoE layers — arithmetic from
the layer count and the autograd graph, not a benchmark:

| Axis | Crossing traffic per microbatch | Count |
|---|---|---:|
| EP, `ep_scope=global` across domains | all-to-all over the EP group per MoE layer: forward dispatch + combine, and their backward mirrors | **240** |
| PP (planned — [not yet available](pipeline-parallelism.md)) | P2P boundary activation, plus its gradient on the way back, per stage edge | **2 × (pp_size − 1)** |

Four per layer, not two, because `DeepEPDispatchFunction.backward` *is* a combine and
`DeepEPCombineFunction.backward` *is* a dispatch (`src/distributed/expert_parallel/autograd.py`).
Gradient checkpointing does not multiply the count — recompute replays cached results rather than
re-issuing the collective.

The gap is worse than the counts suggest: cross-domain dispatch is **latency**-bound, so its fixed
per-collective cost does not amortize as the message grows
([DeepEP → AWS EFA](../infrastructure/deepep.md#expert-parallelism-over-aws-efa)).

**The fork:**

- **Domain = 8** (a standard 8 × B300 node). Where a model fits at `ep_group_size <= 8`, node-local
  EP groups replicate across domains as DP replicas, their cross-replica average deferred to a
  post-backward sweep ([Multi-Node](multi-node.md#deferred-cross-replica-sync)) — no per-layer RDMA.
  At 397B that does not fit (386 GB of experts per rank), so the EP group must cross the domain
  boundary at global scope; when the fabric is latency-bound, narrow the dispatch group with EP+ETP
  (see [Recommended cell](#recommended-cell)). PP, the planned escape that would keep EP node-local,
  is [not yet available in this release](pipeline-parallelism.md).
- **Domain = 72** (GB200/GB300 NVL72). A node-scope EP group rides NVLink at any width that *divides*
  the declared domain, `requires_rdma` stays false, and wide EP becomes the attractive shape. On
  this model that means declaring a **64**-wide domain, not 72 — see
  [Recommended cell](#recommended-cell). Rack-wide domains are **scaffolded and unvalidated** in
  this toolkit — no rack-scale run backs them
  ([Scale & Limits](../reference/scale-and-limitations.md)).

## Model shape

Source: `hf_hub_download("Qwen/Qwen3.5-397B-A17B", "config.json")`, revision
`8472618112abcbd45acbcdc58436aff4233c23f7`. The checkpoint is a VLM —
`architectures: ["Qwen3_5MoeForConditionalGeneration"]`, `model_type: qwen3_5_moe` — and every field
below is from `text_config` (`qwen3_5_moe_text`).

| Field | Value | Field | Value |
|---|---|---|---|
| `num_hidden_layers` | 60 (all MoE, `mlp_only_layers: []`) | `num_attention_heads` | 32 |
| `hidden_size` | 4096 | `num_key_value_heads` | 2 |
| `num_experts` | **512** | `head_dim` | 256 |
| `num_experts_per_tok` | 10 | `attn_output_gate` | `true` (q_proj emits 2 × `NH·head_dim`) |
| `moe_intermediate_size` | 1024 | `vocab_size` | 248320 |
| `shared_expert_intermediate_size` | 1024 (one shared expert per layer) | `tie_word_embeddings` | **false** (top level) |
| `intermediate_size` | absent — no dense FFN | `max_position_embeddings` | 262144 |

`layer_types` is `["linear_attention"]*3 + ["full_attention"]` repeated 15× —
**45 gated-DeltaNet layers + 15 full-attention layers**, `full_attention_interval: 4`. The linear
layers carry `linear_num_value_heads: 64`, `linear_num_key_heads: 16`, head dims 128,
`linear_conv_kernel_dim: 4`.

Parameter counts derived from those fields (fused expert layout
`gate_up_proj [512, 2048, 4096]` + `down_proj [512, 1024, 4096]`):

| Group | Params | Sharded by |
|---|---:|---|
| Routed experts `P_e` | **386.55 B** | `X = ep_size × expert_tp_size` |
| Gated-DeltaNet (45) | 5.31 B | FSDP2 only |
| Full attention (15) | 1.57 B | FSDP2, or TP |
| `embed_tokens` + `lm_head` (untied) | 2.03 B | FSDP2 only |
| Vision tower | 0.45 B | FSDP2 only |
| **Shared experts + routers `P_r`** | **0.88 B** | **nothing** — see below |
| Non-expert `P_ne` (all four rows above + norms) | **10.25 B** | |
| **Total** | **396.80 B** | |
| Active per token (top-10 + shared + attention + head) | **17.35 B** | |

**`P_r` is the trap.** `EPQwen3_5MoELayer` owns the router `gate`, the `shared_expert` and
`shared_expert_gate`; the whole layer lands in FSDP2's ignored set (`_ep_fsdp_ignored_modules`), TP
never visits it (`apply_tp_to_attention_only`), and EP does not shard it — `replicated_named_params`
grad-syncs it by hand. So **0.88 B stays as a full 8 B/param copy on every rank ≈ 7 GB that no axis
shrinks.**

## Per-rank memory model

`P_f = P_ne − P_r = 9.37 B` is the FSDP-managed non-expert share, `F` the FSDP shard width for it.
`F` is **not** `data_parallel_size`. It defaults to every rank of the job (`F = stage_world_size`), so
`ep4 + etp8` on 4 nodes still shards non-expert params 32-way even at `DP = 4` — ETP partners hold
identical replicas of them. Two topologies narrow it: under TP the `(dp, tp)` mesh drops `F` to `DP`,
and under `EPConfig.is_deferred_dp` — multi-group EP across domains, the `× N replicas` rows below —
FSDP shards over the **EP group**, so `F = ep_group_size` and the cross-replica average moves to the
post-backward sweep ([Multi-Node](multi-node.md#deferred-cross-replica-sync)).

| Term | Bytes per rank | 397B value |
|---|---|---|
| Routed experts | `8·P_e/X` | 193 / 97 / 48 GB at `X` = 16 / 32 / 64 |
| Replicated router + shared expert | `8·P_r` | **7 GB, always** |
| FSDP non-expert, ZeRO-2 (default) | `2·P_f + 8·P_f/F` | 19 GB resident + 5 / 2 / 1 GB at `F` = 16 / 32 / 64 |
| Activations, GC on | `(2·L·H + 6·k·I_moe)·b·S` | **4.5 GB** at `b1 s8192`, **18.1 GB** at `b1 s32768` |
| Load-time peak (separate) | `2·(P_ne + P_e/ep_size)` | see the `Load` column |

`8 B/param` = 2 weights + 2 grads + 4 optimizer state; `AdamWBF16` (`src/optimizers/adamw_bf16.py`)
allocates `exp_avg` and `exp_avg_sq` with `zeros_like(p)`, so both are bf16 and there is no master
copy and no error-feedback buffer. `fp32_grad_reduce` upcasts only inside the reduce — no storage.
Stock AdamW would cost 12 B/param and no cell below survives.

The activation term is one bf16 layer input per layer (`2·L·H·b·S`) plus the recomputed layer's
expert intermediates (`6·k·I_moe·b·S` — gate_up, SiLU product, down input at top-`k` = 10). The
45 linear layers additionally hold an fp32 recurrent state of
`64 × 128 × 128 × 4 B ≈ 4 MB` per sequence per layer (≈190 MB at `b=1`).

**ZeRO-3 would remove the 19 GB resident term but is rejected** under EP and under TP+DP
(`_validate_fsdp_settings`) — every layout on this page is EP, so
`fsdp_reshard_after_forward: false` is mandatory, not a choice.

**Budget.** 288 GB HBM per B300; the CUDA context, NCCL buffers and allocator fragmentation take the
practical ceiling to **≈ 260 GB**. Cells are ⚠ between 230 and 260 GB, ✗ above.

**The DeepEP arena is a separate term**, paid once for the model rather than once per layer. It is
receive-side and linear in the dispatch-group width: at this model's `hidden=4096`, `top_k=10` an
`ep64` group at 8192 tokens per rank costs **4.06 GiB** on top of the columns below. Formula and
growth law: [DeepEP → Buffer sizing](../infrastructure/deepep.md#buffer-sizing). It does not shape
the choice of `ep_size` at any cell here — a wide dispatch group keeps its expert-memory saving
instead of spending it back — with one exception: `HALO_EP_CAPACITY_DEDUP=0` gives every layer a
private arena, multiplying that figure by the 60 MoE layers to 243.8 GiB, past the card. When the
arena does not fit, the failure is a CUDA OOM inside `ElasticBuffer` at the **first MoE forward**,
after the whole model has loaded, with nothing in the message naming EP capacity.

## What loads at all

Only three loaders shard at load time: `WeightAction.EXPERT_SHARD`
(`src/models/loading/lazy_safetensors/`, planned by `EPWeightPlanner` in
`src/distributed/expert_parallel/lazy_loader.py`), HF `tp_plan="auto"` for **dense** models, and the
PP stage loader. Every other branch of `_dispatch_model_loading` materializes all 396.8 B on one
GPU — **794 GB, never starts**. That kills plain FSDP2, pure TP (MoE TP has no shard-aware load),
pure ETP (`ep_size == 1` sets `experts_per_rank = num_experts`, so every rank reads every expert)
and pure CP before any memory argument.

The EP lazy loader slices by **`ep_size`, not `X`** — ETP partners share `dispatch_ep_rank` and load
the same experts at full intermediate width. The intermediate dim is split later, at EP patching
(`_init_fused_glu_params` / `_store_separate_glu_params`,
`src/distributed/expert_parallel/base_layer.py`), which also holds one layer's transposed full-width
copy alongside the new shard. So `ep2 + etp8` has the same steady state as `ep16` and a **407 GB**
load peak.

**The planned PP cells would ride the stage loader.** `load_pp_stage_model`
(`src/distributed/pipeline_parallel/lazy_loader.py`) — a shipped seam — streams a stage's own
decoder layers and composes with the EP expert slice (`PPWeightPlanner.filter(EPWeightPlanner(...))`),
giving `2·(P_ne/pp + P_e/(pp·ep_size))` plus the embedding and head. The schedule engine that would
use it does not ship ([Pipeline Parallelism](pipeline-parallelism.md)).

## Scenario matrix

`X = ep_size × expert_tp_size`. `Steady` = experts + replicated + FSDP non-expert + activations at
`b1`. `Load` = the separate per-rank peak while weights stream in. All EP cells need
`512 % ep_size == 0`, rejected in the first second of the job by
`ParallelismConfig.validate_against_model_config` — it reads `config.json` before the process groups
and the meta shell exist.

Read the `Layout` column for the fabric too: on an 8-GPU domain every **`global`** row above one node
puts its dispatch and combine on RDMA, 240 times per microbatch; every **`node`** row keeps them on
NVLink. The `pp*` rows are planned PP shapes kept as forward-looking arithmetic —
[PP is not yet available in this release](pipeline-parallelism.md). Each is sized by stage 0 under
the head-weighted default split, with an un-checkpointed activation estimate.

| Nodes | Layout | X | DP | Experts | Non-exp | Steady 8k | Steady 32k | Load | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **1** (8) | `ep8` node | 8 | 8 | 386 | 35 | **>420** | — | 117 | ✗ experts alone are 1.5× HBM |
| 1 | `ep4`+`etp2` · `ep2`+`etp4` | 8 | 4 · 2 | 386 | 35 | **>420** | — | 214 · 407 | ✗ same expert share — `X` cannot exceed the 8 GPUs |
| **2** (16) | **`ep16` global** | 16 | 16 | 193 | 31 | **228** | 242 | 69 | ⚠ fits, zero headroom |
| 2 | `ep16` global + `tp8` | 16 | 2 | 193 | 55 | **253** | 266 | 69 | ✗ TP collapses `F` to 2 |
| 2 | `ep2` + `etp8` global | 16 | 2 | 193 | 31 | 228 | 242 | **407** | ✗ load peak |
| 2 | `ep8` node · `ep4` node · `ep2` node | ≤8 | 16 | 386 | 31 | **>420** | — | 117–407 | ✗ experts alone exceed HBM |
| 2 | `pp2` + `ep8` node | 8 | 8 | 206 | 21 | **≥306** | — | 66 | ✗ planned PP shape — not yet available |
| **3** (24) | `pp3` + `ep8` node | 8 | 8 | 155 | 14 | **≥243** | — | 50 | ✗ planned PP shape — would be the only fit at 24 GPUs ([The 3-node case](#the-3-node-case)) |
| 3 | `ep3/6/12/24` global | 3–24 | 24 | — | — | — | — | — | ✗ `512 % ep_size ≠ 0` at load |
| 3 | `ep2/4/8` node | ≤8 | 24 | 386 | 31 | **>420** | — | 117–407 | ✗ experts alone exceed HBM |
| 3 | `ep3` + `etp8` global | 24 | 3 | 129 | 24 | 158 | 171 | **278** | ✗ `512 % 3 ≠ 0`, and the `ep_size`-governed load peak would not fit either |
| 3 | `ep2/4/8/16` global | — | — | — | — | — | — | — | ✗ rejected at config (`_validate_ep_group`) |
| **4** (32) | **`ep32` global** | 32 | 32 | 97 | 28 | **129** | 143 | 45 | ✓ |
| 4 | `ep4` + `etp8` global | 32 | 4 | 97 | 28 | 129 | 143 | **214** | ⚠ load peak, experimental axis |
| 4 | `ep32` global + `tp8` | 32 | 4 | 97 | 39 | 140 | 154 | 45 | ✓ but strictly worse than `ep32` |
| 4 | `ep16` global × 2 replicas | 16 | 32 | 193 | 31 | 228 | 242 | 69 | ⚠ half the sharding; both groups still span all 4 domains (4 contiguous ranks per domain — column-block layout), so dispatch stays on RDMA. The win is the narrower 16-rank dispatch group, not domain locality |
| 4 | `pp4` + `ep8` node | 8 | 8 | 103 | 12 | ≥194 | — | 35 | ✗ planned PP shape — not yet available |
| **8** (64) | **`ep64` global** | 64 | 64 | 48 | 27 | **80** | 93 | 33 | ✓ memory to spare; tokens/rank still capped at 8192 |
| 8 | `ep8` + `etp8` global | 64 | 8 | 48 | 27 | 80 | 93 | 117 | ✓ EFA-latency pick, experimental axis |
| 8 | `ep32` global × 2 replicas | 32 | 64 | 97 | 28 | 129 | 143 | 45 | ✓ narrower EP group |
| 8 | `ep16` global × 4 replicas | 16 | 64 | 193 | 31 | 228 | 242 | 69 | ⚠ |
| 8 | `pp8` + `ep8` node | 8 | 8 | 52 | 8 | ≥138 | — | 20 | ✗ planned PP shape — not yet available |
| any | plain FSDP2 · pure TP · pure ETP · pure CP | — | — | — | — | — | — | **794** | ✗ no shard-aware load |
| any | any EP + CP | — | — | — | — | — | — | — | ✗ `validate_model_for_ulysses` |

**One node cannot train this model, in any shape.** Routed experts are only ever divided by `X`,
which on 8 GPUs is at most 8, so every legal single-node shape puts the same 386 GB of expert state
on each GPU against a ~260 GB practical ceiling. PP, the planned axis that would shorten a rank's
layer range, is [not yet available](pipeline-parallelism.md) — and would need a second NVLink domain
anyway. Two nodes with `ep16 --ep_scope=global` is the floor; four is
the first comfortable size. Freezing the experts is the single-node escape — LoRA on EP experts, or
`unfreeze_layers_patterns` — which drops them from 8 to 2 bytes/param; that is arithmetic, not a
validated cell.

**`Steady 32k` is not reachable on the `global` rows.** Cross-node EP dispatches
`per_device_train_batch_size × max_length` tokens per rank in one MoE forward, and the dispatcher
rejects anything above `HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK` (8192) at buffer sizing — above it a
proxy-GIN dispatch wedges in transit instead of erroring. So every `ep_scope=global` row is capped at
8192 tokens/rank whatever its memory column says. The `node`-scope rows are unaffected:
intra-node dispatch is validated to 65k tokens/rank.

**TP is a net loss on this model.** `apply_tp_to_attention_only` shards only q/k/v/o of the 15
full-attention layers — 1.57 B of 10.25 B. It leaves the 45 gated-DeltaNet layers untouched
(`_find_attention` probes `self_attn`/`attention`/`attn`, and the linear block is `linear_attn`, so
the loop skips them — warning which layers it left replicated) and leaves embeddings and `lm_head`
replicated by design. What
TP does do is divide `data_parallel_size`, which divides `F` — so it *raises* the non-expert term.

**CP is rejected outright.** `validate_model_for_ulysses`
(`src/distributed/context_parallel/validation.py`) raises on `Qwen3_5MoeGatedDeltaNet` and, on a meta
shell, on any `linear_attention` entry in `layer_types`: `causal_conv1d_fn` needs a cross-rank halo
and `chunk_gated_delta_rule` is a recurrent scan. Long context comes from the activation headroom at
4–8 nodes, not from CP.

## The 3-node case

24 GPUs is three NVLink domains, and nothing in the code requires a power of two — `world_size %
nvlink_domain_size == 0` passes and `ParallelismConfig` validates world 24 exactly as it does world
16. The model is what fails, in two independent ways.

**Global-scope EP on 24 must be a multiple of 3.** `cross_node_layout` (called from
`_validate_ep_group`) requires `ep_group_size % num_domains == 0` with `num_domains = 3`, and
`nvlink_domain_size % members_per_domain == 0`, leaving exactly `ep_size ∈ {3, 6, 12, 24}`. `512` is
`2⁹`, so **every one of them fails `512 % ep_size == 0`** — raised by
`validate_against_model_config` off `config.json`, at the top of the model load and before any
process group. `_validate_ep_group` refuses the rest at global scope in order: `ep_size=16` first, on
`reject_cross_node_ep_group` (`24 % 16 ≠ 0`), then `ep_size ∈ {2, 4, 8}` on `cross_node_layout` —
they cannot tile 3 domains equally. The single EP+ETP shape at
world 24 is `ep_size=3, expert_tp_size=8`, which dies on the same divisor.

**Node-scope EP caps `X` at 8.** `_validate_ep_group` requires `nvlink_domain_size % ep_group_size ==
0`, so `ep_size ∈ {2, 4, 8}` — and `8·P_e/8 = 386 GB` of experts per rank before anything else. The
same ceiling makes node-scope EP useless at 2, 4 and 8 nodes too.

**Nothing survives: 24 GPUs has no legal layout for this model in this release.** The shape that
would survive is `pp3` + `ep8` — `pp_size=3` gives `stage_world_size = 8`, satisfying
`stage_world_size % nvlink_domain_size == 0` where `pp2`, `pp4`, `pp6`, `pp8` and `pp12` at world 24
all fail it — but pipeline parallelism is
[not yet available in this release](pipeline-parallelism.md). Use 2 nodes (`ep16` global) or 4
(`ep32` global) instead.

## Recommended cell

On an 8-GPU domain every layout puts the EP dispatch on RDMA — pick the widest EP group that loads,
and narrow the dispatch group (EP+ETP) when the fabric is latency-bound.

| Nodes | Cell | Crosses on | Steady `b1 s8192` | Why |
|---|---|---|---:|---|
| 2 | `ep16 --ep_scope=global` + `unfreeze_layers_patterns` | RDMA all-to-all | 228 GB full FT / ≈120 GB partial | the only layout that loads and fits; no headroom for 32k or `b>1` |
| 3 | — | — | — | no legal layout for this model in this release ([The 3-node case](#the-3-node-case)) |
| 4 | **`ep32 --ep_scope=global`** | RDMA all-to-all | 129 GB | first comfortable point on memory — but the binding limit is the 8192 tokens/rank cross-node dispatch ceiling, not the 120 GB spare |
| 8 | **`ep64 --ep_scope=global`** | RDMA all-to-all | 80 GB | memory headroom to spare, same 8192 tokens/rank ceiling on `b × max_length`; `ep8 + etp8` if the fabric is EFA |

Pipeline parallelism would add the alternative that keeps EP node-local and puts only P2P boundary
activations on the fabric — [not yet available in this release](pipeline-parallelism.md).

**On NVL72, declare a 64-GPU domain — not 72.** `_validate_ep_group` requires
`nvlink_domain_size % ep_group_size == 0`, and `72 % 64 = 8`. At `NVLINK_DOMAIN_SIZE=72` the only EP
widths that survive validation are `ep2` and `ep72` — `ep4`/`ep8` fall to the racy-EP guard
(scope-blind on a single domain), `ep16`/`ep32`/`ep64` to the divisibility rule — and `512 % 72 ≠ 0`
then kills `ep72` at model load. **A 512-expert model has no legal EP shape on a full 72-wide rack.**

The working recipe is `NVLINK_DOMAIN_SIZE=64` over 16 of the rack's 18 compute trays, with
`ep_size=64, ep_scope=node`: one dispatch group per rack, 8 experts/rank, ~48 GB of expert state,
`requires_rdma` false. It costs 8 of 72 GPUs, 11% idle. The general form
— the domain is the tiling unit, so the legal `ep_group_size` values are its divisors — is on
[Multi-Node](multi-node.md#gb200gb300-nvl72-multi-node-nvlink).

Scaling out, `world=512` over 8 racks at domain 64 accepts `ep8`, `ep32` and `ep64` node-scope (the
racy guard is inert across multiple domains); `ep64` gives 8 EP groups, one per rack, averaged by the
deferred cross-replica sweep. Unvalidated on hardware — treat it as a plan.

**Frozen layers are the lever at 2 nodes.** `unfreeze_layers_patterns` leaves untouched parameters at
2 B/param — weights only, no gradient and no optimizer state. Unfreezing the top 20 of 60 layers plus
routers and `lm_head` takes the expert term from `8·P_e/X` to `4·P_e/X` — 193 → 97 GB at `X=16`.

**When the fabric is the bottleneck, narrow the dispatch group.**
Cross-node EP over EFA is proxy-Gin latency-bound
([DeepEP → AWS EFA](../infrastructure/deepep.md#expert-parallelism-over-aws-efa)).

`ep64` gives each rank 56 cross-node dispatch partners; `ep8 + expert_tp_size=8` gives the same `X=64` with **7**,
because `_validate_expert_tp` forces exactly one ETP group per NVLink domain
(`expert_tp_size == members_per_domain`, `ep_size == num_domains`). EP+ETP is an experimental axis —
its expert-TP reduce runs in token space outside the DeepEP dispatch→combine span.

On InfiniBand with IBGDA the latency wall does not exist; use plain `ep64` — but size the microbatch against the arena
term above, which `ep8 + etp8` does not pay (its dispatch group is 8 wide, so its arena is an eighth
of `ep64`'s at the same tokens/rank).

## Launch recipes

One `torchrun` per node. No 397B config ships — derive one from the 122B sibling per
[YAML knobs](#yaml-knobs).

```bash
torchrun --nnodes=$NNODES --node_rank=$NODE_RANK --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
  scripts/training/sft.py <your-397b-config>.yaml <layout flags>
```

| Nodes | Layout flags |
|---|---|
| 2 | `--expert_parallel_size=16 --ep_scope=global` |
| 3 | — no legal layout for this model in this release ([The 3-node case](#the-3-node-case)) |
| 4 | `--expert_parallel_size=32 --ep_scope=global` |
| 8 | `--expert_parallel_size=64 --ep_scope=global` |
| 8, EFA | `--expert_parallel_size=8 --expert_tensor_parallel_size=8 --ep_scope=global` |

### Environment

The fabric block (EFA vs InfiniBand, proxy GIN, `NCCL_SOCKET_IFNAME`) is the standard one from
[Launch Recipes](launch-recipes.md#environment-variables). What this model changes:

```bash
export DIST_NCCL_TIMEOUT_MINUTES=90    # default 30; a 397B gathered save takes minutes
export DIST_STORE_TIMEOUT_HOURS=8      # default 4; a single-rank 794 GB download can outlast it
# GB200/GB300 NVL72 only — 64, not 72: ep_group_size must DIVIDE the domain, and 512 % 72 != 0.
# Launch on 16 of the rack's 18 compute trays; the other 8 GPUs sit idle.
export NVLINK_DOMAIN_SIZE=64
```

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` composes with the `ElasticBuffer` on single-node
runs. Set it when variable-shape packing at `per_device_train_batch_size > 1` fragments the
allocator; no shipped config sets it.

### YAML knobs

Base the config on `examples/sft/qwen3_5/qwen3.5-122b-a10b-ep.yaml` — the closest shipped shape.
Values that differ from it or are load-bearing at 397B:

| Knob | Value | Why |
|---|---|---|
| `expert_parallel_size` / `ep_scope` | per the table above / `global` | node-scope EP caps `X` at 8 and cannot hold `P_e` |
| `attn_implementation` | `flash_attention_2` | **pin it explicitly.** Auto-detection downgrades this family to `sdpa` on Blackwell ([FA4 backward NaN](../models/qwen3_5.md)), and `sdpa` is not a varlen backend, so `packing: true` would pay a dense mask over the flattened batch instead of consuming `cu_seqlens` ([Collators](../data/collators.md#constraints)). An explicit FA2 is kept as-is |
| `packing` / `padding_free` | `true` / `false` | fixed-length only — the M-RoPE varlen path crashes FA2 |
| `fsdp_reshard_after_forward` | `false` | ZeRO-3 is rejected under EP |
| `gradient_checkpointing` | `true` | the activation column assumes it |
| `bf16` + `optim` | as shipped | AdamWBF16 auto-enables whenever `bf16: true` and `optim` is `adamw_torch_fused`/`adamw_torch` outside accelerate-managed DDP — both are already the defaults, so neither key has to be written to get the 8 B/param cost. `optim: muon` or `flash_adamw` opts out |
| `fp32_experts` / `fp32_non_ep_params` | **omit** | the 35B configs set them; fp32 masters would take the expert term to 16 B/param and OOM every cell |
| `moe_balancing` | `aux_loss` | Under `text_only_model: true` the model is `Qwen3_5MoeForCausalLM`, whose forward declares `output_router_logits`, so `aux_loss` trains and exports by construction. Strict `bias_update` raises: Qwen3.5 has no checkpoint slot for a routing bias ([Callbacks](../training-methods/callbacks.md#moe-balancing-modes)) |
| `router_balancing_rate` | well below `1e-3` | the sign step lands on softmax probabilities, whose uniform scale is `1/512` here — the default γ is ~50% of it |
| `fp32_router` | `true` | cheap, and 512 experts at top-10 is where bf16 routing logits start to tie |
| `fp32_grad_reduce` | `true` | bf16 sums lose precision at 16–64 ranks; no storage cost |
| `use_grouped_gemm` / `use_liger_kernel` / `fp32_output_conversion` | `true` / `true` / `false` (all defaults) | the fused `[512, 2048, 4096]` layout is what `grouped_mm` wants on SM100, and fused linear cross-entropy plus the disabled upcast keep `[b, S, 248320]` — 4 GB per copy at `S=8192` — off the card |
| `max_concurrent_loading` | leave unset | node-local wave gate; unset adapts to the node (`min(4, max(1, local_world_size // 2))` — 4 on an 8-GPU node, 2 on a 4-GPU tray), and any explicit value is used verbatim. The lazy EP path bypasses it, so this only bounds the fallback |
| `save_sharded_ep` | `true` on the pure-EP cells (`ep16`/`ep32`/`ep64`) | every rank writes its own shard instead of funneling 794 GB through one. Requires a single EP group spanning all ranks (`ep_group_size == world_size` — exactly these cells) and a shared output filesystem; **rejected under ETP**, so the `ep8+etp8` cell takes the layer-streaming gathered save instead. Merge before serving ([Checkpoints](../reference/checkpoints.md#expert-parallelism-ep)) |
| `per_device_train_batch_size` | `1` | the activation column is `b=1`; raise only at 8 nodes and only after step 1's peak is known |
| `max_length` | `8192` on every `ep_scope=global` cell | doubling it doubles the activation column — but on cross-node EP the hard limit is the 8192 tokens/rank dispatch ceiling above, not memory. Raise the ceiling only after validating a larger dispatch end-to-end on your fabric ([DeepEP → AWS EFA](../infrastructure/deepep.md#expert-parallelism-over-aws-efa)) |
| `unfreeze_layers_patterns` | required at 2 nodes | see [Recommended cell](#recommended-cell) |
| `save_total_limit: 1` + `save_only_model: true` | as shipped | a bf16 checkpoint is ~800 GB per save |

Pre-shard the corpus with `scripts/before_training/prepare_dataset.py --num-shards N`,
`N >= data_parallel_size` — at 64 ranks the on-the-fly path materializes the dataset per rank.

## Other 300B-class checkpoints

The same arithmetic — 8 B/param on the trained experts, `X = ep_size × expert_tp_size`, the
non-expert weights gathered whole under ZeRO-2 — for the three other checkpoints in the 200–320B
class, on 8 × B300 (≈260 GB usable — [Budget](#per-rank-memory-model)). All three ship as composite VLMs with no text-only CausalLM
sibling; every figure is a plan, not a measurement. The PP rows are planned shapes — pipeline
parallelism is [not yet available in this release](pipeline-parallelism.md).

| Checkpoint | Routed experts | Cell | Expert leg / rank | Static / rank | What binds |
|---|---:|---|---:|---:|---|
| **GLM-5.3-Flash** (321B; 42 MoE layers × 288 × `M=2048, H=4096`) | 304 B | 2 × 8, `ep16 --ep_scope=global` | 19.0 B → 152 GB | ~200 GB (+33 GB gathered non-expert bf16, +13 GB fp32 non-EP masters/state) | the 8192 tokens/rank cross-node dispatch ceiling; `ep8` on one node is 304 GB of experts alone. Needs the bf16 conversion (~650 GB) |
| GLM-5.3-Flash on EFA | 304 B | 2 × 8, `ep2 --expert_tensor_parallel_size=8 --ep_scope=global` | same `X=16` | same | one cross-node dispatch partner per rank instead of 8 (one ETP group per domain, [EP+ETP](expert-tensor-parallelism.md#process-groups-epetp-combo)); experimental axis |
| GLM-5.3-Flash, planned PP shape ([not yet available](pipeline-parallelism.md)) | 304 B | 2 × 8, `pp2 --expert_parallel_size=8` (stage = one node) | 19.0 B → 152 GB | ~175 GB | forward-looking arithmetic: dispatch would stay on NVLink, bounding `max_length` by memory rather than the Gin ceiling |
| **Step-3.7-Flash** (198B; 42 MoE layers × 288 × `M=1280, H=4096`) | 190 B | 1 × 8, `ep8` | 23.8 B → 190 GB | ~213 GB (+15 GB gathered non-expert, +6 GB shards) | ~60 GB left for activations at `max_length: 8192` with checkpointing — marginal, unmeasured |
| Step-3.7-Flash | 190 B | 2 × 8, `ep16 --ep_scope=global` | 11.9 B → 95 GB | ~115 GB | the 8192 tokens/rank dispatch ceiling |
| **DeepSeek-V4-Flash** (284B; 43 MoE layers × 256 × `M=2048, H=4096`, 3 of them hash-routed) | 277 B | 2 × 8, `ep16 --ep_scope=global` | 17.3 B → 138 GB | ~155 GB | the 8192 tokens/rank dispatch ceiling. Needs the bf16 conversion (~750 GB) |
| DeepSeek-V4-Flash, planned PP shape ([not yet available](pipeline-parallelism.md)) | 277 B | 2 × 8, `pp2 --expert_parallel_size=8` (stage = one node) | 17.3 B → 138 GB | ~155 GB | forward-looking arithmetic: dispatch would stay on NVLink, bounding `max_length` by memory rather than the Gin ceiling |

The `ep16` rows all sit on cross-node EP and inherit its `per_device_train_batch_size × max_length
≤ 8192` contract; the two planned PP rows are forward-looking arithmetic for the shape that would
keep every MoE all-to-all on NVLink.

## Pre-flight checklist

**Before the launch**

- `512 % ep_size == 0` and `expert_tp_size | moe_intermediate_size` are checked off `config.json` in
  the first second (`ParallelismConfig.validate_against_model_config`, at the top of
  `load_distributed_model`). The other model-dependent gates still fire minutes in — TP head
  divisibility, `validate_model_for_ulysses`.
- Scratch: a single checkpoint is ~800 GB with `save_only_model: true` (as shipped), ~2.4 TB with
  optimizer shards (AdamWBF16 adds 4 B/param). Rotation deletes the previous checkpoint only after
  the new one completes, so `save_total_limit: 1` still transiently holds **both**: budget ~1.6 TB
  free per save, ~4.8 TB with optimizer shards. Confirm the volume with `findmnt`/`df -h` before
  pointing `HF_HOME`, `HF_DATASETS_CACHE`, `TMPDIR` and `HALO_DATA_ROOT` at it. On the
  `save_sharded_ep` cells budget for the merge too: `merge_ep_shards.py` streams (peak host RAM ≈ one
  merged MoE layer, not the ~800 GB artifact), and `--delete_input_shards` frees the per-rank shards
  only once the merged checkpoint is complete, so the volume briefly holds both.
- Host RAM: the EP lazy path reads safetensors straight to GPU and needs almost none at load —
  every slice is a view over the mmapped shard, so anonymous RSS stays at the process baseline and
  `max_concurrent_loading` never applies; what `free` shows is page cache the size of the checkpoint
  (~800 GB here; 398 / 628 / 582 GB for the Step-3.7 / GLM-5.3 / DeepSeek-V4 Flash artifacts of
  [the 300B table](#other-300b-class-checkpoints)), shared by the node's ranks and reclaimable, and
  each rank's file-backed RSS reaches its dense + own expert slices while its shard handles are
  open. Not a leak. The gate+up fan-in's 2× transient lives on the GPU (~1 GB per layer). If
  `ep_lazy_loading` is off or the checkpoint layout is lazy-incompatible, the fallback stages the
  full model in CPU RAM per concurrent loader — set `max_concurrent_loading: 1` when `free -g` is
  tight. A run that keeps optimizer state (`save_only_model: false`) also copies each rank's
  optimizer shard to host RAM at every save and resume (`cpu_offload`): ~28 GB/rank at `ep64`,
  ~102 GB/rank at `ep16` — ~820 GB per 8-GPU node. Every cell here runs **one** EP group, so each
  rank's expert state is unique; a job wide enough to run several EP groups (they are DP replicas)
  writes that half once per replica group instead
  ([Checkpoints](../reference/checkpoints.md#warm-restart-vs-exact-resume-torchrun)).
- Hardware gates on every node: `scripts/profiling/nvlink_health.py` (non-zero exit = a degraded
  lane that silently caps or hangs the EP all-to-all), `nvidia-smi` (8 GPUs, near-zero used memory,
  no ECC/Xid), and the fabric check for your interconnect
  ([RDMA fabrics](multi-node.md#rdma-fabrics)).
- Dataset shards: `num_shards >= data_parallel_size`, which is `stage_world_size / max(tp, cp, etp)`
   — not the GPU count.
- Dry-run the layout on one node with a tiny Qwen3.5 MoE (`Qwen/Qwen3.5-35B-A3B`) to exercise the
  validators before the real weights load.

**After step 1**

| Check | Expected | If not |
|---|---|---|
| Peak memory (`EfficiencyCallback`) | within ~15% of the matrix cell | recheck `P_e`/`P_ne`; the replicated `P_r` term is the usual surprise |
| `grad_norm` | finite, same order as a 35B-A3B run of the same recipe | see [Debugging](../reference/debugging.md) |
| `moe/*` metrics present and non-zero | `RouterBiasBalancingCallback` logs them under either bias-update mode | this checkpoint keeps its expert fields on `text_config`, and `detect_moe_experts_topk` resolves the text config before reading them. Silent `(0, 0)` means the model is not being seen as MoE and `moe_balancing` has resolved to `none` — the router is training unbalanced |
| Packing took the varlen path | no dense-mask warning from `select_data_collator` | a non-varlen `_attn_implementation` pays a dense mask over the flattened batch, and on some families it also stops isolating documents ([Document isolation](../data/collators.md#document-isolation-under-packing)). Pin `flash_attention_2` |
| Expert balance (`MoEMetricsCallback`, or `RouterBiasBalancingCallback` under a bias-update mode) | no expert at 0 tokens across a step | raise `router_balancing_rate` — but see the 512-expert scale caveat in [Callbacks](../training-methods/callbacks.md#routerbiasbalancingcallback) before doubling it |
| tok/s/GPU | compare against the [MFU registry](../optimization/throughput-benchmarks.md) at 35B-A3B | a 2×+ shortfall on EFA is the proxy-Gin latency wall: narrow the dispatch group (`ep8 + etp8`) |
| Step time variance | stable after 3 warmup steps | a slow rank is usually a degraded NVLink lane or a straggler NIC |
| First checkpoint | written, and re-loadable with `from_pretrained` — on the `save_sharded_ep` cells only **after** `merge_ep_shards.py`; the artifact itself is per-rank (`format: "ep_sharded"`) and every loader refuses it unmerged. The gathered `ep8+etp8` row loads directly | raise `DIST_NCCL_TIMEOUT_MINUTES` before blaming the fabric |
