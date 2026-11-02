# Parallelism

On top of data parallelism, four strategies address different bottlenecks: model size (TP), sequence
length (CP), expert distribution (EP), or depth across NVLink domains (PP). They combine within the
supported set below. Each mode is a communication pattern with a price:
[GPU Training Theory §9](../reference/gpu-training-theory.md#9-distributed-training-the-communication-wall).

Multi-GPU DP modes run on FSDP2 for gradient sync (pure TP at DP=1 and single-GPU skip it);
`DistributedTrainerMixin` selects the sharding strategy from the active mode. See the
[Data Parallelism Guide](data-parallelism.md) for launchers and Accelerate configs.

## Parallelism modes

- **Expert parallelism (EP)** — distributes MoE expert layers across GPUs via DeepEP all-to-all (the router stays replicated on every rank). Each GPU holds a subset of experts and routes tokens to the owning GPU. For MoE models; see [Expert Parallelism Guide](expert-parallelism.md).
- **Tensor parallelism (TP)** — shards dense attention and MLP weights across GPUs via DTensor with all-reduce. For large dense models (Llama 70B+, Mistral Large) that exceed single-GPU memory: plain FSDP2 and DDP load the whole model onto every GPU first ([why](data-parallelism.md#fsdp2-strategy-by-mode)), and TP's HF `tp_plan` load, which shards as it reads, is the route past that inside one NVLink domain (PP's stage loader is the one across domains); see [Tensor Parallelism Guide](tensor-parallelism.md).
- **Context parallelism (CP)** — splits long sequences across GPUs via Ulysses-style attention with all-to-all. For long-context training (32K+ tokens); see [Context Parallelism Guide](context-parallelism.md).
- **Pipeline parallelism (PP)** — splits decoder layers into stages, one contiguous rank block each, and is the only axis meant to cross NVLink domains. Off by default ([why](#when-pipeline-parallelism-is-worth-it)). See [Pipeline Parallelism](pipeline-parallelism.md).

Which MoE families each axis wraps: [Supported Models → Compatibility matrix](../models/README.md#compatibility-matrix).

## Supported combinations

All modes use FSDP2 (`fully_shard`, per [unit of decoder layers](data-parallelism.md#shard-groups)). EP expert modules sit in `ignored_params`; their routers sync via EP backward hooks, and their experts (with one EP group) or everything (with more) via the deferred post-backward sweep, except at `ep_group_size == 1`, where `fsdp_shard_ep1_experts` (default on) has FSDP2 shard them; CP wraps the attention path for sequence splitting and lets FSDP2 sync the rest.

| Mode | Data Parallel Size | Notes |
|------|-------------------|-------|
| No parallelism | `world_size` | Standard multi-GPU |
| EP only | `world_size` | EP orthogonal to DP; full DP preserved |
| TP only | `world_size / tp_size` | DTensor mesh reduces effective DP |
| CP only | `world_size / cp_size` | Ulysses attention reduces effective DP |
| EP+CP | `world_size / cp_size` | Only CP reduces DP |
| EP+TP | `world_size / tp_size` | Attention sharded (node-local TP), experts distributed; TP adds no expert sharding |
| ETP (pure, `ep_size=1`) | `world_size / expert_tp_size` | Expert FFN sharded `expert_tp_size`-way; experts replicated. MoE only — a dense model raises. See [ETP guide](expert-tensor-parallelism.md) |
| EP+ETP | `world_size / expert_tp_size` | Experts distributed `ep_size`-way **and** each expert's FFN sharded `expert_tp_size`-way (`ep_group_size = ep_size × expert_tp_size`). Experimental. The expert-TP reduce runs in token space, outside the DeepEP dispatch→combine span |
| PP | `world_size / pp_size` | Stages are contiguous rank blocks of whole NVLink domains; every other axis runs unchanged inside one stage. See [Pipeline Parallelism](pipeline-parallelism.md) |
| PP+EP | `world_size / pp_size` | EP inside each stage; the cross-replica expert average is deferred to a post-backward sweep |
| PP+ETP | `world_size / (pp_size × expert_tp_size)` | Pure ETP (`ep_size=1`) inside each stage. The deferred sweep divides the expert leg by `stage_world_size / expert_tp_size` — ETP partners share a batch and are not DP replicas |
| **Anything else** | **Not supported** | Rejected at config time with the mechanism named — see below |

The supported set is an **allowlist** (`SUPPORTED_AXIS_SETS` in
`halo/distributed/parallelism_rules.py`), checked first by `reject_unsupported_axes`, before any rank
math, so a combination nobody considered is refused rather than run unvalidated. Adding one means
adding the equivalence gate that proves its gradients against an unsplit reference. Rejections quote
the flags you typed and the mechanism from `AXIS_SET_MECHANISMS` that would break, for example:

- **TP+CP** — TP shards attention over a DTensor `(dp, tp)` mesh while CP's Ulysses groups partition the same ranks a second, incompatible way.
- **TP+ETP**, with or without EP — attention TP and expert TP would shard the same ranks along two different axes. Use EP+TP or pure ETP.
- **ETP+CP** — expert-TP partners hold shards of one expert and must see the **same** tokens (`ReduceFromExpertTP` sums their outputs element-wise in token space), but CP hands each rank a different sequence chunk, so that sum would add unrelated tokens.
- **PP with anything but EP or pure ETP** — the full PP refusal list: [Pipeline Parallelism → Rejected combinations](pipeline-parallelism.md#rejected-combinations).

### Topology rules

Inside an allowed set, `reject_invalid_composition` checks the layout against the NVLink domain
(`nvlink_domain_size`: the OS node, or the rack on NVL72), also before the model loads:

- **Locality.** `tp_size`, `cp_size` and `expert_tp_size` divide the domain (`_reject_tp`,
  `_reject_cp_locality`, `_reject_expert_tp`). A node-scope `ep_group_size` divides the domain; a
  global one must tile the stage's ranks as equal contiguous per-domain blocks (`_reject_ep_group`,
  [Multi-Node](multi-node.md#node-local-vs-cross-node-ep)). `world_size` is a multiple of both
  `gpus_per_node` and the domain (`ParallelismConfig._resolve_nvlink_domain`).
- **One domain, one dispatch group.** Within a single domain pure EP runs at `ep_size == 2` or with
  `ep_group_size` filling the domain; `ep4` on 8 GPUs is refused (`_reject_single_domain_multigroup_ep`,
  [mechanism and evidence](expert-parallelism.md#single-domain-multi-group-ep-races-and-hangs)).
- **EP+TP.** `ep_size` is a multiple of `tp_size`. TP leaves `ep_group_size` at `ep_size`, so the
  single-domain rule applies unchanged: `ep2+tp2` and `ep8+tp2` on 8 pass, `ep4+tp2` is refused.
  Across domains the EP group must be a **single global** one (`ep_size == world_size`,
  `ep_scope: global`): multi-group EP there defers its replica average by sharding over the EP group,
  while EP+TP shards over the `(dp, tp)` mesh (`_reject_tp`).
- **EP+CP.** Node-local EP with `ep_group_size == nvlink_domain_size`; global-scope EP is refused
  (`_reject_ep_cp`). `cp_size` only has to divide the domain, so `ep8+cp2`, `ep8+cp4` and `ep8+cp8` all
  run on one 8-GPU node.
- **EP+ETP.** The ETP groups stay inside one domain. EP may cross domains as a single EP group with
  exactly one ETP group per domain; multi-group EP+ETP across domains is refused, since expert TP keeps
  the deferred replica average off (`_reject_expert_tp`,
  [ETP validation rules](expert-tensor-parallelism.md#validation-rules)).
- **HSDP** composes with pure DP and CP only (`_reject_hsdp`); the FSDP shaping knobs carry per-axis
  rules (`_reject_fsdp_settings`, [Data Parallelism → Limitations](data-parallelism.md#limitations)).
- **Model shape.** `reject_unsupported_model` reads `config.json` before any weight: `num_experts`
  divisible by `ep_size`, `expert_tp_size` dividing the expert FFN width, head counts divisible by
  `tp_size`, and the declared per-rank token budget under DeepEP's dispatch ceilings
  ([DeepEP backend](expert-parallelism.md#deepep-backend)). Then `reject_unsupported_structure` checks a
  meta-device shell of the model, still before any weight: the Ulysses refusals and CP head counts under CP
  ([Context Parallelism](context-parallelism.md#requirements)), and the zero-shard check under selective TP
  ([Tensor Parallelism](tensor-parallelism.md#limitations)).

## Communication and data flow

Two ways to group ranks, used where each fits:

- **`DeviceMesh` (DTensor)** for the data-parallel and tensor axes, where sharding is *static* and the
  collective follows from the tensor layout: `fully_shard` reduce-scatters FSDP params, and TP's
  linears sum their activations over the `tp` group.

    The meshes are built in one place, `halo/distributed/mesh.py`: a 1D `(dp,)` mesh, a 2D HSDP
    `(dp_replicate, dp_shard)` mesh (replicate across NVLink domains, shard within one), and the
    `(tp,)` / `(dp, tp)` tensor mesh. The trainer reads the resulting groups through its accessors
    (`tp_group`, `tp_local_rank`, `dp_group`).

- **Hand-built process groups** (`build_groups` in `halo/runtime/collectives.py`) for EP, CP and PP, whose
  communication DTensor cannot express: EP routing is *data-dependent* (the router picks each token's experts
  at runtime), EP and CP run *custom all-to-all kernels* (DeepEP dispatch/combine, Ulysses sequence↔head swap)
  rather than a layout-inferred redistribution, and PP sends point-to-point along a pipeline chain
  (`halo/distributed/pipeline_parallel/groups.py`). On an NCCL default group bound to its device (the
  `init_distributed` launch) every group of one kind comes from one `split_group`, a single communicator split;
  a `dist.new_group` per group would split the world communicator once per group, each split blocking every
  rank (136 world-wide splits for `ep8` on 1,024 ranks, 520 on 4,096).

    DeepEP also needs contiguous rank blocks for its intra-node kernel, which a mesh's row-major
    order cannot produce. `EPConfig` (`halo/distributed/expert_parallel/config.py`) builds the EP,
    expert-replica, dispatch (sub-EP) and expert-TP groups; `CPConfig`
    (`halo/distributed/context_parallel/config.py`) builds the CP groups. All of them, and
    `ParallelismConfig`, take their membership from the two layouts in
    `halo/distributed/group_layout.py`: `NodeLocalLayout` and the cross-node `CrossNodeLayout`.

What crosses the wire:

| Mode | Collective | Over |
|---|---|---|
| FSDP2 | all-gather params (forward), reduce-scatter grads (backward) | `(dp,)` mesh |
| HSDP | all-gather/reduce-scatter within a domain, one all-reduce across replicas per backward | `(dp_replicate, dp_shard)` mesh |
| TP | all-reduce / all-gather per layer | `(tp,)` axis |
| CP | Ulysses all-to-all (sequence ↔ heads) around attention | CP group |
| EP | DeepEP all-to-all dispatch → local expert GEMM → all-to-all combine | EP dispatch group |
| EP+ETP | expert-TP all-reduce in token space, outside the dispatch→combine span | expert-TP group |
| PP | boundary activation forward, its gradient back, point to point | pipeline chain (one rank per stage) |

Backward mirrors the table: every activation all-to-all / all-reduce has an autograd transpose. By
default FSDP2 reduce-scatters after every accumulation micro-step's backward
([`fsdp_defer_grad_sync`](data-parallelism.md#deferred-gradient-reduce-fsdp_defer_grad_sync) holds
it to the last); the EP gradient sync runs on the last micro-step: the router hooks during its backward (once per
recompute under reentrant checkpointing, an average that repeats harmlessly), the deferred sweep after it.

EP gradients sync outside FSDP2: the replicated router all-reduces over the world group (the stage
under PP), and expert grads all-reduce over the expert-replica group only when a run holds more than
one EP group.
`fp32_grad_reduce` runs that reduction in fp32 over bf16 storage.

The global batch is split along the data-parallel axis only: each of `data_parallel_size` ranks takes
a disjoint slice, sharded by `data_parallel_rank` in `halo/trainers/dataloaders.py`. Ranks
inside a TP, CP or ETP group, and the stages of one pipeline chain, see the **same** batch; they
shard weights, the sequence or the layers, not the data.

EP is orthogonal: every rank still draws its own batch, and tokens are all-to-all'd to the ranks
owning their routed experts, then returned in place after combine. CP splits the **sequence**: each
rank holds `seq_len / cp_size` tokens of its batch, and the Ulysses all-to-all reconstructs full
sequences inside attention.

### Stacking the dimensions

`data_parallel_size` counts distinct batches. PP carves the world first (`stage_world_size =
world_size / pp_size`), then TP, CP or ETP divide each stage:

```text
data_parallel_size = world_size / pp_size / max(tp_size, cp_size, expert_tp_size)
```

EP does not reduce it, and the divisor is a max, not a product: no two of TP, CP and ETP may exceed
1 at once. FSDP2's shard group is chosen per mode: the whole stage (the world at `pp_size` 1) for
plain DP, CP, EP and ETP (CP and ETP ranks shard parameters even though they share a batch); the
`dp` dimension of the `(dp, tp)` mesh under TP and EP+TP; and the EP group under multi-group EP
across NVLink domains, where the cross-replica average is deferred to the post-backward sweep.

HSDP is not a separate layer on top of FSDP2; it is how the plain-DP or CP shard group is meshed.
The default is 1D full-shard. Under `--use_hsdp` it is a 2D `(dp_replicate, dp_shard)` mesh that
shards within one NVLink domain (`dp_shard_size = nvlink_domain_size`) and replicates across domains
(`dp_replicate_size = num_nvlink_domains`), so on a single domain it is a no-op
([HSDP](data-parallelism.md#hsdp-hybrid-sharded-data-parallel)).

Node-local EP+CP reaches cross-domain depth without HSDP: EP dispatch/combine and CP Ulysses
all-to-all stay node-local on NVLink, FSDP shards the non-expert params within the EP group, and the
cross-replica DP average is deferred to one post-backward sweep over RDMA per step. EP+TP and EP+ETP
reach the same depth by sharding attention (TP) or the expert FFN (ETP) across the NVLink domain
instead of replicating the DP dimension across domains.

## torch.compile

`torch.compile` does not pay on EP MoE, and PP refuses it; keep `torch_compile` off and Liger on ([torch.compile](../optimization/torch-compile.md)).

## Per-axis limitation surface

Each axis page states its own knob rejections, every one traced to the code that raises:
[EP](expert-parallelism.md#limitations) · [ETP](expert-tensor-parallelism.md#limitations) ·
[TP](tensor-parallelism.md#limitations) · [CP](context-parallelism.md#limitations) ·
[PP](pipeline-parallelism.md#rejected-combinations) · [DP](data-parallelism.md#limitations). No trainer
refuses EP, ETP or TP; CP and PP are declared per class (`SUPPORTS_CP` / `PP_REFUSAL`):
[Trainer Compatibility](../reference/trainer-architecture.md#trainer-compatibility). Which family
runs which axis: [Compatibility matrix](../models/README.md#compatibility-matrix).

## Recommendations

| Model | Short Seq (<32K) | Long Seq (>32K) | Max Memory Efficiency |
|---|:--:|:--:|:--:|
| Dense | Standard or TP | CP where the family has a wrapper, else TP | TP |
| MoE with CP and TP | EP or EP+TP | EP+CP | EP+TP or pure ETP |
| MoE with TP, no CP | EP or EP+TP | EP at max single-GPU seq | EP+TP or pure ETP |
| MoE with CP, no TP | EP | EP+CP | pure ETP |
| MoE with neither | EP | EP at max single-GPU seq | pure ETP |
| VLM without a TP plan (Qwen3-VL) | Standard | Standard, CP on text-only batches | ZeRO-3 FSDP2 (`fsdp_reshard_after_forward: true`) |

A model too large for one NVLink domain adds PP across domains where its family is
[PP-registered](pipeline-parallelism.md#model-support), a MoE with EP (or experimental pure ETP) inside
each stage ([When PP is worth it](#when-pipeline-parallelism-is-worth-it)).

A family's CP, TP and PP verdicts, and its qualifiers (Zaya trains without gradient checkpointing in every
mode): [Compatibility matrix](../models/README.md#compatibility-matrix).

## Additional guides

- [Data Parallelism (FSDP)](data-parallelism.md) — FSDP strategies, Accelerate configs, mixin internals
- [Expert Tensor Parallelism](expert-tensor-parallelism.md) — shard expert FFN weights for extra memory reduction
- [Large-Scale Scenarios](large-scale-scenarios.md) — pick a layout by NVLink-domain size; start here for any multi-node job
- [Multi-Node Parallelism](multi-node.md) — cross-node EP/CP/TP, EP scopes, RDMA setup
- [Multi-Node Launch Recipes](launch-recipes.md) — runnable torchrun/SLURM launches, EP checkpoint save/load
- [Distributed Data Loading](data-loading.md) — data loading across parallelism modes
- [Checkpoints and Resume](../reference/checkpoints.md) — distributed checkpoint formats (gathered vs sharded) and resume

## When pipeline parallelism is worth it

Inside one NVLink domain PP earns nothing: FSDP2 + EP/TP/ETP/CP already run every collective on the fabric, and PP only adds bubbles. Across domains the trade is which traffic crosses RDMA: global-scope EP puts **four** latency-bound all-to-alls there per MoE layer per microbatch (dispatch and combine each repeat in backward), while PP keeps EP node-local and sends `2 × (pp_size − 1)` point-to-point activations per microbatch. On an NVL72 rack the whole EP group rides NVLink and global EP wins outright.

PP is not a drop-in: [what it refuses](pipeline-parallelism.md#rejected-combinations). Layouts without it: [Large-Scale Scenarios](large-scale-scenarios.md) and [Capabilities & Limitations at Scale](../reference/scale-and-limitations.md).
