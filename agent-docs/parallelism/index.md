# Parallelism

On top of data parallelism, three strategies address different bottlenecks — model size (TP), sequence
length (CP), expert distribution (EP) — and they combine within the supported set below. A fourth,
pipeline parallelism (depth across NVLink domains), is [not yet available in this release](pipeline-parallelism.md). Multi-GPU DP modes run on FSDP2 for gradient sync (pure TP at DP=1 and
single-GPU skip it); `DistributedTrainerMixin` selects the sharding strategy from the active mode.
See the [Data Parallelism Guide](data-parallelism.md) for launchers and Accelerate configs.
Each mode is a communication pattern with a price:
[GPU Training Theory §9](../reference/gpu-training-theory.md#9-distributed-training-the-communication-wall).

## Parallelism modes

- **Expert parallelism (EP)** — distributes MoE expert layers across GPUs via DeepEP all-to-all (the router stays replicated on every rank). Each GPU holds a subset of experts and routes tokens to the owning GPU. EP is orthogonal to data parallelism (every GPU still processes its own batch). For MoE models; see [Expert Parallelism Guide](expert-parallelism.md).
- **Tensor parallelism (TP)** — shards dense attention and MLP weights across GPUs via DTensor with all-reduce. For large dense models (Llama 70B+, Mistral Large) that exceed single-GPU memory; see [Tensor Parallelism Guide](tensor-parallelism.md).
- **Context parallelism (CP)** — splits long sequences across GPUs via Ulysses-style attention with all-to-all. For long-context training (32K+ tokens); see [Context Parallelism Guide](context-parallelism.md).
- **Pipeline parallelism (PP)** — would split decoder layers into stages, one contiguous rank block each, as the only axis meant to cross NVLink domains. **Not yet available in this release**: the config surface and seams ship, the schedule engine does not, and `pipeline_parallel_size > 1` is rejected at config time. See [Pipeline Parallelism](pipeline-parallelism.md) and [the rationale](#when-pipeline-parallelism-is-worth-it).

Wrapped MoE families: [Qwen3 MoE](../models/qwen3.md#qwen3-moe), [Qwen3.5/3.6 MoE](../models/qwen3_5.md), [GPT-OSS](../models/gpt-oss.md), [GLM-4 MoE Lite](../models/glm4.md), [Laguna](../models/laguna.md), [Inkling](../models/inkling.md), [Gemma 4](../models/gemma4.md), [Bailing / Ling](../models/bailing.md), [LFM-2](../models/lfm2.md), [Mistral4](../models/mistral4.md), [DeepSeek-V4](../models/deepseek-v4.md), [Zaya](../models/zaya.md), [Cohere2 MoE](../models/cohere2-moe.md), [GLM-5 Next](../models/glm5-next.md), [Step-3.7 Flash](../models/step3p7.md). Full list and per-family quirks: [Supported Models](../models/index.md).

## Supported combinations

All modes use FSDP2 (`fully_shard`, per-layer). EP expert modules sit in `ignored_params` and sync via EP backward hooks; CP wraps the attention path for sequence splitting and lets FSDP2 sync the rest.

| Mode | Data Parallel Size | Notes |
|------|-------------------|-------|
| No parallelism | `world_size` | Standard multi-GPU |
| EP only | `world_size` | EP orthogonal to DP; full DP preserved |
| TP only | `world_size / tp_size` | DTensor mesh reduces effective DP |
| CP only | `world_size / cp_size` | Ulysses attention reduces effective DP |
| EP+CP | `world_size / cp_size` | Only CP reduces DP. Node-local EP requires `ep_group_size == nvlink_domain_size` (= `gpus_per_node` on a standard node; = the rack on NVL72) |
| EP+TP | `world_size / tp_size` | Attention sharded (node-local TP), experts distributed. TP leaves `ep_group_size` untouched, so it adds no expert sharding and does not change which EP topologies are legal. Across NVLink domains the EP group must be a **single global** one (`ep_size == stage_world_size`, `ep_scope=global`) — multi-group EP+TP is rejected |
| ETP (pure, `ep_size=1`) | `world_size / expert_tp_size` | Expert FFN sharded `expert_tp_size`-way; experts replicated. MoE only — a dense model raises. See [ETP guide](expert-tensor-parallelism.md) |
| EP+ETP | `world_size / expert_tp_size` | Experts distributed `ep_size`-way **and** each expert's FFN sharded `expert_tp_size`-way (`ep_group_size = ep_size × expert_tp_size`). Experimental. The expert-TP reduce stays node-local (token space, outside the DeepEP dispatch→combine span) while EP may be node-local **or cross-node** (one ETP group per NVLink domain) |
| PP | `world_size / pp_size` | In the allowlist, but the schedule engine is not shipped in this release — `pipeline_parallel_size > 1` is rejected at config time; seams only. See [Pipeline Parallelism](pipeline-parallelism.md) |
| PP+EP | `world_size / pp_size` | Same rejection — not yet available in this release |
| PP+ETP | `world_size / (pp_size × expert_tp_size)` | Same rejection — not yet available in this release |
| **Anything else** | **Not supported** | Rejected at config time with the mechanism named — see below |

The supported set is an **allowlist** (`SUPPORTED_AXIS_SETS` in
`src/distributed/parallelism_config.py`), checked before any rank math, so a combination nobody
considered is refused rather than run unvalidated. Adding one means adding the equivalence gate that
proves its gradients against an unsplit reference. Rejections quote the flags you typed and the
mechanism that would break, for example:

- **TP+CP** — TP shards attention over a DTensor `(dp, tp)` mesh while CP's Ulysses groups partition the same ranks a second, incompatible way.
- **TP+ETP**, with or without EP — attention TP and expert TP would shard the same ranks along two different axes. Use EP+TP or pure ETP.
- **ETP+CP** — expert-TP partners hold shards of one expert and must see the **same** tokens (`ReduceFromExpertTP` sums their outputs element-wise in token space), but CP hands each rank a different sequence chunk, so that sum would add unrelated tokens.
- **Any `pipeline_parallel_size > 1`** — pipeline parallelism is [not yet available in this release](pipeline-parallelism.md).

## Communication and data flow

Two ways to group ranks, used where each fits:

- **`DeviceMesh` (DTensor)** for the data-parallel and tensor axes, where sharding is *static* and the
  collective follows from the tensor layout: `fully_shard` reduce-scatters FSDP params, TP's
  `ColwiseParallel` / `RowwiseParallel` all-reduce attention weights. The meshes are built in one
  place, `src/distributed/mesh.py` — a 1D `(dp,)` mesh, a 2D HSDP `(dp_replicate, dp_shard)` mesh
  (replicate across NVLink domains, shard within one), and the `(tp,)` / `(dp, tp)` tensor mesh. The
  trainer reads the resulting groups through one `ParallelDims` view
  (`src/distributed/mesh.py`).
- **Hand-built process groups** (`dist.new_group`) for EP and CP, whose communication DTensor cannot
  express: EP routing is *data-dependent* (the router picks each token's experts at runtime), and
  both EP and CP run *custom all-to-all kernels* (DeepEP dispatch/combine, Ulysses sequence↔head
  swap) rather than a layout-inferred redistribution. DeepEP also needs contiguous rank blocks for
  its intra-node kernel, which a mesh's row-major order cannot produce.
  `src/distributed/expert_parallel/config.py` builds the EP dispatch, expert-replica, sub-EP and
  expert-TP groups; the CP path takes its group directly. All rank math — node-local vs cross-node,
  which dim divides `world_size` — lives in `src/distributed/group_layout.py` and `ParallelismConfig`.

What crosses the wire:

| Mode | Collective | Over |
|---|---|---|
| FSDP2 | all-gather params (forward), reduce-scatter grads (backward) | `(dp,)` mesh |
| HSDP | all-gather/reduce-scatter within a domain, one all-reduce across replicas per step | `(dp_replicate, dp_shard)` mesh |
| TP | all-reduce / all-gather per layer | `(tp,)` axis |
| CP | Ulysses all-to-all (sequence ↔ heads) around attention | CP group |
| EP | DeepEP all-to-all dispatch → local expert GEMM → all-to-all combine | EP dispatch group |
| EP+ETP | expert-TP all-reduce in token space, outside the dispatch→combine span | expert-TP group |

Backward mirrors the table — every activation all-to-all / all-reduce has an autograd transpose — but
the gradient *sync* (FSDP2 reduce-scatter, EP hooks) fires once per optimizer step, on the last
accumulation micro-step, not per layer. EP gradients sync outside FSDP2: the replicated router
all-reduces over the world group, and expert grads all-reduce over the expert-replica group only when
a run holds more than one EP group. `fp32_grad_reduce` runs that reduction in fp32 over bf16 storage.

The global batch is split along the data-parallel axis only: each of `data_parallel_size` ranks takes
a disjoint slice, sharded by `data_parallel_rank` in `src/trainers/mixins/dataloader.py`. Ranks
inside a TP, CP, or ETP group see the **same** batch — they shard weights or the sequence, not the
data. EP is orthogonal: every rank still draws its own batch, and tokens are all-to-all'd to the
ranks owning their routed experts, then returned in place after combine. CP splits the **sequence** —
each rank holds `seq_len / cp_size` tokens of its batch, and the Ulysses all-to-all reconstructs full
sequences inside attention.

Microbatching is TRL/Accelerate gradient accumulation (`gradient_accumulation_steps`); EP grad hooks
skip cross-rank sync on accumulation steps. [Pipeline parallelism](pipeline-parallelism.md) — the
outermost, cross-node dimension, which would replace that accumulation with its own microbatch
schedule — is not yet available in this release ([why it will matter](#when-pipeline-parallelism-is-worth-it)).

### Stacking the dimensions

The dimensions nest in a fixed order: EP, CP, TP, and ETP each carve their groups out of `world_size`
first; whatever remains is the data-parallel dimension, and FSDP2 shards the non-expert params over
it. HSDP is not a separate layer on top of FSDP2 — it is how that DP dimension is meshed: 1D
full-shard by default, or a 2D `(dp_replicate, dp_shard)` mesh under `--use_hsdp` that shards within
one NVLink domain (`dp_shard_size = nvlink_domain_size`) and replicates across domains
(`dp_replicate_size = num_nvlink_domains`). It composes only with pure DP and CP — TP, EP, ETP and PP
each reject it for their own reason, and on a single domain it is a no-op
([HSDP](data-parallelism.md#hsdp-hybrid-sharded-data-parallel)).

Node-local EP+CP reaches cross-domain depth without HSDP: EP dispatch/combine and CP Ulysses
all-to-all stay node-local on NVLink, FSDP shards the non-expert params within the EP group, and the
cross-replica DP average is deferred to one post-backward sweep over RDMA per step. EP+TP and EP+ETP
reach the same depth by sharding attention (TP) or the expert FFN (ETP) across the NVLink domain
instead of replicating the DP dimension across domains.

## torch.compile

On EP MoE, `torch_compile` reaches about the same speedup as Liger kernels (the default) and composes with them. DeepEP all-to-all (EP), DTensor dispatch (TP), and Ulysses all-to-all (CP) each break the graph, so compile only fuses the spans between breaks — its ceiling stays near Liger's. Liger has no per-shape warmup cost; add `torch_compile` only when the first-step compile latency is acceptable. See [torch.compile](../optimization/torch-compile.md).

## Per-axis limitation surface

Each axis page states its own supported trainers, supported models and knob rejections, every one
traced to the code that raises. This table is the index into them, not a second copy.

| Axis | Trainers | Models | Signature knob rejections |
|---|---|---|---|
| [EP](expert-parallelism.md#limitations) | all | the wrapped MoE families; a dense model raises | QLoRA, PEFT inside expert layers, `fsdp_reshard_after_forward`, `use_hsdp`, `bf16_optimizer: false`, `accelerate launch` |
| [ETP](expert-tensor-parallelism.md#limitations) | all (gated by `_supports_ep`) | every EP-capable MoE family | expert LoRA, `save_sharded_ep` — plus every EP rule |
| [TP](tensor-parallelism.md#limitations) | all | the attention classes in `TP_SHARDABLE_ATTENTION_CLASSES`; zero sharded layers raises | LoRA/PEFT, QLoRA, `fsdp_reshard_after_forward` at DP > 1, `use_hsdp` |
| [CP](context-parallelism.md#limitations) | SFT and SMPO only | the Ulysses attention wrappers | packing, padding-free, left padding, non-Flash attention, `label_smoothing_factor`, `loss_type: dft`, eval metrics, multimodal |
| [PP](pipeline-parallelism.md) | — (not yet available in this release; `pipeline_parallel_size > 1` is rejected at config time) | — | — |
| [DP](data-parallelism.md#limitations) | all | all | MoE grouped GEMM under `accelerate launch`, multi-device `device_map` |

Trainer support is declared per class (`_supports_ep` / `_supports_tp` / `_supports_cp` /
`_supports_pp`) and enforced in `ParallelismValidationMixin`; there is no `_supports_etp`, since ETP
folds into `ep_group_size`. Full matrix:
[Trainer Compatibility](../reference/trainer-architecture.md#trainer-compatibility). CP's `False`
rows are per-trainer declarations — `logits_to_keep`, global log-probability sums, full-sequence
pooling and dual models are the reasons behind them, not properties CP itself detects.

## Recommendations

| Model Family | Short Seq (<32K) | Long Seq (>32K) | Max Memory Efficiency | CP-Compatible |
|---|:--:|:--:|:--:|:--:|
| **Dense** (Llama, Mistral, Qwen3 dense) | Standard or TP | CP on Qwen3 dense; TP otherwise | TP | Qwen3 dense only |
| **MoE — full-coverage** ([Qwen3 MoE](../models/qwen3.md#qwen3-moe), [GPT-OSS](../models/gpt-oss.md), [GLM-4 MoE Lite](../models/glm4.md), [Mistral4](../models/mistral4.md) ⁵, [Cohere2 MoE](../models/cohere2-moe.md)) | EP or EP+TP | EP+CP | EP+TP or pure ETP | Yes |
| **MoE — EP/TP/ETP, no CP** ([Qwen3.5/3.6 MoE](../models/qwen3_5.md) ¹, [LFM-2](../models/lfm2.md) ⁶) | EP or EP+TP | EP at max single-GPU seq | EP+TP or pure ETP | **No** |
| **MoE — EP/ETP/CP** ([Bailing / Ling 2.0](../models/bailing.md)) | EP | EP+CP ² | pure ETP | Yes ² |
| **MoE — EP/ETP only** ([Inkling](../models/inkling.md) ⁸, Ring-mini-linear-2.0, Ling 3.0) | EP | EP at max single-GPU seq | pure ETP | **No** ² |
| **MoE — EP / pure ETP** ([Gemma 4](../models/gemma4.md), [Laguna](../models/laguna.md) ⁷) | EP | EP at max single-GPU seq | pure ETP ⁷ | **No** ³ |
| **MoE w/ CCA** ([Zaya](../models/zaya.md)) | EP (no GC) ⁴ or pure ETP (no GC) | — | Plain FSDP2 (no GC) | **No** ⁴ |
| **MoE — EP/ETP, no CP/TP** ([DeepSeek-V4](../models/deepseek-v4.md), [GLM-5 Next](../models/glm5-next.md), [Step-3.7 Flash](../models/step3p7.md)) | EP | EP at max single-GPU seq | pure ETP | **No** |
| **VLM** ([Qwen3-VL](../models/qwen3.md#qwen3-vl)) | Standard | CP | CP ⁹ | Yes |

¹ Qwen3.5/3.6 — CP blocked by interleaved linear-attention layers; see [qwen3_5.md](../models/qwen3_5.md).
² CP covers Ling 2.0 only. `Ring-mini-linear-2.0` is rejected by name (its file reuses Ling 2.0's full-attention class names over a Lightning-Attention-2 stack) and Ling 3.0 pairs a KDA linear recurrence with unwrapped MLA. EP+CP itself is untested on this family. See [bailing.md](../models/bailing.md#cp-wrapper).
³ Gemma 4 — TP and CP blocked by KV-shared layers + `attention_k_eq_v`; see [gemma4.md](../models/gemma4.md).
⁴ Zaya — CCA rules out CP/TP, GC unsupported; see [Zaya — Limitations](../models/zaya.md#limitations).
⁵ Mistral4 — the CP wrapper handles the MLA mismatched head dims, shared rope head, and llama-4 position scale (all-gathers `position_ids` across the CP group). See [mistral4.md](../models/mistral4.md).
⁶ LFM-2 — CP blocked by the sequence-axis short-conv layers in the hybrid stack (no Ulysses wrapper); see [lfm2.md](../models/lfm2.md).
⁷ Laguna — `LagunaAttention` is in neither the Ulysses nor the TP registry, so CP and TP both raise; ETP is mechanically reachable but unvalidated. See [laguna.md](../models/laguna.md).
⁸ Inkling — CP blocked by the sequence-axis short convolutions, TP by the RoPE-free relative-logits attention. See [inkling.md](../models/inkling.md).
⁹ Qwen3-VL — a dense checkpoint ships no `base_model_tp_plan`, so `tp_plan="auto"` shards nothing and the loader raises; see [Supported Models](../models/index.md).

Per-family configs and EP wrapper internals: [Supported Models](../models/index.md).

## Additional guides

- [Data Parallelism (FSDP)](data-parallelism.md) — FSDP strategies, Accelerate configs, mixin internals
- [Expert Tensor Parallelism](expert-tensor-parallelism.md) — shard expert FFN weights for extra memory reduction
- [Large-Scale Scenarios](large-scale-scenarios.md) — pick a layout by NVLink-domain size; start here for any multi-node job
- [Multi-Node Parallelism](multi-node.md) — cross-node EP/CP/TP, EP scopes, RDMA setup
- [Multi-Node Launch Recipes](launch-recipes.md) — runnable torchrun/SLURM launches, EP checkpoint save/load
- [Distributed Data Loading](data-loading.md) — data loading across parallelism modes
- [Checkpoints and Resume](../reference/checkpoints.md) — distributed checkpoint formats (gathered vs sharded) and resume

## When pipeline parallelism is worth it

PP is [not yet available in this release](pipeline-parallelism.md); this section is the forward-looking rationale for why it exists on the roadmap.

Inside one NVLink domain PP earns nothing: FSDP2 + EP/TP/ETP/CP already run every collective on the fabric, and PP only adds bubbles. It is the cross-domain axis, and whether it is worth its constraints depends on how big the domain is.

Across nodes the question is which traffic crosses RDMA. Global-scope EP puts the DeepEP dispatch/combine there, and that is **four** latency-bound all-to-alls per MoE layer per microbatch — dispatch and combine each have a genuine second one in backward (`src/distributed/expert_parallel/autograd.py`). PP would instead keep EP node-local and send `2 × (pp_size − 1)` point-to-point activation tensors per microbatch. On an NVL72 rack the whole EP group rides NVLink and global EP wins outright; on NVL8 hosts with EFA/RDMA, PP + node-local EP is the trade PP will exist to offer. Until it lands, take a large MoE across nodes with global-scope EP, node-local EP+CP/EP+TP/EP+ETP, or multi-group EP — see [Large-Scale Scenarios](large-scale-scenarios.md) and [Capabilities & Limitations at Scale](../reference/scale-and-limitations.md).
