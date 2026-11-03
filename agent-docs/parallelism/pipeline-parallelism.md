# Pipeline Parallelism

PP splits a model's decoder layers into contiguous **stages**, each owning a rank block. It is the
outermost parallelism dimension and the only one designed to cross NVLink domains: the only traffic
leaving a domain is the pipeline's point-to-point boundary activations.

**The expert axes are the only other sharding PP composes with: EP, or pure ETP, never both.**
FSDP2 data parallelism runs inside each stage. Every other pairing is refused by the
[capability matrix](README.md#supported-combinations). The bubble arithmetic and the bandwidth
argument: [GPU Training Theory §9](../reference/gpu-training-theory.md#pipeline-parallelism-the-bubble-and-the-boundary).

## Why PP is the cross-node dimension

Per-step traffic that crosses RDMA, by mode:

| Mode | What crosses | Volume |
|---|---|---|
| FSDP2 1D full-shard | per-shard-group param all-gather + grad reduce-scatter | ~2–3× param bytes |
| HSDP | one gradient all-reduce | ~1× param bytes, plus a param replica per domain |
| EP `ep_scope=global` | DeepEP dispatch/combine | sparse token bytes per MoE layer |
| TP / CP | **forbidden cross-node** | — |
| **PP** | **boundary activations, point-to-point** | `B·S·H` per microbatch per stage edge (`× hc_mult` / `× hc_count` for the hyper-connection families — [cost below](#boundary-cost)) |

When that trade pays, and when it does not (inside one domain, or on an NVL72 rack):
[When pipeline parallelism is worth it](README.md#when-pipeline-parallelism-is-worth-it).

## Launching

Set `pipeline_parallel_size` on a script whose trainer declares `PP_REFUSAL = None`: SFT, SMPO, DPO,
KTO, offline GRPO, Bradley-Terry reward and classification
([Trainer support](#trainer-support)). Any other script refuses it before a model loads with
`<Trainer> does not support Pipeline Parallelism (PP): <its PP_REFUSAL>`. The model's backbone must be
registered ([Model support](#model-support)); any other is refused by name on the meta shell, on every
rank, before a weight is read.

Launch one `torchrun` per node (`halo launch` is single-node). Every stage is a whole number of NVLink
domains holding at least two ranks, so a run spans at least two domains — two nodes at the default
domain sizing. Two recipes ship, both one stage per node on 2 × 8 GPUs:

| Recipe | Shape |
|---|---|
| `examples/sft/qwen3/qwen3-8b-multinode-pp2.yaml` | Qwen3-8B, `pp2`, FSDP2 inside each stage |
| `examples/sft/gptoss/gptoss-20b-multinode-pp2-ep8.yaml` | gpt-oss-20b, `pp2` × `ep8`, `moe_balancing: bias_update` |

A per-node output filesystem leaves each node only its own stage's parts; merge the node
directories with `scripts/after_training/merge_pp_stages.py` before `from_pretrained`
([Per-node export](#per-node-export)).

## Rank layout

PP carves the world first; every other mode runs unchanged inside one stage:

```text
stage_world_size = world_size / pipeline_parallel_size
pp_rank          = global_rank // stage_world_size
```

A **pipeline chain** is the set of ranks holding the same intra-stage position across every stage.
Chain members are `stage_world_size` apart, share a data-parallel coordinate, and consume the same
batch — stage 0 reads `input_ids`, the last stage reads `labels`. The trainer's dataloader shards by
the stage-local DP rank, never the global rank.

**Stage boundaries fall on NVLink-domain boundaries** (`stage_world_size % nvlink_domain_size == 0`,
enforced at config time). Every EP/TP/ETP/CP group is a domain-local rank block, so this keeps them
strictly inside one stage. `pp_size` must therefore divide the domain count: `pp2` on one 8-GPU node
leaves `stage_world_size = 4` and is rejected at the default domain sizing. Single-rank stages are
rejected too (the stage FSDP wrap assumes DP ≥ 2).

## Configuration

The four knobs (`pipeline_parallel_size`, `pipeline_microbatches`, `pipeline_schedule`,
`pipeline_split`), their defaults and constraints:
[Configuration reference → ParallelismConfig](../reference/configuration-reference.md#parallelismconfig).
How they interact with the trainer's batch settings:

- The microbatches **are** the gradient accumulation: an unset `pipeline_microbatches` takes
  `gradient_accumulation_steps`, setting both raises, and the trainer then runs with
  `gradient_accumulation_steps=1`. Under `1f1b` a count below `pp_size` is raised to it, with a warning.
- `per_device_train_batch_size` is the chain's rows per optimizer step, split into the microbatches,
  so it must be a multiple of their count (the recipes run 8 rows in 8 microbatches).
- `data_parallel_size` is divided by `pipeline_parallel_size`: a whole chain consumes one batch.
- A `pipeline_split` sums to the layer count with boundaries on the `layer_types` period
  ([Split-offset constraint](#split-offset-constraint)).
- `max_length` must be set: the pipeline freezes its P2P shapes on the first step and pads every batch
  to it. Offline GRPO fills an unset one from `max_prompt_length + max_completion_length` (both
  required) and rejects a smaller explicit one. A `max_length` at the model's context window is
  warned, since every row then pays full-window compute. The train dataloader runs with
  `drop_last=True`, forced and logged when the config leaves it off.

**Microbatch count sizes the bubble.** Below `4 × pp_size` microbatches the trainer warns: at
`2 × pp_size` the 1F1B bubble `(pp − 1)/(m + pp − 1)` is a fifth of the step at `pp_size = 2` and grows
toward a third with `pp_size`. Plan for `9 × (pp_size − 1)` microbatches to hold it at 10%.

**Raising `pp_size` does not shrink the bubble.** With one row per microbatch (what packing requires)
and no ETP, the count is `m = T × pp_size` for `T = global_batch_tokens / (world_size × max_length)`,
so the 1F1B bubble `(pp − 1)/(m + pp − 1)` *rises* with `pp_size`, toward `1/(1 + T)`. The global
batch you accept sets the bubble, not the stage count: at world 512, `max_length` 8192, `pp8`, a
4.2M-token step is `m = 8` and a **47%** bubble; 10% needs `m = 63`, ~33M tokens per step. Only an
interleaved (virtual-stage) schedule changes this, and none is wired.

**Stage partition.** The default split is **head-weighted**: the last stage carries the `lm_head`
matmul + CE (for Qwen3-8B, ~3 decoder layers' worth of compute), so its layer budget shrinks by that
cost, snapped to the `layer_types` period, and the earlier stages absorb the difference. Any
partition is gradient-exact; only throughput changes. The chosen partition is logged at startup.

## Model support

A stage is the **real backbone** with its layer list sliced, the final norm neutralized on non-last
stages, the input embedding (found through `get_input_embeddings`, with any norm it carries) dropped
off non-first stages, and non-first stages driven through `inputs_embeds`, so each family's rotary
embeddings, mask construction, `layer_types` handling and EP layers are inherited.

Families are registered in `PP_FAMILY_SPECS` (`halo/distributed/pipeline_parallel/families.py`), one
`PPFamilySpec` per family keyed by its backbone class. A spec with `validated=True` is registered: its
stages reproduce the unsplit model on the pipeline GPU tier, and
`tests/cpu/conventions/test_pp_family_roster.py` holds every such family to a GPU row that runs it, a
MoE one also with its experts distributed. A spec with a `refusal` is refused with its reason, and a
backbone no spec registers is refused by name —
`<Backbone> is not registered for pipeline parallelism (PP_FAMILY_SPECS); registered: ...`.
`reject_unsupported_pp_structure` (`model_rules.py`) checks, in order, the decoder-layer list, the
family refusal, the tie gate, the head transform, live MTP layers and the registration, so a tied or
MTP-carrying checkpoint names that reason first.

| Family (backbone) | Verdict | Constraint |
|---|---|---|
| Qwen3 dense (`Qwen3Model`) | registered | period = `layer_types` pattern when sliding is enabled. 0.6B, 1.7B and 4B ship `tie_word_embeddings=True` and are refused by the tie gate; 8B and up are untied |
| Qwen3.5 / 3.6 dense and MoE text towers (`Qwen3_5TextModel`, `Qwen3_5MoeTextModel`) | registered | period 4 (`L,L,L,F`), so `pp_size` ≤ 8 for a 32-layer checkpoint, 10 for 40 layers. `text_only_model: true` for the shipped multimodal checkpoints; packed rows keep their documents apart on the markers each stage rebuilds ([Packing](#packing)); the family has no exportable bias slot and `aux_loss` is severed, so balance with `none` |
| GPT-OSS 20b / 120b (`GptOssModel`) | registered | period 2, so `pp_size` ≤ 12 for 20b (24 layers), 18 for 120b (36 layers). `bias_update` or `none` balancing |
| Qwen3 MoE (`Qwen3MoeModel`) | registered | any split offset (no `layer_types`). The family has no exportable bias slot and `aux_loss` is severed, so balance with `none`: `auto` resolves to `aux_loss` and is refused at the hub's `router_aux_loss_coef: 0.001` |
| GLM-4 MoE Lite (`Glm4MoeLiteModel`) | registered | any split offset; layer 0 is dense. `auto` resolves to `bias_update` on the EP layers (`gate.e_score_correction_bias`). The hub's `num_nextn_predict_layers: 1` is metadata and passes |
| Laguna S / XS 2.1 (`LagunaModel`) | registered | period-4 `full,s,s,s` list, so `pp_size` ≤ 12 for S (48 layers), 10 for XS (40 layers). `auto` resolves to `aux_loss`, which the released `router_aux_loss_coef: 0.0` leaves off; `bias_update` balances and exports through `experts.e_score_correction_bias` |
| Inkling (`InklingTextModel`) | registered | period 6 (5 sliding : 1 full), so `pp_size` ≤ 7 for Inkling-Small (42 layers). `auto` resolves to `bias_update` on the EP layers; packing is refused ([Packing](#packing)). The shipped composite `InklingForConditionalGeneration` trains text-only, as Step-3.7's does |
| Step-3.7 Flash text tower (`Step3p7TextModel`) | registered | period-4 `full,s,s,s` list ([split offsets](#split-offset-constraint)). The family ships only the composite `Step3p7ForConditionalGeneration`; a text-only run of it is admitted (tower and projector re-emitted in every checkpoint), image data is refused. The hub's `num_nextn_predict_layers: 3` is metadata and passes. `auto` resolves to `bias_update` on the EP layers |
| DeepSeek-V4 (`DeepseekV4Model`) | registered | hyper-connection boundary, any split offset. Every `hash_moe` layer (`mlp_layer_types`, the first 3 by default) must sit on stage 0 — its `tid2eid` router reads `input_ids`, which only stage 0 has. `bias_update` or `none` balancing; eager attention only (family) |
| GLM-5 Next text tower (`Glm5NextTextModel`) | registered | hyper-connection boundary, any split offset; a stage may not begin on a `shared` DSA indexer layer (`indexer_types`; GLM-5.3-Flash ships all `full`). The composite `Glm5NextForConditionalGeneration` trains text-only, as Step-3.7's does; packing is refused ([Packing](#packing)). `auto` resolves to `aux_loss` and is refused at a live `router_aux_loss_coef`; `bias_update` balances and exports through `gate.e_score_correction_bias` |
| Qwen3.8-Flash-Next text tower (`Qwen4ExpTextModel`) | registered | hyper-connection boundary, any split offset; later stages take the token ids their PLE layers read ([Token ids on later stages](#token-ids-on-later-stages)), and each stage saves its own n-gram tables ([Checkpoints and resume](#checkpoints-and-resume)). `text_only_model: true`, or the composite text-only; full fine-tuning with `moe_balancing: none` ([Qwen3.8-Flash-Next](../models/qwen4_exp.md#pipeline-parallelism)) |
| Zaya (`ZayaModel`) | refused | two tensors cross every layer boundary — the fp32 residual stream and the accumulating EDA router state (`prev_router_hidden_states`) — a mixed-dtype boundary the single-activation contract does not carry; the loop also selects `layer_types` by list position |
| Gemma 4 (`Gemma4TextModel`) | refused | per-layer embeddings sliced by enumerate position, and the trailing KV-shared layers read an earlier layer's K/V through a forward-threaded dict |
| Any other backbone (Bailing, Mistral 4, LFM-2, Cohere2 MoE, Qwen3-Next, …) | not registered | refused by name |

Models with **live** MTP tail layers are refused: MTP re-embeds `input_ids` mid-backbone, a second
cross-stage stream. Live means built — the decoder list is longer than `num_hidden_layers` — so an MTP
count the native families ship as checkpoint metadata (`num_nextn_predict_layers`,
`mtp_num_hidden_layers`, `num_mtp_layers`; transformers drops those weights at load) passes.

**Hyper-connection families** carry the widened residual between layers: DeepSeek-V4 and GLM-5 Next a
`[batch, seq, hc_mult, hidden]` stream collapsed by `hc_head`, Qwen3.8-Flash-Next a flat
`[batch, seq, hc_count * hidden]` one collapsed by `hyper_connection_mixer` (its only post-layers module, declared
as its final norm). Each spec neutralizes the collapse off the last stage like the norm, and binds a mid-chain
forward that feeds the received stream to the layer loop instead of re-widening it, refusing anything but the
full stream. All three select per-layer inputs by the layer's own attributes (`layer_types_rebase_safe`), so any
split offset is exact.

### Token ids on later stages

A family whose layers re-read the token ids mid-network declares the backbone kwarg it takes them under
(`PPFamilySpec.token_ids_kwarg`): Qwen3.8-Flash-Next's per-layer embeddings (PLE) look their n-grams up from
`ple_input_ids`. Every rank of a chain already holds the batch, so the runner hands the non-first stages of such a
family the microbatch's `input_ids` as a `token_ids` kwarg, split with the rows, and nothing crosses a boundary;
a stage that should take them and gets none raises. DeepSeek-V4 declares none: its `hash_moe` layers stay on
stage 0.

### Boundary cost

One bf16 tensor crosses each stage edge per microbatch per direction (the activation forward, its
gradient back) and nothing else: masks, `position_ids` and `labels` are re-sliced on every rank from
the batch it already holds, and shape inference exchanges meta tensors once.

| Boundary | bytes/token, one direction | 32k-token microbatch, one crossing | NVLink (400 GB/s) | EFA share (50 GB/s) |
|---|---:|---:|---:|---:|
| plain residual, `H=4096` | 8 KiB | 256 MiB | 0.7 ms | 5.4 ms |
| hyper-connection stream, `H=4096, hc_mult=4` (GLM-5.3-Flash, DeepSeek-V4-Flash) | 32 KiB | 1 GiB | 2.7 ms | 21 ms |
| hyper-connection stream, `H=2560, hc_count=4` (Qwen3.8-Flash-Next) | 20 KiB | 640 MiB | 1.7 ms | 13 ms |

Forward plus backward doubles each cell; `1f1b` waits on every P2P batch before the dependent
compute, so those figures are exposed, not overlapped. torch keeps `2 × pipeline_microbatches`
boundary buffers per mid-chain stage for the life of the run: 16 GiB at 8 microbatches of 32k tokens
on the `hc_mult=4` stream (4 GiB for a plain residual). Only `pipeline_parallel_size` changes the
traffic; the cut position does not.

Measured between two B300 nodes over EFA, NCCL send/recv from each rank to its peer on the other node
moves 92.8 GB/s per GPU pair at 256 MiB and 96.2 GB/s at 1 GiB, and keeps that rate with all eight pairs
sending at once (776 GB/s in total). That is about twice the EFA column, so the two crossings above take
2.9 ms and 11.2 ms there.

### Split-offset constraint

GptOss, Qwen3, Qwen3.5/3.6, Step-3.7, Laguna and Inkling select each layer's attention mask by *position in the
layer list*. A stage starting off a whole number of `layer_types` periods would give sliding-window layers a
full causal mask, so `validate_partition` (`partition.py`) raises at split time, on every rank; the
hyper-connection families (`layer_types_rebase_safe`) are exempt. The period is the
smallest shift `p` with `layer_types[p:] == layer_types[:-p]`, so a ragged list (Step-3.7-Flash: 45
layers = 11 whole `full,s,s,s` periods plus one) still has period 4.

That caps the stage count at `pp_size <= num_layers // period`. The default splitter snaps every
boundary to the period and leaves the ragged tail on the last stage (45 layers at period 4, pp2 →
`[24, 21]`); a **manual** `pipeline_split` gets no such help: at 60 layers a hand-written `[30, 30]`
is rejected where the default `[32, 28]` passes.

## PP + EP (MoE)

EP dispatch/combine stays inside each stage — the EP groups carry the stage's rank offset — while
only P2P activations cross stages. This is the cross-node shape for a large MoE.

- **Every stage must carry at least one MoE layer** at `ep_group_size > 1` (EP and PP+ETP): a dense
  stage runs a different collective program than its MoE peers and deadlocks at the gradient norm,
  so such a partition raises at setup. Choose a `pipeline_parallel_size` or `pipeline_split` that
  reaches the MoE layers on every stage.
- The stage FSDP wrap is EP-aware: EP layers are FSDP-ignored (`ep_size>1`) or sharded as
  stage-mesh DTensors (`ep_size==1` + `fsdp_shard_ep1_experts`); `fsdp_shard_ep1_experts: false` is
  rejected under PP.
- EP gradient sync runs in one deferred post-backward sweep (`EPConfig.sweep_reduces_across_replicas`), since
  per-backward hooks would fire on every microbatch and re-scale grads already synced. The sweep is
  off at `ep_size == 1` with the default `fsdp_shard_ep1_experts`, where FSDP2's own reduce-scatter
  syncs the experts. Router and replicated averages reduce over the stage's rank block, never the
  world.
- The gradient norm is exact and rank-uniform: one world all-reduce of each rank's local squared
  norm where every stage parameter is a DTensor; otherwise stage-scoped reduces, then one sum across
  the pipeline chain.
- Router balancing: `bias_update` works on a family whose EP layers export the bias (the counter
  all-reduce is stage-scoped), and `bias_update_transient` is refused; the HF **aux-loss path is
  severed** by the split, so `aux_loss` with a live `router_aux_loss_coef` is rejected at split time,
  the refusal naming `bias_update` only where it exports. `MoEMetricsCallback` is skipped under PP.
- Gradient checkpointing works: the recompute replays each frame's DeepEP dispatch/combine instead of
  issuing a second one, so it composes with the schedule's in-flight microbatches
  ([EP checkpoint scope](expert-parallelism.md#gradient-checkpoint-dispatch-replay)). The DeepEP
  capacity generation is registered on `PipelineStageModule`, so every microbatch sizes its own
  dispatch.
- Equivalence gates: gpt-oss against a bf16 ep1 twin (`tests/gpu/parallelism/pp/test_pp_ep_vs_unsplit.py`,
  both sides on the unfused expert-output sum, since the fused un-permute rounds a rank's local top-k
  sum once), and every other registered MoE family against its unsplit ep1 twin through the real
  trainer at `ep2` (`test_pp_family_vs_reference.py`).

## PP + ETP (MoE)

Pure expert-tensor parallelism (`expert_parallel_size: 1`, `expert_tensor_parallel_size: k`) inside
each stage: every rank holds every expert, sharded `k`-way along the FFN intermediate dim, and
`SumAcrossGroup` sums the partners' outputs in token space. The stage keeps `stage_world_size / k`
data-parallel replicas.

- ETP partners hold shards of one expert and consume the same batch, so they are **not** DP
  replicas: the deferred sweep divides the expert leg by `stage_world_size / expert_tp_size`.
- The ETP all-reduce is model math inside the forward/backward and is not deferred — safe only
  because `ep_size == 1` leaves no DeepEP combine to interleave with. Adding EP on top is rejected.
- The equivalence gate (`tests/gpu/parallelism/pp/test_pp_etp_vs_unsplit.py`) runs in **fp32**: ETP
  reorders the summation producing each expert's output, which in bf16 flips near-tied top-k picks
  with no sync bug present. Save, export (each expert whole, under global layer names) and exact
  resume are covered on GPU (`test_pp_ep_resume.py --mode etp2`) and on CPU
  (`tests/cpu/checkpoint/test_pp_sharded_save_roundtrip.py`).

## Rejected combinations

The first five are refused by the [capability-matrix allowlist](README.md#supported-combinations)
before any rank math; the rest by their own gate.

| Combination | Why |
|---|---|
| PP + TP | transformers reconciles a `replicated_with_grad_allreduce` plan entry (Qwen3 `q_norm`/`k_norm`) with a per-backward all-reduce hook the schedule disables on non-final microbatches, so the hook re-reduces accumulated history. |
| PP + CP | The pipeline loss normalizer is stage-wide with no cancelling `× cp_size`, so every gradient comes out `cp_size×` too small, silently. |
| PP + EP + TP | The deferred expert sweep needs FSDP to shard non-expert params over the EP group, while EP+TP shards them over the 2-D `(dp, tp)` mesh. |
| PP + EP + CP | The deferred sweep's divisor counts every stage rank as a distinct DP replica; CP ranks are not, so every expert gradient comes out `cp_size×` too small. |
| PP + EP + ETP | The expert-TP all-reduce cannot be deferred, so at `ep_size > 1` it interleaves with the DeepEP combine of the same backward in an order not guaranteed to agree across ranks ([PP + ETP](#pp--etp-moe)). |
| PP + HSDP | HSDP's 2-D mesh is built over the whole world and cannot be restricted to a stage's rank block. |
| PP + `fsdp_reshard_after_forward` / `fsdp_reshard_after_backward: false` | The schedule pins each stage unsharded across the backward and drives FSDP's post-backward itself. |
| PP + tied embeddings | Stage 0's embedding and the last stage's head diverge from step 1; refused by the structure gate at load, since config validation cannot see the model. |
| PP + an unregistered or refused family | Refused by the structure gate on the meta shell ([Model support](#model-support)). |
| PP + PEFT / expert LoRA | The adapter save writes stage-local layer indices from one stage, the PP resume restores no adapter, and `layers_to_transform` / `all-linear` resolve against re-based layers. A model passed already PEFT-wrapped (a `PeftModel`, or tuner layers injected in place) is refused at the split as well: the split drops the `PeftModel` (and any prompt encoder on it) and keeps only the tuner layers inside the backbone, so no adapter is saved and the stage parts carry `base_model.model.*` / `.base_layer` names `from_pretrained` cannot load. |
| PP + `save_sharded_ep` | The per-rank partial format carries no stage layer offset, so shards from different stages would collide. |
| PP + `push_to_hub` on per-node output storage across nodes | The Trainer uploads from global rank 0, whose node lacks the other stages' parts. Upload from a shared output directory, or from the [merged](#per-node-export) one. |
| PP + `torch_compile` | The schedule captures the stage module at setup, so a compiled wrapper installed afterwards never runs. |
| PP + reentrant GC | Reentrant checkpoint runs its forward under `no_grad`, so FSDP2 skips pre-backward hook registration. `use_reentrant: false` is supported; `every_n_layers` counts from each stage's first local layer. |
| PP + `optim: muon` / `flash_adamw` | The stage-local optimizer build and the per-stage optimizer shards a resume restores are validated with the AdamW family only. |
| PP + `moe_balancing: bias_update_transient` | The transient side-buffer is a plain attribute the stage split and the PP resume never carry. |
| PP + `lowp_precision` other than `bf16` | `lowp_keep_first_blocks`/`lowp_keep_last_blocks` would apply to every stage's own ends instead of the network's. |
| PP + `init_from_scratch` | The stage loader has no checkpoint to stream, so every rank would build the whole model. |
| PP + quantized (QLoRA) loading | The stage loader materializes plain weights, never bitsandbytes `Params4bit`, and the stage's FSDP2 wrap cannot shard `Params4bit` either. The adapter QLoRA trains is PEFT, refused above. |
| PP + `accelerate launch` | Accelerate owns the FSDP/DDP wrap the pipeline replaces. Launch with `torchrun`. |
| PP + padding-free collators | Their flattened width varies per batch while the P2P buffers are frozen, and the single row cannot be microbatched. Use [packing](#packing). |
| PP + `activation_offloading` | TRL implements it by wrapping `training_step`, which the PP schedule never calls. |
| PP + image-bearing VLM training | The vision tower and projector sit in no stage. Refused on image evidence (a dataset image column, embedded image parts, or a collator declaring image columns); a text-only run of a composite checkpoint is admitted and re-emits the tower unchanged, and `text_only_model: true` drops it from the build. |
| PP + layer-indexed `freeze_layers_patterns` / `unfreeze_layers_patterns` | Each stage's layers are re-based to index 0, so a global index selects the wrong layer ([Configuration](../reference/configuration-reference.md#commonscriptarguments)). |
| PP + a stage that is not a whole number of NVLink domains, or a one-rank stage | Every intra-stage group must stay inside one stage, and the stage FSDP wrap assumes DP ≥ 2 ([Rank layout](#rank-layout)). |
| PP + `fsdp_defer_grad_sync` or `fsdp_shard_ep1_experts: false` | The schedule already reduces once per step; with ep1 experts unsharded, a MoE stage and a dense stage run different collective programs and deadlock at the gradient norm ([PP + EP](#pp--ep-moe)). |
| PP-only knobs at `pipeline_parallel_size: 1`; `pipeline_microbatches` with `gradient_accumulation_steps > 1`; a `per_device_train_batch_size` the microbatch count does not divide; an unset `max_length` | Nothing reads the knobs there; the microbatches are the accumulation and split the per-device batch; the P2P shapes freeze at `max_length` ([Configuration](#configuration)). |
| PP + live MTP tail layers, or a head transform the last stage cannot reproduce | MTP re-embeds `input_ids` mid-backbone ([Model support](#model-support)); the last stage applies the family's head path itself ([Fused head](#fused-head-and-loss-on-the-last-stage)). |
| PP + `aux_loss` with a live `router_aux_loss_coef`, or a stage with no MoE layer at `ep_group_size > 1` | The split severs the aux-loss path; a dense stage runs a different collective program than its MoE peers and deadlocks ([PP + EP](#pp--ep-moe)). |
| PP + `compute_metrics` on DPO, KTO or offline GRPO, or an eval batch size other than the train one | No pipeline reproduces their prediction convention; the P2P shapes are frozen ([Evaluation](#evaluation)). |
| PP + DPO/KTO with positional `model`/`args`, a live `ref_model`, or reference log-probs not precomputed | The PP reference contract is precompute-only ([Trainer support](#trainer-support)). |
| PP + `packing` on a model whose conv / linear-attention layers read no boundary at all (GLM-5 Next and Inkling in the roster), on GatedDeltaNet layers without their segment-aware kernels, or with more than one packed row per microbatch | No marker reaches those layers, the torch fallbacks drop the markers, and FlashAttention isolates documents only at batch size 1 ([Packing](#packing)). |

## Fused head and loss on the last stage

The last stage folds `lm_head` and cross-entropy into its forward and returns the summed loss, so the
`[microbatch, seq, vocab]` logits plane and its gradient are never materialized. The head runs over
token chunks of `max(1, 128M // vocab_size)` rows (`logit_chunk_rows`), each under a non-reentrant
checkpoint. At `V=201088`, `S=8192` and one microbatch row that drops 3.29 GB of bf16 logits plus as
much logit gradient; the loss and activation gradients are unchanged, while the head weight's gradient
is reassociated per chunk (~7e-7 relative in fp32, ~1e-2 in bf16).

On by default; `HALO_PP_FUSED_HEAD_LOSS=0` restores the logits path. `PipelineLossAdapter.stage_loss`
fuses only the causal-LM CE with raw-logit predictions (the default `causal_lm_token_loss`, no
`predictions_fn`) — **SFT alone**; every other trainer declares its own token loss. The forward-only
pass that serves `compute_metrics` un-fuses and returns real logits. The gate
(`tests/gpu/parallelism/pp/test_pp_fused_head_loss.py`, pp2×dp2, tiny Qwen3, fp32, seq 1024, vocab
65536, with and without GC) holds loss within `1e-6` relative and every gradient within `1e-5`, at
last-stage peak memory **1045.9 → 470.9 MiB**.

The last stage applies what the family's `*ForCausalLM.forward` does around `lm_head` — Cohere's
`logit_scale`, Gemma's softcap, Granite's division — to the logits it returns and inside the fused
loss: the transform the chunked GRPO log-probs apply, verified against the family's forward
(`halo/models/head_transform.py`). A family whose forward it does not reproduce is refused on every
rank.

## Attention, masks, and checkpointing

- **Padding masks are exact.** Mask construction reads only `inputs_embeds`, the 2D mask and
  `position_ids`, so a mid-stage forward builds a bitwise identical mask (equivalence-gated on ragged
  right-padded batches under eager, SDPA and FA4 varlen).
- **FA4** kernels are pre-warmed by the loader before the split, so no mid-schedule JIT stall.
- **Liger `cross_entropy` and `fused_linear_cross_entropy` are force-disabled** under PP: a stage
  forward runs without labels, so FLCE would claim a saving it does not deliver, and `cross_entropy`
  rebinds `F.cross_entropy` process-wide under the last-stage loss. The stage loss chunks its own
  fp32 CE ([Liger](../optimization/liger-kernels.md)).
- **Gradient checkpointing** (non-reentrant) is grad-exact: the per-microbatch phase holds FSDP's
  reduction off, so recompute cannot trigger a premature reduce-scatter.
- **The stage holds its unsharded gradient across the microbatch loop.** The schedule reduce-scatters
  once per step, so each rank keeps a full gradient of its stage's FSDP-managed parameters until
  then: 2 B/param in bf16, 4 B/param under `fp32_grad_reduce` or `fp32_non_ep_params` (warned at
  config time).
- **Turn GC off where memory allows**, dense or with EP. A microbatch's activations are already `1/m`
  of a step. TRL's `SFTConfig` turns `gradient_checkpointing` on by default; both shipped recipes set
  it off ([Measured throughput](#measured-throughput)).

## Measured throughput

B300, packed `HuggingFaceH4/ultrachat_200k`, tokens per second per GPU over the steady steps. Every
gpt-oss row runs grouped GEMM and `moe_balancing: bias_update`.

**One 8-GPU host**, `max_length` 8192, 524k tokens (64 rows) per step, steps 6–15; peaks are the
allocator's maximum over ranks, allocated / reserved. A single `torchrun` refuses an NVLink domain
smaller than its node (`nvlink_domain_size` must be a multiple of `gpus_per_node`), so the `pp2` rows run
two `torchrun` agents in one container, `--nnodes=2 --nproc_per_node=4` with `--node_rank=0` on
`CUDA_VISIBLE_DEVICES=0,1,2,3` and `--node_rank=1` on `4,5,6,7`: each agent is a 4-GPU node, and each
half is one domain and one stage. The boundary between the halves stays inside the host, so these rows
price the schedule and the narrower groups, not an inter-node link. The `dp8` and `ep8` rows run one
agent on the host's 8-GPU domain.

| Model | Layout | GC | tok/s/GPU | Peak GiB |
|---|---|:-:|---:|---:|
| Qwen3-8B | `dp8` | on | 17,062 | 33.0 / 36.6 |
| Qwen3-8B | `pp2 × dp4`, m=16 | on | 16,291 | 27.9 / 33.1 |
| Qwen3-8B | `pp2 × dp4`, m=16 | off | **19,445** | 66.3 / 71.6 |
| gpt-oss-20b | `ep8` | off | 20,066 | 63.1 / 70.4 |
| gpt-oss-20b | `pp2 × ep4`, m=16 | on | 17,398 | 35.3 / 48.4 |
| gpt-oss-20b | `pp2 × ep4`, m=16 | off | **22,223** | 62.6 / 73.9 |

- Dense: `pp2 × dp4` without GC runs 14% faster than `dp8` with GC, at twice its peak, and 19% faster
  than itself with GC. At m=16 the bubble is 1/17 of the step.
- PP + EP: `pp2 × ep4` without GC runs 11% faster than `ep8` at about the same peak, and 28% faster
  than itself with GC: halving the dispatch group cuts the DeepEP cost by more than the 1/17 bubble
  adds.

**Two nodes**, 2 × 8 B300 over EFA, one stage per node, the shipped recipes at their own
`max_length` and m=8 (a 1/9 bubble); median step of a 10-step run, rank 0's allocated peak:

| Recipe | GC | s/step | tok/s/GPU | Peak GiB |
|---|:-:|---:|---:|---:|
| `qwen3-8b-multinode-pp2.yaml` (`max_length` 4096) | on | 1.090 | 14,970 | 22.1 |
| | off | 0.920 | **17,740** | 41.3 |
| `gptoss-20b-multinode-pp2-ep8.yaml` (`max_length` 8192) | on | 2.565 | 12,755 | 24.2 |
| | off | 2.115 | **15,500** | 45.2 |

GC off raises throughput 18.5% (Qwen3-8B) and 21.5% (gpt-oss-20b) for 1.9× the allocated peak. The
whole-device peaks are 49.2 GiB on stage 0 and 35.7 GiB on stage 1 for Qwen3-8B, 70.4 and 40.8 GiB for
gpt-oss-20b. On the single host the same recipes, gpt-oss at `ep4` since a 4-GPU stage cannot hold
`ep8`, gain about 25–30%.

## Packing

`packing: true` composes with PP and is the way to avoid full-window compute on short rows: PP pads
every batch to `max_length`. Each stage rebuilds the segment markers a model's conv / linear-attention layers
read (`segment_markers_for`: `seq_idx` for GatedDeltaNet and LFM-2 ShortConv, `cu_seq_lens` for the delta rule)
from its microbatch's `position_ids`, as the packing collator builds them for a flattened row, so Qwen3.5 / 3.6 and
Qwen3.8-Flash-Next keep each document of a packed row apart; GatedDeltaNet needs its segment-aware kernels for it,
as off PP. Packing is refused, whatever `allow_cross_document_linear_attention` says, where the layers read no
boundary at all (`find_unmarked_sequence_mixer`: KDA, Lightning Attention-2, Inkling's short convolutions), whose
state would carry from each document into the next. In the roster that is GLM-5 Next and Inkling, whose padded
rows hold one document each and still run; DeepSeek-V4 refuses multi-document rows on every path
([Document isolation](../data/collators.md#document-isolation-under-packing)).
`reject_pipeline_packed_rows` (`halo/models/segment_markers.py`) holds the rule; the collator factory
applies it when PP keeps the packed rows, and `reject_pp_collator` again to the trainer's final collator,
whatever built it.

Under PP the packing collator keeps its `[B, L]` rows instead of flattening to `[1, B*L]`: the
schedule splits dim 0 into microbatches, so **the microbatch split is the flattening**, and each
microbatch reaches the stage as `[1, max_length]`. One requirement, enforced at setup:

```yaml
packing: true
per_device_train_batch_size: 4    # must equal the microbatch count
pipeline_microbatches: 4
```

A microbatch carrying more than one packed row leaks attention across documents under FlashAttention,
which derives `cu_seqlens` from `position_ids` only at batch size 1 (~1.9 nats of logprob error); the
gate does not vary by backend ([Document isolation](../data/collators.md#document-isolation-under-packing)).
The trailing pad region is inert: its positions restart at 0 every `PAD_TAIL_SEGMENT_CHUNK` (256) tokens,
so it forms a few segments of its own with ignored labels.

Each stage builds those markers once per microbatch from one host read of the row
(`PipelineStageModule._segment_kwargs` in `halo/distributed/pipeline_parallel/stage.py`), and on a flash backend
the varlen set with them: `cu_seq_lens_q/k` and the longest segment as a host-int `max_length_q/k`. The schedule
splits only batch-dim tensors, so the markers the packing collator builds for a flattened row cannot ride in with
the batch; without the varlen set every attention layer derives it from `position_ids` again, three host syncs
per layer and again in the recompute. A masked or one-document microbatch gets none, nor does a non-flash
backend on a family without markers.

## Per-step metrics

Only the last stage computes the loss, while HF logs from global rank 0 on the first stage, so the
trainer broadcasts the last stage's per-step scalars down the chain (one fp32 vector per step,
`PipelineLossAdapter.metrics_fn`); every chain rank logs the same values.

## Evaluation

Evaluation runs through the schedule on every PP trainer: `eval_loss` reports normally and every rank
logs the identical broadcast value. Without `compute_metrics` the schedule computes its own
microbatched loss; with it, one forward-only pass yields the last stage's merged output, from which
both the loss and the predictions derive. A forward-only drive skips the schedule's trailing gradient
reduction, so evaluation does not reshard the stage per batch.

`compute_metrics` is served where a pipeline can reproduce the trainer's prediction convention and
**rejected at construction** elsewhere:

| Trainer | Under PP | Prediction |
|---|---|---|
| SFT | served | whatever `preprocess_logits_for_metrics` returns, hoisted onto the last stage (required) |
| Classification | served | pooled `[B, num_labels]` |
| Reward (BT) | served | `[pairs, 2]` chosen/rejected scores |
| SMPO | served | `[pairs, 2]` chosen/rejected rewards |
| DPO | rejected | TRL hands metrics the raw plane of the `[chosen ⧺ rejected]` concat; PP rows are interleaved |
| KTO | rejected | TRL's convention is the raw plane, `[rows, max_length, vocab]` broadcast to every chain rank |
| Offline GRPO | rejected | its `prediction_step` returns no predictions in any mode |

- `per_device_eval_batch_size == per_device_train_batch_size` whenever the run evaluates, since the
  P2P shapes are frozen. Partial final batches are row-padded with all-ignore rows contributing 0,
  then trimmed before scoring.
- SFT's `preprocess_logits_for_metrics` runs on the last stage before the broadcast, so it must be
  row-independent (argmax and top-k are); SFT `compute_metrics` without it is refused.
- The evaluation gather is scoped to the data-parallel replicas, since every chain rank returns the
  same predictions.
- DPO and KTO require their precomputed reference columns in the eval dataset too.

## Loading

Loading is **stage-aware**: each rank builds the whole model shell on the meta device, plans the
checkpoint against that key space, keeps only the plans for its stage's decoder layers — re-basing
`layers.5` on disk to `layers.1` on a stage starting at layer 4 — and streams exactly those tensors
from safetensors (`load_pp_stage_model`, `halo/model_loading/parallel/pp.py`, narrowing the
whole-model plan through `StageKeyFilter`). Peak
per-rank load memory is

```text
P_ne_layers / pp_size + embeddings + final norm + head        (+ the EP expert slice / ep_size)
```

Embeddings, the final norm and the head are materialized on every stage and dropped by
`build_pipeline_stage` right after. PP+EP composes as a filter: the PP layer filter narrows the EP
planner's output.

The structure gate runs on the meta shell before any weight is read and before the loader's first
cross-rank verdict, so an unregistered family, a refused one or a tied checkpoint fails every rank
alike. The coverage gate raises if any parameter or persistent buffer is left on the meta device, or
a re-based key names nothing on the sliced stage; both verdicts are stage-local, so they are joined
and raised on every rank. State the checkpoint does not carry (a `score` head on a base LM) follows
the EP lazy loader's random-init policy ([EP → Model loading](expert-parallelism.md#model-loading)).

The stage loader reads configured FP32 masters (`fp32_non_ep_params`, `fp32_router`, `fp32_experts`)
at FP32 ([Load precision](../models/README.md#load-precision)). PP **requires a safetensors
checkpoint**; `init_from_scratch`, QLoRA loading and `lowp_precision != bf16` are rejected. The stage
load runs on every rank at once; `max_concurrent_loading` does not apply
([Multi-Node → Model loading](multi-node.md#model-loading)).

## Checkpoints and resume

`save_pp_checkpoint` (`halo/checkpoint/parallel/pp.py`) writes one complete-tensor part set per stage
under the unsplit model's hub names plus a merged standard index, so a directory on a shared output
filesystem loads with plain `from_pretrained` and serves like an EP-gathered save. The layout, the
name revert and the index exchange: [Checkpoints → What lands on disk](../reference/checkpoints.md#what-lands-on-disk--four-layouts).
The input embedding, with any norm it carries, lives on stage 0 alone, so no two stages write it. A stage holding
an external table (Qwen3.8-Flash-Next's n-gram tables) writes it beside its parts under the unsplit model's names,
in files under its own part prefix, and the stage-aware loader attaches it by those names, so a resume reads each
stage's table from its own files.

The **writer set** follows the output filesystem (`DIST_OUTPUT_SHARED_FILESYSTEM`, falling back to
the `DIST_SHARED_FILESYSTEM` umbrella):

| Artifact | Shared output FS | Per-node output FS |
|---|---|---|
| Stage parts `model-ppSSSSS-of-PPPPP-*.safetensors`, with the stage's [external tables](../reference/checkpoints.md#external-tables) as `model-ppSSSSS-of-PPPPP-tableNN-*` | one rank per stage (`stage_local_rank == 0`) | one rank per stage per **node** (`is_pp_shard_writer`) |
| Index, `config.json`, tokenizer, a text-only run's untouched vision tower (`model-wrapper-*`) | global rank 0 | one rank per node; the index is the full one |
| `optimizer_meta.pt`, `scheduler.pt`, `trainer_state.json`, `router_balancing_biases.pt`, offline GRPO's `reference_logps.pt` | global rank 0 | one rank per node, whole |
| `optimizer_shard_<rank>.pt`, `rng_state_<rank>.pth` | every rank | every rank, on its own node |

Because the parts carry the unsplit model's names, the **weights are topology-independent**: a
complete directory resumes onto a different `pp_size`. The **optimizer** shards are stage-local:
their restore is gated on the topology fingerprint and `pp_stage_partition` (both in
`optimizer_meta.pt`), and any drift raises. PP+EP loads its expert weights only at construction, so
resume through the training scripts, which repoint the load at the checkpoint
([Resume by parallelism mode](../reference/checkpoints.md#resume-by-parallelism-mode)).

Every PP trainer's save, resume and export runs on GPU through the real trainer, on one host. SFT
resumes in `tests/gpu/parallelism/pp/test_pp_resume.py` (dense `pp2`), exports in
`test_pp_sft_trainer_e2e.py`, and does both with experts in `test_pp_ep_resume.py` (gpt-oss `pp2 × ep2`
and `pp2 × etp2`); on per-node storage its checkpoint is also merged with `merge_pp_stages.py` and resumed
from the merge, and the Qwen3-8B recipe saved, resumed, merged and loaded with `from_pretrained` on two
nodes over EFA ([Verified topologies](multi-node.md#verified-topologies)). The other trainers run in
`test_pp_trainer_checkpoints.py`, one row per trainer and two for offline GRPO (with and without the KL
reference that rides each checkpoint as `reference_logps.pt`); each resume there restores the optimizer
bit for bit and replays the unbroken run.

### Per-node export

On per-node output storage each node's directory holds the full index but only the parts of the
stages it ran, plus the wrapper parts, and only its own ranks' optimizer shards and RNG states. It
resumes only the topology and rank placement that wrote it; a node asked for another stage's layers
fails on every rank, naming the placement, and every after-training tool refuses the directory, naming
its absent parts. To export, or to resume from one directory, copy each node's directory onto one host
and merge:

```bash
python scripts/after_training/merge_pp_stages.py \
    --input_dir /copies/node0/checkpoint-500 /copies/node1/checkpoint-500 \
    --output_dir /models/merged
```

The inputs must carry identical indexes. Each part, a stage's table files included, comes from the first input
holding it; every
input's optimizer shards, RNG and prefetch states come along with one `optimizer_meta.pt`, hard-linked
when the output is on their filesystem (so the optimizer state is not stored twice) and copied with
their mode otherwise, and the config, tokenizer, trainer state and scheduler come from the first input.
A part no input holds, metas that differ, a per-rank file held at two sizes or a rank whose shard no
input holds is refused before the first write; the index lands last. The merged directory loads with
`from_pretrained`, uploads like a shared-filesystem save, and resumes exactly on the save's topology.
A linked file outlives its input's deletion, and a later training save into either directory replaces it
rather than writing through the link
([Checkpoints & Resume](../reference/checkpoints.md#expert-parallelism-ep-eptp-epcp)).

## Constraints inherited from `torch.distributed.pipelining`

- **Fixed shapes.** Input shapes freeze on the first step; a later shape raises
  `PipeliningShapeError`. The trainer pads every batch to `max_length`.
- **Bare tensors only.** Every stage returns a tensor. Per-example side tensors (advantages,
  reference log-probs, pool positions) ride the runner's dict-target side channel, split in lockstep
  with the rows.
- **Microbatch invariance.** Per-microbatch losses must **sum** to the full-batch loss, so a
  whole-batch denominator is passed as a step-level normalizer.
- `PipelineRunner` drives torch's **private** FSDP2 and schedule members, so construction asserts
  their presence: a torch upgrade that renames one raises immediately, naming PP as needing
  re-validation.

## Trainer support

PP is declare-to-enable per trainer (like `SUPPORTS_CP`): `PP_REFUSAL` is the mechanism a refusing
trainer's error names, `None` where PP is supported. An enabled trainer declares its
pipeline-loss contract as a `PipelineLossAdapter` (`_pp_loss_adapter()`,
`halo/trainers/parallelism/pipeline_loss_adapter.py`). `_configure_pipeline_parallel` calls
`_pp_loss_adapter()` once, before anything reads the adapter, so an adapter builder may seed the
per-step state its loss closes over.

The refusals are pure functions in `halo/trainers/parallelism/pipeline_rules.py`. The training-argument,
PEFT and multimodal gates run before the stage split, with each trainer's own
`_reject_unsupported_pp_options` (DPO and KTO: keyword `model`/`args`, no live `ref_model`, the
precompute-only reference, `compute_metrics`); the collator, pad-token and EP-coverage gates run after
`super().__init__`. The rank-local verdicts (image evidence, a stage with no EP layer) are joined
across ranks.

Which trainers enable PP: [Trainer Compatibility](../reference/trainer-architecture.md#trainer-compatibility).
The enabled trainers' pipeline-loss contracts:

| Trainer | Contract notes |
|---|---|
| SFT | canonical causal-LM |
| SMPO | interleaved pairs; percentile clips stay rejected (batch-global quantiles are not microbatch-decomposable) |
| Reward (BT) | interleaved pairs over scores pooled in the loss; needs `num_labels: 1`, and a `margin` column in the train dataset or in no batch |
| Classification | every loss variant; single-label needs `num_labels >= 2`; `predictions_fn` broadcasts pooled `[B, num_labels]` |
| DPO | precompute-only: reference log-probs as dataset columns; limits: [DPO](../training-methods/preference/dpo.md) |
| KTO | `apo_zero_unpaired` only: KL-free, per-row decomposable, precompute-only; the default `kto` loss needs a world-global KL baseline |
| Offline GRPO | all three loss types and both PG formulations; `max_prompt_length` + `max_completion_length` required. `kl_beta > 0` scores the reference once at construction by a forward-only sweep through the pipeline and ships it per completion token as a dataset column (sized, indexable, unsharded datasets); the scores ride every checkpoint (`reference_logps.pt`) |

A refusing trainer's error names its `PP_REFUSAL`: embedding (the in-batch similarity matrix *is* the
loss), SDPG (a second, differently-shaped same-model forward), online and async GRPO (the weight sync
sends the full `named_parameters()` from one rank set, which no PP rank holds), teacher and self
distillation (a second forward per microbatch through a distinct frozen teacher).

## Code map

`halo/distributed/pipeline_parallel/`: `families.py` (the family specs and the hyper-connection
stream forwards), `model_rules.py` (the structure and run-settings gates), `partition.py` (the split
and `validate_partition`), `stage.py` (`build_pipeline_stage` and the stage↔global naming),
`stage_losses.py`, `groups.py` (chain and stage groups) and `runner.py` (`PipelineRunner` over torch's
`Schedule1F1B` / `ScheduleGPipe`). The stage-aware loader is `halo/model_loading/parallel/pp.py`; the
trainer side is `PipelineTrainerMixin` (`halo/trainers/parallelism/pipeline_trainer.py`, inert at
`pp_size == 1`), `pipeline_rules.py`, `pipeline_loss_adapter.py` and the stage clip in
`grad_clipping.py`.
