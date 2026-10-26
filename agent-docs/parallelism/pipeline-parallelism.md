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
| FSDP2 1D full-shard | per-layer param all-gather + grad reduce-scatter | ~2–3× param bytes |
| HSDP | one gradient all-reduce | ~1× param bytes, plus a param replica per domain |
| EP `ep_scope=global` | DeepEP dispatch/combine | sparse token bytes per MoE layer |
| TP / CP | **forbidden cross-node** | — |
| **PP** | **boundary activations, point-to-point** | `B·S·H` per microbatch per stage edge (`× hc_mult` for the hyper-connection families — [cost below](#boundary-cost)) |

Within one NVLink domain PP earns nothing — FSDP2 plus node-local EP/TP/ETP already dominate. Its
case is a model that cannot live in one domain, or a job wide enough that cross-node DP bandwidth is
the wall.

## Launching

Set `pipeline_parallel_size` on a script whose trainer declares `PP_REFUSAL = None`: SFT, SMPO, DPO,
KTO, offline GRPO, Bradley-Terry reward and classification ([Trainer support](#trainer-support)).
Any other script refuses it at config time with `<Trainer> does not support Pipeline Parallelism
(PP)`, before a model loads. The model's backbone must be registered
([Model support](#model-support)); any other is refused by name on the meta shell, on every rank,
before a weight is read.

Launch with `torchrun`, one per node. Every stage is a whole number of NVLink domains holding at
least two ranks, so a run spans at least two domains — two nodes at the default domain sizing. Two
recipes ship, both one stage per node on 2 × 8 GPUs:

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

| Field | Default | Meaning |
|---|---|---|
| `pipeline_parallel_size` | `1` | Number of stages. Must divide `world_size`; each stage owns whole NVLink domains. |
| `pipeline_microbatches` | `0` (auto = `gradient_accumulation_steps`) | Microbatches per optimizer step. The microbatches **are** the gradient accumulation, so setting both this and `gradient_accumulation_steps > 1` is rejected, and the count must divide `per_device_train_batch_size`. |
| `pipeline_schedule` | `1f1b` | `1f1b` (lower activation memory; fewer than `pp_size` microbatches are raised to `pp_size` with a warning) or `gpipe`. |
| `pipeline_split` | `null` | Manual per-stage decoder-layer counts, e.g. `[20, 16]`. Must sum to the layer count, boundaries on the `layer_types` period. |

`data_parallel_size` is divided by `pipeline_parallel_size`: a whole chain consumes one batch.
Setting any of `pipeline_split`, `pipeline_microbatches` or `pipeline_schedule` at
`pipeline_parallel_size=1` is rejected: nothing reads them there.

`max_length` must be set. The pipeline freezes its P2P shapes on the first step and pads every batch
to it, so an unset value is rejected at construction. Offline GRPO fills an unset `max_length` from
`max_prompt_length + max_completion_length` (both required) and rejects a smaller explicit one. For
the same frozen shapes the train dataloader runs with `drop_last=True`, forced and logged when the
config leaves it off; a short final batch would crash mid-epoch.

**Microbatch count sizes the bubble.** Below `4 × pp_size` microbatches the trainer warns: at
`2 × pp_size` the 1F1B bubble measured ~22% of the step at `pp_size = 2` (B300, two simulated NVLink
domains on one host), and the idle share grows with `pp_size`. Plan for `9 × (pp_size − 1)`
microbatches to hold it at 10%.

**Raising `pp_size` does not shrink the bubble.** With no ETP in the stage, `data_parallel_size ==
stage_world_size`. With one row per microbatch (what packing requires) the count is
`m = T × pp_size`, for `T = global_batch_tokens / (world_size × max_length)`, so the 1F1B bubble
`(pp − 1)/(m + pp − 1)` *rises* with `pp_size`, toward a ceiling of `1/(1 + T)`. What sets the bubble
is the global batch you accept, not the stage count. At world 512, `max_length` 8192, `pp8`: a
4.2M-token step is `m = 8` and a **47%** bubble; 10% needs `m = 63`, ~33M tokens per step. Only an
interleaved (virtual-stage) schedule changes this arithmetic, and none is wired.

**Stage partition.** The default split is **head-weighted**: the last stage carries the `lm_head`
matmul + CE (for Qwen3-8B, ~3 decoder layers' worth of compute), so its layer budget shrinks by the
head's layer-equivalent cost, snapped to the model's `layer_types` period, and the earlier stages
absorb the difference. Any partition is gradient-exact; only throughput changes. The chosen partition
is logged at startup.

## Model support

A stage is the **real backbone** with its layer list sliced, the final norm neutralized on non-last
stages, and non-first stages driven through `inputs_embeds`, so each family's rotary embeddings, mask
construction, `layer_types` handling and MoE wrappers are inherited. The input embedding is located
by identity (`get_input_embeddings`).

Families are registered in `PP_FAMILY_SPECS` (`src/distributed/pipeline_parallel/families.py`), one
`PPFamilySpec` per family keyed by its backbone class. A spec with `validated=True` is registered: its
stages reproduce the unsplit model on the pipeline GPU tier. A spec with a `refusal` is refused with
its reason. A backbone no spec registers is refused by name —
`<Backbone> is not registered for pipeline parallelism (PP_FAMILY_SPECS); registered: ...`.
`reject_unsupported_pp_structure` (`model_rules.py`) runs, in order: the decoder-layer probe, the
family refusal, the tie gate, the head-transform check, the live-MTP gate, then the registration
verdict, so a tied or MTP-carrying checkpoint names that reason first.

| Family (backbone) | Verdict | Constraint |
|---|---|---|
| Qwen3 dense (`Qwen3Model`) | registered | period = `layer_types` pattern when sliding is enabled. 0.6B, 1.7B and 4B ship `tie_word_embeddings=True` and are refused by the tie gate; 8B and up are untied |
| Qwen3.5 / 3.6 dense and MoE text towers (`Qwen3_5TextModel`, `Qwen3_5MoeTextModel`) | registered | period 4 (`L,L,L,F`), so `pp_size` ≤ 8 for a 32-layer checkpoint, 10 for 40 layers. `text_only_model: true` for the shipped multimodal checkpoints; packing is refused ([Packing](#packing)); the family has no exportable bias slot and `aux_loss` is severed, so balance with `none` |
| GPT-OSS 20b / 120b (`GptOssModel`) | registered | period 2, so `pp_size` ≤ 12 for 20b (24 layers), 18 for 120b (36 layers). `bias_update` or `none` balancing |
| Step-3.7 Flash text tower (`Step3p7TextModel`) | registered | period-4 `full,s,s,s` list ([split offsets](#split-offset-constraint)). The family ships only the composite `Step3p7ForConditionalGeneration`; a text-only run of it is admitted (tower and projector re-emitted in every checkpoint), image data is refused. The hub's `num_nextn_predict_layers: 3` is metadata and passes |
| DeepSeek-V4 (`DeepseekV4Model`) | registered | hyper-connection boundary, any split offset. Every `hash_moe` layer (`mlp_layer_types`, the first 3 by default) must sit on stage 0 — its `tid2eid` router reads `input_ids`, which only stage 0 has. `bias_update` or `none` balancing; eager attention only (family) |
| GLM-5 Next text tower (`Glm5NextTextModel`) | registered | hyper-connection boundary, any split offset; a stage may not begin on a `shared` DSA indexer layer (`indexer_types`; GLM-5.3-Flash ships all `full`). The composite `Glm5NextForConditionalGeneration` trains text-only, as Step-3.7's does |
| Zaya (`ZayaModel`) | refused | two tensors cross every layer boundary — the fp32 residual stream and the accumulating EDA router state (`prev_router_hidden_states`) — a mixed-dtype boundary the single-activation contract does not carry; the loop also selects `layer_types` by list position |
| Gemma 4 (`Gemma4TextModel`) | refused | per-layer embeddings sliced by enumerate position, and the trailing KV-shared layers read an earlier layer's K/V through a forward-threaded dict |
| Inkling (`InklingTextModel`) | refused | loader-only validation: no schedule, end-to-end or save/resume row. Its embedding module carries `embed_norm`, so the norm leaves later stages with it |
| Qwen3 MoE, GLM-4 MoE Lite, Bailing/Ling, Mistral4, Laguna, LFM-2, Cohere2 MoE, any other | not registered | refused by name. LFM-2 and the shipped Cohere2 MoE checkpoints also tie their head (tie gate) |

Models with **live** MTP tail layers are refused: MTP re-embeds `input_ids` mid-backbone, a second
cross-stage stream. Live means built — the decoder list is longer than `num_hidden_layers`. All three
config spellings count (`num_nextn_predict_layers`, `mtp_num_hidden_layers`, `num_mtp_layers`, read
off `config.get_text_config()`); a field the native families ship as checkpoint metadata
(transformers drops the MTP weights at load) passes.

**Hyper-connection families** (DeepSeek-V4, GLM-5 Next) carry the `hc_mult`-widened
`[batch, seq, hc_mult, hidden]` stream between layers, widened from the embedding before layer 0 and
collapsed by `hc_head` after the last. Their spec declares `hc_head` a tail module (neutralized off
the last stage like the norm) and binds a mid-chain forward (`deepseek_v4_stream_forward`,
`glm5_next_stream_forward`) that feeds the received stream to the layer loop instead of re-widening
it; stage 0 keeps the real forward. Both families select per-layer inputs by the layer's own
attributes (`layer_types_rebase_safe`), so any split offset is exact.

### Boundary cost

One bf16 tensor crosses each stage edge per microbatch per direction (the activation forward, its
gradient back) and nothing else: masks, `position_ids` and `labels` are re-sliced on every rank from
the batch it already holds, `use_cache` is off, and shape inference exchanges meta tensors once. The
received buffer goes straight into the first layer and the last layer's output is sent as-is.

| Boundary | bytes/token, one direction | 32k-token microbatch, one crossing | NVLink (400 GB/s) | EFA share (50 GB/s) |
|---|---:|---:|---:|---:|
| plain residual, `H=4096` | 8 KiB | 256 MiB | 0.7 ms | 5.4 ms |
| hyper-connection stream, `H=4096, hc_mult=4` (GLM-5.3-Flash, DeepSeek-V4-Flash) | 32 KiB | 1 GiB | 2.7 ms | 21 ms |

Forward plus backward doubles each cell; `1f1b` waits on every P2P batch before the dependent
compute, so those figures are exposed, not overlapped. torch keeps `2 × pipeline_microbatches`
boundary buffers per mid-chain stage for the life of the run: 16 GiB at 8 microbatches of 32k tokens
on the `hc_mult=4` stream (4 GiB for a plain residual). Only `pipeline_parallel_size` changes the
traffic; the cut position does not.

### Split-offset constraint

GptOss, Qwen3, Qwen3.5, LFM-2, Laguna and Step-3.7 select each layer's attention mask by *position in
the layer list*. A stage starting off a whole number of `layer_types` periods would give
sliding-window layers a full causal mask, so `validate_partition` (`partition.py`) raises at split
time, on every rank; a family whose spec is `layer_types_rebase_safe` is exempt. The period is the
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
- The stage FSDP wrap is EP-aware: expert wrappers are FSDP-ignored (`ep_size>1`) or sharded as
  stage-mesh DTensors (`ep_size==1` + `fsdp_shard_ep1_experts`); `fsdp_shard_ep1_experts: false` is
  rejected under PP.
- EP gradient sync runs in one deferred post-backward sweep (`EPConfig.defer_grad_sync`): PP forces
  `gradient_accumulation_steps=1`, so per-backward hooks would re-fire after every microbatch. The
  sweep is off at `ep_size == 1` with the default `fsdp_shard_ep1_experts`, where FSDP2's own
  reduce-scatter syncs the experts. Router and replicated averages reduce over the stage's rank
  block, never the world.
- The gradient norm is exact and rank-uniform: one world all-reduce of each rank's local squared
  norm where every stage parameter is a DTensor; otherwise stage-scoped reduces, then one sum across
  the pipeline chain.
- Router balancing: `bias_update` works (the counter all-reduce is stage-scoped); the HF **aux-loss
  path is severed** by the split, so `aux_loss` with a live `router_aux_loss_coef` is rejected at
  split time. `MoEMetricsCallback` is skipped under PP.
- Gradient checkpointing works: the recompute replays each frame's DeepEP dispatch/combine instead of
  issuing a second one, so it composes with the schedule's in-flight microbatches
  ([EP checkpoint scope](expert-parallelism.md#gradient-checkpoint-dispatch-replay)). The DeepEP
  capacity generation is registered on `PipelineStageModule`, so every microbatch sizes its own
  dispatch.
- The equivalence gate (`tests/gpu/parallelism/pp/test_pp_ep_vs_unsplit.py`, the EP modes of
  `test_pp_combos_8gpu.py`) compares against a bf16 ep1 twin with both sides on the unfused
  expert-output sum, since the fused un-permute rounds a rank's local top-k sum once.

## PP + ETP (MoE)

Pure expert-tensor parallelism (`expert_parallel_size: 1`, `expert_tensor_parallel_size: k`) inside
each stage: every rank holds every expert, sharded `k`-way along the FFN intermediate dim, and
`ReduceFromExpertTP` sums the partners' outputs in token space. The stage keeps `stage_world_size / k`
data-parallel replicas.

- ETP partners hold shards of one expert and consume the same batch, so they are **not** DP
  replicas: the deferred sweep divides the expert leg by `stage_world_size / expert_tp_size`.
- The ETP all-reduce is model math inside the forward/backward and is not deferred — safe only
  because `ep_size == 1` leaves no DeepEP combine to interleave with. Adding EP on top is rejected.
- The equivalence gate (`tests/gpu/parallelism/pp/test_pp_etp_vs_unsplit.py`) runs in **fp32**: ETP
  reorders the summation producing each expert's output, which in bf16 flips near-tied top-k picks
  with no sync bug present.

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
| PP + PEFT / expert LoRA | The adapter save writes stage-local layer indices from one stage, the PP resume restores no adapter, and `layers_to_transform` / `all-linear` resolve against re-based layers. |
| PP + `save_sharded_ep` | The per-rank partial format carries no stage layer offset, so shards from different stages would collide. |
| PP + `push_to_hub` on per-node output storage across nodes | The Trainer uploads from global rank 0, whose node lacks the other stages' parts. Upload from a shared output directory, or from the [merged](#per-node-export) one. |
| PP + `torch_compile` | The schedule captures the stage module at setup, so a compiled wrapper installed afterwards never runs. |
| PP + reentrant GC | Reentrant checkpoint runs its forward under `no_grad`, so FSDP2 skips pre-backward hook registration. `use_reentrant: false` is supported; `every_n_layers` counts from each stage's first local layer. |
| PP + `optim: muon` / `flash_adamw` | The stage-local optimizer build and the per-stage optimizer shards a resume restores are validated with the AdamW family only. |
| PP + `moe_balancing: bias_update_transient` | The transient side-buffer is a plain attribute the stage split and the PP resume never carry. |
| PP + `lowp_precision` other than `bf16` | `lowp_keep_first_blocks`/`lowp_keep_last_blocks` would apply to every stage's own ends instead of the network's. |
| PP + `init_from_scratch` / quantized (QLoRA) loading | Neither gives the stage loader a checkpoint to stream. |
| PP + `accelerate launch` | Accelerate owns the FSDP/DDP wrap the pipeline replaces. Launch with `torchrun`. |
| PP + padding-free collators | Their flattened width varies per batch while the P2P buffers are frozen, and the single row cannot be microbatched. Use [packing](#packing). |
| PP + `activation_offloading` | TRL implements it by wrapping `training_step`, which the PP schedule never calls. |
| PP + image-bearing VLM training | The vision tower and projector sit in no stage. Refused on image evidence (a dataset image column, embedded image parts, or a collator declaring image columns); a text-only run of a composite checkpoint is admitted and re-emits the tower unchanged, and `text_only_model: true` drops it from the build. |
| PP + layer-indexed `freeze_layers_patterns` / `unfreeze_layers_patterns` | Each stage's layers are re-based to index 0, so a global index selects the wrong layer ([Configuration](../reference/configuration-reference.md#commonscriptarguments)). |

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
(`src/models/head_transform.py`). A family whose forward it does not reproduce is refused on every
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
- **Turn GC off under dense PP where memory allows.** A microbatch's activations are already `1/m`
  of a step. Measured on Qwen3-8B, seq 8192, one 8×B300 host split into two simulated NVLink
  domains, 524k tokens/step: dp8+GC 151.2k tok/s at 114 GB; pp2×dp4+GC (m=8) 125.3k; **pp2×dp4
  without GC (m=16) 152.1k tok/s at 66 GB**.
- **Under PP + EP the no-GC regime wins too**: halving the dispatch-group width cuts the DeepEP cost
  by more than the bubble adds. Measured on gpt-oss-20b, seq 8192, same host and global batch: ep8
  without GC 106.8k tok/s at 121 GB; **pp2×ep4 (m=16) without GC 135.2k tok/s at 37 GB**.

## Packing

`packing: true` composes with PP and is the way to avoid full-window compute on short rows: PP pads
every batch to `max_length`. The collator factory **raises** for the GatedDeltaNet families
(`qwen3_5*`, `qwen3_next*`) under PP: the kept rows carry no per-row `cu_seq_lens`, so the delta rule
would cross document boundaries.

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
The trailing pad region is an inert document: every `position_ids == 0` is a segment start and an
ignored label, so each pad token is its own length-1 segment carrying no gradient.

## Per-step metrics

Only the last stage runs the loss closure, while HF logs from global rank 0 on the *first* stage.
`PipelineLossAdapter.metrics_fn` closes the gap: the trainer returns this rank's per-step scalars and
the mixin broadcasts the last stage's values down the chain (one fp32 vector per step) into the
trainer's metric store. The names come from configuration, since only values cross; the mixin pins
them at setup and proves every chain rank agrees. SMPO declares it, dividing its metric sums by the
full-batch constants its `step_state_fn` stashes. Offline GRPO's per-sample diagnostics drain over
the stage group and are shared down the chain once per log step.

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
- `PipelineLossAdapter.predictions_fn` runs on the last stage before the broadcast (classification
  sends `[B, num_labels]`, not `[B, S, num_labels]`). Without one, the trainer hoists
  `preprocess_logits_for_metrics` onto the last stage, which must be row-independent (argmax and
  top-k are); `compute_metrics` with neither is refused.
- The evaluation gather is scoped to the data-parallel replicas, since every chain rank returns the
  same predictions.
- The denominator is adapter-declared (`PipelineLossAdapter.eval_normalizer`): the replica's
  loss-token count by default; reward/DPO pair count, KTO row count, SMPO 1, classification its
  pooled denominator, offline GRPO its per-loss-type denominator.
- DPO and KTO require their precomputed reference columns in the eval dataset too.

## Loading

Loading is **stage-aware**: each rank builds the whole model shell on the meta device, plans the
checkpoint against that key space, keeps only the plans for its stage's decoder layers — re-basing
`layers.5` on disk to `layers.1` on a stage starting at layer 4 — and streams exactly those tensors
from safetensors (`load_pp_stage_model`, `src/model_loading/parallel/pp.py`, narrowing the
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

`save_pp_checkpoint` (`src/checkpoint/parallel/pp.py`) writes one complete-tensor safetensors
part set per stage under the hub's names plus a merged standard HF index: the directory loads via
plain `from_pretrained` and serves like an EP-gathered save
([Checkpoints](../reference/checkpoints.md#serving-on-vllm--sglang)). A stage names its tensors
stage-locally, so `build_pipeline_stage` snapshots the unsplit model's load conversions and the save
reverts each decoder layer, EP layer gather and wrapper stash through them. A module applied to the
token embedding alone (the spec's `input_module_attrs`) stays on stage 0, so no two stages write it.

The **writer set** follows the output filesystem (`DIST_OUTPUT_SHARED_FILESYSTEM`, falling back to
the `DIST_SHARED_FILESYSTEM` umbrella):

| Artifact | Shared output FS | Per-node output FS |
|---|---|---|
| Stage parts `model-ppSSSSS-of-PPPPP-*.safetensors` | one rank per stage (`stage_local_rank == 0`) | one rank per stage per **node** (`is_pp_shard_writer`) |
| Index, `config.json`, tokenizer, a text-only run's untouched vision tower (`model-wrapper-*`) | global rank 0 | one rank per node; the index is the full one |
| `optimizer_meta.pt`, `scheduler.pt`, `trainer_state.json`, `router_balancing_biases.pt`, offline GRPO's `reference_logps.pt` | global rank 0 | one rank per node, whole |
| `optimizer_shard_<rank>.pt`, `rng_state_<rank>.pth` | every rank | every rank, on its own node |

Resume reads each stage once. A stage the stage-aware loader built from the checkpoint (every
scripted resume at the default `use_grouped_gemm`, which repoints the load —
[Resuming training](../reference/checkpoints.md#resuming-training)) skips its reload; otherwise each
rank maps the checkpoint's names onto its stage's (`ModuleLayoutView`) and streams only those
tensors. Because the parts carry the unsplit model's names, the **weights are
topology-independent**: a complete directory resumes onto a different `pp_size`. The **optimizer**
shards are stage-local: restore is gated on the topology fingerprint and `pp_stage_partition` (both
in `optimizer_meta.pt`), and any drift raises. PP+EP resumes expert weights at construction, so
launch through the training scripts ([Resume by parallelism mode](../reference/checkpoints.md#resume-by-parallelism-mode)).

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

`merge_pp_stage_parts` (`src/checkpoint/pp_stage_merge.py`) requires identical indexes across the
inputs and copies each part the index names from the first input that holds it, refusing a size
mismatch. It copies every input's optimizer shards, RNG and prefetch states and one
`optimizer_meta.pt`, refusing metas that differ, a per-rank file held at two sizes and a rank whose
shard no input holds. The config, tokenizer, remote code, trainer state, scheduler and export
sidecars come from the first input. It writes the index last and finishes with the absent-shard
check. A part no input holds raises naming it and its stage. The merged directory loads with
`from_pretrained`, uploads like a shared-filesystem save, and resumes exactly on the save's topology.

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
`src/trainers/parallelism/pipeline_loss_adapter.py`). `_configure_pipeline_parallel` calls
`_pp_loss_adapter()` once, before anything reads the adapter, so an adapter builder may seed the
per-step state its loss closes over.

The refusals are pure functions in `src/trainers/parallelism/pipeline_rules.py`.
`_split_pipeline_stage` runs them before the split, in this order: the trainer's own
`_reject_unsupported_pp_options` hook (DPO and KTO: positional `model`/`args`, a live `ref_model`, the
precompute-only reference contract, `compute_metrics`), then `reject_unsupported_pp_training_args`
(`max_length`, `save_sharded_ep`, per-node `push_to_hub`, reentrant GC, eval batch size,
`torch_compile`, TRL's `activation_offloading`), `reject_pp_peft`, `require_pp_stage_model` and,
for a multimodal wrapper, `reject_pp_vlm`. After `super().__init__`, `reject_pp_collator`,
`require_pad_token_id` and `reject_uncovered_pp_ep_layers` gate the collator wrap and the stage wrap.
SMPO and offline GRPO call `reject_pp_peft` and `reject_pp_compute_metrics` on their explicit ctor
parameters themselves. `reject_pp_vlm` and `reject_uncovered_pp_ep_layers` are collective: their
evidence is rank-local, so the verdict is joined across ranks.

| Trainer | Enabled | Contract notes |
|---|---|---|
| SFT | ✅ | canonical causal-LM |
| SMPO | ✅ | interleaved pairs; percentile clips stay rejected (batch-global quantiles are not microbatch-decomposable) |
| Reward (BT) | ✅ | interleaved pairs over scores pooled in the loss; needs `num_labels: 1`, and a `margin` column in the train dataset or in no batch |
| Classification | ✅ | every loss variant; single-label needs `num_labels >= 2`; `predictions_fn` broadcasts pooled `[B, num_labels]` |
| DPO | ✅ precompute-only | reference log-probs as dataset columns; limits: [DPO](../training-methods/preference/dpo.md) |
| KTO | ✅ `apo_zero_unpaired` only | KL-free, per-row decomposable, precompute-only; the default `kto` loss needs a world-global KL baseline |
| Offline GRPO | ✅ | all three loss types and both PG formulations; `max_prompt_length` + `max_completion_length` required. `kl_beta > 0` scores the reference once at construction by a forward-only sweep through the pipeline and ships it per completion token as a dataset column (sized, indexable, unsharded datasets); the scores ride every checkpoint (`reference_logps.pt`) |
| Embedding | ❌ | the in-batch similarity matrix *is* the loss |
| SDPG | ❌ | online GRPO's blockers plus a second, differently-shaped same-model forward |
| Online / async GRPO | ❌ | the weight sync sends the full `named_parameters()` from one rank set, which no PP rank holds |
| Teacher / self distillation | ❌ | a second forward per microbatch through a distinct frozen teacher |

## Code map

`src/distributed/pipeline_parallel/`: `families.py` (`PPFamilySpec`, `PP_FAMILY_SPECS`, the
hyper-connection stream forwards), `model_rules.py` (`reject_unsupported_pp_structure`, the
run-settings gate), `partition.py` (the head-weighted split and `validate_partition`), `stage.py`
(`StageLayout`, `PipelineStageModule`, `build_pipeline_stage` and the stage↔global naming contract),
`stage_losses.py` (`TokenLoss`, `FusedHeadLoss`), `groups.py` (the chain and stage groups), `runner.py`
(`PipelineRunner` over torch's `Schedule1F1B` / `ScheduleGPipe`, `resolve_microbatch_count`); the
stage-aware loader is `src/model_loading/parallel/pp.py`. Trainer side: `PipelineTrainerMixin`
(`src/trainers/parallelism/pipeline_trainer.py`, inert at `pp_size == 1`), the refusals
(`src/trainers/parallelism/pipeline_rules.py`), `PipelineLossAdapter` and the stage clip
`pp_clip_grad_norm` (`src/trainers/parallelism/grad_clipping.py`); the next-token label math the losses
share lives in `src/kernels/logprobs.py`.
