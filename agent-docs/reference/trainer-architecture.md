# Trainer Architecture

All distributed trainers inherit `DistributedTrainerMixin` (`src/trainers/mixins/base.py`), which sits
alongside a base trainer class (`SFTTrainer`, `Trainer`, `GRPOTrainer`) and overrides the
parallelism-sensitive methods while delegating the rest. `base.py` itself owns the core lifecycle:
config extraction and mode dispatch, a custom `Accelerator` (no DDP wrapping, manual gradient sync),
the FSDP2 wraps the modes share, optimizer construction and the parallelism-aware eval loop. It
composes its seven sibling sub-mixins:

| Sub-mixin | Owns |
|---|---|
| `CheckpointingMixin` | Save / resume, plus the LR-scheduler and router-balancing-bias sidecars |
| `DataParallelDataLoaderMixin` | Parallelism-aware DP dataloaders |
| `EpIntrospectionMixin` | EP module/param discovery, EP-safe gradient checkpointing |
| `GradientSyncMixin` | The per-mode FSDP2 wrap, the QLoRA / deferred-EP / TP-replicated grad sweeps, the EP/DTensor-aware global-norm clips |
| `ParallelismValidationMixin` | Mode and LoRA/EP/TP compatibility checks |
| `PipelineTrainerMixin` | PP hooks, inert unless `pp_size > 1` |
| `TokenMetricsMixin` | The loss-token counter behind `train/total_output_tokens`: an on-device per-step accumulator, gathered once per log |

Each is a class because it reads live trainer state; the methods a test or a caller reaches as
`DistributedTrainerMixin.<name>` resolve through the MRO unchanged. `CheckpointingMixin`'s zero-arg
`super()` calls (`_save_checkpoint`, `save_model`, the two loads) must reach the base Trainer —
`tests/cpu/trainers/test_mixin_composition.py` fails if a sibling base ever intercepts one.

Four modules in the same package sit outside that composition. `StoredMetricsMixin`
(`src/trainers/mixins/stored_metrics.py`) is mixed in *directly* by SMPO, teacher and self
distillation, and SDPG for buffered per-step metric logging. Offline GRPO
and the embedding trainer keep their own `log` instead: offline reads the train/eval bucket off
`model.training` rather than the mixin's `"loss" in logs`, and the embedding trainer's eval metrics
go into `output.metrics` for best-model tracking. Under PP the store would be fed from the last
stage ([Pipeline Parallelism](../parallelism/pipeline-parallelism.md), not yet available in this
release).

`GradientSyncMixin` dispatches by method, not by mode string: `_setup_ep_gradient_sync`,
`_setup_cp_gradient_sync` and `_setup_ep_tp_gradient_sync` are called directly from `mixins/base.py`'s
per-mode setup, and the EP one derives the FSDP ignored-module set in a single module-tree walk it
hands to `_apply_ep_aware_dp_fsdp2`.

The other three are imported as plain functions — `grad_clip.py::scale_shards_to_max_norm_` (the
shared clip coefficient, below), `loss_masks.py::effective_loss_mask` (`completion_mask ∧ tool_mask`
where a `tool_mask` exists, else `completion_mask`), and `pp_gates.py` (the shared PP rejection
vocabulary, below).

## Trainer compatibility

| Trainer | Base Class | EP | CP | TP | ETP | PP |
|---------|-----------|:--:|:--:|:--:|:--:|:--:|
| `DistributedSFTTrainer` | `SFTTrainer` | Yes | Yes | Yes | Yes | Yes |
| `SmoothMarginPOTrainer` | `Trainer` | Yes | Yes | Yes | Yes | Yes (no VLM / `padding_free` / clip percentile / PEFT; `label_pad_token_id: -100`) |
| `OfflineGRPOTrainer` | `ChunkedLogprobsCore`, `Trainer` | Yes | No | Yes | Yes | Yes (`kl_beta > 0` via a construction-time reference sweep) |
| `DistributedDPOTrainer` | `DPOTrainer` | Yes | No | Yes | Yes | Yes (precompute-only; `sigmoid`/`hinge`/`ipo`) |
| `DistributedKTOTrainer` | `KTOTrainer` | Yes | No | Yes | Yes | Yes (`apo_zero_unpaired`, precompute-only) |
| `DistributedRewardTrainer` | `RewardTrainer` | Yes | No | Yes | Yes | Yes |
| `ClassificationTrainer` | `Trainer` | Yes | No | Yes | Yes | Yes |
| `DistributedGRPOTrainer` | `GRPOTrainer` | Yes | No | Yes | Yes | No |
| `DistributedSDPGTrainer` | `DistributedGRPOTrainer` | Yes | No | Yes | Yes | No |
| `DistributedAsyncEnvironmentalGRPOTrainer` | `GRPOTrainer` | Yes | No | Yes | Yes | No |
| `DistributedDistillationTrainer` | `Trainer` | Yes | No | Yes | Yes | No |
| `DistributedSelfDistillationTrainer` | `DistributedSFTTrainer` | Yes | No | Yes | Yes | No |
| `EmbeddingTrainer` | `SentenceTransformerTrainer` | Yes | No | Yes | Yes | No |

The PP axis itself is **not yet available in this release** — `pipeline_parallel_size > 1` is
rejected at config time, and the PP column records each trainer's `_supports_pp` declaration: which
trainers take the axis when the engine lands ([Pipeline Parallelism](../parallelism/pipeline-parallelism.md)).

Support is declared per class as `_supports_ep` / `_supports_cp` / `_supports_tp` / `_supports_pp`
and enforced in `ParallelismValidationMixin`; `_pp_unsupported_reason` carries the rejection text.
`DistributedTrainerMixin` defaults them to EP/TP on and CP/PP off, so a trainer states only what it
flips — plus its PP verdict, which is written out even when it matches the default because
`_pp_unsupported_reason` is meaningless without it. There is no `_supports_etp` — ETP folds into
`ep_group_size = ep_size × expert_tp_size`, so it is gated by `_supports_ep`.

**CP** works only where the loss is computable from a sequence chunk. The rest inherit the default
`_supports_cp = False` because the trainer uses `logits_to_keep` (offline GRPO), needs global
log-probability sums (DPO, KTO), needs full-sequence pooling (classification, reward, embedding),
wraps two models (distillation), or runs a separate-length privileged-teacher or rollout sequence
(self distillation, SDPG, online and environmental GRPO).

**PP** needs a single-forward objective on one stage's logits, and the conditional rows above are
constructor-time gates rather than class attributes. Rejections land in three places.

`PipelineTrainerMixin._maybe_prepare_pipeline_model` rejects the generic blockers on every trainer:
an unset `max_length`, `save_sharded_ep`, reentrant gradient checkpointing, `peft_config`, TRL's
`activation_offloading`, `torch_compile`, image-bearing data, and an eval batch size differing
from train. Padding-free collation is rejected at the collator wrap (its flattened width varies
every step); **packing is supported**, at one packed row per microbatch.

`src/trainers/mixins/pp_gates.py` holds the shared vocabulary the preference and offline-GRPO
wrappers add on top — a live `ref_model`, missing precomputed reference columns, and
`compute_metrics` (no PP reduction reproduces their non-PP prediction convention). A tied-embedding
checkpoint, live MTP tail layers, an unsupported model family, and aux-loss MoE balancing are
rejected later, at the model split (`src/distributed/pipeline_parallel/split.py`). See
[Pipeline Parallelism](../parallelism/pipeline-parallelism.md).

## Modality support

Modality routing differs by method. SFT, SMPO, DPO, reward modeling and the distillation scripts key
on the **run** (`is_vlm_run`): the VLM data path needs a multimodal checkpoint *and* declared image
data, so text-only rows on a natively-multimodal model (Qwen3.5/3.6, Gemma 4, Inkling) train through
the text pipeline with packing available ([declaration rules](../data/dataset-formats.md#sft-vlm)).
KTO takes its processor **class** from the checkpoint (`install_resolved_tokenizer` keeps the
processor a multimodal checkpoint resolved to) but routes its data path off the **dataset**, as
`scripts/training/preference/kto.py` states. The model class always follows the checkpoint.

| Method | Vision | Notes |
|---|---|---|
| SFT | Yes | Conversation-embedded images or an `images_field` column; packing/padding-free rejected on the VLM path; CP patches text attention only. See [SFT — VLMs](../training-methods/sft.md#vision-language-models) |
| DPO / KTO | Yes | `images`/`image` column routes to TRL's vision collators. Vision excludes `precompute_ref_log_probs`, so EP DPO needs standard-PEFT adapters |
| SMPO | Yes | `DataCollatorForVLMSMPO` processes images at collation; CP, padding-free and PP are text-only. See [SMPO — VLMs](../training-methods/preference/smpo.md) |
| Teacher distillation | Yes | Student and teacher share the processor's vision geometry; over-length rows pre-filtered |
| Self-distillation (SDPG offline) | Yes | Privileged hint appended to the last user turn; teacher branch fails loud on overflow |
| Reward | Yes, on score-headed families | `DataCollatorForVLMPreference` expands images into the shared prompt at collation. Refused before the distributed init on a multimodal family with no sequence-classification head. See [Reward — VLMs](../training-methods/preference/reward-modeling.md#vision-language) |
| Classification | No | Same head roster as reward modeling, but the classification script has no vision data path; multimodal architectures still train on text |
| GRPO (offline / online / environmental) | No | Text rollouts |
| Embedding | No | Text towers only |

Pipeline parallelism rejects a vision-language **run** on every trainer — an image column, embedded
image parts, or an image-consuming collator — because a stage split keeps the text backbone and the
task head only. A text-only run of a multimodal checkpoint is admitted: the vision tower and
projector are held by no stage, kept untrained on the save rank, and re-emitted unchanged in every
checkpoint.

The data contract is the same across methods: conversations carry typed content parts
(`{"type": "text" | "image", ...}`), a per-modality top-level column is injected into the
conversation at map time, placeholder expansion belongs to the model's processor, and media
extraction is driven by part type. A new modality is an additive part type plus a column field.

## Lifecycle

```python
class MyDistributedTrainer(DistributedTrainerMixin, SomeBaseTrainer):
    _supports_cp = True   # only the flips; EP/TP default on, CP/PP default off
    _supports_pp = False

    def __init__(self, *args, **kwargs):
        kwargs = self._init_distributed_config(kwargs)  # extract parallelism kwargs
        super().__init__(*args, **kwargs)               # base init: accelerator, model, optimizer
        self._setup_distributed_modes()                 # apply parallelism after model exists
```

**`_init_distributed_config(kwargs, training_args=None, ctor_args=(), **explicit)`** pops the toolkit-only kwargs
before the base trainer sees them: `parallelism_config` (a `ParallelismConfig`; passing `None`
raises `ValueError`), the save flag `save_sharded_ep` (default `False`), `moe_balancing`,
`dataset_presharded`, and `bf16_optimizer`. It also reconciles
`save_on_each_node`, the Liger config and the GC `use_reentrant` kwarg with the requested mode.

A trainer that forwards `**kwargs` calls it as above. One whose `__init__` names those parameters
passes them through `**explicit` instead (SMPO, Classification, offline GRPO, teacher distillation);
kwargs-style values win over explicit ones. `training_args` defaults to `kwargs["args"]`, so a
trainer with an explicit `args` parameter passes `training_args=` (online and environmental GRPO)
and `EmbeddingTrainer` passes a synthetic `{"args": args}`.

`ParallelismConfig` validates the combination and exposes mode-flag properties (`is_ep_mode`,
`is_cp_mode`, `is_tp_mode`, `is_expert_tp_mode`, `is_ep_tp_mode`, `is_ep_cp_mode`, `is_pp_mode`);
`is_ep_mode` is `ep_group_size > 1`, covering both EP and pure ETP. Under PP,
`PipelineTrainerMixin._maybe_prepare_pipeline_model` runs *before* `super().__init__()` to split the
model into this rank's stage.

**`create_accelerator_and_postprocess()`** is overridden during base init — on the custom path only
(`_needs_custom_accelerator()`; otherwise it delegates to the base): no DDP wrapping (manual gradient
sync), `gradient_accumulation_steps=1` (the Trainer drives accumulation), bf16/fp16 autocast. On
either path `fp32_output_conversion: false` (the default) clears `accelerator.native_amp`, dropping
the fp32 logits upcast that can cost many GB at long sequence. Under `fp16` it is ignored with a warning —
`native_amp` also gates GradScaler unscaling there, so clearing it would clip and step on scaled
gradients.

**`_setup_distributed_modes()`** dispatches on the mode flags, in this order:

```text
is_pp_mode         -> _setup_pipeline_parallel()   # stage-scoped (EP-aware) FSDP, runtime, clip
is_ep_tp_mode      -> _setup_ep_tp()               # EP + attention TP
is_ep_cp_mode      -> _setup_ep_cp()               # EP + CP
is_cp_mode         -> _setup_cp_only()
is_tp_mode         -> _setup_tp_only()             # FSDP2 on the DP axis + TP clipping (TP applied at load)
is_ep_mode         -> _setup_ep_only()             # EP only, ETP only, or EP+ETP
accelerate-managed -> accelerate FSDP / DDP        # raises if the model carries EP/grouped-GEMM wrappers
needs_ep_wrappers  -> _setup_ep_only() if MoE, else _setup_standard_data_parallel()
else               -> _setup_standard_data_parallel()   # standard FSDP2
```

TP/CP and the accelerate-managed branch are checked before `needs_ep_wrappers` because
`use_grouped_gemm` defaults to True even for dense models — without this ordering a TP-only dense
model would enter the EP path with the wrong DP size, and a dense accelerate launch would be
hijacked onto the mixin's FSDP2. An MoE whose experts are already wrapped under `accelerate launch`
raises here.

`_setup_ep_only()` patches gradient clipping and installs the EP gradient sync (which applies FSDP2
with EP modules in `ignored_params`); it handles EP, pure ETP, and EP+ETP. CP setup validates the
model is already a `UlyssesCPModelWrapper` (wrapped at load time). FSDP2 (`fully_shard`,
`reshard_after_forward` from `fsdp_reshard_after_forward`, default `false`) carries non-EP gradient
sync wherever DP > 1 — pure TP and EP+TP at DP=1 skip the wrap entirely.

QLoRA skips FSDP2 on both the plain-DP and the CP path (`fully_shard` cannot wrap bnb's non-float
`Params4bit`). `_setup_qlora_gradient_sync` sets a flag rather than per-parameter hooks, whose
rank-local firing would hang a job whose microbatch touches different adapters per rank. The sync is
one bucketed all-reduce over every trainable grad per optimizer step, with membership agreed by a
grad-presence mask so no rank reduces alone. It is ordered ahead of clipping — otherwise each rank
clips by its own coefficient — and backstopped by a step-pre-hook for `max_grad_norm: 0`.

## Parallelism modes

| Mode | DP effect | Gradient sync | Model loader |
|------|-----------|---------------|--------------|
| EP | none (orthogonal) | non-expert via FSDP2 (`ignored_params`); expert via DeepEP backward hooks within the EP group, except at `ep_group_size == 1` where FSDP2 shards the experts too. **Any multi-EP-group topology** (`defer_grad_sync`, single-domain included) drops the in-backward hooks and averages in one post-backward sweep (`_sync_deferred_expert_grads`); **across nodes** (`is_deferred_dp`) the non-expert FSDP shards over the EP group as well | `load_ep_model()` / `load_distributed_model()` |
| CP | `world / cp_size` | FSDP2 across all ranks (each holds a partial gradient from its chunk) | `load_model_for_cp()` |
| TP | `world / tp_size` | FSDP2 `fully_shard` on the DP dimension (DTensor weights) | dense: `from_pretrained(distributed_config=DistributedConfig(tp_plan="auto"))`; MoE: `_load_tp_moe_model` (attention-only TP — the auto plan mis-shards expert biases) |
| EP+CP | `world / cp_size` | EP hooks for experts + FSDP2 for non-experts | `load_model_for_ep_cp()` |
| EP+TP | `world / tp_size` | DTensor (TP) + EP hooks (experts) + FSDP2 per-TP-position DP groups | `load_distributed_model()` |
| ETP (pure, `ep_size=1`) | `world / expert_tp_size` | EP hooks with expert-TP sub-group aggregation + FSDP2 | `load_distributed_model()` |
| PP ([not yet available](../parallelism/pipeline-parallelism.md)) | `world / pp_size` | stage-scoped FSDP2 (EP-aware) inside each stage; P2P for boundary activations | `load_distributed_model()`, split per stage |

Which combinations may run, and the domain-locality rules each one carries, are the
`ParallelismConfig` allowlist's business — see
[Supported combinations](../parallelism/index.md#supported-combinations). One consequence lands in
this table: multi-domain multi-group EP+TP is rejected because the deferred cross-replica average
assumes FSDP shards over the EP group, while EP+TP shards over the `(dp, tp)` mesh.

## Gradient synchronization

The mixin bypasses HuggingFace's DDP and manages sync directly, per the table above. The one
exception to the expert/non-expert split is `experts_fsdp_managed` — `fsdp_shard_ep1_experts`
(default `true`) at `ep_group_size == 1`, the default shape of every `ep_size: 1` MoE run: the
`ignored_params` list is emptied, FSDP2 shards the experts, its reduce-scatter becomes their sole
sync, and the EP layer skips its own hooks.

In CP-only mode each rank computes partial gradients from its chunk and FSDP averaging yields the
correct global mean (`effective_grad = (1/cp_size) * sum(partial_grad_i)`). In EP+TP with DP=1,
DTensor and EP hooks alone sync; with DP>1, FSDP2 syncs non-EP params across nodes.

Clipping and sync read their process groups (TP, DP, dispatch-EP, expert-TP, expert-replica) through
one `ParallelDims` view (`src/distributed/mesh.py`) rather than re-deriving mesh lookups
per call site.

### EP-aware gradient clipping

A local norm is wrong when expert params are sharded, so the mixin patches
`accelerator.clip_grad_norm_` to reconstruct the global norm from separate expert and non-expert
norms:

- DTensor params (EP+TP): take the local shard via `._local_tensor` and all-reduce shard norms
  across the TP group into the non-expert norm.
- ETP: all-reduce expert shard norms within `expert_tp_group` first.
- All topologies: all-reduce expert norms within the dispatch (sub-EP) group.
- Multiple EP groups: all-reduce over replica groups and divide by `num_ep_groups` to avoid
  double-counting.

Then `global_norm = sqrt(expert_norm_sq + other_norm_sq)` clips every local gradient. Each path
computes its own global norm — the collective differs per topology — and applies it through
`scale_shards_to_max_norm_` (`src/trainers/mixins/grad_clip.py`), the one device-resident clip
coefficient the EP, TP and pipeline clips share.

## Optimizer construction

Each optimizer module under `src/optimizers/` owns its own builder, and every builder is pure — it
takes the model, args and decay parameters and returns an optimizer without touching trainer state.
`create_optimizer` picks one and owns the `self.optimizer` assignment: `bf16_optimizer` →
`AdamWBF16` (stochastic rounding); an `args.optim` present in the `NAMED_OPTIMIZER_BUILDERS`
registry (`src/optimizers/registry.py`; Muon, FlashAdamW) → that builder; `fp32_non_ep_params` (or
`optim: sgd` on a model with EP layers) → param groups split by `(weight_decay, dtype, is_dtensor)`
with `foreach`/`fused` disabled (PyTorch's foreach/fused optimizers cannot mix DTensor and
plain-tensor params); else the base Trainer optimizer.

The last branch is gated, not a fallthrough: `_refuse_stock_optimizer_on_mixed_params` raises on a
stock fused/foreach AdamW whenever `ep_group_size > 1`, or the EP wrappers hold the experts and
FSDP2 does not (`experts_fsdp_managed` false). `aten._fused_adamw_` raises "mixed torch.Tensor and
DTensor" over that parameter set at the first step; the message names AdamWBF16 and
`fp32_non_ep_params` as the two supported shapes.

It then registers three step-pre-hooks so their grad syncs still run when `max_grad_norm: 0` skips
clipping: `_register_tp_replicated_grad_sync_hook` (TP replicated-grad sync),
`_register_deferred_ep_grad_sync_hook` (the multi-group-EP cross-replica sweep), and
`_register_qlora_grad_sync_hook` (the QLoRA DP sweep). The TP hook no-ops
when `max_grad_norm > 0`; the EP hook must *not* gate on `max_grad_norm` — on a logging step at
`max_grad_norm == 0` transformers still reaches the patched `clip_grad_norm_`, so it gates on
whether the sweep already ran this step.

At `max_grad_norm <= 0` transformers asks for an unclipped norm purely to log it. The mixin's
`_get_grad_norm` answers `None` on the steps that log nothing — replicating `DefaultFlowCallback`'s
rule off replicated trainer state, so every rank answers alike — rather than paying a
`torch._foreach_norm` and an all-reduce per rank per step for a discarded number. The three hooks
above are what keeps the grad sweeps running on those steps. See
[BF16 Optimizer](../optimization/bf16-optimizer.md).

## DataLoader and data parallelism

`DataParallelDataLoaderMixin` overrides `get_train_dataloader()` / `get_eval_dataloader()` to build
dataloaders with a custom DP size/rank so ranks in the same TP/CP/ETP group — and every rank of one
pipeline chain — receive identical data. The custom path fires when `_needs_custom_dataloader()` is
True: TP, CP, ETP, or PP active, or the dataset is pre-sharded per DP rank. EP alone does not trigger
it (EP is orthogonal to DP) unless the dataset is pre-sharded.

Some trainers diverge: SMPO sets custom tokenized signature columns; Classification defaults
`remove_unused_columns=False`; online/environmental GRPO scale the batch by `steps_per_generation`
and rebuild TRL's `RepeatSampler` at the DP consumption rate
([batch geometry](../training-methods/grpo/online-grpo.md#data-flow-and-batch-construction));
Offline GRPO uses `MultiGroupSampler`.

`ParallelismConfig` computes both. DP size is
`(world_size / pp_size) / max(tp_size, cp_size, expert_tp_size)`; `get_data_parallel_rank()` derives
the shard index per mode ([per-mode derivation](../parallelism/data-loading.md)). It divides the
**stage-local** rank, not the global one — that is what makes every rank of one pipeline chain
consume the same batch. `_prepare_dataloader()` passes the computed size/rank to
`accelerate.prepare_data_loader()` as `num_processes` / `process_index`. A dataset already sharded
per DP rank passes `1` / `0` instead, so accelerate places batches on the device without re-sharding
away `(N-1)/N` of each slice; offline GRPO passes the same pair, its `MultiGroupSampler` having
already sharded.

## Training loop integration

HuggingFace's `_inner_training_loop` calls `accelerator.prepare(model)`, which normally wraps in
DDP. The mixin replaces `prepare()` with a no-op for model/optimizer/scheduler when
`_should_skip_ddp_wrapping()` is True — `_fsdp_wrapped or _mixin_manages_gmm() or is_tp_mode or
is_expert_tp_mode or is_pp_mode`. `_mixin_manages_gmm()` is `needs_ep_wrappers and not
accelerate-managed`; `needs_ep_wrappers` alone defaults True even for dense models, so this guards
against hijacking an accelerate-managed DDP/FSDP launch.

In CP mode `DistributedSFTTrainer` disables HuggingFace's `num_items_in_batch` loss scaling
(`model_accepts_loss_kwargs = False`): CP ranks process chunks of the same sequence and FSDP
averaging already gives correct gradients, so without this both loss and `grad_norm` are inflated by
`cp_size`.

HuggingFace infers that flag from the model's forward signature alone — any `**kwargs` makes it True
— and then drops its own `/gradient_accumulation_steps` division. A trainer whose `compute_loss`
returns the batch's own mean therefore declares `_loss_is_own_mean`, which `_setup_distributed_modes`
applies once for every trainer that declares it.

## CP loss and metrics in SFT

`UlyssesCPModelWrapper.forward()` owns the boundary handling and the globally-normalized sum loss
([the two corrections](../parallelism/context-parallelism.md)); `_compute_cp_metrics()` mirrors the
boundary handling for accuracy. `DistributedSFTTrainer` calls base `Trainer.compute_loss`, not
`SFTTrainer.compute_loss`: the TRL version adds metrics that assume logits cover the full sequence,
but under CP they cover only the local chunk.

## Trainer-specific notes

- **SmoothMarginPOTrainer** — reference-free preference optimization; loss types `sigmoid`, `hinge`,
  `ipo`, `smooth_lower_bound`; margin scheduling via `VariableSchedulerCallback`; CP via cross-rank
  all-reduce of log-prob sums and token counts. See [SMPO](../training-methods/preference/smpo.md).
- **OfflineGRPOTrainer** — advantage methods `z_norm`, `minmax`, `quantile_norm`,
  `quantile_uniform`, `robust`; PG formulations `prob_weighted` or `reinforce`. `ChunkedLogprobsCore`
  owns its log-prob path; `use_chunked_grpo_logprobs` has no effect under PP (the schedule never calls
  `_get_per_token_logps`) and warns at construction.
- **DistributedDPOTrainer** — the reference model is not parallelized under EP/TP: use
  `precompute_ref_log_probs=True`, or PEFT/LoRA with `ref_model=None` (LoRA works under EP, not TP).
- **DistributedDistillationTrainer** — losses `kl_divergence`, `mse`, `soft_cross_entropy`,
  `cosine_similarity`, `jensen_shannon`, `earth_mover_distance`, `alpha_beta_divergence`, `slim`;
  the teacher forward runs under `torch.no_grad()` in `eval()` mode.
- **DistributedAsyncEnvironmentalGRPOTrainer** — async
  multi-turn RL with Ray actors and vLLM servers; rollout
  generation overlaps training; environments resolved by `environment_type` through
  `src/environments/registry.py`. See [Environmental GRPO](../training-methods/grpo/environmental-grpo.md).

The `src/trainers/grpo/` package keeps the three trainers (`environmental.py`, `online.py`,
`offline.py`) at the top level, with support code in `objective/` (pure loss-side functions),
`mixins/`, and `rollout/`. `environmental.py` keeps the objective itself: batch assembly,
advantages, the IS trust region and the rank-uniform fences.

`rollout/` holds function modules (`weight_sync.py`, `weight_sync_clients.py`, `trajectory_spans.py`,
`routing_replay.py`, `completions_logging.py`) plus the three mixins the environmental trainer composes:
`AsyncRolloutMixin` (`async_rollouts.py` — Ray actors, the prefetch thread, engine weight sync),
`TrajectoryTokenizeMixin` (`trajectory_tokenize.py` — trajectory → training rows) and
`RolloutMetricsMixin` (`rollout_metrics.py` — completion logs and per-episode diagnostics).

### Online GRPO vLLM weight sync

`DistributedGRPOTrainer._setup_weight_sync` replaces TRL's
`VLLMGeneration.sync_weights` with `_distributed_sync_weights` whenever vLLM generation is set up.
TRL's default only unfolds DTensors when its `DistributedBackend.is_fsdp` flag is set (an accelerate
`fsdp_plugin`), but FSDP2 is applied manually via `fully_shard` with accelerate in MULTI_GPU mode, so the default forwards DTensors
verbatim and the `torch.cat` in `packed_broadcast_producer` triggers a DTensor dispatch that
deadlocks against the trainer↔vLLM NCCL group. It uses the vendored `VLLMWeightSyncClient`
(`src/distributed/nccl/`) instead of TRL's `VLLMClient`, which would import the vLLM package.

`_generate_single_turn` broadcasts the per-rank rollout result from the TP-group leader; without it
TRL slices the broadcast by `process_index`, each TP rank lands on different completions, and the
first forward deadlocks on its first all-reduce.

`_distributed_sync_weights` calls `sync_trainer_weights`
(`src/trainers/grpo/rollout/weight_sync.py`, also used by the environmental trainer), which gathers,
sends, and resets the vLLM prefix cache — see
[the gather](../training-methods/grpo/online-grpo.md#weight-sync) for what it collects.
EP layers are found by `isinstance(module, EPMoELayerBase)` rather than an `ep_config` probe (a PEFT
`modules_to_save` wrapper forwards `__getattr__` and would match the wrapper too), and PEFT/LoRA is
merged into the base for the gather and forwarded under base-model param names.

## FSDP2 output capturing

Transformers models capture auxiliary outputs (router logits, hidden states, attention weights) via
a `capture_outputs` decorator keyed on `_CAN_RECORD_REGISTRY` (`str(class)` → capturable flags).
FSDP2 `fully_shard` creates dynamic subclasses at wrap time (`GptOssModel` → `FSDPGptOssModel`) that
are not in the registry, so capture silently fails. `_register_output_capturing_for_fsdp()` walks
the wrapped model, finds modules with `_can_record_outputs` not yet registered, and adds them. It
runs from `_setup_distributed_modes()` only when `_fsdp_wrapped=True` (the torchrun path);
accelerate-managed FSDP v1 does not create dynamic subclasses and is unaffected.

## Checkpoint save and load

Weight save/load lives in `src/distributed/checkpoint/`. `save_model()` builds a
`CheckpointContext` snapshot (the one place trainer internals are read) and hands it to
`save_checkpoint()`, whose `select_checkpoint_saver()` ladder returns the per-mode `CheckpointSaver`
(`None` = fall through to `Trainer.save_model`); weight resume is driven by
`CheckpointLoader` and the per-rank optimizer shards by `OptimizerShardStore`, both built over the
mixin's `_checkpoint_load_context()` (the `_load_from_checkpoint` / `_load_optimizer_and_scheduler` /
`_load_best_model` hooks are thin delegators). The ladder reads rank-uniform config only, so every
rank picks the identical saver.

Around it the mixin keeps the non-weight parts of a checkpoint: `_save_checkpoint` (which defers
`save_total_limit` rotation until the new checkpoint is complete),
`_persist_lr_scheduler_for_resume`, and `_persist_router_balancing_biases` /
`_restore_router_balancing_biases` for the `router_balancing_biases.pt` sidecar.

`load_best_model_at_end` is refused at construction for every shape whose end-of-run reload is
guaranteed to be refused: `cp_size > 1`, a MoE carrying EP or grouped-GEMM wrappers (`ep_size: 1`
plain FSDP2 included, since `use_grouped_gemm` defaults on), and `tp_size > 1` with
`data_parallel_size > 1`. The reload would otherwise fail after the whole run and export the LAST
weights.

Adapter-only runs reload in place and are exempt — unless `merge_expert_lora_on_save` folds the
adapter into a full base checkpoint, which the loader refuses like any other. For the formats,
multi-node filesystem handling, and resume, see [Checkpoints & Resume](checkpoints.md).

## Related pages

- [Checkpoints & Resume](checkpoints.md) · [Pipeline Parallelism](../parallelism/pipeline-parallelism.md)
- [Expert](../parallelism/expert-parallelism.md) · [Tensor](../parallelism/tensor-parallelism.md) · [Context](../parallelism/context-parallelism.md) Parallelism
- [SMPO](../training-methods/preference/smpo.md) · [Offline GRPO](../training-methods/grpo/offline-grpo.md) · [RLVR Online GRPO](../training-methods/grpo/online-grpo.md) · [Environmental GRPO](../training-methods/grpo/environmental-grpo.md)
