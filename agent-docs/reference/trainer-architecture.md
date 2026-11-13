# Trainer Architecture

All distributed trainers inherit `DistributedTrainerMixin` (`halo/trainers/distributed_trainer.py`), which
sits alongside a base trainer class (`SFTTrainer`, `Trainer`, `GRPOTrainer`) and overrides the
parallelism-sensitive methods while delegating the rest. The spine itself holds construction
(`_init_distributed_config`), the post-wrap orchestration (`_configure_distributed_modes`), the HF
overrides (`training_step`, `prediction_step`, `log`, `create_optimizer`, `get_total_train_batch_size`,
`_move_model_to_device`) and the reference-model hooks. Everything else sits in a module named for
its concern:

| Module | Owns |
|---|---|
| `construction/accelerator.py` | `AcceleratorMixin`: the plain `Accelerator` on the mixin-managed path (`build_plain_accelerator`, no DDP wrapping) and the training loop's no-wrap `prepare`; `accelerate_manages_fsdp` / `accelerate_manages_ddp` say when an `accelerate launch` owns the wrap |
| `construction/optimizer_selection.py` | `resolve_bf16_optimizer` and `build_optimizer` ([below](#optimizer-construction)) |
| `construction/training_args_overrides.py` | The TRL/HF `TrainingArguments` fields the toolkit writes or reads before or around the TRL/HF constructor: `align_save_on_each_node`, `disable_trl_liger`, `keep_all_dataset_columns`, `uses_evaluation` |
| `construction/ctor_slots.py` | Positional ctor slots of a TRL/HF base, read off its installed signature (`resolve_ctor_positions`, `get_ctor_arg`, `get_ctor_config`, `get_ctor_model_and_config`) |
| `loop/checkpointing.py` | `CheckpointingMixin`: save / resume over an explicit `CheckpointContext`, the LR-scheduler and router-balancing-bias sidecars and a per-trainer sidecar hook |
| `loop/dataloaders.py` | `DataParallelLoaderMixin`: the parallelism-aware DP-sharded train / eval / test loaders and the pre-sharded length equalization |
| `loop/evaluation.py` | `EvaluationMixin`: `evaluate` / `evaluation_loop`, the unequal-batch escape hatch, the DP-scoped evaluation gather and the final round's padding cut (`eval_split_rows`) |
| `metrics/token_metrics.py` | `TokenMetricsMixin`: the loss-token counter behind `num_unmasked_output_tokens_seen` (an on-device accumulator) and HF's input-token count (`include_num_input_tokens_seen: all`, counted as attended tokens by `count_attended_tokens` and added up on the device), summed over the world in one all-reduce per log, the input tokens also before each save and at each step the [EfficiencyCallback](../training-methods/callbacks.md#efficiencycallback) measures, where the step's data-wait seconds ride the same all-reduce (the host time of every `next` the train loop makes, timed in `NonFiniteStepGuardMixin.get_batch_samples`, the one seam every trainer draws its window through); the per-document attention-work accumulator the MFU attention term reads; HF's `total_flos` estimate, with its parameter count walked once per model, summed in the log's all-reduce and on its own at a save and at the end of training (`store_flos`); `effective_loss_mask` |
| `parallelism/mode_setup.py` | `ModeSetupMixin`: the mode ladder (`_wrap_for_parallel_mode`), every per-mode FSDP2 wrap and the QLoRA sync install, `_fsdp_excluded_params`, the CP wrapper adoption, `reject_unsynced_trainable_params` (COLLECTIVE) |
| `parallelism/ep_layers.py` | `EPLayersMixin`: the model's EP layers (`_ep_layers()`), their parameter-id sets (one `_param_id_caches` store, cleared after the wrap), EP-safe gradient checkpointing; `resolve_use_reentrant` |
| `parallelism/grad_sync.py` | `GradientSyncMixin`: the grad-accum window, the QLoRA / deferred-EP / TP-replicated grad sweeps and their step-pre-hooks |
| `parallelism/grad_clipping.py` | `GradientClippingMixin`: the EP / TP / QLoRA `clip_grad_norm_` replacements and the EP/DTensor-aware global norm they reduce; `pp_clip_grad_norm` and the clip primitives |
| `parallelism/pipeline_trainer.py` | `PipelineTrainerMixin`: the PP split, setup, schedule-driven step and eval, inert unless `pp_size > 1` |
| `parallelism/tp_leader.py` | `TPLeaderBroadcastMixin`: object and tensor broadcasts from the TP/ETP-group leader (the GRPO generation paths) |
| `loop/step_guard.py` | `NonFiniteStepGuardMixin`: the optimizer step a non-finite global gradient norm would take, skipped or raised on ([Non-finite steps](#non-finite-steps)) |
| `construction/trainer_rules.py`, `parallelism/pipeline_rules.py`, `parallelism/adapter_rules.py` | Every construction refusal as a pure `reject_*` over explicit inputs; `pipeline_rules.py` holds the PP ones, and the adapter rules cover LoRA, PEFT and native expert LoRA under the parallelism modes |

Each mixin is a class because it reads live trainer state; the methods a test or a caller reaches as
`DistributedTrainerMixin.<name>` resolve through the MRO unchanged. The HF overrides' zero-arg
`super()` calls (the checkpoint methods, `evaluate`, `create_accelerator_and_postprocess`,
`_inner_training_loop`) must reach the base Trainer — `tests/cpu/trainers/test_mixin_composition.py`
fails if a sibling base ever intercepts one.

Three mixins sit outside that composition, in `halo/trainers/`. `StoredMetricsMixin`
(`metrics/stored_metrics.py`) is mixed in *directly* by SMPO, teacher and self
distillation, and SDPG for buffered per-step metric logging. `log` reports
each metric's row-weighted world mean through one `WorldMetrics` flush per log call, and ranks that stored
different keys raise on every rank. `store_metrics(..., rows=n)` weights
a value by the real rows it averages. Every eval store passes `rows=eval_split_rows(n)` (pairs for
SMPO) and computes its values over those rows — the distillation and SDPG trainers through
`store_batch_metrics`, which weighs a train micro-batch 1; the eval losses read
the same rows, bar SDPG's GRPO term, which TRL computes over the whole batch.
TRL's DPO, KTO and reward trainers log through their own `_metrics` and accelerate's gather instead,
so their eval metrics keep the final round's padding. Their loss call is TRL's, and so are its metric reductions:
every micro-batch gathers each metric across the world and reads it back to the host (about 15 gathers for DPO,
14 for KTO, 4 for reward; TRL's GRPO loss, which online and async GRPO and SDPG run, about 8 reductions). The
cost is latency that grows with the rank count; SFT and the trainers above fold theirs once per log.

Offline GRPO, the embedding trainer, and online and async GRPO keep their own `log` instead:
offline reads the train/eval bucket off `model.training` rather than the mixin's `"loss" in logs`,
the embedding trainer's eval metrics go into `output.metrics` for best-model tracking, and the two
on-policy trainers decouple the completions table from the metric drain. The embedding trainer's
`eval_embed/*` values are rank 0's per-batch means and count a padded final round's repeated rows
([Embedding](../training-methods/embedding.md#what-to-watch)).

`WorldMetrics` (`metrics/world_metrics.py`) is the one cross-rank metric accumulator: `StoredMetricsMixin`, SFT's loss
metrics, offline GRPO's sign metrics and the GRPO step and rollout metrics record rank-local sums into it. A
flush folds each column where it was recorded (a tensor's on its device, with the arithmetic Python's `sum`,
`max` and `min` apply to a list), then reduces over the `(key, kind)` layout the ranks agreed on last: one
float64 SUM all-reduce carries each rank's fit verdict, the keys it holds and the summed columns, one MAX
all-reduce the maxes and negated mins, and one host read brings both back. Their count, width and host work do
not grow with the rank count. A key outside the layout (the first flush, or a key behind a data-dependent gate the
first time a rank reaches it), ranks holding different keys under a key check, or a `percentile`'s raw values take
one object gather instead, whose host work does grow with it on NCCL, and the next flush reuses the union; the
async GRPO rollout metrics pay that gather every round for `episode/generation_tokens_p90`. `world_sums` is the
one-shot form for a value read outside a window (the trust-region breaker, the KL clamp fraction, the prefetch and
routing-replay counters, the token counters).

`ModeSetupMixin` dispatches by method, not by mode string: `_apply_ep_fsdp2`,
`_apply_cp_fsdp2` and `_apply_ep_tp_fsdp2` are called directly from the mode ladder
(`_wrap_for_parallel_mode`), and the EP one derives the FSDP ignored-module set in a single
module-tree walk it hands to `_apply_ep_aware_dp_fsdp2`.

`TRLDatasetPrepMixin` (`trl_dataset_prep.py`) is mixed in directly by DPO, KTO and the
reward trainer, ahead of TRL: it runs TRL's dataset preparation once per filesystem scope
([Filesystem handling](../data/filesystem-handling.md)).

`ReferenceLogpsMixin` (`reference_logps.py`) is mixed in by DPO/KTO precompute and offline GRPO. It
owns reference identity checks, resume attachment and atomic sidecar persistence; each trainer owns
its score payload and sweep ([Checkpoints](checkpoints.md#what-gets-saved)). Both record the log-prob
precision each split was scored in (`LOGPROB_PRECISION_KEY`;
[DPO](../training-methods/preference/dpo.md#resuming-a-precompute-run),
[offline GRPO](../training-methods/grpo/offline.md#reference-model)).

DPO and KTO share one `TRLPreferenceMixin` (`preference/trl_preference.py`), first in their bases: it
runs their construction (TRL's Liger path off, reference resume, distributed config, reference gates,
TRL's construction-time reference hash skipped), rebinds TRL's log-prob functions to fp32 ones, and runs
the reference sweep on the DP axis. Ahead of `DistributedTrainerMixin` its sidecar hook wins over the
checkpointing default; ahead of the TRL base its rebinds wrap TRL's loss and reference pass. A subclass
declares `REFERENCE_COLUMNS` and its Liger reason; `CTOR_POSITIONS` is derived from the TRL base's
signature per subclass. SMPO, DPO and KTO construction refusals are pure functions in
`preference/preference_rules.py`.
Offline GRPO's `reference_cache.py` streams current-batch scores to filesystem-aware writers and
maps the ordered ragged token values; its reference mixin attaches those buffers without repacking
them ([Reference model](../training-methods/grpo/offline.md#reference-model)).

`metrics/token_metrics.py::effective_loss_mask` (`completion_mask ∧ tool_mask` where a `tool_mask` exists, else
`completion_mask`) is imported as a plain function.

## Trainer compatibility

| Trainer | Base Class | CP | PP |
|---------|-----------|:--:|:--:|
| `DistributedSFTTrainer` | `SFTTrainer` | Yes | Yes |
| `SMPOTrainer` | `Trainer` | Yes | Yes (no VLM / `padding_free` / clip percentile / PEFT) |
| `OfflineGRPOTrainer` | `LogprobScoringMixin`, `Trainer` | Yes (full fine-tuning) | Yes (`kl_beta > 0` via a construction-time reference sweep) |
| `DistributedDPOTrainer` | `DPOTrainer` | No | Yes (precompute-only; `sigmoid`/`hinge`/`ipo`) |
| `DistributedKTOTrainer` | `KTOTrainer` | No | Yes (`apo_zero_unpaired`, precompute-only) |
| `DistributedRewardTrainer` | `RewardTrainer` | No | Yes |
| `ClassificationTrainer` | `Trainer` | No | Yes |
| `DistributedOnlineGRPOTrainer` | `GRPOTrainer` | No | No |
| `DistributedSDPGTrainer` | `DistributedOnlineGRPOTrainer` | No | No |
| `DistributedAsyncEnvironmentalGRPOTrainer` | `GRPOTrainer` | No | No |
| `DistributedTeacherDistillationTrainer` | `Trainer` | No | No |
| `DistributedSelfDistillationTrainer` | `DistributedSFTTrainer` | No | No |
| `EmbeddingTrainer` | `SentenceTransformerTrainer` | No | No |

No trainer refuses EP, TP or ETP. GPU rows run all three under SFT, DPO, offline, online and async GRPO,
SDPG, reward and classification; SMPO, KTO, teacher distillation and embedding are verified under EP and TP,
and self-distillation under EP only. A **No** PP trainer is refused at config time
(`<Trainer> does not support Pipeline Parallelism (PP)`); see
[Pipeline Parallelism](../parallelism/pipeline-parallelism.md).
`DistributedSDPGTrainer` overrides the online trainer's `PACKED_ROWS_REFUSAL` and declares its teacher prompt as a second entry of `ROUND_PROMPT_ROWS` ([Online SDPG](../training-methods/distillation/online-sdpg.md#refusals)).

A trainer declares its capabilities as public class constants on `DistributedTrainerMixin`.
`SUPPORTS_CP` and `PP_REFUSAL` are read by one classmethod, `reject_undeclared_axes(cp_size=,
pp_size=)`, which `parallelism_config_from_args` runs before any model loads and
`_init_distributed_config` runs again at construction:

| ClassVar | Default | Meaning |
|---|---|---|
| `SUPPORTS_CP` | `False` | CP admitted |
| `SUPPORTS_HYBRID_CP` | `False` | dense hybrid CP admitted only when declared on the concrete SFT trainer; subclasses must opt in separately ([restrictions](../parallelism/context-parallelism.md#dense-hybrid-sft)) |
| `PP_REFUSAL` | a generic refusal | the mechanism the PP refusal names; `None` where PP is supported |
| `FORWARD_ROWS_PER_EXAMPLE` | `1` | rows one model call carries per example (`2` for the chosen ⧺ rejected concat of DPO, SMPO and the BT reward trainer); the EP dispatch-ceiling gate multiplies by it |
| `SUPPORTS_LOWP_PRECISION` | `False` | the `lowp_*` knobs apply (SFT) |
| `POLICY_GRADIENT_LOSS` | `False` | the loss runs outside the forward with no labels (the GRPO family): it never adds the router aux loss, and an applied Liger FLCE draws a warning |
| `SYNCS_TO_EXTERNAL_GENERATOR` | `False` | on-policy RL that syncs weights to a rollout engine (online and async GRPO, SDPG) |
| `ROUTER_AUX_LOSS_ADDED_BY` | `None` | what adds the router aux loss: `"model_forward"` (SFT's `labels` forward), `"trl_loss"` (DPO, KTO; TRL is handed the balancing mode's coefficient, every other trainer `0.0`), or nothing, where `moe_balancing: aux_loss` is refused and `auto` turns the router logits back off |
| `LOSS_IS_OWN_MEAN` | `False` | `compute_loss` returns the batch's own mean ([below](#training-loop-integration)) |
| `REFERENCE_ALTERNATIVES` | the preference text | the trainer's way around holding a frozen reference, named by every reference warning and refusal |

The refusal texts are `<Trainer> does not support Context Parallelism (CP). Use Expert Parallelism (EP)
and/or Tensor Parallelism (TP) instead, or remove context_parallel_size.` and `<Trainer> does not
support Pipeline Parallelism (PP): <PP_REFUSAL>. Use ... or remove pipeline_parallel_size.`

Every PP-refusing trainer declares its own `PP_REFUSAL` text.

**CP** works only where the loss is computable from a sequence chunk. The rest keep the default
`SUPPORTS_CP = False`.

The reasons: the trainer uses `logits_to_keep` (Async GRPO with Environments), needs global
log-probability sums (DPO, KTO), needs full-sequence pooling (classification, reward, embedding),
wraps two models (distillation), or runs a separate-length privileged-teacher or rollout sequence
(self distillation, SDPG, online GRPO).

Offline GRPO scores boundary-aligned local hidden states through the chunked head and reduces its
token objective across CP with the shared autograd SUM. Its CP path rejects adapters; its
[full-FT reference](../training-methods/grpo/offline.md#reference-model) is persisted, not
recomputed from trained weights on resume.

**PP** needs a single-forward objective on one stage's logits, and the conditional rows above are
constructor-time gates rather than class attributes. Rejections land in three places.

`PipelineTrainerMixin._split_pipeline_stage` rejects the generic blockers on every trainer
(`halo/trainers/parallelism/pipeline_rules.py`):
an unset `max_length`, `save_sharded_ep`, reentrant gradient checkpointing, `peft_config` or a model
already PEFT-wrapped (a `PeftModel`, or one carrying PEFT tuner layers), TRL's `activation_offloading`,
`torch_compile`, image-bearing data, and an eval batch size differing from train. Padding-free
collation is rejected at the collator wrap (its flattened width varies every step); packing passes
at one packed row per microbatch, except on a model whose sequence mixers read no document boundary
([PP packing](../parallelism/pipeline-parallelism.md#packing)).

`pipeline_rules.py` also holds the shared vocabulary the preference and offline-GRPO
wrappers add on top, through the `_reject_unsupported_pp_options` hook or their own gate — a live `ref_model`, missing precomputed reference columns, and
`compute_metrics` (no PP reduction reproduces their non-PP prediction convention). A tied-embedding
checkpoint, live MTP tail layers, a family that is refused or not registered in `PP_FAMILY_SPECS`,
and aux-loss MoE balancing are rejected later, at the model split
(`halo/distributed/pipeline_parallel/model_rules.py`). See
[Pipeline Parallelism](../parallelism/pipeline-parallelism.md).

## Modality support

Modality routing differs by method. SFT, SMPO, DPO, reward modeling and the distillation scripts key
on the **run** (`is_vlm_run_across_ranks`): the VLM data path needs a multimodal checkpoint *and* declared image
data, so text-only rows on a natively-multimodal model (Qwen3.5/3.6, Gemma 4, Inkling) train through
the text pipeline with packing available ([declaration rules](../data/dataset-formats.md#sft-vlm)).

KTO routes its data path off the **dataset** (TRL's own column probe), as
`scripts/training/preference/kto.py` states. The model class always follows the checkpoint. SFT,
DPO, KTO, SMPO and distillation load the processing class by the rule in
[SFT — VLMs](../training-methods/sft.md#vision-language-models); reward modeling loads the processor
for an image run only.

| Method | Vision | Notes |
|---|---|---|
| SFT | Yes | Conversation-embedded images or an `images_field` column; packing/padding-free rejected on the VLM path; CP is text-only (the CP wrapper raises on a batch carrying `pixel_values`). See [SFT — VLMs](../training-methods/sft.md#vision-language-models) |
| DPO / KTO | Yes | `images`/`image` column routes to TRL's vision collators. Vision excludes `precompute_ref_log_probs`, so EP vision DPO takes standard-PEFT adapters or a frozen copy. A family whose frozen copy does not fit declares `refuses_trl_image_rows` and is refused before the load (Nemotron-H). Refused on a prefix-LM checkpoint (PaliGemma) before the model load (`reject_prefix_lm_image_preference`): the collators mark no completion as its causal suffix, and TRL's calls the PaliGemma processor without images |
| SMPO | Yes | `DataCollatorForVLMSMPO` processes images at collation; CP and padding-free are text-only; a prefix-LM checkpoint (PaliGemma) is refused as for DPO. See [SMPO — VLMs](../training-methods/preference/smpo.md) |
| Teacher distillation | Yes | Student and teacher share the processor's vision geometry; over-length rows pre-filtered |
| Self-distillation (SDPG offline) | Yes | Privileged hint appended to the last user turn; teacher branch fails loud on overflow |
| Reward | Yes, on score-headed families | `DataCollatorForVLMPreference` expands images into the shared prompt at collation. Refused before the distributed init on a multimodal family with no sequence-classification head. See [Reward — VLMs](../training-methods/reward-modeling.md#vision-language) |
| Classification | No | Same head roster as reward modeling, but the classification script has no vision data path; multimodal architectures still train on text |
| GRPO (offline / online / async) | No | Text rollouts |
| Embedding | No | Text towers only |

The data contract is the same across methods: conversations carry typed content parts
(`{"type": "text" | "image", ...}`), a per-modality top-level column is injected into the
conversation at map time, placeholder expansion belongs to the model's processor, and media
extraction is driven by part type. A new modality is an additive part type plus a column field.

## Lifecycle

```python
# Slots of the base the *args are forwarded to, read off its installed signature.
_CTOR_POSITIONS = resolve_ctor_positions(SomeBaseTrainer, "model", "args")


class MyDistributedTrainer(DistributedTrainerMixin, SomeBaseTrainer):
    SUPPORTS_CP = True  # declare only what differs from the mixin's defaults
    PP_REFUSAL = "the loss couples the whole batch, which no pipeline microbatch sees"

    def __init__(self, *args, **kwargs):
        # Extract the parallelism kwargs; a positional model/config is read through the table.
        kwargs = self._init_distributed_config(kwargs, ctor_args=args, ctor_positions=_CTOR_POSITIONS)
        super().__init__(*args, **kwargs)    # base init: accelerator, model, optimizer
        self._configure_distributed_modes()  # apply parallelism after the model exists
```

**`_init_distributed_config(kwargs, training_args=None, ctor_args=(), ctor_positions=None, **explicit)`** pops the toolkit-only kwargs
before the base trainer sees them: `parallelism_config` (a `ParallelismConfig`; passing `None`
raises `ValueError`), the checkpoint knobs `save_sharded_ep` (default `False`), `save_max_shard_size`
and `allow_optimizer_warm_restart`, `max_consecutive_nonfinite_steps` (the [step guard](#non-finite-steps)'s
limit), `moe_balancing`, `dataset_presharded`, and `bf16_optimizer`.
Its `_apply_training_args_overrides` step sets `fsdp = ""` on the training args (the mixin owns sharding),
hands TRL its router aux-loss coefficient, and
reconciles `save_on_each_node`, the Liger config and the GC `use_reentrant` kwarg with the requested
mode and the model ([reentrant rule](../optimization/flash-attention.md#usage)). It then runs the construction refusals
(`reject_undeclared_axes`, `reject_pp_named_optimizer`, `reject_custom_parallelism_under_accelerate`, the determinism
gates) and the PP split.
`distributed_trainer_kwargs` (`halo/entrypoints/script.py`) is the bundle every entry script
passes.

A trainer that forwards `**kwargs` calls it as above. One whose `__init__` names those parameters
passes them through `**explicit` instead (SMPO, Classification, offline GRPO, teacher distillation);
kwargs-style values win over explicit ones. The model and `training_args` default to the ctor's
`model` / `args`, read from `kwargs` or, for positionals, from their slots in `ctor_positions`
(positionals without the table raise). Online and async GRPO pass `training_args=` as well, having
resolved it first, and `EmbeddingTrainer` passes a synthetic `{"model": model, "args": args}` (a
`SentenceTransformer`'s `config` is its backbone's).

`ParallelismConfig` validates the combination and exposes mode-flag properties (`has_ep_group`,
`is_cp_mode`, `is_tp_mode`, `is_expert_tp_mode`, `is_pp_mode`); `has_ep_group` is `ep_group_size > 1`,
covering both EP and pure ETP.

**`create_accelerator_and_postprocess()`** (`AcceleratorMixin`) is overridden during base init, on the
custom path only (`_needs_custom_accelerator()`; otherwise it delegates to the base):
`build_plain_accelerator` applies no DDP wrapping (manual gradient sync), sets
`gradient_accumulation_steps=1` (the Trainer drives accumulation) and bf16/fp16 autocast.

On either path the trainer clears `accelerator.native_amp`, dropping the fp32 logits upcast that can
cost many GB at long sequence. Under `fp16` it stays: `native_amp` also gates GradScaler unscaling
there, so clearing it would clip and step on scaled gradients.

**`_configure_distributed_modes()`** takes back TRL's construction-time writes
(`_restore_trl_overrides`), runs the pre-wrap gates (adapters under TP, inert router logits,
`load_best_model_at_end`), aligns the PEFT and fp32-master dtypes, wraps the model, then runs the
post-wrap gates (`reject_unsynced_trainable_params`, the expert-LoRA rules) and installs
(`_configure_wrapped_model`). The wrap, `ModeSetupMixin._wrap_for_parallel_mode()`, dispatches on the
mode flags in this order:

```text
is_pp_mode                 -> _configure_pipeline_parallel()  # stage-scoped (EP-aware) FSDP, runner, clip
has_ep_group + TP          -> EP clip, loader mesh, EP+TP FSDP2
has_ep_group + CP          -> adopt the CP wrapper, EP clip, EP FSDP2
is_cp_mode                 -> adopt the CP wrapper, CP FSDP2
is_tp_mode                 -> _apply_tp_fsdp2() + TP clip     # FSDP2 on the DP axis (TP applied at load)
has_ep_group, or EP layers -> EP clip, EP FSDP2               # EP, pure ETP, EP+ETP, grouped-GEMM ep1
accelerate-managed         -> accelerate FSDP / DDP           # refuses EP layers
else                       -> plain DP FSDP2 (QLoRA: the DP sweep)
```

TP/CP precede the EP branch because `use_grouped_gemm` defaults to True even for dense models —
without this ordering a TP-only dense model would enter the EP path with the wrong DP size. The EP
branch takes a model whose EP layers the config does not declare only off an accelerate launch, so a
dense accelerate launch is never hijacked onto the mixin's FSDP2, and an MoE whose experts are
already wrapped under `accelerate launch` raises.

The EP branch patches gradient clipping and installs the EP gradient sync (which applies FSDP2 with
EP modules in `ignored_params`, except at `ep_group_size == 1` — [below](#gradient-synchronization)).
CP setup requires the model to be a `UlyssesModel` already (wrapped at load time,
`require_ulysses_model`). FSDP2 (`fully_shard`,
`reshard_after_forward` from `fsdp_reshard_after_forward`, default `false`) carries non-EP gradient
sync wherever DP > 1 — pure TP and EP+TP at DP=1 skip the wrap entirely. The wrap leaves out
frozen parameters of a dtype no trainable parameter shares, and raises if it would leave out a
trainable one nothing else syncs ([Data Parallelism](../parallelism/data-parallelism.md#fsdp2-strategy-by-mode)).

QLoRA skips FSDP2 on both the plain-DP and the CP path (`fully_shard` cannot wrap bnb's non-float
`Params4bit`). `_install_qlora_grad_sync` sets a flag rather than per-parameter hooks, whose
rank-local firing would hang a job whose microbatch touches different adapters per rank, and gives
every floating-point parameter rank 0's copy: the sync averages gradients only, so adapters the ranks
drew from their own RNG streams would otherwise stay apart for the whole run.

The sync is one bucketed all-reduce over every trainable grad per optimizer step, with membership
agreed by a grad-presence mask so no rank reduces alone. It is ordered ahead of clipping (otherwise
each rank clips by its own coefficient) and backstopped by a step-pre-hook for a step that misses the clip.

## Parallelism modes

| Mode | DP effect | Gradient sync | Model loader |
|------|-----------|---------------|--------------|
| EP | none (orthogonal) | non-expert via FSDP2 (`ignored_params`); router via in-backward hooks, expert via the DeepEP combine within the EP group and one post-backward divide (`runs_post_backward_sweep`), except at `ep_group_size == 1` where FSDP2 shards the experts too. **Any multi-EP-group topology** (`sweep_reduces_across_replicas`, single-domain included) drops the router hooks too and averages everything in one post-backward sweep (`_run_deferred_grad_sweep`); **across nodes** (`is_deferred_dp`) the non-expert FSDP shards over the EP group as well | `load_ep_model()` / `load_distributed_model()` |
| CP | `world / cp_size` | FSDP2 across all ranks (each holds a partial gradient from its chunk) | `load_cp_model()` |
| TP | `world / tp_size` | FSDP2 `fully_shard` on the DP dimension (DTensor weights) | dense: `from_pretrained(distributed_config=DistributedConfig(tp_plan="auto"))`; MoE: `load_tp_moe_model` (attention-only TP — the auto plan mis-shards expert biases) |
| EP+CP | `world / cp_size` | EP sync for routers and experts (as EP) + FSDP2 for non-experts | `load_ep_cp_model()` |
| EP+TP | `world / tp_size` | DTensor (TP) + EP sync (routers and experts) + FSDP2 per-TP-position DP groups | `load_distributed_model()` |
| ETP (pure, `ep_size=1`) | `world / expert_tp_size` | EP sync with expert-TP sub-group aggregation + FSDP2 | `load_distributed_model()` |
| [PP](../parallelism/pipeline-parallelism.md) | `world / pp_size` | stage-scoped FSDP2 (EP-aware) inside each stage; P2P for boundary activations | `load_distributed_model()`, split per stage |

Which combinations may run, and the domain-locality rules each one carries, are the
`ParallelismConfig` allowlist's business — see
[Supported combinations](../parallelism/README.md#supported-combinations). One consequence lands in
this table: multi-domain multi-group EP+TP is rejected because the deferred cross-replica average
assumes FSDP shards over the EP group, while EP+TP shards over the `(dp, tp)` mesh.

## Gradient synchronization

The mixin bypasses HuggingFace's DDP and manages sync directly, per the table above. The one
exception to the expert/non-expert split is `experts_fsdp_managed` — `fsdp_shard_ep1_experts`
(default `true`) at `ep_group_size == 1`, the default shape of every `ep_size: 1` MoE run: the
`ignored_params` list is emptied, FSDP2 shards the experts, its reduce-scatter becomes their sole
sync, and the EP layer skips its own hooks and the sweep its expert leg.

In CP-only mode each rank computes partial gradients from its chunk and FSDP averaging yields the
correct global mean (`effective_grad = (1/cp_size) * sum(partial_grad_i)`). In EP+TP with DP=1,
DTensor and the EP sync alone; with DP>1, FSDP2 syncs non-EP params across nodes.

Clipping and sync read the mesh groups (TP, DP) through the mesh accessors (`tp_group`, `dp_group`
in `halo/distributed/mesh.py`) and the expert groups (dispatch-EP, expert-TP, expert-replica) off
`EPConfig`, rather than re-deriving them per call site.

### EP-aware gradient clipping

A local norm is wrong when expert params are sharded, so the mixin patches
`accelerator.clip_grad_norm_` to reconstruct the global norm from separate expert and non-expert
norms. The EP and TP clips share one norm, `_compute_global_grad_norm`:

- DTensor params (TP, EP+TP): take the local shard via `._local_tensor` and all-reduce shard norms
  over the group that tiles them into the non-expert norm.
- ETP: all-reduce expert shard norms within `expert_tp_group` first.
- All EP topologies: all-reduce expert norms within the dispatch (sub-EP) group.
- Multiple EP groups: no replica reduce. Every multi-group topology defers, and the clip runs the
  deferred sweep before the norm, so the replicas already hold identical expert grads.

Then `global_norm = sqrt(expert_norm_sq + other_norm_sq)` sets the clip. Every clip derives one
device-resident coefficient,
`clip_coefficient` (`halo/trainers/parallelism/grad_clipping.py`), and refuses any `norm_type` but 2 (`require_l2`).
The EP clip refuses to install on a model with no EP layers.
The TP clip, and the EP clip under any other optimizer, scale every local gradient by it
in place (`scale_shards_to_max_norm_`). Under EP with AdamWBF16 stepping exactly the clipped parameters
that hold gradients, the EP clip hands the coefficient to the optimizer instead (`defer_grad_scale`),
which multiplies it into each gradient inside its fused step; the gradients stay unscaled until then.

## Optimizer construction

Each optimizer module under `halo/optimizers/` owns its own builder, and every builder is pure: it
takes the model, args and decay parameters and returns an optimizer without touching trainer state.
`create_optimizer` picks one and owns the `self.optimizer` assignment:

| Condition | Optimizer |
|---|---|
| `bf16_optimizer` | `AdamWBF16` (stochastic rounding) |
| `args.optim` present in the `NAMED_OPTIMIZER_BUILDERS` registry (`halo/optimizers/registry.py`; Muon, FlashAdamW) | That builder |
| `fp32_non_ep_params` (or `optim: sgd` on a model with EP layers) | Param groups split by `(weight_decay, dtype, is_dtensor)`, each fused under `optim: adamw_torch_fused` and single-tensor otherwise, `foreach` off (one foreach/fused update cannot mix DTensor and plain-tensor params, and torch refuses foreach beside fused) |
| else | The base Trainer optimizer |

`build_optimizer` (`halo/trainers/construction/optimizer_selection.py`) returns `None` for that last row, and the
stock build is gated, not a fallthrough: `reject_stock_adamw_on_mixed_params` raises on a
stock fused/foreach AdamW whenever `ep_group_size > 1`, or the EP layers hold the experts and
FSDP2 does not (`experts_fsdp_managed` false). `aten._fused_adamw_` raises "mixed torch.Tensor and
DTensor" over that parameter set at the first step; the message names AdamWBF16 and
`fp32_non_ep_params` as the two supported shapes.

Each grad sweep runs inside the patched `clip_grad_norm_`. At `max_grad_norm <= 0` transformers skips
the clip but still calls it on every step for an unclipped norm, at an infinite threshold that
`is_clipping_enabled` treats as no clipping, so no gradient is rescaled; the non-finite step guard reads
that norm. `create_optimizer` also registers three step-pre-hooks (`_register_grad_sync_step_hooks`) as
backstops for a step that misses the call: the TP replicated-grad sync, the multi-group-EP cross-replica
sweep, and the QLoRA DP sweep. Each sweep is marked once per step, so a hook acts only when the clip did
not run it: the TP hook no-ops when `max_grad_norm > 0`, and the EP hook gates on whether the sweep
already ran this step, never on `max_grad_norm`. Every hook sits out the zero-LR step torch takes to
materialize an empty optimizer's state, which a shard restore takes and a shard save takes for an optimizer
that keeps none (plain SGD): a sweep there would claim the step the next one owns. See
[BF16 Optimizer](../optimization/bf16-optimizer.md).

## Non-finite steps

An optimizer step whose global gradient norm is NaN or inf is never applied (`halo/trainers/loop/step_guard.py`).
Without the guard one such step writes NaN into every parameter and both Adam moments (the toolkit's clips
pass a NaN norm through as coefficient 1.0, torch's multiplies it into every gradient), and the model is NaN
from the next forward on. The norm transformers computes right before `optimizer.step()` is all-reduced by
every clip path, so every rank reaches the same verdict. The optimizer picks how the verdict is applied:

| Optimizer | Skip | Host read of the norm |
|---|---|---|
| AdamWBF16 and Muon (`DeviceStepSkipMixin`, `halo/optimizers/step_skip.py`) | On the device: the step runs under a 0-dim flag, set for a non-finite norm, and its kernels write no parameter or moment; transformers' per-step `zero_grad` drops the gradients | Late, without a stall: the norm is copied non-blocking to pinned memory and read at the first of the next step's norm, a step that logs, evaluates or saves, the end of training, and the next `train()` call |
| Any other: torch's, and FlashAdamW, whose third-party kernel takes no flag | On the host: the step is not called, and its gradients are discarded through the optimizer's `zero_grad` | Before every step |

| Step | Outcome |
|---|---|
| Finite norm | Applied. |
| Non-finite norm, finite weights | Skipped: weights, moments and the optimizer's step counts stay as they were (a device skip rolls its step counts and any state its step created back when the host reads the verdict), and the trainer's step counter, LR schedule and data move on. Rank 0 warns naming the step, and the logs carry `nonfinite_steps_skipped` (counted since launch). |
| Non-finite norm, non-finite weights | Raised on every rank, naming the parameters: they went bad before this step's gradients, and a skip cannot repair them. |
| More than `max_consecutive_nonfinite_steps` (default 3) in a row | Raised on every rank. `0` raises at the first. |

On the device path the warning and both raises come at the late read, at the latest before the next step's
update, so no later step is applied and a final non-finite step is reported before training ends. Against no guard
on 8× B300 at s4096 (1.2.0) that path reads TP2 +0.88%, ep1 +0.31%, ep2 +0.69% and CP2 −0.96% (six rounds, ranges
overlapping), where a host read before every step costs TP2 3.26% and ep2 42 ms of sync wait per step.

The skip is installed in `create_optimizer` on the optimizer that steps (the one inside accelerate's wrapper), so
every trainer gets it: before the LR scheduler wraps its `step`, or over the scheduler's wrapper when the caller
passes a pre-built pair (`optimizers=(optimizer, scheduler)`), in which case a skipped step still counts as taken
for the scheduler. The zero-LR step that materializes optimizer state for a shard restore or save always steps.

A skipped step's loss is discarded with its gradients: transformers' loss accumulator returns to its value
before the step's first micro-batch (by a device-side select on the device path), so none of the step's
micro-batches, finite ones included, adds to its window's logged `loss` or to the run's `train_loss`, whatever
the logging strategy, and later windows stay finite. Both divide by the applied steps alone, so each is the mean
over the steps that were applied; a window none of whose steps was applied logs `loss: nan`, with
`logging_nan_inf_filter` on too, and so does `train_loss` for a run none of whose steps was applied.

A step that saves reads the unread verdict first, so a checkpoint's optimizer state counts the applied steps
only, and a `train()` call settles one an interrupted run left before it loads any checkpoint (its warning or
raise lands there). Nothing else of the guard is checkpointed: a resumed run starts with no streak.

Router balancing biases (the bias-update modes'
[BalancingBiasCallback](../training-methods/callbacks.md#balancingbiascallback)) still update on a skipped
step. The update reads no gradient: a sign step of at most `router_balancing_rate` per bias, taken from the
expert-load counts the step's forward recorded, which are finite token counts whatever the gradients held. A
skipped step therefore moves a bias no further than an applied one and cannot make it non-finite.

Under fp16, where accelerate's optimizer wrapper steps through the loss scaler, the scaler owns overflow
steps and the guard takes no verdict; the toolkit's no-wrap prepare keeps the raw optimizer, so there no scaler
steps it and the guard stays armed. Async GRPO's trust-region
[breaker](../training-methods/async-grpo/objective.md#breaker) is the other skip path: it discards the
gradients in `on_pre_optimizer_step`, after the guard's verdict, and either one discarding them skips the
step.

## DataLoader and data parallelism

`DataParallelLoaderMixin` overrides `get_train_dataloader()` / `get_eval_dataloader()` /
`get_test_dataloader()` (the eval body, for `predict()`) to build
dataloaders with a custom DP size/rank so ranks in the same TP/CP/ETP group receive identical data.
The custom path fires when `_needs_custom_dataloader()` is True: TP, CP or ETP active, or the dataset is pre-sharded per DP rank. EP alone does not trigger
it (EP is orthogonal to DP) unless the dataset is pre-sharded.

Some trainers diverge: SMPO sets custom tokenized signature columns; Classification defaults
`remove_unused_columns=False`; online and async GRPO scale the batch by `steps_per_generation`
and rebuild TRL's `RepeatSampler` at the DP consumption rate
([batch geometry](../training-methods/grpo/online.md#data-flow-and-batch-construction));
Offline GRPO uses `MultiGroupSampler`.

`ParallelismConfig` computes both: `data_parallel_size`
(see [Stacking the dimensions](../parallelism/README.md#stacking-the-dimensions)) and, through
`get_data_parallel_rank()`, each rank's shard index per mode
([per-mode derivation](../parallelism/data-loading.md)).

`_prepare_dataloader()` passes the computed size/rank to
`accelerate.prepare_data_loader()` as `num_processes` / `process_index`. A dataset already sharded
per DP rank passes `1` / `0` instead, so accelerate places batches on the device without re-sharding
away `(N-1)/N` of each slice; offline GRPO passes the same pair, its `MultiGroupSampler` having
already sharded. accelerate's dispatching loader (its default for an iterable dataset) ignores both
and slices rank 0's batches by global rank, so a loader sharded fewer ways than the world never
dispatches: an iterable dataset is sharded by DP rank instead, and an explicit
`accelerator_config.dispatch_batches: true` raises. Sharded that way, an iterable eval split's final
round repeats its first rows with no length to count them by, so its metrics count them (the eval
loader warns).

accelerate's even-batches shard fills every rank's last eval batch with the split's first rows, so
all ranks run the same rounds; with `even_batches` off the last batch is short instead, while HF
still repeats each rank's scalar loss `eval_batch_size` times. The split's own rows fill that final
round in rank order, and the loader's `BatchSamplerShard` says how many: the rows its sampler draws
(`num_generations` per prompt under TRL's `RepeatSampler`) modulo one round
(`final_round_split_rows`). That count is trusted only
for torch's `BatchSampler` iteration over a sampler whose length is its rows (sequential, random,
subset, group-by-length, `RepeatSampler`), which sentence-transformers' default batch sampler is.
A batch sampler with its own iteration counts off its dataset only where its trainer declares it
draws every row once (`DATASET_DRAWING_BATCH_SAMPLERS`: the embedding trainer's no-duplicates
sampler, exact while only its last batch is short). Any other is not counted, so nothing is cut:
group-by-label draws fewer rows than its dataset holds.

`eval_split_rows(n)` gives a rank its share of a batch of `n`, identical across TP/CP/ETP siblings.
The evaluation gather installed on every trainer cuts the gathered final round by the same
geometry, per rank: each rank's chunk keeps the head standing for its own real rows, rounded up, so
a rank with any real row keeps its loss even where HF's loss repeat is narrower than the loader
batch. It runs after keeping one chunk per DP rank wherever the loader is DP-sharded; a loader
sharded any other number of ways (one built around `get_eval_dataloader` / `get_test_dataloader`)
raises on its first batch. The object gather (`eval_use_gather_object`, or a non-tensor input) keeps
the same ranks and cuts each one's entries the same way. accelerate's own `remainder` counts the
dataset instead, which miscounts a `RepeatSampler` and trims real rows from a one-process loader
(pre-sharded data, offline GRPO). A half-padded rank's scalar loss still averages over its padding,
except where the eval loss reads `eval_split_rows` (SMPO, classification, teacher and self
distillation). SFT's eval loss is TRL's
model-forward mean over rows that packing and padding-free collation flatten across examples, so no
row cut reaches it.

When the eval dataloader yields a different number of batches per rank (or has no length),
`evaluate()` replaces the per-step metric gather with the identity to avoid a deadlock, and warns:
the eval loss and `compute_metrics` values it logs are then rank 0's shard, not the global value. A dict of
named splits gets this verdict split by split.

## Training loop integration

HuggingFace's `_inner_training_loop` calls `accelerator.prepare(model)`, which normally wraps in
DDP. `AcceleratorMixin` replaces `prepare()` with `no_wrap_prepare` for model/optimizer/scheduler when
`_fsdp_wrapped or _qlora_grad_sync or _mixin_manages_ep_layers or is_tp_mode or is_expert_tp_mode or is_pp_mode`.
`_mixin_manages_ep_layers` is `needs_ep_layers and not accelerate-managed`; `needs_ep_layers` alone
defaults True even for dense models, so this guards against hijacking an accelerate-managed DDP/FSDP
launch.

In CP mode `DistributedSFTTrainer` disables HuggingFace's `num_items_in_batch` loss scaling
(`model_accepts_loss_kwargs = False`): CP ranks process chunks of the same sequence and FSDP
averaging already gives correct gradients, so without this both loss and `grad_norm` are inflated by
`cp_size`.

HuggingFace infers that flag from the model's forward signature alone — any `**kwargs` makes it True
— and then drops its own `/gradient_accumulation_steps` division. A trainer whose `compute_loss`
returns the batch's own mean therefore declares `LOSS_IS_OWN_MEAN`, which `_configure_distributed_modes`
applies once for every trainer that declares it. It also sets `loss_is_scaled_for_ga = False`, since a
`compute_loss_func` drops the same division.

## SFT loss and metrics

`DistributedSFTTrainer` calls base `Trainer.compute_loss`, not `SFTTrainer.compute_loss`, which gathers and reads
its metrics on every micro-batch and assumes the logits cover the whole sequence. It forwards the inputs TRL's
version prepares (`_add_trl_forward_inputs`; `tests/cpu/trainers/test_sft_trl_loss_inputs.py` pins them to the
installed TRL) and records accuracy and entropy over the supervised tokens, Liger's fused token accuracy, the
attended tokens and the router aux loss into a per-mode `WorldMetrics` window (`_record_loss_metrics`) that
flushes once per log. Under CP, `UlyssesModel.forward()` owns the boundary handling and the globally-normalized
sum loss ([the two corrections](../parallelism/context-parallelism.md)), and the recorder mirrors the boundary
handling on the rank's chunk; under its [fused head loss](../parallelism/context-parallelism.md#fused-head-loss) it
weighs the chunk's `token_accuracy` by the chunk's own supervised targets. A run with an [MTP head](../optimization/mtp-heads.md#objective) also records `ntp_ce` and `mtp_ce`, and at an evaluation [its acceptance](../optimization/mtp-heads.md#acceptance). Every rank records its whole batch's attended tokens, and the flush divides the
world sum by the ranks sharing a batch (`non_dp_replication_factor`). A subclass whose own loss records none of
these declares `RECORDS_LOSS_METRICS = False` and skips the flush (self-distillation, which stores its terms through
`StoredMetricsMixin`).

## Trainer-specific notes

- **SMPOTrainer** — reference-free preference optimization; loss types `sigmoid`, `hinge`,
  `ipo`, `smooth_lower_bound`; margin scheduling via `VariableSchedulerCallback`; CP via cross-rank
  all-reduce of log-prob sums and token counts. See [SMPO](../training-methods/preference/smpo.md).
- **OfflineGRPOTrainer** — advantage methods `z_norm`, `minmax`, `quantile_norm`,
  `quantile_uniform`, `robust`; PG formulations `prob_weighted` or `reinforce`. `LogprobScoringMixin`
  (`logprob_scoring.py`) owns its log-prob path.
- **DistributedDPOTrainer** — the reference model is never parallelized: under EP/TP a frozen copy is a
  whole dense replica per rank (warned); `precompute_ref_log_probs=True`, or PEFT/LoRA with
  `ref_model=None` (LoRA works under EP, not TP), avoids it.
- **DistributedTeacherDistillationTrainer** — losses `kl_divergence`, `reverse_kl`, `mse`,
  `soft_cross_entropy`, `cosine_similarity`, `jensen_shannon`, `slim`; the teacher forward runs under
  `torch.no_grad()` in `eval()` mode.
- **DistributedAsyncEnvironmentalGRPOTrainer** — async multi-turn RL with Ray actors against vLLM or
  SGLang servers (`rollout_backend`); rollout generation overlaps training; environments resolved by
  `environment_type` through `halo/environments/registry.py`. See
  [Async GRPO](../training-methods/async-grpo/README.md).

The `halo/trainers/grpo/` package keeps the online trainer (`online.py`), the `offline/` and `environmental/`
packages, and these shared leaves at the top level: `on_policy.py`, the constructor spine
(`OnPolicyGRPOInitMixin`) and generation-round geometry (`GenerationRoundMixin`) both on-policy trainers
compose; `grpo_rules.py`, their construction refusals as pure `reject_*` functions called ahead of TRL's
constructor; `early_stop.py`; `phase_timing.py` (`PhaseTimedMixin`, the phase spans and their report at each log) and
`row_span_stash.py` (`RowSpanStashMixin`, the round's row spans), both composed by the on-policy trainers ahead of
`DistributedTrainerMixin`; `rounds/packed_rounds.py` (`PackedRoundMixin`, the packed micro-batches' plan, slot layout and bin-by-bin
scoring), `rounds/packed_rules.py` (their refusals), `rounds/microbatch_plan.py` and `rounds/row_order.py` (the row permutation and the loss types
that admit a regrouping of rows), which both on-policy trainers share; and `logprob_scoring.py`, the log-prob scoring every GRPO trainer composes (the chunked sweep over
`halo/kernels/head_logprobs.py`, the full-logits memory guard, the protected-token entropy mask, and
`_reference_policy`, the model the KL reference forward runs: a held `ref_model`, else the PEFT policy with
its adapters disabled). Support code lives in `objective/` (pure loss-side functions) and `rollout/`
(`weight_push.py`, the push and its construction gate; `engine_preflight.py`, the COLLECTIVE rollout-server
preflights; `trl_vllm_client.py`; and `completions_logging.py`, whose `DecoupledCompletionsLogMixin` both
on-policy trainers compose to write the completions record under `save_completions` and the console table
under TRL's `log_completions`). `offline/` holds `trainer.py`, `offline_rules.py` (its construction refusals),
`group_rows.py` (one tokenized row per stored completion, carrying its advantage), `group_sampler.py`,
`pipeline_loss.py`, `reference.py` (the reference mode, the run-start
sweep, the checkpoint payload and evaluation reuse) and `reference_cache.py` (the storage of its
frozen-reference scores).

`environmental/` holds the async GRPO trainer, one concern per module: `trainer.py`
(`DistributedAsyncEnvironmentalGRPOTrainer`: construction, the round entry, `compute_loss`, `log`),
`environmental_rules.py` (the pure `reject_*` construction refusals), `step_batch.py` (`StepBatchMixin`:
`_build_training_tensors` as named phases in their collective order, `TrajectoryRowMap`, `StepRows`, `StepScores`, the
`BatchBuildFence`), `loss/importance_sampling.py` (`ImportanceSamplingMixin`: the engine-to-trainer ratio, its mask
stages, the engine re-score and the trust-region breaker), `reasoning.py` (`ReasoningBudgetMixin`: the per-level
thinking contract and the rollout rewards with the reasoning length terms charged), `routing_replay.py` (the
routing-replay injector and `RoutingReplayMixin`), `rollout_routing.py` (the round's ragged per-row routing, R2's captured or R3's engine), `async_rollouts.py` (`AsyncRolloutMixin`: Ray actors, the prefetch
thread, engine weight sync), `trajectory_tokenize.py` and `trajectory_spans.py` (`TrajectoryTokenizeMixin`:
trajectory → training rows), and `rollout_metrics.py` (`RolloutMetricsMixin`: completion logs and per-episode
diagnostics).

### Online GRPO vLLM weight sync

`DistributedOnlineGRPOTrainer._install_weight_sync` replaces TRL's
`VLLMGeneration.sync_weights` with `_distributed_sync_weights` at construction, and raises when TRL
built no `vllm_generation` (server-mode vLLM is required). Before it, the init spine both on-policy
trainers close through (`OnPolicyGRPOInitMixin._finish_on_policy_init`) runs
`reject_unsyncable_model` against the trainer's engine.
TRL builds `TrlVLLMClient` (`halo/trainers/grpo/rollout/trl_vllm_client.py`), TRL's `generate`
contract over the vLLM weight-sync client, in place of its own `VLLMClient`, which would import the
vLLM package ([generation](../training-methods/grpo/online.md#server-mode-only)).

TRL's default only unfolds DTensors when its `DistributedBackend.is_fsdp` flag is set (an accelerate
`fsdp_plugin`). FSDP2 is applied manually via `fully_shard` with accelerate in MULTI_GPU mode, so
the default forwards DTensors verbatim, and the `torch.cat` in `pack_tensors` triggers
a DTensor dispatch that deadlocks against the trainer↔vLLM NCCL group.

`_generate_single_turn` broadcasts the per-rank rollout result from the TP-group leader; without it
TRL slices the broadcast by `process_index`, each TP rank lands on different completions, and the
first forward deadlocks on its first all-reduce.

`_distributed_sync_weights` calls `sync_trainer_weights`
(`halo/trainers/grpo/rollout/weight_push.py`, also used by the async GRPO trainer), which gathers,
sends, and closes the engine's update with `finish_weight_update` — see
[the gather](../training-methods/grpo/online.md#weight-sync) for what it collects.
EP layers are found by `isinstance(module, EPMoELayerBase)` rather than an `ep_config` probe (a PEFT
`modules_to_save` wrapper forwards `__getattr__` and would match the wrapper too), and PEFT/LoRA is
folded out of place into each base weight as it is sent, under base-model param names, without writing
the frozen base ([PEFT](../optimization/peft.md#online-rl--rollout-server-weight-sync)).

## FSDP2 output capturing

Transformers models capture auxiliary outputs (router logits, hidden states, attention weights) via
a `capture_outputs` decorator keyed on `_CAN_RECORD_REGISTRY` (`str(class)` → capturable flags).
FSDP2 `fully_shard` creates dynamic subclasses at wrap time (`GptOssModel` → `FSDPGptOssModel`) that
are not in the registry, so capture silently fails.

`_register_output_capturing_for_fsdp()` walks the wrapped model, finds modules with `_can_record_outputs` not yet registered, and adds them. It
runs from `_configure_distributed_modes()` only when `_fsdp_wrapped=True` (the torchrun path);
accelerate-managed FSDP v1 does not create dynamic subclasses and is unaffected.

## Checkpoint save and load

The save flow, the saver ladder, the on-disk formats and every resume path are on
[Checkpoints & Resume](checkpoints.md). On the trainer side, `_checkpoint_context` is the one place a
saver's trainer state is read, and the ladder reads rank-uniform state only, so every rank picks the
identical saver (`None` = fall through to `Trainer.save_model`).

Weight resume goes through `load_resume_weights()` over `_weights_load_context()`, and the per-rank
optimizer shards through `OptimizerShardStore` (`halo/checkpoint/parallel/`) over
`_checkpoint_load_context()`; where `load_resume_weights()` declines, `_load_weights_through_base` runs
the base Trainer's own load, and `_load_best_model` routes through `_load_from_checkpoint` too. A trainer
whose saves write a sub-module's names overrides the two hooks (the embedding trainer: the backbone,
and sentence-transformers' own reload).

Around it the mixin orchestrates the non-weight parts of a checkpoint: `_save_checkpoint` (which
defers `save_total_limit` rotation until the new checkpoint is complete, and keeps the base's
`optimizer.pt` out of a save whose shard store writes per-rank shards), and the LR-scheduler and
`router_balancing_biases.pt` sidecars of `halo/checkpoint/parallel/sidecars.py`.
`_persist_trainer_sidecars` is a trainer's own hook, called on every rank before rotation, and
`_restore_trainer_sidecars` its read-back, called on every rank of a resume (never a best-model
load). The shared reference mixin overrides the write for DPO/KTO and offline GRPO's `reference_logps.pt`.
DPO/KTO read it back during TRL's `__init__`, and offline GRPO restores it during construction,
not through the resume hook. The async GRPO rollout mixin overrides both for its pending prefetch
rounds. Each trainer lists the overriding mixin ahead of `DistributedTrainerMixin` in its bases,
so the empty defaults do not shadow it.

`load_best_model_at_end` is refused at construction for every shape whose end-of-run reload is
guaranteed to be refused: `cp_size > 1`, a MoE carrying EP layers (`ep_size: 1`
plain FSDP2 included, since `use_grouped_gemm` defaults on), and `tp_size > 1` with
`data_parallel_size > 1`. The reload would otherwise fail after the whole run and export the LAST
weights.

Adapter-only runs reload in place and are exempt — unless `merge_expert_lora_on_save` folds the
adapter into a full base checkpoint, which the loader refuses like any other.

## Related pages

- [Checkpoints & Resume](checkpoints.md) · [Pipeline Parallelism](../parallelism/pipeline-parallelism.md)
- [Expert](../parallelism/expert-parallelism.md) · [Tensor](../parallelism/tensor-parallelism.md) · [Context](../parallelism/context-parallelism.md) Parallelism
- [SMPO](../training-methods/preference/smpo.md) · [Offline GRPO](../training-methods/grpo/offline.md) · [RLVR Online GRPO](../training-methods/grpo/online.md) · [Async GRPO](../training-methods/async-grpo/README.md)
