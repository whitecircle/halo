# Doc ownership — `src/` area → owning doc page(s)

When a `src/` path changes in a way that affects behavior, defaults, supported
configurations, or APIs, update the owning page(s) below in the same change. This
is the doc-actualization discipline: a changed `src/` path must not leave its doc
stale. Derived from the `CLAUDE.md` source tree and the Documentation index.

Always also check whether the change affects: the section **README.md** index, the
**Documentation index** in `CLAUDE.md`, the **Configuration Reference**
(`agent-docs/reference/configuration-reference.md`), and the **Trainer Architecture**
page (`agent-docs/reference/trainer-architecture.md`) when a trainer or the mixin
changes.

## Trainers

| `src/` area | Owning doc page(s) |
|---|---|
| `src/trainers/mixins/base.py` (DistributedTrainerMixin) | `agent-docs/reference/trainer-architecture.md` + every parallelism page under `agent-docs/parallelism/` |
| `src/trainers/mixins/` (sub-mixins: checkpointing (save/resume orchestration), dataloader, EP introspection, grad sync/clip, validation, pipeline, reference_logps, stored_metrics; the loss_masks helpers) | `agent-docs/reference/trainer-architecture.md`; `agent-docs/reference/checkpoints.md` (checkpointing, the `reference_logps.pt` sidecar), `agent-docs/parallelism/pipeline-parallelism.md` (pipeline) |
| `src/trainers/parallelism/pipeline_loss_adapter.py` (`PipelineLossAdapter`, the trainer's pipeline-loss contract, and the pair layout) | `agent-docs/parallelism/pipeline-parallelism.md`, `agent-docs/reference/trainer-architecture.md` |
| `src/trainers/parallelism/trainer_rules.py` (the trainer-construction PP and CP refusals, incl. the named-optimizer one) | `agent-docs/parallelism/pipeline-parallelism.md`; `agent-docs/parallelism/context-parallelism.md`; `agent-docs/reference/trainer-architecture.md` |
| `src/trainers/mixins/trl_dataset_prep.py` (CoordinatedTRLDatasetPrepMixin) | `agent-docs/data/filesystem-handling.md` (the coordination contract); `agent-docs/reference/trainer-architecture.md` (one line + link) |
| `src/trainers/mixins/token_metrics.py` (loss-token accumulation, per-document attention work behind `bind_attention_work_source`) | `agent-docs/training-methods/callbacks.md`, `agent-docs/optimization/throughput-benchmarks.md` |
| `src/trainers/sft.py` | `agent-docs/training-methods/sft.md`, `agent-docs/training-methods/pretraining.md` |
| `src/trainers/preference/` (DPO, SMPO, KTO) | `agent-docs/training-methods/preference/dpo.md`, `agent-docs/training-methods/preference/smpo.md`, `agent-docs/training-methods/preference/kto.md` |
| `src/trainers/preference/logprobs.py` (`FP32LogprobsMixin`: fp32 sequence log-probs on TRL's DPO/KTO loss and reference passes), `precompute.py` (the rank-consistent reference precompute and the precision each split records) | `agent-docs/training-methods/preference/dpo.md` (Log-prob precision, Resuming a precompute run), `agent-docs/training-methods/preference/kto.md` |
| `src/trainers/grpo/` (online, offline, environmental) | `agent-docs/training-methods/grpo/{README,online,offline}.md`, `agent-docs/training-methods/async-grpo/*.md` |
| `src/trainers/grpo/reasoning_terms.py` (the reasoning length terms the environmental trainer charges) | `agent-docs/training-methods/async-grpo/reasoning.md` (Reasoning length reward), `agent-docs/reference/configuration-reference.md` |
| `src/trainers/grpo/mixins/offline_reference.py`, `src/trainers/grpo/reference_cache.py` (offline GRPO's run-start reference scores, their bounded cache and the `reference_logps.pt` sidecar) | `agent-docs/training-methods/grpo/offline.md` (Reference model), `agent-docs/reference/checkpoints.md` |
| `src/rewards/` (`terms.py` the config-parsed reward terms, `samples.py` the sample and its views, `scorers/` the generative judge — scoring or veto — and the served reward model, `composer.py` the scorer catalog and settlement, `trl_reward_functions.py` the TRL adapters, `graders/` the RLVR graders and answer matching) | `agent-docs/training-methods/grpo/rewards/{README,judge,reward-model}.md` (each scorer's metrics on its own page), `agent-docs/training-methods/grpo/online.md`, `agent-docs/training-methods/async-grpo/monitoring.md`, `agent-docs/training-methods/async-grpo/environments/README.md`, `agent-docs/reference/configuration-reference.md` |
| `src/trainers/reward/` | `agent-docs/training-methods/preference/reward-modeling.md`, `agent-docs/training-methods/classification.md` |
| `src/trainers/distillation/` | `agent-docs/training-methods/distillation/{README,teacher-distillation,self-distillation,online-sdpg}.md` |
| `src/trainers/embedding/` (SBERT trainer, `sentence_transformers_compat.py` patches + preloaded-model shim) | `agent-docs/training-methods/embedding.md` |
| any new trainer | `agent-docs/reference/trainer-architecture.md` + new method page + the section `README.md` + `CLAUDE.md` index |

## Distributed / parallelism

| `src/` area | Owning doc page(s) |
|---|---|
| `src/distributed/parallelism_config.py` (ParallelismConfig), `parallelism_rules.py` (axis allowlist, refusals) | `agent-docs/parallelism/*` (all; the combination table in `agent-docs/parallelism/README.md#supported-combinations`), `agent-docs/reference/configuration-reference.md`, the parallelism matrix in `CLAUDE.md` |
| `src/distributed/expert_parallel/` (EP topology, per-family layers; `moe_layer/`: `EPMoELayerBase` and its compute, routing, grad-sync, expert-LoRA (with the adapter gather/load), checkpoint-gather and balancing-bias mixins, `named_ep_layers`) | `agent-docs/parallelism/expert-parallelism.md`, `agent-docs/parallelism/expert-tensor-parallelism.md`, `agent-docs/optimization/grouped-gemm.md`, `agent-docs/infrastructure/deepep.md`, `agent-docs/models/adding-a-model.md` (the `_adopt_*` construction hooks), `agent-docs/reference/architecture.md` |
| `src/distributed/expert_parallel/deepep/` (DeepEP knobs, buffer sizing and the capacity scope, buffer backends, dispatcher), `src/distributed/rank_uniform_env.py` (the job-wide env agreement) | `agent-docs/infrastructure/deepep.md`, `agent-docs/reference/configuration-reference.md` (DeepEP environment variables) |
| `src/models/moe/expert_lora_spec.py` (native expert-LoRA vocabulary) | `agent-docs/optimization/peft.md`, `agent-docs/models/adding-a-model.md` |
| `src/models/moe/expert_layout.py` (each family's runtime-to-hub expert layout, shared by the EP gather and `merge_ep_shards.py`) | `agent-docs/reference/checkpoints.md`, `agent-docs/parallelism/expert-parallelism.md`, `agent-docs/models/adding-a-model.md` |
| `src/distributed/context_parallel/` (Ulysses: `config`, `sequence_chunks`, `all_to_all`, `ulysses_attention`, `families`, `model_rules`, `ulysses_model`), `src/model_loading/parallel/cp.py` (CP loads) | `agent-docs/parallelism/context-parallelism.md` |
| `src/distributed/tensor_parallel/` (DTensor: `hf_plan`, `parallelize_attention`, `plain_params`), `src/model_loading/parallel/tp.py` (TP loads) | `agent-docs/parallelism/tensor-parallelism.md` |
| `src/distributed/pipeline_parallel/` (families — `PP_FAMILY_SPECS`, the registered and refused families — model_rules, partition, stage, stage_losses, groups, the `PipelineRunner` schedule engine in runner) | `agent-docs/parallelism/pipeline-parallelism.md`, `agent-docs/reference/checkpoints.md` (stage shards) |
| `src/model_loading/lazy_safetensors/` (lazy-load core shared by the EP + PP loaders: `conversion.py` op vocabulary + key alignment, `conversion_selection.py` which transformers conversions apply and the family key mapping (`family_key_mapping`), `materialization.py` weight plans + loader + shared stream tail, `meta_shell.py`) | `agent-docs/parallelism/expert-parallelism.md` (lazy loading), `agent-docs/parallelism/pipeline-parallelism.md`, `agent-docs/reference/checkpoints.md` |
| `src/distributed/expert_parallel/lazy_plan.py` (expert-key vocabulary, lazy gate, weight planner, shared prologue), `src/distributed/expert_parallel/expert_fuser.py` (per-expert fusion), `src/model_loading/parallel/ep.py` (EP loads: lazy + `from_pretrained` fallback) | `agent-docs/parallelism/expert-parallelism.md` (Model loading) |
| `src/model_loading/parallel/pp.py` (stage-aware pipeline load, `StageKeyFilter`) | `agent-docs/parallelism/pipeline-parallelism.md` |
| `src/distributed/group_layout.py`, `mesh.py` (EP/CP rank layouts, DeviceMesh construction + TP/DP accessors) | `agent-docs/parallelism/multi-node.md`, `agent-docs/parallelism/data-parallelism.md`, `agent-docs/reference/architecture.md` |
| `src/subclass_registry.py` (class-claim maps over a subclass tree: HF class → EP/CP wrapper, head transform) | `agent-docs/models/adding-a-model.md`, `agent-docs/reference/architecture.md` |
| `src/distributed/nccl/_nccl_wrapper.py` (the `ncclUniqueId` binding the vLLM image unpickles from this path) | `agent-docs/infrastructure/docker.md` (the `ncclUniqueId` ABI assert), `agent-docs/infrastructure/rollout-servers.md` |
| `src/distributed/grad_reduce.py` (bucketed gradient all-reduce — EP cross-replica, TP replicated, QLoRA sweeps) | `agent-docs/parallelism/data-parallelism.md`, `agent-docs/reference/configuration-reference.md` (`HALO_GRAD_BUCKET_MB`) |
| `src/weight_sync/` (the engine clients, the backend registry, the chunking, the multi-server `pool.py` (`WeightSyncPool`), and their NCCL transport and group-address policy under `transport/`) | `agent-docs/infrastructure/rollout-servers.md` (weight sync, NCCL transport, group rendezvous), `agent-docs/training-methods/grpo/online.md`, `agent-docs/reference/configuration-reference.md` (`VLLM_GROUP_HOST` / `SGLANG_GROUP_HOST`, `HALO_WEIGHT_SYNC_BIND_ALL`, `HALO_NCCL_SYNC_TIMEOUT_SECONDS`) |
| `src/trainers/grpo/rollout/weight_push.py` (gather, `reject_unsyncable_model`, LoRA fold, memory bracket) | `agent-docs/infrastructure/rollout-servers.md`, `agent-docs/training-methods/grpo/online.md` (construction gates), `agent-docs/optimization/peft.md` (LoRA fold), `agent-docs/training-methods/async-grpo/setup.md`, `agent-docs/reference/debugging.md` (memory bracket) |
| `src/trainers/grpo/rollout/trl_vllm_client.py` (`TrlVLLMClient`: TRL's `generate` contract over the vLLM weight-sync client, online GRPO) | `agent-docs/training-methods/grpo/online.md` (Server mode only), `agent-docs/reference/trainer-architecture.md`, `agent-docs/reference/configuration-reference.md` (`HALO_VLLM_GENERATION_TIMEOUT_SECONDS`) |
| `src/trainers/grpo/rollout/engine_preflight.py` (COLLECTIVE rollout-server preflights: context window, sampler and prompt log-probs) | `agent-docs/training-methods/async-grpo/setup.md`, `agent-docs/training-methods/grpo/online.md` |
| `src/trainers/grpo/rollout/async_rollouts.py` (Ray-actor collection, prefetch thread, engine weight-sync entry points) | `agent-docs/training-methods/async-grpo/setup.md` |
| `src/trainers/grpo/rollout/routing_replay.py` (R2/R3 capture, wire decode, engine-layer slicing) | `agent-docs/training-methods/async-grpo/objective.md`, `agent-docs/infrastructure/rollout-servers.md` |
| `src/trainers/grpo/rollout/trajectory_tokenize.py`, `trajectory_spans.py` (trajectory → training rows: whole-render spans, per-turn sampled ids) | `agent-docs/training-methods/async-grpo/objective.md` (Training rows) |
| `src/trainers/grpo/rollout/rollout_metrics.py`, `completions_logging.py` (per-episode rollout diagnostics, completion/trajectory artifacts) | `agent-docs/training-methods/async-grpo/monitoring.md`, `agent-docs/training-methods/callbacks.md` |
| `src/trainers/grpo/early_stop.py` (the shared GRPO early stop; knobs in `GRPOEarlyStopArguments`) | `agent-docs/training-methods/async-grpo/monitoring.md` (early stop), `agent-docs/training-methods/grpo/online.md`, `agent-docs/reference/configuration-reference.md` |
| `src/trainers/grpo/world_metrics.py` (`WorldMetrics` / `gathered_fractions`: step metrics folded from every rank's counts, shared by online + environmental GRPO) | `agent-docs/training-methods/async-grpo/monitoring.md`, `agent-docs/training-methods/grpo/online.md` |
| `src/trainers/grpo/objective/` (group-relative advantages, truncated log-ratio corrections, RLRR relative rewards, offline GRPO's loss-type reduction: `offline_loss_numerator` / `offline_loss_normalizer`, `LOSS_TYPES` / `PG_FORMULATIONS`) | `agent-docs/training-methods/async-grpo/objective.md`, `agent-docs/training-methods/grpo/{online,offline}.md` |
| `src/trainers/grpo/mixins/chunked_logprobs.py` (`use_chunked_grpo_logprobs` — the chunked log-prob path past the logits wall) | `agent-docs/training-methods/async-grpo/performance.md`, `agent-docs/reference/configuration-reference.md` |
| `src/environments/ray_actors.py` (actor pool, dispatch, Ray init) | `agent-docs/infrastructure/ray.md`, `agent-docs/training-methods/async-grpo/setup.md` (trainer-side knobs) |
| `src/checkpoint/parallel/save.py` (`save_checkpoint`, `save_tokenizer`, the flat saver ladder, the streamed FSDP2/CP/TP rungs) | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/trainer-architecture.md` (Checkpoint save and load), `agent-docs/parallelism/tensor-parallelism.md` (Checkpoint saving) |
| `src/checkpoint/parallel/ep.py` (the EP gathered and sharded saves, `reject_unmergeable_ep_sharded_save`) | `agent-docs/reference/checkpoints.md`, `agent-docs/parallelism/expert-parallelism.md` (Checkpointing) |
| `src/checkpoint/parallel/pp.py` (the PP stage saves and `restore_pp_stage_weights`) | `agent-docs/reference/checkpoints.md`, `agent-docs/parallelism/pipeline-parallelism.md` |
| `src/checkpoint/parallel/adapters.py` (`save_peft_adapters`, `save_expert_lora_adapters`, `restore_adapters`, the CP adapter-config respell) | `agent-docs/optimization/peft.md`, `agent-docs/reference/checkpoints.md` (PEFT (LoRA) saving) |
| `src/checkpoint/parallel/context.py` (`CheckpointContext` and its `capture`, `CheckpointLoadContext`) | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/trainer-architecture.md` |
| `src/checkpoint/format.py` (every shared on-disk name and name builder) | `agent-docs/reference/checkpoints.md` |
| `src/checkpoint/reading.py` (the layout cascade, the JSON config reader, the streaming reader, partial-checkpoint refusals, shard walks, header sizing, family sniffing) | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/scripts-reference.md` |
| `src/checkpoint/parallel/resume_weights.py` (`load_resume_weights`, the weight resume per mode) | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/trainer-architecture.md`, `agent-docs/parallelism/tensor-parallelism.md` (resume) |
| `src/checkpoint/parallel/resume_source.py` (`resolve_resume_checkpoint`, which weights a resume builds from) | `agent-docs/reference/checkpoints.md` (Checkpoint detection, Interrupted saves), `agent-docs/reference/architecture.md` |
| `src/checkpoint/parallel/reading.py` (`consensus_read`, `joined_streaming_reader`) | `agent-docs/reference/architecture.md`, `agent-docs/reference/checkpoints.md` |
| `src/checkpoint/parallel/completeness.py` (completion markers, the withheld trainer state) | `agent-docs/reference/checkpoints.md` (Interrupted saves) |
| `src/checkpoint/parallel/optimizer.py`, `src/checkpoint/parallel/optimizer_meta.py` (OptimizerShardStore, the topology fingerprint) | `agent-docs/reference/checkpoints.md` (Warm restart vs exact resume), `agent-docs/reference/troubleshooting.md`, `agent-docs/data/filesystem-handling.md` |
| `src/checkpoint/parallel/sidecars.py` (LR scheduler, router balancing biases) | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/trainer-architecture.md` |
| `src/checkpoint/hub_layout.py` (the load conversions a save reverts, `CheckpointNamespace` with its export conversions, `ModuleLayoutView`) | `agent-docs/reference/checkpoints.md` |
| `src/checkpoint/atomic.py` (exclusive staging, `staged_file`, atomic Torch-file publication, directory fsync, the staged directory publish) | `agent-docs/reference/checkpoints.md` |
| `src/checkpoint/config_export.py` (what an exported `config.json` must contain: model_type restore, flat legacy keys, source schema) | `agent-docs/reference/checkpoints.md`, `agent-docs/models/README.md` |
| `src/checkpoint/weights_source.py` (which checkpoint a live model's weights and config schema came from) | `agent-docs/reference/checkpoints.md` |
| `src/checkpoint/adapters.py` (saved-PEFT file layout, its writers and readers, resume-adapter refusals, expert-LoRA shape gates) | `agent-docs/optimization/peft.md`, `agent-docs/reference/checkpoints.md` |
| `src/checkpoint/adapter_merge.py` (the adapter input gates and the merge into a base) | `agent-docs/reference/model-merging.md`, `agent-docs/reference/scripts-reference.md` |
| `src/checkpoint/sidecars.py` (training provenance and the training-state sidecars an export carries) | `agent-docs/reference/checkpoints.md` |
| `src/checkpoint/offline_export.py` (the tools' source resolver, input gates, size preflight, non-weight copy, full-model save), `src/checkpoint/fp8_dequant.py` (streaming fp8 → bf16) | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/scripts-reference.md` |
| `src/checkpoint/writing.py` (save-dtype casts, stale-file sweeps, merged index, whole-dict and streaming shard writers, tie reconcile) | `agent-docs/reference/checkpoints.md` |
| `src/checkpoint/pp_stage_merge.py` (`merge_pp_stage_parts`: one checkpoint from a per-node PP save's node directories) | `agent-docs/reference/checkpoints.md`, `agent-docs/parallelism/pipeline-parallelism.md` |
| `src/checkpoint/model_card.py` (`HUB_TAGS`, `tag_model_card` / `tag_exported_model_card`: the `halo` Hub tag on the model card of every checkpoint Halo writes, tool conversions included) | `agent-docs/reference/checkpoints.md` (Hub model card), `human-docs/checkpoints.md` (upload) |
| `src/checkpoint/parallel/writing.py` (the collective half of a gathered write: retain-gated DTensor resolve of params and buffers with neutralized sinks, conversion-closed chunks, `stream_gathered_checkpoint`, the hand-sliced TP gather) | `agent-docs/reference/checkpoints.md`, `agent-docs/parallelism/data-parallelism.md` |
| `src/models/unwrapping.py` (wrapper peeling, plain module, PEFT and CP adapter-key names, `find_peft_model`), `src/models/param_roles.py` (embedding/head markers, norms, persistent buffers, fp32 pins), `src/models/model_classes.py` (a live model's transformers or remote-code class, the `REMOTE_CODE_MODEL_CLASS_ATTR` stamp) | `agent-docs/reference/checkpoints.md`, `agent-docs/optimization/peft.md` |
| `src/models/module_tree.py` (decoder layers, the backbone and the input-embedding owner, module paths, the attribute probe, embedding modules) | `agent-docs/parallelism/data-parallelism.md` |
| `src/models/peft_adapters.py` (PEFT adapter parameters, the out-of-place LoRA fold) | `agent-docs/optimization/peft.md` |
| `src/models/gradient_checkpointing.py` (recompute scopes, expert-load counting, the scoped GC enable) | `agent-docs/parallelism/expert-parallelism.md` |
| `src/models/head_transform.py` (per-family head transform, verified against the family forward) | `agent-docs/models/adding-a-model.md`, `agent-docs/training-methods/async-grpo/performance.md`, `agent-docs/parallelism/pipeline-parallelism.md` |
| `src/model_loading/parallel/frozen_models.py` (unsharded teacher / reference loads + freeze) | `agent-docs/training-methods/distillation/teacher-distillation.md`, `agent-docs/training-methods/preference/dpo.md` |
| `src/model_loading/parallel/fp32_masters.py` (configured FP32 parameter masters: selection, lazy capture, coverage check, pre-wrap checkpoint restore incl. native TP storage) | `agent-docs/models/README.md` (Load precision), `agent-docs/reference/checkpoints.md`, `agent-docs/parallelism/expert-parallelism.md` (Model loading) |
| `src/model_loading/parallel/model_source.py` (the agreed `ModelSource`: main-first fetch, presence + commit agreement; the rank-0 lazy-checkpoint verdict; `probe_checkpoint`, the agreed modality probe) | `agent-docs/parallelism/multi-node.md` (Model loading), `agent-docs/data/filesystem-handling.md`, `agent-docs/reference/troubleshooting.md` |
| `src/entrypoints/model_setup.py` (the policy load and processing class — `load_policy_model`, `load_model_and_tokenizer` — the tokenizer window pins, the padded-attention default) | `agent-docs/training-methods/sft.md` (Vision-language models), `agent-docs/models/README.md`, `agent-docs/reference/configuration-reference.md` |
| `src/distributed/fsdp.py` (FSDP2 wrapping + reshard) | `agent-docs/parallelism/data-parallelism.md`, `agent-docs/reference/checkpoints.md` |
| `src/runtime/` (ranks, process group and its timeouts, collectives incl. `combine_across_ranks`, cross-rank consensus and c10d-store joins, shared-FS scope and main-first ordering, output-FS probe, load throttle, DTensor access) | `agent-docs/parallelism/multi-node.md`, `agent-docs/data/filesystem-handling.md`, `agent-docs/reference/architecture.md`, `agent-docs/reference/debugging.md` (`assert_rank_uniform`) |
| `src/distributed/nvlink.py` (fabric probes behind `nvlink_domain_size`) | `agent-docs/parallelism/multi-node.md`, `agent-docs/infrastructure/deepep.md` |
| `src/diagnostics/profiling.py` (torch-profiler construction and traces, CUDA memory reads and snapshots), `src/diagnostics/py_spy.py` (py-spy capture) | `agent-docs/reference/debugging.md`, `agent-docs/optimization/throughput-benchmarks.md` |
| `src/weight_sync/transport_check.py` + `scripts/profiling/weight_sync_transport.py` (weight-sync transport check) | `agent-docs/infrastructure/rollout-servers.md` (Servers on other nodes), `agent-docs/reference/scripts-reference.md`, `agent-docs/reference/debugging.md` |
| `src/diagnostics/ep_span_timing.py` (opt-in EP span timing) | `agent-docs/optimization/throughput-benchmarks.md`, `agent-docs/reference/debugging.md` |

## Models

| `src/` area | Owning doc page(s) |
|---|---|
| EP layer wrappers `src/distributed/expert_parallel/families/<family>.py` + `roster.py`, the family specs `src/models/moe/family_specs.py` | the per-family page under `agent-docs/models/` + `agent-docs/parallelism/expert-parallelism.md` (supported-models table) + `agent-docs/optimization/grouped-gemm.md` (supported-models table) |
| `src/distributed/expert_parallel/families/zaya.py`, `src/models/zaya.py` | `agent-docs/models/zaya.md` |
| `src/model_loading/parallel/loader.py`, `src/model_loading/parallel/load_rules.py` (every load refusal before a weight read, `reject_unsupported_load` included), `src/model_loading/parallel/whole_model.py`, `src/model_loading/model_preparation.py` (`apply_family_patches`, `apply_run_model_policies`), `src/model_loading/remote_code.py` (remote-class funnel, v4 shims, whole-file module-cache copies, the list-tie refusal) | `agent-docs/models/README.md`, `agent-docs/models/adding-a-model.md`, the affected per-family page; `agent-docs/optimization/flash-attention.md` (the FA4 and flex warm-up) |
| `src/models/seq_cls_heads.py` (Gemma 4 + MoE Qwen3.5/3.6 seq-cls heads, registered by an import in `src/model_loading/model_preparation.py`, and the multimodal seq-cls roster) | `agent-docs/training-methods/classification.md`, `agent-docs/training-methods/preference/reward-modeling.md` |
| `src/model_loading/tokenizer_setup.py` (processing class, chat template, special tokens, the `model_max_length` pin, `resolve_peft_processing_class`), `src/models/context_window.py` (the context window `max_length: null` resolves to, `is_bounded_length`, `bounded_row_length`) | `agent-docs/getting-started/configuration.md`, `agent-docs/reference/configuration-reference.md` |
| `src/model_loading/buffer_repair.py` (`finalize_loaded_model`: the non-persistent buffer repair every load path ends in) | `agent-docs/models/README.md`, `agent-docs/models/adding-a-model.md` |
| `src/kernels/hub_kernel_dispatch.py` (device-aware repair of transformers' kernel-fallback dispatch, installed at package import by `src/__init__.py`) | `agent-docs/models/README.md`, `agent-docs/models/qwen3_5.md` |
| `src/model_loading/checkpoint_coverage.py` (random-init load gate) | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/troubleshooting.md` |
| `src/model_loading/dtype.py` (run dtype, the loaded-parameter cast: fp32 pins → run dtype, fp8 refused; fp32-matmul precision) | `agent-docs/models/README.md` (Load precision), `agent-docs/reference/configuration-reference.md` (`HALO_FP32_MATMUL_PRECISION`) |
| `src/models/config_access.py` (composite-config field access, run-scoped writes, `config_export_ready`, the embedding-tie readers) | `agent-docs/models/README.md`, `agent-docs/reference/checkpoints.md`, `agent-docs/training-methods/callbacks.md` |
| `src/models/modality.py` (pure multimodal verdict off a config or a name) | `agent-docs/data/dataset-formats.md`, `agent-docs/models/README.md` |
| `src/models/attention/layout.py` (head-dim and KV-head resolution across composite/per-layer configs, and the per-layer attention cost rules off `layer_types` — the MFU attention term), `src/callbacks/model_flops.py` (`RankFlopsModel`, `estimate_rank_flops`: the per-rank training FLOPs behind MFU / S-MFU) | `agent-docs/models/README.md`, `agent-docs/optimization/flash-attention.md`, `agent-docs/training-methods/callbacks.md`, `agent-docs/optimization/throughput-benchmarks.md` |
| `src/models/segment_markers.py` (per-family conv / linear-attention segment markers, the GatedDeltaNet kernel refusal) | `agent-docs/data/collators.md` (Document isolation under packing), `agent-docs/optimization/padding-free-collator.md`, `agent-docs/training-methods/preference/smpo.md` |
| new model support | new `agent-docs/models/<family>.md` + every family roster listed in step 8 (*Wire the docs*) of `agent-docs/models/adding-a-model.md` |

## Collators & data

| `src/` area | Owning doc page(s) |
|---|---|
| `src/data/collators/` (CompletionsOnly, packing, padding-free) | `agent-docs/data/collators.md`, `agent-docs/optimization/padding-free-collator.md` |
| `src/data/collators/offline_grpo.py`, `src/data/collators/vlm_preference.py` + `src/data/pipeline/preferences.py` (per-method collators and their maps) | `agent-docs/training-methods/grpo/offline.md`, `agent-docs/training-methods/preference/reward-modeling.md` |
| `src/data/collators/factory.py` | `agent-docs/data/collators.md` |
| tree packing (`src/data/tree_layout.py`, `src/data/pipeline/tree_rows.py`, `src/data/collators/tree_packing.py`, `src/models/attention/tree_plan.py`, `src/models/attention/tree.py`) | `agent-docs/optimization/tree-packing.md`; the `tree_*` rows of `agent-docs/reference/configuration-reference.md` |
| `src/data/spans.py` (turn terminators, completion spans, the one completion-mask implementation) | `agent-docs/data/collators.md`, `agent-docs/data/dataset-preparation.md` |
| `src/data/` (loading, processing, sharding, VLM) | `agent-docs/data/{dataset-formats,dataset-preparation,filesystem-handling}.md`, `agent-docs/parallelism/data-loading.md` |
| `src/data/sources/s3_client.py`, `src/data/sources/dataset_cache.py`, `src/data/sources/paths.py`, `scripts/before_training/s3_datasets.py` (CLI) | `agent-docs/data/s3-utilities.md` |
| `src/data/pipeline/preprocessing.py` (tokenize/pack/shard bake), `src/data/pipeline/preprocessed_metadata.py` (the `metadata.json` contract), `src/data/shard_index.py` | `agent-docs/data/dataset-preparation.md`, `agent-docs/parallelism/data-loading.md` |
| `src/data/pipeline/rendered.py` (`tokenize_rendered` — special-token ownership on a rendered template), `src/model_loading/tokenizer_backend.py` (`tokenizer_backend`) | `agent-docs/data/dataset-formats.md`, `agent-docs/data/dataset-preparation.md` |
| `src/data/pipeline/vlm_dataset.py` (raw-VLM map, schema, over-length filter) | `agent-docs/data/dataset-formats.md`, `agent-docs/models/README.md` |
| `src/data/deduplication.py` (embedding + FAISS corpus dedup) | `agent-docs/reference/scripts-reference.md` |

## Kernels & optimization

| `src/` area | Owning doc page(s) |
|---|---|
| `src/model_loading/liger/` (dispatch, family specs, appliers, fused modules, the fused-loss forward) | `agent-docs/optimization/liger-kernels.md` |
| `src/optimizers/` (AdamWBF16, Muon, FlashAdamW; `stochastic_rounding.py`, the SR seeds and rounding step AdamWBF16 and Muon share) | `agent-docs/optimization/{bf16-optimizer,muon-optimizer,flash-adamw}.md` |
| `src/kernels/lowp/` (spec: `LowpSpec`, `QuantFormat`, `LOWP_RECIPES`; quantization, linear, deepgemm) | `agent-docs/optimization/low-precision-moe-kernels.md`, `agent-docs/reference/configuration-reference.md` (`lowp`) |
| `src/model_loading/parallel/lowp_compute.py` (`apply_lowp_compute`) | `agent-docs/optimization/low-precision-moe-kernels.md` |
| `src/kernels/histogram.py` (the expert-load histogram behind the `moe/*` metrics and the bias update) | `agent-docs/training-methods/callbacks.md` |
| `src/kernels/expert_gemm.py` (quant-format dispatch), `src/kernels/grouped_mm.py` (bf16 primitive) | `agent-docs/optimization/grouped-gemm.md` |
| `src/kernels/fused_glu.py` (`GluCombine`, activation probes, the `HALO_FUSED_GLU` switch) | `agent-docs/models/glm4.md` (gate), `agent-docs/models/gpt-oss.md` (clamped variant), `agent-docs/optimization/liger-kernels.md` (Fused GLU under an EP wrapper), `agent-docs/reference/configuration-reference.md` (`HALO_FUSED_GLU`), `agent-docs/reference/troubleshooting.md` (Turning off a fused kernel), `agent-docs/optimization/grouped-gemm.md` (packed GLU), `agent-docs/optimization/torch-compile.md`, `agent-docs/parallelism/expert-parallelism.md` (the EP warm-up) |
| `src/kernels/moe_permute.py` (atomic-free token permute: `build_inv_map`, `MoEGatherPermute`, the fused weighted un-permute, `MoEExpertBiasGather`, `warm_permute_kernels`) | `agent-docs/optimization/grouped-gemm.md`, `agent-docs/parallelism/expert-parallelism.md` (the EP warm-up) |
| `src/kernels/logprobs.py` (chunked fp32 log-probs: `selective_logprobs` and the logit chunk budget; `LABEL_IGNORE_INDEX` and the next-token label math — `next_token_logprobs`, `token_logprobs`, `sequence_logprobs`, `loss_token_counts_per_row` — shared by the PP last-stage losses, SMPO, and the DPO/KTO loss paths) | `agent-docs/training-methods/preference/dpo.md` (Log-prob precision), `agent-docs/training-methods/preference/smpo.md`, `agent-docs/parallelism/pipeline-parallelism.md` |
| attention selection (`src/models/attention/selection.py`: `flash_kernel`, `preferred_flash_impl`, `_detect_attention_impl`, the resolver and validator), `src/models/attention/backend_setup.py` (process-wide backend fixes, FA4 warm-up), `src/models/attention/packed_position_ids.py`, `src/kernels/jit_cache.py` (FA4 / Triton JIT caches, CuTe DSL warning filter) | `agent-docs/optimization/flash-attention.md`; `packed_position_ids.py` also `agent-docs/models/mistral4.md`, `agent-docs/models/zaya.md` |
| `src/models/attention/sdpa_flex_sliding.py` (FlexAttention on SDPA sliding-window and wide-head layers) | `agent-docs/optimization/flash-attention.md`, `agent-docs/models/gemma4.md` |
| GptOss sink policy (`src/models/attention/gpt_oss_sinks.py`) | `agent-docs/models/gpt-oss.md` (Attention sinks), the `reset_sinks`/`train_sinks` rows of `agent-docs/reference/configuration-reference.md` |
| `torch.compile` paths | `agent-docs/optimization/torch-compile.md` |
| LoRA/QLoRA (`src/model_loading/parallel/peft_setup.py`), freeze patterns (`src/model_loading/parallel/freeze_patterns.py`) | `agent-docs/optimization/peft.md` |

## Callbacks

| `src/` area | Owning doc page(s) |
|---|---|
| `src/callbacks/` (ParameterStatsCallback, GenerateExamplesCallback, EfficiencyCallback, MoEMetricsCallback, BalancingBiasCallback, VariableSchedulerCallback, TorchProfilerCallback) | `agent-docs/training-methods/callbacks.md` |
| `src/callbacks/factory.py` (`build_run_callbacks`, the roster order, `reorder_integration_callbacks_last`) | `agent-docs/training-methods/callbacks.md`, `agent-docs/optimization/throughput-benchmarks.md` |
| `src/models/moe/balancing.py` (`resolve_balancing_mode`, the export slot contract), `src/models/moe/config_fields.py` (expert-count / top-k / FFN-width spellings), `src/models/moe/router_logits.py` (declared routers), `src/models/moe/router_aux_gradient.py` (the aux-loss gradient under reentrant checkpointing) and `src/callbacks/balancing.py` (`configure_moe_balancing`, `apply_balancing_mode`, the export contract, `BalancingBiasCallback`) | `agent-docs/training-methods/callbacks.md` (MoE balancing modes), `agent-docs/models/README.md` |
| `src/hardware.py` (this rank's device, architecture predicates, `supports_grouped_mm`, GPU model detection, peak-FLOPS registry, host-RAM probe) | `agent-docs/optimization/throughput-benchmarks.md`, `agent-docs/optimization/flash-attention.md`, `agent-docs/optimization/grouped-gemm.md` (Detection and disabling) |

## Environments (Async GRPO with Environments)

| `src/` area | Owning doc page(s) |
|---|---|
| `src/environments/base.py`, `episode.py`, `registry.py` | `agent-docs/training-methods/async-grpo/environments/{README,custom}.md`, `agent-docs/training-methods/async-grpo/{episodes,reasoning}.md` (episode budgets, effort binding, unproductive turns); `agent-docs/training-methods/grpo/rewards/README.md` (What a scorer reads, Settlement) for the scoring hooks and the external-term settlement |
| `src/environments/envs/protocols/` (native, react, mcp) | `agent-docs/training-methods/async-grpo/environments/protocols/{native-tool-use,react,mcp}.md` |
| `src/environments/tasks/coding/` (swe, code_contests with its prompts and submission record, program_source, verdicts, grading, contest_datasets), `tasks/qa.py` | `agent-docs/training-methods/async-grpo/environments/tasks/{swe,qa-benchmarks}.md`, `agent-docs/training-methods/async-grpo/environments/tasks/code-contests/*.md` |
| `src/environments/sandbox/` (`base.py` the abstract executor and session, `results.py`, `languages.py`, `factory.py`, the local, bubblewrap, remote and in-process backends, `repl.py`) | `agent-docs/training-methods/async-grpo/environments/sandbox.md` |
| `src/environments/tools/` (`definitions.py`, `toolsets.py` the tool-set builders) | `agent-docs/training-methods/async-grpo/environments/protocols/native-tool-use.md`, `agent-docs/training-methods/async-grpo/environments/tasks/swe.md` |
| `src/environments/engine_wire.py` (the rollout request wire format: stop tokens, `thinking_token_budget`, reasoning effort) | `agent-docs/infrastructure/rollout-servers.md`, `agent-docs/training-methods/async-grpo/{episodes,reasoning}.md` |
| `src/environments/rollout_config.py` (`RolloutConfig`: the rollout defaults `AsyncTrainingConfig` reads, sampler presets, reasoning template variables and `ReasoningEffortChoice`) | `agent-docs/reference/architecture.md` (leaf modules), `agent-docs/reference/configuration-reference.md` (the `AsyncTrainingConfig` fields it mirrors), `agent-docs/training-methods/async-grpo/evaluation.md` |
| `src/environments/eval_runner.py` (offline episode driver + eval report/trajectory outputs) | `agent-docs/training-methods/async-grpo/evaluation.md`, `agent-docs/reference/scripts-reference.md` |
| `src/environments/tools/web_search.py` (pluggable search backends) | `agent-docs/training-methods/async-grpo/environments/tasks/qa-benchmarks.md`, `agent-docs/reference/configuration-reference.md` |

## Config & args

| `src/` area | Owning doc page(s) |
|---|---|
| `src/configs/` (the YAML/CLI dataclasses: each method family's script arguments and trainer config, the shared bundles) | `agent-docs/reference/configuration-reference.md` + the method page that owns the config |
| `src/configs/parser.py` (`HaloArgumentParser`: the `load_values` value load, toolkit defaults, the unknown-key raise) | `agent-docs/getting-started/configuration.md`, `agent-docs/reference/configuration-reference.md` |
| `src/env.py` (the `env_*` readers, `HALO_DATA_ROOT` / `data_path`), `src/__init__.py` (the import-time process defaults: `HALO_TORCH_NUM_THREADS`, `TOKENIZERS_PARALLELISM`) and any `HALO_`/`DIST_`/`VLLM_`/`SGLANG_`/`NVLINK_`/`EP_` knob wherever it is read | `agent-docs/reference/configuration-reference.md` (Environment variables), `agent-docs/infrastructure/docker.md` |
| `src/config_validation.py` (the bounds guards `require_finite` / `require_int`, `ValidatedConfig`, `differs_from_default`, the argument refusals `reject_unsupported_args` / `reject_non_default_args`) | `agent-docs/reference/configuration-reference.md` |
| `src/configs/parallelism_arguments.py` (`ParallelismArguments` → `ParallelismConfig`, the per-script CP/PP/lowp gates) | `agent-docs/parallelism/*`, `agent-docs/reference/configuration-reference.md` |
| `src/log.py` (root logging setup, CLI verbosity, `warn_once`, `KEY_PREVIEW_COUNT`, the run-log names) | `agent-docs/reference/debugging.md` |
| `src/cli.py` (`halo launch` / `halo run` surface, tool aliases) | `README.md` quick start, `agent-docs/reference/scripts-reference.md` |

## Entry-script plumbing & clients

| `src/` area | Owning doc page(s) |
|---|---|
| `src/entrypoints/run_setup.py` (output-dir validation, HF cache wiring, per-rank transformers verbosity, seed, tracking vars, resume detection, the `run.log` console tee) | `agent-docs/getting-started/configuration.md`, `agent-docs/data/filesystem-handling.md`, `agent-docs/reference/checkpoints.md`, `agent-docs/reference/debugging.md` |
| `src/entrypoints/script.py` (the `scripts/training/**` lifecycle: `parse_script_args`, `ScriptContext`, `init_training_script`, callback assembly, `run_trainer`, `with_distributed_teardown`, the rank-0 backend probe `verify_backend_on_rank0`) | `agent-docs/reference/scripts-reference.md`, `agent-docs/getting-started/configuration.md` |
| `src/entrypoints/data_setup.py` (script dataset loading, the modality verdict `resolve_vlm_run`, the TRL dataset-prep refusal and switch) | `agent-docs/reference/scripts-reference.md`, `agent-docs/parallelism/data-loading.md` |
| `src/callbacks/early_stop.py` (`StopsTrainingEarly`, the marker the script lifecycle reads to exit a stopped run non-zero) | `agent-docs/training-methods/async-grpo/monitoring.md`, `agent-docs/reference/scripts-reference.md` |
| `src/inference/endpoints.py` (endpoint defaults, key resolution, the loopback test), `src/inference/openai_client.py` (async client, one request with the upstream retry, the single-request helper), `src/inference/batch_requests.py` (the parallel resumable batch and its JSONL resume store), `src/inference/response.py` (`OpenAIResponse`, the finish-reason contract) | `agent-docs/reference/scripts-reference.md`, `agent-docs/training-methods/grpo/rewards/README.md` (Probe and retries) |

## Scripts & infrastructure

| `src/` area | Owning doc page(s) |
|---|---|
| `scripts/training/**`, `scripts/inference/**`, `scripts/before_training/**`, `scripts/after_training/**` | `agent-docs/reference/scripts-reference.md` + the relevant method/data page |
| `scripts/profiling/**` | `agent-docs/reference/debugging.md`, `agent-docs/reference/scripts-reference.md` |
| `scripts/environments/**` (env eval runners, their shared `_common.py` flags/output writer, trajectory re-grading, coding-dataset prep) | `agent-docs/reference/scripts-reference.md`, `agent-docs/training-methods/async-grpo/evaluation.md` |
| `scripts/_common.py` (flags shared across subtrees: the checkpoint tools' shard cap, Hub source block, `--dtype`/`--device_map`, `--trust_remote_code`; the OpenAI endpoint block) | `agent-docs/reference/scripts-reference.md` |
| `scripts/after_training/merge_ep_shards.py` | `agent-docs/reference/checkpoints.md` |
| `scripts/after_training/merge_pp_stages.py` | `agent-docs/reference/scripts-reference.md`, `agent-docs/reference/checkpoints.md`, `agent-docs/parallelism/pipeline-parallelism.md` |
| `scripts/after_training/quantize_to_lowp.py` | `agent-docs/optimization/low-precision-moe-kernels.md` |
| `scripts/after_training/convert_to_bf16.py` | `agent-docs/reference/scripts-reference.md`, `agent-docs/reference/checkpoints.md` |
| `scripts/after_training/merge_models.py` | `agent-docs/reference/model-merging.md`, `agent-docs/reference/scripts-reference.md` |
| `Dockerfile*`, `docker-compose*`, `docker/sglang/patches/` | `agent-docs/infrastructure/docker.md`; `Dockerfile.vllm`/`Dockerfile.sglang` + their compose files and server patches (EFA overlays included) also `agent-docs/infrastructure/rollout-servers.md` |
| `docker/vllm/patches/`, `docker/vllm/parity/`, `docker/vllm/plugins/` (server build patches, config-schema parity gate, gpt-oss parser plugins) | `agent-docs/infrastructure/rollout-servers.md`, `agent-docs/infrastructure/docker.md`; plugins also `agent-docs/models/gpt-oss.md` |
| `docker/nccl_pin.py` (the NCCL pin every image reads from `uv.lock`, the DeepEP-V2 floor) | `agent-docs/infrastructure/docker.md` |
| `docker/efa/install_efa_userspace.sh` (the EFA userspace every image shares) | `agent-docs/infrastructure/docker.md` (RDMA networking), `agent-docs/infrastructure/rollout-servers.md` (Servers on other nodes), `agent-docs/parallelism/multi-node.md` (RDMA fabrics) |
| AWS / S3 auth, `src/data/sources/s3_client.py` paths | `agent-docs/infrastructure/aws-auth.md`, `agent-docs/data/s3-utilities.md` |
| DeepEP install / NVSHMEM / CDMC notes | `agent-docs/infrastructure/deepep.md` |
| multi-node / SkyPilot / RunPod / Nomad launch | `agent-docs/parallelism/multi-node.md`, `agent-docs/infrastructure/{skypilot,runpod,nomad}.md` |
| `launcher-configs/**` (SkyPilot task YAMLs, Nomad job specs, accelerate configs) | `agent-docs/infrastructure/{skypilot,nomad}.md`, `human-docs/clusters.md` |
| `.github/workflows/**` (lint and its conventions job, docs, sensitive-label, CPU and GPU tiers), `.github/scripts/**`, `Makefile` test targets, `tests/common/hub_seed.py` | `agent-docs/infrastructure/ci.md`; tier composition lives in `agent-docs/contributing/README.md` ("Tests") |

## Cross-cutting

| Change | Owning doc page(s) |
|---|---|
| a new user-facing fail-fast raise or failure mode (config-time rejection, NCCL/DeepEP fault, OOM path) | `agent-docs/reference/troubleshooting.md` (symptom → cause → fix), plus `agent-docs/reference/debugging.md` when it needs a diagnosis helper |
| a new, moved, renamed, or deleted `src/` subpackage or top-level module | `agent-docs/reference/architecture.md` (source map) + the architecture tree in `CLAUDE.md` |
| code enters or leaves a path in `.github/sensitive-paths` (any module or package move, rename or delete) | `.github/sensitive-paths` |

## Cross-cutting pages

These pages own no `src/` area; they summarize or route. Update them when the fact they summarize
changes on its owner page.

| Page | Summarizes |
|---|---|
| `agent-docs/README.md`, `agent-docs/training-methods/README.md`, `agent-docs/training-methods/preference/README.md`, `agent-docs/data/README.md`, `agent-docs/infrastructure/README.md`, `agent-docs/optimization/README.md`, `agent-docs/reference/README.md` | the section's pages and their one-line scope |
| `agent-docs/getting-started/{installation,quickstart,choosing-a-method}.md` | the image, the first run, the method choice |
| `agent-docs/contributing/development-environment.md` | the host setup, credentials and paths |
| `agent-docs/optimization/halo-vs-stock-trl.md` | the measured toolkit-vs-TRL comparisons |
| `agent-docs/reference/{glossary,gpu-training-theory,scale-and-limitations,why-this-framework}.md` | terms, the performance model, the known limits, the design rationale |

## Human guide

Each `human-docs/` page condenses the owners on its row for a human reader. When a change alters
what a user sets, runs or can combine there, update the page with its owner. A new page goes into
`human-docs/README.md` (a cookbook into `human-docs/cookbooks/README.md`).

| `human-docs/` page | Condenses |
|---|---|
| `human-docs/cli.md` | `src/cli.py`; `agent-docs/reference/scripts-reference.md` |
| `human-docs/configuration.md` | `src/configs/parser.py`, `src/configs/`; `agent-docs/getting-started/configuration.md`, `agent-docs/reference/configuration-reference.md` |
| `human-docs/environment-variables.md` | `src/env.py` and every toolkit knob; `agent-docs/reference/configuration-reference.md` (Environment variables) |
| `human-docs/parallelism.md` | `src/distributed/parallelism_config.py`, `parallelism_rules.py`, `src/configs/parallelism_arguments.py`; `agent-docs/parallelism/` |
| `human-docs/supported-matrix.md` | every support verdict: version pins, attention backends, trainer × mode, the rollout-engine roster (`src/weight_sync/clients/`), families, parallelism modes, PEFT and quantization |
| `human-docs/clusters.md` | multi-node launch, `launcher-configs/**`; `agent-docs/parallelism/{multi-node,launch-recipes}.md`, `agent-docs/infrastructure/{skypilot,runpod,nomad}.md` |
| `human-docs/checkpoints.md` | `src/checkpoint/parallel/`, `src/checkpoint/`, `scripts/after_training/**`; `agent-docs/reference/{checkpoints,model-merging}.md` |
| `human-docs/data.md` | `src/data/`; `agent-docs/data/` |
| `human-docs/monitoring.md` | `src/callbacks/`, `src/entrypoints/run_setup.py` (the `run.log` tee, the tracking vars); `agent-docs/training-methods/callbacks.md`, `agent-docs/reference/debugging.md`, `agent-docs/training-methods/async-grpo/monitoring.md` |
| `human-docs/performance.md` | `src/kernels/`, `src/optimizers/`; `agent-docs/optimization/` (the numbers from `throughput-benchmarks.md`) |
| `human-docs/rollout-servers.md` | `src/weight_sync/`, `Dockerfile.vllm` / `Dockerfile.sglang` + their compose files; `agent-docs/infrastructure/rollout-servers.md` |
| `human-docs/troubleshooting.md` | a new user-facing failure mode; `agent-docs/reference/{troubleshooting,debugging}.md` |
| `human-docs/models.md`, `human-docs/model-integration-cost.md` | the family roster, `src/distributed/expert_parallel/families/`; `agent-docs/models/` |
| `human-docs/cookbooks/*.md` | the family's `examples/` recipes; its `agent-docs/models/` page |
| `human-docs/installation.md`, `human-docs/quickstart.md` | `Dockerfile*`, the image tags, `examples/`; `agent-docs/infrastructure/docker.md`, `agent-docs/getting-started/` |
| `human-docs/choosing-a-method.md`, `human-docs/training-methods/*.md` | `src/trainers/`, `scripts/training/**`; the method's `agent-docs/training-methods/` page |
| `human-docs/ai-tooling.md` | `skills/` |
| `human-docs/contributing.md` | `CONTRIBUTING.md`, `agent-docs/contributing/` |

## When in doubt

- Trainer/mixin behavior → `agent-docs/reference/trainer-architecture.md`.
- Any new config field → `agent-docs/reference/configuration-reference.md`.
- A change to what parallelism combinations are valid → the parallelism matrix in
  `CLAUDE.md` and the Supported combinations table in `agent-docs/parallelism/README.md`, plus the
  owning `agent-docs/parallelism/` page.
- A new page → wire it into its section `README.md` and the `CLAUDE.md` Documentation
  index, then run `./scripts/docs/check_links.sh`.
