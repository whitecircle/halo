# Doc ownership — `halo/` area → owning doc page(s)

When a `halo/` path changes in a way that affects behavior, defaults, supported
configurations, or APIs, update the owning page(s) below in the same change: a changed
`halo/` path must not leave its doc stale. Rows map paths to pages only; what each module
holds is in the module map, [architecture.md](../../agent-docs/reference/architecture.md).

Always also check whether the change affects: the section **README.md** index, the
agent index (`agent-docs/README.md`), the **Configuration Reference**
(`agent-docs/reference/configuration-reference.md`), and the **Trainer Architecture**
page (`agent-docs/reference/trainer-architecture.md`) when a trainer or the mixin
changes.

## Trainers

| `halo/` area | Owning doc page(s) |
|---|---|
| `halo/trainers/distributed_trainer.py`, `halo/trainers/construction/accelerator.py`, `halo/trainers/construction/optimizer_selection.py`, `halo/trainers/construction/training_args_overrides.py`, `halo/trainers/construction/ctor_slots.py`, `halo/trainers/loop/step_guard.py` | `agent-docs/reference/trainer-architecture.md` + every parallelism page under `agent-docs/parallelism/`; `agent-docs/optimization/bf16-optimizer.md` |
| `halo/trainers/parallelism/mode_setup.py`, `halo/trainers/parallelism/tp_leader.py` | `agent-docs/reference/trainer-architecture.md`, `agent-docs/parallelism/data-parallelism.md` |
| `halo/trainers/parallelism/adapter_rules.py` | `agent-docs/optimization/peft.md`, `agent-docs/parallelism/tensor-parallelism.md`, `agent-docs/parallelism/expert-parallelism.md` |
| `halo/trainers/reference_logps.py`, `halo/trainers/metrics/stored_metrics.py` | `agent-docs/reference/trainer-architecture.md`; `agent-docs/reference/checkpoints.md` |
| `halo/trainers/loop/checkpointing.py`, `halo/trainers/loop/dataloaders.py`, `halo/trainers/loop/evaluation.py` | `agent-docs/reference/trainer-architecture.md`; `agent-docs/reference/checkpoints.md`, `agent-docs/parallelism/data-loading.md` |
| `halo/trainers/parallelism/{grad_sync,grad_clipping,ep_layers}.py` | `agent-docs/reference/trainer-architecture.md`; `agent-docs/parallelism/expert-parallelism.md`; `agent-docs/parallelism/tensor-parallelism.md`; `agent-docs/parallelism/data-parallelism.md`; `agent-docs/optimization/bf16-optimizer.md` |
| `halo/trainers/parallelism/{pipeline_trainer,pipeline_rules}.py`, `halo/trainers/parallelism/pipeline_loss_adapter.py`, `halo/trainers/preference/smpo_pipeline_loss.py`, `halo/trainers/grpo/offline/pipeline_loss.py` | `agent-docs/parallelism/pipeline-parallelism.md`, `agent-docs/reference/trainer-architecture.md` |
| `halo/trainers/construction/trainer_rules.py` | `agent-docs/parallelism/context-parallelism.md`; `agent-docs/reference/trainer-architecture.md` |
| `halo/trainers/trl_dataset_prep.py` | `agent-docs/data/filesystem-handling.md`; `agent-docs/reference/trainer-architecture.md` |
| `halo/trainers/metrics/token_metrics.py` | `agent-docs/training-methods/callbacks.md`, `agent-docs/optimization/throughput-benchmarks.md` |
| `halo/trainers/metrics/world_metrics.py` | `agent-docs/reference/trainer-architecture.md`, `agent-docs/training-methods/async-grpo/monitoring.md`, `agent-docs/training-methods/grpo/online.md` |
| `halo/trainers/sft.py`, `halo/trainers/sft_rules.py` | `agent-docs/training-methods/sft.md`, `agent-docs/training-methods/pretraining.md` |
| `halo/trainers/preference/` | `agent-docs/training-methods/preference/dpo.md`, `agent-docs/training-methods/preference/smpo.md`, `agent-docs/training-methods/preference/kto.md` |
| `halo/trainers/preference/trl_preference.py` | `agent-docs/training-methods/preference/dpo.md`, `agent-docs/training-methods/preference/kto.md` |
| `halo/trainers/preference/preference_rules.py` | `agent-docs/training-methods/preference/dpo.md`, `agent-docs/training-methods/preference/kto.md`, `agent-docs/parallelism/pipeline-parallelism.md` |
| `halo/trainers/grpo/` | `agent-docs/training-methods/grpo/{README,online,offline}.md`, `agent-docs/training-methods/async-grpo/*.md` |
| `halo/trainers/grpo/on_policy.py`, `halo/trainers/grpo/grpo_rules.py` | `agent-docs/training-methods/grpo/online.md`, `agent-docs/parallelism/data-loading.md`, `agent-docs/reference/trainer-architecture.md` |
| `halo/trainers/grpo/environmental/reasoning.py` | `agent-docs/training-methods/async-grpo/reasoning.md`, `agent-docs/reference/configuration-reference.md` |
| `halo/trainers/grpo/offline/` | `agent-docs/training-methods/grpo/offline.md`, `agent-docs/reference/checkpoints.md` |
| `halo/rewards/` | `agent-docs/training-methods/grpo/rewards/{README,judge,reward-model}.md`, `agent-docs/training-methods/grpo/online.md`, `agent-docs/training-methods/async-grpo/monitoring.md`, `agent-docs/training-methods/async-grpo/environments/README.md`, `agent-docs/reference/configuration-reference.md` |
| `halo/trainers/reward/` | `agent-docs/training-methods/reward-modeling.md`, `agent-docs/training-methods/classification.md` |
| `halo/trainers/distillation/` | `agent-docs/training-methods/distillation/{README,teacher-distillation,self-distillation,online-sdpg}.md` |
| `halo/trainers/embedding/` | `agent-docs/training-methods/embedding.md` |
| any new trainer | `agent-docs/reference/trainer-architecture.md` + new method page + the section `README.md` |

## Distributed / parallelism

| `halo/` area | Owning doc page(s) |
|---|---|
| `halo/distributed/{parallelism_config,parallelism_rules}.py` | `agent-docs/parallelism/*`, `agent-docs/reference/configuration-reference.md` |
| `halo/distributed/expert_parallel/` | `agent-docs/parallelism/expert-parallelism.md`, `agent-docs/parallelism/expert-tensor-parallelism.md`, `agent-docs/optimization/grouped-gemm.md`, `agent-docs/infrastructure/deepep.md`, `agent-docs/models/adding-a-model.md`, `agent-docs/reference/architecture.md` |
| `halo/distributed/expert_parallel/deepep/`, `halo/distributed/rank_uniform_env.py` | `agent-docs/infrastructure/deepep.md`, `agent-docs/reference/configuration-reference.md` |
| `halo/models/moe/expert_lora_spec.py` | `agent-docs/optimization/peft.md`, `agent-docs/models/adding-a-model.md` |
| `halo/models/moe/expert_layout.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/parallelism/expert-parallelism.md`, `agent-docs/models/adding-a-model.md` |
| `halo/distributed/context_parallel/`, `halo/model_loading/parallel/cp.py` | `agent-docs/parallelism/context-parallelism.md` |
| `halo/distributed/tensor_parallel/`, `halo/model_loading/parallel/tp.py` | `agent-docs/parallelism/tensor-parallelism.md` |
| `halo/distributed/pipeline_parallel/` | `agent-docs/parallelism/pipeline-parallelism.md`, `agent-docs/reference/checkpoints.md`; the PP column and its footnotes in `agent-docs/models/README.md` and `human-docs/supported-matrix.md` |
| `halo/model_loading/lazy_safetensors/`, `halo/model_loading/checkpoint_keys.py` | `agent-docs/parallelism/expert-parallelism.md`, `agent-docs/parallelism/pipeline-parallelism.md`, `agent-docs/reference/checkpoints.md` |
| `halo/model_loading/parallel/lazy_plan.py`, `halo/model_loading/parallel/expert_fuser.py`, `halo/model_loading/parallel/ep.py`, `halo/model_loading/parallel/lazy_load.py`, `halo/models/gradient_checkpointing.py` | `agent-docs/parallelism/expert-parallelism.md` |
| `halo/model_loading/parallel/pp.py` | `agent-docs/parallelism/pipeline-parallelism.md` |
| `halo/distributed/{group_layout,mesh}.py` | `agent-docs/parallelism/multi-node.md`, `agent-docs/parallelism/data-parallelism.md`, `agent-docs/reference/architecture.md` |
| `halo/subclass_registry.py` | `agent-docs/models/adding-a-model.md`, `agent-docs/reference/architecture.md` |
| `halo/weight_sync/transport/nccl_library.py` | `agent-docs/infrastructure/docker.md`, `agent-docs/infrastructure/rollout-servers.md` |
| `halo/distributed/grad_reduce.py` | `agent-docs/parallelism/data-parallelism.md`, `agent-docs/reference/configuration-reference.md` |
| `halo/weight_sync/` | `agent-docs/infrastructure/rollout-servers.md`, `agent-docs/training-methods/grpo/online.md`, `agent-docs/reference/configuration-reference.md` |
| `halo/trainers/grpo/rollout/weight_push.py` | `agent-docs/infrastructure/rollout-servers.md`, `agent-docs/training-methods/grpo/online.md`, `agent-docs/optimization/peft.md`, `agent-docs/training-methods/async-grpo/setup.md`, `agent-docs/reference/debugging.md` |
| `halo/trainers/grpo/rollout/adapter_sync.py` | `agent-docs/infrastructure/rollout-servers.md`, `agent-docs/optimization/peft.md`, `agent-docs/training-methods/async-grpo/setup.md`, `agent-docs/reference/configuration-reference.md` |
| `halo/trainers/grpo/rollout/trl_vllm_client.py` | `agent-docs/training-methods/grpo/online.md`, `agent-docs/reference/trainer-architecture.md`, `agent-docs/reference/configuration-reference.md` |
| `halo/trainers/grpo/rollout/engine_preflight.py` | `agent-docs/training-methods/async-grpo/setup.md`, `agent-docs/training-methods/grpo/online.md` |
| `halo/trainers/grpo/environmental/async_rollouts.py` | `agent-docs/training-methods/async-grpo/setup.md` |
| `halo/trainers/grpo/packed_forward.py`, `halo/trainers/grpo/rounds/{microbatch_plan,packed_rounds,packed_rules,rank_balance,row_order}.py`, `halo/trainers/grpo/environmental/{packed_rounds,row_order}.py` | `agent-docs/training-methods/async-grpo/performance.md`, `agent-docs/reference/configuration-reference.md` |
| `halo/trainers/grpo/environmental/{trainer,environmental_rules}.py` | `agent-docs/training-methods/async-grpo/{setup,objective,reasoning}.md`, `agent-docs/reference/trainer-architecture.md` |
| `halo/trainers/grpo/environmental/step_batch.py`, `halo/trainers/grpo/environmental/loss/{importance_sampling,centered_scoring}.py` | `agent-docs/training-methods/async-grpo/{objective,monitoring}.md` |
| `halo/trainers/grpo/environmental/loss/loss_forward_weights.py` | `agent-docs/training-methods/async-grpo/{performance,objective,monitoring}.md`, `agent-docs/reference/configuration-reference.md` |
| `halo/trainers/grpo/environmental/{routing_replay,rollout_routing}.py` | `agent-docs/training-methods/async-grpo/objective.md`, `agent-docs/infrastructure/rollout-servers.md` |
| `halo/trainers/grpo/environmental/row_order.py` | `agent-docs/training-methods/async-grpo/performance.md`, `agent-docs/reference/configuration-reference.md` |
| `halo/trainers/grpo/environmental/{trajectory_tokenize,trajectory_spans}.py` | `agent-docs/training-methods/async-grpo/objective.md` |
| `halo/trainers/grpo/environmental/rollout_metrics.py`, `halo/trainers/grpo/rollout/completions_logging.py` | `agent-docs/training-methods/async-grpo/monitoring.md`, `agent-docs/training-methods/callbacks.md` |
| `halo/trainers/grpo/early_stop.py` | `agent-docs/training-methods/async-grpo/monitoring.md`, `agent-docs/training-methods/grpo/online.md`, `agent-docs/reference/configuration-reference.md` |
| `halo/trainers/grpo/objective/` | `agent-docs/training-methods/async-grpo/objective.md`, `agent-docs/training-methods/grpo/{online,offline}.md` |
| `halo/trainers/grpo/logprob_scoring.py`, `halo/kernels/head_logprobs.py` | `agent-docs/training-methods/async-grpo/performance.md`, `agent-docs/reference/configuration-reference.md`; the compiled tiles also `agent-docs/optimization/torch-compile.md` |
| `halo/environments/rollouts/ray_actors.py`, `halo/environments/rollouts/rollout_pool.py` | `agent-docs/infrastructure/ray.md`, `agent-docs/training-methods/async-grpo/setup.md` |
| `halo/checkpoint/parallel/save.py`, `halo/checkpoint/parallel/resume_weights.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/trainer-architecture.md`, `agent-docs/parallelism/tensor-parallelism.md` |
| `halo/checkpoint/parallel/ep.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/parallelism/expert-parallelism.md` |
| `halo/checkpoint/parallel/pp.py`, `halo/checkpoint/tools/pp_stage_merge.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/parallelism/pipeline-parallelism.md`, `agent-docs/reference/scripts-reference.md`, `agent-docs/parallelism/multi-node.md` |
| `halo/checkpoint/parallel/adapters.py`, `halo/checkpoint/adapters.py` | `agent-docs/optimization/peft.md`, `agent-docs/reference/checkpoints.md` |
| `halo/checkpoint/parallel/context.py`, `halo/checkpoint/parallel/sidecars.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/trainer-architecture.md` |
| `halo/checkpoint/{format,hub_layout,weights_source,sidecars,writing}.py` | `agent-docs/reference/checkpoints.md` |
| `halo/checkpoint/atomic.py`, `halo/checkpoint/parallel/completeness.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/scripts-reference.md`, `agent-docs/parallelism/expert-parallelism.md`, `agent-docs/parallelism/pipeline-parallelism.md` |
| `halo/checkpoint/reading.py`, `halo/checkpoint/tools/offline_export.py`, `halo/checkpoint/tools/fp8_dequant.py`, `halo/checkpoint/tools/deepseek_v4_dequant.py`, `halo/checkpoint/tools/sink_reset.py`, `halo/checkpoint/tools/expert_unfuse.py`, `halo/checkpoint/tools/vision_reattach.py`, `halo/checkpoint/tools/vocab_patch.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/scripts-reference.md` |
| `halo/checkpoint/parallel/resume_source.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/architecture.md` |
| `halo/checkpoint/parallel/disk_budget.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/architecture.md` |
| `halo/checkpoint/parallel/reading.py` | `agent-docs/reference/architecture.md`, `agent-docs/reference/checkpoints.md` |
| `halo/checkpoint/parallel/optimizer.py`, `halo/checkpoint/parallel/optimizer_meta.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/troubleshooting.md`, `agent-docs/data/filesystem-handling.md` |
| `halo/checkpoint/config_export.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/models/README.md` |
| `halo/checkpoint/tools/adapter_merge.py`, `halo/checkpoint/tools/adapter_stream_merge.py` | `agent-docs/reference/model-merging.md`, `agent-docs/reference/scripts-reference.md`, `agent-docs/optimization/peft.md` |
| `halo/checkpoint/model_card.py` | `agent-docs/reference/checkpoints.md`, `human-docs/checkpoints.md` |
| `halo/checkpoint/parallel/writing.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/parallelism/data-parallelism.md` |
| `halo/models/introspection/unwrapping.py`, `halo/models/introspection/param_roles.py`, `halo/models/introspection/model_classes.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/optimization/peft.md` |
| `halo/models/introspection/module_tree.py` | `agent-docs/parallelism/data-parallelism.md` |
| `halo/models/peft_adapters.py`, `halo/distributed/lora_fold.py` | `agent-docs/optimization/peft.md` |
| `halo/models/head_transform.py` | `agent-docs/models/adding-a-model.md`, `agent-docs/training-methods/async-grpo/performance.md`, `agent-docs/parallelism/pipeline-parallelism.md` |
| `halo/models/forward_capture.py` | `agent-docs/training-methods/async-grpo/performance.md`, `agent-docs/training-methods/embedding.md` |
| `halo/model_loading/parallel/frozen_models.py` | `agent-docs/training-methods/distillation/teacher-distillation.md`, `agent-docs/training-methods/preference/dpo.md` |
| `halo/model_loading/parallel/fp32_masters.py` | `agent-docs/models/README.md`, `agent-docs/reference/checkpoints.md`, `agent-docs/parallelism/expert-parallelism.md` |
| `halo/models/mtp/`, `halo/model_loading/parallel/mtp_head.py`, `halo/callbacks/mtp.py` | `agent-docs/optimization/mtp-heads.md`, `agent-docs/reference/configuration-reference.md`, `agent-docs/models/README.md`, `agent-docs/training-methods/callbacks.md`; a spec or refusal in `spec.py` also that family's `agent-docs/models/` page |
| `halo/model_loading/parallel/model_source.py` | `agent-docs/parallelism/multi-node.md`, `agent-docs/data/filesystem-handling.md`, `agent-docs/reference/troubleshooting.md` |
| `halo/entrypoints/model_setup.py` | `agent-docs/training-methods/sft.md`, `agent-docs/models/README.md`, `agent-docs/reference/configuration-reference.md` |
| `halo/distributed/fsdp.py` | `agent-docs/parallelism/data-parallelism.md`, `agent-docs/reference/checkpoints.md` |
| `halo/runtime/` | `agent-docs/parallelism/multi-node.md`, `agent-docs/data/filesystem-handling.md`, `agent-docs/reference/architecture.md`, `agent-docs/reference/debugging.md`, `agent-docs/reference/troubleshooting.md` |
| `halo/distributed/nvlink.py` | `agent-docs/parallelism/multi-node.md`, `agent-docs/infrastructure/deepep.md` |
| `halo/diagnostics/profiling.py`, `halo/diagnostics/py_spy.py` | `agent-docs/reference/debugging.md`, `agent-docs/optimization/throughput-benchmarks.md` |
| `halo/diagnostics/phase_timers.py`, `halo/callbacks/phase_timer_callback.py`, `halo/trainers/grpo/phase_timing.py` | `agent-docs/training-methods/async-grpo/monitoring.md`, `agent-docs/training-methods/callbacks.md`, `agent-docs/reference/trainer-architecture.md` |
| `halo/trainers/grpo/row_span_stash.py`, `halo/trainers/grpo/logprob_scoring.py` (the row spans) | `agent-docs/training-methods/async-grpo/performance.md`, `agent-docs/reference/trainer-architecture.md` |
| `halo/weight_sync/transport_check.py`, `scripts/profiling/weight_sync_transport.py` | `agent-docs/infrastructure/rollout-servers.md`, `agent-docs/reference/scripts-reference.md`, `agent-docs/reference/debugging.md` |

## Models

| `halo/` area | Owning doc page(s) |
|---|---|
| `halo/distributed/expert_parallel/families/`, `halo/models/moe/family_specs.py` | the family's `agent-docs/models/` page, `agent-docs/models/adding-a-model.md`, `agent-docs/models/README.md`, `agent-docs/parallelism/expert-parallelism.md`, `agent-docs/optimization/grouped-gemm.md`; the attention facts also `agent-docs/optimization/flash-attention.md`; a `weight_sync_refusal` also `agent-docs/infrastructure/rollout-servers.md` and `human-docs/supported-matrix.md` |
| `halo/distributed/expert_parallel/families/zaya.py`, `halo/models/families/zaya.py` | `agent-docs/models/zaya.md` |
| `halo/models/families/mistral4.py` | `agent-docs/models/mistral4.md` |
| `halo/distributed/expert_parallel/families/nemotron_h.py`, `halo/models/families/nemotron_h.py` | `agent-docs/models/nemotron-h.md`, `human-docs/cookbooks/halo-nemotron-3.5-super-cookbook.md` |
| `halo/distributed/expert_parallel/families/qwen3_next.py` | `agent-docs/models/qwen3-next.md` |
| `halo/distributed/expert_parallel/families/qwen4_exp.py`, `halo/models/families/qwen4_exp/` | `agent-docs/models/qwen4_exp.md` |
| `halo/models/external_tables.py`, `halo/model_loading/external_tables.py`, `halo/checkpoint/external_tables.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/models/qwen4_exp.md` |
| `halo/model_loading/parallel/loader.py`, `halo/model_loading/parallel/load_rules.py`, `halo/model_loading/parallel/whole_model.py`, `halo/model_loading/model_preparation.py`, `halo/model_loading/remote_code.py` | `agent-docs/models/README.md`, `agent-docs/models/adding-a-model.md`, the affected per-family page; `agent-docs/optimization/flash-attention.md` |
| `halo/models/upstream/seq_cls_heads.py` | `agent-docs/training-methods/classification.md`, `agent-docs/training-methods/reward-modeling.md` |
| `halo/model_loading/tokenizer_setup.py`, `halo/models/introspection/context_window.py` | `agent-docs/reference/configuration-reference.md`; the vocabulary grow also `agent-docs/reference/checkpoints.md`, `agent-docs/optimization/peft.md` |
| `halo/model_loading/buffer_repair.py`, `halo/model_loading/finalization.py`, `halo/models/kernel_warmups.py` | `agent-docs/models/README.md`, `agent-docs/models/adding-a-model.md` |
| `halo/kernels/hub_kernel_dispatch.py` | `agent-docs/models/README.md`, `agent-docs/models/qwen3_5.md` |
| `halo/kernels/fla_launch.py` | `agent-docs/models/qwen3_5.md`, `agent-docs/models/bailing.md`, `agent-docs/models/glm5-next.md`, `agent-docs/reference/configuration-reference.md` |
| `halo/model_loading/checkpoint_coverage.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/troubleshooting.md`; the full-weight loads under accelerate FSDP also `agent-docs/parallelism/data-parallelism.md` |
| `halo/model_loading/dtype.py` | `agent-docs/models/README.md`, `agent-docs/reference/configuration-reference.md` |
| `halo/models/introspection/config_access.py` | `agent-docs/models/README.md`, `agent-docs/reference/checkpoints.md`, `agent-docs/training-methods/callbacks.md` |
| `halo/models/introspection/modality.py` | `agent-docs/data/dataset-formats.md`, `agent-docs/models/README.md` |
| `halo/models/attention/layout.py`, `halo/callbacks/model_flops.py` | `agent-docs/models/README.md`, `agent-docs/optimization/flash-attention.md`, `agent-docs/training-methods/callbacks.md`, `agent-docs/optimization/throughput-benchmarks.md` |
| `halo/models/upstream/conversion_overrides.py`, `halo/models/row_documents.py`, `halo/models/upstream/upstream_guard.py` | `agent-docs/models/adding-a-model.md`, `agent-docs/reference/architecture.md` |
| `halo/models/segment_markers.py` | `agent-docs/data/collators.md`, `agent-docs/optimization/padding-free-collator.md`, `agent-docs/training-methods/preference/smpo.md` |
| new model support | new `agent-docs/models/<family>.md` + every family roster listed in step 8 of `agent-docs/models/adding-a-model.md` |

## Collators & data

| `halo/` area | Owning doc page(s) |
|---|---|
| `halo/data/collators/` | `agent-docs/data/collators.md`, `agent-docs/optimization/padding-free-collator.md` |
| `halo/data/collators/{offline_grpo,vlm_preference}.py`, `halo/data/pipeline/preferences.py` | `agent-docs/training-methods/grpo/offline.md`, `agent-docs/training-methods/reward-modeling.md` |
| `halo/data/pipeline/prompt_completion.py` | `agent-docs/training-methods/preference/smpo.md`, `agent-docs/training-methods/grpo/offline.md`, `agent-docs/reference/configuration-reference.md` |
| `halo/data/collators/factory.py`, `halo/data/collators/collator_rules.py` | `agent-docs/data/collators.md` |
| `halo/data/tree_layout.py`, `halo/data/pipeline/tree_rows.py`, `halo/data/collators/tree_packing.py`, `halo/models/attention/{tree_plan,tree,tree_sparse}.py` | `agent-docs/optimization/tree-packing.md`, `agent-docs/reference/configuration-reference.md`; GLM-5 Next's in `agent-docs/models/glm5-next.md` (Tree packing) |
| `halo/data/turn_markers.py`, `halo/data/completion_spans.py` | `agent-docs/data/collators.md`, `agent-docs/data/dataset-preparation.md` |
| `halo/data/` | `agent-docs/data/{dataset-formats,dataset-preparation,filesystem-handling}.md`, `agent-docs/parallelism/data-loading.md` |
| `halo/data/sources/s3_client.py`, `halo/data/sources/s3_dataset_store.py`, `halo/data/sources/dataset_cache.py`, `halo/data/sources/paths.py`, `halo/data/sources/s3_commands.py`, `scripts/before_training/s3.py` | `agent-docs/data/s3-utilities.md` |
| `halo/data/preprocessed/`, `scripts/before_training/prepare_dataset.py` | `agent-docs/data/dataset-preparation.md`, `agent-docs/parallelism/data-loading.md` |
| `halo/data/chat_render.py`, `halo/model_loading/tokenizer_backend.py` | `agent-docs/data/dataset-formats.md`, `agent-docs/data/dataset-preparation.md` |
| `halo/data/pipeline/vlm_rows.py` | `agent-docs/data/dataset-formats.md`, `agent-docs/models/README.md` |
| `halo/data/deduplication.py` | `agent-docs/reference/scripts-reference.md` |

## Kernels & optimization

| `halo/` area | Owning doc page(s) |
|---|---|
| `halo/model_loading/liger/` | `agent-docs/optimization/liger-kernels.md` |
| `halo/optimizers/` | `agent-docs/optimization/{bf16-optimizer,muon-optimizer,flash-adamw}.md` |
| `halo/optimizers/step_skip.py` | `agent-docs/reference/trainer-architecture.md`, `agent-docs/optimization/bf16-optimizer.md`, `agent-docs/optimization/muon-optimizer.md` |
| `halo/kernels/lowp/block_fp8.py`, `halo/models/fp8_rollout.py`, `halo/checkpoint/tools/block_fp8_export.py` | `agent-docs/infrastructure/rollout-servers.md`, `agent-docs/reference/scripts-reference.md` |
| `halo/kernels/lowp/` | `agent-docs/optimization/low-precision-moe-kernels.md`, `agent-docs/reference/configuration-reference.md` |
| `halo/model_loading/parallel/lowp_compute.py` | `agent-docs/optimization/low-precision-moe-kernels.md` |
| `halo/kernels/histogram.py` | `agent-docs/training-methods/callbacks.md` |
| `halo/kernels/expert_gemm.py`, `halo/kernels/grouped_mm.py` | `agent-docs/optimization/grouped-gemm.md` |
| `halo/kernels/fused_glu.py` | `agent-docs/models/glm4.md`, `agent-docs/models/gpt-oss.md`, `agent-docs/optimization/liger-kernels.md`, `agent-docs/reference/configuration-reference.md`, `agent-docs/reference/troubleshooting.md`, `agent-docs/optimization/grouped-gemm.md`, `agent-docs/optimization/torch-compile.md`, `agent-docs/parallelism/expert-parallelism.md` |
| `halo/kernels/moe_permute.py` | `agent-docs/optimization/grouped-gemm.md`, `agent-docs/parallelism/expert-parallelism.md`, `agent-docs/reference/troubleshooting.md` |
| `halo/kernels/moe_route_sort.py` | `agent-docs/optimization/grouped-gemm.md`, `agent-docs/infrastructure/deepep.md`, `agent-docs/parallelism/expert-parallelism.md`, `agent-docs/optimization/torch-compile.md`, `agent-docs/reference/troubleshooting.md` |
| `halo/kernels/hyper_connection.py`, `halo/kernels/gated_residual.py`, `halo/models/hyper_connections.py`, `halo/models/families/qwen4_exp/gated_residual.py` | `agent-docs/optimization/hyper-connection-kernels.md`, `agent-docs/optimization/torch-compile.md`, `agent-docs/models/glm5-next.md`, `agent-docs/models/deepseek-v4.md`, `agent-docs/reference/configuration-reference.md`, `agent-docs/reference/troubleshooting.md`, `human-docs/environment-variables.md` |
| `halo/kernels/logprobs.py` | `agent-docs/training-methods/preference/dpo.md`, `agent-docs/training-methods/preference/smpo.md`, `agent-docs/parallelism/pipeline-parallelism.md`, `agent-docs/parallelism/context-parallelism.md` |
| `halo/kernels/fp32_matmul.py` | `agent-docs/models/glm5-next.md`, `agent-docs/optimization/flash-attention.md` |
| `halo/models/attention/masked_matmul.py`, `halo/kernels/masked_softmax.py` | `agent-docs/optimization/flash-attention.md`, `agent-docs/models/glm5-next.md`, `agent-docs/reference/configuration-reference.md`, `agent-docs/reference/troubleshooting.md`, `human-docs/environment-variables.md` |
| `halo/kernels/block_sparse_attention.py` | `agent-docs/optimization/flash-attention.md`, `agent-docs/models/qwen4_exp.md`, `agent-docs/reference/configuration-reference.md` |
| `halo/models/attention/{selection,backend_setup,fa4_entry_points}.py` | `agent-docs/optimization/flash-attention.md` |
| `halo/models/attention/prefix_lm.py` | `agent-docs/optimization/flash-attention.md`, `agent-docs/training-methods/sft.md`, `agent-docs/models/adding-a-model.md` |
| `halo/kernels/jit_cache.py` | `agent-docs/optimization/flash-attention.md`; the Liger implementation pin also `agent-docs/optimization/liger-kernels.md`; the cache and CuTe DSL variables also `agent-docs/infrastructure/docker.md`, `agent-docs/reference/configuration-reference.md` |
| `halo/models/attention/packed_position_ids.py` | `agent-docs/optimization/flash-attention.md`, `agent-docs/models/mistral4.md`, `agent-docs/models/zaya.md` |
| `halo/models/attention/remote_flash_attention.py`, `halo/models/attention/tensor_memo.py` | `agent-docs/models/bailing.md`, `agent-docs/optimization/flash-attention.md`, `agent-docs/data/collators.md` |
| `halo/models/attention/sdpa_flex_sliding.py` | `agent-docs/optimization/flash-attention.md`, `agent-docs/models/gemma4.md`, `agent-docs/models/step3p7.md` |
| `halo/models/attention/gpt_oss_sinks.py` | `agent-docs/models/gpt-oss.md`, `agent-docs/reference/configuration-reference.md` |
| `torch.compile` call sites | `agent-docs/optimization/torch-compile.md` |
| `halo/model_loading/parallel/{peft_setup,freeze_patterns}.py` | `agent-docs/optimization/peft.md` |

## Callbacks

| `halo/` area | Owning doc page(s) |
|---|---|
| `halo/callbacks/` | `agent-docs/training-methods/callbacks.md` |
| `halo/callbacks/gradient_checkpointing_plan.py`, `halo/models/gradient_checkpointing_plan.py` | `agent-docs/training-methods/callbacks.md`, `agent-docs/data/filesystem-handling.md`, `agent-docs/reference/configuration-reference.md` |
| `halo/callbacks/factory.py` | `agent-docs/training-methods/callbacks.md`, `agent-docs/optimization/throughput-benchmarks.md` |
| `halo/callbacks/torch_profiler.py` | `agent-docs/training-methods/callbacks.md`, `agent-docs/reference/debugging.md` |
| `halo/models/moe/{balancing,config_fields,router_logits,router_aux_gradient}.py`, `halo/callbacks/balancing.py` | `agent-docs/training-methods/callbacks.md`, `agent-docs/models/README.md` |
| `halo/hardware.py` | `agent-docs/optimization/throughput-benchmarks.md`, `agent-docs/optimization/flash-attention.md`, `agent-docs/optimization/grouped-gemm.md`; the CPU-count defaults also `agent-docs/reference/configuration-reference.md`, `agent-docs/data/dataset-preparation.md`, `agent-docs/training-methods/async-grpo/environments/sandbox.md` |

## Environments (Async GRPO with Environments)

| `halo/` area | Owning doc page(s) |
|---|---|
| `halo/environments/{environment_base,registry}.py`, `halo/environments/episodes/{messages,episode_keys,turn_recovery,reasoning_effort}.py`, `halo/environments/tools/accounting.py`, `halo/environments/rollouts/episode_loop.py` | `agent-docs/training-methods/async-grpo/environments/{README,custom}.md`, `agent-docs/training-methods/async-grpo/{episodes,reasoning}.md` |
| `halo/environments/episodes/episode_scoring.py` | `agent-docs/training-methods/grpo/rewards/README.md`, `agent-docs/training-methods/async-grpo/environments/custom.md` |
| `halo/environments/protocols/` | `agent-docs/training-methods/async-grpo/environments/protocols/{native-tool-use,react,mcp}.md` |
| `halo/environments/tasks/coding/`, `halo/environments/tasks/qa.py` | `agent-docs/training-methods/async-grpo/environments/tasks/{swe,qa-benchmarks}.md`, `agent-docs/training-methods/async-grpo/environments/tasks/code-contests/*.md` |
| `halo/environments/sandbox/` | `agent-docs/training-methods/async-grpo/environments/sandbox.md` |
| `halo/environments/tools/` | `agent-docs/training-methods/async-grpo/environments/protocols/native-tool-use.md`, `agent-docs/training-methods/async-grpo/environments/tasks/swe.md` |
| `halo/environments/rollouts/engine_wire.py`, `halo/environments/episodes/candidate_logprobs.py` | `agent-docs/infrastructure/rollout-servers.md`, `agent-docs/training-methods/async-grpo/{episodes,reasoning}.md` |
| `halo/environments/rollout_config.py` | `agent-docs/reference/architecture.md`, `agent-docs/reference/configuration-reference.md`, `agent-docs/training-methods/async-grpo/evaluation.md` |
| `halo/environments/rollouts/eval_runner.py` | `agent-docs/training-methods/async-grpo/evaluation.md`, `agent-docs/reference/scripts-reference.md` |
| `halo/environments/tools/web_search.py` | `agent-docs/training-methods/async-grpo/environments/tasks/qa-benchmarks.md`, `agent-docs/reference/configuration-reference.md` |

## Config & args

| `halo/` area | Owning doc page(s) |
|---|---|
| `halo/configs/` | `agent-docs/reference/configuration-reference.md` + the method page that owns the config |
| `halo/configs/parser.py` | `agent-docs/reference/configuration-reference.md` |
| `halo/env.py`, `halo/__init__.py` and any `HALO_`/`DIST_`/`VLLM_`/`SGLANG_`/`NVLINK_`/`EP_` knob wherever it is read | `agent-docs/reference/configuration-reference.md`, `agent-docs/infrastructure/docker.md` |
| `halo/config_validation.py` | `agent-docs/reference/configuration-reference.md` |
| `halo/configs/parallelism_arguments.py` | `agent-docs/parallelism/*`, `agent-docs/reference/configuration-reference.md` |
| `halo/configs/rollout_arguments.py` | `agent-docs/reference/configuration-reference.md`, `agent-docs/training-methods/async-grpo/setup.md` |
| `halo/log.py` | `agent-docs/reference/debugging.md` |
| `halo/cli.py` | `README.md` quick start, `agent-docs/reference/scripts-reference.md` |

## Entry-script plumbing & clients

| `halo/` area | Owning doc page(s) |
|---|---|
| `halo/entrypoints/run_setup.py` | `agent-docs/parallelism/launch-recipes.md`, `agent-docs/data/filesystem-handling.md`, `agent-docs/reference/checkpoints.md`, `agent-docs/reference/debugging.md` |
| `halo/entrypoints/script.py` | `agent-docs/reference/scripts-reference.md`, `agent-docs/reference/configuration-reference.md` |
| `halo/entrypoints/data_setup.py` | `agent-docs/reference/scripts-reference.md`, `agent-docs/parallelism/data-loading.md` |
| `halo/entrypoints/{code_contests_eval,code_contests_regrade}.py` | `agent-docs/training-methods/async-grpo/environments/tasks/code-contests/evaluation.md`, `agent-docs/training-methods/async-grpo/evaluation.md` |
| `halo/entrypoints/generic_env_eval.py` | `agent-docs/training-methods/async-grpo/evaluation.md` |
| `halo/entrypoints/code_pool_preparation.py` | `agent-docs/training-methods/async-grpo/environments/tasks/code-contests/datasets.md` |
| `halo/callbacks/early_stop.py` | `agent-docs/training-methods/async-grpo/monitoring.md`, `agent-docs/reference/scripts-reference.md` |
| `halo/inference/endpoints.py`, `halo/inference/openai_client.py`, `halo/inference/batch_requests.py`, `halo/inference/response.py` | `agent-docs/reference/scripts-reference.md`, `agent-docs/training-methods/grpo/rewards/README.md` |

## Scripts & infrastructure

| `halo/` area | Owning doc page(s) |
|---|---|
| `scripts/training/**`, `scripts/inference/**`, `scripts/before_training/**`, `scripts/after_training/**` | `agent-docs/reference/scripts-reference.md` + the relevant method/data page |
| `scripts/profiling/**` | `agent-docs/reference/debugging.md`, `agent-docs/reference/scripts-reference.md` |
| `scripts/environments/**` | `agent-docs/reference/scripts-reference.md`, `agent-docs/training-methods/async-grpo/evaluation.md` |
| `scripts/cli_arguments.py`, `scripts/inference/inference_arguments.py`, `scripts/inference/reward_model/reward_model_arguments.py`, `halo/entrypoints/{resumable_generation,reward_scoring,rejection_sampling}.py`, `halo/entrypoints/eval_contract.py` | `agent-docs/reference/scripts-reference.md` |
| `scripts/after_training/merge_ep_shards.py`, `halo/checkpoint/tools/ep_shard_merge.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/scripts-reference.md`, `agent-docs/parallelism/expert-parallelism.md` |
| `scripts/after_training/merge_pp_stages.py` | `agent-docs/reference/scripts-reference.md`, `agent-docs/reference/checkpoints.md`, `agent-docs/parallelism/pipeline-parallelism.md` |
| `scripts/after_training/quantize_to_lowp.py`, `halo/checkpoint/tools/lowp_export.py` | `agent-docs/optimization/low-precision-moe-kernels.md` |
| `scripts/after_training/convert_to_bf16.py`, `halo/checkpoint/tools/bf16_conversion.py` | `agent-docs/reference/scripts-reference.md`, `agent-docs/reference/checkpoints.md` |
| `scripts/after_training/merge_models.py`, `halo/checkpoint/tools/model_merge.py` | `agent-docs/reference/model-merging.md`, `agent-docs/reference/scripts-reference.md` |
| `Dockerfile*`, `docker-compose*`, `docker/*/verify_image.py`, `docker/sglang/patches/` | `agent-docs/infrastructure/docker.md`; `Dockerfile.vllm`/`Dockerfile.sglang` + their compose files and server patches also `agent-docs/infrastructure/rollout-servers.md` |
| `docker/engine_source_patch.py`, `docker/vllm/patches/`, `docker/vllm/parity/`, `docker/vllm/plugins/` | `agent-docs/infrastructure/rollout-servers.md`, `agent-docs/infrastructure/docker.md`; plugins also `agent-docs/models/gpt-oss.md` |
| `docker/nccl_pin.py` | `agent-docs/infrastructure/docker.md` |
| `docker/efa/install_efa_userspace.sh` | `agent-docs/infrastructure/docker.md`, `agent-docs/infrastructure/rollout-servers.md`, `agent-docs/parallelism/multi-node.md` |
| AWS / S3 auth, `halo/data/sources/s3_client.py` and `halo/data/sources/s3_dataset_store.py` paths | `agent-docs/infrastructure/aws-auth.md`, `agent-docs/data/s3-utilities.md` |
| DeepEP install / NVSHMEM / CDMC notes | `agent-docs/infrastructure/deepep.md` |
| multi-node / SkyPilot / RunPod / Nomad launch | `agent-docs/parallelism/multi-node.md`, `agent-docs/infrastructure/{skypilot,runpod,nomad}.md` |
| `launcher-configs/**` | `agent-docs/infrastructure/{skypilot,nomad}.md`, `human-docs/clusters.md` |
| `.github/workflows/**`, `.github/scripts/**`, `Makefile` test targets, `tests/common/hub_seed.py` | `agent-docs/infrastructure/ci.md`; tier composition lives in `agent-docs/contributing/README.md` |
| `scripts/docs/**` | `agent-docs/infrastructure/ci.md`, `skills/docs/SKILL.md` |
| `scripts/diagrams/**` | the page embedding the figure; regenerate it in the same commit (`make diagrams`, [Contributing → Docs](../../agent-docs/contributing/README.md#docs)) |
| `tests/baselines/bench/**`, `tests/gpu/profiling/benchmark_sft_{dense,ep}.py` | `agent-docs/contributing/README.md`, `agent-docs/optimization/throughput-benchmarks.md` |

## Cross-cutting

| Change | Owning doc page(s) |
|---|---|
| a new user-facing fail-fast raise or failure mode (config-time rejection, NCCL/DeepEP fault, OOM path) | `agent-docs/reference/troubleshooting.md` (symptom → cause → fix), plus `agent-docs/reference/debugging.md` when it needs a diagnosis helper |
| a new, moved, renamed, or deleted `halo/` subpackage or top-level module | `agent-docs/reference/architecture.md` (source map) |
| code enters or leaves a path in `.github/sensitive-paths` (any module or package move, rename or delete) | `.github/sensitive-paths` |

## Cross-cutting pages

These pages own no `halo/` area; they summarize or route. Update them when the fact they summarize
changes on its owner page.

| Page | Summarizes |
|---|---|
| `agent-docs/README.md`, `agent-docs/training-methods/README.md`, `agent-docs/training-methods/preference/README.md`, `agent-docs/data/README.md`, `agent-docs/infrastructure/README.md`, `agent-docs/optimization/README.md`, `agent-docs/reference/README.md` | the section's pages and their one-line scope |
| `agent-docs/contributing/development-environment.md` | the host setup, credentials and paths |
| `agent-docs/optimization/halo-vs-stock-trl.md` | the measured toolkit-vs-TRL comparisons |
| `agent-docs/reference/{glossary,gpu-training-theory,scale-and-limitations,why-this-framework}.md` | terms, the performance model, the known limits, the design rationale |

## Human guide

Each `human-docs/` page condenses the owners on its row for a human reader. When a change alters
what a user sets, runs or can combine there, update the page with its owner. A new page goes into
`human-docs/README.md` (a cookbook into `human-docs/cookbooks/README.md`).

| `human-docs/` page | Condenses |
|---|---|
| `human-docs/cli.md` | `halo/cli.py`; `agent-docs/reference/scripts-reference.md` |
| `human-docs/configuration.md` | `halo/configs/parser.py`, `halo/configs/`; `agent-docs/reference/configuration-reference.md` |
| `human-docs/environment-variables.md` | `halo/env.py` and every toolkit knob; `agent-docs/reference/configuration-reference.md` (Environment variables) |
| `human-docs/parallelism.md` | `halo/distributed/{parallelism_config,parallelism_rules}.py`, `halo/configs/parallelism_arguments.py`; `agent-docs/parallelism/` |
| `human-docs/supported-matrix.md` | every support verdict: version pins, attention backends, trainer × mode, the rollout-engine roster (`halo/weight_sync/clients/`, `weight_sync_refusal` in `halo/models/moe/family_specs.py`), families, parallelism modes, PEFT and quantization |
| `human-docs/clusters.md` | multi-node launch, `launcher-configs/**`; `agent-docs/parallelism/{multi-node,launch-recipes}.md`, `agent-docs/infrastructure/{skypilot,runpod,nomad}.md` |
| `human-docs/checkpoints.md` | `halo/checkpoint/parallel/`, `halo/checkpoint/`, `scripts/after_training/**`; `agent-docs/reference/{checkpoints,model-merging}.md` |
| `human-docs/data.md` | `halo/data/`; `agent-docs/data/` |
| `human-docs/monitoring.md` | `halo/callbacks/`, `halo/entrypoints/run_setup.py` (the `run.log` tee, the tracking vars); `agent-docs/training-methods/callbacks.md`, `agent-docs/reference/debugging.md`, `agent-docs/training-methods/async-grpo/monitoring.md` |
| `human-docs/performance.md` | `halo/kernels/`, `halo/optimizers/`; `agent-docs/optimization/` (the numbers from `throughput-benchmarks.md`) |
| `human-docs/rollout-servers.md` | `halo/weight_sync/`, `Dockerfile.vllm` / `Dockerfile.sglang` + their compose files; `agent-docs/infrastructure/rollout-servers.md` |
| `human-docs/troubleshooting.md` | a new user-facing failure mode; `agent-docs/reference/{troubleshooting,debugging}.md` |
| `human-docs/models.md`, `human-docs/model-integration-cost.md` | the family roster, `halo/distributed/expert_parallel/families/`; `agent-docs/models/` |
| `human-docs/cookbooks/*.md` | the family's `examples/` recipes; its `agent-docs/models/` page |
| `human-docs/installation.md`, `human-docs/quickstart.md` | `Dockerfile*`, the image tags, `examples/`; `agent-docs/infrastructure/docker.md`, `agent-docs/parallelism/launch-recipes.md` |
| `human-docs/choosing-a-method.md`, `human-docs/training-methods/*.md` | `halo/trainers/`, `scripts/training/**`; the method's `agent-docs/training-methods/` page |
| `human-docs/ai-tooling.md` | `skills/` |
| `human-docs/contributing.md` | `CONTRIBUTING.md`, `agent-docs/contributing/` |

## When in doubt

- Trainer/mixin behavior → `agent-docs/reference/trainer-architecture.md`.
- Any new config field → `agent-docs/reference/configuration-reference.md`.
- A change to what parallelism combinations are valid → the Supported combinations table in
  `agent-docs/parallelism/README.md`, plus the owning `agent-docs/parallelism/` page.
- A new page → wire it into its section `README.md`, then run `./scripts/docs/check_links.sh`.
