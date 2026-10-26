# Doc ownership — `src/` area → owning doc page(s)

When a `src/` path changes in a way that affects behavior, defaults, supported
configurations, or APIs, update the owning page(s) below in the same change: a changed
`src/` path must not leave its doc stale. Rows map paths to pages only; what each module
holds is in the module map, [architecture.md](../../agent-docs/reference/architecture.md).

Always also check whether the change affects: the section **README.md** index, the
**Documentation index** in `CLAUDE.md`, the **Configuration Reference**
(`agent-docs/reference/configuration-reference.md`), and the **Trainer Architecture**
page (`agent-docs/reference/trainer-architecture.md`) when a trainer or the mixin
changes.

## Trainers

| `src/` area | Owning doc page(s) |
|---|---|
| `src/trainers/distributed_trainer.py`, `src/trainers/accelerator.py`, `src/trainers/optimizer_selection.py`, `src/trainers/training_args_overrides.py`, `src/trainers/ctor_slots.py` | `agent-docs/reference/trainer-architecture.md` + every parallelism page under `agent-docs/parallelism/`; `agent-docs/optimization/bf16-optimizer.md` |
| `src/trainers/parallelism/mode_setup.py`, `src/trainers/parallelism/tp_leader.py` | `agent-docs/reference/trainer-architecture.md`, `agent-docs/parallelism/data-parallelism.md` |
| `src/trainers/parallelism/adapter_rules.py` | `agent-docs/optimization/peft.md`, `agent-docs/parallelism/tensor-parallelism.md`, `agent-docs/parallelism/expert-parallelism.md` |
| `src/trainers/reference_logps.py`, `src/trainers/stored_metrics.py` | `agent-docs/reference/trainer-architecture.md`; `agent-docs/reference/checkpoints.md` |
| `src/trainers/checkpointing.py`, `src/trainers/dataloaders.py`, `src/trainers/evaluation.py` | `agent-docs/reference/trainer-architecture.md`; `agent-docs/reference/checkpoints.md`, `agent-docs/parallelism/data-loading.md` |
| `src/trainers/parallelism/{grad_sync,grad_clipping,ep_layers}.py` | `agent-docs/reference/trainer-architecture.md`; `agent-docs/parallelism/expert-parallelism.md`; `agent-docs/parallelism/tensor-parallelism.md`; `agent-docs/parallelism/data-parallelism.md`; `agent-docs/optimization/bf16-optimizer.md` |
| `src/trainers/parallelism/{pipeline_trainer,pipeline_rules}.py`, `src/trainers/parallelism/pipeline_loss_adapter.py` | `agent-docs/parallelism/pipeline-parallelism.md`, `agent-docs/reference/trainer-architecture.md` |
| `src/trainers/parallelism/trainer_rules.py` | `agent-docs/parallelism/context-parallelism.md`; `agent-docs/reference/trainer-architecture.md` |
| `src/trainers/trl_dataset_prep.py` | `agent-docs/data/filesystem-handling.md`; `agent-docs/reference/trainer-architecture.md` |
| `src/trainers/token_metrics.py` | `agent-docs/training-methods/callbacks.md`, `agent-docs/optimization/throughput-benchmarks.md` |
| `src/trainers/world_metrics.py` | `agent-docs/reference/trainer-architecture.md`, `agent-docs/training-methods/async-grpo/monitoring.md`, `agent-docs/training-methods/grpo/online.md` |
| `src/trainers/sft.py`, `src/trainers/sft_rules.py` | `agent-docs/training-methods/sft.md`, `agent-docs/training-methods/pretraining.md` |
| `src/trainers/preference/` | `agent-docs/training-methods/preference/dpo.md`, `agent-docs/training-methods/preference/smpo.md`, `agent-docs/training-methods/preference/kto.md` |
| `src/trainers/preference/trl_preference.py` | `agent-docs/training-methods/preference/dpo.md`, `agent-docs/training-methods/preference/kto.md` |
| `src/trainers/grpo/` | `agent-docs/training-methods/grpo/{README,online,offline}.md`, `agent-docs/training-methods/async-grpo/*.md` |
| `src/trainers/grpo/on_policy.py`, `src/trainers/grpo/grpo_rules.py` | `agent-docs/training-methods/grpo/online.md`, `agent-docs/parallelism/data-loading.md`, `agent-docs/reference/trainer-architecture.md` |
| `src/trainers/grpo/environmental/reasoning.py` | `agent-docs/training-methods/async-grpo/reasoning.md`, `agent-docs/reference/configuration-reference.md` |
| `src/trainers/grpo/offline/` | `agent-docs/training-methods/grpo/offline.md`, `agent-docs/reference/checkpoints.md` |
| `src/rewards/` | `agent-docs/training-methods/grpo/rewards/{README,judge,reward-model}.md`, `agent-docs/training-methods/grpo/online.md`, `agent-docs/training-methods/async-grpo/monitoring.md`, `agent-docs/training-methods/async-grpo/environments/README.md`, `agent-docs/reference/configuration-reference.md` |
| `src/trainers/reward/` | `agent-docs/training-methods/preference/reward-modeling.md`, `agent-docs/training-methods/classification.md` |
| `src/trainers/distillation/` | `agent-docs/training-methods/distillation/{README,teacher-distillation,self-distillation,online-sdpg}.md` |
| `src/trainers/embedding/` | `agent-docs/training-methods/embedding.md` |
| any new trainer | `agent-docs/reference/trainer-architecture.md` + new method page + the section `README.md` + `CLAUDE.md` index |

## Distributed / parallelism

| `src/` area | Owning doc page(s) |
|---|---|
| `src/distributed/{parallelism_config,parallelism_rules}.py` | `agent-docs/parallelism/*`, `agent-docs/reference/configuration-reference.md`, the parallelism matrix in `CLAUDE.md` |
| `src/distributed/expert_parallel/` | `agent-docs/parallelism/expert-parallelism.md`, `agent-docs/parallelism/expert-tensor-parallelism.md`, `agent-docs/optimization/grouped-gemm.md`, `agent-docs/infrastructure/deepep.md`, `agent-docs/models/adding-a-model.md`, `agent-docs/reference/architecture.md` |
| `src/distributed/expert_parallel/deepep/`, `src/distributed/rank_uniform_env.py` | `agent-docs/infrastructure/deepep.md`, `agent-docs/reference/configuration-reference.md` |
| `src/models/moe/expert_lora_spec.py` | `agent-docs/optimization/peft.md`, `agent-docs/models/adding-a-model.md` |
| `src/models/moe/expert_layout.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/parallelism/expert-parallelism.md`, `agent-docs/models/adding-a-model.md` |
| `src/distributed/context_parallel/`, `src/model_loading/parallel/cp.py` | `agent-docs/parallelism/context-parallelism.md` |
| `src/distributed/tensor_parallel/`, `src/model_loading/parallel/tp.py` | `agent-docs/parallelism/tensor-parallelism.md` |
| `src/distributed/pipeline_parallel/` | `agent-docs/parallelism/pipeline-parallelism.md`, `agent-docs/reference/checkpoints.md` |
| `src/model_loading/lazy_safetensors/`, `src/model_loading/checkpoint_keys.py` | `agent-docs/parallelism/expert-parallelism.md`, `agent-docs/parallelism/pipeline-parallelism.md`, `agent-docs/reference/checkpoints.md` |
| `src/distributed/expert_parallel/lazy_plan.py`, `src/distributed/expert_parallel/expert_fuser.py`, `src/model_loading/parallel/ep.py`, `src/model_loading/parallel/lazy_load.py`, `src/models/gradient_checkpointing.py` | `agent-docs/parallelism/expert-parallelism.md` |
| `src/model_loading/parallel/pp.py` | `agent-docs/parallelism/pipeline-parallelism.md` |
| `src/distributed/{group_layout,mesh}.py` | `agent-docs/parallelism/multi-node.md`, `agent-docs/parallelism/data-parallelism.md`, `agent-docs/reference/architecture.md` |
| `src/subclass_registry.py` | `agent-docs/models/adding-a-model.md`, `agent-docs/reference/architecture.md` |
| `src/distributed/nccl/_nccl_wrapper.py` | `agent-docs/infrastructure/docker.md`, `agent-docs/infrastructure/rollout-servers.md` |
| `src/distributed/grad_reduce.py` | `agent-docs/parallelism/data-parallelism.md`, `agent-docs/reference/configuration-reference.md` |
| `src/weight_sync/` | `agent-docs/infrastructure/rollout-servers.md`, `agent-docs/training-methods/grpo/online.md`, `agent-docs/reference/configuration-reference.md` |
| `src/trainers/grpo/rollout/weight_push.py` | `agent-docs/infrastructure/rollout-servers.md`, `agent-docs/training-methods/grpo/online.md`, `agent-docs/optimization/peft.md`, `agent-docs/training-methods/async-grpo/setup.md`, `agent-docs/reference/debugging.md` |
| `src/trainers/grpo/rollout/trl_vllm_client.py` | `agent-docs/training-methods/grpo/online.md`, `agent-docs/reference/trainer-architecture.md`, `agent-docs/reference/configuration-reference.md` |
| `src/trainers/grpo/rollout/engine_preflight.py` | `agent-docs/training-methods/async-grpo/setup.md`, `agent-docs/training-methods/grpo/online.md` |
| `src/trainers/grpo/environmental/async_rollouts.py` | `agent-docs/training-methods/async-grpo/setup.md` |
| `src/trainers/grpo/environmental/{trainer,environmental_rules}.py` | `agent-docs/training-methods/async-grpo/{setup,objective,reasoning}.md`, `agent-docs/reference/trainer-architecture.md` |
| `src/trainers/grpo/environmental/{step_batch,importance_sampling}.py` | `agent-docs/training-methods/async-grpo/{objective,monitoring}.md` |
| `src/trainers/grpo/environmental/routing_replay.py` | `agent-docs/training-methods/async-grpo/objective.md`, `agent-docs/infrastructure/rollout-servers.md` |
| `src/trainers/grpo/environmental/{trajectory_tokenize,trajectory_spans}.py` | `agent-docs/training-methods/async-grpo/objective.md` |
| `src/trainers/grpo/environmental/rollout_metrics.py`, `src/trainers/grpo/rollout/completions_logging.py` | `agent-docs/training-methods/async-grpo/monitoring.md`, `agent-docs/training-methods/callbacks.md` |
| `src/trainers/grpo/early_stop.py` | `agent-docs/training-methods/async-grpo/monitoring.md`, `agent-docs/training-methods/grpo/online.md`, `agent-docs/reference/configuration-reference.md` |
| `src/trainers/grpo/objective/` | `agent-docs/training-methods/async-grpo/objective.md`, `agent-docs/training-methods/grpo/{online,offline}.md` |
| `src/trainers/grpo/logprob_scoring.py`, `src/kernels/head_logprobs.py` | `agent-docs/training-methods/async-grpo/performance.md`, `agent-docs/reference/configuration-reference.md` |
| `src/environments/ray_actors.py` | `agent-docs/infrastructure/ray.md`, `agent-docs/training-methods/async-grpo/setup.md` |
| `src/checkpoint/parallel/save.py`, `src/checkpoint/parallel/resume_weights.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/trainer-architecture.md`, `agent-docs/parallelism/tensor-parallelism.md` |
| `src/checkpoint/parallel/ep.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/parallelism/expert-parallelism.md` |
| `src/checkpoint/parallel/pp.py`, `src/checkpoint/pp_stage_merge.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/parallelism/pipeline-parallelism.md` |
| `src/checkpoint/parallel/adapters.py`, `src/checkpoint/adapters.py` | `agent-docs/optimization/peft.md`, `agent-docs/reference/checkpoints.md` |
| `src/checkpoint/parallel/context.py`, `src/checkpoint/parallel/sidecars.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/trainer-architecture.md` |
| `src/checkpoint/{format,hub_layout,atomic,weights_source,sidecars,writing}.py`, `src/checkpoint/parallel/completeness.py` | `agent-docs/reference/checkpoints.md` |
| `src/checkpoint/reading.py`, `src/checkpoint/offline_export.py`, `src/checkpoint/fp8_dequant.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/scripts-reference.md` |
| `src/checkpoint/parallel/resume_source.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/architecture.md` |
| `src/checkpoint/parallel/reading.py` | `agent-docs/reference/architecture.md`, `agent-docs/reference/checkpoints.md` |
| `src/checkpoint/parallel/optimizer.py`, `src/checkpoint/parallel/optimizer_meta.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/troubleshooting.md`, `agent-docs/data/filesystem-handling.md` |
| `src/checkpoint/config_export.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/models/README.md` |
| `src/checkpoint/adapter_merge.py` | `agent-docs/reference/model-merging.md`, `agent-docs/reference/scripts-reference.md` |
| `src/checkpoint/model_card.py` | `agent-docs/reference/checkpoints.md`, `human-docs/checkpoints.md` |
| `src/checkpoint/parallel/writing.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/parallelism/data-parallelism.md` |
| `src/models/unwrapping.py`, `src/models/param_roles.py`, `src/models/model_classes.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/optimization/peft.md` |
| `src/models/module_tree.py` | `agent-docs/parallelism/data-parallelism.md` |
| `src/models/peft_adapters.py` | `agent-docs/optimization/peft.md` |
| `src/models/head_transform.py` | `agent-docs/models/adding-a-model.md`, `agent-docs/training-methods/async-grpo/performance.md`, `agent-docs/parallelism/pipeline-parallelism.md` |
| `src/model_loading/parallel/frozen_models.py` | `agent-docs/training-methods/distillation/teacher-distillation.md`, `agent-docs/training-methods/preference/dpo.md` |
| `src/model_loading/parallel/fp32_masters.py` | `agent-docs/models/README.md`, `agent-docs/reference/checkpoints.md`, `agent-docs/parallelism/expert-parallelism.md` |
| `src/model_loading/parallel/model_source.py` | `agent-docs/parallelism/multi-node.md`, `agent-docs/data/filesystem-handling.md`, `agent-docs/reference/troubleshooting.md` |
| `src/entrypoints/model_setup.py` | `agent-docs/training-methods/sft.md`, `agent-docs/models/README.md`, `agent-docs/reference/configuration-reference.md` |
| `src/distributed/fsdp.py` | `agent-docs/parallelism/data-parallelism.md`, `agent-docs/reference/checkpoints.md` |
| `src/runtime/` | `agent-docs/parallelism/multi-node.md`, `agent-docs/data/filesystem-handling.md`, `agent-docs/reference/architecture.md`, `agent-docs/reference/debugging.md` |
| `src/distributed/nvlink.py` | `agent-docs/parallelism/multi-node.md`, `agent-docs/infrastructure/deepep.md` |
| `src/diagnostics/profiling.py`, `src/diagnostics/py_spy.py` | `agent-docs/reference/debugging.md`, `agent-docs/optimization/throughput-benchmarks.md` |
| `src/weight_sync/transport_check.py`, `scripts/profiling/weight_sync_transport.py` | `agent-docs/infrastructure/rollout-servers.md`, `agent-docs/reference/scripts-reference.md`, `agent-docs/reference/debugging.md` |
| `src/diagnostics/ep_span_timing.py` | `agent-docs/optimization/throughput-benchmarks.md`, `agent-docs/reference/debugging.md` |

## Models

| `src/` area | Owning doc page(s) |
|---|---|
| `src/distributed/expert_parallel/families/`, `src/models/moe/family_specs.py` | the family's `agent-docs/models/` page, `agent-docs/parallelism/expert-parallelism.md`, `agent-docs/optimization/grouped-gemm.md` |
| `src/distributed/expert_parallel/families/zaya.py`, `src/models/zaya.py` | `agent-docs/models/zaya.md` |
| `src/model_loading/parallel/loader.py`, `src/model_loading/parallel/load_rules.py`, `src/model_loading/parallel/whole_model.py`, `src/model_loading/model_preparation.py`, `src/model_loading/remote_code.py` | `agent-docs/models/README.md`, `agent-docs/models/adding-a-model.md`, the affected per-family page; `agent-docs/optimization/flash-attention.md` |
| `src/models/seq_cls_heads.py` | `agent-docs/training-methods/classification.md`, `agent-docs/training-methods/preference/reward-modeling.md` |
| `src/model_loading/tokenizer_setup.py`, `src/models/context_window.py` | `agent-docs/reference/configuration-reference.md` |
| `src/model_loading/buffer_repair.py` | `agent-docs/models/README.md`, `agent-docs/models/adding-a-model.md` |
| `src/kernels/hub_kernel_dispatch.py` | `agent-docs/models/README.md`, `agent-docs/models/qwen3_5.md` |
| `src/model_loading/checkpoint_coverage.py` | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/troubleshooting.md` |
| `src/model_loading/dtype.py` | `agent-docs/models/README.md`, `agent-docs/reference/configuration-reference.md` |
| `src/models/config_access.py` | `agent-docs/models/README.md`, `agent-docs/reference/checkpoints.md`, `agent-docs/training-methods/callbacks.md` |
| `src/models/modality.py` | `agent-docs/data/dataset-formats.md`, `agent-docs/models/README.md` |
| `src/models/attention/layout.py`, `src/callbacks/model_flops.py` | `agent-docs/models/README.md`, `agent-docs/optimization/flash-attention.md`, `agent-docs/training-methods/callbacks.md`, `agent-docs/optimization/throughput-benchmarks.md` |
| `src/models/segment_markers.py` | `agent-docs/data/collators.md`, `agent-docs/optimization/padding-free-collator.md`, `agent-docs/training-methods/preference/smpo.md` |
| new model support | new `agent-docs/models/<family>.md` + every family roster listed in step 8 of `agent-docs/models/adding-a-model.md` |

## Collators & data

| `src/` area | Owning doc page(s) |
|---|---|
| `src/data/collators/` | `agent-docs/data/collators.md`, `agent-docs/optimization/padding-free-collator.md` |
| `src/data/collators/{offline_grpo,vlm_preference}.py`, `src/data/pipeline/preferences.py` | `agent-docs/training-methods/grpo/offline.md`, `agent-docs/training-methods/preference/reward-modeling.md` |
| `src/data/pipeline/prompt_completion.py` | `agent-docs/training-methods/preference/smpo.md`, `agent-docs/training-methods/grpo/offline.md`, `agent-docs/reference/configuration-reference.md` |
| `src/data/collators/factory.py`, `src/data/collators/collator_rules.py` | `agent-docs/data/collators.md` |
| `src/data/tree_layout.py`, `src/data/pipeline/tree_rows.py`, `src/data/collators/tree_packing.py`, `src/models/attention/{tree_plan,tree}.py` | `agent-docs/optimization/tree-packing.md`, `agent-docs/reference/configuration-reference.md` |
| `src/data/turn_markers.py`, `src/data/completion_spans.py` | `agent-docs/data/collators.md`, `agent-docs/data/dataset-preparation.md` |
| `src/data/` | `agent-docs/data/{dataset-formats,dataset-preparation,filesystem-handling}.md`, `agent-docs/parallelism/data-loading.md` |
| `src/data/sources/s3_client.py`, `src/data/sources/s3_dataset_store.py`, `src/data/sources/dataset_cache.py`, `src/data/sources/paths.py`, `scripts/before_training/s3_datasets.py` | `agent-docs/data/s3-utilities.md` |
| `src/data/preprocessed/`, `scripts/before_training/prepare_dataset.py` | `agent-docs/data/dataset-preparation.md`, `agent-docs/parallelism/data-loading.md` |
| `src/data/pipeline/chat_render.py`, `src/model_loading/tokenizer_backend.py` | `agent-docs/data/dataset-formats.md`, `agent-docs/data/dataset-preparation.md` |
| `src/data/pipeline/vlm_rows.py` | `agent-docs/data/dataset-formats.md`, `agent-docs/models/README.md` |
| `src/data/deduplication.py` | `agent-docs/reference/scripts-reference.md` |

## Kernels & optimization

| `src/` area | Owning doc page(s) |
|---|---|
| `src/model_loading/liger/` | `agent-docs/optimization/liger-kernels.md` |
| `src/optimizers/` | `agent-docs/optimization/{bf16-optimizer,muon-optimizer,flash-adamw}.md` |
| `src/kernels/lowp/` | `agent-docs/optimization/low-precision-moe-kernels.md`, `agent-docs/reference/configuration-reference.md` |
| `src/model_loading/parallel/lowp_compute.py` | `agent-docs/optimization/low-precision-moe-kernels.md` |
| `src/kernels/histogram.py` | `agent-docs/training-methods/callbacks.md` |
| `src/kernels/expert_gemm.py`, `src/kernels/grouped_mm.py` | `agent-docs/optimization/grouped-gemm.md` |
| `src/kernels/fused_glu.py` | `agent-docs/models/glm4.md`, `agent-docs/models/gpt-oss.md`, `agent-docs/optimization/liger-kernels.md`, `agent-docs/reference/configuration-reference.md`, `agent-docs/reference/troubleshooting.md`, `agent-docs/optimization/grouped-gemm.md`, `agent-docs/optimization/torch-compile.md`, `agent-docs/parallelism/expert-parallelism.md` |
| `src/kernels/moe_permute.py` | `agent-docs/optimization/grouped-gemm.md`, `agent-docs/parallelism/expert-parallelism.md` |
| `src/kernels/logprobs.py` | `agent-docs/training-methods/preference/dpo.md`, `agent-docs/training-methods/preference/smpo.md`, `agent-docs/parallelism/pipeline-parallelism.md` |
| `src/models/attention/{selection,backend_setup}.py`, `src/kernels/jit_cache.py` | `agent-docs/optimization/flash-attention.md` |
| `src/models/attention/packed_position_ids.py` | `agent-docs/optimization/flash-attention.md`, `agent-docs/models/mistral4.md`, `agent-docs/models/zaya.md` |
| `src/models/attention/sdpa_flex_sliding.py` | `agent-docs/optimization/flash-attention.md`, `agent-docs/models/gemma4.md` |
| `src/models/attention/gpt_oss_sinks.py` | `agent-docs/models/gpt-oss.md`, `agent-docs/reference/configuration-reference.md` |
| `torch.compile` call sites | `agent-docs/optimization/torch-compile.md` |
| `src/model_loading/parallel/{peft_setup,freeze_patterns}.py` | `agent-docs/optimization/peft.md` |

## Callbacks

| `src/` area | Owning doc page(s) |
|---|---|
| `src/callbacks/` | `agent-docs/training-methods/callbacks.md` |
| `src/callbacks/factory.py` | `agent-docs/training-methods/callbacks.md`, `agent-docs/optimization/throughput-benchmarks.md` |
| `src/models/moe/{balancing,config_fields,router_logits,router_aux_gradient}.py`, `src/callbacks/balancing.py` | `agent-docs/training-methods/callbacks.md`, `agent-docs/models/README.md` |
| `src/hardware.py` | `agent-docs/optimization/throughput-benchmarks.md`, `agent-docs/optimization/flash-attention.md`, `agent-docs/optimization/grouped-gemm.md` |

## Environments (Async GRPO with Environments)

| `src/` area | Owning doc page(s) |
|---|---|
| `src/environments/{environment_base,messages,episode_keys,tool_accounting,turn_recovery,reasoning_effort,episode,registry}.py` | `agent-docs/training-methods/async-grpo/environments/{README,custom}.md`, `agent-docs/training-methods/async-grpo/{episodes,reasoning}.md` |
| `src/environments/episode_scoring.py` | `agent-docs/training-methods/grpo/rewards/README.md`, `agent-docs/training-methods/async-grpo/environments/custom.md` |
| `src/environments/protocols/` | `agent-docs/training-methods/async-grpo/environments/protocols/{native-tool-use,react,mcp}.md` |
| `src/environments/tasks/coding/`, `src/environments/tasks/qa.py` | `agent-docs/training-methods/async-grpo/environments/tasks/{swe,qa-benchmarks}.md`, `agent-docs/training-methods/async-grpo/environments/tasks/code-contests/*.md` |
| `src/environments/sandbox/` | `agent-docs/training-methods/async-grpo/environments/sandbox.md` |
| `src/environments/tools/` | `agent-docs/training-methods/async-grpo/environments/protocols/native-tool-use.md`, `agent-docs/training-methods/async-grpo/environments/tasks/swe.md` |
| `src/environments/engine_wire.py` | `agent-docs/infrastructure/rollout-servers.md`, `agent-docs/training-methods/async-grpo/{episodes,reasoning}.md` |
| `src/environments/rollout_config.py` | `agent-docs/reference/architecture.md`, `agent-docs/reference/configuration-reference.md`, `agent-docs/training-methods/async-grpo/evaluation.md` |
| `src/environments/eval_runner.py` | `agent-docs/training-methods/async-grpo/evaluation.md`, `agent-docs/reference/scripts-reference.md` |
| `src/environments/tools/web_search.py` | `agent-docs/training-methods/async-grpo/environments/tasks/qa-benchmarks.md`, `agent-docs/reference/configuration-reference.md` |

## Config & args

| `src/` area | Owning doc page(s) |
|---|---|
| `src/configs/` | `agent-docs/reference/configuration-reference.md` + the method page that owns the config |
| `src/configs/parser.py` | `agent-docs/reference/configuration-reference.md` |
| `src/env.py`, `src/__init__.py` and any `HALO_`/`DIST_`/`VLLM_`/`SGLANG_`/`NVLINK_`/`EP_` knob wherever it is read | `agent-docs/reference/configuration-reference.md`, `agent-docs/infrastructure/docker.md` |
| `src/config_validation.py` | `agent-docs/reference/configuration-reference.md` |
| `src/configs/parallelism_arguments.py` | `agent-docs/parallelism/*`, `agent-docs/reference/configuration-reference.md` |
| `src/configs/rollout_arguments.py` | `agent-docs/reference/configuration-reference.md`, `agent-docs/training-methods/async-grpo/setup.md` |
| `src/log.py` | `agent-docs/reference/debugging.md` |
| `src/cli.py` | `README.md` quick start, `agent-docs/reference/scripts-reference.md` |

## Entry-script plumbing & clients

| `src/` area | Owning doc page(s) |
|---|---|
| `src/entrypoints/run_setup.py` | `agent-docs/parallelism/launch-recipes.md`, `agent-docs/data/filesystem-handling.md`, `agent-docs/reference/checkpoints.md`, `agent-docs/reference/debugging.md` |
| `src/entrypoints/script.py` | `agent-docs/reference/scripts-reference.md`, `agent-docs/reference/configuration-reference.md` |
| `src/entrypoints/data_setup.py` | `agent-docs/reference/scripts-reference.md`, `agent-docs/parallelism/data-loading.md` |
| `src/callbacks/early_stop.py` | `agent-docs/training-methods/async-grpo/monitoring.md`, `agent-docs/reference/scripts-reference.md` |
| `src/inference/endpoints.py`, `src/inference/openai_client.py`, `src/inference/batch_requests.py`, `src/inference/response.py` | `agent-docs/reference/scripts-reference.md`, `agent-docs/training-methods/grpo/rewards/README.md` |

## Scripts & infrastructure

| `src/` area | Owning doc page(s) |
|---|---|
| `scripts/training/**`, `scripts/inference/**`, `scripts/before_training/**`, `scripts/after_training/**` | `agent-docs/reference/scripts-reference.md` + the relevant method/data page |
| `scripts/profiling/**` | `agent-docs/reference/debugging.md`, `agent-docs/reference/scripts-reference.md` |
| `scripts/environments/**` | `agent-docs/reference/scripts-reference.md`, `agent-docs/training-methods/async-grpo/evaluation.md` |
| `scripts/cli_arguments.py`, `scripts/inference/inference_arguments.py`, `scripts/inference/reward_model/reward_model_arguments.py` | `agent-docs/reference/scripts-reference.md` |
| `scripts/after_training/merge_ep_shards.py` | `agent-docs/reference/checkpoints.md` |
| `scripts/after_training/merge_pp_stages.py` | `agent-docs/reference/scripts-reference.md`, `agent-docs/reference/checkpoints.md`, `agent-docs/parallelism/pipeline-parallelism.md` |
| `scripts/after_training/quantize_to_lowp.py` | `agent-docs/optimization/low-precision-moe-kernels.md` |
| `scripts/after_training/convert_to_bf16.py` | `agent-docs/reference/scripts-reference.md`, `agent-docs/reference/checkpoints.md` |
| `scripts/after_training/merge_models.py` | `agent-docs/reference/model-merging.md`, `agent-docs/reference/scripts-reference.md` |
| `Dockerfile*`, `docker-compose*`, `docker/sglang/patches/` | `agent-docs/infrastructure/docker.md`; `Dockerfile.vllm`/`Dockerfile.sglang` + their compose files and server patches also `agent-docs/infrastructure/rollout-servers.md` |
| `docker/engine_source_patch.py`, `docker/vllm/patches/`, `docker/vllm/parity/`, `docker/vllm/plugins/` | `agent-docs/infrastructure/rollout-servers.md`, `agent-docs/infrastructure/docker.md`; plugins also `agent-docs/models/gpt-oss.md` |
| `docker/nccl_pin.py` | `agent-docs/infrastructure/docker.md` |
| `docker/efa/install_efa_userspace.sh` | `agent-docs/infrastructure/docker.md`, `agent-docs/infrastructure/rollout-servers.md`, `agent-docs/parallelism/multi-node.md` |
| AWS / S3 auth, `src/data/sources/s3_client.py` and `src/data/sources/s3_dataset_store.py` paths | `agent-docs/infrastructure/aws-auth.md`, `agent-docs/data/s3-utilities.md` |
| DeepEP install / NVSHMEM / CDMC notes | `agent-docs/infrastructure/deepep.md` |
| multi-node / SkyPilot / RunPod / Nomad launch | `agent-docs/parallelism/multi-node.md`, `agent-docs/infrastructure/{skypilot,runpod,nomad}.md` |
| `launcher-configs/**` | `agent-docs/infrastructure/{skypilot,nomad}.md`, `human-docs/clusters.md` |
| `.github/workflows/**`, `.github/scripts/**`, `Makefile` test targets, `tests/common/hub_seed.py` | `agent-docs/infrastructure/ci.md`; tier composition lives in `agent-docs/contributing/README.md` |

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
| `human-docs/configuration.md` | `src/configs/parser.py`, `src/configs/`; `agent-docs/reference/configuration-reference.md` |
| `human-docs/environment-variables.md` | `src/env.py` and every toolkit knob; `agent-docs/reference/configuration-reference.md` (Environment variables) |
| `human-docs/parallelism.md` | `src/distributed/{parallelism_config,parallelism_rules}.py`, `src/configs/parallelism_arguments.py`; `agent-docs/parallelism/` |
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
| `human-docs/installation.md`, `human-docs/quickstart.md` | `Dockerfile*`, the image tags, `examples/`; `agent-docs/infrastructure/docker.md`, `agent-docs/parallelism/launch-recipes.md` |
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
