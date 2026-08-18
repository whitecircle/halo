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
| `src/trainers/mixins/` (sub-mixins: checkpointing, dataloader, EP introspection, grad sync/clip, validation, pipeline) | `agent-docs/reference/trainer-architecture.md`; `agent-docs/reference/checkpoints.md` (checkpointing), `agent-docs/parallelism/pipeline-parallelism.md` (pipeline) |
| `src/trainers/sft.py` | `agent-docs/training-methods/sft.md`, `agent-docs/training-methods/pretraining.md` |
| `src/trainers/preference/` (DPO, SMPO, KTO) | `agent-docs/training-methods/preference/dpo.md`, `agent-docs/training-methods/preference/smpo.md`, `agent-docs/training-methods/preference/kto.md` |
| `src/trainers/grpo/` (online, offline, environmental) | `agent-docs/training-methods/grpo/{online-grpo,offline-grpo,environmental-grpo,grpo-comparison}.md` |
| `src/trainers/reward/` | `agent-docs/training-methods/preference/reward-modeling.md`, `agent-docs/training-methods/classification.md` |
| `src/trainers/distillation/` | `agent-docs/training-methods/distillation/{index,teacher-distillation,self-distillation,online-sdpg}.md` |
| `src/trainers/embedding/` (SBERT trainer, `sentence_transformers_compat.py` patches + preloaded-model shim) | `agent-docs/training-methods/embedding.md` |
| any new trainer | `agent-docs/reference/trainer-architecture.md` + new method page + the section `README.md` + `CLAUDE.md` index |

## Distributed / parallelism

| `src/` area | Owning doc page(s) |
|---|---|
| `src/distributed/parallelism_config.py` (ParallelismConfig, validation) | `agent-docs/parallelism/*` (all), `agent-docs/reference/configuration-reference.md`, the parallelism matrix in `CLAUDE.md` + `agent-docs/README.md` |
| `src/distributed/expert_parallel/` (DeepEP, all-to-all, per-family layers, expert compute) | `agent-docs/parallelism/expert-parallelism.md`, `agent-docs/parallelism/expert-tensor-parallelism.md`, `agent-docs/optimization/grouped-gemm.md`, `agent-docs/infrastructure/deepep.md` |
| `src/distributed/context_parallel/` (Ulysses) | `agent-docs/parallelism/context-parallelism.md` |
| `src/distributed/tensor_parallel/` (DTensor) | `agent-docs/parallelism/tensor-parallelism.md` |
| `src/distributed/pipeline_parallel/` (stage split, schedules, stage loader) | `agent-docs/parallelism/pipeline-parallelism.md`, `agent-docs/reference/checkpoints.md` (stage shards) |
| `src/models/loading/lazy_safetensors/` (safetensors lazy-load core shared by the EP + PP loaders) | `agent-docs/parallelism/expert-parallelism.md` (lazy loading), `agent-docs/parallelism/pipeline-parallelism.md`, `agent-docs/reference/checkpoints.md` |
| `src/distributed/group_layout.py`, `mesh.py` (rank math, DeviceMesh + the typed group view) | `agent-docs/parallelism/multi-node.md`, `agent-docs/parallelism/data-parallelism.md`, `agent-docs/reference/architecture.md` |
| `src/distributed/module_registry.py` (HF-class → wrapper registries) | `agent-docs/models/adding-a-model.md`, `agent-docs/reference/architecture.md` |
| `src/distributed/grad_reduce.py` (bucketed gradient all-reduce — EP cross-replica, TP replicated, QLoRA sweeps) | `agent-docs/parallelism/data-parallelism.md`, `agent-docs/reference/configuration-reference.md` (`HALO_GRAD_BUCKET_MB`) |
| `src/distributed/nccl/` (weight sync clients + transport) | `agent-docs/infrastructure/rollout-servers.md` (weight sync, NCCL transport), `agent-docs/training-methods/grpo/online-grpo.md` |
| `src/trainers/grpo/rollout/weight_sync.py` (gather/gates/memory bracket) | `agent-docs/infrastructure/rollout-servers.md`, `agent-docs/training-methods/grpo/environmental-grpo.md`, `agent-docs/reference/debugging.md` (memory bracket) |
| `src/trainers/grpo/rollout/weight_sync_clients.py` (per-server weight-sync client pool, `/v1/models` context preflight) | `agent-docs/training-methods/grpo/environmental-grpo.md`, `agent-docs/training-methods/grpo/online-grpo.md` |
| `src/trainers/grpo/rollout/async_rollouts.py` (Ray-actor collection, prefetch thread, engine weight-sync entry points) | `agent-docs/training-methods/grpo/environmental-grpo.md` |
| `src/trainers/grpo/rollout/trajectory_tokenize.py` (trajectory → training rows: whole-render spans, per-turn sampled ids) | `agent-docs/training-methods/grpo/environmental-grpo.md` |
| `src/trainers/grpo/rollout/rollout_metrics.py` (completion logs, per-episode rollout diagnostics) | `agent-docs/training-methods/grpo/environmental-grpo.md`, `agent-docs/training-methods/callbacks.md` |
| `src/environments/ray_actors.py` (actor pool, dispatch, Ray init) | `agent-docs/infrastructure/ray.md`, `agent-docs/training-methods/grpo/environmental-grpo.md` (trainer-side knobs) |
| `src/distributed/checkpoint/` (save ladder, weight loader, OptimizerShardStore, PeftAdapterSaver) | `agent-docs/reference/checkpoints.md` |
| `src/checkpoint/format.py` (on-disk spellings, save-dtype casts, the layout cascade, state-dict IO) | `agent-docs/reference/checkpoints.md` |
| `src/checkpoint/config_export.py` (what an exported `config.json` must contain: model_type restore, flat legacy keys, source schema) | `agent-docs/reference/checkpoints.md`, `agent-docs/models/README.md` |
| `src/checkpoint/adapters.py` (saved-PEFT file layout, expert-LoRA shape gates, merge-into-base) | `agent-docs/optimization/peft.md`, `agent-docs/reference/checkpoints.md`, `agent-docs/reference/scripts-reference.md` |
| `src/checkpoint/tool_io.py` (tool-side checkpoint walks, input gates, staged publish, training-state sidecars), `src/checkpoint/fp8_dequant.py` (streaming fp8 → bf16) | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/scripts-reference.md` |
| `src/checkpoint/shard_writer.py` (`StageShardWriter`: incremental safetensors parts + index) | `agent-docs/reference/checkpoints.md` |
| `src/distributed/checkpoint/write.py` (the collective half of a write: retain-gated DTensor resolve of params and buffers with neutralized sinks, the streamed part writer, the shard-index exchange) | `agent-docs/reference/checkpoints.md`, `agent-docs/parallelism/data-parallelism.md` |
| `src/models/structure.py` (module-tree introspection: unwrap, PEFT names, decoder layers, norms) | `agent-docs/reference/checkpoints.md`, `agent-docs/parallelism/data-parallelism.md` |
| `src/distributed/fsdp.py` (FSDP2 wrapping + reshard) | `agent-docs/parallelism/data-parallelism.md`, `agent-docs/reference/checkpoints.md` |
| `src/distributed/runtime.py` (rank/world state, barriers, cross-rank consensus, group timeouts), `src/distributed/filesystem.py` (c10d-store phases, main-first ordering, output-FS probe, load throttle) | `agent-docs/parallelism/multi-node.md`, `agent-docs/data/filesystem-handling.md`, `agent-docs/reference/architecture.md` |
| `src/distributed/nvlink.py` (fabric probes behind `nvlink_domain_size`) | `agent-docs/parallelism/multi-node.md`, `agent-docs/infrastructure/deepep.md` |
| `src/diagnostics/profiling.py` (torch-profiler traces, CUDA memory snapshots), `src/diagnostics/debugging.py` (opt-in consistency checks, py-spy capture) | `agent-docs/reference/debugging.md`, `agent-docs/optimization/throughput-benchmarks.md` |
| `src/diagnostics/performance_monitor.py` | `agent-docs/optimization/throughput-benchmarks.md`, `agent-docs/reference/debugging.md` |

## Models

| `src/` area | Owning doc page(s) |
|---|---|
| EP layer wrappers `src/distributed/expert_parallel/layers/<family>.py` | the per-family page under `agent-docs/models/` + `agent-docs/parallelism/expert-parallelism.md` (supported-models table) + `agent-docs/optimization/grouped-gemm.md` (supported-models table) |
| `src/distributed/expert_parallel/layers/zaya.py`, `src/models/patches/zaya.py` | `agent-docs/models/zaya.md` |
| `src/distributed/loading/model_loading.py`, `src/distributed/loading/warmup.py`, `src/models/loading/model_preparation.py`, `src/models/patches/remote_code_compat.py`, `src/models/patches/remote_code_hooks.py` | `agent-docs/models/README.md`, `agent-docs/models/adding-a-model.md`, the affected per-family page |
| `src/models/seq_cls_heads.py` (Gemma 4 + MoE Qwen3.5/3.6 seq-cls heads, registered by an import in `src/models/loading/model_preparation.py`) | `agent-docs/training-methods/classification.md`, `agent-docs/training-methods/preference/reward-modeling.md` |
| `src/models/loading/checkpoint_coverage.py` (random-init load gate) | `agent-docs/reference/checkpoints.md`, `agent-docs/reference/troubleshooting.md` |
| `src/models/loading/config_levels.py` (composite-config field access, run-scoped writes, `config_export_ready`) | `agent-docs/models/README.md`, `agent-docs/reference/checkpoints.md`, `agent-docs/training-methods/callbacks.md` |
| `src/models/modality.py` (multimodal checkpoint detection) | `agent-docs/data/dataset-formats.md`, `agent-docs/models/README.md` |
| `src/models/attention_geometry.py` (head-dim and KV-head resolution across composite/per-layer configs) | `agent-docs/models/README.md`, `agent-docs/optimization/flash-attention.md` |
| new model support | new `agent-docs/models/<family>.md` + model matrices in `expert-parallelism.md`/`grouped-gemm.md` + `agent-docs/models/README.md` + `CLAUDE.md` index |

## Collators & data

| `src/` area | Owning doc page(s) |
|---|---|
| `src/data/collators/` (CompletionsOnly, packing, padding-free) | `agent-docs/data/collators.md`, `agent-docs/optimization/padding-free-collator.md` |
| `src/data/collators/offline_grpo.py`, `src/data/collators/vlm_preference.py` + `src/data/pipeline/preferences.py` (per-method collators and their maps) | `agent-docs/training-methods/grpo/offline-grpo.md`, `agent-docs/training-methods/preference/reward-modeling.md` |
| `src/data/collators/factory.py` | `agent-docs/data/collators.md` |
| `src/data/spans.py` (turn terminators, completion spans, the one completion-mask implementation) | `agent-docs/data/collators.md`, `agent-docs/data/dataset-preparation.md` |
| `src/data/` (loading, processing, sharding, VLM) | `agent-docs/data/{dataset-formats,dataset-preparation,filesystem-handling}.md`, `agent-docs/parallelism/data-loading.md` |
| `src/data/sources/s3_client.py`, `src/data/sources/dataset_cache.py`, `src/data/sources/paths.py`, `scripts/before_training/s3_datasets.py` (CLI) | `agent-docs/data/s3-utilities.md` |
| `src/data/pipeline/preprocessing.py` (tokenize/pack/shard bake), `src/data/pipeline/preprocessed_metadata.py` (the `metadata.json` contract), `src/data/shard_index.py` | `agent-docs/data/dataset-preparation.md`, `agent-docs/parallelism/data-loading.md` |
| `src/data/pipeline/vlm_dataset.py` (raw-VLM map, schema, over-length filter) | `agent-docs/data/dataset-formats.md`, `agent-docs/models/README.md` |
| `src/data/probe_consensus.py` (cross-rank agreement for the data probes) | `agent-docs/data/filesystem-handling.md` |
| `src/data/deduplication.py` (embedding + FAISS corpus dedup) | `agent-docs/reference/scripts-reference.md` |

## Kernels & optimization

| `src/` area | Owning doc page(s) |
|---|---|
| `src/kernels/liger/` (orchestrator + appliers) | `agent-docs/optimization/liger-kernels.md` |
| `src/optimizers/` (AdamWBF16, Muon, FlashAdamW) | `agent-docs/optimization/{bf16-optimizer,muon-optimizer,flash-adamw}.md` |
| `src/kernels/lowp/` (quantization, linear, deepgemm, mixed_precision) | `agent-docs/optimization/low-precision-moe-kernels.md` |
| `src/kernels/grouped_gemm.py` (precision dispatch), `src/kernels/grouped_mm_autograd.py` (bf16 primitive) | `agent-docs/optimization/grouped-gemm.md` |
| `src/kernels/fused_glu.py` (fused SwiGLU kernels + `is_silu_activation` gate) | `agent-docs/models/glm4.md` (gate), `agent-docs/models/gpt-oss.md` (clamped variant) |
| attention selection (`src/models/patches/attention.py`, `_detect_attention_impl`) | `agent-docs/optimization/flash-attention.md` |
| GptOss sink policy (`src/models/patches/gpt_oss_sinks.py`) | `agent-docs/models/gpt-oss.md` (Attention sinks), the `reset_sinks`/`train_sinks` rows of `agent-docs/reference/configuration-reference.md` |
| `torch.compile` paths | `agent-docs/optimization/torch-compile.md` |
| LoRA/QLoRA (`src/distributed/loading/peft_setup.py`) | `agent-docs/optimization/peft.md` |

## Callbacks

| `src/` area | Owning doc page(s) |
|---|---|
| `src/callbacks/` (ParameterStatsCallback, GenerateExamplesCallback, EfficiencyCallback, MoEMetricsCallback, RouterBiasBalancingCallback, VariableSchedulerCallback, TorchProfilerCallback) | `agent-docs/training-methods/callbacks.md` |
| `src/callbacks/wiring.py` (`build_perf_callbacks`, `moe_balancing`) | `agent-docs/training-methods/callbacks.md`, `agent-docs/optimization/throughput-benchmarks.md` |
| `src/models/moe_balancing.py` (`resolve_balancing_mode`, the router field registries) and `src/distributed/expert_parallel/balancing_strategy.py` (`apply_balancing_strategy`, the export contract) | `agent-docs/training-methods/callbacks.md` (MoE balancing modes), `agent-docs/models/README.md` |
| `src/hardware.py` (architecture predicates, GPU model detection, peak-FLOPS registry, host-RAM probe) | `agent-docs/optimization/throughput-benchmarks.md`, `agent-docs/optimization/flash-attention.md` |

## Environments (Environmental GRPO)

| `src/` area | Owning doc page(s) |
|---|---|
| `src/environments/base.py`, `episode.py`, `registry.py` | `agent-docs/training-methods/grpo/environments/README.md`, `custom-environments.md` |
| `src/environments/envs/protocols/` (native, react, mcp) | `agent-docs/training-methods/grpo/environments/{native-tool-use,react}.md` |
| `src/environments/envs/tasks/coding/` (swe, code_contests, grading, datasets), `tasks/qa.py` | `agent-docs/training-methods/grpo/environments/{swe-environment,code-contests,mcp,benchmarks}.md` |
| `src/environments/rewards.py` (shared grader: answer extraction + match chain, also used by `scripts/training/online_grpo/rlvr.py`) | `agent-docs/training-methods/grpo/environments/benchmarks.md`, `agent-docs/training-methods/grpo/online-grpo.md` |
| `src/environments/sandbox/` (in-process + remote code execution) | `agent-docs/training-methods/grpo/environments/sandbox.md` |
| `src/environments/tools/` | `agent-docs/training-methods/grpo/environments/{native-tool-use,swe-environment}.md` |
| `src/environments/engine_wire.py` (the rollout request wire format: stop tokens, `thinking_token_budget`, reasoning effort) | `agent-docs/infrastructure/rollout-servers.md`, `agent-docs/training-methods/grpo/environmental-grpo.md` |
| `src/environments/eval_runner.py` (offline episode driver + eval report/trajectory outputs) | `agent-docs/training-methods/grpo/environments/benchmarks.md`, `agent-docs/reference/scripts-reference.md` |
| `src/environments/tools/web_search.py` (pluggable search backends) | `agent-docs/training-methods/grpo/environments/benchmarks.md`, `agent-docs/reference/configuration-reference.md` |

## Config & args

| `src/` area | Owning doc page(s) |
|---|---|
| `src/configs/`, `src/args/` (config/arg dataclasses) | `agent-docs/reference/configuration-reference.md` + the method page that owns the config |
| `src/training/parser.py` (H4ArgumentParser, toolkit defaults, the unknown-key raise) | `agent-docs/getting-started/configuration.md`, `agent-docs/reference/configuration-reference.md` |
| `src/env.py` (every `HALO_`/`DIST_`/`VLLM_` knob and its default) | `agent-docs/reference/configuration-reference.md` (Environment variables), `agent-docs/infrastructure/docker.md` |
| `src/log.py` (root logging setup, CLI verbosity, `warn_once`) | `agent-docs/reference/debugging.md` |
| `src/cli.py` (`halo launch` / `halo run` surface, tool aliases) | `README.md` quick start, `agent-docs/reference/scripts-reference.md` |

## Entry-script plumbing & clients

| `src/` area | Owning doc page(s) |
|---|---|
| `src/training/environment.py` (output-dir validation, HF cache wiring, seed, tracking vars, resume detection) | `agent-docs/getting-started/configuration.md`, `agent-docs/data/filesystem-handling.md`, `agent-docs/reference/checkpoints.md` |
| `src/training/script_runner.py` (the `scripts/training/**` backbone: window pins, tokenizer/attention resolution, callback assembly, `reject_*` guards) | `agent-docs/reference/scripts-reference.md`, `agent-docs/getting-started/configuration.md` |
| `src/training/parallelism_args.py` (`DistributedArguments` → `ParallelismConfig`, the per-script CP/PP/lowp gates) | `agent-docs/parallelism/*`, `agent-docs/reference/configuration-reference.md` |
| `src/training/run_logging.py` (per-rank transformers verbosity, the `run.log` console tee) | `agent-docs/reference/debugging.md` |
| `src/inference/openai_client.py` (OpenAI-compatible endpoint defaults, async client, parallel-request helpers), `src/inference/response.py` (`OpenAIResponse`, the finish-reason contract), `src/inference/resume_store.py` (resumable request checkpoints) | `agent-docs/reference/scripts-reference.md` |

## Scripts & infrastructure

| `src/` area | Owning doc page(s) |
|---|---|
| `scripts/training/**`, `scripts/inference/**`, `scripts/before_training/**`, `scripts/after_training/**` | `agent-docs/reference/scripts-reference.md` + the relevant method/data page |
| `scripts/profiling/**` | `agent-docs/reference/debugging.md`, `agent-docs/reference/scripts-reference.md` |
| `scripts/environments/**` (env eval runners, their shared `_common.py` flags/output writer, trajectory re-grading, coding-dataset prep) | `agent-docs/reference/scripts-reference.md`, `agent-docs/training-methods/grpo/environments/benchmarks.md` |
| `scripts/_common.py` (the checkpoint tools' shared flags: shard cap, Hub source block, `--trust_remote_code`) | `agent-docs/reference/scripts-reference.md` |
| `scripts/after_training/merge_ep_shards.py` | `agent-docs/reference/checkpoints.md` |
| `scripts/after_training/{quantize_to_lowp,convert_to_bf16}.py` | `agent-docs/optimization/low-precision-moe-kernels.md` |
| `scripts/after_training/merge_models.py` | `agent-docs/reference/scripts-reference.md` |
| `Dockerfile*`, `docker-compose*` | `agent-docs/infrastructure/docker.md`; `Dockerfile.vllm`/`Dockerfile.sglang` + their compose files also `agent-docs/infrastructure/rollout-servers.md` |
| AWS / S3 auth, `src/data/sources/s3_client.py` paths | `agent-docs/infrastructure/aws-auth.md`, `agent-docs/data/s3-utilities.md` |
| DeepEP install / NVSHMEM / CDMC notes | `agent-docs/infrastructure/deepep.md` |
| multi-node / SkyPilot / RunPod / Nomad launch | `agent-docs/parallelism/multi-node.md`, `agent-docs/infrastructure/{skypilot,runpod,nomad}.md` |
| `launcher-configs/**` (SkyPilot task YAMLs, Nomad job specs, accelerate configs) | `agent-docs/infrastructure/{skypilot,nomad}.md`, `human-docs/clusters.md` |
| `.github/workflows/**` (lint, docs, GPU tier), `Makefile` test targets | `agent-docs/infrastructure/ci.md`; tier composition lives in `agent-docs/contributing/README.md` ("Tests") |

## Cross-cutting

| Change | Owning doc page(s) |
|---|---|
| a new user-facing fail-fast raise or failure mode (config-time rejection, NCCL/DeepEP fault, OOM path) | `agent-docs/reference/troubleshooting.md` (symptom → cause → fix), plus `agent-docs/reference/debugging.md` when it needs a diagnosis helper |
| a new, moved, renamed, or deleted `src/` subpackage or top-level module | `agent-docs/reference/architecture.md` (source map) + the architecture tree in `CLAUDE.md` |

## When in doubt

- Trainer/mixin behavior → `agent-docs/reference/trainer-architecture.md`.
- Any new config field → `agent-docs/reference/configuration-reference.md`.
- A change to what parallelism combinations are valid → the parallelism matrix in
  `CLAUDE.md` and the mode list in `agent-docs/README.md`, plus the owning
  `agent-docs/parallelism/` page.
- A new page → wire it into its section `README.md` and the `CLAUDE.md` Documentation
  index, then run `./scripts/docs/check_links.sh`.
