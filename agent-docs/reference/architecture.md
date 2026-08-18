# Architecture

Halo extends HuggingFace — Transformers, Accelerate, TRL — with Expert, Context, Tensor, and
Expert-Tensor parallelism plus alignment methods TRL does not ship (SMPO, Offline
GRPO, Environmental GRPO). Every trainer subclasses a TRL, Transformers, or SentenceTransformers
trainer and adds one mixin. The default save is a standard HuggingFace checkpoint and there is no
Megatron conversion step; the one opt-in per-rank format (`save_sharded_ep`) needs a merge script
before reload.

The runtime is the prebuilt Docker image: PyTorch 2.11+cu130, DeepEP, and Flash Attention live only
inside `halo:blackwell` (B200/B300, SM100/SM103, FA2+FA4) and `halo:hopper`
(H100/H200, SM90, FA2+FA3). The host has no usable Python — everything runs inside the image
(tools on `PATH`, no prefix). See [Docker](../infrastructure/docker.md).

## Component map

| Layer | Owns | Source | Deeper page |
|---|---|---|---|
| Config | YAML parse, toolkit defaults, strict rejection of unknown or retired keys; env knobs read through one seam; the per-method argument/config dataclasses | `src/training/parser.py`, `src/env.py`, `src/args/`, `src/configs/` | [Configuration](../getting-started/configuration.md) |
| Trainers | Training loop, loss, gradient clipping, saving | `src/trainers/mixins/` (the spine is `base.py`) | [Trainer Architecture](trainer-architecture.md) |
| Distributed | EP/CP/TP/ETP/PP setup, process groups, validation, FSDP2 sharding | `src/distributed/parallelism_config.py`, `src/distributed/fsdp.py` | [Parallelism](../parallelism/index.md) |
| Data | Dataset loading, sharding, collators | `src/data/`, `src/data/collators/` | [Dataset Formats](../data/dataset-formats.md) |
| Callbacks | Throughput, MoE load, router-bias balancing | `src/callbacks/`, `src/hardware.py` (GPU detection + peak-FLOPs table) | [Callbacks](../training-methods/callbacks.md) |
| Diagnostics | Opt-in and off by default: cross-rank consistency checks, py-spy capture, torch-profiler traces, CUDA memory snapshots | `src/diagnostics/` | [Debugging](debugging.md) |
| Kernels | Grouped GEMM, Liger, Flash Attention, low-precision quantization | `src/kernels/`, `src/models/patches/attention.py` | [Grouped GEMM](../optimization/grouped-gemm.md) |
| Checkpoints | The per-mode saver ladder, weight resume, per-rank optimizer shards + LR scheduler, PEFT adapters, load-coverage gate — over the sharding-agnostic on-disk layer | `src/distributed/checkpoint/`, `src/checkpoint/`, `src/models/loading/checkpoint_coverage.py` | [Checkpoints](checkpoints.md) |
| Optimizers | AdamWBF16 (SR), Muon, FlashAdamW | `src/optimizers/` | [BF16 Optimizer](../optimization/bf16-optimizer.md) |
| Models | The sharding-agnostic model side: module-tree introspection, load-time patches (attention selection, GptOss sinks, Zaya), MoE router balancing, the `Auto*`/tokenizer/dtype preparation, and the sequence-classification heads transformers does not ship (`src/models/seq_cls_heads.py`, registered by an import in `src/models/loading/model_preparation.py`) | `src/models/` | [Adding a Model](../models/adding-a-model.md) |
| Environments | RL environment registry, Ray rollout actors, tools, sandboxes, eval runner | `src/environments/` | [Environments](../training-methods/grpo/environments/index.md) |
| Entry-script plumbing | Environment setup (output dir, HF caches, seed, resume detection), the `scripts/training/**` backbone, the `run.log` tee, root/CLI logging, and the `halo launch` / `halo run` surface | `src/training/`, `src/cli.py`, `src/log.py` | [Scripts](scripts-reference.md) |
| Served endpoints | The OpenAI-compatible client every rollout and batch-generation script talks through, its finish-reason contract, and resumable request logs | `src/inference/` | [Rollout Servers](../infrastructure/rollout-servers.md) |

Leaf modules keep those imports one-way, each holding a contract several layers share:

| Leaf | Holds | Read by |
|---|---|---|
| `src/data/spans.py` | turn terminators, completion spans, the one completion-mask implementation and the completion-only label builder over it | the collators, the offline label bake and the PP losses — none imports a collator to reach a span helper |
| `src/models/structure.py` | module-tree introspection: wrapper peeling, PEFT name normalization, decoder-layer discovery, persistent buffers, norm / fp32-pin classification | FSDP2/TP/PP wraps, the attention patches, every checkpoint writer |
| `src/checkpoint/format.py` | the on-disk checkpoint spellings, save-dtype casts, config/state-dict read-write — torch + safetensors only, no `torch.distributed` | the parallel save paths and the standalone `scripts/after_training/` tools |
| `src/data/sources/paths.py` | S3 / Hub / local classification of a dataset source or destination, pure string rules | the loader, the preprocessing pipeline and the scripts — without a boto3 import |
| `src/data/sources/dataset_cache.py` | the local cache-publish protocol (lock, completion marker, content fingerprint, atomic publish, stale-temp sweep) — `os`/`shutil`/`filelock`, the fetch injected | the S3 dataset cache and the per-shard cache of a sharded pre-processed dataset, so their crash and staleness semantics cannot drift |
| `src/data/sources/s3_client.py` | the boto3 `S3Client`, the s3fs control-file reads and the default-bucket helpers | the loader, the preprocessing pipeline, `ShardedDatasetLoader`, the inference scripts and the `scripts/before_training/s3_datasets.py` CLI |
| `src/distributed/context_parallel/key_mapping.py` | the one CP→HF attention key mapping | the EP gathered save and the PEFT adapter save — without pulling the CP wrapper stack |
| `src/distributed/checkpoint/write.py` | the collective half of a write: retain-gated DTensor resolve of params AND buffers (with the neutralized GptOss sinks), the streamed part writer, the shard-index exchange | the FSDP2/EP gathered save, the TP state dict and the CP save — the leg every gathered writer must run symmetrically or hang |
| `src/distributed/checkpoint/coordination.py` | the rank consensus and key-preview cap a resume's two halves share | `checkpoint/loader.py` (weights) and `checkpoint/optimizer.py` (optimizer shards) |
| `src/data/shard_index.py` | the torch-free `shard_index.json` contract and the stamped-sidecar writer both halves of a preprocessed artifact use | written by the preprocessing pipeline, read by `ShardedDatasetLoader` |
| `src/data/vlm.py` | the VLM chat render, the processor call and the over-length refusal | the runtime collators, the offline bake and the run-intent probe, so a batch and a bake of one row tokenize identically |
| `src/data/pipeline/preprocessed_metadata.py` | the `metadata.json` contract: the recorded `PreprocessingConfig`, the stamp and the compatibility verdicts | the training entry points and the loader, which read the stamp without importing the bake that wrote the rows |
| `src/configs/rollout_config.py` | `RolloutConfig` | built by `AsyncTrainingConfig`, received pickled by the Ray rollout actors — keeping the Ray import out of `src.configs` |

`src/distributed/runtime.py` therefore holds rank/world state, barriers, the cross-rank
rejection/consensus seams, the process-group timeouts and DTensor resolution only; anything a
single rank can compute alone lives in the leaves above. It is itself the package leaf:
`nvlink.py` (fabric probes, read by `ParallelismConfig`) and `filesystem.py` (the c10d-store
phase, main-first ordering, the output-FS probe, the load throttle) import it, never the reverse.

## Trainers

Every distributed trainer uses multiple inheritance: a base trainer (`trl.SFTTrainer`,
`transformers.Trainer`, `trl.GRPOTrainer`, …) plus `DistributedTrainerMixin`
(`src/trainers/mixins/base.py`), which composes its seven sibling sub-mixins — checkpointing,
dataloaders, EP introspection, gradient sync, parallelism validation, pipeline hooks, and token
metrics. The mixin overrides the parallelism-sensitive methods (accelerator creation,
dataloader sharding, gradient clipping, model saving) and delegates the rest.

Thirteen trainers share this shape: SFT, SMPO, DPO, KTO, offline/online/async environmental GRPO,
online SDPG, teacher and self distillation, reward, classification, and
embedding. All support EP, TP, and ETP. CP is limited to SFT and SMPO. PP is
[not yet available in this release](../parallelism/pipeline-parallelism.md); `_supports_pp` marks
SFT, SMPO, DPO, KTO, reward, classification, and offline GRPO for when it lands. The per-trainer
matrix and the reason behind each exclusion are in
[Trainer Architecture](trainer-architecture.md#trainer-compatibility).

## Distributed layer

`ParallelismConfig` (`src/distributed/parallelism_config.py`) is the single source of truth: it
validates the requested combination against an allowlist, computes `data_parallel_size`, and
creates the process groups. The trainer reads mode flags (`is_ep_mode`, `is_cp_mode`, …) off it and
never re-derives them.

| Mode | What it shards | Data parallel size | Backend |
|---|---|---|---|
| FSDP2 (default) | All params, per layer | `world_size` | `fully_shard` |
| EP | MoE experts across ranks | `world_size` (orthogonal) | DeepEP all-to-all |
| CP | The sequence axis | `world_size / cp_size` | Ulysses attention |
| TP | Attention Q/K/V/O + dense MLP; attention only on MoE (embeddings/lm_head replicated) | `world_size / tp_size` | DTensor |
| ETP (`ep_size=1`) | Expert FFN weights | `world_size / expert_tp_size` | EP wrappers |
| PP ([not yet available](../parallelism/pipeline-parallelism.md)) | The layer stack, into sequential stages | `world_size / pp_size` | torch pipelining — seams ship, the engine does not |

In full: `data_parallel_size = (world_size / pp_size) / max(tp_size, cp_size, expert_tp_size)`. EP
is orthogonal — each rank still sees different data.

Four subpackages own the mechanics: `expert_parallel/` (DeepEP dispatch/combine, per-family MoE
wrappers, grouped-GEMM expert compute, gradient-sync hooks, expert gather/export), `context_parallel/`
(`UlyssesCPModelWrapper`, sequence splitting), `tensor_parallel/` (DTensor weight sharding), and
`pipeline_parallel/` (layer split, stage module, stage-aware loading, P2P groups, the
`torch.distributed.pipelining` seam). A fifth, `loading/`, sits above all four: `load_distributed_model`
picks the per-mode loader off a `ParallelismConfig`, so it is the one place that reaches into every
implementation — which is why it lives here and not under `src/models/loading/`.

The lazy-loading machinery both the EP and PP loaders share — safetensors index resolution,
checkpoint-key alignment, per-key weight plans, hub-conversion op math, meta-shell instantiation —
lives in `src/models/loading/lazy_safetensors/` (`conversion.py`, `weights.py`, `meta_shell.py`) —
model-loading machinery with no parallelism knowledge, so it sits under `src/models/loading/`, the
sharding-agnostic half. The EP package keeps only the expert-domain knowledge: key patterns,
planner, fuser, per-family conversion/rename resolution.

The FSDP2, HSDP, and TP axes ride a torch `DeviceMesh` (`src/distributed/mesh.py`); EP and CP use
hand-built `dist.new_group` groups whose all-to-all patterns do not map to a mesh. The trainer reads
every group through one `ParallelDims` view (`src/distributed/mesh.py`), and the bucketed
gradient all-reduce every post-backward sweep shares — deferred EP cross-replica, TP replicated,
QLoRA — is a torch-only leaf (`src/distributed/grad_reduce.py`) that imports no parallelism
implementation.

Which axis combinations may run is an allowlist, not a denylist — see
[Parallelism](../parallelism/index.md#communication-and-data-flow).

## A training step

```text
YAML config
   │  H4ArgumentParser: toolkit defaults (use_liger_kernel, bf16, logging_nan_inf_filter)
   ▼
ParallelismConfig  ── validate mode, build process groups
   │
   ▼
load_distributed_model            src/distributed/loading/model_loading.py
   │  • resolve_attn_implementation → FA4 (SM100+) / FA3 (SM90) / FA2; per-model SDPA/eager overrides
   │  • patch_moe_model_for_ep → EP wrappers (when ep_size > 1 or grouped-GEMM)
   │  • UlyssesCPModelWrapper (when cp_size > 1)
   │  • load_pp_stage_model → this stage's decoder layers only (when pp_size > 1)
   ▼
DistributedTrainerMixin._setup_distributed_modes
   │  • FSDP2 fully_shard (EP modules in ignored_params)
   │  • router/expert grad-sync hooks already attached at EP-wrapper construction (load);
   │    PEFT modules_to_save router copies re-hooked here
   ▼
trainer.train()  ── HF training loop, DDP wrapping skipped
   │  • forward → loss → backward
   │  • EP-aware global grad-norm clip (DTensor + replica groups)
   │  • AdamWBF16 step (bf16 weights, stochastic rounding — no fp32 master)
   │  • callbacks: EfficiencyCallback (tokens/s/GPU), MoEMetricsCallback,
   │               RouterBiasBalancingCallback
   ▼
trainer.save_model  ── save_checkpoint (src/distributed/checkpoint/) gathers shards → HF checkpoint
```

The parser applies `use_liger_kernel: true`, `bf16: true` and `logging_nan_inf_filter: false` when
the YAML omits them. It migrates no spelling: every key no config declares raises the unknown-key
error. See the [Configuration Guide](../getting-started/configuration.md).

`resolve_attn_implementation` (`src/models/patches/attention.py`) auto-selects the attention
backend from compute capability: `flash_attention_4` on SM100+ when `flash_attn.cute` imports,
`flash_attention_3` on Hopper, FA2 otherwise. Per-model overrides then redirect the families whose
head geometry or sinks a flash kernel cannot serve — Qwen3.5/3.6/Qwen3-Next, GLM-4 MoE Lite and
Gemma4 to SDPA, DeepSeek-V4 to eager. See [Flash Attention](../optimization/flash-attention.md) for
the per-model table and the reason behind each redirect.

For MoE models the loader replaces each MoE block with the per-family EP wrapper holding this
rank's expert slice. FSDP2 `fully_shard` then shards the non-expert params, with EP modules in
`ignored_params` so their gradients sync through the manual hooks instead. Saving gathers the
distributed shards back into a standard HuggingFace checkpoint — see
[Checkpoints & Resume](checkpoints.md).

## RL: the rollout engine as a separate container

Online and Environmental GRPO generate completions with vLLM (0.26.0). vLLM pins its own
torch/transformers stack, so it is never imported into the training environment — it runs as its
own container (`Dockerfile.vllm` + `docker-compose.vllm.yml`). Environmental GRPO can target SGLang
instead (`rollout_backend: sglang`, `Dockerfile.sglang` + `docker-compose.sglang.yml`), under
narrower model and expert-distribution limits — see
[Rollout Servers](../infrastructure/rollout-servers.md).

The training process talks to it over two channels: HTTP for generation, and a vendored NCCL client
(`src/distributed/nccl/`, `VLLMWeightSyncClient`) for weight sync, replacing TRL's `VLLMClient`
which would pull in the vLLM package.

Before each generation round that follows a weight update (environmental GRPO: every
`sync_weights_every_n_steps`), the trainer gathers EP expert shards, unfolds FSDP2 DTensors via
`full_tensor()`, gathers TP shards, pushes the weights to vLLM over NCCL, and resets the prefix
cache. See [Online GRPO](../training-methods/grpo/online-grpo.md) and
[Environmental GRPO](../training-methods/grpo/environmental-grpo.md).

## Related pages

- [Trainer Architecture](trainer-architecture.md) — the mixin lifecycle, gradient sync, per-trainer notes
- [Why This Framework](why-this-framework.md) — comparison to TRL, Unsloth, Megatron-LM, veRL, MS-SWIFT
- [Parallelism](../parallelism/index.md) — EP, CP, TP, ETP, PP and valid combinations
- [Configuration](../getting-started/configuration.md) — the YAML system and toolkit defaults
- [Checkpoints & Resume](checkpoints.md) — checkpoint formats and the two resume paths
