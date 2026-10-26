# Architecture

Halo extends HuggingFace — Transformers, Accelerate, TRL — with Expert, Context, Tensor, and
Expert-Tensor parallelism plus alignment methods TRL does not ship (SMPO, Offline GRPO) and Async
GRPO with Environments. Every trainer subclasses a TRL, Transformers, or SentenceTransformers trainer
and adds `DistributedTrainerMixin`.

The default save is a standard HuggingFace checkpoint and there is no Megatron conversion step; the
one opt-in per-rank format (`save_sharded_ep`) needs a merge script before reload.

Everything runs inside the prebuilt image (`halo:blackwell` / `halo:hopper`); the
host has no usable Python ([Docker](../infrastructure/docker.md)).

## Component map

| Layer | Owns | Source | Deeper page |
|---|---|---|---|
| Config | YAML parse, toolkit defaults, strict rejection of unknown or retired keys; env knobs read through one seam; the per-method argument/config dataclasses and the field guards and argument refusals they share | `src/configs/` (`parser.py` among them), `src/env.py`, `src/config_validation.py` | [Configuration](../getting-started/configuration.md) |
| Trainers | Training loop, loss, gradient clipping, saving | `src/trainers/mixins/` (the spine is `base.py`) | [Trainer Architecture](trainer-architecture.md) |
| Runtime | Process plumbing every layer above shares: rank and node math, process-group construction and timeouts, barriers and broadcasts, cross-rank verdicts and the c10d-store joins, the shared-filesystem scope and main-first ordering, the per-node load throttle, DTensor access | `src/runtime/` | [Multi-Node](../parallelism/multi-node.md) |
| Distributed | EP/CP/TP/ETP/PP setup, process groups, validation, FSDP2 sharding | `src/distributed/parallelism_config.py`, `src/distributed/fsdp.py` | [Parallelism](../parallelism/README.md) |
| Data | Dataset loading, sharding, collators | `src/data/`, `src/data/collators/` | [Dataset Formats](../data/dataset-formats.md) |
| Callbacks | Throughput, MoE load, router-bias balancing | `src/callbacks/`, `src/hardware.py` (GPU detection + peak-FLOPs table) | [Callbacks](../training-methods/callbacks.md) |
| Diagnostics | Opt-in and off by default: py-spy capture, torch-profiler traces, CUDA memory snapshots, EP span timing, the weight-sync transport preflight | `src/diagnostics/` | [Debugging](debugging.md) |
| Kernels | Grouped GEMM, the fused GLU and MoE permute kernels, chunked fp32 log-probs, Liger, Flash Attention and FlexAttention on sliding-window layers, low-precision quantization | `src/kernels/`, `src/models/attention/selection.py`, `src/models/attention/sdpa_flex_sliding.py` | [Grouped GEMM](../optimization/grouped-gemm.md) |
| Checkpoints | The flat saver ladder and its rungs (PEFT and expert-LoRA adapters, PP, EP, CP, TP, FSDP2), weight resume, per-rank optimizer shards + LR scheduler, load-coverage gate — over the sharding-agnostic on-disk layer | `src/checkpoint/parallel/`, `src/checkpoint/`, `src/model_loading/checkpoint_coverage.py` | [Checkpoints](checkpoints.md) |
| Optimizers | AdamWBF16 (SR), Muon, FlashAdamW | `src/optimizers/` | [BF16 Optimizer](../optimization/bf16-optimizer.md) |
| Models | The sharding-agnostic model side: module-tree introspection, composite-config access, the context window, load-time patches (attention selection, GptOss sinks, Zaya), MoE router balancing, the head-path transform contract, and the sequence-classification heads transformers does not ship (`src/models/seq_cls_heads.py`, registered by an import in `src/model_loading/model_preparation.py`) | `src/models/` | [Adding a Model](../models/adding-a-model.md) |
| Model loading | Sharding-agnostic model construction: the `Auto*` class and the coverage-verified `from_pretrained` call, the family patches and run policies every loaded model takes, the run dtype and the loaded-parameter cast, the processing class and its `model_max_length` pin, remote-code compatibility, the post-load buffer repair, Liger applied at load, and the lazy safetensors core the EP and PP loaders share. Above it, `parallel/` holds the parallelism-aware loads: the `load_distributed_model` dispatcher (`loader.py`), the refusals it runs before any weight (`load_rules.py`), the `ModelSource` every rank agrees on (`model_source.py`), the whole-model load core (`whole_model.py`), the TP and CP loaders (`tp.py`, `cp.py`), the frozen reference and teacher loads, and the PEFT and freeze-pattern setup | `src/model_loading/` | [Models](../models/README.md) |
| Rewards | Config-parsed reward terms (`terms.py`), the sample and the views a scorer reads (`samples.py`), the scorers behind the external sources (`scorers/`: the contract, the generative judge — scoring or veto — the served reward model, their catalog), the settlement of verdicts into one reward (`composer.py`), the TRL adapters (`functions.py`) and the rule-based graders (`graders/`: RLVR graders, answer matching) | `src/rewards/` | [Reward Terms](../training-methods/grpo/rewards/README.md) |
| Environments | RL environment registry, Ray rollout actors, tools, sandboxes, eval runner | `src/environments/` | [Environments](../training-methods/async-grpo/environments/README.md) |
| Weight sync | Trainer-to-rollout-engine weight sync: one client per engine (`VLLMWeightSyncClient`, `SGLangWeightSyncClient`) and the backend registry, the chunking (the chunk budget, the co-load groups, the snapshots on the sync device), and the NCCL transport — the rendezvous listener and the group-address policy, the bounded broadcast waits, the vLLM pynccl communicator and packed broadcast, the SGLang c10d weight-update group, the send-stream buffers | `src/weight_sync/` | [Rollout Servers](../infrastructure/rollout-servers.md) |
| Entry-script plumbing | The `scripts/training/**` backbone (`parse_script_args`, `ScriptContext`, the run lifecycle and teardown), run setup (output dir, HF caches, seed, resume detection, the `run.log` tee), the policy load and window pins, script dataset loading; root/CLI logging and the `halo launch` / `halo run` surface | `src/entrypoints/`, `src/cli.py`, `src/log.py` | [Scripts](scripts-reference.md) |
| Served endpoints | The OpenAI-compatible client every rollout, judge and batch-generation script talks through: endpoint defaults, key resolution and the loopback test (`endpoints.py`), the async client, one request with the upstream retry and the single-request helper that normalizes a reply (`openai_client.py`), the parallel resumable batch with its JSONL resume store (`batch_requests.py`), and the response record with its finish-reason contract (`response.py`) | `src/inference/` | [Rollout Servers](../infrastructure/rollout-servers.md) |

Leaf modules keep those imports one-way, each holding a contract several layers share:

| Leaf | Holds | Read by |
|---|---|---|
| `src/data/spans.py` | turn terminators, completion spans, the one completion-mask implementation and the completion-only label builder over it | the collators, the offline label bake and the PP losses — none imports a collator to reach a span helper |
| `src/models/unwrapping.py` | wrapper peeling (framework, toolkit, PEFT) and the plain hub spelling of a name read off a wrapped tree | the FSDP2 wrap, the CP wrapper, the weight sync, the MoE balancing, every checkpoint writer |
| `src/models/module_tree.py` | module-tree lookups: the decoder-layer list and the layer index in a dotted name, the backbone owning the list, the module consuming the input embedding, the embedding modules, module paths by identity, the attribute probe | the FSDP2/TP/PP wraps, the attention patches, the PEFT setup, the low-precision install, the routing replay |
| `src/models/param_roles.py` | parameter roles: embedding/head markers, normalization modules, persistent buffers, fp32 pins | every checkpoint writer, the FSDP2 wrap and the lazy loaders (fp32 pins), Muon and the PEFT setup |
| `src/models/peft_adapters.py` | PEFT adapter parameters and the out-of-place LoRA fold, the value PEFT's merge would write | the weight sync, the merged EP and FSDP2 saves, the embedding trainer's save, the PEFT validation |
| `src/models/gradient_checkpointing.py` | the recompute scope both passes of one checkpoint invocation share, expert-load counting off it, and the scoped gradient-checkpointing enable | the EP layers' dispatch replay, the router-bias load counters, the EP trainer setup |
| `src/models/modality.py` | the multimodal verdict off a config in hand or a name; no hub read, no collective (the collective probe is `probe_checkpoint` in `src/model_loading/parallel/model_source.py`) | the loaders, the run-level `is_vlm_run`, the PP gate |
| `src/models/segment_markers.py` | which families' conv / linear-attention mixers read per-document segment markers, the GatedDeltaNet kernel refusal, and the markers built from a row's `position_ids` | the collator factory, the packing and padding-free collators, and SMPO's padding-free forward |
| `src/models/attention/layout.py` | per-layer attention cost rules off `layer_types` and head geometry — the MFU attention term | the token-metrics mixin and the efficiency callbacks |
| `src/models/moe/family_specs.py` | every MoE family's facts, one frozen `MoEFamilySpec` each (model types, balancing slot, export renames and config rewrites, expert-weight roots, capability flags), indexed by `model_type` | the EP layers (`FAMILY`), the loaders, the config writers, the weight sync and the offline checkpoint tools — none imports the EP package to read a family fact |
| `src/models/head_transform.py` | the head-path contract: each family's declared transform around `lm_head` (scale, softcap, vocabulary cut), verified against its own forward on a meta-device shell | the chunked GRPO log-prob sweep (and the PP stage seam), which apply the same verdict |
| `src/checkpoint/atomic.py` | exclusive staging with ordinary umask permissions, atomic Torch-file publication, directory fsync — no rank coordination | model-card writes, reference sidecars, the optimizer metadata, the async GRPO prefetch sidecar, the trainer-state commit, and checkpoint export filtering |
| `src/checkpoint/format.py` | every on-disk checkpoint name two modules share, and the builders that spell them — no model | the parallel save and load paths and the standalone `scripts/after_training/` tools |
| `src/checkpoint/reading.py` | raw checkpoint reads: the layout cascade, the JSON configs, the streaming reader, the partial-checkpoint refusals, the tools' shard walks and header sizing — no model | the resume loaders, the lazy loaders and the standalone tools |
| `src/checkpoint/hub_layout.py` | the transformers load conversions a save reverts to write the hub's tensor names, and `ModuleLayoutView`, which reads a saved layout under the live names | every gathered, TP, PP and EP writer, the RL weight sync and the resume loaders |
| `src/checkpoint/writing.py` | save-dtype casts, the stale-file sweep, the merged index, the whole-dict and streaming shard writers, the tied-embedding reconcile | the parallel save paths and the standalone `scripts/after_training/` tools |
| `src/data/sources/paths.py` | S3 / Hub / local classification of a dataset source or destination, pure string rules | the loader, the preprocessing pipeline and the scripts — without a boto3 import |
| `src/data/sources/dataset_cache.py` | the local cache-publish protocol (lock, completion marker, content fingerprint, atomic publish, stale-temp sweep) — `os`/`shutil`/`filelock`, the fetch injected | the S3 dataset cache and the per-shard cache of a sharded pre-processed dataset, so their crash and staleness semantics cannot drift |
| `src/data/sources/s3_client.py` | the boto3 `S3Client`, the s3fs control-file reads and the default-bucket helpers | the loader, the preprocessing pipeline, `ShardedDatasetLoader`, the inference scripts and the `scripts/before_training/s3_datasets.py` CLI |
| `src/checkpoint/parallel/writing.py` | the collective half of a gathered write: retain-gated DTensor resolve of params AND buffers (with the neutralized GptOss sinks), the conversion-closed chunks and the streamed writer they feed | the FSDP2, CP, TP and EP gathered saves, the PP stage writes and the embedding save — the leg every gathered writer must run symmetrically or hang |
| `src/checkpoint/parallel/reading.py` | the two reads a resume shares: one file every rank agrees on (`consensus_read`), and a checkpoint's shards opened on every rank (`joined_streaming_reader`) | the weight resume, the optimizer shards, the adapter restore and the sidecars |
| `src/checkpoint/parallel/resume_source.py` | which checkpoint a run resumes from (the newest complete on every rank, incomplete ones moved out of rotation's sight) and which weights its policy is built from | resume detection (`src/entrypoints/run_setup.py`, the trainer's `train(resume_from_checkpoint=True)`) |
| `src/data/shard_index.py` | the torch-free `shard_index.json` contract and the stamped-sidecar writer both halves of a preprocessed artifact use | written by the preprocessing pipeline, read by `ShardedDatasetLoader` |
| `src/data/vlm.py` | the VLM chat render, the processor call and the over-length refusal | the runtime collators, the offline bake and the run-intent probe, so a batch and a bake of one row tokenize identically |
| `src/data/pipeline/preprocessed_metadata.py` | the `metadata.json` contract: the recorded `PreprocessingConfig`, the stamp and the compatibility verdicts | the training entry points and the loader, which read the stamp without importing the bake that wrote the rows |
| `src/environments/rollout_config.py` | `RolloutConfig` and the rollout defaults `AsyncTrainingConfig` reads | built by `AsyncTrainingConfig`, received pickled by the Ray rollout actors — keeping the Ray import out of `src.configs` |
| `src/config_validation.py` | the bounds guards, the validation base, the declared-default comparison and the argument refusals — standard library only | the argument and config dataclasses, the reward terms, the training scripts and the Muon builder |
| `src/subclass_registry.py` | the subclass-tree walk, the class-claim maps (a key claimed twice raises), the swap of claimed modules and the package import that registers every family — torch only | the EP and CP patchers, the head-path contract, the EP `model_type` roster and the weight-sync backend registry, so a family registers by subclassing, with no list to edit |
| `src/rewards/terms.py` | the typed reward terms and their config parse — the endpoint defaults and the `src/config_validation.py` guards are its only first-party imports | the argument dataclasses and `EnvironmentConfig`, which parse `rewards:` at config time without pulling in the scorers' SDK client |
| `src/inference/endpoints.py` | the OpenAI-compatible endpoint defaults, key resolution and the loopback test (`is_loopback`) — `src/env.py` only | the reward terms, the CLI endpoint flag block (`scripts/_common.py`), the judge clients, the Ray rollout actors and the weight-sync group-address policy — none of them imports the SDK |

`src/runtime/` therefore holds only what needs a process group or the launcher: rank and node math
(`ranks.py`), `init_distributed` and the subgroup timeout pin (`process_group.py`), barriers,
broadcasts and fixed-order group creation (`collectives.py`), the cross-rank rejection/consensus seams
and the c10d-store joins (`consensus.py`), the shared-filesystem scope, FS-aware writer ranks,
main-first ordering and the output-FS probe (`filesystem.py`), the per-node load throttle
(`load_coordination.py`) and DTensor access (`dtensor.py`); anything a single rank can compute alone
lives in the leaves above. It imports only `src.env`, `src.log` and `src.hardware`, and the
parallelism strategies in `src/distributed/` import it, never the reverse.

## Trainers

Every distributed trainer uses multiple inheritance: a base trainer (`trl.SFTTrainer`,
`transformers.Trainer`, `trl.GRPOTrainer`, …) plus `DistributedTrainerMixin`
(`src/trainers/mixins/base.py`), which composes the sub-mixins listed in
[Trainer Architecture](trainer-architecture.md). The mixin overrides the parallelism-sensitive
methods (accelerator creation, dataloader sharding, gradient clipping, model saving) and delegates
the rest.

Thirteen trainers share this shape: SFT, SMPO, DPO, KTO, offline/online/async environmental GRPO,
online SDPG, teacher and self distillation, reward, classification, and
embedding. All support EP, TP, and ETP. CP supports SFT, SMPO, and offline GRPO full fine-tuning.
[PP](../parallelism/pipeline-parallelism.md) supports SFT, SMPO, DPO, KTO, reward, classification, and
offline GRPO. The per-trainer
matrix and the reason behind each exclusion are in
[Trainer Architecture](trainer-architecture.md#trainer-compatibility).

Offline GRPO's frozen-reference scores span three modules: `src/trainers/grpo/reference_cache.py`
(bounded, filesystem-aware score storage and the memory-mapped token buffers),
`src/trainers/grpo/mixins/offline_reference.py` (the ragged payload, the training-checkpoint contract
and evaluation's original-reference scores), and the shared `src/trainers/mixins/reference_logps.py` (checkpoint identity and
persistence, which DPO/KTO precompute also use). Storage, resume and recovery:
[Offline GRPO → Reference model](../training-methods/grpo/offline.md#reference-model).

## Distributed layer

`ParallelismConfig` (`src/distributed/parallelism_config.py`) is the single source of truth: it
validates the requested combination against the allowlist and refusals in
`src/distributed/parallelism_rules.py`, computes `data_parallel_size`, and builds the `EPConfig` /
`CPConfig` process groups, whose membership comes from the layouts in `src/distributed/group_layout.py`; the DeviceMeshes come from
`src/distributed/mesh.py`. The trainer reads mode flags (`has_ep_group`, `is_cp_mode`, …) off it and
never re-derives them.

What each mode shards and its backend: [Parallelism → Parallelism modes](../parallelism/README.md#parallelism-modes);
its data-parallel size: [Supported combinations](../parallelism/README.md#supported-combinations).

Four subpackages own the mechanics: `expert_parallel/` (DeepEP dispatch/combine, per-family MoE
wrappers, grouped-GEMM expert compute, gradient-sync hooks, expert gather), `context_parallel/`
(`UlyssesModel`, sequence splitting), `tensor_parallel/` (DTensor weight sharding), and
`pipeline_parallel/` (layer split, stage module, stage-aware loading, P2P groups, the
`torch.distributed.pipelining` seam). Inside `expert_parallel/`, `moe_layer/` composes `EPMoELayerBase` from
one mixin per concern; its `compute`, `routing`, `grad_sync` and `balancing_bias` modules are imported only
from inside `expert_parallel/`, and everything else reaches them through layer instances
(`named_ep_layers`).

`src/model_loading/parallel/` sits above all four: `load_distributed_model` picks the per-mode loader
off a `ParallelismConfig`, so it is the one place that reaches into every implementation, and the rest
of `src/model_loading/` stays sharding-agnostic. The collective checkpoint half, `src/checkpoint/parallel/`
(the saver ladder in `save.py`, the EP saves in `ep.py`, the PP saves in `pp.py`, the adapter saves in
`adapters.py`, and the resume beside them), sits above them too: it gathers through each implementation.
These packages import each other acyclically
(`tests/cpu/conventions/test_import_layering.py`).

No parallelism package imports `src/model_loading/parallel/`, so what both sides need sits in the
parallelism packages: `expert_parallel/lazy_plan.py` holds the expert-key vocabulary, the lazy gate, the
per-key weight planner, the family key mapping and the prologue both lazy loaders share, and
`expert_parallel/expert_fuser.py` stacks a per-expert checkpoint's local experts. The loaders themselves
sit in the dispatch package: `ep.py`, `pp.py`, and `fp32_masters.py`, which selects the configured FP32
parameter masters off the EP layer registry, captures them for a lazy load and streams their stored
values back for eager construction ([Load precision](../models/README.md#load-precision)).

The machinery both lazy loaders share — conversion-op math and key alignment (`conversion.py`), which
transformers conversions apply (`conversion_selection.py`), per-key weight plans and the shared stream
tail (`materialization.py`), meta-shell instantiation (`meta_shell.py`) — lives in
`src/model_loading/lazy_safetensors/`, with no parallelism knowledge.

The FSDP2, HSDP, and TP axes ride a torch `DeviceMesh` (`src/distributed/mesh.py`); EP and CP use
hand-built `dist.new_group` groups whose all-to-all patterns do not map to a mesh. The trainer reads
the mesh groups (DP, TP) through the `ParallelDims` view (`src/distributed/mesh.py`) and the expert
groups (dispatch, expert-TP, expert-replica) off `EPConfig`. The bucketed
gradient all-reduce and the grad-presence agreement every post-backward sweep shares — deferred EP
cross-replica, TP replicated, QLoRA — live in a leaf (`src/distributed/grad_reduce.py`) that
imports torch, `src.env` and `src.runtime`, and no parallelism implementation.

Which axis combinations may run is an allowlist, not a denylist — see
[Parallelism](../parallelism/README.md#supported-combinations).

## A training step

```text
YAML config
   │  HaloArgumentParser: toolkit defaults (use_liger_kernel, bf16, logging_nan_inf_filter)
   ▼
ParallelismConfig  ── validate mode, build process groups
   │
   ▼
load_distributed_model            src/model_loading/parallel/loader.py
   │  • resolve_attn_implementation → FA4 (SM100+) / FA3 (SM90) / FA2; per-model SDPA/eager overrides
   │  • apply_ep_layers → EP wrappers (when ep_group_size > 1 — EP or ETP — or grouped GEMM on an MoE)
   │  • UlyssesModel (when cp_size > 1)
   ▼
DistributedTrainerMixin._setup_distributed_modes
   │  • FSDP2 fully_shard (EP modules in ignored_params, except at ep_group_size == 1)
   │  • router/expert grad-sync hooks already attached at EP-wrapper construction (load);
   │    PEFT modules_to_save router copies re-hooked here
   ▼
trainer.train()  ── HF training loop, DDP wrapping skipped
   │  • forward → loss → backward
   │  • EP-aware global grad-norm clip (DTensor + replica groups)
   │  • AdamWBF16 step (bf16 weights, stochastic rounding — no fp32 master)
   │  • callbacks: EfficiencyCallback (tokens/s/GPU), MoEMetricsCallback,
   │               BalancingBiasCallback
   ▼
trainer.save_model  ── save_checkpoint (src/checkpoint/parallel/) gathers shards → HF checkpoint
```

The parser applies `use_liger_kernel: true`, `bf16: true` and `logging_nan_inf_filter: false` when
the YAML omits them. It migrates no spelling: every key no config declares raises the unknown-key
error. See the [Configuration Guide](../getting-started/configuration.md).

`resolve_attn_implementation` (`src/models/attention/selection.py`) auto-selects the attention
backend from compute capability: `flash_attention_4` on SM100+ when `flash_attn.cute` imports,
`flash_attention_3` on Hopper, FA2 otherwise. Per-model overrides then redirect the families whose
head geometry or sinks a flash kernel cannot serve; the per-model table and the reason behind each
redirect are on [Flash Attention](../optimization/flash-attention.md).

For MoE models the loader replaces each MoE block with the per-family EP wrapper holding this
rank's expert slice. FSDP2 `fully_shard` then shards the non-expert params, with EP modules in
`ignored_params` so their gradients sync through the manual hooks instead — except at
`ep_group_size == 1`, where `fsdp_shard_ep1_experts` (default on) has FSDP2 shard the experts too
and its reduce-scatter is their only sync. Saving gathers the
distributed shards back into a standard HuggingFace checkpoint — see
[Checkpoints & Resume](checkpoints.md).

## RL: the rollout engine as a separate container

Online and async GRPO generate completions with vLLM (0.26.0). vLLM pins its own
torch/transformers stack, so it is never imported into the training environment — it runs as its
own container (`Dockerfile.vllm` + `docker-compose.vllm.yml`). Async GRPO can target SGLang
instead (`rollout_backend: sglang`, `Dockerfile.sglang` + `docker-compose.sglang.yml`; the sync needs that image, whose
`docker/sglang/patches/` repair two loaders and the prefill log-prob offset). Each
engine's pinned loaders refuse a few families — see
[Rollout Servers](../infrastructure/rollout-servers.md#which-families-each-engine-serves).

The training process talks to it over two channels: HTTP for generation, and an NCCL weight-sync client
(`src/weight_sync/clients/`, one per engine: `VLLMWeightSyncClient` / `SGLangWeightSyncClient`,
over the NCCL transport in `src/weight_sync/transport/`) for weight sync. Online GRPO hands TRL
`TrlVLLMClient` (`src/trainers/grpo/rollout/`), the vLLM client plus TRL's generation contract, in place
of TRL's `VLLMClient`, which would pull in the vLLM package. A server on another node syncs over EFA
([Rollout Servers](../infrastructure/rollout-servers.md#servers-on-other-nodes-efa)).

Before each generation round that follows a weight update (async GRPO: every
`sync_weights_every_n_steps`), the trainer gathers EP expert shards, unfolds FSDP2 DTensors via
`full_tensor()`, gathers TP shards, pushes the weights to the rollout server over NCCL, and resets
the prefix cache. See [Online GRPO](../training-methods/grpo/online.md) and
[Async GRPO](../training-methods/async-grpo/README.md).

## Related pages

- [Trainer Architecture](trainer-architecture.md) — the mixin lifecycle, gradient sync, per-trainer notes
- [Why This Framework](why-this-framework.md) — comparison to TRL, Unsloth, Megatron-LM, veRL, MS-SWIFT
- [Parallelism](../parallelism/README.md) — EP, CP, TP, ETP, PP and valid combinations
- [Configuration](../getting-started/configuration.md) — the YAML system and toolkit defaults
- [Checkpoints & Resume](checkpoints.md) — checkpoint formats and the two resume paths
