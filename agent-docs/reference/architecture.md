# Architecture

Halo extends HuggingFace — Transformers, Accelerate, TRL — with Expert, Context, Tensor,
Expert-Tensor, and Pipeline parallelism plus alignment methods TRL does not ship (SMPO, Offline GRPO) and Async
GRPO with Environments. Every trainer subclasses a TRL, Transformers, or SentenceTransformers trainer
and adds `DistributedTrainerMixin`.

The default save is a standard HuggingFace checkpoint and there is no Megatron conversion step; the
one opt-in per-rank format (`save_sharded_ep`) needs a merge script before reload.

Everything runs inside the prebuilt image (`halo:blackwell` / `halo:hopper`); the
host has no usable Python ([Docker](../infrastructure/docker.md)).

## Component map

| Layer | Owns | Source | Deeper page |
|---|---|---|---|
| Config | YAML parse, toolkit defaults, strict rejection of unknown or retired keys; env knobs read through one seam; the per-method argument/config dataclasses and the field guards and argument refusals they share | `halo/configs/` (`parser.py` among them), `halo/env.py`, `halo/config_validation.py` | [Configuration](configuration-reference.md#parsing) |
| Trainers | Training loop, loss, gradient clipping, saving | `halo/trainers/` (the spine is `distributed_trainer.py`) | [Trainer Architecture](trainer-architecture.md) |
| Runtime | Process plumbing every layer above shares: rank and node math, process-group construction and timeouts, barriers and broadcasts, cross-rank verdicts and the c10d-store joins, the shared-filesystem scope and main-first ordering, the per-node load throttle, DTensor access | `halo/runtime/` | [Multi-Node](../parallelism/multi-node.md) |
| Distributed | EP/CP/TP/ETP/PP setup, process groups, validation, FSDP2 sharding | `halo/distributed/parallelism_config.py`, `halo/distributed/fsdp.py` | [Parallelism](../parallelism/README.md) |
| Data | Dataset loading, sharding, collators; the dataset tools' drivers (semantic dedup, the offline preparation's run and publish, the S3 commands) | `halo/data/`, `halo/data/collators/` | [Dataset Formats](../data/dataset-formats.md) |
| Callbacks | Throughput, MoE load, router-bias balancing, the checkpointing plan, the MTP acceptance gate | `halo/callbacks/`, `halo/hardware.py` (GPU detection + peak-FLOPs table) | [Callbacks](../training-methods/callbacks.md) |
| Diagnostics | Opt-in and off by default: py-spy capture, torch-profiler traces, CUDA memory snapshots | `halo/diagnostics/` | [Debugging](debugging.md) |
| Kernels | Grouped GEMM, the fused GLU, MoE routing-sort and permute kernels, the hyper-connection kernels, the DSA attention backward (`masked_softmax.py`), chunked fp32 log-probs, Flash Attention and FlexAttention on sliding-window and packed layers, low-precision quantization, the JIT kernel caches and the Liger implementation pin (`jit_cache.py`), FLA's launch options (`fla_launch.py`) | `halo/kernels/`, `halo/models/attention/selection.py`, `halo/models/attention/sdpa_flex_sliding.py` | [Grouped GEMM](../optimization/grouped-gemm.md) |
| Checkpoints | The flat saver ladder and its rungs (PEFT and expert-LoRA adapters, PP, EP, CP, TP, FSDP2), weight resume, per-rank optimizer shards + LR scheduler, load-coverage gate — over the sharding-agnostic on-disk layer; beside it, `tools/` holds the offline checkpoint tools' drivers (shard and model merges, the bf16, low-precision and fp8 conversions, sink reset, expert unfuse, vision re-attach, vocabulary patch) | `halo/checkpoint/parallel/`, `halo/checkpoint/`, `halo/checkpoint/tools/`, `halo/model_loading/checkpoint_coverage.py` | [Checkpoints](checkpoints.md) |
| Optimizers | AdamWBF16 (SR), Muon, FlashAdamW | `halo/optimizers/` | [BF16 Optimizer](../optimization/bf16-optimizer.md) |
| Models | The sharding-agnostic model side: module-tree introspection, composite-config access, the context window, load-time patches (attention selection, GLM-5 Next's DSA masked matmul, GptOss sinks, Zaya, Nemotron-H's Mamba scan, Qwen3.8-Flash-Next's host n-gram table, hyper-connections), MoE router balancing, the head-path transform contract, the released MTP heads (`halo/models/mtp/`), and the sequence-classification heads transformers does not ship (`halo/models/seq_cls_heads.py`, registered by an import in `halo/model_loading/model_preparation.py`) | `halo/models/` | [Adding a Model](../models/adding-a-model.md) |
| Model loading | Sharding-agnostic model construction: the `Auto*` class and the coverage-verified `from_pretrained` call that leaves the full weights on every rank, the family patches and run policies every loaded model takes, the run dtype and the loaded-parameter cast, the processing class and its `model_max_length` pin, remote-code compatibility, the post-load repair (non-persistent buffers, FLA's launch options, the external tables a family holds outside its module state), Liger applied at load (`liger/`), and the lazy safetensors core the EP and PP loaders share. Above it, `parallel/` holds the parallelism-aware loads: the `load_distributed_model` dispatcher (`loader.py`), the refusals it runs before any weight (`load_rules.py`), the `ModelSource` every rank agrees on (`model_source.py`), the whole-model load core (`whole_model.py`), the TP and CP loaders (`tp.py`, `cp.py`), the frozen reference and teacher loads, and the PEFT and freeze-pattern setup | `halo/model_loading/` | [Models](../models/README.md) |
| Rewards | Config-parsed reward terms (`terms.py`), the sample and the views a scorer reads (`samples.py`), the scorers behind the external sources (`scorers/`: the contract, the generative judge — scoring or veto — the served reward model), the scorer catalog and the settlement of verdicts into one reward (`composer.py`), the TRL adapters (`trl_reward_functions.py`) and the rule-based graders (`graders/`: RLVR graders, answer matching) | `halo/rewards/` | [Reward Terms](../training-methods/grpo/rewards/README.md) |
| Environments | RL environment registry, Ray rollout actors, tools, sandboxes, eval runner, the code-contest test-suite compaction | `halo/environments/` | [Environments](../training-methods/async-grpo/environments/README.md) |
| Weight sync | Trainer-to-rollout-engine weight sync: one client per engine (`VLLMWeightSyncClient`, `SGLangWeightSyncClient`), the backend registry and the pool over several servers, with the one NCCL group of `rollout_sync_topology: single_group` (`group.py`, its rank layout in `group_layout.py`, the `auto` resolution in `group_eligibility.py`), both over the update stream they share (`update_stream.py`: the chunk buffer, the open, the send and the close), the chunking (the chunk budget, the co-load groups, the snapshots on the sync device), and the NCCL transport — the rendezvous listener and the group-address policy, the bounded broadcast waits, the vLLM pynccl communicator and packed broadcast, the SGLang c10d weight-update group, the send-stream buffers; and the transport preflight (`transport_check.py`) | `halo/weight_sync/` | [Rollout Servers](../infrastructure/rollout-servers.md) |
| Entry-script plumbing | The `scripts/training/**` backbone (`parse_script_args`, `ScriptContext`, the run lifecycle and teardown), run setup (output dir, HF caches, seed, resume detection, the `run.log` tee), the policy load and window pins, script dataset loading; the generation CLIs' runs (resumable S3 generation, reward-model scoring, rejection sampling); the environment tools' runs (the code-contest and generic evals, the re-grade, the code-pool preparation) and the evals' training contract and output writer; root/CLI logging and the `halo launch` / `halo run` surface | `halo/entrypoints/`, `halo/cli.py`, `halo/log.py` | [Scripts](scripts-reference.md) |
| Served endpoints | The OpenAI-compatible client every rollout, judge and batch-generation script talks through: endpoint defaults, key resolution and the loopback test (`endpoints.py`), the async client, one request with the upstream retry and the single-request helper that normalizes a reply (`openai_client.py`), the parallel resumable batch with its JSONL resume store (`batch_requests.py`), and the response record with its finish-reason contract (`response.py`) | `halo/inference/` | [Rollout Servers](../infrastructure/rollout-servers.md) |

Leaf modules keep those imports one-way, each holding a contract several layers share:

| Leaf | Holds |
|---|---|
| `halo/data/turn_markers.py` | what delimits an assistant turn: the response marker, the terminator set, and the construction-time marker refusals |
| `halo/data/completion_spans.py` | the named span policies, the span search, the one span-to-labels rule (`build_span_labels`) and the completion-only label builders over it |
| `halo/models/unwrapping.py` | wrapper peeling (framework, toolkit, PEFT) and the plain hub spelling of a name read off a wrapped tree |
| `halo/models/module_tree.py` | module-tree lookups: the decoder-layer list and the layer index in a dotted name, the backbone owning the list, the module consuming the input embedding, the embedding modules, module paths by identity, the attribute probe |
| `halo/models/param_roles.py` | parameter roles: embedding/head markers, normalization modules, persistent buffers, fp32 pins |
| `halo/models/peft_adapters.py` | PEFT adapter parameters and the out-of-place LoRA fold, the value PEFT's merge would write |
| `halo/models/gradient_checkpointing.py` | the recompute scope both passes of one checkpoint invocation share, expert-load counting off it, the scoped gradient-checkpointing enable, and the input a reentrant checkpoint needs for a layer's trainable parameters to take a gradient (patched onto `GradientCheckpointingLayer.__call__` at import) |
| `halo/models/modality.py` | the multimodal verdict off a config in hand or a name; no hub read, no collective (the collective probe is `probe_checkpoint` in `halo/model_loading/parallel/model_source.py`) |
| `halo/models/conversion_overrides.py` | family corrections to transformers' conversion mapping, registered at import behind one guard |
| `halo/models/row_documents.py` | the documents of a training row as spans: where each sits and which of its tokens padding leaves real |
| `halo/models/upstream_guard.py` | the guard on a toolkit forward written against one transformers release, which raises at load on a release that moved it |
| `halo/models/segment_markers.py` | which families' conv / linear-attention mixers read per-document segment markers, the linear-attention kernel refusal, and the markers built from a row's `position_ids` |
| `halo/models/attention/layout.py` | per-layer attention cost rules off `layer_types` and head geometry — the MFU attention term; `attention_call`, the per-module kernel-call geometry (heads, KV heads, width, window, softcap) both load-time warm-ups and the FlexAttention call check read |
| `halo/models/attention/tree_plan.py` | the tree-attention plan of a tree-packed row: node recovery from the per-token columns, the index tensors each sequence mixer reads and their batch key |
| `halo/models/moe/family_specs.py` | every MoE family's facts, one frozen `MoEFamilySpec` each (model types, balancing slot, export renames and config rewrites, expert-weight roots, capability flags), indexed by `model_type`, so no reader imports the EP package for a family fact |
| `halo/models/mtp/spec.py` | each family's released MTP head, one `MtpHeadSpec`, or the reason it is refused, and the objective a run asks of it |
| `halo/models/external_tables.py` | the external-table contract the loaders and the gathered writers share: a frozen table held outside the module state as views of its checkpoint parts, its part keys and the files it was last read from or written to |
| `halo/models/head_transform.py` | the head-path contract: each family's declared transform around `lm_head` (scale, softcap, vocabulary cut), verified against its own forward on a meta-device shell |
| `halo/checkpoint/atomic.py` | exclusive staging with ordinary umask permissions (safetensors files included), atomic Torch-file publication, directory fsync — no rank coordination |
| `halo/checkpoint/format.py` | every on-disk checkpoint name two modules share, and the builders that spell them — no model |
| `halo/checkpoint/reading.py` | raw checkpoint reads: the layout cascade, the JSON configs, the streaming reader, the partial-checkpoint refusals, the tools' shard walks and header sizing — no model |
| `halo/model_loading/checkpoint_keys.py` | where each checkpoint key lands on the live names, planned once on transformers' rename rules from key names alone (no shard read, no collective), and the conversion mapping it plans with, so a lazy build and a resume read the same keys |
| `halo/checkpoint/hub_layout.py` | the transformers load conversions a save reverts to write the hub's tensor names, and `ModuleLayoutView`, which reads a saved layout under the live names |
| `halo/checkpoint/writing.py` | save-dtype casts, the stale-file sweep, the merged index, the streaming shard writer, the tied-embedding reconcile |
| `halo/data/sources/paths.py` | S3 / Hub / local classification of a dataset source or destination, string rules plus the ambiguous-spelling refusal's one directory probe, with no boto3 import |
| `halo/data/sources/dataset_cache.py` | the local cache-publish protocol (lock, completion marker, content fingerprint, atomic publish, stale-temp sweep) — `os`/`shutil`/`filelock`, the fetch injected; one protocol for both dataset caches, so their crash and staleness semantics cannot drift |
| `halo/data/sources/s3_client.py` | S3 transport: the boto3 `S3Client`, the bounded parallel transfer pool and the default-bucket helpers |
| `halo/data/sources/s3_dataset_store.py` | dataset trees on S3: the push protocol, the cached load and the mirrored control-file reads |
| `halo/checkpoint/parallel/writing.py` | the collective half of a gathered write: retain-gated DTensor gather of params AND buffers (with the neutralized GptOss sinks), the conversion-closed chunks and the streamed writer they feed — a leg every gathered writer runs symmetrically or hangs |
| `halo/checkpoint/parallel/reading.py` | the two reads a resume shares: one file every rank agrees on (`consensus_read`), and a checkpoint's shards opened on every rank (`joined_streaming_reader`) |
| `halo/checkpoint/parallel/resume_source.py` | which checkpoint a run resumes from (the newest complete on every rank, incomplete ones moved out of rotation's sight) and which weights its policy is built from |
| `halo/data/preprocessed/shard_index.py` | the torch-free `shard_index.json` contract and the stamped-sidecar writer both halves of a preprocessed artifact use |
| `halo/data/conversation.py` | conversation shaping the text and VLM renders share: JSON columns, the image-part spelling, system-prompt folding and the template kwargs |
| `halo/data/chat_render.py` | the text chat render and the tokenization of rendered text, with probe-based special-token ownership; the text twin of `vlm_render.py` |
| `halo/data/vlm_render.py` | the VLM chat render, the processor call, the over-length refusal, image decoding and the placeholder token ids; one render for the batch and the offline bake, so both tokenize a row identically |
| `halo/data/vlm_detection.py` | whether a run takes the VLM data path: the image-column spellings, a dataset's image evidence and the cross-rank verdicts and refusals off them |
| `halo/data/preprocessed/metadata.py` | the `metadata.json` contract: the recorded `PreprocessingConfig`, the stamp and the compatibility verdicts, readable without importing the bake that wrote the rows |
| `halo/environments/rollout_config.py` | `RolloutConfig`, the rollout defaults `AsyncRolloutConfig` reads, the reasoning-effort levels, and the stop-token and reasoning-end id resolvers; pickled to the Ray rollout actors, which keeps the Ray import out of `halo.configs` |
| `halo/config_validation.py` | the bounds guards, the validation base, the declared-default comparison and the argument refusals — standard library only |
| `halo/subclass_registry.py` | the subclass-tree walk, the class-claim maps (a key claimed twice raises), the swap of claimed modules and the package import that registers every family — torch only, so a family registers by subclassing, with no list to edit |
| `halo/rewards/terms.py` | the typed reward terms and their config parse — the endpoint defaults and the `halo/config_validation.py` guards are its only first-party imports, so `rewards:` parses at config time without the scorers' SDK client |
| `halo/inference/endpoints.py` | the OpenAI-compatible endpoint defaults, key resolution and the loopback test (`is_loopback`) — `halo/env.py` only, so no reader imports the SDK |

`halo/runtime/` therefore holds only what needs a process group or the launcher: rank and node math
(`ranks.py`), `init_distributed` and the subgroup timeout pin (`process_group.py`), barriers,
broadcasts, fixed-order group creation and the rank-0 object gather (`collectives.py`; a ring reduce, since a
per-peer send connects every peer to rank 0 and NCCL keeps those buffers, about 0.3 GiB per NVLink peer within
one NVLink domain, for the run, 2.94 GiB on rank 0 of 8 GPUs against 0.75 GiB over the ring), the cross-rank rejection/consensus seams
and the c10d-store joins (`consensus.py`), the shared-filesystem scope, FS-aware writer ranks,
main-first ordering and the output-FS probe (`filesystem.py`), the per-node load throttle
(`load_coordination.py`) and DTensor access (`dtensor.py`); anything a single rank can compute alone
lives in the leaves above. It imports only `halo.env`, `halo.log` and `halo.hardware`, and the
parallelism strategies in `halo/distributed/` import it, never the reverse.

## Layer table

A unit imports only units of strictly lower rows; `tests/cpu/conventions/test_import_layering.py`
enforces it and asserts this block equals its `_LAYERS`.

```text
L0  env
L1  log
L2  hardware, config_validation, subclass_registry, cli
L3  runtime, kernels, inference, checkpoint.model_free
L4  models, diagnostics, rewards
L5  model_loading, optimizers, weight_sync
L6  checkpoint.model_aware, data
L7  distributed, environments
L8  model_loading.parallel, checkpoint.parallel, callbacks, configs
L9  trainers
L10 entrypoints
```

`checkpoint.model_free` is the `halo/checkpoint/` modules that read no model (`format`, `atomic`,
`reading`, `model_card`); the rest of `halo/checkpoint/` outside `parallel/`, `tools/` included, is
`checkpoint.model_aware`.

## Trainers

Every distributed trainer uses multiple inheritance: a base trainer (`trl.SFTTrainer`,
`transformers.Trainer`, `trl.GRPOTrainer`, …) plus `DistributedTrainerMixin`
(`halo/trainers/distributed_trainer.py`), which composes the concern mixins listed in
[Trainer Architecture](trainer-architecture.md). The mixin overrides the parallelism-sensitive
methods (accelerator creation, dataloader sharding, gradient clipping, model saving) and delegates
the rest.

Thirteen trainers share this shape; which parallelism axes each supports, and why, is in
[Trainer Architecture](trainer-architecture.md#trainer-compatibility).

Offline GRPO's frozen-reference scores span three modules: `halo/trainers/grpo/offline/reference_cache.py`
(bounded, filesystem-aware score storage and the memory-mapped token buffers),
`halo/trainers/grpo/offline/reference.py` (the run-start sweep, the ragged payload, the training-checkpoint
contract and evaluation's original-reference scores), and the shared `halo/trainers/reference_logps.py` (checkpoint identity and
persistence, which DPO/KTO precompute also use). Storage, resume and recovery:
[Offline GRPO → Reference model](../training-methods/grpo/offline.md#reference-model).

## Distributed layer

`ParallelismConfig` (`halo/distributed/parallelism_config.py`) is the single source of truth: it
validates the requested combination against the allowlist and refusals in
`halo/distributed/parallelism_rules.py`, computes `data_parallel_size`, and builds the `EPConfig` /
`CPConfig` process groups, whose membership comes from the layouts in `halo/distributed/group_layout.py`; the DeviceMeshes come from
`halo/distributed/mesh.py`. The trainer reads mode flags (`has_ep_group`, `is_cp_mode`, …) off it and
never re-derives them.

What each mode shards and its backend: [Parallelism → Parallelism modes](../parallelism/README.md#parallelism-modes);
its data-parallel size: [Supported combinations](../parallelism/README.md#supported-combinations).

Four subpackages own the mechanics: `expert_parallel/` (DeepEP dispatch/combine, per-family EP
layers, grouped-GEMM expert compute, gradient-sync hooks, expert gather), `context_parallel/`
(`UlyssesModel`, sequence splitting), `tensor_parallel/` (DTensor weight sharding), and
`pipeline_parallel/` (layer split, stage module, stage-aware loading, P2P groups, the
`torch.distributed.pipelining` seam). Inside `expert_parallel/`, `moe_layer/` composes `EPMoELayerBase` from
one mixin per concern; its `compute`, `routing`, `grad_sync` and `balancing_bias` modules are imported only
from inside `expert_parallel/`, and everything else reaches them through layer instances
(`named_ep_layers`).

`halo/model_loading/parallel/` sits above all four: `load_distributed_model` picks the per-mode loader
off a `ParallelismConfig`, so it is the one place that reaches into every implementation, and the rest
of `halo/model_loading/` stays sharding-agnostic. The collective checkpoint half, `halo/checkpoint/parallel/`
(the saver ladder in `save.py`, the EP saves in `ep.py`, the PP saves in `pp.py`, the adapter saves in
`adapters.py`, and the resume beside them), sits above them too: it gathers through each implementation.
These packages import each other acyclically
(`tests/cpu/conventions/test_import_layering.py`).

The lazy load sits whole in the dispatch package, since only its loaders use it: `lazy_plan.py` holds the
expert-key vocabulary, the lazy gate, the per-key weight planner and the prologue the EP and PP loaders share
(a dense PP load runs it too); `expert_fuser.py` stacks a per-expert checkpoint's local experts; beside them
sit the loaders, `ep.py` and `pp.py`, the weight load both run once their plans are final
(`lazy_load.py`), and `fp32_masters.py`, which selects the configured FP32 parameter masters off the EP
layer registry and streams their stored values back for eager construction
([Load precision](../models/README.md#load-precision)).

The machinery both lazy loaders share — the conversion-op vocabulary and the key plan translated into it
(`conversion.py`), which transformers conversions a load plans with (`conversion_selection.py`), per-key
weight plans and the shared stream tail (`materialization.py`), meta-shell instantiation (`meta_shell.py`) —
lives in `halo/model_loading/lazy_safetensors/`, with no parallelism knowledge.

The FSDP2, HSDP, and TP axes ride a torch `DeviceMesh` (`halo/distributed/mesh.py`); EP and CP use
hand-built groups (`build_groups`) whose all-to-all patterns do not map to a mesh. The trainer reads
the mesh groups (DP, TP) through the mesh accessors (`dp_group`, `tp_group`) and the expert
groups (dispatch, expert-TP, expert-replica) off `EPConfig`. The bucketed
gradient all-reduce and the grad-presence agreement every post-backward sweep shares — deferred EP
cross-replica, TP replicated, QLoRA — live in a leaf (`halo/distributed/grad_reduce.py`) that
imports torch, `halo.env` and `halo.runtime`, and no parallelism implementation.

Which axis combinations may run is an allowlist, not a denylist — see
[Parallelism](../parallelism/README.md#supported-combinations).

## A training step

```text
YAML config
   │  HaloArgumentParser: toolkit defaults (use_liger_kernel, bf16, logging_nan_inf_filter, non_blocking)
   ▼
ParallelismConfig  ── validate mode, build process groups
   │
   ▼
load_distributed_model            halo/model_loading/parallel/loader.py
   │  • resolve_attn_implementation → FA4 (SM100+) / FA3 (SM90) / FA2; per-model SDPA/eager overrides
   │  • apply_ep_layers → EP layers (when ep_group_size > 1 — EP or ETP — or grouped GEMM on an MoE)
   │  • UlyssesModel (when cp_size > 1)
   │  • load_pp_stage_model: this rank's stage only (when pp_size > 1)
   │  • attach_mtp_head: the release's head and its forward hooks (when mtp_loss_weight > 0)
   ▼
DistributedTrainerMixin._configure_distributed_modes
   │  • FSDP2 fully_shard (EP modules in ignored_params, except at ep_group_size == 1)
   │  • router grad-sync hooks already attached at EP-layer construction (load);
   │    PEFT modules_to_save router copies re-hooked here
   ▼
trainer.train()  ── HF training loop, DDP wrapping skipped
   │  • forward → loss → backward
   │  • EP-aware global grad-norm clip (DTensor + replica groups)
   │  • AdamWBF16 step (bf16 weights, stochastic rounding — no fp32 master)
   │  • callbacks: EfficiencyCallback (tokens/s/GPU), MoEMetricsCallback,
   │               BalancingBiasCallback
   ▼
trainer.save_model  ── save_checkpoint (halo/checkpoint/parallel/) gathers shards → HF checkpoint
```

The backend choice and its per-model redirects are on [Flash Attention](../optimization/flash-attention.md);
the EP layers and their FSDP2 treatment on [Expert Parallelism](../parallelism/expert-parallelism.md); the
gathered save on [Checkpoints & Resume](checkpoints.md).

## RL: the rollout engine as a separate container

Online and async GRPO generate on a rollout engine (vLLM 0.32.0, or SGLang 0.5.22 under async GRPO's
`rollout_backend: sglang`) that runs in its own container and is never imported into the training
environment, since it pins its own torch/transformers stack. The trainer talks to it over HTTP for
generation and through an NCCL weight-sync client (`halo/weight_sync/`) for weights, which it gathers
from every parallel layout before each generation round that follows an update. Online GRPO hands TRL
`TrlVLLMClient` (`halo/trainers/grpo/rollout/`) in place of TRL's `VLLMClient`, which would import the vLLM
package. Images, patches, the
families each engine serves and the sync itself: [Rollout Servers](../infrastructure/rollout-servers.md);
the trainers: [Online GRPO](../training-methods/grpo/online.md), [Async GRPO](../training-methods/async-grpo/README.md).

## Related pages

- [Trainer Architecture](trainer-architecture.md) — the mixin lifecycle, gradient sync, per-trainer notes
- [Why This Framework](why-this-framework.md) — comparison to TRL, Unsloth, Megatron-LM, veRL, MS-SWIFT
- [Parallelism](../parallelism/README.md) — EP, CP, TP, ETP, PP and valid combinations
- [Configuration](configuration-reference.md#parsing) — the YAML system and toolkit defaults
- [Checkpoints & Resume](checkpoints.md) — checkpoint formats and the two resume paths
