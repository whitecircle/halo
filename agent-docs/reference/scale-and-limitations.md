# Capabilities & Limitations at Scale

What the toolkit can and cannot do for very large models, multi-week runs, huge corpora, and many
nodes. Multi-node mechanics: [Multi-Node Training](../parallelism/multi-node.md). Debugging:
[Debugging & Profiling](debugging.md).

Which axis combinations may run at all is an **allowlist** (`SUPPORTED_AXIS_SETS` in
`src/distributed/parallelism_config.py`): plain data parallelism; EP, ETP, TP, CP or PP alone; and
EP+TP, EP+CP, EP+ETP, PP+EP, PP+ETP — the PP sets not yet runnable this release
([Pipeline Parallelism](../parallelism/pipeline-parallelism.md)). Anything else is rejected at
config time, before any rank math.

## What works at scale

| Capability | Notes |
|------------|-------|
| Fine-tuning (SFT/DPO/SMPO/GRPO/reward/…) | The primary, most-exercised path |
| **From-scratch pre-training (dense)** | `init_from_scratch` on dense data-parallel / FSDP2 — see [Pre-training](../training-methods/pretraining.md) |
| Continued pre-training (any mode) | Load a checkpoint, train on raw text prepared with `scripts/before_training/prepare_dataset.py --mode text` |
| Expert Parallelism (MoE), incl. cross-node | DeepEP all-to-all; multi-group EP across nodes syncs cross-replica gradients in a deferred post-backward sweep |
| Tensor / Context / Expert-Tensor Parallelism | TP, CP and ETP are NVLink-domain-local; see parallelism guides |
| Pipeline Parallelism | **Not yet available in this release** — the config surface, rank math, trainer gates and stage/loss seams ship; the schedule engine does not, and `pipeline_parallel_size > 1` is rejected at config time ([Pipeline Parallelism](../parallelism/pipeline-parallelism.md)) |
| Multi-node (torchrun / SLURM / SkyPilot / RunPod / Nomad) | FS-aware checkpoints, NCCL timeout knob |
| Pre-tokenized **sharded** datasets (Megatron-style) | Per-rank shards from S3 or a local path; the recommended large-corpus path |
| Raw-text data prep + document packing | `--mode text` (per-doc EOS, bfd/wrapped packing) |
| Step-based training (`max_steps`) + WSD/long-cosine LR | Standard HF Trainer knobs; see [Pre-training](../training-methods/pretraining.md) |
| Sharded HF checkpoints (`max_shard_size`) | All gathered saves (FSDP2/CP/EP/TP) auto-shard; large models don't write one monolithic file |
| Exact optimizer-state resume (all sharded modes) | Per-rank optimizer shards (`optimizer_shard_XXXXX.pt`) for every FSDP2-wrapped run (DP, EP, EP+CP, EP+TP, EP+ETP, CP) plus pure TP, gated by a topology fingerprint; same world size required. See [Checkpoints & Resume](checkpoints.md) |
| Gradient clipping across DP/EP/TP shards | Global grad-norm reduced over all shard groups (correct, rank-consistent) |
| Profiling / flame graphs / memory snapshots | See [Debugging & Profiling](debugging.md) |

## Known limitations & roadmap

**RL weight-sync buffers a full-model host snapshot.** For online/env GRPO the vendored NCCL client
copies each un-sharded policy weight into a persistent pinned-host buffer pool (~1× model in host
RAM), then broadcasts it to the vLLM workers pack-by-pack (~1 GB packs, double-buffered), so only
~2 GB transits the forwarding rank's GPU at a time. The ceiling is host RAM and the single-producer
fan-out below, not rank-0 GPU memory.

**Checkpoint save is synchronous, and the gathered save funnels through one rank.** Training pauses
during each gathered save — there is no async or overlapped save. Every gathered save streams (EP
layer by layer, FSDP2/CP/TP chunk by chunk through the shared writer), so all of them are bounded
by write throughput on one rank rather than by host RAM.

The parallel-write escape hatch covers the flagship pure-EP layouts only. `save_sharded_ep` requires
a single EP group spanning all ranks (`ep_group_size == world_size` — exactly what
`ep16`/`ep32`/`ep64 --ep_scope=global` run) and a shared output filesystem on multi-node;
`validate_ep_sharded_save` refuses replicated-EP layouts (multiple EP groups), ETP, CP and expert
LoRA; PP is refused by the pipeline mixin. Dense, FSDP2, CP and TP have no parallel-write mode at all. `merge_ep_shards.py` reassembles
the artifact, streaming group by group — peak host RAM is one merged MoE layer plus one pending
output shard, not the model — and `--delete_input_shards` frees the per-rank shards once the merged
checkpoint is complete.

Optimizer state escapes the funnel (per-rank shards, written in parallel), but its write copies each
rank's state to host RAM first (`cpu_offload`): at 120B/EP8 with AdamWBF16 that is ~60 GB/rank,
~480 GB per 8-GPU node transient at every save and again at resume. Preflight free host RAM
accordingly.

Where `save_sharded_ep` does not qualify, tune `save_steps` — at 400B a gathered save is one rank
writing ~800 GB per checkpoint, so budget it against your write bandwidth. `save_only_model: true`
drops the optimizer shards but not the funnel. Parallel model writes for a replicated layout are a
roadmap item.

| Limitation | Impact | Workaround / status |
|------------|--------|--------------------|
| **Pipeline parallelism is not yet available** | The config surface, stage-scoped rank math, trainer gates (`src/trainers/mixins/pp_gates.py`) and stage/loss/split seams (`src/distributed/pipeline_parallel/`) ship; the schedule engine (`PipelineRuntime`) does not. `pipeline_parallel_size > 1` is rejected at config time. | Shard with EP/TP/CP and their supported combinations; the engine lands in a future release. See [Pipeline Parallelism](../parallelism/pipeline-parallelism.md). |
| **From-scratch with EP/TP/CP/ETP/PP** | Random-init pre-training is dense-only; `init_from_scratch` raises `NotImplementedError` under EP, TP, CP, ETP or PP (no distributed random-init of sharded experts; PP has nothing to be stage-aware about without a checkpoint). Only `scripts/training/sft.py` threads the flag through. | Pre-train dense from scratch; or **continued** pre-training (load a small/seed checkpoint) works in every mode. MoE-from-scratch is a roadmap item. |
| **Streaming (on-the-fly) datasets** | The data pipeline materializes datasets (full or per-rank shards) rather than streaming an infinite corpus; exact data-position resume across a multi-week run is not yet supported. | Use **pre-tokenized sharded datasets** (`scripts/before_training/prepare_dataset.py` + `--num-shards`) — each rank loads only its shards, which scales to very large corpora. Set `--num-shards` to a multiple of the data-parallel degree (`>= data_parallel_size`; `k×world_size` is safe); fewer shards than DP ranks hard-errors for the train split. Checkpoint model frequently (`save_steps`) for restart. |
| **Exact resume requires same world size + layout** | All sharded modes (FSDP2 DP, EP, EP+CP, EP+TP, EP+ETP, CP, pure TP) save per-rank optimizer shards; a topology fingerprint gates the restore, so a different GPU count or parallelism shape cannot remap them (no resharding / elastic). | Resume on the same world size and layout for exact optimizer state. A fingerprint mismatch (or `save_only_model: true`) warm-restarts instead — weights, LR schedule, and trainer step still resume, only the Adam moments reinit. Weights themselves are topology-free — every checkpoint is a standard HF model directory, so export-and-reload across topologies works. Elastic/DCP-resharding resume is a roadmap item. |
| **Non-shared-FS checkpoints duplicate per node** | On `DIST_SHARED_FILESYSTEM=0`, each node's local rank 0 writes a full checkpoint (the price of per-node local disk). | Use a shared FS (NFS/Lustre) to write once; otherwise this is expected. |
| **vLLM weight sync fans out from rank 0** | Online/Env GRPO broadcasts updated weights from the trainer to inference workers from a single producer — fine for a handful of vLLM nodes, not a large inference fleet. | Keep the inference fleet modest, or shard weight sync (roadmap). |
| **QLoRA (bitsandbytes 4-/8-bit) + EP / TP / PP / grouped-GEMM MoE** | The EP and grouped-GEMM loaders materialize plain de-quantized weights (the EP lazy loader streams raw tensors), and TP shards into DTensors, so bitsandbytes `Params4bit` are lost and PEFT's 4-bit adapter dispatch fails; PP rejects PEFT outright. Rejected at model-load time (`load_distributed_model`), including an `ep_size=1` MoE run that only wants the grouped-GEMM wrappers. CP on a **dense** model is unaffected — it keeps the standard `from_pretrained` loader; a CP MoE run is rejected by the same guard (the CP loader applies the grouped-GEMM expert wrappers internally). | Use QLoRA with standard **DDP/FSDP** (`accelerate launch`) or **CP on a dense model**, or **plain LoRA** (no quantization) for EP. TP has no LoRA path at all — full fine-tune there. |
| **NVL72 / MNNVL rack-wide domains** | `NVLINK_DOMAIN_SIZE` makes the rack the locality unit for EP/CP/TP/ETP grouping. The fabric check (`validate_nvlink_domain_against_fabric`, run from `ParallelismConfig.__post_init__`) **raises** when a declared domain block straddles two fabric cliques or when ranks disagree on the value, warns when the clique is wider, and no-ops where any rank cannot read its clique — it never builds groups. The path is exercised only against simulated domain sizes in the test suite — no rack-scale hardware run backs it. | Treat rack-wide grouping as unvalidated: bring up a small NVLink-wide job first, and keep `NVLINK_DOMAIN_SIZE` unset on ≤8-GPU nodes. |
| **EP requires a de-quantized (BF16) checkpoint** | The EP path materializes experts as plain `nn.Parameter`s, so a natively-quantized MoE checkpoint (e.g. the MXFP4 `openai/gpt-oss-20b`, whose expert blocks are `uint8` on disk) fails to load — the EP patcher checks every expert tensor and raises with the workaround. | Point EP configs at a BF16-dequantized checkpoint (`unsloth/gpt-oss-20b-BF16` or a locally patched BF16 export). |
| **EP+CP loss equivalence on short sequences (gpt-oss)** | The EP-vs-EP+CP forward-loss correctness check (`tests/gpu/parallelism/combined/test_ep_cp_correctness.py`) can exceed its 10% relative tolerance on an occasional gpt-oss input at very short sequence length (128 tokens, cp=2 → 64/rank), where router top-k ties are more likely to flip between the two modes. Full-length training is unaffected. | Expected numerical edge at tiny seq len; use realistic sequence lengths for gpt-oss + CP. |

## Choosing a large-scale layout

- **Dense model, fits with FSDP2**: plain `torchrun` DP (+ TP node-local for very
  large attention). From-scratch via `init_from_scratch`.
- **MoE model**: EP (node-local or cross-node / NVL72-wide), optionally + TP for
  attention, or + ETP to also shard each expert's FFN (EP+ETP, node-local), or
  pure ETP (`ep_size=1`) when experts stay replicated. Continued pre-training or
  fine-tuning.
- **Huge corpus**: pre-tokenize with `scripts/before_training/prepare_dataset.py --num-shards N`
  (`N >= data_parallel_size`; `k×world_size` is safe). Loads per-rank with
  bounded RAM.
- **NVL72 rack**: set `NVLINK_DOMAIN_SIZE` to run TP/CP/EP NVLink-wide — unvalidated on
  rack hardware, see the
  [NVL72 section](../parallelism/multi-node.md#gb200gb300-nvl72-multi-node-nvlink).

See also: [Why This Framework](why-this-framework.md) · [Pre-training](../training-methods/pretraining.md) · [Checkpoints & Resume](checkpoints.md)
