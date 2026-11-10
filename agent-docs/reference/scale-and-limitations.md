# Capabilities & Limitations at Scale

What the toolkit can and cannot do for very large models, long sequences, multi-week runs, huge
corpora, and many nodes. Multi-node mechanics: [Multi-Node Training](../parallelism/multi-node.md).
Debugging: [Debugging & Profiling](debugging.md). What has run on real machines:
[Verified topologies](../parallelism/multi-node.md#verified-topologies) and
[Release validation](../infrastructure/ci.md#release-validation).

Which axis combinations may run at all is an **allowlist**
([Supported combinations](../parallelism/README.md#supported-combinations)); anything else is
rejected at config time, before any rank math.

## What works at scale

| Capability | Notes |
|------------|-------|
| Fine-tuning (SFT/DPO/SMPO/GRPO/reward/…) | The primary, most-exercised path |
| **From-scratch pre-training (dense)** | `init_from_scratch` on dense data-parallel / FSDP2 — see [Pre-training](../training-methods/pretraining.md) |
| Continued pre-training (any mode) | Load a checkpoint, train on raw text prepared with `scripts/before_training/prepare_dataset.py --mode text` |
| Expert Parallelism (MoE), incl. cross-node | DeepEP all-to-all; multi-group EP across nodes syncs cross-replica gradients in a deferred post-backward sweep |
| Tensor / Context / Expert-Tensor Parallelism | TP, CP and ETP are NVLink-domain-local; see parallelism guides |
| Pipeline Parallelism | The one axis meant to cross NVLink domains; composes with EP or pure ETP ([Pipeline Parallelism](../parallelism/pipeline-parallelism.md)) |
| Multi-node (torchrun / SLURM / SkyPilot / RunPod / Nomad) | FS-aware checkpoints, NCCL timeout knob |
| Pre-tokenized **sharded** datasets (Megatron-style) | Per-rank shards from S3 or a local path; the recommended large-corpus path |
| Raw-text data prep + document packing | `--mode text` (per-doc EOS, bfd/wrapped packing) |
| Step-based training (`max_steps`) + WSD/long-cosine LR | Standard HF Trainer knobs; see [Pre-training](../training-methods/pretraining.md) |
| Sharded HF checkpoints (`save_max_shard_size`) | All gathered saves (FSDP2/CP/EP/TP) auto-shard; large models don't write one monolithic file |
| Exact optimizer-state resume (all sharded modes) | Per-rank optimizer shards (`optimizer_shard_XXXXX.pt`) for every FSDP2-wrapped run (DP, EP, EP+CP, EP+TP, EP+ETP, CP, PP) plus pure TP, gated by a topology fingerprint; same world size required. See [Checkpoints & Resume](checkpoints.md) |
| Gradient clipping across DP/EP/TP shards | Global grad-norm reduced over all shard groups (correct, rank-consistent) |
| Profiling / flame graphs / memory snapshots | See [Debugging & Profiling](debugging.md) |

## Known limitations

**RL weight sync stages one chunk on the forwarding rank's GPU.** For Online GRPO and Async GRPO with Environments the NCCL
weight-sync client (`halo/weight_sync/`) stages each un-sharded policy weight on the sync GPU and streams it to the engine in
byte-bounded chunks (`HALO_WEIGHT_SYNC_CHUNK_MB`, 1 GiB), never the model and never host RAM.

The forwarding rank's peak is the assembled EP layer being sent plus the chunk, the largest tensor's
snapshot and the engine path's buffers. Each is at least the chunk budget, grown past it only when a
tensor exceeds it (a fused expert pair on a 100B+ MoE)
([Rollout Servers](../infrastructure/rollout-servers.md#weight-sync)). The ceiling is the
fabric behind one GPU (its NICs over EFA) and the single-producer fan-out below.

**Checkpoint save is synchronous.** Training pauses during each gathered save; there is no async save.
Every gathered save streams (EP layer by layer, FSDP2/CP/TP chunk by chunk through the shared writer),
so all of them are bounded by write throughput rather than by host RAM. The EP save funnels through one
rank per node (`HALO_CHECKPOINT_WRITERS_PER_NODE`), so a multi-node shared-filesystem save scales with
the node count; the FSDP2, CP and TP saves keep one writer. `HALO_CHECKPOINT_WRITE_THREADS` (opt-in) overlaps the save
rank's part writes with the next chunk's gather inside that pause ([Checkpoints](checkpoints.md)).

The parallel-write escape hatch covers the flagship pure-EP layouts only. `save_sharded_ep` requires
a single EP group spanning all ranks (`ep_group_size == world_size`, exactly what
`ep16`/`ep32`/`ep64 --ep_scope=global` run) and a shared output filesystem on multi-node.
`reject_unmergeable_ep_sharded_save` refuses replicated-EP layouts (multiple EP groups), ETP, CP and expert
LoRA. Dense, FSDP2, CP and TP have no parallel-write mode at all.

`merge_ep_shards.py` reassembles the artifact, streaming group by group: peak host RAM is one merged
MoE layer plus the output shard being filled and one per write thread, not the model.
`--delete_input_shards` frees the per-rank shards once the merged checkpoint is complete.

Optimizer state escapes the funnel (per-rank shards), but each rank's state passes through host RAM:
~60 GB/rank at 120B/EP8 with AdamWBF16. A save holds its copy from the `cpu_offload` to the end of its
shard write, inside the node's `max_concurrent_loading` throttle: ~240 GB per 8-GPU node at the default
4, ~480 GB at `0`. A resume reads under the same throttle, but every rank keeps its copy until the
collective apply, so a node holds all of them at any setting: ~480 GB. Preflight free host RAM for the
resume.

Where `save_sharded_ep` does not qualify, tune `save_steps` — at 400B a gathered save is one rank
writing ~800 GB per checkpoint, so budget it against your write bandwidth. `save_only_model: true`
drops the optimizer shards but not the funnel.

| Limitation | Impact | Workaround / status |
|------------|--------|--------------------|
| **Dense load is whole-model per GPU** | Plain FSDP2 and DDP load the whole model onto every GPU before the wrap shards it, so a dense model must fit on one GPU at its load dtype; `fsdp_reshard_after_forward` does not change the load. | Run TP node-local: its HF-native `tp_plan` load shards the model as it loads ([Data Parallelism](../parallelism/data-parallelism.md#fsdp2-strategy-by-mode)). A [PP-registered family](../parallelism/pipeline-parallelism.md#model-support) also loads per stage. |
| **From-scratch with EP/TP/CP/ETP/PP** | Random-init pre-training is dense-only; `init_from_scratch` raises `NotImplementedError` under EP, TP, CP, ETP or PP (no distributed random-init of sharded experts, and no checkpoint for a stage to stream). Only `scripts/training/sft.py` threads the flag through. | Pre-train dense from scratch; or **continued** pre-training (load a small/seed checkpoint) works in every mode. |
| **Streaming (on-the-fly) datasets** | The data pipeline materializes datasets (full or per-rank shards) rather than streaming an infinite corpus; exact data-position resume across a multi-week run is not supported. | Use **pre-tokenized sharded datasets** (`scripts/before_training/prepare_dataset.py` + `--num_shards`) — each rank loads only its shards, which scales to very large corpora. Set `--num_shards` to a multiple of the data-parallel degree (`>= data_parallel_size`; `k×world_size` is safe); fewer shards than DP ranks hard-errors for the train split. Checkpoint model frequently (`save_steps`) for restart. |
| **Exact resume requires same world size + layout** | All sharded modes (FSDP2 DP, EP, EP+CP, EP+TP, EP+ETP, CP, PP, pure TP) save per-rank optimizer shards; a topology fingerprint gates the restore, so a different GPU count or parallelism shape cannot remap them (no resharding / elastic). | Resume on the same world size and layout for exact optimizer state. A fingerprint mismatch (or `save_only_model: true`) warm-restarts instead — weights, LR schedule, and trainer step still resume, only the Adam moments reinit. Under PP a mismatch or a different layer split raises instead, since stage-local names would land the moments on the wrong layers. Weights themselves are topology-free — every checkpoint is a standard HF model directory, so export-and-reload across topologies works. |
| **Non-shared-FS checkpoints duplicate per node** | On a non-shared output filesystem (`DIST_OUTPUT_SHARED_FILESYSTEM=0`, or the `DIST_SHARED_FILESYSTEM=0` umbrella), each node's local rank 0 writes a full checkpoint (the price of per-node local disk); under PP a node writes only the stages it runs, and `merge_pp_stages.py` assembles them ([Per-node export](../parallelism/pipeline-parallelism.md#per-node-export)). | Use a shared FS (NFS/Lustre) to write once; otherwise this is expected. |
| **TRL loss metrics sync per micro-batch** | TRL's DPO, KTO, GRPO and reward losses, and the GRPO term of SDPG, read their logged metrics to the host and gather them across ranks on every micro-batch (about 15 scalar all-gathers per DPO micro-batch): 1–2 ms on one node, latency-bound at 1k+ ranks. Halo's own losses (SFT, SMPO, offline GRPO, distillation) keep theirs on device until the log. | Upstream: the metrics belong on device, reduced once per log; no Halo-side patch. |
| **Weight sync fans out from one producer** | Online and async GRPO broadcast updated weights from the trainer's forwarding rank to every rollout server, and the push time grows with the server count (the fan-out costs the sum of the pushes, not the slowest) — fine for a handful of servers, not a large inference fleet. | Keep the inference fleet modest. |
| **QLoRA (bitsandbytes 4-/8-bit) + EP / TP / grouped-GEMM MoE** | The EP and grouped-GEMM loaders materialize plain de-quantized weights (the EP lazy loader streams raw tensors), and TP shards into DTensors, so bitsandbytes `Params4bit` are lost and PEFT's 4-bit adapter dispatch fails. Rejected at model-load time (`load_distributed_model`), including an `ep_size=1` MoE run that only wants the grouped-GEMM EP layers. CP on a **dense** model is unaffected — it keeps the standard `from_pretrained` loader; a CP MoE run is rejected by the same guard (the CP loader applies the grouped-GEMM EP layers internally). | Use QLoRA under plain **DDP/FSDP** (either launcher; a MoE also needs `use_grouped_gemm: false`) or **CP on a dense model**, or **plain LoRA** (no quantization) for EP. TP has no LoRA path at all — full fine-tune there. |
| **Pipeline parallelism trains full fine-tunes on registered families** | Every adapter shape is refused, a `PeftModel` included, along with online and async GRPO, distillation, embedding, `torch_compile`, reentrant checkpointing and interleaved schedules; the bubble rises with `pp_size` at a fixed global batch. Every registered MoE family runs tiny models on the pipeline GPU tier; of released checkpoints, the Qwen3-8B and gpt-oss-20b recipes, GLM-5.3-Flash's full fine-tune and a Qwen3-30B-A3B correctness smoke have run, as have Qwen3.5-35B-A3B and Qwen3.6-35B-A3B, and 13 steps each of Qwen3.8-Flash-Next and Nemotron 3.5 Super VL with GC on. | Train adapters without PP. Limits and the refusal list: [Pipeline Parallelism](../parallelism/pipeline-parallelism.md#rejected-combinations). |
| **NVL72 / MNNVL rack-wide domains** | `NVLINK_DOMAIN_SIZE` makes the rack the locality unit for EP/CP/TP/ETP grouping. The fabric check (`validate_nvlink_domain_against_fabric`, run from `ParallelismConfig.__post_init__`) **raises** when a declared domain block straddles two fabric cliques, warns when the clique is wider, and no-ops where any rank cannot read its clique — it never builds groups. Ranks that disagree on the value are refused earlier in `__post_init__` (`reject_divergent_settings`). The test suite drives the path with simulated domain sizes. | Bring up a small NVLink-wide job first, and keep `NVLINK_DOMAIN_SIZE` unset on ≤8-GPU nodes. |
| **EP requires a de-quantized (BF16) checkpoint, gpt-oss's MXFP4 under the lazy loaders aside** | The EP path materializes experts as plain `nn.Parameter`s. The lazy EP and PP loaders decode the MXFP4 `openai/gpt-oss-20b` (expert blocks `uint8` on disk) to bf16 while reading ([GPT-OSS](../models/gpt-oss.md#configs)); any other natively-quantized MoE checkpoint fails to load — the EP patcher checks every expert tensor and raises with the workaround. A whole-model load of that release (TP, CP, `ep_lazy_loading: false`) is left to transformers, which dequantizes it in the Halo images at the whole decoded model per rank. | Point those configs at a BF16-dequantized checkpoint (`unsloth/gpt-oss-20b-BF16` or a locally patched BF16 export). |
| **Tokens per rank through the DeepEP dispatch** | One rank's tokens per MoE forward are uncapped within an NVLink domain and capped at 8,192 when the EP group crosses domains. | For the cross-domain cap, batch 1, a shorter `max_length` or node-local EP with DP across nodes. The ceiling, where it is enforced, and the override in [Dispatch token ceiling](../infrastructure/deepep.md#dispatch-token-ceiling) and [EP over EFA](../infrastructure/deepep.md#expert-parallelism-over-aws-efa). |
| **Context-parallel trainer and model limits** | CP runs only under SFT, SMPO and offline GRPO full fine-tuning; every other trainer refuses it, as do families with sequence-mixing layers or no CP wrapper. | Requirements, including the head-count rule, in [Context Parallelism](../parallelism/context-parallelism.md#requirements). |
| **Full logits at long context** | GRPO without chunked log-probs, and SFT without fused linear cross-entropy (off by default on most families), materialize the full `[B, T, vocab]` logits, which EP does not shard. | `use_chunked_grpo_logprobs` ([The logits wall](../training-methods/async-grpo/performance.md#the-logits-wall)); `fused_linear_cross_entropy: true` for SFT where it is not forced off ([Liger configuration](../optimization/liger-kernels.md#configuration)). |
| **Async GRPO trajectories are never truncated** | A trajectory grows across turns up to the context window; a row over it raises on every rank and ends the run. | Budget the episode under the served `max_model_len`: [Context budget](../training-methods/async-grpo/performance.md#context-budget); the rule: [Trajectory length](../training-methods/async-grpo/episodes.md#trajectory-length). |
| **Vision-language runs** | An image-bearing run refuses CP and cannot pack. The toolkit's VLM collators do not truncate: a runtime batch over `max_length` raises, and offline preparation drops over-length rows. DPO's and KTO's TRL collators truncate. GRPO, classification and embedding have no vision path. PaliGemma trains on images through SFT and the distillation scripts only, never on a flash kernel, and every row of its image run carries an image. | [SFT — VLMs](../training-methods/sft.md#vision-language-models), [Pre-Processing](../data/dataset-preparation.md) and [Modality support](trainer-architecture.md#modality-support). |

## Choosing a large-scale layout

- **Dense model**: plain `torchrun` DP while it fits on one GPU, TP node-local past that (the load limit
  above). From-scratch via `init_from_scratch`.
- **MoE model**: EP (node-local or cross-node / NVL72-wide), optionally + TP for
  attention, or + ETP to also shard each expert's FFN (EP+ETP; each ETP group stays in one NVLink domain), or
  pure ETP (`ep_size=1`) when experts stay replicated. Continued pre-training or
  fine-tuning.
- **Beyond one NVLink domain**: PP across domains for the
  [PP-registered families](../parallelism/pipeline-parallelism.md#model-support), a MoE with EP (or
  experimental pure ETP) inside each stage; each stage is a whole number of domains.
- **Huge corpus**: pre-tokenize with `scripts/before_training/prepare_dataset.py --num_shards N`
  (`N >= data_parallel_size`; `k×world_size` is safe). Loads per-rank with
  bounded RAM.
- **NVL72 rack**: set `NVLINK_DOMAIN_SIZE` to run TP/CP/EP NVLink-wide; see the
  [NVL72 section](../parallelism/multi-node.md#gb200gb300-nvl72-multi-node-nvlink).

See also: [Why This Framework](why-this-framework.md) · [Pre-training](../training-methods/pretraining.md) · [Checkpoints & Resume](checkpoints.md)
