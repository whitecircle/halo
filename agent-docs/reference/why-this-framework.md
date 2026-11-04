# Why this framework

Built on HuggingFace Transformers, Accelerate, and TRL — every distributed trainer subclasses a TRL, Transformers, or SentenceTransformers trainer. What it adds over that stack:

- **EP/CP/TP/ETP/PP on native HF models, no Megatron conversion** (DeepEP for MoE expert parallelism).
- **EP+TP and pure ETP** shapes that other HF-native tools don't compose (Axolotl composes EP with CP, not TP).
- **Full BF16 masters + optimizer state** (`AdamWBF16` + stochastic rounding, ~6 bytes/param) — others keep fp32 masters.
- **SMPO** and **Offline GRPO** (pre-scored completions) — methods absent elsewhere.
- **Fully asynchronous multi-turn RL** — Ray rollout actors overlap training via a prefetch queue, a clean Transformers ↔ vLLM/SGLang split with NCCL weight sync, and built-in environments; no Megatron backend, no veRL.
- Native `s3://` datasets and FA4-on-Blackwell.

## Framework comparison

Versions compared: TRL 1.14.2 and Accelerate 1.11.0 (the pinned releases), Unsloth, Axolotl 0.19.0, MS-SWIFT 4.0, veRL (Megatron backend), Megatron-LM/Core. "Via Megatron" = needs a Megatron backend + an HF↔MCore checkpoint conversion; "—" = out of scope (a layer the others build on).

| Feature | This Toolkit | HF TRL | Accelerate | Unsloth | Axolotl | MS-SWIFT | veRL | Megatron-LM |
|---------|:------------:|:------:|:----------:|:-------:|:-------:|:--------:|:----:|:-----------:|
| SMPO (reference-free PO) | **Yes** | No | — | No | No | No | No | No |
| Offline GRPO (pre-scored) | **Yes** | No | — | No | No | No | No | No |
| EP for MoE (no conversion) | **DeepEP** | No | No | No | **DeepEP** (v0.17) | Via Megatron | Via Megatron | MCore format |
| Context Parallelism | **Ulysses (SFT, SMPO, offline GRPO full-FT)** | SFT-only; needs a newer Accelerate | Yes (FSDP2) | No | Yes (ring/seq) | Via Megatron | Via Megatron | MCore |
| Tensor Parallelism | Yes (DTensor) | Via Accelerate; needs a newer Accelerate | Yes (ND-parallel) | No | Yes (experimental) | Via Megatron | Via Megatron | MCore |
| Pipeline Parallelism | **Yes (registered families; EP inside each stage)** | No | Inference only (PiPPy) | No | No | Via Megatron | Via Megatron | MCore |
| Combined EP+CP / EP+TP / pure ETP | **Yes** | No | No (no EP) | No | Partial (EP×CP and EP×CP×FSDP; EP×TP raises) | Via Megatron | Via Megatron | Yes |
| Full BF16 (~6 bytes/param) | **AdamWBF16 + SR** | No (FP32 masters) | Plumbing only | No | bf16 MP only | No | No | No |
| Multi-turn RL | **Async Ray + built-in envs** | Async GRPO + env adapters (OpenEnv, Harbor, OpenReward) | — | Via ART | Async GRPO + NeMo Gym | GYM env | AgentLoop + SGLang | No |
| Native HF models (no conversion) | **Yes (all modes)** | Yes (no EP) | Yes (plumbing) | Yes (no EP/CP/TP) | Yes | MCore for EP/CP/TP | MCore for EP/CP/TP | No |
| Flash Attention (FA4 Blackwell / FA3 Hopper / FA2 fallback) | **Yes** | Inherited (FA2/3) | Inherited | No (FA2) | Yes (FA2/3/4) | Yes | Inherited (FA2) | Via TE |
| Native `s3://` datasets | **Yes** | No | No | No | Yes | No | No | Via MSC |
| Multi-node | **Yes** ([verified topologies](../parallelism/multi-node.md#verified-topologies)) | Via Accelerate | Yes | Paid tiers | Yes | Yes | Yes | Yes |

Megatron-Core stores `torch_dist` sharded checkpoints split by TP/PP/EP/ETP — a separate format from HuggingFace's, so MS-SWIFT, veRL, and Megatron-LM all require an HF↔MCore conversion before EP/CP/TP/PP. This toolkit, Accelerate, TRL, Unsloth, and Axolotl work directly on HuggingFace weights; of those, this toolkit and Axolotl run DeepEP expert parallelism on the upstream HF MoE modules. Two neighbors sit close: NeMo AutoModel keeps HF checkpoints without a conversion step but ships its own implementation of each supported architecture behind the `transformers` API, and transformers itself shards a model at load (`from_pretrained(distributed_config=DistributedConfig(...))`) with FSDP2, TP and EP: `ep_size` dispatches tokens through a torch all-to-all (`ep_dispatch_experts`, most MoE families' default plan), and a few families keep a router that all-reduces the full MoE output across `ep_size == tp_size` ranks. On the pinned transformers 5.20.2 and Accelerate 1.11.0, the transformers `Trainer` trains such a model only at `tp_size=1` (FSDP2 plus token dispatch) and without resume; it, and TRL's trainers that subclass it, run neither TP, CP nor the all-reduce EP path: a load-time TP model needs a newer Accelerate, and CP needs Accelerate's `ParallelismConfig` (`TrainingArguments.parallelism_config`), on which that `Trainer` fails at construction under this Accelerate release.

### When to use which

| Use case | Recommended | Why |
|----------|:-----------:|-----|
| EP+TP or pure ETP on MoE | **This Toolkit** | Expert shapes other HF-native tools don't compose |
| Memory-constrained full fine-tuning | **This Toolkit** | AdamWBF16 — ~6 bytes/param |
| Long-context training (scales with nGPUs) | **This Toolkit** | CP splits sequences; combines with EP |
| MoE wider than one NVLink domain, on HF weights | **This Toolkit** | PP across domains, EP inside each stage |
| SMPO or Offline GRPO | **This Toolkit** | Unique methods |
| Multi-turn RL with tool environments | **This Toolkit** | Ray actors, NCCL sync, built-in envs |
| Config-driven SFT/DPO/GRPO over many models (incl. FSDP+EP) | Axolotl | YAML, DeepEP EP (v0.17), s3/GCS data, FA2/3/4 |
| A hand-written loop needing only FSDP2 + TP + CP | Accelerate | The plumbing layer, no trainer overhead |
| Single-GPU LoRA/QLoRA on consumer hardware | Unsloth | Extreme VRAM efficiency, free tier |
| SFT/DPO on single node, standard models | HF TRL | Simplest setup, largest community |
| Online PPO/GRPO with many algorithm variants | veRL or MS-SWIFT | Purpose-built for online RL |
| Broad model coverage + no-code UI | MS-SWIFT | 600+ models, Gradio Web UI |
| Dense pre-training at 1000+ GPU scale | Megatron-LM | TE/FP8 kernels, peak utilization |

Measured throughput against NeMo AutoModel, Megatron-LM, Axolotl, MS-SWIFT, Unsloth and transformers' own
expert-parallel path: [Throughput Benchmarks](../optimization/throughput-benchmarks.md#published-numbers-in-similar-setups).

Notes on individual frameworks:

- **Accelerate** is the foundation layer this toolkit builds on (launcher + FSDP2/DTensor TP/CP plumbing over native HF), not a competing trainer — it ships no trainers, RL, MoE/EP, attention kernels, or `s3://` loading.
- **Megatron-LM** adds deeper TE/FP8 optimization for 1000+ GPU dense pre-training; its alignment is split across four repos and needs MCore conversion. On large NVLink domains FSDP2 + EP/TP avoids the pipeline bubble; this toolkit keeps [pipeline parallelism](../parallelism/pipeline-parallelism.md) for crossing domains — see [When PP is worth it](../parallelism/README.md#when-pipeline-parallelism-is-worth-it).
- **Axolotl** (v0.19.0) closes most of the EP gap (DeepEP on native HF, no conversion, composed with CP and FSDP) and offers config-driven YAML, s3/GCS data, FA2/3/4, and async-GRPO + NeMo Gym. This toolkit still adds SMPO, Offline GRPO, EP+TP and pure ETP (Axolotl's EP×TP raises), and full BF16.
- **NVIDIA NeMo RL** — alignment on DTensor (FSDP2+TP+CP) or Megatron-Core; MCore conversion on the Megatron path; capabilities span four repos.
- **SkyRL** — RL plus a native SFT trainer (v0.3.0), no DPO/reward/distillation; async dispatcher ~1.55× over naive batching; EP/CP/TP need its Megatron backend.

## Model compatibility

Any HuggingFace `AutoModelForCausalLM` works with standard FSDP, and any model carrying a `tp_plan`
gets TP. Advanced parallelism (EP, CP, ETP, PP) needs per-family registration, not a model
reimplementation or a checkpoint conversion.

For EP, that is a frozen `MoEFamilySpec` holding the family's facts plus an EP layer under
`halo/distributed/expert_parallel/families/` (88 lines or fewer; GPT-OSS's is 171) subclassing `EPMoELayerBase`, declaring
its `HF_MODULE_NAMES` and pointing `FAMILY` at the spec, with `MOE_LAYER_MAP` built from the roster's
imports; for PP, a validated `PPFamilySpec` in `PP_FAMILY_SPECS`. Seventeen MoE families ship an EP layer;
the per-family × per-mode matrix is
[Supported Models](../models/README.md), and [Adding a New Model](../models/adding-a-model.md) is the
procedure. Trainer × parallelism support is separate — see
[Trainer Compatibility](trainer-architecture.md#trainer-compatibility).

## Key dependencies

Pinned versions: [Docker](../infrastructure/docker.md) (images) and `pyproject.toml` / `uv.lock` (Python).

> [!WARNING]
> **Rollout servers run as separate Docker containers**
>
> vLLM and SGLang each pin their own torch/transformers stack — neither is in the training environment. Each runs as an isolated container (`Dockerfile.vllm` + `docker-compose.vllm.yml`, `Dockerfile.sglang` + `docker-compose.sglang.yml`). The training container talks to the server over HTTP (generation) and an NCCL weight-sync client (`halo/weight_sync/`).
