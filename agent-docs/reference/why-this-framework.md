# Why this framework

Built on HuggingFace Transformers, Accelerate, and TRL — every distributed trainer subclasses a TRL, Transformers, or SentenceTransformers trainer. What it adds over that stack:

- **EP/CP/TP/ETP on native HF models, no Megatron conversion** (DeepEP for MoE expert parallelism).
- **EP+TP and pure ETP** shapes that other HF-native tools don't compose (Axolotl composes EP with CP, not TP).
- **Full BF16 masters + optimizer state** (`AdamWBF16` + stochastic rounding, ~6 bytes/param) — others keep fp32 masters.
- **SMPO** and **Offline GRPO** (pre-scored completions) — methods absent elsewhere.
- **Fully asynchronous multi-turn RL** — Ray rollout actors overlap training via a prefetch queue, a clean Transformers ↔ vLLM/SGLang split with NCCL weight sync, and built-in environments; no Megatron backend, no veRL.
- Native `s3://` datasets and FA4-on-Blackwell.

## Framework comparison

Versions compared: TRL 1.6.0 and Accelerate 1.11.0 (the pinned releases), Unsloth, Axolotl 0.19.0, MS-SWIFT 4.0, veRL (Megatron backend), Megatron-LM/Core. "Via Megatron" = needs a Megatron backend + an HF↔MCore checkpoint conversion; "—" = out of scope (a layer the others build on).

| Feature | This Toolkit | HF TRL | Accelerate | Unsloth | Axolotl | MS-SWIFT | veRL | Megatron-LM |
|---------|:------------:|:------:|:----------:|:-------:|:-------:|:--------:|:----:|:-----------:|
| SMPO (reference-free PO) | **Yes** | No | — | No | No | No | No | No |
| Offline GRPO (pre-scored) | **Yes** | No | — | No | No | No | No | No |
| EP for MoE (no conversion) | **DeepEP** | No | No | No | **DeepEP** (v0.17) | Via Megatron | Via Megatron | MCore format |
| Context Parallelism | **Ulysses (SFT + SMPO)** | SFT-only; needs a newer Accelerate | Yes (FSDP2) | No | Yes (ring/seq) | Via Megatron | Via Megatron | MCore |
| Tensor Parallelism | Yes (DTensor) | Via Accelerate; needs a newer Accelerate | Yes (ND-parallel) | No | Yes (experimental) | Via Megatron | Via Megatron | MCore |
| Combined EP+CP / EP+TP / pure ETP | **Yes** | No | No (no EP) | No | Partial (EP×CP and EP×CP×FSDP; EP×TP raises) | Via Megatron | Via Megatron | Yes |
| Full BF16 (~6 bytes/param) | **AdamWBF16 + SR** | No (FP32 masters) | Plumbing only | No | bf16 MP only | No | No | No |
| Multi-turn RL | **Async Ray + built-in envs** | Async GRPO (thin envs) | — | Via ART | Async GRPO + NeMo Gym | GYM env | AgentLoop + SGLang | No |
| Native HF models (no conversion) | **Yes (all modes)** | Yes (no EP) | Yes (plumbing) | Yes (no EP/CP/TP) | Yes | MCore for EP/CP/TP | MCore for EP/CP/TP | No |
| Flash Attention (FA4 Blackwell / FA3 Hopper / FA2 fallback) | **Yes** | Inherited (FA2) | Inherited | No (FA2) | Yes (FA2/3/4) | Yes | Inherited (FA2) | Via TE |
| Native `s3://` datasets | **Yes** | No | No | No | Yes | No | No | Via MSC |
| Multi-node | **Tested** | Via Accelerate | Yes | Paid tiers | Yes | Yes | Yes | Yes |

Megatron-Core stores `torch_dist` sharded checkpoints split by TP/PP/EP/ETP — a separate format from HuggingFace's, so MS-SWIFT, veRL, and Megatron-LM all require an HF↔MCore conversion before EP/CP/TP/PP. This toolkit, Accelerate, TRL, Unsloth, and Axolotl work directly on HuggingFace weights; of those, this toolkit and Axolotl run DeepEP expert parallelism on the upstream HF MoE modules. Two neighbors sit close: NeMo AutoModel keeps HF checkpoints without a conversion step but ships its own implementation of each supported architecture behind the `transformers` API, and transformers itself has an expert-parallel path (`DistributedConfig(enable_expert_parallel=True)`) whose router all-reduces the full MoE output on every rank instead of dispatching tokens. On the pinned transformers 5.16.1 and Accelerate 1.11.0, the transformers `Trainer`, and TRL's trainers that subclass it, run neither TP, CP nor that EP path: each needs Accelerate's `ParallelismConfig` (`TrainingArguments.parallelism_config`), and that `Trainer` fails at construction on any `ParallelismConfig` under this Accelerate release.

### When to use which

| Use case | Recommended | Why |
|----------|:-----------:|-----|
| EP+TP or pure ETP on MoE | **This Toolkit** | Expert shapes other HF-native tools don't compose |
| Memory-constrained full fine-tuning | **This Toolkit** | AdamWBF16 — ~6 bytes/param |
| Long-context training (scales with nGPUs) | **This Toolkit** | CP splits sequences; combines with EP |
| SMPO or Offline GRPO | **This Toolkit** | Unique methods |
| Multi-turn RL with tool environments | **This Toolkit** | Ray actors, NCCL sync, built-in envs |
| Config-driven SFT/DPO/GRPO over many models (incl. FSDP+EP) | Axolotl | YAML, DeepEP EP (v0.17), s3/GCS data, FA2/3/4 |
| A hand-written loop needing only FSDP2 + TP + CP | Accelerate | The plumbing layer, no trainer overhead |
| Single-GPU LoRA/QLoRA on consumer hardware | Unsloth | Extreme VRAM efficiency, free tier |
| SFT/DPO on single node, standard models | HF TRL | Simplest setup, largest community |
| Online PPO/GRPO with many algorithm variants | veRL or MS-SWIFT | Purpose-built for online RL |
| Broad model coverage + no-code UI | MS-SWIFT | 600+ models, Gradio Web UI |
| Pre-training at 1000+ GPU scale | Megatron-LM | Pipeline parallelism, peak utilization |

### Published numbers in similar setups

Vendor-reported figures beside Halo's, with the setup stated; only the Megatron-LM and Axolotl rows
share hardware. B300 has 2.3× the bf16 peak of H100, so read a cross-GPU row as a ceiling on the
gap, not the gap.

- **gpt-oss-20b, seq 4096, bf16.** Halo: 24,456 tok/s/GPU at EP1 (8× B300, batch 4, GC off; 20,174
  GC on) and 10,051 at EP8 (batch 4, GC on, 57 GB). NeMo AutoModel reports 13,058 on 8× H100 (FSDP
  without EP, mock data, forced-balanced routing). Megatron-LM on the same 8× B300 (batch 2, GC off): OOM at EP1 where
  Halo runs 27,707; 13,932 vs Halo 21,642 at EP2; 15,876 vs 17,519 at EP4; 14,734 vs 11,856 at EP8
  (Megatron ahead there), at 246 vs 158 GB peak at EP1 and 104 vs 58 GB at EP8.
- **Qwen3-30B-A3B, bf16.** transformers' own expert-parallel path: 3,485 tok/s/GPU at 38.6 GB
  (8× H100, seq 2048, `tp_size=8`). Halo EP2 on 8× B300 at seq 4096: 6,898 / 11,343 / 14,536 at
  batch 1 / 2 / 4. NeMo AutoModel reports 12,040 on 8× H100 at EP8, seq 4096 (mock data,
  forced-balanced routing).
- **Same hardware.** Gemma 4 26B-A4B and Mistral Small 4 119B against NeMo AutoModel, Axolotl,
  Megatron Bridge, MS-SWIFT and Unsloth: [Throughput Benchmarks](../optimization/throughput-benchmarks.md#full-parameter-sft-framework-comparison).
  Laguna-S 2.1 on 4× B300: Halo 11,298 vs Axolotl 5,617 tok/s/GPU at 15% less memory. LFM2.5-8B-A1B:
  Halo 31,262 vs Axolotl 28,330 at seq 8192, 35,517 vs 29,520 at seq 16384.

The Megatron-LM, Laguna and LFM2.5 rows are from the Halo research page (whitecircle.com/research/halo).

Notes on individual frameworks:

- **Accelerate** is the foundation layer this toolkit builds on (launcher + FSDP2/DTensor TP/CP plumbing over native HF), not a competing trainer — it ships no trainers, RL, MoE/EP, attention kernels, or `s3://` loading.
- **Megatron-LM** adds deeper TE/FP8 optimization for 1000+ GPU dense pre-training; its alignment is split across four repos and needs MCore conversion. This toolkit's pipeline parallelism is [not yet available](../parallelism/pipeline-parallelism.md) (the seams ship, the engine lands in a future release); on large NVLink domains FSDP2 + EP/TP avoids the pipeline bubble — see [When PP is worth it](../parallelism/README.md#when-pipeline-parallelism-is-worth-it).
- **Axolotl** (v0.19.0) closes most of the EP gap (DeepEP on native HF, no conversion, composed with CP and FSDP) and offers config-driven YAML, s3/GCS data, FA2/3/4, and async-GRPO + NeMo Gym. This toolkit still adds SMPO, Offline GRPO, EP+TP and pure ETP (Axolotl's EP×TP raises), and full BF16.
- **NVIDIA NeMo RL** — alignment on DTensor (FSDP2+TP+CP) or Megatron-Core; MCore conversion on the Megatron path; capabilities span four repos.
- **SkyRL** — RL plus a native SFT trainer (v0.3.0), no DPO/reward/distillation; async dispatcher ~1.55× over naive batching; EP/CP/TP need its Megatron backend.

## Model compatibility

Any HuggingFace `AutoModelForCausalLM` works with standard FSDP, and any model carrying a `tp_plan`
gets TP. Advanced parallelism (EP, CP, ETP) requires per-family wrappers, not a model
reimplementation or a checkpoint conversion.

For EP, that is a wrapper under `src/distributed/expert_parallel/layers/` (under 140 lines; GPT-OSS
the outlier at 366) subclassing `EPMoELayerBase` and declaring its `HF_MODULE_NAMES` /
`HF_MODEL_TYPES`, with `MOE_LAYER_MAP` derived from the subclass tree so the family self-registers on
import. Fifteen MoE families ship one; the per-family × per-mode matrix is
[Supported Models](../models/README.md), and [Adding a New Model](../models/adding-a-model.md) is the
procedure.

PP — [not yet available in this release](../parallelism/pipeline-parallelism.md) — is orthogonal to
that matrix and ships two model gates. A family whose `PPModelSpec` sets
`SUPPORTS_PP = False` (Zaya, Gemma 4) is rejected at split time; separately, the stage
loader streams layers from safetensors, so a family whose EP layer sets
`_supports_lazy_loading = False` is rejected at load. Trainer × parallelism support is separate — see
[Trainer Compatibility](trainer-architecture.md#trainer-compatibility).

## Key dependencies

| Package | Version | Purpose |
|---|---|---|
| PyTorch | 2.11.x (+cu130) | Training framework |
| Transformers | 5.16.x | Model loading and tokenization |
| TRL | 1.6.x | GRPO, DPO, and RL trainers |
| Accelerate | 1.11.x | FSDP distributed training |
| PEFT | 0.18.x | LoRA and parameter-efficient fine-tuning |
| vLLM | 0.26.0 | Online generation for GRPO (separate container) |
| DeepEP | V2 (commit `af9a040`) | MoE expert parallelism |
| Flash Attention | 2.x / 3.x / 4.x (FA4 on Blackwell) | Attention acceleration |
| Liger Kernel | 0.8.x | Triton kernel optimizations |
| FlashAdamW | 0.1.x (extra) | Quantized AdamW states (~5 bytes/param) |

`pyproject.toml` is PEP 621; each pin is a bounded range (e.g. `transformers>=5.16.1,<5.17.0`, `torch>=2.11.0,<2.12.0`) resolved by `uv` into `uv.lock`.

> [!WARNING]
> **vLLM runs as a separate Docker container**
>
> vLLM pins its own torch/transformers stack — it is **not** in the training environment. It runs as an isolated container (`Dockerfile.vllm` + `docker-compose.vllm.yml`). The training container talks to vLLM over HTTP (generation) and a vendored NCCL client (`src/distributed/nccl/`, weight sync).
