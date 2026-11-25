<p align="center">
    <img src="/logo.png" height="100" alt="Halo logo: a white ring on black">
    <h3 align="center">Halo</h3>
</p>

<p align="center">
  Frontier-Lab Training for Everyone
</p>

<p align="center">
  <a href="https://whitecircle.com/research/halo"><strong>Blog post</strong></a> ·
  <a href="https://whitecircle.com/halo/docs"><strong>Docs</strong></a> ·
  <a href="examples/"><strong>Examples</strong></a> ·
  <a href="https://whitecircle.com/halo/docs/performance"><strong>Benchmarks</strong></a>
</p>

<br/>

## What is Halo

Halo is an open-source framework built by [White Circle](https://whitecircle.com) for training large language and multimodal models. It supports pre-training, SFT, preference optimization, fully asynchronous multi-turn RL, and much more.

The same codebase runs on one GPU or across multiple nodes, with expert, context, tensor, expert-tensor and pipeline parallelism (EP, CP, TP, ETP, PP), FSDP2, fused kernels, BF16 training, and distributed rollouts.

Halo trains Hugging Face models directly. Checkpoints still load with `from_pretrained`, and the EP layer a MoE family adds is under 90 lines for every family but Inkling 2, at 117, and GPT-OSS, at 168 ([Model Integration Cost](human-docs/model-integration-cost.md)).

On 8× B300, Halo 1.2.0 trains Qwen3.6-35B-A3B on packed 64k and 128k rows at 1.10–1.12× Megatron Bridge, 1.47× TRL 1.17.1 and 1.9–2.0× NeMo AutoModel 0.7.0, and Qwen3.5-122B-A10B at 1.50× TRL and 1.7–1.9× NeMo; it trails a tuned Megatron by 11.5% on dense Qwen3.5-9B at 64k ([Benchmarks](#benchmarks)). It runs 1.11–2.07× as fast as Halo 1.1.0 on the twelve shapes benchmarked against it.

`Pre- & Post-Training` · `EP / CP / TP / ETP / PP` ·
`Multi-Node` · `Verifiable & Multi-Turn RL` · `FA4 + Liger + Grouped GEMM` · `Full BF16`

</br>

## Why Halo

- Trainers subclass Hugging Face or TRL trainers, and checkpoints stay standard SafeTensors.

- Halo adds parallelism to the model you already have. EP wraps the MoE blocks, CP wraps attention, and TP and ETP shard weights in place. No second, distributed copy of the model code exists.

- The distributed stack is built on PyTorch. FSDP2, DTensor, and DeviceMesh provide the primitives, and DeepEP handles the all-to-all communication for Expert Parallelism.

- Each parallelism axis has its own setting. EP, CP, TP, ETP, and PP are configured separately. EP combines with CP, TP, or ETP, PP combines with EP or ETP, and other pairings are rejected at config time.

- Training methods share the same infrastructure. Pre-training, SFT, preference optimization, distillation, and RL use the same parallelism and checkpointing code, so a new method doesn't need its own distributed implementation.

- SFT can train a release's MTP head. The multi-token-prediction head that ships for speculative decoding trains beside a fine-tune or over a frozen trunk ([MTP heads](agent-docs/optimization/mtp-heads.md)).

- Performance optimizations are integrated. Halo includes DeepEP V2, FlashAttention 4, Liger, Grouped GEMM, padding-free and tree packing, AdamWBF16, and kernels for GLM-5 Next, with fallbacks for Blackwell, Hopper, and older GPUs.

- RL is asynchronous, with training and rollout serving in separate processes. Multi-turn environment rollouts run as Ray actors against vLLM or SGLang servers, and the trainer pushes updated weights to them over native NCCL. The training side stays plain Transformers, with no Megatron backend or veRL dependency. See [Async GRPO with Environments](human-docs/training-methods/async-grpo-environments.md).

</br>

## Latest updates

<!-- Newest first, a few lines per entry. The details of a release belong in its release notes. -->

Highlights only. Every release's full notes, with breaking changes and upgrade steps, are in
[GitHub Releases](https://github.com/whitecircle/halo/releases).

- **2026-11-25 — Halo 1.3.0rc2.** Adds Inkling 2: Inkling-2-Small trains with LoRA on its attention, alone or
  beside its routed experts, while the released NVFP4/MXFP8 expert weights stay frozen
  ([Inkling 2](agent-docs/models/inkling2.md),
  [release notes](https://github.com/whitecircle/halo/releases/tag/v1.3.0rc2)).
  - **Adapter sync on vLLM:** environmental GRPO serves each adapter version from a shared directory, and
    Inkling 2 is served from its release with no full-weight push.
  - **Qwen3.8-Flash-Next:** block-sparse attention and gated-residual kernels are on by default (1.57× and 1.22×
    faster LoRA steps at EP8 on 8× B300), and its weights sync to vLLM and SGLang for RL.
  - **Also:** rollouts without reasoning, per-row reasoning effort, fused MXFP4 and NVFP4 fake-quantization
    kernels, chunked distillation losses, MaxRL advantages, and Transformers 5.23.
- **2026-11-24 — Halo Competition.** Dec 1–10, with tracks for making Halo faster on a model family, post-training
  one for a task you pick, and best-use tracks from the partners and White Circle. $200k in cash prizes and $1.5M
  in compute and API credits from OpenAI, Mistral, Nebius, Modal, Lovable and others. Apply by Nov 30
  ([details](https://whitecircle.com/halo/competition), [apply](https://luma.com/ykoj0t48)).
- **2026-11-07 — Halo 1.3.0rc1.** Faster RL at scale: a synchronous GRPO step on Qwen3.5-35B-A3B over two 8× B300
  nodes takes 0.137 s per 1k response tokens against Miles v0.1.3's 0.176
  ([GRPO against Miles](agent-docs/optimization/throughput-benchmarks.md#grpo-against-miles),
  [release notes](https://github.com/whitecircle/halo/releases/tag/v1.3.0rc1)).
- **2026-11-04 — Halo 1.2.0.** Training steps are 1.11× to 2.07× faster than in 1.1.0 on all twelve shapes
  benchmarked against it. Adds pipeline parallelism, tree packing, MTP heads and three model families
  ([benchmarks](human-docs/performance.md),
  [release notes](https://github.com/whitecircle/halo/releases/tag/v1.2.0)).
- **2026-10-11 — Halo 1.1.0.** Faster MoE training, exact resume, and more reliable async RL
  ([release notes](https://github.com/whitecircle/halo/releases/tag/v1.1.0)).
- **2026-08-20 — Halo 1.0.0.** First public release: EP / CP / TP / ETP on native HuggingFace models, 15 MoE
  families, and pre-training through multi-turn RL with vLLM or SGLang rollouts
  ([release notes](https://github.com/whitecircle/halo/releases/tag/v1.0.0)).

</br>

## Quick start

### Installation
Halo ships as separate Docker images for Blackwell and Hopper, with PyTorch, Flash Attention, and
DeepEP V2 built in. Pull them anonymously, with no account:

```bash
# B200/B300: FA4 + FA2 + DeepEP
docker pull public.ecr.aws/whitecircle/halo:blackwell

# H100/H200: FA2 + FA3 + DeepEP
docker pull public.ecr.aws/whitecircle/halo:hopper
```

Versioned tags (`:blackwell-1.3.0rc2`, `:hopper-1.3.0rc2`) pin the release. There is no `latest` tag, since
the images are architecture-specific. The RL inference images are published alongside them
(`:vllm-0.32.0`, `:sglang-0.5.22`), pinned as `:vllm-0.32.0-1.3.0rc2` and `:sglang-0.5.22-1.3.0rc2`.
Inkling 2 is served from its own vLLM image ([Inkling 2](agent-docs/models/inkling2.md#rollouts)).

To build locally instead:
```bash
# B200/B300: FA4 + FA2 + DeepEP
make build-blackwell

# H100/H200: FA2 + FA3 + DeepEP
make build-hopper
```
The commands below run inside a container started from the image, so the host needs no Python environment.
[Installation](human-docs/installation.md) has the retag to the local image name, the `.env` file and the
`docker run` line that starts the container.

### Launch a training run
`halo launch` takes a training method and a YAML config:

```bash
# SFT + LoRA (fits into a single GPU)
halo launch sft examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml

# SFT + Expert Parallelism (MoE, 8 GPUs → torchrun)
halo launch sft examples/sft/qwen3_5/qwen3.5-35b-a3b-ultrachat-ep.yaml -n 8

# FSDP-2 + Accelerate
halo launch sft examples/sft/qwen3/qwen3-4b-ultrachat.yaml \
    -a launcher-configs/accelerate/fsdp2_gradop_config.yaml -n 8

# all training methods (sft, preference/smpo, grpo/online, grpo/environmental, ...)
halo launch --list
```

`halo` selects `python`, `torchrun`, or `accelerate launch` based on the arguments, and `--dry_run`
prints the command instead of running it. Options after the config that are not the launcher's own are
passed to the trainer and override the YAML config:

```bash
halo launch sft examples/sft/qwen3_5/qwen3.5-35b-a3b-ultrachat-ep.yaml \
    -n 8 \
    --expert_parallel_size=8 \
    --learning_rate=1e-5 \
    --max_length=32000
```

### Other tools

Checkpoint, inference, data, and environment utilities use `halo run`:

```bash
halo run merge-ep-shards --input_dir <ep-checkpoint-dir> --output_dir <merged-dir>
halo run quantize-to-lowp --input_dir <bf16-model-dir> --output_dir <nvfp4-out-dir> --format nvfp4

# all tools (merge-*, quantize-to-lowp, inference/reward-model/score, eval-env, …)
halo run --list
```

The underlying scripts can also be run directly:

```bash
python scripts/training/sft.py \
    examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml

torchrun --nproc_per_node=8 scripts/training/sft.py \
    examples/sft/qwen3_5/qwen3.5-35b-a3b-ultrachat-ep.yaml
```

### Coding agents

The training images ship repo-specific agent skills (discoverable by Claude Code and Codex; add
`--build-arg INSTALL_CLAUDE_CODE=1` to bake the CLI into the image) for [launching jobs](skills/launch),
[choosing parallelism settings](skills/parallelism), [debugging distributed runs](skills/debug), [adding models](skills/add-model), [managing checkpoints](skills/checkpoints),
[preparing data](skills/data), [optimizing throughput](skills/optimize), [setting up RL](skills/rl-setup),
[writing tests](skills/write-tests), [keeping docs current](skills/docs), [writing for people](skills/voice)
and [making figures](skills/visuals).

</br>

## Documentation

We keep two versions of the docs, depending on whether or not you're AI:

- **[`human-docs/`](human-docs/README.md)**: shorter, task-oriented guides for installing, configuring,
  and running Halo.
- **[`agent-docs/`](agent-docs/README.md)**: the full technical reference, written for coding agents,
  covering model integrations, parallelism, optimization internals, infrastructure, and troubleshooting.

The full reference is also perfectly readable if you're human and want the details.

**The human guide:**
[Installation](human-docs/installation.md) ·
[Quickstart](human-docs/quickstart.md) ·
[Choosing a Method](human-docs/choosing-a-method.md) ·
[The halo CLI](human-docs/cli.md) ·
[Writing a Config](human-docs/configuration.md) ·
[Datasets](human-docs/data.md) ·
[Training Methods](human-docs/training-methods/README.md) ·
[Rollout Servers](human-docs/rollout-servers.md) ·
[Parallelism](human-docs/parallelism.md) ·
[Clusters & Multi-Node](human-docs/clusters.md) ·
[Performance](human-docs/performance.md) ·
[Supported Matrix](human-docs/supported-matrix.md) ·
[Supported Models](human-docs/models.md) ·
[Model Cookbooks](human-docs/cookbooks/README.md) ·
[Model Integration Cost](human-docs/model-integration-cost.md) ·
[Checkpoints & Export](human-docs/checkpoints.md) ·
[Monitoring](human-docs/monitoring.md) ·
[Troubleshooting](human-docs/troubleshooting.md) ·
[Environment Variables](human-docs/environment-variables.md) ·
[AI Tooling](human-docs/ai-tooling.md) ·
[Contributing](human-docs/contributing.md)

</br>

## Repository layout

A map of the repository:

| Path | What's inside |
|---|---|
| **`halo/`** | Trainers, parallelism, optimizers, kernels, collators, and callbacks |
| **`scripts/`** | Training entry points, inference, data-prep, and checkpoint tools |
| **`examples/`** | Ready-to-run YAML configs, grouped by method and model |
| **`launcher-configs/`** | Accelerate, SkyPilot, and Nomad launch configs |
| **`human-docs/`** | Short, task-oriented documentation |
| **`agent-docs/`** | Full technical reference |
| **`skills/`** | Agent skills (linked from `.claude/skills` and `.agents/skills`) |
| **`tests/`** | CPU and distributed GPU tests |
| **`Dockerfile` · `Makefile`** | Image builds, tests, and training commands |
| **`AGENTS.md`** | Contributor and agent instructions |

A typical run picks a config from `examples/` and launches it inside the Docker image with `halo launch <method> <config>` (or `python scripts/training/<method>.py <config>`).

</br>

## Benchmarks

All rows ran on 8× B300 with the Blackwell image unless a row says otherwise. Rates are tokens/s per GPU on a warm
kernel cache; a fresh Triton cache costs the first steps (Qwen3.5-35B-A3B's first 20 or so steps run at about 40%
of the steady rate).

### Against other frameworks

Full-parameter SFT on UltraChat 200k packed into rows of exactly 64k or 128k tokens with documents isolated, the same rows
for every framework. Each cell is tok/s/GPU (peak GiB); † is an nvidia-smi peak, which reads higher than the others.

| Model, tokens per row | Halo 1.2.0 | Megatron Bridge | TRL 1.17.1 | NeMo AutoModel 0.7.0 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B, 64k | 17,206 (54.3) | **19,431** (245.5†) | 14,735 (56.0) | 10,360 (99.3) |
| Qwen3.5-9B, 128k | 17,093 (82.1) | **17,348** (241.1†) | 14,601 (86.8) | 9,451 (166.4) |
| Qwen3.6-35B-A3B, 64k | **23,637** (139.1) | 21,559 (224.6†) | 16,126 (138.8) | 12,340 (99.1) |
| Qwen3.6-35B-A3B, 128k | **23,647** (139.1) | 21,109 (244.2†) | 16,112 (138.8) | 12,026 (139.3) |
| Qwen3.5-122B-A10B, 64k | **8,933** (188.7) | 4,781 (254.0†) | 5,936 (194.5) | 5,397 (219.3) |
| Qwen3.5-122B-A10B, 128k | **8,943** (188.7) | does not fit | 5,951 (194.5) | 4,683 (227.1) |

Megatron Bridge is a 0.7.0 release-candidate commit with Megatron Core 0.20.0. How to read the table:

- Optimizer regimes differ: Halo trains bf16 parameters with stochastic rounding, TRL and NeMo bf16 parameters with bf16 Adam
  state, Megatron fp32 main parameters (fp16 master on the 122B).
- Megatron's peaks are nvidia-smi samples; the others are torch allocated peaks. Do not compare them directly.
- Each Halo row is the median of 3 runs of a chosen setup; the others are single runs of their best layouts.

Rates per parallelism layout, the 16k and 4k tables, and the other conditions: [Throughput Benchmarks](agent-docs/optimization/throughput-benchmarks.md#full-parameter-sft-framework-comparison);
the TRL side: [Halo vs stock TRL](agent-docs/optimization/halo-vs-stock-trl.md).

### GRPO against Miles

GRPO on Qwen3.5-35B-A3B (GSM8K, 32 prompts × 8 samples, one 8× B300 node training and one serving SGLang
rollouts, same prompts on both sides): on-policy, Halo takes 0.137 s per 1k response tokens against Miles v0.1.3's 0.176, 22% less,
with a 14.0 s step against 18.6 s and 63 against 109 GiB on the trainer node. With rollouts overlapped it takes 0.102
against 0.111, 8% less. Layouts and conditions:
[GRPO against Miles](agent-docs/optimization/throughput-benchmarks.md#grpo-against-miles).

### Halo 1.2.0 against 1.1.0

| Result | Configuration |
|---|---|
| **1.11–2.07× Halo 1.1.0** | Twelve SFT shapes, none slower: TP2 1.82×, EP4+ETP2 2.07×, EP 1.12–1.29×, CP2 1.15×, FSDP2 1.11×. Qwen3-8B, gpt-oss-20b and Qwen3.5-35B-A3B at batch 1, gradient checkpointing (GC) on. |
| **2.2–2.7× on GLM-5.3-Flash LoRA** | EP8, real text: 2,840 vs 1,282 at 8k, 2,390 vs 893 at 16,384. Same peak memory. The baseline is 1.2.0 without the two new kernels. |
| **1.65–2.3× on GLM-5.3-Flash full fine-tune** | Two 8-GPU machines: EP16 1,504 vs 913, PP2 + EP8 2,423 vs about 1,060. EP8 with DP runs out of memory. |
| **PP across two machines** | gpt-oss-20b PP2 + EP8 at 8k: 15,715, 15.9% below EP8 with DP. Qwen3-8B PP2 at 4k: 17,608, 4.0% below DP16. Qwen3.5-35B-A3B PP2 + EP8 reaches 0.97× EP8 with DP, at 126.7 GiB on the first stage and 62.7 on the last, against 148.7 (GC off, packed rows of 8k). |
| **13.8× DP16 on plain TCP sockets** | Machines forced onto plain TCP sockets: Qwen3-8B PP2 holds 15,414 while DP16 falls to 1,117. |

### Kernels and memory (Halo 1.1.0 or earlier)

No later release has rerun these rows.

| Result | Configuration |
|---|---|
| **28,700 tok/s/GPU** | gpt-oss-20b, dense EP1, 4k, batch 4, GC off: ~230k tok/s across 8 GPUs. |
| **16,447 tok/s/GPU** | Qwen3.5-35B-A3B, 4k, batch 4, EP2, SDPA. |
| **25–76 GiB/GPU at 16k tokens** | EP8+CP8, one 16k row over 8 GPUs: 24.6 GiB at 7,149 tok/s/GPU. Dense EP1, 4 × 4k per GPU: 75.8 GiB at 23,590 tok/s/GPU. |
| **256k context** | gpt-oss-20b, dense EP1 ZeRO-3 with fused linear cross-entropy: 5,728 tok/s/GPU at 112 GiB (measured on Halo 1.0.0). |
| **2.45× Grouped GEMM** | Qwen3-30B-A3B, 2× B300, EP2, 8k, batch 4, GC on; 4.10× at batch 1. |
| **3.6–3.9× FA4 kernel throughput** | FA4 vs FA2 on the isolated kernel (B2×S8192, head_dim 128); up to 2.3× end-to-end on dense Qwen3-4B at 32k (1× B300). |

[Throughput Benchmarks](agent-docs/optimization/throughput-benchmarks.md) has the full results and the setup behind them.

</br>

## Parallelism

Each axis has its own flag. EP combines with one of CP, TP, or ETP, and PP with EP or ETP; any other pairing is rejected at config time:

```bash
--expert_parallel_size=N          # MoE experts across GPUs (DeepEP)
--context_parallel_size=N         # long sequences (Ulysses attention)
--tensor_parallel_size=N          # model weights (DTensor; attention only on MoE)
--expert_tensor_parallel_size=N   # expert-MLP sharding (MoE, experimental)
--pipeline_parallel_size=N        # decoder layers in stages across nodes (registered families)
```

| Mode | Best for                                  |
|---|-------------------------------------------|
| EP | Distribute MoE experts across GPUs        |
| TP | Shard model weights across GPUs (a dense model past one GPU) |
| CP | Split long sequences across GPUs          |
| ETP | Shard individual expert MLPs              |
| EP+CP | MoE + long-context training               |
| EP+TP | MoE + weight sharding                     |
| EP+ETP | MoE + sharded experts                     |
| PP | Split a model's layers across nodes       |
| PP+EP, PP+ETP | MoE across nodes, experts inside each stage |

Between machines on a fast link, PP is slower than data parallelism when the model fits. It is faster on slow
links, against layouts that send attention or experts across machines, and for models that do not fit at
EP8 on one machine. On one machine it gained only with EP, where gpt-oss-20b at PP2 + EP4 ran 9.2% above
EP8 (8k rows, GC off, one run each). PP does not support LoRA yet.

Pass them directly to `torchrun`, or after the config with `halo launch`:

```bash
halo launch sft config.yaml -n 8 --expert_parallel_size=8
```

Multi-node runs support shared (NFS/Lustre) and node-local filesystems.
See [Parallelism](human-docs/parallelism.md) to pick a mode, [Supported Matrix](human-docs/supported-matrix.md)
for the per-model matrix, and [Clusters & Multi-Node](human-docs/clusters.md) for the multi-node recipes.

</br>

## Optimizations

Built in. Most apply by default where supported; QAT, the Muon and FlashAdamW optimizers, LoRA and QLoRA, and the packing modes are set per run in the config:

- **Attention and kernels.** Flash Attention 4 on Blackwell, FA3 on Hopper, FA2 elsewhere, Liger fused
  CE/SwiGLU/RMSNorm (Qwen3 MoE, GLM-4.7-Flash, GPT-OSS, Gemma 4, Nemotron-H and Inkling 2 take torch's fused RMSNorm instead),
  fused GLU kernels, and Grouped GEMM for MoE expert compute.

- **Family kernels.** GLM-5 Next's DSA attention on Blackwell (`HALO_DSA_MATMUL`) and the hyper-connections of
  GLM-5 Next and DeepSeek-V4 (`HALO_HC_KERNELS`) run on dedicated kernels
  ([GLM-5 Next](agent-docs/models/glm5-next.md), [hyper-connection kernels](agent-docs/optimization/hyper-connection-kernels.md)).
  So do Qwen3.8-Flash-Next's block-sparse attention (`HALO_QSA_KERNEL`) and gated residuals
  (`HALO_GATED_RESIDUAL_KERNELS`), both on by default
  ([kernel defaults](agent-docs/models/qwen4_exp.md#kernel-defaults)). Inkling 2's attention runs on compiled
  FlexAttention, and each frozen NVFP4 or MXFP8 expert bank dequantizes in one Triton kernel
  ([Inkling 2](agent-docs/models/inkling2.md#measured)).

- **Parallel steps.** TP linears on their local shards, one Ulysses all-to-all per exchange, a sync-free
  MoE routing sort, and two decoder layers per FSDP2 shard group under ZeRO-2.

- **Precision and optimizers.** `AdamWBF16` with stochastic rounding keeps weights and optimizer
  state in 6 bytes/parameter instead of 12 with FP32 master weights and moments, plus the 2-byte bf16
  working copy. Muon and FlashAdamW are also
  supported. FP8/FP4 QAT runs on MoE experts and dense MLPs through a fake-quantization path, with an opt-in
  DeepGEMM forward; `quantize-to-lowp` writes block-scaled mxfp8/mxfp4/nvfp4 weights.

- **Memory and PEFT.** Padding-free packing with `cu_seq_lens`, SFT tree packing, LoRA, and QLoRA. The
  [Qwen3-4B QLoRA example](examples/sft/qwen3/qwen3-4b-ultrachat-qlora.yaml) peaks at 7.9 GiB (batch 1, 4k)
  and Qwen3-8B at 32k at 20.9 GiB with the fused loss ([PEFT](agent-docs/optimization/peft.md)). Single-GPU
  LoRA/QLoRA also runs on Ampere and Ada cards from `halo:blackwell`, but that path is not validated.

- **Checkpoints.** Large gathered checkpoints are automatically sharded, with tools for merging per-rank
  EP shards, per-node PP saves and PEFT adapters back into a standard HuggingFace checkpoint
  ([Checkpoints](human-docs/checkpoints.md)).

- **Data.** Offline tokenization, packing, and sharding, plus native `s3://` dataset loading with a local cache.

[Performance](human-docs/performance.md) covers the levers that help, what each one gains, and the ones that do not help at MoE shapes.

</br>

## Built with AI

We use GPT and Claude extensively while developing Halo, through
[PI-Agent](https://github.com/earendil-works/pi),
[Claude Code](https://github.com/anthropics/claude-code), and
[Codex](https://github.com/openai/codex). 

[`AGENTS.md`](AGENTS.md) contains the instructions and conventions used by both coding agents and
human contributors.

AI-assisted contributions are welcome. We ask that contributors understand and stand behind
the code they submit. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

</br>

## Acknowledgments

Halo builds on a lot of great open-source work, in particular:

- **[Hugging Face](https://huggingface.co)**: Transformers, TRL, Accelerate, Datasets, and PEFT.
- **[Liger Kernel](https://github.com/linkedin/Liger-Kernel)**: fused cross-entropy, SwiGLU, and RMSNorm.
- **[FlashAdamW](https://github.com/databricks/flashoptim)** by Databricks AI Research: memory-efficient
  AdamW with quantized optimizer states and compact master weights.
- **[DeepEP](https://github.com/deepseek-ai/DeepEP)** and **[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)**
  by DeepSeek-AI: expert communication and grouped GEMM. Halo's aux-loss-free router balancing also
  follows the approach introduced with DeepSeek-V3.
- **[Flash Attention](https://github.com/Dao-AILab/flash-attention)** by Tri Dao and contributors:
  FA2, FA3, and FA4.
- **[vLLM](https://github.com/vllm-project/vllm)**: the rollout/serving engine behind online and
  environmental RL.
- **[PyTorch](https://pytorch.org)**: DTensor, FSDP2, and `torch.distributed`.
- **[SGLang](https://github.com/sgl-project/sglang)**: the alternative rollout engine, with NCCL weight
  synchronization, for multi-turn environment RL.
- **[SDPG](https://arxiv.org/abs/2606.04036)**, led by researchers at UCLA: self-distilled policy
  gradient for online reinforcement learning.

DeepEP is built from upstream at a pinned commit in the training image and keeps its upstream license;
the build pins, Flash Attention's included, live in the [`Dockerfile`](Dockerfile).
See [`pyproject.toml`](pyproject.toml) for the full dependency set.

Halo is built and maintained by **[White Circle](https://whitecircle.com)**.

</br>

## License

Halo is released under the **Halo License**: the Apache License 2.0
([`APACHE-2.0.txt`](APACHE-2.0.txt)) as modified by the Supplemental Terms in [`LICENSE`](LICENSE).
Copyright © 2026 White Circle, PBC.

In plain terms: free to use, modify, self-host, and to train your own models and ship them in your
own products, at any scale. The one commercial condition is offering Halo's training functionality
to third parties as a service: that's free up to US \$20M in such revenue over any 12 months, and
above that requires a commercial agreement with White Circle. Models trained with Halo by anyone
over \$20M in annual revenue must carry a "Trained with Halo" credit.

Questions or commercial licensing: [hello@whitecircle.com](mailto:hello@whitecircle.com).
