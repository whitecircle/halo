<p align="center">
    <img src="https://whitecircle.com/favicon.ico" height="88">
    <h3 align="center">Halo</h3>
</p>

<p align="center">
  Frontier-Lab Training for Everyone
</p>

<p align="center">
  <a href="https://whitecircle.com/research/halo"><strong>Blog post</strong></a> ·
  <a href="human-docs/README.md"><strong>Docs</strong></a> ·
  <a href="examples/"><strong>Examples</strong></a> ·
  <a href="agent-docs/optimization/throughput-benchmarks.md"><strong>Benchmarks</strong></a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Halo%20License-1f6feb.svg" alt="License: Halo License"></a>
</p>
<br/>

## What is Halo

Halo is an open-source framework built by [White Circle](https://whitecircle.com) for training large language and multimodal models. It supports pre-training, SFT, preference optimization, fully asynchronous multi-turn RL, and much more.

The same codebase runs on one GPU or across multiple nodes, with EP, CP, TP, ETP, FSDP2, fused kernels, BF16 training, and distributed rollouts.

Halo trains Hugging Face models directly. Checkpoints still load with `from_pretrained`, and supporting a new model family typically requires about 100 lines of integration code.

On 8× B300, Halo delivers up to ~2.8× the training throughput of stock TRL while using less peak memory, with larger margins over the other frameworks benchmarked.

`Pre- & Post-Training` · `EP / CP / TP / ETP` ·
`Multi-Node` · `Verifiable & Multi-Turn RL` · `FA4 + Liger + Grouped GEMM` · `Full BF16`

</br>

## Why Halo

- **Halo trains Hugging Face models directly.** Trainers subclass Hugging Face or TRL trainers, checkpoints remain standard SafeTensors, and `from_pretrained` continues to work.

- **Parallelism is added to the existing model.** EP wraps MoE blocks, CP wraps attention, and TP/ETP shard weights in place. There is no separate distributed implementation of the model.

- **The distributed stack is built on PyTorch.** FSDP2, DTensor, and DeviceMesh provide the underlying primitives, while DeepEP handles the all-to-all communication for Expert Parallelism. EP, CP, TP, and ETP can be configured independently and combined across GPUs and nodes.

- **Training methods share the same infrastructure.** Pre-training, SFT, preference optimization, distillation, and RL use the same parallelism and checkpointing code, so new methods don't need their own distributed implementation.

- **Performance optimizations are integrated.** Halo includes DeepEP V2, FlashAttention 4, Liger, Grouped GEMM, padding-free packing, and AdamWBF16, with implementations and fallbacks for Blackwell, Hopper, and older GPUs.

- **RL is asynchronous, with a clean Transformers ↔ vLLM/SGLang split.** Multi-turn environment rollouts run as Ray actors against a vLLM or SGLang server and overlap training through a prefetch queue; the trainer pushes updated weights to the server over native NCCL. There is no Megatron backend and no veRL dependency — the training side stays plain Transformers, with the same parallelism and checkpointing stack. See [Environmental GRPO](agent-docs/training-methods/grpo/environmental-grpo.md).

</br>

## Latest updates

<!-- Newest first. This feed and GitHub Releases are Halo's changelog. -->

- **2026-08-20 — Halo 1.0.0.** First public release: EP / CP / TP / ETP on native HuggingFace models
  (LLM and VLM), 15 MoE families, pre-training through multi-turn RL with vLLM or SGLang rollouts,
  FlashAttention-4, DeepEP V2, DeepGEMM, and the bf16 `AdamW` optimizer.

</br>

## Quick start

### Installation
Halo ships as separate Docker images for Blackwell and Hopper, with PyTorch, Flash Attention, and
DeepEP V2 already built in — pull anonymously, no account required:

```bash
# B200/B300 — FA4 + FA2 + DeepEP
docker pull public.ecr.aws/whitecircle/halo:blackwell

# H100/H200 — FA2 + FA3 + DeepEP
docker pull public.ecr.aws/whitecircle/halo:hopper
```

Versioned tags (`:blackwell-1.0.0`, `:hopper-1.0.0`) pin the release. There is
 no `latest` tag, since the images are architecture-specific. The RL
inference images are published alongside them (`:vllm-0.26.0`, `:sglang-0.5.17`).

To build locally instead: 
```bash
# B200/B300 — FA4 + FA2 + DeepEP
make build-blackwell

# H100/H200 — FA2 + FA3 + DeepEP
make build-hopper
```
No host Python environment is required. See [Docker](agent-docs/infrastructure/docker.md) for image details and build options.

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

# all training methods (SFT, SMPO, DPO, online-GRPO, environmental-GRPO, ...)
halo launch --list
```

`halo` selects `python`, `torchrun`, or `accelerate launch` based on the arguments. Options after `--`
are passed to the trainer and override the YAML config:

```bash
halo launch sft examples/sft/qwen3_5/qwen3.5-35b-a3b-ultrachat-ep.yaml \
    -n 8 -- \
    --expert_parallel_size=8 \
    --learning_rate=1e-5 \
    --max_length=32000
```

Use `--dry-run` to print the command without running it, and `--list` to see the available methods:

```bash
halo launch --list
halo launch sft config.yaml --dry-run
```

### Other tools

Checkpoint, inference, data, and environment utilities use halo run:

```bash
halo run merge-ep-shards -- --input_dir <ep-checkpoint-dir> --output_dir <merged-dir>
halo run quantize-to-lowp -- --input_dir <bf16-model-dir> --output_dir <nvfp4-out-dir> --format nvfp4

# all tools (merge-*, quantize-to-lowp, rm-scoring, run-env, …)
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
[preparing data](skills/data), [optimizing throughput](skills/optimize), and [setting up RL](skills/rl-setup).

See [`skills/`](skills/) for the available skills.

</br>

## Documentation

We keep two versions of the docs, depending on whether or not you're AI:

- **[`human-docs/`](human-docs/README.md)** — shorter, task-oriented guides for installing, configuring,
  and running Halo.
- **[`agent-docs/`](agent-docs/README.md)** — the full technical reference about model integrations,
  parallelism, optimization internals, infrastructure, and troubleshooting.

The full reference is also perfectly readable if you're human and want the details.

**The human guide:**
[Installation](human-docs/installation.md) ·
[Quickstart](human-docs/quickstart.md) ·
[The halo CLI](human-docs/cli.md) ·
[Choosing a Method](human-docs/choosing-a-method.md) ·
[Configuration](human-docs/configuration.md) ·
[Datasets](human-docs/data.md) ·
[Parallelism](human-docs/parallelism.md) ·
[Clusters & Multi-Node](human-docs/clusters.md) ·
[Supported Models](human-docs/supported-matrix.md) ·
[Checkpoints & Export](human-docs/checkpoints.md) ·
[Monitoring](human-docs/monitoring.md) ·
[Troubleshooting](human-docs/troubleshooting.md) ·
[Environment Variables](human-docs/environment-variables.md) ·
[AI Tooling](human-docs/ai-tooling.md) ·
[Model-Integration Cost](human-docs/model-integration-cost.md) ·
[Contributing](human-docs/contributing.md)

**Go deeper in the reference (`agent-docs/`):**
[Installation](agent-docs/getting-started/installation.md) ·
[Models](agent-docs/models/README.md) · [Adding a Model](agent-docs/models/adding-a-model.md) ·
[Parallelism (EP/CP/TP/ETP)](agent-docs/parallelism/expert-parallelism.md) ·
[Multi-Node](agent-docs/parallelism/multi-node.md) ·
[Pre-training](agent-docs/training-methods/pretraining.md) ·
[Optimization & Benchmarks](agent-docs/optimization/throughput-benchmarks.md) ·
[Docker](agent-docs/infrastructure/docker.md) ·
[Scale & Limitations](agent-docs/reference/scale-and-limitations.md) ·
[Why This Framework](agent-docs/reference/why-this-framework.md)

</br>

## Repository layout

A map of the repository:

| Path | What's inside |
|---|---|
| **`src/`** | Trainers, parallelism, optimizers, kernels, collators, and callbacks |
| **`scripts/`** | Training entry points, inference, data-prep, and checkpoint tools |
| **`examples/`** | Ready-to-run YAML configs, grouped by method and model |
| **`launcher-configs/`** | Accelerate, SkyPilot, and Nomad launch configs |
| **`human-docs/`** | Short, task-oriented documentation |
| **`agent-docs/`** | Full technical reference |
| **`skills/`** | Agent skills (linked from `.claude/skills` and `.agents/skills`) |
| **`tests/`** | CPU and distributed GPU tests |
| **`Dockerfile` · `Makefile`** | Image builds, tests, and training commands |
| **`AGENTS.md`** | Contributor and agent instructions |

A typical run is: **pick an `examples/` config → `halo launch <method> <config>`** (or `python scripts/training/<method>.py <config>`), inside the Docker image.

</br>

## Benchmarks

Benchmarks below were run on 8× B300. The stock TRL baseline uses `trl.SFTTrainer` with
Transformers v5 and FSDP2 ZeRO-3, with the same model, data, bf16 precision, FlashAttention 4,
Liger kernels, and grouped GEMM.

| Result | Configuration |
|---|---|
| **2.3–2.8× stock TRL throughput** | gpt-oss-20b, 4k–16k; EP2 / dense EP1. Loss matches the baseline to ~1% by step 100. |
| **24,456 tok/s/GPU** | gpt-oss-20b, 4k, batch 4, GC off — ~196k tok/s across 8 GPUs. |
| **12,584 tok/s/GPU at 1,435 TFLOPS** | Qwen3.5-35B-A3B, 4k, batch 4, EP2 — the highest achieved TFLOPS of the MoE rosters. |
| **Up to 256k context** | gpt-oss-20b; dense EP1 is 2.1× faster than TRL at 64k and 1.28× at 256k. CP configurations cut per-GPU memory to roughly half the baseline. |
| **23–76 GB/GPU on the same 16k workload** | EP8+CP8: 23 GB at 5.4k tok/s/GPU. Dense EP1: 76 GB at 18.3k tok/s/GPU. |
| **2.12× Grouped GEMM** | Qwen3-30B-A3B, EP2; 3.43× at batch 1. |
| **2.1–3.7× FA4 kernel throughput** | FA4 vs FA2; up to 2.3× end-to-end on dense long-context training. |

Full results and methodology:
[Throughput Benchmarks](agent-docs/optimization/throughput-benchmarks.md) ·
[Halo vs stock TRL](agent-docs/optimization/halo-vs-stock-trl.md)

</br>

## Parallelism

EP, CP, TP, and ETP are configured independently:

```bash
--expert_parallel_size=N          # MoE experts across GPUs (DeepEP)
--context_parallel_size=N         # long sequences (Ulysses attention)
--tensor_parallel_size=N          # attention weights (DTensor)
--expert_tensor_parallel_size=N   # expert-FFN sharding (MoE, experimental)
```

| Mode | Best for                                  |
|---|-------------------------------------------|
| EP | Distribute MoE experts across GPUs        |
| TP | Shard model weights across GPUs           |
| CP | Split long sequences across GPUs          |
| ETP | Shard individual expert FFNs              |
| EP+CP | MoE + long-context training               |
| EP+TP | MoE + weight sharding                     |
| EP+ETP | MoE + sharded experts                     |

Pass them directly to `torchrun`, or after `--` with `halo launch`:

```bash
halo launch sft config.yaml -n 8 -- --expert_parallel_size=8
```

Multi-node runs support shared (NFS/Lustre) and node-local filesystems.
See [Supported Models](human-docs/supported-matrix.md) for the per-model support matrix and
[Parallelism](agent-docs/parallelism/expert-parallelism.md) for implementation details.

</br>

## Optimizations

Enabled by default where supported:

- **Attention and kernels** — Flash Attention 4 on Blackwell, FA2/FA3 on Hopper, Liger fused
  CE/SwiGLU/RMSNorm, and Grouped GEMM for MoE expert compute.

- **Precision and optimizers** — `AdamWBF16` with stochastic rounding uses 6 bytes/parameter of
  optimizer state instead of 12 with FP32 master weights. Muon and FlashAdamW are also supported.
  FP8/FP4 MoE training is available through fake-quant QAT and DeepGEMM, with mxfp8/nvfp4 export.

- **Memory and PEFT** — padding-free, boundary-aware packing with `cu_seq_lens`, plus LoRA and QLoRA.
  QLoRA runs on a single 24 GB consumer GPU. Attention falls back to FA2/SDPA on older hardware.

- **Checkpoints** — large gathered checkpoints are automatically sharded, with tools for merging
  EP/TP shards and PEFT adapters back into a standard HuggingFace checkpoint.
  → [Checkpoints](human-docs/checkpoints.md)

- **Data** — offline tokenization, packing, and sharding, plus native `s3://` dataset streaming.

→ [Optimization docs](agent-docs/optimization/throughput-benchmarks.md) ·
[GPU Training Theory](agent-docs/reference/gpu-training-theory.md)

</br>

## Built with AI

We use GPT and Claude extensively while developing Halo, through
[PI-Agent](https://github.com/earendil-works/pi),
[Claude Code](https://github.com/anthropics/claude-code), and
[Codex](https://github.com/openai/codex). 

[`AGENTS.md`](AGENTS.md) contains the instructions and conventions used by both coding agents and
human contributors.

AI-assisted contributions are welcome. We just ask that contributors understand and stand behind
the code they submit. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

</br>

## Acknowledgments

Halo builds on a lot of great open-source work, in particular:

- **[Hugging Face](https://huggingface.co)** — Transformers, TRL, Accelerate, Datasets, and PEFT.
- **[Liger Kernel](https://github.com/linkedin/Liger-Kernel)** — fused cross-entropy, SwiGLU, and RMSNorm.
- **[FlashAdamW](https://github.com/databricks/flashoptim)** by Databricks AI Research — memory-efficient
  AdamW with quantized optimizer states and compact master weights.
- **[DeepEP](https://github.com/deepseek-ai/DeepEP)** and **[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)**
  by DeepSeek-AI — expert communication and grouped GEMM. Halo's aux-loss-free router balancing also
  follows the approach introduced with DeepSeek-V3.
- **[Flash Attention](https://github.com/Dao-AILab/flash-attention)** by Tri Dao and contributors —
  FA2, FA3, and FA4.
- **[vLLM](https://github.com/vllm-project/vllm)** — the rollout/serving engine behind online and
  environmental RL.
- **[PyTorch](https://pytorch.org)** — DTensor, FSDP2, and `torch.distributed`.
- **[SGLang](https://github.com/sgl-project/sglang)** — generation and NCCL weight synchronization for
  online and multi-turn RL.
- **[SDPG](https://arxiv.org/abs/2606.04036)**, led by researchers at UCLA — self-distilled policy
  gradient for online reinforcement learning.

DeepEP and Flash Attention are vendored as git submodules under [`vendors/`](vendors/) and retain
their upstream licenses; the build pins live in the [`Dockerfile`](Dockerfile).
See [`pyproject.toml`](pyproject.toml) for the full dependency set.

Halo is built and maintained by **[White Circle](https://whitecircle.com)**.

</br>

## License

Halo is released under the **Halo License** — the Apache License 2.0
([`APACHE-2.0.txt`](APACHE-2.0.txt)) as modified by the Supplemental Terms in [`LICENSE`](LICENSE).
Copyright © 2026 White Circle, PBC.

In plain terms: free to use, modify, self-host, and to train your own models and ship them in your
own products — at any scale. The one commercial condition is offering Halo's training functionality
to third parties as a service: that's free up to US \$20M in such revenue over any 12 months, and
above that requires a commercial agreement with White Circle. Models trained with Halo by anyone
over \$20M in annual revenue must carry a "Trained with Halo" credit.

Questions or commercial licensing: [hello@whitecircle.com](mailto:hello@whitecircle.com).
