# Installation

!!! warning "Run inside the Docker image — the host has no usable Python environment"
    The prebuilt images are the supported path for the compiled GPU dependencies (PyTorch 2.11+cu130, FlashAttention, DeepEP/NVSHMEM) — nothing on the host provides them. Anything that executes — training, tests, one-off Python — runs inside `halo:blackwell` (B200/B300) or `halo:hopper` (H100/H200). The image installs the project plus all deps into the system interpreter with `uv`, so `halo`/`python`/`torchrun`/`accelerate`/`pytest` are on `PATH` — no prefix. Building the same stack yourself against a matching CUDA 13.2 toolkit works but is unsupported; the [FlashAttention](#flash-attention) and [DeepEP](#deepep-required-for-expert-parallelism) sections below carry the pins it needs.

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.12 | `requires-python = ">=3.12,<3.13"` |
| GPU | Hopper (SM90) or Blackwell (SM100/103) | The two release targets; Ampere runs only via a best-effort source build with the FA2 fallback |
| CUDA | 13.2 | Image pinned to NGC `nvcr.io/nvidia/pytorch:26.03-py3`; local installs need a matching toolkit for source-built wheels |
| OS | Ubuntu 24.04 LTS | The NGC base |
| GPU driver | R580+ | CUDA 13.2-compatible |

## Install with uv

[uv](https://docs.astral.sh/uv/) is the package manager (PEP 621 `pyproject.toml` + `uv.lock`). `make install` shells out via `docker run`, so [build or pull the image](#docker-image) first; it installs an editable `-e .` plus the exported locked requirements (with the `gigatoken` and `flash-optimizers` extras) over the compiled torch/FA/DeepEP the image already ships.

```bash
pip install uv            # or: curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/whitecircle/halo.git
cd halo
make install
```

For a host `.venv` that powers IDE go-to-definition (`uv sync`), see [Development Environment](../contributing/development-environment.md). Optional: point `HF_HOME` at a **verified** large volume (`df -h` / `findmnt` — a `/mnt` path is not always a big array), `hf auth login` (gated models), `wandb login` (only for `report_to: wandb`).

## Docker image

| Image | ECR Public tag | Attention | GPUs |
|---|---|---|---|
| Blackwell | `blackwell` | FA2 + FA4 (CuTe DSL) + DeepEP | B200 (SM100) / B300 / GB200/GB300 (SM103) |
| Hopper | `hopper` | FA2 + FA3 + DeepEP | H100 / H200 (SM90) |

<!-- markdownlint-disable MD046 -- pymdownx tabbed content is indented by syntax -->

=== "Pull from ECR Public"

    Anonymous — no AWS account or login:

    ```bash
    docker pull public.ecr.aws/whitecircle/halo:blackwell   # or :hopper
    docker tag  public.ecr.aws/whitecircle/halo:blackwell halo:blackwell
    ```

    Each moving tag has immutable SemVer pins (`blackwell-1.0.0`); there is deliberately no `latest` — it would let a Hopper host silently pull a Blackwell image.

=== "Build locally"

    Credential-free — no token or secret needed. The `make` targets pass the `SOURCE_REVISION` build arg that busts the source-COPY cache:

    ```bash
    make build-blackwell     # or: make build-hopper
    ```

    `make push-blackwell ECR_ACCOUNT_ID=<yours>` pushes to your own registry under a moving and a SemVer tag.

<!-- markdownlint-enable MD046 -->

The full run invocation (`--gpus all`, `--ipc=host`, `--shm-size=128g`, `--env-file .env`, volume mounts) is in the [Docker Guide](../infrastructure/docker.md). Cluster provisioning: [SkyPilot](../infrastructure/skypilot.md) · [Nomad](../infrastructure/nomad.md).

## DeepEP (required for expert parallelism)

[DeepEP](https://github.com/deepseek-ai/DeepEP) provides the MoE all-to-all and is required for EP; dense training does not need it. NVSHMEM ships transitively as `nvidia-nvshmem-cu13` with PyTorch 2.11+cu130 — do **not** install `nvidia-nvshmem-cu12` on top, it clobbers the cu13 headers and breaks device-side linking.

A bare local build on CUDA 13.2 needs the `cccl` headers on `CPATH`, the pinned commit, `TORCH_CUDA_ARCH_LIST="9.0+PTX"`, and `nvidia-nccl-cu13` at `uv.lock`'s exact pin (`python docker/nccl_pin.py uv.lock`, at or above the DeepEP Gin floor — every image shares that one version, and a skew fails the RL weight-sync `ncclCommInitRank`); a naive `pip install .` fails. The image does this for you — for a manual build follow the [DeepEP Installation Guide](../infrastructure/deepep.md).

## Flash Attention

The backend is auto-detected by `_detect_attention_impl()` (`src/models/patches/attention.py`): `flash_attention_4` on Blackwell (SM100+), `flash_attention_3` on Hopper, `flash_attention_2` otherwise, each falling back to FA2 when the preferred kernel is absent. No `attn_implementation` config is needed.

The Blackwell image adds FA4 (`flash-attn-4[cu13]==4.0.0b16`, the `flash_attn.cute` submodule) on top of the NGC base's FA2, under one `flash_attn` namespace; transformers 5.7+ dispatches FA4 natively. FA4 is JIT-compiled (~10 s per distinct kernel on first use), so `ensure_fa4_kernel_cache_env()` turns on a persistent CuTe DSL cache under `$HF_HOME` — shared across models and ranks, and across nodes where `HF_HOME` sits on a shared filesystem. With `HF_HOME` unset it anchors on `TMPDIR` instead, which no other node can read.

Models with a head_dim-256 partial-rotary stack fall back from FA4 to SDPA, gated by `model_fa4_backward_nan_prone` on the `qwen3_5*`, `qwen3_next*`, and `glm4_moe_lite` model types (Qwen3.6 ships under the 3.5 types, so it is covered). GPT-OSS keeps FA4. Benchmarks and the per-model matrix: [Flash Attention](../optimization/flash-attention.md).

!!! note "Hopper + CUDA 13.2: split-K kernel stub"
    Under CUDA 13.2 `ptxas` hangs on 24 FA2 split-K kernel files targeting `sm_90`. The Hopper build replaces them with a single throw-on-call stub (`docker/training/flash_attn_split_stubs_hopper.cpp`). The varlen forward and the whole backward never reach them, so packed/padding-free training is unaffected; the *non-varlen* forward can select split-K by occupancy heuristic at small batch/head/seq, which is why CP prefers FA3 on Hopper. Mirror the patch if you build FA2 manually for Hopper.

Verify:

```bash
python -c "import flash_attn; print(flash_attn.__version__)"
python -c "import flash_attn_3; print('FA3 OK')"                          # Hopper
python -c "from flash_attn.cute import flash_attn_func; print('FA4 OK')"  # Blackwell
```

## Troubleshooting

- **PyTorch does not detect CUDA** — check `nvidia-smi` (driver) and `nvcc --version` (toolkit); the PyTorch build must match the CUDA version.
- **uv fails to resolve** — `uv lock --upgrade && make install`, or pin the interpreter with `uv python pin 3.12`.
- **DeepEP build fails** — never mix `nvidia-nvshmem-cu13` and `nvidia-nvshmem-cu12`. See the [DeepEP guide](../infrastructure/deepep.md).
- **Missing system libraries** — `apt install build-essential ninja-build zlib1g-dev libffi-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev liblzma-dev libncurses-dev tk-dev`.
