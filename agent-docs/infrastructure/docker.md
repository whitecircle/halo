# Docker Guide

The host has no usable Python — PyTorch, DeepEP, and Flash Attention live only inside the images. Anything
that executes runs inside an image.

- `halo:hopper` — H100/H200 (SM90): FA2 + FA3 + DeepEP.
- `halo:blackwell` — B200 (SM100) / B300, GB200/GB300 (SM103): FA2 + FA4 + DeepEP (no FA3).
- `vllm-server:0.26.0` — vLLM inference server with native NCCL weight transfer (separate container).
- `sglang-server:0.5.17` — SGLang inference server, NCCL matched to the training image so it can
  receive weight updates too ([Rollout Servers](rollout-servers.md)). Serving-only use needs no custom build.

All four are published to Amazon ECR Public under `public.ecr.aws/whitecircle/halo` — anonymous pulls,
no AWS account ([Registry](#registry)) — or build credential-free from source ([Building](#building)).

## Standard training launch

The `Makefile` wraps the `docker run` incantation once:

```bash
make train CONFIG=examples/sft/qwen3/qwen3-4b-ultrachat.yaml NPROC=8 EXTRA="--expert_parallel_size=8"
```

The Makefile's `DOCKER_RUN` is not identical to a hand-rolled launch: it runs in the foreground, adds
`--network host` (the vLLM e2e GPU tests reach the compose server at `localhost:8000`; bridge
networking would silently fail them) plus `-e PYTHONPATH=/workspace -e CUDA_DEVICE_MAX_CONNECTIONS=1`,
and makes `--env-file` and the `~/.aws` mount conditional on `ENV_FILE`/`AWS_DIR` so CI can run
creds-free. It also **omits `--cap-add=SYS_PTRACE`**, so py-spy cannot attach to a job started through
a `make` target.

The equivalent detached background job:

```bash
# Resolve the largest real (non-overlay/tmpfs) mount instead of assuming /mnt.
D=$(findmnt -rbno TARGET,AVAIL,FSTYPE | awk '$3!~/tmpfs|overlay|squashfs|nfs|fuse|autofs/ && $2+0>20e9{print $2,$1}' | sort -rn | head -1 | awk '{print $2}')
docker run -d --rm --name <job> --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=128g \
    --cap-add=SYS_PTRACE --env-file .env \
    -e HF_HOME="$D/hf" -e HF_DATASETS_CACHE="$D/hf/datasets" \
    -e TMPDIR="$D/tmp" -e HALO_DATA_ROOT="$D" \
    -v $(pwd):/workspace -v "$D:$D" -v ~/.aws:/root/.aws -w /workspace \
    halo:blackwell \
    bash -lc "torchrun --nproc_per_node=8 scripts/training/sft.py <config> > \"$D/<job>.log\" 2>&1"
```

Load-bearing flags:

- `--env-file .env` — supplies `WANDB_API_KEY` / `HF_TOKEN` / `AWS_DEFAULT_REGION`. The code does not
  auto-load `.env`; start from the repo-root template (`cp .env.example .env`), which the `make`
  targets and the dev container both expect to exist.
- `-e HF_HOME`/`HF_DATASETS_CACHE`/`TMPDIR` — redirect HuggingFace caches and temp off the small root FS.
- `-e HALO_DATA_ROOT` — toolkit scratch root: the S3 dataset cache resolves to `$HALO_DATA_ROOT/s3_datasets`
  and profiler artifacts to `$HALO_DATA_ROOT/profiling`. Defaults to `~/.cache/halo` when unset.
- `-v ~/.aws:/root/.aws` — host AWS config for S3.
- `--ipc=host --shm-size=128g` and the `ulimit` settings — required for NCCL and the DataLoader.
- `--cap-add=SYS_PTRACE` — lets py-spy attach for hang triage (`scripts/profiling/py_spy_diag.py`).

None of those four paths is a code default — each falls back to its own default when unset
(`~/.cache/huggingface`, `~/.cache/halo`, the system `/tmp`), so the redirect is a convention you
supply. The Makefile defaults them to `/mnt` via `HALO_SCRATCH ?= /mnt` — the same variable the
vLLM compose file and the devcontainer read, so exporting `HALO_SCRATCH` once points the whole
toolchain at the host's large volume. `/mnt` is
**not a guaranteed large volume**: on some hosts it shares the small root device. Confirm with
`findmnt` / `df -h` before any multi-GB write.

## Image matrix

| Component | Hopper | Blackwell |
|-----------|--------|-----------|
| Base NGC `nvcr.io/nvidia/pytorch:26.03-py3` (Ubuntu 24.04, Python 3.12, CUDA 13.2) | Yes | Yes |
| PyTorch 2.11.0+cu130 (reinstalled from the cu130 index over the base's build) | Yes | Yes |
| Transformers 5.16.x · TRL 1.6.x · Accelerate 1.11.x | Yes | Yes |
| Flash Attention 2 | Yes (built from tag `v2.8.3.post1`, split-K kernels are throw-stubs) | Yes — inherited from `BASE_IMAGE`, which is the pin; the build asserts it still imports |
| Flash Attention 3 (SM90) | Yes | No |
| Flash Attention 4 (`flash-attn-4[cu13]==4.0.0b16`, `flash_attn.cute`) | No | Yes |
| DeepEP V2 (commit `af9a040`, built `9.0+PTX`) | Yes | Yes |
| NCCL `nvidia-nccl-cu13` at `uv.lock`'s exact pin, shared with the vLLM/SGLang images — weight sync needs one runtime (asserted at build) | Yes | Yes |
| NVSHMEM (`nvidia-nvshmem-cu13`, transitive via torch) | Yes | Yes |
| `nvidia-cutlass-dsl` 4.5.2 + `quack-kernels` 0.5.0 | Yes | Yes |
| FlashAdamW (`flashoptim==0.1.4`) | Yes | Yes |
| DeepGEMM (commit `559d79f`, native fp8/fp4 grouped MoE GEMM behind `HALO_DEEPGEMM_NATIVE=1`; JIT-compiles at runtime like DeepEP V2) | No | Yes |
| Gigatoken tokenizer backend (`gigatoken` 0.9.x, `tokenizer_backend: gigatoken`) | Yes | Yes |
| uv 0.10.x, AWS CLI v2 (Claude Code via `INSTALL_CLAUDE_CODE=1`, off by default) | Yes | Yes |

`TARGET_GPU` writes the build's CUDA arch list to `/etc/cuda_arch` — `9.0` for Hopper, `10.0+PTX` for
Blackwell (one image serves SM100 and SM103) — which each source-build step exports as
`TORCH_CUDA_ARCH_LIST`. It is not an image `ENV`; only the login shell re-exports it (`.bashrc`), so a
`docker run ... python` sees it unset. The DeepEP build overrides it — see [DeepEP](deepep.md) for why
that arch list does not bound the kernels the toolkit actually runs.

`transformer_engine` is uninstalled at the end of the build.

On Blackwell, FA4 is the auto-selected default (`_detect_attention_impl`); FA2 stays available. See
[Flash Attention](../optimization/flash-attention.md) and [DeepEP](deepep.md).

Either training image is ~45–50 GB; a cold build takes tens of minutes, longest on Hopper (FA2 and FA3
build from source there).

## Building

The build needs no GPU (it cross-compiles via `TORCH_CUDA_ARCH_LIST`).

```bash
make build-hopper
make build-blackwell
make build-vllm
make build-sglang

# Equivalent raw build (SOURCE_REVISION is what `make` stamps; pass it yourself here)
docker build --build-arg SOURCE_REVISION=$(git rev-parse --short HEAD) -t halo:hopper .
docker build --build-arg SOURCE_REVISION=$(git rev-parse --short HEAD) \
  --build-arg TARGET_GPU=blackwell -t halo:blackwell .
```

- **Credential-free.** Every dependency is public — no build secret or token is required.
- **Built with uv, not Poetry.** Deps install into the system interpreter via
  `uv export --locked` into a requirements file, then `uv pip install --system --no-deps -r` it (hatchling
  backend, no venv). `--no-deps` is load-bearing: without it the resolver reinstalls torch, Flash Attention
  and DeepEP over the source builds earlier layers compiled.
- **`SOURCE_REVISION`** busts the source-COPY cache; `make build-*` stamps it with the short git SHA, so a
  code-only change reruns only the COPY + editable reinstall while every heavy dep layer stays cached. It
  guards against BuildKit occasionally serving a stale `src/` snapshot. The same value is stamped as
  `org.opencontainers.image.revision`, so
  `docker inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' halo:blackwell`
  reports which commit an image carries. Pass it explicitly on a raw `docker build`.
- **`BASE_IMAGE`** defaults to `nvcr.io/nvidia/pytorch:26.03-py3`.
- **License.** The image is labeled `org.opencontainers.image.licenses=LicenseRef-Halo` and carries
  `LICENSE` + `APACHE-2.0.txt` in `/workspace`.
- **Every upstream clone is pinned to a commit or tag.** DeepEP `af9a040`; Hopper's FA2
  `v2.8.3.post1` and its FA3 a `main` commit (`c46b8144`) — no FA release supports CUDA 13;
  `aws-ofi-nccl` and `gdrcopy` at the `AWS_OFI_NCCL_COMMIT` / `GDRCOPY_COMMIT` build args, so a rebuild
  reproduces the shipped images. Blackwell never builds FA2 — it inherits it from `BASE_IMAGE`, which is
  the pin, and the build asserts the base still delivers it rather than depending on it silently.
  `aws-ofi-nccl` is configured `--disable-tests` because its functional tests need `mpi.h`, which the
  image does not carry.
- **Hopper FA2 split-K stub.** CUDA 13.2 ptxas hangs on the 24 sm_90 split-K kernels, so the build swaps
  them for throw-stubs (`docker/training/flash_attn_split_stubs_hopper.cpp`) and sets
  `FLASH_ATTN_CUDA_ARCHS=90`. Split-K is unreachable from the varlen forward and the whole backward, but the
  **non-varlen** forward sets `num_splits=0` and can select it by occupancy heuristic at small
  batch/head/seq — so a non-varlen caller must prefer FA3 on sm_90, which
  `src/distributed/context_parallel/base_layer.py` does. Blackwell installs the prebuilt FA4 wheel and skips
  this.

Rebuild when `pyproject.toml`/`uv.lock` change (run `uv lock` first), or when Flash Attention / DeepEP / the
base NGC image need updating.

## Baked-in environment variables

| Variable | Value | Effect |
|----------|-------|--------|
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` | Latched by the driver at `cuInit` (DeepEP import time), so it must be in the environment from PID 1. Free default: neutral on dense and `ep_size=2`, +9.7% on `ep_size=8`. It does **not** make racy single-domain multi-group EP safe — `ParallelismConfig` rejects that shape. Override `-e CUDA_DEVICE_MAX_CONNECTIONS=8` for pure-dense FSDP all-gather/compute overlap. See [DeepEP](deepep.md). |
| `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` | `0` | The NGC base defaults fp32 matmuls to TF32, whose 10-bit mantissa collapses adjacent long-context RoPE positions past 2048. Forced off. |
| `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED` | `1` | Persist the FA4 CuTe DSL kernel cache (~10 s JIT per kernel on first use). |
| `CUTE_DSL_ENABLE_TVM_FFI` | `1` | TVM-FFI direct-invocation ABI for CuTe DSL kernels. |
| `EP_SUPPRESS_NCCL_CHECK` | `1` | Silences DeepEP V2's duplicate-NCCL-runtime check (the NGC base ships an HPC-X NCCL transport plugin alongside the pip `libnccl.so.2`). |
| `NCCL_IB_DISABLE=0`, `NCCL_NET_GDR_LEVEL=2`, `NCCL_IB_HCA=mlx5`, `NCCL_P2P_LEVEL=NVL`, `NCCL_DEBUG=WARN` | — | Mellanox IB/RoCE defaults; IB clusters work with no extra env. |
| `PYTHONPATH` | `/workspace` | Lets `python tests/.../foo.py` resolve the `tests` package. |
| `TOKENIZERS_PARALLELISM` | `false` | — |
| `HF_XET_HIGH_PERFORMANCE` | `1` | HF Xet high-throughput hub transfers (supersedes the deprecated `HF_HUB_ENABLE_HF_TRANSFER`). Already baked — SkyPilot configs need not re-set it. |
| `TRANSFORMERS_NO_ADVISORY_WARNINGS` | `1` | Silences Transformers' advisory warnings (e.g. the per-call sequence-length notice). |

`FLASH_ATTENTION_CUTE_DSL_CACHE_DIR` and `TRITON_CACHE_DIR` are not baked — the FA4 cache and the Triton
kernel/autotune cache both derive their directories from `HF_HOME` (or the temp dir) at runtime, so one
mounted volume carries every kernel cache across `--rm` containers. Triton matters more than it looks:
fla's autotuners persist measured configs there (`FLA_CACHE_RESULTS` defaults on), and some of their keys
are shape-derived, so an ephemeral cache re-benchmarks kernels per fresh sequence length on every run.

The runtime NCCL is the pip wheel `nvidia-nccl-cu13`, pinned in `uv.lock` — read the version there,
not from here; the NGC `NCCL_VERSION` env is the base-image system label and does not reflect it. All three images read the version out of `uv.lock` through one helper
(`docker/nccl_pin.py`), which also owns the DeepEP-V2 floor: the post-install guard runs
`nccl_pin.py --verify`, which fails the build if a later pip step re-resolved the wheel — or if the
`libnccl.so.2` a process resolves once torch has preloaded is not that wheel's, which is how the
base image's own older system copy would otherwise reach the weight-sync communicator unnoticed.

## RDMA networking (InfiniBand and EFA)

Both training images are EFA-ready as built. IB works on the baked defaults; **EFA is a per-job
opt-in** (its env vars degrade IB clusters, so they are not baked). The image matrix keys on GPU arch only —
no separate `-efa` tag.

- **InfiniBand / RoCE** (no extra env) — HPC-X `libnccl-net.so` (`/opt/hpcx`), loaded by NCCL by default.
- **AWS EFA** (opt-in) — libfabric (`/opt/amazon/efa`) + a GIN-capable `aws-ofi-nccl` built at the pinned
  commit (exporting `ncclGinPlugin_v13`; the NGC-bundled 1.17.3 exports no `ncclGin`) exposed as
  `libnccl-gin.so`, plus GDRCopy `libgdrapi`. Select it per job with the libfabric env block on the
  page linked below, and for DeepEP cross-node EP add `NCCL_GIN_TYPE=2` (proxy GIN; EFA has no IBGDA)
  and `--device /dev/gdrdrv` (host `gdrdrv` module).

Prerequisites and measured EFA ceilings: [DeepEP → EFA](deepep.md#expert-parallelism-over-aws-efa);
per-fabric launch env: [Multi-Node → RDMA fabrics](../parallelism/multi-node.md#rdma-fabrics).

## vLLM inference server

vLLM runs as a separate container built from upstream `vllm/vllm-openai:v0.26.0` plus toolkit patches
(`Dockerfile.vllm`; the unsuffixed upstream tag is the CUDA 13 build). It pins its own torch/transformers
stack, so the training environment uses a vendored NCCL weight-sync client (`src/distributed/nccl/`) instead
of importing vLLM.

The image reads `nvidia-nccl-cu13` out of `uv.lock` and installs that **exact** version, so the weight-sync
communicator sees an identical NCCL runtime on both ends. Build asserts check the installed version against
the lock, the resolved `libnccl.so.2`, and the `ncclUniqueId` ABI identity between the two stacks.

The image bakes the RL serving contract: native weight transfer (`VLLM_SERVER_DEV_MODE=1`; the
trainer's client drives the phased update protocol the server exposes), R3
routed-experts capture, and two `sitecustomize`-applied patches — layerwise reload and
weight-transfer re-init — whose targets are asserted against the live vLLM at build, so an upstream
refactor fails the image build. What the patches do, the
`--moe-backend triton` rule, serving flags, networking, GPU assignment, and troubleshooting:
[Rollout Servers](rollout-servers.md).

Which families this image can serve for RL, and why a family is refused, is stated once in
[Rollout Servers → Weight sync](rollout-servers.md#weight-sync). Image-side quirk: vLLM loads the
`ForConditionalGeneration` archs, so a text-only `qwen3_5_moe_text` checkpoint needs its
`config.json` patched to `qwen3_5_moe` first.

```bash
make build-vllm            # docker build -f Dockerfile.vllm -t vllm-server:0.26.0 .
```

Standalone `docker run` gotcha: the ENTRYPOINT is `vllm serve`, so any args REPLACE the compose
`CMD` wholesale — restate every default flag or the server silently loses NCCL weight transfer and
falls back to the expert-corrupting auto MoE backend. `VLLM_TOOL_PARSER` and friends are
compose-only interpolations; vLLM itself reads no such env.

The healthcheck calls `python3`, not `python` — the `vllm-openai` base ships no bare `python`, and an
exit-127 healthcheck marks the service permanently unhealthy, wedging the training service's
`service_healthy` gate.

**GPT-OSS.** The image applies `docker/vllm/patches/patch_vllm_disable_gptoss.sh` at build time,
disabling harmony mode; with harmony off the model emits plain-text tool calls only the two baked
plugins parse (`/opt/gpt_oss_text_tool_parser.py`; `/opt/gpt_oss_reasoning_parser.py` restores
`thinking_token_budget`). Serve flags: [Rollout Servers](rollout-servers.md#vllm); model page:
[GPT-OSS](../models/gpt-oss.md).

## Registry

Prebuilt images live on Amazon ECR Public — anonymous pulls, no AWS account:

```bash
docker pull public.ecr.aws/whitecircle/halo:blackwell     # or :hopper
docker pull public.ecr.aws/whitecircle/halo:vllm-0.26.0
docker pull public.ecr.aws/whitecircle/halo:sglang-0.5.17
```

Each moving tag has immutable SemVer pins (`blackwell-1.0.0`); there is deliberately no `latest` — it
would let a Hopper host silently pull a Blackwell image. Roll back by pulling a pinned tag and
retagging locally:

```bash
docker pull public.ecr.aws/whitecircle/halo:hopper-1.0.0
docker tag  public.ecr.aws/whitecircle/halo:hopper-1.0.0 halo:hopper
```

Maintainers publish with `make push-public-all` (`ecr-public-login` + the four `push-public-*`
targets). To host the images in your own registry, `docker tag` and `docker push` the locally built
images wherever you like — the build is credential-free, so nothing sensitive rides along.

## Verifying and debugging

```bash
docker run --gpus all --rm halo:hopper bash -c "
    python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\" && \
    python -c \"import transformers, trl, accelerate; print(transformers.__version__, trl.__version__)\" && \
    python -c \"import flash_attn; print('FA2', flash_attn.__version__)\" && \
    python -c \"from flash_attn_interface import flash_attn_func; print('FA3 OK')\" && \
    python -c \"import deep_ep; print('DeepEP OK')\"
"

docker exec -it <container_id> bash        # shell into a running container
docker logs <container_id>                 # view job logs
docker system prune -a                     # free disk on "no space left"
```

The FA3 line is Hopper-only. DeepEP imports only with `--gpus` (it needs `libcuda.so.1`) — a runtime DeepEP
import error usually means the container ran without `--gpus all`.

Containers have no AWS credentials by default; mount the host config (`-v ~/.aws:/root/.aws`) for S3. If
S3 fails with `SSOTokenLoadError`, re-run `aws sso login` on the host. See [AWS Auth](aws-auth.md).

[Claude Code](https://code.claude.com) is not installed by default: pass
`--build-arg INSTALL_CLAUDE_CODE=1` to install it at `/root/.local/bin/claude` (best-effort, non-fatal).
Repo-aware skills live under `skills/` (symlinked from `.claude/skills` and `.agents/skills`) and ship
inside the image (`skills/` is copied in and the symlinks recreated) and inside any container that
mounts the repo at `/workspace`.

## Cloud and dev setup

- **RunPod** — override the CMD with `sleep infinity` (the default CMD prints a banner and exits, which
  stops a RunPod pod). See [RunPod](runpod.md).
- **SkyPilot** — see [SkyPilot Deployment](skypilot.md).
- **VS Code / dev container / host `.venv`** — see
  [Development Environment](../contributing/development-environment.md).
