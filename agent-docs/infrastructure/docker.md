# Docker Guide

The host has no usable Python: PyTorch, DeepEP and Flash Attention live only inside the images, so
anything that executes runs inside one.

- `halo:hopper` — H100/H200 (SM90): FA2 + FA3 + DeepEP.
- `halo:blackwell` — B200 (SM100) / B300 (SM103): FA2 + FA4 + DeepEP (no FA3).
- `halo:vllm-0.32.0` — the vLLM rollout server (separate container).
- `halo:sglang-0.5.22` — the SGLang rollout server (separate container).

All four are published to Amazon ECR Public under `public.ecr.aws/whitecircle/halo` with the same tags
(anonymous pulls, no AWS account; [Registry](#registry)), or build credential-free from source
([Building](#building)) under these local names. The images are x86_64-only: Grace-based GB200/GB300
hosts (aarch64) need an arm64 build the Dockerfile does not provide.

The host needs an NVIDIA driver R580 or newer, the CUDA 13 floor the images' CUDA 13.2 runs on (the GPU
tiers run on 580.159.03), and the NVIDIA Container Toolkit.

## Standard training launch

The `Makefile` wraps the `docker run` once; `train` runs `halo launch` with `METHOD` (default `sft`):

```bash
make train CONFIG=examples/sft/gptoss/gptoss-20b-multinode-ep.yaml NPROC=8 EXTRA="--expert_parallel_size=8 --use_grouped_gemm=true"
```

The Makefile's `DOCKER_RUN` runs in the foreground and adds `--network host` (the server GPU tiers
reach the compose server at `localhost:8000`) plus `-e PYTHONPATH=/workspace -e
CUDA_DEVICE_MAX_CONNECTIONS=1`. It mounts `--env-file` and `~/.aws` only while `ENV_FILE` / `AWS_DIR`
are non-empty, so CI runs creds-free. It omits `--cap-add=SYS_PTRACE`, so py-spy attaches to a `make`
job only where `kernel.yama.ptrace_scope` is 0
([Debugging](../reference/debugging.md#1c-py-spy--cpu-flame-graph-dataloader--python-stalls)).

The detached equivalent:

```bash
# The local mount other than / (not tmpfs, overlay or a network FS) with the most free space over 20 GB.
D=$(findmnt -rbno TARGET,AVAIL,FSTYPE | awk '$1!="/" && $3!~/tmpfs|overlay|squashfs|nfs|fuse|autofs/ && $2+0>20e9{print $2,$1}' | sort -rn | head -1 | awk '{print $2}')
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

- `--env-file .env` supplies `WANDB_API_KEY` / `HF_TOKEN` / `AWS_DEFAULT_REGION`. Nothing auto-loads
  `.env`; start from the template (`cp .env.example .env`), which the GPU `make` targets and the dev
  container require.
- `HF_HOME` / `HF_DATASETS_CACHE` / `TMPDIR` / `HALO_DATA_ROOT` move the caches, temp files and the
  toolkit scratch root (S3 dataset cache under `s3_datasets/`, profiler artifacts under `profiling/`)
  off the small root FS. None is a code default: unset, they fall back to `~/.cache/huggingface`, the
  system `/tmp` and `~/.cache/halo`. Keep `TMPDIR` a short path: a multiprocessing socket under it
  caps at 108 bytes ([Troubleshooting](../reference/troubleshooting.md#symptom--cause--fix)).
- `--ipc=host`, `--shm-size=128g` and the two `ulimit`s are required by NCCL and the DataLoader.
- `--cap-add=SYS_PTRACE` lets py-spy attach for hang triage (`halo run py-spy-diag`).

`HALO_SCRATCH` is the one knob for that volume: the Makefile derives the mount and the four variables
from it, the dev container requires it, and `docker-compose.vllm.yml` mounts it. Unset, the Makefile
resolves it with the recipe above and stops a target that needs it when no mount qualifies; export it
on a host with more than one large volume (the recipe picks by free space) and on a CI runner whose
seeded HF cache must stay put. The compose file has no fallback
([Rollout server images](#rollout-server-images)). Confirm the volume with `findmnt` / `df -h` before
any multi-GB write.

On a host whose docker default runtime rejects `--gpus` (sysbox-runc, for one), set
`DOCKER_RUNTIME=nvidia` on any `make` target.

A CPU-only job (CPU tests, a one-off script) drops `--gpus all`, the ulimits and the shm size, as the
Makefile's `DOCKER_RUN_CPU` does. Keep both mounts: without `-v "$D:$D"` the log dies with the `--rm`
container, and the CPU tests that call `from_pretrained` directly fail without an HF cache once the Hub
is unreachable.

```bash
docker run -d --rm --name <job> \
    -e PYTHONPATH=/workspace -e HF_HOME="$D/hf" \
    -v $(pwd):/workspace -v "$D:$D" -w /workspace \
    halo:blackwell \
    bash -lc "pytest -m cpu tests/cpu > \"$D/<job>.log\" 2>&1"
```

## Image matrix

| Component | Hopper | Blackwell |
|-----------|--------|-----------|
| Base NGC `nvcr.io/nvidia/pytorch:26.03-py3` (Ubuntu 24.04, Python 3.12, CUDA 13.2) | Yes | Yes |
| PyTorch 2.11.0+cu130 (reinstalled from the cu130 index over the base's build) | Yes | Yes |
| Transformers 5.20.2 · TRL 1.14.2 · Accelerate 1.11.0 · PEFT 0.18.1 · sentence-transformers 6.1.0 · Liger 0.8.4 (`uv.lock`) | Yes | Yes |
| Flash Attention 2 | Built from tag `v2.8.3.post1`, split-K kernels stubbed | Inherited from `BASE_IMAGE`, which is the pin; the build asserts it imports |
| Flash Attention 3 (SM90) | Yes | No |
| Flash Attention 4 (`flash-attn-4[cu13]==4.0.0b37`, `flash_attn.cute`) | No | Yes |
| DeepEP V2 (commit `af9a040`, built `9.0+PTX`) | Yes | Yes |
| NCCL `nvidia-nccl-cu13` at `uv.lock`'s exact pin, shared with the rollout images (asserted at build) | Yes | Yes |
| EFA userspace: rdma-core, AWS libfabric and one pinned `aws-ofi-nccl` build (`docker/efa/install_efa_userspace.sh`), shared with the rollout images | Yes | Yes |
| cuBLAS `nvidia-cublas` 13.8.0.4 (CUDA 13.4.3), overriding torch's locked 13.1.0.3; asserted at build | Yes | Yes |
| NVSHMEM (`nvidia-nvshmem-cu13`, transitive via torch) | Yes | Yes |
| `nvidia-cutlass-dsl` 4.6.2 (CUDA 13 runtime flavor only) + `quack-kernels` 0.6.4 | Yes | Yes |
| `flash-linear-attention` 0.4.2 + `causal-conv1d` 1.6.2.post1, built at `uv.lock`'s pins (FLA's Blackwell state-kernel restriction: [Qwen3.5](../models/qwen3_5.md#linear-attention-qwen3_5moegateddeltanet)) | Yes | Yes |
| FlashAdamW (`flashoptim==0.1.4`) | Yes | Yes |
| DeepGEMM (commit `559d79f`, native fp8/fp4 grouped MoE GEMM behind `HALO_DEEPGEMM_NATIVE=1`; JIT-compiles at runtime) | No | Yes |
| Gigatoken tokenizer backend (`gigatoken` 0.9.x, `tokenizer_backend: gigatoken`) | Yes | Yes |
| uv 0.10.x, AWS CLI v2, bubblewrap (Claude Code via `INSTALL_CLAUDE_CODE=1`, off by default) | Yes | Yes |

`TARGET_GPU` writes the build's CUDA arch list to `/etc/cuda_arch`: `9.0` for Hopper, `10.0+PTX` for
Blackwell (one image serves SM100 and SM103). Each source build exports it as `TORCH_CUDA_ARCH_LIST`;
it is not an image `ENV`, and only the login shell re-exports it (`.bashrc`), so a bare
`docker run ... python` sees it unset. DeepEP overrides it ([DeepEP](deepep.md) explains why that list
does not bound the kernels the toolkit runs). `transformer_engine` is uninstalled.

On Blackwell FA4 is the auto-selected attention and FA2 stays available
([Flash Attention](../optimization/flash-attention.md)). Either training image is ~30–32 GB; a cold
build takes tens of minutes, longest on Hopper, which builds FA2 and FA3 from source.

## Building

The build needs no GPU (it cross-compiles via `TORCH_CUDA_ARCH_LIST`) and no credentials.

```bash
make build-hopper
make build-blackwell
make build-vllm      # halo:vllm-0.32.0
make build-sglang    # halo:sglang-0.5.22

# Raw equivalent; pass the stamps `make` adds yourself
docker build --build-arg SOURCE_REVISION=$(git rev-parse --short HEAD) \
  --build-arg BUILD_DATE=$(date -u +%Y-%m-%d) --build-arg TARGET_GPU=blackwell -t halo:blackwell .
```

A bare `docker build` targets Hopper (`ARG TARGET_GPU=hopper`). The server tags carry the engine
release read off `Dockerfile.vllm` / `Dockerfile.sglang` (`ARG VLLM_VERSION`, `ARG SGLANG_VERSION`).

- **uv, not Poetry.** Dependencies install into the system interpreter: `uv export --locked` (the dev
  group plus the `gigatoken` and `flash-optimizers` extras) into a requirements file, then
  `uv pip install --system --no-deps -r` it. `--no-deps` is load-bearing: without it the resolver
  reinstalls torch, Flash Attention and DeepEP over the source builds. The source-built pre-installs
  (`causal-conv1d`, `flash-linear-attention`) take it too, so their dependencies come from the lock.
  `make check-lock` replays this install in a throwaway container and fails on a stale lock.
- **`SOURCE_REVISION`** busts the source-COPY cache: `make build-*` stamps the short git SHA, so a
  code-only change reruns only the COPY and the editable reinstall. All four images carry it as
  `org.opencontainers.image.revision`, with `VERSION` and `BUILD_DATE` as `.version` and `.created`
  (a raw build without them labels `dev` and an empty date), so
  `docker inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' halo:blackwell`
  names the commit an image carries.
- **Every upstream clone is pinned** to a commit or tag: DeepEP `af9a040`; Hopper's FA2 `v2.8.3.post1`
  and FA3 at the `main` commit `c46b8144` (no FA release supports CUDA 13); `aws-ofi-nccl` at the commit
  the EFA script pins, configured `--disable-tests` (its tests need `mpi.h`); `gdrcopy` at
  `GDRCOPY_COMMIT`. Blackwell's FA2 comes from `BASE_IMAGE` (default
  `nvcr.io/nvidia/pytorch:26.03-py3`).
- **Hopper FA2 split-K stub.** CUDA 13.2 ptxas hangs on the 24 sm_90 split-K kernels, so the build
  swaps them for throw-stubs (`docker/training/flash_attn_split_stubs_hopper.cpp`) and pins the dense
  and varlen forwards to one split (`num_splits` 0 → 1 in `flash_api.cpp`, asserted in the build): their
  occupancy heuristic would otherwise pick split-K for a small unpadded batch and for decode under GQA.
  The backward never uses split-K; only `flash_attn_with_kvcache`, which nothing in Halo calls, still
  can reach a stub.
- **No `flash_attn.cute` on Hopper.** FA2's source build also installs an old copy of FA4's CuTe DSL
  sources, which does not import against the pinned cutlass-dsl; the Hopper build removes it, so every
  FA4 probe reads FA4 as not installed there.
- **Known `pip check` gaps.** The cutlass-dsl metapackage requires its CUDA 12 runtime flavor, which
  neither image installs (both run CUDA 13), and neither carries the `nvidia-cuda-nvdisasm` wheel it
  declares: the pinned set is installed without its dependencies, and cutlass-dsl runs `nvdisasm` only
  to keep SASS (`CUTE_DSL_KEEP=sass`).
- **Build-time checks.** `docker/training/verify_image.py` imports the stack, the target GPU's
  attention kernels, the cutlass-dsl the locked install re-pins and the extras; the build then runs
  `halo --help` and checks the bubblewrap jail's subordinate id maps. DeepEP needs `libcuda.so.1`, so
  it is checked at runtime ([Verifying](#verifying-and-debugging)).
- **License.** The images are labeled `LicenseRef-Halo` and carry `LICENSE` + `APACHE-2.0.txt`.

Rebuild after a `pyproject.toml` / `uv.lock` change (run `uv lock` first) or to move Flash Attention,
DeepEP or the base image.

## Baked-in environment variables

| Variable | Value | Effect |
|----------|-------|--------|
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` | Latched by the driver at `cuInit` (DeepEP import), so it must be set from PID 1. The setting EP is validated with ([DeepEP](deepep.md#environment-variables)); it does not make racy single-domain multi-group EP safe, which `ParallelismConfig` rejects. |
| `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` | `0` | The NGC base defaults fp32 matmuls to TF32, whose 10-bit mantissa collapses adjacent RoPE positions past 2048 in remote code that computes RoPE with a matmul. |
| `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED` | `1` | Persist the FA4 CuTe DSL kernel cache (~10 s JIT per kernel on first use). |
| `CUTE_DSL_ENABLE_TVM_FFI` | `1` | TVM-FFI direct-invocation ABI for CuTe DSL kernels. Liger's CuTe DSL RoPE returns wrong results under it, so the toolkit pins Liger's RoPE to Triton ([Liger → Kernel implementation](../optimization/liger-kernels.md#kernel-implementation)). |
| `EP_SUPPRESS_NCCL_CHECK` | `1` | Silences DeepEP's duplicate-NCCL-runtime check, which flags the NGC base's HPC-X NCCL transport plugin beside the pip `libnccl.so.2`. |
| `NCCL_IB_DISABLE=0`, `NCCL_NET_GDR_LEVEL=2`, `NCCL_IB_HCA=mlx5`, `NCCL_P2P_LEVEL=NVL`, `NCCL_DEBUG=WARN` | — | Mellanox IB/RoCE defaults; IB clusters need no extra env. |
| `PYTHONPATH` | `/workspace` | Lets `python tests/.../foo.py` resolve the `tests` package. |
| `TOKENIZERS_PARALLELISM` | `false` | — |
| `HF_XET_HIGH_PERFORMANCE` | `1` | High-throughput Hub transfers (supersedes `HF_HUB_ENABLE_HF_TRANSFER`). |
| `TRANSFORMERS_NO_ADVISORY_WARNINGS` | `1` | Silences transformers' advisory warnings. |

`FLASH_ATTENTION_CUTE_DSL_CACHE_DIR` and `TRITON_CACHE_DIR` are not baked: the toolkit points each at
`$HF_HOME/<subdir>` (else the temp dir) when unset, so one mounted volume keeps every kernel cache
across `--rm` containers. An unwritable location falls back with a warning; set the variables to keep
the caches elsewhere. The Triton cache matters: fla's autotuners persist measured configs there
(`FLA_CACHE_RESULTS` defaults on), and some keys are shape-derived, so an ephemeral cache re-benchmarks
per sequence length every run. Fresh caches can pick differently, and a different pick changes bf16 rounding,
which routing and index selection can amplify: across separate cold caches tiny GLM-5 Next's logits moved up to
12.4%, while Qwen3.5-35B-A3B, Qwen3.6-35B-A3B and GLM-5.3-Flash cut to six layers stayed bit-identical. On one cold
cache shared by two runs, tiny Qwen3-Next's gradients moved up to 6.1e-5, Qwen3.5 MoE's 2.4e-7 and
Qwen3.8-Flash-Next's 6e-8, and the 13 other tiny families stayed bit-identical. Runs that must match bit for bit share
an already warm cache, and no other run fills it meanwhile.

The FA4 cache serializes each kernel file's writer with `flock`. Where `flock` fails (Lustre without
`flock`), the first use of each kernel raises after a 15 s lock timeout, mid-run included; where locks
are node-local (NFS `nolock` or `local_lock=flock`, Lustre `localflock`), one node can load a file
another is still writing. On either mount, set `FLASH_ATTENTION_CUTE_DSL_CACHE_DIR` to node-local
storage.

The runtime NCCL is the pip wheel `nvidia-nccl-cu13` at the exact `uv.lock` pin; the NGC
`NCCL_VERSION` env is the base image's label and does not reflect it. All three Dockerfiles read the
pin through `docker/nccl_pin.py`, which also owns the DeepEP V2 floor, and its `--verify` fails the
build when a later step re-resolved the wheel or when the `libnccl.so.2` a process resolves is not
that wheel's (the base's older system copy would otherwise reach the weight-sync communicator).

The runtime cuBLAS is the pip wheel `nvidia-cublas`, pinned by an `override-dependencies` entry in
`pyproject.toml` because torch's `cuda-toolkit[cublas]` extra locks 13.1.0.3, which drops the trailing
rows of some bf16 matmuls ([pytorch/pytorch#203481](https://github.com/pytorch/pytorch/issues/203481)).
13.8.0.4 passes the reproducer cases; 13.7.0.27 was reported to still fail. The wheel's `nvidia-cuda-nvrtc`
dependency is unpinned and stays at torch's lock. `docker/training/verify_image.py` fails the build when
the library torch loads is not that wheel. The pin reaches the two training images only: the vLLM and SGLang
rollout images keep the cuBLAS of their own bases and neither pin nor check it.

## RDMA networking (InfiniBand and EFA)

Every image is EFA-ready as built. `docker/efa/install_efa_userspace.sh` installs the EFA installer
1.46.0's rdma-core (60) and AWS libfabric 2.3.1amzn4.0 (`/opt/amazon/efa`), replacing the NGC base's
MOFED rdma-core in the training images and the distro rdma-core of the server bases, and builds
`aws-ofi-nccl` at one pinned commit against the `uv.lock` NCCL. The plugin is GIN-capable and exposed
as `libnccl-gin.so`; GDRCopy's `libgdrapi` ships beside it in the training images. Both ends of a
weight-sync group must run this one build, rdma-core included
([Rollout Servers → Servers on other nodes](rollout-servers.md#servers-on-other-nodes-efa)).

InfiniBand and RoCE run on the [baked defaults](#baked-in-environment-variables); EFA is selected per
job, so there is no separate `-efa` tag. The launch env per fabric:
[Multi-Node → RDMA fabrics](../parallelism/multi-node.md#rdma-fabrics).

## Rollout server images

`halo:vllm-0.32.0` is upstream `vllm/vllm-openai:v0.32.0` and `halo:sglang-0.5.22` upstream
`lmsysorg/sglang:v0.5.22`, each with the `uv.lock` NCCL, the EFA userspace and the weight-sync patches
asserted at build. Each pins its own torch and transformers, so the trainer never imports either
engine. Their compose files require one host path each, with no fallback: `HALO_SCRATCH` for
`docker-compose.vllm.yml` (even to start the server alone) and `SGLANG_MODEL_DIR` for
`docker-compose.sglang.yml`; set them in the shell or the repo-root `.env`. Build gates, patches,
serving flags and the families each serves: [Rollout Servers](rollout-servers.md).

## Registry

Prebuilt images live on Amazon ECR Public:

```bash
docker pull public.ecr.aws/whitecircle/halo:blackwell     # or :hopper
docker pull public.ecr.aws/whitecircle/halo:vllm-0.32.0
docker pull public.ecr.aws/whitecircle/halo:sglang-0.5.22
```

Each moving tag has an immutable release pin: `blackwell-1.3.0rc1`, `hopper-1.3.0rc1`, `vllm-0.32.0-1.3.0rc1`,
`sglang-0.5.22-1.3.0rc1`. There is deliberately no `latest`, which would let a Hopper host pull a
Blackwell image. Pair a trainer with the server images of the same release: the vLLM image unpickles
the trainer's `ncclUniqueId` through a shim at the trainer's module path. Pin a release by pulling its
tag and retagging it under the local name the `make` targets use:

```bash
docker pull public.ecr.aws/whitecircle/halo:hopper-1.3.0rc1
docker tag  public.ecr.aws/whitecircle/halo:hopper-1.3.0rc1 halo:hopper
```

The compose files take the server images from `VLLM_IMAGE` / `SGLANG_IMAGE` (default: the local
`halo:vllm-0.32.0` / `halo:sglang-0.5.22`) and build nothing, so a pulled server image needs no
retag: `export VLLM_IMAGE=public.ecr.aws/whitecircle/halo:vllm-0.32.0`.

Maintainers publish with `make push-public-all` (`ecr-public-login` plus the four `push-public-*`
targets). To host the images elsewhere, `docker tag` and `docker push` the local builds; nothing
sensitive rides along. `tests/cpu/config/test_release_version_in_sync.py` holds every statement of the
release to `pyproject.toml`'s `version`: the `uv.lock` entry, the `Makefile` `VERSION`, each Dockerfile's
`ARG VERSION`, `CITATION.cff`, the README's newest release entry and every documented image pin.

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

The FA3 line is Hopper-only. DeepEP imports only with `--gpus`: a runtime DeepEP import error usually
means the container ran without `--gpus all`. Containers carry no AWS credentials; mount
`-v ~/.aws:/root/.aws` for S3 and re-run `aws sso login` on the host after an `SSOTokenLoadError`
([AWS Auth](aws-auth.md)).

[Claude Code](https://code.claude.com) installs only with `--build-arg INSTALL_CLAUDE_CODE=1`, at
`/root/.local/bin/claude` (best-effort, non-fatal). The skills (`skills/`, symlinked from
`.claude/skills` and `.agents/skills`) are copied into `/workspace` at build; a checkout mounted at
`/workspace` replaces them with its own.

## Cloud and dev setup

- **RunPod** — override the CMD with `sleep infinity`: the default CMD prints a banner and exits, which
  stops a pod ([RunPod](runpod.md)).
- **SkyPilot / Nomad** — [SkyPilot](skypilot.md), [Nomad](nomad.md).
- **VS Code, dev container, host `.venv`** — [Development Environment](../contributing/development-environment.md).
