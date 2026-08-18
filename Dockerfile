# Halo training image — NGC PyTorch base (nvcr.io/nvidia/pytorch:26.03-py3: Ubuntu 24.04,
# Python 3.12, CUDA 13.2) + stable PyTorch 2.11.0+cu130, the uv.lock dependency closure
# (transformers 5.16.1, TRL 1.6), Flash Attention (FA2+FA3 on Hopper, FA4 on Blackwell), DeepEP.
# vLLM is NOT installed here: the two stacks pin torch/transformers sets that cannot share one
# interpreter, so vLLM runs as a separate container (Dockerfile.vllm) the trainer reaches over the
# vendored NCCL weight-sync client (src/distributed/nccl/). See agent-docs/infrastructure/docker.md.
#
# TARGET_GPU build arg:
#   - hopper (default): H100/H200, SM90, FA2+FA3, DeepEP
#   - blackwell: B200 (SM100) / B300 / GB200/GB300 (SM103), FA4 (on top of base-image FA2) +
#     cutlass-dsl + DeepEP. One image covers SM100 and SM103 via `+PTX` (JIT) — see /etc/cuda_arch.
#     cutlass-dsl backs Muon's quack Newton-Schulz kernels (runtime-probed, pure-torch fallback);
#     native block-scaled MoE GEMM is DeepGEMM (SM100, opt-in HALO_DEEPGEMM_NATIVE=1).
#
# SkyPilot-compatible: Debian-based, passwordless sudo, apt.
#
# Build:  make build-hopper / make build-blackwell (pass TARGET_GPU, SOURCE_REVISION)
# Run:    docker run --gpus all -it halo:hopper

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.03-py3
FROM ${BASE_IMAGE}

# Static identity labels only — the version/created labels live below, after the heavy layers,
# so a release bump never invalidates the dep-compile cache.
LABEL maintainer="White Circle <hello@whitecircle.ai>"
LABEL org.opencontainers.image.title="Halo"
LABEL org.opencontainers.image.description="Halo — HuggingFace-native LLM/VLM training and alignment toolkit (EP/CP/TP/ETP, MoE, FA4, DeepEP)"
LABEL org.opencontainers.image.vendor="White Circle"
LABEL org.opencontainers.image.source="https://github.com/whitecircle/halo"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Build-only knobs — ARG, not ENV: BuildKit exports them to every RUN in this stage, and nothing
# at runtime reads them (a persisted DEBIAN_FRONTEND/PIP_* would only leak into training subprocesses).
ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_NO_CACHE_DIR=1
ARG PIP_DISABLE_PIP_VERSION_CHECK=1

ENV TZ=UTC
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# uv configuration. We install the locked dependency set into the SYSTEM interpreter's
# site-packages (NGC's Debian layout puts packages in /usr/local/.../dist-packages, where
# the pre-built compiled torch/FA/DeepEP live) via `uv pip install --system` from an exported
# requirements file — NOT `uv sync`, which is venv-oriented and rejects the split /usr vs
# /usr/local prefix (no /usr/local/bin/python). --break-system-packages opts past Debian's
# PEP 668 externally-managed guard. Already-satisfied pins (torch, the nvidia/triton CUDA
# stack, causal-conv1d) are detected and skipped, so the compiled ABI is preserved.
ARG UV_VERSION=0.10.12
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/tmp/uv_cache

# HF Xet high-throughput transfer (supersedes the deprecated HF_HUB_ENABLE_HF_TRANSFER).
ENV HF_XET_HIGH_PERFORMANCE=1
ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Persist the FlashAttention-4 (flash_attn.cute) CuTe DSL kernel cache. FA4 JIT-compiles
# its kernels on first use (~10s each); the library's on-disk AOT cache is OFF by default,
# so without this every process/rank/launch recompiles from scratch. The code anchors the
# cache dir on HF_HOME (a mounted volume) at runtime; this env just turns the cache on for
# any process — incl. ad-hoc python — that doesn't go through setup_training_environment.
ENV FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
# CuTe DSL TVM-FFI fast-invocation ABI for ALL CuTe DSL kernels (FA4 + block-scaled mxfp8/nvfp4
# GEMMs). The default DSL host-launch shim is slow; TVM-FFI's direct-invocation path cut the grouped
# block-scaled MoE GEMM's per-call host overhead ~30% on B300 (bit-identical). Read at DSL import time,
# so set here for any process that doesn't go through ensure_fa4_kernel_cache_env() first.
ENV CUTE_DSL_ENABLE_TVM_FFI=1

# NGC sets TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 → fp32 matmuls run in TF32, whose 10-bit mantissa
# collapses adjacent RoPE token positions past 2048 and corrupts long context on every model. Force
# true fp32 (bf16 matmuls unaffected); the toolkit also enforces it in-process.
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

# NCCL optimizations for multi-GPU and InfiniBand
ENV NCCL_DEBUG=WARN
ENV NCCL_IB_DISABLE=0
ENV NCCL_NET_GDR_LEVEL=2
ENV NCCL_IB_HCA=mlx5
ENV NCCL_P2P_LEVEL=NVL

# RDMA fabric: the NCCL_* above target Mellanox InfiniBand/RoCE (RunPod, Nebius) — the
# default fabric, for which NCCL auto-loads the NGC HPC-X libnccl-net.so plugin. AWS EFA is
# also supported WITHOUT adding packages: the NGC base bundles libfabric (/opt/amazon/efa)
# and aws-ofi-nccl (/opt/amazon/aws-ofi-nccl, in the ldconfig cache). EFA is opt-in PER JOB
# (NCCL_NET_PLUGIN=ofi + FI_PROVIDER=efa + FI_EFA_USE_DEVICE_RDMA=1 + NCCL_PROTO=simple) —
# NOT baked here, because NCCL_PROTO=simple / the OFI plugin would degrade IB clusters.
# DeepEP V2 EP over EFA additionally needs proxy GIN (NCCL_GIN_TYPE=2; no IBGDA on EFA) and
# may need a newer aws-ofi-nccl. See agent-docs/parallelism/multi-node.md#rdma-fabrics + agent-docs/infrastructure/deepep.md.

# Serialize device work onto a single hardware queue — the standard Megatron-LM setting for
# training alongside custom communication kernels, and free as a default here: neutral on dense
# and ep_size=2, +9.7% on ep_size=8 (8xB300, gpt-oss-20b, seq 4096, GC on). The CUDA driver
# latches this at driver init (cuInit), which the DeepEP/NVSHMEM C-extensions trigger at *import*
# time — before any training script reaches init_distributed — so it must be in the process
# environment from PID 1 (a late os.environ write inside Python is a no-op). It does NOT make
# single-node multi-group >2-rank pure EP safe: `ParallelismConfig` rejects that shape at config
# time (agent-docs/parallelism/expert-parallelism.md). Pure-dense runs wanting maximal FSDP
# all-gather/compute overlap can override at launch: `docker run -e CUDA_DEVICE_MAX_CONNECTIONS=8 ...`.
ENV CUDA_DEVICE_MAX_CONNECTIONS=1

# DeepEP V2's duplicate-NCCL-runtime guard flags the NGC image's HPC-X `libnccl-net.so`
# (RDMA/SHARP transport *plugin*, ~900KB) as a second NCCL runtime alongside the pip
# `libnccl.so.2` (~250MB). The plugin is complementary, not a conflicting runtime, so the
# check is suppressed. DeepEP latches it inside `check_nccl_so()` at `import deep_ep`, so it must be
# in the process environment — the dispatcher can only warn when it is unset, never set it.
ENV EP_SUPPRESS_NCCL_CHECK=1

# GPU target: "hopper" (SM90, H100/H200) or "blackwell" (B200 SM100 / B300 SM103).
# CUDA arch (TORCH_CUDA_ARCH_LIST for the DeepEP / causal-conv1d source builds): hopper -> 9.0;
# blackwell -> "10.0+PTX" so one image serves BOTH B200 (sm_100, native SASS) and B300 (sm_103, via the
# embedded compute_100 PTX JIT-compiled at load) — without +PTX an sm_100-only cubin may not load on
# sm_103. (FA4 + cutlass-dsl are prebuilt wheels, unaffected by this list.)
ARG TARGET_GPU=hopper
RUN echo $([ "$TARGET_GPU" = "blackwell" ] && echo "10.0+PTX" || echo "9.0") > /etc/cuda_arch

# Install system dependencies (SkyPilot requires sudo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    wget \
    vim \
    htop \
    tmux \
    rsync \
    openssh-client \
    openssh-server \
    build-essential \
    ninja-build \
    bubblewrap \
    sudo \
    bc \
    unzip \
    screen \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Install AWS CLI v2 (for S3/ECR access from containers)
RUN curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscliv2.zip /tmp/aws

# SkyPilot requirement: passwordless sudo for default user
RUN echo "ALL ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set working directory
WORKDIR /workspace

# Put the repo root on the import path so test files run by path resolve the
# `tests` package (`from tests.common...`); `src` is already importable via the
# editable install (pyproject hatchling `packages = ["src"]`), but `tests` is
# not a packaged module. This removes the need to pass PYTHONPATH=/workspace per
# command when running `python tests/.../foo.py`.
ENV PYTHONPATH=/workspace

# Install uv and ninja (pip ninja needed for CUDA device linking).
# Pre-remove the NGC-bundled copies the uv reconcile below cannot cleanly replace in place. Only
# numba and notebook stay absent from the final image (nothing in the lockfile pulls them);
# grpcio, torchvision, scikit-learn, matplotlib and pytest come back at their locked versions.
RUN pip install --upgrade pip uv==${UV_VERSION} ninja \
    && pip uninstall -y numba notebook grpcio torchvision scikit-learn matplotlib pytest || true

# Install PyTorch first — Flash Attention and DeepEP compile C++/CUDA extensions
# against PyTorch's ABI, so the correct version must be present before building.
# NGC 26.03 ships PyTorch 2.11.0a0; we replace with stable 2.11.0+cu130 (required
# for working torch._grouped_mm CUTLASS kernels on B200/B300 SM100+).
RUN pip install "torch~=2.11.0" --index-url https://download.pytorch.org/whl/cu130

# =============================================================================
# Flash Attention Installation
# =============================================================================
# Hopper: FA3 from a `main` SHA (only main handles CUDA 13) plus FA2 from the v2.8.3.post1 tag
# (only the tag is a known-good FA2) — both source builds; changing either pin needs a Hopper
# revalidation. Blackwell: FA4 (`flash_attn.cute`, prebuilt CuTe DSL wheel) added ON TOP OF the
# base image's FA2 under the same namespace — FA2 is inherited, never built, and stays the
# fallback `_detect_attention_impl` selects when FA4 is unusable. FA3 is Hopper-only (wgmma is
# incompatible with SM100). See agent-docs/optimization/flash-attention.md.
# Source: https://github.com/Dao-AILab/flash-attention

# The cutlass-dsl / quack pair FA4 and the Muon + block-scaled MoE kernels all need. Declared once
# here because two layers force-reinstall it (this one, and again after the uv reconcile below).
ARG CUTLASS_DSL_VERSION=4.5.2
ARG QUACK_VERSION=0.5.0

# Stub .cpp providing throw-on-call full specializations for the 24 FA2 split-K
# template instantiations that CUDA 13.2 ptxas hangs on for sm_90 (Hopper only).
# Used by the Hopper FA2 build path below; ignored on Blackwell.
COPY docker/training/flash_attn_split_stubs_hopper.cpp /tmp/flash_attn_split_stubs_hopper.cpp

# Blackwell: install FA4 (prebuilt CuTe DSL wheel — no source compile, ~minutes vs ~25 min for FA2).
# Hopper: build FA2 (+FA3) from source. FA2's setup.py ignores TORCH_CUDA_ARCH_LIST; it uses its own
# FLASH_ATTN_CUDA_ARCHS to decide which SM archs to compile — restrict it to sm_90 only.
# MAX_JOBS=16 caps concurrent nvcc jobs (NGC base sets MAX_JOBS=N_CORES → 100+ ptxas procs).
RUN if [ "$TARGET_GPU" = "blackwell" ]; then \
      echo "=== Installing FlashAttention-4 (Blackwell, prebuilt CuTe DSL wheel) ===" \
      && pip install "flash-attn-4[cu13]==4.0.0b16" \
      # FA4 declares only floors, so it resolves the newest cutlass-dsl and installs the 10KB meta
      # pkg without the '-libs-base' .pth that exposes top-level `cutlass`. Force the exact trio.
      && pip install --force-reinstall --no-deps \
           "nvidia-cutlass-dsl==${CUTLASS_DSL_VERSION}" \
           "nvidia-cutlass-dsl-libs-base==${CUTLASS_DSL_VERSION}" \
           "nvidia-cutlass-dsl-libs-cu13==${CUTLASS_DSL_VERSION}" \
           "quack-kernels==${QUACK_VERSION}" \
      # Orphans of the newer floor resolve: the 4.6+-only '-libs-core' split and the cu12 libs,
      # neither of which exists at the pinned version.
      && pip uninstall -y nvidia-cutlass-dsl-libs-core nvidia-cutlass-dsl-libs-cu12 \
      && python -c "import cutlass.cute.nvgpu; from flash_attn.cute import flash_attn_func; print('FA4 + cutlass.cute.nvgpu: OK')" \
      # No layer here installs FA2 on Blackwell — it is inherited from the base image and is the
      # FA4 fallback, so a base bump that drops it would silently remove that fallback. BASE_IMAGE
      # is the pin; this only asserts the base still delivers.
      && python -c "import flash_attn; from flash_attn import flash_attn_func; \
print(f'FA2 inherited from base: {flash_attn.__version__}')"; \
    else \
      export TORCH_CUDA_ARCH_LIST=$(cat /etc/cuda_arch) \
      && export FLASH_ATTN_CUDA_ARCHS="90" \
      && export MAX_JOBS=16 \
      && export NVCC_THREADS=4 \
      # Same CUDA 13 CCCL relocation the DeepEP build handles: the headers moved out of
      # /usr/local/cuda/include, and the FA2 tag's vendored cutlass (fast_math.h) includes
      # <cuda/std/utility> without adding the new path. FA3's main SHA adds it itself.
      && export CPATH=/usr/local/cuda/targets/x86_64-linux/include/cccl:${CPATH} \
      && echo "=== Building Flash Attention 2+3 from source for Hopper (TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}) ===" \
      && echo "Building FA3 from hopper/ (SM90)..." \
      # FA3 builds from a main SHA while FA2 below builds from the v2.8.3.post1 tag: no FA release
      # supports CUDA 13 yet. The tag vendors CUTLASS 4.0, which reads the unversioned
      # PFN_cuTensorMapEncodeTiled typedef CUDA 13 removed, and its hopper/setup.py swaps the
      # compiler for a downloaded CUDA 12.6 nvcc on any toolkit that is not exactly 12.8 — against
      # this image's CTK 13.2 headers CCCL then rejects the 12.6-compiler/13.2-header skew before a
      # single kernel compiles. c46b814 is upstream's CUDA 13 path: it keeps the system nvcc on
      # CTK >= 13 and vendors CUTLASS 4.3. Bump it only with an FA3 revalidation on Hopper.
      && git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention-fa3 \
      && git -C /tmp/flash-attention-fa3 checkout c46b8144f2d5039d3d3de05da1b668325130bb35 \
      && cd /tmp/flash-attention-fa3/hopper \
      && python setup.py install \
      # Leave the tree before deleting it — a shell whose CWD no longer exists cannot start git
      # ("fatal: Unable to read current working directory"), so the FA2 clone below would die.
      && cd /tmp \
      && rm -rf /tmp/flash-attention-fa3 \
      && echo "Building FA2 from source..." \
      # FA2 stays on a RELEASE tag — an untagged main build is neither identifiable nor
      # reproducible. v2.8.3.post1 is upstream's newest tag and its sm80-style kernels do compile
      # against CTK 13.2 (they never reach CUTLASS's TMA host adapter). Bump it only with an FA2
      # revalidation on Hopper — the GPU tier's attention tests are the gate.
      && git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention \
      && git -C /tmp/flash-attention checkout v2.8.3.post1 \
      && cd /tmp/flash-attention \
      && echo "Hopper: replacing 24 FA2 split-K kernel files with throw-stubs" \
      && echo "  (CUDA 13.2 ptxas hangs indefinitely on these for sm_90;" \
      && echo "   split-K is only used by KV-cache inference paths, not training)" \
      && cp /tmp/flash_attn_split_stubs_hopper.cpp csrc/flash_attn/src/flash_fwd_split_stubs.cpp \
      && sed -i '/csrc\/flash_attn\/src\/flash_fwd_split_/d' setup.py \
      && sed -i '/csrc\/flash_attn\/flash_api.cpp/a\                "csrc/flash_attn/src/flash_fwd_split_stubs.cpp",' setup.py \
      && echo "Patched setup.py: removed split .cu lines, added 1 stub .cpp" \
      && pip install --no-build-isolation . \
      && cd /tmp \
      && rm -rf /tmp/flash-attention; \
    fi

# Set up FA3 (Hopper only) package structure for unified import paths.
# FA3's hopper/setup.py installs `flash_attn_interface` / `flash_attn_config` as TOP-LEVEL modules
# beside a near-empty flash_attn_3/ package, but transformers imports both spellings. Re-export the
# top-level modules through flash_attn_3/__init__.py and write a dist-info so
# importlib.metadata.version() resolves. Handles both install layouts (setuptools .egg, or the
# modern dir layout NGC 26.03 produces).
RUN if [ "$TARGET_GPU" = "hopper" ]; then \
      cd /usr/local/lib/python3.12/dist-packages \
      && FA3_EGG=$(ls flash_attn_3-*.egg 2>/dev/null | head -1) \
      && if [ -n "$FA3_EGG" ] && [ -f "$FA3_EGG" ]; then \
           echo "Found FA3 .egg, extracting: $FA3_EGG" \
           && unzip -o "$FA3_EGG" \
           && rm "$FA3_EGG" \
           && sed -i "/$FA3_EGG/d" easy-install.pth 2>/dev/null || true \
           && rm -rf EGG-INFO; \
         else \
           echo "No FA3 .egg present (modern wheel/dist install layout)"; \
         fi \
      # Determine FA3 version: try installed metadata, then dist-info dir, fall back to 3.0.0.
      # Use a single inline shell expression — chained `&& [ -z X ] && X=...` semantics break
      # because `[ -z ]` returns 1 when the var is already set, killing the && chain.
      && FA3_VERSION=$( \
           v=$(python -c "import importlib.metadata as m; \
               [print(d.version) for d in m.distributions() if d.metadata['Name'] == 'flash_attn_3']" 2>/dev/null \
               | head -1); \
           [ -z "$v" ] && v=$(ls -d flash_attn_3-*.dist-info 2>/dev/null | head -1 \
               | sed 's/flash_attn_3-\(.*\)\.dist-info/\1/'); \
           [ -z "$v" ] && v="3.0.0"; \
           echo "$v" \
         ) \
      && echo "FA3 version detected: $FA3_VERSION" \
      # Re-export top-level py_modules through flash_attn_3 package
      && mkdir -p flash_attn_3 \
      && echo "from flash_attn_interface import *" > flash_attn_3/__init__.py \
      && echo "from flash_attn_config import *" >> flash_attn_3/__init__.py \
      # Ensure a clean dist-info exists for importlib.metadata
      && DIST_INFO="flash_attn_3-${FA3_VERSION}.dist-info" \
      && if [ ! -d "$DIST_INFO" ]; then \
           mkdir -p "$DIST_INFO" \
           && printf "Metadata-Version: 2.1\nName: flash_attn_3\nVersion: %s\n" "$FA3_VERSION" > "$DIST_INFO/METADATA" \
           && echo "flash_attn_3" > "$DIST_INFO/top_level.txt" \
           && touch "$DIST_INFO/INSTALLER" \
           && echo "Created $DIST_INFO"; \
         fi \
      && python -c "from flash_attn_3 import flash_attn_func; print('FA3 via flash_attn_3: OK')" \
      && python -c "from flash_attn_interface import flash_attn_func; print('FA3 via flash_attn_interface: OK')" \
      && python -c "import importlib.metadata; print(f'FA3 version via metadata: {importlib.metadata.version(\"flash_attn_3\")}')"; \
    else \
      echo "Skipping FA3 setup (Blackwell target)"; \
    fi

# =============================================================================
# DeepEP V2 (EPv2) Installation (MoE Expert Parallelism - 2-3x faster than NCCL)
# =============================================================================
# DeepEP V2 unifies EP under `ElasticBuffer` over the NCCL Gin backend for cross-node scale-out
# (e.g. AWS EFA) and a non-Gin NVLink path intra-node. See agent-docs/infrastructure/deepep.md.
#
# NCCL is installed at uv.lock's EXACT pin, not a floor: `deep_ep._C` is compiled against whatever
# lands here, and the same lock pins both inference images — a floor would pair the built extension
# with a different runtime and skew the weight-sync communicator. NVSHMEM arrives transitively with
# torch (nvidia-nvshmem-cu13); installing nvidia-nvshmem-cu12 would clobber the cu13 headers.
# CPATH is widened because CUDA 13 moved the CCCL headers out of /usr/local/cuda/include, which
# DeepEP's host-compiler invocation still assumes. TORCH_CUDA_ARCH_LIST=9.0+PTX governs only the
# AOT half of `deep_ep._C` (the `legacy` V1 kernels, PTX-JIT'd on Blackwell) — the `elastic` V2
# kernels this toolkit runs are compiled at runtime for the live device, so no Blackwell arch is
# needed here and this list deliberately overrides /etc/cuda_arch.
COPY uv.lock docker/nccl_pin.py /tmp/
RUN NCCL_V=$(python /tmp/nccl_pin.py /tmp/uv.lock) \
    && pip install "nvidia-nccl-cu13==${NCCL_V}" --no-deps \
    && rm /tmp/uv.lock /tmp/nccl_pin.py \
    && export TORCH_CUDA_ARCH_LIST="9.0+PTX" \
    && export CPATH=/usr/local/cuda/targets/x86_64-linux/include/cccl:${CPATH} \
    && git clone https://github.com/deepseek-ai/DeepEP.git /tmp/DeepEP \
    && cd /tmp/DeepEP \
    && git checkout af9a040 \
    && pip install --no-build-isolation . \
    && rm -rf /tmp/DeepEP

# =============================================================================
# DeepGEMM (native fp8/fp4 grouped MoE GEMM) — Blackwell only
# =============================================================================
# The kernel behind `HALO_DEEPGEMM_NATIVE=1` (`src/kernels/deepgemm.py`). Without it
# `deepgemm_available()` is False and the opt-in silently does nothing: the simulated fake-quant path
# runs instead, at different speed and different numerics from what the flag asks for. It also leaves
# `tests/gpu/kernels/test_deepgemm.py` exiting 0 without running either kernel check.
#
# Blackwell only: the kernels this toolkit calls are the SM100 block-scaled grouped ones, and the
# Hopper image has no caller. Pinned like DeepEP — upstream main moves and a rebuild must stay
# reproducible. Recursive clone: CUTLASS is a submodule and the build needs its headers. Like
# DeepEP's V2 path, DeepGEMM JIT-compiles at runtime against the live device, so it relies on the
# full CUDA toolkit the image already keeps.
RUN if [ "$TARGET_GPU" = "blackwell" ]; then \
      echo "=== Installing DeepGEMM (native fp8/fp4 grouped MoE GEMM, SM100) ===" \
      && export TORCH_CUDA_ARCH_LIST="10.0+PTX" \
      && export CPATH=/usr/local/cuda/targets/x86_64-linux/include/cccl:${CPATH} \
      && git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git /tmp/DeepGEMM \
      && cd /tmp/DeepGEMM \
      && git checkout 559d79f \
      && git submodule update --init --recursive \
      && pip install --no-build-isolation . \
      && cd / && rm -rf /tmp/DeepGEMM; \
    else \
      echo "=== Skipping DeepGEMM (Blackwell-only kernels) ==="; \
    fi

# =============================================================================
# GIN-capable aws-ofi-nccl + GDRCopy (DeepEP V2 cross-node EP over AWS EFA)
# =============================================================================
# The NGC base bundles aws-ofi-nccl 1.17.3, which exports NO `ncclGin` symbol, so NCCL's GIN plugin
# init fails and DeepEP V2 inter-node EP aborts at the first all-to-all with "NCCL GIN is
# unavailable". Build a GIN-capable aws-ofi-nccl (exports `ncclGinPlugin_v13`, matching the NCCL the
# lock pins) over the bundled one and expose the same .so as `libnccl-gin.so`, which is the separate
# file NCCL loads the GIN plugin from. GDRCopy's `libgdrapi` follows because proxy GIN — the only
# EFA path, EFA has no IBGDA — needs GDRCopy >= 2.5 for its host->GPU copy; the `gdrdrv` module and
# /dev/gdrdrv come from the HOST (pass `--device /dev/gdrdrv`). Enabling it per job:
# agent-docs/infrastructure/deepep.md#expert-parallelism-over-aws-efa.
#
# Both are pinned to the commits the shipped images were built from. `--disable-tests` is load-
# bearing at this pin: the functional tests `#include <mpi.h>`, which this image does not carry.
ARG AWS_OFI_NCCL_COMMIT=1f0a976f537f859d8ea70c6699f7d92ac89eb7af
ARG GDRCOPY_COMMIT=fcec3ce0bb40a97a6cc45dd4afeec4bccb509712
RUN apt-get update && apt-get install -y --no-install-recommends libtool hwloc libhwloc-dev \
    && rm -rf /var/lib/apt/lists/* \
    && NCCL_HOME="$(python -c 'import nvidia.nccl; print(nvidia.nccl.__path__[0])')" \
    && git clone https://github.com/aws/aws-ofi-nccl.git /tmp/aws-ofi-nccl \
    && cd /tmp/aws-ofi-nccl \
    && git checkout ${AWS_OFI_NCCL_COMMIT} \
    && git submodule update --init --recursive \
    && ./autogen.sh \
    && ./configure --prefix=/opt/amazon/aws-ofi-nccl \
        --with-libfabric=/opt/amazon/efa \
        --with-cuda=/usr/local/cuda \
        --with-nccl="${NCCL_HOME}" \
        --enable-platform-aws \
        --disable-tests \
    && make -j"$(nproc)" && make install \
    && ln -sf /opt/amazon/aws-ofi-nccl/lib/libnccl-net-ofi.so /usr/lib/x86_64-linux-gnu/libnccl-gin.so \
    && git clone https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && git -C /tmp/gdrcopy checkout ${GDRCOPY_COMMIT} \
    && make -C /tmp/gdrcopy PREFIX=/usr/local lib lib_install \
    && ldconfig \
    && rm -rf /tmp/aws-ofi-nccl /tmp/gdrcopy

# Re-install apt-shipped Python packages with proper pip RECORD files.
# NGC 26.03 ships PyYAML, Pygments, and wheel via debian dpkg in /usr/lib/python3/dist-packages/
# without pip RECORD files. The uv reconcile below tries to swap them for the lockfile versions
# and fails because pip cannot uninstall packages without a RECORD file.
# We use `pip install --ignore-installed` (NOT --force-reinstall, which still attempts the
# no-RECORD uninstall and aborts) to write fresh, properly-recorded copies into
# /usr/local/lib/python3.12/dist-packages/ — that path has sys.path priority over the
# apt-shipped one, so uv can later uninstall those cleanly.
RUN pip install --ignore-installed --no-deps PyYAML pygments wheel

# Pre-install packages whose source builds need our system torch (2.11+cu130 + CUDA 13.2).
# uv's PEP-517 builder creates an isolated venv and pulls torch from PyPI — that's
# 2.10+cu128, which mismatches the CUDA 13.2 toolkit and aborts (see torch's _check_cuda_version).
# We pip-install with --no-build-isolation so they compile against system torch; uv then
# detects the lockfile version is already satisfied and skips them.
# Pin to the EXACT uv.lock versions: a mismatch makes uv rebuild the locked version in an
# isolated PEP-517 env and abort (causal-conv1d's setup.py needs torch at build time).
# Bump these when uv.lock bumps them.
RUN export TORCH_CUDA_ARCH_LIST=$(cat /etc/cuda_arch) \
    && pip install --no-build-isolation "causal-conv1d==1.6.2.post1" "flash-linear-attention==0.4.2"

# Install the remaining locked dependencies. `uv export` renders uv.lock flat; the pre-built
# compiled deps already satisfy their locked pins, so uv skips them and their ABI survives.
# --no-emit-project keeps `src` out (installed editable below); --no-deps installs exactly the
# locked closure. The export carries the `dev` group, making this image the test/docs/lint runtime
# too. Placed after FA/DeepEP so a lockfile change does not invalidate those builds.
COPY pyproject.toml uv.lock* ./
RUN uv export --locked --no-emit-project --no-hashes --format requirements-txt -o /tmp/requirements.txt \
    && uv pip install --system --no-deps -r /tmp/requirements.txt \
    && rm -rf ${UV_CACHE_DIR} /tmp/requirements.txt

# gram-newton-schulz hard-pins nvidia-cutlass-dsl==4.4.2, so the uv reconcile above re-installs that
# and breaks `import cutlass` (the 4.4.2 wheel exposes no top-level `cutlass` with cute.nvgpu /
# _mlir), taking the Muon Newton-Schulz and Blackwell block-scaled MoE/dense GEMM kernels
# (agent-docs/optimization/low-precision-moe-kernels.md) with it — and, on Blackwell, FA4's
# flash_attn.cute. Restore the pinned pair here; --no-deps keeps the FA4-pulled libs. The
# flash_attn.cute assertion is Blackwell-only: Hopper installs FA2+FA3 and never has FA4.
RUN pip install --force-reinstall --no-deps \
      "nvidia-cutlass-dsl==${CUTLASS_DSL_VERSION}" \
      "nvidia-cutlass-dsl-libs-base==${CUTLASS_DSL_VERSION}" \
      "nvidia-cutlass-dsl-libs-cu13==${CUTLASS_DSL_VERSION}" \
      "quack-kernels==${QUACK_VERSION}" \
    && python -c "import cutlass, cutlass.cute.nvgpu, cutlass._mlir; print('cutlass-dsl import OK:', cutlass.__file__)" \
    && if [ "$TARGET_GPU" = "blackwell" ]; then \
         python -c "from flash_attn.cute import flash_attn_func; print('FA4 flash_attn.cute import OK')"; \
       else \
         echo "FA4 flash_attn.cute assertion skipped (Hopper installs FA2+FA3, not FA4)"; \
       fi

# FlashAdamW optimizer (optim: flash_adamw) — quantized-state optimizer (~57% less
# per-param memory). Ships as the optional `flash-optimizers` extra; install it here so
# the optimizer is available out-of-the-box. Pure-Triton, no source build needed.
# --no-deps: torch + triton are already installed; without it pip re-resolves
# `torch>=2.6.0` and re-enforces torch's hard `nvidia-nccl-cu13==2.28.9` pin, downgrading the
# locked NCCL that DeepEP V2 was built against (→ `undefined symbol: ncclCommQueryProperties`).
RUN pip install "flashoptim==0.1.4" --no-deps \
    && python -c "from flashoptim import FlashAdamW; print('flashoptim OK')"

# Gigatoken tokenizer backend (tokenizer_backend: gigatoken) — Rust bulk encoder for dataset
# tokenization. Ships as the optional `gigatoken` extra; install it here so the backend (and its
# CPU parity test suite) works out-of-the-box. Pure Rust wheel + awkward-array, no torch deps.
# Exact pin = uv.lock's version; bump both together.
RUN pip install "gigatoken==0.9.0" \
    && python -c "import gigatoken; print('gigatoken OK')"

# Guard: any pip step above that re-resolved torch can silently downgrade nvidia-nccl-cu13 to
# torch's pinned 2.28.9, leaving deep_ep._C with an undefined Gin symbol at import. `--verify`
# asserts the DeepEP V2 floor, equality with uv.lock, AND that the libnccl.so.2 an actual process
# resolves is that wheel's — the NGC base ships its own older system copy under /usr/lib, which a
# metadata-only check would not notice winning the soname race. deep_ep._C was compiled against the
# lock's version above, and the inference images pin from the same lock, so any drift skews the
# weight-sync communicator too. The same helper both inference images call: one check, one place.
COPY docker/nccl_pin.py /tmp/nccl_pin.py
RUN python /tmp/nccl_pin.py /workspace/uv.lock --verify && rm /tmp/nccl_pin.py

# Cache-bust the source COPY layers: BuildKit's local COPY cache can serve a stale src/ snapshot,
# baking old code into a "rebuilt" image. The caller passes SOURCE_REVISION (the git SHA) so this
# cheap RUN — and every COPY + editable reinstall below it — reruns whenever the code moves, while
# the heavy dep-compile layers above stay cached. See agent-docs/infrastructure/docker.md.
ARG SOURCE_REVISION=dev
RUN echo "Building Halo source revision: ${SOURCE_REVISION}"

# Record the revision on the image so `docker inspect` can answer "which commit is in here?" without
# checksumming the tree. Declared after the heavy layers, so stamping it costs no rebuild — the
# release version LABEL lives here too, so a version bump never invalidates the dep-compile cache.
LABEL org.opencontainers.image.revision="${SOURCE_REVISION}"
ARG VERSION=1.0.0
LABEL version="${VERSION}"

# Copy project source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY launcher-configs/ ./launcher-configs/
COPY jinja_templates/ ./jinja_templates/
COPY examples/ ./examples/
COPY agent-docs/ ./agent-docs/
# Skills live at skills/; .claude/skills and .agents/skills point at it so the bundled
# Claude Code (and Codex) discover them without a repo bind-mount. Only the skills ship:
# the rest of .claude/.agents is local agent state. mkdocs.yml makes `mkdocs serve` usable in-image.
COPY skills/ ./skills/
RUN mkdir -p .claude .agents && ln -s ../skills .claude/skills && ln -s ../skills .agents/skills
COPY mkdocs_hooks/ ./mkdocs_hooks/
COPY README.md CLAUDE.md AGENTS.md mkdocs.yml LICENSE APACHE-2.0.txt ./

# Install the project package in editable mode. Not masked with `|| echo`: swallowing a failure here
# ships an image with no `halo` CLI and no installed `src` distribution.
RUN uv pip install --system -e . --no-deps

# Drop transformer_engine — shipped by the NGC base image but unused by Halo (no import
# anywhere in src/scripts/tests, absent from pyproject.toml/uv.lock). Removed here, after the
# FA/DeepEP builds and the uv reconcile, so nothing can re-pull it; trims a few hundred MB.
RUN pip uninstall -y transformer_engine || true

# =============================================================================
# Verification (build-time checks - GPU-dependent packages verified at runtime)
# =============================================================================
# Note: deep_ep requires libcuda.so.1 which is only available at runtime with --gpus
RUN echo "=== Verifying Installation (TARGET_GPU=${TARGET_GPU}) ===" \
    && python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')" \
    && python -c "import transformers; print(f'Transformers: {transformers.__version__}')" \
    && python -c "import trl; print(f'TRL: {trl.__version__}')" \
    && python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')" \
    && python -c "import flash_attn; print(f'Flash Attention 2: {flash_attn.__version__}')" \
    && python -c "from flash_attn.flash_attn_interface import flash_attn_func; print('  - FA2 flash_attn_func: OK')" \
    && if [ "$TARGET_GPU" = "hopper" ]; then \
         python -c "import flash_attn_3; print('Flash Attention 3: OK')" \
         && python -c "from transformers.utils import is_flash_attn_3_available; print(f'  - transformers FA3 detection: {is_flash_attn_3_available()}')"; \
       else \
         echo "  - FA3: skipped (Blackwell target)"; \
       fi \
    && python -c "import nvidia.nvshmem; print('NVSHMEM: installed')" \
    && python -c "from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient; print('Vendored NCCL client: OK')" \
    && halo --help >/dev/null && echo "halo CLI: $(command -v halo)" \
    && echo "=== Build-time verification complete ===" \
    && echo "=== Note: DeepEP will be verified at runtime with GPU access ==="

# =============================================================================
# Claude Code CLI (Anthropic's agentic coding tool)
# =============================================================================
# Native installer (standalone binary, symlinked at ~/.local/bin/claude). Non-fatal: a dev
# convenience, not a training dependency — a claude.ai blip must not fail the build.
# The trailing rm scrubs installer residue in the same layer that creates it (~/.claude.json is
# first-run state the installer seeds; backups/downloads are build residue) — the CLI itself stays.
RUN (curl -fsSL https://claude.ai/install.sh | bash && test -x /root/.local/bin/claude \
    && echo "Claude Code installed: $(ls -la /root/.local/bin/claude)" \
    && rm -rf /root/.claude.json /root/.claude/backups /root/.claude/downloads) \
    || echo "Claude Code install skipped (installer unreachable); not required for training."

# Set PATH globally via ENV (works in all contexts: interactive, non-interactive, docker exec, CMD)
ENV PATH="/root/.local/bin:${PATH}"
RUN echo "export TORCH_CUDA_ARCH_LIST=$(cat /etc/cuda_arch)" >> /root/.bashrc

# Default command
CMD ["python", "-c", "print('Halo ready.\\nRun training: python scripts/training/sft.py <config.yaml>\\nDocs: mkdocs serve')"]
