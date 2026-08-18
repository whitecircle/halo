# Halo training image for Hopper and Blackwell GPUs. vLLM uses a separate image (Dockerfile.vllm):
# its torch/transformers stack is incompatible with this one.
# TARGET_GPU: hopper (H100/H200, SM90) | blackwell (B200 SM100 / B300 SM103 via +PTX).
# Build: make build-hopper / make build-blackwell. See agent-docs/infrastructure/docker.md.

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.03-py3
FROM ${BASE_IMAGE}

# Static identity labels. The release labels sit below the dependency layers to preserve the
# dependency cache.
LABEL maintainer="White Circle <hello@whitecircle.ai>"
LABEL org.opencontainers.image.title="Halo"
LABEL org.opencontainers.image.description="Halo is an open-source framework built by White Circle for training large language and multimodal models"
LABEL org.opencontainers.image.vendor="White Circle"
LABEL org.opencontainers.image.source="https://github.com/whitecircle/halo"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_NO_CACHE_DIR=1
ARG PIP_DISABLE_PIP_VERSION_CHECK=1

ENV TZ=UTC
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# uv installs the locked set into the system interpreter — NGC's split /usr vs /usr/local layout
# rejects `uv sync`; --break-system-packages opts past PEP 668.
ARG UV_VERSION=0.10.12
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/tmp/uv_cache

ENV HF_XET_HIGH_PERFORMANCE=1
ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Enables FA4's on-disk JIT kernel cache, off by default (the toolkit anchors the dir on HF_HOME).
ENV FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
# CuTe DSL fast-invocation ABI (FA4 + block-scaled GEMMs). Read at DSL import time, so it has to be
# in the environment before the process starts.
ENV CUTE_DSL_ENABLE_TVM_FFI=1

# NGC defaults fp32 matmuls to TF32, which corrupts long-context RoPE positions. Use full fp32
# matmul precision; the toolkit also sets this policy at runtime.
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

# NCCL defaults for Mellanox IB/RoCE, the default fabric. AWS EFA needs no extra packages but is
# opt-in per job — see agent-docs/parallelism/multi-node.md#rdma-fabrics.
ENV NCCL_DEBUG=WARN
ENV NCCL_IB_DISABLE=0
ENV NCCL_NET_GDR_LEVEL=2
ENV NCCL_IB_HCA=mlx5
ENV NCCL_P2P_LEVEL=NVL

# Serialize device work onto one hardware queue. Set here because the driver latches it at CUDA
# initialization, which `import deep_ep` triggers — a Python write is too late.
ENV CUDA_DEVICE_MAX_CONNECTIONS=1

# DeepEP's duplicate-NCCL-runtime guard flags NGC's HPC-X libnccl-net.so transport plugin;
# it is complementary, not a second runtime. Latched at `import deep_ep`.
ENV EP_SUPPRESS_NCCL_CHECK=1

# CUDA arch list for the source builds below: hopper -> 9.0; blackwell -> 10.0+PTX so one image
# serves B200 (native SASS) and B300 (JIT via embedded PTX).
ARG TARGET_GPU=hopper
RUN echo $([ "$TARGET_GPU" = "blackwell" ] && echo "10.0+PTX" || echo "9.0") > /etc/cuda_arch

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

RUN curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscliv2.zip /tmp/aws

# SkyPilot requires passwordless sudo.
RUN echo "ALL ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /workspace

# Repo root on the import path so test files run by path resolve the `tests` package.
ENV PYTHONPATH=/workspace

# uv + ninja; pre-remove the NGC-bundled copies the uv reconcile below cannot replace in place.
RUN pip install --upgrade pip uv==${UV_VERSION} ninja \
    && pip uninstall -y numba notebook grpcio torchvision scikit-learn matplotlib pytest || true

# Stable torch before the C++/CUDA extension builds below — they compile against its ABI, and
# `torch._grouped_mm` on SM100+ needs the stable wheel rather than NGC's prerelease.
RUN pip install "torch~=2.11.0" --index-url https://download.pytorch.org/whl/cu130

# ---------------------------------------------------------------------------
# Flash Attention. Hopper: FA2 (v2.8.3.post1 tag) + FA3 (main SHA — the only CUDA 13 path), both
# source builds for SM90. Blackwell: prebuilt FA4 wheel on top of the base image's inherited FA2,
# the fallback when FA4 is unusable. See agent-docs/optimization/flash-attention.md.
# ---------------------------------------------------------------------------

# cutlass-dsl/quack pair FA4, Muon and the block-scaled MoE kernels all need; two layers
# force-reinstall it (here and after the uv reconcile).
ARG CUTLASS_DSL_VERSION=4.5.2
ARG QUACK_VERSION=0.5.0

# Throw-on-call stubs for the FA2 split-K instantiations CUDA 13.2 ptxas hangs on for sm_90
# (split-K serves KV-cache inference paths, not training). Hopper build only.
COPY docker/training/flash_attn_split_stubs_hopper.cpp /tmp/flash_attn_split_stubs_hopper.cpp

RUN if [ "$TARGET_GPU" = "blackwell" ]; then \
      echo "=== Installing FlashAttention-4 (Blackwell, prebuilt CuTe DSL wheel) ===" \
      && pip install "flash-attn-4[cu13]==4.0.0b16" \
      # Force the pinned cutlass-dsl package set; FA4 itself declares floors only.
      && pip install --force-reinstall --no-deps \
           "nvidia-cutlass-dsl==${CUTLASS_DSL_VERSION}" \
           "nvidia-cutlass-dsl-libs-base==${CUTLASS_DSL_VERSION}" \
           "nvidia-cutlass-dsl-libs-cu13==${CUTLASS_DSL_VERSION}" \
           "quack-kernels==${QUACK_VERSION}" \
      && pip uninstall -y nvidia-cutlass-dsl-libs-core nvidia-cutlass-dsl-libs-cu12 \
      && python -c "import cutlass.cute.nvgpu; from flash_attn.cute import flash_attn_func; print('FA4 + cutlass.cute.nvgpu: OK')" \
      # Verify that the base image still provides the FA2 fallback.
      && python -c "import flash_attn; from flash_attn import flash_attn_func; \
print(f'FA2 inherited from base: {flash_attn.__version__}')"; \
    else \
      export TORCH_CUDA_ARCH_LIST=$(cat /etc/cuda_arch) \
      && export FLASH_ATTN_CUDA_ARCHS="90" \
      && export MAX_JOBS=16 \
      && export NVCC_THREADS=4 \
      # CUDA 13 moved the CCCL headers; both FA builds still expect the old include path.
      && export CPATH=/usr/local/cuda/targets/x86_64-linux/include/cccl:${CPATH} \
      && echo "=== Building Flash Attention 2+3 from source for Hopper (TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}) ===" \
      && echo "Building FA3 from hopper/ (SM90)..." \
      # c46b814 is upstream's CUDA 13 path: it keeps system nvcc on CTK >= 13 and vendors CUTLASS 4.3.
      && git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention-fa3 \
      && git -C /tmp/flash-attention-fa3 checkout c46b8144f2d5039d3d3de05da1b668325130bb35 \
      && cd /tmp/flash-attention-fa3/hopper \
      && python setup.py install \
      # Leave the directory before deleting it — git cannot start from a deleted CWD.
      && cd /tmp \
      && rm -rf /tmp/flash-attention-fa3 \
      && echo "Building FA2 from source..." \
      # FA2 stays on a release tag for reproducibility; v2.8.3.post1 compiles against CTK 13.2.
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

# FA3 (Hopper only): re-export the top-level flash_attn_interface/flash_attn_config modules
# through flash_attn_3/ and write a dist-info so both transformers import spellings and
# importlib.metadata.version() resolve. Handles both .egg and modern install layouts.
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
      # Single inline expression — a chained `[ -z X ] && X=...` breaks the && chain when set.
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
      && mkdir -p flash_attn_3 \
      && echo "from flash_attn_interface import *" > flash_attn_3/__init__.py \
      && echo "from flash_attn_config import *" >> flash_attn_3/__init__.py \
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

# ---------------------------------------------------------------------------
# DeepEP V2 (MoE expert parallelism). NCCL at uv.lock's exact pin — deep_ep._C compiles against
# it and the inference images pin from the same lock; drift skews the weight-sync communicator.
# TORCH_CUDA_ARCH_LIST here governs only the AOT legacy kernels (V2 elastic kernels JIT at
# runtime), so 9.0+PTX deliberately overrides /etc/cuda_arch.
# ---------------------------------------------------------------------------
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

# DeepGEMM (Blackwell only): the native fp8/fp4 grouped MoE GEMM, opt-in via HALO_DEEPGEMM_NATIVE=1.
# Pinned for reproducibility; JIT-compiles at runtime like DeepEP V2.
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

# GIN-capable aws-ofi-nccl + GDRCopy (DeepEP V2 cross-node EP over AWS EFA): the NGC-bundled
# aws-ofi-nccl exports no ncclGin symbol, so V2 inter-node EP aborts without this build. The same
# .so is exposed as libnccl-gin.so (the file NCCL loads the GIN plugin from); GDRCopy >= 2.5 backs
# proxy GIN, whose host gdrdrv module must be loaded at runtime. --disable-tests: they need mpi.h.
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

# Re-install apt-shipped Python packages with pip RECORD files so the uv reconcile can
# uninstall them cleanly (--ignore-installed writes fresh copies with sys.path priority).
RUN pip install --ignore-installed --no-deps PyYAML pygments wheel

# Pre-install source-built deps against the system torch (uv's isolated PEP-517 env would pull a
# mismatched PyPI torch and abort). Exact uv.lock versions; bump together with the lock.
RUN export TORCH_CUDA_ARCH_LIST=$(cat /etc/cuda_arch) \
    && pip install --no-build-isolation "causal-conv1d==1.6.2.post1" "flash-linear-attention==0.4.2"

# Remaining locked deps (flat export; compiled deps already satisfy their pins and are skipped).
# The export carries the dev group — this image is also the test/docs/lint runtime.
COPY pyproject.toml uv.lock* ./
RUN uv export --locked --no-emit-project --no-hashes --format requirements-txt -o /tmp/requirements.txt \
    && uv pip install --system --no-deps -r /tmp/requirements.txt \
    && rm -rf ${UV_CACHE_DIR} /tmp/requirements.txt

# gram-newton-schulz hard-pins cutlass-dsl 4.4.2, which the reconcile re-installs, breaking
# `import cutlass` (and with it Muon/quack, block-scaled GEMMs, FA4). Restore the pinned trio.
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

# FlashAdamW (optional `flash-optimizers` extra, pure Triton). --no-deps: a re-resolve would
# re-enforce torch's own NCCL pin and downgrade the locked one DeepEP was built against.
RUN pip install "flashoptim==0.1.4" --no-deps \
    && python -c "from flashoptim import FlashAdamW; print('flashoptim OK')"

# Gigatoken (optional `gigatoken` extra) — Rust bulk tokenizer backend. Exact lock version.
RUN pip install "gigatoken==0.9.0" \
    && python -c "import gigatoken; print('gigatoken OK')"

# Guards against a silent NCCL downgrade above: asserts the DeepEP floor, equality with uv.lock,
# and that a live process resolves that wheel's libnccl.so.2 rather than the system copy.
COPY docker/nccl_pin.py /tmp/nccl_pin.py
RUN python /tmp/nccl_pin.py /workspace/uv.lock --verify && rm /tmp/nccl_pin.py

# SOURCE_REVISION (the git SHA) cache-busts the source COPY layers below whenever the code moves,
# while the dependency layers above stay cached.
ARG SOURCE_REVISION=dev
RUN echo "Building Halo source revision: ${SOURCE_REVISION}"

LABEL org.opencontainers.image.revision="${SOURCE_REVISION}"
ARG VERSION=1.0.0
LABEL version="${VERSION}"

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY launcher-configs/ ./launcher-configs/
COPY jinja-templates/ ./jinja-templates/
COPY examples/ ./examples/
COPY agent-docs/ ./agent-docs/
COPY skills/ ./skills/
RUN mkdir -p .claude .agents && ln -s ../skills .claude/skills && ln -s ../skills .agents/skills
COPY README.md CLAUDE.md AGENTS.md ./

# Editable install, deliberately unmasked: a swallowed failure would ship an image with no `halo`
# CLI and no installed `src` distribution.
RUN uv pip install --system -e . --no-deps

# transformer_engine ships with the NGC base and is unused here. Removed after the reconcile, so
# nothing re-pulls it.
RUN pip uninstall -y transformer_engine || true

# Build-time verification (GPU-dependent packages verify at runtime with --gpus).
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

# Optional development tool, disabled by default: --build-arg INSTALL_CLAUDE_CODE=1. Non-fatal when
# enabled — not a training dependency.
ARG INSTALL_CLAUDE_CODE=0
RUN if [ "$INSTALL_CLAUDE_CODE" = "1" ]; then \
      (curl -fsSL https://claude.ai/install.sh | bash && test -x /root/.local/bin/claude \
       && echo "Claude Code installed: $(ls -la /root/.local/bin/claude)" \
       && rm -rf /root/.claude.json /root/.claude/backups /root/.claude/downloads) \
      || echo "Claude Code install skipped (installer unreachable); not required for training."; \
    else \
      echo "Claude Code install disabled (INSTALL_CLAUDE_CODE=0)"; \
    fi

ENV PATH="/root/.local/bin:${PATH}"
RUN echo "export TORCH_CUDA_ARCH_LIST=$(cat /etc/cuda_arch)" >> /root/.bashrc

CMD ["python", "-c", "print('Halo ready.\\nRun training: python scripts/training/sft.py <config.yaml>\\nDocs: agent-docs/README.md')"]
