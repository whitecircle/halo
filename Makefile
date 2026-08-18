# Halo — task runner. The Docker incantation lives here ONCE; contributors, CI, and the
# Claude Code skills all call these targets instead of re-typing `docker run ...`.
#
# The host has no usable Python (torch/DeepEP/Flash-Attention live only inside the images),
# so everything that executes runs inside the image. Pick the image for your GPU:
#   make ... IMAGE=halo:blackwell   # B200/B300 (default)
#   make ... IMAGE=halo:hopper      # H100/H200
#
# Lint/format run on the host via `uvx ruff` (the only host-runnable gates); docs need the
# image because mkdocs and its plugins ship there, not on the host.

# bash, not dash (the default /bin/sh on Debian/Ubuntu): several recipes use bashisms.
SHELL := /bin/bash

IMAGE        ?= halo:blackwell
# SemVer for the immutable image tags the push targets publish alongside the moving tags.
VERSION      ?= 1.0.0
NPROC        ?= 8
# Training method name for `make train` (any `halo launch --list` entry).
METHOD       ?= sft
RUFF_VERSION ?= 0.9.10
# Git SHA stamped into the image to bust the source-COPY cache (see the SOURCE_REVISION note in
# Dockerfile). `dev-nogit` if not in a git checkout. Overridable, but the default is what you want.
SOURCE_REVISION ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev-nogit)
PYTEST_ARGS  ?=          # extra pytest flags, e.g. PYTEST_ARGS="--junitxml=cpu-junit.xml" (CI)
# Ports the server tiers listen on. The same names drive docker-compose.{vllm,sglang}.yml
# (`${VLLM_PORT:-8000}` / `${SGLANG_PORT:-30000}`), so exporting one moves the server AND the URL
# the tests dial; hardcoding the URL here would probe :8000 against a server started elsewhere.
VLLM_PORT   ?= 8000
SGLANG_PORT ?= 30000
# Where the vLLM tier's server answers (docker-compose.vllm.yml, network_mode: host).
VLLM_SERVER_URL ?= http://localhost:$(VLLM_PORT)
# Where the SGLang tier's server answers (docker-compose.sglang.yml, network_mode: host).
SGLANG_SERVER_URL ?= http://localhost:$(SGLANG_PORT)

# Credential + data mounts — overridable so untrusted CI can run creds-free. The defaults mount the
# maintainer's `.env` (WANDB/HF/AWS keys), `~/.aws`, and the full scratch volume for local dev; a
# self-hosted CI job running contributor code drops the secrets with:
#   make test-gpu-core ENV_FILE= AWS_DIR=      # creds-free (still mounts $(HALO_SCRATCH) read-write)
# HALO_SCRATCH is the one home for this host's large volume, shared with the vLLM compose file and the
# devcontainer: the bind mount, the in-container cache/temp env, and the `clean` prune all derive
# from it, so a host whose big disk is not /mnt exports or overrides it once.
HALO_SCRATCH ?= /mnt
ENV_FILE  ?= .env
AWS_DIR   ?= ~/.aws
MNT_MOUNT ?= -v $(HALO_SCRATCH):$(HALO_SCRATCH)

# Shared `docker run` flags for anything that needs the image. $(CURDIR) is the Make builtin for
# the repo root — a bare $(PWD) is an undefined Make variable (empty), not shell substitution.
# CUDA_DEVICE_MAX_CONNECTIONS=1 is also baked into the image ENV; setting it here too keeps the
# EP throughput default (+9.7% on ep8) on an older pulled image that predates the baked ENV.
# --network host: the vLLM e2e tests (online/env GRPO) reach the compose server at localhost:8000;
# bridge networking would silently fail them (torchrun master ports are unique per manifest entry).
# On a host whose docker default runtime rejects --gpus/--ipc host (e.g. sysbox-runc),
# set DOCKER_RUNTIME=nvidia to pin the NVIDIA runtime explicitly; empty means the host default.
DOCKER_RUNTIME ?=
DOCKER_RUN = docker run --rm $(if $(strip $(DOCKER_RUNTIME)),--runtime $(DOCKER_RUNTIME),) --gpus all --network host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=128g \
  $(if $(strip $(ENV_FILE)),--env-file $(ENV_FILE),) \
  -e HF_HOME=$(HALO_SCRATCH)/hf -e HF_DATASETS_CACHE=$(HALO_SCRATCH)/hf/datasets \
  -e TMPDIR=$(HALO_SCRATCH)/tmp -e HALO_DATA_ROOT=$(HALO_SCRATCH) \
  -e PYTHONPATH=/workspace -e CUDA_DEVICE_MAX_CONNECTIONS=1 $(EXTRA_DOCKER_ENV) \
  -v $(CURDIR):/workspace $(MNT_MOUNT) $(if $(strip $(AWS_DIR)),-v $(AWS_DIR):/root/.aws,) -w /workspace \
  $(IMAGE)
# Per-target additions to the run above (see test-gpu-vllm).
EXTRA_DOCKER_ENV ?=
# CPU-only variant (no --gpus): CPU tests, lint, docs inside the image. The HF cache is mounted
# because ~60 CPU tests load a real tokenizer: without it they HARD-FAIL (not skip) the moment the
# hub is unreachable, which is exactly the state the self-hosted CPU runner can be in. Mounted
# read-WRITE (like DOCKER_RUN) so a cache miss can populate rather than erroring on a read-only FS.
# Override HF_CACHE= to run genuinely cache-less.
HF_CACHE ?= $(HALO_SCRATCH)/hf
DOCKER_RUN_CPU = docker run --rm $(if $(strip $(DOCKER_RUNTIME)),--runtime $(DOCKER_RUNTIME),) \
  $(if $(strip $(HF_CACHE)),-e HF_HOME=$(HF_CACHE) -v $(HF_CACHE):$(HF_CACHE),) \
  -e PYTHONPATH=/workspace -v $(CURDIR):/workspace -w /workspace $(IMAGE)

.DEFAULT_GOAL := help
.PHONY: help install lint format precommit test-cpu test-gpu-core test-gpu-full test-gpu-vllm test-gpu-sglang bench \
        docs docs-serve diagrams build-blackwell build-hopper build-vllm build-sglang build-all \
        require-ecr-registry ecr-login push-blackwell push-hopper push-vllm push-sglang push-all \
        ecr-public-login push-public-blackwell push-public-hopper push-public-vllm \
        push-public-sglang push-public-all train clean

help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## sync deps inside the image (uv pip install from the lock; preserves compiled torch/FA/DeepEP)
	$(DOCKER_RUN_CPU) bash -lc "uv pip install --system --break-system-packages -e . \
	  -r <(uv export --locked --no-emit-project --no-hashes --extra gigatoken --extra flash-optimizers --format requirements-txt)"

lint: ## ruff check (pinned binary on host)
	uvx ruff@$(RUFF_VERSION) check . || ruff check .

format: ## ruff format the tree
	uvx ruff@$(RUFF_VERSION) format . || ruff format .

precommit: lint ## format-check + lint (CI gate)
	uvx ruff@$(RUFF_VERSION) format --check . || ruff format --check .

test-cpu: ## pytest CPU tier inside the image
	$(DOCKER_RUN_CPU) bash -lc "pytest -m cpu tests/cpu $(PYTEST_ARGS)"

# Both entrypoints are named explicitly: pointing pytest at `tests/gpu/` would collect the manifest
# scripts as modules (executing their top-level torchrun code), which the launcher design avoids.
GPU_ENTRYPOINTS = tests/gpu/test_suite.py tests/gpu/test_launcher_contract.py

test-gpu-core: ## pytest core GPU tier (pre-merge, GPU changes) via the manifest launcher
	$(DOCKER_RUN) bash -lc "pytest -m 'gpu and core' $(GPU_ENTRYPOINTS) $(PYTEST_ARGS)"

test-gpu-full: ## pytest full GPU tier (heavy, many-GPU)
	$(DOCKER_RUN) bash -lc "pytest -m gpu $(GPU_ENTRYPOINTS) $(PYTEST_ARGS)"

# GPUs the TRAINER may use — must exclude the server's (VLLM_CUDA_DEVICES in docker-compose.vllm.yml):
# weight sync is an NCCL broadcast, and a rank cannot broadcast to itself.
TRAINER_CUDA_DEVICES ?= 0,1,2,3,4,5,6
# These tests broadcast the trainer's weights into the served model and assert the served policy
# MOVED, so both ends must be the same checkpoint and no one server satisfies both tiers. Run each
# in two passes, restarting the server in between (the MoE halves need different checkpoints —
# SGLang gathers only GptOss's fused expert layout):
#   dense:        VLLM_MODEL/SGLANG_MODEL=Qwen/Qwen3-0.6B          make test-gpu-vllm / test-gpu-sglang
#   MoE (vLLM):   VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507      make test-gpu-vllm SERVER_TIER=moe
#   MoE (SGLang): SGLANG_MODEL=unsloth/gpt-oss-20b-BF16            make test-gpu-sglang SERVER_TIER=moe
SERVER_TIER ?= not moe
# NCCL_IB_DISABLE/NCCL_NET: the trainer↔server weight-transfer group is NCCL over the host network.
# On a host without InfiniBand the image's OFI/Gin defaults wedge its first collective (both GPUs
# spin at 100% until the 120 s formation deadline) rather than falling back, so force sockets.
test-gpu-vllm: EXTRA_DOCKER_ENV = -e NCCL_IB_DISABLE=1 -e NCCL_NET=Socket \
  -e CUDA_VISIBLE_DEVICES=$(TRAINER_CUDA_DEVICES) \
  -e VLLM_SERVER_URL=$(VLLM_SERVER_URL) -e HALO_TEST_REQUIRE_SERVER=vllm
test-gpu-vllm: ## pytest the vLLM-server GPU tier (server on a GPU outside TRAINER_CUDA_DEVICES; SERVER_TIER=moe for the MoE half)
	@curl -sf $(VLLM_SERVER_URL)/health >/dev/null || { echo "No vLLM server at $(VLLM_SERVER_URL). Start it on a \
	  GPU the trainer does not use: VLLM_CUDA_DEVICES=7 VLLM_REASONING_PARSER=qwen3 \
	  VLLM_USE_V2_MODEL_RUNNER=0 docker compose -f docker-compose.vllm.yml up -d vllm-server \
	  (both are required by the benchmarks: their per-effort CoT budget draws a 400 without a \
	  reasoning parser, and another under Model Runner V2)"; exit 1; }
	$(DOCKER_RUN) bash -lc "pytest -m 'gpu and vllm_server and ($(SERVER_TIER))' $(GPU_ENTRYPOINTS) $(PYTEST_ARGS)"

# NCCL_P2P_DISABLE/NCCL_SHM_DISABLE: the trainer and SGLang run in separate containers, and NCCL
# takes the same-node CUDA-IPC P2P path whose shareable-buffer import fails across that boundary
# (`ncclP2pImportShareableBuffer ... invalid argument`), leaving the engine partially written. The
# server sets the same five (docker-compose.sglang.yml).
# NCCL_NET_PLUGIN=none: the bundled aws-ofi plugin outranks the socket transport and wedges group
# formation on a host with no OFI fabric, rather than falling back.
test-gpu-sglang: EXTRA_DOCKER_ENV = -e NCCL_IB_DISABLE=1 -e NCCL_NET=Socket -e NCCL_NET_PLUGIN=none \
  -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 -e CUDA_VISIBLE_DEVICES=$(TRAINER_CUDA_DEVICES) \
  -e SGLANG_SERVER_URL=$(SGLANG_SERVER_URL) -e HALO_TEST_REQUIRE_SERVER=sglang
test-gpu-sglang: ## pytest the SGLang-server GPU tier (server on a GPU outside TRAINER_CUDA_DEVICES; SERVER_TIER=moe for the MoE half)
	@curl -sf $(SGLANG_SERVER_URL)/health >/dev/null || { echo "No SGLang server at $(SGLANG_SERVER_URL). Start it on a \
	  GPU the trainer does not use: SGLANG_CUDA_DEVICES=7 SGLANG_MODEL=Qwen/Qwen3-0.6B docker compose -f docker-compose.sglang.yml up -d"; exit 1; }
	$(DOCKER_RUN) bash -lc "pytest -m 'gpu and sglang_server and ($(SERVER_TIER))' $(GPU_ENTRYPOINTS) $(PYTEST_ARGS)"

bench: ## run the EP/TP throughput benchmarks
	$(DOCKER_RUN) bash -lc "./tests/gpu/profiling/run_ep_tp_benchmarks.sh --gpus=$(NPROC)"

docs: ## mkdocs build --strict (agent-docs/, in-image) + relative-link check (human-docs, skills, root markdown)
	$(DOCKER_RUN_CPU) bash -lc "mkdocs build --strict"
	./scripts/docs/check_links.sh

docs-serve: ## mkdocs serve on :8000 (in-image; --network host to reach it)
	docker run --rm --network host \
	  -e PYTHONPATH=/workspace -v $(CURDIR):/workspace -w /workspace $(IMAGE) \
	  bash -lc "mkdocs serve --dev-addr 0.0.0.0:8000"

# Each generator writes its figures at import time and most carry no `__main__` guard, so they are
# run one per `python` invocation rather than imported. `_style_base.py` / `_flow_style.py` / `_theory_style.py` are
# shared style modules, not generators — the `gen_*.py` glob leaves them out.
diagrams: ## regenerate agent-docs/assets figures from scripts/diagrams (in-image; matplotlib ships there)
	$(DOCKER_RUN_CPU) bash -lc 'set -e; for g in scripts/diagrams/gen_*.py; do echo "$$g"; python "$$g"; done'

build-blackwell: ## build the Blackwell image (credential-free; every dep is public)
	docker build -t halo:blackwell \
	  --build-arg TARGET_GPU=blackwell --build-arg SOURCE_REVISION=$(SOURCE_REVISION) \
	  --build-arg VERSION=$(VERSION) .

build-hopper: ## build the Hopper image (TARGET_GPU=hopper is the Dockerfile default)
	docker build -t halo:hopper \
	  --build-arg TARGET_GPU=hopper --build-arg SOURCE_REVISION=$(SOURCE_REVISION) \
	  --build-arg VERSION=$(VERSION) .

BUILD_DATE ?= $(shell date -u +%Y-%m-%d)

build-vllm: ## build the vLLM inference image (credential-free; NCCL pinned from uv.lock)
	docker build -f Dockerfile.vllm -t vllm-server:0.26.0 \
	  --build-arg VERSION=$(VERSION) --build-arg BUILD_DATE=$(BUILD_DATE) .

build-sglang: ## build the SGLang inference image (NCCL pinned from uv.lock, matching the training images)
	docker build -f Dockerfile.sglang -t sglang-server:0.5.17 \
	  --build-arg VERSION=$(VERSION) --build-arg BUILD_DATE=$(BUILD_DATE) .

build-all: build-blackwell build-hopper build-vllm build-sglang ## build all four images

# --- Publishing to a container registry -------------------------------------------------
# Each image gets a moving tag (halo-blackwell) and an immutable SemVer tag
# (halo-blackwell-$(VERSION)). Point ECR_REGISTRY/ECR_REPO at your own registry (e.g. AWS ECR)
# and authenticate (`make ecr-login` needs a valid AWS session). Override on the CLI:
# `make push-blackwell ECR_REGISTRY=... ECR_REPO=...`.
ECR_REGION   ?= us-east-1
ECR_REGISTRY ?= $(ECR_ACCOUNT_ID).dkr.ecr.$(ECR_REGION).amazonaws.com
ECR_REPO     ?= halo
# Unset by design: these targets publish to YOUR private registry, and there is no sane default.
# Angle-bracket placeholders cannot be used here — `<` and `>` are shell redirections inside a
# recipe, so an unset value would fail with a mangled redirect instead of an actionable message.
ECR_ACCOUNT_ID ?=

# An explicit ECR_REGISTRY is a complete answer on its own — the account id only exists to build
# the default AWS registry host, so requiring it as well would refuse the documented override.
require-ecr-registry:
	@test -n "$(ECR_ACCOUNT_ID)" -o "$(origin ECR_REGISTRY)" != "file" || { \
	  echo "ECR_ACCOUNT_ID is unset. These targets push to your own private ECR:"; \
	  echo "  make $(MAKECMDGOALS) ECR_ACCOUNT_ID=123456789012 ECR_REGION=us-east-1"; \
	  echo "Or override ECR_REGISTRY directly:"; \
	  echo "  make $(MAKECMDGOALS) ECR_REGISTRY=ghcr.io/you ECR_REPO=halo"; \
	  echo "To pull the prebuilt public images instead, see"; \
	  echo "  docker pull public.ecr.aws/whitecircle/halo:blackwell"; \
	  exit 1; }

ecr-login: require-ecr-registry ## authenticate docker to the container registry
	aws ecr get-login-password --region $(ECR_REGION) \
	  | docker login --username AWS --password-stdin $(ECR_REGISTRY)

push-blackwell: ecr-login ## push halo:blackwell -> registry (halo-blackwell + halo-blackwell-$(VERSION))
	docker tag halo:blackwell $(ECR_REGISTRY)/$(ECR_REPO):halo-blackwell
	docker tag halo:blackwell $(ECR_REGISTRY)/$(ECR_REPO):halo-blackwell-$(VERSION)
	docker push $(ECR_REGISTRY)/$(ECR_REPO):halo-blackwell
	docker push $(ECR_REGISTRY)/$(ECR_REPO):halo-blackwell-$(VERSION)

push-hopper: ecr-login ## push halo:hopper -> registry (halo-hopper + halo-hopper-$(VERSION))
	docker tag halo:hopper $(ECR_REGISTRY)/$(ECR_REPO):halo-hopper
	docker tag halo:hopper $(ECR_REGISTRY)/$(ECR_REPO):halo-hopper-$(VERSION)
	docker push $(ECR_REGISTRY)/$(ECR_REPO):halo-hopper
	docker push $(ECR_REGISTRY)/$(ECR_REPO):halo-hopper-$(VERSION)

push-vllm: ecr-login ## push vllm-server:0.26.0 -> registry (moving + SemVer tag)
	docker tag vllm-server:0.26.0 $(ECR_REGISTRY)/$(ECR_REPO):vllm-server-0.26.0
	docker tag vllm-server:0.26.0 $(ECR_REGISTRY)/$(ECR_REPO):vllm-server-0.26.0-$(VERSION)
	docker push $(ECR_REGISTRY)/$(ECR_REPO):vllm-server-0.26.0
	docker push $(ECR_REGISTRY)/$(ECR_REPO):vllm-server-0.26.0-$(VERSION)

push-sglang: ecr-login ## push sglang-server:0.5.17 -> registry (moving + SemVer tag)
	docker tag sglang-server:0.5.17 $(ECR_REGISTRY)/$(ECR_REPO):sglang-server-0.5.17
	docker tag sglang-server:0.5.17 $(ECR_REGISTRY)/$(ECR_REPO):sglang-server-0.5.17-$(VERSION)
	docker push $(ECR_REGISTRY)/$(ECR_REPO):sglang-server-0.5.17
	docker push $(ECR_REGISTRY)/$(ECR_REPO):sglang-server-0.5.17-$(VERSION)

push-all: push-blackwell push-hopper push-vllm push-sglang ## push all four images to a private registry

# --- Publishing to Amazon ECR Public (gallery.ecr.aws) ----------------------------------
# A DIFFERENT service from the private ECR above: its own IAM actions (`ecr-public:*` plus
# `sts:GetServiceBearerToken`), auth always in us-east-1 whatever the repo region, and the
# result is world-readable. Only the images built from this repo are published here.
# Tags are arch-qualified because one repo serves both GPU targets; deliberately no
# `latest`, which would let a Hopper host silently pull a Blackwell image.
ECR_PUBLIC_HOST ?= public.ecr.aws
# Registry alias (the registry's numeric id resolves to the same images).
ECR_PUBLIC_NS   ?= whitecircle
ECR_PUBLIC_REPO ?= halo
ECR_PUBLIC      = $(ECR_PUBLIC_HOST)/$(ECR_PUBLIC_NS)/$(ECR_PUBLIC_REPO)

ecr-public-login: ## authenticate docker to ECR Public (region is always us-east-1)
	aws ecr-public get-login-password --region us-east-1 \
	  | docker login --username AWS --password-stdin $(ECR_PUBLIC_HOST)

push-public-blackwell: ecr-public-login ## publish halo:blackwell -> ECR Public (blackwell + blackwell-$(VERSION))
	./docker/scan_image.sh halo:blackwell
	docker tag halo:blackwell $(ECR_PUBLIC):blackwell
	docker tag halo:blackwell $(ECR_PUBLIC):blackwell-$(VERSION)
	docker push $(ECR_PUBLIC):blackwell
	docker push $(ECR_PUBLIC):blackwell-$(VERSION)

push-public-hopper: ecr-public-login ## publish halo:hopper -> ECR Public (hopper + hopper-$(VERSION))
	./docker/scan_image.sh halo:hopper
	docker tag halo:hopper $(ECR_PUBLIC):hopper
	docker tag halo:hopper $(ECR_PUBLIC):hopper-$(VERSION)
	docker push $(ECR_PUBLIC):hopper
	docker push $(ECR_PUBLIC):hopper-$(VERSION)

push-public-vllm: ecr-public-login ## publish vllm-server:0.26.0 -> ECR Public (vllm-0.26.0 + -$(VERSION))
	./docker/scan_image.sh vllm-server:0.26.0
	docker tag vllm-server:0.26.0 $(ECR_PUBLIC):vllm-0.26.0
	docker tag vllm-server:0.26.0 $(ECR_PUBLIC):vllm-0.26.0-$(VERSION)
	docker push $(ECR_PUBLIC):vllm-0.26.0
	docker push $(ECR_PUBLIC):vllm-0.26.0-$(VERSION)

push-public-sglang: ecr-public-login ## publish sglang-server:0.5.17 -> ECR Public (sglang-0.5.17 + -$(VERSION))
	./docker/scan_image.sh sglang-server:0.5.17
	docker tag sglang-server:0.5.17 $(ECR_PUBLIC):sglang-0.5.17
	docker tag sglang-server:0.5.17 $(ECR_PUBLIC):sglang-0.5.17-$(VERSION)
	docker push $(ECR_PUBLIC):sglang-0.5.17
	docker push $(ECR_PUBLIC):sglang-0.5.17-$(VERSION)

push-public-all: push-public-blackwell push-public-hopper push-public-vllm push-public-sglang ## publish all four to ECR Public

train: ## run a training config: make train CONFIG=... [METHOD=sft] NPROC=8 EXTRA="--expert_parallel_size=8"
	$(DOCKER_RUN) bash -lc "python -m src.cli launch $(strip $(METHOD)) $(CONFIG) \
	  --nproc $(NPROC) -- $(EXTRA)"

# checkpoints/ is NOT pruned: it is the default output_dir of every shipped example, so a run's
# weights live there and an unprompted rm would delete them. Remove a finished run by hand.
clean: ## prune wandb/, the ruff/pytest caches and $(HALO_SCRATCH) test scratch (leaves checkpoints/)
	rm -rf wandb/ .ruff_cache .pytest_cache
	rm -rf $(HALO_SCRATCH)/tmp/*_cache_r* $(HALO_SCRATCH)/tmp/*_out_* 2>/dev/null || true
