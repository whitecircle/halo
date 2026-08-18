# Halo — common Docker build, training, verification and publishing targets.
#
# Everything that executes runs inside the image; pick the one for your GPU:
#   make ... IMAGE=halo:blackwell   # B200/B300 (default)
#   make ... IMAGE=halo:hopper      # H100/H200
#
# Lint/format and the docs link check run on the host.

# bash, not dash (the default /bin/sh on Debian/Ubuntu): several recipes use bashisms.
SHELL := /bin/bash

IMAGE        ?= halo:blackwell
# SemVer for the versioned image tags the push targets publish alongside the moving tags.
VERSION      ?= 1.0.0
NPROC        ?= 8
# Training method name for `make train` (any `halo launch --list` entry).
METHOD       ?= sft
RUFF_VERSION ?= 0.9.10
# Git SHA stamped into the image to bust the source-COPY cache (see the SOURCE_REVISION note in
# Dockerfile). `dev-nogit` if not in a git checkout.
SOURCE_REVISION ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev-nogit)
PYTEST_ARGS  ?=          # extra pytest flags, e.g. PYTEST_ARGS="--junitxml=cpu-junit.xml" (CI)
# Ports the compose services listen on and the URLs the tests dial: docker-compose.{vllm,sglang}.yml
# read the same `${VLLM_PORT}` / `${SGLANG_PORT}`, so exporting one moves the server and the URL together.
VLLM_PORT   ?= 8000
SGLANG_PORT ?= 30000
VLLM_SERVER_URL ?= http://localhost:$(VLLM_PORT)
SGLANG_SERVER_URL ?= http://localhost:$(SGLANG_PORT)

# Credential + data mounts. Set ENV_FILE= or AWS_DIR= to disable the credential mounts, e.g. for a
# CI job running contributor code:
#   make test-gpu-core ENV_FILE= AWS_DIR=      # creds-free (still mounts $(HALO_SCRATCH) read-write)
# HALO_SCRATCH must point at an existing large host path: the bind mount, the in-container cache and
# temp env, and the `clean` prune all derive from it.
HALO_SCRATCH ?= /mnt
ENV_FILE  ?= .env
AWS_DIR   ?= ~/.aws
MNT_MOUNT ?= -v $(HALO_SCRATCH):$(HALO_SCRATCH)

# Shared GPU container settings for anything that needs the image. Host networking lets a job reach
# a rollout server started by docker compose on the same host; $(CURDIR) is the Make builtin for the
# repo root — a bare $(PWD) is an undefined Make variable (empty), not shell substitution.
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
# CPU-only variant (no --gpus): CPU tests, lint, docs inside the image. The Hugging Face cache is
# mounted read-write so the tests that load a real tokenizer work without the hub; set HF_CACHE= to
# disable that mount.
HF_CACHE ?= $(HALO_SCRATCH)/hf
DOCKER_RUN_CPU = docker run --rm $(if $(strip $(DOCKER_RUNTIME)),--runtime $(DOCKER_RUNTIME),) \
  $(if $(strip $(HF_CACHE)),-e HF_HOME=$(HF_CACHE) -v $(HF_CACHE):$(HF_CACHE),) \
  -e PYTHONPATH=/workspace -v $(CURDIR):/workspace -w /workspace $(IMAGE)

.DEFAULT_GOAL := help
.PHONY: help install lint format precommit test-cpu test-gpu-core test-gpu-full test-gpu-vllm test-gpu-sglang bench \
        docs diagrams build-blackwell build-hopper build-vllm build-sglang build-all \
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

# GPUs the trainer may use — must exclude the server's (VLLM_CUDA_DEVICES in docker-compose.vllm.yml):
# weight sync is an NCCL broadcast, and a rank cannot broadcast to itself.
TRAINER_CUDA_DEVICES ?= 0,1,2,3,4,5,6
# Both ends of a weight-sync test must serve the same checkpoint, so the dense and MoE halves are
# separate passes with the server restarted in between; SERVER_TIER=moe selects the MoE half.
SERVER_TIER ?= not moe
# NCCL_IB_DISABLE/NCCL_NET: the trainer↔server weight-transfer group is NCCL over the host network.
# On a host without InfiniBand the image's OFI/Gin defaults hang on the first collective rather than
# falling back, so force sockets.
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

# NCCL_P2P_DISABLE/NCCL_SHM_DISABLE: the trainer and SGLang run in separate containers, across which
# NCCL's same-node CUDA-IPC path cannot import a shareable buffer. NCCL_NET_PLUGIN=none: the bundled
# aws-ofi plugin outranks the socket transport and wedges group formation on a host with no OFI
# fabric. The server sets the same five (docker-compose.sglang.yml).
test-gpu-sglang: EXTRA_DOCKER_ENV = -e NCCL_IB_DISABLE=1 -e NCCL_NET=Socket -e NCCL_NET_PLUGIN=none \
  -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 -e CUDA_VISIBLE_DEVICES=$(TRAINER_CUDA_DEVICES) \
  -e SGLANG_SERVER_URL=$(SGLANG_SERVER_URL) -e HALO_TEST_REQUIRE_SERVER=sglang
test-gpu-sglang: ## pytest the SGLang-server GPU tier (server on a GPU outside TRAINER_CUDA_DEVICES; SERVER_TIER=moe for the MoE half)
	@curl -sf $(SGLANG_SERVER_URL)/health >/dev/null || { echo "No SGLang server at $(SGLANG_SERVER_URL). Start it on a \
	  GPU the trainer does not use: SGLANG_CUDA_DEVICES=7 SGLANG_MODEL=Qwen/Qwen3-0.6B docker compose -f docker-compose.sglang.yml up -d"; exit 1; }
	$(DOCKER_RUN) bash -lc "pytest -m 'gpu and sglang_server and ($(SERVER_TIER))' $(GPU_ENTRYPOINTS) $(PYTEST_ARGS)"

bench: ## run the EP/TP throughput benchmarks
	$(DOCKER_RUN) bash -lc "./tests/gpu/profiling/run_ep_tp_benchmarks.sh --gpus=$(NPROC)"

docs: ## relative-link check over agent-docs/, human-docs/, skills/ and the root markdown
	./scripts/docs/check_links.sh

# Each generator writes its figures at import time and most carry no `__main__` guard, so they run
# one per `python` invocation.
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

# --- Publishing to Amazon ECR Public (gallery.ecr.aws) ----------------------------------
# Amazon ECR Public images are world-readable.
# ECR Public authentication uses us-east-1.
ECR_PUBLIC_HOST ?= public.ecr.aws
ECR_PUBLIC_NS   ?= whitecircle
ECR_PUBLIC_REPO ?= halo
ECR_PUBLIC      = $(ECR_PUBLIC_HOST)/$(ECR_PUBLIC_NS)/$(ECR_PUBLIC_REPO)

ecr-public-login: ## authenticate docker to ECR Public (region is always us-east-1)
	aws ecr-public get-login-password --region us-east-1 \
	  | docker login --username AWS --password-stdin $(ECR_PUBLIC_HOST)

push-public-blackwell: ecr-public-login ## publish halo:blackwell -> ECR Public (blackwell + blackwell-$(VERSION))
	docker tag halo:blackwell $(ECR_PUBLIC):blackwell
	docker tag halo:blackwell $(ECR_PUBLIC):blackwell-$(VERSION)
	docker push $(ECR_PUBLIC):blackwell
	docker push $(ECR_PUBLIC):blackwell-$(VERSION)

push-public-hopper: ecr-public-login ## publish halo:hopper -> ECR Public (hopper + hopper-$(VERSION))
	docker tag halo:hopper $(ECR_PUBLIC):hopper
	docker tag halo:hopper $(ECR_PUBLIC):hopper-$(VERSION)
	docker push $(ECR_PUBLIC):hopper
	docker push $(ECR_PUBLIC):hopper-$(VERSION)

push-public-vllm: ecr-public-login ## publish vllm-server:0.26.0 -> ECR Public (vllm-0.26.0 + -$(VERSION))
	docker tag vllm-server:0.26.0 $(ECR_PUBLIC):vllm-0.26.0
	docker tag vllm-server:0.26.0 $(ECR_PUBLIC):vllm-0.26.0-$(VERSION)
	docker push $(ECR_PUBLIC):vllm-0.26.0
	docker push $(ECR_PUBLIC):vllm-0.26.0-$(VERSION)

push-public-sglang: ecr-public-login ## publish sglang-server:0.5.17 -> ECR Public (sglang-0.5.17 + -$(VERSION))
	docker tag sglang-server:0.5.17 $(ECR_PUBLIC):sglang-0.5.17
	docker tag sglang-server:0.5.17 $(ECR_PUBLIC):sglang-0.5.17-$(VERSION)
	docker push $(ECR_PUBLIC):sglang-0.5.17
	docker push $(ECR_PUBLIC):sglang-0.5.17-$(VERSION)

push-public-all: push-public-blackwell push-public-hopper push-public-vllm push-public-sglang ## publish all four to ECR Public

train: ## run a training config: make train CONFIG=... [METHOD=sft] NPROC=8 EXTRA="--expert_parallel_size=8"
	$(DOCKER_RUN) bash -lc "python -m src.cli launch $(strip $(METHOD)) $(CONFIG) \
	  --nproc $(NPROC) -- $(EXTRA)"

# clean does not remove checkpoints/: it is the default output_dir of every shipped example.
clean: ## prune wandb/, the ruff/pytest caches and $(HALO_SCRATCH) test scratch (leaves checkpoints/)
	rm -rf wandb/ .ruff_cache .pytest_cache
	rm -rf $(HALO_SCRATCH)/tmp/*_cache_r* $(HALO_SCRATCH)/tmp/*_out_* 2>/dev/null || true
