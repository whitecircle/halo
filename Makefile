# Halo — common Docker build, training, verification and publishing targets. Everything that
# executes runs inside the image; pick the one for your GPU:
#   make ... IMAGE=halo:blackwell   # B200/B300 (default)
#   make ... IMAGE=halo:hopper      # H100/H200
# Lint/format and the docs link check run on the host.

# bash, not dash (the default /bin/sh on Debian/Ubuntu), so a recipe may use bash syntax.
SHELL := /bin/bash

IMAGE        ?= halo:blackwell
# SemVer for the versioned image tags the push targets publish alongside the moving tags.
VERSION      ?= 1.2.0
# The engine release a server image builds, read off its Dockerfile ARG (the one place it is pinned).
VLLM_VERSION   := $(shell sed -n 's/^ARG VLLM_VERSION=//p' Dockerfile.vllm)
SGLANG_VERSION := $(shell sed -n 's/^ARG SGLANG_VERSION=//p' Dockerfile.sglang)
NPROC        ?= 8
# Training method name for `make train` (any `halo launch --list` entry).
METHOD       ?= sft
# ruff at uv.lock's pin, as CI runs it (pre-commit's literal rev is held to the lock by a test).
RUFF_VERSION ?= $(shell sed -n '/^name = "ruff"$$/{n;s/^version = "\(.*\)"$$/\1/p;}' uv.lock)
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
# HALO_SCRATCH is the large host volume the bind mount, the in-container cache and temp env, and the
# `clean` prune derive from. Unset, it resolves as the docker.md launch recipe does: the local mount other
# than `/` with the most free space, over 20 GB; a target that needs it stops when none is found.
ifeq ($(origin HALO_SCRATCH),undefined)
HALO_SCRATCH := $(shell findmnt -rbno TARGET,AVAIL,FSTYPE 2>/dev/null | \
  awk '$$1!="/" && $$3!~/tmpfs|overlay|squashfs|nfs|fuse|autofs/ && $$2+0>20e9{print $$2,$$1}' | sort -rn | head -1 | awk '{print $$2}')
endif
# Lowercase: make re-exports an environment SCRATCH (common on clusters), which would run this guard
# in every recipe.
scratch_dir = $(or $(HALO_SCRATCH),$(error HALO_SCRATCH is empty and no local mount besides / has 20 GB free: set it to a large host path (findmnt / df -h)))
ENV_FILE  ?= .env
AWS_DIR   ?= ~/.aws
MNT_MOUNT ?= -v $(scratch_dir):$(scratch_dir)

# Shared GPU container settings for anything that needs the image. Host networking lets a job reach
# a rollout server started by docker compose on the same host; $(CURDIR) is the Make builtin for the
# repo root — a bare $(PWD) is an undefined Make variable (empty), not shell substitution.
# On a host whose docker default runtime rejects --gpus/--ipc host (e.g. sysbox-runc),
# set DOCKER_RUNTIME=nvidia to pin the NVIDIA runtime explicitly; empty means the host default.
DOCKER_RUNTIME ?=
DOCKER_RUN = docker run --rm $(if $(strip $(DOCKER_RUNTIME)),--runtime $(DOCKER_RUNTIME),) --gpus all --network host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=128g \
  $(if $(strip $(ENV_FILE)),--env-file $(ENV_FILE),) \
  -e HF_HOME=$(scratch_dir)/hf -e HF_DATASETS_CACHE=$(scratch_dir)/hf/datasets \
  -e TMPDIR=$(scratch_dir)/tmp -e HALO_DATA_ROOT=$(scratch_dir) \
  -e PYTHONPATH=/workspace -e CUDA_DEVICE_MAX_CONNECTIONS=1 $(EFA_DOCKER_FLAGS) $(NCCL_PROTO_ENV) \
  $(SERVER_TIER_DOCKER_ENV) $(EQUIVALENCE_DOCKER_FLAGS) $(EXTRA_DOCKER_ENV) \
  -v $(CURDIR):/workspace $(MNT_MOUNT) $(if $(strip $(AWS_DIR)),-v $(AWS_DIR):/root/.aws,) -w /workspace \
  $(IMAGE)
# The server tiers' own flags, set per target (see test-gpu-vllm); kept apart from the caller's
# EXTRA_DOCKER_ENV so a caller's flags add to them instead of replacing them.
SERVER_TIER_DOCKER_ENV =
# Extra docker flags from the caller for this run and the CPU one below, e.g.
# EXTRA_DOCKER_ENV="-e HF_HUB_OFFLINE=1" for an offline CPU tier.
EXTRA_DOCKER_ENV ?=
# Fabric for NCCL in the container — the weight-sync group to a rollout server and every other
# trainer collective. EFA=1 passes the EFA devices and names the aws-ofi-nccl net, so a missing
# plugin fails loudly instead of falling back to sockets; the server must run under the matching
# compose overlay. Its interface default excludes lo (an excluded-only list still ranks loopback
# first, and a 127.0.0.1 bootstrap never reaches another node); the no-fabric default keeps lo for the
# same-host path. NCCL_SOCKET_IFNAME overrides either (make does not read .env); NCCL_PROTO passes through.
EFA ?=
NCCL_SOCKET_IFNAME ?=
EFA_SOCKET_IFNAME_DEFAULT = ^lo,docker,veth,tailscale
NO_FABRIC_SOCKET_IFNAME_DEFAULT = ^docker,veth
EFA_DOCKER_FLAGS = $(if $(filter 1,$(EFA)),--device=/dev/infiniband -e NCCL_NET=Libfabric -e NCCL_NET_PLUGIN=ofi \
  -e NCCL_IB_DISABLE=0 -e NCCL_SOCKET_IFNAME=$(or $(NCCL_SOCKET_IFNAME),$(EFA_SOCKET_IFNAME_DEFAULT)),)
NO_FABRIC_ENV = $(if $(filter 1,$(EFA)),,-e NCCL_IB_DISABLE=1 -e NCCL_NET=Socket \
  -e NCCL_SOCKET_IFNAME=$(or $(NCCL_SOCKET_IFNAME),$(NO_FABRIC_SOCKET_IFNAME_DEFAULT)))
NCCL_PROTO_ENV = $(if $(strip $(NCCL_PROTO)),-e NCCL_PROTO=$(NCCL_PROTO),)
# CPU-only variant (no --gpus): check-lock, the CPU tier, seed-hf-cache and diagrams. The Hugging Face
# cache is mounted so the tests that load a real tokenizer work without the hub, read-write so
# seed-hf-cache can fill it; set HF_CACHE= to disable that mount. check-lock and diagrams read no
# cache, so they run without it and without a scratch volume.
HF_CACHE ?= $(scratch_dir)/hf
check-lock diagrams: HF_CACHE =
DOCKER_RUN_CPU = docker run --rm $(if $(strip $(DOCKER_RUNTIME)),--runtime $(DOCKER_RUNTIME),) \
  $(if $(strip $(HF_CACHE)),-e HF_HOME=$(HF_CACHE) -v $(HF_CACHE):$(HF_CACHE),) \
  -e PYTHONPATH=/workspace $(EQUIVALENCE_DOCKER_FLAGS) $(EXTRA_DOCKER_ENV) -v $(CURDIR):/workspace -w /workspace $(IMAGE)
# Set by the equivalence targets, which run git in the image as the caller (see verify-move).
EQUIVALENCE_DOCKER_FLAGS =

.DEFAULT_GOAL := help
.PHONY: help check-lock lint format precommit test-cpu seed-hf-cache test-gpu-core test-gpu-full test-gpu-vllm test-gpu-sglang bench bench-band \
        verify-move verify-artifacts verify-ep verify-cp verify-efficiency docs diagrams build-blackwell build-hopper build-vllm build-sglang build-all \
        ecr-public-login push-public-blackwell push-public-hopper push-public-vllm \
        push-public-sglang push-public-all train clean

help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

check-lock: ## check uv.lock installs into the image (throwaway container; fails on a stale lock)
	$(DOCKER_RUN_CPU) bash -lc "set -euo pipefail; \
	  uv export --locked --no-emit-project --no-hashes --extra gigatoken --extra flash-optimizers \
	    --format requirements-txt -o /tmp/requirements.txt; \
	  uv pip install --system --break-system-packages --no-deps -r /tmp/requirements.txt; \
	  uv pip install --system --break-system-packages --no-deps -e ."

lint: ## ruff check on the host (pinned via uvx, else the ruff on PATH)
	if command -v uvx >/dev/null 2>&1; then uvx ruff@$(RUFF_VERSION) check .; else ruff check .; fi

format: ## ruff format the tree
	if command -v uvx >/dev/null 2>&1; then uvx ruff@$(RUFF_VERSION) format .; else ruff format .; fi

precommit: lint ## format-check + lint (CI gate)
	if command -v uvx >/dev/null 2>&1; then uvx ruff@$(RUFF_VERSION) format --check .; else ruff format --check .; fi

test-cpu: ## pytest CPU tier inside the image
	$(DOCKER_RUN_CPU) bash -lc "pytest -m cpu tests/cpu $(PYTEST_ARGS)"

seed-hf-cache: ## fetch the Hub configs and tokenizers the CPU tier reads into HF_CACHE (tests/common/hub_seed.py)
	@test -n "$(strip $(HF_CACHE))" || { echo "HF_CACHE is empty: there is no cache to seed"; exit 1; }
	$(DOCKER_RUN_CPU) bash -lc "HF_HUB_DISABLE_PROGRESS_BARS=1 python -m tests.common.hub_seed"

# Passed whole: tests/gpu/conftest.py collects only the LAUNCHER_ENTRYPOINTS of tests/gpu/manifest.py,
# the manifest launcher (test_suite.py, one torchrun launch per (script, args) row) and its contract
# tests.
GPU_TEST_DIR = tests/gpu

test-gpu-core: ## pytest core GPU tier (pre-merge, GPU changes) via the manifest launcher
	$(DOCKER_RUN) bash -lc "pytest -m 'gpu and core' $(GPU_TEST_DIR) $(PYTEST_ARGS)"

test-gpu-full: ## pytest full GPU tier (heavy, many-GPU)
	$(DOCKER_RUN) bash -lc "pytest -m gpu $(GPU_TEST_DIR) $(PYTEST_ARGS)"

# GPUs the trainer may use — must exclude the server's (VLLM_CUDA_DEVICES / SGLANG_CUDA_DEVICES in the
# compose files): weight sync is an NCCL broadcast, and a rank cannot broadcast to itself.
TRAINER_CUDA_DEVICES ?= 0,1,2,3,4,5,6
# Both ends of a weight-sync test must serve the same checkpoint, so the dense and MoE halves are
# separate passes with the server restarted in between; SERVER_TIER selects the pass.
#   dense: Qwen/Qwen3-0.6B on either engine (VLLM_MODEL / SGLANG_MODEL)
#   MoE:   SERVER_TIER='moe and not gptoss' with VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507, and
#          SERVER_TIER='moe and gptoss' with VLLM_MODEL / SGLANG_MODEL=unsloth/gpt-oss-20b-BF16
#   The Step-3.7 sync suite serves its own checkpoint (see its script header) and skips without HALO_TEST_STEP3P7_MODEL.
# The vLLM compose default parses tool calls with hermes, which reads JSON <tool_call> calls only: the
# gpt-oss pass serves with VLLM_GPTOSS_PARSERS below, and a per-family pass (HALO_TEST_ENV_GRPO_MODEL) with
# its family's parser from agent-docs/infrastructure/rollout-servers.md. Under a wrong one every tool call
# stays text, every draw episode scores 0 and the env-GRPO rows fail step_1_rewards_spread_within_a_group.
# SGLang's default parser (auto) reads the right one off the chat template.
SERVER_TIER ?= not moe
VLLM_GPTOSS_PARSERS = VLLM_TOOL_PARSER_PLUGIN=/opt/gpt_oss_text_tool_parser.py VLLM_TOOL_PARSER=gpt_oss_text \
  VLLM_REASONING_PARSER_PLUGIN=/opt/gpt_oss_reasoning_parser.py VLLM_REASONING_PARSER=openai_gptoss
# The trainer↔server weight-transfer group is NCCL between two containers. Without EFA=1 both ends
# use the socket recipe the compose bases default to (InfiniBand off, socket net; NCCL_SOCKET_IFNAME
# keeps it off Docker's bridge and the per-container veth pairs, which NCCL otherwise enumerates and
# cannot carry host-to-host traffic on). The SGLang server needs only cuMem parity on top
# (docker-compose.sglang.yml).
test-gpu-vllm: SERVER_TIER_DOCKER_ENV = $(NO_FABRIC_ENV) \
  -e CUDA_VISIBLE_DEVICES=$(TRAINER_CUDA_DEVICES) \
  -e VLLM_SERVER_URL=$(VLLM_SERVER_URL) -e HALO_TEST_REQUIRE_SERVER=vllm
test-gpu-vllm: ## pytest the vLLM-server GPU tier (server on a GPU outside TRAINER_CUDA_DEVICES; SERVER_TIER picks the dense or a MoE pass, see above; EFA=1 on an EFA host)
	@curl -sf $(VLLM_SERVER_URL)/health >/dev/null || { echo "No vLLM server at $(VLLM_SERVER_URL). Start it on a \
	  GPU the trainer does not use: make build-vllm (or set VLLM_IMAGE), then \
	  HALO_SCRATCH=$(HALO_SCRATCH) VLLM_CUDA_DEVICES=7 VLLM_REASONING_PARSER=qwen3 docker compose -f docker-compose.vllm.yml up -d vllm-server \
	  (the benchmarks need the reasoning parser: their per-effort CoT budget draws a 400 without one; \
	  the gpt-oss pass serves VLLM_MODEL=unsloth/gpt-oss-20b-BF16 $(VLLM_GPTOSS_PARSERS) in its place, \
	  another family its parsers from agent-docs/infrastructure/rollout-servers.md; \
	  EFA=1: add -f docker-compose.vllm.efa.yml)"; exit 1; }
	$(DOCKER_RUN) bash -lc "pytest -m 'gpu and vllm_server and ($(SERVER_TIER))' $(GPU_TEST_DIR) $(PYTEST_ARGS)"

test-gpu-sglang: SERVER_TIER_DOCKER_ENV = $(NO_FABRIC_ENV) \
  -e CUDA_VISIBLE_DEVICES=$(TRAINER_CUDA_DEVICES) \
  -e SGLANG_SERVER_URL=$(SGLANG_SERVER_URL) -e HALO_TEST_REQUIRE_SERVER=sglang
test-gpu-sglang: ## pytest the SGLang-server GPU tier (server on a GPU outside TRAINER_CUDA_DEVICES; SERVER_TIER picks the dense or a MoE pass, see above; EFA=1 on an EFA host)
	@curl -sf $(SGLANG_SERVER_URL)/health >/dev/null || { echo "No SGLang server at $(SGLANG_SERVER_URL). Start it on a \
	  GPU the trainer does not use: make build-sglang (or set SGLANG_IMAGE), then \
	  SGLANG_CUDA_DEVICES=7 SGLANG_MODEL=Qwen/Qwen3-0.6B SGLANG_MODEL_DIR=$(HALO_SCRATCH) docker compose -f docker-compose.sglang.yml up -d \
	  (EFA=1: add -f docker-compose.sglang.efa.yml)"; exit 1; }
	$(DOCKER_RUN) bash -lc "pytest -m 'gpu and sglang_server and ($(SERVER_TIER))' $(GPU_TEST_DIR) $(PYTEST_ARGS)"

# make bench runs the rows of tests/gpu/profiling/run_baseline_benchmarks.sh on one 8-GPU node BENCH_REPEATS
# times, one log per repeat under evidence/. BAND=1 then holds the run to the committed baselines in
# tests/baselines/bench; RECORD=1 makes those baselines the run's, which takes a clean tree (a baseline names
# the commit it was measured at) and the 5 repeats it defaults to, MIN_REPEATS in
# tests/common/equivalence/bench_band.py.
BAND ?=
RECORD ?=
BENCH_MODE ?= $(if $(filter 1,$(RECORD)),record,check)
BENCH_REPEATS ?= $(if $(filter record,$(BENCH_MODE)),5,1)
BENCH_REVISION = $(shell git rev-parse HEAD 2>/dev/null)$(if $(shell git status --porcelain --untracked-files=no 2>/dev/null),-dirty)
BENCH_LOGS = $(foreach repeat,$(shell seq $(BENCH_REPEATS)),evidence/bench-$(BENCH_REVISION)-$(repeat).log)

bench: ## throughput and peak-memory rows on one 8-GPU node (BAND=1: hold them to tests/baselines/bench; RECORD=1: rewrite those)
	$(if $(and $(filter 1,$(RECORD)),$(findstring -dirty,$(BENCH_REVISION))),$(error RECORD=1 needs a clean tree: a baseline names the commit it was measured at))
	mkdir -p evidence
	set -o pipefail; for log in $(BENCH_LOGS); do \
	  $(DOCKER_RUN) bash -lc ./tests/gpu/profiling/run_baseline_benchmarks.sh | tee $$log || exit 1; \
	done
	$(if $(or $(filter 1,$(BAND)),$(filter 1,$(RECORD))),$(MAKE) --no-print-directory bench-band BENCH_LOGS="$(BENCH_LOGS)")

# bench-band runs as the caller, so the evidence and the rewritten baselines are the caller's.
bench-band: ## hold the logs make bench wrote at HEAD to tests/baselines/bench (BENCH_MODE=record: record them; no GPUs)
	$(DOCKER_RUN_CPU) python -m tests.common.equivalence.bench_band $(BENCH_MODE) $(BENCH_LOGS) \
	  --revision $(BENCH_REVISION) --image "$(IMAGE)@$(IMAGE_ID)"

# The equivalence harnesses compare HEAD_REF with BASE, both read from git, and write
# evidence/<harness>-<head sha>.json. They run as the caller, so the evidence is the caller's to remove,
# and git runs in the image: a worktree's checkout names its git directory outside the repo, so that
# directory is mounted at its own path, read-only, and git is told to trust a mount it does not own.
# Every harness but the move kit runs its own module in both trees, so BASE must already carry it.
BASE ?=
HEAD_REF ?= HEAD
# The move kit's declared renames: a JSON file under the repo, by its repo-relative path (only the repo
# is mounted; see tests/common/equivalence/move_kit.py).
RENAMES ?=
# Further move-kit flags, e.g. MOVE_ARGS="--require renames side_effects goldens --min-similarity 70".
MOVE_ARGS ?=
# The cases whose deltas a flagged change declares, space-separated: a case path or any prefix of one, e.g.
# ACCEPT="cpu/qwen3_moe" for verify-ep. Their deltas stay in the evidence without failing it.
ACCEPT ?=
# The tiny families a harness runs, space-separated (default: every family it covers).
FAMILIES ?=
# Where verify-ep and verify-cp run: cpu (the image, no GPUs) or gpu.
TIER ?= cpu
EQUIVALENCE_RUN = $(if $(filter gpu,$(TIER)),$(DOCKER_RUN),$(DOCKER_RUN_CPU))
GIT_COMMON_DIR = $(shell git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)
IMAGE_ID = $(shell docker image inspect --format '{{.Id}}' $(IMAGE) 2>/dev/null)
EQUIVALENCE_ARGS = --base-ref $(BASE) --head-ref $(HEAD_REF) --image "$(IMAGE)@$(IMAGE_ID)"
COMPARISON_ARGS = $(EQUIVALENCE_ARGS) $(if $(strip $(ACCEPT)),--accept $(ACCEPT))
FAMILY_ARGS = $(if $(strip $(FAMILIES)),--families $(FAMILIES))
verify-move verify-artifacts verify-ep verify-cp verify-efficiency bench-band: EQUIVALENCE_DOCKER_FLAGS = --user $(shell id -u):$(shell id -g) -e HOME=/tmp \
  $(if $(GIT_COMMON_DIR),-v $(GIT_COMMON_DIR):$(GIT_COMMON_DIR):ro) \
  -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e 'GIT_CONFIG_VALUE_0=*'

verify-move: ## move kit: HEAD_REF only moves BASE's code (BASE=<sha> [RENAMES=<map.json>] [MOVE_ARGS=...])
	@test -n "$(BASE)" || { echo "BASE=<commit> is required"; exit 1; }
	$(DOCKER_RUN_CPU) python -m tests.common.equivalence.move_kit $(EQUIVALENCE_ARGS) \
	  $(if $(RENAMES),--renames $(RENAMES)) $(MOVE_ARGS)

verify-artifacts: ## checkpoint artifacts of every CPU saver mode, BASE vs HEAD_REF (BASE=<sha> [FAMILIES=...] [ACCEPT=...])
	@test -n "$(BASE)" || { echo "BASE=<commit> is required"; exit 1; }
	$(DOCKER_RUN_CPU) python -m tests.common.equivalence.checkpoint_artifacts $(COMPARISON_ARGS) $(FAMILY_ARGS)

verify-ep: ## EP forward, backward and gradients bit for bit, BASE vs HEAD_REF (BASE=<sha> [TIER=cpu|gpu] [FAMILIES=...] [ACCEPT=...])
	@test -n "$(BASE)" || { echo "BASE=<commit> is required"; exit 1; }
	$(EQUIVALENCE_RUN) python -m tests.common.equivalence.ep_bitwise $(COMPARISON_ARGS) --tier $(TIER) $(FAMILY_ARGS)

verify-cp: ## Ulysses CP attention and model, BASE vs HEAD_REF (BASE=<sha> [TIER=cpu|gpu] [FAMILIES=...] [ACCEPT=...])
	@test -n "$(BASE)" || { echo "BASE=<commit> is required"; exit 1; }
	$(EQUIVALENCE_RUN) python -m tests.common.equivalence.cp_equivalence $(COMPARISON_ARGS) --tier $(TIER) $(FAMILY_ARGS)

verify-efficiency: ## what the EfficiencyCallback reports, BASE vs HEAD_REF (BASE=<sha> [ACCEPT=...])
	@test -n "$(BASE)" || { echo "BASE=<commit> is required"; exit 1; }
	$(DOCKER_RUN_CPU) python -m tests.common.equivalence.efficiency_callback $(COMPARISON_ARGS)

docs: ## links, anchors and code references in agent-docs/, human-docs/, skills/ and the root markdown
	./scripts/docs/check_links.sh

# One `python` per generator: most write their figures at import time with no `__main__` guard, and
# `_style_base.py` / `_theory_style.py` / `_pipeline_style.py` are shared style, not generators.
diagrams: ## regenerate agent-docs/assets figures from scripts/diagrams (in-image; matplotlib ships there)
	$(DOCKER_RUN_CPU) bash -lc 'set -e; for g in scripts/diagrams/gen_*.py; do echo "$$g"; python "$$g"; done'

BUILD_DATE ?= $(shell date -u +%Y-%m-%d)

build-blackwell: ## build the Blackwell image (credential-free; every dep is public)
	docker build -t halo:blackwell \
	  --build-arg TARGET_GPU=blackwell --build-arg SOURCE_REVISION=$(SOURCE_REVISION) \
	  --build-arg VERSION=$(VERSION) --build-arg BUILD_DATE=$(BUILD_DATE) .

build-hopper: ## build the Hopper image (TARGET_GPU=hopper is the Dockerfile default)
	docker build -t halo:hopper \
	  --build-arg TARGET_GPU=hopper --build-arg SOURCE_REVISION=$(SOURCE_REVISION) \
	  --build-arg VERSION=$(VERSION) --build-arg BUILD_DATE=$(BUILD_DATE) .

build-vllm: ## build the vLLM inference image halo:vllm-$(VLLM_VERSION) (credential-free; NCCL pinned from uv.lock)
	docker build -f Dockerfile.vllm -t halo:vllm-$(VLLM_VERSION) \
	  --build-arg VERSION=$(VERSION) --build-arg BUILD_DATE=$(BUILD_DATE) \
	  --build-arg SOURCE_REVISION=$(SOURCE_REVISION) .

build-sglang: ## build the SGLang inference image halo:sglang-$(SGLANG_VERSION) (NCCL pinned from uv.lock, matching the training images)
	docker build -f Dockerfile.sglang -t halo:sglang-$(SGLANG_VERSION) \
	  --build-arg VERSION=$(VERSION) --build-arg BUILD_DATE=$(BUILD_DATE) \
	  --build-arg SOURCE_REVISION=$(SOURCE_REVISION) .

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

push-public-vllm: ecr-public-login ## publish halo:vllm-$(VLLM_VERSION) -> ECR Public (vllm-$(VLLM_VERSION) + -$(VERSION))
	docker tag halo:vllm-$(VLLM_VERSION) $(ECR_PUBLIC):vllm-$(VLLM_VERSION)
	docker tag halo:vllm-$(VLLM_VERSION) $(ECR_PUBLIC):vllm-$(VLLM_VERSION)-$(VERSION)
	docker push $(ECR_PUBLIC):vllm-$(VLLM_VERSION)
	docker push $(ECR_PUBLIC):vllm-$(VLLM_VERSION)-$(VERSION)

push-public-sglang: ecr-public-login ## publish halo:sglang-$(SGLANG_VERSION) -> ECR Public (sglang-$(SGLANG_VERSION) + -$(VERSION))
	docker tag halo:sglang-$(SGLANG_VERSION) $(ECR_PUBLIC):sglang-$(SGLANG_VERSION)
	docker tag halo:sglang-$(SGLANG_VERSION) $(ECR_PUBLIC):sglang-$(SGLANG_VERSION)-$(VERSION)
	docker push $(ECR_PUBLIC):sglang-$(SGLANG_VERSION)
	docker push $(ECR_PUBLIC):sglang-$(SGLANG_VERSION)-$(VERSION)

push-public-all: push-public-blackwell push-public-hopper push-public-vllm push-public-sglang ## publish all four to ECR Public

train: ## run a training config: make train CONFIG=... [METHOD=sft] NPROC=8 EXTRA="--expert_parallel_size=8"
	$(DOCKER_RUN) bash -lc "python -m halo.cli launch $(strip $(METHOD)) $(CONFIG) \
	  --nproc $(NPROC) -- $(EXTRA)"

# clean does not remove checkpoints/: it is the default output_dir of every shipped example.
clean: ## prune wandb/, the ruff/pytest caches and $(HALO_SCRATCH) test scratch (leaves checkpoints/)
	rm -rf wandb/ .ruff_cache .pytest_cache
	$(if $(HALO_SCRATCH),rm -rf $(HALO_SCRATCH)/tmp/halo-test-* 2>/dev/null || true)
