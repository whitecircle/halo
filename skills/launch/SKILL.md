---
name: launch
description: >-
  Assemble and run a Halo training or test job inside the prebuilt
  Docker image. Picks the image (halo:blackwell for B200/B300 vs
  halo:hopper for H100/H200), sets GPU count, mounts ($(pwd)->/workspace,
  the host's large scratch volume, /root/.aws), passes --env-file .env, runs
  torchrun/pytest directly inside the container, and detaches the job with logs
  onto that volume.
  Prefers `make` targets (the single source of truth for the docker incantation)
  and falls back to a raw `docker run -d` template when no target fits.
  USER-INVOKED ONLY — invoke when the user explicitly asks to launch/run a
  training run, a GPU/CPU test, or a benchmark inside the image.
disable-model-invocation: true
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
---

# launch

Run training, tests, or benchmarks for **Halo** inside the prebuilt
Docker image. The host has **no usable Python** (PyTorch 2.11+cu130, DeepEP,
Flash Attention live only inside the images), so anything that *executes* must
run inside the container. The image is built with `uv` (`uv pip install` into the
system interpreter — no venv, no Poetry), so `python`/`torchrun`/`accelerate`/
`pytest` are on `PATH` — call them directly, **no prefix**.

## 1. ALWAYS prefer a `make` target first

The Docker incantation lives **once** in the repo `Makefile` — contributors,
CI, and this skill all call these targets instead of re-typing `docker run`.
Read the Makefile (`make help` or open `Makefile`)
and use a target if one fits. The relevant targets:

```bash
# Training — CONFIG is required; METHOD defaults to sft; NPROC defaults to 8;
# IMAGE defaults to halo:blackwell; EXTRA passes through extra CLI flags.
# The recipe routes through `python -m src.cli launch METHOD CONFIG` (the halo CLI
# picks python/torchrun/accelerate), so METHOD selects any scripts/training method.
make train CONFIG=examples/sft/qwen3/qwen3-4b-ultrachat.yaml
make train CONFIG=examples/sft/gptoss/gptoss-20b-multinode-ep.yaml \
           NPROC=8 EXTRA="--expert_parallel_size=8"
make train METHOD=offline-grpo CONFIG=examples/grpo/offline/gptoss/... \
           NPROC=8            # non-SFT method via METHOD=
make train CONFIG=examples/sft/qwen3/qwen3-4b-ultrachat.yaml \
           IMAGE=halo:hopper        # run on Hopper instead

# Tests
make test-cpu                                # pytest CPU tier (no GPU, runs in image)
make test-gpu-core                           # core GPU tier (PR gate)
make test-gpu-full                           # full GPU tier (nightly)

# Benchmarks
make bench                                   # EP/TP throughput benchmarks (NPROC GPUs)

# Image builds (credential-free — every dep is public)
make build-blackwell
make build-hopper
make build-vllm                              # vllm-server:0.26.0
make build-sglang                            # sglang-server:0.5.17 (NCCL matched to the training images)
make build-all                               # all four

# Rollout-server GPU tiers (the server must already serve on a GPU outside TRAINER_CUDA_DEVICES;
# both force NCCL_IB_DISABLE=1 NCCL_NET=Socket and check /health first; the SGLang target adds
# NCCL_NET_PLUGIN=none NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1).
# SERVER_TIER=moe runs the MoE half against a MoE-serving server — no one server satisfies both.
# The vLLM server needs VLLM_REASONING_PARSER=qwen3 VLLM_USE_V2_MODEL_RUNNER=0: the benchmarks
# send a per-effort CoT budget, which draws a 400 without a reasoning parser and another under
# Model Runner V2. The health guard prints the exact compose line when it is missing.
make test-gpu-vllm
make test-gpu-sglang

# Host-only gates (pinned uvx ruff, no image)
make lint
make format
make precommit

# Docs link check (plain GitHub markdown; runs on the host)
make docs
make diagrams                                # regenerate agent-docs/assets from scripts/diagrams (CI byte-compares)
make install

make clean                                   # prune wandb/, the ruff and pytest caches, $(HALO_SCRATCH) test scratch (leaves checkpoints/)
```

Notes that come straight from the Makefile recipes:

- `make train` selects the method via `METHOD=` (default `sft`) and dispatches
  through the `halo` CLI (`python -m src.cli launch METHOD CONFIG`), which resolves
  the right `scripts/training/` script and launcher. Method names come from
  `halo launch --list` (e.g. `sft`, `offline-grpo`, `classification`, `embedding`,
  `environmental-grpo`, and nested names like `preference/smpo`). For a raw one-off
  or an unusual flag layout the target doesn't fit, use the template in §2.
- Every GPU target already bakes in `--gpus all --network host --ipc=host --ulimit
  memlock=-1 --ulimit stack=67108864 --shm-size=128g`, the `-e HF_HOME /
  HF_DATASETS_CACHE / TMPDIR / HALO_DATA_ROOT / PYTHONPATH /
  CUDA_DEVICE_MAX_CONNECTIONS=1` env, and the `$(pwd):/workspace`, `$(HALO_SCRATCH):$(HALO_SCRATCH)`,
  `/root/.aws:/root/.aws` mounts. `--env-file .env` and the `~/.aws` mount are
  conditional on `ENV_FILE`/`AWS_DIR` (CI drops both to run creds-free). Don't
  re-specify those by hand when a target works. `DOCKER_RUN` does **not** add
  `--cap-add=SYS_PTRACE`, so py-spy cannot attach to a `make`-launched job — use §2
  when you may need hang triage.
- **`HALO_SCRATCH` (default `/mnt`) is the one override for this host's large volume** — the
  bind mount, the in-container `HF_HOME`/`HF_DATASETS_CACHE`/`TMPDIR`/`HALO_DATA_ROOT`,
  and `make clean`'s prune all derive from it. `/mnt` is **not** guaranteed large (on
  some hosts it shares the small root device), so verify with `findmnt`/`df -h` and pass
  `HALO_SCRATCH=/path/to/big/volume` when it isn't. `docker-compose.vllm.yml` and the devcontainer
  read the same variable.
- `make` targets run in the **foreground** (`docker run --rm`, no `-d`). For a
  long detached job that survives the turn and logs to the scratch volume, use §2.

## 2. Raw detached `docker run` template (when `make` doesn't fit)

This is the incantation from `CLAUDE.md`, parameterized. Use it for non-SFT
scripts, custom flags, disjoint-GPU runs, or long detached jobs. Fill in `<job>`
(container + log name), `<nproc>`, `<script>`, `<config>`, and any extra flags.
`$D` is the scratch volume, **resolved dynamically — never hardcode a path**: a
directory's name does not prove its capacity.

```bash
# Resolve the scratch volume dynamically (see Preflight in CLAUDE.md).
D=$(findmnt -rbno TARGET,AVAIL,FSTYPE | awk '$3!~/tmpfs|overlay|squashfs|nfs|fuse|autofs/ && $2+0>20e9{print $2,$1}' | sort -rn | head -1 | awk '{print $2}')
docker run -d --rm --name <job> --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=128g \
  --cap-add=SYS_PTRACE --env-file .env \
  -e HF_HOME="$D/hf" -e HF_DATASETS_CACHE="$D/hf/datasets" \
  -e TMPDIR="$D/tmp" -e HALO_DATA_ROOT="$D" \
  -v $(pwd):/workspace -v "$D:$D" -v /root/.aws:/root/.aws -w /workspace \
  halo:blackwell \
  bash -lc "torchrun --nproc_per_node=<nproc> <script> <config> > \"$D/<job>.log\" 2>&1"
```

Mandatory pieces (do NOT drop them):

- **`--env-file .env`** — the repo-root `.env` supplies `WANDB_API_KEY`,
  `HF_TOKEN`, `AWS_DEFAULT_REGION`. The code does **not** auto-load it.
- **No prefix** inside the container — call `python`/`torchrun`/`accelerate`/
  `pytest` directly (they are on `PATH`; the image is uv-built, no Poetry, no venv).
- **`-v $(pwd):/workspace -w /workspace`** — the code mounts live; edits on the
  host take effect immediately.
- **`-v "$D:$D"`** — outputs, logs, HF cache all go here. Verify `$D` is a large
  volume before any multi-GB write (`findmnt` / `df -h` / `readlink -f`); the root FS
  is small and on some hosts shares a device with `/mnt`.
- **`--cap-add=SYS_PTRACE`** — lets py-spy attach for hang triage (`agent-docs/reference/debugging.md`).
- **`-e HALO_DATA_ROOT="$D"`** — toolkit scratch root: S3 dataset cache →
  `$D/s3_datasets`, profiler artifacts → `$D/profiling` (one knob; defaults to
  `~/.cache/halo` when unset).
- **`-v /root/.aws:/root/.aws`** — S3 dataset/checkpoint access. Pre-cached S3
  datasets live under `$HALO_DATA_ROOT/s3_datasets` (keyed by `md5("<bucket>/<key>")`)
  and load without live AWS.

Single-GPU / accelerate / pytest variants just swap the `bash -lc "..."` body,
e.g.:

```bash
# Single GPU, plain python
bash -lc "python <script> <config> > \"$D/<job>.log\" 2>&1"

# Accelerate DDP / FSDP (prefer the FSDP2 config; CLAUDE.md flags an Accelerate FSDP-v1 corruption bug)
bash -lc "accelerate launch --config_file launcher-configs/accelerate/fsdp2_gradop_config.yaml \
          <script> <config> > \"$D/<job>.log\" 2>&1"

# A specific pytest selection not covered by the make targets
bash -lc "pytest -m gpu tests/gpu/test_suite.py -k <pattern> > \"$D/<job>.log\" 2>&1"
```

A **CPU-only** job (CPU tests, lint-in-image, one-off Python with no GPU) drops
`--gpus all` and the GPU ulimits/shm, matching the Makefile's `DOCKER_RUN_CPU`:

```bash
docker run -d --rm --name <job> \
  -e PYTHONPATH=/workspace -e HF_HOME="$D/hf" \
  -v $(pwd):/workspace -v "$D:$D" -w /workspace \
  halo:blackwell \
  bash -lc "pytest -m cpu tests/cpu > \"$D/<job>.log\" 2>&1"
```

Both mounts are load-bearing: without `-v "$D:$D"` the log is written into the
container's ephemeral filesystem and destroyed by `--rm`, and without the HF cache
~60 CPU tests **hard-fail** (not skip) the moment the Hub is unreachable.

## 3. Image selection — by GPU architecture

| GPU | Arch | Image |
|-----|------|-------|
| B200 / B300 / GB200 / GB300 | SM100 / SM103 (Blackwell) | `halo:blackwell` — FA4 (default) + FA2 + DeepEP, no FA3 |
| H100 / H200 | SM90 (Hopper) | `halo:hopper` — FA2 + FA3 + DeepEP, no FA4 |

Default to **blackwell** (the Makefile `IMAGE` default); confirm the host arch with `nvidia-smi`.
Confirm with `docker run --gpus all --rm halo:blackwell nvidia-smi`
(or `nvidia-smi` if available) when unsure of the host arch. If the local tag is
missing, pull it from ECR — see `agent-docs/infrastructure/docker.md`.

For `make`, override with `IMAGE=halo:hopper`. For the raw template,
swap the image line.

## 4. GPU count and disjoint / concurrent runs

- **GPU count** is the `torchrun --nproc_per_node=<n>` value (and `NPROC=<n>`
  for `make`). EP/CP/TP parallelism flags (e.g. `--expert_parallel_size`,
  `--context_parallel_size`, `--tensor_parallel_size`) pass through `EXTRA=` or
  directly after the config in the raw template.
- **Subset of GPUs** — restrict the devices the container sees with
  `--gpus '"device=0,1,2,3"'` (note the nested quoting). Inside, torchrun uses
  `--nproc_per_node` equal to that count.
- **Two jobs at once on disjoint GPUs** — give each its own device set, a
  **unique `--master_port`**, and a distinct `--name`/log:

  ```bash
  # Job A on GPUs 0-3
  docker run -d --rm --name jobA --gpus '"device=0,1,2,3"' \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=128g \
    --env-file .env -e HF_HOME="$D/hf" -e HF_DATASETS_CACHE="$D/hf/datasets" \
    -e TMPDIR="$D/tmp" -e HALO_DATA_ROOT="$D" -v $(pwd):/workspace -v "$D:$D" -v /root/.aws:/root/.aws \
    -w /workspace halo:blackwell \
    bash -lc "torchrun --nproc_per_node=4 --master_port=29801 <script> <config> > \"$D/jobA.log\" 2>&1"

  # Job B on GPUs 4-7 — different port + name + log
  docker run -d --rm --name jobB --gpus '"device=4,5,6,7"' \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=128g \
    --env-file .env -e HF_HOME="$D/hf" -e HF_DATASETS_CACHE="$D/hf/datasets" \
    -e TMPDIR="$D/tmp" -e HALO_DATA_ROOT="$D" -v $(pwd):/workspace -v "$D:$D" -v /root/.aws:/root/.aws \
    -w /workspace halo:blackwell \
    bash -lc "torchrun --nproc_per_node=4 --master_port=29802 <script> <config> > \"$D/jobB.log\" 2>&1"
  ```

  `CUDA_VISIBLE_DEVICES` (`-e CUDA_VISIBLE_DEVICES=0,1`) also works to scope
  devices inside a `--gpus all` container, but `--gpus '"device=..."'` is
  cleaner. Always pair concurrent torchrun jobs with **distinct master ports**.

## 5. Logging to the scratch volume and reading it back

Detached jobs write stdout+stderr to `$D/<job>.log` (root FS is small — keep big
outputs and checkpoints under `$D`). After launching:

```bash
docker ps                          # confirm the job is up
tail -f "$D/<job>.log"             # follow live (Ctrl-C to stop following)
docker logs -f <job>               # equivalent, via the container name
grep -E "loss|error|Traceback" "$D/<job>.log"   # quick health check
docker stop <job>                  # stop early (--rm auto-removes it)
```

Training configs should set `output_dir` (and checkpoints) under `$D` too.

## 6. Reminders checklist before launching

- [ ] A `make` target fits? → use it (`make train …`, `make test-gpu-core`, …).
- [ ] Right image for the host GPU arch (blackwell vs hopper)?
- [ ] `--env-file .env` present (raw runs)?
- [ ] In-container command called directly (no `poetry run`/venv prefix)?
- [ ] Scratch volume resolved (not hardcoded), `-v "$D:$D"`, log redirected to `$D/<job>.log`?
- [ ] `HF_HOME=$D/hf`, `HF_DATASETS_CACHE=$D/hf/datasets`, `TMPDIR=$D/tmp`,
      `HALO_DATA_ROOT=$D` (S3 cache + profiler scratch) — all on a verified large volume?
- [ ] Concurrent jobs → unique `--master_port`, `--name`, log, disjoint GPUs?
- [ ] Detached (`-d`) for anything long-running, so it survives the turn?

References: `Makefile` (canonical recipes),
`CLAUDE.md` (work principles), `agent-docs/infrastructure/docker.md` (image registry,
ECR pull, troubleshooting).

## Sources of truth
The `Makefile` + `agent-docs/infrastructure/docker.md` document the run recipes. The code is the **ultimate**
authority: when the `Makefile` recipe, this skill, or memory disagrees with what a script actually
expects (its argparse) or the `Dockerfile`, read the real file before launching. (`CLAUDE.md`:
docs-first, the code wins.) Related skills: `checkpoints` (turn a finished run into a usable artifact +
resume correctly), `data` (dataset format / offline prep).
