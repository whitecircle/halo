---
name: launch
description: >-
  Assemble and run a Halo training or test job inside the prebuilt
  Docker image. Picks the image (halo:blackwell for B200/B300 vs
  halo:hopper for H100/H200), sets GPU count, mounts ($(pwd)->/workspace,
  the host's large scratch volume, ~/.aws), passes --env-file .env, runs
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

The Docker incantation lives **once** in the repo `Makefile` — contributors, CI, and this skill all call
its targets instead of re-typing `docker run`. Run `make help` and use a target if one fits:

```bash
# Training — CONFIG is required; METHOD defaults to sft (any `halo launch --list` entry, e.g.
# offline-grpo, environmental-grpo, preference/smpo); NPROC defaults to 8; IMAGE to halo:blackwell;
# EXTRA passes through extra CLI flags.
make train CONFIG=examples/sft/qwen3/qwen3-4b-ultrachat.yaml
make train CONFIG=examples/sft/gptoss/gptoss-20b-multinode-ep.yaml \
           NPROC=8 EXTRA="--expert_parallel_size=8"

# Tests, benchmarks, images, host-only gates
make seed-hf-cache && make test-cpu          # CPU tier (no GPU, in the image)
make test-gpu-core                           # core GPU tier; test-gpu-full for the heavy one
make bench BAND=1                            # throughput + peak memory held to tests/baselines/bench (8 GPUs)
make build-blackwell                         # build-hopper / build-vllm / build-sglang / build-all
make lint && make docs                       # host-only (pinned uvx ruff; links, anchors, code references)
```

The rollout-server GPU tiers (`make test-gpu-vllm` / `make test-gpu-sglang`) need the server already
serving the same checkpoint on a GPU outside `TRAINER_CUDA_DEVICES`, one pass per served model
(`SERVER_TIER`): [Contributing → Dev loop](../../agent-docs/contributing/README.md#dev-loop).

What a `make` run differs in from a hand-rolled launch (foreground, `--network host`, no
`--cap-add=SYS_PTRACE`, conditional `.env`/`~/.aws`) and the one `HALO_SCRATCH` override for the large
volume: [Docker → Standard training launch](../../agent-docs/infrastructure/docker.md#standard-training-launch).
Unset, `make` resolves `HALO_SCRATCH` to the local mount other than `/` with the most free space — verify the pick with
`findmnt`/`df -h` and pass `HALO_SCRATCH=<path>` to choose another.

## 2. Raw detached `docker run` (when `make` doesn't fit)

Use the detached recipe in
[Docker → Standard training launch](../../agent-docs/infrastructure/docker.md#standard-training-launch)
for custom flags, disjoint-GPU runs, or long detached jobs: fill in `<job>` (container + log name), the
`torchrun` line and any extra flags, and keep every flag it calls load-bearing. `$D` is the scratch volume,
**resolved dynamically — never hardcode a path**: a directory's name does not prove its capacity.

Single-GPU / accelerate / pytest variants just swap the `bash -lc "..."` body, e.g.:

```bash
# Single GPU, plain python
bash -lc "python <script> <config> > \"$D/<job>.log\" 2>&1"

# Accelerate DDP / FSDP (prefer the FSDP2 config; CLAUDE.md flags an Accelerate FSDP-v1 corruption bug)
bash -lc "accelerate launch --config_file launcher-configs/accelerate/fsdp2_gradop_config.yaml \
          <script> <config> > \"$D/<job>.log\" 2>&1"

# A specific pytest selection not covered by the make targets
bash -lc "pytest -m gpu tests/gpu/test_suite.py -k <pattern> > \"$D/<job>.log\" 2>&1"
```

A **CPU-only** job (CPU tests, lint-in-image, one-off Python with no GPU) uses the CPU-only variant on
that Docker page.

## 3. Image selection — by GPU architecture

`halo:blackwell` on B200/B300 (the Makefile `IMAGE` default), `halo:hopper` on H100/H200; confirm the host
arch with `nvidia-smi`. What each image carries, and pulling a missing tag from ECR:
[Docker](../../agent-docs/infrastructure/docker.md#image-matrix). For `make`, override with
`IMAGE=halo:hopper`; for the raw template, swap the image line.

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
    -e TMPDIR="$D/tmp" -e HALO_DATA_ROOT="$D" -v $(pwd):/workspace -v "$D:$D" -v ~/.aws:/root/.aws \
    -w /workspace halo:blackwell \
    bash -lc "torchrun --nproc_per_node=4 --master_port=29801 <script> <config> > \"$D/jobA.log\" 2>&1"

  # Job B on GPUs 4-7 — different port + name + log
  docker run -d --rm --name jobB --gpus '"device=4,5,6,7"' \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=128g \
    --env-file .env -e HF_HOME="$D/hf" -e HF_DATASETS_CACHE="$D/hf/datasets" \
    -e TMPDIR="$D/tmp" -e HALO_DATA_ROOT="$D" -v $(pwd):/workspace -v "$D:$D" -v ~/.aws:/root/.aws \
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

## Sources of truth

The `Makefile` + `agent-docs/infrastructure/docker.md` document the run recipes. The code is the **ultimate**
authority: when the `Makefile` recipe, this skill, or memory disagrees with what a script actually
expects (its argparse) or the `Dockerfile`, read the real file before launching. (`CLAUDE.md`:
docs-first, the code wins.) Related skills: `checkpoints` (turn a finished run into a usable artifact +
resume correctly), `data` (dataset format / offline prep).
