# Development Environment

The compiled GPU dependencies (PyTorch 2.11+cu130, FlashAttention, DeepEP/NVSHMEM) live inside the
prebuilt Docker images. Pick the variant by what you are doing:

| Variant | Use it for | How | GPU deps (FA/DeepEP) | Runs GPU code |
|---|---|---|---|---|
| **In-image** | Training, tests, any script that executes | `make` targets / `docker run` ([Docker](../infrastructure/docker.md)) | Yes | Yes |
| **Host `.venv` (uv)** | IDE go-to-definition, hover, find-usages | `uv sync` | No | No |
| **Dev Container** | IDE *inside* the image — full env, terminal that runs GPU code | `.devcontainer/devcontainer.json` | Yes | Yes |

In-image is canonical for running; the other two only add IDE intelligence.

## Host `.venv` for the IDE

```bash
uv sync                # creates .venv/ from uv.lock; installs src editable
```

Python is pinned to 3.12 (`requires-python = ">=3.12,<3.13"`), the image's interpreter. `uv sync`
resolves `torch 2.11.0+cu130`, `transformers`, `trl`, `accelerate`, `peft`, `datasets`, `ray`, and
`src` editable — enough for Pylance across the tree. Does **not** resolve `flash_attn` or `deep_ep`
(both source-built in the `Dockerfile`, not in `[project.dependencies]`); navigation elsewhere is
unaffected because both are imported behind optional-dependency guards. `vllm` is never imported
into the training env — it runs in a separate container reached over HTTP plus the vendored NCCL
client (`src/distributed/nccl/`).

The `.venv` torch is the cu130 wheel: it imports without a GPU, but CUDA operations fail. The
`.venv` is git-ignored.

## VS Code / IDE setup

Agent skills live at `skills/` (`.claude/skills` and `.agents/skills` are tracked symlinks to it, so Claude Code and Codex both discover them). `.vscode/` is git-ignored — create your own `settings.json` with the values below.

| Setting | Value | Why |
|---|---|---|
| `python.defaultInterpreterPath` | `${workspaceFolder}/.venv/bin/python` | Binds Pylance to the host `.venv` |
| `editor.defaultFormatter` (Python) | `charliermarsh.ruff` | Format + import-sort on save (line length 119) |
| `python.analysis.diagnosticMode` | `workspace` | Diagnostics cover the whole workspace; check mode comes from `[tool.pyright]` |
| `python.testing.pytestEnabled` | `true` | Test discovery under `tests/` |

Type checking is Pyright in **basic** mode — full checks need the deps, so run those in the image.

## Dev Container

`.devcontainer/devcontainer.json` runs the editor inside `halo:blackwell`, so Pylance
resolves `flash_attn` / `deep_ep` and the integrated terminal runs `torchrun` / `pytest` with no
`docker run` wrapper. It carries `--gpus all`, `--ipc=host`, `--ulimit`, `--shm-size=128g`,
`--cap-add=SYS_PTRACE` (py-spy attaches) and `--env-file .env` — so **`cp .env.example .env` before
first open**, or Docker refuses to start the container.

The host scratch volume mounts at `/scratch`, from `$HALO_SCRATCH` (default `/mnt`); export it in
your local shell to point elsewhere, after confirming with `findmnt` / `df -h` that the target really
is a large device — it is the same knob `make` and `docker compose` read, so exporting it once points
the whole toolchain at that volume. `containerEnv` then pins `HF_HOME`, `HF_DATASETS_CACHE`, `TMPDIR` and
`HALO_DATA_ROOT` under it, so HF caches, temp files, the S3 dataset cache and profiler traces all
stay off the small in-container root FS.

The **Dev Containers** extension (`ms-vscode-remote.remote-containers`) is a UI extension — install
it on your local VS Code, not on an SSH remote (VS Code refuses the remote install), which is why
it is absent from the Dev Container's `extensions` list in `.devcontainer/devcontainer.json`. Then **Cmd/Ctrl+Shift+P → "Dev Containers: Reopen in
Container"**; Pylance binds to `/usr/bin/python` inside the image. "Reopen Folder Locally" drops
back to the host `.venv`. For Hopper, change `"image"` to `halo:hopper`.

## Environment variables

The code does not auto-load `.env` — it is passed with `docker run --env-file .env`. Only the GPU
`make` targets do this (`DOCKER_RUN`, and only when `ENV_FILE` is non-empty); the CPU targets
(`install`, `test-cpu`, `docs`, `diagrams`, `docs-serve`) pass no credentials at all.

All of those but `docs-serve` (which mounts only the repo) mount `HF_CACHE` (default `$HALO_SCRATCH/hf`)
read-write, because ~60 CPU tests load a real tokenizer. A test calling `from_pretrained` directly
hard-fails when the cache is missing and the Hub is unreachable; one going through
`tests/common/tokenizers.py` skips instead. Secrets live in the repo-root `.env`; cache and path
redirects are `-e` flags pointed at a **verified** large volume (the root filesystem is small, and a
path named `/mnt` is not guaranteed to be a separate device — check `findmnt` / `df -h` first). On a
host whose docker defaults to a runtime that rejects `--gpus`/`--ipc host` (e.g. sysbox-runc), set
`DOCKER_RUNTIME=nvidia` — every `make` docker invocation then pins `--runtime nvidia` explicitly.

| Variable | Purpose | Where set | Owner page |
|---|---|---|---|
| `WANDB_API_KEY` | Weights & Biases logging | `.env` | — |
| `HF_TOKEN` | Gated HuggingFace model/dataset access | `.env` | — |
| `AWS_DEFAULT_REGION` | Region for your S3 bucket / ECR registry | `.env` | [AWS Auth](../infrastructure/aws-auth.md) |
| `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `TAVILY_API_KEY`, `SERPER_API_KEY`, `BRAVE_API_KEY` | External LLM + web-search backends | `.env` | [Native Tool-Use](../training-methods/grpo/environments/native-tool-use.md) |
| `HF_HOME` / `HF_DATASETS_CACHE` | HF cache roots on the scratch volume | `-e` flag | — |
| `TMPDIR` | Temp dir on the scratch volume | `-e` flag | — |
| `HALO_DATA_ROOT` | Toolkit scratch root — S3 dataset cache (`<root>/s3_datasets`) + profiler artifacts (`<root>/profiling`) | `-e` flag | [S3 Utilities](../data/s3-utilities.md) |
| `DIST_SHARED_FILESYSTEM` | `1` shared NFS/Lustre (default), `0` per-node local; `DIST_INPUT_SHARED_FILESYSTEM` / `DIST_OUTPUT_SHARED_FILESYSTEM` override the read and write sides separately | env | [Filesystem Handling](../data/filesystem-handling.md) |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` for EP throughput; baked into the image `ENV` (do not unset) | preset in image | [DeepEP](../infrastructure/deepep.md) |
| `HALO_SANDBOX_BACKEND`, `HALO_SANDBOX_URL` | Code-execution sandbox backend for coding RL envs | env | [Sandboxes](../training-methods/grpo/environments/sandbox.md) |

## Paths

| Path | Role |
|---|---|
| `/workspace` | Repo bind-mount and working dir in the image (`-v $(pwd):/workspace -w /workspace`); also `PYTHONPATH` |
| scratch volume (the `make` targets use `/mnt`) | Checkpoints, logs, HF cache, temp — verify its capacity before a large write |
| `.env` (repo root) | Secrets, passed via `--env-file .env` |
| `/root/.aws` | AWS credentials, bind-mounted for S3 / ECR |
| `.venv/` (repo root) | Host-side uv environment for the IDE (git-ignored) |

## Next steps

- [Installation](../getting-started/installation.md) · [Docker](../infrastructure/docker.md) — the runtime in detail
- [Contributing](index.md) — the `make`-driven dev loop and the gates
