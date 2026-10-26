# Development Environment

The compiled GPU dependencies (PyTorch 2.11+cu130, Flash Attention, DeepEP/NVSHMEM) live only inside
the images. Pick the variant by what you are doing:

| Variant | Use it for | How | GPU deps (FA/DeepEP) | Runs GPU code |
|---|---|---|---|---|
| **In-image** | Training, tests, any script that executes | `make` targets / `docker run` ([Docker](../infrastructure/docker.md)) | Yes | Yes |
| **Host `.venv` (uv)** | IDE go-to-definition, hover, find-usages | `uv sync` | No | No |
| **Dev Container** | IDE *inside* the image — full env, terminal that runs GPU code | `.devcontainer/devcontainer.json` | Yes | Yes |

In-image is canonical for running; the other two only add IDE intelligence.

## Host `.venv` for the IDE

```bash
uv sync                # creates .venv/ from uv.lock; installs halo editable
```

Python is pinned to 3.12 (`requires-python = ">=3.12,<3.13"`), the image's interpreter. `uv sync`
resolves `torch 2.11.0+cu130`, `transformers`, `trl`, `accelerate`, `peft`, `datasets`, `ray`, and
`halo` editable, enough for Pylance across the tree.

It does **not** resolve `flash_attn` or `deep_ep`, which the `Dockerfile` installs outside
`[project.dependencies]`: Blackwell takes FA2 from the NGC base and FA4 from a prebuilt wheel, Hopper
source-builds FA2 and FA3, and DeepEP is source-built in both. Navigation elsewhere is unaffected
because both are imported behind optional-dependency guards. `vllm` is never imported into the training env; it runs in a separate
container reached over HTTP plus the weight-sync clients (`halo/weight_sync/`).

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

The host scratch volume mounts at `/scratch`, from `$HALO_SCRATCH`, which the container refuses to
start without: export it in your local shell after confirming with `findmnt` / `df -h` that the target
really is a large device. `make` (which resolves it when unset) and the `training` service of
`docker-compose.vllm.yml` (which refuses to start without it) read the same knob, so one export points them at
that volume too.

`containerEnv` then pins `HF_HOME`, `HF_DATASETS_CACHE`, `TMPDIR` and `HALO_DATA_ROOT` under it, so HF
caches, temp files, the S3 dataset cache and profiler traces all stay off the small in-container root FS.

The **Dev Containers** extension (`ms-vscode-remote.remote-containers`) is a UI extension — install
it on your local VS Code, not on an SSH remote (VS Code refuses the remote install), which is why
it is absent from the Dev Container's `extensions` list in `.devcontainer/devcontainer.json`. Then **Cmd/Ctrl+Shift+P → "Dev Containers: Reopen in
Container"**; Pylance binds to `/usr/bin/python` inside the image. "Reopen Folder Locally" drops
back to the host `.venv`. For Hopper, change `"image"` to `halo:hopper`.

## Environment variables

The code does not auto-load `.env` — it is passed with `docker run --env-file .env`. Only the GPU
`make` targets do this (`DOCKER_RUN`, and only when `ENV_FILE` is non-empty); the CPU targets
(`check-lock`, `test-cpu`, `seed-hf-cache`, `diagrams`, `verify-move`, `verify-artifacts`, `verify-efficiency`,
`verify-ep` and `verify-cp` at `TIER=cpu`, and `make bench`'s band step) pass no credentials
at all; `docs` runs on the host.

The CPU targets other than `check-lock` and `diagrams` mount `HF_CACHE` (default `$HALO_SCRATCH/hf`)
read-write, because many CPU tests load a real tokenizer. A test calling `from_pretrained` directly
hard-fails when the cache is missing and the Hub is unreachable; one going through
`tests/common/tokenizers.py` skips instead, or fails under `HALO_TEST_REQUIRE_HUB_CACHE=1`.
`make seed-hf-cache` fills that cache ([what the seed holds](../infrastructure/ci.md#hub-seed)).

Secrets live in the repo-root `.env`; the cache and path redirects onto a verified large volume and
`DOCKER_RUNTIME` are on [Docker → Standard training launch](../infrastructure/docker.md#standard-training-launch),
and the toolkit and cache variables are cataloged in
[Environment variables](../reference/configuration-reference.md#environment-variables).

| Variable | Purpose | Where set | Owner page |
|---|---|---|---|
| `WANDB_API_KEY` | Weights & Biases logging | `.env` | — |
| `HF_TOKEN` | Gated HuggingFace model/dataset access | `.env` | — |
| `AWS_DEFAULT_REGION` | Region for your S3 bucket / ECR registry | `.env` | [AWS Auth](../infrastructure/aws-auth.md) |
| `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `TAVILY_API_KEY`, `SERPER_API_KEY`, `BRAVE_API_KEY` | External LLM + web-search backends | `.env` | [Native Tool-Use](../training-methods/async-grpo/environments/protocols/native-tool-use.md) |
| `VLLM_API_KEY` | The key a local OpenAI-compatible server expects (its `--api-key`), else `OPENAI_API_KEY` | `.env` | [Environment variables](../reference/configuration-reference.md#environment-variables) |
| `HALO_SCRATCH` | The large host volume the `make` targets, the dev container and `docker-compose.vllm.yml` mount | shell export (compose also reads `.env`; `make` does not) | [Docker](../infrastructure/docker.md#standard-training-launch) |
| `SGLANG_MODEL_DIR` | The host directory `docker-compose.sglang.yml` mounts (required, no fallback) | shell export or `.env` | [Rollout Servers](../infrastructure/rollout-servers.md#sglang) |
| `VLLM_GROUP_HOST`, `SGLANG_GROUP_HOST` | The trainer address a rollout server on another host dials back to | `.env` | [Rollout Servers](../infrastructure/rollout-servers.md#group-rendezvous) |

## Paths

| Path | Role |
|---|---|
| `/workspace` | Repo bind-mount and working dir in the image (`-v $(pwd):/workspace -w /workspace`); also `PYTHONPATH` |
| scratch volume (`HALO_SCRATCH`) | Checkpoints, logs, HF cache, temp — verify its capacity before a large write |
| `.env` (repo root) | Secrets, passed via `--env-file .env` |
| `/root/.aws` | AWS credentials, bind-mounted for S3 / ECR |
| `.venv/` (repo root) | Host-side uv environment for the IDE (git-ignored) |

## Next steps

- [Installation](../../human-docs/installation.md) · [Docker](../infrastructure/docker.md) — the runtime in detail
- [Contributing](README.md) — the `make`-driven dev loop and the gates
