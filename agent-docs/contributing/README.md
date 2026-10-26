# Contributing

Every command goes through `make` — contributors, CI, and the Claude Code skills call the
same targets. The Docker incantation lives once in the `Makefile`; `make help` lists them.

## Ground rules

- **The host has no usable Python.** PyTorch 2.11+cu130, DeepEP, and Flash Attention live only
  inside the prebuilt images, so anything that *runs* runs inside one. Tools are on `PATH` in the
  container (uv installs into the system interpreter — no venv, no Poetry): call `python`,
  `torchrun`, `pytest` directly.
- **`make lint`, `make format`, `make precommit`, and `make docs` are the only host-runnable gates**
  (ruff via `uvx ruff@$(RUFF_VERSION)`, `RUFF_VERSION ?= 0.9.10`, falling back to a `ruff` already on
  `PATH`; the docs target, on stdlib Python 3.12+, checks links, anchors and the code the docs cite).
  Everything else — tests, benchmarks — runs inside the image.
- **Credentials live in the repo-root `.env`.** `cp .env.example .env` and fill it in: the GPU
  `make` targets pass `--env-file .env` and fail outright without the file.
- **Markdown lives in `agent-docs/` (the detailed reference), `human-docs/` (the concise human
  guide) and `skills/` (the agent skills) — all plain GitHub markdown.** The root files
  (`README`, `CONTRIBUTING`, `SECURITY`, `CODE_OF_CONDUCT`, `AGENTS.md`, `CLAUDE.md`, `.github/*`)
  are the documented exception; scratch markdown goes to `/tmp`.
- **Large outputs go to a verified large volume.** The root filesystem is small; the `make` targets
  put `HF_HOME` and `TMPDIR` under `HALO_SCRATCH` (default `/mnt`) — point it at a volume you have
  checked with `findmnt` / `df -h`.

## The gate (read first)

Halo is a public, gated project. Every change starts as an **accepted issue** (use a
`.github/ISSUE_TEMPLATE`), and a maintainer must comment `/approve @you` on it — which adds you to
`.github/APPROVED_CONTRIBUTORS` (kept on the `allowlist` branch) — **before** you open a PR. A PR from anyone not on that list and
without write access is auto-commented and closed by `.github/workflows/pr-gate.yml`. A reaction, a
comment, a branch, or a draft reserves nothing. `CONTRIBUTING.md` is the short version; the workflow
mechanics are in [Continuous Integration](../infrastructure/ci.md#contribution-gate).

## Dev setup

```bash
# Outside contributors: fork on GitHub first — there is no push access to whitecircle/halo.
git clone https://github.com/<your-handle>/halo.git   # maintainers: whitecircle/halo
cd halo

cp .env.example .env          # the GPU make targets pass it with --env-file and fail without it
# Either pull the prebuilt image under the name the make targets use ...
docker pull public.ecr.aws/whitecircle/halo:blackwell && docker tag public.ecr.aws/whitecircle/halo:blackwell halo:blackwell
# ... or build it: B200 (SM100) / B300 (SM103); build-hopper for H100/H200 (SM90)
make build-blackwell
make install                  # check the lock installs into the image (throwaway container)
make seed-hf-cache            # configs + tokenizers the CPU tests read into HF_CACHE (no weights)
make test-cpu                 # sanity check; needs Docker, not a GPU
```

Both builds are credential-free — no token, no registry login, no BuildKit secret.
`make build-blackwell` builds `halo:blackwell` (B200/B300), `make build-hopper` builds
`halo:hopper` (H100/H200); a bare `docker build` defaults to hopper
(`ARG TARGET_GPU=hopper`). Point any *run* target at a specific image with `IMAGE=`:
`make test-cpu IMAGE=halo:hopper` (the `build-*` and `push-*` targets hardcode their tags).
`make build-vllm` / `build-sglang` build the two inference images; the `push-*` targets publish to
a registry you configure — see [Registry](../infrastructure/docker.md#registry). What each image
carries: [Docker → Image matrix](../infrastructure/docker.md#image-matrix).

## Dev loop

| Step | Command | Where it runs |
|------|---------|---------------|
| Format | `make format` | host (`ruff format`) |
| Lint | `make lint` | host (`ruff check`) |
| Lint + format-check | `make precommit` | host |
| Same gates on every commit (optional) | `pipx install pre-commit && pre-commit install` | host (`.pre-commit-config.yaml`) |
| CPU tests | `make test-cpu` | image (no GPUs) |
| Core GPU tests (pre-merge, GPU changes) | `make test-gpu-core` | image (`-m "gpu and core"`) |
| Full GPU tests (heavy tier) | `make test-gpu-full` | image (`-m gpu`) |
| vLLM-server GPU tests | `make test-gpu-vllm` | image (`-m "gpu and vllm_server and ($SERVER_TIER)"`) |
| SGLang-server GPU tests | `make test-gpu-sglang` | image (`-m "gpu and sglang_server and ($SERVER_TIER)"`) |
| Throughput and peak memory hold to the committed baselines | `make bench BAND=1` | image (8 GPUs) |
| A structural change only moves code | `make verify-move BASE=<sha>` | image (no GPUs) |
| Every CPU saver writes what it wrote at `BASE` | `make verify-artifacts BASE=<sha>` | image (no GPUs) |
| EP forward, backward and gradients match `BASE` bit for bit | `make verify-ep BASE=<sha> [TIER=gpu]` | image (`TIER=gpu`: 4 GPUs) |
| Ulysses CP attention and model match `BASE` | `make verify-cp BASE=<sha> [TIER=gpu]` | image (`TIER=gpu`: 2 GPUs) |
| The efficiency callback reports what it reported at `BASE` | `make verify-efficiency BASE=<sha>` | image (no GPUs) |
| Docs link check | `make docs` | host (`scripts/docs/check_links.sh`) |

`make test-gpu-vllm` is the `vllm_server` slice of the full tier and needs the `docker-compose.vllm.yml`
server already serving on a GPU outside `TRAINER_CUDA_DEVICES` (default `0,1,2,3,4,5,6`, which the target
pins as the trainer's `CUDA_VISIBLE_DEVICES`). Weight sync is an NCCL broadcast, and a rank cannot
broadcast to itself.

Without `EFA=1` it forces the no-fabric recipe both compose bases default to
(`NCCL_IB_DISABLE=1 NCCL_NET=Socket`; why: [Rollout Servers → vLLM](../infrastructure/rollout-servers.md#vllm)).
`EFA=1` passes `/dev/infiniband` and the fabric variables instead, matching a server started with
its compose EFA overlay ([Servers on other nodes](../infrastructure/rollout-servers.md#servers-on-other-nodes-efa)).

Each server slice takes **two passes**, because the tests broadcast the trainer's own weights into the
served model and assert the served policy moved — server and trainer must hold the same checkpoint, and
no one server covers both halves. `SERVER_TIER` defaults to `not moe` (serve a dense model); restart the
server on a MoE checkpoint and rerun with `SERVER_TIER=moe`.

`make test-gpu-sglang` is the `sglang_server` slice with the same server requirement and the same
`EFA=1` switch; the server needs only cuMem parity on top — `NCCL_CUMEM_ENABLE=1`, the compose default
(why: [Rollout Servers → NCCL transport](../infrastructure/rollout-servers.md#nccl-transport-sglang)).

```bash
make train CONFIG=examples/sft/qwen3/qwen3-4b-ultrachat.yaml NPROC=8
make train CONFIG=examples/sft/gptoss/gptoss-20b-multinode-ep.yaml \
  NPROC=8 EXTRA="--expert_parallel_size=8"
make train METHOD=preference/smpo CONFIG=<config>    # METHOD defaults to sft
```

`METHOD` is any `halo launch --list` entry.

`make clean` prunes `wandb/`, the ruff and pytest caches, and the `$HALO_SCRATCH` test scratch.
`checkpoints/` is left alone — every example writes its run there.

## Code style

- **ruff, line length 119.** Selected: `E, F, I, W, UP, B, C4, SIM, T20, PLC0415`. `make lint`
  enforces the full set, and the blocking CI job runs `ruff format --check` plus the same full set.
  See [Continuous Integration](../infrastructure/ci.md).
- **No inline imports** (`PLC0415`), in `tests/` as in `src/`. A function-level import is reserved
  for a genuinely optional or arch-specific dependency (`flash_attn`, `deep_ep`, `kernels`). A
  circular import is never that exception — fix it structurally.
- **No stray `print()` in `src/`** (`T20`). `tests/**` relaxes `E402`, `T20`, `E741`, `B007`,
  `B023`, and `SIM117`; `scripts/**` relaxes `T20`.
- **Reuse over re-implementation.** New trainers extend `DistributedTrainerMixin`; new parallelism
  wrappers reuse the EP-layer base hooks. See
  [Trainer Architecture](../reference/trainer-architecture.md).
- **Naming, named constants, module order, comments and docstrings** follow the Code style section
  of `CLAUDE.md`.

## AI-assisted contributions

Halo is built with AI assistance, and AI-assisted contributions are welcome. `CLAUDE.md` and
`AGENTS.md` are the rulebook for agents and humans alike. Three conditions:

- **Disclose** the scaffold and models in the PR (the template has a checkbox).
- **Own it** — you are accountable for every line, at the same bar as hand-written code.
- **House rules apply** — `CLAUDE.md`, the ruff gate, the test conventions, and the docs charter
  bind AI-generated code exactly as they bind yours.

Unreviewed AI output — code nobody read, tests nobody ran, docs in marketing register — is closed.
A diff over **~2,000 lines** drops off the fast path; split it.

## Tests

CPU tests run directly under pytest; GPU tests are external `torchrun` scripts launched through a
manifest.

- **Test behavior, not implementation.** Assert on loss finiteness, loss decrease, output shapes,
  and rejection of unsupported configs — never on private internals.
- **A CPU test file is a plain pytest module.** It ends in
  `if __name__ == "__main__": raise SystemExit(pytest.main([__file__, "-v"]))`, so a standalone
  `python tests/cpu/<file>.py` runs exactly what `pytest` collects. The conventions test requires
  every CPU test file to end in exactly that block.

    A hand-listed runner silently drops any test missing from its list, and a printed pass/fail summary
    hides a FAIL from pytest; both are rejected by `tests/cpu/conventions/test_test_conventions.py`.

- **The `cpu` marker is applied by path, once.** `tests/conftest.py` marks every item under
  `tests/cpu/` in `pytest_collection_modifyitems`, so `-m cpu` selects the tier without a per-file
  `pytestmark`; re-declaring `cpu` in one duplicates a marker the collector already applied.
- **Imports resolve from the image, not from `sys.path` surgery.** `PYTHONPATH=/workspace` is baked
  into the training images (and `tests/conftest.py` puts the repo root on the path for a pytest
  run), so `from src...` / `from tests.common...` work in both `pytest` and a standalone
  `torchrun tests/gpu/<script>.py`.

    A per-file `sys.path.insert` is dead weight that only masks running outside the image. `scripts/`
    is a namespace package, so a test imports a tool by dotted path
    (`from scripts.after_training.merge_models import merge_models`);
    `tests.common.repo.load_script_module` loads one by file path where a test needs a module object
    of its own (a fresh copy per case, a name registered for `patch`) or the file sits outside the
    packages (`docker/`).

- **Every first-party reference resolves without running.** `tests/cpu/conventions/test_references_resolve.py`
  statically resolves each first-party import, `patch` / `monkeypatch` target (string or object form),
  attribute chain on an imported module or class, dotted name and repo-root path in the Python under
  `src/`, `scripts/`, `tests/`, `docker/` and `.github/`, in the other text files of those trees (the
  GPU benchmark runners, the workflows), and in the image builds, `Makefile`, `pyproject.toml`,
  `.env.example`, `.pre-commit-config.yaml` and the recipes, so a moved module, a renamed method or a
  dead patch target fails the CPU tier even where only a GPU script names it. A class attribute must be
  defined in the first-party hierarchy; an attribute a third-party base supplies (`Function.apply`) is
  listed per base in `_EXTERNAL_BASE_ATTRIBUTES` (`scripts/docs/references.py`). Text that only reads
  as a reference (a scratch tree a test builds) is exempted per file in the test's `_NOT_REFERENCES`.

    The same sweep holds each reference to the binding it means. An import names the module that
    defines what it takes: a module that only re-binds the name by an import of its own is a second
    path, which would keep a moved symbol reachable through its old home (one in a sensitive path waits
    in `_SENSITIVE_REEXPORT_IMPORTS`). An attribute read through a module object only has to exist, since
    the binding a module imports is what a patch replaces and restores through it. A patch of a module's
    own top-level name, a `module.name = fake` assignment included, must target a name some function reads
    when it runs, one of that module or, outside the tests, one reading it through the module object;
    patching a name the module only binds intercepts nothing, so the test would pass whatever the code does.

- **A move rewrites every reference in the same change.** One symbol keeps one import path, so moving a
  module or a symbol rewrites the imports that name it (a `from` line split per destination), its
  `patch` / `monkeypatch` / `caplog` / `sys.modules` / `import_module` strings and its repo-root paths,
  and leaves no re-export behind. Rewrite them by splicing the source text at the AST nodes' offsets, or
  with libcst in a throwaway environment (`uv run --no-project --with libcst <script>`, which leaves
  `uv.lock` alone); never regenerate a file with `ast.unparse`, which drops its comments. The reference
  sweep fails whatever the rewrite missed, and `make verify-move BASE=<sha>` proves the move changed no
  code ([Move kit](#move-kit)).

- **The import layering is read off the import graph.** `tests/cpu/conventions/test_import_layering.py`
  fails a `src.distributed` or `torch.distributed` import in `src/checkpoint/` or `src/models/loading/`,
  an import of the loader dispatch from the rest of `src/distributed`, a cycle among the members of
  `src/distributed` and a raw toolkit env read. It also lays the final layer table (`_LAYERS`: a unit
  imports only units of lower rows, `cli` imports nothing from the package, the tool-safe units reach no
  process group, parallelism or rollout sync, the member orders inside `src/distributed`, one lazy
  `deep_ep` importer, package-internal modules, no module cycle) over today's tree, each module that has
  not moved yet placed where it lands through `_PENDING_MOVES`. A break outside `_KNOWN_LAYER_BREAKS`
  (the ones today's tree still makes, reported as one warning) fails, and so do a module the table
  cannot place and an entry of either table that is out of date. `TYPE_CHECKING` imports are exempt.

- **Convention tests also run on the hosted runner.** The `conventions` job in `.github/workflows/lint.yml`
  runs the files it lists on the standard library and pytest alone, so such a test reads the source
  with `ast` instead of importing it and takes repo helpers from `tests.common.repo`;
  `tests/cpu/conventions/test_hosted_conventions_are_stdlib.py` fails when a listed file, a package
  `__init__.py` or conftest pytest runs for it, or a file one of them imports or loads by path reaches
  anything else. It cannot read what a probe script imports, so a listed test that runs one is refused
  unless its `_PROBE_COLLECTS` entry names the files the probe has pytest collect.

- **Remote code loads into the test process's own module cache.** `tests/cpu/conftest.py` points
  `HF_MODULES_CACHE` at a scratch dir per process before transformers is imported, so remote-code
  loads need no writable HF home and no worker shares the copies another writes. A standalone
  `python tests/cpu/<file>.py` that imports transformers keeps the environment's cache
  (`HF_MODULES_CACHE`, by default `$HF_HOME/modules`).

    Do not redirect the cache inside a test around a `trust_remote_code` load. transformers puts
    each cache it copies into on `sys.path` and never takes it off, so the `transformers_modules`
    package can stay bound to the test's temp dir for the rest of the worker, and every later local
    remote-code checkpoint there fails to import.

- **Deterministic, seeded data.** GPU tests generate synthetic problems from a fixed seed; only
  rank 0 prints.

    A seed alone does not fix a chat-templated batch: `strftime_now` is a jinja global transformers
    injects into every template, and gpt-oss's harmony prompt stamps the live date into its system
    message.

    The shared `tests.common.ep_reference.fixed_chat_batch` shadows that global with
    `CHAT_TEMPLATE_NOW`, so the correctness thresholds measured against its batch (EP/TP loss bounds,
    the rotated-expert control floor) do not drift with the calendar. Build a fixed batch through it
    instead of re-rolling one per file.

- **GPU tests register in the manifest.** Drop the script under `tests/gpu/`, add one `TestSpec`
  line to `tests/gpu/manifest.py` (`nproc`, `markers`, `args_matrix`, `timeout`, `flaky`). A script
  missing from the manifest fails collection, so coverage cannot silently rot.

    A `bench*.py` that is in neither the manifest nor `_UNMANIFESTED_BENCHMARKS` fails the same way.
    World-size strictness is not a manifest field; the script declares it itself via
    `gpu_test_main(exact_world_size=N)`.

- **Run GPU tests through pytest**, not by hand: `make test-gpu-core`, or a narrower marker
  expression over the directory (`pytest -m "gpu and ep" tests/gpu`). The launcher entry points
  (`LAUNCHER_ENTRYPOINTS` in `tests/gpu/manifest.py`) are the only modules pytest collects under
  `tests/gpu/`: `tests/gpu/conftest.py` ignores every manifest script, so no collection (a
  `pytest -m cpu` from the repo root included) imports a torchrun program.

    A CPU test or a `tests/common` helper never reaches a GPU script either — not by import, not by
    `load_script_module`, not inside a probe script's source. A fixture both tiers need lives in
    `tests/common/`; `tests/cpu/conventions/test_test_conventions.py` fails the suite on a reach.

    The launcher allocates a free `--master_port` per node from `tests/common/ports.py` — a pool from
    20000 up to the kernel's ephemeral range, one slice per pytest-xdist worker — and points `TMPDIR`
    at a per-run dir under pytest's basetemp; never hardcode either. A script run standalone under
    `torchrun --nproc_per_node=N <script>` lets torchrun pick the port.

- **Scratch goes through the launcher's `TMPDIR`.** `setup_cache_dirs` for per-rank output/cache
  dirs, `shared_scratch_dir` (`tests/common/distributed.py`) for a synthetic checkpoint rank 0
  writes and the peers read. The per-rank dirs carry the `SCRATCH_DIR_TAG` prefix
  (`tests/common/scratch.py`), and the root `tests/conftest.py` removes tagged dirs older than 6 h at
  session start, since a crash skips the in-test cleanup; untagged names in a shared `TMPDIR` are never
  touched.

    A literal `/mnt/...` in a test escapes basetemp, is never reclaimed, and assumes a volume layout
    this host may not have; checkpoint locations belong in `tests/common/models.py`.

- **Use the `gpu_test_main` harness** (`tests/common/harness.py`). It owns the lifecycle
  (`init_distributed` → setup → body → teardown → `sys.exit`); the body is *load → train → assert*
  and returns `{"checks": {name: bool}, "metrics": {...}}`, exiting `0` pass / `1` fail / `2` bad
  launch.

    Every manifest script runs under it, except a script whose lifecycle the harness cannot express,
    listed in `_OWN_LIFECYCLE` (`tests/cpu/conventions/test_gpu_harness_conventions.py`) with the
    reason, and an entry whose script moves onto the harness fails the suite until it is dropped.

    The harness sets `CAUSAL_CONV1D_DETERMINISTIC=1` unless the caller exported a value: causal_conv1d's
    default backward sums the conv weight gradient with atomics, so the gated-DeltaNet families miss
    an exact resume replay now and then. An EP body that replays a run exactly without
    `full_determinism` calls `pin_deterministic_ep_dispatch` (`tests/common/distributed.py`) itself:
    DeepEP's default dispatch claims receive slots with atomics, so the order an expert's tokens arrive
    in, and with it the rounding of every expert weight gradient, changes from run to run (under
    `full_determinism` the dispatcher builds the deterministic buffer on its own). A forward-only
    comparison on the grouped-GEMM path needs no pin: each expert row is computed independently of that
    order and the combine sums a token's partials in top-k slot order, so it holds the bound a non-EP
    run does, as the per-expert loop does ([EP → Determinism](../parallelism/expert-parallelism.md#determinism)).
- **Build checks from the shared helpers** rather than re-deriving them per file:
  `training_run_checks` (`tests/common/utils.py`, the finished-run verdicts), `parallel_shape_checks`
  (`parallel_shape.py`, each enabled axis read off the model, not the config echo), `run_sft_suite`
  (`sft_modes.py`, the SFT smoke body with one `--mode` per manifest row), `run_gloo_ranks` (`gloo.py`,
  spawned gloo ranks for a CPU test that needs a real process group) and the synthetic-checkpoint
  builders in `tiny_models.py`. A worst-difference bound goes through `max_or_nan` (`utils.py`): the
  builtin `max` drops a NaN anywhere but first, so the bound would pass on it. The full table is in
  `skills/write-tests/harness.md`.
- **Emit perf and memory.** Return `ctx.metrics(trainer)` so the result line carries tokens/s/GPU
  and peak memory; a `benchmark_*` script's `emit_benchmark(key, callback)` writes the line the
  committed throughput baselines are compared against.
- **Cover the matrix and the rejections.** Markers select the tier (`core` or `full`), the world size
  every entry declares (`1gpu 2gpu 4gpu 8gpu`) and capabilities (`ep cp tp etp pp hsdp vlm lora moe` +
  model family).

    `vllm_server`/`sglang_server` mark a test needing the live vLLM/SGLang container; those are always
    `full`. A new parallelism mode needs a correctness test per supported combination *and* a test
    asserting the unsupported ones are rejected.

The GPU launcher (`tests/gpu/conftest.py`) reads its knobs with raw `os.environ` — importing
`src/env.py` would pull torch and transformers into the launcher process — so each one is matched as
a literal string, not through the toolkit's `1/true/yes/on` flag parsing. The ones shipped
infrastructure sets:

| Var | Default | Meaning |
|---|---|---|
| `VLLM_SERVER_URL` / `SGLANG_SERVER_URL` | `http://localhost:8000` / `:30000` | Live rollout server the `vllm_server` / `sglang_server` tiers probe and drive. Both are set by the shipped infrastructure itself (`Makefile`, `docker-compose.vllm.yml`), which is why they carry no `HALO_TEST_` prefix — every other test knob does. |
| `HALO_TEST_LAUNCH_ID` | per launch | Set BY the launcher, not for it: a unique id stamped into every torchrun launch's environment so the orphan sweep can identify surviving workers from `/proc` without matching on a script name a co-tenant might also be running. Do not export it. |
| `HALO_TEST_REQUIRE_SERVER` | unset | The **engine name** (`vllm` / `sglang`, set by the `make` server tiers) whose tests must not be skipped: a node carrying the `<value>_server` marker raises a `UsageError` instead of skipping when the endpoint is unreachable, so a dead container cannot pass as a skip. Any other engine's tests still skip. |

Test scripts read their own knobs through `src/env.py`, `HALO_TEST_`-prefixed so a stray `export` or
a co-tenant compose file cannot collide with them — the one exception is the server-side `VLLM_MODEL`
below. The cross-suite ones:

| Var | Meaning |
|---|---|
| `HALO_TEST_MODEL` | Checkpoint of the suites that sweep architectures: the expert-LoRA, router `modules_to_save`, mixed merged-save and expert-LoRA reference suites (default gpt-oss-20b; swept with `HALO_TEST_ATTN` / `HALO_TEST_EP`) and the dense self-distillation text suite (default Qwen3-0.6B). |
| `HALO_TEST_EP` / `HALO_TEST_CP` / `HALO_TEST_TP` / `HALO_TEST_ETP` | Parallel size a sweep-capable suite builds its `ParallelismConfig` with; unset = the suite's own default (often `world_size` for EP). The suffix is the parallelism axis as the rest of the toolkit spells it (`ep_size` / `cp_size` / `tp_size` / `expert_tp_size`), so the knob and the config field it feeds read the same. |
| `HALO_TEST_REQUIRE_HUB_CACHE` | For an offline run over the seed `tests/common/hub_seed.py` derives ([Hub seed](../infrastructure/ci.md#hub-seed)): a CPU test whose Hub repo is not in the local HF cache fails instead of skipping. Gated repos (`GATED_REPOS`) and local checkpoint paths still skip. Load a tokenizer, processor, config or template through the `tests/common/tokenizers.py` helpers, and name its repo in `tests/common/models.py` so the seed carries it. |
| `HALO_TEST_ATTN` / `HALO_TEST_GC` / `HALO_TEST_REVISION` | Attention implementation, gradient checkpointing (default **on**), hub revision for the suites that sweep them. The per-family `HALO_TEST_ZAYA_GC` defaults the other way — see the per-suite table. |
| `HALO_TEST_OFFGRPO_PARALLEL` | `tp` (default, dense Qwen3) or `ep` (gpt-oss MoE) leg of `trainers/grpo/test_offline_grpo_tp_resume.py`. |
| `HALO_TEST_MAX_STEPS`, `HALO_TEST_BATCH_SIZE`, `HALO_TEST_GRAD_ACCUM`, `HALO_TEST_NUM_GENERATIONS`, `HALO_TEST_NUM_WORKERS`, `HALO_TEST_MAX_CONCURRENT`, `HALO_TEST_ROLLOUT_MAX_TOKENS`, `HALO_TEST_MAX_COMPLETION` | Step count and rollout sizing for `trainers/grpo/test_environmental_grpo_benchmarks.py`, whose defaults are sized for one vLLM server. |
| `HALO_TEST_VLLM_GROUP_PORT` / `HALO_TEST_SGLANG_GROUP_PORT` | Pins the trainer-side weight-transfer group port the rollout e2e suites bind. Unset, each run draws one with `free_port()` (`tests/common/ports.py`), below the kernel's ephemeral range, so no outbound connection on the host can hold it; the engine learns the port from the trainer's init request, so the server needs no matching setting. The online-GRPO MoE / dense pair offsets a pinned port by the row's index, so back-to-back rows bind different ones. A resume row rebinds its port for phase 2, and both weight-transfer re-init suites rebind it every cycle. |
| `HALO_TEST_VLLM_SERVER_URLS` | Comma-separated rollout endpoints the environmental legs of `trainers/grpo/test_online_grpo_vllm_e2e.py` drive (default: the single `VLLM_SERVER_URL`). The leg always pushes through the single training process's `InferenceClientManager`; two or more URLs make it pause every server together and stream each chunk to all of them concurrently, and the leg then asserts the served policy moved on **every** one of them. Each server serves the same model on its own GPU; the leg binds one trainer-side group port per server, each allocated by `free_port()` (`tests/common/ports.py`) rather than pinned, so back-to-back rows cannot contend. |
| `VLLM_MODEL` | Checkpoint the running vLLM server serves, read by the rollout benchmark itself — not by the launcher, and **not** forwarded by the `make` tiers, so put it in `.env` or pass `EXTRA_DOCKER_ENV="-e VLLM_MODEL=…"` (a host export never reaches the container). It must be the model the test trains: unset, the benchmark falls back to Qwen3-0.6B and a MoE-serving tier silently tests the wrong pairing. (`SGLANG_MODEL` is the compose file's server-side spelling; no test reads it.) |

Suites that pin one family or one phase add their own `HALO_TEST_<SUITE>_*` knobs on top; the manifest
entry and the script name them, and the non-obvious ones are:

| Var | Meaning |
|---|---|
| `HALO_TEST_<SUITE>_MODEL` | Swap the checkpoint a family or phase suite loads without editing it; the per-suite spelling keeps a global export from pointing a family test at a checkpoint of another family. Per family (`HALO_TEST_ZAYA_MODEL`, `HALO_TEST_GLM4_MODEL`, `HALO_TEST_GEMMA4_MODEL`, `HALO_TEST_QWEN3_5_MODEL`, …) and per phase: `HALO_TEST_EP_RT_MODEL` (EP round-trip, default a local Gemma4-26B-A4B checkpoint, built as shown below the table), `HALO_TEST_EP_CP_RT_MODEL` (EP+CP round-trip, the same for gpt-oss-20b), `HALO_TEST_EP1_KNOB_MODEL` (ep1 weight-sync, default gpt-oss), `HALO_TEST_RESUME_MODEL` / `HALO_TEST_RESUME_EP_MODEL` (SFT resume: dense default Qwen3-0.6B, and the MoE the `ep` mode needs), `HALO_TEST_LORA_CP_MODEL` / `HALO_TEST_LORA_SAVE_LOAD_MODEL` (both default Qwen3-0.6B; point them at a 4B for a scale check), `HALO_TEST_STEP3P7_MODEL` (the Step-3.7 vLLM sync suite serves its own `--write-checkpoint` tree, so it has no default and skips without one). A suite whose local default checkpoint is absent skips rather than failing. |
| `HALO_TEST_ZAYA_EP_STEPS` / `HALO_TEST_ZAYA_EP_SEQ` | Training steps (default `4`) and `max_length` (default `512`) of the ZAYA1-8B EP smoke suite (`tests/gpu/parallelism/ep/test_zaya_ep.py`); the step count is asserted, so a longer sweep stays self-checking. |
| `HALO_TEST_ZAYA_FSDP_STEPS` | Training steps (default `4`) of the ZAYA1-8B FSDP2 suite (`tests/gpu/trainers/sft/test_zaya_fsdp.py`); its sequence length is fixed. |
| `HALO_TEST_ZAYA_GC` / `HALO_TEST_ZAYA_GC_REENTRANT` | Gradient checkpointing (default **off**) and its `use_reentrant` kwarg (default on) for both Zaya suites. Zaya's load patch refuses GC, so `=1` asserts that refusal rather than training with it. |
| `HALO_TEST_EP_RT_ATTN` / `HALO_TEST_EP_RT_SCOPE` / `HALO_TEST_EP_RT_EXPERT_TP` / `HALO_TEST_EP_RT_KEEP` | EP save/reload round-trip (`tests/gpu/parallelism/ep/test_ep_save_reload_roundtrip.py`): attention impl (default `sdpa`), `ep_scope` (default `auto`; `node` / `global` pick the node-local vs cross-node gathered-save path), `expert_tp_size` (default `1`; at `HALO_TEST_EP=1` this is pure ETP), and `_KEEP` to leave the gathered checkpoint on disk instead of deleting it. |
| `HALO_TEST_EP_CP_RT_ATTN` / `HALO_TEST_EP_CP_RT_KEEP` | The same two knobs for the EP+CP round-trip (`tests/gpu/parallelism/combined/test_ep_cp_save_reload_roundtrip.py`); attention defaults to `flash_attention_2` there. |
| `HALO_TEST_EP1_KNOB_ATTN` / `HALO_TEST_EP1_KNOB_LAZY` | ep1 `fsdp_shard_ep1_experts` weight-sync suite (`tests/gpu/parallelism/ep/test_ep1_knob_weight_sync.py`): attention impl (default `flash_attention_2`) and `ep_lazy_loading` (default on; `=0` routes the load through `from_pretrained` + EP patching instead). |
| `HALO_TEST_EP_FORWARD_DELAY_MS` | Milliseconds every EP layer of `tests/gpu/profiling/benchmark_sft_ep.py` holds the GPU idle before each forward, a device synchronize then a host sleep (default `0`): the bench band's negative control ([Golden performance baselines](#golden-performance-baselines)). |
| `HALO_TEST_RESUME_FSDP_RESHARD` | `fsdp_reshard_after_forward` (default off = ZeRO-2) for the `fsdp` and `cp` modes of `tests/gpu/trainers/sft/test_sft_checkpoint_resume.py`; the `tp` / `ep` modes ignore it (TP+DP+FULL_SHARD is config-rejected). |
| `HALO_TEST_VLLM_DENSE_SERVER_URL` | Endpoint of the dense half of the online-GRPO/SDPG e2e pair (`trainers/grpo/test_online_grpo_vllm_dense_e2e.py`) and of the vLLM weight-transfer re-init and pool-reconnect suites, which serve the same checkpoint. The pair's two files train different checkpoints and each asserts on its own server's logprobs, so one `VLLM_SERVER_URL` cannot carry both; unset, both read `VLLM_SERVER_URL`, then `http://localhost:8010`. |
| `HALO_TEST_ONLINE_GRPO_MOE_MODEL` / `HALO_TEST_ONLINE_GRPO_DENSE_MODEL` | Checkpoints of the online-GRPO/SDPG e2e pair (`trainers/grpo/test_online_grpo_vllm_moe_e2e.py` and its dense sibling): the MoE half's default `Qwen/Qwen3-30B-A3B-Instruct-2507` (EP/ETP rows) and the dense half's `Qwen/Qwen3-0.6B` (TP and FSDP2-DP rows). Every row asserts on the served logprobs, so the running server must serve the same checkpoint. |
| `HALO_TEST_ENV_GRPO_MODEL` / `HALO_TEST_ENV_GRPO_MAX_STEPS` | Checkpoint (default `Qwen/Qwen3-30B-A3B-Instruct-2507`) and step count (default `2`) of the Environmental-GRPO e2e body (`tests/common/env_grpo_e2e.py`), the vLLM 2-GPU leg's; the SGLang and 4-GPU wrappers own their own checkpoint knobs. |
| `HALO_TEST_ENV_GRPO_SGLANG_MODEL` | Checkpoint of the SGLang Environmental-GRPO e2e wrappers (`trainers/grpo/test_env_grpo_sglang_e2e.py` and its four-rank sibling `test_env_grpo_sglang_4gpu_e2e.py`, default `unsloth/gpt-oss-20b-BF16`); a per-family pass points it at the family and serves the same checkpoint. Its own knob because its server and default family differ from the vLLM leg's. |
| `HALO_TEST_ENV_GRPO_ATTN_IMPL` / `HALO_TEST_ENV_GRPO_LORA_TARGETS` | Per-family overrides for the Environmental-GRPO e2e body: the policy's `attn_implementation` (unset = the loader's auto-selection; `sdpa` for a remote-code family without FA4) and a comma-separated `lora_target_modules` for the `--peft lora` rows (unset = the attention projections read off the checkpoint's index, which MLA families resolve to their own names; needed where the projections do not end in `_proj`, such as Ling's `query_key_value,dense`). |
| `HALO_TEST_ENV_GRPO_4GPU_MODEL` | Checkpoint of the 4-rank Environmental-GRPO e2e (`trainers/grpo/test_env_grpo_vllm_4gpu_e2e.py`, default `unsloth/gpt-oss-20b-BF16`), whose rows hold two parallelism axes at once (EP+ETP, EP+TP, ep4). Its own knob because it shares the 2-GPU file's server but not its default family. |
| `HALO_TEST_VLLM_REINIT_MODEL` / `HALO_TEST_VLLM_REINIT_CYCLES` | Checkpoint (default `Qwen/Qwen3-0.6B`, the dense endpoint's) and connect/sync/disconnect cycle count (default `12`) of `trainers/grpo/test_vllm_weight_transfer_reinit.py`; the pool-reconnect suite (`trainers/grpo/test_weight_sync_pool_reconnect.py`) drives the same server and reads the same checkpoint knob. An unreleased communicator costs ~633 MiB per cycle on each end, so twelve cycles of a leak overshoot the suite's 1 GiB growth budget several times over. |
| `HALO_TEST_VLLM_SERVER_GPU` | The vLLM server's GPU as `nvidia-smi` indexes it, for the same suite's device-memory read. Unset means "every GPU the trainer does not own", which is exactly the server's on the tier's own topology (`TRAINER_CUDA_DEVICES` covers the rest); set it when another job holds a third GPU. |
| `HALO_TEST_SGLANG_REINIT_MODEL` / `HALO_TEST_SGLANG_REINIT_CYCLES` / `HALO_TEST_SGLANG_SERVER_GPU` | The same three knobs for the SGLang re-init suite (`trainers/grpo/test_sglang_weight_transfer_reinit.py`), against `SGLANG_SERVER_URL`: checkpoint (default `Qwen/Qwen3-0.6B`, the dense SGLang pass's), cycle count (default `12`) and the server's GPU for the device-memory read. |

The two local checkpoints the per-suite overrides default to, Gemma4-26B-A4B (`HALO_TEST_GEMMA4_MODEL`,
`HALO_TEST_EP_RT_MODEL`) and gpt-oss-20b (`HALO_TEST_EP_CP_RT_MODEL`), are
`scripts/before_training/patch_vocab.py` outputs at `$HALO_DATA_ROOT/models/<repo name>-patched`
(`patched_checkpoint_dir` in `tests/common/models.py`), which resolves under
`$(HALO_SCRATCH)/models/` in the `make` tiers (they set `HALO_DATA_ROOT=$(HALO_SCRATCH)`). The
suites need a local checkpoint directory, not added tokens, so the tool runs without `--patterns`
and re-saves the source vocabulary unchanged:

```bash
python scripts/before_training/patch_vocab.py --model_id google/gemma-4-26B-A4B-it \
    --output_dir "${HALO_DATA_ROOT:?}/models/gemma-4-26B-A4B-it-patched"
python scripts/before_training/patch_vocab.py --model_id unsloth/gpt-oss-20b-BF16 \
    --output_dir "${HALO_DATA_ROOT:?}/models/gpt-oss-20b-BF16-patched"
```

**Env knob or `args_matrix` row?** `nproc`, `markers`, `timeout` and the tier are per-`TestSpec`, not
per-row, so every row of a matrix runs at the same size, under the same marker set, in the same tier.

A leg that keeps the model, the GPU count and the cost becomes a CLI flag with one row per leg, so
each gets its own pytest node and verdict (`--mode` on `trainers/grpo/test_online_grpo_vllm_e2e.py`;
one `--mode` per parallel shape on `trainers/sft/test_sft_qwen3_dense.py` and
`trainers/sft/test_sft_gptoss_modes.py`; `--ep-loading lazy` / `eager` on
`trainers/grpo/test_offline_grpo_ep_reference.py` and `test_offline_grpo_ep_cp.py`, one row per
`ep_lazy_loading` value, pinned by `tests/cpu/conventions/test_offline_grpo_ep_loading_rows.py`).
Such an entry's markers are the union over its rows, so `-m "gpu and cp"` also selects its non-CP
rows.

A leg that changes the model family, the GPU count or the runtime stays an env override:
registering it as a row would file an EP-on-20B run under a dense entry's markers and its
neighbor's timeout.

`core` is the pre-merge gate, and small-and-fast is its *intent*: ≤2 GPUs, tiny model. Size the host
from the manifest, not from that intent. Over half of `tests/gpu/manifest.py` carries `core`.

Within that tier several entries need 4 GPUs, many declare timeouts of 1500–2400 s, and a large
minority load a real multi-billion-parameter checkpoint (gpt-oss-20b, ZAYA1-8B, Qwen3-30B-A3B,
Qwen3.5-2B, Qwen3-VL-2B and three Ling/Ring checkpoints), so summed worst-case timeouts
run to tens of hours. This page owns tier composition; the manifest is the only place exact counts live.

Where a big-checkpoint entry stays `core`, it is because it is a *correctness gate* — a comparison
against an independent reference that catches a silently wrong result, like
`parallelism/ep/test_ep_correctness.py` (gpt-oss ep2 vs the dense reference). Smoke runs on the same
checkpoint are `full`: `trainers/sft/test_sft_gptoss_modes.py`, `trainers/sft/test_sft_oss20b_default.py`
and `trainers/lora/test_lora_mixed_merged_save.py` all assert only that training completed on the
requested shape and stayed finite, which the gate already covers.

Run `make test-gpu-core` deliberately, not as a quick check, and mark a new heavy or many-GPU test
`full`. The tier measures about 4 h 15 m on 8×B300, and `gpu-tests.yml` budgets 6 h for the whole
`-m "gpu and core"` selection; a new core entry that pushes past that budget belongs in `full`. See
[Continuous Integration](../infrastructure/ci.md).

### A test must not be AI-slop

A test earns its place only if it **fails when the behavior breaks**: mutate the function under
test (flip a sign, drop a clause, return the input) and a real test goes red. These patterns stay
green:

| Anti-pattern | What to do instead |
|---|---|
| **Tautology** — `assert True`, `assert x == x`, or asserting a value the test just set | Assert the computed output against an independently-known expected value |
| **Smoke-only** — call it, assert it "did not raise" | Assert the return value / shape / loss |
| **Mock the thing under test**, then assert the mock returned what you told it to | Mock only the boundary (network, GPU, clock); call the real function |
| **Vacuous assertion** — `assert out is not None`, `assert len(out) >= 0` | Assert the exact value, correct length, or a tight bound with a reason |
| **Test the framework** — `assert isinstance(cfg, ParallelismConfig)` | Assert the property your code computes (`cfg.data_parallel_size == 4`) |
| **Swallow-and-pass** — `try: real_call() except Exception: pass` | Let it raise, or `pytest.raises(ValueError, match=...)` |
| **Golden with no meaning** — a literal pasted from one run | Derive the expected value from the math; if it is a true golden, say why |

`tests/cpu/parallelism/test_parallelism_config.py` is the voice to copy. Cover the edge cases —
empty input, the unsupported combination, the degenerate near-zero case — that is where
regressions hide.

### Golden performance baselines

`make bench` runs the rows of `tests/gpu/profiling/run_baseline_benchmarks.sh` on one 8-GPU node, each one
`torchrun` launch at seq 4096, batch 1, with gradient checkpointing and the production attention and
grouped-GEMM defaults:

| Rows | Script | Keys |
|---|---|---|
| Qwen3-8B under FSDP2, CP2, TP2 | `benchmark_sft_dense.py` | `sft_dense_qwen3-8b_{fsdp,cp2,tp2}_s4096` |
| gpt-oss-20b under EP1, EP2, EP8, EP8 × TP2, EP4 × ETP2 | `benchmark_sft_ep.py` | `sft_ep_gpt-oss-20b_{ep1,ep2,ep8,ep8_tp2,ep4_etp2}_s4096` |
| the EP8 row's gathered save (`--measure_save`) | `benchmark_sft_ep.py` | `sft_ep_gpt-oss-20b_ep8_gathered_save` |

Every rank prints one `__HALO_BENCH__` line per key (`tests/common/reporting.py`) with its rank, the world
size, the command and the metrics. The key names the golden `tests/baselines/bench/<key>.json`: the commit,
image, date, world size and command it was measured with, and the metrics of each of its repeats, at least
three. `make bench BAND=1` holds the run to them (`tests/common/equivalence/bench_band.py`, evidence in
`evidence/bench_band-<revision>.json`); `make bench-band` re-checks the logs `make bench` left at HEAD
without GPUs. Each gated metric has a band around the mean of its baseline repeats, give or take the larger
of a margin and twice their spread. `tokens_per_second` (mean over ranks) and `save_seconds` must stay within
theirs, on either side, at a 2% margin: a faster run fails too, since a band a speedup left behind would pass
the regression that undoes it, so a change that is meant to speed a row up re-records its baseline.
`peak_allocated_gb` and `save_peak_rss_gib` (the save's peak resident host memory), each the highest over
ranks, must not rise above their band's upper edge, at a 0.5% margin. That edge is never below the highest
repeat, and it is not the highest repeat itself because a peak that varies between runs would fail one run
in four against the highest of three. MFU, S-MFU and TFLOPS stay under `metrics.diagnostics`, never gated.
A key compares only with a baseline measured by the same command on as many ranks of the same GPU model, so
editing a row means re-recording it; a measured key with no baseline, and a baseline the run did not
measure, fail too.

`make bench RECORD=1` runs the rows three times on a clean tree and writes the baselines only when each
repeat falls inside the band of the other two, removing any baseline the run no longer measures; it touches
nothing else under `tests/baselines/`. Commit the files it writes. The gate's negative control holds the GPU
idle before every EP layer's forward:
`make bench BAND=1 EXTRA_DOCKER_ENV="-e HALO_TEST_EP_FORWARD_DELAY_MS=2"` must fail every EP row's
`tokens_per_second` and pass the dense rows and the save.

**EP4 runs with ETP2**: pure `ep_size > 2` with `ep_group_size < nvlink_domain_size` is rejected at config
time, since its DeepEP combine races FSDP2's DP-wide collectives
([Expert Parallelism](../parallelism/expert-parallelism.md#single-domain-multi-group-ep-races-and-hangs)).

### Rank-layout golden

The rank-layout golden pins, for every parallel shape, the EP, dispatch,
expert-TP, expert-replica, DP-scope and CP groups and the pipeline chains (each as its distinct groups
plus the one each rank holds), each rank's EP, dispatch, expert-TP, CP and data-parallel coordinates,
and the `EPConfig` flags that decide where expert gradients are synced. The TP and FSDP/HSDP
`DeviceMesh` dimensions are not recorded. The shapes come from `tests/common/goldens/shape_grid.py`:
every shape `ParallelismConfig` accepts with a pipeline stage of at most 32 ranks, at every NVLink
domain dividing the stage, and the same sweep at 64, 72, 128, 144 and 256 ranks on 8-GPU nodes and
72-GPU NVL72 racks. `tests/common/goldens/rank_layout.py` builds every rank's `ParallelismConfig`,
`EPConfig` and `CPConfig` against a stubbed `torch.distributed`, and `tests/common/goldens/rank_layout.json`
holds a digest of each shape's layout plus the full layout of 40 shapes, so a reviewer can read the
layout and a failure reads as a diff. `tests/cpu/parallelism/test_grid_goldens.py` checks it, and every
other golden over the same grid (each a `GridGolden` in `shape_grid.py`), against today's code.

A refactor of the rank math leaves the file unchanged. Regenerate it only for a deliberate layout
change, in a commit that says so:

```bash
python -m tests.common.goldens.rank_layout          # exits non-zero when the committed file is stale
python -m tests.common.goldens.rank_layout --regen  # rewrites it from today's code
```

`tests/cpu/parallelism/test_parallelism_config_gate_order.py` pins the order of
`ParallelismConfig`'s gates: a shape breaking two adjacent rules reports the earlier one.

### EP group-order golden

The EP group-order golden pins, for every shape of the same grid, the `new_group` calls `EPConfig`
makes, in order and with their keyword arguments, which of those calls built each group a rank keeps,
and each rank's `ep_group_idx` and `expert_replica_ranks`. `new_group` is a collective over the default
group, so a rank making other calls than rank 0 hangs a real job; `tests/common/goldens/ep_group_order.py`
raises there. Two calls over the same members build two communicators, so a reordered call fails this
golden even where the rank-layout golden, which compares memberships, passes. The record is built with
`DIST_NCCL_TIMEOUT_MINUTES` off its default, so a timeout is written as `get_nccl_timeout()` only when
the group reads it from that call. `tests/common/goldens/ep_group_order.json` holds a digest per shape
plus the full record of node-local and cross-node EP at expert TP 1, 2 and 4 on one pipeline stage, EP
or expert TP alone on two, and shapes whose one-rank EP groups are skipped;
`tests/cpu/parallelism/test_grid_goldens.py` checks it.

A refactor of `EPConfig`'s group construction leaves the file unchanged. Regenerate it only for a
deliberate change to the calls, in a commit that says so:

```bash
python -m tests.common.goldens.ep_group_order          # exits non-zero when the committed file is stale
python -m tests.common.goldens.ep_group_order --regen  # rewrites it from today's code
```

### Loader collective-trace golden

`tests/cpu/parallelism/test_load_collective_trace.py` pins what `load_distributed_model` issues on
ranks of a 4-rank job on two 2-GPU nodes, for every branch of its loader ladder from a local directory
and from the Hub, and what the frozen reference load issues (the cases are `LOAD_CASES` in the
generator): process-group builds in order with their members, collectives with their operation,
payload and group, device meshes, c10d-store operations with their keys, node-load phases with their
labels, and the model builds between them. `tests/common/collective_recorder.py` runs a load as one
simulated rank over torch's fake process group and an in-memory store, with only the launcher's
variables in the environment. It replaces each `torch.distributed` function under every name a module
holds it by, so a collective DTensor issues through `distributed_c10d` is recorded too, and a
collective or functional collective it has no model of raises. Another rank receives the object
broadcasts rank 0 recorded. `tests/common/goldens/load_collective_trace.py` loads tiny checkpoints,
from their directories or from a fake Hub cache, through the real loaders, with only the model builds
kept on the CPU, and `tests/common/goldens/load_collective_trace.json` holds the traces.

A refactor of the loaders leaves the file unchanged. Regenerate it only for a deliberate change to what
a load issues, in a commit that says so:

```bash
python -m tests.common.goldens.load_collective_trace          # exits non-zero when the committed file is stale
python -m tests.common.goldens.load_collective_trace --regen  # rewrites it from today's code
```

### Weight-sync wire golden

`tests/cpu/weight_sync/test_wire_pins.py` pins the vLLM weight-transfer bootstrap's wire contract
against `tests/common/goldens/weight_sync_wire.json`: the arguments rank 0's rendezvous `TCPStore` takes
(a master that waits for the engine's workers, listening on the advertised listener), its one write, the
pickled `ncclUniqueId` under `broadcast_from/0/0`, the id the communicator initializes with, and the
class's module, qualname and fields. `tests/common/goldens/weight_sync_wire.py` runs the real bootstrap
against a recording store and a faked libnccl that hands out a fixed id. The vLLM image unpickles that id
as its own class through a shim at the same module path, so the test also checks that `Dockerfile.vllm`
writes the shim, and asserts it at build, for the module the golden names.

Moving `ncclUniqueId` changes the pickle bytes: regenerate the golden in the commit that moves the class,
and move the shim in `Dockerfile.vllm` with it:

```bash
python -m tests.common.goldens.weight_sync_wire          # exits non-zero when the committed file is stale
python -m tests.common.goldens.weight_sync_wire --regen  # rewrites it from today's code
```

### Weight-sync request-transcript golden

`tests/cpu/weight_sync/test_request_transcript.py` pins what each engine's weight-sync client sends
through its public lifecycle (construction, `init_communicator`, `update_named_param`,
`reset_prefix_cache`, `abort_weight_update`, `close_communicator` and the at-exit hook) against
`tests/common/goldens/weight_sync_transcript.json`, in twelve scenarios per engine: a dense and a MoE
sync, a group formation failing on the server before and after the trainer formed its half, on the
trainer and at its thread's start, a finish failing once and every time, an abort before any chunk landed
and one mid-stream, an unusable client at exit, and an exit after a sync. Each step records its outcome;
every request with its route, query, body, timeout, whether it went through the client's retrying
session, the retry policy it went out under (every field set away from urllib3's defaults) and how many
times the engine received it; what went on the NCCL wire; what happened to the local group; and the
at-exit hooks the client held after it. The clients run against `FakeVLLMServer` and `FakeSGLangServer`
(`tests/common/fake_engines.py`) with NCCL and CUDA faked on the CPU.

A refactor of the clients leaves the file unchanged. Regenerate it only for a deliberate change to what
a client sends, in a commit that says so:

```bash
python -m tests.common.goldens.weight_sync_transcript          # exits non-zero when the committed file is stale
python -m tests.common.goldens.weight_sync_transcript --regen  # rewrites it from today's code
```

### Registry-order golden

`tests/cpu/conventions/test_registry_order.py` pins what each toolkit registry holds, in its own order,
against `tests/common/goldens/registry_order.json`: the EP layer classes, the EP MoE-block and `model_type`
claims and the balancing registries those classes fill as they are defined, the CP attention, PP backbone
and causal-LM head claims, the weight-sync clients, the Liger family specs, the sequence-classification
heads, the named optimizer builders and the environment types. A registry fills as its classes are
defined and its modules imported, so moving modules or reordering imports can change the order without
changing any class. The check renders in a fresh interpreter, so the order is the one the code and the
generator's imports produce, whatever a test imported first. Classes are recorded by name, not module, so a
move that keeps each class's name and order leaves the file unchanged. Regenerate it only for a deliberate
change, in a commit that says so:

```bash
python -m tests.common.goldens.registry_order          # exits non-zero when the committed file is stale
python -m tests.common.goldens.registry_order --regen  # rewrites it from today's code
```

### Move kit

A structural change — a module moved or renamed, code carved between modules, a package renamed —
proves it changed no code with `make verify-move BASE=<sha>`
(`tests/common/equivalence/move_kit.py`). The kit reads both commits from git (`HEAD_REF`, default
`HEAD`, against `BASE`), parses the wheel's packages and `scripts/` without importing them, and writes
`evidence/move_kit-<head sha>.json` (git-ignored) with one verdict per check:

| Check | Fails when |
|---|---|
| `renames` | a file rename the change declares is missing from `git diff -M50%`, or below `--min-similarity` (90 by default) |
| `definitions` | a module docstring, function, class header, method or module-level assignment hashes differently once each first-party name it reads is spelled in head terms, or head adds one the change does not declare |
| `public_names` | a public top-level name appears or disappears |
| `string_literals`, `raise_templates` | the multiset of string literals (docstrings aside) or of raised expressions changes |
| `side_effects` | a module's ordered import-time statements change: bare calls, assignments that call first-party code or set an attribute or item, first-party decorators, control flow, and imports marked `noqa: F401` or `E402` |
| `goldens` | a golden file (`tests/common/goldens/*.json`) the change does not declare changes, a declared one does not, or a golden's generator fails its check against head's code |

Spelling a name in head terms follows every rename git finds, each symbol rename the change declares,
and each definition that moved to another module under its own name; a name a function binds hides the
module-level one. A path in a string follows its file's rename, or its directory's when every Python file
of that directory moved to one new directory. A library decorator (`@dataclass`, `@lru_cache`) is the
definition's business, not an import-time effect, so a carve may move it. Each patch in head on a name that
a touched module only imports is listed under `patch_targets` for review: the patch replaces that module's
binding while the defining module keeps its own. A renamed method's callers through an instance read as
changed.

Declare what the definitions do not show in a JSON file under the repo, passed by its repo-relative path as
`RENAMES=`:

```json
{
  "files": {"src/<old path>.py": "src/<new path>.py"},
  "symbols": {"src.<module>:<old qualname>": "src.<module>:<new qualname>"},
  "added": ["src.<new module>:__doc__", "src.<new module>:logger"],
  "goldens": ["weight_sync_wire.json"]
}
```

`added` names the definitions a carve adds (a new module's docstring and logger), `goldens` the golden
files the change regenerates on purpose (a golden that names a moved module). A declaration the trees
contradict fails. `MOVE_ARGS` passes further flags: `--require` names the checks a carve's proof rests on
(one that extracts new functions fails `definitions` by design), `--min-similarity` lowers the bar for a
rename that carries edits, and `--ignore-docstrings` compares a prose-only change:

```bash
make verify-move BASE=<sha> RENAMES=<map.json> \
  MOVE_ARGS="--require renames string_literals raise_templates side_effects goldens --min-similarity 70"
```

`make verify-move` exits non-zero when a required check fails, and runs as the caller, so the evidence is
the caller's. `tests/cpu/conventions/test_move_kit.py` seeds each kind of change on a scratch repository and
moves a real module and a real package.

### Comparison harnesses

The checkpoint, EP, CP and efficiency harnesses below share one contract. Each exports `BASE` and
`HEAD_REF`, runs its own module in each tree with that tree's code (so `BASE` must already carry it), and
writes `evidence/<harness>-<head sha>.json` (git-ignored) with every delta keyed by its path in the output.
`ACCEPT="<case> ..."` names the cases whose deltas a flagged change declares, each a case path or any prefix
of one (`ACCEPT=cpu/qwen3_moe` covers every EP record of that family); their deltas stay in the evidence under
`accepted` and do not fail the verdict. The EP, CP and efficiency harnesses also run a negative control: the
head tree reruns one case with a deliberate perturbation, and the verdict fails when that record matches its
base one or raised, since a perturbation that raises proves nothing about what the comparison sees.

### Checkpoint artifact harness

`make verify-artifacts BASE=<sha>` (`tests/common/equivalence/checkpoint_artifacts.py`) saves the same
artifacts in each tree with that tree's own savers and compares them file by file: the file set and
permission bits, each safetensors header and tensor (dtype, shape and a digest of its bytes), each JSON file
by value, each torch file by key and tensor, any other file by its bytes. It fails on any delta, and on any
case outside `KNOWN_RAISING` that raises. `FAMILIES="gpt_oss qwen3_moe"` limits the run to those tiny
families. Both producers run under one string hash seed, since a set's order reaches `adapter_config.json`.

The cases save a tiny model of every family (the dense roster, every EP family and its multimodal
wrappers) at fp32 and at bf16 (an EP router kept at fp32 by `fp32_router`), through every saver mode one CPU
process runs, as an export and as a training checkpoint: the FSDP2 gathered stream over the per-layer DP
wrap on a single-rank mesh, CP over plain parameters (one DP rank) and over that FSDP2 wrap, the TP saver,
EP with FSDP2-managed experts (`ep_size == 1`) and with FSDP-ignored EP layers (`ep_size > 1`), native
expert LoRA as an adapter and merged (with its resume adapter), and attention LoRA through the PEFT saver. A
training checkpoint of a full-weight mode adds the optimizer shard and its meta after one AdamW step. A case
that raises writes `raised.json` in place of its files. The per-rank EP save, the PP stage save and every
multi-rank gather need more than one rank, which the GPU checkpoint suites cover. The seeded saver changes
it catches are in `tests/cpu/checkpoint/test_checkpoint_artifacts.py`.

### EP bitwise harness

`make verify-ep BASE=<sha>` (`tests/common/equivalence/ep_bitwise.py`) runs every EP family's tiny model in
each tree and compares the records bit for bit. A forward case records the loss, the logits, every parameter
gradient, the routing each EP layer dispatched and the expert-load counts; every case records the ordered
parameters (name, dtype, shape, trainability), the state-dict keys, the weight-decay param groups and each
parameter's gradient hooks. A hook is recorded by what it does to a gradient of ones (each all-reduce it
issues, with its op, dtype and group ranks, and the scale it leaves), so a router hook swapped for an expert
hook is a delta while a rebuilt hook that acts alike is not.

The CPU tier (`TIER=cpu`, the default) runs the per-expert loop on one process. Its forward variants are the
defaults, an fp32 router on FSDP-ignored experts, native expert LoRA with dropout, an armed routing-replay
mask and bias-update balancing; its construction variants build every rank of a layout on stubbed process
groups: one two-rank EP group (in-backward router and expert hooks), two EP groups (deferred sync, no hook),
that one group with PEFT saving the router (`modules_to_save`, re-hooked), and expert TP 2 (each rank's
shard). `TIER=gpu` runs the grouped-GEMM path in bf16 under `torchrun` and `full_determinism` (DeepEP's
deterministic dispatch included): the forward variants on one rank, `fp32_experts` added to the fp32 one, then
`ep2` (2 GPUs) and `ep2+etp2` (4 GPUs), each with `HALO_EP_SHARED_OVERLAP` off and on. The verdict also fails
on a raise outside `KNOWN_RAISING`. Each launch's negative control reruns its first case outside
`KNOWN_RAISING` with every expert bank rotated by one expert. `tests/cpu/parallelism/test_ep_bitwise_harness.py`
holds the seeded EP changes it catches.

### CP equivalence harness

`make verify-cp BASE=<sha>` (`tests/common/equivalence/cp_equivalence.py`) runs a two-rank CP group under
`torchrun` in each tree, on one tiny model per Ulysses wrapper class, at two levels: the first attention layer
alone on fixed hidden states with the position embeddings its decoder hands it, and the model under
`UlyssesCPModelWrapper` through its loss (MoE experts on the `ep_size == 1` wrappers). Each rank records the
outputs and every gradient. The base tree runs twice and must repeat bit for bit before base and head are
compared bit for bit; a record whose base does not repeat is compared forward bit for bit with each gradient
inside its base-vs-base max-abs envelope, and the evidence names it under `enveloped`. The tiny Mistral4
steps its llama-4 query scale every 32 positions, so the scale varies along the sequence.

`TIER=gpu` runs bf16 under `full_determinism`, with the flash kernel CP picks for each family (FA4 on
Blackwell where the family allows it, FA2 otherwise, FA3 on Hopper) called with `deterministic=True`. The CPU
tier runs fp32 on gloo, with a plain torch attention under the flash contract in place of the kernel (it
refuses a keyword it does not implement) and the list all-to-all carried over `all_to_all_single`. The
verdict also fails on any raise. The negative control moves one attention weight of the first family by a few
ulps. The seeded CP changes it catches are in `tests/cpu/parallelism/test_cp_equivalence_harness.py`.

### Efficiency harness

`make verify-efficiency BASE=<sha>` (`tests/common/equivalence/efficiency_callback.py`) drives each tree's
`EfficiencyCallback` through the same scripted steps and compares, per case, every logs dict it fills, every
record logged under the package while it runs, and the `snapshot_efficiency` row. CUDA answers as a scripted
B300, the clock and the token counter follow a fixed schedule and a bound source supplies each step's
documents, so every figure is a function of the code. At tiny scale the setup summary rounds the parameter
and FLOPs figures to zero, so a FLOPs change shows in the per-step figures and the snapshot.

The cases cover every family on one rank and under EP with FSDP2, three families on every parallel shape (EP,
ETP, TP, CP and their supported pairs, FSDP2 with DTensor or plain experts, and the first and last stage of a
two-stage pipeline alone, with EP and with ETP) built with the real EP, TP and FSDP2 wraps and pipeline stage
cut on torch's fake process group, and every scenario: an unknown GPU or peak, fp8, fp32 and tf32, a model
handed over only at train begin, uncounted or implausible tokens, the config attention term, a frozen LoRA
base, full-model params, an estimate that raises, one expert, no top-k, no layer list (also on a pipeline
stage, which owns its share of the config's depth) and no parameters. Each tree runs its own roster, so a
case only one of them has is listed under `new` or `dropped` in the evidence rather than compared. The
verdict also fails on any raise. The negative control moves one step's clock by a tick. The seeded changes it
catches are in `tests/cpu/callbacks/test_efficiency_harness.py`.

## Docs

Update the owning doc page in the same PR when you change `src/` (`skills/docs/docs-ownership.md` is the map) and
follow the anti-slop charter carried by the `/docs` skill: American English, active voice, short
sentences, tables only for real matrices, no marketing register.

`docs.yml` runs three jobs, and `make docs` runs only the first: the link, anchor and code-reference
check over `agent-docs/`, `human-docs/`, `skills/` and the root markdown, which blocks a merge; a
`diagrams` job that re-runs every `scripts/diagrams/gen_*.py` and byte-compares the result against the
committed PNGs under `agent-docs/assets/` — touch a generator and you owe `make diagrams` plus the
regenerated figure in the same commit; and `markdownlint`. The last two are advisory.

## Proof of Value

| Change type | What to show |
|-------------|--------------|
| No-op / refactor | Loss is **bitwise-identical** to `main` at a fixed seed, and the original checkpoint still loads |
| Behavior change | An end-to-end test demonstrating the new behavior or convergence |
| Performance | Before/after **tokens/s/GPU** and **peak memory** (utilization is a diagnostic, not the headline) |

State the GPU, model, and sequence/batch shape with any number. A measured net-slower result is
first-class content — report it with the reason.

## Submitting a PR

1. **Get approved first** — an accepted issue plus a maintainer's `/approve @your-handle` on it. An un-approved PR is closed by
   `pr-gate.yml`; once approved, reopen it, or ask on the issue and a maintainer will — the gate re-runs on
   reopen. Merging a PR adds you to the allowlist, so the gate applies once. `/approve` assigns you to the
   issue, which keeps it open while you work; an unassigned idle issue goes stale after 30 days and closes
   7 days later.
2. **Branch** off `main` — in your fork, unless you have write access. Every PR is squash-merged, and
   every commit in it must carry a verified signature (SSH or GPG), forks included: GitHub does not
   merge a PR while any of its commits lacks one. Re-sign earlier commits with
   `git rebase --exec 'git commit --amend --no-edit -S' <base>` (the `main` commit the branch starts
   from) and force-push; the one-time signing setup is in `CONTRIBUTING.md`.
3. **Pass the gates.** `make lint`, `make format`, `make seed-hf-cache` (again when
   `tests/common/models.py` or `examples/` gain a repo), then `make test-cpu` (plus
   `make test-gpu-core` for GPU-affecting changes), `make docs`. Hosted CI runs `ruff`,
   `actionlint`, the convention tests and the docs checks; the test tiers run locally, so report their
   result in the PR. A PR that changes a path listed in `.github/sensitive-paths` also needs the
   `sensitive` label, which a maintainer adds once the change is approved.
4. **Fill the PR template** — what and why, type of change, Proof-of-Value evidence, checklist.
5. **No secrets.** Never add `keys/`, `.env`, `*.pem`, or any credential.

Report vulnerabilities **privately** — see `SECURITY.md`, never a public issue.

## Where to look next

- [Development Environment](development-environment.md) — host `.venv`, dev container, env vars
- [Trainer Architecture](../reference/trainer-architecture.md) — how trainers compose with the mixin
- [Troubleshooting](../reference/troubleshooting.md) · [Debugging](../reference/debugging.md)
- [Adding a New Model](../models/adding-a-model.md)
