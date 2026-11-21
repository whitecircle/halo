# Contributing

Every command goes through `make`: contributors, CI and the agent skills call the same targets, and
the Docker invocation lives once in the `Makefile` (`make help` lists them).

## Ground rules

- **The host has no usable Python.** PyTorch, DeepEP and Flash Attention live only in the images, so
  anything that runs, runs inside one, where `python`, `torchrun`, `pytest` and `ruff` are on `PATH`
  (uv installs into the system interpreter; no venv).
- **`make lint`, `make format`, `make precommit` and `make docs` are the host-runnable gates.** ruff runs
  through `uvx ruff@$(RUFF_VERSION)` at `uv.lock`'s pin (else the `ruff` on `PATH`); the docs target
  needs stdlib Python 3.12. Tests and benchmarks run in the image.
- **Credentials live in the repo-root `.env`** (`cp .env.example .env`): the GPU `make` targets pass it
  with `--env-file` and fail without the file.
- **Markdown lives in `agent-docs/`, `human-docs/` and `skills/`**, plain GitHub markdown; the root files
  (`README`, `CONTRIBUTING`, `SECURITY`, `CODE_OF_CONDUCT`, `AGENTS.md`, `CLAUDE.md`, `.github/*`) are
  the exception, and scratch markdown goes to `/tmp`.
- **Large outputs go to a verified large volume.** The `make` targets put the caches and `TMPDIR` under
  `HALO_SCRATCH`, which resolves to the local mount other than `/` with the most free space when unset;
  confirm it with `findmnt` / `df -h` or export it
  ([Docker → Standard training launch](../infrastructure/docker.md#standard-training-launch)).

## The gate (read first)

Every change starts as an accepted issue and a maintainer's `/approve @you` on it, **before** any PR;
an unapproved PR is closed automatically. The rules are in [`CONTRIBUTING.md`](../../CONTRIBUTING.md),
the workflow mechanics in [Continuous Integration](../infrastructure/ci.md#contribution-gate).

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
make check-lock               # check uv.lock installs into the image
make seed-hf-cache            # configs + tokenizers the CPU tests read into HF_CACHE (no weights)
make test-cpu                 # sanity check; needs Docker, not a GPU
```

Point any run target at another image with `IMAGE=` (`make test-cpu IMAGE=halo:hopper`); the `build-*`
and `push-*` targets name their own tags. What each image carries and how it builds:
[Docker](../infrastructure/docker.md).

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
| Docs links, anchors and code references | `make docs` | host (`scripts/docs/check_links.sh`) |

The server tiers need their compose server already serving on a GPU outside `TRAINER_CUDA_DEVICES`
(default `0,1,2,3,4,5,6`, which the target pins as the trainer's `CUDA_VISIBLE_DEVICES`): weight sync is
an NCCL broadcast, and a rank cannot broadcast to itself. Without `EFA=1` the targets force the
no-fabric recipe both compose bases default to; `EFA=1` passes `/dev/infiniband` and the fabric
variables instead, matching a server started with its EFA overlay
([Rollout Servers → Servers on other nodes](../infrastructure/rollout-servers.md#servers-on-other-nodes-efa)).
The SGLang server needs only cuMem parity on top, the compose default
([NCCL transport](../infrastructure/rollout-servers.md#nccl-transport-sglang)).

Each server tier runs **one pass per served model**: the tests broadcast the trainer's own weights into
the served model and assert the served policy moved, so server and trainer hold the same checkpoint.
`SERVER_TIER` defaults to `not moe` (serve `Qwen/Qwen3-0.6B`); the MoE passes restart the vLLM server on
`Qwen/Qwen3-30B-A3B-Instruct-2507` for `SERVER_TIER='moe and not gptoss'` and on
`unsloth/gpt-oss-20b-BF16` for `SERVER_TIER='moe and gptoss'`, while SGLang's MoE tests are gpt-oss only
(`SERVER_TIER=moe`). The vLLM server needs `VLLM_REASONING_PARSER=qwen3`: the benchmarks send a
per-effort CoT budget, which draws a 400 without a reasoning parser (the target's health guard prints
the compose line).

The env-GRPO rows train on a native tool call, so the vLLM server also needs the served family's tool
parser ([Tool parsers are per family](../infrastructure/rollout-servers.md#running-it)). The compose
default, `hermes`, reads JSON `<tool_call>` calls only. The gpt-oss pass serves the image's parser
plugins instead (the Makefile's `VLLM_GPTOSS_PARSERS`, which the health guard prints into its gpt-oss
compose line), and they replace the `qwen3` reasoning parser. A per-family pass
(`HALO_TEST_ENV_GRPO_MODEL`) takes its family's row of that table. Under a wrong parser every call stays
text, so each env-GRPO row on the draw task (all but the one-process `react_math` rows, which send no tool schema)
raises by its second round.
SGLang's default parser (`auto`) reads it off the chat template where it can and disables it elsewhere: the
Qwen3-30B pass needs `SGLANG_TOOL_PARSER=qwen25`
([Tool parsers on SGLang](../infrastructure/rollout-servers.md#tool-parsers-on-sglang)).

```bash
make train CONFIG=examples/sft/qwen3/qwen3-4b-ultrachat.yaml NPROC=8
make train CONFIG=examples/sft/gptoss/gptoss-20b-multinode-ep.yaml \
  NPROC=8 EXTRA="--expert_parallel_size=8 --use_grouped_gemm=true"
make train METHOD=preference/smpo CONFIG=<config>    # METHOD: any `halo launch --list` entry; default sft
```

`make clean` prunes `wandb/`, the ruff and pytest caches and the `$HALO_SCRATCH` test scratch, and
leaves `checkpoints/`, where every example writes its run.

## Code style

- **ruff, line length 119.** Selected: `E, F, I, W, UP, B, C4, SIM, T20, PLC0415, D200, D202, D210, D300,
  D419, ERA001`; `make lint` and the blocking CI job enforce the full set, CI with
  `ruff format --check` ([Continuous Integration](../infrastructure/ci.md)).
- **No inline imports** (`PLC0415`), in `tests/` as in `halo/`. A function-level import is reserved for a
  genuinely optional or arch-specific dependency (`flash_attn`, `deep_ep`, `deep_gemm`); a circular import
  is never that exception.
- **One docstring format** (`D200`, `D202`, `D210`, `D300`, `D419`): triple double quotes, a one-line
  docstring on one line, no padding inside the quotes, no blank line after a function's docstring, never
  empty.
- **No commented-out code** (`ERA001`); a note that merely parses as Python (`# Default: n * 4`) is
  reworded as prose. **No stray `print()` in `halo/`** (`T20`); `tests/**` relaxes `E402`, `T20`,
  `E741`, `B007`, `B023` and `SIM117`, `scripts/**` relaxes `T20`.
- **Reuse over re-implementation.** New trainers extend `DistributedTrainerMixin`; new parallelism layers
  reuse the EP-layer base hooks ([Trainer Architecture](../reference/trainer-architecture.md)).
- **Naming, named constants, module order, comments and docstrings** follow the Code style section of
  `CLAUDE.md`.
- **A field help is short.** A config or script-argument `help` gives the effect and what the default
  means in one or two sentences, optionally closed by a `See agent-docs/<page>#<heading>.` pointer the
  reference check resolves; rationale and measurements go to the
  [Configuration Reference](../reference/configuration-reference.md) or the method's page.

## AI-assisted contributions

AI-assisted contributions are welcome on the terms in [`CONTRIBUTING.md`](../../CONTRIBUTING.md#the-one-rule);
`CLAUDE.md` and `AGENTS.md` bind agents and humans alike.

## Tests

CPU tests run under pytest. GPU tests are scripts launched through a manifest (`tests/gpu/manifest.py`),
each row under `torchrun`, or as one process that starts its own ranks for a `launcher="python"` row
(a `halo launch` driver, such as `trainers/other/test_entry_scripts.py`). Most of the conventions below
are enforced by `tests/cpu/conventions/`.

- **Test behavior, not implementation.** Assert loss finiteness, loss decrease, output shapes and the
  rejection of unsupported configs, never private internals.
- **A CPU test file is a plain pytest module** ending in
  `if __name__ == "__main__": raise SystemExit(pytest.main([__file__, "-v"]))`, so a standalone run
  executes exactly what `pytest` collects; `tests/cpu/conventions/test_test_conventions.py` refuses any
  other entry, a hand-listed runner or a printed pass/fail summary.
- **The `cpu` marker is applied by path.** `tests/conftest.py` marks every item under `tests/cpu/`, so no
  file re-declares it in `pytestmark`.
- **Imports resolve from the image, not from `sys.path` surgery.** `PYTHONPATH=/workspace` is baked in
  and `tests/conftest.py` puts the repo root on the path, so `from halo...` / `from tests.common...` work
  under `pytest` and a standalone `torchrun`. A tool's driver lives in `halo/` and a test imports it from
  there (`from halo.checkpoint.tools.model_merge import merge_models`); `scripts/` is a namespace package, so a
  test that drives a CLI's own parser imports it by dotted path
  (`from scripts.environments.inference.eval_code_contests import parse_args`), and
  `tests.common.repo.load_script_module` loads a file by path where a test needs its own module object or
  the file sits outside the packages (`docker/`).
- **Every first-party reference resolves without running.**
  `tests/cpu/conventions/test_references_resolve.py` statically resolves each first-party import,
  `patch` / `monkeypatch` target, attribute chain on an imported module or class, dotted name and
  repo-root path in the Python and text files under `halo/`, `scripts/`, `tests/`, `docker/` and
  `.github/`, the image builds, `Makefile`, `pyproject.toml`, `.env.example`, `.pre-commit-config.yaml`
  and the recipes, so a moved module or a dead patch target fails the CPU tier even where only a GPU
  script names it. An attribute a third-party base supplies (`Function.apply`) is listed per base in
  `_EXTERNAL_BASE_ATTRIBUTES` (`scripts/docs/references.py`); text that only reads as a reference is
  exempted per file in the test's `_NOT_REFERENCES`. The sweep also holds each reference to the binding
  it means: an import names the module that defines what it takes (a re-binding module is a second path
  that keeps a moved symbol reachable), and a patch of a module's own top-level name must target a name
  some function reads when it runs, or it intercepts nothing.
- **A message names symbols, not source paths.** `tests/cpu/conventions/test_messages_name_symbols.py`
  fails a string in `halo/` or `scripts/` that names a package file by path: an error, warning or field
  help names the function, class or constant its reader acts on. Docstrings are exempt, and a `scripts/`
  tool a user runs keeps its path.
- **A move rewrites every reference in the same change**: the imports (a `from` line split per
  destination), the `patch` / `monkeypatch` / `caplog` / `sys.modules` / `import_module` strings and the
  repo-root paths, leaving no re-export. Splice the source text at the AST nodes' offsets, or use libcst
  in a throwaway environment (`uv run --no-project --with libcst <script>`, which leaves `uv.lock` alone);
  never regenerate a file with `ast.unparse`, which drops its comments. `make verify-move BASE=<sha>`
  proves the move changed no code ([Move kit](#move-kit)).
- **The import layering is read off the import graph.** `tests/cpu/conventions/test_import_layering.py`
  holds the layer table (`_LAYERS`: a unit imports only units of lower rows, `cli` imports nothing from
  the package, the tool-safe units reach no process group, parallelism or rollout sync, one lazy
  `deep_ep` importer, no module cycle), refuses a raw toolkit env read, and asserts `_LAYERS` equals the
  table [Architecture](../reference/architecture.md#layer-table) renders. `TYPE_CHECKING` imports are exempt.
- **Convention tests also run on the hosted runner.** The `conventions` job in `.github/workflows/lint.yml`
  runs the files it lists on the standard library and pytest alone, so such a test reads the source with
  `ast` and takes repo helpers from `tests.common.repo`; a listed test that runs a probe script names the
  files the probe collects in `_PROBE_COLLECTS` (`tests/cpu/conventions/test_hosted_conventions_are_stdlib.py`).
- **Remote code loads into the test process's own module cache.** `tests/cpu/conftest.py` points
  `HF_MODULES_CACHE` at a scratch dir per process before transformers is imported. Do not redirect the
  cache inside a test around a `trust_remote_code` load: transformers puts each cache it copies into on
  `sys.path` and never removes it, so the `transformers_modules` package can stay bound to the test's
  temp dir and break every later local remote-code load in that worker.
- **An oracle a test loads itself is finalized as production loads are.** `finalize_loaded_model` runs once its
  weights sit on their device: transformers 5 leaves a remote-code family's rotary table as uninitialized memory,
  and a bare oracle reads that noise as the parallel run's error. `tests/cpu/model_loading/test_load_finalization.py`
  fails a load by a family's own `trust_remote_code` in a module that runs a forward without it.
- **Deterministic, seeded data.** GPU tests generate synthetic problems from a fixed seed; only rank 0
  prints. A seed alone does not fix a chat-templated batch, because `strftime_now` stamps the live date
  (gpt-oss's harmony system message), so build fixed batches through
  `tests.common.ep_reference.fixed_chat_batch`, which pins `CHAT_TEMPLATE_NOW`.
- **GPU tests register in the manifest.** Drop the script under `tests/gpu/` and add one `TestSpec`
  ([fields](../../skills/write-tests/harness.md#register-in-the-manifest-testsgpumanifestpy)); a script
  missing from the manifest fails collection, as does a `bench*.py` in neither the manifest nor
  `_UNMANIFESTED_BENCHMARKS`.
- **Run GPU tests through pytest**: `make test-gpu-core`, or a marker expression over the directory
  (`pytest -m "gpu and ep" tests/gpu`). Only the launcher entry points (`LAUNCHER_ENTRYPOINTS`) are
  collected under `tests/gpu/`, so no collection imports a GPU script, and no CPU test or `tests/common`
  helper reaches one (a fixture both tiers need lives in `tests/common/`). The launcher allocates a free
  `--master_port` per launch from `tests/common/ports.py` (a pool from 20000 to the kernel's ephemeral
  floor, one slice per xdist worker) and points `TMPDIR` at a per-run dir under pytest's basetemp; never
  hardcode either.
- **Scratch goes through the launcher's `TMPDIR`**: `setup_cache_dirs` for per-rank dirs,
  `shared_scratch_dir` (`tests/common/distributed.py`) for a checkpoint rank 0 writes and the peers read.
  Tagged dirs (`SCRATCH_DIR_TAG`) older than 6 h are removed at session start; checkpoint locations
  belong in `tests/common/models.py`, never a literal `/mnt/...`.
- **Use the `gpu_test_main` harness** (`tests/common/harness.py`) and build checks from the shared helpers;
  return `ctx.metrics(trainer)` so the result line carries tokens/s/GPU and peak memory
  ([`skills/write-tests/harness.md`](../../skills/write-tests/harness.md)). A script whose lifecycle the
  harness cannot express is listed in `_OWN_LIFECYCLE`
  (`tests/cpu/conventions/test_gpu_harness_conventions.py`) with its reason. Shared helpers in
  `tests/common/` keep one concern per module, with no grab-bag word in a module name
  (`tests/cpu/conventions/test_module_names.py`).
- **Cover the matrix and the rejections.** Markers select the tier (`core` or `full`), the world size
  (`1gpu 2gpu 4gpu 8gpu`, matching the row's `nproc`) and capabilities (`ep cp tp etp pp hsdp vlm lora
  moe` plus the model family); `vllm_server` / `sglang_server` need a live container and are always
  `full`. A new parallelism mode needs a correctness test per supported combination *and* a test that the
  unsupported ones are rejected.

### Test knobs

The GPU launcher (`tests/gpu/conftest.py`) reads its knobs with raw `os.environ`, since importing
`halo/env.py` would pull torch into the launcher, so each is matched as a literal string:

| Var | Meaning |
|---|---|
| `VLLM_SERVER_URL` / `SGLANG_SERVER_URL` | The live server the `vllm_server` / `sglang_server` tiers drive (default `http://localhost:8000` / `:30000`); set by the shipped infrastructure (`Makefile`, `docker-compose.vllm.yml`), hence no `HALO_TEST_` prefix |
| `HALO_TEST_LAUNCH_ID` | Set **by** the launcher on every launch so the orphan sweep finds surviving workers in `/proc`; do not export it |
| `HALO_TEST_NPROC` | Set by the launcher for a `launcher="python"` row: the rank count its script starts (`halo launch -n`) |
| `HALO_TEST_REQUIRE_SERVER` | The engine name (`vllm` / `sglang`, set by the `make` server tiers) whose tests must not skip: an unreachable endpoint raises a `UsageError` instead |

Test scripts read their own knobs through `halo/env.py`, prefixed `HALO_TEST_` so a stray export
cannot collide with them (the one exception is the server-side `VLLM_MODEL` below). Cross-suite knobs:

| Var | Meaning |
|---|---|
| `HALO_TEST_MODEL` | Checkpoint of the suites that sweep architectures: the expert-LoRA, router `modules_to_save`, mixed merged-save and expert-LoRA reference suites (default gpt-oss-20b) and the dense self-distillation text suite (default Qwen3-0.6B) |
| `HALO_TEST_EP` / `HALO_TEST_CP` / `HALO_TEST_TP` / `HALO_TEST_ETP` | Parallel size a sweep-capable suite builds its `ParallelismConfig` with (unset: the suite's default); the suffix spells the axis as the config does (`ep_size` / `cp_size` / `tp_size` / `expert_tp_size`) |
| `HALO_TEST_ATTN` / `HALO_TEST_GC` / `HALO_TEST_REVISION` | Attention implementation, gradient checkpointing (default **on**) and hub revision for the suites that sweep them |
| `HALO_TEST_REQUIRE_HUB_CACHE` | A CPU test whose Hub repo is missing from the local cache fails instead of skipping ([Hub seed](../infrastructure/ci.md#hub-seed)); gated repos and local paths still skip. Load through the `tests/common/tokenizers.py` helpers and name the repo in `tests/common/models.py` |
| `HALO_TEST_OFFGRPO_PARALLEL` | `tp` (default, dense Qwen3) or `ep` (gpt-oss MoE) leg of `trainers/grpo/test_offline_grpo_tp_resume.py` |
| `HALO_TEST_MAX_STEPS`, `HALO_TEST_BATCH_SIZE`, `HALO_TEST_GRAD_ACCUM`, `HALO_TEST_NUM_GENERATIONS`, `HALO_TEST_NUM_WORKERS`, `HALO_TEST_MAX_CONCURRENT`, `HALO_TEST_ROLLOUT_MAX_TOKENS`, `HALO_TEST_MAX_COMPLETION` | Step count and rollout sizing of `trainers/grpo/test_environmental_grpo_benchmarks.py`, sized for one vLLM server |
| `HALO_TEST_BENCHMARK_DATASETS_CACHE` | A writable datasets cache holding `openai/gsm8k` (`main`), `basicv8vc/SimpleQA`, `TIGER-Lab/MMLU-Pro` and `deepmind/code_contests`, which the same suite reads instead of the empty per-run cache, so it runs offline. The `make` tiers do not forward it: pass `EXTRA_DOCKER_ENV="-e HALO_TEST_BENCHMARK_DATASETS_CACHE=..."` with a path under the mounted `$(HALO_SCRATCH)` |
| `HALO_TEST_VLLM_GROUP_PORT` / `HALO_TEST_SGLANG_GROUP_PORT` | Pin the trainer-side group port the rollout e2e suites bind; unset, each run draws one with `free_port()` below the ephemeral range. The online-GRPO rows offset a pinned port by the row's index, a resume row rebinds for phase 2, and the re-init suite every cycle |
| `HALO_TEST_SGLANG_SERVER_URLS` / `HALO_TEST_SYNC_SERVER_CONTROL` | The SGLang servers of the `single_group` weight-sync rows (`trainers/grpo/test_single_group_sync.py`, `--backend sglang --servers N`; the vLLM rows read `HALO_TEST_VLLM_SERVER_URLS`), each serving Qwen3-0.6B; a row skips with fewer URLs than `--servers`. `trainers/grpo/test_single_group_failure.py` also needs an executable in `HALO_TEST_SYNC_SERVER_CONTROL`, called `<it> kill <index>`, `<it> restart <index>` or `<it> restart-all` and returning once the servers it touched serve again; it skips without one |
| `HALO_TEST_VLLM_SERVER_URLS` | Comma-separated vLLM endpoints of the one-process (`*_world1`) rows of `trainers/grpo/test_online_grpo_vllm_e2e.py`, each serving the same checkpoint on its own GPU (default: `HALO_TEST_VLLM_DENSE_SERVER_URL`). An `--trainer environmental` row pushes through a `WeightSyncPool` over the first `--servers` of them and asserts the policy moved on every one; `--servers 2` skips with fewer URLs. The two-rank `--servers` row of `trainers/grpo/test_env_grpo_e2e.py` reads the same list (default: `VLLM_SERVER_URL`), each URL serving that row's checkpoint |
| `VLLM_MODEL` | Checkpoint the running vLLM server serves, read by the rollout benchmark and **not** forwarded by the `make` tiers (put it in `.env` or `EXTRA_DOCKER_ENV`). It must be the model the test trains: unset, the benchmark falls back to Qwen3-0.6B |

Suites that pin one family or one phase add `HALO_TEST_<SUITE>_*` knobs, named by the manifest entry and
the script:

| Var | Meaning |
|---|---|
| `HALO_TEST_<SUITE>_MODEL` | Swap the checkpoint a family or phase suite loads; the per-suite spelling keeps a global export from pointing a family test at another family. Per family (`HALO_TEST_ZAYA_MODEL`, `HALO_TEST_GLM4_MODEL`, `HALO_TEST_GEMMA4_MODEL`, `HALO_TEST_QWEN3_5_MODEL`, …) and per phase: `HALO_TEST_EP_RT_MODEL` (EP round-trip, default a local Gemma4-26B-A4B checkpoint), `HALO_TEST_EP_CP_RT_MODEL` (EP+CP round-trip, local gpt-oss-20b), `HALO_TEST_EP1_KNOB_MODEL` (ep1 weight sync, default gpt-oss), `HALO_TEST_RESUME_MODEL` / `HALO_TEST_RESUME_EP_MODEL` (SFT resume: dense Qwen3-0.6B, and the MoE the `ep` mode needs), `HALO_TEST_STEP3P7_MODEL` (the Step-3.7 vLLM sync suite serves its own `--write-checkpoint` tree, so it has no default and skips without one) and `HALO_TEST_GLM5_NEXT_MODEL` (the same for GLM-5 Next's `test_weight_push_exact.py` rows), `HALO_TEST_QWEN4_EXP_MODEL` (the Qwen3.8-Flash-Next real-checkpoint suite, `tests/gpu/models/test_qwen4_exp_real_checkpoint.py`: a repo id it reads from the local Hub cache only, or a directory), `HALO_TEST_NEMOTRON_MODEL` and `HALO_TEST_NEMOTRON_LIGHTNING_MODEL` (the Nemotron-H release suite, `tests/gpu/models/test_nemotron_h_real_checkpoint.py`, and the MTP real and serving rows, one per `--release`, default that Hub release). A suite whose local default is absent skips |
| `HALO_TEST_MTP_REAL_MODEL` / `HALO_TEST_MTP_REAL_STALE_HEAD` | The trunk (default `Qwen/Qwen3.6-35B-A3B`) and the checkpoint holding the stale head (default `Qwen/Qwen3.5-35B-A3B`) of `tests/gpu/trainers/sft/test_sft_mtp_head_real.py` and the trunk of `test_sft_mtp_head_cost.py`. Set to anything else, a row records its numbers and holds no bar read off the defaults |
| `HALO_TEST_ZAYA_EP_STEPS` / `HALO_TEST_ZAYA_EP_SEQ` | Steps (default `4`, asserted) and `max_length` (default `512`) of the ZAYA1-8B EP suite (`tests/gpu/parallelism/ep/test_zaya_ep.py`) |
| `HALO_TEST_ZAYA_FSDP_STEPS` | Steps (default `4`) of the ZAYA1-8B FSDP2 suite (`tests/gpu/trainers/sft/test_zaya_fsdp.py`) |
| `HALO_TEST_EP_RT_ATTN` / `HALO_TEST_EP_RT_SCOPE` / `HALO_TEST_EP_RT_EXPERT_TP` / `HALO_TEST_EP_RT_KEEP` | EP save/reload round-trip (`tests/gpu/parallelism/ep/test_ep_save_reload_roundtrip.py`): attention (default `sdpa`), `ep_scope` (default `auto`; `node` / `global` pick the node-local or cross-node gathered save), `expert_tp_size` (default `1`; at `HALO_TEST_EP=1` this is pure ETP), and `_KEEP` to leave the checkpoint on disk |
| `HALO_TEST_EP_CP_RT_ATTN` / `HALO_TEST_EP_CP_RT_KEEP` | The same two for the EP+CP round-trip (`tests/gpu/parallelism/combined/test_ep_cp_save_reload_roundtrip.py`; attention defaults to `flash_attention_2`) |
| `HALO_TEST_EP1_KNOB_ATTN` / `HALO_TEST_EP1_KNOB_LAZY` | ep1 `fsdp_shard_ep1_experts` weight-sync suite (`tests/gpu/parallelism/ep/test_ep1_knob_weight_sync.py`): attention (default `flash_attention_2`) and `ep_lazy_loading` (default on; `=0` loads through `from_pretrained` + EP patching) |
| `HALO_TEST_EP_FORWARD_DELAY_MS` | Milliseconds every EP layer of `tests/gpu/profiling/benchmark_sft_ep.py` idles the GPU before each forward (default `0`): the bench band's negative control |
| `HALO_TEST_RESUME_FSDP_RESHARD` | `fsdp_reshard_after_forward` (default off) for the `fsdp`, `fsdp_base` and `cp` modes of `tests/gpu/trainers/sft/test_sft_checkpoint_resume.py` (TP+DP+FULL_SHARD is config-rejected) |
| `HALO_TEST_VLLM_DENSE_SERVER_URL` | Endpoint of the dense and one-process rows of `trainers/grpo/test_online_grpo_vllm_e2e.py` and of the vLLM re-init and pool-reconnect suites; the MoE rows train another checkpoint, so one `VLLM_SERVER_URL` cannot carry both (unset: `VLLM_SERVER_URL`, then `http://localhost:8010`) |
| `HALO_TEST_ONLINE_GRPO_MOE_MODEL` / `HALO_TEST_ONLINE_GRPO_DENSE_MODEL` | Checkpoints of that e2e: MoE rows `Qwen/Qwen3-30B-A3B-Instruct-2507` (EP/ETP), dense rows `Qwen/Qwen3-0.6B` (TP, FSDP2-DP, one-process); the server must serve the same one |
| `HALO_TEST_ENV_GRPO_MODEL` / `HALO_TEST_ENV_GRPO_MAX_STEPS` | Checkpoint (default `Qwen/Qwen3-30B-A3B-Instruct-2507`) and steps (default `2`; a `--prefetch-depth N` row runs `N + 2`: its `N` cold rounds, then two that pop a prefetched one) of the two-rank vLLM rows of the Environmental-GRPO e2e (`tests/common/env_grpo_e2e.py`) |
| `HALO_TEST_ENV_GRPO_SGLANG_MODEL` / `HALO_TEST_ENV_GRPO_4GPU_MODEL` | Checkpoints of the SGLang rows and of the four-rank vLLM rows (EP+ETP, EP+TP, ep4) of `trainers/grpo/test_env_grpo_e2e.py`, both default `unsloth/gpt-oss-20b-BF16` |
| `HALO_TEST_ENV_GRPO_ATTN_IMPL` / `HALO_TEST_ENV_GRPO_LORA_TARGETS` / `HALO_TEST_ENV_GRPO_TRUST_REMOTE_CODE` | Per-family overrides for that e2e: the policy's `attn_implementation` (unset: auto-selection; `sdpa` for a remote-code family without FA4) and a comma-separated `lora_target_modules` for the `--peft lora` rows (unset: the attention projections read off the index; needed where they do not end in `_proj`, such as Ling's `query_key_value,dense`) The last (default on) is off for Step-3.7, whose `auto_map` is for the server: its trainer takes the native class. |
| `HALO_TEST_PUSH_EXACT_FRESH` / `HALO_TEST_PUSH_EXACT_MODEL` | `trainers/grpo/test_weight_push_exact.py`: the first states the server has taken no other suite's push, without which the suite skips, since a server another push moved cannot be told from one this push moved; the second is a local path standing in for the row's checkpoint on both sides (the script header has the server launch) |
| `HALO_TEST_VLLM_REINIT_MODEL` / `HALO_TEST_VLLM_REINIT_CYCLES` / `HALO_TEST_VLLM_SERVER_GPU` | Checkpoint (default `Qwen/Qwen3-0.6B`), connect/sync/disconnect cycles (default `12`, enough for a stranded communicator to overshoot the 1 GiB growth budget) and the server's `nvidia-smi` index (unset: every GPU the trainer does not own) for `trainers/grpo/test_weight_transfer_reinit.py --backend vllm`; the pool-reconnect suite reads the same checkpoint knob |
| `HALO_TEST_SGLANG_REINIT_MODEL` / `HALO_TEST_SGLANG_REINIT_CYCLES` / `HALO_TEST_SGLANG_SERVER_GPU` | The same three for `--backend sglang`, against `SGLANG_SERVER_URL` |

The local checkpoints the per-suite overrides default to (Gemma4-26B-A4B for `HALO_TEST_GEMMA4_MODEL` and
`HALO_TEST_EP_RT_MODEL`, gpt-oss-20b for `HALO_TEST_EP_CP_RT_MODEL`) are `patch_vocab.py` outputs at
`$HALO_DATA_ROOT/models/<repo name>-patched` (`patched_checkpoint_dir` in `tests/common/models.py`),
under `$(HALO_SCRATCH)/models/` in the `make` tiers. The suites need a local directory, not added tokens,
so the tool runs without `--patterns`:

```bash
python scripts/before_training/patch_vocab.py --model_id google/gemma-4-26B-A4B-it \
    --output_dir "${HALO_DATA_ROOT:?}/models/gemma-4-26B-A4B-it-patched"
python scripts/before_training/patch_vocab.py --model_id unsloth/gpt-oss-20b-BF16 \
    --output_dir "${HALO_DATA_ROOT:?}/models/gpt-oss-20b-BF16-patched"
```

### Manifest rows and tiers

A spec launches its `args_matrix` entries under its `nproc`, `markers`, `timeout` and `flaky`; a `Row`
entry sets its own launch values, so one script runs at several widths, tiers or engines, each row
holding `gpu`, one tier and the `Ngpu` tag of its own `nproc`
(`tests/cpu/conventions/test_gpu_harness_conventions.py`). A leg the script can name becomes a CLI flag
with one row per leg, so each gets its own pytest node and verdict (`--trainer` / `--mode` on
`trainers/grpo/test_online_grpo_vllm_e2e.py`; `--backend vllm` / `sglang` on
`trainers/grpo/test_env_grpo_e2e.py`; one `--mode` per parallel shape on `trainers/sft/test_sft_families.py`
and `trainers/sft/test_sft_gptoss_modes.py`; `--ep-loading lazy` / `eager` on
`trainers/grpo/test_offline_grpo_reference.py` and `test_offline_grpo_ep_cp.py`, pinned by
`tests/cpu/conventions/test_offline_grpo_ep_loading_rows.py`). Rows that keep the spec's markers share
their union, so `-m "gpu and cp"` also selects the non-CP rows of such an entry. A leg that only swaps the
checkpoint stays an env override (`HALO_TEST_<SUITE>_MODEL`): a per-family pass is a run of that row with
the override set.

`core` is the pre-merge gate, and small-and-fast (≤2 GPUs, tiny model) is its intent; size the host from
the manifest, not from that intent. Most manifest entries carry a `core` row; within the tier some rows
need 4 GPUs, many declare 1500–2400 s timeouts, and a large minority load a real multi-billion-parameter
checkpoint (gpt-oss-20b, ZAYA1-8B, Qwen3-30B-A3B, Qwen3.5-2B, Qwen3-VL-2B and three Ling/Ring
checkpoints), so the summed worst-case timeouts pass 100 hours. The manifest is the only place
exact counts live. A big-checkpoint entry stays `core` only as a *correctness gate*, a comparison
against an independent reference (`parallelism/ep/test_ep_modes_vs_reference.py`: gpt-oss and Qwen3-MoE
parallel modes vs the dense reference); smokes on the same checkpoint (`trainers/sft/test_sft_gptoss_modes.py`,
`trainers/lora/test_lora_mixed_merged_save.py`) are `full`.

Run `make test-gpu-core` deliberately, not as a quick check, and mark a new heavy or many-GPU test
`full`. `gpu-tests.yml` budgets the tier 6 h; a new core entry that pushes past that belongs in `full`.

### A test must not be AI-slop

A test earns its place only if it **fails when the behavior breaks**: mutate the function under test
(flip a sign, drop a clause, return the input) and a real test goes red. These stay green:

| Anti-pattern | What to do instead |
|---|---|
| **Tautology** — `assert True`, `assert x == x`, or asserting a value the test just set | Assert the computed output against an independently known expected value |
| **Smoke-only** — call it, assert it "did not raise" | Assert the return value / shape / loss |
| **Mock the thing under test**, then assert the mock returned what you told it to | Mock only the boundary (network, GPU, clock); call the real function |
| **Vacuous assertion** — `assert out is not None`, `assert len(out) >= 0` | Assert the exact value, correct length, or a tight bound with a reason |
| **Test the framework** — `assert isinstance(cfg, ParallelismConfig)` | Assert the property your code computes (`cfg.data_parallel_size == 4`) |
| **Swallow-and-pass** — `try: real_call() except Exception: pass` | Let it raise, or `pytest.raises(ValueError, match=...)` |
| **Golden with no meaning** — a literal pasted from one run | Derive the expected value from the math; if it is a true golden, say why |

`tests/cpu/parallelism/test_parallelism_config.py` is the voice to copy. Cover the edge cases (empty
input, the unsupported combination, the degenerate near-zero case): that is where regressions hide.

### Golden performance baselines

`make bench` runs the rows of `tests/gpu/profiling/run_baseline_benchmarks.sh` on one 8-GPU node, each a
`torchrun` launch at seq 4096, batch 1, with gradient checkpointing and the production attention and
grouped-GEMM defaults:

| Rows | Script | Keys |
|---|---|---|
| Qwen3-8B under FSDP2, CP2, TP2 | `benchmark_sft_dense.py` | `sft_dense_qwen3-8b_{fsdp,cp2,tp2}_s4096` |
| gpt-oss-20b under EP1, EP2, EP8, EP8 × TP2, EP4 × ETP2 | `benchmark_sft_ep.py` | `sft_ep_gpt-oss-20b_{ep1,ep2,ep8,ep8_tp2,ep4_etp2}_s4096` |
| the EP8 row's gathered save (`--measure_save`) | `benchmark_sft_ep.py` | `sft_ep_gpt-oss-20b_ep8_gathered_save` |

Every rank prints one `__HALO_BENCH__` line per key (`tests/common/reporting.py`). The key names the
golden `tests/baselines/bench/<key>.json`: the commit, image, date, world size and command it was
measured with, and the metrics of at least five repeats. `make bench BAND=1` holds the run to them
(`tests/common/equivalence/bench_band.py`, evidence in `evidence/bench_band-<revision>.json`);
`make bench-band` re-checks the logs `make bench` left at HEAD without GPUs.

- **The band.** Each gated metric's band is the mean of its baseline repeats, give or take the larger of
  a margin and twice their spread. `tokens_per_second` (mean over ranks, 2%) and `save_seconds` (6%)
  must stay inside it on either side: a faster run fails too, so a speed-up re-records its baseline.
  `peak_allocated_gb` (0.5%) and `save_peak_rss_gib` (1%, the save's peak resident host memory), each the
  highest over ranks, must not rise above the upper edge, which is never below the highest repeat. MFU,
  S-MFU and TFLOPS stay under `metrics.diagnostics`, never gated.
- **The corpus.** Every row trains on `create_benchmark_dataset` (`tests/common/datasets.py`): seeded
  uniform token ids no model can fit, so an MoE row's routing, and the hottest EP rank's step time and
  peak, stay put (EP8, EP8 × TP2 and EP4 × ETP2 repeat within 1% in throughput and 0.02 GB in peak; on a
  corpus the model fits, EP8 moved 4% and 1.2 GB between runs).
- **A quiet host.** A row sharing the host with another job's CPU or network load runs 4–17% slow;
  recording refuses that repeat and a band run that meets one fails.
- **Flags.** Each training line carries its rank's mean and median step time (`mean_step_time_ms`,
  `median_step_time_ms`; the band reads rank 0's). A repeat whose mean is more than 2% off its median is
  flagged, never failed: re-run it before reading its throughput as a regression.
- **Same command.** A key compares only with a baseline measured by the same command on as many ranks of
  the same GPU model; a measured key with no baseline, and a baseline the run did not measure, fail.
- **Another machine.** A baseline can fail on another machine of its type with no code effect: gpt-oss-20b
  EP1 read 2.4% under its baseline on a second B300 machine, at the baseline's commit too. Split a miss with
  an A/B of that commit and HEAD on the machine that missed.

`make bench RECORD=1` runs the rows five times on a clean tree and writes the baselines only when no
repeat is outlying (a rate or duration outside the other repeats' band widened to four times their
spread, a peak only above its upper edge), removing any baseline the run did not measure; commit the
files it writes. The negative control,
`make bench BAND=1 EXTRA_DOCKER_ENV="-e HALO_TEST_EP_FORWARD_DELAY_MS=2"`, must fail every EP row's
`tokens_per_second` and pass the dense rows and the save. EP4 runs with ETP2 because pure `ep_size > 2`
with `ep_group_size < nvlink_domain_size` is rejected at config time
([Expert Parallelism](../parallelism/expert-parallelism.md#single-domain-multi-group-ep-races-and-hangs)).

### Goldens

Each golden is a JSON file under `tests/common/goldens/`, written by the generator module beside it and
checked against today's code by a CPU test. A refactor leaves it unchanged. Regenerate one only for a
deliberate change, in a commit that says so:

```bash
python -m tests.common.goldens.<name>          # exits non-zero when the committed file is stale
python -m tests.common.goldens.<name> --regen  # rewrites it from today's code
```

| Golden (`<name>`) | Pins | Checked by |
|---|---|---|
| `rank_layout` | For every parallel shape of the grid: the EP, dispatch, expert-TP, expert-replica, DP-scope and CP groups and the pipeline chains, each rank's coordinates, and the `EPConfig` flags that decide where expert gradients sync (not the TP / FSDP `DeviceMesh` dimensions) | `tests/cpu/parallelism/test_grid_goldens.py` |
| `ep_group_order` | The `new_group` calls `EPConfig` makes, in order with their keyword arguments, which call built each group a rank keeps, and each rank's `ep_group_idx` and `expert_replica_ranks` | `tests/cpu/parallelism/test_grid_goldens.py` |
| `load_collective_trace` | What `load_distributed_model` issues on ranks of a 4-rank job on two 2-GPU nodes, for every branch of its loader ladder from a local directory and the Hub, and what the frozen reference load issues (`LOAD_CASES`) | `tests/cpu/parallelism/test_load_collective_trace.py` |
| `balancing_collectives` | What configuring MoE balancing and building the run's callbacks issue on rank 0 of a two-rank job, for every `moe_balancing` mode, loss kind and weight-sync flag over tiny dense and MoE stand-ins: the collectives in order, the log records between them, the raise, the callback roster and the router config fields left | `tests/cpu/callbacks/test_balancing_collectives.py` |
| `weight_sync_wire` | The vLLM weight-transfer bootstrap: the rendezvous `TCPStore` arguments, its one write (the pickled `ncclUniqueId` under `broadcast_from/0/0`), the id the communicator initializes with, and the class's module, qualname and fields | `tests/cpu/weight_sync/test_wire_pins.py` |
| `weight_sync_transcript` | What each engine's client sends through its public lifecycle in twelve scenarios per engine (dense and MoE syncs, formation and finish failures, aborts before and mid-stream, an unusable client at exit): every request with its route, body, timeout and retry policy, what went on the NCCL wire, and the at-exit hooks left | `tests/cpu/weight_sync/test_request_transcript.py` |
| `registry_order` | What each toolkit registry holds, in its own order: the EP layer classes and their claims, the CP, PP and head claims, the weight-sync clients, the Liger and MoE family specs, the sequence-classification heads, the named optimizers and the environment types | `tests/cpu/conventions/test_registry_order.py` |

The grid (`tests/common/goldens/shape_grid.py`) is every shape `ParallelismConfig` accepts with a pipeline
stage of at most 32 ranks, at every NVLink domain dividing the stage, plus the same sweep at 64, 72, 128,
144 and 256 ranks on 8-GPU nodes and 72-GPU NVL72 racks; each grid golden (a `GridGolden`) holds a digest
per shape plus the full record of a readable subset, so a failure reads as a diff. Notes per golden:

- **`ep_group_order`.** Recorded on a gloo default group, where every group is a `new_group` call (an NCCL group
  bound to its device splits each call's groups at once: `tests/cpu/parallelism/test_group_split.py`).
  `new_group` is a collective over the default group, so a rank making other calls
  than rank 0 hangs a real job (the generator raises there), and two calls over the same members build
  two communicators, so a reordered call fails here even where `rank_layout`, which compares memberships,
  passes. `tests/cpu/parallelism/test_parallelism_config_gate_order.py` separately pins the order of
  `ParallelismConfig`'s gates.
- **`load_collective_trace`.** `tests/common/collective_recorder.py` runs a load as one simulated rank
  over torch's fake process group and an in-memory store, replacing each `torch.distributed` function
  under every name a module holds it by (a collective DTensor issues is recorded too); a collective it
  has no model of raises.
- **`weight_sync_wire`.** The vLLM image unpickles the id as its own class through a shim at the same
  module path, so the test also checks that `Dockerfile.vllm` writes the shim for the module the golden
  names. Moving `ncclUniqueId` changes the pickle bytes: regenerate the golden in the commit that moves
  the class, and move the shim with it.
- **`weight_sync_transcript`.** The clients run against `FakeVLLMServer` and `FakeSGLangServer`
  (`tests/common/fake_engines.py`) with NCCL and CUDA faked on the CPU.
- **`registry_order`.** The check renders in a fresh interpreter, so the order is the one the code and
  the generator's imports produce, and classes are recorded by name, so a move that keeps each name and
  its order leaves the file unchanged.

### Move kit

A structural change (a module moved or renamed, code carved between modules, a package renamed) proves it
changed no code with `make verify-move BASE=<sha>` (`tests/common/equivalence/move_kit.py`). The kit
reads both commits from git (`HEAD_REF`, default `HEAD`, against `BASE`), parses the wheel's packages and
`scripts/` without importing them, and writes `evidence/move_kit-<head sha>.json` (git-ignored) with one
verdict per check:

| Check | Fails when |
|---|---|
| `renames` | a file rename the change declares is missing from `git diff -M50%`, or below `--min-similarity` (90 by default) |
| `definitions` | a module docstring, function, class header, method or module-level assignment hashes differently once each first-party name it reads is spelled in head terms, or head adds one the change does not declare |
| `public_names` | a public top-level name appears or disappears |
| `string_literals`, `raise_templates` | the multiset of string literals (docstrings aside) or of raised expressions changes |
| `side_effects` | a module's ordered import-time statements change: bare calls, assignments that call first-party code or set an attribute or item, first-party decorators, control flow, and imports marked `noqa: F401` or `E402` |
| `goldens` | a golden file the change does not declare changes, a declared one does not, or a golden's generator fails its check against head's code |

Spelling a name in head terms follows every rename git finds, each declared symbol rename, and each
definition that moved to another module under its own name; a name a function binds hides the
module-level one. A path in a string follows its file's rename, or its directory's when every Python file
of that directory moved to one new directory. A library decorator (`@dataclass`, `@lru_cache`) is the
definition's business, so a carve may move it. Each patch in head on a name a touched module only
imports is listed under `patch_targets` for review, and a renamed method's callers through an instance
read as changed.

Declare what the definitions do not show in a JSON file under the repo, passed by its repo-relative path
as `RENAMES=`:

```json
{
  "files": {"halo/<old path>.py": "halo/<new path>.py"},
  "symbols": {"halo.<module>:<old qualname>": "halo.<module>:<new qualname>"},
  "added": ["halo.<new module>:__doc__", "halo.<new module>:logger"],
  "goldens": ["weight_sync_wire.json"]
}
```

`added` names the definitions a carve adds, `goldens` the golden files the change regenerates on purpose;
a declaration the trees contradict fails. `MOVE_ARGS` passes further flags: `--require` names the checks a
carve's proof rests on (one that extracts new functions fails `definitions` by design),
`--min-similarity` lowers the bar for a rename that carries edits, and `--ignore-docstrings` compares a
prose-only change:

```bash
make verify-move BASE=<sha> RENAMES=<map.json> \
  MOVE_ARGS="--require renames string_literals raise_templates side_effects goldens --min-similarity 70"
```

`make verify-move` exits non-zero when a required check fails and runs as the caller, so the evidence is
the caller's. `tests/cpu/conventions/test_move_kit.py` seeds each kind of change on a scratch repository.

### Comparison harnesses

The checkpoint, EP, CP and efficiency harnesses share one contract: each runs its own module in `BASE`
and in `HEAD_REF` with that tree's code (so `BASE` must already carry it) and writes
`evidence/<harness>-<head sha>.json` (git-ignored) with every delta keyed by its path in the output.
`ACCEPT="<case> ..."` names the cases whose deltas a flagged change declares, each a case path or a
prefix of one (`ACCEPT=cpu/qwen3_moe` covers every EP record of that family); their deltas stay in the
evidence under `accepted` without failing the verdict, and `FAMILIES=` limits a run to some tiny
families. The checkpoint and EP harnesses run only the families both trees' `FAMILIES` rosters hold, so
a change that adds, drops or renames a family still compares the rest against its base; each family left
out is logged by the tree whose roster lacks it and listed under `skipped`
(`tests/cpu/conventions/test_harness_family_rosters.py`). Every verdict also fails on an unexpected raise.
The EP, CP and efficiency harnesses run a negative control: the head tree reruns one case with a
deliberate perturbation, and the verdict fails when that record matches its base or raised.

- **Checkpoint artifacts** (`make verify-artifacts`, `tests/common/equivalence/checkpoint_artifacts.py`)
  saves a tiny model of every family (the dense roster, every EP family and its multimodal wrappers) at
  fp32 and at bf16 through every saver mode one CPU process runs, as an export and as a training
  checkpoint: the FSDP2 gathered stream, CP over plain parameters and over FSDP2, the TP saver, EP with
  FSDP2-managed experts (`ep_size == 1`) and with FSDP-ignored EP layers, native expert LoRA as an
  adapter and merged, and attention LoRA through the PEFT saver; a full-weight training checkpoint adds
  the optimizer shard and its meta after one AdamW step. It compares file by file: the file set and
  permission bits, each safetensors header and tensor digest, each JSON by value, each torch file by key
  and tensor. Both producers run under one string hash seed, since a set's order reaches
  `adapter_config.json`. The per-rank EP save, the PP stage save and every multi-rank gather need more
  ranks, which the GPU checkpoint suites cover (`tests/cpu/checkpoint/test_checkpoint_artifacts.py` holds
  the seeded changes it catches).
- **EP bitwise** (`make verify-ep`, `tests/common/equivalence/ep_bitwise.py`) compares every EP family's
  records bit for bit: per forward case the loss, logits, every parameter gradient, the routing each EP
  layer dispatched and the expert-load counts; per case the ordered parameters, state-dict keys,
  weight-decay groups and each parameter's gradient hooks, recorded by what they do to a gradient of
  ones (each all-reduce with its op, dtype and group ranks, and the scale left). `TIER=cpu` (the default)
  runs the per-expert loop on one process (defaults, an fp32 router on FSDP-ignored experts, native
  expert LoRA with dropout, an armed routing-replay mask, bias-update balancing) and builds every rank of
  a few layouts on stubbed groups; `TIER=gpu` runs the grouped-GEMM path in bf16 under `full_determinism`
  (DeepEP's deterministic dispatch included) on one rank, `ep2` (2 GPUs) and `ep2+etp2` (4 GPUs), each
  with `HALO_EP_SHARED_OVERLAP` off and on. Records outside `KNOWN_RAISING` must not raise; the negative
  control rotates every expert bank by one expert (`tests/cpu/parallelism/test_ep_bitwise_harness.py`). Both
  trees read one Triton cache, so it does not compare autotune picks across caches, and another run filling that
  cache beside it can read as deltas on the families that call fla's kernels
  ([Docker](../infrastructure/docker.md#baked-in-environment-variables)).
- **CP equivalence** (`make verify-cp`, `tests/common/equivalence/cp_equivalence.py`) runs a two-rank CP
  group in each tree on the CP roster, tiny models covering every Ulysses wrapper class (the Qwen3 wrapper on
  its dense, MoE and VL classes), at two levels: the first attention layer on fixed hidden states, and the
  model under `UlyssesModel` through its loss. A family the base tree's roster lacks has nothing to compare
  with: it is skipped and listed under `skipped`. The base tree runs twice
  and must repeat bit for bit; a record whose base does not repeat is compared forward bit for bit with
  each gradient inside its base-vs-base envelope (listed under `enveloped`). `TIER=gpu` runs bf16 under
  `full_determinism` with the flash kernel CP picks per family (FA4 on Blackwell where the family allows
  it, FA2 otherwise, FA3 on Hopper); the CPU tier runs fp32 on gloo with a plain torch attention under the
  flash contract. The negative control moves one attention weight by a few ulps
  (`tests/cpu/parallelism/test_cp_equivalence_harness.py`).
- **Efficiency** (`make verify-efficiency`, `tests/common/equivalence/efficiency_callback.py`) drives each
  tree's `EfficiencyCallback` through the same scripted steps and compares every logs dict it fills,
  every record logged under the package, and the `snapshot_efficiency` row. CUDA answers as a scripted
  B300 and the clock, token counter and document source follow a fixed schedule, so every figure is a
  function of the code. The cases cover every family on one rank and under EP with FSDP2, three families
  on every parallel shape (pipeline stages included) built with the real wraps on torch's fake process
  group, and the edge scenarios (an unknown GPU, fp8 / fp32 / tf32, a frozen LoRA base, an estimate that
  raises, no layer list, no parameters); a case only one tree has is listed under `new` or `dropped`. The
  negative control moves one step's clock by a tick (`tests/cpu/callbacks/test_efficiency_harness.py`).

## Docs

Update the owning doc page in the same PR when you change `halo/` (`skills/docs/docs-ownership.md` is the
map), following the style charter of the `docs` skill (`skills/docs/style.md`). `make docs` runs the
link, anchor and code-reference check that blocks a merge; editing a `scripts/diagrams/gen_*.py`
generator owes `make diagrams` and the regenerated figure in the same commit
([Continuous Integration → Hosted tier](../infrastructure/ci.md#hosted-tier)).

## Proof of Value

| Change type | What to show |
|-------------|--------------|
| No-op / refactor | Loss is **bitwise-identical** to `main` at a fixed seed, and the original checkpoint still loads |
| Behavior change | An end-to-end test demonstrating the new behavior or convergence |
| Performance | Before/after **tokens/s/GPU** and **peak memory** (utilization is a diagnostic, not the headline) |

State the GPU, model and sequence/batch shape with any number. Report a measured net-slower result
too, with the reason.

## Submitting a PR

1. **Get approved first and work from a signed fork branch**: the gate, the fork workflow, the
   signed-commit setup and the secrets rule are in [`CONTRIBUTING.md`](../../CONTRIBUTING.md).
2. **Pass the gates.** `make lint`, `make format`, `make seed-hf-cache` (again when
   `tests/common/models.py` or `examples/` gain a repo), then `make test-cpu` (plus `make test-gpu-core`
   for GPU-affecting changes) and `make docs`. The test tiers run locally, so report their result in the
   PR.
3. **Fill the PR template**: what and why, type of change, Proof-of-Value evidence, checklist.

## Where to look next

- [Development Environment](development-environment.md) — host `.venv`, dev container, env vars
- [Trainer Architecture](../reference/trainer-architecture.md) — how trainers compose with the mixin
- [Troubleshooting](../reference/troubleshooting.md) · [Debugging](../reference/debugging.md)
- [Adding a New Model](../models/adding-a-model.md)
