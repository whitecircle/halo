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
  `PATH`; the docs target is a pure link check). Everything else — tests, benchmarks — runs inside
  the image.
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
make build-blackwell          # B200 (SM100) / B300 (SM103); or build-hopper for H100/H200 (SM90)
make install                  # uv install inside the image
make test-cpu                 # sanity check
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
| Throughput benchmarks | `make bench` | image |
| Docs link check | `make docs` | host (`scripts/docs/check_links.sh`) |

`make test-gpu-vllm` is the `vllm_server` slice of the full tier and needs the `docker-compose.vllm.yml`
server already serving on a GPU outside `TRAINER_CUDA_DEVICES` (default `0,1,2,3,4,5,6`, which the target
pins as the trainer's `CUDA_VISIBLE_DEVICES`) — weight sync is an NCCL broadcast, and a rank cannot
broadcast to itself. It forces `NCCL_IB_DISABLE=1 NCCL_NET=Socket` (why:
[Rollout Servers → vLLM](../infrastructure/rollout-servers.md#vllm)).

Each server slice takes **two passes**, because the tests broadcast the trainer's own weights into the
served model and assert the served policy moved — server and trainer must hold the same checkpoint, and
no one server covers both halves. `SERVER_TIER` defaults to `not moe` (serve a dense model); restart the
server on a MoE checkpoint and rerun with `SERVER_TIER=moe`.

`make test-gpu-sglang` is the `sglang_server` slice and has the same server requirement, but forces
three more variables — `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET_PLUGIN=none` (why:
[Rollout Servers → NCCL transport](../infrastructure/rollout-servers.md#nccl-transport-sglang-only)).

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
- **No inline imports in `src/`** (`PLC0415`). A function-level import is reserved for a genuinely
  optional or arch-specific dependency (`flash_attn`, `deep_ep`, `kernels`). A circular import is
  never that exception — fix it structurally.
- **No stray `print()` in `src/`** (`T20`). `tests/**` relaxes `PLC0415`, `E402`, `T20`, `E741`,
  `B007`, `B023`, and `SIM117`; `scripts/**` relaxes `T20`.
- **Reuse over re-implementation.** New trainers extend `DistributedTrainerMixin`; new parallelism
  wrappers reuse the EP-layer base hooks. See
  [Trainer Architecture](../reference/trainer-architecture.md).

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
  `if __name__ == "__main__": raise SystemExit(pytest.main([__file__, "-v"]))` — the conventions test
  requires a `pytest.main([__file__, …])` entry in every CPU test file — so a standalone
  `python tests/cpu/<file>.py` runs exactly what `pytest` collects. A hand-listed runner silently
  drops any test missing from its list, and a printed pass/fail summary hides a FAIL from pytest;
  both are rejected by `tests/cpu/conventions/test_test_conventions.py`.
- **The `cpu` marker is applied by path, once.** `tests/conftest.py` marks every item under
  `tests/cpu/` in `pytest_collection_modifyitems`, so `-m cpu` selects the tier without a per-file
  `pytestmark`; adding one duplicates a marker the collector already applied.
- **Imports resolve from the image, not from `sys.path` surgery.** `PYTHONPATH=/workspace` is baked
  into the training images (and `tests/conftest.py` puts the repo root on the path for a pytest
  run), so `from src...` / `from tests.common...` work in both `pytest` and a standalone
  `torchrun tests/gpu/<script>.py`. A per-file `sys.path.insert` is dead weight that only masks
  running outside the image. A `scripts/` entry point — not an importable package — is loaded with
  `tests.common.utils.load_script_module`.
- **Deterministic, seeded data.** GPU tests generate synthetic problems from a fixed seed; only
  rank 0 prints. A seed alone does not fix a chat-templated batch: `strftime_now` is a jinja global
  transformers injects into every template, and gpt-oss's harmony prompt stamps the live date into
  its system message. The shared `tests.common.ep_reference.fixed_chat_batch` shadows that global
  with `CHAT_TEMPLATE_NOW`, so the correctness thresholds measured against its batch (EP/TP loss
  bounds, the rotated-expert control floor) do not drift with the calendar — build a fixed batch
  through it instead of re-rolling one per file.
- **GPU tests register in the manifest.** Drop the script under `tests/gpu/`, add one `TestSpec`
  line to `tests/gpu/manifest.py` (`nproc`, `markers`, `args_matrix`, `timeout`, `flaky`). A script
  missing from the manifest fails collection, so coverage cannot silently rot; a `bench*.py` that is
  in neither the manifest nor `_UNMANIFESTED_BENCHMARKS` fails the same way. World-size strictness is
  not a manifest field — the script declares it itself via `gpu_test_main(exact_world_size=N)`.
- **Run GPU tests through pytest**, not by hand: `make test-gpu-core`, or a narrower marker
  expression over the two entrypoints — `pytest -m "gpu and ep" tests/gpu/test_suite.py
  tests/gpu/test_launcher_contract.py`. Name the entrypoints: pointed at `tests/gpu/` instead, pytest
  collects the manifest scripts as modules and executes their top-level torchrun code. The launcher
  allocates a free `--master_port` per node and points `TMPDIR` at a
  per-run dir under pytest's basetemp — never hardcode either. A script run standalone under
  `torchrun --nproc_per_node=N <script>` lets torchrun pick the port.
- **Scratch goes through the launcher's `TMPDIR`.** `setup_cache_dirs` for per-rank output/cache
  dirs, `shared_scratch_dir` (`tests/common/distributed.py`) for a synthetic checkpoint rank 0
  writes and the peers read. A literal `/mnt/...` in a test escapes basetemp, is never reclaimed,
  and assumes a volume layout this host may not have; checkpoint locations belong in
  `tests/common/models.py`.
- **Use the `gpu_test_main` harness** (`tests/common/harness.py`). It owns the lifecycle
  (`init_distributed` → setup → body → teardown → `sys.exit`); the body is *load → train → assert*
  and returns `{"checks": {name: bool}, "metrics": {...}}`, exiting `0` pass / `1` fail / `2` bad
  launch.
- **Emit perf and memory.** Return `ctx.metrics(trainer)` so the result line carries tokens/s/GPU
  and peak memory; a `benchmark_*` script's `emit_benchmark(key, callback)` writes the line the
  committed throughput baselines are compared against.
- **Cover the matrix and the rejections.** Markers select the tier (`core` or `full`), the world size
  every entry declares (`1gpu 2gpu 4gpu 8gpu`) and capabilities (`ep cp tp etp hsdp vlm lora moe` +
  model family). `vllm_server`/`sglang_server` mark a test
  needing the live vLLM/SGLang container; those are always `full`. A new parallelism mode needs a
  correctness test per supported combination *and* a test asserting the unsupported ones are
  rejected.

The GPU launcher (`tests/gpu/conftest.py`) reads its knobs with raw `os.environ` — importing
`src/env.py` would pull torch and transformers into the launcher process — so each one is matched as
a literal string, not through the toolkit's `1/true/yes/on` flag parsing. The ones shipped
infrastructure sets:

| Var | Default | Meaning |
|---|---|---|
| `VLLM_SERVER_URL` / `SGLANG_SERVER_URL` | `http://localhost:8000` / `:30000` | Live rollout server the `vllm_server` / `sglang_server` tiers probe and drive. Both are set by the shipped infrastructure itself (`Makefile`, `docker-compose.vllm.yml`), which is why they carry no `HALO_TEST_` prefix — every other test knob does. |
| `HALO_TEST_LAUNCH_ID` | per launch | Set BY the launcher, not for it: a unique id stamped into every torchrun launch's environment so the orphan sweep can identify surviving workers from `/proc` without matching on a script name a co-tenant might also be running. Do not export it. |
| `HALO_TEST_REQUIRE_SERVER` | unset | The **engine name** (`vllm` / `sglang`, set by the `make` server tiers) whose tests must not be skipped: a node carrying the `<value>_server` marker raises a `UsageError` instead of skipping when the endpoint is unreachable, so a dead container cannot pass as a skip. Any other engine's tests still skip. |
| `HALO_TEST_MODEL` and the per-suite `HALO_TEST_<SUITE>_MODEL` overrides | per suite | Swap the checkpoint a suite loads without editing it. One spelling for every per-suite checkpoint override: `HALO_TEST_<SUITE>_MODEL`, so a global override cannot point a family test at a checkpoint of another family. Per family (`HALO_TEST_ZAYA_MODEL`, `HALO_TEST_GLM4_MODEL`, `HALO_TEST_GEMMA4_MODEL`, `HALO_TEST_QWEN3_5_MODEL`, …) and per phase: `HALO_TEST_EP_RT_MODEL` (EP round-trip, default a local vocab-patched Gemma4-26B-A4B — `scripts/before_training/patch_vocab.py` output, path in `tests/common/models.py`), `HALO_TEST_EP_CP_RT_MODEL` (EP+CP round-trip, the same for gpt-oss-20b), `HALO_TEST_EP1_KNOB_MODEL` (ep1 weight-sync, default gpt-oss), `HALO_TEST_RESUME_MODEL` / `HALO_TEST_RESUME_EP_MODEL` (SFT resume: dense default Qwen3-0.6B, and the MoE the `ep` mode needs), `HALO_TEST_LORA_CP_MODEL` / `HALO_TEST_LORA_SAVE_LOAD_MODEL` (both default Qwen3-0.6B; point them at a 4B for a scale check), `HALO_TEST_STEP3P7_MODEL` (**required** — the Step-3.7 vLLM sync suite serves its own `--write-checkpoint` tree and has no default). A suite whose local default checkpoint is absent skips rather than failing. |

Test scripts read their own knobs through `src/env.py`, `HALO_TEST_`-prefixed so a stray `export` or
a co-tenant compose file cannot collide with them — the one exception is the server-side `VLLM_MODEL`
below. The cross-suite ones:

| Var | Meaning |
|---|---|
| `HALO_TEST_EP` / `HALO_TEST_CP` / `HALO_TEST_TP` / `HALO_TEST_ETP` | Parallel size a sweep-capable suite builds its `ParallelismConfig` with; unset = the suite's own default (often `world_size` for EP). The suffix is the parallelism axis as the rest of the toolkit spells it (`ep_size` / `cp_size` / `tp_size` / `expert_tp_size`), so the knob and the config field it feeds read the same. |
| `HALO_TEST_ATTN` / `HALO_TEST_GC` / `HALO_TEST_REVISION` | Attention implementation, gradient checkpointing (default **on**), hub revision for the suites that sweep them. The per-family `HALO_TEST_ZAYA_GC` defaults the other way — see the per-suite table. |
| `HALO_TEST_OFFGRPO_PARALLEL` | `tp` (default, dense Qwen3) or `ep` (gpt-oss MoE) leg of `trainers/grpo/test_offline_grpo_tp_resume.py`. |
| `HALO_TEST_MAX_STEPS`, `HALO_TEST_BATCH_SIZE`, `HALO_TEST_GRAD_ACCUM`, `HALO_TEST_NUM_GENERATIONS`, `HALO_TEST_NUM_WORKERS`, `HALO_TEST_MAX_CONCURRENT`, `HALO_TEST_ROLLOUT_MAX_TOKENS`, `HALO_TEST_MAX_COMPLETION` | Step count and rollout sizing for `trainers/grpo/test_environmental_grpo_benchmarks.py`, whose defaults are sized for one vLLM server. |
| `HALO_TEST_VLLM_GROUP_PORT` / `HALO_TEST_SGLANG_GROUP_PORT` | Weight-transfer NCCL group port the e2e rollout suites open (default `51216`; `51220` for the Step-3.7 sync suite, `51340` / `51380` for the online-GRPO MoE / dense e2e pair — each row owns `base + 2×row-index`, its resume phase 2 the next port up, so back-to-back rows never contend through TIME_WAIT; `51240` for the 4-GPU env file; `51228` for the weight-transfer re-init suite, which rebinds it once per cycle); the server must be started on the same one. |
| `HALO_TEST_VLLM_SERVER_URLS` | Comma-separated rollout endpoints the environmental legs of `trainers/grpo/test_online_grpo_vllm_e2e.py` drive (default: the single `VLLM_SERVER_URL`). Two or more put the weight sync on the rolling multi-server path — one server updated at a time while the rest keep serving — and the leg then asserts the served policy moved on **every** one of them. Each server serves the same model on its own GPU; the leg binds one trainer-side group port per server, counting up from its own base (`51219` for the environmental leg, `51230` for the LoRA one). |
| `VLLM_MODEL` | Checkpoint the running vLLM server serves, read by the rollout benchmark itself — not by the launcher, and **not** forwarded by the `make` tiers, so put it in `.env` or export it. It must be the model the test trains: unset, the benchmark falls back to Qwen3-0.6B and a MoE-serving tier silently tests the wrong pairing. (`SGLANG_MODEL` is the compose file's server-side spelling; no test reads it.) |

Suites that pin one family or one phase add their own `HALO_TEST_<SUITE>_*` knobs on top; the manifest
entry and the script name them, and the non-obvious ones are:

| Var | Meaning |
|---|---|
| `HALO_TEST_ZAYA_EP_STEPS` / `HALO_TEST_ZAYA_EP_SEQ` | Training steps (default `4`) and `max_length` (default `512`) of the ZAYA1-8B EP smoke suite (`tests/gpu/parallelism/ep/test_zaya_ep.py`); the step count is asserted, so a longer sweep stays self-checking. |
| `HALO_TEST_ZAYA_FSDP_STEPS` | Training steps (default `4`) of the ZAYA1-8B FSDP2 suite (`tests/gpu/trainers/sft/test_zaya_fsdp.py`); its sequence length is fixed. |
| `HALO_TEST_ZAYA_GC` / `HALO_TEST_ZAYA_GC_REENTRANT` | Gradient checkpointing (default **off**) and its `use_reentrant` kwarg (default on) for both Zaya suites. Zaya's load patch refuses GC, so `=1` asserts that refusal rather than training with it. |
| `HALO_TEST_EP_RT_ATTN` / `HALO_TEST_EP_RT_SCOPE` / `HALO_TEST_EP_RT_EXPERT_TP` / `HALO_TEST_EP_RT_KEEP` | EP save/reload round-trip (`tests/gpu/parallelism/ep/test_ep_save_reload_roundtrip.py`): attention impl (default `sdpa`), `ep_scope` (default `auto`; `node` / `global` pick the node-local vs cross-node gathered-save path), `expert_tp_size` (default `1`; at `HALO_TEST_EP=1` this is pure ETP), and `_KEEP` to leave the gathered checkpoint on disk instead of deleting it. |
| `HALO_TEST_EP_CP_RT_ATTN` / `HALO_TEST_EP_CP_RT_KEEP` | The same two knobs for the EP+CP round-trip (`tests/gpu/parallelism/combined/test_ep_cp_save_reload_roundtrip.py`); attention defaults to `flash_attention_2` there. |
| `HALO_TEST_EP1_KNOB_ATTN` / `HALO_TEST_EP1_KNOB_LAZY` | ep1 `fsdp_shard_ep1_experts` weight-sync suite (`tests/gpu/parallelism/ep/test_ep1_knob_weight_sync.py`): attention impl (default `flash_attention_2`) and `ep_lazy_loading` (default on; `=0` routes the load through `from_pretrained` + EP patching instead). |
| `HALO_TEST_RESUME_FSDP_RESHARD` | `fsdp_reshard_after_forward` (default off = ZeRO-2) for the `fsdp` and `cp` modes of `tests/gpu/trainers/sft/test_sft_checkpoint_resume.py`; the `tp` / `ep` modes ignore it (TP+DP+FULL_SHARD is config-rejected). |
| `HALO_TEST_VLLM_DENSE_SERVER_URL` | Endpoint of the dense half of the online-GRPO/SDPG e2e pair (`trainers/grpo/test_online_grpo_vllm_dense_e2e.py`, default `http://localhost:8010`) and of the weight-transfer re-init suite, which serves the same checkpoint. The pair's two files train different checkpoints and each asserts on its own server's logprobs, so one `VLLM_SERVER_URL` cannot carry both; unset, both fall back to `VLLM_SERVER_URL`. |
| `HALO_TEST_ONLINE_GRPO_MOE_MODEL` / `HALO_TEST_ONLINE_GRPO_DENSE_MODEL` | Checkpoints of the online-GRPO/SDPG e2e pair (`tests/common/online_grpo_e2e.py`): the MoE half's default `Qwen/Qwen3-30B-A3B-Instruct-2507` (EP/ETP rows) and the dense half's `Qwen/Qwen3-0.6B` (TP and FSDP2-DP rows). Every row asserts on the served logprobs, so the running server must serve the same checkpoint. |
| `HALO_TEST_ENV_GRPO_MODEL` / `HALO_TEST_ENV_GRPO_MAX_STEPS` | Checkpoint (default `Qwen/Qwen3-30B-A3B-Instruct-2507`) and step count (default `2`) of the Environmental-GRPO e2e body (`tests/common/env_grpo_e2e.py`). The model reaches the **vLLM** leg only — the SGLang and 4-GPU legs pass their own — and the served model must be the same checkpoint. |
| `HALO_TEST_ENV_GRPO_4GPU_MODEL` | Checkpoint of the 4-rank Environmental-GRPO e2e (`trainers/grpo/test_env_grpo_vllm_4gpu_e2e.py`, default `unsloth/gpt-oss-20b-BF16`), whose rows hold two parallelism axes at once (EP+ETP, EP+TP, ep4). Its own knob because it shares the 2-GPU file's server but not its default family. |
| `HALO_TEST_VLLM_REINIT_MODEL` / `HALO_TEST_VLLM_REINIT_CYCLES` | Checkpoint (default `Qwen/Qwen3-0.6B`, the dense endpoint's) and connect/sync/disconnect cycle count (default `12`) of `trainers/grpo/test_vllm_weight_transfer_reinit.py`. Each cycle leaked ~633 MiB of NCCL communicator on both ends before the re-init patch, so twelve overshoot the suite's 1 GiB growth budget several times over. |
| `HALO_TEST_VLLM_SERVER_GPU` | The vLLM server's GPU as `nvidia-smi` indexes it, for the same suite's device-memory read. Unset means "every GPU the trainer does not own", which is exactly the server's on the tier's own topology (`TRAINER_CUDA_DEVICES` covers the rest); set it when another job holds a third GPU. |

**Env knob or `args_matrix` row?** `nproc`, `markers`, `timeout` and the tier are per-`TestSpec`, not
per-row, so every row of a matrix runs at the same size, under the same marker set, in the same tier.

A leg that only changes *which phase runs* — same model, same axis, same cost — becomes a CLI flag
with one row per leg, so each gets its own pytest node and verdict (`--mode` on
`trainers/grpo/test_online_grpo_vllm_e2e.py`, `--mode` on `trainers/sft/test_sft_qwen3_dense.py`). A
leg that changes the model family, the parallelism axis, or the runtime stays an env override:
registering it as a row would file an EP-on-20B run under the entry's `tp`/dense markers and its
neighbour's timeout. That is also why the twelve `gpt-oss` SFT scripts under `trainers/sft/`
(`test_sft_ep*`, `test_sft_oss20b_*`) stay separate entries rather than collapsing into one matrix —
`-m "gpu and cp"` must select their two CP legs and nothing else.

`core` is the pre-merge gate, and small-and-fast is its *intent* — ≤2 GPUs, tiny model. Size the host
from the manifest, not from that intent: over half of `tests/gpu/manifest.py` carries `core`
(120 of 195 entries, more pytest nodes once the `args_matrix` rows expand), and within that tier some
entries need 4 GPUs, 18 declare a timeout ≥1500 s (three at 2400 s), and a large minority load a real
multi-billion-parameter checkpoint (gpt-oss-20b, GLM-4.7-Flash, ZAYA1-8B, Qwen3-30B-A3B, Qwen3.5-2B,
Qwen3-VL-2B and three Ling/Ring checkpoints), so summed worst-case timeouts run to tens of hours.
This page owns tier composition; the manifest is the only place exact counts live.

Where a big-checkpoint entry stays `core`, it is because it is a *correctness gate* — a comparison
against an independent reference that catches a silently wrong result, like
`parallelism/ep/test_ep_correctness.py` (gpt-oss ep2 vs the dense reference). Smoke runs on the same
checkpoint are `full`: `trainers/sft/test_sft_ep.py`, `trainers/sft/test_sft_oss20b_*.py` and
`trainers/lora/test_lora_mixed_merged_save.py` all assert only that training completed and stayed
finite, which the gate already covers.

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

`tests/baselines/<key>.json` is a golden throughput/memory snapshot for one benchmark config. Its
`metrics` block comes from a `benchmark_*` run's `__HALO_BENCH__` line
(`tests/common/reporting.py::emit_benchmark`), keyed by the string that script passes to
`emit_benchmark`; the `provenance` block is hand-maintained.

Comparison is by hand — no CI job reads these files. Diff the headline `metrics.tokens_per_second`
and `metrics.peak_allocated_gb` against the committed payload, and treat `metrics.diagnostics`
(MFU / S-MFU / TFLOPS) as setup-dependent context, not a target. To refresh after an intended perf
change, re-run the benchmark on hardware matching the file's provenance block, overwrite the file with the new payload, and
keep `provenance` accurate.

The committed set is gpt-oss-20b SFT under Expert Parallelism at **ep1 / ep2 / ep8** — 8× B300, seq
4096, batch 1, gradient checkpointing on, `CUDA_DEVICE_MAX_CONNECTIONS=1`, bf16 + FA4 + grouped GEMM
on `halo:blackwell`: 9,401 / 10,551 / 8,225 tok/s/GPU. **ep4 is intentionally excluded** —
pure `2 < ep_size < gpus_per_node` deadlocks the DeepEP combine barrier and the trainer fails fast
([DeepEP](../infrastructure/deepep.md)).

## Docs

Update the owning doc page in the same PR when you change `src/` (`skills/docs/docs-ownership.md` is the map) and
follow the anti-slop charter carried by the `/docs` skill: American English, active voice, short
sentences, tables only for real matrices, no marketing register.

`docs.yml` runs three jobs, and `make docs` runs only the first: the relative-link check over
`agent-docs/`, `human-docs/`, `skills/` and the root markdown, which blocks a merge; a `diagrams` job
that re-runs every `scripts/diagrams/gen_*.py` and byte-compares the result against the committed PNGs
under `agent-docs/assets/` — touch a generator and you owe `make diagrams` plus the regenerated figure in
the same commit; and `markdownlint`. The last two are advisory.

## Proof of Value

| Change type | What to show |
|-------------|--------------|
| No-op / refactor | Loss is **bitwise-identical** to `main` at a fixed seed, and the original checkpoint still loads |
| Behavior change | An end-to-end test demonstrating the new behavior or convergence |
| Performance | Before/after **tokens/s/GPU** and **peak memory** (utilization is a diagnostic, not the headline) |

State the GPU, model, and sequence/batch shape with any number. A measured net-slower result is
first-class content — report it with the reason.

## Submitting a PR

1. **Get approved first** — an accepted issue plus `/approve` on it. An un-approved PR is closed by
   `pr-gate.yml`. Merging a PR adds you to the allowlist, so the gate applies once. An idle issue
   goes stale after 30 days and closes 7 days later.
2. **Branch** off `main` — in your fork, unless you have write access — and sign your commits
   (SSH or GPG): the repo requires signed commits, so a PR with unsigned commits can only be
   squash-merged.
3. **Pass the gates.** `make lint`, `make format`, `make test-cpu` (plus `make test-gpu-core` for
   GPU-affecting changes), `make docs`.
4. **Fill the PR template** — what and why, type of change, Proof-of-Value evidence, checklist.
5. **No secrets.** Never add `keys/`, `.env`, `*.pem`, or any credential.

Report vulnerabilities **privately** — see `SECURITY.md`, never a public issue.

## Where to look next

- [Development Environment](development-environment.md) — host `.venv`, dev container, env vars
- [Trainer Architecture](../reference/trainer-architecture.md) — how trainers compose with the mixin
- [Troubleshooting](../reference/troubleshooting.md) · [Debugging](../reference/debugging.md)
- [Adding a New Model](../models/adding-a-model.md)
