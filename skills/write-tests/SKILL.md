---
name: write-tests
description: Write CPU or GPU correctness/benchmark tests for Halo using the shared harness + manifest. Use when adding a test for a trainer, parallelism mode, kernel, optimizer, model family, or a perf benchmark — covering the support matrix and the required rejection tests. User-invoked only.
disable-model-invocation: true
allowed-tools:
  - Read
  - Edit
  - Write
  - Grep
  - Glob
  - Bash
---

# write-tests

Write a test that follows this repo's shared harness and manifest. **Test behaviour, not
implementation.** Use deterministic seeded synthetic data, never download on the hot path,
emit perf+mem on every GPU test, assert cross-rank invariants, and cover the support matrix
plus the rejection tests.

Detail lives in the sibling files — read them before writing:

- **`harness.md`** — the `gpu_test_main` API, a copy-pasteable GPU skeleton, manifest
  registration, ports/tolerances/reporting usage, how the launcher runs it.
- **`matrix.md`** — the trainer × parallelism coverage checklist, the EP-family
  `ep_vs_fsdp` baselines, and the combos that **must raise** (rejection tests).

## First: decide CPU vs GPU

- **CPU test** — pure logic: config validation, dataclasses, collator routing, YAML parsing,
  shard merge, loss math on tiny tensors, class-attribute inspection. No GPU, no model load.
- **GPU test** — anything that loads a model, trains a step, or exercises a CUDA/NCCL/DeepEP
  path (parallelism, kernels, optimizers, save/load round-trips).

Everything runs **inside the Docker image** (`halo:blackwell` / `:hopper`) — the
host has no usable Python. Never run tests on the host.

## CPU test workflow

Put it under `tests/cpu/<area>/`. CPU tests are **pytest-native**: module-level
`def test_*()` functions using plain `assert` (and `pytest.raises` for rejection cases).
Run via `python tests/cpu/<area>/<file>.py` or `pytest -m cpu` inside the image.

```python
import pytest
from src.distributed.parallelism_config import ParallelismConfig

def test_ep_tp_etp_combo_rejected():
    """EP+TP+ETP is unsupported and must raise at config time."""
    with pytest.raises(ValueError, match="not supported"):
        ParallelismConfig(ep_size=2, tp_size=2, expert_tp_size=2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
```

Rules:
- End the file with that `__main__` guard, declare no `pytestmark` (the `cpu` marker is applied
  by path) and no `sys.path` bootstrap — `tests/cpu/conventions/test_test_conventions.py` fails the
  suite over a hand-rolled runner or a printed pass/fail summary.
- Assert the **behaviour/invariant**, not the internal call sequence. A test that mirrors the
  implementation breaks on every refactor and catches nothing.
- **It must fail when the behaviour breaks.** Mentally mutate the function (flip a sign, drop a
  clause) — a real test goes red. Reject the slop patterns: tautologies (`assert True`, asserting a
  value you just set), smoke-only "didn't raise" checks, vacuous `assert out is not None` /
  `assert len(x) >= 0`, mocking the thing under test, and `try/except: pass` that turns a raise into a
  pass. Assert the **exact** computed value or a tight bound with a reason. (Full guide:
  `agent-docs/contributing/README.md` → "A test must not be AI-slop".)
- Deterministic data only — seed `torch`/`random` if you generate anything; no network.
- Rejection cases use `pytest.raises` with a `match=` substring so the *reason* is checked.
- No manifest entry needed for CPU tests (the manifest is GPU-only).

## GPU test workflow

1. **Write the body** as `@gpu_test_main(...)` over `def run(ctx) -> dict` returning
   `{"checks": {name: bool}, "metrics": ...}`. The decorator owns init → validate world size
   → cache dirs → teardown (`cleanup_ep → cleanup_memory → cleanup_dirs → barrier →
   teardown`) → the `__HALO_TEST_RESULT__` line → `sys.exit`. See `harness.md` for the full
   skeleton; do not hand-roll init/finally.
2. **Register in `tests/gpu/manifest.py`** — add one `TestSpec(nproc=..., markers=(...),
   timeout=..., ...)`. A script on disk but absent from the manifest fails collection
   (`unregistered_scripts`). Markers must come from `ALL_MARKERS`.
3. **Ports** — never hardcode `--master_port`; the launcher allocates a free one via
   `tests/common/ports.py`. Your script reads it from the env (handled by `init_distributed`).
4. **Tolerances** — import named constants from `tests/common/tolerances.py` (`TOL.*`); never
   re-inline a literal like `0.05`. The name documents what the bound guards.
5. **Perf + mem on every GPU test** — return `ctx.metrics(trainer)` so the headline
   tokens/s/GPU + peak mem + step time are snapshotted (`tests/common/reporting.py`). It
   returns `{}` if no `EfficiencyCallback` is attached — attach one (or pass the callback) so
   the metrics aren't empty.

### What every GPU correctness test must assert

- **Behaviour, not implementation** — loss is finite *and* decreases over ≥2 real steps
  (a 1-step run that skips the decrease check is a silent pass — require `len(losses) >= 2`).
- **Deterministic seeded synthetic data** — generate problems from a fixed seed; use
  `ctx.broadcast_seed()` when every rank must produce the *same* data (e.g. a parallel run
  compared to a single-GPU reference).
- **No hot-path downloads** — load from a cached snapshot / tiny local config; never hit the
  Hub mid-test.
- **Cross-rank invariants** — the DP-averaged loss agrees across ranks
  (`TOL.rank_loss_consistency_abs`); TP grad-norm is identical across the TP group
  (`TOL.tp_grad_norm_spread_abs`). Compute the verdict on **all** ranks (all-gather /
  broadcast) — setting `checks[...] = True` off rank 0 or on a `world_size < 2` fallback
  masks rank-skew bugs.
- **Parallel-vs-reference** — EP/CP/TP/ETP step-0 loss matches the dense reference within
  `TOL.parallel_vs_baseline_loss_abs`; the trend within `TOL.parallel_vs_baseline_train_loss_abs`.

### Benchmark tests (`benchmark_*`)

`test_*` = correctness (exit 0/1). `benchmark_*` = perf and live in
`tests/gpu/profiling/`. They MUST `sys.exit(main())` and return non-zero on crash (a crashed
benchmark that reports PASS is the worst failure mode). Emit the machine-readable line via
`emit_benchmark("<key>", efficiency_callback)` and diff the **headline**
tokens/s/GPU + peak mem against the committed golden in `tests/baselines/<key>.json`. MFU /
S-MFU / TFLOPS are opt-in diagnostics — never the gated number.

## Cover the matrix

Open `matrix.md` and tick off the trainer × parallelism cells the change touches, the
`ep_vs_fsdp` baseline for any new MoE family, and **every required rejection test**. The allowlist
is `SUPPORTED_AXIS_SETS` — plain DP, each axis alone (EP/ETP/TP/CP/PP), EP+TP, EP+CP, EP+ETP, PP+EP,
PP+ETP; everything else must raise, including TP+CP, TP+ETP, EP+TP+ETP, ETP+CP, and every PP pairing
outside the expert axes (PP+TP, PP+CP, PP+EP+TP, PP+EP+ETP). Also: QLoRA+EP/TP, LoRA+TP, and `_supports_cp=False` / `_supports_pp=False` trainers
rejecting `cp_size>1` / `pp_size>1` at init. A support-matrix change isn't done until its rejection
counterpart exists. **PP is not available in this release** — `pipeline_parallel_size > 1` is
rejected at config time and `PipelineRuntime` raises, so every PP cell takes a rejection test and
none takes a correctness suite (`matrix.md`).

## Verify before finishing

Inside the image, sanity-check registration and (if a GPU is present) a smoke run:

```bash
python -c "from tests.gpu.manifest import unregistered_scripts, stale_entries; \
    print('unregistered:', unregistered_scripts()); print('stale:', stale_entries())"
# CPU:  python tests/cpu/<area>/<file>.py   (or: pytest -m cpu)
# GPU:  torchrun --nproc_per_node=<N> tests/gpu/<path>/<file>.py
```

## Sources of truth
`harness.md` / `matrix.md` + `agent-docs/contributing/README.md` document the conventions. The code is the
**ultimate** authority: `tests/common/harness.py`, `tests/gpu/manifest.py`, and the `src/` behavior you
are testing are what actually hold — when a doc, this skill, or memory disagrees, or you are unsure,
read the real file before writing the assertion. (`CLAUDE.md`: docs-first, the code wins.) Related
skill: `data` (the dataset format your fixtures must mimic).
