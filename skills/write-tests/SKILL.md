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

Write a test that follows this repo's shared harness and manifest. **Test behavior, not
implementation**, and make it **fail when the behavior breaks**. The conventions (CPU entry, markers,
imports, manifest, ports, scratch, tiers, the anti-slop table, goldens and harnesses) are owned by
`agent-docs/contributing/README.md#tests`; read it before writing. The sibling files:

- **`harness.md`** — the `gpu_test_main` API, the shared check helpers, a copy-pasteable GPU skeleton,
  `TestSpec` fields, tolerances and reporting, how the launcher runs and classifies a script.
- **`matrix.md`** — the trainer × parallelism coverage checklist, the EP-family EP-vs-stock rows,
  and the combos that **must raise** (rejection tests).

## First: decide CPU vs GPU

- **CPU test** — pure logic: config validation, dataclasses, collator routing, YAML parsing,
  shard merge, loss math on tiny tensors, class-attribute inspection. No GPU, no model load.
- **GPU test** — anything that loads a model, trains a step, or exercises a CUDA/NCCL/DeepEP
  path (parallelism, kernels, optimizers, save/load round-trips).

Everything runs **inside the Docker image** (`halo:blackwell` / `:hopper`) — the
host has no usable Python. Never run tests on the host.

## CPU test workflow

Put it under `tests/cpu/<area>/`: module-level `def test_*()` functions with plain `assert`, and
`pytest.raises(..., match=...)` for a rejection so the *reason* is checked. No manifest entry.

```python
import pytest
from halo.distributed.parallelism_config import ParallelismConfig

def test_ep_tp_etp_combo_rejected():
    """EP+TP+ETP is unsupported and must raise at config time."""
    with pytest.raises(ValueError, match="is not a supported parallelism combination"):
        ParallelismConfig(ep_size=2, tp_size=2, expert_tp_size=2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
```

Mentally mutate the function under test (flip a sign, drop a clause): a real test goes red. Assert the
**exact** computed value or a tight bound with a reason; deterministic data only, no network.

## GPU test workflow

1. **Write the body** as `@gpu_test_main(...)` over `def run(ctx) -> dict` returning
   `{"checks": {name: bool}, "metrics": ctx.metrics(trainer)}` — the skeleton in `harness.md`; do not
   hand-roll init/finally.
2. **Register it** with one `TestSpec` in `tests/gpu/manifest.py` (markers from `ALL_MARKERS`; a
   pipeline suite carries `pp` and lives under `tests/gpu/parallelism/pp/`; a PP test on a family not in
   `PP_FAMILY_SPECS` opts it in with `registered_pp_backbones(...)`, `tests/common/pp_fakes.py`).
3. **Assert behavior on every rank**: loss finite *and* decreasing over ≥ 2 real steps; cross-rank
   verdicts computed on all ranks (a logged loss is already the world mean, so its spread cannot fail);
   parallel-vs-reference within the named `TOL.*` bounds; the parallel axis read off the model, not the
   config echo. Deterministic seeded synthetic data, no Hub download on the hot path.

A perf script is a `benchmark_*` / `bench_*` file run via shell, listed in `_UNMANIFESTED_BENCHMARKS`;
it must `sys.exit(main())` non-zero on crash and emit its line with `emit_benchmark` — gating:
`agent-docs/contributing/README.md#golden-performance-baselines`.

## Cover the matrix

Open `matrix.md` and tick off the trainer × parallelism cells the change touches, the EP-vs-stock
row for any new MoE family, and **every required rejection test**: the allowlist and its refusals
are `agent-docs/parallelism/README.md#supported-combinations`. A support-matrix change isn't done until
its rejection counterpart exists.

## Verify before finishing

Inside the image, sanity-check registration and (if a GPU is present) a smoke run:

```bash
python -c "from tests.gpu.manifest import unregistered_scripts, stale_entries; \
    print('unregistered:', unregistered_scripts()); print('stale:', stale_entries())"
python tests/cpu/conventions/test_test_conventions.py            # the CPU-test conventions
python tests/cpu/parallelism/test_matrix_correctness_coverage.py # renaming a parallelism test breaks this
# CPU:  python tests/cpu/<area>/<file>.py   (or: pytest -m cpu)
# GPU:  torchrun --nproc_per_node=<N> tests/gpu/<path>/<file>.py
```

## Sources of truth

`agent-docs/contributing/README.md#tests` + `harness.md` / `matrix.md` document the conventions. The code
is the **ultimate** authority: `tests/common/harness.py`, `tests/gpu/manifest.py`, and the `halo/` behavior
you are testing are what actually hold — when a doc, this skill, or memory disagrees, or you are unsure,
read the real file before writing the assertion. (`CLAUDE.md`: docs-first, the code wins.) Related
skill: `data` (the dataset format your fixtures must mimic).
