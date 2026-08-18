# GPU test harness reference

Concrete API for `tests/common/harness.py` and how a GPU test is launched. Every snippet
below matches the real harness — do not invent fields.

## `gpu_test_main` — the lifecycle decorator

```python
def gpu_test_main(
    *,
    min_world_size: int = 1,         # fewer GPUs than this = BAD LAUNCH → exit 2 (infra ERROR, not FAIL)
    exact_world_size: int | None = None,  # if set, world_size must equal it exactly
    prefix: str = "halo_test",       # temp-dir prefix for this test's isolated output/cache dirs
    partial_state: bool = True,      # build accelerate.PartialState() (needed by Trainer tests; off for pure-kernel)
): ...
```

It wraps a `def run(ctx) -> dict` body and owns the full lifecycle so the body is just
*load → train → assert*:

1. `init_distributed()` → `(rank, world_size, local_rank)`, optional `PartialState()`.
2. **Validate the launch before allocating** — wrong world size emits an `error` result and
   `sys.exit(2)` (the launcher classifies this as ERROR, not FAIL).
3. `setup_cache_dirs(prefix, rank)` → per-rank isolated `output_dir` / `cache_dir`.
4. Run the body inside `try`; in `finally`, **guaranteed teardown order**:
   `ctx._run_finalizers()` (LIFO) → `cleanup_memory()` → `cleanup_dirs(...)`, then — **only when the
   body did not raise** — `ctx.barrier()` → `teardown_distributed()`. A rank that raised has
   abandoned a collective its peers are still inside, so it exits immediately instead of blocking
   both on the watchdog and turning one rank's traceback into a job-wide hang.
5. Rank 0 prints the metrics table + checks, then `emit_result(...)` writes the
   `__HALO_TEST_RESULT__` line.
6. `sys.exit`: **0** = pass (all checks True), **1** = fail (a check False, body raised, or no
   checks returned), **2** = bad launch.

### The `ctx` object

| Member | Meaning |
|---|---|
| `ctx.rank`, `ctx.world_size`, `ctx.local_rank` | distributed coords |
| `ctx.device` | `torch.device(f"cuda:{local_rank}")` |
| `ctx.output_dir`, `ctx.cache_dir` | per-rank isolated dirs (auto-cleaned) |
| `ctx.on_teardown(fn)` | register a finalizer (e.g. `trainer.cleanup_ep`); run LIFO before teardown |
| `ctx.barrier()` | `dist.barrier()` if initialized |
| `ctx.broadcast_seed(seed=42)` | seed torch/cuda/random identically on all ranks (rank-0 value wins); returns the shared seed. Use when every rank must generate the **same** data. |
| `ctx.broadcast_checks(checks)` | AND rank 0's verdict into this rank's dict, for a check only rank 0 can make (a served model's response, an HTTP probe). Merges rather than replaces, so a check that failed only on rank 1 survives — the harness exits per rank, and a rank-0-only failure would otherwise read as a teardown race. |
| `ctx.metrics(trainer_or_cb)` | snapshot headline metrics from a trainer (or an `EfficiencyCallback`); returns `{}` if none attached |

### Return shape

```python
return {
    "checks":  {"loss_finite": True, "loss_decreased": True, "rank_loss_consistent": True},
    "metrics": ctx.metrics(trainer),   # {} is allowed for pure-correctness tests
}
```

The decorator computes `all(checks.values())`. **Returning no checks is an error** — always
return at least one. `metrics` is optional but should be present on any test that trains.

## Minimal copy-pasteable skeleton

```python
#!/usr/bin/env python
"""SFT under <mode>: <behaviour> matches the dense reference.

Run: torchrun --nproc_per_node=2 tests/gpu/<area>/test_<name>.py
"""
import math

from tests.common.harness import gpu_test_main
from tests.common.tolerances import TOL


@gpu_test_main(min_world_size=2, prefix="test_sft_ep")
def run(ctx) -> dict:
    ctx.broadcast_seed(42)               # every rank generates the SAME synthetic data

    # 1. Build a tiny model + deterministic synthetic dataset (NO Hub download on hot path).
    #    Load from a cached snapshot or a small local config.
    model, tokenizer = build_tiny_model(ctx)
    dataset = make_synthetic_sft_dataset(seed=42)   # seeded, reproducible

    # 2. Train a few REAL steps (>= 2 so the decrease check has signal).
    #    The body builds its own ParallelismConfig — ctx carries the launch, not the mode.
    pc = ParallelismConfig(ep_size=2, use_grouped_gemm=has_grouped_mm())
    trainer = DistributedSFTTrainer(model=model, ..., parallelism_config=pc)
    ctx.on_teardown(trainer.cleanup_ep)   # finalizer the decorator can't reach
    trainer.train()
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]

    # 3. Assert BEHAVIOUR + cross-rank invariants (verdict computed on ALL ranks).
    loss_finite = all(math.isfinite(x) for x in losses)
    loss_decreased = len(losses) >= 2 and losses[-1] < losses[0]
    rank_loss_consistent = all_ranks_agree(losses[-1], atol=TOL.rank_loss_consistency_abs)

    return {
        "checks": {
            "loss_finite": loss_finite,
            "loss_decreased": loss_decreased,
            "rank_loss_consistent": rank_loss_consistent,
        },
        "metrics": ctx.metrics(trainer),   # headline tokens/s/GPU + peak mem + step time
    }


if __name__ == "__main__":
    run()   # the decorator calls sys.exit itself
```

For a **pure-kernel** test (no Trainer / accelerate) pass `partial_state=False` and assert
numerical equivalence with `TOL.kernel_atol` / `TOL.kernel_rtol`.

## Register in the manifest (`tests/gpu/manifest.py`)

A GPU script is invisible until it has a `TestSpec`. Add one row (path relative to
`tests/gpu/`):

```python
MANIFEST: dict[str, TestSpec] = {
    ...
    "trainers/sft/test_sft_ep.py": TestSpec(
        nproc=2,                                       # --nproc_per_node
        markers=('gpu', 'full', '2gpu', 'ep', 'moe', 'gptoss'),  # all from ALL_MARKERS
        timeout=1000,                                  # seconds; process group killed on expiry
        # flaky=True,                                   # scoped reruns on transient errors
    ),
}
```

`TestSpec` fields: `nproc` (required), `markers` (must include `'gpu'` + a tier
`core` or `full` + a `Ngpu` tag + capability/model tags, all in `ALL_MARKERS`),
`args_matrix` (default `("",)`; a multi-mode script like `('--mode fsdp', '--mode ep')`
becomes several nodes), `timeout` (default 1200) and `flaky` (default `False`). World-size
strictness is **not** a manifest field:
the script owns it via `gpu_test_main(exact_world_size=N)`, which pins an exact count rather
than a bool, and the launcher always launches exactly `nproc`.

Drift guards: `unregistered_scripts()` (on-disk `test_*.py` missing from the manifest) and
`stale_entries()` (manifest rows whose file vanished) both fail collection. Run:

```bash
python -c "from tests.gpu.manifest import unregistered_scripts, stale_entries; \
    print(unregistered_scripts(), stale_entries())"
```

If you add a brand-new marker, add it to `ALL_MARKERS` — `tests/conftest.py` registers every
`ALL_MARKERS` entry with pytest, so `--strict-markers` (set in `pyproject.toml`) rejects typos
without a second list.

## Live-server e2e tests reuse a shared body

A test that drives a real rollout server does **not** hand-roll the round. The trainer-agnostic
half — served-policy probe, policy loader, parallelism verdict, the perturb → sync → prove-it-moved
round, the sink round — is `tests/common/on_policy_e2e.py`; the per-flavor bodies are
`tests/common/env_grpo_e2e.py` (`run_env_grpo_e2e`) and `tests/common/online_grpo_e2e.py`
(`run_online_grpo_e2e`), with `tests/common/thinking_budget.py` for the live
`rollout_max_thinking_tokens` check. A new dense/MoE arm is a thin wrapper script over one of
those, kept separate only so the manifest can attach its family markers, timeout and
`vllm_server` / `sglang_server` marker per file.

## Ports, tolerances, reporting

- **Ports** (`tests/common/ports.py`) — never hardcode `--master_port`. The launcher calls
  `free_port()` (bind `:0`, dedup per process) and passes it via env; your script's
  `init_distributed()` reads it. Hardcoded ports race across CI shards (`Errno 98`).
- **Tolerances** (`tests/common/tolerances.py`) — `from tests.common.tolerances import TOL`,
  then use the named constant: `TOL.rank_loss_consistency_abs`, `TOL.tp_grad_norm_spread_abs`,
  `TOL.parallel_vs_baseline_loss_abs`, `TOL.parallel_vs_baseline_train_loss_abs`,
  `TOL.ep_rank_loss_abs`, `TOL.logprob_atol/rtol`, `TOL.weight_atol`, `TOL.resume_loss_abs`,
  `TOL.grad_norm_rel`, `TOL.kernel_atol/rtol`. Never re-inline a literal — the name is the
  contract.
- **Reporting** (`tests/common/reporting.py`):
  - Correctness: `ctx.metrics(trainer)` → `snapshot_efficiency(cb)` flat dict. Headline at
    top level: `tokens_per_second` (per-GPU), `cluster_tokens_per_second`, `peak_allocated_gb`,
    `training_peak_allocated_gb`, `avg_step_time_seconds`. MFU/S-MFU/TFLOPS live under `"diagnostics"` and
    are never gated.
  - Benchmarks: `emit_benchmark("<key>", efficiency_cb)` prints a `__HALO_BENCH__` line a
    refresh run uses to seed `tests/baselines/<key>.json`; `format_benchmark_report(cb)` for
    the human log. The result/bench lines (`RESULT_SENTINEL` = `__HALO_TEST_RESULT__`,
    `BENCH_SENTINEL` = `__HALO_BENCH__`) survive interleaved torchrun stdout.

## How it's launched

The pytest launcher (`tests/gpu/conftest.py`) reads the manifest and, per `(script, args)`
node, shells out:

```
python -m torch.distributed.run --nproc_per_node=<nproc> --master_port=<free> <script> <args>
```

with the spec `timeout` (the whole process **group** is killed on expiry — NCCL/FA hangs are
live) and a `TORCHELASTIC_ERROR_FILE`. It then classifies:

- **PASS** — exit 0 (and `status="pass"` in the parsed result line).
- **FAIL** — a `__HALO_TEST_RESULT__` line with `status="fail"`, or non-zero exit that did
  emit checks (a real assertion failure).
- **ERROR** — non-zero exit with **no** result line → infra / hang / import crash.
- **SKIP** — fewer GPUs than `nproc` available, or an OOM on the 8-GPU `full` tier (an OOM on
  a 2-GPU `core` smoke is a real ERROR — that config must fit).

All of this runs inside the Docker image; selection is by marker, e.g. `pytest -m "gpu and
core"` (PR) or `pytest -m gpu` (nightly).
