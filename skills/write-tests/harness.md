# GPU test harness reference

Concrete API for `tests/common/harness.py` and how a GPU test is launched. Every snippet
below matches the real harness — do not invent fields.

## `gpu_test_main` — the lifecycle decorator

```python
def gpu_test_main(
    *,
    min_world_size: int = 1,         # fewer GPUs than this = BAD LAUNCH → status="error" result line + exit 2 (the launcher reports it as FAIL)
    exact_world_size: int | None = None,  # if set, world_size must equal it exactly
    prefix: str = "halo_test",       # temp-dir prefix for this test's isolated output/cache dirs
    partial_state: bool = True,      # build accelerate.PartialState() (needed by Trainer tests; off for pure-kernel)
): ...
```

It wraps a `def run(ctx) -> dict` body and owns the full lifecycle so the body is just
*load → train → assert*:

1. `CAUSAL_CONV1D_DETERMINISTIC=1` unless the caller exported a value (causal_conv1d's default
   backward sums the conv weight gradient with atomics, so the gated-DeltaNet families would miss an
   exact resume replay now and then), then `init_distributed()` → `(rank, world_size, local_rank)`,
   optional `PartialState()`.
2. **Validate the launch before allocating** — wrong world size emits an `error` result and
   `sys.exit(2)` (the launcher reports the `status="error"` line as a FAIL naming the launch).
3. `setup_cache_dirs(prefix, rank)` → per-rank isolated `output_dir` / `cache_dir`.
4. Run the body inside `try`; in `finally`, **guaranteed teardown order**:
   `ctx._run_finalizers()` (LIFO) → `cleanup_memory()` → `cleanup_dirs(...)`, then — **only on the
   clean path** (`status != "error"`, so a body that raised *or* returned no checks skips it) —
   `ctx.barrier()` → `teardown_distributed()`. A rank that raised has
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
| `ctx.broadcast_seed(seed=42)` | seed torch/cuda/random identically on all ranks (rank-0 value wins); returns the shared seed. Use when every rank must generate the same data. |
| `ctx.broadcast_checks(checks)` | AND rank 0's verdict into this rank's dict, for a check only rank 0 can make (a served model's response, an HTTP probe). Merges rather than replaces, so a check that failed only on rank 1 survives — the harness exits per rank, and a rank-0-only failure would otherwise read as a teardown race. |
| `ctx.metrics(trainer_or_cb)` | snapshot headline metrics from a trainer (or an `EfficiencyCallback`); returns `{}` if none attached |

### Return shape

```python
return {
    "checks":  {"loss_finite": True, "all_steps_finite": True, "expert_bank_split_ep_way": True},
    "metrics": ctx.metrics(trainer),   # {} is allowed for pure-correctness tests
}
```

The decorator computes `all(checks.values())`. **Returning no checks is an error** — always
return at least one. `metrics` is optional but should be present on any test that trains.

Every check must be able to fail. A cross-rank spread of a logged loss cannot (HF logs the world
mean), and neither can `trainer.parallelism_config.has_ep_group` / `is_tp_mode` / `is_cp_mode`, which
read back the config the trainer was handed; read the axis off the model instead.

### Shared checks and probes

Build checks from these rather than re-deriving them per file:

| Helper | Module | What it gives |
|---|---|---|
| `step_losses(trainer)` | `tests/common/utils.py` | the per-step training losses, eval entries and the run summary excluded |
| `training_run_checks(result, trainer, max_steps, *, loss_band=, grad_norms=, loss_decreased=)` | `tests/common/utils.py` | `loss_finite`, `all_steps_finite`, `steps_completed`, and on request `loss_reasonable`, `grad_norms_finite`, `loss_decreased` |
| `parallel_shape_checks(model, parallelism_config)` | `tests/common/parallel_shape.py` | one model-side probe per axis the config enables: EP wrappers, the expert bank split EP-way, ETP sharding, TP-sharded params, Ulysses attention layers |
| `ep_layers(model)` | `tests/common/ep_reference.py` | every EP/ETP-wrapped MoE layer |
| `group_max_abs_diff(tensor, group)` | `tests/common/distributed.py` | the replica-identity probe: the largest elementwise difference across a group, NaN-propagating (collective) |
| `pin_deterministic_ep_dispatch()` | `tests/common/distributed.py` | every DeepEP `ElasticBuffer` of the process built in deterministic mode, for an EP body that replays a run exactly without `full_determinism` (under it the dispatcher does this itself); the default dispatch claims receive slots with atomics, so expert weight gradients round differently run to run. A forward-only comparison on the grouped-GEMM path needs no pin: each expert row is independent of arrival order and the combine sums in top-k slot order ([EP → Determinism](../../agent-docs/parallelism/expert-parallelism.md#determinism)) |
| `model_save_checks` / `resume_checkpoint_checks` / `resume_continuity_checks` | `tests/common/checkpoint_io.py` | the files a `save_model` or a mid-training checkpoint must hold, and what a resume restored at its first step |
| `run_two_phase_resume(ctx, build_trainer, *, output_dir=, probe=, optimizer_state=)` | `tests/common/checkpoint_io.py` | the whole resume body: train to a checkpoint, resume a fresh trainer from it, and return both phases' checks, optionally with the optimizer state compared bit-exact (collective) |
| `run_sft_suite(ctx, SFTSuite(...), {key: SFTMode(...)}, default_mode=)` | `tests/common/sft_modes.py` | the whole SFT smoke body: one `--mode` per manifest row, load → train → the checks above; a mode with `adapters` loads through the production PEFT path, and its `before_train` hook returns the checks run after training |
| `run_lora_sft(ctx, *, model=, mode=, attn=, qlora=, convergence=, save_reload=, resume=)` | `tests/common/lora_sft_modes.py` | the LoRA SFT rows on that body: only the adapters train and every one moves, plus per row a save reloaded on a fresh unsharded base, a restore onto the zeroed live adapters, or a 25-step loss curve |
| `train_recording_first_step` / `score_first_step` / `first_step_checks` / `first_step_gradient_checks` | `tests/common/first_step.py` | a parallel run's first optimizer step (microbatch losses, logged loss, sharded gradients, c10d autograd fallbacks) scored against a reference trainer on the same microbatches, for an objective a parallel axis could miscount |
| `skip_unless_local_checkpoint(path, env_var)` | `tests/common/harness.py` | the `SKIP:` exit for a suite whose local checkpoint is absent, called under `__main__` before `run()` |
| `skip_unless_blackwell(what)` | `tests/common/harness.py` | the same `SKIP:` exit for a suite or row that runs FlashAttention 4, on a pre-Blackwell GPU |
| `max_or_nan(values, *, default=)` | `tests/common/utils.py` | the largest value, NaN when any is NaN; the builtin `max` drops a NaN anywhere but first, so a bound on it would pass |
| `run_gloo_ranks(worker, nprocs, *args, pg_timeout=, env=)` | `tests/common/gloo.py` | `nprocs` spawned ranks sharing one gloo group, for a CPU test that needs a real process group; a rank that raises, dies or hangs fails the call. The rendezvous store is served from the calling process, and no worker starts before every rank has initialized the group: a rank that finishes first tears down neither the store nor a pair a peer is still connecting |
| `UnsplitReference` / `build_pp_sft_trainer` / `compare_stage_grads_vs_reference` / `check_expert_grads_vs_reference` | `tests/common/pp_equivalence.py` | the PP gradient-equivalence blocks: an unsplit reference's loss and gradients, a real trainer on a hand-built PP config, and each stage's parameter and expert gradients scored against that reference |
| `build_pp_trainer` / `build_pp_args` / `expect_construction_error` / `losses_identical_across_ranks` | `tests/common/pp_trainers.py` | a real trainer under PP with its runner's `step` and `eval_loss` spied (each call's batch, normalizer and extra targets), its PP training args, a construction refusal recorded as a check, and whether every rank logged the same losses (collective) |
| `hf_eval_loss_reference` / `eval_loss_over_examples` | `tests/common/distributed.py` | two independent oracles for a run's `eval_loss` from per-rank reference losses: HF's gather with the DP-scoped dedup and accelerate's tail trim, and an example-by-example walk of the eval set (collective) |
| `tiny_family_model` / `build_tiny_family_checkpoint` / `shared_tiny_family_checkpoint` | `tests/common/tiny_models.py` | one seeded random-init tiny model per EP MoE family, and a synthetic checkpoint of it the production loaders read (`shared_…` builds it once on rank 0 into a rank-shared scratch dir) |

## Minimal copy-pasteable skeleton

```python
#!/usr/bin/env python
"""SFT under <mode>: <behavior> matches the dense reference.

Run: torchrun --nproc_per_node=2 tests/gpu/<area>/test_<name>.py
"""
from tests.common.harness import gpu_test_main
from tests.common.parallel_shape import parallel_shape_checks
from tests.common.utils import training_run_checks

MAX_STEPS = 5


@gpu_test_main(min_world_size=2, prefix="test_sft_ep")
def run(ctx) -> dict:
    # 1. Build a tiny model + deterministic synthetic dataset (NO Hub download on hot path).
    #    Load from a cached snapshot or a small local config.
    model, tokenizer = build_tiny_model(ctx)
    dataset = make_synthetic_sft_dataset(seed=42)   # same seed on every rank: the SAME data

    # 2. Train a few REAL steps (>= 2 so the decrease check has signal).
    #    The body builds its own ParallelismConfig — ctx carries the launch, not the mode.
    pc = ParallelismConfig(ep_size=2)
    trainer = DistributedSFTTrainer(model=model, ..., parallelism_config=pc)
    ctx.on_teardown(trainer.cleanup_ep)   # finalizer the decorator can't reach
    checks = parallel_shape_checks(model, pc)   # the axis, read off the model
    result = trainer.train()

    # 3. Assert BEHAVIOR (verdict computed on ALL ranks). A logged loss is already the world mean
    #    (HF all-gathers it at every log step), so a cross-rank spread of it cannot fail.
    checks |= training_run_checks(result, trainer, MAX_STEPS, grad_norms=True, loss_decreased=True)

    return {
        "checks": checks,
        "metrics": ctx.metrics(trainer),   # headline tokens/s/GPU + peak mem + step time
    }


if __name__ == "__main__":
    run()   # the decorator calls sys.exit itself
```

For a **pure-kernel** test (no Trainer / accelerate) pass `partial_state=False` and assert
numerical equivalence with `TOL.kernel_atol` / `TOL.kernel_rtol`.

`record_check(checks, name, fn)` (same module) is the sanctioned way to record many independent
verdicts in one launch without aborting at the first failure — the conventions test names it as the
replacement for a printed pass/fail summary. Every raise inside an `fn` must be rank-symmetric or come
after its last collective, or the ranks desynchronize. Cross-rank verdicts come from
`tests/common/distributed.py` (`world_mean`, `world_any`, `world_min`, `world_spread`,
`group_max_abs_diff`).

## Register in the manifest (`tests/gpu/manifest.py`)

A GPU script is invisible until it has a `TestSpec`. Add one row (path relative to
`tests/gpu/`):

```python
MANIFEST: dict[str, TestSpec] = {
    ...
    "trainers/sft/test_sft_example.py": TestSpec(
        nproc=2,                                       # --nproc_per_node
        markers=('gpu', 'full', '2gpu', 'ep', 'moe', 'qwen3'),  # all from ALL_MARKERS
        timeout=1000,                                  # seconds; process group killed on expiry
        # flaky=True,                                   # scoped reruns on transient errors
    ),
}
```

`TestSpec` fields: `nproc` (required), `markers` (by convention `'gpu'` + a tier
`core` or `full` + a `Ngpu` tag + capability/model tags; only membership in `ALL_MARKERS` is
enforced, by `tests/cpu/conventions/test_gpu_harness_conventions.py`),
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
`ALL_MARKERS` entry with pytest without a second list. Pytest only warns on an unregistered mark, so
the CPU conventions test is what fails a manifest marker missing from `ALL_MARKERS`; a typo would
otherwise drop its row from every selection by that marker.

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

- **Ports** (`tests/common/ports.py`) — the launcher calls `free_port()` (a port below the kernel's
  ephemeral range, from this xdist worker's own slice of the pool) and passes it via env, where your
  script's `init_distributed()` reads it; a CPU test that spawns ranks or starts a server takes its port
  from `free_port()` too. Never hardcode one ([conventions](../../agent-docs/contributing/README.md#tests)).
- **Hub files** (`tests/common/tokenizers.py`) — load a real tokenizer, processor, config or
  chat template through these helpers, and name its repo in `tests/common/models.py`. `make
  seed-hf-cache` fetches a seed derived from that module and the `examples/` configs, and an
  offline run over it fails a cache miss (`HALO_TEST_REQUIRE_HUB_CACHE`; see
  [Contributing → Tests](../../agent-docs/contributing/README.md#tests)).
- **Tolerances** (`tests/common/tolerances.py`) — `from tests.common.tolerances import TOL`,
  then use the named constant: `TOL.ep_identical_batch_rank_spread_abs`,
  `TOL.parallel_vs_baseline_loss_abs`, `TOL.parallel_vs_baseline_train_loss_abs`,
  `TOL.logprob_atol`, `TOL.weight_atol`, `TOL.resume_loss_abs`, `TOL.kernel_atol/rtol`. Never
  re-inline a literal — the name is the contract. An EP-vs-reference gradient test scores its `name -> (EP grad, reference)` pairs with
  `tests.common.ep_reference.score_ep_grad_pairs(pairs, checks, metrics, cos_min=TOL.ep_grad_cosine_min)`;
  its norm-ratio band defaults to `TOL.ep_grad_norm_ratio_band`.
- **Reporting** (`tests/common/reporting.py`):
    - Correctness: `ctx.metrics(trainer)` → `snapshot_efficiency(cb)` flat dict. Headline at
    top level: `tokens_per_second` (per-GPU), `cluster_tokens_per_second`, `peak_allocated_gb`,
    `training_peak_allocated_gb`, `avg_step_time_seconds`. MFU/S-MFU/TFLOPS live under `"diagnostics"` and
    are never gated.
    - Benchmarks: `emit_benchmark("<key>", efficiency_cb)` (`emit_save_benchmark` for a checkpoint
    save) prints this rank's `__HALO_BENCH__` line, which `make bench BAND=1` holds to
    `tests/baselines/bench/<key>.json` and `make bench RECORD=1` records; `format_benchmark_report(cb)` for
    the human log. The result/bench lines (`RESULT_SENTINEL` = `__HALO_TEST_RESULT__`,
    `BENCH_SENTINEL` = `__HALO_BENCH__`) survive interleaved torchrun stdout.

## How it's launched

The pytest launcher (`tests/gpu/conftest.py`) reads the manifest and, per `(script, args)`
node, shells out:

```bash
python -m torch.distributed.run --nproc_per_node=<nproc> --master_port=<free> <script> <args>
```

with the spec `timeout` (on expiry the agent and every worker are killed — NCCL/FA hangs are
live) and `MASTER_PORT` / `HALO_TEST_LAUNCH_ID` / `TMPDIR` in the env. `TORCHELASTIC_ERROR_FILE` is
deliberately left unset: one shared path makes the last writer win, and the agent's collateral
SIGTERMs would overwrite the real cause. It then classifies:

- **PASS** — exit 0 with `status="pass"` in the parsed result line.
- **FAIL** — a `__HALO_TEST_RESULT__` line with `status="fail"` or `"error"` (a check False, a body
  that raised, no checks returned, a bad launch), or a non-zero exit whose line says `pass` (a
  non-zero rank failed).
- **ERROR** — non-zero exit with **no** result line → infra / hang / import crash. Plus TIMEOUT.
- **SKIP** — fewer GPUs than `nproc` available; an OOM on the 8-GPU `full` tier (an OOM on
  a 2-GPU `core` smoke is a real ERROR — that config must fit); an unreachable
  `vllm_server`/`sglang_server` engine, unless `HALO_TEST_REQUIRE_SERVER` names it (then a
  `UsageError`); or an exit-0 run that printed a `SKIP:` line (`skip_unless_local_checkpoint`,
  `skip_unless_blackwell`).
  **Zero** visible GPUs is a `UsageError`, never a skip.

All of this runs inside the Docker image; selection is by marker, e.g. `pytest -m "gpu and
core"` (`make test-gpu-core`, pre-merge) or `pytest -m gpu` (`make test-gpu-full`).
