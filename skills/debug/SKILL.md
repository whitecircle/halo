---
name: debug
description: >-
  Diagnose distributed-training failures in Halo — hangs/deadlocks,
  CUDA OOM, loss=NaN/Inf, DeepEP faults, and NCCL collective timeouts — by
  routing the symptom to the exact opt-in helper, env var, or known failure
  mode. Auto-fire when a training run hangs/deadlocks, OOMs, produces
  loss=NaN/Inf, or hits a DeepEP/NCCL fault, or when the user asks to debug,
  triage, or diagnose such a problem. Grounds every recommendation in
  halo/diagnostics/, halo/callbacks/torch_profiler.py, and
  agent-docs/reference/debugging.md.
allowed-tools:
  - Read
  - Grep
  - Glob
  - Bash
---

# Distributed-training failure triage

Diagnose a hang / OOM / NaN-loss / DeepEP fault / NCCL timeout. Every helper and
env var below matches its owner **exactly** — `halo/runtime/consensus.py` (`assert_rank_uniform` consistency
check), `halo/diagnostics/py_spy.py` (py-spy capture), `halo/diagnostics/profiling.py`
(`cuda_memory_history`, `log_cuda_memory`), `halo/env.py` (`DIST_NCCL_TIMEOUT_MINUTES`),
`halo/models/attention/selection.py` (`is_fa4_backward_nan_prone`) — do not invent flags. Before a
verdict on anything non-trivial, find the symptom's row in
[Troubleshooting](../../agent-docs/reference/troubleshooting.md#symptom--cause--fix) and the helper's
enable recipe in [Debugging](../../agent-docs/reference/debugging.md).

**First, classify the symptom**, then follow the matching branch.

## Triage decision tree

### Hang / deadlock (job stuck, no progress, GPUs idle or one busy)

A hang is almost always a **collective mismatch**: one rank issues a collective
the others never reach.

1. Dump every rank's stack — the rank *not* in a collective is the culprit:
   `python scripts/profiling/py_spy_diag.py dump` from a shell in the training container attaches
   py-spy to every torchrun rank on that node. No launch-time setup, so it works on a job already hung;
   one file per rank under `$HALO_DATA_ROOT/profiling/py_spy/stacks`. Attaching needs
   `--cap-add=SYS_PTRACE` when the host's `kernel.yama.ptrace_scope` is 1 or 2 (0 needs nothing, 3 never
   attaches), else py-spy fails with `Permission denied`.
2. Suspect a shape/value divergence upstream of the stuck collective →
   `assert_rank_uniform(tuple(t.shape), group=..., label=..., strict=True)`.
3. An EP job that stops at startup naming `concurrent >2-rank DeepEP dispatch groups` is a rejected
   topology, not a hang — see the **DeepEP fault** branch.
4. Recipes: [Debugging §4](../../agent-docs/reference/debugging.md#4-diagnosing-multi-node-hangs).

### OOM (CUDA out of memory)

1. Find *what* holds memory: `enable_torch_profiler: true` + `profiler_record_memory_snapshot: true`
   (or `cuda_memory_history(...)`) → `.pickle` onto <https://pytorch.org/memory_viz> for the
   per-allocation flame graph.
2. Reduce levers, cheapest first: **gradient checkpointing** on; lower
   `per_device_train_batch_size`; lower `max_length` / seq len; switch to a
   parallelism that lowers DP (CP/TP). GC is refused for Zaya in every mode (cuDNN CCA fault)
   and forced reentrant under EP/CP/any MoE (the `CheckpointError` row of Troubleshooting), except
   under PP, which runs non-reentrant and refuses `use_reentrant: true`.
   Recipes: [Debugging §3](../../agent-docs/reference/debugging.md#3-gpu-memory-profiling).
3. Quick textual check: `log_cuda_memory("after forward")` / `EfficiencyCallback` peak mem.

### NaN / Inf loss

1. **Which model + attention impl?** qwen3.5/3.6/Qwen3-Next and GLM-4 MoE Lite under FA4 NaN the
   first backward (head_dim-256 + partial rotary) — the fix is the automatic fallback from FA4 to SDPA,
   or to FA2 under CP (`is_fa4_backward_nan_prone`, applied in `resolve_attn_implementation` in
   `halo/models/attention/selection.py`); confirm it engaged, don't force FA4.
2. bf16 path: AdamWBF16 stochastic rounding (SR on weight write + `exp_avg_sq`) is what
   makes tiny LRs converge; a NaN right after an optimizer step suggests SR/precision —
   check `fp32_grad_reduce` / `fp32_non_ep_params`.
3. `assert_rank_uniform` does not find a NaN: it compares a hash across ranks, so a NaN every rank
   computes identically passes. It helps only where ranks that must agree (TP-replicated
   activations) diverge — add the call there, with `strict=True` to raise.

### DeepEP fault (EP crash, combine-barrier deadlock)

- **Multiple >2-rank dispatch groups in one NVLink domain are rejected at config time** (`ep_size > 2`
  with `ep_group_size` below the NVLink domain, e.g. ep4 on 8 GPUs). **Fix: `ep_size=2` or
  `ep_size = nvlink_domain_size`**, or EP with **ETP** (`ep4+etp2`) for finer sharding — attention TP
  leaves `ep_group_size` untouched. Mechanism and evidence:
  [Expert Parallelism](../../agent-docs/parallelism/expert-parallelism.md#single-domain-multi-group-ep-races-and-hangs).
- Import, GIN, OOM-masquerading-as-launch-failure and EFA payload faults:
  [Troubleshooting → DeepEP](../../agent-docs/reference/troubleshooting.md#deepep-build--runtime-faults).

### NCCL timeout (watchdog fires after a stall)

1. Raise the window for legit-slow collectives (big all-to-all): `DIST_NCCL_TIMEOUT_MINUTES`, the one
   NCCL watchdog setting (a non-default `ddp_timeout` is refused). Checkpoint-write joins and
   rank-coordinated dataset work wait on the c10d store instead, bounded by `DIST_STORE_TIMEOUT_HOURS`
   ([defaults](../../agent-docs/reference/configuration-reference.md#environment-variables)).
2. Capture *which* collective each rank missed: `TORCH_NCCL_TRACE_BUFFER_SIZE=20000`,
   `TORCH_NCCL_DUMP_ON_TIMEOUT=1`, `TORCH_NCCL_DEBUG_INFO_TEMP_FILE=<path>`.
3. Subgroup reachability: a timeout in an EP/CP/TP **subgroup** means a rank took a
   different branch — confirm with the stack dump (hang branch) which subgroup is stuck.
4. Transport/topology (RDMA not engaging): `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,NET`.
5. Marginal NVLink fabric (intermittent stalls, Xid 145 flood in `dmesg`): preflight-gate with
   `python scripts/profiling/nvlink_health.py` (`--per_link` / `--json`) — flags hard link errors
   / deep FEC bins and down links, exits non-zero when a link is unhealthy or down or a GPU has fewer
   active links than its peers; `nvidia-smi`-only, safe alongside a live job.

### Slow but not stuck (throughput / straggler)

- Per-rank skew: capture `profiler_ranks: "all"` traces and run the TraceLens collective report
  (per-collective latency/bandwidth/skew): `python scripts/profiling/trace_report.py`.
- CPU stalls (dataloader/tokenize): `python scripts/profiling/py_spy_diag.py record --duration 30`
  → per-rank flame SVGs (in-script equivalent: `record_distributed_flamegraph`).
- Operator/kernel picture: `enable_torch_profiler: true` (`TorchProfilerCallback`) → Chrome trace;
  the EP phases are labeled ranges (`ep.dispatch` / `ep.expert_compute` / `ep.combine`, plus
  `grad_sync.reduce_bucketed`), so the comm fraction reads directly off the Perfetto timeline.
  `trace_report.py` turns the trace into TraceLens workbooks (gpu_timeline compute/comm/idle split,
  op tables, roofline).

The per-failure-mode rows (DeepEP ep4 rejection, Qwen3.5 FA4 NaN, Gemma 4 attention, Zaya GC refusal,
the MoE `CheckpointError`, tokenizer-cache embedding OOB, a fused EP export an engine rejects) are in
[Troubleshooting](../../agent-docs/reference/troubleshooting.md#symptom--cause--fix); reading
`TorchProfilerCallback` artifacts is [Debugging §1a](../../agent-docs/reference/debugging.md#1a-torchprofiler--gpuoperator-trace).

## Sources of truth

`agent-docs/reference/troubleshooting.md` + `debugging.md` capture the known failure modes. The code is the
**ultimate** authority: `halo/diagnostics/` (the opt-in helpers) and the failing
`halo/` path itself are what actually behave — when a doc, this skill, or memory disagrees, or you are
unsure, read the real file before concluding. (`CLAUDE.md`: docs-first, the code wins.) Related skill:
`data` (data loading: `num_shards < DP` raises at train load, S3 path split), `parallelism` (rejected/deadlocking combos).
