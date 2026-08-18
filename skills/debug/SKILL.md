---
name: debug
description: >-
  Diagnose distributed-training failures in Halo — hangs/deadlocks,
  CUDA OOM, loss=NaN/Inf, DeepEP faults, and NCCL collective timeouts — by
  routing the symptom to the exact opt-in helper, env var, or known failure
  mode. Auto-fire when a training run hangs/deadlocks, OOMs, produces
  loss=NaN/Inf, or hits a DeepEP/NCCL fault, or when the user asks to debug,
  triage, or diagnose such a problem. Grounds every recommendation in
  src/diagnostics/debugging.py, src/callbacks/profiler.py, and
  agent-docs/reference/debugging.md.
allowed-tools:
  - Read
  - Grep
  - Glob
  - Bash
---

# Distributed-training failure triage

Diagnose a hang / OOM / NaN-loss / DeepEP fault / NCCL timeout. Every helper and
env var below matches `src/diagnostics/debugging.py` **exactly** — do not
invent flags. The full symptom→cause→fix table and helper-enable recipes are in
**[playbook.md](playbook.md)**; read it before giving a verdict on anything
non-trivial. User-facing guide: `agent-docs/reference/debugging.md`.

**First, classify the symptom**, then follow the matching branch.

## Triage decision tree

### Hang / deadlock (job stuck, no progress, GPUs idle or one busy)
A hang is almost always a **collective mismatch**: one rank issues a collective
the others never reach.
1. Dump every rank's stack — the rank *not* in a collective is the culprit:
   `python scripts/profiling/py_spy_diag.py dump` from a shell in the training container attaches
   py-spy to every torchrun rank. No launch-time setup, so it works on a job already hung; one file
   per rank under `$TMPDIR/halo_diag_stacks`. The container must have `--cap-add=SYS_PTRACE`, else
   py-spy fails with `Permission denied`.
2. Suspect a shape/value divergence upstream of the stuck collective →
   `HALO_TP_CONSISTENCY_CHECK=1` + `assert_tensor_shape_consistent(t, group=..., label=...)`.
3. If it crashes ~30s into EP, jump to the **DeepEP fault** branch (the ep4 deadlock).

### OOM (CUDA out of memory)
1. Find *what* holds memory: `profiler_record_memory_snapshot: true` (or `cuda_memory_history(...)`)
   → `.pickle` onto <https://pytorch.org/memory_viz> for the per-allocation flame graph.
2. Reduce levers, cheapest first: **gradient checkpointing** on; lower
   `per_device_train_batch_size`; lower `max_length` / seq len; switch to a
   parallelism that lowers DP (CP/TP). Note GC is **broken** in a couple of cases
   (Zaya FSDP2+GC cuDNN regression; EP multi-group GC) — see playbook before enabling.
3. Quick textual check: `log_cuda_memory("after forward")` / `EfficiencyCallback` peak mem.

### NaN / Inf loss
1. **Which model + attention impl?** qwen3.5/3.6/Qwen3-Next and GLM-4 MoE Lite under FA4 NaN the
   first backward (head_dim-256 + partial rotary) — the fix is the auto FA4→SDPA fallback
   (`model_fa4_backward_nan_prone`, applied in `resolve_attn_implementation` in
   `src/models/patches/attention.py`); confirm it engaged, don't force FA4.
2. bf16 path: AdamWBF16 stochastic rounding (SR on weight write + `exp_avg_sq`) is what
   makes tiny LRs converge; a NaN right after an optimizer step suggests SR/precision —
   check `fp32_grad_reduce` / `fp32_non_ep_params`.
3. Localize the first NaN with `HALO_TP_CONSISTENCY_CHECK=1` + `assert_consistent` on
   the suspect tensor across the group.

### DeepEP fault (EP crash, combine-barrier deadlock)
- **Multiple >2-rank dispatch groups in one NVLink domain FAIL** (`ep_size > 2` with
  `nvlink_domain_size > ep_group_size`, e.g. ep4 on an 8-GPU domain): the job dies ~30s in,
  and `EpIntrospectionMixin._setup_ep_gradient_checkpointing`
  (`src/trainers/mixins/ep_introspection.py`) fails fast on this topology. **Fix: use
  ep_size=2 or ep_size = nvlink_domain_size** (one group per domain). For finer sharding
  combine EP with **ETP** (`ep4+etp2`) — TP leaves `ep_group_size` untouched.
  Per-symptom row: [playbook.md](playbook.md) §1. Mechanism and measured evidence live once,
  in the `parallelism` skill (`matrix.md`, row *Multi-group >2-rank EP on one NVLink domain*).

### NCCL timeout (watchdog fires after a stall)
1. Raise the window for legit-slow collectives (big all-to-all / gathered save):
   `DIST_NCCL_TIMEOUT_MINUTES` (default 30; PyTorch's own is 10).
2. Capture *which* collective each rank missed: `TORCH_NCCL_TRACE_BUFFER_SIZE=20000`,
   `TORCH_NCCL_DUMP_ON_TIMEOUT=1`, `TORCH_NCCL_DEBUG_INFO_TEMP_FILE=<path>`.
3. Subgroup reachability: a timeout in an EP/CP/TP **subgroup** means a rank took a
   different branch — confirm with the stack dump (hang branch) which subgroup is stuck.
4. Transport/topology (RDMA not engaging): `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,NET`.
5. Marginal NVLink fabric (intermittent stalls, Xid 145 flood in `dmesg`): preflight-gate with
   `python scripts/profiling/nvlink_health.py` (`--per-link` / `--json`) — flags hard link errors
   / deep FEC bins, exits non-zero when a link is unhealthy; `nvidia-smi`-only, safe alongside a
   live job.

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

See **[playbook.md](playbook.md)** for the per-failure-mode table (DeepEP ep4
deadlock, qwen3.5 FA4 NaN, gemma4 attn, Zaya FSDP2+GC cuDNN, tokenizer-cache
embedding-OOB, EP-save bias loss), the exact env-var enable recipe for each
`debugging.py` helper, and how to read `TorchProfilerCallback` artifacts.

## Sources of truth
`agent-docs/reference/debugging.md` + `playbook.md` capture the known failure modes. The code is the
**ultimate** authority: `src/diagnostics/debugging.py` (the opt-in helpers) and the failing
`src/` path itself are what actually behave — when a doc, this skill, or memory disagrees, or you are
unsure, read the real file before concluding. (`CLAUDE.md`: docs-first, the code wins.) Related skill:
`data` (data-loading hangs: `num_shards < DP`, S3 path split), `parallelism` (rejected/deadlocking combos).
