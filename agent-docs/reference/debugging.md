# Debugging & Profiling Distributed Pipelines

Every helper here is opt-in and costs nothing when disabled.

| Goal | Tool | Enable |
|------|------|--------|
| Kernel/operator flame graph + Chrome trace | `TorchProfilerCallback` / `torch_profiler_session` | `enable_torch_profiler` YAML flag |
| Automated trace analysis (compute/comm/idle, roofline, collective skew) | TraceLens | `scripts/profiling/trace_report.py` |
| GPU memory timeline + per-allocation flame graph | CUDA memory snapshot | `profiler_record_memory_snapshot` / `cuda_memory_history` |
| CPU-side stalls (dataloader, tokenization) | py-spy flame graph | `scripts/profiling/py_spy_diag.py record` |
| Step time, tokens/s, peak memory, MFU/S-MFU | `EfficiencyCallback` | `enable_efficiency_metrics` YAML flag; MFU/S-MFU also need `report_mfu_diagnostics` |
| Rank skew on a collective | TraceLens collective report | `profiler_ranks: "all"`, then `scripts/profiling/trace_report.py` |
| Cross-rank value divergence | `assert_consistent` | add the call, then `HALO_TP_CONSISTENCY_CHECK=1` |
| Hang stack dump (every rank on the node) | `scripts/profiling/py_spy_diag.py dump` | run standalone; needs `--cap-add=SYS_PTRACE` |
| NVLink lane health (hard errors: Rx, symbol, link recovery, link integrity) | `scripts/profiling/nvlink_health.py` | run standalone; exit 1 = hard lane errors, while a failed `nvidia-smi` or an unrecognized counter layout raises |
| NCCL watchdog timeout | `init_distributed` | `DIST_NCCL_TIMEOUT_MINUTES` (default 30) |

The consistency helpers live in `src/diagnostics/debugging.py`. They are **manual
instrumentation** — the toolkit ships no call sites for them, so the env flag does nothing until you
add the call to the code you are investigating. The profiler session and CUDA-memory helpers live in
`src/diagnostics/profiling.py`; the in-loop callbacks in `src/callbacks/`.

## 1. Flame graphs

Use the **torch profiler** for the GPU/operator picture (which kernels dominate, CPU↔GPU gaps) and
**py-spy** for the pure-CPU picture (Python stalls between kernels).

### 1a. torch.profiler — GPU/operator trace {#1a-torchprofiler--gpuoperator-trace--flame-graph}

```yaml
enable_torch_profiler: true
profiler_output_dir: /mnt/profiling/torch   # default: $HALO_DATA_ROOT/profiling/torch
profiler_wait: 5        # skip the first 5 steps (warm caches, allocator settles)
profiler_warmup: 1      # 1 warmup step (profiler on, not yet recording)
profiler_active: 3      # record 3 steps into the trace
profiler_ranks: "0"     # "0" | "all" | "0,8"
profiler_record_memory_snapshot: false   # also dump a CUDA memory snapshot (§3)
```

Each selected rank writes into `profiler_output_dir`; artifacts carry a `-cycleN` label (one per wait→active cycle):

| Artifact | View with |
|----------|-----------|
| `<label>-cycleN-rankNN.trace.json.gz` (Chrome trace) | `chrome://tracing`, [Perfetto](https://ui.perfetto.dev), TensorBoard |
| `<label>-cycleN-rankNN.memory_timeline.html` | browser |
| `<label>-cycleN-rankNN.top_ops.txt` (top 25 ops by self-CUDA) | eyeball |

Traces label the toolkit's own phases as named ranges: `ep.dispatch` / `ep.expert_compute` /
`ep.combine` around the DeepEP spans of every EP layer, and `grad_sync.reduce_bucketed` around the
post-backward grad-sync sweep — so the all-to-all and grad-sync fractions read off the timeline
directly.

`with_stack=True` puts the Python frames in the trace itself (`python_function` events, readable in
Perfetto and the TraceLens workbook of §1b); torch's collapsed-stacks export writes nothing on the
image's torch, so `export_profiler_artifacts` drops the empty file rather than leaving a 0-byte one.
For a collapsed-stack flame graph use py-spy (§1c).

Keep the window small — stacks and shapes add real overhead, and the capture is one-shot so the
rest of training runs at full speed. Profile an arbitrary region the same way:

```python
from src.diagnostics.profiling import torch_profiler_session

with torch_profiler_session("/mnt/profiling/gen", label="generate", ranks="0"):
    out = model.generate(**inputs)
```

### 1b. TraceLens — automated trace analysis

[TraceLens](https://github.com/AMD-AGI/TraceLens) (in the images via the `profiling` dependency
group) turns the Chrome traces into workbooks: hierarchical compute/communication/idle attribution,
unique-op launcher tables, per-op roofline placement, and — for `profiler_ranks: "all"` captures —
per-collective latency, bus bandwidth, and cross-rank skew.

```bash
python scripts/profiling/trace_report.py                    # $HALO_DATA_ROOT/profiling/torch
python scripts/profiling/trace_report.py /mnt/profiling/torch --label trace-cycle1
```

Writes `<label>-rankNN.tracelens.xlsx` per rank and, for multi-rank captures dense from 0,
`<label>.collective.xlsx`, next to the traces (`--no-collective` skips the second). A CUDA-only
capture (`cpu_activity=False`) is refused — the report hangs every kernel off the `cpu_op` that
launched it — and any trace set left without a report exits 1 unless `--keep-going`. For short-kernel
studies, CSV output, or report diffs, call the TraceLens CLIs directly
(`TraceLens_generate_perf_report_pytorch`, `TraceLens_compare_perf_reports_pytorch`).

### 1c. py-spy — CPU flame graph (dataloader / Python stalls)

When GPUs are idle between steps, the bottleneck is CPU-side. Attach to the running job from a
shell in the same container — no launch-time flags:

```bash
# One SVG per rank → $TMPDIR/halo_diag_flamegraphs/<ts>-rank00/pid<pid>.svg (--output-dir moves it)
python scripts/profiling/py_spy_diag.py record --duration 30
```

One SVG per torchrun rank spots a single straggler; `--pid` narrows the target, `--rate` sets the
sampling rate (default 100/s), `--native` adds C/CUDA-runtime frames. In-script,
`record_distributed_flamegraph` defaults to `this_rank_only=True` — one SVG for the calling process;
pass `this_rank_only=False` for the per-rank sweep. Without py-spy on `PATH` the CLI exits 2;
attaching also needs ptrace permission — start the container with `--cap-add=SYS_PTRACE`, or py-spy
itself fails with `Permission denied`.

## 2. Finding throughput bottlenecks

One `nvidia-smi` sample narrows it before any trace — read power, not util %
([GPU Training Theory §11](gpu-training-theory.md#watch-power-not-utilization)).

Open the Chrome trace in Perfetto and read the gaps on the CUDA stream:

- Large gaps with NCCL kernels (`ncclDevKernel…`, all-to-all, reduce-scatter) →
  communication-bound. On multi-node, check whether the slow collective is cross-node (RDMA) that
  could be kept NVLink-local ([Multi-Node](../parallelism/multi-node.md)).
- CPU thread busy, GPU idle → dataloader / host overhead; confirm with a py-spy flame graph (§1c)
  and raise `dataloader_num_workers`.
- Back-to-back GEMM/attention, GPU ~100% busy → compute-bound (the good case); read `top_ops.txt`
  for the dominant kernels.

`enable_efficiency_metrics: true` logs per-step time, tokens/s, and peak memory. MFU/S-MFU are
computed every step but reach the headline log only with `report_mfu_diagnostics: true` — MFU as
`step_mfu_percent` / `avg_mfu_percent`, S-MFU as `step_smfu_percent` / `avg_smfu_percent` (MoE
only, `sparsity_factor < 1.0`).

For MoE, read **S-MFU** (and achieved TFLOPS / tokens/s/GPU), not plain MFU: plain MFU tracks local
params per rank, so low EP keeps most params local and over-reads while high EP shrinks `N_local`
and under-reads at the same throughput
([why](../optimization/throughput-benchmarks.md#why-moe-utilization-reads-low)). Judge against the
model class and EP degree, not the dense ceiling.

**Rank skew (stragglers)** — capture every rank with `profiler_ranks: "all"` and read the TraceLens
collective report (`scripts/profiling/trace_report.py`), which attributes wait time per collective
per rank without a code change.

**Fine-grained op timing** — `get_performance_monitor().time_operation(name)`
(`src/diagnostics/performance_monitor.py`) times a span on CUDA events where available, else the
wall clock, and accumulates it into the monitor's `.stats` map: one `TimingStats` (`count`,
`total_time`, `.avg_time`) per operation name. The EP layers wrap their
dispatch/expert-compute/combine phases with it when `HALO_EP_PERF_PROFILE=1`; the per-phase syncs
serialize the timing, so use it on a diagnostic run only.
`tests/gpu/profiling/benchmark_sft_ep.py --comm_profile` reads those stats and prints the
dispatch / expert-compute / combine split.

## 3. GPU memory profiling

```yaml
enable_torch_profiler: true
profiler_record_memory_snapshot: true   # dumps mem-<label>-rankNN.pickle
```

Recording starts when the first active step ends and stops one step past the window, so it is offset
from the trace: at `wait/warmup/active` 5/1/3 the history covers the last two active steps plus the
one after.

Or wrap any region manually:

```python
from src.diagnostics.profiling import cuda_memory_history

with cuda_memory_history("/mnt/profiling/oom", ranks="all"):
    trainer.train()      # or the step that OOMs
```

Drag the `.pickle` onto <https://pytorch.org/memory_viz> for the allocation timeline and
per-allocation flame graph, each block traced to its Python call site — the fastest way to find an
OOM source or a leak. For quick textual checks, `log_cuda_memory("after forward")` and
`reset_peak_memory_stats()` live in `src.diagnostics.profiling`; `EfficiencyCallback`
already reports peak memory per step.

`HALO_WEIGHT_SYNC_MEM_LOG=1` brackets every collective weight sync with per-rank
`[mem rankNN] weight-sync pre/post` lines (allocated / reserved / peaks, GiB). It is off by default,
and env-GRPO's single-process path emits nothing either way. *post − pre* isolates the sync, and the
per-rank spread exposes forwarding-rank asymmetry. Nothing resets the peak counters at a sync, so the
*pre* peaks run since process start — or since the current step began under
`enable_efficiency_metrics: true`, which resets them every step. `reserved` far above `peak_alloc` on
one rank means allocator pools are stranding blocks (e.g. allocations on short-lived CUDA streams),
not live tensors.

## 4. Diagnosing multi-node hangs

A hang is almost always a **collective mismatch**: one rank issues a collective (all-reduce,
all-gather, broadcast, barrier) that another never reaches, so NCCL blocks until the watchdog fires.

### Data-dependent backward graph desyncs FSDP/EP collectives

Signature: every GPU reads **100% utilization at idle power** (~190 W on a B300), every rank spins a
full CPU core, and no NCCL error is raised until the watchdog fires. The NCCL flight recorder names
it exactly: at the same `collective_seq_id` one rank issues a *different* collective than its peers
(e.g. `_reduce_scatter_base` on one rank, `_all_gather_base` on the others).

Root cause is a **per-rank backward graph** — every rank must run the same FSDP2 / EP grad
collectives in the same order, so any rank that skips a module's backward deadlocks the rest. The
usual trigger is a masked or empty row (variable-row RL padding, a fully-masked turn) whose loss
contribution is zero: if its forward output is left disconnected from the loss, autograd prunes that
row's backbone backward and its reduce-scatter never fires — on the rank that padded more such rows.
Keep every row connected to the loss (a value masked downstream is fine) so the backward is
identical on all ranks. This is *not* a `CUDA_DEVICE_MAX_CONNECTIONS` issue; CDMC only shifts the
timing that decides which step the mismatch lands on.

### Dump every rank's stack

```bash
python scripts/profiling/py_spy_diag.py dump   # → $TMPDIR/halo_diag_stacks/<ts>-rank00/pid<pid>.txt (--output-dir moves it)
```

It attaches to every training process on the node, so the container needs `--cap-add=SYS_PTRACE`.
The rank *not* in a collective (still in the dataloader, or a different `if` branch) is the culprit.

### Catch divergence before it deadlocks

`assert_consistent` / `assert_tensor_shape_consistent` compare a value across a process group and
raise a clear error instead of a downstream NCCL hang. Add the call at the suspect point — nothing
in the toolkit calls them for you:

```bash
export HALO_TP_CONSISTENCY_CHECK=1   # make assert_consistent raise instead of warn
```

```python
from src.diagnostics.debugging import assert_tensor_shape_consistent
assert_tensor_shape_consistent(hidden_states, group=tp_group, label="attn_in")
```

### NCCL flight recorder + watchdog timeout

PyTorch's NCCL watchdog defaults to 10 minutes; `init_distributed()` raises it via
`DIST_NCCL_TIMEOUT_MINUTES` (default 30). Raise it further for very large cross-node all-to-all or
gathered checkpoint saves. To capture every rank's NCCL state at the moment of a timeout — the
definitive "which collective did rank K miss?":

```bash
export DIST_NCCL_TIMEOUT_MINUTES=60
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000   # ring buffer of recent collectives
export TORCH_NCCL_DUMP_ON_TIMEOUT=1         # dump the buffer when the watchdog fires
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$TMPDIR/halo_diag/nccl_trace
```

For NCCL transport/topology problems (RDMA not engaging, falling back to TCP), set
`NCCL_DEBUG=INFO` with `NCCL_DEBUG_SUBSYS=INIT,NET`. The NCCL/IB env vars themselves
(`NCCL_IB_DISABLE`, `NCCL_NET_GDR_LEVEL`, `NCCL_SOCKET_IFNAME`) are documented in
[Launch Recipes](../parallelism/launch-recipes.md).

## Related

- [Troubleshooting](troubleshooting.md) — symptom → cause → fix
- [Multi-Node Parallelism](../parallelism/multi-node.md) — NCCL/IB env vars, launch, FS coordination
- [Throughput Benchmarks](../optimization/throughput-benchmarks.md) — throughput methodology
