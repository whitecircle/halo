# Debug playbook — symptom → cause → fix

Companion to `SKILL.md`. Sources: `src/diagnostics/debugging.py`,
`src/callbacks/profiler.py` (`TorchProfilerCallback`),
`src/callbacks/wiring.py` (wiring), and the user-facing guide
`agent-docs/reference/debugging.md`. Every env var / helper name below is verbatim
from those files — do not paraphrase them.

## 1. Known failure modes

| Symptom | Likely cause | Enable / inspect | Fix |
|---------|--------------|------------------|-----|
| EP job dies ~30s in: `CUDA error: Invalid access of peer GPU memory over nvlink` on the `elastic` default, or a combine-barrier deadlock around step 2 on `legacy` | Multiple >2-rank DeepEP dispatch groups inside ONE NVLink domain — `num_nvlink_domains == 1 and ep_size > 2 and nvlink_domain_size > ep_group_size` (e.g. ep4 on an 8-GPU domain; the unit is the **domain**, not the OS node). Their combine barriers race FSDP2's DP-wide NCCL — full mechanism and evidence in the `parallelism` skill (`matrix.md`, row *Multi-group >2-rank EP on one NVLink domain*) | Stack dump (§2.2); `EpIntrospectionMixin._setup_ep_gradient_checkpointing` (`src/trainers/mixins/ep_introspection.py`) already fails fast on this topology | Use **ep_size=2 or ep_size = nvlink_domain_size** (one dispatch group per domain). For finer expert sharding combine EP with **ETP** (`ep4+etp2` fills the domain and passes) — attention TP leaves `ep_group_size` untouched and lands on the same rejection |
| loss=NaN on the **first backward**, qwen3.5 / qwen3.6 / Qwen3-Next / GLM-4 MoE Lite | FA4 beta + head_dim 256 + partial rotary (qwen3.x output-gate + GQA 16:2; GLM-4 MoE Lite MLA 256-wide qk/v) | Confirm the auto fallback fired (`model_fa4_backward_nan_prone` → SDPA in `resolve_attn_implementation`, `src/models/patches/attention.py`) | Do **not** force `flash_attention_4`; let it fall back to SDPA. gpt-oss is unaffected |
| OOM / attention error on **gemma4** at long seq | head_dim=512 blocks FA2; needs mem-efficient SDPA + manual KV repeat | — | Auto-applied in `load_distributed_model` (no `enable_gqa`); cap seq (~20k) if still tight |
| `gradient_checkpointing_enable` raises on **Zaya** | A toolkit patch clears the family's GC support flag: backward recompute through CCA's `nn.Conv1d` is an env-level cuDNN/CUDA 13.2 fault on the Blackwell image | — | Run Zaya **without GC** (plain FSDP2, EP, or EP+ETP all work GC-off). EP+GC unsupported by design (CCA+EDA recompute cascades); TP/CP incompatible |
| `CheckpointError` (recomputed tensor shape ≠ forward) under **EP + gradient checkpointing**, often in GRPO | `use_reentrant=False` validates recompute shapes; the extra no-grad GRPO forwards make the recompute take a different path than the forward it must reproduce | Watch for the override warning at trainer init | The mixin forces `use_reentrant=True` for EP+GC everywhere except PP (which requires non-reentrant) even when the config (every GRPO template) sets `false` — let it; do not pin `use_reentrant: false` under EP |
| Embedding index OOB / CUDA assert early in training after switching model | A stale HF `map` cache keyed without the tokenizer → cross-model token-ID reuse | — | The cache key folds the closure-captured tokenizer and `functools.partial` bound args (`src/data/pipeline/processing.py`). A value it cannot fingerprint takes a one-time `Dataset-map cache fingerprint … skips a value of type …` warning — thread that value through `cache_key_extras`; clearing `HF_DATASETS_CACHE` is the blunt fallback |
| EP checkpoint reload → meta-tensor / missing-bias error | A stale checkpoint whose EP gather dropped 2D `gate_up_proj_bias`/`down_proj_bias`, or exported fused experts vLLM can't read | — | The EP gather keeps the biases (`src/distributed/expert_parallel/expert_weights.py`). Repair an old checkpoint from source biases, or re-emit the family's per-expert layout with `scripts/after_training/unfuse_moe_experts.py` |
| Hang at a collective, GPUs idle | Collective mismatch — one rank never reaches it (different branch, dataloader stall, shape divergence) | Stack dump (§2.2) → find the rank NOT in a collective; or consistency check (§2.1) | Make all ranks issue the same collectives; fix the divergent branch / shape |
| NCCL watchdog timeout on a legit-slow collective | Big cross-node all-to-all or gathered checkpoint save exceeds the 30-min default | `DIST_NCCL_TIMEOUT_MINUTES` (raise); flight recorder (§2.3) to see which collective | Raise the timeout; or reduce the collective (node-local EP, sharded save) |
| One rank consistently slow → all collectives wait | Straggler / rank skew | `profiler_ranks: "all"` + the TraceLens collective report (§4) | Address the slow rank (data imbalance, CPU contention, bad GPU) |
| GPUs idle between steps, CPU busy | Dataloader / tokenization / collation host overhead | `record_distributed_flamegraph(duration=30)` py-spy SVG (§2.4) | Raise `dataloader_num_workers`; pre-process dataset offline |
| CUDA OOM mid-step | Activation/state memory over budget | Memory snapshot (§3) → memory_viz allocation flame graph | GC on (mind Zaya/EP-GC limits above); lower batch / `max_length`; CP or TP to cut DP |

## 2. Enabling the `debugging.py` helpers

All helpers are opt-in and zero-cost when their env var is unset. The accepted
truthy values are `1`, `true`, `yes`, `on` (`env_flag`, `src/env.py`); a set-but-empty
value counts as unset and yields the default.

### 2.1 Cross-rank consistency — `assert_consistent` / `assert_tensor_shape_consistent`
Catches a shape/value divergence as a clear error instead of a downstream NCCL hang.
```bash
export HALO_TP_CONSISTENCY_CHECK=1   # makes assert_consistent RAISE on mismatch (else just warns)
```
```python
from src.diagnostics.debugging import assert_tensor_shape_consistent, assert_consistent
assert_tensor_shape_consistent(hidden_states, group=tp_group, label="attn_in")
assert_consistent(some_scalar, group=ep_group, label="n_tokens")
```
`group=None` or world≤1 is a no-op. Pass `strict=True`/`False` to override the env var per call.

### 2.2 Stack dump on hang — `py_spy_diag.py dump`
Attaches py-spy out of process, so it works on a job that is already hung — nothing to enable at
launch. Attach from the job's pid namespace (`docker exec` into the training container first).
```bash
python scripts/profiling/py_spy_diag.py dump              # every torchrun rank on this node
python scripts/profiling/py_spy_diag.py dump --pid 1234   # explicit pid, repeatable
```
Output: a timestamped `$TMPDIR/halo_diag_stacks/<ts>-rank00/pid<pid>.txt` per pid (`--output-dir` to
change; it walks torchrun's python children, so one invocation dumps the whole job). The rank NOT
inside a collective is the culprit. Requires `py-spy` on `PATH` (ships in the image's `profiling`
group) and `--cap-add=SYS_PTRACE` on the container (in the standard launch command), else py-spy
fails with `Permission denied`. In-script equivalent: `dump_distributed_stacks`.

### 2.3 NCCL watchdog + flight recorder
```bash
export DIST_NCCL_TIMEOUT_MINUTES=60          # watchdog window (default 30; applied by init_distributed; PyTorch's own default is 10)
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000    # ring buffer of recent collectives
export TORCH_NCCL_DUMP_ON_TIMEOUT=1          # dump the buffer when the watchdog fires → "which collective did rank K miss?"
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/mnt/halo_diag/nccl_trace
# transport/topology (RDMA not engaging, TCP fallback):
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
```

### 2.4 CPU flame graph — `py_spy_diag.py record` / `record_distributed_flamegraph`
For CPU-side stalls (dataloader/tokenize/collation), NOT for hangs (use §2.2 for those).
```bash
python scripts/profiling/py_spy_diag.py record --duration 30   # every torchrun rank → $TMPDIR/halo_diag_flamegraphs/<ts>-rank00/pid<pid>.svg
```
In-script equivalent:
```python
from src.diagnostics.debugging import record_distributed_flamegraph
record_distributed_flamegraph(duration=30)                  # this rank only
record_distributed_flamegraph(duration=30, this_rank_only=False)  # every rank on node (spot a straggler)
```
py-spy ships in the image (`profiling` dependency group); attach needs `--cap-add=SYS_PTRACE`.

## 3. GPU memory profiling (OOM)
Wrap the OOMing region or enable alongside the profiler, then drop the `.pickle`
on <https://pytorch.org/memory_viz> for a per-allocation flame graph:
```yaml
enable_torch_profiler: true
profiler_record_memory_snapshot: true  # dumps mem-<label>-rankNN.pickle over the active window
```
```python
from src.diagnostics.profiling import cuda_memory_history, log_cuda_memory, reset_peak_memory_stats
with cuda_memory_history("/mnt/profiling/oom", ranks="all"):
    trainer.train()                  # → /mnt/profiling/oom/snapshot-rankNN.pickle
reset_peak_memory_stats(); log_cuda_memory("after forward")   # quick textual allocated/reserved/peak
```

## 4. Reading `TorchProfilerCallback` traces
Enable from any training YAML (wired in `wiring.build_perf_callbacks` when
`enable_torch_profiler: true`):
```yaml
enable_torch_profiler: true
profiler_output_dir: /mnt/profiling/torch   # default: $HALO_DATA_ROOT/profiling/torch
profiler_wait: 5        # skip first 5 steps (warm caches)
profiler_warmup: 1      # 1 warmup step (on, not recording)
profiler_active: 3      # record 3 steps
profiler_ranks: "0"     # "0" | "all" | "0,8" — which global ranks profile
profiler_record_memory_snapshot: false
```
Artifacts per selected rank, with a `-cycleN` suffix per wait→active cycle:

| Artifact | What it is | How to read |
|----------|-----------|-------------|
| `<label>-cycleN-rankNN.trace.json.gz` | Chrome trace (CPU+CUDA timeline) | Perfetto / `chrome://tracing`. Gaps with NCCL kernels → comm-bound; CPU busy + GPU idle → input-bound; back-to-back GEMM/attn → compute-bound |
| `<label>-cycleN-rankNN.stacks_cuda.txt` / `.stacks_cpu.txt` | Collapsed stacks (`with_stack`) — torch's exporter comes back empty on the image's torch, and the empty file is dropped | Python frames are in the trace itself (`python_function` events); for a collapsed-stack flame graph use py-spy (§2.4) |
| `<label>-cycleN-rankNN.top_ops.txt` | Top 25 ops by self-CUDA time | quick eyeball of the dominant kernel |
| `<label>-cycleN-rankNN.memory_timeline.html` | Allocation timeline (needs `profile_memory`) | open in browser |
| `mem-<label>-rankNN.pickle` | CUDA memory snapshot (`memory_snapshot=True`) | memory_viz (§3) |

Traces carry named ranges for the toolkit's own phases — `ep.dispatch` / `ep.expert_compute` /
`ep.combine` per EP layer and `grad_sync.reduce_bucketed` for the post-backward sweep — so
comm-vs-compute attribution reads off the timeline directly. For automated analysis run
`python scripts/profiling/trace_report.py` (TraceLens): per-rank workbooks (gpu_timeline
compute/comm/idle, op tables, roofline) + a multi-rank collective latency/bandwidth/skew
report for `ranks: "all"` captures.

Notes from the callback source:
- Default profiles **only rank 0**; set `ranks: "all"` to catch rank skew/stragglers.
- On a large multi-GPU MoE, `ProfilerActivity.CPU` per-op tracing starves the GPUs
  (first step crawls). Construct `TorchProfilerCallback(cpu_activity=False)` for a
  CUDA-kernel-only trace (top-ops still resolves; you lose host-side op attribution).
- Window is intentionally small and one-shot (`repeat=1`) — profiling adds real overhead.

For MFU / TPS / peak-memory per step (not a trace), use `EfficiencyCallback`
(`enable_efficiency_metrics: true`). Production MoE MFU is routinely 5–12% (sparse
expert compute) — judge against the model class, not the dense ceiling.

## References
- `agent-docs/reference/debugging.md` — full user-facing guide (this playbook mirrors it)
- `src/diagnostics/debugging.py` — consistency-check + py-spy capture helpers (authoritative for env vars/signatures)
- `src/diagnostics/profiling.py` — CUDA-memory helpers (`log_cuda_memory`, `cuda_memory_history`, `reset_peak_memory_stats`)
- `src/callbacks/profiler.py` — `TorchProfilerCallback`
- `agent-docs/parallelism/multi-node.md` — NCCL/IB env vars, FS coordination
