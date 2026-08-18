# DeepEP Installation Guide

DeepEP provides the all-to-all GPU kernels (dispatch and combine) for MoE Expert Parallelism. EP has no
NCCL fallback, so DeepEP is required for it — without it `DeepEPDispatcher.__init__` raises an
`ImportError` pointing here. Every other training mode (SFT, SMPO, GRPO) works without it. The prebuilt
images already ship it; what follows is the from-source runbook and the runtime reference.

The images pin **DeepEP V2 at commit `af9a040`** (mirrored as the `vendors/DeepEP` submodule for
traceability; the image builds from a fresh clone, not from it), built with
`TORCH_CUDA_ARCH_LIST=9.0+PTX` on both. That
list governs only the ahead-of-time half — the `legacy` (V1) kernels compiled into `deep_ep._C`. The
`elastic` V2 dispatch/combine kernels ship as sources under `deep_ep/include/deep_ep/impls/*.cuh` and
DeepEP compiles them at runtime for the live device (`--gpu-architecture=sm_<device>`), which is why the
image keeps the full CUDA toolkit and why the arch list needs no Blackwell entry.

V2 unifies EP under `deep_ep.ElasticBuffer` over the **NCCL Gin** backend (GPU-Initiated Networking /
RDMA) for cross-node scale-out, with a non-Gin NVLink path intra-node. The legacy `deep_ep.Buffer` (CUDA
IPC over NVLink) stays selectable for intranode EP via `ep_buffer_backend`.

## Prerequisites

- NVLink for intranode communication: Blackwell (B200/B300, GB200/GB300) or Hopper (H100/H200).
- RDMA is required for cross-node EP via the Gin backend (`DeepEPDispatcher.is_inter_node`, set when
  `num_nodes > 1` and `node_local=False`); intra-node runs the non-Gin path and needs no RDMA.
- Python 3.12 (`requires-python = ">=3.12,<3.13"`); PyTorch 2.11+ (`pyproject.toml` pins `torch>=2.11.0,<2.12.0`); CUDA 12.3+ (image 13.2);
  ninja (nvcc device linking under `-rdc=true`).
- **NCCL 2.30.4+** — the Gin backend's floor; `uv.lock` pins `2.31.2` and every image shares it. `torch 2.11+cu130` ships `nvidia-nccl-cu13==2.28.9`.

## Build from source

```bash
# 1. NCCL. Install uv.lock's EXACT pin, the version every image shares — a floor lets two images
#    resolve different wheels and the RL weight-sync ncclCommInitRank then fails on the skew.
#    Upgrading the pip wheel moves torch's own collectives and DeepEP together with no
#    bundled-NCCL conflict — torch's libtorch_cuda.so links that same nvidia/nccl/lib/libnccl.so.2.
#    (torch.cuda.nccl.version() still prints the 2.28.9 compile-time constant; the runtime lib is upgraded.)
pip install "nvidia-nccl-cu13==$(python docker/nccl_pin.py uv.lock)" --no-deps
pip install ninja

# 2. Build DeepEP against the PyTorch in this environment.
git clone https://github.com/deepseek-ai/DeepEP.git && cd DeepEP && git checkout af9a040
export TORCH_CUDA_ARCH_LIST="9.0+PTX"
export CPATH=/usr/local/cuda/targets/x86_64-linux/include/cccl:${CPATH}   # CUDA 13 moved the CCCL headers
PYBIN=$(python -c "import sys; print(sys.prefix + '/bin')")               # ninja must be on PATH for nvcc
export PATH="$PYBIN:/usr/local/bin:/usr/bin:/usr/local/cuda/bin:$PATH"
pip install --no-build-isolation .

# 3. Verify.
python -c "import deep_ep; print(deep_ep.__version__, deep_ep.ElasticBuffer, deep_ep.topk_idx_t)"
```

**NVSHMEM** is needed for DeepEP's device linking (and the legacy `deep_ep.Buffer` path); V2 internode uses
Gin, not NVSHMEM. With PyTorch 2.11+cu130 it arrives transitively as `nvidia-nvshmem-cu13` — **do not**
`pip install nvidia-nvshmem-cu12` on top, which clobbers the cu13 headers and breaks device-side linking
against CUDA 13's `nvshmem_device.a`. On older CUDA 12 setups install `nvidia-nvshmem-cu12` explicitly.

A PyTorch upgrade makes the compiled `deep_ep/_C*.so` ABI-incompatible (undefined-symbol errors). Clean
(`rm -rf build/ *.egg-info deep_ep/_C*.so` plus any `site-packages/deep_ep*`) and rebuild with the same
exports.

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EP_DISABLE_GIN` | `1` disables the Gin (RDMA) backend → non-Gin NVLink path. The dispatcher sets it from EP topology (`1` intra-node, `0` inter-node); an explicit value is honored and logged. | dispatcher-set |
| `EP_SUPPRESS_NCCL_CHECK` | Suppress DeepEP's duplicate-NCCL-runtime guard (the NGC image's HPC-X `libnccl-net` transport *plugin* trips it; complementary, not a conflicting runtime). Must be in the **process** environment — DeepEP reads it inside `check_nccl_so()` at `import deep_ep`, so a Python write is too late; the dispatcher warns whenever the value it sees is not `1`. | `1` (image `ENV`, both images) |
| `CUDA_DEVICE_MAX_CONNECTIONS` | Hardware work-queue count. `1` is DeepEP's free default (measured below), latched at `cuInit` — a launch outside the image exports it before the process starts. The toolkit only warns on another value; it does not make the racy single-domain multi-group shape safe (below). | `1` (image ENV) |
| `HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK` | Cross-node (Gin) dispatch ceiling in tokens/rank, checked against the all-reduced capacity at buffer sizing; a larger dispatch wedges instead of erroring ([AWS EFA](#expert-parallelism-over-aws-efa)). `0` disables. | `8192` |
| `HALO_DEEPEP_NUM_SMS` | Pin the dispatch/combine SM count (else auto from `get_theoretical_num_sms`). Applies to both backends; legacy requires an even count. | auto |
| `HALO_DEEPEP_NUM_QPS` | Override the RDMA queue-pair count for dispatch **and** combine (also sets the buffer's allocation). **elastic only** — V1 takes no per-call QP count. On non-IBGDA fabrics (EFA proxy Gin) more QPs can raise the latency-bound internode all-to-all parallelism — A/B it. | auto (`0`) |
| `HALO_DEEPEP_GPU_TIMEOUT_SECONDS` | Device-side spin budget for the dispatch/combine barrier (below). **elastic only** — the V1 buffer's ctor takes no timeout, so a value set under `legacy` is ignored with a warning. | `100` |
| `HALO_EP_SHARED_OVERLAP` | `1` runs the always-active shared-expert FFN on a side stream concurrent with the routed dispatch all-to-all (shared-expert families: Qwen3.5/3.6, GLM4 MoE Lite/Laguna, Bailing, Mistral4, DeepSeek-V4, Inkling, Cohere2 MoE). | off |
| `HALO_EP_CAPACITY_DEDUP` | `0` restores the per-MoE-layer buffer-capacity all-reduce, and with it a private arena per layer. | `1` (on) |

**These must agree across every rank of the job**: `HALO_EP_CAPACITY_DEDUP`,
`HALO_DEEPEP_GPU_TIMEOUT_SECONDS`, `HALO_DEEPEP_NUM_SMS`, `HALO_DEEPEP_NUM_QPS`,
`HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK`, `HALO_GRAD_BUCKET_MB`, `DIST_NCCL_TIMEOUT_MINUTES`,
`DIST_STORE_TIMEOUT_HOURS`, `EP_DISABLE_GIN`.

Dedup changes how many collectives a rank runs. The Gin ceiling gates a raise against an all-reduced
(rank-uniform) capacity. The grad-bucket size sets the chunk boundaries of the bucketed gradient
reductions (EP, TP and QLoRA sweeps alike). The two coordination timeouts decide which rank gives up
on a join first: a node left at the default aborts its process group while its peers still hold
budget, and the survivors then die on the *next* collective and blame it. The rest are wire
parameters both ends of the all-to-all share.

`verify_rank_uniform_env` gathers them in distributed setup — **before the weight load**, so a mismatch
fails at startup rather than at the first EP collective — and refuses a job whose ranks disagree
(`Rank-uniform toolkit environment differs across ranks`, naming rank 0's values). It compares the
**resolved** settings, not the raw strings, so a knob left unset and the same knob exported at its own
default are one behavior: the ordinary per-node `--env-file` that sets a default on the head node only
is not a divergence. `EP_DISABLE_GIN` is DeepEP-owned and read by DeepEP as a raw string, so it is
compared as one.

The NCCL watchdog (`DIST_NCCL_TIMEOUT_MINUTES`, default 30) applies to every `dist.new_group()` the EP
config creates. It does **not** bound the DeepEP dispatch/combine barrier, which is a GPU spin-wait on
DeepEP's own budget — `HALO_DEEPEP_GPU_TIMEOUT_SECONDS`, default 100, long enough to absorb FA4's
first-use JIT compile and transient host stalls.

**That budget bounds rank skew, not idle time.** The device-side spin starts its clock when a rank
*enters* the barrier, so a gap between steps is invisible to it — no dispatch kernel is resident,
nothing is counting. What it bounds is how far behind a peer may be when this rank arrives. A
single-writer checkpoint is the case that can exceed it at scale, which is why every save path fences
with a host-side barrier first: the peers then wait inside a process-group collective under the much
larger `DIST_NCCL_TIMEOUT_MINUTES`, not inside the DeepEP spin. Raise
`HALO_DEEPEP_GPU_TIMEOUT_SECONDS` only for a legitimately skewed phase that outlasts it — raising it
also delays how fast a genuine hang surfaces.

## Buffer sizing

`ElasticBuffer` is sized from the **global max** per-rank token count, so the dispatcher all-reduces that max
before it builds or grows the buffer — under a padding-free or variable-length batch the per-rank counts
differ. Every MoE layer in one forward sees the same count, so the reduce runs **once per forward**: the
first MoE layer decides the capacity (rounded up to a 256-token alignment) and the rest reuse it,
dropping N−1 collectives and their `.item()` device syncs per forward. A forward-pre-hook on the
EP-patched model invalidates the cache, so the first-layer miss lands on every rank together. Sizing is
grow-only.

**One buffer per EP group, on both backends.** Every `EPMoELayerBase` dispatching the same shape on
the same group shares one arena, so the cost is paid once for the model rather than once per layer —
`_ElasticArena` keyed on group, padded hidden, top-k and QP count; `_LegacyArena` on group and buffer
bytes. V1's buffer is token-count-independent (measured: identical bytes at 1k and 8k tokens/rank) and
flat at ~100 MiB per layer at `hidden=4096`/`ep4`, which is 5.9 GiB across 60 layers unshared.

Sharing is safe because a dispatch copies its results into allocator-owned tensors and the handle it
returns carries the routing layout rather than buffer state, so a later layer's dispatch does not
disturb an earlier layer's pending backward. Bit-exactness against per-layer arenas is pinned on both
backends by `tests/gpu/parallelism/ep/test_ep_shared_arena.py`.

The elastic arena **grows with `ep_size`**: it is a receive-side all-to-all region sized for every peer
sending its whole batch, so it is linear in the dispatch-group width rather than divided by it.

```text
bytes ≈ ep_size × tokens_per_rank × (2·hidden + 4·top_k)
```

At `hidden=4096`, `top_k=10`: 0.51 GiB at `ep8`/8192 tokens per rank, 4.06 GiB at `ep64`/8192 —
exactly 2.0× per doubling of `ep_size` on B300. That law is **within-domain**: DeepEP sizes the arena
from `num_scaleout_ranks × num_scaleup_ranks`, so widening a group inside one NVLink domain scales
every term, while widening it by adding nodes grows only the scale-out factor and the growth is
sub-linear. Treat the formula as an upper bound cross-node and as the real number for a single-domain
group (an NVL72 `ep64` is single-domain). See
[Large-scale scenarios](../parallelism/large-scale-scenarios.md#per-rank-memory-model).

**Freeing a buffer is a collective**, so it never happens from a Python finalizer — a rank that enters
the NVLink barrier alone on its garbage collector's schedule burns the whole
`HALO_DEEPEP_GPU_TIMEOUT_SECONDS` budget and aborts the job with `cudaErrorLaunchFailure`. A dropped
dispatcher only relinquishes its claim; `free_unclaimed_ep_buffers()` sweeps up the remainder at the
explicit teardown (`cleanup_ep`, `destroy_all_dispatchers`), where every rank is aligned
(`tests/gpu/parallelism/ep/test_ep_buffer_gc_safety.py`).

**Budget the retained generations, not just the live arena.** Grow-only means the capacity latches to
the largest count any rank of the group has ever presented, so the first oversized batch — or a larger
`per_device_eval_batch_size` at the first eval — raises it for the rest of the run. A grow *retires*
the previous arena rather than freeing it (handles held by in-flight pipeline microbatches may still
name it) and releases both at teardown.

Until teardown the group's footprint is therefore the **sum** of every capacity it has held: ~4 GiB per
generation at `ep64` with 8k tokens/rank, so three growths cost ~12 GiB of HBM for the run on a model
that is already HBM-bound. Nothing reclaims a retirement earlier, so avoid the growths instead — pack
the corpus, or let the first forward present the run's maximum tokens per rank — and watch the
`DeepEP arena grew … retained so far` line, which counts them (one per EP group, from that group's rank
0). Packed batches above batch 1 flatten to a row of variable width (real tokens only), so a later step
can set a new high-water mark; the capacity is the all-reduced MAX aligned to 256, so this costs at most
a handful of generations, bounded by `per_device_train_batch_size × max_length`.

One shape the dedup cannot size: a model whose **later** MoE layers dispatch more tokens/rank than its
first. No family in the roster does; if one did, the dispatcher raises rather than under-size the wire
buffer. The reuse guard compares each layer's local token count against the cached capacity, so the
raise is rank-local — the rank that outgrows the arena raises while its peers reach the dispatch
collective, and `HALO_DEEPEP_GPU_TIMEOUT_SECONDS` bounds that wait rather than hanging the job.
`HALO_EP_CAPACITY_DEDUP=0` gives every layer its own all-reduce and, since one capacity can then no
longer cover the whole forward, its own arena.

The same raise fires when the per-forward generation that scopes the cache stops advancing. A forward
pre-hook bumps it, so it must sit on the **outermost** module the loop calls: a pipeline stage registers
it on itself, and the trainer re-registers it after PEFT wrapping — a task-typed `PeftModel` reaches the
model it wraps through `.forward()`, which runs no pre-hook on it. A caller entering the backbone
directly is out of the hook's reach entirely (TRL's chunked log-prob path peels `base_model` off the
wrapper) and opens its own scope through `bump_forward_generation()`.

## EP grouping: what is reliable

`CUDA_DEVICE_MAX_CONNECTIONS=1` serializes device work onto one hardware queue, and it is **free** as a
default: neutral on dense and `ep_size=2`, **+9.7%** on `ep_size=8` (8×B300, gpt-oss-20b, seq 4096, GC on).

**It does not make single-domain multi-group >2-rank pure EP reliable.** `ep_size > 2` with
`ep_group_size < nvlink_domain_size` inside one NVLink domain is rejected at config time — the unit is
the domain, not the OS node, so on NVL72 with `NVLINK_DOMAIN_SIZE=72` that covers `ep8`. The groups'
combine barriers race FSDP2's DP-wide NCCL collectives, and the two transports fail differently on
8×B300: `legacy` (V1 `Buffer`) deadlocks around step 2, the `elastic` default (V2 over NCCL Gin) faults
with `CUDA error: Invalid access of peer GPU memory over nvlink`. Safe shapes, the `ep4+etp2` exemption
and the full mechanism:
[Expert Parallelism](../parallelism/expert-parallelism.md#single-domain-multi-group-ep-races-and-hangs).

**Multi-node multi-group EP is supported** — EP groups that span nodes and act as data-parallel replicas
(`num_ep_groups>1`, `num_nodes>1`, no expert-TP). Nothing races the combine there: every backward-time
collective stays **within** the EP group, and the cross-replica DP average is deferred to a
post-backward sweep. Node-local `ep8×2` and cross-node `ep8` (two replicas) both converge and match
single-group `ep16` on 2× `p6-b300.48xlarge`. Mechanism:
[Multi-Node](../parallelism/multi-node.md#deferred-cross-replica-sync).

`ep_size` must divide the expert count exactly: DeepEP dispatch assumes a uniform expert→rank
division. `ParallelismConfig.validate_against_model_config` raises off `config.json` at the top of
the model load; `EPConfig.finalize_expert_assignment` re-checks it once the EP groups exist.

## Transport backend {#transport-backend}

`ep_buffer_backend` selects the transport for the EP all-to-all. The dispatcher hides the choice behind one
interface (`_DeepEPBackend`), so it is transparent to the MoE layer and the autograd path.

| `ep_buffer_backend` | Buffer | Transport | Dispatch limit | Cross-node |
|---|---|---|---|---|
| `elastic` (= `auto`, default) | `deep_ep.ElasticBuffer` (V2) | NCCL Gin (RDMA cross-node; non-Gin NVLink intra-node) | intra-node: arbitrary length up to the 32-bit wire index (~175k tok/rank); cross-node Gin: [~8k tok/rank](#expert-parallelism-over-aws-efa) | **Yes** (AWS EFA) |
| `legacy` | `deep_ep.Buffer` (V1) | CUDA IPC P2P over NVLink | fixed-size chunked pipeline sized by hidden, streams arbitrary length | **No** (rejected at config time) |

The two are **numerically identical** (bit-identical loss + matching expert/router gradients on
gpt-oss-20b ep2) and throughput-comparable (legacy ≈ 1.04× elastic step time at ep2/seq4096). `auto`
resolves to `elastic`, which fits every topology; the backend is fixed for the run.

Pick explicit **`legacy`** (intranode / node-local) for long-context ep8 training: elastic ep8 at extreme
tok/rank (≥~64k·b1) fails at step 1, its combine barrier racing FSDP2's reduce-scatter into a deadlock.
Legacy's CUDA-IPC buffer has no such barrier and trains gpt-oss-20b ep8 at 64k. EP+CP is the other
long-context ep8 path.

**`legacy` is intranode-only.** `ParallelismConfig` rejects it at config time for any cross-node EP group
(`ep_scope=global` / `node_local=False` spanning NVLink domains), for `ep_size` above 8 or above
`gpus_per_node`, and for an `ep_size` outside DeepEP V1's tuned rank table (`DEEPEP_V1_CONFIG_RANKS`).
The dispatcher re-checks at backend selection and also rejects a node-local group spanning OS nodes: the
buffer is built with `num_rdma_bytes=0`, and V1's node-major rank layout does not match the column-block
cross-node layout.

## SM control

`ElasticBuffer` computes the optimal SM count analytically from the MoE shape via
`get_theoretical_num_sms(num_experts, num_topk)`; the dispatcher uses this by default and passes a fixed
count through only if one is given. When that analytic call divides by zero on inter-node Blackwell, the
dispatcher warns and falls back to 24 SMs. `HALO_DEEPEP_NUM_SMS=<n>` overrides it to A/B the comm/compute
SM split (the dispatch/combine kernels get `n` SMs, leaving the rest for the local expert GEMM). On
8×B300 qwen3.5-35b ep8 (seq 4096) a pinned `16`–`32` ran ~5% over the auto count; sweep per shape.

The pin applies to **both** backends, by different routes: elastic passes it per dispatch/combine call,
while V1 carries the SM count inside its per-rank-count `Config` tables, so the legacy backend applies it
via `Buffer.set_num_sms` before those tables are read. V1 splits SMs into send/recv channel pairs and
therefore requires an **even** count — an odd pin is rejected at buffer creation rather than asserted from
inside DeepEP.

`HALO_DEEPEP_NUM_QPS` is **elastic-only**. The pin is allocated on the buffer and passed to every dispatch **and** combine. Both halves are
needed: dispatch asserts `num_qps <= num_allocated_qps` against DeepEP's automatic allocation (17 queue
pairs, 65/129 in hybrid mode), and combine inherits only `num_sms` from the handle, recomputing its own
QP count otherwise.

## Dispatch is the EP bottleneck

The **dispatch** all-to-all dominates the EP step, more than its byte volume implies: it computes the
routing layout and exchanges per-expert recv counts across the group, so it is latency/sync-bound, not
bandwidth-bound. The combine reuses that layout via the `handle` and runs several times faster. The
count exchange earns its cost — it lets the local GEMM process only real tokens instead of a
capacity-padded buffer.

Levers: `HALO_EP_SHARED_OVERLAP=1` hides part of the dispatch behind the shared-expert FFN, and
`HALO_DEEPEP_NUM_SMS` tunes the SM split. Per-config breakdown:
[Throughput Benchmarks](../optimization/throughput-benchmarks.md#measured-bottleneck-case-study--gpt-oss-20b-ep-on-8-b300).

## Dispatch wire-index limit {#token-count-ceiling}

ElasticBuffer **forwards** arbitrary sequence length intra-node (gpt-oss-20b ep8 to 65536;
cross-node Gin has its own ~8k tokens/rank ceiling — [EFA](#expert-parallelism-over-aws-efa)). The one
guarded limit is DeepEP's 32-bit **wire index**: the kernels offset the per-rank wire buffer
(`num_max_tokens_per_rank × num_topk` rows of `padded_hidden`) with 32-bit indices, so
`reject_oversized_dispatch` (`DEEPEP_INDEX_LIMIT = 2³¹`) fails the buffer build loud rather than let
an extent at or above `2³¹` wrap and illegal-access. The boundary is
`2³¹ / (num_topk × padded_hidden)` ≈ **~175k tokens/rank** for gpt-oss-20b (topk 4, padded hidden
3072).

This bites only a **single very long sequence** (`per_device_train_batch_size × sequence_length` tokens
in one MoE forward), far beyond any training sequence. The buffer is per-rank, so EP size does not lower
it; at high `ep_size` the symmetric buffer OOMs at the build around the same point anyway. **Levers:**
`per_device_train_batch_size=1`, a shorter `max_length`, or in env-GRPO fewer generations / shorter
trajectories.

Both this ceiling and the cross-node Gin one are applied **at config time** as well, before any weight
is read: `ParallelismConfig.validate_against_model_config` sizes the run's declared budget
(`per_device_train_batch_size × max_length`, divided by `cp_size`) through the same
`ep_dispatch_capacity` alignment the dispatcher uses and refuses it there, naming the budget, the
capacity and the EP group; the dispatcher's check stays the backstop for the batch actually in hand.
Whether the Gin ceiling applies is decided by the NVLink **domain** — a rack-wide NVL72 group is not
bound by it.

There is **no lower "symmetric-window" ceiling.** What carries long-context ep8 is the int64 program
offset in the fused GptOss SwiGLU kernel (`src/kernels/fused_glu.py`): the grouped expert activation
crosses `2³¹` elements when a skewed router piles >745k tokens onto one rank (gpt-oss-20b ep8 at 64k),
and an int32 offset there illegal-accesses. That fault async-surfaces on peer ranks at DeepEP's
`symmetric.hpp:136` (`CUDA 700` / async `719`), which reads like a transport ceiling but is the kernel.
Hitting `symmetric.hpp` `719` on a long sequence means the activation kernel or an OOM, not the
all-to-all.

## How the toolkit drives it

The toolkit uses `DeepEPDispatcher` (`src/distributed/expert_parallel/dispatcher.py`), not the raw buffer. It
owns the `ElasticBuffer` lifetime, selects the Gin / non-Gin backend from topology, lazily sizes the buffer,
and pads hidden for the combine kernel. `dispatch` / `combine` are autograd-aware (`autograd.py`): the
dispatch backward is a combine and the combine backward is a dispatch that reuses the cached layout via the
`handle`.

```python
# Inside an EP MoE layer (EPMoELayerBase._dispatch_compute_combine); _gc_dispatch / _gc_combine are
# the checkpoint-replay wrappers around the dispatcher calls.
recv_x, recv_topk_idx, recv_topk_weights, handle = self._gc_dispatch(flat, experts, weights)
output = self._compute_experts(recv_x, recv_topk_idx, recv_topk_weights, input_dtype)
combined = self._gc_combine(output, recv_topk_weights, handle)
```

Per-expert received token counts (for grouped GEMM) come from `recv_topk_idx` by a stable sort on the
expert id, a `scatter_add_` histogram over the host-known `experts_per_rank`, then `cumsum` into the
grouped-GEMM `offs` — sync-free, where `torch.unique_consecutive` would force a device read-back per
MoE layer.

`destroy_all_dispatchers()` must run **before** `dist.destroy_process_group()` — Gin frees the symmetric heap
through the group communicator, and the reverse order raises `cudaErrorIllegalAddress`.

### Hidden padding (`hidden % 256`)

V2's TMA-vectorized combine kernel processes hidden in warp-cooperative int4 tiles and requires
`hidden % 256 == 0`. The dispatcher zero-pads the token feature dim up to the next multiple of 256 on the
wire (GPT-OSS 2880 → 3072) and slices it back before results re-enter the autograd graph, symmetrically
across forward and backward so gradients are exact. Padding is confined to the elastic backend's own
wire buffers — the MoE layer and expert compute see the real hidden — and is a no-op for conforming models (Qwen3 MoE, hidden
4096/2048). Cost is ~`padded/hidden − 1` extra transport bandwidth (≈6.7% for GPT-OSS). The legacy backend
needs no padding.

## Expert parallelism over AWS EFA

Inter-node EP runs the Gin backend over RDMA, and DeepEP asserts `props.ginType != NCCL_GIN_TYPE_NONE` at the
first dispatch. NCCL sources Gin from a GIN plugin — a `libnccl-gin.so` exporting `ncclGinPlugin_v<N>` (the
image builds `v13`, which the pinned NCCL still accepts). On Mellanox IB, Gin can be device-initiated
(IBGDA); **AWS EFA has no IBGDA**, so it uses
host-driven **proxy Gin** (`NCCL_GIN_TYPE=2`), copying completions to GPU memory via **GDRCopy**.

| Need | Provided by | Detail |
|------|-------------|--------|
| GIN plugin | **image** | The Dockerfile builds a GIN-capable `aws-ofi-nccl` (exports `ncclGinPlugin_v13`) over the NGC-bundled 1.17.3 (which exports no `ncclGin`) and exposes it as `libnccl-gin.so`. NCCL ≥ 2.30.4 is the GIN-API floor. |
| GDRCopy ≥ 2.5 | **image + host** | `libgdrapi` is built into the image; the `gdrdrv` kernel module + `/dev/gdrdrv` come from the host. Pass `--device /dev/gdrdrv`. |
| Launch env | **per job** | `NCCL_NET_PLUGIN=ofi`, `FI_PROVIDER=efa`, `FI_EFA_USE_DEVICE_RDMA=1`, `NCCL_PROTO=simple`, `NCCL_GIN_TYPE=2` — the same block as [Launch Recipes](../parallelism/launch-recipes.md#environment-variables) plus the proxy-GIN type. |

Without these, NCCL reports `globalGinSupport 0` / `GIN/Plugin: Failed to initialize any GIN plugin` and
DeepEP aborts with `NCCL GIN is unavailable`; `Failed to initialize GDRCopy` means the host is missing
`/dev/gdrdrv`. The node's Mellanox `mlx5` devices are not a cross-node NCCL path on p6 (NCCL falls back to
TCP socket, ~18 GB/s) — EFA is the fabric.

Cross-node EP over EFA is bound by **proxy-Gin per-operation latency, not bandwidth**: the dispatch's
layout-exchange round-trips through a CPU proxy, and the cost scales with the number of cross-node
operations, far below the wire ceiling (NCCL all-to-all tops near 173 GB/s busbw and all-reduce near
767 GB/s on 2× `p6-b300.48xlarge`, yet a measured per-step dispatch sits ~100× over what that bandwidth
alone predicts). Three consequences:

- **Fewer cross-node partners per rank → faster.** On a 2-node 8×B300 EFA test, dropping gpt-oss-20b from
  EP=16 (8 cross-node partners/rank) to **EP=2 global** (1 partner) raised throughput ~2.4× (≈510 → ≈1210
  tok/s/GPU), at 16 experts/GPU. Combine EP=2 with `expert_tp_size` to shard those experts back down — see
  [Expert Tensor Parallelism](../parallelism/expert-tensor-parallelism.md#process-groups-epetp-combo).
- **At most ~8k tokens/rank per dispatch.** Above that a proxy-GIN dispatch **wedges instead of erroring**:
  the receive counts never arrive, so the run dies in DeepEP's CPU wait (`Dispatch CPU wait ... received
  count 0`) or in a GPU spin-wait (Xid 109 `CTX SWITCH TIMEOUT`, cascading into Xid 43 and surfacing as an
  async `unspecified launch failure` in whatever kernel is on-stream). Measured on 2× 8×B300: 8,192
  tokens/rank trains, 16,384 hangs — on every node pair, at ep8 and ep16, `num_sms` 24 and 48. The
  dispatcher rejects an oversized cross-node dispatch at buffer sizing
  (`HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK`, default 8192, `0` disables); intra-node NVLink dispatch is
  unaffected to 65k tokens/rank.
- **A lower-latency fabric removes the wall.** Mellanox IB with IBGDA, or rack-wide NVLink (GB200/GB300
  NVL72, where cross-node EP rides NVLink), collapse the dispatch latency that dominates here. The limit is
  EFA's proxy path, not cross-node EP itself.

Node-local FSDP on the same setup runs ~122,000 cluster tok/s, so reserve cross-node EP for a MoE too
large for one node.

## Troubleshooting

**Build: "ninja is required" / "gcc: No such file" / "Failed to preprocess host compiler properties."** nvcc
needs ninja and the host compiler in `PATH`; the interpreter's `bin` dir is not on the system `PATH` by
default. Use the full `PATH` export above.

### Runtime errors

**`undefined symbol: _ZN5torch9TypeErrorC1EPKcz`.** The `.so` was built against a different PyTorch. Rebuild
from source.

**`NVSHMEM_DIR is not specified`.** Install `nvidia-nvshmem-cu13` (CUDA 13.x) or `nvidia-nvshmem-cu12`
(12.x), or set `NVSHMEM_DIR`. With torch 2.11+cu130 the cu13 build is already transitive.

**`No device id is provided via init_process_group`.** A warning, not an error — the training launcher binds
the device eagerly via `device_id=`, so it does not appear in normal runs.

**`DeepEP timeout check failed` → `CUDA error: unspecified launch failure` (~step 2).** Either multi-group
>2-rank single-node pure EP (rejected at config time), or sustained severe host contention tripping the
100 s GPU barrier — avoid running EP training alongside a heavy concurrent `docker build` on the same node.

**Memory allocation failures.** The `ElasticBuffer` is sized automatically and grow-only. Reduce per-rank
tokens (lower batch / `max_length`, or raise `ep_size`) or free GPU memory. Fragmentation OOMs on
variable-shape packed runs respond to `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which
composes with the buffer on single-node runs.

**`CUDA error: an illegal memory access` (Xid 31 MMU fault) at a large `num_max_tokens_per_rank`.** The
32-bit wire-index limit, normally caught before the kernel faults. Keep `per_device_train_batch_size = 1` for
long trajectories, or shorten them. In env-GRPO the log-prob precompute is skipped when unneeded and
otherwise chunked to `per_device_train_batch_size`, so it does not dispatch the whole
`grad_accum × num_generations` batch at once.

The DeepEP-side constraint under EP+ETP (experimental): the EP group splits into `expert_tp_size`
dispatch groups coupled by the strided expert-TP all-reduce, and that reduce must stay in **token
space** (`SumGradAcrossGroup` before dispatch, `ReduceFromExpertTP` after combine). A cross-group
all-reduce placed *between* dispatch and combine deadlocks the coupled groups under FSDP2's
multi-stream execution. Configuration:
[Expert Parallelism](../parallelism/expert-parallelism.md) ·
[Expert-Tensor Parallelism](../parallelism/expert-tensor-parallelism.md).
