# DeepEP

DeepEP provides the all-to-all GPU kernels (dispatch and combine) for MoE Expert Parallelism. EP has no
NCCL fallback: without DeepEP, `DeepEPDispatcher.__init__` raises an `ImportError` pointing here. Every
other mode runs without it. The images ship it; this page is the from-source runbook and the runtime
reference.

The images pin **DeepEP V2 at commit `af9a040`**, built with `TORCH_CUDA_ARCH_LIST=9.0+PTX` on both. That
list governs only the ahead-of-time `legacy` (V1) kernels in `deep_ep._C`. The `elastic` V2
dispatch/combine kernels ship as sources (`deep_ep/include/deep_ep/impls/*.cuh`) and JIT-compile for
the live device, which is why the image keeps the full CUDA toolkit and the arch list needs no
Blackwell entry.

V2 runs EP through `deep_ep.ElasticBuffer`: the **NCCL Gin** backend (GPU-Initiated Networking over
RDMA) across NVLink domains, a non-Gin NVLink path within one. The legacy `deep_ep.Buffer` (CUDA IPC
over NVLink) stays selectable for intranode EP ([Transport backend](#transport-backend)).

## Prerequisites

- NVLink within a node: Blackwell (B200/B300) or Hopper (H100/H200).
- RDMA for an EP group that spans NVLink domains (`EPConfig.spans_domains`: more than one domain and
  `node_local=False`); an intranode group runs the non-Gin path and needs none.
- Python 3.12, PyTorch 2.11 (`torch>=2.11.0,<2.12.0`), CUDA 12.3+ (the image runs 13.2), ninja (nvcc
  device linking under `-rdc=true`).
- **NCCL 2.30.4+**, the Gin floor. `uv.lock` pins the exact version every image shares, overriding the
  `nvidia-nccl-cu13==2.28.9` that torch 2.11+cu130 declares.

## Build from source

```bash
# 1. NCCL at uv.lock's EXACT pin, the version every image shares: a floor lets two images resolve
#    different wheels, and the RL weight-sync ncclCommInitRank then fails on the skew. torch's
#    libtorch_cuda.so links the same nvidia/nccl/lib/libnccl.so.2, so torch moves with it
#    (torch.cuda.nccl.version() still prints its 2.28.9 compile-time constant).
pip install "nvidia-nccl-cu13==$(python docker/nccl_pin.py uv.lock)" --no-deps
pip install ninja

# 2. Build DeepEP against the PyTorch in this environment.
git clone https://github.com/deepseek-ai/DeepEP.git && cd DeepEP && git checkout af9a040
export TORCH_CUDA_ARCH_LIST="9.0+PTX"
export CPATH=/usr/local/cuda/targets/x86_64-linux/include/cccl:${CPATH}   # CUDA 13 moved the CCCL headers
PYBIN=$(python -c "import sys; print(sys.prefix + '/bin')")               # ninja must be on PATH for nvcc
export PATH="$PYBIN:/usr/local/bin:/usr/bin:/usr/local/cuda/bin:$PATH"
pip install --no-build-isolation .

# 3. Verify (needs a GPU: the import runs cuInit).
python -c "import deep_ep; print(deep_ep.__version__, deep_ep.ElasticBuffer, deep_ep.topk_idx_t)"
```

**NVSHMEM** is needed for DeepEP's device linking and the legacy buffer; V2 internode uses Gin. With
PyTorch 2.11+cu130 it arrives as `nvidia-nvshmem-cu13`. **Do not** install `nvidia-nvshmem-cu12` on top:
it clobbers the cu13 headers and breaks device linking against CUDA 13's `nvshmem_device.a`.

A PyTorch upgrade makes the compiled `deep_ep/_C*.so` ABI-incompatible (undefined-symbol errors): clean
`build/`, `*.egg-info`, `deep_ep/_C*.so` and any `site-packages/deep_ep*`, then rebuild with the same
exports.

## Environment variables

The DeepEP knobs (`EP_DISABLE_GIN`, `EP_SUPPRESS_NCCL_CHECK`, `CUDA_DEVICE_MAX_CONNECTIONS`,
`HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK`, `HALO_DEEPEP_NUM_SMS`, `HALO_DEEPEP_NUM_QPS`,
`HALO_DEEPEP_GPU_TIMEOUT_SECONDS`, `HALO_EP_SHARED_OVERLAP`, `HALO_EP_CAPACITY_DEDUP`) and their
defaults are cataloged in
[Environment variables](../reference/configuration-reference.md#environment-variables).

`CUDA_DEVICE_MAX_CONNECTIONS=1` (an image `ENV`) serializes device work onto one hardware queue. It is
the setting the EP suites are validated with, and the trainer warns at startup when it differs. No
supported shape is known to need it: on 8 GPUs, `ep_size=2` (four 2-rank dispatch groups) and
`ep4+etp2` (two 4-rank groups) each train 120 steps cleanly at `8` too (gpt-oss-20b, seq 4096, Halo 1.1.0). It
costs no throughput: against `8`, `ep_size=8` reads +1.5% and `ep_size=2` +0.2% (8× B300, gpt-oss-20b, seq
4096, batch 1, GC on, Halo 1.1.0). The driver latches it at `cuInit`, so a launch outside the image exports it
before the process starts. It does not make the racy single-domain multi-group shape safe
([EP grouping](#ep-grouping-what-is-reliable)).

**These must agree across every rank of the job**: `HALO_EP_CAPACITY_DEDUP`,
`HALO_DEEPEP_GPU_TIMEOUT_SECONDS`, `HALO_DEEPEP_NUM_SMS`, `HALO_DEEPEP_NUM_QPS`,
`HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK`, `HALO_GRAD_BUCKET_MB`, `HALO_CHECKPOINT_WRITE_THREADS`, `HALO_CHECKPOINT_WRITERS_PER_NODE`,
`DIST_NCCL_TIMEOUT_MINUTES`, `DIST_STORE_TIMEOUT_HOURS`, `DIST_FS_PROBE_TIMEOUT_SECONDS`, `EP_DISABLE_GIN`,
`FLASH_ATTENTION_DETERMINISTIC`, `HALO_FLEX_SLIDING`, `HALO_FP32_MATMUL_PRECISION`. A divergence is a hang
or a silent drift, not a slow run: the dedup changes how many
collectives a rank runs, the grad-bucket size sets the chunk boundaries of the bucketed gradient
reductions (EP, TP and QLoRA alike), the write-thread count decides whether a rank refuses a save its
peers have entered (a negative count raises on every rank once the ranks agree), the timeouts decide
which rank gives up on a join first (the survivors then die on the next collective and blame it) and how
long each rank looks for the filesystem probe's file, transformers' determinism switch picks a head_dim-256 model's attention backend (FA4 or its
fallback, [Flash Attention](../optimization/flash-attention.md#model-specific-handling)), the flex-sliding
switch an SDPA run's backend, the matmul precision whether a rank's fp32 matmuls round through TF32, and the
rest are wire parameters both ends of the all-to-all share.

`agree_rank_uniform_env` (`halo/distributed/rank_uniform_env.py`) gathers them after the device bind and
**before the weight load**, and `load_distributed_model` repeats it as a latched backstop; a mismatch
raises `Rank-uniform toolkit environment differs across ranks` with rank 0's values. It compares the
resolved settings, so a knob left unset and one exported at its default are the same; `EP_DISABLE_GIN`,
which DeepEP reads as a raw string, is compared as one with a blank export reading as unset, and
`FLASH_ATTENTION_DETERMINISTIC` by transformers' own rule (only `1` turns it on).

The NCCL watchdog (`DIST_NCCL_TIMEOUT_MINUTES`) applies to every group the EP config creates. It does
**not** bound the dispatch/combine barrier, a GPU spin-wait on DeepEP's own budget,
`HALO_DEEPEP_GPU_TIMEOUT_SECONDS` (default 100, long enough for FA4's first-use JIT and transient host
stalls). That budget **bounds rank skew, not idle time**: the clock starts when a rank enters the
barrier, so a gap between steps is invisible to it. A single-writer checkpoint can exceed it at scale,
so every save path and the teardown fence on a host-side collective first, under the much larger
watchdog. Raise the budget only for a legitimately skewed phase that outlasts it; a raise also delays
how fast a genuine hang surfaces.

## Buffer sizing

`ElasticBuffer` is sized from the **global max** per-rank token count, so the dispatcher all-reduces
that max before it builds or grows the buffer. Sizing is grow-only, and the capacity rounds up to 256
tokens.

Every MoE layer in one forward sees the same count, so with `HALO_EP_CAPACITY_DEDUP` on (the default)
the first MoE layer all-reduces the capacity and the rest reuse it, dropping N−1 collectives and their
`.item()` syncs per forward. A forward pre-hook on the outermost module the loop calls opens each
forward's scope (`capacity_scope`, a `CapacityScope` in
`halo/distributed/expert_parallel/deepep/capacity.py`), so the first-layer miss lands on every rank
together. The trainer re-registers the hook after PEFT wrapping: a task-typed `PeftModel` reaches the
model it wraps through `.forward()`, which runs no pre-hook. A caller entering the backbone directly
(the chunked log-prob path peels the hidden-state backbone off the wrapper) opens its own scope with
`bump_forward_generation()`.

A layer that dispatches more tokens than the cached capacity raises rather than under-size the wire
buffer: a model whose later MoE layers dispatch more than its first (no family in the roster does), or
a forward that never opened its own scope. The check is rank-local, so the rank that outgrows the arena
raises while its peers wait in the dispatch for `HALO_DEEPEP_GPU_TIMEOUT_SECONDS`.
`HALO_EP_CAPACITY_DEDUP=0` sizes every layer with its own all-reduce.

**One buffer per EP group, on both backends.** Every EP layer dispatching the same shape on the same
group shares one arena (under `HALO_EP_CAPACITY_DEDUP=0` each layer keeps its own elastic one): `_ElasticArena` is keyed on group, padded hidden, top-k, QP count and
deterministic mode, `_LegacyArena` on group and buffer bytes. Sharing is safe because a dispatch copies
its results into allocator-owned tensors and the handle carries the routing layout, not buffer state,
so a later layer leaves an earlier layer's pending backward intact (bit-exact against per-layer
arenas: `tests/gpu/parallelism/ep/test_ep_shared_arena.py`). Unshared, V1's buffer alone is ~100 MiB per
layer at `hidden=4096`/`ep4`, token-count-independent: 5.9 GiB across 60 layers.

The elastic arena **grows with `ep_size`**: it is a receive-side region sized for every peer sending
its whole batch.

```text
bytes ≈ ep_size × tokens_per_rank × (2·hidden + 4·top_k)
```

At `hidden=4096`, `top_k=10`: 0.51 GiB at `ep8` with 8192 tokens per rank, 4.06 GiB at `ep64`, exactly
2.0× per doubling on B300. The law holds within one NVLink domain: DeepEP sizes the arena from
`num_scaleout_ranks × num_scaleup_ranks`, so widening a group by adding nodes grows only the scale-out
factor, sub-linearly. Treat the formula as an upper bound cross-node and as the real number for a
single-domain group (an NVL72 `ep64` is single-domain;
[Large-scale scenarios](../parallelism/large-scale-scenarios.md#per-rank-memory-model)).

**Budget the retained generations.** The capacity latches to the largest count any rank of the group
has presented, so the first oversized batch, or a larger `per_device_eval_batch_size` at the first
eval, raises it for the rest of the run. A grow *retires* the previous arena rather than freeing it (an
in-flight microbatch's backward may still name it); the trainer frees the retired arenas once the
optimizer step's last backward is done, on every rank of the group. Within the step that grew, the group
holds the **sum** of the capacities it grew through (~4 GiB per generation at `ep64` with 8k
tokens/rank), and from the next step only the live arena. A build or grow takes the tokens/rank the forward
needs times 1.5 (`ARENA_HEADROOM`, a constant so every rank sizes alike), rounded up to 256 and held to the largest
aligned capacity both limits of the dispatch check allow (the 32-bit wire index; across NVLink domains the Gin
ceiling), so a rising high-water mark rebuilds the arena a few times per run, and an arena already at a ceiling takes
no headroom. The capacity dedup still caches the exact need, and the legacy arena is never grown. Each growth still
rebuilds the arena: pack the corpus, or let the first forward present the run's maximum tokens per rank. The
`DeepEP arena grew …` line (once per EP group, old and new capacity) marks each one;
`tests/gpu/profiling/bench_ep_arena_headroom.py` times dispatch and combine at an exact and a headroomed capacity. Packed batches above batch 1 flatten to a
row of variable width, so a later step can still set a new high-water mark, bounded by
`per_device_train_batch_size × max_length`.

**Freeing a buffer is a collective**, so it never happens from a Python finalizer: a rank that enters
the NVLink barrier alone on its garbage collector's schedule burns the whole
`HALO_DEEPEP_GPU_TIMEOUT_SECONDS` and aborts with `cudaErrorLaunchFailure`. A dropped dispatcher only
relinquishes its claim. `destroy_all_dispatchers()`
(`halo/distributed/expert_parallel/deepep/dispatcher.py`) barriers, drops every live claim and frees
every buffer in build order, claimed or not; every other free (the `free_unclaimed_ep_buffers()` sweep,
a layer's `destroy()` under `cleanup_ep`, a top-k or determinism change moving a layer to a new buffer)
happens only once no rank of the buffer's group claims it
(`tests/cpu/parallelism/test_ep_buffer_teardown_order.py`,
`tests/gpu/parallelism/ep/test_ep_buffer_gc_safety.py`). It must run **before**
`dist.destroy_process_group()`: Gin frees the symmetric heap through the group communicator, and the
reverse order raises `cudaErrorIllegalAddress`.

## EP grouping: what is reliable

Single-domain multi-group pure EP with `ep_size > 2` (`ep4` on one 8-GPU domain) is refused at config
time; the race, both transports' failures and the safe shapes:
[Expert Parallelism → Single-domain multi-group EP](../parallelism/expert-parallelism.md#single-domain-multi-group-ep-races-and-hangs).

## Transport backend

`ep_buffer_backend` selects the EP all-to-all transport, fixed for the run; the dispatcher hides it
behind one interface (`DeepEPBackend`, built by `build_ep_backend` in
`halo/distributed/expert_parallel/deepep/backends.py`).

| `ep_buffer_backend` | Buffer | Transport | Dispatch limit | Cross-node |
|---|---|---|---|---|
| `elastic` (= `auto`, default) | `deep_ep.ElasticBuffer` (V2) | NCCL Gin across domains; NVLink within one | intranode: up to the 32-bit wire index (~175k tok/rank); cross-node Gin: [~8k tok/rank](#expert-parallelism-over-aws-efa) | **Yes** (AWS EFA) |
| `legacy` | `deep_ep.Buffer` (V1) | CUDA IPC P2P over NVLink | fixed-size chunked pipeline sized by hidden, streams any length | **No** (rejected at config time) |

The two are numerically identical (bit-identical loss and matching expert/router gradients on
gpt-oss-20b ep2). Legacy runs 9,826 tok/s/GPU against elastic's 11,896 at ep8 (gpt-oss-20b, seq 4096, batch 1,
GC on, 8× B300, [1.2.0](../optimization/throughput-benchmarks.md#120-against-110));
`tests/gpu/profiling/bench_ep_buffer_backends.py` A/Bs the two.

Pick **`legacy`** for long-context ep8 within a node: on gpt-oss-20b, elastic ep8 at extreme tokens/rank
(≥~64k·b1) fails at step 1, its combine barrier racing FSDP2's reduce-scatter into a deadlock, while legacy's
CUDA-IPC buffer has no such barrier and trains ep8 at 64k. EP+CP is the other long-context ep8 path. Elastic ep8
ran Qwen3.6-35B at `max_length` 65,536, about 63,200 tokens per rank, with no fault over its 8 steps, at
[8,113 tok/s/GPU](../optimization/throughput-benchmarks.md#recipes-and-real-text). That row sits just under the
threshold, so it says nothing about elastic past ~64k tokens per rank.

`legacy` is intranode-only: `ParallelismConfig` rejects it for an EP group spanning NVLink domains, for
a dispatch group wider than 8 or than `gpus_per_node`, and for a width outside DeepEP V1's tuned rank
table (`DEEPEP_V1_CONFIG_RANKS`). The dispatcher re-applies the rules for a hand-built `EPConfig`; the
`gpus_per_node` bound is what refuses a node-local group spanning OS nodes (an NVL72 `ep8` over 4-GPU
trays), since the V1 buffer is built with `num_rdma_bytes=0`.

**Deterministic dispatch.** Under `full_determinism` the `elastic` buffer is built with
`deterministic=True`; a cross-domain group (hybrid kernels, no deterministic mode) is refused
([Expert Parallelism → Determinism](../parallelism/expert-parallelism.md#determinism)).

## SM control

`ElasticBuffer` computes the SM count from the MoE shape (`get_theoretical_num_sms(num_experts,
num_topk)`); when that divides by zero on an inter-node Blackwell topology the dispatcher warns and
uses 24. `HALO_DEEPEP_NUM_SMS=<n>` pins it to A/B the comm/compute split; sweep per shape.

The pin applies to both backends: elastic passes it per call, legacy through `Buffer.set_num_sms`
before V1 reads its per-rank `Config` tables. V1 splits SMs into send/recv channel pairs, so an **odd**
pin is rejected at buffer creation. `HALO_DEEPEP_NUM_QPS` is elastic-only (legacy warns): the pin is
allocated on the buffer and passed to every dispatch **and** combine, since dispatch asserts
`num_qps <= num_allocated_qps` (17 automatic, 65/129 in hybrid mode) and a combine inherits only
`num_sms` from the handle.

## Dispatch cost

The **dispatch** computes the routing layout and exchanges per-expert receive counts across the group. The
combine reuses that layout through the `handle`. The count exchange earns its cost: the local GEMM processes only
real tokens instead of a capacity-padded buffer. Within one NVLink domain the dispatch is 9–13% of the EP phases, and
at ep8 the combine's larger share is mostly the wait for the busiest rank
([measured split](../optimization/throughput-benchmarks.md#where-the-ep-steps-time-goes)). Levers:
`HALO_EP_SHARED_OVERLAP=1` hides part of the dispatch behind the shared-expert FFN (+2.1–2.5% on Qwen3.5-35B-A3B and
Qwen3.6-35B-A3B EP8 at 16k, with a small reduction-order shift in the gradients; it cost 4.6% on Qwen3.5-122B-A10B EP8
with activation offload at 64k), and
`HALO_DEEPEP_NUM_SMS` tunes the split.

## Dispatch wire-index limit

`ElasticBuffer` forwards arbitrary sequence length intranode (gpt-oss-20b ep8 to 65536); cross-node Gin
has its own ~8k tokens/rank ceiling ([EFA](#expert-parallelism-over-aws-efa)). The guarded limit is
DeepEP's 32-bit **wire index**: the kernels offset the per-rank wire buffer (`num_max_tokens_per_rank ×
num_topk` rows of `padded_hidden`) with 32-bit indices, so `reject_oversized_dispatch`
(`DEEPEP_INDEX_LIMIT = 2³¹`) refuses an extent that would wrap and illegal-access. The boundary is
`2³¹ / (num_topk × padded_hidden)`, ~175k tokens/rank for gpt-oss-20b (topk 4, padded hidden 3072). It
bites only a single very long sequence in one MoE forward; the buffer is per-rank, so a larger
`ep_size` does not lower it. Levers: `per_device_train_batch_size=1`, a shorter `max_length`, or fewer
generations / shorter trajectories in async GRPO.

Both ceilings are also applied before any weight is read: `reject_unsupported_model` sizes the run's
declared budget (`per_device_train_batch_size` × the row length, doubled for the trainers that forward
chosen and rejected together, divided by `cp_size`) through the dispatcher's own
capacity alignment and refuses it, naming the budget, the capacity and the EP group. The row length is
`max_length`; unbounded, `max_prompt_length + max_completion_length` where the config bounds both, else
the model's context window. Online GRPO takes `max_prompt_length` (script argument) plus `max_completion_length`;
async GRPO takes `max_train_row_tokens` alone, as the turn and episode budgets bound what is sampled, not the tool
and observation tokens a multi-turn row carries. Both also size `per_device_eval_batch_size` rows when an evaluation
runs. SDPG adds the byte length of its hint template's literal text (a bound on its tokens) to that row, since its teacher forwards prompt, hint and completion; the answer a template names has no config bound and is not counted. A row with no such bound is judged against the context window on a group spanning NVLink domains only, so
cross-node EP needs `max_prompt_length` or `max_train_row_tokens` at or under 8,192 tokens per row, while a
node-local run with no bound declares nothing. The dispatcher's check stays the backstop for the batch in hand.
`packed_microbatch_tokens` (either trainer) is a per-rank budget checked the same way at construction, and its planner
caps every bin at the same wire and Gin ceiling. Whether the Gin ceiling applies is
decided by the NVLink **domain**: a rack-wide NVL72 group is not bound by it.

There is no lower "symmetric-window" ceiling. What carries long-context ep8 is the int64 program offset
in the fused GptOss SwiGLU kernel (`halo/kernels/fused_glu.py`): a skewed router piling >745k tokens
onto one rank (gpt-oss-20b ep8 at 64k) takes the grouped activation past `2³¹` elements. An int32
offset there faults, and the fault surfaces on peer ranks at DeepEP's `symmetric.hpp:136` (`CUDA 700` /
async `719`), which reads like a transport ceiling: on a long sequence it means the activation kernel or
an OOM, not the all-to-all.

## How the toolkit drives it

The toolkit drives `DeepEPDispatcher` (`halo/distributed/expert_parallel/deepep/dispatcher.py`), not the
raw buffer: it claims a shared buffer through its backend, which picks the Gin or non-Gin path from the
topology and sizes it lazily. `dispatch` / `combine` are autograd-aware: the dispatch backward is a
combine, and the combine backward is a dispatch reusing the cached layout through the `handle`. Under
gradient checkpointing the recompute replays the original pass's cached results, since a second
dispatch would invalidate the handle the original backward holds. The replay trades memory for no extra
all-to-all: each MoE layer keeps its received tokens, their top-k ids and weights and its combined
output from the forward until its backward, several times the GC activation term
([sized per rank](../parallelism/large-scale-scenarios.md#per-rank-memory-model)).

```python
# Inside an EP MoE layer (EPMoELayerBase._dispatch_compute_combine); owner= keys the checkpoint-replay cache.
recv_x, recv_topk_idx, recv_topk_weights, handle, recv_counts = self.dispatcher.dispatch(
    flat, experts, weights, owner=self
)
output = self._compute_experts(recv_x, recv_topk_idx, recv_topk_weights, input_dtype, recv_counts=recv_counts)
combined = self.dispatcher.combine(output, recv_topk_weights, handle, owner=self)
```

At `ep_size > 1` the expert compute reads nothing back from the device to size its indices. The dispatch
already blocks the host until the receive counts land (`do_cpu_sync`, DeepEP's default), and both backends
hand them on as `recv_counts`, the (token, expert) slots each local expert received: V2 from
`EPHandle.num_recv_tokens_per_expert_list`, V1 from its fourth dispatch output. The GC replay caches them
with the dispatch. On the grouped path their sum sizes the routing sort's outputs, and its counting sort
takes the group offsets from its own histogram, so no count is copied to the device
([Grouped GEMM](../optimization/grouped-gemm.md)). The per-expert loop (`use_grouped_gemm: false`) selects
each local expert's slots with a `nonzero_static` of that expert's count. Counts that disagree with the
received ids where a path reads them (the sum on the grouped path, each count on the loop) trip a
device-side assert (`torch._assert_async`, checked before any slot is placed): on CUDA it surfaces as a
`device-side assert triggered` at the rank's next synchronizing call, and its peers then wait out DeepEP's
timeout. Besides each dispatch's own wait, the EP layer code reads back only the first MoE layer's capacity
all-reduce (`cap.item()`, once per forward; per layer under `HALO_EP_CAPACITY_DEDUP=0`) and, at
`ep_size == 1` (EP1, pure ETP), where no dispatch counts exist, the loop's `torch.where` per local expert.

### Hidden padding (`hidden % 256`)

V2's TMA-vectorized combine kernel requires `hidden % 256 == 0`. The elastic backend zero-pads the
feature dim to the next multiple of 256 on the wire (GPT-OSS 2880 → 3072) and slices it back as a
row-strided view before results re-enter autograd, symmetrically in forward and backward, so gradients
are exact and the MoE layer sees the real hidden. The view keeps its padded buffer alive, so the GC
replay cache pins received tensors at the padded width, `padded/hidden − 1` more (≈6.7% on GPT-OSS,
≈+0.5 GiB peak on gpt-oss-20b EP8; more on Gemma 4, 2816 → 3072), with the same extra transport
bandwidth. Conforming models (Qwen3 MoE) and the legacy backend need no padding.

## Expert parallelism over AWS EFA

Inter-node EP runs the Gin backend over RDMA, and DeepEP asserts `props.ginType != NCCL_GIN_TYPE_NONE`
at the first dispatch. NCCL sources Gin from a plugin, a `libnccl-gin.so` exporting
`ncclGinPlugin_v<N>` (the image builds `v13`). On Mellanox IB, Gin can be device-initiated (IBGDA);
**AWS EFA has no IBGDA**, so it uses host-driven **proxy Gin** (`NCCL_GIN_TYPE=2`), copying completions
to GPU memory through **GDRCopy**.

| Need | Provided by | Detail |
|------|-------------|--------|
| GIN plugin | **image** | `docker/efa/install_efa_userspace.sh` builds a GIN-capable `aws-ofi-nccl` (exports `ncclGinPlugin_v13`) into every image and exposes it as `libnccl-gin.so`; the NGC-bundled 1.17.3 exports no `ncclGin` |
| GDRCopy ≥ 2.5 | **image + host** | `libgdrapi` is in the training images; the `gdrdrv` kernel module and `/dev/gdrdrv` come from the host. Pass `--device /dev/gdrdrv` |
| Launch env | **per job** | `NCCL_GIN_TYPE=2` on top of the EFA env ([Multi-Node → RDMA fabrics](../parallelism/multi-node.md#rdma-fabrics)) |

Without these, NCCL reports `globalGinSupport 0` / `GIN/Plugin: Failed to initialize any GIN plugin`
and DeepEP aborts with `NCCL GIN is unavailable`; `Failed to initialize GDRCopy` means the host lacks
`/dev/gdrdrv`.

Cross-node EP over EFA is bound by **proxy-Gin per-operation latency, not bandwidth**: the dispatch's
layout exchange round-trips through a CPU proxy, and the cost scales with the number of cross-node
operations (on two 8-GPU B300 nodes NCCL all-to-all reaches ~173 GB/s busbw, yet a measured per-step
dispatch sits ~100× over what that bandwidth predicts). Three consequences:

- **Fewer cross-node partners per rank is faster.** On 2-node 8× B300 EFA, gpt-oss-20b at **EP=2
  global** (1 cross-node partner, 16 experts/GPU) ran ~2.4× the throughput of EP=16 (8 partners), at a
  batch shape the run did not record. `gptoss-20b-multinode-ep.yaml` at EP=16 (one 8,192-token row
  per rank, `gradient_accumulation_steps: 4`, loop path, `aux_loss`, no gradient checkpointing) runs
  3,636 tok/s/GPU at 8.97 s/step and a 55.4 GiB rank-0 allocated peak. The PP recipe's 15,715 is not
  comparable: its dispatch stays on NVLink, with grouped GEMM and `bias_update`, and the same arm at EP=16 with
  those two settings runs [4,115](../parallelism/pipeline-parallelism.md#measured-throughput). Combine EP=2 with
  `expert_tp_size` to shard those experts back down
  ([Expert Tensor Parallelism](../parallelism/expert-tensor-parallelism.md#process-groups-epetp-combo));
  the ETP partners share one batch, so tok/s/GPU falls as the ETP width grows.
- **At most ~8k tokens/rank per dispatch.** Above that a proxy-Gin dispatch **wedges instead of
  erroring**: on 2× 8× B300, 8,192 tokens/rank trains and 16,384 hangs, at ep8 and ep16, `num_sms` 24
  and 48. The receive counts never arrive, and the run dies in DeepEP's CPU wait (`Dispatch CPU wait ...
  received count 0`) or a GPU spin-wait (Xid 109 `CTX SWITCH TIMEOUT`, cascading into Xid 43 and an
  async `unspecified launch failure`). The dispatcher refuses a larger cross-node dispatch at buffer
  sizing (`HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK`, default 8192, `0` disables).
- **A lower-latency fabric removes the wall.** Mellanox IB with IBGDA, or rack-wide NVLink (GB200/GB300
  NVL72, where cross-node EP rides NVLink), collapses the dispatch latency; the limit is EFA's proxy
  path, not cross-node EP itself.

Reserve cross-node EP for a MoE too large for one node: the same recipe at `ep8` on one 8× B300 host, on
the same loop path and tokens per step (under `bias_update`), runs 14,402 tok/s/GPU against that 3,636.

## Troubleshooting

**Build: "ninja is required" / "gcc: No such file" / "Failed to preprocess host compiler properties."**
nvcc needs ninja and the host compiler on `PATH`; use the full `PATH` export above.

### Runtime errors

| Error | Cause → fix |
|---|---|
| `undefined symbol: _ZN5torch9TypeErrorC1EPKcz` | The `.so` was built against another PyTorch → rebuild from source |
| `NVSHMEM_DIR is not specified` | The cu13 NVSHMEM wheel is missing → `pip install nvidia-nvshmem-cu13` (torch 2.11+cu130 normally brings it) or set `NVSHMEM_DIR`; never add `nvidia-nvshmem-cu12` |
| `No device id is provided via init_process_group` | A warning; the launcher binds the device with `device_id=`, so it does not appear in normal runs |
| `DeepEP timeout check failed` → `CUDA error: unspecified launch failure` (~step 2) | Sustained severe host contention tripping the 100 s barrier → do not run EP training beside heavy CPU/IO load (a large build) on the same node |
| Memory allocation failures | The grow-only arena ([Buffer sizing](#buffer-sizing)) → fewer tokens per rank (batch, `max_length`) or a larger `ep_size`. Fragmentation OOMs on variable-shape packed runs respond to `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which costs throughput under EP: set it only to fit; the EP examples leave it off ([measured](../optimization/throughput-benchmarks.md#allocator-expandable-segments)) |
| `DeepEP receive counts disagree with the received expert ids` in a `device-side assert triggered` on one rank, its peers then timing out in DeepEP's barrier | The receive counts the dispatch returned do not match the expert ids it delivered ([How the toolkit drives it](#how-the-toolkit-drives-it)) → check that `deep_ep` is the image's build at `af9a040` ([Build from source](#build-from-source)); on that build, report it with the EP shape (`ep_size`, `ep_buffer_backend`) |
| `CUDA error: an illegal memory access` (Xid 31) at a large `num_max_tokens_per_rank` | The 32-bit wire-index limit, normally refused first → `per_device_train_batch_size: 1` for long trajectories, or shorten them. Async GRPO chunks its log-prob precompute to `per_device_train_batch_size`, so it never dispatches the whole `grad_accum × num_generations` batch at once |

Under EP+ETP the EP group splits into `expert_tp_size` dispatch groups coupled by the strided expert-TP
all-reduce, which must stay in **token space** (`SumGradAcrossGroup` before dispatch,
`SumAcrossGroup` after combine): a cross-group all-reduce between dispatch and combine deadlocks the
coupled groups under FSDP2's multi-stream execution
([Expert-Tensor Parallelism](../parallelism/expert-tensor-parallelism.md)).
