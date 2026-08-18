# GPU Training Theory

Every "should I use X?" reduces to one question: which bottleneck does X attack, and is that the one you have? The **roofline** ([§2](#2-the-roofline-arithmetic-intensity-and-the-ridge-point)) answers it; [§3](#3-the-four-bottlenecks-a-step-can-hit) names the four bottlenecks a step can hit. The [Optimization](../optimization/README.md) pages each describe one lever — this page describes the machine they pull on.

**How to read it.** §1–§2 build the one tool the rest of the page uses (the roofline); §3–§8 apply it on a single GPU; §9 adds GPUs and the communication wall; §10–§11 cover inference vs training and how to measure your own step. Read in order once; afterwards §3's table and the [rules of thumb](#rules-of-thumb) are the index.

## Vocabulary

- **FLOP** — one multiply or add. A matmul `(M×K) @ (K×N)` costs `2·M·N·K` FLOPs. A B300 does ~1.8×10¹⁵ bf16 FLOP/s (§1).
- **Precision / dtype** — bits per number: `fp32` (4 B), `bf16` (2 B), `fp8` (1 B), `fp4` (0.5 B). Fewer bits = less memory, faster tensor-core math, coarser accuracy. The toolkit trains in **bf16** by default.
- **HBM** — the GPU's main memory (288 GB on a B300), far from compute. **SRAM** (shared memory + registers) is the small fast on-SM scratchpad.
- **SM / warp / tensor core** — 148 **Streaming Multiprocessors** on a B300; threads run in lockstep groups of 32 (**warps**); **tensor cores** do a small matmul per instruction and carry almost all transformer FLOPs.
- **Occupancy** — fraction of an SM's warp capacity in use; high occupancy lets the GPU hide HBM latency by switching among resident warps (§1).
- **Kernel** — one GPU program launched from the CPU; issuing one costs a few microseconds of host time (§3).
- **Activations** — the intermediate tensors a forward produces. Autograd keeps the ones the backward needs, so they occupy HBM for the whole step and grow with `batch × seq`, not with parameter count (§8).
- **Gradient checkpointing (GC)** — keep few activations and recompute the rest in the backward: much less memory, tens of percent more wall-clock (§8).
- **Collective** — coordinated cross-GPU communication (e.g. gradient sum).
- **DP / EP / TP / CP / PP / ETP** — the axes a job splits along: data (FSDP2 here), expert, tensor, context, pipeline, and expert-tensor (TP applied inside the expert FFN). §9 prices each one as a communication pattern; [Parallelism](../parallelism/README.md) configures them.
- **MoE** — an FFN split into many **experts**; a **router** sends each token to its **top-k** (e.g. 4 of 128). Many total params, few active. Each expert processes only its routed tokens, so its per-expert token count `M` is small (§2).
- **MFU (Model FLOPs Utilization)** — fraction of bf16 peak a whole step achieves; where on the roofline the step landed. Definition and MoE caveats in [§11](#11-measuring-it-yourself).

---

## 1. The GPU is two machines: ALUs and memory

Every number below is measured on a Blackwell B300 (`NVIDIA B300 SXM6`, SM 10.3, 148 SMs, 288 GB HBM3e) — achieved, not spec peak, and wobbling ±10–20% run-to-run with boost clocks. A Hopper part lands at different absolutes; the *relationships* carry across. Reproduce them with the script in [§11](#11-measuring-it-yourself).

Two capabilities scale independently, and almost every performance question is which one you ran out of first.

1. **Arithmetic throughput** (tensor cores): **~1818 TFLOP/s** bf16 measured on a large square GEMM (B300). The die's architectural ceiling is 148 SMs x 2.032 GHz x 8192 FLOP/SM/clk = **2464 TFLOP/s**, so that measurement is 74% of what the silicon can physically issue — a normal large-GEMM cuBLAS efficiency. Any quoted bf16 figure above 2464 is impossible, whatever the benchmark claims.
2. **Memory bandwidth** (HBM ↔ chip): ~6.6 TB/s (B300, large device-to-device copy).

Arithmetic outruns memory ~275 to 1: the tensor cores can do ~275 FLOPs in the time HBM delivers one byte. A kernel reaches peak FLOP/s only if it does that much arithmetic per byte loaded. Otherwise it is memory-bound, however fast the tensor cores are. §2 turns this ratio into the page's central tool.

### Memory hierarchy

| Level | Size (order) | Bandwidth (order) | Role |
|---|---|---|---|
| **HBM (global)** | 100s of GB | single-digit TB/s | weights, activations, optimizer states |
| **L2 cache** | 10s of MB | ~10× HBM | shared across SMs |
| **Shared memory / SMEM** (+ Blackwell *tensor memory*) | ~100s KB/SM | ~10× L2 | the kernel's tile scratchpad |
| **Registers** | KB/thread | fastest | operands the ALU reads |

A fast kernel stages a tile up this hierarchy once, does as much math on it as possible in SMEM/registers, then writes back, amortizing the HBM trip. A slow kernel streams from HBM, touches once, writes back. Flash Attention, grouped GEMM, and fusion are all ways to do more work per HBM byte. A `nvidia-smi` util % is not a busy signal for the tensor cores ([§11](#11-measuring-it-yourself)).

### Latency hiding

Both ceilings above are *bandwidths*, and both hide a second quantity: latency, the time one HBM access takes (~hundreds of ns). A CPU fights it with caches and speculation; a GPU hides it with parallelism, keeping many warps resident per SM and switching to a ready one the cycle another stalls. The fraction of that capacity in use is **occupancy**.

A kernel therefore reaches its ceiling only once the problem is big enough to fill the machine: a per-expert M of 128 leaves too few warps to hide latency and lands below the roofline its AI alone predicts. The same shortfall resurfaces as §4's small-M ramp and §9's strong-scaling collapse.

---

## 2. The roofline — arithmetic intensity and the ridge point {#2-the-roofline-arithmetic-intensity-and-the-ridge-point}

Everything below runs on one operation: the matrix multiply (**GEMM**) inside every linear layer. Its three dimensions recur on every page, so fix them now.

![Anatomy of a GEMM: Y = X @ W. X is the input activation [M×K], W the weight [K×N], Y the output [M×N]. M is the shared row dimension of X and Y, the token count (batch×seq, or tokens routed per expert), the knob you turn. K and N are the weight's fixed input and output width. The weight W is read from HBM regardless of M, so a small M wastes that load](../assets/diagrams/gemm_anatomy.png){ .diagram-mid }

**M** is the row count, the tokens flowing through: `batch × seq` for a dense layer, the tokens routed to one expert for MoE. It is the **knob you turn** — batch, sequence length, packing, and parallel degree all move it. K and N are the weight's input and output width, fixed by the model. The weight `W` (`K × N` numbers) is read from HBM whatever M is, while only the activations `X` and `Y` grow with M. One weight load spreads over M token-rows, and that ratio is what the roofline measures.

Define a kernel's **arithmetic intensity**:

```text
AI = (useful FLOPs) / (bytes moved from HBM)     [FLOP/byte]
```

Plot achievable FLOP/s against AI: a diagonal (capped by `AI × bandwidth`) rising to a horizontal ceiling (peak FLOP/s). The corner is the **ridge point**:

```text
ridge AI = achievable_FLOPs / achievable_bandwidth ≈ 1.818e15 / 6.6e12 ≈ 275 FLOP/byte   (B300, bf16)
```

![Roofline plot on log-log axes: a diagonal memory ceiling (throughput = AI × 6.6 TB/s) rises to meet the flat compute ceiling (~1818 TFLOP/s measured, against a 2464 TFLOP/s architectural ceiling) at the ridge point, arithmetic intensity ≈ 275 FLOP/byte. Left of the ridge is memory-bound, where fine-grained MoE experts at small per-expert M sit; right is compute-bound, where large-M GEMM and dense MLP sit. Three measured expert-GEMM points (K=N=2880) are plotted: M=8192 approaches the roof at three quarters of it, M=512 clears the ridge but reaches only a third, and M=128 lands far below even the memory ceiling, too few tiles to fill the 148 SMs, the occupancy shortfall of §1](../assets/diagrams/roofline.png){ .diagram-mid }

- **AI below ridge → memory-bound.** Achieved FLOP/s = `AI × bandwidth`. Fix: move fewer bytes, or use faster HBM. More tensor cores won't help.
- **AI above ridge → compute-bound.** Fix: more or faster tensor cores, or lower precision. Faster HBM won't help.

### Worked example — why small per-expert M is slow

Sweeping `M` (tokens routed to one expert) at the gpt-oss square-expert shape `K=N=2880`, B300:

| M | TFLOP/s | AI | regime |
|---|---|---|---|
| 128 | 180 | 118 | below ridge → **weight-bandwidth-bound** |
| 512 | 548 | 378 | above ridge → compute-bound |
| 8192 | 1375 | 1225 | compute-bound |

The crossover at M≈256–512 is fixed by the shape. Absolute TFLOP/s vary ±10–20% with clocks, more so at the smallest M, whose ~12 µs runtime sits near the launch floor. At `M=128` the kernel hits ~180 TFLOP/s, about 10% of the 1818 TFLOP/s measured ceiling: ~92% of its bytes are the `2880×2880` weight (16 MB bf16), streamed in full while only 128 token-rows multiply, so the tensor cores finish and idle. Growing `M` amortizes that one weight load over more rows.

Per-expert `M` is set by routing:

```text
M ≈ (batch × seq × top_k) / num_experts
```

`batch × seq` is the token count the EP group dispatches — EP relocates experts across ranks, it does
not change how many tokens reach one expert.

A fine-grained MoE (many narrow experts, high top-k — Qwen3.6 has 256) puts few tokens on each one, deep in the bandwidth-bound regime. There the per-expert loop fires hundreds of tiny launches, so grouped GEMM's single batched launch wins big; it also collapses hundreds of autograd nodes, each pinning its own saved activations, into two. A coarse MoE (gpt-oss: 32–128 experts, `2880²`) gives each expert more tokens and bigger matmuls that a single cuBLAS call already handles, so the grouped win shrinks ([Grouped GEMM](../optimization/grouped-gemm.md)).

The cheapest M is the one you are already paying for. Padding spends it on nothing: a varlen Flash Attention kernel skips pad tokens (SDPA computes and masks them), but every pad row still flows through the projections and the MLP at full GEMM cost. Packing documents into full rows takes it back — 9.2× the real tokens/s on Qwen3-30B-A3B at `max_length` 4096 with ~1024-token documents (2× B300, EP=2) ([Padding-Free Collator](../optimization/padding-free-collator.md#benchmark-results)).

First question of any step: are its dominant kernels' M above the ridge? If not, the fix is **a bigger M** — longer packed sequences, a bigger batch, or a smaller parallel degree — not a faster kernel. It buys nothing once the step is compute-bound and saturated, or already at the memory limit.

---

## 3. The four bottlenecks a step can hit

A step is thousands of kernels, and at any moment one of four resources limits it. Each has a different fix. Profile first, then pull only the lever that attacks the bottleneck you have.

| Bottleneck | Signature | Lever |
|---|---|---|
| **Compute-bound** | tensor cores saturated, power near board limit | bigger dims, lower precision if it pays ([§7](#7-precision-low-precision-compute-and-bf16-numerics)) |
| **Memory-bound** | high util %, **low power** (~50–70%), AI below ridge | move fewer bytes: **fusion**, **bigger M**, **flash attention** |
| **Latency / launch-bound** | many tiny kernels, gaps between them | **batch** the work: grouped GEMM, CUDA graphs |
| **Communication-bound** | GPUs idle on NCCL, util drops in bursts | bigger M, smaller parallel degree, compute–comm overlap |

The launch floor is a few microseconds of host time per kernel — the sustained issue cost with framework dispatch included, not the bare driver call; [§11](#11-measuring-it-yourself)'s script measures it on your GPU. Run a MoE layer as a per-expert loop and that cost dominates. Qwen3-30B-A3B's `128 experts × 2 fused projections = 256` tiny matmuls take 1.33 ms in the forward against 0.11 ms for one grouped kernel (B300, 4k tokens) ([Grouped GEMM](../optimization/grouped-gemm.md)). Batching into one kernel matters at high expert count independent of the roofline.

### Anatomy of one step

1. **Load batch** — CPU dataloader plus PCIe copy. If the GPU outruns it, the step is input-bound (a fifth bottleneck, off the GPU; fix with prefetch/overlap, not a faster kernel).
2. **Forward (~`2P` FLOPs per token)** — matmuls (compute-bound at good M), attention (flash), and elementwise norms/activations (memory-bound). `P` is the parameter count (the standard `6N` rule, renamed so `N` stays the GEMM width of §2).
3. **Loss** — fp32 reduction.
4. **Backward (~`4P` FLOPs per token)** — two gradient GEMMs per layer (`dgrad` + `wgrad`); GC recomputes the forward first.
5. **Gradient sync** — cross-GPU collective, comm-bound, overlapped where possible.
6. **Optimizer step** — memory-bound elementwise update (AdamWBF16).

Forward plus backward dominate FLOPs (`6P`). The collective and optimizer are fixed per-step overheads that a bigger `M` amortizes.

---

## 4. How a matmul runs: tiles, kernels, launches

Tensor-core kernels break the output into **tiles**, commonly 128 rows along M on SM90/SM100. Each tile's warp group loads matching input strips into SRAM and multiplies there, and the hardware wants the problem to divide into whole tiles:

- **Alignment** — the MMA consumes fixed-size fragments, so the contraction dim `K` (and ideally `M`, `N`) should be multiples of 16 in bf16; a dim that isn't a multiple of 8 forces padding or a slower path. Round hidden sizes and padded vocabularies exist for this reason.
- **Tail effect** — M not a tile multiple leaves the last tile partly empty. At M=130 against a 128-tile, the second tile fills only 2 of 128 rows. This is pure geometry, and its throughput cost folds into the small-M penalty below.
- **Wave quantization** — tiles schedule in waves across 148 SMs, so a tile count just past a multiple of 148 runs a near-empty final wave. At its worst (149 tiles on 148 SMs) that would nearly double the time for the same work; measured, it is a mild ripple.

The geometry suggests throughput should sawtooth and collapse just past each boundary. It doesn't. Sweep M on the same gpt-oss expert and read efficiency as a fraction of the shape's own peak, so boost-clock wobble cancels. It ramps first — small M means few tiles and most SMs idle, the §2 penalty — then ripples near peak. The ramp dominates; wave and tile quantization are the texture on top rather than a cliff.

![Measured bf16 GEMM efficiency versus M on a B300, gpt-oss expert (K=N=2880), as a percentage of the shape's own peak. Efficiency ramps steeply from a few percent at small M, where few tiles leave most of the 148 SMs idle, to near 90% by M≈1100, then ripples between roughly 70% and 95% as tiles repack across the SMs: a ramp plus a ripple, not the deep sawtooth the tile/wave geometry predicts in the abstract](../assets/diagrams/tile_quantization.png){ .diagram-mid }

This is why `F.grouped_mm` and per-shape cuBLAS trade places. The loop lets cuBLAS pick a per-shape tile and absorb the tail, at one launch per expert; grouped GEMM uses one tile shape for all groups and eats the tail on small ones, but pays a single launch. Which wins is set by `M` and expert count, roofline plus launch floor.

### Blackwell vs Hopper: same idea, different tensor cores

| | Hopper (H100/H200, SM90) | Blackwell (B200/B300, SM100/103) |
|---|---|---|
| Tensor-core MMA | `wgmma` (4th-gen) | **`tcgen05`** (5th-gen, 2-CTA cluster) |
| Async movement | **TMA** | TMA + **tensor memory (`tmem`)** |
| Native low precision | FP8 | FP8 **+ FP6 + FP4**, block-scaled (**MXFP8/6/4, NVFP4**) |
| HBM | HBM3 (3.35 TB/s) / HBM3e (4.8 TB/s) | HBM3e, ~8 TB/s spec (**6.6 measured**) |

An Ampere-era `mma` kernel runs on Blackwell, but it feeds the 5th-gen cores through a narrow straw: no cluster MMAs, no async TMA. This is the FA2 situation (§6), the same math with most of the Blackwell core idle. Peak is reachable only by an arch-matched kernel, which is why the toolkit ships FA4 rather than FA2 on Blackwell and FA3 on Hopper, and why hand-rolled Triton tops out well below cuBLAS/CUTLASS (below).

### The kernel software stack

A step leans on a handful of kernel kinds: standard matmuls (projections), fused elementwise/norm (§5, Liger), attention (§6, flash), and MoE grouped GEMM ([Grouped GEMM](../optimization/grouped-gemm.md)). The library ladder runs from ease to control:

- **cuBLAS** — the closed-source BLAS PyTorch calls for `a @ b`; near-peak on *standard* GEMMs only (no grouped, no fused epilogues, no exotic low-precision).
- **CUTLASS** — open-source C++ GEMM building blocks for what cuBLAS doesn't ship: grouped GEMM, fused activations, block-scaled fp8/fp4. The bf16 grouped path (`torch.nn.functional.grouped_mm`) dispatches here.
- **CuTe** — the layout abstraction inside CUTLASS 3.x; the **CuTe DSL** is its Python frontend, and FA4 is written in it.
- **Triton** — a Python kernel DSL, easier than CUTLASS and right for elementwise/fusion (Liger is Triton). A hand-written Triton dense matmul tops out at ~0.6–0.7× cuBLAS on Blackwell: it cannot express the warp-specialized pipeline plus cluster MMA.
- **DeepGEMM** — DeepSeek's fp8/fp4 grouped-GEMM library for MoE; the verdict on it waits for [§7](#7-precision-low-precision-compute-and-bf16-numerics).

### Streams, launches, CUDA graphs

The CPU enqueues each kernel into a **stream** that the GPU drains asynchronously, and the host races ahead to keep SMs fed. Launch cost is per kernel and on the host (the launch floor of §3), so a kernel doing less GPU work than that is launch-bound.

A stream is an **ordering constraint, not a hardware lane**. Kernels in one stream run in issue order. Kernels in different streams *may* run concurrently, but they share the same 148 SMs, and PyTorch issues everything onto one current stream per device unless code opts into side streams. The block scheduler fills SMs from any resident grid, so real overlap happens only when one kernel leaves SMs idle: two large GEMMs mostly serialize, while an NCCL collective (a few SMs' worth of copy work) does run beside compute.

Every overlap trick in §9 rests on that. Data-parallel training hides each layer's gradient collective behind the next layer's backward, and the EP wrapper can run the always-active shared experts beside the token exchange (`HALO_EP_SHARED_OVERLAP`). Cross-stream ordering is expressed with **events**: record on one stream, wait on it from another.

Streams also shape **memory**: the caching allocator pools freed blocks *per stream*, since reuse within a stream is safe by ordering alone while cross-stream reuse would need per-block event tracking. So allocations made on a short-lived stream are stranded when it dies — the reason the RL weight broadcast (§10) keeps one persistent stream pair per server client ([Rollout Servers](../infrastructure/rollout-servers.md#weight-sync)) — and `reserved` far above `peak_alloc` in a memory reading means stranded pools, not live tensors ([Debugging](debugging.md#3-gpu-memory-profiling)).

**CUDA graphs** record a kernel sequence once and replay it with a single launch, collapsing that host cost — a win for many-small-kernel steps. The catch is rigidity: fixed shapes and addresses fight dynamic control flow, so graphs are awkward for MoE, where per-expert token counts change every step. The toolkit does not capture the EP path.

---

## 5. Memory-bound kernels and why fusion pays

Most non-matmul work does about 1 FLOP per element loaded (AI ≈ 1, far below the ridge), so it runs purely memory-bound: RMSNorm, SwiGLU/GELU, residual adds, RoPE, softmax, the optimizer update.

On a B300, eager `silu(a)*b` on `8192×8192` bf16 tensors takes ~0.12 ms — pure HBM traffic across its two kernels (silu writes an intermediate that mul rereads: ~670 MB over 5 passes; the fused minimum is read `a`, read `b`, write out — 400 MB). A full `8192³` GEMM moves that same 400 MB and does 1.1 TFLOP on it — thousands of times the arithmetic — yet at §1's 1818 TFLOP/s ceiling it takes ~0.6 ms, only ~5× as long. At AI ≈ 1 the arithmetic is effectively free and the bytes set the runtime.

A naive activation is several such kernels, each a separate HBM round trip. Fusion collapses them into one kernel that loads once, does the math in registers, and writes once, cutting HBM traffic ~3× for a 3-op SwiGLU:

![Fusion means fewer HBM round trips. y = silu(a)·b run naively is three elementwise kernels (sigmoid, multiply, multiply), each reading from HBM and writing an intermediate back, 8 tensor passes in total. Fused, one kernel reads a and b, keeps intermediates in registers, and writes y: 3 passes, ~2.7× less traffic for the same math and FLOPs](../assets/diagrams/fusion_roundtrips.png){ .diagram-mid }

The toolkit gets this fusion from two places:

- **Liger**: hand-written Triton kernels (fused RMSNorm, RoPE, SwiGLU, cross-entropy, fused-linear-cross-entropy), auto-applied per model type. Conflicting ones auto-disable: SwiGLU when an EP wrapper replaced the MoE layer, the cross-entropy variants when TP shards the vocab, CP computes its own loss, or a per-microbatch PP loss would own the reduction. [Liger Kernels](../optimization/liger-kernels.md).
- **torch.compile** (Inductor) auto-fuses pointwise chains. [torch.compile](../optimization/torch-compile.md).

### Fused-linear-cross-entropy (FLCE)

An LM's final step multiplies each position's hidden state by `lm_head` to produce logits, one score per vocab token per position. At 150k vocab and seq=32k the logit tensor is `[32768 × 151936]`, tens of GB once upcast to fp32, materialized only to reduce to a loss scalar.

FLCE never materializes it. It fuses the `lm_head` projection and cross-entropy into one kernel that processes the sequence in chunks, computing each block's logits, loss, and gradient and freeing them before the next — often the difference between a long sequence and OOM.

The cost is that `outputs.logits` never exists. That makes FLCE SFT-only (preference and distillation need logits for log-probs), turns off entropy logging, and forces it off under TP, CP and PP. It is default-on for the large-vocab families whose logits plane is the binding limit (`zaya`, `deepseek_v4`, `glm4_moe_lite`), opt-in elsewhere (`liger_kernel_config: {fused_linear_cross_entropy: true}`). Per-model defaults and the measured savings: [Liger Kernels](../optimization/liger-kernels.md).

### What fusion does not buy

- **Chains, not isolated ops.** `torch.compile(mode="max-autotune")` on a single `silu(a)*b` is slower than eager on a B300, since autotune and guard overhead exceed the one-op saving.
- **Compile carries guard and recompile cost.** On parallel models the DeepEP all-to-all (EP), DTensor dispatch (TP), and Ulysses all-to-all (CP) each break the graph, so compile fuses only the spans between breaks and its ceiling stays near Liger's ([torch.compile](../optimization/torch-compile.md)).
- **Nothing for compute-bound matmuls**, which have no HBM pass to remove.

---

## 6. Attention — the canonical memory-bound fix

Naive attention computes `S = QKᵀ` (an `[seq, seq]` score matrix per head), softmaxes it, then `S·V`. The `seq²` score matrix at seq=32k is a billion elements per head, and writing it to HBM then reading it back for the softmax is the most memory-bound op in the model.

FlashAttention never materializes `S`. It tiles the sequence, computes scores for a tile in SRAM, keeps a running max and sum (online softmax, where the running max doubles as the overflow guard, see §7), and accumulates output tile-by-tile. HBM traffic drops from `O(seq²)` to `O(seq)`, which is why long-context training is possible at all. The backward recomputes tiles instead of storing `S`.

The compute stays `O(seq²)`: this is an IO optimization, not a cheaper algorithm. In roofline terms (§2) it moves attention across the ridge. Naive AI ≈ `head_dim` (64–128 for most models, well below the ~275 ridge, so memory-bound), while Flash's effective AI ≈ `seq/2` lands attention on the compute-bound side.

The versions are arch-tuning of one algorithm, each pairing the same math with its generation's tensor cores: FA2 on Ampere `mma` (the portable baseline, which under-feeds Hopper/Blackwell — the §4 narrow straw), FA3 on Hopper `wgmma` + TMA, FA4 on Blackwell `tcgen05` + tensor memory + 2-CTA clusters. Auto-selected per architecture; selection logic, GptOss sink handling, and the per-family NaN→SDPA fallbacks are in [Flash Attention & Backends](../optimization/flash-attention.md).

---

## 7. Precision — low-precision compute and bf16 numerics {#7-precision-low-precision-compute-and-bf16-numerics}

### When low-precision compute helps

FP8/FP4 doubles or quadruples tensor-core throughput and raises the compute ceiling. The roofline tells you when that pays off:

- **Compute-bound (AI above ridge):** the ceiling is what you are hitting, so raising it makes the step faster. This is the large-M / wide-matmul regime (dense MLPs at long seq, very wide experts).
- **Memory-bound (AI below ridge):** you're on the diagonal, nowhere near the ceiling, so raising it does nothing. Most schemes also add a per-step quantization pass (read operand, compute scales, write quantized copy), so on a bandwidth-bound kernel the net result is slower.

Fine-grained MoE experts live in the second regime: small per-expert M, weight-bandwidth-bound (§2 table). That is the measured reason bf16 is the toolkit default for MoE expert compute. Low precision earns its keep as a *memory* lever (smaller weights, activations, optimizer states, EP-dispatch payloads) and for inference, not as a compute lever at fine-grained MoE shapes.

A frontier-scale result therefore doesn't transfer down. DeepSeek-V3 (671B total, 37B active) trained its core GEMMs in FP8 and roughly doubled their throughput. At 671B with an enormous global batch, those matmuls sit above the ridge. Even there it kept embeddings, output head, router gating, norms, and attention in higher precision. At the toolkit's model sizes and per-expert M the same GEMMs fall below the ridge, where FP8 buys nothing. Check which regime *your* shape lands in.

### Why a faster kernel doesn't change the verdict

An isolated block-scaled nvfp4 grouped kernel does beat bf16 — but only with the group layout known ahead of time. MoE routing is dynamic, so each group's size and offset change every step and the fixed-layout CuTe grouped API has to rebuild its per-expert pointer arrays and swizzle metadata on the CPU every call, leaving the GPU idle behind a host wall.

DeepGEMM removes that wall (expert per token-row resolved on-device from an int32 `m_indices` array, one compiled kernel) and still loses: the per-token activation-quant pass it exposes never amortizes at these per-expert M. It stays opt-in (`HALO_DEEPGEMM_NATIVE=1` plus a shape gate) and net-slower than bf16 in a real step. Measurements: [Low-Precision MoE Kernels](../optimization/low-precision-moe-kernels.md).

### bf16 numerics (what stays fp32)

| format | exp / mantissa bits | dynamic range | sig digits |
|---|---|---|---|
| **fp32** | 8 / 23 | wide | ~7 |
| **fp16** | 5 / 10 | **narrow** | ~3 |
| **bf16** | 8 / 7 | **wide (= fp32)** | ~2 |

Training runs in bf16 rather than fp16 because gradients span an enormous dynamic range that fp16's narrow exponent overflows (fp16 needs loss scaling). bf16 keeps fp32's range and pays for it in precision, roughly 2 significant digits. That's survivable only because the precision-sensitive steps stay fp32:

- **Reductions accumulate in fp32.** Summing thousands of bf16 numbers (softmax, RMSNorm denominator, loss, long dot product) would otherwise swamp the small terms. Tensor cores already accumulate every matmul in fp32 and round the result to bf16.
- **The optimizer keeps the update honest.** A step often nudges a weight by less than the bf16 ULP (the gap between neighboring representable values), which round-to-nearest would drop. **AdamWBF16** uses stochastic rounding on the weight write and on the always-positive `exp_avg_sq`, where nearest-rounding also biased upward ~50%. [BF16 Optimizer](../optimization/bf16-optimizer.md).

**Master weights** are the textbook version of that second defense: keep the authoritative copy of every weight and both moments in fp32, and cast down to bf16 only for the forward and backward. That state costs 12 B/param, with the transient bf16 copy on top. AdamWBF16 keeps no master copy — weights and moments live in bf16 at 6 B/param, and the kernel upcasts to fp32 in registers, does the update there, and rounds the write back stochastically where nearest would bias (the weight and `exp_avg_sq`, above). At 20B params that is 120 GB instead of 240 GB, and the loss curve tracks the fp32-master run (gap ~8e-5 after 200 steps on a 50M-param test) ([BF16 Optimizer](../optimization/bf16-optimizer.md)).

So "bf16 training" means bf16 storage plus fp32 accumulation, fp32 reductions, and a stochastic-rounding optimizer. The low-precision formats add a fourth defense: block scaling, one MXFP/NVFP4 scale per 16–32 values, so a few large values don't crush the rest to zero.

---

## 8. Memory — what fills HBM, and the compute↔memory trade

HBM holds four things during training. Per parameter, typical bf16 setup:

| What | Bytes/param | Notes |
|---|---|---|
| **Weights** | 2 (bf16) | the model |
| **Gradients** | 2 (bf16) | `fp32_grad_reduce` reduces in fp32 over bf16 storage |
| **Optimizer state** | 4 (bf16) – 8 (fp32) | AdamW keeps `exp_avg` (momentum) + `exp_avg_sq` (mean-square). fp32 = 8 B; **AdamWBF16** stores both bf16 = 4 B. [BF16 Optimizer](../optimization/bf16-optimizer.md) |
| **Activations** | *not per-param* | grow with `batch × seq × hidden × num_layers` |

The first three scale with parameter count and shrink with sharding (FSDP, EP).

### Activations: the memory sharding doesn't touch

FSDP and EP shard weights, gradients, and optimizer state, but not activations. Every data-parallel rank runs its own micro-batch and stores that micro-batch's full activation set, so adding DP ranks shrinks the per-rank weight footprint and nothing else. A model whose weights fit can still OOM at long sequence, and more DP won't help. The fix is less activation: gradient checkpointing (below), Flash attention (§6), or context parallelism, the one mode that shards the sequence/activation axis (§9).

Activation memory grows linearly in batch, sequence, hidden size, and layer count, and not at all in parameter count. The per-layer constant is small — the few tensors autograd keeps for the backward. One bf16 hidden state at `seq=32768, hidden=4096, batch=1` is `32768 × 4096 × 2 ≈ 268 MB`, and a transformer block keeps several of them (attention and MLP inputs), paid once per layer. That product is why every lever against long-context memory (GC, Flash, CP) targets *seq* rather than weights.

### One weight through a step

Take `Y = X @ W`:

![Timeline of one weight through a training step: forward runs Y = X @ W (2P FLOPs) and saves X; backward runs two matmuls, dgrad dX = dY @ Wᵀ (2P, flows to the previous layer) and wgrad dW = Xᵀ @ dY (2P, needs the saved X), so forward 2P plus backward 4P is 6P FLOPs per step (P = parameters); update is a cheap elementwise AdamW step. Below, a proportional bar of HBM bytes per parameter: weights 2 B, gradients 2 B, optimizer state 4 B (8 B total, bf16 AdamW), with activations set apart because they scale with batch and sequence, not parameter count](../assets/diagrams/step_dataflow.png){ .diagram-mid }

Two backward matmuls per forward — `dgrad` to the previous layer, `wgrad` needing the saved `X` — is why a step totals `6P`, and keeping `X` alive is exactly what GC trades for compute. Dtypes follow §7: `X`, `W`, `dY` are bf16; every matmul accumulates in fp32 and rounds to bf16; `dW` is bf16 (fp32-reduced under `fp32_grad_reduce`); `m` and `v` are bf16 under AdamWBF16; the `W` write uses stochastic rounding.

### Optimizer variants

The two moments cost 4–8 B/param. Each variant trades one axis:

- **Memory — quantize state.** AdamW-8bit (bitsandbytes) stores `m`, `v` as 8-bit with block scaling; **AdamWBF16** is the lighter default (bf16 + SR, 4 B/param). Use 8-bit when even that won't fit.
- **Speed — fuse the kernel.** [FlashAdamW](../optimization/flash-adamw.md) fuses the elementwise update with quantized states, but the quant/dequant adds step time — it is for OOM relief, not free speed.
- **Quality — change the update.** [Muon](../optimization/muon-optimizer.md) orthogonalizes the momentum (Newton–Schulz) for 2D weights and keeps one moment (no `v`), lighter as a side effect.

### Gradient checkpointing

GC keeps only a few activation checkpoints and recomputes the rest in the backward. Memory drops sharply, so longer sequences fit; the cost is one extra forward over the checkpointed regions, `6P → ~8P` — measured +29% wall-clock on gpt-oss-20b ep8 (batch 4, seq 4096, 8× B300), under the `8P/6P ≈ +33%` ceiling because only checkpointed regions recompute and comm is unchanged ([Throughput Benchmarks](../optimization/throughput-benchmarks.md)).

The corollary: if the batch fits without GC, turning it off is a free speedup, so keep GC on only when you need the memory. **Selective recomputation** (Megatron) checkpoints only the regions with the best memory-saved-per-recompute-FLOP; FlashAttention is already this idea specialized to the attention block.

---

## 9. Distributed training: the communication wall

Split a model across GPUs and a third speed enters the picture: inter-GPU bandwidth. It has its own roofline, and at scale it is usually the real ceiling.

### Interconnect tiers

| Link | Connects | Bandwidth/GPU | vs HBM |
|---|---|---|---|
| **HBM** | a GPU to its own memory | **~6.6 TB/s** (measured) | 1× |
| **NVLink / NVSwitch** | GPUs within a node / NVL72 rack | **~1.8 TB/s** spec (NVLink 5); NCCL all-reduce bus bw **767 GB/s** measured on 16 GPUs across 2 nodes | ~4× slower (spec) |
| **PCIe** | CPU ↔ GPU | ~64 GB/s/dir (Gen5 ×16) | ~100× slower |
| **InfiniBand / Ethernet** | across nodes | ~50 GB/s (400 Gb/s NDR; ~46–48 effective) | ~130× slower |

![The bandwidth ladder on a log scale: HBM at 6.6 TB/s measured (spec 8 TB/s), NVLink/NVSwitch inside the node at 1.8 TB/s spec (~4× below HBM), with the repo's one recorded all-reduce bus bandwidth, 767 GB/s across 16 GPUs on 2 nodes, drawn as the measured bar, PCIe Gen5 ×16 at 64 GB/s per direction (~100× below HBM), and InfiniBand NDR across nodes at ~48 GB/s (~130× below HBM, the node boundary). Filled bars are measured, dashed outlines are spec; a collective is priced by the slowest tier it touches](../assets/diagrams/interconnect_tiers.png){ .diagram-mid }

One ratio drives distributed design. Crossing a node boundary costs ~36× more per byte than staying on NVLink (1.8 TB/s vs 50 GB/s, spec against spec). A collective inside the NVLink domain is cheap. Stretch it across nodes and it hits the InfiniBand wall.

- **NVLink** is direct GPU-to-GPU (Blackwell NVLink 5 ≈ 1.8 TB/s/GPU, Hopper NVLink 4 ≈ 0.9) and bypasses the CPU.
- **NVSwitch** is an on-board crossbar: all 8 GPUs reach each other at full NVLink bandwidth, any-to-any. NCCL's effective all-reduce bus bandwidth sits below the link spec because of algorithm and protocol overhead, the same measured-vs-spec gap HBM shows. The repo's one recorded number, 767 GB/s, was measured on 16 GPUs across two 8-GPU B300 nodes — NVLink inside each node plus a cross-node RDMA hop — so a single node's all-reduce lands above it ([DeepEP](../infrastructure/deepep.md)).
- **NVLink SHARP (NVLS)** does the all-reduce sum inside the NVSwitch ASIC. Each GPU then sends its tensor once and receives the sum once, against ~2× the tensor over a ring — a theoretical ~1.75× at 8 GPUs, rising toward 2× at large `R`, and tens of percent measured. NCCL enables it where the fabric supports it.
- **Rack-scale (NVL72).** GB200/GB300 NVL72 extends one NVLink domain across 72 GPUs over ~18 trays. The NVLink boundary is then the rack, not the OS node. The operator declares the domain with `NVLINK_DOMAIN_SIZE`, and the toolkit sizes every node-local group to that value; on an 8-GPU server it is the node. The declaration is checked against the fabric's clique membership, and a block spanning two cliques raises. [Multi-Node](../parallelism/multi-node.md).

### Collectives

![The four collectives as before/after grids of 4 ranks × 4 chunk slots. All-gather (FSDP forward): each rank starts with one shard, ends with every shard, (R−1)/R × D bytes per GPU. Reduce-scatter (FSDP backward, all-gather's dual): each rank starts with a full different-valued tensor, ends with the sum of one chunk, same bytes. All-reduce (DDP gradient sync, TP layer outputs): everyone ends with every summed chunk, a reduce-scatter followed by an all-gather, 2·(R−1)/R × D. All-to-all (EP dispatch, CP Ulysses): a transpose of the rank×chunk grid, each chunk with exactly one destination. Backward runs the dual: the gradient of an all-gather is a reduce-scatter and vice versa; an all-to-all's is another all-to-all](../assets/diagrams/collective_ops.png){ .diagram-mid }

Three identities do most of the reasoning about them.

- **all-reduce = reduce-scatter + all-gather.** Each half moves `(R−1)/R × D` bytes per GPU over `R` ranks (ring, `D` = tensor size); composing them doubles it. FSDP is this identity pulled apart around the compute: all-gather the weight shards before a layer's forward, reduce-scatter its gradients after the backward. At the toolkit's `reshard_after_forward=False` its per-microstep wire bytes therefore match DDP's all-reduce, and the sharded storage in between is what it buys.
- **The backward runs the dual.** The gradient of an all-gather is a reduce-scatter and vice versa, and an all-to-all's is another all-to-all. Choosing a forward comm pattern fixes the backward's bill too.
- **An all-to-all sends each byte to one destination.** All-reduce and all-gather push nearly the whole tensor across every link on the path, whereas all-to-all chunks travel point-to-point. That is why the toolkit pins TP NVLink-local: its all-reduce sits on every layer's forward critical path, at full activation width. EP is allowed to cross nodes, because its dispatch sends each routed token to one expert owner and home again.

Cost has two regimes. The crossover message size is roughly `link_bandwidth × per-hop latency`, a few hundred KB at NVLink bandwidths and microsecond hops. Below it, round count dominates and trees (`log₂ R` rounds) beat rings (`R−1` rounds); above it, rings deliver more of the wire bandwidth. NCCL switches algorithm per message size and topology.

Comm does no FLOPs. The defenses are amortize (bigger M per step), shrink the degree (fewer ranks make a cheaper collective), and overlap (own stream, hidden behind compute). Gradient accumulation amortizes when memory caps the micro-batch: run several micro-batches, sum gradients locally, then fire the optimizer step and collective once. It buys a larger global batch than fits, but it is not itself a throughput lever, since it adds compute proportionally.

### Each parallelism mode is a communication pattern

Each mode buys memory headroom by spending a specific collective.

- **FSDP (data parallel)** shards weights, grads, and optimizer state, paying that first identity once per FSDP unit (a decoder layer). Both halves overlap well (prefetch the next layer), so FSDP reaches the highest efficiency when the model fits. [Data Parallelism](../parallelism/data-parallelism.md).
- **EP (expert parallel)** shards experts and adds a token all-to-all before and after each MoE layer (DeepEP). It is orthogonal to DP and tolerates cross-node links. The catch is load balance: real routing is lumpy, so the busiest expert and the all-to-all gate the step and skew utilization. The toolkit tracks per-expert load and rebalances the router (aux loss or DeepSeek-V3 bias update). [Expert Parallelism](../parallelism/expert-parallelism.md).
- **TP (tensor parallel)** splits each matmul and needs an all-reduce on the forward's critical path that the toolkit does not overlap, so TP must stay NVLink-local. [Tensor Parallelism](../parallelism/tensor-parallelism.md).
- **CP (context parallel)** splits the sequence; a pair of all-to-alls swaps sequence-sharding for head-sharding around each attention (Ulysses). Use it for sequences too long for one GPU. [Context Parallelism](../parallelism/context-parallelism.md).

**Where ZeRO fits.** Most readers arrive with DeepSpeed's stage vocabulary. DDP replicates everything; ZeRO-1 shards optimizer state, ZeRO-2 adds the gradients, ZeRO-3 shards the parameters too. FSDP2's `fully_shard` shards all three, and the toolkit's default `reshard_after_forward=False` keeps each unit's gathered parameters resident from its forward to its backward — the ZeRO-2 analog. `fsdp_reshard_after_forward: true` is the ZeRO-3 analog: it drops the gathered copy after the forward and re-gathers it for the backward, ~1.5× the wire bytes for a lower peak, and it is available only where no expert-distribution group exists. No DeepSpeed runs here; ZeRO is the naming only ([Data Parallelism](../parallelism/data-parallelism.md#zero-2-vs-zero-3-reshard_after_forward)).

### Why N× GPUs is not N× throughput

Adding a GPU adds compute but also adds communication that does no FLOPs. **Strong scaling** (fixed work, more GPUs) stalls twice over: each GPU's compute slice shrinks (its `M` slides back down the small-M ramp of §4) while the per-step collective does not, so comm's fraction climbs. This is why over-sharding a model that nearly fits scales badly. **Weak scaling** (grow work with GPUs, bigger global batch) scales far better, because each GPU keeps a full `M`. Keep per-GPU `M` large, keep collectives on the fastest tier they tolerate, and overlap them.

### Pipeline parallelism: the bubble and the boundary

The schedule engine behind PP is **not yet available in this release** — `pipeline_parallel_size > 1` is rejected at config time, so what follows is the design argument rather than a runbook ([Pipeline Parallelism](../parallelism/pipeline-parallelism.md)).

PP cuts the layer stack into `p` contiguous **stages**, one rank block each, and streams `m` **microbatches** through them. It carves the world first. Every other mode then runs unchanged inside one stage, on `stage_world_size = world_size / p` ranks. That is what "outermost" means, and it is why each stage must be a whole number of NVLink domains. Even once it lands PP would stay off by default — its idle time buys nothing inside a domain where FSDP2 and EP already fit.

**The bandwidth argument.** A stage boundary carries one tensor: the hidden state forward, its gradient back. Per step per edge that is `4 · T · H` bytes (bf16, both directions), for `T` tokens per step and hidden size `H` — `× hc_mult` for the hyper-connection families, whose boundary is the widened stream. At `T = 8192, H = 4096` that is 134 MB, against the ~32 GB FSDP2 moves per rank per step on an 8B model. That is ~250× apart, and point-to-point rather than every-link (the third identity above). Every other model-splitting axis prices its traffic per *parameter*; PP prices it per *token × hidden*. So PP is the axis worth spending a ~50 GB/s cross-node link on. The same pricing is why the toolkit forces stage boundaries onto NVLink-domain boundaries: every intra-stage collective then stays inside a domain, and only the P2P activation crosses.

**The bubble.** Fill and drain leave each rank idle for `p−1` of a step's `m + p − 1` microbatch slots (the figure below splits each slot into its forward and backward halves, 6 idle of 22):

```text
bubble fraction = (p − 1) / (m + p − 1)
```

![A 1F1B pipeline schedule drawn as a grid: 4 stages down, 22 time slots across, one slot per microbatch forward or backward. Every stage runs the same 16 busy slots — 8 forwards and their 8 backwards, alternating one for one once the pipe is full — and sits idle for the other 6. Later stages wait at both ends, while the pipe fills and while the last backwards drain; stage 0 waits in the middle instead, after its warmup forwards, until the first backward comes back down. Those 6 of 22 slots are the bubble, (p−1)/(m+p−1) = 3/11 ≈ 27% of the step](../assets/diagrams/pp_bubble.png){ .diagram-wide }

The formula assumes equal per-stage cost and no cross-step overlap. What sets it is the ratio `m/p`, not either term alone. At `m = 2p` the idle share runs from a fifth (`p = 2`) toward a third (large `p`); planning `m = 9·(p−1)` puts it at 10%.

**The tension with M.** Under PP the microbatches *are* the gradient accumulation. Raising `m` to shrink the bubble therefore splits the per-device batch into thinner slices and slides every GEMM's `M` down §4's ramp. Fixed shapes push the same way: the P2P buffers freeze on the first step, so every batch pads to a constant width. That is why packing (fixed-width rows) would be the shape to run under PP, and why the padding-free collator, whose width is the summed document length, is refused. Deep pipelines want many microbatches; the roofline wants few and fat.

**Memory.** A stage holds `1/p` of the parameters, and torch's schedule pins them unsharded across the backward and defers FSDP's reduce-scatter to the optimizer step. Per rank that is `2P/p` bytes of resident weights and `2P/p` of accumulating gradient, while optimizer state stays sharded at `4P/world_size` (§8); `fp32_grad_reduce` doubles the gradient term to `4P/p` ([BF16 Optimizer](../optimization/bf16-optimizer.md#master-weight-and-grad-reduce-options)). PP divides the one term FSDP2 leaves whole.

Activations are the sharper win. A **1F1B** (one-forward-one-backward) stage holds at most `p` microbatches over `L/p` layers at `B/m` rows each — `1/m` of what the same rank stores without PP. (GPipe's all-forwards-first schedule holds `B·L/p`, which is why 1F1B is the default.) PP hands a dense stage the headroom GC exists to buy, so a dense PP run would want GC off, and nothing would disable it automatically.

**Composition.** The expert axes are the only other sharding PP admits: EP, whose dispatch stays inside a stage, or pure ETP (`ep_size=1`), never both. The rejections all share one shape — a per-microbatch schedule breaks any collective or normalizer that assumes one backward per step: TP's replicated-grad hook would re-reduce accumulated history, CP's `× cp_size` factor never cancels, and a reentrant checkpoint's `no_grad` forward leaves FSDP2 without pre-backward hooks — which is why gradient checkpointing under PP must be non-reentrant. Tied embeddings break too: stage 0's embedding and the last stage's head are one parameter landing on two ranks.

### Megatron-LM

Megatron-LM popularized TP, PP, sequence parallelism, and the expert-parallel MoE path, and the toolkit inherits these patterns (ETP reuses its scatter-gather autograd; `scripts/before_training/prepare_dataset.py` writes Megatron-style sharded datasets, see [Data Loading](../parallelism/data-loading.md)). It does not adopt Megatron-Core, which carries its own model definitions and sharded checkpoint format — EP/CP/TP apply to native `transformers` classes through FSDP2 and DTensor, with no conversion and no parallel model zoo ([Why This Framework](why-this-framework.md)).

Two Megatron techniques are forgone. Sequence parallelism in Megatron's sense (splitting LayerNorm/dropout along sequence inside the TP region) is replaced by CP for sequence-axis memory relief. TP communication overlap (Userbuffers ring decomposition, which needs sequence parallelism and static shapes) is unnecessary while TP stays NVLink-local, where the all-reduce is cheap enough not to need hiding.

---

## 10. Training vs inference: different bottlenecks

The most common wrong choice starts "fp8 made inference 2× faster, so it'll speed up training."

Inference compute splits into prefill (all prompt tokens, large M, compute-bound, like a training forward) and decode (one token at a time, M=1 per sequence). Decode is the §2 small-M regime at its extreme — weight-bandwidth-bound, its speed set by how fast weights stream rather than by FLOPs. That is why low precision pays there: quantizing weights halves (fp8) or quarters (fp4/int4) the bytes per token and shrinks the KV cache. None of it transfers to training, where weights stay in training precision, the streamed bytes don't shrink, and §7's quant-pass overhead is all that remains.

| | Training | Inference (decode) |
|---|---|---|
| Passes/layer | `fprop`+`dgrad`+`wgrad` (3 GEMMs) | `fprop` only |
| Dominant memory | activations (∝ seq×batch) + optimizer state | weights + KV cache |
| Per-matmul M | batch×seq (large, packed) | 1 per sequence (tiny) |
| Typical regime | compute-bound at good M; comm-bound when sharded | weight-bandwidth-bound |
| Low precision | mostly a *memory* lever; bf16 compute at these MoE shapes | a real *speed* lever |
| Optimize | bigger M, fusion, GC trade-off, comm overlap | weight quant, KV compression, batching |

Hence `scripts/after_training/quantize_to_lowp.py`, which converts a trained bf16 checkpoint to mxfp8/nvfp4: fp8/fp4 is an inference and memory capability here, and training stays bf16.

### RL is both at once

An on-policy RL step (online / environmental GRPO) contains both columns of the table: **generate** rollouts (decode — the weight-bandwidth-bound regime), grade them, **train** on them (the compute-bound regime), then **broadcast** the updated weights back to the generator (comm). No single machine configuration is right for both, which is why the toolkit splits them across GPUs: an inference engine (vLLM/SGLang) owns some and the trainer owns the rest ([Rollout Servers](../infrastructure/rollout-servers.md)).

The engine does what decode wants — KV cache, CUDA-graph decode, continuous batching — with one decode lever forgone: weight quantization. The sync writes bf16 checkpoint-layout tensors into the server's parameter storage in place, so a quantized engine would store transformed tensors the broadcast cannot update; the engine serves bf16.

A multi-turn RL step is usually **rollout-bound**. Decode at M≈1 per sequence emits tens of thousands of tokens while the trainer waits, so trainer GPUs at idle-class power during generation are expected rather than a stall — read power per §11, but per *phase*. The levers are the same three ideas at step granularity instead of kernel granularity — overlap (prefetch the next round's generation behind the current training step), batching (a bigger *generation batch* saturates the decode engine the way a bigger M saturates a GEMM, while the concurrency cap only throttles work that already exists), and amortization (`sync_weights_every_n_steps` spreads the broadcast over more steps).

The weight sync itself is a bandwidth question with the same shape as §9's ladder: a full-model broadcast (~42 GB at 20B) is cheap over NVLink and still ~1–2 s over host loopback, a tier below the ladder's bottom rung. The SGLang socket transport's sharper cost is collateral: it is process-global, so it also demotes the trainer's *own* FSDP2 collectives to TCP. Hence that backend's `fsdp_reshard_after_backward: false`. Throughput anatomy and the tuning order live in [Environmental GRPO](../training-methods/grpo/environmental-grpo.md#throughput-tuning).

---

## 11. Measuring it yourself

### Watch power, not utilization %

A `nvidia-smi` util % of 100 means an SM had a warp scheduled, not that tensor cores did math. A memory- or launch-bound kernel reads near 100% util while the ALUs idle. Power is the honest signal:

| Power (fraction of board limit) | Interpretation |
|---|---|
| near TDP (95–100%) | tensor cores saturated → **compute-bound** (good) |
| ~50–70% at high util % | stalled on HBM → **memory/launch-bound** |
| < 40% at high util % | wrong kernel for the shape |

Sample during a warm step: `nvidia-smi --query-gpu=power.draw,utilization.gpu --format=csv -l 1`.

### MFU and why MoE complicates it

```text
MFU = (6 × parameters × tokens) / (peak_FLOP/s × step_time)
```

This is the roofline applied to a whole step: 1.0 means every nanosecond ran at the compute ceiling. The denominator matters — `EfficiencyCallback` uses its per-chip registry value (2250 TF bf16 on a B300), not §1's measured 1818 TF GEMM ceiling, so a step that matched the best-possible GEMM reports ~0.81, not 1.0.

The `6 × parameters` numerator assumes every parameter works on every token, which is false for MoE — each token visits only its top-k — so plain MFU credits FLOPs that never ran. The toolkit reports **S-MFU** instead, scaling the expert term so it counts active work only ([definition](../optimization/throughput-benchmarks.md#why-moe-utilization-reads-low)). For comparing MoE configs, raw achieved TFLOPS and tokens/sec are the more reliable signals; only tokens/sec is comparable across GPUs, since the peak-FLOP/s denominator differs by chip. Measured numbers for real configs live in [Throughput Benchmarks](../optimization/throughput-benchmarks.md).

### Reproduce the anchors

Every roofline number comes from `tests/gpu/profiling/benchmark_roofline.py`, on one GPU in about a minute:

```bash
# inside the training image (see docker.md), on one GPU:
docker run --rm --gpus '"device=0"' --ipc=host -v "$(pwd)":/workspace -w /workspace \
  halo:blackwell \
  bash -lc "python tests/gpu/profiling/benchmark_roofline.py --expert_kn 2880"
```

It measures HBM copy bandwidth, large-square bf16 GEMM peak, the small-M→large-M roofline crossover at a `K=N` expert shape (`--expert_kn`, default 2880; pass your model's `moe_intermediate`), the launch floor, and elementwise fusion traffic. Point `--expert_kn` at your expert width and read off which side of the ridge each `M` sits on.

For a full training step, enable the profiler and efficiency callback to get the real compute/comm/elementwise split. See [Debugging](debugging.md) and [Throughput Benchmarks](../optimization/throughput-benchmarks.md).

---

## Rules of thumb

- **Profile first, then check the dominant GEMM's `M` against the ridge.** Below it, spend on bytes moved; above it, on FLOPs (§2, §3).
- **Watch power, not util %.** 100% utilization at 50–70% of the board limit is memory- or launch-bound (§11).
- **Pad tokens cost full GEMM FLOPs.** Pack the batch before reaching for a faster kernel (§2).
- **Gradient checkpointing is a memory lever with a compute price.** Leave it off whenever the batch fits without it (§8).
- **Low precision is a memory lever at these MoE shapes, not a speed one.** Train in bf16; export to fp8/fp4 (§7, §10).
- **N× GPUs is not N× throughput.** Grow the batch with the world, or per-GPU `M` slides down the ramp (§9).
- **Keep TP inside an NVLink domain and let EP cross nodes; PP is the axis a cross-node link would be spent on** (§9).
- **Compare runs in tokens/s per GPU.** It survives a change of config and chip; MFU compares only within one chip (§11).

---

## Related pages

Each lever in depth: [Optimization](../optimization/README.md). Each mode in depth:
[Parallelism](../parallelism/README.md). Measured throughput of real configs:
[Throughput Benchmarks](../optimization/throughput-benchmarks.md).

## Further reading (external)

- Horace He, [*Making Deep Learning Go Brrrr From First Principles*](https://horace.io/brrr_intro.html) — compute- vs memory- vs overhead-bound.
- NVIDIA, [*Matrix Multiplication Background*](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) and [*GPU Performance Background*](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html) — tiles, tail effect, wave quantization, arithmetic intensity.
- Colfax Research, [CUTLASS / CuTe tutorials](https://research.colfax-intl.com/category/papers/deep-learning/) and the [FlashAttention-3 write-up](https://research.colfax-intl.com/gpu-mode-cutlass-and-flashattention-3/) — the kernel stack and arch-native attention.
- Aleksa Gordić, [*Inside TPU and GPU Clusters: The Anatomy of Collective Communication*](https://www.aleksagordic.com/blog/collective-operations) — ring/tree algorithms, cost models per topology, SHARP in practice.
- Hugging Face / nanotron, [*The Ultra-Scale Playbook*](https://huggingface.co/spaces/nanotron/ultrascale-playbook) — DP/TP/PP/CP, activation recomputation, the communication wall.
- Stas Bekman, [*Machine Learning Engineering Open Book*](https://github.com/stas00/ml-engineering) — collective patterns, interconnect realities, multi-node debugging.
- [GPU MODE](https://www.youtube.com/@GPUMODE) (code in the [lectures repo](https://github.com/gpu-mode/lectures)) — *Flash Attention* (lecture 12), *Quantization* (lecture 7), Jay Shah's *CUTLASS and FlashAttention-3*.
