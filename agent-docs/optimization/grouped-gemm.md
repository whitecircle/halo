# Grouped GEMM for MoE Expert Compute

Grouped GEMM (`torch.nn.functional.grouped_mm`) batches a rank's per-expert matmuls into one kernel launch, eliminating per-expert dispatch overhead in MoE layers. Available on SM90+ (Hopper, Blackwell) with the pinned PyTorch 2.11.x. Enabled by default (`use_grouped_gemm: true`).

The MoE expert step sorts tokens by expert (contiguous per-expert blocks), builds an int32 cumulative-offset tensor, runs one `F.grouped_mm` over all groups, then scatters outputs back and applies routing weights. The sort (`base_layer._sort_tokens_for_grouped_mm`) drops `-1` padding from DeepEP dispatch (expert IDs outside `[0, experts_per_rank)`) before sorting; an unfiltered padding ID misaligns the groups and produces NaN.

## Performance

**Isolated kernel** — B300 (SM103, PyTorch 2.11), Qwen3-30B-A3B dims (hidden 2048, moe_intermediate 768), median over 50 iters: **≈11.93×** over the per-expert loop (forward and fwd+bwd alike), e.g. 128 experts / 4K tokens fwd+bwd 17.24 ms → 1.45 ms (11.93×), forward 1.33 ms → 0.11 ms (achieved TFLOPS 9.68 → 115.43). The ratio scales with expert count (more launches saved). These are launch-dominated; end-to-end is far smaller.

**End-to-end** — Qwen3-30B-A3B (128 experts, top_k=8), 2× B300 EP=2, FA4+Liger, seq 8192, GC on, 8 steps: **3.43×** at batch 1 (10,711 vs 3,122 tokens/s/GPU), narrowing to **2.12×** at batch 4 (15,307 vs 7,211). With 64 local experts/rank the launch saving is worth most at small batch; as per-expert GEMMs grow, the win narrows.

### Why grouped GEMM also saves memory

The launch saving (one kernel vs many) and the memory saving come from different mechanisms. A real MoE layer runs `E × P` matmuls in the loop (`P=2` for fused gate_up families, `P=3` for separate-projection families — see [Supported models](#supported-models)), each creating an autograd node that pins its saved forward activations (input slice + weight) alive across forward→backward.

Grouped collapses those to `P` nodes (Qwen3.6: 256×2=512 → 2), so far fewer saved-tensor buffers and less allocator fragmentation. The saving scales with `expert_count × projections` — largest on fine-grained 256-expert models, smallest on 32-expert GPT-OSS-20B.

## When the loop path wins

Some MoE / parallelism / hardware combinations make `use_grouped_gemm: false` competitive or faster.

**1. Few local experts per rank (primary lever).** Grouped's advantage is fusing launches across a rank's `num_experts / ep_size` local experts. Many (low EP) → large saving → grouped wins. Few (high EP) → the loop's per-shape-optimal CUTLASS tile wins, since each expert's GEMM gets a tile fit to its actual per-expert `M = ep_size × tokens_per_rank × top_k / num_experts` (tokens pool across the dispatch group, so per-rank rows `tokens_per_rank × top_k` are EP-invariant and M grows with `ep_size`). Per-expert M modulates the trend: `grouped_mm` runs one shared ~128-wide M tile for all groups, so its edge is largest at small M and erodes as M grows.

gpt-oss-20b (32 experts, top-4, seq 8192, 8× B300, FA4), grouped vs loop, plus Qwen3-30B (128 experts, ep2 → 64 local/rank) as the high-local-count anchor:

| Model | EP | local experts/rank | b1 | b2 | b4 | b8 |
|---|----|:------------------:|:---:|:---:|:---:|:---:|
| gpt-oss-20b | ep2 | 16 | grouped +60% | +39% | +21% | +8% |
| gpt-oss-20b | ep8 | 4 | grouped +3.7% | loop +0.9% | loop +4.9% | OOM |
| Qwen3-30B | ep2 | 64 | grouped +243% | — | +112% | — |
| Qwen3.5-35B | ep2 | 128 | — | — | grouped +137% | — |

At seq 8192 grouped wins ep2 at every batch and ep8 only at batch 1; the loop edges ahead from ep8 batch 2 upward. Local-expert *count* is the primary lever; batch (per-expert M) sets how far past the crossover you are. **Rule: grouped (default) wins at low EP and at high EP up to moderate batch; reach for `use_grouped_gemm: false` only at high EP with the largest batches.** The roofline reasoning is in [GPU Training Theory §2](../reference/gpu-training-theory.md#worked-example-why-small-per-expert-m-is-slow).

The 288-expert rosters (GLM-5.3-Flash, Step-3.7-Flash; top-8) sit inside the grouped-wins regime by the same rule, but between the measured anchors: `ep8` holds 36 local experts at per-expert M = 1,820 (8192 tokens/rank) to 7,282 (32k), `ep16` 18 at 3,641 (8192 — the cross-node ceiling). Derived from the table, not measured.

**2. Square experts with K not tile-aligned.** If `hidden == intermediate` (no 4× expansion) and `hidden % 128 != 0`, the CUTLASS K-tail epilogue runs on both projections; the loop picks a per-matmul tile and absorbs the misalignment.

**3. Hardware ↔ CUTLASS tuning gap.** `F.grouped_mm` selects kernels by SM major version, so a kernel tuned for one SKU may underuse occupancy/L2 on a sibling. The loop is more forgiving (cuBLAS/CUTLASS picks per-shape). Only visible in measurement.

**Diagnose with power, not `nvidia-smi` util %.** Util % means "any SM had work," not "SMs did arithmetic." Power draw is honest: near TDP = compute-bound (kernel good); ~50–70% TDP at high util = memory/launch-bound; <40% TDP at high util = wrong kernel for this shape/SKU. A/B one config flip over 10 steady-state steps and sample `nvidia-smi --query-gpu=power.draw`.

## The duplicate-index gather trap {#the-duplicate-index-gather-trap}

Around the kernel sit the token permutation and, for expert-bias models (GptOss), the per-expert bias broadcast. Both gather rows by an index with heavy duplication (every token of an expert shares its id). The forward gather (`x[idx]` / `x.index_select(0, idx)`) is cheap; the trap is the backward. PyTorch's default backward for a duplicate-valued gather is `index_add_`, whose bf16 kernel has no native atomic add and emulates one with a CAS loop that serializes under the duplicate-row contention. On a profiled gpt-oss-20b EP step that atomic scatter took ~5,000 ms over 4 warm steps (≈20% of all GPU time) and inflated the DeepEP combine wait.

**Rule: never let a duplicate-valued gather fall back to the default `index_add_` backward in a training forward.** Both gathers route through custom autograd Functions in `src/distributed/expert_parallel/autograd.py` that keep the gather forward and replace the backward.

The bias gather (`MoEExpertBiasGather`, applied in `EPGptOssMoELayer`'s grouped path) computes `grad_bias` as one GEMM (`onehot(eids)ᵀ @ grad_out`, fp32 tensor-core accumulation) — atomic-free and numerically identical (more accurate than the bf16 atomic add). The atomic-free path runs the EP step at ~6,310 vs ~1,256 tok/s/GPU for the default-backward path (≈5×, board power ~34% → ~56% of limit), and ~6,773 vs ~4,100 tok/s for plain FSDP grouped (~1.65×); the loss curve is identical.

### The atomic-free gather-reduce permute

The token permute/unpermute (`MoEGatherPermute`, `MoEScatterUnpermute`) is the larger lever for high-top_k MoE. Qwen3.6 (top-8, 256 experts, 32 local/rank at EP8) measures the scatter-back `index_add_` at ~32% of the step — 4.3 ms/call vs gpt-oss (top-4, 4 local/rank) 0.38 ms/call (11× gap, pure collision rate). The fix expresses both directions with no atomics via a precomputed `inv_map` (`[recv_N, top_k]` of sorted positions feeding each recv token, sentinel-padded), turning the scatter into gather + reduction (numerically identical to `index_add_`, float64-checked fwd+bwd).

It is gated on **`top_k ≥ ep_size`** (`base_layer._sort_tokens_for_grouped_mm` builds `inv_map` via `_build_inv_map`); below that (gpt-oss top-4 at EP8, the top-8 families at `ep16`, DeepSeek-V4-Flash top-6 at `ep8`) the plain `index_select` + `index_add_` is kept, since the extra `top_k`× read would cost ~4%. Above the gate the gather is materialized as `[recv_N, top_k, H]` before its sum — `top_k`× the recv buffer per MoE layer as a transient, in the forward unpermute and again in the permute's backward: ~5.6 GB per layer at GLM-5.3-Flash / Step-3.7-Flash shapes (16k tokens/rank at ep8, top-8, `H=4096`), most of it sentinel rows since a recv token averages `top_k / ep_size` local experts.

| Qwen3.6-35b EP=8 | `index_add_` | atomic-free | win |
|---|---|---|---|
| seq 4096  | 2,736 tok/s/GPU | 3,231 | +18% |
| seq 16384 | 2,465 tok/s/GPU | 4,056 | +65% |

The win grows with sequence length (larger recv buffers → worse contention).

Per-device batch multiplies the per-call recv buffer exactly like sequence length, so on the
families the gate leaves on the CAS path (`top_k < ep_size`) batch shape is a real lever:
gpt-oss-120b at EP8, same 64-sequence effective batch, bs2 × GA4 measures ~20% slower than
bs1 × GA8 at high router skew (`moe/load_max` ~11), converging to parity once balancing has
flattened the load (`moe/load_max` ~2). At high router skew scale with GA,
not per-device batch.

## Throughput tuning beyond the kernel

The grouped GEMM is one part of an EP step (also: all-to-all dispatch/combine, permute, attention, optimizer). The general sequence/batch playbook is in [Throughput Benchmarks](throughput-benchmarks.md#maximizing-throughput-sequence-batch); the kernel-side levers, measured on 8× B300 (SM 10.3, PyTorch 2.11+cu130, FA4, bf16):

1. **Pick parallelism by fit.** If it fits FSDP2, FSDP has no all-to-all and reaches higher achieved TFLOPS — gpt-oss-20b 1,014 TFLOPS (FSDP) vs 218 (EP=8) at seq 4096 — but is memory-heavy (148 GB at b1, near OOM at larger batch). Use EP only when FSDP OOMs.
2. **Smallest EP degree that fits the experts.** Fewer ranks = smaller all-to-all + larger per-rank GEMMs. gpt-oss EP2 (DP8) 516 TFLOPS vs EP8 218 at seq 4096.
3. **GC off when the batch fits** — recompute is ~+19% overhead on 192 GB B300 at moderate seq.
4. **Atomic-free expert permute** (above) — automatic for `top_k ≥ ep_size`, +18% (seq 4k) to +65% (seq 16k) on qwen3.6.
5. **Do not use low precision** (fp8/fp4) — measured net-slower (experts are tiny-M / bandwidth-bound, bf16 at the roofline). See [Low-Precision Kernels](low-precision-moe-kernels.md).

Compute–comm overlap is the remaining structural lever. One narrow form ships opt-in: `HALO_EP_SHARED_OVERLAP=1` runs the shared-expert FFN on a side stream concurrent with the dispatch all-to-all, on the families that have one (see [DeepEP](../infrastructure/deepep.md)). The broader async dispatch/combine restructure (hiding the all-to-all behind the routed compute) is not implemented — it needs a larger forward restructure, and gpt-oss is already compute-bound at production seq lengths.

## Low precision

This is the bf16 path, the MoE production default. fp8/fp4 grouped GEMM and the opt-in DeepGEMM kernel (`HALO_DEEPGEMM_NATIVE=1`) are net-slower than bf16 at the toolkit's MoE shapes — the per-expert GEMM is bandwidth-bound at the bf16 roofline. See [Low-Precision Kernels](low-precision-moe-kernels.md). The precision dispatch entry point is `src/kernels/grouped_gemm.py` (`grouped_gemm()`, called by `base_layer._grouped_mm`).

## Requirements and compatibility

SM90+ (Hopper H100/H200, Blackwell B200/B300) and the pinned PyTorch 2.11.x. DeepEP is required only for EP token distribution, not for standalone grouped GEMM at `ep_size=1`. Every available parallelism mode composes — EP, TP, CP, gradient checkpointing and FSDP2 — with one exception: GptOss under ETP falls back to the loop path (see the note below).

### Blackwell (SM100+) backward fix

`F.grouped_mm` has a backward bug on Blackwell (B200, B300): zero-stride gradients (from `.sum()` or any scalar reduction broadcasting back through `grouped_mm`) are rejected with `"Invalid strides/sizes, got [0, 0, 0]"`, and `torch.cumsum` upcasts int32 offsets to int64, rejected with `"Offsets have to be int32"`.

The bf16 primitive in `src/kernels/grouped_mm_autograd.py`, where `grouped_gemm()` dispatches every EP MoE layer's bf16 path, materializes zero-stride grads and auto-casts offsets to int32. An empty group takes a fresh `torch.empty` instead of `.contiguous()`: at `M == 0` the copy is a no-op that keeps the `[0, 0]` strides the kernel still validates — reachable under EP whenever a rank routes zero tokens. Overhead vs native is <3%; backward on `.sum()`-style losses runs 200–240× faster than the Python loop fallback.

### Detection and disabling

At model load the EP MoE layer enables grouped GEMM when `use_grouped_gemm` is `True` (default) and `has_grouped_mm()` (`src/distributed/expert_parallel/base_layer.py`) holds — `F.grouped_mm` exists and compute capability is ≥ 9.0. That predicate is the single authority; a test asking for grouped GEMM on pre-SM90 hardware still gets the loop path. Opt out with `use_grouped_gemm: false` in the YAML config.

## Supported models

All EP MoE families use grouped GEMM. GEMM-call count follows the weight layout:

| Model | Layer Class | Calls | Notes |
|---|---|---|---|
| Qwen3 MoE | `EPQwen3MoELayer` | 3 (gate+up+down) | Pre-fused tensors split + transposed at init |
| Qwen3.5 MoE | `EPQwen3_5MoELayer` | 2 (gate_up+down) | Fused GLU + shared expert, sigmoid gating |
| GptOss | `EPGptOssMoELayer` | 3 (gate+up+down) | Interleaved weights de-interleaved at init; grouped path uses a fused Triton clamped-SwiGLU (`src/kernels/fused_glu.py`) |
| GLM4 MoE Lite | `EPGlm4MoELayer` | 2 (gate_up+down) | SwiGLU, shared expert unaffected |
| Laguna | `EPLagunaMoELayer` | 2 (gate_up+down) | Subclasses the GLM4 layer; differs only in the top-k weight default and export key names |
| Bailing MoE | `EPBailingMoELayer` | 3 (gate+up+down) | Per-expert modules stacked into 3D tensors at init, shared expert |
| LFM2 MoE | `EPLfm2MoELayer` | 2 (gate_up+down) | Sigmoid routing, no shared expert |
| Inkling | `EPInklingMoELayer` | 2 (gate_up+down) | Fused GLU (GLM4 layout); joint routed+shared normalization |
| Gemma4 MoE | `EPGemma4MoELayer` | 2 (gate_up+down) | Fused Triton tanh-GeGLU; the router is a sibling module outside the wrapper |
| Mistral4 MoE | `EPMistral4MoELayer` | 2 (gate_up+down) | Fused Triton SwiGLU + group-routed softmax + shared expert |
| DeepSeek-V4 | `EPDeepseekV4MoELayer` | 2 (gate_up+down) | Fused GLU (clamped SwiGLU) + shared expert; hash + top-k routing |
| Zaya MoE | `EPZayaMoELayer` | 2 (gate_up+down) | Fused GLU + cross-layer EDA state |
| Cohere2 MoE | `EPCohere2MoELayer` | 2 (gate_up+down) | Fused GLU, sigmoid/softmax top-k-then-activate gating, averaged shared expert |
| GLM-5 Next | `EPGlm5NextMoELayer` | 2 (gate_up+down) | Fused `[E, 2M, H]` storage; fused Triton clamped SwiGLU (`swiglu_limit`) overrides the fused-GLU seam; shared expert |
| Step-3.7 Flash | `EPStep3p7MoELayer` | 2 (gate_up+down) | Fused `[E, 2M, H]` storage; per-layer **post-activation** fused Triton clamped SwiGLU (`swiglu_limits`, layers 43–44 only — unclamped layers take the base fused-SiLU combine); shared expert |

Families whose activation is a standard SiLU gate fuse the activation and the multiply into one Triton kernel, armed by a behavioral probe — see [GLM-4 → Fused SwiGLU](../models/glm4.md#fused-swiglu).

> [!NOTE]
> **Expert Tensor Parallelism**
>
> When `expert_tp_size > 1`, GptOss falls back to the loop path (interleaved weights cannot be pre-de-interleaved once TP-sharded). The other families use grouped GEMM regardless of ETP, on the 3-call separate-projection path (`gate_up_proj` is split into `gate_proj`/`up_proj` before the intermediate dim is sharded).

## Standalone grouped GEMM mode

`--use_grouped_gemm` applies EP wrappers with `ep_size=1` (all experts local, DeepEP dispatch/combine no-op), giving the grouped GEMM benefit on SM90+ without multi-GPU EP:

```bash
# Multi-GPU FSDP, no EP distribution (--nproc_per_node=1 for a single GPU)
torchrun --nproc_per_node=8 scripts/training/sft.py \
    examples/sft/qwen3_5/qwen3.5-35b-a3b-ultrachat-ep.yaml \
    --expert_parallel_size=1 --use_grouped_gemm=true
```

The wrapper keeps the packed-3D layout used at EP>1, so checkpoints stay shape-compatible across `ep_size`. Requires `torchrun`: an MoE with `use_grouped_gemm: true` under any `accelerate launch` is rejected at load (the wrappers need the mixin-managed FSDP2 path) — launch with torchrun or set `use_grouped_gemm: false`. Standalone benchmark: **3.07× over the naive loop** at 128 experts × 4096 tokens on B300 (naive 51.66 → grouped 158.66 TFLOPS).

`LigerExperts` (Liger's fused single-process expert FFN) is inert wherever an EP wrapper owns the routed experts — including this EP=1 path — because the wrapper replaces the very module the swap targets, and at EP>1 the rank holds only its expert slice. Non-expert Liger kernels (RMSNorm, RoPE, CrossEntropy/FusedLinearCE) still apply, and a family whose toolkit spec also names the dense and shared-expert MLPs keeps its fused GLU: [Liger Kernels](liger-kernels.md#ep-cp-tp-behavior).

## Weight layout

All EP MoE layers store expert weights in matmul convention `[E, K, N]` (so `A @ B[group]` works directly, no runtime transpose). Per-family init conversions:

- **Qwen3**: pre-fused `gate_up_proj [E, 2M, H]` chunked to gate/up and transposed to `[E, H, M]`; `down_proj [E, H, M]` → `[E, M, H]` (`EPQwen3MoELayer` raises at construction if the experts container is not pre-fused).
- **Fused-GLU families** (everything but Qwen3, Bailing and GptOss): transpose from `F.linear` convention (`gate_up_proj → [E, H, 2M]`, `down_proj → [E, M, H]`) in `_init_fused_glu_params`.
- **GptOss**: matmul convention natively; the interleaved `gate_up_proj [E, H, 2M]` (`[g0,u0,g1,u1,…]`) is de-interleaved at init into `gate_proj_gmm`/`up_proj_gmm [E, H, M]` (even/odd cols), avoiding stride-2 slicing on `grouped_mm` output.

Each family's `gather_expert_state_dict` reverses its conversion on save, so checkpoints stay compatible with the original architecture and non-EP inference. See [Checkpoints](../reference/checkpoints.md).

## Running benchmarks

```bash
# Isolated kernel: Qwen3-30B-A3B dims
python tests/gpu/profiling/benchmark_grouped_mm.py \
    --num_experts 128 --hidden 2048 --intermediate 768 --total_tokens 4096

# End-to-end with grouped GEMM (default on SM90+)
torchrun --nproc_per_node=2 \
    tests/gpu/profiling/benchmark_sft_ep.py --model qwen3-30b-a3b --ep 2 --seq 16384

# End-to-end loop baseline
torchrun --nproc_per_node=2 \
    tests/gpu/profiling/benchmark_sft_ep.py --model qwen3-30b-a3b --ep 2 --seq 16384 \
    --no_grouped_gemm
```

## Related pages

- [Expert Parallelism](../parallelism/expert-parallelism.md) — EP architecture and configuration
- [Expert Tensor Parallelism](../parallelism/expert-tensor-parallelism.md) — ETP sharding
- [torch.compile](torch-compile.md) — where compilation does and does not stack on the fused kernels
