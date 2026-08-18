# torch.compile

On EP MoE, `torch.compile` (inductor) reaches roughly the same speedup as [Liger kernels](liger-kernels.md) (~+30% throughput, −9 GB) and **composes with them**: compile on top of Liger holds the gain. DeepEP all-to-all and Flash Attention break the compiled graph at every MoE/attention boundary, but inductor compiles the spans between breaks (norms, projections) well enough to pay off. Liger is the default (no per-shape warmup cost); add `torch_compile` when you can absorb the first-step compile latency. What fusion buys and what it does not: [GPU Training Theory §5](../reference/gpu-training-theory.md#what-fusion-does-not-buy).

## Expert activations are not compiled

**No expert activation is wrapped in `torch.compile`.** Every GLU combine on the roster is a Triton kernel taking its shape-dependent and numeric arguments at runtime (`src/kernels/fused_glu.py`): the plain SwiGLU / tanh-GELU pair, DeepSeek-V4 and GLM-5 Next's clamped SwiGLU (`fused_clamped_silu_mul`), Step-3.7 Flash's post-activation clamp (`fused_silu_then_clamp_mul`), and GptOss's `fused_gptoss_glu` on the loop, grouped-GEMM and ETP paths alike. A `torch.compile`d combine taking a bound (or `alpha`) as a Python float is re-traced as a symbolic input once the token count goes dynamic, and the inductor C++ backend then serves **every later value from the first graph** — a silently wrong clamp on every layer after the first.

The clamped combines latch into the shared `_glu_combine` seam. DeepSeek-V4 arms its combine only when the block's activation probes as exactly SiLU and falls back to the family's eager clamped GLU otherwise; Step-3.7 takes the same probe but only on a layer whose `swiglu_limit` is finite — its unclamped layers keep the plain fused SiLU combine. GLM-5 Next's experts hardcode clamp-then-SiLU structurally, so its combine runs unconditionally.

The low-precision weight quantize/dequantize round-trip *is* compiled, for the power-of-two-scale formats (mxfp8/mxfp4; nvfp4's weight stays eager), with a permanent eager fallback on any compile or runtime failure — see [Low-Precision MoE](low-precision-moe-kernels.md).

## Whole-model compilation (opt-in)

```yaml
torch_compile: true
torch_compile_mode: reduce-overhead
torch_compile_backend: inductor
```

Or via CLI: `torchrun ... scripts/training/sft.py --torch_compile=true --torch_compile_mode=reduce-overhead`.

`torch_compile_mode` and `torch_compile_backend` are HF `TrainingArguments` fields, both `None` by default. Left unset, the mixin's own compile call falls back to `reduce-overhead` / `inductor` (`_apply_torch_compile`); the accelerate-managed path (no custom parallelism) leaves the mode at inductor's default instead.

`DistributedTrainerMixin` applies compilation **after FSDP wrapping** (`_apply_torch_compile`, at the end of
`_setup_distributed_modes`) — compiling before FSDP fails because FSDP restructures the parameter layout. To
stop HF Trainer from compiling too early, the mixin clears Accelerate's `ACCELERATE_DYNAMO_*` env vars while
building its plain accelerator and re-applies `torch.compile` itself, assigning both `self.model` and
`self.model_wrapped` (the training loop runs the latter).

**Rejected under pipeline parallelism** (itself [not yet available in this release](../parallelism/pipeline-parallelism.md)):
a pipeline schedule captures the stage module at setup, so a compiled wrapper installed afterwards would
never run — `torch_compile: true` with PP raises. No other parallelism mode blocks it.

Two costs come with it: `torch.compile()` returns immediately and the **first forward pass compiles**
(2–5 min for MoE models), and `reduce-overhead` allocates extra GPU memory for its recorded CUDA-graph
command streams.

## Benchmark results

Qwen3-30B-A3B (128 experts, top_k=8), 2× B300 (SM103), EP=2, seq 16384, BF16, FA4, gradient checkpointing, 12 steps / 2 warmup, batch 1:

| Mode | Liger | Compile | Step (s) | tokens/s/GPU | Peak mem (GB) |
|------|:-----:|:-------:|---------:|-------------:|--------------:|
| `neither` | OFF | OFF | 1.70 | 9,610 | 128.6 |
| `liger_only` | ON | OFF | 1.31 (−23%) | 12,467 (+30%) | 119.4 |
| `compile_only` | OFF | ON | 1.29 (−24%) | 12,676 (+32%) | 119.4 |
| `liger_compile` | ON | ON | **1.30 (−24%)** | **12,608 (+31%)** | 119.5 |

Liger, compile, and the two stacked all land within ~2% of each other (+30% to +32%, −9 GB) — they target the same non-EP spans, so stacking adds little. These are batch-1 (communication-bound) numbers: DeepEP all-to-all dominates and varies run-to-run, so the throughput percentages are directional. Measure at batch ≥ 4 for a stable comparison. The −9 GB memory saving holds at any batch.

## Why the speedup is bounded on EP MoE

Graph breaks cap how much compile can fuse — it speeds up the spans between them (norms, projections), not across them:

- **DeepEP dispatch/combine** — splits the graph at every MoE layer.
- **Flash Attention** — opaque; the compiler cannot fuse across FA boundaries.
- **Gradient checkpointing** — EP and CP force `use_reentrant=True`, which adds graph breaks; without them the config's `use_reentrant` stands.
- **TP DTensor** — sharded-op dispatch breaks the graph at every sharded operation.
- **CP (Ulysses)** — all-to-all in every attention layer breaks the graph at every block.

These breaks keep compile's ceiling near Liger's rather than far above it. The expert activation, the one hot op Liger does not cover under an EP wrapper, is already a hand-written Triton kernel — compile has nothing left to win there.

## Running benchmarks

`tests/gpu/profiling/benchmark_torch_compile.py` runs the 2×2 Liger × Compile matrix (`--mode neither|compile_only|liger_only|liger_compile`, default all four):

```bash
torchrun --nproc_per_node=2 \
    tests/gpu/profiling/benchmark_torch_compile.py --model qwen3-30b-a3b --ep 2 --seq 16384 --steps 12

# 8-GPU
torchrun --nproc_per_node=8 \
    tests/gpu/profiling/benchmark_torch_compile.py --model qwen3-30b-a3b --ep 8 --seq 16384 --steps 12
```
