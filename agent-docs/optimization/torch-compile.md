# torch.compile

On EP MoE, `torch.compile` buys nothing, and in `reduce-overhead` mode — the trainer's fallback when `torch_compile_mode` is unset — it costs throughput: keep it off and keep [Liger kernels](liger-kernels.md) on. On Qwen3-30B-A3B EP=2 at seq 16384, Liger adds **27%**; compile in `default` mode ties eager, and `reduce-overhead` is **7% slower** alone and **10% slower** on top of Liger ([Benchmark results](#benchmark-results)). DeepEP all-to-all and Flash Attention break the graph at every MoE and attention boundary, so inductor compiles only the short spans between breaks, which Liger already fuses.

What fusion buys and what it does not: [GPU Training Theory §5](../reference/gpu-training-theory.md#what-fusion-does-not-buy).

## Expert activations are not compiled

**No expert activation is wrapped in `torch.compile`.** Every GLU combine on the roster is a Triton kernel taking its shape-dependent and numeric arguments at runtime (`src/kernels/fused_glu.py`): the plain SwiGLU / tanh-GELU pair, DeepSeek-V4 and GLM-5 Next's clamped SwiGLU (`fused_clamped_silu_mul`), Step-3.7 Flash's post-activation clamp (`fused_silu_then_clamp_mul`), and GptOss's `fused_gptoss_glu` on the loop, grouped-GEMM and ETP paths alike.

A `torch.compile`d combine taking a bound (or `alpha`) as a Python float is re-traced as a symbolic input once the token count goes dynamic, and the inductor C++ backend then serves **every later value from the first graph**: a silently wrong clamp on every layer after the first.

The clamped combines latch into the shared `_glu_combine` seam. DeepSeek-V4 arms its combine only when the block's activation probes as exactly SiLU and falls back to the family's eager clamped GLU otherwise; Step-3.7 takes the same probe but only on a layer whose `swiglu_limit` is finite — its unclamped layers keep the plain fused SiLU combine. GLM-5 Next's experts hardcode clamp-then-SiLU structurally, so its combine runs unconditionally.

The low-precision weight quantize/dequantize round-trip *is* compiled, for the power-of-two-scale formats (mxfp8/mxfp4; nvfp4's weight stays eager), with a permanent eager fallback on any compile or runtime failure — see [Low-Precision MoE](low-precision-moe-kernels.md).

## Whole-model compilation (opt-in)

```yaml
torch_compile: true
torch_compile_mode: default
torch_compile_backend: inductor
```

Or via CLI: `torchrun ... scripts/training/sft.py --torch_compile=true --torch_compile_mode=default`.

`torch_compile_mode` and `torch_compile_backend` are HF `TrainingArguments` fields, both `None` by default. Left unset, the mixin's own compile call falls back to `reduce-overhead` / `inductor` (`_apply_torch_compile`); the accelerate-managed path (no custom parallelism) leaves the mode at inductor's default instead.

`DistributedTrainerMixin` applies compilation **after FSDP wrapping** (`_apply_torch_compile`, at the end of
`_setup_distributed_modes`) — compiling before FSDP fails because FSDP restructures the parameter layout. To
stop HF Trainer from compiling too early, the mixin clears Accelerate's `ACCELERATE_DYNAMO_*` env vars while
building its plain accelerator and re-applies `torch.compile` itself, assigning both `self.model` and
`self.model_wrapped` (the training loop runs the latter).

**Rejected under pipeline parallelism** (itself [not yet available in this release](../parallelism/pipeline-parallelism.md)):
a pipeline schedule captures the stage module at setup, so a compiled wrapper installed afterwards would
never run — `torch_compile: true` with PP raises. No other parallelism mode blocks it.

Compiling costs steps up front: `torch.compile()` returns immediately, the **first step compiles**
(6–15 s on Qwen3-30B-A3B EP=2, against 4–5 s eager), and each rank recompiles for dynamic shapes when it
first meets a second sequence length, a 2–7 s step each time. Under `reduce-overhead` slow steps keep coming
after that (up to 2.4 s against a 1.5 s median), which the `default` mode, without CUDA graphs, does not
show.

## Benchmark results

Qwen3-30B-A3B-Instruct-2507 (128 experts, top_k=8), 2× B300, EP=2, seq 16384, batch 1, bf16, FA4, gradient checkpointing. 20 steps with the first 5 excluded, so the recompiles fall outside the window; one process per cell, mean of 2–3 runs, spread = (max − min) / mean. Measured 2026-10-03 on the Blackwell image, training code at commit 0bc3a22a5, with the [benchmark below](#running-benchmarks):

| Mode | Liger | Compile | Step (s) | tokens/s/GPU | Spread | Peak mem (GiB) |
|------|:-----:|:-------:|---------:|-------------:|-------:|--------------:|
| `neither` | OFF | OFF | 1.44 | 11,399 | 0.4% | 128.5 |
| `liger_only` | ON | OFF | **1.13** | **14,490** (+27%) | 0.0% | 126.6 |
| `compile_only` | OFF | `default` | 1.42 | 11,536 (+1%) | 0.4% | 126.9 |
| `liger_compile` | ON | `default` | 1.13 | 14,539 (+0.3% vs `liger_only`) | 0.5% | 149.8 |
| `compile_only` | OFF | `reduce-overhead` | 1.54 | 10,600 (−7%) | 5.4% | 127.0 |
| `liger_compile` | ON | `reduce-overhead` | 1.26 | 13,024 (−10% vs `liger_only`) | 7.5% | 135.8 |

Liger is the lever: +27% at 2 GiB less memory. Compile adds nothing to it in `default` mode and takes 10% away in `reduce-overhead`, whose slow steps also make the result noisy (5–8% between identical runs, against ≤ 0.5% for every other cell). With Liger on, compile also raises peak memory by 9–23 GiB. Peaks are from warm-cache runs; each compiled cell's first run, on a cold compile cache, peaked 14–37 GiB higher.

## Why compile does not pay off on EP MoE

Graph breaks cap what compile can fuse — it compiles the spans between them (norms, projections), not across them:

- **DeepEP dispatch/combine** — splits the graph at every MoE layer.
- **Flash Attention** — opaque; the compiler cannot fuse across FA boundaries.
- **Gradient checkpointing** — EP, CP and every MoE force `use_reentrant=True`, which adds graph breaks; on a dense model outside EP/CP the config's `use_reentrant` stands.
- **TP DTensor** — sharded-op dispatch breaks the graph at every sharded operation.
- **CP (Ulysses)** — all-to-all in every attention layer breaks the graph at every block.

Those spans are the ones Liger already fuses. The expert activation, the one hot op Liger does not cover under an EP wrapper, is already a hand-written Triton kernel — compile has nothing left to win there.

## Running benchmarks

`tests/gpu/profiling/benchmark_torch_compile.py` runs one cell of the Liger × compile matrix per process (`--mode neither|liger_only|compile_only|liger_compile`, required). Liger patches the model classes process-wide, so a second cell in the same process would inherit the first one's kernels. `--compile_mode` picks the compile mode (default `reduce-overhead`, the trainer's fallback):

```bash
for mode in neither liger_only compile_only liger_compile; do
  torchrun --nproc_per_node=2 \
      tests/gpu/profiling/benchmark_torch_compile.py --model qwen3-30b-a3b --ep 2 --seq 16384 \
      --steps 20 --warmup 5 --mode "$mode"
done

# the default-mode rows: add --compile_mode default; 8-GPU: --nproc_per_node=8 --ep 8
```
