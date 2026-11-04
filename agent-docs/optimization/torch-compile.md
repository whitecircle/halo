# torch.compile

On EP MoE `torch.compile` gains at most 1%, and in `reduce-overhead` mode it costs throughput, so the trainer falls back to `default` when `torch_compile_mode` is unset. On Qwen3-30B-A3B EP=2 at seq 16384, batch 1, `default` mode gains **1%** without Liger and **0.3%** on top of it, `reduce-overhead` is **7% slower** alone and **10% slower** on top of Liger, and Liger alone adds **27%** ([Benchmark results](#benchmark-results)). At that shape keep `torch_compile` off and [Liger kernels](liger-kernels.md) on.

What fusion buys and what it does not: [GPU Training Theory §5](../reference/gpu-training-theory.md#what-fusion-does-not-buy).

## Expert activations are not compiled

**No expert activation is wrapped in `torch.compile`.** Every GLU combine on the roster is a Triton kernel taking its shape-dependent and numeric arguments at runtime (`halo/kernels/fused_glu.py`): the plain SwiGLU / tanh-GELU pair, DeepSeek-V4 and GLM-5 Next's clamped SwiGLU (`GluCombine.clamped_silu`), Step-3.7 Flash's post-activation clamp (`GluCombine.clamped_silu(..., clamp_activated=True)`), and GptOss's `GluCombine.gptoss` on the loop, grouped-GEMM and ETP paths alike.

A `torch.compile`d combine taking a bound (or `alpha`) as a Python float is re-traced as a symbolic input once the token count goes dynamic, and the inductor C++ backend then serves **every later value from the first graph**: a silently wrong clamp on every layer after the first.

The clamped combines latch into `_fused_glu_mul`, the GLU-combine seam every base expert path reads. DeepSeek-V4 arms its combine only when the block's activation probes as exactly SiLU and falls back to the family's eager clamped GLU otherwise; Step-3.7 takes the same probe but only on a layer whose `swiglu_limit` is finite — its unclamped layers keep the plain fused SiLU combine. GLM-5 Next's experts hardcode clamp-then-SiLU structurally, so its combine runs unconditionally.

The EP routing sort stays eager too (`torch.compiler.disable` on the counting sort in `halo/kernels/moe_route_sort.py`, which every grouped-GEMM path runs on CUDA): its three Triton kernels are already fused, and traced, the slot counts that size it (the dispatch's host-side receive counts) would become graph inputs that change every step. A compiled EP layer breaks its graph there.

The low-precision weight quantize/dequantize round-trip *is* compiled, for the power-of-two-scale formats (mxfp8/mxfp4; nvfp4's weight stays eager), with a permanent eager fallback on any compile or runtime failure — see [Low-Precision MoE](low-precision-moe-kernels.md).

## Chunked log-prob head

The chunked log-prob head's tile math is compiled on CUDA, one graph per direction serving every tile, with no fallback: a compile failure raises. Its temperature, logit scale and softcap are Python floats traced as symbolic inputs, the case [above](#expert-activations-are-not-compiled): `tests/cpu/kernels/test_head_logprob_tiles.py` holds each new value to the eager result on the C++ backend without a recompile, and `tests/gpu/kernels/test_chunked_logprob_precision.py` on CUDA. What compiles a graph, and the limit of 8 per tile function: [Chunked log-probs](../training-methods/async-grpo/performance.md#chunked-log-probs).

## Whole-model compilation (opt-in)

```yaml
torch_compile: true
torch_compile_mode: default
torch_compile_backend: inductor
```

Or via CLI: `torchrun ... scripts/training/sft.py --torch_compile=true --torch_compile_mode=default`.

`torch_compile_mode` and `torch_compile_backend` are HF `TrainingArguments` fields, both `None` by default. Left unset, the mixin's compile call (`_apply_torch_compile`) uses `default` / `inductor`. On an accelerate-managed launch the mixin compiles in `__init__`, before Accelerate's `prepare`: a single-process `prepare` does not compile it again, while a multi-process DDP one wraps it and compiles the DDP wrapper with its `TorchDynamoPlugin`, which takes the same `torch_compile_mode` and also resolves an unset one to `default`. `reduce-overhead` (CUDA graphs) runs only when set explicitly.

On the mixin-managed path `DistributedTrainerMixin` applies compilation **after FSDP wrapping** (`_apply_torch_compile`, at the end of
`_configure_distributed_modes`) — compiling before FSDP fails because FSDP restructures the parameter layout. To
stop HF Trainer from compiling too early, the mixin clears Accelerate's `ACCELERATE_DYNAMO_*` env vars while
building its plain accelerator and re-applies `torch.compile` itself, assigning both `self.model` and
`self.model_wrapped` (the training loop runs the latter).

Pipeline parallelism refuses `torch_compile`: the schedule captures the stage module at setup, so a compiled
wrapper installed afterwards never runs ([PP rejected combinations](../parallelism/pipeline-parallelism.md#rejected-combinations)).

Compiling costs steps up front: `torch.compile()` returns immediately, the **first step compiles**
(6–15 s on Qwen3-30B-A3B EP=2, against 4–5 s eager), and each rank recompiles for dynamic shapes when it
first meets a second sequence length, a 2–7 s step each time. Under `reduce-overhead` slow steps keep coming
after that (up to 2.4 s against a 1.5 s median), which the `default` mode, without CUDA graphs, does not
show.

## Benchmark results

Qwen3-30B-A3B-Instruct-2507 (128 experts, top_k=8), 2× B300, EP=2, seq 16384, batch 1, bf16, FA4, gradient checkpointing. 20 steps with the first 5 excluded, so the recompiles fall outside the window; one process per cell, mean of 2–3 runs, spread = (max − min) / mean. Measured on the Blackwell image, Halo 1.1.0 or earlier, with the [benchmark below](#running-benchmarks):

| Mode | Liger | Compile | Step (s) | tokens/s/GPU | Spread | Peak mem (GiB) |
|------|:-----:|:-------:|---------:|-------------:|-------:|--------------:|
| `neither` | OFF | OFF | 1.44 | 11,399 | 0.4% | 128.5 |
| `liger_only` | ON | OFF | **1.13** | **14,490** (+27%) | 0.0% | 126.6 |
| `compile_only` | OFF | `default` | 1.42 | 11,536 (+1%) | 0.4% | 126.9 |
| `liger_compile` | ON | `default` | 1.13 | 14,539 (+0.3% vs `liger_only`) | 0.5% | 149.8 |
| `compile_only` | OFF | `reduce-overhead` | 1.54 | 10,600 (−7%) | 5.4% | 126.9 |
| `liger_compile` | ON | `reduce-overhead` | 1.26 | 13,024 (−10% vs `liger_only`) | 7.5% | 135.8 |

Liger is the lever: +27% at 2 GiB less memory. Compile adds 0.3% to it in `default` mode and takes 10% away in `reduce-overhead`, whose slow steps also make the result noisy (5–8% between identical runs, against ≤ 0.5% for every other cell). With Liger on, compile also raises peak memory by 9–23 GiB. Peaks are from warm-cache runs; each compiled cell's first run, on a cold compile cache, peaked 14–37 GiB higher.

## Graph breaks

Compile fuses the spans between graph breaks (norms, projections), not across them:

- **DeepEP dispatch/combine** — splits the graph at every MoE layer.
- **Flash Attention** — opaque; the compiler cannot fuse across FA boundaries. FA4's `autograd.Function` and its cute-DSL launcher do not trace, so each FA4 call also breaks the graph.
- **Hyper-connection kernels** — `torch.compiler.disable`, on GLM-5 Next and DeepSeek-V4: already fused, and traced, the token count would become a graph input.
- **Gradient checkpointing** — EP, CP and every MoE force `use_reentrant=True`, which adds graph breaks; on a dense model outside EP/CP the config's `use_reentrant` stands.
- **TP** — no break: the TP linears run on their local shards, and they and their collectives trace without a graph break; with the replicated-grad hooks Halo retargets after transformers' `tp_plan` load, the rest of a dense decoder layer's DTensor dispatch compiles into aten ops plus functional collectives. transformers' own module-level backward hooks would break it.
- **CP (Ulysses)** — no break of its own: each exchange is one `all_to_all_single` inside an autograd function. The reentrant checkpointing CP forces (above) still breaks the graph.

## Compile caches at scale

Every DP/EP rank compiles the same graphs. Inductor keeps them in `TORCHINDUCTOR_CACHE_DIR` (default `$TMPDIR/torchinductor_<user>`, `/tmp` when `TMPDIR` is unset, so the launch recipe's `TMPDIR="$D/tmp"` already puts it on the verified scratch volume), and each process compiles with up to 32 worker threads, so an 8-GPU node starts up to 256 compile workers at warm-up. On a large job:

- where `$D` is one NFS/Lustre directory shared by every node, point `TORCHINDUCTOR_CACHE_DIR` at a node-local directory: the cache's file-locked writes on a shared directory put every rank of the job on the same few keys;
- set `TORCHINDUCTOR_COMPILE_THREADS` to the node's CPUs divided by its local ranks.

## Running benchmarks

`tests/gpu/profiling/benchmark_torch_compile.py` runs one cell of the Liger × compile matrix per process (`--mode neither|liger_only|compile_only|liger_compile`, required). Liger patches the model classes process-wide, so a second cell in the same process would inherit the first one's kernels. `--compile_mode` picks the compile mode (default `default`, the trainer's fallback):

```bash
for mode in neither liger_only compile_only liger_compile; do
  torchrun --nproc_per_node=2 \
      tests/gpu/profiling/benchmark_torch_compile.py --model qwen3-30b-a3b --ep 2 --seq 16384 \
      --steps 20 --warmup 5 --mode "$mode"
done

# the reduce-overhead rows: add --compile_mode reduce-overhead; 8-GPU: --nproc_per_node=8 --ep 8
```
