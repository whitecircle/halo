# Hyper-connection kernels

Two families of kernels: the mHC residual of GLM-5 Next and DeepSeek-V4 (below), and [Flash-Next's gated residual](#flash-next-gated-residual).

GLM-5 Next and DeepSeek-V4 run both hyper-connections of every decoder layer, and DeepSeek-V4 its final `hc_head`, on Triton kernels on CUDA (`halo/kernels/hyper_connection.py`, installed over upstream's forwards at load by `halo/models/hyper_connections.py`). `HALO_HC_KERNELS=0` leaves upstream's classes untouched, for an A/B or a GPU where a kernel fails to build; a kernel that fails otherwise raises. An input off CUDA or in fp64 runs upstream's forward.

Upstream cycles the `[T, N, H]` streams through fp32 temporaries of `[T, N·H]` for the normalize, the projection and the collapse, then runs the Sinkhorn loop as 39 divisions on `[T, N, N]`. Each step that reads the streams is one launch here, the Sinkhorn projection is another, and no temporary of the streams' size exists: the graph holds the inputs, the projection's output and per-token normalizer, the final matrix and one scale per Sinkhorn step.

| Step | Kernel | Backward |
|---|---|---|
| Normalize + project | `project_streams` | `dx = r·dw@fn − x·r²·(dw·w)/(N·H)` for `r = 1/rms(x)`; `dfn` in a second launch only when `fn` trains |
| Sinkhorn | `sinkhorn_knopp` | for `y = x/s`: `dx = (dy − sum(dy·y))/s` and `x = y·s`, from the final matrix and the saved scales |
| Collapse | `collapse_streams` | `dx = pre·dy`, `dpre = sum(x·dy)` |
| Mix | `mix_streams` | each input's own closed form |

The decoder layer's own forward is replaced for the mix, which upstream writes inline. Everything between the kernels (the sigmoids, the softmax, the `post`/`comb` casts to the streams' dtype) is upstream's line by line under autograd.

## Numbers

One B300, `hc_mult` 4, `H` 4096, 20 Sinkhorn iterations, bf16, frozen hyper-connection weights as under LoRA, 8192 tokens per call (`tests/gpu/profiling/benchmark_hyper_connection.py`):

- **Weights:** 2.26 ms forward and 4.58 ms backward upstream, 0.28 ms and 1.07 ms here; peak 3.31 GiB, 0.56 GiB here. A trained `fn` takes the backward to 1.55 ms against 4.89 ms.
- **Mix:** 1.83 ms forward and 2.71 ms backward upstream, 0.10 ms and 0.48 ms here; peak 1.00 GiB, 0.56 GiB here.
- **Launches:** a hyper-connection and its mix take 143 forward and 306 backward upstream, 16 and 14 here.

Each takes a fixed launch cost plus a per-token term, so doubling the tokens costs less than double; at 16384 the weights take 0.44 ms and 1.29 ms, the mix 0.20 ms and 0.81 ms. The first use compiles for about 2 s on a cold `TRITON_CACHE_DIR`, outside any DeepEP dispatch-to-combine window, so no warm-up runs ahead of it.

In a whole step, on GLM-5.3-Flash's LoRA at `ep8` and 8,192 tokens on a per-part clock, the weights take 1.31 s against 7.17 (5.5×) and the mix 0.44 s against 3.39 (7.7×) ([GLM-5 Next → Throughput](../models/glm5-next.md#throughput)). No DeepSeek-V4 release has run on these kernels, so that family has no rate. Tiny DeepSeek-V4 and GLM-5 Next models hold them to upstream's forwards through reentrant recompute, frozen weights, LoRA, FSDP2 and pipeline stages.

The tile is tuned for a production run's bf16 streams. On fp32 streams the weights forward takes 1.48 ms against upstream's 1.78 ms; the rest of the gain holds.

## Numerics

Every output and gradient matches upstream's autograd within the rounding of both against an fp64 reference (`tests/gpu/kernels/test_hyper_connection_kernels.py`). At bf16 the kernels compute in fp32 and round once at the store, where upstream rounds the product and the sum separately. Under the bf16 autocast of a LoRA or `fp32_non_ep_params` run the projection stays fp32, where upstream's `F.linear` rounds it to bf16.

## Flash-Next gated residual

Qwen3.8-Flash-Next's hyper-connection (`Qwen4ExpTextGatedResidual`, two per decoder layer) is a different block: no Sinkhorn matrix, projection or collapse. It norms each of the four streams on its own, mixes them into the block input through a low-rank gate (`down` → `silu(·/N)` → `up` → sigmoid, then the mean over streams of gate × normed), and returns injection gates `2·sigmoid(inject(normed)/N)`; the block's output re-enters every stream as `x + out·gate`. Only that injection is the mHC mix at `comb = I`.

It runs on four small Triton kernels, with cuBLAS and autograd doing every linear (`halo/kernels/gated_residual.py`, `inject_streams` in `halo/kernels/hyper_connection.py`, patched over upstream's forward by `halo/models/families/qwen4_exp/gated_residual.py`). `HALO_GATED_RESIDUAL_KERNELS` is on by default, and `0` restores upstream's forward; `HALO_HC_KERNELS` keeps meaning GLM-5 Next and DeepSeek-V4.

| Step | Kernel | Backward |
|---|---|---|
| Per-stream norm | `grouped_rms_norm` | `dx = r·dy·(1+w) − x·r³·mean(dy·(1+w)·x)` from the saved `r`; `dw` in fp32 row chunks only when the norm weight trains |
| Gate and mix | `gate_mix` | recomputes `up` from `down`, so the graph holds `normed` and `down`; `dup` and the direct `dnormed` in one launch, `ddown` and `dw_up` as matmuls |
| Injection | `inject_streams` | `dres = dy`, `dblock = sum(dy·gate)`, `dgate = sum(dy·block)` |

Autograd sums `normed`'s three gradients (the mix, the down projection, the injection logits). Every kernel works one row per program with no atomic, so a run repeats bit for bit, and `full_determinism` needs no fallback. The hidden size is a runtime argument, so one compile per stream dtype and count serves every pack length; `finalize_loaded_model` compiles them at load for the weights' dtype, rank-local and unfenced (a run that sets the knob on some ranks only compiles on those).

A module the kernels cannot serve runs upstream's saved forward, logging the reason once on rank 0: non-CUDA or empty input, a stream dtype other than bf16, fp16 or fp32 (or off the active autocast's), mix weights in another dtype than the streams (`fp32_non_ep_params` runs), a bias, a subclassed `nn.Linear`, a forward hook on the norm or a projection, a parameter that is not a plain tensor, or a norm group other than the hidden size; the injection falls back to the eager sum, logging once, where the block output, the gates and the streams differ in dtype. Frozen and trained weights both run on the kernels. A stream count off a power of two is padded.

The layer is local, so EP, ETP, FSDP2 (the layer's pre-forward hook hands it plain parameters), pipeline stages, LoRA, and reentrant and non-reentrant checkpointing run it as they run upstream's; TP and CP stay refused for the family. With the knob on, a layer without PLE takes the toolkit's per-document layer body too, so its injection runs on the kernel; a cached call keeps upstream's layer.

`inject_streams` equals `mix_streams` at an identity `comb` except for that rounding, which is why it has its own kernels; a GPU test pins the two in fp32. The injection rounds the product and the sum to the streams' dtype, as upstream does, so the forward matches upstream's bit for bit apart from the norm's and the mix's reduction order. Tests hold every output and gradient to the larger of a dtype floor and 1.5× upstream's own error against fp64 (`tests/gpu/kernels/test_gated_residual_kernels.py`).

## Flash-Next gated residual numbers

One B300, `hidden_size` 2560, four streams, `hc_lowrank` 320, bf16, one gated residual plus its injection per call, 8192 tokens, upstream's forward against the kernels (`tests/gpu/profiling/benchmark_gated_residual.py`):

- **Frozen weights:** 1.56 ms forward and 3.21 ms backward upstream, 0.50 ms and 1.12 ms here; the graph one call holds 685 MiB against 205 MiB, its peak 1920 MiB against 865 MiB.
- **Trained weights:** the backward takes 3.61 ms upstream and 2.21 ms here, the norm weight's gradient and the two matmuls for the mix's costing the difference; the graph 1010 MiB against 205 MiB.
- **16384 tokens:** 3.00 ms and 6.00 ms upstream, 0.92 ms and 1.77 ms here (frozen); trained backward 6.67 ms against 3.84 ms.

The release runs 96 of these per step (two on each of 48 layers), so the frozen numbers are about 0.3 s of an 8192-token step.

On the real checkpoint the gated residual alone takes the LoRA step at 16k packed tokens from 28.95 s to 23.66 s (1.22×) and the peak from 98.71 to 96.52 GB, on the same loss trajectory ([Flash-Next kernel defaults](../models/qwen4_exp.md#kernel-defaults)).
