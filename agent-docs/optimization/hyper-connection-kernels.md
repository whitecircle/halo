# Hyper-connection kernels

GLM-5 Next and DeepSeek-V4 run both hyper-connections of every decoder layer on Triton kernels on CUDA (`halo/kernels/hyper_connection.py`, installed over upstream's forwards at load by `halo/models/hyper_connections.py`). `HALO_HC_KERNELS=0` leaves upstream's classes untouched, for an A/B or a GPU where a kernel fails to build; a kernel that fails otherwise raises. An input off CUDA runs upstream's forward.

Upstream cycles the `[T, N, H]` streams through fp32 temporaries of `[T, N·H]` for the normalize, the projection and the collapse, then runs the Sinkhorn loop as 39 divisions on `[T, N, N]`. Each step that reads the streams is one launch here, the Sinkhorn projection is another, and no temporary of the streams' size exists: the graph holds the inputs, the projection's output and per-token normalizer, the final matrix and one scale per Sinkhorn step.

| Step | Kernel | Backward |
|---|---|---|
| Normalize + project | `project_streams` | `dx = r·dw@fn − x·r²·(dw·w)/(N·H)` for `r = 1/rms(x)`; `dfn` in a second launch only when `fn` trains |
| Sinkhorn | `sinkhorn_knopp` | for `y = x/s`: `dx = (dy − sum(dy·y))/s` and `x = y·s`, from the final matrix and the saved scales |
| Collapse | `collapse_streams` | `dx = pre·dy`, `dpre = sum(x·dy)` |
| Mix | `mix_streams` | each input's own closed form, only those `requires_grad` asks for |

The decoder layer's own forward is replaced for the mix, which upstream writes inline. Everything between the kernels (the sigmoids, the softmax, the `post`/`comb` casts to the streams' dtype) is upstream's line by line under autograd.

## Numbers

One B300, `hc_mult` 4, `H` 4096, 20 Sinkhorn iterations, bf16, frozen hyper-connection weights as under LoRA, 8192 tokens per call (`tests/gpu/profiling/benchmark_hyper_connection.py`):

- **Weights:** 2.26 ms forward and 4.58 ms backward upstream, 0.28 ms and 1.07 ms here; peak 3.31 GiB, 0.56 GiB here. A trained `fn` takes the backward to 1.55 ms against 4.89 ms.
- **Mix:** 1.83 ms forward and 2.71 ms backward upstream, 0.10 ms and 0.48 ms here; peak 1.00 GiB, 0.56 GiB here.
- **Launches:** a hyper-connection and its mix take 143 forward and 306 backward upstream, 16 and 14 here.

Both scale linearly in tokens; at 16384 the weights take 0.44 ms and 1.29 ms, the mix 0.20 ms and 0.81 ms. The first use compiles for about 2 s on a cold `TRITON_CACHE_DIR`, outside any DeepEP dispatch-to-combine window, so no warm-up runs ahead of it.

The tile is tuned for a production run's bf16 streams. On fp32 streams the weights forward takes 1.48 ms against upstream's 1.78 ms; the rest of the gain holds.

## Numerics

Every output and gradient matches upstream's autograd within the rounding of both against an fp64 reference (`tests/gpu/kernels/test_hyper_connection_kernels.py`). At bf16 the kernels compute in fp32 and round once at the store, where upstream rounds the product and the sum separately. Under the bf16 autocast of a LoRA or `fp32_non_ep_params` run the projection stays fp32, where upstream's `F.linear` rounds it to bf16.
