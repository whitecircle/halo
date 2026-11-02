"""Compute kernels: MoE expert GEMMs, fused GLU combines, the routing sort and token permutes, routing histograms,
chunked fp32 log-probs, the hub kernel dispatch repair, FLA's launch options, the FA4 / Triton JIT cache anchoring and
the opt-in fp8/fp4 stack (``lowp/``); Liger's fused modules live under ``model_loading/liger/``."""
