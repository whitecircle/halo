"""Compute kernels: MoE expert GEMMs, fused GLU, routing sort and permutes, histograms, hyper-connections, chunked fp32
log-probs, the hub kernel dispatch repair and the opt-in fp8/fp4 stack (``lowp/``); Liger's modules are in
``model_loading/liger/``."""
