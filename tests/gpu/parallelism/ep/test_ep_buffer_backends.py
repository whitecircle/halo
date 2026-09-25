#!/usr/bin/env python
"""
DeepEP transport-backend equivalence: legacy Buffer (V1) == ElasticBuffer (V2).

``ep_buffer_backend`` selects the EP all-to-all transport: ``elastic`` (default, DeepEP V2
ElasticBuffer over NCCL Gin) or ``legacy`` (DeepEP V1 CUDA-IPC Buffer, intranode). The two route
tokens identically, so on the same input they must produce the same forward loss and the same
expert/router gradients. (Long-context equivalence past the 2³⁰ arena bound lives in
``test_ep_long_context.py``; this is the short-sequence gradient-equivalence guard.)

Loads the SAME MoE model at EP=2 once per backend, runs forward+backward on identical input, and
checks the loss matches and the expert + router gradients match (cosine ~1). This both proves V1 can
substitute V2 for intranode EP and guards the V2 path against regressions in the shared dispatcher.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_ep_buffer_backends.py

Requirements:
    - 2x GPUs with >=80GB memory each, single node (legacy is intranode-only)
    - DeepEP installed; Model: unsloth/gpt-oss-20b-BF16 (auto-downloaded)
"""

import torch
from transformers import AutoTokenizer

from src.distributed.expert_parallel.base_layer import has_grouped_mm
from src.distributed.expert_parallel.dispatcher import _ElasticBackend, _LegacyBackend
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from tests.common.distributed import ensure_model_downloaded
from tests.common.ep_reference import ep_layers, find_router_weight, fixed_chat_batch, full_grad
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.utils import cleanup_memory, cos_sim, log, log_all

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 2
SEQ_LEN = 256
SEED = 42

LOSS_ABS_TOL = 5e-3  # same routing + transport → loss matches to bf16 reduction-order noise
GRAD_COSINE_MIN = 0.999


def named_grads(model) -> dict[str, tuple[str, torch.Tensor | None]]:
    """This rank's first local expert weight grad and the first router weight grad, flattened fp32.

    Each parameter is chosen by name before its gradient is read, so a severed one stays None here
    instead of falling through to the next layer's. The caller fails it after the last collective:
    a missing gradient can be rank-specific (an expert bank no token reached), and an assert here
    would leave the peer rank inside the other backend's run.
    """
    expert_name, expert_weight = ep_layers(model)[0].expert_named_params()[0]
    router_name, router_weight = find_router_weight(model)
    return {
        role: (name, None if param.grad is None else full_grad(param).flatten())
        for role, (name, param) in (("expert", (expert_name, expert_weight)), ("router", (router_name, router_weight)))
    }


def run_backend(backend, tokenizer, local_rank):
    device = f"cuda:{local_rank}"
    log(f"\n{'=' * 70}\nBACKEND ep_buffer_backend={backend} (EP={EP_SIZE})\n{'=' * 70}")
    parallelism_config = ParallelismConfig(
        ep_size=EP_SIZE, use_grouped_gemm=has_grouped_mm(), ep_buffer_backend=backend
    )
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_liger_kernel=True,
    )
    model.train()
    input_ids, attention_mask, labels = fixed_chat_batch(tokenizer, SEQ_LEN, device, seed=SEED, broadcast=True)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    loss_val = loss.item()
    grads = named_grads(model)
    log_all(f"  [{backend}] loss={loss_val:.6f}")

    # confirm the dispatcher actually selected the requested backend
    want = _LegacyBackend if backend == "legacy" else _ElasticBackend
    backends = [
        m.dispatcher.backend
        for _, m in model.named_modules()
        if hasattr(m, "dispatcher") and m.dispatcher.backend is not None
    ]
    seen = {type(b).__name__ for b in backends}
    backend_ok = seen == {want.__name__}
    log(f"  [{backend}] dispatcher backends in use: {seen} (expected {want.__name__})")

    # BOTH transports serve every MoE layer from one buffer. Per-layer is what a wide-EP job cannot
    # afford — the V2 receive arena is ~4 GiB each, and V1's flat ~100 MiB reaches 5.9 GiB over 60
    # layers. A regression to one buffer per layer makes this len(buffers) == len(live) instead of 1.
    live = [b for b in backends if b.buffer is not None]
    buffers = {id(b.buffer) for b in live}
    shared_ok = len(live) > 1 and len(buffers) == 1
    log(f"  [{backend}] {len(live)} dispatching layers share {len(buffers)} buffer(s)")

    del model, outputs, loss
    cleanup_memory()
    return {"loss": loss_val, "grads": grads, "backend_ok": backend_ok, "shared_ok": shared_ok}


def run(ctx) -> dict:
    ensure_model_downloaded(MODEL_NAME, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    v2 = run_backend("elastic", tokenizer, ctx.local_rank)
    barrier()
    v1 = run_backend("legacy", tokenizer, ctx.local_rank)
    barrier()

    for backend, result in (("elastic", v2), ("legacy", v1)):
        for name, grad in result["grads"].values():
            assert grad is not None, f"[{backend}] {name} has no gradient: backward did not reach it"
    expert_name, v1_expert = v1["grads"]["expert"]
    router_name, v1_router = v1["grads"]["router"]
    expert_cos = cos_sim(v1_expert, v2["grads"]["expert"][1], label=expert_name)
    router_cos = cos_sim(v1_router, v2["grads"]["router"][1], label=router_name)
    loss_diff = abs(v1["loss"] - v2["loss"])

    log(f"\n{'=' * 70}\nRESULTS\n{'=' * 70}")
    log(f"  loss:        V2={v2['loss']:.6f}  V1={v1['loss']:.6f}  |Δ|={loss_diff:.2e}")
    log(f"  expert grad cosine: {expert_cos:.6f}  (min {GRAD_COSINE_MIN})")
    log(f"  router grad cosine: {router_cos:.6f}")

    return {
        "checks": {
            "elastic_selected": v2["backend_ok"],
            "legacy_selected": v1["backend_ok"],
            "elastic_buffer_shared_by_all_layers": v2["shared_ok"],
            "legacy_buffer_shared_by_all_layers": v1["shared_ok"],
            "loss_match": loss_diff <= LOSS_ABS_TOL,
            "expert_grad_match": expert_cos >= GRAD_COSINE_MIN,
            "router_grad_match": router_cos >= GRAD_COSINE_MIN,
        }
    }


main = gpu_test_main(min_world_size=2, prefix="ep_buffer_backends")(run)

if __name__ == "__main__":
    main()
