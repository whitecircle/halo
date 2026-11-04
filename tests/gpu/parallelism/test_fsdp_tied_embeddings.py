#!/usr/bin/env python
"""FSDP2 must not sever ``tie_word_embeddings``, asserted against the single-GPU embedding gradient.

``fully_shard`` rebinds a shared parameter only within one call, and the mixin wraps per layer, backbone, then root. If
two calls claim the tied weight it splits silently, each half gets its own gradient share and diverges from step 1, and
the save flips ``tie_word_embeddings`` to False so the checkpoint looks self-consistent.
"""

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM

from halo.distributed.fsdp import apply_fsdp2_for_dp
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.rank_log import log

SEED = 42
SEQ_LEN = 16
BATCH = 2

# A severed tie drops the ratio to 0.82, while bf16 forward/backward puts the intact ratio at 1.0 to a few 1e-3.
GRAD_RATIO_TOL = 5e-3


def build_model(tie: bool):
    torch.manual_seed(SEED)
    cfg = AutoConfig.for_model("qwen3", **{**TINY_QWEN3_CONFIG, "tie_word_embeddings": tie})
    return AutoModelForCausalLM.from_config(cfg).cuda().to(torch.bfloat16)


def embedding_grad_norm(model) -> float:
    grad = model.get_input_embeddings().weight.grad
    assert grad is not None, "input embedding has no gradient after backward"
    if hasattr(grad, "full_tensor"):
        grad = grad.full_tensor()
    return grad.detach().float().norm().item()


def run(ctx):
    torch.manual_seed(SEED)
    ids = torch.randint(0, TINY_QWEN3_CONFIG["vocab_size"], (BATCH, SEQ_LEN), device="cuda")
    dist.broadcast(ids, src=0)

    # Same weights, no FSDP, one rank's own gradient.
    ref = build_model(tie=True)
    ref(input_ids=ids, labels=ids).loss.backward()
    ref_norm = embedding_grad_norm(ref)
    del ref
    torch.cuda.empty_cache()

    model = build_model(tie=True)
    tied_before = model.get_input_embeddings().weight is model.lm_head.weight
    apply_fsdp2_for_dp(model, dp_size=ctx.world_size, compute_dtype=torch.bfloat16, fp32_reduce=False)
    tied_after = model.get_input_embeddings().weight is model.lm_head.weight

    model(input_ids=ids, labels=ids).loss.backward()
    wrapped_norm = embedding_grad_norm(model)
    ratio = wrapped_norm / ref_norm

    checks = {
        # Without this the test passes vacuously on an untied model.
        "config_is_tied": tied_before,
        # One parameter object, both names, after the wrap.
        "tie_survives_fsdp": tied_after,
        "embedding_grad_matches_reference": abs(ratio - 1.0) <= GRAD_RATIO_TOL,
    }

    log(f"\n{'=' * 70}\nFSDP2 TIED EMBEDDINGS (world_size={ctx.world_size})\n{'=' * 70}")
    log(f"  tied before wrap: {tied_before}")
    log(f"  tied after  wrap: {tied_after}")
    log(f"  ||g_embed|| ref={ref_norm:.6f}  fsdp={wrapped_norm:.6f}  ratio={ratio:.4f}")
    log(f"  tolerance:        |ratio - 1| <= {GRAD_RATIO_TOL}")

    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="fsdp_tied_embeddings", partial_state=False)(run)

if __name__ == "__main__":
    main()
