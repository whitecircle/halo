#!/usr/bin/env python
"""FSDP2 must not sever ``tie_word_embeddings``.

``fully_shard`` rebinds a shared parameter only among the modules of the *same* call. The mixin
wraps per layer, then the embedding-consuming backbone, then the model root — so a tied weight is
read through ``model.embed_tokens.weight`` (backbone call) and ``lm_head.weight`` (root call). If
both calls claim it, the tie is silently broken into two independent parameters that each receive
only their own half of the true tied gradient and diverge from step 1; the save path then flips
``tie_word_embeddings`` to False and the checkpoint looks self-consistent.

The guard is exact, not approximate: with the tie intact the embedding gradient equals the
single-GPU reference, so this asserts equality at bf16 noise; a severed tie lands the ratio at 0.82.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/test_fsdp_tied_embeddings.py
"""

from types import SimpleNamespace

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM

from src.distributed.fsdp import setup_fsdp2_for_dp
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.utils import log

SEED = 42
SEQ_LEN = 16
BATCH = 2

# The tied gradient is the sum of the embedding and lm_head contributions. A severed tie gives the
# embedding only its own share, so the ratio drops far below 1, while bf16 forward/backward puts the
# intact-tie ratio at 1.0 to a few 1e-3.
GRAD_RATIO_TOL = 5e-3


def build_model(tie: bool):
    """Tiny Qwen3 dense with ``tie_word_embeddings`` forced on/off, deterministic init."""
    torch.manual_seed(SEED)
    cfg = AutoConfig.for_model("qwen3", **{**TINY_QWEN3_CONFIG, "tie_word_embeddings": tie})
    return AutoModelForCausalLM.from_config(cfg).cuda().to(torch.bfloat16)


def embedding_grad_norm(model) -> float:
    """Full (gathered) L2 norm of the input-embedding gradient."""
    grad = model.get_input_embeddings().weight.grad
    assert grad is not None, "input embedding has no gradient after backward"
    if hasattr(grad, "full_tensor"):
        grad = grad.full_tensor()
    return grad.detach().float().norm().item()


def run(ctx):
    torch.manual_seed(SEED)
    ids = torch.randint(0, TINY_QWEN3_CONFIG["vocab_size"], (BATCH, SEQ_LEN), device="cuda")
    dist.broadcast(ids, src=0)

    # Reference: same weights, no FSDP, one rank's own gradient.
    ref = build_model(tie=True)
    ref(input_ids=ids, labels=ids).loss.backward()
    ref_norm = embedding_grad_norm(ref)
    del ref
    torch.cuda.empty_cache()

    model = build_model(tie=True)
    tied_before = model.get_input_embeddings().weight is model.lm_head.weight
    setup_fsdp2_for_dp(model, dp_size=ctx.world_size, args=SimpleNamespace(bf16=True, fp16=False))
    tied_after = model.get_input_embeddings().weight is model.lm_head.weight

    model(input_ids=ids, labels=ids).loss.backward()
    wrapped_norm = embedding_grad_norm(model)
    ratio = wrapped_norm / ref_norm

    checks = {
        # The premise: without this the test would pass vacuously on an untied model.
        "config_is_tied": tied_before,
        # The invariant: one parameter object, both names, after the wrap.
        "tie_survives_fsdp": tied_after,
        # The consequence: a severed tie drops the embedding's share of the gradient.
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
