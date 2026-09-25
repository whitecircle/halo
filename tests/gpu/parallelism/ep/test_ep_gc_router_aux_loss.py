#!/usr/bin/env python
"""The router aux loss must train routers under reentrant checkpointing, on a real DeepEP EP layer.

Every non-PP MoE run with gradient checkpointing is forced reentrant, whose original forward runs
under ``no_grad``: transformers collects ``outputs.router_logits`` there, so without the aux-gradient
routing each checkpointed layer's aux term reaches the loss with no graph and the router gradient is
exactly what it would be at ``router_aux_loss_coef: 0``. What only a GPU shows is the live path: EP=2
with a DeepEP ``ElasticBuffer``, the checkpoint scope replaying dispatch/combine in the recompute, the
router gradient's in-backward DP hook, and the backward running on the device thread.

The aux term's share of the router gradient is measured as ``grad(coef) - grad(0)`` on the same
weights and batch, once without checkpointing and once under reentrant checkpointing with
``moe_balancing: aux_loss`` applied through the strategy seam; the two must agree. A forward counter
proves the recompute ran, so agreement cannot come from checkpointing silently doing nothing.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 tests/gpu/parallelism/ep/test_ep_gc_router_aux_loss.py --family gpt_oss
"""

import sys

import torch
from transformers import GptOssConfig, GptOssForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from src.distributed.expert_parallel.balancing_strategy import apply_balancing_strategy
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.patching import (
    create_ep_buffers,
    enable_ep_gradient_checkpointing,
    patch_moe_model_for_ep,
)
from src.distributed.parallelism_config import ParallelismConfig
from src.models.moe_balancing import declared_routers
from tests.common.harness import gpu_test_main, log
from tests.common.models import TINY_GPTOSS_CONFIG, TINY_QWEN3_MOE_CONFIG

FAMILY = "gpt_oss"
EP_SIZE = 2
BATCH, SEQ = 2, 64
COEF = 0.5
SEED = 1234
# bf16 recompute through the replayed dispatch: the aux share must agree to well inside this, while a
# dropped aux gradient is off by 100%.
AUX_REL_TOL = 5e-2

_FAMILIES = {
    "gpt_oss": (GptOssForCausalLM, GptOssConfig, TINY_GPTOSS_CONFIG),
    "qwen3_moe": (Qwen3MoeForCausalLM, Qwen3MoeConfig, TINY_QWEN3_MOE_CONFIG),
}


def build_model(device):
    model_class, config_class, config = _FAMILIES[FAMILY]
    torch.manual_seed(SEED)
    model = model_class(config_class(**{**config, "router_aux_loss_coef": COEF}))
    for router in declared_routers(model):
        torch.nn.init.normal_(router.module.weight, std=0.1)
    return model.to(torch.bfloat16).to(device)


def router_gradient(model, batch, coef: float) -> torch.Tensor:
    """The routers' gradient after one forward+backward with the head's aux coefficient at ``coef``."""
    model.zero_grad(set_to_none=True)
    model.router_aux_loss_coef = coef
    model(**batch).loss.backward()
    return torch.cat([router.module.weight.grad.float().flatten() for router in declared_routers(model)])


def run(ctx):
    checks, metrics = {}, {}
    model = build_model(ctx.device)
    patch_moe_model_for_ep(model, ParallelismConfig(ep_size=EP_SIZE).create_ep_config())
    create_ep_buffers(model)
    apply_balancing_strategy(model, "aux_loss", is_moe=True)
    model.train()
    ep_layers = [module for module in model.modules() if isinstance(module, EPMoELayerBase)]
    checks["ep_layers_patched"] = len(ep_layers) == len(declared_routers(model)) > 0

    forward_calls = {"n": 0}
    ep_layers[0].register_forward_pre_hook(
        lambda _module, _args: forward_calls.__setitem__("n", forward_calls["n"] + 1)
    )

    generator = torch.Generator().manual_seed(SEED + ctx.rank)
    input_ids = torch.randint(0, model.config.vocab_size, (BATCH, SEQ), generator=generator).to(ctx.device)
    batch = {"input_ids": input_ids, "labels": input_ids}

    plain = router_gradient(model, batch, COEF)
    plain_aux = plain - router_gradient(model, batch, 0.0)

    enable_ep_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": True})
    forward_calls["n"] = 0
    checkpointed = router_gradient(model, batch, COEF)
    checks["recompute_ran"] = forward_calls["n"] == 2
    checkpointed_aux = checkpointed - router_gradient(model, batch, 0.0)

    rel = float((checkpointed_aux - plain_aux).norm() / plain_aux.norm())
    metrics["aux_share_of_router_grad"] = float(plain_aux.norm() / plain.norm())
    metrics["checkpointed_aux_rel_err"] = rel
    metrics["checkpointed_total_rel_err"] = float((checkpointed - plain).norm() / plain.norm())
    checks["aux_term_moves_the_routers"] = metrics["aux_share_of_router_grad"] > 1e-2
    checks["checkpointed_aux_gradient_matches_unchecked"] = rel < AUX_REL_TOL
    log(f"[rank {ctx.rank}] {FAMILY}: aux share {metrics['aux_share_of_router_grad']:.3f}, rel err {rel:.2e}")
    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(exact_world_size=2, prefix="ep_gc_router_aux")(run)

if __name__ == "__main__":
    if "--family" in sys.argv:
        i = sys.argv.index("--family")
        FAMILY = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    if FAMILY not in _FAMILIES:
        raise SystemExit(f"--family must be one of {sorted(_FAMILIES)}, got {FAMILY!r}")
    main()
