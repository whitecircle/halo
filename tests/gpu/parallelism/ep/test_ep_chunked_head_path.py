#!/usr/bin/env python
"""The chunked log-probs of an EP model against its own forward, on Nemotron-H's text class and its composite, whose
hidden path enters the tower below the forward: padded and per-row, the gradient reaching the embedding, the head and
each rank's experts."""

import argparse
from types import SimpleNamespace

import torch
from trl.trainer.utils import selective_log_softmax

from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.models.nemotron_h import apply_nemotron_h_patches
from halo.trainers.grpo.logprob_scoring import LogprobScoringMixin
from tests.common.distributed import world_spread
from tests.common.ep_reference import random_token_batch
from tests.common.fla_autotune import pin_fla_autotune_first_fit
from tests.common.harness import gpu_test_main
from tests.common.rank_log import log
from tests.common.tiny_models import (
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    fill_release_scale_bias,
    tiny_family_model,
)
from tests.common.tolerances import TOL

SEED = 42
SEQ, COMPLETION = 64, 16
# Rows per forward: 1 runs each row trimmed, 2 the padded batch.
ROW_COUNTS = (1, 2)
# bf16 logits against the sweep's fp32 ones; a wrong backbone moves every completion token by whole nats.
LOGP_MEAN_TOL = 5e-2
LOGP_MAX_TOL = 0.5
FAMILIES = {"nemotron_h": TINY_MOE_FAMILIES["nemotron_h"], "nemotron_h_omni": TINY_MOE_VLM_FAMILIES["nemotron_h_omni"]}

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
ARGS = parser.parse_args()


class _Scorer(LogprobScoringMixin):
    """What the chunked path reads off a trainer."""

    temperature = 1.0
    accelerator = SimpleNamespace(unwrap_model=lambda module: module)


def _has_gradient(tensor: torch.Tensor) -> bool:
    return tensor.grad is not None and bool((tensor.grad != 0).any())


@gpu_test_main(exact_world_size=2, prefix=f"ep_chunked_head_path_{ARGS.family}")
def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    torch.cuda.set_device(ctx.device)
    apply_nemotron_h_patches()
    torch.manual_seed(SEED)
    model = tiny_family_model(FAMILIES[ARGS.family]).to(ctx.device, torch.bfloat16)
    fill_release_scale_bias(model)
    model.to(ctx.device)
    pin_fla_autotune_first_fit(ctx)
    model = apply_ep_layers(model, ParallelismConfig(ep_size=ctx.world_size).ep_config())
    layers = [layer for _, layer in named_ep_layers(model)]
    checks["ep_layers_in_place"] = bool(layers) and all(layer.ep_size == ctx.world_size for layer in layers)
    model.train()
    vocab = model.get_output_embeddings().out_features

    for rows in ROW_COUNTS:
        ids, _ = random_token_batch(vocab, rows, SEQ, ctx.device, seed=SEED)
        mask = torch.ones_like(ids)
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        forward = selective_log_softmax(logits[:, :-1][:, -COMPLETION:].float(), ids[:, -COMPLETION:])
        logps, _ = _Scorer()._chunked_completion_logps(model, ids, mask, COMPLETION, rows)
        difference = (logps - forward).abs()
        metrics[f"rows{rows}_logp_mean"], metrics[f"rows{rows}_logp_max"] = (
            difference.mean().item(),
            difference.max().item(),
        )
        log(f"  {ARGS.family} rows={rows}: |Δ logp| mean {metrics[f'rows{rows}_logp_mean']:.2e}")
        checks[f"rows{rows}_scores_as_the_forward"] = (
            metrics[f"rows{rows}_logp_mean"] < LOGP_MEAN_TOL and metrics[f"rows{rows}_logp_max"] < LOGP_MAX_TOL
        )
        checks[f"rows{rows}_ranks_agree"] = TOL.identical_batch_ranks_agree(
            world_spread(logps.sum().item(), ctx.device)
        )
        logps.sum().backward()
        checks[f"rows{rows}_gradient_reaches_the_embedding"] = _has_gradient(model.get_input_embeddings().weight)
        checks[f"rows{rows}_gradient_reaches_the_head"] = _has_gradient(model.get_output_embeddings().weight)
        checks[f"rows{rows}_gradient_reaches_the_experts"] = all(_has_gradient(layer.up_proj) for layer in layers)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
