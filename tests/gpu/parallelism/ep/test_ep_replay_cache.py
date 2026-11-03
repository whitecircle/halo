#!/usr/bin/env python
"""A checkpointed EP layer replays its cached DeepEP dispatch instead of running it twice, in both checkpoint modes.

A fresh recompute dispatch reuses the ``ElasticBuffer`` and invalidates the handle the backward node holds, corrupting
every gradient silently. Only the reentrant original runs under ``no_grad``, so the signal is the scope's pass counter.
A bare checkpoint (restored by TRL's ``disable_gradient_checkpointing``) must fail loud.
"""

import math
import sys

import torch
import torch.distributed as dist

from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.gradient_checkpointing import active_recompute_scope, enable_scoped_gradient_checkpointing
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import GPT_OSS_20B
from tests.common.rank_log import log

EP_SIZE = 2
SEQ_LEN = 128
# A bf16 recompute reproduces the forward only to its last bits; the failure is ~2.0 relative, three orders above this.
GRAD_REL_TOL = 5e-3


def fixed_batch(device, vocab: int):
    generator = torch.Generator(device="cpu").manual_seed(42)
    input_ids = torch.randint(0, vocab, (1, SEQ_LEN), generator=generator).to(device)
    dist.broadcast(input_ids, src=0)
    labels = input_ids.clone()
    labels[:, :8] = -100
    return input_ids, torch.ones_like(input_ids), labels


def run_step(model, device, vocab: int, *, collect: bool = False):
    """Returns gradients when ``collect``, else resident memory after ``zero_grad``, constant unless a layer retained
    state across steps."""
    input_ids, attention_mask, labels = fixed_batch(device, vocab)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()
    torch.cuda.synchronize()
    grads = (
        {name: p.grad.detach().float().clone() for name, p in model.named_parameters() if p.grad is not None}
        if collect
        else None
    )
    model.zero_grad(set_to_none=True)
    del outputs
    return grads if collect else torch.cuda.memory_allocated()


def load_ep_model(gradient_checkpointing: bool | None):
    """EP=2 gpt-oss; ``gradient_checkpointing`` selects the ``use_reentrant`` mode, None = no GC."""
    model, _ = load_distributed_model(
        model_name_or_path=GPT_OSS_20B,
        parallelism_config=ParallelismConfig(ep_size=EP_SIZE, cp_size=1, tp_size=1),
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    if gradient_checkpointing is not None:
        enable_scoped_gradient_checkpointing(
            model, gradient_checkpointing_kwargs={"use_reentrant": gradient_checkpointing}
        )
    model.train()
    return model


def worst_rel_err(test: dict, reference: dict) -> tuple[float, str, int]:
    """A non-finite error on either side reads as ``inf``."""
    worst, name, compared = 0.0, "", 0
    for key, ref in reference.items():
        if ref.norm() == 0:
            continue
        compared += 1
        err = float((test[key] - ref).norm() / ref.norm())
        if not math.isfinite(err):
            worst, name = math.inf, key
        elif err > worst:
            worst, name = err, key
    return worst, name, compared


@gpu_test_main(exact_world_size=EP_SIZE, prefix="ep_replay_cache")
def run(ctx):
    checks, metrics = {}, {}
    ensure_model_downloaded(GPT_OSS_20B, ctx.rank)

    model = load_ep_model(None)
    vocab = model.config.get_text_config().vocab_size
    reference = run_step(model, ctx.device, vocab, collect=True)
    del model
    cleanup_memory()

    for reentrant in (True, False):
        model = load_ep_model(reentrant)
        ep_layers = [m for m in model.modules() if isinstance(m, EPMoELayerBase)]
        label = "reentrant" if reentrant else "non_reentrant"

        grads = run_step(model, ctx.device, vocab, collect=True)
        # The frame-local scope releases the dispatch with its checkpoint frame, so resident memory is flat across
        # steps.
        resident_first = run_step(model, ctx.device, vocab)
        resident_second = run_step(model, ctx.device, vocab)

        worst, worst_name, compared = worst_rel_err(grads, reference)
        log(f"[rank {ctx.rank}] {label}: worst {worst:.3e} @{worst_name} over {compared} params")
        checks[f"{label}_grads_match_no_gc"] = worst < GRAD_REL_TOL
        checks[f"{label}_compared_every_param"] = compared == len(reference) and compared > 0
        checks[f"{label}_grads_finite"] = all(torch.isfinite(g).all() for g in grads.values())
        checks[f"{label}_has_ep_layers"] = len(ep_layers) > 0
        checks[f"{label}_no_scope_leaked"] = active_recompute_scope() is None
        checks[f"{label}_no_cache_growth_across_steps"] = resident_second <= resident_first * 1.001
        metrics[f"{label}_worst_grad_rel_err"] = worst
        metrics[f"{label}_resident_gb_step2"] = resident_first / 2**30
        metrics[f"{label}_resident_gb_step3"] = resident_second / 2**30

        del model, grads
        cleanup_memory()

    # The recompute's second DeepEP dispatch must fail loud.
    model = load_ep_model(None)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    try:
        run_step(model, ctx.device, vocab)
        checks["unscoped_checkpoint_rejected"] = False
    except RuntimeError as error:
        checks["unscoped_checkpoint_rejected"] = "no EP checkpoint scope" in str(error)
    del model
    cleanup_memory()

    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
