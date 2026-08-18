#!/usr/bin/env python
"""The EP checkpoint scope on real DeepEP dispatch (gpt-oss-20b, ep2 on 2 GPUs).

A checkpointed EP layer must NOT run DeepEP twice. The recompute's fresh dispatch would reuse the
same ``ElasticBuffer`` and invalidate the handle the original forward's backward node still holds —
which corrupts every gradient in the model and raises nothing. The scope makes the recompute replay
the frame's cached dispatch/combine instead. This test pins that on real dispatch, in BOTH
checkpoint modes:

  * gradients under GC match the no-GC gradients (the corruption this guards against is a measured
    2.0 relative error on nearly every parameter, so the tight tolerance here is the whole test);
  * both ``use_reentrant`` modes replay: the recompute signal is the scope's pass counter, not grad
    mode (only the reentrant original forward runs under ``no_grad``);
  * no scope survives the step, and resident memory is flat across steps (the frame-local cache
    dies with its checkpoint frame — the layer holds no cache of its own);
  * checkpointing re-enabled WITHOUT the scope (TRL's ``disable_gradient_checkpointing`` restores a
    bare checkpoint on exit) fails loud instead of silently corrupting the gradients.

Run: torchrun --nproc_per_node=2 tests/gpu/parallelism/ep/test_ep_replay_cache.py
"""

import sys

import torch
import torch.distributed as dist

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.gc_scope import active_checkpoint_scope
from src.distributed.expert_parallel.patching import enable_ep_gradient_checkpointing
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.utils import cleanup_memory, log

EP_SIZE = 2
SEQ_LEN = 128
# bf16 recompute reproduces the forward only to its last bits; the failure this guards against is
# ~2.0 relative on nearly every parameter, three orders of magnitude above this gate.
GRAD_REL_TOL = 5e-3


def fixed_batch(device, vocab: int):
    """One deterministic batch, identical on every rank."""
    generator = torch.Generator(device="cpu").manual_seed(42)
    input_ids = torch.randint(0, vocab, (1, SEQ_LEN), generator=generator).to(device)
    dist.broadcast(input_ids, src=0)
    labels = input_ids.clone()
    labels[:, :8] = -100
    return input_ids, torch.ones_like(input_ids), labels


def run_step(model, device, vocab: int, *, collect: bool = False):
    """One forward+backward. Returns the gradients when ``collect``, else memory resident after it.

    The resident reading is taken after ``zero_grad``, so within a leg it holds constant unless a
    layer retained something across steps — which is exactly what a cache that outlived its
    checkpoint frame would do.
    """
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
        enable_ep_gradient_checkpointing(
            model, gradient_checkpointing_kwargs={"use_reentrant": gradient_checkpointing}
        )
    model.train()
    return model


def worst_rel_err(test: dict, reference: dict) -> tuple[float, str, int]:
    worst, name, compared = 0.0, "", 0
    for key, ref in reference.items():
        if ref.norm() == 0:
            continue
        compared += 1
        err = float((test[key] - ref).norm() / ref.norm())
        if err > worst:
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
        # A layer-level cache would keep one step's dispatch alive into the next; the frame-local
        # scope releases it with the checkpoint frame, so resident memory is flat across steps.
        resident_first = run_step(model, ctx.device, vocab)
        resident_second = run_step(model, ctx.device, vocab)

        worst, worst_name, compared = worst_rel_err(grads, reference)
        log(f"[rank {ctx.rank}] {label}: worst {worst:.3e} @{worst_name} over {compared} params")
        checks[f"{label}_grads_match_no_gc"] = worst < GRAD_REL_TOL
        checks[f"{label}_compared_every_param"] = compared == len(reference) and compared > 0
        checks[f"{label}_grads_finite"] = all(torch.isfinite(g).all() for g in grads.values())
        checks[f"{label}_has_ep_layers"] = len(ep_layers) > 0
        checks[f"{label}_no_scope_leaked"] = active_checkpoint_scope() is None
        checks[f"{label}_no_cache_growth_across_steps"] = resident_second <= resident_first * 1.001
        metrics[f"{label}_worst_grad_rel_err"] = worst
        metrics[f"{label}_resident_gb_step2"] = resident_first / 2**30
        metrics[f"{label}_resident_gb_step3"] = resident_second / 2**30

        del model, grads
        cleanup_memory()

    # Checkpointing without the scope: the recompute's second DeepEP dispatch must fail loud.
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
