#!/usr/bin/env python
"""Benchmark matched DPO steps with a synthetic large vocabulary on two GPUs."""

import argparse
import time

import torch
import torch.distributed as dist

from halo.runtime.dtensor import to_local
from tests.common.distributed import world_all
from tests.common.dpo_scoring import _build_trainer
from tests.common.harness import gpu_test_main
from tests.common.tiny_models import TINY_DENSE_FAMILY, shared_tiny_family_checkpoint, word_level_tokenizer

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--scorer", choices=("bounded", "full"), required=True)
PARSER.add_argument("--length", type=int, required=True)
ARGS = PARSER.parse_args()
VOCAB = 262_144


def _world_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor)


@gpu_test_main(exact_world_size=2, prefix=f"dpo_benchmark_{ARGS.scorer}_{ARGS.length}")
def run(ctx):
    if not 32 <= ARGS.length <= 16_384:
        raise ValueError("length must be between 32 and 16384")
    tokenizer = word_level_tokenizer([f"t{i}" for i in range(2, 48)])
    checkpoint = shared_tiny_family_checkpoint(
        ctx,
        TINY_DENSE_FAMILY,
        "dpo_large_vocab_benchmark",
        tokenizer,
        102,
        overrides={"max_position_embeddings": 16_384},
        edit=lambda model: model.resize_token_embeddings(VOCAB, mean_resizing=False),
    )
    start = time.perf_counter()
    trainer, _ = _build_trainer(
        ctx,
        checkpoint,
        family="dense",
        dtype=torch.bfloat16,
        bounded=ARGS.scorer == "bounded",
        oracle_head=False,
    )
    startup = _world_max(time.perf_counter() - start, ctx.device)
    trainer.model.train()
    generator = torch.Generator().manual_seed(102 + ctx.rank)
    ids = torch.randint(2, 48, (4, ARGS.length), generator=generator).to(ctx.device)
    completion = torch.ones_like(ids)
    completion[:, :16] = 0
    batch = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "completion_mask": completion,
        "ref_chosen_logps": torch.zeros(2, device=ctx.device),
        "ref_rejected_logps": torch.zeros(2, device=ctx.device),
    }
    timings, peaks = [], []
    for step in range(3):
        trainer.state.global_step = step
        trainer.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(ctx.device)
        torch.cuda.reset_peak_memory_stats(ctx.device)
        ctx.barrier()
        start = time.perf_counter()
        with trainer.compute_loss_context_manager():
            loss = trainer.compute_loss(trainer.model, batch)
        trainer.accelerator.backward(loss)
        trainer._finish_fsdp2_backward(trainer.model)
        grad_norm = trainer._get_grad_norm(trainer.model)
        local_norm = to_local(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
        finite = bool(loss.isfinite()) and local_norm is not None and bool(
            torch.isfinite(torch.as_tensor(local_norm)).all()
        )
        if not world_all(finite, ctx.device):
            raise AssertionError("DPO benchmark loss or gradient norm is nonfinite on at least one rank")
        trainer.optimizer.step()
        torch.cuda.synchronize(ctx.device)
        elapsed = _world_max(time.perf_counter() - start, ctx.device)
        peak = _world_max(torch.cuda.max_memory_allocated(ctx.device) / 2**30, ctx.device)
        with torch.no_grad():
            finite_weights = all(bool(torch.isfinite(to_local(param)).all()) for param in trainer.model.parameters())
        if not world_all(finite_weights, ctx.device):
            raise AssertionError("DPO benchmark stored weights are nonfinite on at least one rank")
        if step:
            timings.append(elapsed)
            peaks.append(peak)
    return {
        "checks": {"three_optimizer_steps_finite": True},
        "metrics": {
            "scorer": ARGS.scorer,
            "length": ARGS.length,
            "vocab": VOCAB,
            "pairs_per_step_global": 4,
            "trainer_init_and_reference_sweep_at_32_tokens_s": startup,
            "mean_step_s_after_warmup": sum(timings) / len(timings),
            "pairs_per_s_after_warmup": 4 * len(timings) / sum(timings),
            "max_rank_peak_allocated_gib": max(peaks),
        },
    }


if __name__ == "__main__":
    run()
