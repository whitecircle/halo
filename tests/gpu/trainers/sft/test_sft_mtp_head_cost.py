#!/usr/bin/env python
"""What a released head costs a real step: median seconds and peak memory of Qwen3.6-35B under EP8 on packed rows of
33000 tokens, off against each scope, hard and soft. One launch sweeps the arms in one order and repeats the head-less
one last, so the row reads its own noise. No arm stores fp32 masters, which ``head_only`` refuses, and determinism is
off, as it costs about 2%; the bit-for-bit claims are the replay rows'.

    torchrun --nproc_per_node=8 tests/gpu/trainers/sft/test_sft_mtp_head_cost.py
    torchrun --nproc_per_node=8 tests/gpu/trainers/sft/test_sft_mtp_head_cost.py --text_only

``--text_only`` loads the CausalLM class, whose loss takes Liger's fused kernel; a composite forces it off.
"""

import argparse
import os
import statistics
import time
from itertools import pairwise

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, TrainerCallback
from trl import SFTConfig

from halo.data.collators.factory import select_data_collator
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.liger.dispatch import finalize_liger_after_direct_load
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.config_access import text_config
from halo.models.mtp.head import find_mtp_head
from halo.models.mtp.spec import MtpHeadRequest, MtpObjective
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.distributed import shared_output_dir
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.mtp_heads import packed_token_rows
from tests.common.mtp_real_rows import REAL_BARS_APPLY, REAL_MODEL, ArmCost, CostBars, cached_source, cost_checks
from tests.common.rank_log import log

_WEIGHT = 0.3
_SEED = 42
_LEARNING_RATE = 1e-5
_ROWS_PER_RANK = 8
# Packed documents in a row, as the SFT recipe's bfd pass leaves them on ultrachat.
_DOCUMENTS = 32
_ARMS = (
    ("off", None),
    ("joint_hard", MtpObjective(_WEIGHT, "hard", "joint")),
    ("joint_soft", MtpObjective(_WEIGHT, "soft", "joint")),
    ("detached_hard", MtpObjective(_WEIGHT, "hard", "detached")),
    ("detached_soft", MtpObjective(_WEIGHT, "soft", "detached")),
    ("head_only_hard", MtpObjective(_WEIGHT, "hard", "head_only")),
    ("head_only_soft", MtpObjective(_WEIGHT, "soft", "head_only")),
    ("off_again", None),
)
# Past this at an arm's start, its peak carries an earlier arm's tensors.
_CLEAN_START_BYTES = 2 * 1024**3
# What one recompute of a decoder layer holds, in bf16 [T, H] planes: attention and every routed expert's activations.
_LAYER_RECOMPUTE_PLANES = 64


class _StepClock(TrainerCallback):
    """Stamps each optimizer step past the device's queue, so one step's time lands on it."""

    def __init__(self):
        self._stamps: list[float] = []

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._stamps.append(time.perf_counter())

    def median_seconds(self, drop: int) -> float:
        return statistics.median(after - before for before, after in pairwise(self._stamps[drop:]))

    def quartiles(self, drop: int) -> list[float]:
        return statistics.quantiles((after - before for before, after in pairwise(self._stamps[drop:])), n=4)


def _memory_allowance(model, length: int) -> int:
    """A head's weights and AdamWBF16's two moments, the fused kernel's fp32 ``[V, H]`` accumulator, and one layer's
    recompute. The head's stored size is its gathered size, which a forward holds under ZeRO-2."""
    text = text_config(model.config)
    stored = sum(param.numel() * param.element_size() for param in find_mtp_head(model).parameters())
    return (
        3 * stored + 4 * text.vocab_size * text.hidden_size + _LAYER_RECOMPUTE_PLANES * length * text.hidden_size * 2
    )


def _world_peak(ctx, reading=torch.cuda.max_memory_allocated) -> int:
    """COLLECTIVE."""
    peak = reading() if torch.cuda.is_available() else 0
    if ctx.world_size == 1:
        return peak
    gathered: list = [None] * ctx.world_size
    dist.all_gather_object(gathered, peak)
    return max(gathered)


def _run_arm(ctx, cli, root: str, tag: str, objective: MtpObjective | None, tokenizer) -> dict:
    """COLLECTIVE: one arm over a fresh load."""
    cleanup_memory()
    started_clean = True
    reserved_at_start = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        started_clean = torch.cuda.memory_allocated() <= _CLEAN_START_BYTES
        reserved_at_start = _world_peak(ctx, torch.cuda.memory_reserved)
    fused = cli.kernel == "fused"
    liger = {"use_liger_kernel": fused, "liger_kernel_config": {"fused_linear_cross_entropy": True} if fused else None}
    parallelism = ParallelismConfig(ep_size=cli.ep_size)
    model, _ = load_distributed_model(
        REAL_MODEL,
        parallelism,
        dtype=torch.bfloat16,
        attn_implementation=cli.attn,
        trust_remote_code=False,
        text_only_model=cli.text_only,
        mtp_head=None if objective is None else MtpHeadRequest(objective),
        **liger,
    )
    config = SFTConfig(
        output_dir=os.path.join(root, tag),
        max_steps=cli.steps,
        per_device_train_batch_size=1,
        learning_rate=_LEARNING_RATE,
        lr_scheduler_type="constant",
        bf16=True,
        gradient_checkpointing=objective is None or objective.scope != "head_only",
        logging_steps=cli.steps,
        save_strategy="no",
        report_to="none",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        seed=_SEED,
        **liger,
    )
    finalize_liger_after_direct_load(config, fused, model)
    clock = _StepClock()
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=packed_token_rows(
            text_config(model.config).vocab_size,
            _ROWS_PER_RANK * ctx.world_size,
            seed=_SEED,
            length=cli.length,
            documents=cli.documents,
        ),
        processing_class=tokenizer,
        data_collator=select_data_collator(
            tokenizer,
            train_on_completions_only=False,
            packing=True,
            model_config=model.config,
            emit_varlen_one_document=True,
        ),
        parallelism_config=parallelism,
        callbacks=[clock],
    )
    allowance = 0 if objective is None else _memory_allowance(model, cli.length)
    trainer.train()
    logged = [entry for entry in trainer.state.log_history if "loss" in entry]
    cost = ArmCost(clock.median_seconds(cli.drop), _world_peak(ctx))
    reserved = _world_peak(ctx, torch.cuda.max_memory_reserved)
    quartiles = clock.quartiles(cli.drop)
    log(
        f"  {tag}: {cost.step_seconds:.3f} s/step (quartiles {quartiles[0]:.3f} {quartiles[2]:.3f}), world peak "
        f"{cost.peak_bytes / 1024**3:.1f} GiB allocated, {reserved / 1024**3:.1f} GiB reserved, "
        f"{reserved_at_start / 1024**3:.1f} GiB reserved at its start, attended tokens "
        f"{logged[-1].get('num_attended_tokens_seen')}"
    )
    result = {
        "cost": cost,
        "reserved": reserved,
        "quartiles": quartiles,
        "allowance": allowance,
        "started_clean": started_clean,
        "took_its_term": (objective is not None) == ("mtp_ce" in logged[-1]),
    }
    trainer.cleanup_ep()
    del trainer, model
    return result


@gpu_test_main(min_world_size=1)
def run(ctx) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep_size", type=int, default=8)
    parser.add_argument("--length", type=int, default=33000)
    parser.add_argument("--documents", type=int, default=_DOCUMENTS)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--drop", type=int, default=10, help="Leading steps left out: first-use compiles land there.")
    parser.add_argument("--attn", default="flash_attention_2")
    parser.add_argument("--kernel", choices=("fused", "chunked"), default="fused")
    parser.add_argument("--text_only", action="store_true")
    parser.add_argument("--arms", default=None, help="Comma-separated arms to run in order; the default is every arm.")
    cli = parser.parse_args()
    if cached_source(REAL_MODEL) is None:
        log(f"SKIP: {REAL_MODEL} is not in the local Hub cache (HALO_TEST_MTP_REAL_MODEL)")
        raise SystemExit(0)

    tokenizer = AutoTokenizer.from_pretrained(REAL_MODEL)
    root = shared_output_dir(ctx)
    chosen = dict(_ARMS) if cli.arms is None else {tag: dict(_ARMS)[tag] for tag in cli.arms.split(",")}
    results = {tag: _run_arm(ctx, cli, root, tag, objective, tokenizer) for tag, objective in chosen.items()}
    arms = {tag: result["cost"] for tag, result in results.items()}
    checks = {f"{tag}_took_its_term": result["took_its_term"] for tag, result in results.items()}
    checks["every_arm_started_clean"] = all(result["started_clean"] for result in results.values())
    if REAL_BARS_APPLY and cli.arms is None:
        checks |= cost_checks(arms, max(result["allowance"] for result in results.values()), CostBars())
    tokens = cli.length * ctx.world_size
    metrics = {
        tag: {
            "step_seconds": cost.step_seconds,
            "step_seconds_quartiles": results[tag]["quartiles"],
            "tokens_per_second": tokens / cost.step_seconds,
            "tokens_per_second_per_rank": cli.length / cost.step_seconds,
            "peak_gib": cost.peak_bytes / 1024**3,
            "reserved_gib": results[tag]["reserved"] / 1024**3,
            "allowance_gib": results[tag]["allowance"] / 1024**3,
            "of_off": cost.step_seconds / arms["off"].step_seconds,
        }
        for tag, cost in arms.items()
    }
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    raise SystemExit(run())
