#!/usr/bin/env python
"""Low-precision EP expert weights re-quantize after every fused AdamWBF16 step, through the trainer.

``cached_fake_quant`` keys on ``weight._version``. The fused kernel writes through raw pointers ATen never sees, so
only ``increment_version`` advances the key; without it the cache serves the step-0 quantization for the whole run.
"""

import argparse
import os

import torch
from transformers import AutoTokenizer, TrainerCallback
from trl import SFTConfig

from halo.distributed.parallelism_config import ParallelismConfig
from halo.kernels.lowp import quantization
from halo.kernels.lowp.spec import LowpSpec
from halo.model_loading.parallel.loader import load_distributed_model
from halo.optimizers.adamw_bf16 import AdamWBF16
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import pin_deterministic_ep_dispatch, world_all
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.memory import finish_phase
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log
from tests.common.tiny_models import TINY_MOE_FAMILIES, shared_tiny_family_checkpoint
from tests.common.training_checks import step_losses

# Numerically identical to the compiled path, without a per-rank compile warm-up.
quantization._compile_failed = True

SEED = 42
TOTAL_STEPS = 3
# Two arms quantizing the same weights agree up to bf16 reduction noise; a stale step-0 quantization moves the loss by
# far more.
LOSS_RTOL = 1e-3

parser = argparse.ArgumentParser()
parser.add_argument("--precision", choices=("fp8", "mxfp4"), required=True)
ARGS = parser.parse_args()


class _ExpertVersions(TrainerCallback):
    def __init__(self, model):
        self.weights = [param for layer in ep_layers(model) for _, param in layer.expert_named_params()]
        self.per_step: list[list[int]] = []

    def on_step_begin(self, args, state, control, **kwargs):
        self.per_step.append([param._version for param in self.weights])


def _train(ctx, checkpoint: str, tokenizer, output_dir: str) -> tuple[list[float], _ExpertVersions, bool]:
    """The run's losses, the expert version counters it saw, and whether AdamWBF16 stepped it."""
    pc = ParallelismConfig(ep_size=2, lowp=LowpSpec(precision=ARGS.precision))
    model, _ = load_distributed_model(
        checkpoint, pc, torch.bfloat16, trust_remote_code=False, attn_implementation="eager", use_liger_kernel=False
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=output_dir,
            max_steps=TOTAL_STEPS,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=5e-3,
            lr_scheduler_type="constant",
            max_length=128,
            bf16=True,
            gradient_checkpointing=False,
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            logging_nan_inf_filter=False,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            seed=SEED,
        ),
        train_dataset=create_sft_dataset(64, tokenizer, seed=SEED),
        processing_class=tokenizer,
        parallelism_config=pc,
    )
    versions = _ExpertVersions(trainer.model)
    trainer.add_callback(versions)
    trainer.train()
    optimizer = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
    losses = step_losses(trainer)
    finish_phase(trainer)
    return losses, versions, isinstance(optimizer, AdamWBF16)


@gpu_test_main(exact_world_size=2, prefix=f"lowp_ep_trainer_{ARGS.precision}")
def run(ctx):
    torch.cuda.set_device(ctx.device)
    pin_deterministic_ep_dispatch()
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    checkpoint = shared_tiny_family_checkpoint(ctx, TINY_MOE_FAMILIES["qwen3_moe"], "lowp_ep_trainer", tokenizer, SEED)

    cached, versions, adamw_bf16 = _train(ctx, checkpoint, tokenizer, os.path.join(ctx.output_dir, "cached"))
    os.environ["HALO_LOWP_WEIGHT_CACHE"] = "0"
    uncached, _, _ = _train(ctx, checkpoint, tokenizer, os.path.join(ctx.output_dir, "uncached"))
    del os.environ["HALO_LOWP_WEIGHT_CACHE"]

    steps = versions.per_step
    advanced = (
        len(steps) == TOTAL_STEPS
        and bool(versions.weights)
        and all(
            all(later > earlier for earlier, later in zip(before, after, strict=True))
            for before, after in zip(steps, steps[1:], strict=False)
        )
    )
    checks = {
        "fused_adamw_bf16_stepped_the_run": adamw_bf16,
        "expert_weight_versions_advance_every_step_on_every_rank": world_all(advanced),
        "both_arms_ran_every_step": len(cached) == len(uncached) == TOTAL_STEPS,
    }
    errors = [abs(a - b) / max(abs(b), 1e-6) for a, b in zip(cached, uncached, strict=False)]
    checks["cached_losses_match_the_uncached_quantization"] = bool(errors) and max(errors) <= LOSS_RTOL
    log(f"  {ARGS.precision}: cached {cached}, uncached {uncached}; expert versions per step {[s[:2] for s in steps]}")
    return {"checks": checks, "metrics": {"loss_max_rel_err": max(errors, default=float("nan"))}}


if __name__ == "__main__":
    run()
