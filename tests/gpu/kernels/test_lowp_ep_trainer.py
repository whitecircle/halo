#!/usr/bin/env python
"""Low-precision EP expert weights re-quantize after every fused AdamWBF16 step, through the trainer.

``cached_fake_quant`` keys on ``weight._version``. The fused kernel writes through raw pointers ATen never sees, so
only ``increment_version`` advances the key; without it the cache serves the step-0 quantization for the whole run.
``--native`` runs the same step through the DeepGEMM kernel, whose cast keeps the same per-version cache; the Hopper
image has no ``deep_gemm``, so there the flag must stay inert.
"""

import argparse
import math
import os

import torch
from transformers import AutoTokenizer, TrainerCallback
from trl import SFTConfig

from halo.distributed.parallelism_config import ParallelismConfig
from halo.kernels import expert_gemm
from halo.kernels.lowp import deepgemm, quantization
from halo.kernels.lowp.spec import LOWP_RECIPES, LowpSpec
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
# Two arms quantizing the same weights agree up to bf16 reduction noise. A stale step-0 quantization on this tiny model
# can stay under this bar too, so the cached entry itself is checked against the weight it stands for.
LOSS_RTOL = 1e-3

parser = argparse.ArgumentParser()
parser.add_argument("--precision", choices=tuple(LOWP_RECIPES), required=True)
parser.add_argument("--native", action="store_true", help="Train with HALO_DEEPGEMM_NATIVE=1")
ARGS = parser.parse_args()
QUANT_FORMAT = LOWP_RECIPES[ARGS.precision].quant_format
# Expert weights are stored ``[E, K, N]``.
CONTRACTION_AXIS = 1


def _simulated_entry_is_current(weight: torch.Tensor) -> bool | None:
    entry = quantization._weight_dequant_cache.get(weight, (weight._version, QUANT_FORMAT, CONTRACTION_AXIS))
    if entry is None:
        return None
    return torch.equal(entry, quantization._block_round_trip(weight, QUANT_FORMAT, CONTRACTION_AXIS).to(weight.dtype))


def _native_entry_is_current(weight: torch.Tensor) -> bool | None:
    entry = deepgemm._weight_cache.get(weight, (weight._version, QUANT_FORMAT))
    if entry is None:
        return None
    k_pad = entry[0].shape[-1] * 2 if QUANT_FORMAT == "mxfp4" else weight.shape[CONTRACTION_AXIS]
    fresh = deepgemm._quant_weight(weight, QUANT_FORMAT, k_pad, cacheable=False)
    # As floats, the one read of fp8 elements the production path itself runs on CUDA.
    return all(torch.equal(served.float(), recast.float()) for served, recast in zip(entry, fresh, strict=True))


class _ExpertVersions(TrainerCallback):
    """Before each write, whatever the step's forwards left in the weight cache must be the quantization of the weight
    as it is now: a write that does not advance the key keeps serving an older one. A rank with no token for an expert
    leaves no entry for it, so ``compared`` keeps the check from passing on nothing."""

    def __init__(self, model, entry_is_current):
        self.weights = [param for layer in ep_layers(model) for _, param in layer.expert_named_params()]
        self.per_step: list[list[int]] = []
        self._entry_is_current = entry_is_current
        self.compared = 0
        self.stale = 0

    def on_step_begin(self, args, state, control, **kwargs):
        self.per_step.append([param._version for param in self.weights])

    @torch.no_grad()
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self._entry_is_current is None:
            return
        verdicts = [self._entry_is_current(param) for param in self.weights]
        self.compared += sum(verdict is not None for verdict in verdicts)
        self.stale += sum(verdict is False for verdict in verdicts)


def _count_calls(module, name: str) -> list[int]:
    """Replace ``module.name`` for the process with a wrapper counting its calls."""
    real, calls = getattr(module, name), [0]

    def counted(*args, **kwargs):
        calls[0] += 1
        return real(*args, **kwargs)

    setattr(module, name, counted)
    return calls


def _train(
    ctx, checkpoint: str, tokenizer, output_dir: str, entry_is_current
) -> tuple[list[float], _ExpertVersions, bool]:
    """The run's losses, the expert version counters and weight-cache checks it saw, and whether AdamWBF16 stepped it."""
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
    versions = _ExpertVersions(trainer.model, entry_is_current)
    trainer.add_callback(versions)
    trainer.train()
    optimizer = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
    losses = step_losses(trainer)
    finish_phase(trainer)
    return losses, versions, isinstance(optimizer, AdamWBF16)


def _versions_advanced(versions: _ExpertVersions) -> bool:
    steps = versions.per_step
    return (
        len(steps) == TOTAL_STEPS
        and bool(versions.weights)
        and all(
            all(later > earlier for earlier, later in zip(before, after, strict=True))
            for before, after in zip(steps, steps[1:], strict=False)
        )
    )


def _shared_checks(versions: _ExpertVersions, adamw_bf16: bool) -> dict[str, bool]:
    return {
        "fused_adamw_bf16_stepped_the_run": adamw_bf16,
        "expert_weight_versions_advance_every_step_on_every_rank": world_all(_versions_advanced(versions)),
        "cached_entries_were_compared_on_every_rank": world_all(versions.compared > 0),
        "cached_entries_quantize_the_current_weight": world_all(versions.stale == 0),
    }


def _run_simulated(ctx, checkpoint: str, tokenizer) -> dict:
    cached, versions, adamw_bf16 = _train(
        ctx, checkpoint, tokenizer, os.path.join(ctx.output_dir, "cached"), _simulated_entry_is_current
    )
    os.environ["HALO_LOWP_WEIGHT_CACHE"] = "0"
    uncached, _, _ = _train(ctx, checkpoint, tokenizer, os.path.join(ctx.output_dir, "uncached"), None)
    del os.environ["HALO_LOWP_WEIGHT_CACHE"]

    checks = _shared_checks(versions, adamw_bf16)
    checks["both_arms_ran_every_step"] = len(cached) == len(uncached) == TOTAL_STEPS
    errors = [abs(a - b) / max(abs(b), 1e-6) for a, b in zip(cached, uncached, strict=False)]
    checks["cached_losses_match_the_uncached_quantization"] = bool(errors) and max(errors) <= LOSS_RTOL
    log(
        f"  {ARGS.precision}: cached {cached}, uncached {uncached}; expert versions {[s[:2] for s in versions.per_step]}"
    )
    return {"checks": checks, "metrics": {"loss_max_rel_err": max(errors, default=float("nan"))}}


def _run_native(ctx, checkpoint: str, tokenizer) -> dict:
    available = deepgemm.deepgemm_available()
    native_calls = _count_calls(deepgemm, "_deepgemm_forward")
    simulated_calls = _count_calls(expert_gemm, "cached_fake_quant")
    os.environ["HALO_DEEPGEMM_NATIVE"] = "1"
    entry_is_current = _native_entry_is_current if available else _simulated_entry_is_current
    losses, versions, adamw_bf16 = _train(
        ctx, checkpoint, tokenizer, os.path.join(ctx.output_dir, "native"), entry_is_current
    )
    del os.environ["HALO_DEEPGEMM_NATIVE"]

    checks = _shared_checks(versions, adamw_bf16)
    checks["ran_every_step"] = len(losses) == TOTAL_STEPS and all(math.isfinite(loss) for loss in losses)
    took_the_kernel = native_calls[0] > 0 and simulated_calls[0] == 0
    stayed_simulated = native_calls[0] == 0 and simulated_calls[0] > 0
    checks["one_route_for_the_whole_run"] = world_all(took_the_kernel if available else stayed_simulated)
    log(
        f"  {ARGS.precision} native (available={available}): {losses}; {native_calls[0]} kernel, {simulated_calls[0]} simulated"
    )
    return {"checks": checks, "metrics": {"native_calls": native_calls[0], "simulated_calls": simulated_calls[0]}}


@gpu_test_main(exact_world_size=2, prefix=f"lowp_ep_trainer_{ARGS.precision}{'_native' if ARGS.native else ''}")
def run(ctx):
    torch.cuda.set_device(ctx.device)
    pin_deterministic_ep_dispatch()
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    checkpoint = shared_tiny_family_checkpoint(ctx, TINY_MOE_FAMILIES["qwen3_moe"], "lowp_ep_trainer", tokenizer, SEED)
    return (_run_native if ARGS.native else _run_simulated)(ctx, checkpoint, tokenizer)


if __name__ == "__main__":
    run()
