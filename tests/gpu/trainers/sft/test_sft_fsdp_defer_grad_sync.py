#!/usr/bin/env python
"""``fsdp_defer_grad_sync: true`` must reduce once per grad-accum window and train like the default.

The knob turns FSDP2's gradient reduce off for microbatches 1..n-1 of a window
(``set_requires_gradient_sync``) and back on for the last, so the unsharded gradients accumulate
locally and are reduce-scattered once. Two arms train the same data from the same weights, knob off
and on, and the test separates the three states the knob can be in:

  * **Correct** — the sharded (DTensor) gradients stay unset after microbatches 1..n-1, where the
    default arm has already reduced into them, and at the optimizer step every sharded param
    carries a gradient; loss, grad norm and final weights track the default arm.
  * **Never disarmed** (the toggle does nothing) — sharded grads are set mid-window: caught by the
    mid-window probe, the only check this state fails.
  * **Never re-armed** — no reduce reaches the sharded params, so the step sees no gradients.

Modes (``--mode``, 4 GPUs):
  dp    dense, FSDP2 over the world
  hsdp  dense, HSDP 2x2 (two simulated NVLink domains)
  tp    dense, TP2 x DP2: FSDP2 over the mesh's dp dimension, plus the TP replicated-grad sweep
  cp    dense, CP2 x DP2
  ep1   MoE at ep_size=1: experts are FSDP2-sharded DTensors, deferred with everything else
  ep1_fp32_router
        ep1 with ``fp32_router``: each fp32 router is a nested FSDP2 group inside its bf16 layer
  ep    MoE, one EP group over the world: experts FSDP-ignored, synced by the in-backward hooks
  ep2   MoE, two EP groups over two simulated domains: non-expert params sharded over the EP group,
        experts and the cross-replica average synced by the post-backward sweep

Run with 4 GPUs:
    torchrun --nproc_per_node=4 tests/gpu/trainers/sft/test_sft_fsdp_defer_grad_sync.py --mode dp
"""

import argparse
import math
import os

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)
from transformers.trainer_callback import TrainerCallback
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_MOE_CONFIG
from tests.common.utils import cleanup_memory, log, max_or_nan, step_losses

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["dp", "hsdp", "tp", "cp", "ep1", "ep1_fp32_router", "ep", "ep2"], default="dp")
ARGS = parser.parse_args()

MOE_MODES = ("ep1", "ep1_fp32_router", "ep", "ep2")
SEED = 42
MAX_STEPS = 4
GRAD_ACCUM = 4
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-4
# The arms sum the same bf16 gradients in a different order (microsteps then ranks), and runs are
# otherwise bitwise reproducible (knob off twice: zero deviation). Measured across the modes: |dloss|
# <= 4.1e-4 (DP and HSDP, both correct, differ by 1.2e-3), grad-norm rel dev <= 3.2e-4 (up to one
# bf16 step, 0.8%, where the clip computes the norm in bf16), final weights within about 2% of their
# movement. The bounds sit 2.5-5x above that, far under a dropped or doubled microstep (a quarter of
# the window's gradient).
LOSS_ABS_TOL = 2e-3
GRAD_NORM_RTOL = 2e-2
WEIGHT_REL_TOL = 0.1

_TINY_DENSE = {
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "max_position_embeddings": 4096,
    "tie_word_embeddings": False,
}


def _parallelism_config(mode: str, world_size: int, defer: bool) -> ParallelismConfig:
    """The mode's topology; two simulated NVLink domains where the mode needs more than one."""
    half = world_size // 2
    shapes = {
        "dp": {},
        "hsdp": {"gpus_per_node": half, "use_hsdp": True},
        "tp": {"tp_size": 2},
        "cp": {"cp_size": 2},
        "ep1": {"ep_size": 1},
        "ep1_fp32_router": {"ep_size": 1, "ep_fp32_router": True},
        "ep": {"ep_size": world_size},
        "ep2": {"ep_size": half, "gpus_per_node": half, "ep_scope": "node"},
    }
    return ParallelismConfig(fsdp_defer_grad_sync=defer, **shapes[mode])


def _exercises_mode(mode: str, pc: ParallelismConfig, model) -> bool:
    """Whether the run resolved to the gradient-sync regime the mode names, so a pass covers it."""
    if mode == "hsdp":
        return pc.is_hsdp
    if mode == "ep1":
        return pc.experts_fsdp_managed
    if mode == "ep1_fp32_router":
        return pc.experts_fsdp_managed and any(
            isinstance(p, DTensor) and p.dtype == torch.float32 and p.requires_grad for p in model.parameters()
        )
    if mode in ("ep", "ep2"):
        # ep: in-backward expert/router hooks; ep2: the post-backward sweep over EP-group shards.
        ep_config = pc.create_ep_config()
        sweep = mode == "ep2"
        return ep_config.defer_grad_sync == sweep and ep_config.is_deferred_dp == sweep
    return pc.data_parallel_size > 1


def _build_tiny_checkpoint(mode: str, target_dir: str) -> None:
    """Seeded random-init tiny model + Qwen tokenizer, identical wherever it is built.

    The hub model's padded vocab, not ``len(tokenizer)``: TP's colwise ``lm_head`` needs it divisible
    by ``tp_size``. Written with ``save_file`` because ``save_pretrained`` writes on global rank 0
    only, which leaves every other node of a multi-node launch without weights.
    """
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B, trust_remote_code=True)
    vocab_size = AutoConfig.from_pretrained(QWEN3_0_6B).vocab_size
    torch.manual_seed(SEED)
    if mode in MOE_MODES:
        config = Qwen3MoeConfig(**{**TINY_QWEN3_MOE_CONFIG, "vocab_size": vocab_size, "num_hidden_layers": 2})
        model = Qwen3MoeForCausalLM(config)
    else:
        model = Qwen3ForCausalLM(Qwen3Config(**_TINY_DENSE, vocab_size=vocab_size))
    model.config.save_pretrained(target_dir)
    save_file(
        {name: t.contiguous() for name, t in model.to(torch.bfloat16).state_dict().items()},
        os.path.join(target_dir, "model.safetensors"),
        metadata={"format": "pt"},
    )
    tokenizer.save_pretrained(target_dir)


class WindowProbe(TrainerCallback):
    """Count sharded (FSDP2 DTensor) params carrying a gradient at both ends of a window.

    ``on_substep_end`` fires after microbatches 1..n-1: deferred, nothing has been reduce-scattered
    onto the sharded params yet. ``on_pre_optimizer_step`` fires after the window's last backward
    and the clip, where every sharded param must carry its reduced gradient.
    """

    def __init__(self, model):
        self.model = model
        self.mid_window: list[int] = []
        self.at_step: list[int] = []
        self.n_sharded: list[int] = []

    def _sharded(self):
        return [p for p in self.model.parameters() if isinstance(p, DTensor) and p.requires_grad]

    def on_substep_end(self, args, state, control, **kwargs):
        self.mid_window.append(sum(p.grad is not None for p in self._sharded()))

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        sharded = self._sharded()
        self.n_sharded.append(len(sharded))
        self.at_step.append(sum(p.grad is not None for p in sharded))


def _local_weights(model) -> dict[str, torch.Tensor]:
    """This rank's slice of every trainable param (the DTensor's local shard), as fp32 on the host."""
    return {
        name: (p.to_local() if isinstance(p, DTensor) else p).detach().float().cpu().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def _global_sq_sum(value: float, device) -> float:
    """World sum of a per-rank square sum. A replicated slice counts once per holder, alike in the
    numerator and the denominator of the ratio it feeds."""
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t)
    return float(t.item())


def run_arm(defer: bool, model_dir: str, tokenizer, train_dataset, output_dir: str) -> dict:
    log(f"\n--- {ARGS.mode}: fsdp_defer_grad_sync={defer} ---")
    pc = _parallelism_config(ARGS.mode, dist.get_world_size(), defer)
    model, _ = load_distributed_model(
        model_name_or_path=model_dir,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        # CP needs a flash kernel (auto picks FA4 on Blackwell); sdpa elsewhere.
        attn_implementation=None if pc.is_cp_mode else "sdpa",
        use_liger_kernel=False,
    )
    config = SFTConfig(
        output_dir=output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",
        warmup_steps=0,
        bf16=True,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        seed=SEED,
        data_seed=SEED,
    )
    trainer = DistributedSFTTrainer(
        model=model, args=config, train_dataset=train_dataset, processing_class=tokenizer, parallelism_config=pc
    )
    probe = WindowProbe(trainer.model)
    trainer.add_callback(probe)
    initial = _local_weights(trainer.model)
    trainer.train()

    result = {
        "exercises_mode": _exercises_mode(ARGS.mode, pc, trainer.model),
        "losses": step_losses(trainer),
        "grad_norms": [e["grad_norm"] for e in trainer.state.log_history if "grad_norm" in e],
        "probe": probe,
        "initial": initial,
        "final": _local_weights(trainer.model),
    }
    log(f"  losses:     {[f'{x:.6f}' for x in result['losses']]}")
    log(f"  grad_norms: {[f'{g:.6f}' for g in result['grad_norms']]}")
    log(f"  sharded params with grad mid-window: {probe.mid_window}")
    log(f"  sharded params with grad at step:    {probe.at_step} of {probe.n_sharded}")
    trainer.cleanup_ep()
    del trainer, model
    cleanup_memory()
    return result


@gpu_test_main(min_world_size=4, prefix=f"fsdp_defer_grad_sync_{ARGS.mode}")
def run(ctx):
    mode = ARGS.mode
    # Built per node (local rank 0): a multi-node launch need not share a filesystem, and the seeded
    # build writes the same weights everywhere.
    model_dir = os.path.join(shared_scratch_dir("fsdp_defer_grad_sync"), mode)
    if ctx.local_rank == 0:
        _build_tiny_checkpoint(mode, model_dir)
    ctx.barrier()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = create_sft_dataset(4 * MAX_STEPS * GRAD_ACCUM * ctx.world_size, tokenizer, seed=SEED)

    off = run_arm(False, model_dir, tokenizer, train_dataset, ctx.output_dir)
    ctx.barrier()
    on = run_arm(True, model_dir, tokenizer, train_dataset, ctx.output_dir)

    checks, metrics = {}, {}
    for label, arm in (("off", off), ("on", on)):
        checks[f"{label}_ran_all_steps"] = len(arm["losses"]) == MAX_STEPS and len(arm["probe"].at_step) == MAX_STEPS
        checks[f"{label}_grad_norm_finite_nonzero"] = len(arm["grad_norms"]) == MAX_STEPS and all(
            math.isfinite(g) and g > 0.0 for g in arm["grad_norms"]
        )
    checks["exercises_mode"] = off["exercises_mode"] and on["exercises_mode"]
    checks["off_loss_decreased"] = off["losses"][-1] < off["losses"][0]

    # The deferral itself: the default arm reduce-scatters every microstep (sharded grads set
    # mid-window), the knob's arm only on the window's last backward.
    n_mid = MAX_STEPS * (GRAD_ACCUM - 1)
    checks["off_reduces_every_microstep"] = len(off["probe"].mid_window) == n_mid and all(
        count > 0 for count in off["probe"].mid_window
    )
    checks["on_defers_the_reduce"] = len(on["probe"].mid_window) == n_mid and all(
        count == 0 for count in on["probe"].mid_window
    )
    # ...and the last backward delivers to exactly the sharded params the default arm reduced into.
    checks["on_delivers_at_window_end"] = on["probe"].at_step == off["probe"].at_step and all(
        count > 0 for count in on["probe"].at_step
    )

    loss_dev = max_or_nan(abs(a - b) for a, b in zip(off["losses"], on["losses"], strict=True))
    norm_dev = max_or_nan(abs(a - b) / a for a, b in zip(off["grad_norms"], on["grad_norms"], strict=True))
    diff_sq = sum(float((on["final"][n] - w).square().sum()) for n, w in off["final"].items())
    moved_sq = sum(float((w - off["initial"][n]).square().sum()) for n, w in off["final"].items())
    weight_dev = math.sqrt(_global_sq_sum(diff_sq, ctx.device) / _global_sq_sum(moved_sq, ctx.device))
    metrics.update(max_loss_abs_dev=loss_dev, max_grad_norm_rel_dev=norm_dev, final_weight_rel_dev=weight_dev)
    checks["losses_match"] = loss_dev <= LOSS_ABS_TOL
    checks["grad_norms_match"] = norm_dev <= GRAD_NORM_RTOL
    checks["final_weights_match"] = weight_dev <= WEIGHT_REL_TOL

    log(
        f"\n  {mode}: max |dloss| {loss_dev:.3e} (tol {LOSS_ABS_TOL}), max grad-norm rel dev {norm_dev:.3e} "
        f"(tol {GRAD_NORM_RTOL}), final-weight rel dev {weight_dev:.3e} (tol {WEIGHT_REL_TOL})"
    )
    for name, ok in checks.items():
        log(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
