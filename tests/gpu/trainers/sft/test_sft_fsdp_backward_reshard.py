#!/usr/bin/env python
"""``fsdp_reshard_after_backward: false`` honors torch's per-window reshard contract and tracks the default arm.

Pinned off for the run, ``post_backward`` never re-registers the sharded DTensors: ``grad_norm`` is 0.0 and the loss
never moves. Never armed looks like the default arm, so only the microstep-end check catches it. At DP=1 FSDP2 is
skipped and the flag is a no-op.
"""

import math

import torch
from datasets import Dataset
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer, TrainerCallback
from trl import SFTConfig

from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.runtime.collectives import barrier
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import QWEN3_0_6B
from tests.common.numerics import max_or_nan
from tests.common.rank_log import log

MODEL_NAME = QWEN3_0_6B
MAX_STEPS = 4
GRAD_ACCUM = 4
DATASET_ROWS = 64
MAX_SEQ_LENGTH = 512
# Moves the loss well past HF's 4-decimal log rounding.
LEARNING_RATE = 5e-5
SEED = 42
# Same reduce-scatters on the same values; only the number of value-preserving all-gathers differs.
CURVE_ABS_TOL = 1e-3


class ReshardInvariantCallback(TrainerCallback):
    """``on_substep_end`` must see parameters unsharded, the half an inert lever drops. ``on_pre_optimizer_step``,
    where every consumer of post-backward parameters reads them, must see them sharded and carrying gradients."""

    def __init__(self, model):
        self.model = model
        self.n_params: list[int] = []
        self.n_dtensor: list[int] = []
        self.n_with_grad: list[int] = []
        self.microstep_dtensor: list[int] = []

    def _dtensor_count(self) -> int:
        return sum(isinstance(p.data, DTensor) for p in self.model.parameters())

    def on_substep_end(self, args, state, control, **kwargs):
        self.microstep_dtensor.append(self._dtensor_count())

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        params = list(self.model.parameters())
        self.n_params.append(len(params))
        self.n_dtensor.append(sum(isinstance(p.data, DTensor) for p in params))
        self.n_with_grad.append(sum(p.grad is not None for p in params))


def run_sft(reshard_after_backward, tokenizer, train_dataset, output_dir):
    log(f"\n--- fsdp_reshard_after_backward={reshard_after_backward} ---")

    # AdamWBF16 stochastically rounds every write, so two arms would drift on RNG; stock AdamW keeps them comparable.
    parallelism_config = ParallelismConfig(fsdp_reshard_after_backward=reshard_after_backward, bf16_optimizer=False)
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_liger_kernel=True,
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
        model=model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    invariants = ReshardInvariantCallback(trainer.model)
    trainer.add_callback(invariants)
    trainer.train()

    history = [e for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]
    losses = [e["loss"] for e in history]
    grad_norms = [e.get("grad_norm") for e in history]
    log(f"  losses:     {[f'{x:.5f}' for x in losses]}")
    log(f"  grad_norms: {[('None' if g is None else f'{g:.5f}') for g in grad_norms]}")
    log(
        f"  at optimizer step: params={invariants.n_params} "
        f"dtensor={invariants.n_dtensor} with_grad={invariants.n_with_grad}"
    )
    log(f"  dtensor at microstep ends: {invariants.microstep_dtensor}")

    del trainer, model
    cleanup_memory()
    return {"losses": losses, "grad_norms": grad_norms, "invariants": invariants}


def _every_step_moved(values) -> bool:
    """Stale weights make every window bit-identical; the net decrease keeps a diverging run from passing on churn
    alone."""
    return (
        len(values) >= 2
        and values[-1] < values[0]
        and all(b != a for a, b in zip(values[:-1], values[1:], strict=True))
    )


def run(ctx) -> dict:
    log(f"\n{'=' * 70}\n  fsdp_reshard_after_backward window contract (world_size={ctx.world_size})\n{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Every microbatch carries one loss surface, so the logged loss moves only if an update reached a forward.
    row = create_sft_dataset(1, tokenizer, seed=SEED, multi_turn_ratio=0.0)[0]
    train_dataset = Dataset.from_list([row] * DATASET_ROWS)

    arms = {}
    for label, flag in (("default", True), ("lever", False)):
        arms[label] = run_sft(flag, tokenizer, train_dataset, ctx.output_dir)
        barrier()

    checks, metrics = {}, {}
    for label, arm in arms.items():
        losses, norms, inv = arm["losses"], arm["grad_norms"], arm["invariants"]
        n_steps = len(inv.n_params)
        checks[f"{label}_ran_all_steps"] = n_steps == MAX_STEPS and len(losses) == MAX_STEPS
        # What the pin breaks: post-backward parameters are the optimizer's sharded DTensors, carrying this window's
        # gradients.
        checks[f"{label}_params_resharded"] = n_steps > 0 and inv.n_dtensor == inv.n_params
        checks[f"{label}_params_carry_grads"] = n_steps > 0 and inv.n_with_grad == inv.n_params
        checks[f"{label}_grad_norm_usable"] = bool(norms) and all(
            g is not None and math.isfinite(g) and g > 0.0 for g in norms
        )
        checks[f"{label}_loss_responds_to_update"] = _every_step_moved(losses)
        # What a never-armed lever breaks: microbatches 1..n-1 end unsharded under the lever, sharded on torch's
        # default.
        expected_mid = 0 if label == "lever" else (inv.n_params[0] if n_steps else -1)
        mid = inv.microstep_dtensor
        checks[f"{label}_microstep_sharding"] = len(mid) == MAX_STEPS * (GRAD_ACCUM - 1) and all(
            count == expected_mid for count in mid
        )
        metrics[f"{label}_first_loss"] = losses[0] if losses else float("nan")
        metrics[f"{label}_last_loss"] = losses[-1] if losses else float("nan")
        metrics[f"{label}_first_grad_norm"] = norms[0] if norms and norms[0] is not None else float("nan")

    default_losses, lever_losses = arms["default"]["losses"], arms["lever"]["losses"]
    aligned = len(default_losses) == len(lever_losses) and bool(default_losses)
    max_dev = (
        max_or_nan(abs(a - b) for a, b in zip(default_losses, lever_losses, strict=True)) if aligned else float("inf")
    )
    checks["curves_match"] = max_dev <= CURVE_ABS_TOL
    metrics["max_curve_deviation"] = max_dev

    log(f"\n  max |default - lever| loss deviation: {max_dev:.6f} (tol {CURVE_ABS_TOL})")
    for name, ok in checks.items():
        log(f"  {'PASS' if ok else 'FAIL'}  {name}")

    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(min_world_size=2, prefix="test_sft_fsdp_backward_reshard")(run)

if __name__ == "__main__":
    main()
