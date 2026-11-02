#!/usr/bin/env python
"""A non-finite optimizer step is skipped on every rank; weights already non-finite stop the run.

Each row trains ``DistributedSFTTrainer`` on one clip and sharding path:

  * ``--mode ep1``: a tiny Qwen3-MoE at ep1 under FSDP2 (DTensor experts), whose EP clip hands its
    coefficient to AdamWBF16 (``defer_grad_scale``) instead of scaling the gradients.
  * ``--mode ep2``: the same model at ep2, whose experts are FSDP-ignored plain tensors.
  * ``--mode ep2_dp2``: ep2 on four ranks, two EP groups: the deferred cross-replica expert sweep.
  * ``--mode etp2``: pure expert-tensor parallelism, each expert FFN split over both ranks.
  * ``--mode dense``: dense Qwen3-0.6B under FSDP2; also launchable through ``accelerate launch``, whose
    optimizer wrapper the guard steps through.
  * ``--mode tp``: dense Qwen3-0.6B at ``tp_size=2``, whose TP clip reduces DTensor and replicated norms.
  * ``--mode cp``: dense Qwen3-0.6B at ``cp_size=2`` (Ulysses, padded rows).
  * ``--mode qlora``: QLoRA on the dense model, plain DP with the QLoRA gradient sweep.
  * ``--mode pp``: a tiny dense Qwen3 at ``pp_size=2`` on four ranks, two per stage: stage-local parameters,
    the pipeline clip.

The last rank poisons a step by multiplying its final norm's output by NaN, which leaves every forward
collective (DeepEP dispatch included) finite and makes the backward NaN; under PP that rank is the last
stage. Three phases per row:

  1. **Skip.** The last rank poisons step ``POISON_STEP``. The gradient reductions carry the NaN to every
     rank, so every rank reads a NaN global norm and skips: every parameter shard and optimizer moment
     after the step equals its value before it. The later steps train, the windows after the skipped one
     log finite losses, and the logs carry ``nonfinite_steps_skipped``.
  2. **Streak.** At ``max_consecutive_nonfinite_steps=1``, two poisoned steps in a row raise on every rank,
     the first skipped and the second refused.
  3. **Corrupted weights.** Rank 0 writes NaN into one trainable ``q_proj`` shard before a step. The step's
     norm is NaN, and every rank raises naming that parameter instead of skipping.

Run:
    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_nonfinite_step_guard.py --mode ep1
    torchrun --nproc_per_node=4 tests/gpu/trainers/sft/test_sft_nonfinite_step_guard.py --mode ep2_dp2
    torchrun --nproc_per_node=4 tests/gpu/trainers/sft/test_sft_nonfinite_step_guard.py --mode pp
    accelerate launch --config_file launcher-configs/accelerate/fsdp2_gradop_config.yaml --num_processes=2 \\
        tests/gpu/trainers/sft/test_sft_nonfinite_step_guard.py --mode dense
"""

import argparse
import math
import os
from typing import NamedTuple

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, TrainerCallback, default_data_collator
from trl import SFTConfig

from halo.data.collators.factory import select_data_collator
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.runtime.collectives import broadcast_from_rank0
from halo.runtime.consensus import rank_consensus
from halo.runtime.dtensor import to_local
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.adapters import load_peft_model
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import shared_output_dir
from tests.common.harness import gpu_test_main
from tests.common.memory import finish_phase
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_CONFIG
from tests.common.pinned_params import SEED, load_row_model, tiny_family_checkpoint
from tests.common.pp_equivalence import build_dense_model
from tests.common.pp_trainers import build_sft_rows
from tests.common.rank_log import log

FAMILY = "qwen3_moe"
TOTAL_STEPS = 5
POISON_STEP = 3
BATCH_SIZE = 2
MAX_LENGTH = 128
PP_MAX_LENGTH = 64
LEARNING_RATE = 2e-3


class _Mode(NamedTuple):
    world_size: int
    # "moe": the tiny family checkpoint; "dense" / "qlora": Qwen3-0.6B; "pp": the tiny dense Qwen3.
    model: str
    axes: dict


MODES = {
    "ep1": _Mode(2, "moe", {"ep_size": 1}),
    "ep2": _Mode(2, "moe", {"ep_size": 2}),
    "ep2_dp2": _Mode(4, "moe", {"ep_size": 2}),
    "etp2": _Mode(2, "moe", {"ep_size": 1, "expert_tp_size": 2}),
    "dense": _Mode(2, "dense", {}),
    "tp": _Mode(2, "dense", {"tp_size": 2}),
    "cp": _Mode(2, "dense", {"cp_size": 2}),
    "qlora": _Mode(2, "qlora", {}),
    # A stage needs two ranks and must end on an NVLink-domain boundary: one two-GPU domain per stage.
    "pp": _Mode(4, "pp", {"pp_size": 2, "gpus_per_node": 2, "nvlink_domain_size": 2}),
}

_PARSER = argparse.ArgumentParser(description=__doc__)
_PARSER.add_argument("--mode", choices=MODES, required=True)
ARGS = _PARSER.parse_args()
MODE = MODES[ARGS.mode]


def _final_norm(model) -> torch.nn.Module:
    """The decoder's final norm, the one module named ``norm`` (the per-layer and per-head norms are not)."""
    (norm,) = [module for name, module in model.named_modules() if name.split(".")[-1] == "norm"]
    return norm


class _PoisonSteps(TrainerCallback):
    """The last rank multiplies its final norm's output by NaN on the optimizer steps in ``steps``."""

    def __init__(self, trainer, steps):
        self.steps = frozenset(steps)
        self.armed = False
        if self.steps and dist.get_rank() == dist.get_world_size() - 1:
            _final_norm(trainer.model).register_forward_hook(self._poison)

    def _poison(self, module, inputs, output):
        return output * float("nan") if self.armed else None

    def on_step_begin(self, args, state, control, **kwargs):
        self.armed = state.global_step + 1 in self.steps

    def on_step_end(self, args, state, control, **kwargs):
        self.armed = False


def _local_state(trainer) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    """This rank's parameter shards and optimizer moments, copied."""
    model = trainer.model
    reshard_fsdp2_modules(model)
    shards = {name: to_local(param.detach()).clone() for name, param in model.named_parameters()}
    moments = [
        to_local(value).clone()
        for state in trainer.optimizer.state.values()
        for key, value in sorted(state.items())
        if key in ("exp_avg", "exp_avg_sq")
    ]
    return shards, moments


class _AroundStep(TrainerCallback):
    """This rank's parameter shards and optimizer moments right before and right after optimizer step ``step``."""

    def __init__(self, trainer, step: int):
        self.trainer, self.step = trainer, step
        self.before = self.after = None

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step + 1 == self.step:
            self.before = _local_state(self.trainer)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            self.after = _local_state(self.trainer)


class _CorruptShard(TrainerCallback):
    """Rank 0 writes NaN into the first element of ``name``'s shard before optimizer step ``step``."""

    def __init__(self, trainer, name: str, step: int):
        self.trainer, self.name, self.step = trainer, name, step

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step + 1 != self.step or dist.get_rank() != 0:
            return
        model = self.trainer.model
        reshard_fsdp2_modules(model)
        with torch.no_grad():
            to_local(dict(model.named_parameters())[self.name]).view(-1)[0] = float("nan")


def _sft_config(output_dir: str) -> SFTConfig:
    pipelined = MODE.model == "pp"
    return SFTConfig(
        output_dir=output_dir,
        max_steps=TOTAL_STEPS,
        per_device_train_batch_size=4 if pipelined else BATCH_SIZE,
        # Under PP the accumulation steps become the pipeline's microbatches.
        gradient_accumulation_steps=2 if pipelined else 1,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",
        max_length=PP_MAX_LENGTH if pipelined else MAX_LENGTH,
        packing=False,
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
    )


def _trainer_inputs(pc: ParallelismConfig, moe_checkpoint: str | None) -> dict:
    """The row's model, tokenizer, dataset and the trainer kwargs that go with them."""
    if MODE.model == "pp":
        return {
            "model": build_dense_model(),
            "processing_class": AutoTokenizer.from_pretrained(QWEN3_0_6B),
            "train_dataset": build_sft_rows(TINY_QWEN3_CONFIG["vocab_size"], n_rows=128, seq_len=PP_MAX_LENGTH),
            "data_collator": default_data_collator,
        }
    peft_config = None
    if MODE.model == "moe":
        model, tokenizer, _ = load_row_model(moe_checkpoint, "full", pc)
    elif MODE.model == "qlora":
        model, tokenizer, peft_config = load_peft_model(
            "qlora", pc, model_name=QWEN3_0_6B, attn_implementation="eager", use_liger_kernel=False
        )
    else:
        model, tokenizer = load_distributed_model(
            QWEN3_0_6B,
            pc,
            trust_remote_code=False,
            attn_implementation="flash_attention_2" if pc.is_cp_mode else "eager",
            use_liger_kernel=False,
        )
    inputs = {
        "model": model,
        "processing_class": tokenizer,
        "train_dataset": create_sft_dataset(64, tokenizer, seed=SEED),
        "peft_config": peft_config,
    }
    if pc.is_cp_mode:
        # Ulysses has no per-document boundaries: padded rows whose length divides cp_size.
        inputs["data_collator"] = select_data_collator(
            tokenizer, train_on_completions_only=False, use_context_parallel=True, pad_to_multiple_of=pc.cp_size
        )
    return inputs


def _same(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def _raised(trainer) -> str | None:
    """The RuntimeError ``trainer.train()`` raised on this rank, or None."""
    try:
        trainer.train()
    except RuntimeError as error:
        return str(error)
    return None


@gpu_test_main(exact_world_size=MODE.world_size, prefix=f"sft_nonfinite_step_guard_{ARGS.mode}")
def run(ctx):
    torch.cuda.set_device(ctx.device)
    moe_checkpoint = tiny_family_checkpoint(ctx, FAMILY) if MODE.model == "moe" else None
    out = os.path.join(shared_output_dir(ctx), "train_out")

    def make_trainer(*, poison_steps=(), **guard_kwargs):
        pc = ParallelismConfig(**MODE.axes)
        trainer = DistributedSFTTrainer(
            args=_sft_config(out), parallelism_config=pc, **_trainer_inputs(pc, moe_checkpoint), **guard_kwargs
        )
        trainer.add_callback(_PoisonSteps(trainer, poison_steps))
        ctx.on_teardown(trainer.cleanup_ep)
        return trainer

    checks: dict[str, bool] = {}

    log(f"\n[1/3] the last rank poisons step {POISON_STEP} of {TOTAL_STEPS}")
    trainer = make_trainer(poison_steps=(POISON_STEP,))
    around = _AroundStep(trainer, POISON_STEP)
    trainer.add_callback(around)
    result = trainer.train()
    before, after = around.before, around.after
    final, _ = _local_state(trainer)
    skipped = before is not None and after is not None
    checks["every_rank_skipped_the_poisoned_step"] = rank_consensus(
        skipped and _same(list(before[0].values()), list(after[0].values())) and _same(before[1], after[1])
    ).all_ok
    checks["moments_existed_to_compare"] = rank_consensus(skipped and bool(before[1])).all_ok
    checks["later_steps_trained"] = rank_consensus(
        skipped
        and not _same(list(after[0].values()), list(final.values()))
        and all(torch.isfinite(t).all() for t in final.values())
    ).all_ok
    logs = {e["step"]: e for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e}
    log(f"  logged losses: {[round(logs[s]['loss'], 4) for s in sorted(logs)]}")
    checks["poisoned_step_logged_nan"] = math.isnan(logs.get(POISON_STEP, {}).get("loss", 0.0))
    checks["later_windows_logged_finite_losses"] = sorted(logs) == list(range(1, TOTAL_STEPS + 1)) and all(
        math.isfinite(logs[s]["loss"]) for s in logs if s != POISON_STEP
    )
    checks["skip_counted_in_the_logs"] = [logs[s].get("nonfinite_steps_skipped", 0) for s in sorted(logs)] == [
        0 if s < POISON_STEP else 1 for s in range(1, TOTAL_STEPS + 1)
    ]
    checks["run_mean_loss_finite"] = math.isfinite(result.training_loss)
    finish_phase(trainer)
    ctx.barrier()

    log("\n[2/3] two poisoned steps in a row at max_consecutive_nonfinite_steps=1")
    trainer = make_trainer(poison_steps=(2, 3), max_consecutive_nonfinite_steps=1)
    raised = _raised(trainer)
    log(f"  raised: {raised}")
    checks["streak_raises_on_every_rank"] = rank_consensus(
        raised is not None and "Optimizer steps 2-3: 2 consecutive steps" in raised
    ).all_ok
    finish_phase(trainer)
    ctx.barrier()

    log("\n[3/3] rank 0 corrupts one weight shard before step 2")
    trainer = make_trainer()
    # Rank 0's name: under PP the other stage holds other layers.
    name = broadcast_from_rank0(
        next(n for n, p in trainer.model.named_parameters() if p.requires_grad and "q_proj" in n)
    )
    trainer.add_callback(_CorruptShard(trainer, name, step=2))
    raised = _raised(trainer)
    log(f"  raised: {raised}")
    checks["corrupted_weights_raise_on_every_rank_naming_the_parameter"] = rank_consensus(
        raised is not None and "already hold NaN/inf before the update" in raised and name in raised
    ).all_ok
    finish_phase(trainer)
    return {"checks": checks, "metrics": {"final_train_loss": result.training_loss}}


if __name__ == "__main__":
    run()
