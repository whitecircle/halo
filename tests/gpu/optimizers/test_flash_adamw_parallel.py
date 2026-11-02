#!/usr/bin/env python
"""FlashAdamW (``optim: flash_adamw``) through ``DistributedSFTTrainer`` on two ranks: EP2, dense FSDP2, and an
unevenly sharded model.

FlashAdamW steps per parameter on the local shard, so one weight-decay group holds FSDP2 DTensor shards beside the
plain rank-local tensors EP leaves (``--mode ep2``, a tiny Qwen3-MoE) or only DTensor shards (``--mode fsdp``, a
tiny Qwen3):

  1. **Train.** The last rank poisons step ``POISON_STEP``, which FlashAdamW's host-read step guard must skip (no
     shard moves) while a clean step moves every leg; the losses of the finite steps are finite.
  2. **Save, exact resume, export** (:func:`tests.common.resume_export.run_resume_export`): the sharded optimizer
     save holds FlashAdamW's quantized state, and the resumed run restores it bit for bit and replays the unbroken
     one.

``--mode uneven`` gives the embedding an odd row count, which FSDP2 shards unevenly: the optimizer build must warn
naming ``save_only_model``, the first checkpoint save with optimizer state must raise, and the same run with
``save_only_model`` must save.

Run: torchrun --nproc_per_node=2 tests/gpu/optimizers/test_flash_adamw_parallel.py --mode ep2
"""

import argparse
import math
import os

import torch
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, default_data_collator
from trl import SFTConfig

from halo.distributed.parallelism_config import ParallelismConfig
from halo.optimizers import flash_adamw
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.checkpoint_io import save_schedule
from tests.common.distributed import pin_deterministic_ep_dispatch, shared_output_dir, world_all
from tests.common.harness import gpu_test_main
from tests.common.memory import finish_phase
from tests.common.models import TINY_QWEN3_CONFIG, TINY_QWEN3_MOE_CONFIG
from tests.common.poisoned_step import AroundStep, PoisonSteps
from tests.common.pp_trainers import PAD_ID, build_sft_rows, tiny_vocab_tokenizer
from tests.common.rank_log import log, logged_messages
from tests.common.resume_export import ResumeExportRoute, load_route_model, run_resume_export, save_base_checkpoint
from tests.common.tiny_models import TINY_DENSE_FAMILY, TINY_MOE_FAMILIES, tiny_family_model
from tests.common.training_checks import step_losses

SEED = 42
TOTAL_STEPS = 6
CONTROL_STEP = 2
POISON_STEP = 4
SAVE_STEP = 3
SEQ_LEN = 64
# An odd row count FSDP2 cannot split evenly over two ranks.
UNEVEN_VOCAB = TINY_QWEN3_CONFIG["vocab_size"] - 1
# Each mode's tiny family, text-config overrides and ParallelismConfig fields.
MODES = {
    "ep2": (TINY_MOE_FAMILIES["qwen3_moe"], {"vocab_size": TINY_QWEN3_MOE_CONFIG["vocab_size"]}, {"ep_size": 2}),
    "fsdp": (TINY_DENSE_FAMILY, {}, {}),
    "uneven": (TINY_DENSE_FAMILY, {"vocab_size": UNEVEN_VOCAB}, {}),
}

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=sorted(MODES), required=True)
ARGS = parser.parse_args()
FAMILY, OVERRIDES, FIELDS = MODES[ARGS.mode]


def sft_args(output_dir: str, *, max_steps: int = TOTAL_STEPS, save_every: int | None = None, **fields) -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        optim="flash_adamw",
        per_device_train_batch_size=4,
        learning_rate=1e-3,
        lr_scheduler_type="constant",
        max_length=SEQ_LEN,
        bf16=True,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        logging_steps=1,
        **save_schedule(save_every),
        report_to=[],
        logging_nan_inf_filter=False,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        seed=SEED,
        **fields,
    )


def mixed_decay_group(optimizer) -> bool:
    """Whether one parameter group holds both DTensor shards and plain tensors."""
    return any(
        len({isinstance(param.data, DTensor) for param in group["params"]}) == 2 for group in optimizer.param_groups
    )


@gpu_test_main(exact_world_size=2, prefix=f"flash_adamw_parallel_{ARGS.mode}")
def run(ctx):
    torch.cuda.set_device(ctx.device)
    if FIELDS.get("ep_size", 1) > 1:
        pin_deterministic_ep_dispatch()
    tokenizer = tiny_vocab_tokenizer()
    root = shared_output_dir(ctx)
    torch.manual_seed(SEED)
    model = tiny_family_model(FAMILY, overrides={**OVERRIDES, "pad_token_id": PAD_ID, "eos_token_id": 1})
    base = save_base_checkpoint(ctx, model.to(torch.bfloat16), tokenizer, os.path.join(root, "base"))
    vocab = OVERRIDES.get("vocab_size", TINY_QWEN3_CONFIG["vocab_size"])
    dataset = build_sft_rows(vocab, n_rows=96, seed=SEED, seq_len=SEQ_LEN)
    pc = ParallelismConfig(**FIELDS)

    def route_trainer(model, output_dir, *, max_steps, save_every, resume_checkpoint, **fields):
        return DistributedSFTTrainer(
            model=model,
            args=sft_args(output_dir, max_steps=max_steps, save_every=save_every, **fields),
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=default_data_collator,
            parallelism_config=ParallelismConfig(**FIELDS),
        )

    route = ResumeExportRoute(route_trainer, AutoModelForCausalLM, dtype=torch.bfloat16, attn_implementation="eager")

    if ARGS.mode == "uneven":
        checks = {}
        with logged_messages(flash_adamw.logger) as messages:
            trainer = route_trainer(
                load_route_model(route, base, pc, resume=False),
                os.path.join(root, "with_state"),
                max_steps=1,
                save_every=1,
                resume_checkpoint=None,
            )
            refusal = None
            try:
                trainer.train()
            # Whichever error flashoptim's state_dict raises: the property is that the save refuses, not its type.
            except Exception as error:
                refusal = f"{type(error).__name__}: {error}"
        if ctx.rank == 0:
            checks["build_warned_naming_save_only_model"] = any("save_only_model" in m for m in messages)
        checks["first_save_with_optimizer_state_raised"] = world_all(refusal is not None)
        log(f"  uneven: refusal {refusal.splitlines()[0][:160] if refusal else '<none>'}")
        finish_phase(trainer)
        weights_only = route_trainer(
            load_route_model(route, base, pc, resume=False),
            os.path.join(root, "weights_only"),
            max_steps=1,
            save_every=1,
            resume_checkpoint=None,
            save_only_model=True,
        )
        weights_only.train()
        checks["weights_only_save_written"] = os.path.isdir(os.path.join(root, "weights_only", "checkpoint-1"))
        finish_phase(weights_only)
        return {"checks": ctx.broadcast_checks(checks)}

    trainer = route_trainer(
        load_route_model(route, base, pc, resume=False),
        os.path.join(root, "poisoned"),
        max_steps=TOTAL_STEPS,
        save_every=None,
        resume_checkpoint=None,
    )
    ctx.on_teardown(trainer.cleanup_ep)
    control, poisoned = AroundStep(trainer, CONTROL_STEP), AroundStep(trainer, POISON_STEP)
    for callback in (PoisonSteps(trainer, {POISON_STEP}), control, poisoned):
        trainer.add_callback(callback)
    trainer.train()
    optimizer = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
    checks = {"optimizer_is_flash_adamw": type(optimizer).__name__ == "FlashAdamW"}
    if FIELDS.get("ep_size", 1) > 1:
        checks["one_decay_group_mixes_dtensor_and_plain_params"] = mixed_decay_group(optimizer)
    # Some tensor, not every one: a sub-ULP update of a norm weight lands in the master-weight correction bits alone.
    trainable = {name for name, param in trainer.model.named_parameters() if param.requires_grad}
    checks["a_clean_step_moved_trainable_tensors"] = bool(trainable & control.moved())
    checks["the_poisoned_step_moved_nothing"] = not poisoned.moved()
    finite = [loss for loss in step_losses(trainer) if math.isfinite(loss)]
    checks["the_other_steps_logged_finite_losses"] = len(finite) >= TOTAL_STEPS - 1
    finish_phase(trainer)

    phase_checks, phase_metrics = run_resume_export(
        ctx, route, pc, base=base, root=os.path.join(root, "resume"), save_step=SAVE_STEP, total_steps=TOTAL_STEPS
    )
    return {"checks": checks | phase_checks, "metrics": phase_metrics}


if __name__ == "__main__":
    run()
