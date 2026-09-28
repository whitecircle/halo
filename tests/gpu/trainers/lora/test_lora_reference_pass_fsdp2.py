#!/usr/bin/env python
"""
Test: a reference pass under ``disable_adapter()`` leaves FSDP2-sharded LoRA adapters trainable.

On-policy RL scores its KL reference inside ``PeftModel.disable_adapter()`` right after an optimizer
step, when the post-backward reshard has registered the sharded params. peft clears
``requires_grad`` on the params registered at entry (the sharded ones) and restores it on those
registered at exit (the unsharded copies the forward leaves behind), and FSDP2 copies the sharded
flag onto the unsharded param at every unshard. Unless the trainer's wrapper
(``make_disable_adapter_fsdp2_safe``) reshards before peft's exit, the sharded adapters stay frozen,
and every micro-step that unshards afresh (the second of each accumulation window) trains without
them — silently, since gradient checkpointing's input-grad hook keeps the loss attached.

This drives that order through the real SFT trainer on FSDP2 data parallelism: a callback runs a
no-grad forward inside the trainer's ``disable_adapter()`` after every optimizer step, as TRL's GRPO
scoring does. Checks:

  1. every step completes with a finite loss, with a reference pass after each;
  2. every training forward (the gradient-checkpoint recomputes included) sees a trainable adapter;
  3. after the last reference pass every sharded adapter is trainable.

Usage:
    torchrun --nproc_per_node=2 tests/gpu/trainers/lora/test_lora_reference_pass_fsdp2.py
"""

import torch
from torch.distributed.tensor import DTensor
from transformers import TrainerCallback
from trl import SFTConfig

from src.distributed.checkpoint.peft import find_peft_model
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.peft_helpers import adapter_param_items, load_peft_model
from tests.common.utils import log, training_run_checks

MAX_STEPS = 3
# Two micro-steps per window: the second unshards afresh after the first one's post-backward reshard.
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-3
NUM_TRAIN_SAMPLES = 64
MAX_SEQ_LENGTH = 256
SEED = 42
REFERENCE_PROMPT = "What is 12 + 30? Answer with one number."


class _ReferencePassAfterStep(TrainerCallback):
    """TRL's GRPO scoring order: after the optimizer step, a no-grad forward under ``disable_adapter()``."""

    def __init__(self, reference_inputs: dict):
        self.reference_inputs = reference_inputs
        self.passes = 0

    def on_step_end(self, args, state, control, model=None, **kwargs):
        with torch.no_grad(), find_peft_model(model).disable_adapter():
            model(**self.reference_inputs)
        self.passes += 1


def run(ctx) -> dict:
    parallelism_config = ParallelismConfig()
    model, tokenizer, peft_config = load_peft_model("lora", parallelism_config, use_liger_kernel=False)
    config = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        fsdp="",
    )
    callback = _ReferencePassAfterStep(dict(tokenizer(REFERENCE_PROMPT, return_tensors="pt").to(ctx.device)))
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED),
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[callback],
        parallelism_config=parallelism_config,
    )
    ctx.on_teardown(trainer.cleanup_ep)

    # One adapter's weight as each grad-enabled forward sees it; the reference passes run without grad.
    adapter = next(module for name, module in trainer.model.named_modules() if name.endswith("lora_A.default"))
    forward_trainable: list[bool] = []
    adapter.register_forward_pre_hook(
        lambda module, args: forward_trainable.append(module.weight.requires_grad) if torch.is_grad_enabled() else None
    )
    train_result = trainer.train()

    # Register the sharded params: they carry the flag every unshard copies, and the optimizer's state.
    reshard_fsdp2_modules(trainer.model)
    adapters = adapter_param_items(trainer.model)
    sharded_trainable = [isinstance(p, DTensor) and p.requires_grad for _, p in adapters]
    log(
        f"  steps {trainer.state.global_step}/{MAX_STEPS}, reference passes {callback.passes}, training forwards "
        f"with a trainable adapter {sum(forward_trainable)}/{len(forward_trainable)}, sharded trainable "
        f"{sum(sharded_trainable)}/{len(sharded_trainable)}"
    )
    checks = training_run_checks(train_result, trainer, MAX_STEPS)
    checks |= {
        "every_step_and_reference_pass_ran": trainer.state.global_step == callback.passes == MAX_STEPS,
        "training_forwards_see_trainable_adapters": len(forward_trainable) >= MAX_STEPS * GRADIENT_ACCUMULATION_STEPS
        and all(forward_trainable),
        "sharded_adapters_trainable": bool(sharded_trainable) and all(sharded_trainable),
    }
    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="lora_reference_pass_fsdp2")(run)

if __name__ == "__main__":
    main()
