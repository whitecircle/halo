#!/usr/bin/env python
"""Save and exact resume of a PEFT attention adapter beside expert LoRA over frozen quantized expert banks (the Inkling 2
release layout) at EP=2, the shipped expert-LoRA recipe's targets.

A checkpoint holds both halves and no bank tensor: the banks reload from the release. A resume at the saved step,
started from zeroed live adapters, must replay the later steps of the uninterrupted run; a half the restore missed
stays at zero.

Both phases share one model load, since a second EP model in one process deadlocks DeepEP on the first's live process
groups.
"""

import os
import random

import torch
from accelerate.utils import extract_model_from_parallel
from datasets import Dataset
from safetensors.torch import load_file
from trl import SFTConfig

from halo.distributed.expert_parallel.families.inkling2 import EPInkling2MoELayer
from halo.distributed.expert_parallel.moe_layer.expert_lora import gather_expert_lora_adapters
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.distributed.parallelism_config import ParallelismConfig
from halo.models.moe.family_specs import INKLING2
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.adapters import load_peft_model_from_config, peft_model_config, portable_adapter_key
from tests.common.distributed import cleanup_dirs, shared_scratch_dir
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.inkling2 import TINY_INKLING2_TEXT, write_tiny_inkling2_release
from tests.common.rank_log import log
from tests.common.tiny_models import word_level_tokenizer
from tests.common.tolerances import TOL
from tests.common.training_checks import snapshot_trainable, step_losses

EP_SIZE = 2
SAVE_AT_STEP = 2
MAX_STEPS = 4
MAX_SEQ_LENGTH = 64
LEARNING_RATE = 1e-3
SEED = 42
WORDS = [f"w{i}" for i in range(TINY_INKLING2_TEXT["unpadded_vocab_size"] - 2)]
# The shipped Inkling 2 expert-LoRA recipe's: the linear inside o_proj for PEFT, the experts peeled into native adapters.
LORA_TARGET_MODULES = ["o_proj.linear", "gate_up_proj", "down_proj"]


def _text_dataset(rows: int) -> Dataset:
    rng = random.Random(SEED)
    return Dataset.from_dict({"text": [" ".join(rng.choices(WORDS, k=48)) for _ in range(rows)]})


def _is_peft_adapter(name: str) -> bool:
    return ".lora_A." in name or ".lora_B." in name


def _zero_live_adapters(model) -> None:
    """Both halves; a zero A and B take no gradient. The PEFT half is FSDP2-sharded, and a forward leaves unsharded
    copies registered that the next unshard discards, so the sharded ones are put back first."""
    reshard_fsdp2_modules(model)
    with torch.no_grad():
        for module in ep_layers(model):
            for attr in module._expert_lora_attrs:
                getattr(module, f"{attr}_lora_A").zero_()
                getattr(module, f"{attr}_lora_B").zero_()
        for name, param in model.named_parameters():
            if _is_peft_adapter(name):
                param.zero_()


def _adapter_halves(model) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """COLLECTIVE. The expert adapters and the PEFT ones, whole on the host and keyed as the checkpoint keys them."""
    reshard_fsdp2_modules(model)
    experts = {key: value.clone() for key, value in gather_expert_lora_adapters(model).items()}
    peft = {
        portable_adapter_key(name): value
        for name, value in snapshot_trainable(model).items()
        if _is_peft_adapter(name)
    }
    return experts, peft


def _relative_l2(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> float:
    difference = sum((actual[key].float() - expected[key].float()).pow(2).sum().item() for key in expected)
    norm = sum(expected[key].float().pow(2).sum().item() for key in expected)
    return (difference / norm) ** 0.5


def run(ctx) -> dict:
    torch.cuda.set_device(ctx.device)
    release_dir = shared_scratch_dir("inkling2_frozen_resume_release")
    reference_dir = shared_scratch_dir("inkling2_frozen_resume_reference")
    if ctx.rank == 0:
        ctx.on_teardown(lambda: cleanup_dirs(release_dir, reference_dir))
        write_tiny_inkling2_release(release_dir, reference_dir, seed=SEED)
        word_level_tokenizer(WORDS).save_pretrained(release_dir)
    ctx.barrier()
    # EP writes the gathered adapter to one directory, which every rank's restore reads.
    shared_out = [ctx.output_dir]
    torch.distributed.broadcast_object_list(shared_out, src=0)
    output_dir = shared_out[0]
    checkpoint = os.path.join(output_dir, f"checkpoint-{SAVE_AT_STEP}")

    parallelism_config = ParallelismConfig(ep_size=EP_SIZE, frozen_quantized_experts=True)
    model_config = peft_model_config(
        "mixed", parallelism_config, model_name=release_dir, lora_target_modules=LORA_TARGET_MODULES
    )
    model, tokenizer, peft_config = load_peft_model_from_config(
        model_config, parallelism_config, use_liger_kernel=False
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=output_dir,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=2,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="linear",
            warmup_steps=1,
            bf16=True,
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="steps",
            save_steps=SAVE_AT_STEP,
            report_to="none",
            logging_nan_inf_filter=False,
            max_length=MAX_SEQ_LENGTH,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            seed=SEED,
        ),
        train_dataset=_text_dataset(64),
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
        peft_config=peft_config,
    )
    unwrapped = extract_model_from_parallel(trainer.model, recursive=True)
    layers = [module for module in unwrapped.modules() if isinstance(module, EPInkling2MoELayer)]

    log(f"\n[1/2] {MAX_STEPS} uninterrupted steps, checkpoint at step {SAVE_AT_STEP}")
    trainer.train()
    continuous = step_losses(trainer)
    trained_experts, trained_peft = _adapter_halves(unwrapped)
    saved = load_file(os.path.join(checkpoint, "adapter_model.safetensors"))
    bank_stems = tuple(slot.hub_stem for slot in INKLING2.frozen_expert_slots)
    log(
        f"  losses {[f'{loss:.6f}' for loss in continuous]}; {len(trained_experts)} expert and {len(trained_peft)} "
        f"PEFT adapter tensors; checkpoint files {sorted(os.listdir(checkpoint))}"
    )

    log(f"\n[2/2] zero the live adapters, resume from step {SAVE_AT_STEP}")
    _zero_live_adapters(unwrapped)
    trainer.train(resume_from_checkpoint=checkpoint)
    resumed = step_losses(trainer)
    replayed_experts, replayed_peft = _adapter_halves(unwrapped)
    loss_gaps = [abs(a - b) for a, b in zip(resumed[SAVE_AT_STEP:], continuous[SAVE_AT_STEP:], strict=True)]
    expert_gap = _relative_l2(replayed_experts, trained_experts)
    peft_gap = _relative_l2(replayed_peft, trained_peft)
    log(
        f"  resumed losses {[f'{loss:.6f}' for loss in resumed]}; gaps {loss_gaps}; adapter rel L2 experts "
        f"{expert_gap:.3e}, PEFT {peft_gap:.3e}"
    )

    trained = {**trained_experts, **trained_peft}
    checks = {
        "every_layer_holds_frozen_banks": bool(layers) and all(layer.has_frozen_banks for layer in layers),
        "ran_every_step": len(continuous) == MAX_STEPS and len(resumed) == MAX_STEPS,
        "both_halves_present": bool(trained_experts) and bool(trained_peft),
        "adapters_trained": all(value.abs().sum().item() > 0 for value in trained.values()),
        "checkpoint_holds_every_adapter": set(saved) == set(trained),
        "checkpoint_holds_no_bank": not any(
            "_frozen_" in key or key.rpartition(".")[2].startswith(bank_stems) for key in saved
        )
        and not any(name.startswith("model") and name.endswith(".safetensors") for name in os.listdir(checkpoint)),
        "resumed_losses_replay_the_run": max(loss_gaps) <= TOL.replayed_resume_loss_abs,
        "resumed_expert_adapters_replay_the_run": expert_gap <= TOL.replayed_resume_weight_rtol,
        "resumed_peft_adapters_replay_the_run": peft_gap <= TOL.replayed_resume_weight_rtol,
    }
    # Every rank reads rank 0's checkpoint directory, and rank 0's teardown removes it.
    ctx.barrier()
    return {
        "checks": checks,
        "metrics": {
            "max_loss_gap": max(loss_gaps),
            "expert_adapter_rel_l2": expert_gap,
            "peft_adapter_rel_l2": peft_gap,
        },
    }


main = gpu_test_main(exact_world_size=EP_SIZE, prefix="lora_frozen_banks_resume")(run)

if __name__ == "__main__":
    main()
