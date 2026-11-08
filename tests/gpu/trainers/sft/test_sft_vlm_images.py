#!/usr/bin/env python
"""SFT on image batches through the raw VLM path, on the attention a run with it unset resolves to: the pixels reach the
forward, the vision tower trains or stays frozen as asked, and the steps complete.

``--data mixed`` holds an image on some rows and text on the rest, so ranks of one step differ in whether they reach the
tower; ``--freeze-tower`` takes the tower out of their gradient reduce, ``--freeze-embedding`` leaves a text row's input
carrying a gradient only through checkpointing's hook, ``--zero3`` reshards every group after its forward, and
``--tower-adapters`` trains adapters on the tower's checkpointed blocks alone, behind its frozen patch embedding, under
reentrant checkpointing whatever the family's default.

    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_vlm_images.py --family qwen3_5 --data all
    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_vlm_images.py --family gemma4 --data mixed --freeze-tower
    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_vlm_images.py --family qwen3_5 --data mixed --zero3
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset, DatasetDict, Sequence
from datasets import Image as HFImage
from peft import LoraConfig
from PIL import Image
from transformers.modeling_layers import GradientCheckpointingLayer
from trl import SFTConfig

from halo.configs.script_arguments import ModelArguments
from halo.configs.sft import SFTScriptArguments
from halo.distributed.parallelism_config import ParallelismConfig
from halo.entrypoints.data_setup import apply_vision_options, prepare_raw_vlm_split
from halo.entrypoints.model_setup import install_resolved_tokenizer, load_policy_model
from halo.models.attention.selection import effective_attn_implementation
from halo.models.introspection.module_tree import embedding_module, media_module_names
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.distributed import shared_output_dir
from tests.common.harness import gpu_test_main
from tests.common.models import GEMMA4_26B_A4B_PATCHED, QWEN3_5_2B, QWEN3_6_MOE_35B, QWEN3_VL_2B
from tests.common.nemotron_h import write_tiny_hub_checkpoint
from tests.common.rank_log import log
from tests.common.tiny_processors import tiny_processor
from tests.common.training_checks import training_run_checks

# Each family's checkpoint and the expert parallelism it trains at. Nemotron-H's is written here at its tiny shape.
FAMILIES = {
    "qwen3_vl": (QWEN3_VL_2B, 1),
    "qwen3_5": (QWEN3_5_2B, 1),
    "qwen3_5_moe": (QWEN3_6_MOE_35B, 2),
    "gemma4": (GEMMA4_26B_A4B_PATCHED, 2),
    "nemotron_h_omni": (None, 2),
}
STEPS = 12
IMAGE_SIDE = 64
MAX_LENGTH = 1024
SEED = 0

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
parser.add_argument("--data", choices=["all", "mixed"], required=True)
parser.add_argument("--freeze-tower", action="store_true")
parser.add_argument("--freeze-embedding", action="store_true")
parser.add_argument("--zero3", action="store_true")
parser.add_argument("--tower-adapters", action="store_true")
ARGS = parser.parse_args()


def _rows(count: int) -> Dataset:
    """Half the rows of ``mixed`` carry no image, in the order a seeded sampler shuffles."""
    rows = []
    for index in range(count):
        has_image = ARGS.data == "all" or index % 2 == 0
        pixels = (np.random.RandomState(index).rand(IMAGE_SIDE, IMAGE_SIDE, 3) * 255).astype("uint8")
        rows.append(
            {
                "prompt": [
                    {"role": "user", "content": f"Describe picture {index}."},
                    {"role": "assistant", "content": "A square of coloured noise."},
                ],
                "images": [Image.fromarray(pixels)] if has_image else [],
            }
        )
    return Dataset.from_list(rows).cast_column("images", Sequence(HFImage()))


def _written_checkpoint(ctx) -> str:
    directory = Path(shared_output_dir(ctx)) / "checkpoint"
    if ctx.rank == 0:
        write_tiny_hub_checkpoint(directory, weights_dtype=torch.bfloat16, processor=tiny_processor(ARGS.family))
    dist.barrier()
    return str(directory)


def _tower_block_linears(model) -> list[str]:
    return [
        f"{tower}.{block_name}.{name}"
        for tower in media_module_names(model)
        for block_name, block in model.get_submodule(tower).named_modules()
        if isinstance(block, GradientCheckpointingLayer)
        for name, module in block.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]


def _adapter_weights(model) -> dict[str, torch.Tensor]:
    """``lora_B``, as ``lora_A`` takes no gradient until ``lora_B`` leaves its zero init. Every rank gathers every one."""
    return {name: _materialized(param).clone() for name, param in model.named_parameters() if ".lora_B." in name}


@gpu_test_main(exact_world_size=2, prefix="sft_vlm_images")
def run(ctx):
    torch.cuda.set_device(ctx.device)
    model_name, ep_size = FAMILIES[ARGS.family]
    model_name = model_name or _written_checkpoint(ctx)
    config = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=STEPS,
        per_device_train_batch_size=1,
        learning_rate=2e-5,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True} if ARGS.tower_adapters else None,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=MAX_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        dataset_kwargs={"skip_prepare_dataset": True},
        seed=SEED,
    )
    parallelism_config = ParallelismConfig(
        **({"ep_size": ep_size} if ep_size > 1 else {}), fsdp_reshard_after_forward=ARGS.zero3
    )
    model_config = ModelArguments(model_name_or_path=model_name)
    args = SFTScriptArguments(
        dataset=[model_name],
        images_field="images",
        train_on_completions_only=False,
        freeze_vision_tower=ARGS.freeze_tower,
    )
    context = SimpleNamespace(
        parallelism_config=parallelism_config, model_source=model_name, policy_from_checkpoint=False
    )
    policy = load_policy_model(context, config, model_config, args, vlm_run=True)
    model = policy.model
    resolved = effective_attn_implementation(model.config)
    log(f"{ARGS.family}: the unset attention resolved to {resolved}")
    processor = install_resolved_tokenizer(policy.processing_class, policy.tokenizer)
    if ARGS.freeze_embedding:
        embedding_module(model, "input").requires_grad_(False)
    apply_vision_options(args, model, model_config, processor, is_vlm=True)

    split = _rows(STEPS * ctx.world_size * 2)
    train_dataset, _, collator = prepare_raw_vlm_split(
        DatasetDict({"train": split, "test": split}),
        args,
        config,
        processor,
        policy.tokenizer,
        model.config,
        desc=f"sft_vlm_images {ARGS.family} {ARGS.data}",
    )
    towers = tuple(f"{tower}." for tower in media_module_names(model))
    tower_start = {name: param.detach().clone() for name, param in model.named_parameters() if name.startswith(towers)}
    targets = _tower_block_linears(model)
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=processor,
        parallelism_config=parallelism_config,
        peft_config=LoraConfig(r=2, target_modules=targets, task_type="CAUSAL_LM") if ARGS.tower_adapters else None,
    )
    ctx.on_teardown(trainer.cleanup_ep)
    adapters_start = _adapter_weights(trainer.model)

    pixels_reached: list[bool] = []
    trainer.model.register_forward_pre_hook(
        lambda _module, _args, kwargs: pixels_reached.append(kwargs.get("pixel_values") is not None),
        with_kwargs=True,
    )
    result = trainer.train()

    checks = training_run_checks(result, trainer, STEPS)
    gathered = [None] * ctx.world_size
    dist.all_gather_object(gathered, pixels_reached)
    per_rank = torch.tensor(gathered)
    checks["pixels_reach_the_forward"] = bool(per_rank.any())
    if ARGS.data == "all":
        checks["every_row_carries_pixels"] = bool(per_rank.all())
    else:
        checks["a_step_mixes_image_and_text_ranks"] = bool((per_rank.any(dim=0) & ~per_rank.all(dim=0)).any())
    if ARGS.tower_adapters:
        adapters_end = _adapter_weights(trainer.model)
        checks["every_tower_adapter_trains"] = bool(adapters_start) and all(
            not torch.equal(adapters_end[name], start) for name, start in adapters_start.items()
        )
        return {"checks": checks, "metrics": {"final_loss": result.training_loss}}
    # Not a generator: ``full_tensor`` is collective, so every rank must gather every parameter.
    moved = [
        not torch.equal(_materialized(param), tower_start[name.replace("_checkpoint_wrapped_module.", "")])
        for name, param in model.named_parameters()
        if name.startswith(towers)
    ]
    tower_moved = any(moved)
    checks["tower_trains_only_when_not_frozen"] = tower_moved != ARGS.freeze_tower
    return {"checks": checks, "metrics": {"final_loss": result.training_loss}}


def _materialized(param: torch.Tensor) -> torch.Tensor:
    return param.full_tensor() if hasattr(param, "full_tensor") else param


if __name__ == "__main__":
    run()
