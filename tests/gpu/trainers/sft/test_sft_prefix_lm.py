#!/usr/bin/env python
"""PaliGemma and PaliGemma 2 SFT on CUDA trains each answer token on its past alone, with the image and prompt
kept a bidirectional prefix.

transformers builds a PaliGemma decoder non-causal and derives the prefix-LM mask from ``token_type_ids``, which no
flash kernel reads and the processor leaves all prefix in a rendered conversation, so an answer trained on its own
next tokens fits to a plausible loss. Each row holds the tiny checkpoints of ``tests/common/prefix_lm.py``, loaded
through ``load_distributed_model`` with the attention unset as a run leaves it, to:

  * the backend the load resolves: SDPA for PaliGemma and FlexAttention for the softcapped PaliGemma 2, never the
    flash kernel the GPU auto-detects;
  * a causal decoder beside a bidirectional SigLIP tower;
  * on the resolved CUDA backend in fp32: swapping an answer token moves no logit before it, swapping a prompt token
    moves an image position's, an all-prefix ``token_type_ids`` lets the earlier answer tokens see the swap, and the
    loss matches eager's on the same weights;
  * every training forward reaching ``PaliGemmaModel.forward`` with the collator's ``token_type_ids`` (1 from the
    first scored label on), over finite steps that move the loss.

``--layout fsdp-gc`` trains on two ranks under FSDP2 with gradient checkpointing, whose decoder layers must recompute
in the backward.

    torchrun --nproc_per_node=1 tests/gpu/trainers/sft/test_sft_prefix_lm.py --family paligemma2
    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_prefix_lm.py --family paligemma --layout fsdp-gc
"""

import argparse
import math
import shutil

import torch
from datasets import Dataset, Sequence
from datasets import Image as HFImage
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaModel
from trl import SFTConfig

from halo.distributed.fsdp import fsdp2_param_groups
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.attention.prefix_lm import prefix_lm_token_type_ids
from halo.models.attention.selection import effective_attn_implementation
from halo.models.gradient_checkpointing import in_backward_pass
from halo.models.introspection.module_tree import backbone_with_layers, decoder_layers
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.distributed import shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.prefix_lm import (
    digit_row,
    first_scored_position,
    paligemma_collator,
    tiny_paligemma_config,
    tiny_paligemma_processor,
)
from tests.common.rank_log import log, log_all
from tests.common.training_checks import step_losses, training_run_checks

# Each family's text model and the backend a load with the attention unset must resolve to.
FAMILIES = {"paligemma": ("gemma", "sdpa"), "paligemma2": ("gemma2", "flex_attention")}
# fsdp-gc also trains with gradient checkpointing.
LAYOUTS = {"single": 1, "fsdp-gc": 2}
SEED = 0
STEPS = 5
BATCH = 2
LEARNING_RATE = 1e-3
# fp32 logit movement that counts as attention reaching a token; a token outside a position's mask moves it by 0.
REACHED = 1e-4
# fp32 loss on the resolved backend against eager on the same weights and batch.
LOSS_RTOL = 1e-4
SWAP_IN_WORD = "The"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
parser.add_argument("--layout", choices=sorted(LAYOUTS), default="single")
ARGS = parser.parse_args()
TEXT_MODEL_TYPE, EXPECTED_BACKEND = FAMILIES[ARGS.family]
GRADIENT_CHECKPOINTING = ARGS.layout == "fsdp-gc"


def build_checkpoint(ctx) -> str:
    """COLLECTIVE: rank 0 saves the random-init tiny model and its processor to a rank-shared dir."""
    path = shared_scratch_dir(f"prefix_lm_{ARGS.family}")
    if ctx.rank == 0:
        shutil.rmtree(path, ignore_errors=True)
        ctx.on_teardown(lambda: shutil.rmtree(path, ignore_errors=True))
        processor = tiny_paligemma_processor()
        torch.manual_seed(SEED)
        model = PaliGemmaForConditionalGeneration(tiny_paligemma_config(processor, text_model_type=TEXT_MODEL_TYPE))
        model.save_pretrained(path)
        processor.save_pretrained(path)
    ctx.barrier()
    return path


def load(path: str, dtype: torch.dtype, attn_implementation: str | None, device):
    """COLLECTIVE: the checkpoint through the production loader, on ``device``."""
    model, _ = load_distributed_model(
        path,
        ParallelismConfig(),
        dtype,
        trust_remote_code=False,
        attn_implementation=attn_implementation,
        use_liger_kernel=False,
    )
    return model.to(device)


def causal_flags(module: torch.nn.Module) -> list[bool]:
    return [child.is_causal for child in module.modules() if hasattr(child, "is_causal")]


def logits(model, batch: dict, **overrides) -> torch.Tensor:
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    inputs.update(overrides)
    with torch.no_grad():
        return model(**inputs).logits[0].float()


def movement(model, batch: dict, position: int, swap_in: int, **overrides) -> torch.Tensor:
    swapped = batch["input_ids"].clone()
    swapped[0, position] = swap_in
    before = logits(model, batch, **overrides)
    after = logits(model, batch, **{**overrides, "input_ids": swapped})
    return (before - after).abs().amax(dim=-1)


def mask_probe_checks(path: str, batch: dict, swap_in: int, device) -> tuple[dict, dict]:
    """COLLECTIVE (the loads): the prefix-LM mask on the resolved backend in fp32, and its loss against eager's."""
    probe = load(path, torch.float32, EXPECTED_BACKEND, device).eval()
    length = batch["input_ids"].shape[1]
    start = first_scored_position(batch["labels"][0].cpu())
    answer_token = length - 2
    leak = movement(probe, batch, answer_token, swap_in)[:answer_token].max().item()
    prefix = movement(probe, batch, start - 2, swap_in)[1].item()
    all_prefix = torch.zeros_like(batch["input_ids"])
    unmasked = (
        movement(probe, batch, answer_token, swap_in, token_type_ids=all_prefix)[start:answer_token].max().item()
    )
    with torch.no_grad():
        loss = probe(**batch).loss.item()
    del probe
    eager = load(path, torch.float32, "eager", device).eval()
    with torch.no_grad():
        eager_loss = eager(**batch).loss.item()
    del eager
    loss_rel = abs(loss - eager_loss) / abs(eager_loss)
    log(
        f"  fp32 {EXPECTED_BACKEND}: answer-swap movement before it {leak:.3e}, prompt-swap movement at an image "
        f"position {prefix:.3e}, all-prefix movement {unmasked:.3e}; loss {loss:.6f} vs eager {eager_loss:.6f}"
    )
    checks = {
        "answer_token_moves_no_earlier_logit": leak == 0.0,
        "prefix_is_bidirectional": prefix > REACHED,
        "probe_sees_an_unmasked_suffix": unmasked > REACHED,
        "loss_matches_eager": loss_rel < LOSS_RTOL,
    }
    metrics = {"answer_swap_leak": leak, "prompt_swap_image_movement": prefix, "loss_rel_vs_eager": loss_rel}
    return checks, metrics


def record_forwards(model) -> tuple[list, list]:
    """Each training forward's ``labels`` and the ``token_type_ids`` its ``PaliGemmaModel`` receives."""
    labels, token_type_ids = [], []
    model.register_forward_pre_hook(
        lambda _module, _args, kwargs: labels.append(kwargs.get("labels")), with_kwargs=True
    )
    model.model.register_forward_pre_hook(
        lambda _module, _args, kwargs: token_type_ids.append(kwargs.get("token_type_ids")), with_kwargs=True
    )
    return labels, token_type_ids


def count_recomputes(model) -> list[int]:
    """A running count of decoder-layer forwards inside a backward: the recomputes gradient checkpointing runs."""
    calls = [0]

    def count(_module, _args) -> None:
        if in_backward_pass():
            calls[0] += 1

    for layer in decoder_layers(backbone_with_layers(model)):
        layer.register_forward_pre_hook(count)
    return calls


@gpu_test_main(exact_world_size=LAYOUTS[ARGS.layout], prefix="sft_prefix_lm")
def run(ctx):
    torch.cuda.set_device(ctx.device)
    checks, metrics = {}, {}
    path = build_checkpoint(ctx)
    processor = tiny_paligemma_processor()
    swap_in = processor.tokenizer.convert_tokens_to_ids(SWAP_IN_WORD)

    model = load(path, torch.bfloat16, None, ctx.device)
    resolved = effective_attn_implementation(model.config)
    log(f"{ARGS.family} {ARGS.layout}: the unset attention resolved to {resolved}")
    checks["resolves_off_flash"] = resolved == EXPECTED_BACKEND
    decoder, tower = causal_flags(model.model.language_model), causal_flags(model.model.vision_tower)
    checks["decoder_is_causal"] = bool(decoder) and all(decoder)
    checks["vision_tower_is_bidirectional"] = bool(tower) and not any(tower)

    collator = paligemma_collator(processor, model.config, completions_only=True)
    probe_batch = {key: value.to(ctx.device) for key, value in collator([digit_row()]).items()}
    probe_checks, probe_metrics = mask_probe_checks(path, probe_batch, swap_in, ctx.device)
    checks.update(probe_checks)
    metrics.update(probe_metrics)

    rows = [digit_row(index % 10) for index in range(BATCH * STEPS * ctx.world_size)]
    dataset = Dataset.from_list(rows).cast_column("images", Sequence(HFImage()))
    config = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=STEPS,
        per_device_train_batch_size=BATCH,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=collator.max_length,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        seed=SEED,
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
        parallelism_config=ParallelismConfig(),
    )
    ctx.on_teardown(trainer.cleanup_ep)
    if not isinstance(model.model, PaliGemmaModel):
        raise AssertionError(f"the trained model's backbone is a {type(model.model).__name__}, not PaliGemmaModel")
    labels, token_type_ids = record_forwards(model)
    recomputes = count_recomputes(model)
    result = trainer.train()

    checks.update(training_run_checks(result, trainer, STEPS, grad_norms=True))
    losses = step_losses(trainer)
    checks["loss_moved"] = len(losses) >= 2 and abs(losses[-1] - losses[0]) > REACHED
    reached = [
        tti is not None and label is not None and torch.equal(tti, prefix_lm_token_type_ids(label))
        for label, tti in zip(labels, token_type_ids, strict=True)
    ]
    checks["every_training_forward_seen"] = len(reached) == STEPS
    checks["token_type_ids_reach_the_forward"] = bool(reached) and all(reached)
    checks["token_type_ids_mark_a_suffix"] = all(
        tti is not None and tti.any() and not tti.all() for tti in token_type_ids
    )
    if not all(reached):
        log_all(f"  forwards without the collator's token_type_ids: {[i for i, ok in enumerate(reached) if not ok]}")
    if ARGS.layout == "fsdp-gc":
        checks["fsdp2_wrapped"] = bool(fsdp2_param_groups(model))
        checks["decoder_layers_recomputed_in_the_backward"] = recomputes[0] > 0
    metrics.update(final_loss=losses[-1] if losses else math.nan, backward_recomputes=recomputes[0])
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
