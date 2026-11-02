#!/usr/bin/env python
"""Dense per-row scoring under FSDP2 and reentrant checkpointing trains on every row's gradient, on a tiny MoE.

env-GRPO scores each turn row in its own forward, so one backward recomputes every checkpointed decoder
layer once per row; the trainer runs MoE gradient checkpointing reentrant. torch 2.11's FSDP2 hooks only
one of those recomputes, and with one EP group a post-accumulate expert divide repeated per recompute.
``DistributedSFTTrainer`` here scores each of its 4 rows per microbatch in its own forward and runs the
production wrap, checkpointing and EP clip through three of HF's steps (``training_step``, clip,
``model.zero_grad()``) with no optimizer update, so the weights stay fixed and steps 2 and 3 expose any
gradient carried across ``model.zero_grad()``. Each step's gradients, read off the sharded parameters an
optimizer holds, are compared with an unsharded ep1 model of the same checkpoint run without checkpointing
over every rank's rows: non-expert parameters whole, experts per rank slice. Every shard group must be
sharded after each step.

Layouts: ``ep1`` (2 GPUs, experts FSDP-sharded), ``ep_single`` (4 GPUs, one EP group of 4) and
``ep_multi`` (4 GPUs, two EP groups of 2); ``--adapter lora`` trains attention LoRA over frozen experts;
``--reshard-after-forward`` (ep1 only) runs ZeRO-3; ``--fp32-router`` (ep1) splits each layer's router into
a nested shard group.

    torchrun --nproc_per_node=2 tests/gpu/parallelism/test_fsdp2_reentrant_recompute.py --family qwen3_moe --layout ep1
    torchrun --nproc_per_node=4 tests/gpu/parallelism/test_fsdp2_reentrant_recompute.py --family gpt_oss --layout ep_single
"""

import argparse
import copy
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import get_peft_model
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer
from trl import SFTConfig

from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.adapters import load_peft_model
from tests.common.datasets import create_sft_dataset
from tests.common.ep_reference import compare_ep_grad
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log, log_all
from tests.common.tiny_models import TINY_MOE_FAMILIES, shared_tiny_family_checkpoint
from tests.common.tolerances import TOL

SEED = 42
STEPS = 3
ROWS = 4
ROW_LENGTHS = (24, 64)
# Sum-reduced row cross entropy over ~4 x 40 tokens, scaled to keep bf16 gradients well inside range.
LOSS_SCALE = 256.0
# Far above any norm here: the clip runs its EP sweep and scales by exactly 1.
NO_CLIP = 1e9
# Ranks and ParallelismConfig fields of each layout.
LAYOUTS = {"ep1": (2, {"ep_size": 1}), "ep_single": (4, {"ep_size": 4}), "ep_multi": (4, {"ep_size": 2})}
FAMILIES = ("qwen3_moe", "gpt_oss", "gemma4_text")
LORA_TARGETS = ["q_proj", "o_proj"]


def rows_loss(model, inputs) -> torch.Tensor:
    """Next-token cross entropy summed over the batch's rows, each row its own forward without padding."""
    loss = 0.0
    for ids, mask, labels in zip(inputs["input_ids"], inputs["attention_mask"], inputs["labels"], strict=True):
        keep = mask.bool()
        logits = model(input_ids=ids[keep][None]).logits.float()
        loss = loss + F.cross_entropy(logits[0, :-1], labels[keep][1:], ignore_index=-100, reduction="sum")
    return loss / LOSS_SCALE


class RowsTrainer(DistributedSFTTrainer):
    """SFT whose loss scores each row in its own forward, as env-GRPO scores turn rows."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return rows_loss(model, inputs)


def make_batch(vocab: int, step: int, rank: int, device) -> dict[str, torch.Tensor]:
    """``ROWS`` right-padded rows of different lengths, distinct per step and rank."""
    generator = torch.Generator().manual_seed(SEED + 1000 * step + rank)
    lengths = torch.randint(*ROW_LENGTHS, (ROWS,), generator=generator).tolist()
    width = max(lengths)
    ids = torch.randint(0, vocab, (ROWS, width), generator=generator)
    mask = (torch.arange(width)[None] < torch.tensor(lengths)[:, None]).long()
    labels = ids.masked_fill(mask == 0, -100)
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device), "labels": labels.to(device)}


def full(tensor: torch.Tensor) -> torch.Tensor:
    """COLLECTIVE for a DTensor."""
    return (tensor.full_tensor() if isinstance(tensor, DTensor) else tensor).detach().float()


def expert_slices(model) -> dict[str, slice]:
    """Each EP-sharded expert parameter's name and the expert range this rank holds; FSDP-sharded ep1 experts
    are whole once gathered, and absent."""
    names = {id(param): name for name, param in model.named_parameters()}
    return {
        names[id(param)]: slice(layer.expert_start, layer.expert_end)
        for _, layer in named_ep_layers(model)
        if not layer.ep_config.experts_fsdp_managed
        for _, param in layer.expert_named_params()
        if param.requires_grad
    }


def captured_grads(named_params: list[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor | None]:
    """COLLECTIVE: each parameter's gradient, DTensors gathered (every rank in one order)."""
    return {name: None if param.grad is None else full(param.grad).clone() for name, param in named_params}


def load_policy(args, checkpoint: str, pc: ParallelismConfig):
    """The policy model through the production loader, with its tokenizer and PEFT config (None: full FT)."""
    if args.adapter == "lora":
        model, tokenizer, peft_config = load_peft_model(
            "lora",
            pc,
            model_name=checkpoint,
            attn_implementation="eager",
            use_liger_kernel=False,
            lora_target_modules=LORA_TARGETS,
        )
        # Nonzero B, so A trains too: at B == 0 every A gradient is zero and compares vacuously.
        peft_config.init_lora_weights = False
        return model, tokenizer, peft_config
    model, tokenizer = load_distributed_model(
        checkpoint, pc, torch.bfloat16, trust_remote_code=True, attn_implementation="eager", use_liger_kernel=False
    )
    return model, tokenizer, None


def reference_grads(reference, batches: list[list[dict]], world: int) -> list[dict[str, torch.Tensor]]:
    """Per step, the gradient of the ranks' mean loss over every rank's rows, without checkpointing."""
    per_step = []
    named = [(name, param) for name, param in reference.named_parameters() if param.requires_grad]
    for step_batches in batches:
        reference.zero_grad(set_to_none=True)
        (sum(rows_loss(reference, batch) for batch in step_batches) / world).backward()
        per_step.append({name: param.grad.detach().float().clone() for name, param in named})
    return per_step


def compare_step(step, grads, ref_grads, slices, checks: dict, metrics: dict) -> None:
    ratios, cosines, missing = [], [], []
    for name, grad in grads.items():
        expected = ref_grads.get(name)
        if expected is None:
            missing.append(name)
            continue
        if name in slices:
            expected = expected[slices[name]]
        if not expected.any():
            continue
        ratio, cosine = compare_ep_grad(grad, expected, f"step{step}/{name}")
        ratios.append(ratio)
        cosines.append(cosine)
    low, high = TOL.ep_grad_norm_ratio_band
    checks[f"step{step}_every_parameter_has_a_reference"] = not missing
    checks[f"step{step}_grads_compared"] = len(ratios) > 0
    checks[f"step{step}_grad_norms_match_reference"] = all(low < r < high for r in ratios)
    checks[f"step{step}_grad_directions_match_reference"] = all(c > TOL.ep_grad_cosine_min for c in cosines)
    metrics[f"step{step}_min_ratio"] = min(ratios, default=float("nan"))
    metrics[f"step{step}_max_ratio"] = max(ratios, default=float("nan"))
    metrics[f"step{step}_min_cos"] = min(cosines, default=float("nan"))
    if missing:
        log_all(f"  step {step}: no reference for {missing[:4]}")


def all_groups_sharded(model) -> bool:
    groups = (
        module._get_fsdp_state()._fsdp_param_group for module in model.modules() if isinstance(module, FSDPModule)
    )
    return all(group.is_sharded for group in groups if group is not None)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--family", choices=FAMILIES, required=True)
parser.add_argument("--layout", choices=sorted(LAYOUTS), required=True)
parser.add_argument("--adapter", choices=("full", "lora"), default="full")
parser.add_argument("--reshard-after-forward", action="store_true")
parser.add_argument("--fp32-router", action="store_true")
ARGS = parser.parse_args()


@gpu_test_main(exact_world_size=LAYOUTS[ARGS.layout][0], prefix="fsdp2_reentrant_recompute")
def run(ctx):
    torch.cuda.set_device(ctx.device)
    checks, metrics = {}, {}
    world, fields = LAYOUTS[ARGS.layout]
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    checkpoint = shared_tiny_family_checkpoint(
        ctx, TINY_MOE_FAMILIES[ARGS.family], f"reentrant_{ARGS.family}", tokenizer, SEED
    )
    pc = ParallelismConfig(
        **fields, ep_fp32_router=ARGS.fp32_router, fsdp_reshard_after_forward=ARGS.reshard_after_forward
    )
    log(f"{ARGS.family} {ARGS.layout} {ARGS.adapter}: {pc.mode_string}")
    model, tokenizer, peft_config = load_policy(ARGS, checkpoint, pc)
    reference, _ = load_distributed_model(
        checkpoint,
        # Unwrapped, so the router computes in the run dtype, as the policy's FSDP2-cast router does.
        ParallelismConfig(),
        torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        use_liger_kernel=False,
    )

    config = SFTConfig(
        output_dir=os.path.join(ctx.output_dir, "out"),
        max_steps=STEPS,
        per_device_train_batch_size=ROWS,
        max_grad_norm=NO_CLIP,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        use_liger_kernel=False,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        average_tokens_across_devices=False,
    )
    trainer = RowsTrainer(
        model=model,
        args=config,
        train_dataset=create_sft_dataset(8, tokenizer, seed=SEED),
        processing_class=tokenizer,
        parallelism_config=pc,
        peft_config=peft_config,
    )
    ctx.on_teardown(trainer.cleanup_ep)
    policy = trainer.model
    if peft_config is not None:
        reference = get_peft_model(reference, copy.deepcopy(peft_config))
        reference_params = dict(reference.named_parameters())
        with torch.no_grad():
            for name, param in policy.named_parameters():
                if param.requires_grad:
                    reference_params[name].copy_(full(param))
    reference.train()
    # The trainable parameters an optimizer would hold: the sharded ones, registered until a forward gathers.
    trainable = [(name, param) for name, param in policy.named_parameters() if param.requires_grad]
    slices = expert_slices(policy)
    vocab = policy.get_input_embeddings().num_embeddings

    # HF's per-step order, with gradient accumulation 1.
    trainer.current_gradient_accumulation_steps = 1
    step_grads, batches = [], []
    for step in range(STEPS):
        trainer.state.global_step = step
        trainer.accelerator.gradient_state._set_sync_gradients(True)
        batch = make_batch(vocab, step, ctx.rank, ctx.device)
        trainer.training_step(policy, batch)
        trainer.accelerator.clip_grad_norm_(policy.parameters(), NO_CLIP)
        step_grads.append(captured_grads(trainable))
        policy.zero_grad()
        checks[f"step{step}_every_group_sharded"] = all_groups_sharded(policy)
        gathered = [None] * world
        dist.all_gather_object(gathered, {key: value.cpu() for key, value in batch.items()})
        batches.append([{key: value.to(ctx.device) for key, value in b.items()} for b in gathered])

    for step, ref_grads in enumerate(reference_grads(reference, batches, world)):
        compare_step(step, step_grads[step], ref_grads, slices, checks, metrics)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
