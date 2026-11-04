#!/usr/bin/env python
"""QLoRA's gradient sweep is the only sync on two real GPUs. A 4-bit base cannot take FSDP2 and the training loop adds no
DDP wrap over the sweep, so a missed reduce would let the replicas drift at finite losses.

The run trains through the production adapter path and checks: no DDP on the wrapped model, one sweep per optimizer
step, every floating-point parameter and the packed base equal across ranks after training, every adapter moved, and
the loss on the first batch moved.

One mid-run step's synced gradient is then held to a one-rank reference. Rank 0 restores the adapters that step read and
runs both ranks' batches alone, with no collective; rank 0's own unsynced gradient must miss that reference, or the
comparison shows nothing.

``--model gpt_oss_20b`` and ``qwen3_0_6b`` run at ``use_grouped_gemm: false``. ``mixtral_tiny`` is a MoE no EP layer
class wraps, on the default config: it must load at grouped GEMM on, with no EP layers and no EP groups.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 tests/gpu/trainers/lora/test_qlora_grad_sweep.py --model qwen3_0_6b
"""

import argparse
import os
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, TrainerCallback
from trl import SFTConfig

from halo.distributed.grad_reduce import reduce_grads_bucketed
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.dtype import has_packed_quantized_params
from halo.runtime.collectives import barrier
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.adapters import check_adapters_moved, load_peft_model, snapshot_adapters, unwrap
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import max_abs_diff_across
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.lora_sft_modes import MODELS as LORA_MODELS
from tests.common.lora_sft_modes import WORLD_SIZE
from tests.common.memory import gpu_mem_gb
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log
from tests.common.sft_collator import build_sft_collator
from tests.common.sft_modes import BASE_SFT_ARGS, load_suite_tokenizer
from tests.common.tiny_models import TINY_UNWRAPPED_MOE_FAMILY, shared_tiny_family_checkpoint
from tests.common.training_checks import step_losses, training_run_checks

STEPS = 5
# Past the first update, so the reference is taken from a state the sweep's own earlier steps produced.
REFERENCE_STEP = 3
SEED = 42
NUM_SAMPLES = 32
BATCH_SIZE = 2
# The largest relative gap to the reference, per adapter tensor. The synced gradient measured 0.0 on every row; a rank's
# own gradient missed by 0.77 to 2.2.
GRAD_TOLERANCE = 1e-3
MIN_OWN_GRADIENT_GAP = 0.1
# Rounding of the rank mean the trainer logs.
LOGGED_LOSS_TOLERANCE = 1e-3


@dataclass(frozen=True)
class SweepModel:
    attn_implementation: str
    lora_target_modules: tuple[str, ...]
    parallelism: dict[str, Any]
    max_length: int
    learning_rate: float


def _lora_row(name: str) -> SweepModel:
    spec = LORA_MODELS[name]
    return SweepModel(
        spec.attn_implementation,
        spec.lora_target_modules,
        {"use_grouped_gemm": False},
        spec.sft_args["max_length"],
        spec.sft_args["learning_rate"],
    )


MODELS = {
    "gpt_oss_20b": _lora_row("gpt_oss_20b"),
    "qwen3_0_6b": _lora_row("qwen3_0_6b"),
    "mixtral_tiny": SweepModel("sdpa", ("q_proj", "k_proj", "v_proj", "o_proj"), {}, 128, 1e-3),
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", choices=sorted(MODELS), required=True)
ARGS = parser.parse_args()


class _RecordingTrainer(DistributedSFTTrainer):
    """The batches and token counts the first and the reference step computed their losses on."""

    def __init__(self, *args, **kwargs):
        self.recorded: dict[int, dict] = {}
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        step = self.state.global_step
        if step in (0, REFERENCE_STEP):
            self.recorded[step] = {
                "inputs": {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in inputs.items()},
                "num_items": None if num_items_in_batch is None else int(num_items_in_batch),
            }
        return super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )


class _ReferenceStep(TrainerCallback):
    """The adapters the reference step read, and the gradients the optimizer stepped with: at ``on_optimizer_step`` the
    sweep has run, by the clip or the step-pre-hook, and no ``zero_grad`` has."""

    def __init__(self, model):
        self.model = model
        self.before: dict[str, torch.Tensor] = {}
        self.synced: dict[str, torch.Tensor] = {}

    def _trainable(self):
        return ((name, param) for name, param in self.model.named_parameters() if param.requires_grad)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if state.global_step == REFERENCE_STEP:
            self.before = {name: param.detach().clone() for name, param in self._trainable()}

    def on_optimizer_step(self, args, state, control, **kwargs):
        if state.global_step == REFERENCE_STEP:
            self.synced = {name: param.grad.detach().clone() for name, param in self._trainable()}


def _source(ctx) -> tuple[str, Any]:
    if ARGS.model == "mixtral_tiny":
        tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
        return shared_tiny_family_checkpoint(
            ctx, TINY_UNWRAPPED_MOE_FAMILY, "qlora_mixtral", tokenizer, SEED
        ), tokenizer
    name = LORA_MODELS[ARGS.model].name
    return name, load_suite_tokenizer(ctx, name)


def _loss(trainer, model, recorded: dict) -> torch.Tensor:
    """The forward the trainer's loss runs, as one rank, with no collective on the path. The token count is the
    trainer's, which under ``average_tokens_across_devices`` counts every rank's batch."""
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in recorded["inputs"].items()}
    inputs["use_cache"] = False
    if trainer.aux_loss_enabled:
        inputs["output_router_logits"] = True
    if trainer.model_accepts_loss_kwargs and recorded["num_items"] is not None:
        inputs["num_items_in_batch"] = torch.tensor(recorded["num_items"], device=device)
    with trainer.compute_loss_context_manager():
        return model(**inputs).loss


def _rank_weight(trainer, recorded: dict) -> float:
    """What each rank's plain loss weighs in the averaged gradient: a loss normalized over every rank's tokens is scaled
    up by the world size before the average, so the sweep leaves their plain sum; else the sweep's average stays."""
    normalized_globally = (
        trainer.args.average_tokens_across_devices
        and trainer.model_accepts_loss_kwargs
        and recorded["num_items"] is not None
    )
    return 1.0 if normalized_globally else 1.0 / WORLD_SIZE


def _relative_gap(actual: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> float:
    return max(
        ((actual[name].float() - grad.float()).norm() / grad.float().norm().clamp_min(1e-30)).item()
        for name, grad in reference.items()
    )


def _one_rank_reference(trainer, model, step: _ReferenceStep, batches: list[dict]) -> dict[str, float]:
    """Rank 0 only: the reference step's gradient from both ranks' batches on the adapters it read."""
    trainable = {name: param for name, param in model.named_parameters() if param.requires_grad}
    with torch.no_grad():
        for name, param in trainable.items():
            param.copy_(step.before[name])
    model.zero_grad()
    model.train()
    weight = _rank_weight(trainer, batches[0])
    losses, own = [], {}
    for index, batch in enumerate(batches):
        loss = _loss(trainer, model, batch) * weight
        loss.backward()
        losses.append(loss.item())
        if index == 0:
            # The size rank 0's gradient holds before the sweep averages it with its peer's.
            own = {name: param.grad.detach().clone() * WORLD_SIZE for name, param in trainable.items()}
    reference = {name: param.grad.detach().clone() for name, param in trainable.items()}
    logged = [e["loss"] for e in trainer.state.log_history if e.get("step") == REFERENCE_STEP + 1 and "loss" in e]
    return {
        "reference_loss": sum(losses),
        "logged_loss": logged[0] if logged else float("nan"),
        "synced_vs_reference": _relative_gap(step.synced, reference),
        "own_vs_reference": _relative_gap(own, reference),
    }


@gpu_test_main(exact_world_size=WORLD_SIZE, prefix=f"qlora_grad_sweep_{ARGS.model}")
def run(ctx):
    spec = MODELS[ARGS.model]
    source, tokenizer = _source(ctx)
    pc = ParallelismConfig(**spec.parallelism)
    model, _, peft_config = load_peft_model(
        "qlora",
        pc,
        model_name=source,
        attn_implementation=spec.attn_implementation,
        use_liger_kernel=False,
        lora_target_modules=list(spec.lora_target_modules),
    )
    log(f"  Loaded {type(model).__name__} ({pc.summary()}), GPU memory {gpu_mem_gb():.1f}GB")
    config = SFTConfig(
        output_dir=os.path.join(ctx.output_dir, "out"),
        **{
            **BASE_SFT_ARGS,
            "per_device_train_batch_size": BATCH_SIZE,
            "max_steps": STEPS,
            "max_length": spec.max_length,
            "learning_rate": spec.learning_rate,
            # No clip scaling, so the synced gradient reads as the reference does; the sweep still runs ahead of the
            # norm transformers takes for its log.
            "max_grad_norm": 0.0,
        },
    )
    trainer = _RecordingTrainer(
        model=model,
        args=config,
        train_dataset=create_sft_dataset(NUM_SAMPLES, tokenizer, seed=SEED),
        processing_class=tokenizer,
        parallelism_config=pc,
        data_collator=build_sft_collator(tokenizer, config, model.config, pc),
        peft_config=peft_config,
    )
    peft_model = unwrap(trainer.model)
    step = _ReferenceStep(peft_model)
    trainer.add_callback(step)
    before = snapshot_adapters(peft_model, expert_lora=False)
    barrier()
    with patch("halo.trainers.parallelism.grad_sync.reduce_grads_bucketed", wraps=reduce_grads_bucketed) as sweep:
        train_result = trainer.train()

    checks = training_run_checks(train_result, trainer, STEPS, grad_norms=True)
    checks |= {
        "ep_layers_absent": not ep_layers(peft_model),
        "no_ep_groups_built": pc._ep_config is None and trainer._ep_config is None,
        "packed_base": has_packed_quantized_params(peft_model),
        "qlora_sweep_installed": trainer._qlora_grad_sync,
        "no_ddp_wrap": not any(
            isinstance(module, DistributedDataParallel) for module in trainer.model_wrapped.modules()
        ),
        "one_sweep_per_optimizer_step": sweep.call_count == STEPS,
    }
    if ARGS.model == "mixtral_tiny":
        checks["grouped_gemm_left_on"] = pc.use_grouped_gemm
    moved, why = check_adapters_moved(before, snapshot_adapters(peft_model, expert_lora=False), require_all=True)
    log(f"  Every adapter moved: {moved} ({why})")
    checks["every_adapter_moved"] = moved

    parameters = list(peft_model.parameters())
    floating_spread = max_abs_diff_across(param.data for param in parameters if param.is_floating_point())
    packed_spread = max_abs_diff_across(param.data for param in parameters if not param.is_floating_point())
    log(f"  Across ranks: floating-point {floating_spread}, packed {packed_spread}")
    checks["floating_params_identical_across_ranks"] = floating_spread == 0.0
    checks["packed_base_identical_across_ranks"] = packed_spread == 0.0

    batches, first_batches = [None] * WORLD_SIZE, [None] * WORLD_SIZE
    dist.all_gather_object(batches, trainer.recorded[REFERENCE_STEP])
    dist.all_gather_object(first_batches, trainer.recorded[0])
    verdict = [None]
    if ctx.rank == 0:
        with torch.no_grad():
            # Ahead of the reference, which puts earlier adapters back.
            first_batches_after = sum(
                _loss(trainer, peft_model, batch).item() * _rank_weight(trainer, batch) for batch in first_batches
            )
        verdict[0] = _one_rank_reference(trainer, peft_model, step, batches) | {
            "first_batches_loss_after": first_batches_after,
            "first_step_logged_loss": step_losses(trainer)[0],
        }
    dist.broadcast_object_list(verdict, src=0)
    metrics = verdict[0]
    log(f"  Reference: {metrics}")
    checks |= {
        "reference_reproduces_the_logged_loss": abs(metrics["reference_loss"] - metrics["logged_loss"])
        <= LOGGED_LOSS_TOLERANCE,
        "synced_gradient_matches_one_rank_reference": metrics["synced_vs_reference"] <= GRAD_TOLERANCE,
        "own_gradient_misses_the_reference": metrics["own_vs_reference"] >= MIN_OWN_GRADIENT_GAP,
        "first_batch_loss_moved": abs(metrics["first_batches_loss_after"] - metrics["first_step_logged_loss"])
        > LOGGED_LOSS_TOLERANCE,
    }
    trainer.cleanup_ep()
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
