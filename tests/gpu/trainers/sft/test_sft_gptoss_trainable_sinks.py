#!/usr/bin/env python
"""Trainable GptOss attention sinks (``train_sinks``) train end-to-end under FA4.

The kernel-level exactness of the FA4 rescale is pinned by ``tests/gpu/kernels/test_fa4_trainable_sink_rescale.py``;
this test pins the trainer path: the loader stamps ``SinksPolicy.TRAINABLE``, the sinks stay trainable
through the parallelism wrap, EVERY layer's sinks receive a non-zero gradient at the optimizer step,
the optimizer moves them, and the loss and grad norms stay finite. Modes: plain FSDP2 (``--mode fsdp``,
sinks DP-sharded as DTensors), TP=2 (``--mode tp``, sinks hand-sliced plain tensors gathered through
the TP seam), EP=2 (``--mode ep``, sinks FSDP-managed next to EP-ignored experts). Packing is on so the
varlen kernel runs; the dense kernel path is covered by the kernel test.

The gradient is probed directly rather than inferred from movement: the sinks are bf16 parameters
under stochastic rounding, and a few steps at a small learning rate move only a fraction of them
(the update is far below one bf16 ULP), so "every layer moved" would be flaky by construction.

Fails if: the policy is not stamped, the sinks are frozen or dropped from the optimizer by any wrap,
the FA4 rescale stops delivering a sink gradient in any layer, or training diverges.
"""

import argparse

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer, TrainerCallback
from trl import SFTConfig

from src.distributed.expert_parallel.base_layer import has_grouped_mm
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import materialize_dtensor
from src.distributed.tensor_parallel.state_dict import iter_tp_sharded_non_dtensor_full
from src.models.patches.gpt_oss_sinks import SinksPolicy, stamped_sinks_policy
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.utils import log

MODEL_NAME = GPT_OSS_20B
MODE_CONFIGS = {"fsdp": {}, "tp": {"tp_size": 2}, "ep": {"ep_size": 2}}
NUM_TRAIN_SAMPLES = 32
MAX_SEQ_LENGTH = 2048
NUM_TRAIN_STEPS = 4
LEARNING_RATE = 5e-5
SEED = 42


class SinkGradProbe(TrainerCallback):
    """Accumulate |grad| of every sink parameter at each optimizer step (local shard / slice)."""

    def __init__(self, model):
        self.model = model
        self.grad_mass: dict[str, float] = {}

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        for name, param in self.model.named_parameters():
            if not name.endswith("self_attn.sinks"):
                continue
            grad = param.grad
            if grad is None:
                self.grad_mass.setdefault(name, 0.0)
                continue
            local = grad.to_local() if isinstance(grad, DTensor) else grad
            self.grad_mass[name] = self.grad_mass.get(name, 0.0) + local.float().abs().sum().item()


def _full_sinks(model) -> dict[str, torch.Tensor]:
    """Every layer's full sink vector on this rank — DTensor shards materialized, TP hand-slices gathered."""
    full = {}
    tp_gathered = dict(iter_tp_sharded_non_dtensor_full(model))
    for name, param in model.named_parameters():
        if not name.endswith("self_attn.sinks"):
            continue
        tensor = tp_gathered.get(name)
        if tensor is None:
            tensor = materialize_dtensor(param.data)
        full[name] = tensor.detach().float().cpu()
    return full


def run(ctx) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(MODE_CONFIGS), required=True)
    mode = parser.parse_args().mode
    log(f"\n{'#' * 70}\n  Trainable GptOss sinks — mode={mode}, world={ctx.world_size}\n{'#' * 70}")

    ensure_model_downloaded(MODEL_NAME, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)

    parallelism_config = ParallelismConfig(use_grouped_gemm=has_grouped_mm(), **MODE_CONFIGS[mode])
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_4",
        use_liger_kernel=True,
        reset_sinks=False,
        train_sinks=True,
    )
    checks = {"policy_stamped_trainable": stamped_sinks_policy(model) is SinksPolicy.TRAINABLE}
    checks["sinks_require_grad_after_load"] = all(
        p.requires_grad for n, p in model.named_parameters() if n.endswith("self_attn.sinks")
    )

    trainer = DistributedSFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=ctx.output_dir,
            max_steps=NUM_TRAIN_STEPS,
            per_device_train_batch_size=1,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,
            packing=True,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            fsdp="",
        ),
        train_dataset=train_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    before = _full_sinks(trainer.model)
    checks["sinks_present_in_wrapped_model"] = len(before) > 0
    checks["sinks_trainable_after_wrap"] = all(
        p.requires_grad for n, p in trainer.model.named_parameters() if n.endswith("self_attn.sinks")
    )
    probe = SinkGradProbe(trainer.model)
    trainer.add_callback(probe)

    trainer.train()
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    grad_norms = [e["grad_norm"] for e in trainer.state.log_history if "grad_norm" in e]
    checks["loss_finite"] = bool(losses) and all(torch.isfinite(torch.tensor(losses)).tolist())
    checks["grad_norm_finite"] = bool(grad_norms) and all(torch.isfinite(torch.tensor(grad_norms)).tolist())

    # Every rank holds a slice of every layer's sinks (DTensor shard or TP head range): sum the
    # per-layer gradient mass over the world before judging, in one fixed name order.
    names = sorted(before)
    mass = torch.tensor([probe.grad_mass.get(name, 0.0) for name in names], device=ctx.device)
    dist.all_reduce(mass)
    gradless = [name for name, m in zip(names, mass.tolist(), strict=True) if m == 0.0]
    log(
        f"  sink gradient reached {len(names) - len(gradless)}/{len(names)} layers"
        + (f"; none in: {gradless[:4]}" if gradless else "")
    )
    checks["every_layer_sinks_get_a_gradient"] = bool(names) and not gradless

    after = _full_sinks(trainer.model)
    moved = sum(not torch.equal(before[name], after[name]) for name in names)
    log(f"  optimizer moved the sinks in {moved}/{len(names)} layers")
    checks["optimizer_moves_sinks"] = moved > 0
    return {
        "checks": checks,
        "metrics": {
            "first_loss": losses[0] if losses else float("nan"),
            "last_loss": losses[-1] if losses else float("nan"),
        },
    }


main = gpu_test_main(exact_world_size=2, prefix="sft_gptoss_trainable_sinks")(run)

if __name__ == "__main__":
    main()
