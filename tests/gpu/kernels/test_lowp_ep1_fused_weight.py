#!/usr/bin/env python
"""FSDP2-managed ep1 expert weights take the fused fake-quant kernel through a real trainer, not the eager chain.

At ``ep_size=1`` with ``fsdp_shard_ep1_experts`` (the default), FSDP2 manages the experts, so ``expert_gemm``
quantizes the all-gathered ``[E, K, N]`` bank on axis 1 every forward (``weight_cacheable=False``). For tiny Qwen3-MoE
and gpt-oss checkpoints at mxfp8, mxfp4 and nvfp4, one forward and backward of the SFT trainer's FSDP2-wrapped model
must, on the step after the first (whose forward also warms each layer's expert kernels)
- reach ``expert_gemm`` with every expert weight as a local tensor, not a DTensor, and hand the kernel local tensors,
- quantize every one of those weights in the kernel, with no eager fallback for a 3-D tensor,
- give, for each, exactly ``dequantize(QUANTIZERS[fmt](w, axis=1))``, the quantizer the compressed-tensors export
  calls.
"""

import math

import torch
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer
from trl import SFTConfig

import halo.kernels.expert_gemm as expert_gemm_module
from halo.distributed.parallelism_config import ParallelismConfig
from halo.kernels.lowp import quantization
from halo.kernels.lowp.spec import LOWP_RECIPES, LowpSpec
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import world_all
from tests.common.ep_reference import ep_layers, fixed_chat_batch
from tests.common.harness import gpu_test_main
from tests.common.memory import finish_phase
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log
from tests.common.tiny_models import TINY_MOE_FAMILIES, shared_tiny_family_checkpoint

SEED = 42
SEQ_LEN = 128
FAMILIES = ("qwen3_moe", "gpt_oss")
# The expert weight's contraction axis in ``[E, K, N]``.
AXIS = 1


class _Routes:
    """For one step: each weight ``expert_gemm`` quantized (its input and output), and the 3-D tensors the kernel and
    the eager quantizer took."""

    def __init__(self):
        self.weights: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.kernel_3d = 0
        self.kernel_dtensors = 0
        self.eager_3d = 0

    def __enter__(self):
        real_fake_quant = expert_gemm_module.fake_quant
        real_kernel = quantization.fake_quant_triton
        real_eager = quantization._block_round_trip
        self._restore = (real_fake_quant, real_kernel, real_eager)

        def fake_quant(t, fmt, axis=-1):
            out = real_fake_quant(t, fmt, axis)
            if axis == AXIS:
                self.weights.append((t.detach().clone(), out.detach().clone()))
            return out

        def kernel(fused_kernel, t, *args, **kwargs):
            self.kernel_dtensors += isinstance(t, DTensor)
            self.kernel_3d += t.ndim == 3
            return real_kernel(fused_kernel, t, *args, **kwargs)

        def eager(t, fmt, axis):
            self.eager_3d += t.ndim == 3
            return real_eager(t, fmt, axis)

        expert_gemm_module.fake_quant = fake_quant
        quantization.fake_quant_triton = kernel
        quantization._block_round_trip = eager
        return self

    def __exit__(self, *_):
        expert_gemm_module.fake_quant, quantization.fake_quant_triton, quantization._block_round_trip = self._restore


def _one_step(ctx, checkpoint: str, tokenizer, precision: str, label: str) -> dict[str, bool]:
    fmt = LOWP_RECIPES[precision].quant_format
    pc = ParallelismConfig(ep_size=1, fsdp_shard_ep1_experts=True, lowp=LowpSpec(precision=precision))
    model, _ = load_distributed_model(
        checkpoint, pc, torch.bfloat16, trust_remote_code=False, attn_implementation="eager", use_liger_kernel=False
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=ctx.output_dir,
            max_steps=1,
            per_device_train_batch_size=1,
            bf16=True,
            gradient_checkpointing=False,  # one forward, so each weight is quantized once
            use_liger_kernel=False,
            max_length=SEQ_LEN,
            save_strategy="no",
            report_to=[],
            dataloader_num_workers=0,
            seed=SEED,
        ),
        train_dataset=create_sft_dataset(8, tokenizer, seed=SEED),
        processing_class=tokenizer,
        parallelism_config=pc,
    )
    wrapped = trainer.model
    layers = ep_layers(wrapped)
    expert_weights = [p for layer in layers for _, p in layer.expert_named_params() if p.ndim == 3]
    fsdp_managed = (
        bool(expert_weights)
        and all(isinstance(p, DTensor) for p in expert_weights)
        and all(layer._lowp_weight_cacheable is False for layer in layers)
    )

    input_ids, attention_mask, labels = fixed_chat_batch(tokenizer, SEQ_LEN, ctx.device, seed=SEED, broadcast=True)
    wrapped.train()
    # The first forward also runs each layer's latched kernel warm-up through expert_gemm, so the second is recorded.
    wrapped(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss.backward()
    wrapped.zero_grad(set_to_none=True)
    with _Routes() as routes:
        loss = wrapped(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        loss.backward()
    loss_value = loss.item()

    local = [not isinstance(w, DTensor) for w, _ in routes.weights]
    exact = [
        torch.equal(out, quantization.dequantize(quantization.QUANTIZERS[fmt](w, axis=AXIS)).to(w.dtype))
        for w, out in routes.weights
    ]
    seen = len(routes.weights)
    log(
        f"  {label}: {len(expert_weights)} expert weights, {seen} quantized on axis {AXIS} "
        f"(local {sum(local)}, bit-exact {sum(exact)}), kernel 3-D calls {routes.kernel_3d}, "
        f"eager 3-D calls {routes.eager_3d}, DTensors at the kernel {routes.kernel_dtensors}, loss {loss_value:.6f}"
    )
    finish_phase(trainer)
    return {
        f"{label}:experts_fsdp_managed": world_all(fsdp_managed),
        f"{label}:every_expert_weight_quantized_once_per_forward": world_all(seen == len(expert_weights)),
        f"{label}:weights_reach_expert_gemm_local": world_all(seen > 0 and all(local)),
        f"{label}:kernel_sees_no_dtensor": world_all(routes.kernel_dtensors == 0),
        f"{label}:every_weight_takes_the_kernel": world_all(routes.kernel_3d == seen and routes.eager_3d == 0),
        f"{label}:weights_bit_exact_with_the_export_quantizer": world_all(seen > 0 and all(exact)),
        f"{label}:loss_finite": world_all(math.isfinite(loss_value)),
    }


@gpu_test_main(exact_world_size=2, prefix="lowp_ep1_fused_weight")
def run(ctx):
    torch.cuda.set_device(ctx.device)
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    checks: dict[str, bool] = {}
    for family in FAMILIES:
        checkpoint = shared_tiny_family_checkpoint(
            ctx, TINY_MOE_FAMILIES[family], f"lowp_ep1_fused_{family}", tokenizer, SEED
        )
        for precision in LOWP_RECIPES:
            checks.update(_one_step(ctx, checkpoint, tokenizer, precision, f"{family}/{precision}"))
    return {"checks": checks}


if __name__ == "__main__":
    run()
