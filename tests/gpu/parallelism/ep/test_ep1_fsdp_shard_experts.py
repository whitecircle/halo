#!/usr/bin/env python
"""``fsdp_shard_ep1_experts``: FSDP's reduce-scatter as the sole expert sync gives the EP hook's gradient at less
memory."""

import torch
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer
from trl import SFTConfig

from halo.distributed.parallelism_config import ParallelismConfig
from halo.hardware import supports_grouped_mm
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.ep_reference import fixed_chat_batch, full_grad
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import GPT_OSS_20B
from tests.common.numerics import cos_sim
from tests.common.rank_log import log, log_all

MODEL_NAME = GPT_OSS_20B
SEQ_LEN = 128
SEED = 42

# Identical input, so the forward loss must be bit-for-bit equal across modes.
LOSS_ABS_TOL = 1e-4
# The two mechanisms reduce in a different float order, so gradients match only to bf16/fp32-reduce noise, tight enough
# to catch a doubled, dropped or per-rank sync.
GRAD_COSINE_MIN = 0.999
GRAD_REL_L2_MAX = 5e-3


def first_expert_weight(model):
    """Through ``expert_named_params()``, independent of the per-model expert attribute layout."""
    ep_layers = [m for _, m in model.named_modules() if hasattr(m, "expert_named_params")]
    for name, module in model.named_modules():
        if hasattr(module, "expert_named_params"):
            params = module.expert_named_params()
            if params:
                pname, p = params[0]
                return f"{name}.{pname}", p, len(ep_layers)
    return None, None, len(ep_layers)


def run_mode(fsdp_shard_ep1_experts, tokenizer, local_rank, output_dir):
    device = f"cuda:{local_rank}"
    mode = "ON" if fsdp_shard_ep1_experts else "OFF"
    log(f"\n{'=' * 70}\nMODE fsdp_shard_ep1_experts={mode} (ep_size=1)\n{'=' * 70}")

    torch.cuda.reset_peak_memory_stats(device)

    parallelism_config = ParallelismConfig(
        ep_size=1,
        use_grouped_gemm=supports_grouped_mm(),
        fsdp_shard_ep1_experts=fsdp_shard_ep1_experts,
    )
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_liger_kernel=True,
    )

    sft_config = SFTConfig(
        output_dir=output_dir,
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        bf16=True,
        gradient_checkpointing=False,  # Keeps the backward graph simple for a clean grad readout.
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=SEQ_LEN,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=True,
    )
    # Required to construct the trainer; the backward below uses a fixed broadcast batch so both modes see identical
    # input.
    train_dataset = create_sft_dataset(num_samples=8, tokenizer=tokenizer, seed=SEED)
    trainer = DistributedSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )

    wrapped = trainer.model
    name, expert_w, n_ep_layers = first_expert_weight(wrapped)
    assert expert_w is not None, (
        f"no EP expert weight found ({n_ep_layers} EP layers detected) — model was not EP-wrapped at ep1. "
        "Confirm use_grouped_gemm enables EP layers for this MoE."
    )
    is_dtensor = isinstance(expert_w, DTensor)
    log(f"  expert weight: {name}  DTensor={is_dtensor}  (FSDP-sharded={is_dtensor})")

    input_ids, attention_mask, labels = fixed_chat_batch(tokenizer, SEQ_LEN, device, seed=SEED, broadcast=True)
    wrapped.train()
    wrapped.zero_grad(set_to_none=True)
    outputs = wrapped(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    loss_val = loss.item()
    log_all(f"  loss={loss_val:.6f}")

    grad = full_grad(expert_w)
    grad_norm = grad.norm().item()
    log(f"  full expert-grad norm={grad_norm:.6f}, shape={tuple(grad.shape)}")

    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    log(f"  peak allocated: {peak_gb:.2f} GB")

    trainer.cleanup_ep()
    del trainer, model, wrapped, outputs, loss
    cleanup_memory()

    return {
        "name": name,
        "is_dtensor": is_dtensor,
        "loss": loss_val,
        "grad": grad,
        "grad_norm": grad_norm,
        "peak_gb": peak_gb,
    }


def run(ctx) -> dict:
    ensure_model_downloaded(MODEL_NAME, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    off = run_mode(False, tokenizer, ctx.local_rank, ctx.output_dir)
    ctx.barrier()
    on = run_mode(True, tokenizer, ctx.local_rank, ctx.output_dir)
    ctx.barrier()

    cos = cos_sim(off["grad"], on["grad"], label=off["name"])
    rel_l2 = ((on["grad"] - off["grad"]).norm() / (off["grad"].norm() + 1e-12)).item()
    loss_diff = abs(on["loss"] - off["loss"])

    checks = {}
    checks["flag_flips_fsdp_ownership"] = (on["is_dtensor"] is True) and (off["is_dtensor"] is False)
    checks["same_expert_weight_compared"] = on["name"] == off["name"]
    checks["loss_parity"] = loss_diff <= LOSS_ABS_TOL
    checks["grad_cosine"] = cos >= GRAD_COSINE_MIN
    checks["grad_rel_l2"] = rel_l2 <= GRAD_REL_L2_MAX
    checks["loss_finite"] = all(
        not (torch.isnan(torch.tensor(v)) or torch.isinf(torch.tensor(v))) for v in (off["loss"], on["loss"])
    )
    checks["memory_win"] = on["peak_gb"] < off["peak_gb"]

    if ctx.rank == 0:
        log(f"\n{'=' * 70}\nRESULTS (world_size={ctx.world_size})\n{'=' * 70}")
        log(f"  expert DTensor:   OFF={off['is_dtensor']}  ON={on['is_dtensor']}")
        log(f"  loss:             OFF={off['loss']:.6f}  ON={on['loss']:.6f}  |Δ|={loss_diff:.2e}")
        log(f"  expert grad norm: OFF={off['grad_norm']:.6f}  ON={on['grad_norm']:.6f}")
        log(f"  grad cosine:      {cos:.6f}  (min {GRAD_COSINE_MIN})")
        log(f"  grad rel-L2:      {rel_l2:.2e}  (max {GRAD_REL_L2_MAX})")
        log(
            f"  peak GB:          OFF={off['peak_gb']:.2f}  ON={on['peak_gb']:.2f}  "
            f"(saved {100 * (off['peak_gb'] - on['peak_gb']) / off['peak_gb']:.1f}%)"
        )

    return {"checks": checks}


main = gpu_test_main(min_world_size=1, prefix="ep1_fsdp_shard_experts")(run)

if __name__ == "__main__":
    main()
