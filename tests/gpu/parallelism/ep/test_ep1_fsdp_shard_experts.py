#!/usr/bin/env python
"""
Grad-equivalence + memory test for ``fsdp_shard_ep1_experts`` (ep_size==1).

At ep_size==1 the MoE experts are replicated (the DeepEP dispatch is a no-op), so
there are two ways to keep them in sync across DP ranks:

  * ``fsdp_shard_ep1_experts=False``: the experts are handed to FSDP as
    ``ignored_params`` (replicated on every rank), and the EP layer's own
    post-backward hook DP-averages their gradient.
  * ``fsdp_shard_ep1_experts=True``: FSDP shards the experts like any other param
    and its reduce-scatter is their *sole* gradient sync (the EP layer skips its
    hook). Frees the replicated-expert memory that otherwise grows with DP size.

Both paths must produce the SAME (DP-averaged) expert gradient. This test builds
the real ``DistributedSFTTrainer`` for each mode (so the trainer-side and
layer-side gates both fire), runs one forward+backward on identical input, and
checks:

  1. Structural — the flag flips FSDP ownership: with the flag ON the expert
     weight is a sharded ``DTensor``; with it OFF it is a plain replicated param.
  2. Loss parity — identical input → identical forward loss (the sync only
     changes the backward).
  3. Gradient equivalence — the full (gathered) expert gradient matches between
     the two modes (cosine ~1, small relative L2). A doubled / dropped / per-rank
     sync would blow past the tolerance.
  4. Memory — peak allocated memory with the flag ON is strictly below OFF (the
     point of the feature); the win grows with DP size.

Run with 2 GPUs (correctness) or 8 GPUs (the largest-all-gather scale gate):
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_ep1_fsdp_shard_experts.py

Requirements:
    - 2x (or 8x) GPUs with >=80GB memory each
    - DeepEP installed
    - Model: unsloth/gpt-oss-20b-BF16 (auto-downloaded)
"""

import random

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.expert_parallel.base_layer import has_grouped_mm
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.ep_reference import full_grad
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.utils import cleanup_memory, log, log_all

MODEL_NAME = GPT_OSS_20B
SEQ_LEN = 128
SEED = 42

# Identical input → the forward loss must be bit-for-bit equal across modes.
LOSS_ABS_TOL = 1e-4
# The two sync mechanisms (FSDP reduce-scatter vs EP-hook all-reduce) reduce in a
# different floating-point order, so the gathered expert gradient matches only to
# bf16/fp32-reduce noise — tight enough to catch a doubled/dropped/per-rank sync.
GRAD_COSINE_MIN = 0.999
GRAD_REL_L2_MAX = 5e-3


def create_fixed_batch(tokenizer, device):
    """Deterministic input batch, identical on every rank."""
    torch.manual_seed(SEED)
    random.seed(SEED)
    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "What is 42 + 58?"},
            {"role": "assistant", "content": "The sum of 42 and 58 is 100."},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    tokens = tokenizer(text, return_tensors="pt", padding="max_length", max_length=SEQ_LEN, truncation=True)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    dist.broadcast(input_ids, src=0)
    dist.broadcast(attention_mask, src=0)
    dist.broadcast(labels, src=0)
    return input_ids, attention_mask, labels


def first_expert_weight(model):
    """Return (name, param, num_ep_layers) for the first EP layer's expert weight.

    Uses the EP layer's canonical ``expert_named_params()`` accessor (the same set the
    gradient-sync path operates on), so detection is independent of the per-model expert
    attribute layout (fused ``gate_up_proj`` vs split ``gate_proj``/``up_proj``).
    """
    ep_layers = [m for _, m in model.named_modules() if hasattr(m, "expert_named_params")]
    for name, module in model.named_modules():
        if hasattr(module, "expert_named_params"):
            params = module.expert_named_params()
            if params:
                pname, p = params[0]
                return f"{name}.{pname}", p, len(ep_layers)
    return None, None, len(ep_layers)


def run_mode(fsdp_shard_ep1_experts, tokenizer, local_rank, output_dir):
    """Build the trainer at ep1 with the flag set, run one backward, return a result dict."""
    device = f"cuda:{local_rank}"
    mode = "ON" if fsdp_shard_ep1_experts else "OFF"
    log(f"\n{'=' * 70}\nMODE fsdp_shard_ep1_experts={mode} (ep_size=1)\n{'=' * 70}")

    torch.cuda.reset_peak_memory_stats(device)

    parallelism_config = ParallelismConfig(
        ep_size=1,
        use_grouped_gemm=has_grouped_mm(),
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
        gradient_checkpointing=False,  # keep the backward graph simple for a clean grad readout
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_length=SEQ_LEN,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=True,
        fsdp="",  # the mixin handles FSDP wrapping
    )
    # A tiny dataset is required to construct the trainer; the backward below uses a
    # fixed broadcast batch instead so both modes see identical input.
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
        "Confirm use_grouped_gemm enables EP wrappers for this MoE."
    )
    is_dtensor = isinstance(expert_w, DTensor)
    log(f"  expert weight: {name}  DTensor={is_dtensor}  (FSDP-sharded={is_dtensor})")

    input_ids, attention_mask, labels = create_fixed_batch(tokenizer, device)
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

    return {"is_dtensor": is_dtensor, "loss": loss_val, "grad": grad, "grad_norm": grad_norm, "peak_gb": peak_gb}


def run(ctx) -> dict:
    ensure_model_downloaded(MODEL_NAME, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    off = run_mode(False, tokenizer, ctx.local_rank, ctx.output_dir)
    ctx.barrier()
    on = run_mode(True, tokenizer, ctx.local_rank, ctx.output_dir)
    ctx.barrier()

    # ---- compare on rank 0 ----
    cos = torch.nn.functional.cosine_similarity(
        off["grad"].flatten().unsqueeze(0), on["grad"].flatten().unsqueeze(0)
    ).item()
    rel_l2 = ((on["grad"] - off["grad"]).norm() / (off["grad"].norm() + 1e-12)).item()
    loss_diff = abs(on["loss"] - off["loss"])

    checks = {}
    checks["flag_flips_fsdp_ownership"] = (on["is_dtensor"] is True) and (off["is_dtensor"] is False)
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
