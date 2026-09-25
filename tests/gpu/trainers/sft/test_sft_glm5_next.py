#!/usr/bin/env python
"""GLM-5 Next EP=2 SFT smoke + gathered-save round-trip on a tiny random-init composite model.

Rank 0 materializes a tiny ``Glm5NextForConditionalGeneration`` checkpoint (real tokenizer vocab;
the family ships no text-only CausalLM sibling) on disk via ``save_pretrained`` — which reverts to
the HUB layout: per-expert ``experts.{i}.{gate,up,down}_proj`` tensors plus the vendor-namespace
KDA/hyper-connection keys (``hc_attn_fn``, ``self_attn.f_a_proj``, split ``q/k/v_conv1d``). Then
all ranks:

  1. Load it through ``load_distributed_model`` with EP=2 — the lazy loader route, which replays
     the family's hub conversion per key (``_HUB_CONVERSION_KEYS``: the vendor-namespace renames,
     the three-source ``q/k/v_conv1d → conv1d`` fan-in) and fuses the per-expert projections
     locally; its bit-exactness against ``from_pretrained`` is pinned by
     ``tests/gpu/parallelism/ep/test_lazy_load_converted.py``.
  2. Run a short DistributedSFTTrainer run over text-only conversations (forward + backward +
     optimizer under FSDP2 + EP, dense+sparse MLP span, KDA/DSA attention interleave).
  3. Save via the gathered EP save and reload the checkpoint as a PLAIN HF model — the reloaded
     loss must match the EP model's post-training loss, and the fp32 ``e_score_correction_bias``
     buffer must survive the round-trip.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_glm5_next.py
"""

import os
import shutil

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration
from trl import SFTConfig

from src.distributed.expert_parallel.layers.glm5_next import EPGlm5NextMoELayer
from src.distributed.expert_parallel.lazy_loader import lazy_loader_supports_checkpoint
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.sft import DistributedSFTTrainer
from tests.common.checkpoint_io import fixed_batch_loss
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded, shared_scratch_dir
from tests.common.ep_reference import random_token_batch
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B, TINY_GLM5_CONFIG, TINY_GLM5_VISION_CONFIG
from tests.common.tolerances import TOL
from tests.common.utils import cleanup_memory, log

SEED = 42
NUM_TRAIN_STEPS = 3
MAX_SEQ_LENGTH = 256

_NUM_SPARSE = TINY_GLM5_CONFIG["mlp_layer_types"].count("sparse")


def _materialize_checkpoint(base_dir: str, tokenizer) -> None:
    """Rank 0: build + save the tiny composite model with the real tokenizer's vocab.

    Special-token ids sit inside that vocab (the family defaults index a 154k vocab)."""
    torch.manual_seed(SEED)
    config = Glm5NextConfig(
        text_config={
            **TINY_GLM5_CONFIG,
            "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
        },
        vision_config=dict(TINY_GLM5_VISION_CONFIG),
        image_token_id=2000,
        video_token_id=2001,
        image_start_token_id=2002,
        image_end_token_id=2003,
        video_start_token_id=2004,
        video_end_token_id=2005,
    )
    config._attn_implementation = "eager"
    model = Glm5NextForConditionalGeneration(config).to(torch.bfloat16)
    model.save_pretrained(base_dir)
    tokenizer.save_pretrained(base_dir)


def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    device = ctx.device
    torch.cuda.set_device(device)

    ensure_model_downloaded(QWEN3_0_6B, ctx.rank)  # tokenizer only
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Rank-shared: rank 0 writes the base checkpoint and the gathered save that every rank reads back.
    scratch = shared_scratch_dir("glm5_next_tiny")
    base_dir, save_dir = os.path.join(scratch, "base"), os.path.join(scratch, "trained")
    if ctx.rank == 0:
        shutil.rmtree(scratch, ignore_errors=True)
        ctx.on_teardown(lambda: shutil.rmtree(scratch, ignore_errors=True))
        _materialize_checkpoint(base_dir, tokenizer)
    barrier()

    # The hub layout is lazy-loadable (declared conversion keys), so every rank takes the lazy route.
    checks["lazy_loading_admitted"] = lazy_loader_supports_checkpoint(base_dir) is True

    pc = ParallelismConfig(ep_size=ctx.world_size)
    model, _ = load_distributed_model(
        model_name_or_path=base_dir,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        use_liger_kernel=False,  # no glm5_next Liger applier
    )

    ep_layers = [m for m in model.modules() if isinstance(m, EPGlm5NextMoELayer)]
    checks["ep_layers_patched"] = len(ep_layers) == _NUM_SPARSE
    checks["native_bias_adoptable"] = all(ep.can_adopt_native_balancing() for ep in ep_layers)
    vocab_size = model.config.text_config.vocab_size

    train_dataset = create_sft_dataset(16, tokenizer, seed=SEED)
    sft_config = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=1,
        max_length=MAX_SEQ_LENGTH,
        bf16=True,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",
        report_to=[],
        logging_nan_inf_filter=False,
        dataloader_num_workers=0,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        fsdp="",
        ddp_find_unused_parameters=True,
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        parallelism_config=pc,
    )
    result = trainer.train()
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    log(f"train losses: {losses}")
    metrics["final_train_loss"] = result.training_loss
    checks["trained_all_steps"] = result.global_step == NUM_TRAIN_STEPS
    checks["train_losses_finite"] = all(torch.isfinite(torch.tensor(losses)).tolist()) and len(losses) > 0

    ids, labels = random_token_batch(vocab_size, batch=2, seq=64, device=device, seed=SEED + 7)
    ep_loss = fixed_batch_loss(model, ids, labels)
    metrics["ep_loss_post_train"] = ep_loss
    checks["ep_loss_finite"] = bool(torch.isfinite(torch.tensor(ep_loss)))

    barrier()
    trainer.save_model(save_dir)
    barrier()

    reloaded = (
        AutoModelForImageTextToText.from_pretrained(save_dir, dtype=torch.bfloat16, attn_implementation="eager")
        .to(device=device)
        .eval()
    )
    # The fp32 correction-bias buffer must survive the gathered save (fp32-strict on reload).
    first_sparse = TINY_GLM5_CONFIG["mlp_layer_types"].index("sparse")
    reloaded_bias = reloaded.model.language_model.layers[first_sparse].mlp.gate.e_score_correction_bias
    checks["e_score_bias_roundtrip"] = torch.equal(
        reloaded_bias.float().cpu(), ep_layers[0].gate.e_score_correction_bias.float().cpu()
    )
    rl_loss = fixed_batch_loss(reloaded, ids, labels)
    metrics["reload_loss"] = rl_loss
    delta = abs(rl_loss - ep_loss)
    metrics["reload_loss_delta"] = delta
    log(f"EP loss {ep_loss:.6f} vs reloaded plain-HF loss {rl_loss:.6f} (|Δ|={delta:.2e})")
    # Same weights on both sides: only the EP forward's numerics (grouped GEMM vs the plain expert
    # loop) separate them. At this model scale an expert-layout bug moves the loss by less than the
    # bound, so this pins the round-trip, not the expert layout.
    checks["reload_loss_matches"] = delta < TOL.parallel_vs_baseline_loss_abs

    del reloaded
    cleanup_memory()
    ctx.on_teardown(lambda: trainer.cleanup_ep() if hasattr(trainer, "cleanup_ep") else None)
    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(min_world_size=2, exact_world_size=2, prefix="sft_glm5_next")(run)

if __name__ == "__main__":
    main()
