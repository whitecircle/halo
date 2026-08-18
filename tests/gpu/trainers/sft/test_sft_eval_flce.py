#!/usr/bin/env python
"""
Eval-mode fused-linear-cross-entropy test for DistributedSFTTrainer.

Liger's ``lce_forward`` defaults ``skip_logits`` to ``self.training and labels
is not None``, so evaluation falls back to materializing the full
``[B, S, vocab]`` fp32 logits — ~43 GiB at 75k eval tokens x 155k vocab
(GLM-4.7-Flash), an OOM on B300s. ``DistributedSFTTrainer.prediction_step``
injects ``skip_logits=True`` for loss-only evaluation so the fused kernel is
used in eval too.

Test validates (tiny random-weights Glm4MoeLite, no downloads):
1. Loss-only evaluation never invokes ``lm_head`` (the fused kernel consumes
   ``lm_head.weight`` directly — full logits are never materialized).
2. The fused eval loss matches the unfused (full-logits) eval loss.
3. Eval loss is finite.

Usage:
    torchrun --nproc_per_node=1 \
        tests/gpu/trainers/sft/test_sft_eval_flce.py
"""

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from transformers.models.glm4_moe_lite import Glm4MoeLiteConfig
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteForCausalLM
from trl import SFTConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.kernels.liger.orchestrator import resolve_liger_applier
from src.trainers.sft import DistributedSFTTrainer
from tests.common.harness import gpu_test_main
from tests.common.utils import log

VOCAB_SIZE = 4096
SEQ_LENGTH = 256
PROMPT_LENGTH = 64  # leading tokens masked to -100
NUM_EVAL_SAMPLES = 4
BATCH_SIZE = 2
SEED = 42
LOSS_ATOL = 0.02  # fused vs unfused CE: both bf16 matmul + fp32 reduction


def create_tiny_glm4_moe_lite() -> Glm4MoeLiteForCausalLM:
    """Random-weights Glm4MoeLite small enough for a seconds-scale eval test."""
    config = Glm4MoeLiteConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=16,
        max_position_embeddings=SEQ_LENGTH,
    )
    torch.manual_seed(SEED)
    model = Glm4MoeLiteForCausalLM(config)
    return model.to(device="cuda", dtype=torch.bfloat16)


def create_pretokenized_dataset(num_samples: int) -> Dataset:
    """Pre-tokenized samples: random ids, prompt prefix masked to -100."""
    generator = torch.Generator().manual_seed(SEED)
    rows = []
    for _ in range(num_samples):
        input_ids = torch.randint(0, VOCAB_SIZE, (SEQ_LENGTH,), generator=generator).tolist()
        labels = [-100] * PROMPT_LENGTH + input_ids[PROMPT_LENGTH:]
        rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * SEQ_LENGTH,
                "labels": labels,
            }
        )
    return Dataset.from_list(rows)


def collate_pretokenized(features):
    return {
        key: torch.tensor([f[key] for f in features], dtype=torch.long)
        for key in ("input_ids", "attention_mask", "labels")
    }


def create_stub_tokenizer() -> PreTrainedTokenizerFast:
    """Offline stand-in — the dataset is pre-tokenized, TRL just needs an object."""
    vocab = {"<pad>": 0, "<eos>": 1}
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<pad>")),
        pad_token="<pad>",
        eos_token="<eos>",
    )


def run(ctx) -> dict:
    checks: dict[str, bool] = {}

    log(f"\n{'=' * 70}")
    log(f"SFT EVAL FLCE TEST (world_size={ctx.world_size})")
    log(f"{'=' * 70}")

    model = create_tiny_glm4_moe_lite()
    log(f"Model: glm4_moe_lite, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # The REAL registered applier, driven the way production resolves it — same FLCE path
    # GLM-4.7-Flash long-context runs take. Only the loss role is enabled: this suite is about the
    # eval-time skip_logits injection, not the family's norm/GLU swaps.
    resolve_liger_applier("glm4_moe_lite")(
        cross_entropy=False,
        fused_linear_cross_entropy=True,
        rms_norm=False,
        swiglu=False,
        model=model,
    )

    eval_dataset = create_pretokenized_dataset(NUM_EVAL_SAMPLES)

    config = SFTConfig(
        output_dir=ctx.output_dir,
        per_device_eval_batch_size=BATCH_SIZE,
        max_length=SEQ_LENGTH,
        bf16=True,
        use_liger_kernel=True,
        liger_kernel_config={"fused_linear_cross_entropy": True},
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        eval_strategy="no",
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        seed=SEED,
    )

    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=eval_dataset,
        eval_dataset=eval_dataset,
        processing_class=create_stub_tokenizer(),
        data_collator=collate_pretokenized,
        parallelism_config=ParallelismConfig(use_grouped_gemm=False),
    )

    # FLCE-patched forward must expose skip_logits — Liger lce_forward contract changed?
    checks["forward_accepts_skip_logits"] = bool(trainer._forward_accepts_skip_logits)

    lm_head_calls = {"count": 0}

    def count_lm_head(_module, _inputs, _output):
        lm_head_calls["count"] += 1

    hook = trainer.model.lm_head.register_forward_hook(count_lm_head)

    log("\n--- Run A: loss-only evaluation (fused path expected) ---")
    metrics_fused = trainer.evaluate()
    fused_loss = metrics_fused["eval_loss"]
    fused_lm_head_calls = lm_head_calls["count"]
    log(f"Fused eval_loss: {fused_loss:.6f}, lm_head calls: {fused_lm_head_calls}")

    # TRL 1.6's SFTTrainer injects skip_logits natively whenever args.use_liger_kernel is set, so
    # the reference must clear it too; lce_forward then defaults to the unfused full-logits path.
    log("\n--- Run B: unfused reference evaluation ---")
    trainer.__dict__["_forward_accepts_skip_logits"] = False
    trainer.args.use_liger_kernel = False
    lm_head_calls["count"] = 0
    metrics_unfused = trainer.evaluate()
    trainer.args.use_liger_kernel = True
    unfused_loss = metrics_unfused["eval_loss"]
    unfused_lm_head_calls = lm_head_calls["count"]
    log(f"Unfused eval_loss: {unfused_loss:.6f}, lm_head calls: {unfused_lm_head_calls}")

    hook.remove()

    checks["fused_loss_finite"] = bool(torch.isfinite(torch.tensor(fused_loss)))

    # lm_head invoked during loss-only eval == full [B, S, vocab] logits materialized instead of the
    # fused FLCE path: the eval-mode OOM the fix targets.
    checks["fused_eval_skips_lm_head"] = fused_lm_head_calls == 0

    # The reference run must take the unfused full-logits path (injection disabled); lm_head never
    # invoked there would mean the comparison is vacuous.
    checks["unfused_reference_uses_lm_head"] = unfused_lm_head_calls > 0

    loss_diff = abs(fused_loss - unfused_loss)
    checks["fused_matches_unfused"] = loss_diff <= LOSS_ATOL
    log(
        f"\nFused eval loss {fused_loss:.6f} vs unfused reference {unfused_loss:.6f} "
        f"(diff {loss_diff:.6f}, atol={LOSS_ATOL})"
    )

    return {"checks": checks}


main = gpu_test_main(min_world_size=1, prefix="sft_eval_flce_test")(run)

if __name__ == "__main__":
    main()
