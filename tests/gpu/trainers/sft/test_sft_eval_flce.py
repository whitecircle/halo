#!/usr/bin/env python
"""Loss-only eval uses the fused FLCE kernel: Liger's ``skip_logits`` defaults off in eval, materializing ~43 GiB of
logits at 75k tokens x 155k vocab."""

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from transformers.models.glm4_moe_lite import Glm4MoeLiteConfig
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteForCausalLM
from trl import SFTConfig

from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.harness import gpu_test_main
from tests.common.liger_appliers import liger_applier
from tests.common.rank_log import log

VOCAB_SIZE = 4096
SEQ_LENGTH = 256
PROMPT_LENGTH = 64  # Leading tokens masked to -100.
NUM_EVAL_SAMPLES = 4
BATCH_SIZE = 2
SEED = 42
LOSS_ATOL = 0.02  # Both bf16 matmul + fp32 reduction.


def create_tiny_glm4_moe_lite() -> Glm4MoeLiteForCausalLM:
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

    # The real applier, on the FLCE path GLM-4.7-Flash long-context runs take; only the loss role is enabled.
    liger_applier("glm4_moe_lite")(
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
        logging_nan_inf_filter=False,
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

    # Fails if Liger's lce_forward contract changed.
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

    # TRL 1.14 sets skip_logits itself whenever args.use_liger_kernel is set, so the reference clears it too.
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

    # lm_head invoked means full logits materialized: the eval-mode OOM the fused path avoids.
    checks["fused_eval_skips_lm_head"] = fused_lm_head_calls == 0

    # lm_head never invoked here would mean the comparison is vacuous.
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
