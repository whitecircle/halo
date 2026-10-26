#!/usr/bin/env python
"""SmoothMarginPOTrainer at TP=2 saves a mid-training checkpoint and resumes from it on Qwen3-0.6B.

The DTensor parameters are gathered for the save and re-sharded on the resume, which streams the
checkpoint's weights into a base-loaded TP model; the checks are
:func:`~tests.common.checkpoint_io.run_two_phase_resume`'s.
"""

import torch
from transformers import AutoTokenizer

from halo.configs.preference import SmoothMarginPOConfig
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.preference.smpo import SmoothMarginPOTrainer
from tests.common.checkpoint_io import RESUME_PROBE_TEXT, fixed_text_batch, run_two_phase_resume
from tests.common.datasets import create_preference_dataset
from tests.common.distributed import shared_output_dir
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B

MODEL_NAME = QWEN3_0_6B
TP_SIZE = 2
SEED = 42


def run(ctx) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = create_preference_dataset(64, tokenizer, seed=SEED)
    eval_dataset = create_preference_dataset(8, tokenizer, seed=SEED + 100)

    def build_trainer(checkpoint: str | None, **schedule) -> SmoothMarginPOTrainer:
        # Both phases build on the base weights: the resume's own TP reload streams the checkpoint's in.
        parallelism_config = ParallelismConfig(tp_size=TP_SIZE)
        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            use_liger_kernel=True,
            attn_implementation="sdpa",  # padded preference: avoid FA4 varlen-bwd cold-compile (see test_smpo_tp)
        )
        return SmoothMarginPOTrainer(
            model=model,
            args=SmoothMarginPOConfig(
                **schedule,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=5e-6,
                bf16=True,
                gradient_checkpointing=True,
                use_liger_kernel=False,
                logging_steps=1,
                report_to="none",
                logging_nan_inf_filter=False,
                max_length=2048,
                max_prompt_length=1024,
                dataloader_drop_last=True,
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )

    checks = run_two_phase_resume(
        ctx,
        build_trainer,
        output_dir=shared_output_dir(ctx),
        probe=fixed_text_batch(tokenizer, ctx.device, RESUME_PROBE_TEXT),
    )
    return {"checks": checks}


main = gpu_test_main(min_world_size=TP_SIZE, prefix="smpo_tp_resume")(run)

if __name__ == "__main__":
    main()
