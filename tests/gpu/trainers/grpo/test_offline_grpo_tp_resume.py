#!/usr/bin/env python
"""OfflineGRPOTrainer saves a mid-training checkpoint and resumes from it, under TP=2 or EP=2
(``HALO_TEST_OFFGRPO_PARALLEL``)."""

from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from halo.checkpoint.parallel.resume_source import resolve_resume_weights_source
from halo.configs.grpo_offline import OfflineGRPOConfig
from halo.distributed.parallelism_config import ParallelismConfig
from halo.env import env_str
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.grpo.offline.trainer import OfflineGRPOTrainer
from tests.common.checkpoint_io import RESUME_PROBE_TEXT, fixed_text_batch, run_two_phase_resume
from tests.common.datasets import create_offline_grpo_dataset
from tests.common.distributed import shared_output_dir
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B, QWEN3_0_6B

IS_EP = env_str("HALO_TEST_OFFGRPO_PARALLEL", "tp") == "ep"
MODEL_NAME = GPT_OSS_20B if IS_EP else QWEN3_0_6B
SEED = 42


def run(ctx) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = create_offline_grpo_dataset(tokenizer, 64, seed=SEED)
    eval_dataset = create_offline_grpo_dataset(tokenizer, 8, seed=SEED + 100)

    def build_trainer(checkpoint: str | None, **schedule) -> OfflineGRPOTrainer:
        parallelism_config = ParallelismConfig(ep_size=2) if IS_EP else ParallelismConfig(tp_size=2)
        # EP skips the loader's weight reload, so its trained weights come only from a model built on the checkpoint.
        weights = resolve_resume_weights_source(
            checkpoint, SimpleNamespace(model_name_or_path=MODEL_NAME), parallelism_config
        )
        model, _ = load_distributed_model(
            model_name_or_path=weights,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            use_liger_kernel=True,
        )
        return OfflineGRPOTrainer(
            model=model,
            args=OfflineGRPOConfig(
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
                max_prompt_length=1024,
                max_completion_length=1024,
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


main = gpu_test_main(min_world_size=2, prefix="offline_grpo_tp_resume")(run)

if __name__ == "__main__":
    main()
