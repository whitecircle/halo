#!/usr/bin/env python
"""DistributedSFTTrainer saves a mid-training checkpoint and resumes from it, with its optimizer bit-exact, in each
2-GPU parallelism mode."""

import argparse
import os
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig

from halo.checkpoint.parallel.resume_source import resolve_resume_weights_source
from halo.distributed.parallelism_config import ParallelismConfig
from halo.env import env_flag, env_str
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.checkpoint_io import RESUME_PROBE_TEXT, fixed_text_batch, run_two_phase_resume
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import shared_output_dir
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B, QWEN3_0_6B
from tests.common.rank_log import log

# Overridable, so one harness validates resume across model families.
DEFAULT_MODEL = env_str("HALO_TEST_RESUME_MODEL", QWEN3_0_6B)
EP_MODEL = env_str("HALO_TEST_RESUME_EP_MODEL", GPT_OSS_20B)  # EP requires an MoE model.
SEED = 42
# ZeRO-3 on the DP axis. Only the fsdp modes and cp honor it: ParallelismConfig rejects TP+DP+FULL_SHARD (no DTensor
# all-gather strategy for the backward re-gather), and EP too.
RESHARD = env_flag("HALO_TEST_RESUME_FSDP_RESHARD")


@dataclass(frozen=True)
class ResumeMode:
    """``base_weights`` builds the resumed trainer on the base model, so the resume loads weights into the FSDP2
    shards; otherwise the production rule builds it from the checkpoint."""

    parallelism: dict
    max_length: int
    model: str = DEFAULT_MODEL
    base_weights: bool = False


MODES = {
    "fsdp": ResumeMode({"fsdp_reshard_after_forward": RESHARD}, 512),
    "fsdp_base": ResumeMode({"fsdp_reshard_after_forward": RESHARD}, 512, base_weights=True),
    "cp": ResumeMode({"cp_size": 2, "fsdp_reshard_after_forward": RESHARD}, 4096),
    "tp": ResumeMode({"tp_size": 2}, 512),
    "ep": ResumeMode({"ep_size": 2}, 2048, model=EP_MODEL),
}


def load_model(parallelism_config: ParallelismConfig, model_path: str):
    if not parallelism_config.has_custom_parallelism:
        return AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa"
        )
    model, _ = load_distributed_model(
        model_name_or_path=model_path,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        use_liger_kernel=True,
    )
    return model


def run_mode(ctx, mode: str) -> dict[str, bool]:
    log(f"\n{'=' * 60}\n  Mode: {mode.upper()}\n{'=' * 60}")
    spec = MODES[mode]
    tokenizer = AutoTokenizer.from_pretrained(spec.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = create_sft_dataset(32, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(8, tokenizer, seed=SEED + 1)

    def build_trainer(checkpoint: str | None, **schedule) -> DistributedSFTTrainer:
        parallelism_config = ParallelismConfig(**spec.parallelism)
        # A base returned where the loader skips the weight reload would resume untrained weights, which the probe
        # catches.
        weights = (
            spec.model
            if spec.base_weights
            else resolve_resume_weights_source(
                checkpoint, SimpleNamespace(model_name_or_path=spec.model), parallelism_config
            )
        )
        return DistributedSFTTrainer(
            model=load_model(parallelism_config, weights),
            args=SFTConfig(
                **schedule,
                per_device_train_batch_size=1,
                learning_rate=2e-5,
                bf16=True,
                gradient_checkpointing=True,
                logging_steps=1,
                report_to="none",
                logging_nan_inf_filter=False,
                max_length=spec.max_length,
                dataloader_drop_last=True,
                dataloader_num_workers=0,
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )

    checks = run_two_phase_resume(
        ctx,
        build_trainer,
        output_dir=os.path.join(shared_output_dir(ctx), mode),
        # Padded so CP can split it, through the same split before the save and after the resume, so a corrupting
        # reload shifts the value.
        probe=fixed_text_batch(tokenizer, ctx.device, RESUME_PROBE_TEXT, pad=True),
        optimizer_state=True,
    )
    return {f"{mode}_{name}": ok for name, ok in checks.items()}


def run(ctx) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=[*MODES, "all"])
    mode = parser.parse_args().mode
    modes = list(MODES) if mode == "all" else [mode]
    log(f"  SFT checkpoint resume: modes {modes}, default model {DEFAULT_MODEL}, EP model {EP_MODEL}")

    checks: dict[str, bool] = {}
    for mode in modes:
        checks |= run_mode(ctx, mode)
    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="sft_resume")(run)

if __name__ == "__main__":
    main()
