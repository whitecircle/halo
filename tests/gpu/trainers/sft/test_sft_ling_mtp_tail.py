#!/usr/bin/env python
"""Ling-3.0's remote head under real DeepEP, trained the way its docs say. At a weight the tail's experts take gradient on
every rank and move, and every save holds them under the release's keys; at weight 0 neither weights nor a checkpoint
change. The trunk moves in both, so a run that trained nothing cannot pass.

    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_ling_mtp_tail.py --mode ep
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import TrainerCallback
from trl import SFTConfig

from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.liger.dispatch import finalize_liger_after_direct_load
from halo.model_loading.parallel.loader import load_distributed_model
from halo.runtime.collectives import barrier
from halo.runtime.consensus import rank_consensus
from halo.runtime.dtensor import to_local
from halo.runtime.filesystem import is_fs_save_rank
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.distributed import shared_output_dir
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import QWEN3_0_6B, TINY_BAILING_LING3_MLA_CONFIG
from tests.common.mtp_heads import (
    MTP_GPU_MODES,
    collate_tokenized_rows,
    moved_keys,
    saved_tensors,
    tokenized_rows,
)
from tests.common.tiny_models import TINY_BAILING_LING3_MLA_FAMILY, build_tiny_family_checkpoint
from tests.common.tokenizers import load_cached_tokenizer

_MODES = ("ep", "ep_etp", "ep1", "ep1_ignored")
_WEIGHT = 0.3
_TAIL_PREFIX = f"model.layers.{TINY_BAILING_LING3_MLA_CONFIG['num_hidden_layers']}."
_STEPS = 3
_LENGTH = 64
_PROMPT = 8
_SEED = 42


class _TailExpertGradients(TrainerCallback):
    """Whether this rank's tail experts held a non-zero gradient at the first step, past the reduce. Read then, as the
    FSDP2 wrap replaces the parameters a ``fsdp_shard_ep1_experts`` run shards."""

    def __init__(self, model):
        self._model = model
        self.nonzero: bool | None = None

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if state.global_step == 0:
            experts = [
                param
                for path, layer in named_ep_layers(self._model)
                if f"{path}.".startswith(_TAIL_PREFIX)
                for _name, param in layer.expert_named_params()
            ]
            self.nonzero = bool(experts) and all(
                param.grad is not None and to_local(param.grad).abs().sum() > 0 for param in experts
            )


def _run_arm(ctx, cli, root: str, base: str, weight: float) -> tuple[bool | None, dict[str, torch.Tensor]]:
    parallelism = ParallelismConfig(**MTP_GPU_MODES[cli.mode][0])
    model, _ = load_distributed_model(
        base,
        parallelism,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        model_config_overrides={"mtp_loss_scaling_factor": weight},
    )
    output = Path(root) / f"weight_{weight}"
    config = SFTConfig(
        output_dir=str(output),
        max_steps=_STEPS,
        save_steps=_STEPS,
        save_strategy="steps",
        per_device_train_batch_size=1,
        learning_rate=1e-3,
        lr_scheduler_type="constant",
        max_grad_norm=0,
        bf16=True,
        logging_steps=1,
        report_to="none",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        seed=_SEED,
        use_liger_kernel=False,
    )
    finalize_liger_after_direct_load(config, False, model)
    gradients = _TailExpertGradients(model)
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=tokenized_rows(
            model.config.vocab_size, 8 * ctx.world_size, seed=_SEED, length=_LENGTH, prompt=_PROMPT
        ),
        processing_class=load_cached_tokenizer(QWEN3_0_6B),
        data_collator=collate_tokenized_rows,
        parallelism_config=parallelism,
        callbacks=[gradients],
    )
    trainer.train()
    trainer.cleanup_ep()
    del trainer, model
    cleanup_memory()
    barrier()
    checkpoint = output / f"checkpoint-{_STEPS}"
    return gradients.nonzero, saved_tensors(checkpoint) if is_fs_save_rank() else {}


@gpu_test_main(min_world_size=1)
def run(ctx) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=_MODES, required=True)
    cli = parser.parse_args()
    if ctx.world_size != MTP_GPU_MODES[cli.mode][1]:
        raise ValueError(
            f"--mode {cli.mode} runs on {MTP_GPU_MODES[cli.mode][1]} ranks, launched on {ctx.world_size}."
        )

    tokenizer = load_cached_tokenizer(QWEN3_0_6B)
    root = shared_output_dir(ctx)
    base = os.path.join(root, "base")
    if is_fs_save_rank():
        build_tiny_family_checkpoint(
            TINY_BAILING_LING3_MLA_FAMILY,
            base,
            tokenizer,
            overrides={"num_nextn_predict_layers": 1, "mtp_loss_scaling_factor": 0},
        )
    barrier()
    stored = saved_tensors(base) if is_fs_save_rank() else {}

    def tail(saved: dict) -> dict:
        return {key: value for key, value in saved.items() if key.startswith(_TAIL_PREFIX)}

    def trunk(saved: dict) -> dict:
        return {key: value for key, value in saved.items() if not key.startswith(_TAIL_PREFIX)}

    trained_gradients, trained = _run_arm(ctx, cli, root, base, _WEIGHT)
    idle_gradients, idle = _run_arm(ctx, cli, root, base, 0.0)
    saves = is_fs_save_rank()
    checks = {
        "weighted_tail_experts_take_gradient_on_every_rank": rank_consensus(trained_gradients is True).all_ok,
        "weighted_tail_moved": rank_consensus(not saves or bool(moved_keys(tail(trained), stored))).all_ok,
        "every_save_holds_the_tail_under_its_keys": rank_consensus(
            not saves or (bool(tail(trained)) and set(tail(trained)) == set(tail(stored)))
        ).all_ok,
        "idle_tail_experts_take_no_gradient": rank_consensus(idle_gradients is False).all_ok,
        "idle_tail_stayed": rank_consensus(not saves or not moved_keys(tail(idle), stored)).all_ok,
        "the_trunk_moved_in_both": rank_consensus(
            not saves or (bool(moved_keys(trunk(trained), stored)) and bool(moved_keys(trunk(idle), stored)))
        ).all_ok,
    }
    return {"checks": checks, "metrics": {}}


if __name__ == "__main__":
    raise SystemExit(run())
