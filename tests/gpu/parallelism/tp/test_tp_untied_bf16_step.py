#!/usr/bin/env python
"""An untied dense model takes AdamWBF16 steps at TP=2.

With ``tie_word_embeddings`` off the input embedding is whole on every TP rank, so each rank rounds the same update
stochastically. The table must move and stay bit-identical across the TP group, or the ranks train different models.
"""

import torch
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer
from trl import SFTConfig

from halo.callbacks.efficiency import EfficiencyCallback
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.optimizers.adamw_bf16 import AdamWBF16
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import group_max_abs_diff
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log
from tests.common.tiny_models import TINY_DENSE_FAMILY, shared_tiny_family_checkpoint
from tests.common.training_checks import training_run_checks

TP_SIZE = 2
NUM_STEPS = 3
SEED = 42
# High enough that every step moves a bf16 embedding row past its rounding step.
LEARNING_RATE = 2e-3


def _local(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


@gpu_test_main(exact_world_size=TP_SIZE, prefix="tp_untied_bf16_step")
def run(ctx):
    torch.cuda.set_device(ctx.device)
    checkpoint = shared_tiny_family_checkpoint(
        ctx, TINY_DENSE_FAMILY, "tiny_untied_dense_tp", AutoTokenizer.from_pretrained(QWEN3_0_6B), SEED
    )
    pc = ParallelismConfig(tp_size=TP_SIZE)
    model, tokenizer = load_distributed_model(
        model_name_or_path=checkpoint,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        use_liger_kernel=False,
        trust_remote_code=False,
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=ctx.output_dir,
            max_steps=NUM_STEPS,
            per_device_train_batch_size=2,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="constant",
            max_length=128,
            bf16=True,
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            logging_nan_inf_filter=False,
            dataloader_drop_last=True,
            seed=SEED,
        ),
        train_dataset=create_sft_dataset(16, tokenizer, seed=SEED),
        processing_class=tokenizer,
        parallelism_config=pc,
        callbacks=[EfficiencyCallback(pc, n_warmup_steps=1)],
    )
    embedding = trainer.model.get_input_embeddings().weight
    head = trainer.model.get_output_embeddings().weight
    embedding_before, head_before = _local(embedding).clone(), _local(head).clone()

    result = trainer.train()

    checks = training_run_checks(result, trainer, NUM_STEPS)
    checks["embeddings_untied"] = embedding is not head
    checks["optimizer_is_adamw_bf16"] = isinstance(trainer.optimizer, AdamWBF16)
    checks["embedding_whole_on_every_rank"] = tuple(_local(embedding).shape) == (
        trainer.model.config.vocab_size,
        trainer.model.config.hidden_size,
    )
    checks["embedding_moved"] = not torch.equal(_local(embedding), embedding_before)
    checks["head_moved"] = not torch.equal(_local(head), head_before)
    cross_rank = group_max_abs_diff(_local(embedding))
    log(f"  embedding max |diff| across the TP group after {NUM_STEPS} steps: {cross_rank:.3e}")
    checks["embedding_identical_across_tp_ranks"] = cross_rank == 0.0
    return {"checks": checks, "metrics": {**ctx.metrics(trainer), "final_train_loss": result.training_loss}}


if __name__ == "__main__":
    run()
