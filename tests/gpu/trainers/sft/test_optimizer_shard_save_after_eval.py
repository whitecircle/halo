#!/usr/bin/env python
"""Per-rank optimizer shard save after an eval-only forward (2 GPUs).

With ``eval_strategy: steps`` transformers' flow callback forces a final evaluation at
``max_steps`` and ``_maybe_log_save_evaluate`` runs it BEFORE the end-of-training save — so the
production save sequence is eval → save. An eval-only forward has no backward to reshard FSDP2
(and ``reshard_after_forward=False``, the toolkit default, skips the post-forward reshard), which
leaves the transient UNSHARDED plain params registered on the modules; ``get_optimizer_state_dict``
maps optimizer params to FQNs by identity against ``named_parameters()`` while the optimizer holds
the sharded DTensors, so an unresharded save leaves every FSDP2 param unmapped and skips the shard
save with ``KeyError: 0`` — a checkpoint with no optimizer state, exit 0. The save path reshards
first; this test pins that on the exact sequence.

``--mode fsdp``  tiny dense Qwen3, plain FSDP2 — the mechanism is family-agnostic.
``--mode ep``    tiny Qwen3.5-MoE composite (ForConditionalGeneration), ep_size=2 — the hardest
                 shape: FSDP2 DTensors + FSDP-ignored plain expert tensors + AdamWBF16's split
                 param groups.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_optimizer_shard_save_after_eval.py --mode fsdp
"""

import argparse
import os

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.utils import cleanup_memory, log

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["fsdp", "ep"], default="fsdp")
ARGS, _ = parser.parse_known_args()

SEED = 42
MAX_STEPS = 2
BATCH_SIZE = 2
MAX_SEQ_LENGTH = 256
# Larger than MAX_STEPS so no mid-run eval fires: the only evaluation is the END-of-training one the
# flow callback forces (max_steps % eval_steps != 0), reproducing the production eval → save order.
EVAL_STEPS = 100


def _build_tiny_checkpoint(mode: str, target_dir: str) -> None:
    """Rank 0: random-init tiny model + Qwen tokenizer saved as a loadable HF checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B, trust_remote_code=True)
    torch.manual_seed(SEED)
    if mode == "ep":
        config = Qwen3_5MoeConfig()
        text_config = config.text_config
        text_config.vocab_size = len(tokenizer)
        text_config.hidden_size = 256
        text_config.intermediate_size = 512
        text_config.moe_intermediate_size = 128
        text_config.shared_expert_intermediate_size = 128
        text_config.num_hidden_layers = 4
        text_config.layer_types = ["linear_attention"] * 3 + ["full_attention"]
        text_config.num_experts = 8
        text_config.num_experts_per_tok = 2
        text_config.num_attention_heads = 4
        text_config.num_key_value_heads = 2
        text_config.head_dim = 64
        text_config.linear_conv_kernel_dim = 4
        text_config.linear_key_head_dim = 32
        text_config.linear_value_head_dim = 32
        text_config.linear_num_key_heads = 2
        text_config.linear_num_value_heads = 4
        text_config.max_position_embeddings = 4096
        text_config.tie_word_embeddings = False
        vision_config = config.vision_config
        vision_config.hidden_size = 64
        vision_config.intermediate_size = 128
        vision_config.depth = 2
        vision_config.num_heads = 2
        vision_config.out_hidden_size = 256
        model = Qwen3_5MoeForConditionalGeneration(config)
    else:
        config = Qwen3Config(
            vocab_size=len(tokenizer),
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            max_position_embeddings=4096,
            tie_word_embeddings=False,
        )
        model = Qwen3ForCausalLM(config)
    model.to(torch.bfloat16).save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)


def _all_ranks_true(local: bool, device) -> bool:
    t = torch.tensor([1 if local else 0], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())


def _shard_covers_optimizer(ckpt_dir: str, rank: int, optimizer) -> tuple[bool, str]:
    """This rank's shard holds FQN-keyed state for EVERY optimizer param — not just some."""
    shard_path = os.path.join(ckpt_dir, f"optimizer_shard_{rank:05d}.pt")
    if not os.path.exists(shard_path):
        return False, f"missing {os.path.basename(shard_path)}"
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    state = shard.get("state", {})
    non_fqn = [k for k in state if not isinstance(k, str)]
    if non_fqn:
        return False, f"non-FQN state keys: {non_fqn[:3]}"
    num_params = len({id(p) for group in optimizer.param_groups for p in group["params"]})
    if len(state) != num_params:
        return False, f"state covers {len(state)} of {num_params} optimizer params"
    return True, "ok"


@gpu_test_main(min_world_size=2, prefix=f"optim_shard_after_eval_{ARGS.mode}")
def run(ctx):
    mode = ARGS.mode
    checks: dict[str, bool] = {}
    device = ctx.device

    dirs = [ctx.output_dir]
    dist.broadcast_object_list(dirs, src=0)
    shared_dir = dirs[0]
    tiny_dir = os.path.join(shared_dir, "tiny_model")
    train_out = os.path.join(shared_dir, "train_out")

    if ctx.rank == 0:
        _build_tiny_checkpoint(mode, tiny_dir)
    dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(tiny_dir, trust_remote_code=True)
    pc = ParallelismConfig(ep_size=2) if mode == "ep" else ParallelismConfig()
    model, _ = load_distributed_model(
        model_name_or_path=tiny_dir,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        use_liger_kernel=False,
    )

    config = SFTConfig(
        output_dir=train_out,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=1e-4,
        bf16=True,
        optim="adamw_torch_fused",  # auto-replaced by AdamWBF16 — the production optimizer
        logging_steps=1,
        save_strategy="steps",
        save_steps=MAX_STEPS,
        save_only_model=False,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        dataloader_num_workers=0,
        seed=SEED,
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=create_sft_dataset(16, tokenizer),
        eval_dataset=create_sft_dataset(8, tokenizer),
        processing_class=tokenizer,
        parallelism_config=pc,
    )
    trainer.train()

    evaluated = any("eval_loss" in e for e in trainer.state.log_history)
    checks["final_eval_ran_before_save"] = _all_ranks_true(evaluated, device)

    ckpt_dir = os.path.join(train_out, f"checkpoint-{MAX_STEPS}")
    files = set(os.listdir(ckpt_dir)) if os.path.isdir(ckpt_dir) else set()
    shards_present = all(f"optimizer_shard_{r:05d}.pt" in files for r in range(ctx.world_size))
    checks["optimizer_shards_present"] = _all_ranks_true(shards_present and "optimizer_meta.pt" in files, device)
    checks["no_stale_single_rank_optimizer"] = _all_ranks_true("optimizer.pt" not in files, device)

    covered, detail = _shard_covers_optimizer(ckpt_dir, ctx.rank, trainer.optimizer)
    if not covered:
        log(f"[rank {ctx.rank}] shard coverage: {detail}")
    checks["shard_covers_all_optimizer_params"] = _all_ranks_true(covered, device)

    del trainer
    cleanup_memory()
    return {"checks": checks}


if __name__ == "__main__":
    run()
