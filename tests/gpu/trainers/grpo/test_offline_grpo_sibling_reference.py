#!/usr/bin/env python
"""TP2×DP2 and ETP2×DP2 run-start KL: ordered scores over two DP shards of collective-model siblings, exact
resume and HF export. ``--mode tp`` trains a local tiny Qwen3, ``--mode etp`` Qwen3-MoE under pure ETP.
Run: torchrun --nproc_per_node=4 tests/gpu/trainers/grpo/test_offline_grpo_sibling_reference.py --mode tp
"""

import argparse
from functools import partial

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.distributed.parallelism_config import ParallelismConfig
from tests.common.harness import gpu_test_main
from tests.common.offline_grpo import OFFLINE_VOCAB, make_offline_tokenizer, save_offline_moe_base
from tests.common.offline_grpo_reference import run_reference_lifecycle


def _save_dense_base(path, seed):
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=len(OFFLINE_VOCAB),
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
    )
    Qwen3ForCausalLM(config).to(torch.bfloat16).save_pretrained(path)
    make_offline_tokenizer().save_pretrained(path)


def run(ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("tp", "etp"), required=True)
    mode = parser.parse_args().mode
    return run_reference_lifecycle(
        ctx,
        partial(ParallelismConfig, tp_size=2) if mode == "tp" else partial(ParallelismConfig, expert_tp_size=2),
        f"{mode.upper()}2×DP2",
        save_base=_save_dense_base if mode == "tp" else save_offline_moe_base,
        layout_checks=lambda trainer: {
            "two_dp_shards_with_two_siblings_each": sorted(trainer._data_parallel_rank_by_global_rank())
            == [0, 0, 1, 1]
        },
    )


main = gpu_test_main(exact_world_size=4, prefix="offline_grpo_sibling_reference")(run)

if __name__ == "__main__":
    main()
