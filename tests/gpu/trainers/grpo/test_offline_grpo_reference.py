#!/usr/bin/env python
"""Offline GRPO full fine-tuning without context parallelism: run-start KL scores survive train, resume,
evaluation and HF export, held to a full-logits oracle (:func:`~tests.common.offline_grpo_reference.run_reference_lifecycle`).

``--mode ep8`` trains local tiny Qwen3-MoE weights at EP8, whose fp32 router masters must restore exactly
(``--ep-loading`` picks the checkpoint loader); ``--mode tp`` (a local tiny Qwen3) and ``--mode etp``
(Qwen3-MoE under pure ETP) score over two DP shards of collective-model siblings on four ranks.

Run: torchrun --nproc_per_node=8 tests/gpu/trainers/grpo/test_offline_grpo_reference.py --mode ep8 --ep-loading eager
"""

from functools import partial

import torch
from torch.distributed.tensor import DTensor
from transformers import Qwen3Config, Qwen3ForCausalLM

from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.distributed.parallelism_config import ParallelismConfig
from tests.common.checkpoint_io import RestorePointSnapshot
from tests.common.distributed import pin_deterministic_ep_dispatch
from tests.common.harness import gpu_test_main
from tests.common.offline_grpo import OFFLINE_VOCAB, ep_loading_parser, make_offline_tokenizer, save_offline_moe_base
from tests.common.offline_grpo_reference import run_reference_lifecycle

parser = ep_loading_parser()
parser.add_argument("--mode", choices=("ep8", "tp", "etp"), required=True)
parser.set_defaults(ep_loading=None)
ARGS = parser.parse_args()
if (ARGS.ep_loading is None) == (ARGS.mode == "ep8"):
    parser.error("--ep-loading picks the EP8 checkpoint loader: name it with --mode ep8, and only there")


class _MastersSnapshot(RestorePointSnapshot):
    """The save point plus every plain fp32 master, which the resumed trainer must hold bit for bit before its
    first forward."""

    def extra(self):
        return {
            "masters": {
                name: parameter.detach().cpu().clone()
                for name, parameter in self.trainer.model.named_parameters()
                if not isinstance(parameter, DTensor) and parameter.dtype == torch.float32
            }
        }

    @staticmethod
    def restored_checks(resumed, captured) -> dict:
        reshard_fsdp2_modules(resumed.model)
        masters = captured.get("masters", {})
        current = dict(resumed.model.named_parameters())
        return {
            "fp32_router_restore_has_coverage": bool(masters),
            "fp32_masters_restored_exactly_before_forward": bool(masters)
            and all(
                current[name].dtype == torch.float32 and torch.equal(before, current[name].detach().cpu())
                for name, before in masters.items()
            ),
        }


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
    if ARGS.mode == "ep8":
        pin_deterministic_ep_dispatch()
        parallelism = partial(
            ParallelismConfig, ep_size=8, ep_fp32_router=True, ep_lazy_loading=ARGS.ep_loading == "lazy"
        )
        return run_reference_lifecycle(
            ctx, parallelism, "EP8 non-CP", snapshot=_MastersSnapshot, restore_checks=_MastersSnapshot.restored_checks
        )
    return run_reference_lifecycle(
        ctx,
        partial(ParallelismConfig, tp_size=2) if ARGS.mode == "tp" else partial(ParallelismConfig, expert_tp_size=2),
        f"{ARGS.mode.upper()}2×DP2",
        save_base=_save_dense_base if ARGS.mode == "tp" else save_offline_moe_base,
        layout_checks=lambda trainer: {
            "two_dp_shards_with_two_siblings_each": sorted(trainer._data_parallel_rank_by_global_rank())
            == [0, 0, 1, 1]
        },
    )


main = gpu_test_main(exact_world_size=8 if ARGS.mode == "ep8" else 4, prefix=f"offline_grpo_{ARGS.mode}_reference")(
    run
)

if __name__ == "__main__":
    main()
