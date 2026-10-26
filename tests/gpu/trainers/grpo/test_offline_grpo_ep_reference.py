#!/usr/bin/env python
"""EP8 full fine-tuning: run-start KL scores survive train, resume, evaluation and HF export, and the fp32
router masters restore exactly. Local tiny Qwen3-MoE weights, a full-logits EP8 oracle, no context parallelism.
Run: torchrun --nproc_per_node=8 tests/gpu/trainers/grpo/test_offline_grpo_ep_reference.py [--ep-loading eager]
"""

from functools import partial

import torch
from torch.distributed.tensor import DTensor

from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.checkpoint_io import RestorePointSnapshot
from tests.common.distributed import pin_deterministic_ep_dispatch
from tests.common.harness import gpu_test_main
from tests.common.offline_grpo import ep_loading_parser
from tests.common.offline_grpo_reference import run_reference_lifecycle


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


def run(ctx):
    ep_loading = ep_loading_parser().parse_args().ep_loading
    pin_deterministic_ep_dispatch()
    parallelism = partial(ParallelismConfig, ep_size=8, ep_fp32_router=True, ep_lazy_loading=ep_loading == "lazy")
    return run_reference_lifecycle(
        ctx, parallelism, "EP8 non-CP", snapshot=_MastersSnapshot, restore_checks=_MastersSnapshot.restored_checks
    )


main = gpu_test_main(exact_world_size=8, prefix="offline_grpo_ep_reference")(run)

if __name__ == "__main__":
    main()
