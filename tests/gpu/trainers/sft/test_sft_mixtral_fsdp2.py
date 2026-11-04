#!/usr/bin/env python
"""A MoE no EP layer class wraps, on the default config with grouped GEMM on: it trains on FSDP2 with its stock experts.

The loader reads whether a family wraps its experts off the family spec, so this run must build no EP layer and no EP
process group. Tiny Mixtral, two ranks through the production trainer. Every trainable parameter is an FSDP2 DTensor,
every weight moves, and the full weights match across ranks after training.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_mixtral_fsdp2.py
"""

import torch
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from halo.distributed.fsdp import reshard_fsdp2_modules
from tests.common.distributed import max_abs_diff_across, snapshot_full_weights
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log
from tests.common.sft_modes import SFTMode, SFTSuite, run_sft_mode
from tests.common.tiny_models import TINY_UNWRAPPED_MOE_FAMILY, shared_tiny_family_checkpoint

WORLD_SIZE = 2
STEPS = 8
SEED = 42
MAX_LENGTH = 128
LEARNING_RATE = 1e-3


def _parameter_weights(model) -> dict[str, torch.Tensor]:
    """COLLECTIVE: full copies of the trainable parameters, which FSDP2 holds sharded once resharded."""
    reshard_fsdp2_modules(model)
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    return {name: tensor for name, tensor in snapshot_full_weights(model).items() if name in trainable}


def _before_train(trainer):
    model = trainer.model
    pc = trainer.parallelism_config
    params = list(model.named_parameters())
    plain = [name for name, param in params if param.requires_grad and not isinstance(param.data, DTensor)]
    log(f"  Trainable parameters outside FSDP2: {plain}")
    checks = {
        "grouped_gemm_left_on": pc.use_grouped_gemm,
        "ep_layers_absent": not ep_layers(model),
        "no_ep_groups_built": pc._ep_config is None and trainer._ep_config is None,
        "stock_moe_blocks_kept": any(isinstance(module, MixtralSparseMoeBlock) for module in model.modules()),
        "fsdp2_wrapped": trainer._fsdp_wrapped,
        "every_trainable_param_sharded": bool(params) and not plain,
    }
    before = _parameter_weights(model)

    def after_train(trainer) -> dict[str, bool]:
        after = _parameter_weights(trainer.model)
        unmoved = sorted(name for name, tensor in after.items() if torch.equal(tensor, before[name]))
        spread = max_abs_diff_across(tensor.to(torch.cuda.current_device()) for tensor in after.values())
        log(f"  Unmoved: {unmoved}; across ranks: {spread}")
        return checks | {
            "every_weight_moved": bool(after) and not unmoved,
            "weights_identical_across_ranks": spread == 0.0,
        }

    return after_train


@gpu_test_main(exact_world_size=WORLD_SIZE, prefix="sft_mixtral_fsdp2")
def run(ctx):
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    source = shared_tiny_family_checkpoint(ctx, TINY_UNWRAPPED_MOE_FAMILY, "sft_mixtral_fsdp2", tokenizer, SEED)
    suite = SFTSuite(
        model_name=source,
        attn_implementation="sdpa",
        sft_args={"max_steps": STEPS, "max_length": MAX_LENGTH, "learning_rate": LEARNING_RATE},
        num_samples=(32, 0),
        evaluate=False,
        loss_decreased=True,
    )
    return {"checks": run_sft_mode(ctx, suite, "fsdp2", SFTMode(before_train=_before_train), tokenizer)}


if __name__ == "__main__":
    run()
