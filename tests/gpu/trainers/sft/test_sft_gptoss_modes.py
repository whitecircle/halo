#!/usr/bin/env python
"""SFT smoke of GptOss-20B under each parallel shape, one ``--mode`` per launch. No loss is compared against a
reference, so a wrong finite loss passes.

``ep`` keeps pretrained sinks live on flex, as on-policy RL loads them; ``ep_fa2`` drops them. The correctness gates
live under ``tests/gpu/parallelism/``.
"""

import argparse
import os
from functools import partial

from halo.runtime.collectives import barrier
from tests.common.checkpoint_io import model_save_checks
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.sft_modes import SFTMode, SFTSuite, TrainerChecks, run_sft_suite

SUITE = SFTSuite(
    model_name=GPT_OSS_20B,
    attn_implementation="flex_attention",
    sft_args={"max_steps": 5, "gradient_accumulation_steps": 1, "learning_rate": 2e-5, "max_length": 4096},
    num_samples=(32, 0),
    evaluate=False,
    loss_decreased=True,
)
FA2 = "flash_attention_2"


def _dropped_sinks(model) -> list[bool]:
    return [module.sinks is None for module in model.modules() if hasattr(module, "sinks")]


def sink_shape_checks(trainer, *, dropped: bool) -> TrainerChecks:
    """Dropped under FA2, whose ``s_aux`` path rejects any tensor, a parameter on every other backend. Returns the
    check that training kept it."""
    before = _dropped_sinks(trainer.model)
    return lambda trainer: {
        "sink_layers_found": bool(before),
        "sink_reset_shape_matches_backend": bool(before) and all(layer == dropped for layer in before),
        "sink_shape_stable_across_training": _dropped_sinks(trainer.model) == before,
    }


def saved_model_checks(trainer) -> dict[str, bool]:
    """COLLECTIVE: save, and check on rank 0 that the save holds ``config.json`` and weights."""
    save_dir = os.path.join(trainer.args.output_dir, "saved_model")
    trainer.save_model(save_dir)
    barrier()
    return model_save_checks(save_dir, trainer.args.process_index)


MODES = {
    "fsdp": SFTMode(),
    "fsdp_fa2": SFTMode(attn_implementation=FA2, extra_checks=saved_model_checks),
    "ep": SFTMode({"ep_size": 2}, before_train=partial(sink_shape_checks, dropped=False), reset_sinks=False),
    "ep_fa2": SFTMode({"ep_size": 2}, attn_implementation=FA2, before_train=partial(sink_shape_checks, dropped=True)),
    "cp": SFTMode({"cp_size": 2}),
    "tp": SFTMode({"tp_size": 2, "max_concurrent_loading": 1}, attn_implementation=FA2),
    "ep_cp": SFTMode({"ep_size": 2, "cp_size": 2}),
    "ep_tp": SFTMode({"ep_size": 2, "tp_size": 2}),
    "ep_etp": SFTMode({"ep_size": 1, "expert_tp_size": 2}, attn_implementation=FA2),
}


@gpu_test_main(min_world_size=2, prefix="sft_gptoss_modes")
def run(ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=[*MODES, "all"], default="ep")
    return run_sft_suite(ctx, SUITE, MODES, parser.parse_args().mode)


if __name__ == "__main__":
    run()
