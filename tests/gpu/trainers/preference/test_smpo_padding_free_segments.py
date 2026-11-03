#!/usr/bin/env python
"""SMPO padding-free on the CUDA kernels keeps each flattened document's mixer state its own.

Each document's mean log-prob matches the document run alone; each leak control withholds markers and must drift far
outside the tolerance.
"""

from types import SimpleNamespace

import torch

from halo.configs.preference import SMPOConfig
from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.preference.smpo import SMPOTrainer
from tests.common.harness import gpu_test_main
from tests.common.rank_log import log
from tests.common.segment_isolation import (
    FAMILIES,
    LEAK_CONTROLS,
    PAD_ID,
    lone_document_logps,
    preference_batch,
    tiny_model,
)

# bf16 rounding differs between varlen and dense kernels: isolated rows land at ~1e-2, a row missing its marker drifts
# by ~1.
ISOLATION_TOL = 5e-2
LEAK_FLOOR = 0.3


def _padding_free_logps(trainer: SMPOTrainer, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        out = trainer.concatenated_forward(trainer.model, preference_batch(device))
    return torch.cat([out["chosen_logps"], out["rejected_logps"]]).float()


def run(ctx):
    checks, metrics = {}, {}
    for family in FAMILIES:
        trainer = SMPOTrainer(
            model=tiny_model(family, "flash_attention_2", dtype=torch.bfloat16, device=ctx.device),
            # The rejected-side clips reshape the reported log-probs; the comparison needs them raw.
            args=SMPOConfig(
                output_dir=ctx.output_dir,
                padding_free=True,
                lower_clip_percentile=None,
                min_log_prob=None,
                report_to="none",
                logging_nan_inf_filter=False,
                save_strategy="no",
            ),
            processing_class=SimpleNamespace(pad_token_id=PAD_ID, eos_token_id=1),
            parallelism_config=ParallelismConfig(),
            is_vlm=False,
        )
        reference = lone_document_logps(trainer.model)

        drift = (_padding_free_logps(trainer, ctx.device) - reference).abs().max().item()
        log(f"  {family}: drift {drift:.3e}")
        checks[f"{family}_documents_isolated"] = drift <= ISOLATION_TOL
        metrics[f"{family}_drift"] = drift

        for control, markers in LEAK_CONTROLS[family].items():
            trainer._segment_markers = markers
            leak = (_padding_free_logps(trainer, ctx.device) - reference).abs().max().item()
            log(f"  {family} ({control}): drift {leak:.3e}")
            checks[f"{family}_{control}_row_leaks"] = leak > LEAK_FLOOR
            metrics[f"{family}_{control}_drift"] = leak
    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(exact_world_size=1, prefix="smpo_padding_free_segments")(run)

if __name__ == "__main__":
    main()
