#!/usr/bin/env python
"""SMPO padding-free on the CUDA kernels keeps each flattened document's mixer state its own.

The production path of the CPU test (``tests/cpu/trainers/test_smpo_padding_free_segments.py``): a
real ``SmoothMarginPOTrainer`` built with ``padding_free`` on bf16 flash attention, whose forward
hands ``seq_idx`` to the ``causal_conv1d`` kernel (LFM2 ShortConv, the GatedDeltaNet conv) and
``cu_seq_lens_*`` to the ``fla`` chunked delta rule and flash attention; dense Qwen3 gets no marker
and relies on flash attention deriving the boundaries from ``position_ids``. Every document's mean
log-prob from the flattened row must match the same document run alone, and each leak control —
the row rerun with markers withheld — must drift far outside that tolerance.

Single GPU::

    torchrun --nproc_per_node=1 tests/gpu/trainers/preference/test_smpo_padding_free_segments.py
"""

from types import SimpleNamespace

import torch

from src.configs.smpo_config import SmoothMarginPOConfig
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.preference.smpo import SmoothMarginPOTrainer
from tests.common.harness import gpu_test_main
from tests.common.segment_isolation import (
    FAMILIES,
    LEAK_CONTROLS,
    PAD_ID,
    lone_document_logps,
    preference_batch,
    tiny_model,
)
from tests.common.utils import log

# bf16 rounding differs between the varlen and dense kernels: the isolated rows land at ~1e-2 on this
# batch, while a row missing a marker its family reads drifts by ~1.
ISOLATION_TOL = 5e-2
LEAK_FLOOR = 0.3


def _padding_free_logps(trainer: SmoothMarginPOTrainer, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        out = trainer.concatenated_forward(trainer.model, preference_batch(device))
    return torch.cat([out["chosen_logps"], out["rejected_logps"]]).float()


def run(ctx):
    checks, metrics = {}, {}
    for family in FAMILIES:
        trainer = SmoothMarginPOTrainer(
            model=tiny_model(family, "flash_attention_2", dtype=torch.bfloat16, device=ctx.device),
            # The rejected-side clips reshape the reported log-probs; the comparison needs them raw.
            args=SmoothMarginPOConfig(
                output_dir=ctx.output_dir,
                padding_free=True,
                lower_clip_percentile=None,
                min_log_prob=None,
                report_to="none",
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
