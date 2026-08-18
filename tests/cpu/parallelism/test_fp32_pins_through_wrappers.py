#!/usr/bin/env python
"""A family's fp32 pins must survive every wrapper a checkpoint writer holds the model through.

``_keep_in_fp32_modules(_strict)`` is a class attribute of the family's ``PreTrainedModel``; a
writer that reads it off the type of what it holds — a pipeline stage, the CP wrapper, a PEFT
model — sees no pins and casts DeepSeek-V4's hyper-connection tensors and norms, GLM-5's KDA
``A_log``/``dt_bias``/conv, Inkling's short convolutions to bf16: an export the reload's re-pin
cannot repair. The derivation reads every class in the module tree instead.

Run: ``pytest -m cpu tests/cpu/parallelism/test_fp32_pins_through_wrappers.py``
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import DeepseekV4Config, DeepseekV4ForCausalLM

from src.checkpoint.format import save_dtype_caster
from src.distributed.pipeline_parallel.stage import build_pipeline_stage
from src.models.structure import fp32_pinned_param_names
from tests.common.models import TINY_DSV4_CONFIG

PP_SIZE = 2


def _model() -> DeepseekV4ForCausalLM:
    torch.manual_seed(0)
    return DeepseekV4ForCausalLM(DeepseekV4Config(**TINY_DSV4_CONFIG))


def test_pins_are_read_through_a_plain_wrapper():
    model = _model()
    pinned = fp32_pinned_param_names(model)
    assert pinned and any("hc" in name for name in pinned), sorted(pinned)[:5]

    class _Holder(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

    assert fp32_pinned_param_names(_Holder(model)) == {f"inner.{name}" for name in pinned}


def test_pipeline_stages_pin_exactly_the_unsplit_models_tensors():
    """Union of the stages' pins (in global names) == the unsplit model's; the stage caster keeps
    each pinned tensor at its trained dtype while an unpinned one goes to the save dtype."""
    pinned = fp32_pinned_param_names(_model())
    seen: set[str] = set()
    for pp_rank in range(PP_SIZE):
        stage = build_pipeline_stage(_model(), pp_rank, PP_SIZE, moe_balancing="none")
        local = fp32_pinned_param_names(stage)
        assert local, f"stage {pp_rank} derives no pins"
        seen |= {stage.global_parameter_name(name) for name in local}

        cast = save_dtype_caster(stage)
        params = dict(stage.named_parameters())
        kept = next(iter(local))
        assert cast(kept, params[kept].float()).dtype == torch.float32
        unpinned = next(name for name in params if name not in local and "norm" not in name)
        assert cast(unpinned, params[unpinned].float()).dtype != torch.float32
    assert seen == pinned


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
