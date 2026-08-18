"""CPU tests for text-backbone scoping in ``src/kernels/lowp/mixed_precision.py``.

``keep_first/last_n_blocks`` is an NVFP4 recipe about the TEXT decoder's precision-sensitive end
blocks. Block indices must therefore come from the model's own decoder (``get_decoder()``), never
from a vision tower's ``layers.N`` — a deeper vision stack would otherwise inflate ``n_blocks`` and
shift the kept set off the real text end-blocks.

    python tests/cpu/kernels/test_mixed_precision_scoping.py
"""

import sys

import pytest
import torch
import torch.nn as nn

from src.kernels.lowp.linear import LowPrecisionLinear
from src.kernels.lowp.mixed_precision import apply_mixed_precision_compute


class _MLPBlock(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.gate_proj = nn.Linear(width, width)
        self.up_proj = nn.Linear(width, width)
        self.down_proj = nn.Linear(width, width)


class _Stack(nn.Module):
    def __init__(self, n_layers, width=8):
        super().__init__()
        self.layers = nn.ModuleList(_MLPBlock(width) for _ in range(n_layers))


class _VLM(nn.Module):
    """Text decoder with 4 blocks; vision tower with MORE (8) blocks to expose index skew."""

    def __init__(self):
        super().__init__()
        self.model = _Stack(4)
        self.vision_tower = _Stack(8)

    def get_decoder(self):
        return self.model


def test_keep_last_blocks_counts_text_backbone_only():
    model = _VLM()
    summary = apply_mixed_precision_compute(model, precision="fp8", keep_last_n_blocks=1)
    # Kept set derives from the 4 TEXT blocks (last = 3), not the 8 vision blocks (last = 7).
    assert summary["kept_blocks"] == [3]
    assert not isinstance(model.model.layers[3].gate_proj, LowPrecisionLinear)  # kept in bf16
    assert isinstance(model.model.layers[0].gate_proj, LowPrecisionLinear)
    # Vision modules sit outside the backbone: never converted, as the export's tower fence assumes.
    assert not any(isinstance(m, LowPrecisionLinear) for m in model.vision_tower.modules())
    assert summary["dense_linears"] == 3 * 3


def test_conversion_keeps_module_hooks_and_attributes():
    """The conversion retypes in place; anything attached to the module object must survive.

    Parallelism attaches its state to the module (torch's ``parallelize_module`` registers forward
    hooks, transformers stamps ``_is_hooked``); a conversion that constructs a replacement drops it,
    and the run trains on unsynced partial sums with a finite, plausible loss.
    """
    model = _Stack(2, width=32)  # fp8 fake-quant blocks the contraction axis by 32
    fired = []
    for i, block in enumerate(model.layers):
        block.down_proj.register_forward_hook(lambda m, inp, out, i=i: fired.append(i) or out)
        block.down_proj._is_hooked = True

    apply_mixed_precision_compute(model, precision="fp8")

    for i, block in enumerate(model.layers):
        assert isinstance(block.down_proj, LowPrecisionLinear), f"layer {i} down_proj not converted"
        assert len(block.down_proj._forward_hooks) == 1, f"layer {i} lost its forward hook"
        assert block.down_proj._is_hooked is True, f"layer {i} lost _is_hooked"

    for block in model.layers:
        block.down_proj(torch.randn(2, 32))
    assert fired == [0, 1], f"converted modules did not run their hooks: {fired}"


def test_plain_text_model_unchanged():
    model = _Stack(4)  # no get_decoder resolving elsewhere → whole model is the backbone
    summary = apply_mixed_precision_compute(model, precision="fp8", keep_first_n_blocks=1, keep_last_n_blocks=1)
    assert summary["kept_blocks"] == [0, 3]
    assert not isinstance(model.layers[0].gate_proj, LowPrecisionLinear)
    assert not isinstance(model.layers[3].gate_proj, LowPrecisionLinear)
    assert isinstance(model.layers[1].gate_proj, LowPrecisionLinear)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
