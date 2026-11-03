"""Low-precision ``nn.Linear`` for dense MLPs: a bf16/fp32 master, block-scaled fake-quant operands and a
straight-through gradient. Simulated only (no native kernel for dense linears) and slower than bf16."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from halo.kernels.lowp.quantization import fake_quant
from halo.kernels.lowp.spec import QuantFormat


def _fake_quant_contraction_axis(t: torch.Tensor, quant_format: QuantFormat) -> torch.Tensor:
    """Shard-local for a DTensor, as a TP placement leaves the axis whole or splits it on block boundaries, so
    quantizing the shard is bit-identical and skips DTensor dispatch, which has no rule for fp4's ``bucketize``."""
    if isinstance(t, DTensor):
        local = fake_quant(t.to_local(), quant_format, axis=-1)
        return DTensor.from_local(local, t.device_mesh, t.placements, run_check=False)
    return fake_quant(t, quant_format, axis=-1)


def lowp_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, *, quant_format: QuantFormat
) -> torch.Tensor:
    """Fake-quantized every call, as a dense FSDP2 weight's ``_version`` does not track the sharded update."""
    return F.linear(
        _fake_quant_contraction_axis(x, quant_format), _fake_quant_contraction_axis(weight, quant_format), bias
    )


class LowPrecisionLinear(nn.Linear):
    """:meth:`convert_` is the only way to make one, so a direct construction has no ``quant_format`` and raises."""

    quant_format: QuantFormat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return lowp_linear(x, self.weight, self.bias, quant_format=self.quant_format)

    @classmethod
    def convert_(cls, linear: nn.Linear, quant_format: QuantFormat) -> LowPrecisionLinear:
        """Retype ``linear`` in place, keeping the DTensor params and hooks TP attached to the module object. An
        instance-level ``forward`` (transformers' TP transforms) would shadow the class one, so it is rejected."""
        if (instance_forward := vars(linear).get("forward")) is not None:
            raise TypeError(
                f"{type(linear).__name__} carries an instance-level forward "
                f"({getattr(instance_forward, '__qualname__', instance_forward)!r}) that shadows the class "
                f"forward: retyping it to {cls.__name__} would leave that wrapper calling the captured "
                f"nn.Linear.forward and the module computing in bf16. Strip the wrapper, retype, then "
                f"re-install it around the retyped module."
            )
        linear.__class__ = cls
        linear.quant_format = quant_format
        return linear
