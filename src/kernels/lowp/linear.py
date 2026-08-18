"""Low-precision linear for dense-model MLPs: bf16/fp32 master, block-scale fake-quant compute.

Straight-through gradient — the forward GEMM sees fp8/fp4 numerics, the grad flows to the unchanged
master. Simulated only (no native DeepGEMM for dense linears), and slower than bf16.
"""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from src.kernels.lowp.quantization import fake_quant


class LinearPrecision(str, Enum):
    """Compute precision for :func:`lowp_linear` (master weight stays bf16/fp32).

    No ``BF16`` member: bf16 is the plain :func:`F.linear` path and is gated out upstream, so a
    low-precision linear is never built for it.
    """

    FP8 = "fp8"  # mxfp8 block-scale (e4m3 + e8m0 1×32)
    FP4 = "fp4"  # nvfp4 block-scale (e2m1 + e4m3 1×16) — most accurate fp4
    MXFP4 = "mxfp4"  # mxfp4 block-scale (e2m1 + e8m0 1×32) — fast fp4


PRECISION_TO_FORMAT = {LinearPrecision.FP8: "mxfp8", LinearPrecision.FP4: "nvfp4", LinearPrecision.MXFP4: "mxfp4"}


def _fake_quant_contraction_axis(t: torch.Tensor, fmt: str) -> torch.Tensor:
    """Block-scale fake-quant along the last (contraction) axis, shard-local for a DTensor.

    A TP placement either leaves the contraction axis whole (colwise) or splits it into chunks the block
    boundaries tile exactly (rowwise ``Shard(-1)``), so quantizing the local shard is bit-identical to
    quantizing the full tensor — and it keeps the kernel off DTensor dispatch, which has no sharding rule
    for fp4's ``bucketize``.
    """
    if isinstance(t, DTensor):
        local = fake_quant(t.to_local(), fmt, axis=-1)
        return DTensor.from_local(local, t.device_mesh, t.placements, run_check=False)
    return fake_quant(t, fmt, axis=-1)


def lowp_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    precision: LinearPrecision,
) -> torch.Tensor:
    """Low-precision linear ``x @ weight.T + bias`` (master weight stays bf16/fp32).

    Block-scale fake-quants both operands along the contraction axis (straight-through grad) and runs
    a bf16 matmul.
    """
    fmt = PRECISION_TO_FORMAT[precision]
    # Quantized every call, not step-cached like EP experts: a dense FSDP2 weight's `_version` does not
    # track the sharded update.
    return F.linear(_fake_quant_contraction_axis(x, fmt), _fake_quant_contraction_axis(weight, fmt), bias)


class LowPrecisionLinear(nn.Linear):
    """``nn.Linear`` drop-in computing in low precision (fake-quant) while keeping a bf16/fp32 master.

    :func:`convert_` is the only supported way to make one; ``precision`` is an annotation, not an
    assignment, so a direct construction raises instead of silently picking a precision.
    """

    precision: LinearPrecision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return lowp_linear(x, self.weight, self.bias, precision=self.precision)

    @classmethod
    def convert_(cls, linear: nn.Linear, precision: LinearPrecision) -> LowPrecisionLinear:
        """Retype ``linear`` to this class in place, preserving everything else about the module.

        Never rebuild it: tensor parallelism attaches its semantics to the module object (DTensor params,
        forward transforms, hooks), so a replacement drops the collectives and the model trains on unsynced
        partial sums. The retype works because the CLASS ``forward`` resolves — an instance-level
        ``forward`` (transformers' TP installs its transforms as one, closed over the bound
        ``nn.Linear.forward``) shadows it and keeps computing in bf16, so it is refused here and the caller
        re-installs the transforms around the retyped module
        (:func:`~src.kernels.lowp.mixed_precision.apply_mixed_precision_compute`).
        """
        if (instance_forward := vars(linear).get("forward")) is not None:
            raise TypeError(
                f"{type(linear).__name__} carries an instance-level forward "
                f"({getattr(instance_forward, '__qualname__', instance_forward)!r}) that shadows the class "
                f"forward: retyping it to {cls.__name__} would leave that wrapper calling the captured "
                f"nn.Linear.forward and the module computing in bf16. Strip the wrapper, retype, then "
                f"re-install it around the retyped module."
            )
        linear.__class__ = cls
        linear.precision = precision
        return linear
