"""The low-precision vocabulary, and the projections and layer window both the training-time conversion and the export
read, so a checkpoint quantizes what its forward quantized."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal, get_args

LowpPrecision = Literal["bf16", "fp8", "fp4", "mxfp4"]
# Simulated is the fake-quant QAT oracle; native runs the MoE experts on real low-precision tensor-core kernels.
LowpBackend = Literal["simulated", "native"]
LowpBackwardPrecision = Literal["bf16", "fp8"]
# Block-scaled: mxfp8 = e4m3 + e8m0/32, mxfp4 = e2m1 + e8m0/32, nvfp4 = e2m1 + e4m3/16 (two-level).
QuantFormat = Literal["mxfp8", "mxfp4", "nvfp4"]

# Attention q/k/v/o stay bf16 in training, as small GEMMs are FA4's domain.
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True)
class LowpRecipe:
    """A low-precision mode's GEMM operand format, the ``GPU_PEAK_FLOPS`` key its matmuls issue at, and whether a
    native expert kernel takes it."""

    quant_format: QuantFormat
    peak_flops_key: str
    native: bool


# nvfp4 is the most accurate fp4, mxfp4 the fast one; both issue the same 4-bit MMA. No native kernel takes nvfp4's
# e4m3 block scales.
LOWP_RECIPES: dict[LowpPrecision, LowpRecipe] = {
    "fp8": LowpRecipe("mxfp8", "fp8", native=True),
    "fp4": LowpRecipe("nvfp4", "fp4", native=False),
    "mxfp4": LowpRecipe("mxfp4", "fp4", native=True),
}


@dataclass(frozen=True)
class LowpSpec:
    """Masters and checkpoints stay bf16/fp32. ``keep_{first,last}_blocks`` pin decoder layers at the ends to bf16."""

    precision: LowpPrecision = "bf16"
    backend: LowpBackend = "simulated"
    backward_precision: LowpBackwardPrecision = "bf16"
    apply_dense_mlp: bool = True
    apply_moe_experts: bool = True
    keep_first_blocks: int = 0
    keep_last_blocks: int = 0

    @property
    def enabled(self) -> bool:
        return self.precision != "bf16"

    @property
    def native(self) -> bool:
        return self.enabled and self.backend == "native"

    def validate(self) -> None:
        for name, value, choices in (
            ("lowp_precision", self.precision, LowpPrecision),
            ("lowp_backend", self.backend, LowpBackend),
            ("lowp_backward_precision", self.backward_precision, LowpBackwardPrecision),
        ):
            if value not in get_args(choices):
                raise ValueError(f"{name} must be one of {', '.join(map(repr, get_args(choices)))}, got {value!r}")
        if self.native:
            native = [p for p, r in LOWP_RECIPES.items() if r.native]
            if not LOWP_RECIPES[self.precision].native:
                raise ValueError(
                    f"lowp_backend='native' has no kernel for lowp_precision={self.precision!r}; it takes "
                    f"{', '.join(map(repr, native))}. Use lowp_backend='simulated' for {self.precision!r}."
                )
            if self.apply_dense_mlp or not self.apply_moe_experts:
                raise ValueError(
                    "lowp_backend='native' runs the MoE expert GEMMs only, and dense MLP projections have no native "
                    "kernel: set lowp_apply_dense_mlp: false and keep lowp_apply_moe_experts: true."
                )
        if self.backward_precision == "fp8" and not self.native:
            raise ValueError(
                "lowp_backward_precision='fp8' needs lowp_backend='native' and a low lowp_precision: the simulated "
                "path's backward is bf16 by construction (straight-through gradient)."
            )
        if self.keep_first_blocks < 0 or self.keep_last_blocks < 0:
            raise ValueError(
                f"lowp_keep_first_blocks / lowp_keep_last_blocks must be >= 0, got "
                f"{self.keep_first_blocks} / {self.keep_last_blocks}"
            )
        if self.enabled and not (self.apply_dense_mlp or self.apply_moe_experts):
            raise ValueError(
                f"lowp_precision={self.precision!r} is set but both lowp_apply_dense_mlp "
                f"and lowp_apply_moe_experts are False — low precision would apply to nothing. "
                f"Enable at least one, or set lowp_precision='bf16'."
            )

    def non_default_shape_fields(self) -> list[str]:
        default = LowpSpec()
        return [
            f"lowp_{f.name}"
            for f in fields(self)
            if f.name != "precision" and getattr(self, f.name) != getattr(default, f.name)
        ]


def kept_block_indices(n_blocks: int, keep_first: int, keep_last: int) -> set[int]:
    """Out-of-range ends clamp, so keeping more blocks than exist keeps all of them."""
    kept = set(range(keep_first)) | {n_blocks - 1 - i for i in range(keep_last)}
    return {block for block in kept if 0 <= block < n_blocks}
