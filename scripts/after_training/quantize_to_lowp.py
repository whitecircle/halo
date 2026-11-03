"""Quantize a bf16/fp32 checkpoint to a block-scaled low-precision checkpoint (mxfp8 / mxfp4 / nvfp4).

Pairs with QAT (``lowp_precision: fp8|fp4|mxfp4``): quantizing the master weights reproduces the training forward
exactly (``relerr = 0``). The manifest is Halo's own ``block_scaled``, which serving engines do not load as-is. Each
quantized ``<name>.weight`` is stored as:

  - ``<name>.weight_packed``: ``float8_e4m3fn`` (mxfp8) or packed ``uint8`` with two ``e2m1`` nibbles per byte (nvfp4),
  - ``<name>.weight_scale``: one scale per block along the contraction axis: ``uint8`` e8m0 (mxfp8, block 32) or
    ``float8_e4m3fn`` (nvfp4, block 16),
  - ``<name>.weight_shape``: the original ``[out, in]`` shape (the packed nvfp4 data halves the last dim),
  - ``<name>.weight_global_scale``: nvfp4 only; an element is ``code x weight_scale / weight_global_scale``, so the
    stored value is the reciprocal of :class:`~halo.kernels.lowp.quantization.BlockScaledTensor`'s multiplier.

The quantized weights derive from the rosters that decide what QAT quantized, so the export cannot quantize a weight
the forward left in bf16; all else is copied unchanged. The ``lowp_*`` scope knobs the run trained with must be passed
back in, as the checkpoint does not carry them."""

import argparse

from halo.checkpoint.tools.lowp_export import DEFAULT_EXCLUDE, DEFAULT_INCLUDE, quantize_checkpoint
from halo.kernels.lowp.quantization import QUANTIZERS
from halo.kernels.lowp.spec import LowpSpec
from halo.log import configure_cli_logging


def parse_args():
    p = argparse.ArgumentParser(description="Quantize a bf16/fp32 checkpoint to block-scaled mxfp8/mxfp4/nvfp4.")
    p.add_argument("--input_dir", required=True, help="Source checkpoint directory (safetensors).")
    p.add_argument("--output_dir", required=True, help="Output directory for the quantized checkpoint.")
    p.add_argument("--format", choices=list(QUANTIZERS), required=True, help="Block-scaled target format.")
    p.add_argument(
        "--include",
        default=DEFAULT_INCLUDE,
        help="Regex narrowing the quantized set (dense MLP projections and declared expert weights).",
    )
    p.add_argument("--exclude", default=DEFAULT_EXCLUDE, help="Regex of weight names to keep high precision.")
    # The training run's lowp scope, which the checkpoint does not carry; same names and defaults as ParallelismConfig.
    p.add_argument(
        "--lowp_apply_dense_mlp",
        action=argparse.BooleanOptionalAction,
        default=LowpSpec.apply_dense_mlp,
        help="Whether training quantized dense MLP projections (lowp_apply_dense_mlp; default: %(default)s).",
    )
    p.add_argument(
        "--lowp_apply_moe_experts",
        action=argparse.BooleanOptionalAction,
        default=LowpSpec.apply_moe_experts,
        help="Whether training quantized MoE expert FFNs (lowp_apply_moe_experts; default: %(default)s).",
    )
    p.add_argument(
        "--lowp_keep_first_blocks",
        type=int,
        default=LowpSpec.keep_first_blocks,
        help="Leading transformer blocks training kept in bf16 (lowp_keep_first_blocks; default: %(default)s).",
    )
    p.add_argument(
        "--lowp_keep_last_blocks",
        type=int,
        default=LowpSpec.keep_last_blocks,
        help="Trailing transformer blocks training kept in bf16 (lowp_keep_last_blocks; default: %(default)s).",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Report the max round-trip dequant relerr, and fail when it exceeds the format's tolerance.",
    )
    return p.parse_args()


def main() -> None:
    configure_cli_logging()
    a = parse_args()
    quantize_checkpoint(
        a.input_dir,
        a.output_dir,
        a.format,
        include=a.include,
        exclude=a.exclude,
        apply_dense_mlp=a.lowp_apply_dense_mlp,
        apply_moe_experts=a.lowp_apply_moe_experts,
        keep_first_blocks=a.lowp_keep_first_blocks,
        keep_last_blocks=a.lowp_keep_last_blocks,
        verify=a.verify,
    )


if __name__ == "__main__":
    main()
