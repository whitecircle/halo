"""Quantize a bf16/fp32 checkpoint to a block-scaled low-precision checkpoint (mxfp8 / mxfp4 / nvfp4).

Pairs with the mixed-precision QAT training path (``lowp_precision: fp8|fp4|mxfp4``, i.e.
mxfp8/nvfp4/mxfp4): a model trained that way keeps bf16/fp32 master weights and its forward already
computes in the block-scaled format. Quantizing those master weights to the same format here
reproduces the training forward exactly (``relerr = 0`` for both mxfp8 and nvfp4), so the converted
model runs in low precision with no accuracy loss beyond what training already saw. A model trained
in plain bf16 can also be converted, but that is post-training quantization rather than QAT.

For each quantized weight ``<name>.weight`` the output stores, following the ``compressed-tensors``
block-scaled convention:

  - ``<name>.weight_packed`` — quantized data: ``float8_e4m3fn`` (mxfp8) or packed ``uint8`` with two
    ``e2m1`` nibbles per byte (nvfp4),
  - ``<name>.weight_scale``  — one scale per block along the contraction axis: ``uint8`` e8m0 (mxfp8,
    block 32) or ``float8_e4m3fn`` (nvfp4, block 16),
  - ``<name>.weight_shape``  — the original ``[out, in]`` shape (the packed nvfp4 data halves the last dim),
  - ``<name>.weight_global_scale`` — nvfp4 only: the per-tensor fp32 scale of its two-level scaling, in
    compressed-tensors' spelling: an element is ``code x weight_scale / weight_global_scale``, so the
    stored value is the reciprocal of :class:`~halo.kernels.lowp.quantization.BlockScaledTensor`'s
    multiplier (dropping it rescales the weight).

Which weights those are is derived from the rosters that decide what QAT quantized (the trainer's
dense ``MLP_PROJECTIONS`` plus the family specs' expert layouts), so the export cannot quantize a
weight the training forward left in bf16. Every other tensor (attention, embeddings, norms, lm_head,
biases, and a VLM's vision tower / projector) is copied unchanged in its original dtype. The four
``lowp_*`` scope knobs the run trained with must be passed back in (``--lowp_apply_dense_mlp`` /
``--lowp_apply_moe_experts`` / ``--lowp_keep_first_blocks`` / ``--lowp_keep_last_blocks``, defaulting
to the training defaults); they are not recoverable from the checkpoint. A
``quantization_config.json`` records the scheme so a loader (an inference engine, or the toolkit's
:mod:`halo.kernels.lowp.quantization` dequantizer) can reconstruct the weights. The block layout is OCP
microscaling (mxfp8) / NVFP4 (1x16 e4m3), the formats TensorRT-LLM and vLLM consume via
``compressed-tensors`` / ModelOpt; mapping this manifest to a specific engine's loader remains an
integration step.

Usage:
  python scripts/after_training/quantize_to_lowp.py \\
      --input_dir <large-volume>/models/my-bf16-ckpt --output_dir <large-volume>/models/my-mxfp8-ckpt \\
      --format mxfp8 --verify
"""

import argparse

from halo.checkpoint.lowp_export import DEFAULT_EXCLUDE, DEFAULT_INCLUDE, quantize_checkpoint
from halo.kernels.lowp.quantization import QUANTIZERS
from halo.kernels.lowp.spec import LowpSpec
from halo.log import configure_cli_logging


def parse_args():
    p = argparse.ArgumentParser(description="Quantize a bf16/fp32 checkpoint to block-scaled mxfp8/mxfp4/nvfp4.")
    p.add_argument("--input_dir", required=True, help="Source checkpoint directory (safetensors).")
    p.add_argument("--output_dir", required=True, help="Output directory for the quantized checkpoint.")
    p.add_argument("--format", choices=list(QUANTIZERS), required=True, help="Block-scaled target format.")
    p.add_argument(
        "--contraction_axis",
        type=int,
        default=-1,
        help="Reduction axis to block along, for 2-D `*.weight` matrices only (default -1 = in_features "
        "for [out,in] Linear weights). Fused 3-D expert tensors use the axis their family spec "
        "declares and ignore this flag.",
    )
    p.add_argument(
        "--include",
        default=DEFAULT_INCLUDE,
        help="Regex narrowing the quantized set (dense MLP projections and declared expert weights).",
    )
    p.add_argument("--exclude", default=DEFAULT_EXCLUDE, help="Regex of weight names to keep high precision.")
    # The training run's lowp scope, which the checkpoint does not carry. Same names and defaults as
    # the ParallelismConfig knobs, so a config's values transfer verbatim.
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
        contraction_axis=a.contraction_axis,
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
