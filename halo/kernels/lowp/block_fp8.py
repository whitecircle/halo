"""DeepSeek-style block FP8 for served weights: float8_e4m3fn values with one fp32 power-of-two scale per 128x128
block, stored beside the weight as ``<weight>_scale_inv``. One quantizer for the rollout weight push and the
served-checkpoint tool, so what is pushed is what a fresh boot reads."""

import torch
import torch.nn.functional as F

from halo.kernels.lowp.quantization import E4M3_MAX

BLOCK = 128
SCALE_SUFFIX = "_scale_inv"
# A block of zeros still gets a finite scale.
_AMAX_FLOOR = 1e-12


def _pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """The smallest power of two at or above each positive fp32 ``x``. Read off the exponent bits, so the result is
    the same on every device, where ``exp2(ceil(log2(x)))`` leans on the device's log2."""
    mantissa, exponent = torch.frexp(x)
    return torch.ldexp(torch.ones_like(x), exponent - (mantissa == 0.5).to(exponent.dtype))


def quantize_block_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q, scale_inv)`` for ``weight`` ``[..., N, K]``: ``q`` has its shape in float8_e4m3fn, ``scale_inv`` is fp32
    ``[..., ceil(N/128), ceil(K/128)]`` and ``q * scale_inv`` (broadcast over blocks) reconstructs the weight. A
    dimension off the block size pads with zeros, which change no block's maximum."""
    n, k = weight.shape[-2:]
    padded = F.pad(weight.float(), (0, -k % BLOCK, 0, -n % BLOCK))
    blocks = padded.reshape(*padded.shape[:-2], padded.shape[-2] // BLOCK, BLOCK, padded.shape[-1] // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=(-3, -1), keepdim=True).clamp_min(_AMAX_FLOOR)
    scale = _pow2_ceil(amax / E4M3_MAX)
    quantized = (blocks / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    quantized = quantized.reshape(padded.shape)[..., :n, :k].contiguous()
    return quantized, scale[..., 0, :, 0].contiguous()


def block_fp8_quantization_config(modules_to_not_convert: list[str]) -> dict:
    """The ``config.json`` ``quantization_config`` both engines read to serve the checkpoint. vLLM matches
    ``modules_to_not_convert`` by exact module path."""
    return {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [BLOCK, BLOCK],
        "modules_to_not_convert": sorted(modules_to_not_convert),
    }
