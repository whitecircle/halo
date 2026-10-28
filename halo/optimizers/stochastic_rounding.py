"""Stochastic rounding to bf16: the rounding-noise seeds and the rounding step every SR optimizer shares.

The noise is a pure function of a parameter's optimizer step and name, so replicas round alike and a resumed run rounds
as the uninterrupted one did, with no generator state to checkpoint. An element's noise is keyed by its local-shard
offset: reproducible for one sharding layout (the one an optimizer-shard resume must keep), not across layouts."""

import functools
import hashlib

import torch
import triton
import triton.language as tl
from torch import Tensor

# Launch tile every SR kernel shares: each makes the same elementwise pass over a flattened parameter.
SR_BLOCK_SIZE = 1024
# Seeds stay inside int32, so a kernel's seed argument keeps one Triton type specialization.
_SR_SEED_MASK = (1 << 30) - 1
_MASK64 = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    """SplitMix64's output function: a 64-bit avalanche mix, so adjacent inputs map to unrelated outputs."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


@functools.cache
def sr_param_index(name: str) -> int:
    """The rounding-noise key of the parameter named ``name``: a stable 64-bit hash of the name.

    Not its position: ranks can hold different parameter lists (an expert-TP rank owns a bias its
    partners do not), and a replicated parameter must still round as it does on every other rank.
    """
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "little")


def require_param_names(param_groups: list[dict], optimizer: str) -> None:
    """Refuse param groups built without names, which :func:`sr_param_index` keys the rounding by."""
    if any("param_names" not in group for group in param_groups):
        raise ValueError(
            f"{optimizer} keys its stochastic rounding by parameter name: build it from named parameters "
            f"(decay_groups, or (name, param) pairs)."
        )


def sr_seed_pair(key: int, step: int, index: int) -> tuple[int, int]:
    """The SR seed pair of the parameter keyed ``index`` (:func:`sr_param_index`), at its ``step``.

    ``step`` is the count the restored state carries. Nothing is drawn, so the zero-LR step that
    materializes state before a restore consumes nothing a later step reads. The eager step seeds its
    two SR writes with the pair; the kernel derives both noise streams from the first.
    """
    mixed = _splitmix64(_splitmix64(key ^ step) ^ index)
    return mixed & _SR_SEED_MASK, (mixed >> 32) & _SR_SEED_MASK


def stochastic_round_to_bf16(x_fp32: Tensor, seed: int) -> Tensor:
    """Convert an fp32 tensor to bf16 by stochastic rounding, modifying it in-place.

    Noise comes from ``seed`` (:func:`sr_seed_pair`), not torch's default generator whose CUDA state
    drifts per rank, so replicated bf16 params round identically across replicas.
    """
    x_fp32 = x_fp32.contiguous()
    bits = x_fp32.view(torch.int32)
    gen = torch.Generator(device=bits.device)
    gen.manual_seed(seed)
    bits += torch.randint(0, 0x10000, bits.shape, dtype=bits.dtype, device=bits.device, generator=gen)
    bits &= 0xFFFF0000
    return x_fp32.to(torch.bfloat16)


@triton.jit
def sr_round_to_bf16(x_fp32, noise_int32):
    """Stochastically round ``x_fp32`` to bf16 in a kernel: add ``noise_int32`` (16 random bits) below the
    bf16 mantissa, then truncate them (``-65536`` is ``0xFFFF0000`` in two's complement)."""
    return ((x_fp32.to(tl.int32, bitcast=True) + noise_int32) & -65536).to(tl.float32, bitcast=True).to(tl.bfloat16)
