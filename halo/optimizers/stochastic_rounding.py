"""Stochastic rounding to bf16. The noise is a pure function of a parameter's optimizer step and name, so replicas
round alike and a resume rounds as the uninterrupted run did. An element's noise is keyed by its local-shard offset, so
it reproduces within one sharding layout only."""

import functools
import hashlib

import torch
import triton
import triton.language as tl
from torch import Tensor

# Launch tile every SR kernel shares.
SR_BLOCK_SIZE = 1024
# Seeds stay inside int32, so a kernel's seed argument keeps one Triton type specialization.
_SR_SEED_MASK = (1 << 30) - 1
_MASK64 = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    """SplitMix64's output function, so adjacent inputs map to unrelated outputs."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


@functools.cache
def sr_param_index(name: str) -> int:
    """Hashes the name, as ranks can hold different parameter lists (an expert-TP rank owns a bias its partners do not)
    and a replicated parameter must still round alike on every rank."""
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "little")


def require_param_names(param_groups: list[dict], optimizer: str) -> None:
    if any("param_names" not in group for group in param_groups):
        raise ValueError(
            f"{optimizer} keys its stochastic rounding by parameter name: build it from named parameters "
            f"(decay_groups, or (name, param) pairs)."
        )


def sr_seed_pair(key: int, step: int, index: int) -> tuple[int, int]:
    """Nothing is drawn, so the zero-LR step that materializes state before a restore consumes no later noise."""
    mixed = _splitmix64(_splitmix64(key ^ step) ^ index)
    return mixed & _SR_SEED_MASK, (mixed >> 32) & _SR_SEED_MASK


def stochastic_round_to_bf16(x_fp32: Tensor, seed: int) -> Tensor:
    """In place. Noise comes from ``seed``, as the default generator's CUDA state drifts per rank."""
    x_fp32 = x_fp32.contiguous()
    bits = x_fp32.view(torch.int32)
    gen = torch.Generator(device=bits.device)
    gen.manual_seed(seed)
    bits += torch.randint(0, 0x10000, bits.shape, dtype=bits.dtype, device=bits.device, generator=gen)
    bits &= 0xFFFF0000
    return x_fp32.to(torch.bfloat16)


@triton.jit
def sr_round_to_bf16(x_fp32, noise_int32):
    """Add ``noise_int32`` (16 random bits) below the bf16 mantissa, then truncate (``-65536`` is ``0xFFFF0000``)."""
    return ((x_fp32.to(tl.int32, bitcast=True) + noise_int32) & -65536).to(tl.float32, bitcast=True).to(tl.bfloat16)
