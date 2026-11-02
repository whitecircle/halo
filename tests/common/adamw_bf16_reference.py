"""The per-parameter AdamWBF16 step the fused launch replaced: the bit-for-bit oracle both test tiers hold it to.

``_per_parameter_kernel`` and ``PerParameterAdamWBF16.step`` are the one-launch-per-parameter implementation;
:func:`per_parameter_step` launches it with its float arguments rounded to fp32 first, which the CUDA launcher
does itself and the interpreter does not. :func:`fused_kernel_mismatches` runs the fused kernel and the oracle over
the same parameter set and names every output that differs. It runs on CUDA and, with ``TRITON_INTERPRET=1`` set
before Triton is imported, on CPU. :func:`lookup_mismatches` holds the fused kernel's block lookup to the per-block
scan of the block-start row (``_block_scan_kernel``), the one-block-per-program lookup the fused kernel ran before.
"""

import math
import struct
from collections import defaultdict

import torch
import triton
import triton.language as tl
from torch import Tensor

from halo.optimizers.adamw_bf16 import (
    _ADAM_NUM_WARPS,
    _BLOCKS_PER_PROGRAM,
    _DIVISIBILITY,
    _MAX_TENSORS_PER_LAUNCH,
    _SR_KEY,
    AdamWBF16,
    _divisibility,
    _eager_adam_bf16_step,
    _FusedUpdate,
    _launch_fused,
    _launch_tables,
    _program_blocks,
    _scaled_grad,
)
from halo.optimizers.stochastic_rounding import SR_BLOCK_SIZE, sr_param_index, sr_round_to_bf16, sr_seed_pair
from halo.runtime.dtensor import to_local

LR = 1e-2
BETAS = (0.9, 0.999)
EPS = 1e-8
# (element count, optimizer step, weight decay, aligned): empty, sub-block, block-boundary and multi-block tensors at
# mixed steps, as a resumed run can hold them, and one tensor of three programs, the last one short. ``aligned``
# starts all four of a tensor's roles on 16-byte boundaries; otherwise its grad, and every second param, sits at an
# odd element offset. The launches split by specialization: the three aligned tensors (a padded table), five with
# only the param and moments aligned (the three-program one among them), two with only the moments, and the unaligned
# 1024-element one alone (its element count is a multiple of 16).
CASE = (
    (0, 1, 0.0, False),
    (1, 1, 0.1, False),
    (3, 3, 0.1, False),
    (1023, 7, 0.0, False),
    (1024, 1, 0.1, False),
    (1025, 2, 0.0, False),
    (4099, 5, 0.1, False),
    (2 * 1024 + 7, 9, 0.1, False),
    (1024, 3, 0.1, True),
    (2048, 4, 0.0, True),
    (3 * 1024 + 16, 6, 0.1, True),
    ((2 * _BLOCKS_PER_PROGRAM + 3) * 1024 + 5, 4, 0.1, False),
)
# bf16 elements per 16-byte boundary.
_BOUNDARY = _DIVISIBILITY.value // torch.bfloat16.itemsize
# Block counts a lookup table's tensors draw from: empty, one-block, program-boundary and multi-program tensors.
_LOOKUP_BLOCKS = (0, 1, 2, *(_BLOCKS_PER_PROGRAM + d for d in (-1, 0, 1)), 2 * _BLOCKS_PER_PROGRAM + 3)


@triton.jit
def _per_parameter_kernel(
    p_ptr,
    grad_ptr,
    ea_ptr,
    easq_ptr,
    beta1,
    beta2,
    step_size,
    bc2_sqrt,
    eps,
    wd_factor,
    n_elements,
    seed,
    grad_scale_ptr,
    HAS_GRAD_SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    QUARTER: tl.constexpr = BLOCK_SIZE // 4
    pid = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, QUARTER)
    offsets = pid * BLOCK_SIZE + (lane[:, None] * 4 + tl.arange(0, 4)[None, :])
    mask = offsets < n_elements

    p = tl.load(p_ptr + offsets, mask=mask).to(tl.float32)
    grad_raw = tl.load(grad_ptr + offsets, mask=mask)
    grad = grad_raw.to(tl.float32)
    if HAS_GRAD_SCALE:
        scale = tl.load(grad_scale_ptr).to(grad_raw.dtype).to(tl.float32)
        grad = (grad * scale).to(grad_raw.dtype).to(tl.float32)
    ea = tl.load(ea_ptr + offsets, mask=mask).to(tl.float32)
    easq = tl.load(easq_ptr + offsets, mask=mask).to(tl.float32)

    ea = beta1 * ea + (1.0 - beta1) * grad
    easq = beta2 * easq + (1.0 - beta2) * grad * grad

    tl.store(ea_ptr + offsets, ea.to(tl.bfloat16), mask=mask)

    r0, r1, r2, r3 = tl.randint4x(seed, pid * QUARTER + lane)
    bits = tl.reshape(tl.join(tl.join(r0, r1), tl.join(r2, r3)), (QUARTER, 4)).to(tl.uint32, bitcast=True)
    easq_noise = (bits >> 16).to(tl.int32)
    rand_noise = (bits & 0xFFFF).to(tl.int32)

    tl.store(easq_ptr + offsets, sr_round_to_bf16(easq, easq_noise), mask=mask)

    p = p * wd_factor
    denom = tl.math.sqrt(easq) / bc2_sqrt + eps
    p = p - step_size * ea / denom
    tl.store(p_ptr + offsets, sr_round_to_bf16(p, rand_noise), mask=mask)


def _fp32(value: float) -> float:
    """``value`` rounded to fp32, as the CUDA launcher passes a float argument; the interpreter keeps the double."""
    return struct.unpack("f", struct.pack("f", value))[0]


def per_parameter_step(
    p: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step_size: float,
    bc2_sqrt: float,
    eps: float,
    wd_factor: float,
    beta1: float,
    beta2: float,
    seed: int,
    grad_scale: Tensor | None = None,
) -> None:
    """One launch of the per-parameter kernel over one parameter's local shard."""
    p_flat = to_local(p.detach()).view(-1)
    grad_flat = to_local(grad).view(-1)
    n = p_flat.numel()
    _per_parameter_kernel[(triton.cdiv(n, SR_BLOCK_SIZE),)](
        p_flat,
        grad_flat,
        to_local(exp_avg).view(-1),
        to_local(exp_avg_sq).view(-1),
        _fp32(beta1),
        _fp32(beta2),
        _fp32(step_size),
        _fp32(bc2_sqrt),
        _fp32(eps),
        _fp32(wd_factor),
        n,
        seed,
        grad_scale if grad_scale is not None else grad_flat,
        HAS_GRAD_SCALE=grad_scale is not None,
        BLOCK_SIZE=SR_BLOCK_SIZE,
        num_warps=_ADAM_NUM_WARPS,
    )
    torch.autograd.graph.increment_version(p)


class PerParameterAdamWBF16(AdamWBF16):
    """AdamWBF16 stepping each bf16 CUDA parameter with its own per-parameter launch."""

    @torch.no_grad()
    def step(self, closure=None):
        grad_scale, self._grad_scale = self._grad_scale, None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p, name in zip(group["params"], group["param_names"], strict=True):
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                if p.grad is None:
                    continue
                step = state["step"]
                bc2_sqrt = math.sqrt(1.0 - beta2**step)
                step_size = lr / (1.0 - beta1**step)
                if p.dtype == torch.bfloat16:
                    seeds = sr_seed_pair(_SR_KEY, step, sr_param_index(name))
                    args = (p, p.grad, state["exp_avg"], state["exp_avg_sq"], step_size, bc2_sqrt, eps, 1.0 - lr * wd)
                    if self._use_triton and p.is_cuda:
                        per_parameter_step(*args, beta1, beta2, seeds[0], grad_scale)
                    else:
                        _eager_adam_bf16_step(*args, beta1, beta2, seeds, grad_scale)
                    continue
                p_data = to_local(p.detach())
                exp_avg = to_local(state["exp_avg"])
                exp_avg_sq = to_local(state["exp_avg_sq"])
                grad_fp32 = _scaled_grad(to_local(p.grad), grad_scale).float()
                exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)
                if wd != 0.0:
                    p_data.mul_(1.0 - lr * wd)
                denom = (exp_avg_sq.sqrt() / bc2_sqrt).add_(eps)
                p_data.addcdiv_(exp_avg, denom, value=-step_size)
        return None


def _packed_views(buffer: Tensor, order: list[int]) -> list[Tensor]:
    """Views of :data:`CASE`'s tensors packed into ``buffer`` in ``order``: an aligned one starts on a 16-byte
    boundary, any other at an odd element offset."""
    views: list[Tensor | None] = [None] * len(CASE)
    cursor = 0
    for i in order:
        n, _, _, aligned = CASE[i]
        cursor = -(-cursor // _BOUNDARY) * _BOUNDARY + (0 if aligned else 1)
        views[i] = buffer.narrow(0, cursor, n)
        cursor += n
    return views


def _case_tensors(device: str) -> list[list[Tensor]]:
    """``[params, grads, exp_avgs, exp_avg_sqs]`` over :data:`CASE`. Every second param is a view into one shared
    buffer, as fused-parameter slices are; every grad is a view into another, packed in reverse as an FSDP2
    reduce-scatter output can be, so the grad role carries negative offsets from its base."""
    generator = torch.Generator().manual_seed(1)
    sizes = [n for n, _, _, _ in CASE]
    room = sum(sizes) + _BOUNDARY * (len(CASE) + 1)
    shared_params = _packed_views(torch.randn(room, generator=generator).bfloat16().to(device), list(range(len(CASE))))
    params = [
        shared_params[i] if i % 2 == 0 else torch.randn(n, generator=generator).bfloat16().to(device)
        for i, n in enumerate(sizes)
    ]
    grads = _packed_views(
        torch.randn(room, generator=generator).bfloat16().to(device), list(reversed(range(len(CASE))))
    )
    exp_avgs = [(torch.randn(n, generator=generator) * 1e-3).bfloat16().to(device) for n in sizes]
    exp_avg_sqs = [(torch.rand(n, generator=generator) * 1e-6).bfloat16().to(device) for n in sizes]
    return [params, grads, exp_avgs, exp_avg_sqs]


def fused_kernel_mismatches(device: str, *, seed_shift: int = 0) -> list[str]:
    """Step :data:`CASE` three times (grad scale off, on, off) through the fused kernel, one launch per
    specialization as ``step()`` groups them, and through the oracle; name each differing output and each tensor
    whose divisibility is not what :data:`CASE` placed.

    ``seed_shift`` offsets the fused launch's seeds: a comparison that still reports nothing then is blind to the
    rounding noise.
    """
    oracle, fused = _case_tensors(device), _case_tensors(device)
    mismatches = []
    for grad_scale in (None, torch.tensor(0.3, device=device), None):
        launches: dict[tuple, list[_FusedUpdate]] = defaultdict(list)
        for i, (n, step, wd, aligned) in enumerate(CASE):
            bc2_sqrt = math.sqrt(1.0 - BETAS[1] ** step)
            step_size = LR / (1.0 - BETAS[0] ** step)
            seed = sr_seed_pair(_SR_KEY, step, sr_param_index(f"p{i}"))[0]
            tensors = tuple(role[i] for role in fused)
            hparams = (BETAS[0], BETAS[1], step_size, bc2_sqrt, EPS, 1.0 - LR * wd)
            update = _FusedUpdate(tensors[0], tensors, hparams, seed + seed_shift, _divisibility(tensors))
            if n and all(update.divisible) != aligned:
                mismatches.append(f"divisible[{i}] is {update.divisible}, placed {'' if aligned else 'un'}aligned")
            if n:
                launches[update.group_key].append(update)
            per_parameter_step(
                *(role[i] for role in oracle), step_size, bc2_sqrt, EPS, 1.0 - LR * wd, *BETAS, seed, grad_scale
            )
        for live in launches.values():
            index, hparams, num_programs = _launch_tables(live)
            _launch_fused(live, index.to(device), hparams.to(device), num_programs, grad_scale, None)
        for role, name in enumerate(("param", "grad", "exp_avg", "exp_avg_sq")):
            for i, (expected, got) in enumerate(zip(oracle[role], fused[role], strict=True)):
                if not torch.equal(expected, got):
                    mismatches.append(f"scale={grad_scale is not None} {name}[{i}] ({CASE[i][0]} elements)")
    return mismatches


@triton.jit
def _block_scan_kernel(block_starts, num_tensors, out, TABLE_SIZE: tl.constexpr):
    """The block-start scan: one program per block scans the whole block-start row for the last tensor that starts
    at or before it, and stores that tensor and the block's index within it."""
    pid = tl.program_id(0).to(tl.int64)
    slots = tl.arange(0, TABLE_SIZE)
    starts = tl.load(block_starts + slots, mask=slots < num_tensors, other=2**63 - 1)
    owned = starts <= pid
    tl.store(out + 2 * pid, tl.sum(owned.to(tl.int32), axis=0) - 1)
    tl.store(out + 2 * pid + 1, pid - tl.max(tl.where(owned, starts, 0), axis=0))


@triton.jit
def _program_blocks_kernel(
    index_table,
    num_tensors,
    out,
    TABLE_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Each program's tensor and block indices as the fused kernel resolves them, one row per block it steps."""
    tensor, _, first, end = _program_blocks(
        index_table, num_tensors, False, TABLE_SIZE, BLOCKS_PER_PROGRAM, BLOCK_SIZE
    )
    for block in range(first, end):
        row = tl.program_id(0) * BLOCKS_PER_PROGRAM + block - first
        tl.store(out + 2 * row, tensor)
        tl.store(out + 2 * row + 1, block)


def _lookup_tables() -> list[tuple[list[int], int]]:
    """(block counts per tensor, TABLE_SIZE): one tensor and two (TABLE_SIZE 1 and 2), three in a table padded to the
    cap, a full table at the cap whose first and last tensors are empty, and a table just past a power of two."""
    generator = torch.Generator().manual_seed(5)

    def ragged(count: int) -> list[int]:
        return [_LOOKUP_BLOCKS[i] for i in torch.randint(len(_LOOKUP_BLOCKS), (count,), generator=generator).tolist()]

    full = [0, *ragged(_MAX_TENSORS_PER_LAUNCH - 2), 0]
    return [
        ([3], 1),
        ([1, 2 * _BLOCKS_PER_PROGRAM + 3], 2),
        ([2, 0, _BLOCKS_PER_PROGRAM + 1], _MAX_TENSORS_PER_LAUNCH),
        (full, _MAX_TENSORS_PER_LAUNCH),
        (ragged(_MAX_TENSORS_PER_LAUNCH // 2 + 1), _MAX_TENSORS_PER_LAUNCH),
    ]


def lookup_mismatches() -> list[str]:
    """Resolve every program of each :func:`_lookup_tables` table, built by :func:`_launch_tables`, to the tensor and
    block indices it steps, and every block to them by the block scan; name each table where the two lists differ.

    The fused lookup also runs on each table with every tensor but the first moved one program on: if no table then
    differs, the comparison is blind to the lookup and that is reported too.
    """
    mismatches, blind = [], True
    for blocks, table_size in _lookup_tables():
        # Alternate full and short last blocks; the element count, not the block count, is what the table holds.
        numels = [b * SR_BLOCK_SIZE - (i % 2) * 7 if b else 0 for i, b in enumerate(blocks)]
        storage = torch.empty(max(numels), dtype=torch.bfloat16)
        updates = [_FusedUpdate(storage, (storage[:n],) * 4, (0.0,) * 6, 0, (False,) * 5) for n in numels]
        index, _, num_programs = _launch_tables(updates)
        block_starts = torch.tensor([sum(blocks[:i]) for i in range(len(blocks))], dtype=torch.int64)
        expected = torch.empty((sum(blocks), 2), dtype=torch.int64)
        _block_scan_kernel[(sum(blocks),)](block_starts, len(blocks), expected, TABLE_SIZE=table_size)
        for shift in (0, 1):
            table = index.clone()
            table[0, 1:] += shift
            found = torch.full((num_programs * _BLOCKS_PER_PROGRAM, 2), -1, dtype=torch.int64)
            _program_blocks_kernel[(num_programs,)](
                table,
                len(blocks),
                found,
                TABLE_SIZE=table_size,
                BLOCKS_PER_PROGRAM=_BLOCKS_PER_PROGRAM,
                BLOCK_SIZE=SR_BLOCK_SIZE,
            )
            matches = torch.equal(found[found[:, 0] >= 0], expected)
            if shift:
                blind &= matches
            elif not matches:
                mismatches.append(f"{len(blocks)} tensors at TABLE_SIZE {table_size}")
    if blind:
        mismatches.append("a table shifted by one program resolved like the scan: the comparison is blind")
    return mismatches
