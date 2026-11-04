"""AdamW with bf16 master weights and moments (6 B/param), stochastically rounded so a sub-ULP ``lr*step`` update
and the non-negative second moment stay unbiased. The fused step bumps ``_version`` itself, as its Triton stores are
invisible to ATen and the low-precision weight cache keys on the counter."""

import functools
import logging
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple

import torch
import triton
import triton.language as tl
from torch import Tensor

from halo.optimizers.param_groups import decay_groups
from halo.optimizers.step_skip import DeviceStepSkipMixin, write_unless_skipped
from halo.optimizers.stochastic_rounding import (
    SR_BLOCK_SIZE,
    require_param_names,
    sr_param_index,
    sr_round_to_bf16,
    sr_seed_pair,
    stochastic_round_to_bf16,
)
from halo.runtime.dtensor import to_local
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# Key of this optimizer's rounding noise; Muon keys its own, so the two never share a stream.
_SR_KEY = 0xB165EED
# One thread per Philox lane of four elements, which beats Triton's default 4 warps on a memory-bound step.
_ADAM_NUM_WARPS = SR_BLOCK_SIZE // 4 // 32
# Rows of the int64 index table (_launch_tables): first program, element count, seed, each role's element offset.
_ROW_FIRST_PROGRAM = tl.constexpr(0)
_ROW_NUMEL = tl.constexpr(1)
_ROW_SEED = tl.constexpr(2)
_ROW_P = tl.constexpr(3)
_ROW_GRAD = tl.constexpr(4)
_ROW_EA = tl.constexpr(5)
_ROW_EASQ = tl.constexpr(6)
_ROLE_ROWS = tuple(row.value for row in (_ROW_P, _ROW_GRAD, _ROW_EA, _ROW_EASQ))
_INDEX_ROWS = _ROW_EASQ.value + 1
# Rows of the fp32 scalar table, in _FusedUpdate.hparams order.
_ROW_BETA1 = tl.constexpr(0)
_ROW_BETA2 = tl.constexpr(1)
_ROW_STEP_SIZE = tl.constexpr(2)
_ROW_BC2_SQRT = tl.constexpr(3)
_ROW_EPS = tl.constexpr(4)
_ROW_WD_FACTOR = tl.constexpr(5)
# Each program binary-searches its launch's first-program row, one probe per halving: 8 probes at this cap.
_MAX_TENSORS_PER_LAUNCH = 256
# B200 and B300, where the multi-tensor launch is measured. Elsewhere each launch steps one tensor at TABLE_SIZE 1,
# which compiles to a lone tensor's per-element code.
_MULTI_TENSOR_LAUNCH_CAPABILITIES = frozenset({(10, 0), (10, 3)})
# A program's lookup and table reads run once for all its blocks, so a block costs about what a lone launch's program
# does; more would leave a launch of small tensors too few programs to fill the GPU.
_BLOCKS_PER_PROGRAM = 16
# 64K registers over 2,048 threads, a lone tensor's launch's count. Uncapped, the per-tensor scalars a program holds
# across its blocks take 48, and an SM fits 5 programs instead of 8.
_ADAM_MAX_REGISTERS = 32
# Triton's launcher specializes an integer argument as divisible when it is a multiple of this, and a pointer
# argument when its address is.
_DIVISIBILITY = tl.constexpr(16)
_DIVISIBILITY_FLAGS = ("NUMEL_DIVISIBLE", "P_ALIGNED", "GRAD_ALIGNED", "EA_ALIGNED", "EASQ_ALIGNED")


@triton.constexpr_function
def _halvings(size: int) -> int:
    return size.bit_length() - 1


@triton.jit
def _program_blocks(
    index_table,
    num_tensors,
    NUMEL_DIVISIBLE: tl.constexpr,
    TABLE_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """The tensor slot this program steps, its element count and the range ``[first, end)`` of its blocks. The slot is
    binary-searched in the sorted first-program row; a probe past ``num_tensors`` reads the last tensor's entry."""
    pid = tl.program_id(0)
    tensor = 0
    for level in tl.static_range(_halvings(TABLE_SIZE)):
        probe = tl.minimum(tensor + (TABLE_SIZE >> (level + 1)), num_tensors - 1)
        tensor = tl.where(tl.load(index_table + probe) <= pid, probe, tensor)
    n_elements = tl.load(index_table + _ROW_NUMEL * num_tensors + tensor)
    # Hinted on the load, as Triton drops a hint applied to the value this call returns.
    if NUMEL_DIVISIBLE:
        n_elements = tl.multiple_of(n_elements, _DIVISIBILITY)
    first = (pid - tl.load(index_table + tensor)) * BLOCKS_PER_PROGRAM
    return tensor, n_elements, first, tl.minimum(first + BLOCKS_PER_PROGRAM, tl.cdiv(n_elements, BLOCK_SIZE))


@triton.jit(do_not_specialize=["num_tensors"])
def _adam_bf16_sr_kernel(
    p_base,
    grad_base,
    ea_base,
    easq_base,
    index_table,
    hparam_table,
    num_tensors,
    grad_scale_ptr,
    skip_ptr,
    HAS_GRAD_SCALE: tl.constexpr,
    HAS_SKIP: tl.constexpr,
    NUMEL_DIVISIBLE: tl.constexpr,
    P_ALIGNED: tl.constexpr,
    GRAD_ALIGNED: tl.constexpr,
    EA_ALIGNED: tl.constexpr,
    EASQ_ALIGNED: tl.constexpr,
    TABLE_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Adam step for a group of bf16 params: reads bf16, computes fp32, stochastically rounds the writes.
    A block's index within its tensor keys the noise exactly as a launch over that tensor alone would. The clip scale
    applies as CUDA's ``grad.mul_(scale)`` computes it, scale and product rounded to the gradient's dtype. The
    divisibility flags restore what the table loads hide from the compiler: the per-element code, each element's FMA
    association included, depends on them, so a tensor matches its lone launch bit for bit only under its own flags.
    ``num_tensors`` is unspecialized, so how many tensors a launch steps selects no variant."""
    # One Philox call per lane of four elements: 4 x 32 bits give each element two 16-bit draws (high half for the
    # second moment, low half for the weight). A call per element would make the Philox rounds the bound over memory.
    QUARTER: tl.constexpr = BLOCK_SIZE // 4
    tensor, n_elements, first_block, end_block = _program_blocks(
        index_table, num_tensors, NUMEL_DIVISIBLE, TABLE_SIZE, BLOCKS_PER_PROGRAM, BLOCK_SIZE
    )
    if HAS_SKIP:
        end_block = tl.where(tl.load(skip_ptr), first_block, end_block)
    seed = tl.load(index_table + _ROW_SEED * num_tensors + tensor)
    p_off = tl.load(index_table + _ROW_P * num_tensors + tensor)
    grad_off = tl.load(index_table + _ROW_GRAD * num_tensors + tensor)
    ea_off = tl.load(index_table + _ROW_EA * num_tensors + tensor)
    easq_off = tl.load(index_table + _ROW_EASQ * num_tensors + tensor)
    beta1 = tl.load(hparam_table + _ROW_BETA1 * num_tensors + tensor)
    beta2 = tl.load(hparam_table + _ROW_BETA2 * num_tensors + tensor)
    step_size = tl.load(hparam_table + _ROW_STEP_SIZE * num_tensors + tensor)
    bc2_sqrt = tl.load(hparam_table + _ROW_BC2_SQRT * num_tensors + tensor)
    eps = tl.load(hparam_table + _ROW_EPS * num_tensors + tensor)
    wd_factor = tl.load(hparam_table + _ROW_WD_FACTOR * num_tensors + tensor)
    if P_ALIGNED:
        p_off = tl.multiple_of(p_off, _DIVISIBILITY // p_base.dtype.element_ty.itemsize)
    if GRAD_ALIGNED:
        grad_off = tl.multiple_of(grad_off, _DIVISIBILITY // grad_base.dtype.element_ty.itemsize)
    if EA_ALIGNED:
        ea_off = tl.multiple_of(ea_off, _DIVISIBILITY // ea_base.dtype.element_ty.itemsize)
    if EASQ_ALIGNED:
        easq_off = tl.multiple_of(easq_off, _DIVISIBILITY // easq_base.dtype.element_ty.itemsize)

    for block in range(first_block, end_block):
        lane = tl.arange(0, QUARTER)
        offsets = block * BLOCK_SIZE + (lane[:, None] * 4 + tl.arange(0, 4)[None, :])
        mask = offsets < n_elements

        p = tl.load(p_base + p_off + offsets, mask=mask).to(tl.float32)
        grad_raw = tl.load(grad_base + grad_off + offsets, mask=mask)
        grad = grad_raw.to(tl.float32)
        if HAS_GRAD_SCALE:
            scale = tl.load(grad_scale_ptr).to(grad_raw.dtype).to(tl.float32)
            grad = (grad * scale).to(grad_raw.dtype).to(tl.float32)
        ea = tl.load(ea_base + ea_off + offsets, mask=mask).to(tl.float32)
        easq = tl.load(easq_base + easq_off + offsets, mask=mask).to(tl.float32)

        ea = beta1 * ea + (1.0 - beta1) * grad
        easq = beta2 * easq + (1.0 - beta2) * grad * grad

        # exp_avg is a signed ~zero-mean EMA, so nearest rounding is already unbiased.
        tl.store(ea_base + ea_off + offsets, ea.to(tl.bfloat16), mask=mask)

        r0, r1, r2, r3 = tl.randint4x(seed, block * QUARTER + lane)
        bits = tl.reshape(tl.join(tl.join(r0, r1), tl.join(r2, r3)), (QUARTER, 4)).to(tl.uint32, bitcast=True)
        easq_noise = (bits >> 16).to(tl.int32)
        rand_noise = (bits & 0xFFFF).to(tl.int32)

        # Near the bf16 underflow floor nearest rounding biases this accumulator, and the step, by tens of percent.
        tl.store(easq_base + easq_off + offsets, sr_round_to_bf16(easq, easq_noise), mask=mask)

        # The weight update reads the un-rounded second moment.
        p = p * wd_factor
        denom = tl.math.sqrt(easq) / bc2_sqrt + eps
        p = p - step_size * ea / denom
        tl.store(p_base + p_off + offsets, sr_round_to_bf16(p, rand_noise), mask=mask)


class _FusedUpdate(NamedTuple):
    param: Tensor
    tensors: tuple[Tensor, Tensor, Tensor, Tensor]  # local p, grad, exp_avg, exp_avg_sq
    hparams: tuple[float, float, float, float, float, float]  # in _ROW_BETA1 .. _ROW_WD_FACTOR order
    seed: int
    divisible: tuple[bool, bool, bool, bool, bool]  # _divisibility(tensors)

    @property
    def group_key(self) -> tuple:
        """A launch specializes on each role's dtype and device and on the divisibility flags."""
        return (*((tensor.device, tensor.dtype) for tensor in self.tensors), self.divisible)


def _step_hparams(group: dict, step: int) -> tuple[float, float, float, float, float, float]:
    """In ``_ROW_BETA1 .. _ROW_WD_FACTOR`` order."""
    beta1, beta2 = group["betas"]
    lr = group["lr"]
    step_size = lr / (1.0 - beta1**step)
    bc2_sqrt = math.sqrt(1.0 - beta2**step)
    return beta1, beta2, step_size, bc2_sqrt, group["eps"], 1.0 - lr * group["weight_decay"]


def _flagged_fp32_operands(
    skip: Tensor, dtype: torch.dtype, beta1: float, beta2: float, wd_factor: float, eps: float
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """0-dim operands that let the skip flag ride the update's own ops: a -0.0 gradient, the moment decays, the weight
    decay factor and ``eps``. Unset, each holds its float's value, so the step writes the same bits. Set, the factors
    are one and ``eps`` is infinite: the moments keep their values and the weight update is a signed zero, which keeps
    every weight except a -0.0 one beside a negative (or -0.0) ``exp_avg``, which turns +0.0."""
    factors = tuple(
        torch.full((), value, dtype=dtype, device=skip.device).masked_fill_(skip, identity)
        for value, identity in ((beta1, 1.0), (beta2, 1.0), (wd_factor, 1.0), (eps, math.inf))
    )
    return torch.full((), -0.0, dtype=torch.float32, device=skip.device), *factors


def _divisibility(tensors: Sequence[Tensor]) -> tuple[bool, bool, bool, bool, bool]:
    """What Triton would mark divisible in a launch over ``tensors`` alone: the element count, each tensor's start."""
    divisor = _DIVISIBILITY.value
    return (tensors[0].numel() % divisor == 0, *(tensor.data_ptr() % divisor == 0 for tensor in tensors))


def _specialization(divisible: Iterable[tuple[bool, ...]]) -> dict[str, bool]:
    """The flags must agree, as flags other than a tensor's own compile other per-element code for it."""
    distinct = set(divisible)
    if len(distinct) != 1:
        raise ValueError(f"a fused AdamWBF16 launch takes one specialization, got {sorted(distinct)}")
    return dict(zip(_DIVISIBILITY_FLAGS, distinct.pop(), strict=True))


def _fused_update(p: Tensor, name: str, state: dict, group: dict) -> _FusedUpdate:
    tensors = (to_local(p), to_local(p.grad), to_local(state["exp_avg"]), to_local(state["exp_avg_sq"]))
    step = state["step"]
    # The kernel derives both noise streams from the first seed.
    seed = sr_seed_pair(_SR_KEY, step, sr_param_index(name))[0]
    return _FusedUpdate(p, tensors, _step_hparams(group, step), seed, _divisibility(tensors))


def _launch_tables(updates: Sequence[_FusedUpdate]) -> tuple[Tensor, Tensor, int]:
    """The int64 index table, the fp32 scalar table (rounded from Python floats as the launcher rounds a float
    argument) and the total program count. An empty update owns no program."""
    # One launch shares each role's dtype, so the first update's tensors give every role's base and element size.
    bases = [tensor.data_ptr() for tensor in updates[0].tensors]
    sizes = [tensor.element_size() for tensor in updates[0].tensors]
    per_program = SR_BLOCK_SIZE * _BLOCKS_PER_PROGRAM
    rows: list[list[int]] = [[] for _ in range(_INDEX_ROWS)]
    first_programs, numels, seeds = rows[_ROW_FIRST_PROGRAM], rows[_ROW_NUMEL], rows[_ROW_SEED]
    offsets = [rows[row] for row in _ROLE_ROWS]
    num_programs = 0
    for update in updates:
        n = update.tensors[0].numel()
        first_programs.append(num_programs)
        numels.append(n)
        seeds.append(update.seed)
        for column, tensor, base, size in zip(offsets, update.tensors, bases, sizes, strict=True):
            byte_offset = tensor.data_ptr() - base
            if tensor.numel() != n or not tensor.is_contiguous() or byte_offset % size:
                raise ValueError(
                    f"AdamWBF16 steps contiguous, element-aligned local shards: got a {tuple(tensor.shape)} "
                    f"tensor with strides {tensor.stride()} at byte offset {byte_offset} beside a "
                    f"{n}-element parameter"
                )
            column.append(byte_offset // size)
        num_programs += (n + per_program - 1) // per_program
    index = torch.tensor(rows, dtype=torch.int64)
    hparams = torch.tensor(list(zip(*(update.hparams for update in updates), strict=True)), dtype=torch.float32)
    return index, hparams, num_programs


def _launch_fused(
    updates: Sequence[_FusedUpdate],
    index: Tensor,
    hparams: Tensor,
    num_programs: int,
    grad_scale: Tensor | None,
    skip: Tensor | None,
    table_size: int,
) -> None:
    if len(updates) > table_size or table_size & (table_size - 1):
        raise ValueError(
            f"a fused AdamWBF16 launch searches a power-of-two table no smaller than its tensors: got a table of "
            f"{table_size} for {len(updates)} tensors"
        )
    p, grad, exp_avg, exp_avg_sq = updates[0].tensors
    _adam_bf16_sr_kernel[(num_programs,)](
        p,
        grad,
        exp_avg,
        exp_avg_sq,
        index,
        hparams,
        len(updates),
        grad_scale if grad_scale is not None else grad,
        skip if skip is not None else grad,
        HAS_GRAD_SCALE=grad_scale is not None,
        HAS_SKIP=skip is not None,
        TABLE_SIZE=table_size,
        BLOCKS_PER_PROGRAM=_BLOCKS_PER_PROGRAM,
        BLOCK_SIZE=SR_BLOCK_SIZE,
        num_warps=_ADAM_NUM_WARPS,
        maxnreg=_ADAM_MAX_REGISTERS,
        **_specialization(update.divisible for update in updates),
    )


@functools.cache
def _tensors_per_launch(device: torch.device) -> int:
    if torch.cuda.get_device_capability(device) in _MULTI_TENSOR_LAUNCH_CAPABILITIES:
        return _MAX_TENSORS_PER_LAUNCH
    return 1


def _fused_adam_bf16_step(updates: Sequence[_FusedUpdate], grad_scale: Tensor | None, skip: Tensor | None) -> None:
    """Step non-empty ``updates``, at most :func:`_tensors_per_launch` sharing a group key, in one kernel launch."""
    index, hparams, num_programs = _launch_tables(updates)
    device = updates[0].tensors[0].device
    # A pageable copy would synchronize the stream at the step's head.
    index = index.pin_memory().to(device, non_blocking=True)
    hparams = hparams.pin_memory().to(device, non_blocking=True)
    _launch_fused(updates, index, hparams, num_programs, grad_scale, skip, _tensors_per_launch(device))


def _scaled_grad(grad: Tensor, grad_scale: Tensor | None) -> Tensor:
    return grad if grad_scale is None else grad * grad_scale


def _eager_adam_bf16_step(
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
    sr_seeds: tuple[int, int],
    grad_scale: Tensor | None = None,
    skip: Tensor | None = None,
):
    """Eager Adam step for one bf16 parameter, seeding its two SR writes with ``sr_seeds``."""
    easq_seed, weight_seed = sr_seeds
    p_data = to_local(p).detach()
    grad_fp32 = _scaled_grad(to_local(grad), grad_scale).float()
    exp_avg = to_local(exp_avg)
    exp_avg_sq = to_local(exp_avg_sq)

    # In fp32, as in the kernel, since in-place bf16 ops would lose small gradients.
    ea_fp32 = exp_avg.float()
    ea_fp32.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
    write_unless_skipped(exp_avg, ea_fp32.bfloat16(), skip)

    easq_fp32 = exp_avg_sq.float()
    easq_fp32.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

    # sqrt() stays out of place, as the SR store below reads the un-rounded second moment.
    p_fp32 = p_data.float() * wd_factor
    denom = easq_fp32.sqrt().div_(bc2_sqrt).add_(eps)
    p_fp32.addcdiv_(ea_fp32, denom, value=-step_size)
    del denom

    write_unless_skipped(exp_avg_sq, stochastic_round_to_bf16(easq_fp32, seed=easq_seed), skip)
    write_unless_skipped(p_data, stochastic_round_to_bf16(p_fp32, seed=weight_seed), skip)
    # A DTensor's copy_ bumps only the local shard's counter, not the wrapper's cached_fake_quant reads.
    torch.autograd.graph.increment_version(p)


class AdamWBF16(DeviceStepSkipMixin, torch.optim.Optimizer):
    """bf16 params take the fused Triton step (eager with ``use_triton=False`` or off CUDA), many per launch on
    sm_100/103 and one elsewhere; fp32 params (``fp32_router`` / ``fp32_experts``) take plain AdamW."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        use_triton: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})
        require_param_names(self.param_groups, type(self).__name__)
        self._use_triton = use_triton
        self._grad_scale: Tensor | None = None

    def defer_grad_scale(self, scale: Tensor) -> None:
        """Multiply every gradient by ``scale`` (a device fp32 scalar) once, inside the next :meth:`step`. Each is
        scaled exactly as ``grad.mul_(scale)`` would, product rounded to its dtype, so the step matches clip-then-step
        bit for bit and bf16 rounding absorbs the coefficient's last-bit run-to-run noise as the in-place clip does."""
        self._grad_scale = scale.detach().to(dtype=torch.float32).reshape(())

    def zero_grad(self, set_to_none: bool = True) -> None:
        # A scale deferred for gradients that are being discarded must not reach the next step's.
        self._grad_scale = None
        super().zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grad_scale, self._grad_scale = self._grad_scale, None
        skip = self._take_step_skip(self.param_groups)
        # A full launch goes out at once, so the GPU starts on it while the host builds the rest.
        pending: dict[tuple, list[_FusedUpdate]] = defaultdict(list)
        fused_params = []
        for group in self.param_groups:
            # They carry the group's hyperparameters, so they live per group (and dtype).
            flagged_operands: dict[torch.dtype, tuple[Tensor, ...]] = {}
            for p, name in zip(group["params"], group["param_names"], strict=True):
                # A param with no grad counts too, so replicas whose grad presence differs agree on its seed's step.
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                if p.grad is None:
                    continue

                if p.dtype == torch.bfloat16 and self._use_triton and p.is_cuda:
                    fused_params.append(p)
                    update = _fused_update(p, name, state, group)
                    if not update.tensors[0].numel():
                        continue
                    launch = pending[update.group_key]
                    launch.append(update)
                    if len(launch) == _tensors_per_launch(update.tensors[0].device):
                        _fused_adam_bf16_step(launch, grad_scale, skip)
                        launch.clear()
                    continue
                beta1, beta2, step_size, bc2_sqrt, eps, wd_factor = _step_hparams(group, state["step"])
                if p.dtype == torch.bfloat16:
                    _eager_adam_bf16_step(
                        p,
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        step_size,
                        bc2_sqrt,
                        eps,
                        wd_factor,
                        beta1,
                        beta2,
                        sr_seed_pair(_SR_KEY, state["step"], sr_param_index(name)),
                        grad_scale,
                        skip,
                    )
                    continue

                # _foreach_* cannot mix FSDP2 DTensor params with plain tensors. All four tensors share one sharding,
                # so the pointwise update on local shards writes through.
                p_data = to_local(p).detach()
                exp_avg = to_local(state["exp_avg"])
                exp_avg_sq = to_local(state["exp_avg_sq"])
                grad_fp32 = _scaled_grad(to_local(p.grad), grad_scale).float()
                ea_decay, easq_decay, p_decay, denom_eps = beta1, beta2, wd_factor, eps
                if skip is not None:
                    if p_data.dtype not in flagged_operands:
                        flagged_operands[p_data.dtype] = _flagged_fp32_operands(
                            skip, p_data.dtype, beta1, beta2, wd_factor, eps
                        )
                    skipped_grad, ea_decay, easq_decay, p_decay, denom_eps = flagged_operands[p_data.dtype]
                    # A select, as a non-finite gradient survives multiplication by zero.
                    grad_fp32 = torch.where(skip, skipped_grad, grad_fp32)
                exp_avg.mul_(ea_decay).add_(grad_fp32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(easq_decay).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)
                if group["weight_decay"] != 0.0:
                    p_data.mul_(p_decay)
                denom = (exp_avg_sq.sqrt() / bc2_sqrt).add_(denom_eps)
                p_data.addcdiv_(exp_avg, denom, value=-step_size)

        for launch in pending.values():
            if launch:
                _fused_adam_bf16_step(launch, grad_scale, skip)
        if fused_params:
            torch.autograd.graph.increment_version(fused_params)
        return loss


def adamw_bf16_optimizer_from_args(model: torch.nn.Module, args: Any, decay_parameters: Sequence[str]):
    # Two groups even when one is empty, as the count defines the index space of a saved state dict.
    grouped = decay_groups(model.named_parameters(), decay_parameters, args.weight_decay, keep_empty=True)
    optimizer = AdamWBF16(
        grouped,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )
    if is_global_main_process():
        bf16_params = [p for g in grouped for p in g["params"] if p.dtype == torch.bfloat16]
        bf16_count = sum(p.numel() for p in bf16_params)
        fp32_count = sum(p.numel() for g in grouped for p in g["params"] if p.dtype == torch.float32)
        launch_device = next((p.device for p in bf16_params if p.is_cuda), None)
        bf16_step = (
            "eager step"
            if launch_device is None
            else f"fused Triton step, up to {_tensors_per_launch(launch_device)} per launch"
        )
        logger.info(
            f"✓ AdamWBF16 optimizer: {bf16_count / 1e6:.1f}M bf16 params (SR, {bf16_step}), "
            f"{fp32_count / 1e6:.1f}M fp32 params (standard)"
        )
    return optimizer
