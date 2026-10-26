"""AdamW with bf16 master weights and moments (6 B/param), stochastically rounded.

Stochastic rounding keeps the bf16 writes unbiased, where nearest rounding would truncate a sub-ULP
``lr*step`` update to zero and inflate the non-negative second moment. Writes go through
``p.detach()`` and bump ``_version`` explicitly: the Triton stores are invisible to ATen, and the
low-precision weight cache keys on the counter.
"""

import logging
import math
from collections.abc import Sequence
from typing import Any

import torch
import triton
import triton.language as tl
from torch import Tensor

from halo.optimizers.param_groups import decay_groups
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
# One thread per Philox lane of four elements: beats Triton's default 4 warps on a large parameter's
# memory-bound step.
_ADAM_NUM_WARPS = SR_BLOCK_SIZE // 4 // 32


@triton.jit
def _adam_bf16_sr_kernel(
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
    """Fused Adam step for bf16 params: reads bf16, computes fp32, stochastically rounds the writes.

    ``grad_scale_ptr`` (with ``HAS_GRAD_SCALE``) is the deferred clip coefficient, applied as CUDA's
    in-place ``grad.mul_(scale)`` computes it: scale and product rounded to the gradient's dtype.
    """
    # One Philox call per lane of four elements: its 4 x 32 bits give each element two 16-bit draws
    # (high half for the second moment, low half for the weight). A call per element would make the
    # Philox rounds, not memory, the bound.
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

    # exp_avg is a signed ~zero-mean EMA, so nearest rounding is already unbiased.
    tl.store(ea_ptr + offsets, ea.to(tl.bfloat16), mask=mask)

    r0, r1, r2, r3 = tl.randint4x(seed, pid * QUARTER + lane)
    bits = tl.reshape(tl.join(tl.join(r0, r1), tl.join(r2, r3)), (QUARTER, 4)).to(tl.uint32, bitcast=True)
    easq_noise = (bits >> 16).to(tl.int32)
    rand_noise = (bits & 0xFFFF).to(tl.int32)

    # Near the bf16 underflow floor nearest rounding biases this non-negative accumulator by tens of
    # percent, moving sqrt(v) and the effective step away from lr.
    tl.store(easq_ptr + offsets, sr_round_to_bf16(easq, easq_noise), mask=mask)

    # The weight update reads the un-rounded second moment.
    p = p * wd_factor
    denom = tl.math.sqrt(easq) / bc2_sqrt + eps
    p = p - step_size * ea / denom
    tl.store(p_ptr + offsets, sr_round_to_bf16(p, rand_noise), mask=mask)


def _triton_adam_bf16_step(
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
):
    """Launch :func:`_adam_bf16_sr_kernel` on one parameter's local shard."""
    p_flat = to_local(p.detach()).view(-1)
    grad_flat = to_local(grad).view(-1)
    n = p_flat.numel()
    _adam_bf16_sr_kernel[(triton.cdiv(n, SR_BLOCK_SIZE),)](
        p_flat,
        grad_flat,
        to_local(exp_avg).view(-1),
        to_local(exp_avg_sq).view(-1),
        beta1,
        beta2,
        step_size,
        bc2_sqrt,
        eps,
        wd_factor,
        n,
        sr_seeds[0],  # the kernel derives both noise streams from the first seed
        grad_scale if grad_scale is not None else grad_flat,
        HAS_GRAD_SCALE=grad_scale is not None,
        BLOCK_SIZE=SR_BLOCK_SIZE,
        num_warps=_ADAM_NUM_WARPS,
    )
    torch.autograd.graph.increment_version(p)


def _scaled_grad(grad: Tensor, grad_scale: Tensor | None) -> Tensor:
    """``grad`` times the deferred clip scale (:meth:`AdamWBF16.defer_grad_scale`)."""
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
):
    """Eager Adam step for one bf16 parameter, seeding its two SR writes with ``sr_seeds``."""
    easq_seed, weight_seed = sr_seeds
    p_data = to_local(p.detach())
    grad_fp32 = _scaled_grad(to_local(grad), grad_scale).float()
    exp_avg = to_local(exp_avg)
    exp_avg_sq = to_local(exp_avg_sq)

    # The EMAs run in fp32, as in the kernel: in-place bf16 ops would lose small gradients.
    ea_fp32 = exp_avg.float()
    ea_fp32.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
    exp_avg.copy_(ea_fp32.bfloat16())

    easq_fp32 = exp_avg_sq.float()
    easq_fp32.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

    # sqrt() stays out of place: the SR store below reads the un-rounded second moment.
    p_fp32 = p_data.float() * wd_factor
    denom = easq_fp32.sqrt().div_(bc2_sqrt).add_(eps)
    p_fp32.addcdiv_(ea_fp32, denom, value=-step_size)
    del denom

    exp_avg_sq.copy_(stochastic_round_to_bf16(easq_fp32, seed=easq_seed))
    p_data.copy_(stochastic_round_to_bf16(p_fp32, seed=weight_seed))
    # A DTensor's copy_ bumps only the local shard's counter, not the wrapper's cached_fake_quant reads.
    torch.autograd.graph.increment_version(p)


class AdamWBF16(torch.optim.Optimizer):
    """AdamW with stochastically rounded bf16 weights and moments.

    bf16 params take the fused Triton step (eager with ``use_triton=False`` or off CUDA); fp32 params
    (``fp32_router`` / ``fp32_experts``) take plain in-place AdamW.
    """

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
        """Multiply every gradient by ``scale`` (a device fp32 scalar) once, inside the next :meth:`step`.

        The gradient clip hands its coefficient here, so the scale rides the step's own read of each
        gradient instead of a separate pass. Each gradient is scaled exactly as ``grad.mul_(scale)``
        would (product rounded to the gradient's dtype), so the step matches clip-then-step bit for bit
        and bf16 rounding absorbs the coefficient's last-bit run-to-run noise as the in-place clip does.
        """
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
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p, name in zip(group["params"], group["param_names"], strict=True):
                # The step count advances for every param, including one with no grad, so replicas
                # whose grad presence differs still agree on the step a param's seed is keyed by.
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
                    # Triton needs CUDA storage, so the param's own device picks the path.
                    update_fn = _triton_adam_bf16_step if self._use_triton and p.is_cuda else _eager_adam_bf16_step
                    update_fn(
                        p,
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        step_size,
                        bc2_sqrt,
                        eps,
                        1.0 - lr * wd,
                        beta1,
                        beta2,
                        sr_seed_pair(_SR_KEY, step, sr_param_index(name)),
                        grad_scale,
                    )
                    continue

                # Per param, not _foreach_*: FSDP2 DTensor params cannot mix with plain tensors. All four
                # tensors share one sharding, so the pointwise update on local shards writes through.
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

        return loss


def adamw_bf16_optimizer_from_args(model: torch.nn.Module, args: Any, decay_parameters: Sequence[str]):
    """AdamWBF16 built from the training args."""
    # Two groups even when one is empty: the count defines the index space of a saved state dict.
    grouped = decay_groups(model.named_parameters(), decay_parameters, args.weight_decay, keep_empty=True)
    optimizer = AdamWBF16(
        grouped,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )
    if is_global_main_process():
        bf16_count = sum(p.numel() for g in grouped for p in g["params"] if p.dtype == torch.bfloat16)
        fp32_count = sum(p.numel() for g in grouped for p in g["params"] if p.dtype == torch.float32)
        logger.info(
            f"✓ AdamWBF16 optimizer: {bf16_count / 1e6:.1f}M bf16 params (SR), "
            f"{fp32_count / 1e6:.1f}M fp32 params (standard)"
        )
    return optimizer
