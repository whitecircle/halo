"""AdamW with bf16 master weights and moments (6 B/param), stochastically rounded.

Stochastic rounding keeps the bf16 writes unbiased, where nearest rounding would truncate a sub-ULP
``lr*step`` update to zero and inflate the non-negative second moment. Writes go through
``p.detach()`` (``p.data`` carries its own version counter) and bump ``_version`` explicitly, since
the raw-pointer Triton stores are invisible to ATen and the low-precision weight cache keys on it.
"""

import logging
import math
import random
from collections.abc import Sequence
from typing import Any

import torch
import triton
import triton.language as tl
from torch import Tensor

from src.distributed.runtime import is_global_main_process, to_local
from src.optimizers.param_groups import decay_groups

logger = logging.getLogger(__name__)

# SR-seed RNG kept off global ``random`` (the data path advances that per-rank) and seeded
# identically everywhere, so replicated bf16 params (HSDP/DDP/EP) round the same way.
_SR_RNG = random.Random(0xB165EED)

# Launch tile shared by every SR kernel here and in Muon: all of them make the same elementwise pass
# over a flattened parameter.
BLOCK_SIZE = 1024


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
    seed,  # one Philox call yields both noise streams
    grad_scale_ptr,
    HAS_GRAD_SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Adam update with stochastic rounding for bf16 params (reads bf16, computes fp32, SR-writes bf16).

    ``grad_scale_ptr`` (with ``HAS_GRAD_SCALE``) is a device fp32 scalar multiplied into the gradient as it
    is read: the deferred gradient-clip coefficient, which then costs no pass of its own over the grads.
    """
    # Each lane owns four consecutive elements and one Philox call: its 4 x 32 random bits give every
    # element its own two 16-bit noise draws (high half for the second moment, low half for the weight).
    # The Philox rounds, not memory, bounded the one-call-per-element form.
    QUARTER: tl.constexpr = BLOCK_SIZE // 4
    pid = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, QUARTER)
    offsets = pid * BLOCK_SIZE + (lane[:, None] * 4 + tl.arange(0, 4)[None, :])
    mask = offsets < n_elements

    p = tl.load(p_ptr + offsets, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
    if HAS_GRAD_SCALE:
        grad = grad * tl.load(grad_scale_ptr)
    ea = tl.load(ea_ptr + offsets, mask=mask).to(tl.float32)
    easq = tl.load(easq_ptr + offsets, mask=mask).to(tl.float32)

    ea = beta1 * ea + (1.0 - beta1) * grad
    easq = beta2 * easq + (1.0 - beta2) * grad * grad

    # exp_avg stored nearest: a signed ~zero-mean EMA, so truncation is already unbiased.
    tl.store(ea_ptr + offsets, ea.to(tl.bfloat16), mask=mask)

    r0, r1, r2, r3 = tl.randint4x(seed, pid * QUARTER + lane)
    bits = tl.reshape(tl.join(tl.join(r0, r1), tl.join(r2, r3)), (QUARTER, 4)).to(tl.uint32, bitcast=True)
    easq_noise = (bits >> 16).to(tl.int32)
    rand_noise = (bits & 0xFFFF).to(tl.int32)

    # SR the second moment: nearest rounding biases this non-negative accumulator upward, inflating
    # sqrt(v) and shrinking the effective step below lr.
    easq_bits = easq.to(tl.int32, bitcast=True)
    easq_bits = (easq_bits + easq_noise) & (-65536)
    tl.store(easq_ptr + offsets, easq_bits.to(tl.float32, bitcast=True).to(tl.bfloat16), mask=mask)

    # Weight update in fp32 using pre-rounding easq (exact second moment).
    p = p * wd_factor
    denom = tl.math.sqrt(easq) / bc2_sqrt + eps
    p = p - step_size * ea / denom

    p_bits = p.to(tl.int32, bitcast=True)
    p_bits = (p_bits + rand_noise) & (-65536)  # -65536 == 0xFFFF0000 (two's complement)
    p_sr = p_bits.to(tl.float32, bitcast=True).to(tl.bfloat16)

    tl.store(p_ptr + offsets, p_sr, mask=mask)


def _draw_sr_seeds(use_triton: bool) -> tuple[int, int]:
    """Draw the SR seed pair one bf16 parameter consumes, in the range its update path needs.

    Both paths draw a pair (the kernel derives both noise streams from one Philox call and uses only
    the first), so the number of draws per parameter does not depend on the path taken. ``step``
    draws unconditionally, including for a param it skips, so rank-nonuniform grad presence cannot
    shift the rank-synchronized ``_SR_RNG`` stream and drift replicated params apart.
    """
    if use_triton:
        return _SR_RNG.randint(0, 2**30), _SR_RNG.randint(0, 2**30)
    return _SR_RNG.randint(0, 2**31 - 1), _SR_RNG.randint(0, 2**31 - 1)


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
    sr_seeds: tuple[int, int] | None = None,
    grad_scale: Tensor | None = None,
):
    """Launch the fused Adam+SR Triton kernel for a single parameter (``grad_scale``: see the kernel)."""
    p_data = to_local(p.detach())
    grad_local = to_local(grad)
    ea_local = to_local(exp_avg)
    easq_local = to_local(exp_avg_sq)

    p_flat = p_data.view(-1)
    grad_flat = grad_local.view(-1)
    ea_flat = ea_local.view(-1)
    easq_flat = easq_local.view(-1)
    n = p_flat.numel()

    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Pre-drawn by ``step`` (structural) or drawn here for direct calls; the kernel uses only the
    # first of the pair (see :func:`_draw_sr_seeds`).
    seed = (sr_seeds if sr_seeds is not None else _draw_sr_seeds(use_triton=True))[0]

    _adam_bf16_sr_kernel[grid](
        p_flat,
        grad_flat,
        ea_flat,
        easq_flat,
        beta1,
        beta2,
        step_size,
        bc2_sqrt,
        eps,
        wd_factor,
        n,
        seed,
        grad_scale if grad_scale is not None else grad_flat,
        HAS_GRAD_SCALE=grad_scale is not None,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    torch.autograd.graph.increment_version(p)  # the raw-pointer store above is invisible to ATen


def stochastic_round_to_bf16(x_fp32: Tensor, seed: int | None = None) -> Tensor:
    """Convert an fp32 tensor to bf16 by stochastic rounding, modifying it in-place.

    Noise comes from the rank-synchronized ``_SR_RNG``, not torch's default generator whose CUDA
    state drifts per rank, so replicated bf16 params round identically across replicas.
    """
    x_fp32 = x_fp32.contiguous()
    bits = x_fp32.view(torch.int32)
    gen = torch.Generator(device=bits.device)
    gen.manual_seed(seed if seed is not None else _SR_RNG.randint(0, 2**31 - 1))
    bits += torch.randint(0, 0x10000, bits.shape, dtype=bits.dtype, device=bits.device, generator=gen)
    bits &= 0xFFFF0000
    return x_fp32.to(torch.bfloat16)


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
    sr_seeds: tuple[int, int] | None = None,
    grad_scale: Tensor | None = None,
):
    """Eager (non-Triton) Adam+SR step for a single bf16 parameter."""
    if sr_seeds is None:
        sr_seeds = _draw_sr_seeds(use_triton=False)
    easq_seed, weight_seed = sr_seeds
    p_data = to_local(p.detach())
    grad = to_local(grad)
    if grad_scale is not None:
        grad = grad.float() * grad_scale
    exp_avg = to_local(exp_avg)
    exp_avg_sq = to_local(exp_avg_sq)

    # EMA upcast to fp32; in-place bf16 ops would lose precision on small grads and diverge from Triton.
    ea_fp32 = exp_avg.float()
    ea_fp32.mul_(beta1).add_(grad.float(), alpha=1.0 - beta1)
    exp_avg.copy_(ea_fp32.bfloat16())

    easq_fp32 = exp_avg_sq.float()
    grad_fp32 = grad.float()
    easq_fp32.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

    # Un-rounded second moment: sqrt() stays out-of-place so easq_fp32 survives for the SR store below.
    p_fp32 = p_data.float() * wd_factor
    denom = easq_fp32.sqrt().div_(bc2_sqrt).add_(eps)
    p_fp32.addcdiv_(ea_fp32, denom, value=-step_size)
    del denom

    exp_avg_sq.copy_(stochastic_round_to_bf16(easq_fp32, seed=easq_seed))

    p_data.copy_(stochastic_round_to_bf16(p_fp32, seed=weight_seed))
    # For a DTensor param the copy_ bumps only the local shard's counter, not the wrapper's that
    # cached_fake_quant keys on; for plain tensors this is a harmless second bump.
    torch.autograd.graph.increment_version(p)


class AdamWBF16(torch.optim.Optimizer):
    """AdamW with stochastic rounding for bf16 master weights (6 B/param).

    bf16 params take the fused Triton (or eager) Adam+SR path; fp32 params (e.g. ``fp32_router`` /
    ``fp32_experts``) take standard in-place AdamW without SR. use_triton=False uses eager ops (slower).
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

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        # Resolved per parameter at step time from the tensor's own device: the eager path is
        # equivalent, and gating on `torch.cuda.is_available()` would send a CPU param to Triton.
        self._use_triton = use_triton
        self._grad_scale: Tensor | None = None

    def defer_grad_scale(self, scale: Tensor) -> None:
        """Multiply every gradient by ``scale`` (a device fp32 scalar) inside the next :meth:`step`.

        The gradient clip hands its coefficient here instead of rescaling the gradients in place, so the
        scale rides the optimizer's own read of each gradient rather than costing a separate pass over
        all of them. The gradients themselves stay unscaled until the step; the scale applies once.
        """
        self._grad_scale = scale.detach().to(dtype=torch.float32).reshape(())

    def zero_grad(self, set_to_none: bool = True) -> None:
        # A scale deferred for gradients that are being discarded must not reach the next step's.
        self._grad_scale = None
        super().zero_grad(set_to_none=set_to_none)

    def _triton_for(self, param: torch.Tensor) -> bool:
        """Whether ``param`` takes the fused kernel. Its device decides — Triton needs CUDA storage."""
        return self._use_triton and param.is_cuda

    @torch.no_grad()
    def step(self, closure=None):
        """Single step: bf16 params → fused Triton Adam+SR (or eager); fp32 params → standard in-place AdamW."""
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

            bf16_params = []
            fp32_params = []

            for p in group["params"]:
                # Seed draw and step count advance for every param, including one with no grad;
                # otherwise rank-nonuniform grad presence would drift replicas apart.
                sr_seeds = _draw_sr_seeds(self._triton_for(p)) if p.dtype == torch.bfloat16 else None

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1

                if p.grad is None:
                    continue

                if p.dtype == torch.bfloat16:
                    bf16_params.append((p, p.grad, state, sr_seeds))
                else:
                    fp32_params.append((p, p.grad, state))

            if bf16_params:
                for p, grad, state, sr_seeds in bf16_params:
                    update_fn = _triton_adam_bf16_step if self._triton_for(p) else _eager_adam_bf16_step
                    step = state["step"]
                    bc1 = 1.0 - beta1**step
                    bc2_sqrt = math.sqrt(1.0 - beta2**step)
                    step_size = lr / bc1
                    wd_factor = 1.0 - lr * wd

                    update_fn(
                        p,
                        grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        step_size,
                        bc2_sqrt,
                        eps,
                        wd_factor,
                        beta1,
                        beta2,
                        sr_seeds,
                        grad_scale,
                    )

            # Per-param updates (no _foreach_* — FSDP2 DTensor params can't mix with plain Tensors).
            for p, grad, state in fp32_params:
                step = state["step"]
                bc1 = 1.0 - beta1**step
                bc2_sqrt = math.sqrt(1.0 - beta2**step)
                step_size = lr / bc1

                # All four share one sharding, so the pointwise update on local shards writes through.
                p_data = to_local(p.detach())
                exp_avg = to_local(state["exp_avg"])
                exp_avg_sq = to_local(state["exp_avg_sq"])
                grad_fp32 = to_local(grad)
                grad_fp32 = grad_fp32.float() if grad_fp32.dtype != torch.float32 else grad_fp32
                if grad_scale is not None:
                    grad_fp32 = grad_fp32 * grad_scale

                exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

                if wd != 0.0:
                    p_data.mul_(1.0 - lr * wd)

                denom = (exp_avg_sq.sqrt() / bc2_sqrt).add_(eps)
                p_data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def build_bf16_optimizer(model: torch.nn.Module, args: Any, decay_parameters: Sequence[str]):
    """AdamWBF16 with stochastic rounding, built from the training args.

    Handles mixed dtypes internally: bf16 params use the fused Triton SR kernel, fp32 params use
    standard in-place AdamW.
    """
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
