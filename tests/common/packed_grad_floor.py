"""The reference the packed GRPO suites judge a packed window's gradient against: the per-row window in fp32 on the
same rows, so the bar is the per-row bf16 path's own distance from the exact answer, which a packed varlen kernel's
rounding may match but not exceed by much."""

import copy
import math
from dataclasses import dataclass

import torch

from halo.distributed.fsdp import apply_fsdp2_for_dp
from tests.common.numerics import cos_sim, relative_l2

GRAD_FLOOR_FACTOR = 2.0
GRAD_COSINE_MIN = 0.999
# The unscaled control's error must exceed max(floor, packed error) by this factor. The rescale this suite exercises
# (rows_per_bin * slots / round_rows, 1.5) moves the gradient by about 1 - 1/1.5 = 0.33, 7.5x the ~0.045 bf16 floor
# measured against the fp32 reference; a bf16-noise error cannot reach 5x its own measured level.
CONTROL_MARGIN = 5
# Two bf16 windows over identical rows and plan must agree to this relative L2, or to REPEAT_MARGIN times the gap
# between two runs of the same packed window where the backward is not deterministic (FlashAttention's varlen dq
# accumulates with atomics): a rescale the loss should not carry moves the gradient by about 0.33.
IDENTICAL_WINDOW_TOLERANCE = 1e-3
REPEAT_MARGIN = 3


@dataclass(frozen=True)
class PackedGradVerdict:
    floor: float
    packed_err: float
    order_floor: float
    cosine: float
    unscaled_err: float
    unscaled_vs_packed: float
    repeat_vs_packed: float

    @property
    def metrics(self) -> dict[str, float]:
        return {
            "grad_floor": self.floor,
            "grad_packed": self.packed_err,
            "grad_order_floor": self.order_floor,
            "grad_cosine": self.cosine,
            "grad_unscaled": self.unscaled_err,
            "grad_unscaled_vs_packed": self.unscaled_vs_packed,
            "grad_repeat_vs_packed": self.repeat_vs_packed,
        }

    @property
    def inside_the_floor(self) -> bool:
        return (
            math.isfinite(self.packed_err) and self.floor > 0.0 and self.packed_err <= GRAD_FLOOR_FACTOR * self.floor
        )

    @property
    def points_the_same_way(self) -> bool:
        return self.cosine >= GRAD_COSINE_MIN

    @property
    def control_bites(self) -> bool:
        return self.unscaled_err > CONTROL_MARGIN * max(self.floor, self.packed_err)

    @property
    def rescale_changes_nothing(self) -> bool:
        return self.unscaled_vs_packed <= max(IDENTICAL_WINDOW_TOLERANCE, REPEAT_MARGIN * self.repeat_vs_packed)


def build_fp32_reference(model: torch.nn.Module, *, world_size: int) -> torch.nn.Module:
    """A copy of ``model`` computing in fp32 on sdpa attention (FlashAttention has no fp32 kernels), wrapped by FSDP2
    like it. Call before ``model`` is wrapped; the weights are ``model``'s upcast. Dense models only: sdpa shares the
    flash kernels' semantics there."""
    reference = copy.deepcopy(model).to(torch.float32)
    reference.config._attn_implementation = "sdpa"
    apply_fsdp2_for_dp(
        reference, world_size, compute_dtype=torch.float32, fp32_reduce=True, reshard_after_forward=True
    )
    return reference.train()


def _by_index(grads: list[torch.Tensor]) -> dict[int, torch.Tensor]:
    return dict(enumerate(grads))


def _flat(grads: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([grad.flatten() for grad in grads])


def judge_packed_grads(
    per_row: list[torch.Tensor],
    reorder: list[torch.Tensor],
    packed: list[torch.Tensor],
    unscaled: list[torch.Tensor],
    reference: list[torch.Tensor] | None,
    *,
    repeat: list[torch.Tensor] | None = None,
) -> PackedGradVerdict:
    """The bf16 per-row window's distance from the fp32 per-row ``reference`` is the floor; the packed window must stay
    within :data:`GRAD_FLOOR_FACTOR` of it. ``reorder`` (a per-row window in another shuffle) is reported as the order
    floor. Every error, the unscaled control's included, is measured against the same anchor (the fp32 ``reference``);
    ``unscaled_vs_packed`` is the direct distance of the two packed gradients and ``repeat_vs_packed`` that of a second
    run of the packed window (0 without ``repeat``). Without a ``reference`` (an arm with no fp32 model) the order floor
    is the floor and the per-row window the reference."""
    order_floor = relative_l2(_by_index(reorder), _by_index(per_row))
    anchor = _by_index(per_row if reference is None else reference)
    return PackedGradVerdict(
        floor=order_floor if reference is None else relative_l2(_by_index(per_row), anchor),
        packed_err=relative_l2(_by_index(packed), anchor),
        order_floor=order_floor,
        cosine=cos_sim(_flat(packed), _flat(per_row), label="packed vs per-row gradient"),
        unscaled_err=relative_l2(_by_index(unscaled), anchor),
        unscaled_vs_packed=relative_l2(_by_index(unscaled), _by_index(packed)),
        repeat_vs_packed=0.0 if repeat is None else relative_l2(_by_index(repeat), _by_index(packed)),
    )
