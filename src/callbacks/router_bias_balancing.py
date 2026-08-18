"""DeepSeek-V3 style bias-update router balancing: ``b_i <- b_i + gamma * sign(mean_count - count_i)``.

Router contract: a ``balancing_biases`` buffer (shape [E] or [E+1]) added to the *detached* top-k
scores, so gradients keep flowing through the un-biased gate, plus a lazily-initialised
``expert_load_counter`` this callback all-reduces, sign-steps and zeros every step. That counter is a
plain attribute, never a buffer: it must stay out of ``state_dict()`` exports and out of the PP lazy
loader, which materializes state-dict tensors only.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist

from src.callbacks.moe_metrics import MoELoadMetricsCallback, compute_moe_load_metrics
from src.distributed.runtime import get_global_world_size, is_global_main_process
from src.models.moe_balancing import is_transient_balancing_router, iter_balancing_routers

logger = logging.getLogger(__name__)

# Overshoot guard for the TRANSIENT side-buffer, whose bias adds to selection scores of uniform
# scale 1/num_experts: warn once one sign step exceeds this fraction of uniform (the controller
# then oscillates instead of converging), and quote a rate that would keep it at the second.
_TRANSIENT_STEP_WARN_FRACTION_OF_UNIFORM = 0.25
_TRANSIENT_STEP_SUGGESTED_FRACTION_OF_UNIFORM = 0.1

# Value the discard/null slot is pinned to under ``exclude_last_slot``: far enough below the routed
# slots' bias range that the discard never wins a top-k comparison it would not have won unbiased.
_DISCARD_SLOT_BIAS = -1.0


def _ensure_counter(router, num_slots: int) -> torch.Tensor:
    """Return ``router.expert_load_counter``, lazy-initialising on the bias device."""
    counter = router.expert_load_counter
    if counter is None:
        counter = torch.zeros(num_slots, dtype=torch.float32, device=router.balancing_biases.device)
        router.expert_load_counter = counter
    return counter


class RouterBiasBalancingCallback(MoELoadMetricsCallback):
    """Post-step DeepSeek-V3 bias update for any router with a balancing-bias buffer.

    Args (beyond the base's ``exclude_last_slot`` / ``log_every_n_steps``):
        update_rate: Sign-step magnitude (``gamma``). DeepSeek-V3 used ``1e-3``.

    ``exclude_last_slot`` additionally excludes the trailing slot from the sign update and clamps it
    to :data:`_DISCARD_SLOT_BIAS`. The bias update itself always runs every step — only the metric
    summary is periodic, since it costs a host sync.
    """

    def __init__(
        self,
        update_rate: float = 1e-3,
        exclude_last_slot: bool = False,
        log_every_n_steps: int = 1,
    ):
        if update_rate <= 0:
            raise ValueError(f"update_rate must be > 0, got {update_rate}")
        super().__init__(exclude_last_slot=exclude_last_slot, log_every_n_steps=log_every_n_steps)
        self.update_rate = float(update_rate)

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        routers = list(iter_balancing_routers(model))
        if not routers:
            if is_global_main_process():
                logger.warning(
                    "RouterBiasBalancingCallback enabled but no router with a "
                    "balancing_biases buffer was found in the model — callback is a no-op."
                )
            return
        for router in routers:
            # Arms recording on routers whose bias state exists regardless of balancing (Zaya's
            # native buffer); EP layers key recording on the bias the enable created, so for them
            # this is a no-op marker.
            router.balancing_active = True
        # Transient routers only (:data:`_TRANSIENT_STEP_WARN_FRACTION_OF_UNIFORM`): a native slot
        # lives in the family's own score space, where GPT-OSS logits even want a LARGER rate.
        transient_widths = {r.balancing_biases.numel() for r in routers if is_transient_balancing_router(r)}
        if transient_widths:
            num_experts = max(transient_widths) - (1 if self.exclude_last_slot else 0)
            step_fraction_of_uniform = self.update_rate * num_experts
            if step_fraction_of_uniform > _TRANSIENT_STEP_WARN_FRACTION_OF_UNIFORM and is_global_main_process():
                suggested = _TRANSIENT_STEP_SUGGESTED_FRACTION_OF_UNIFORM / num_experts
                logger.warning(
                    f"router_balancing_rate={self.update_rate} is {step_fraction_of_uniform:.2f}x "
                    f"the uniform routing probability at {num_experts} experts — the transient bias "
                    f"step overshoots and load will oscillate. Scale it down in proportion "
                    f"(~{suggested:.1e} keeps one step at "
                    f"{_TRANSIENT_STEP_SUGGESTED_FRACTION_OF_UNIFORM:.0%} of uniform); watch moe/load_cv."
                )
        if is_global_main_process():
            logger.info(
                f"RouterBiasBalancingCallback active: {len(routers)} routers, "
                f"update_rate={self.update_rate}, exclude_last_slot={self.exclude_last_slot}"
            )

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return

        routers = list(iter_balancing_routers(model))
        if not routers:
            return

        device = routers[0].balancing_biases.device
        widths = {r.balancing_biases.numel() for r in routers}
        if len(widths) > 1:
            raise ValueError(
                f"RouterBiasBalancingCallback requires one expert-slot width across routers, got "
                f"{sorted(widths)}. The per-slot bias update and its all-reduce assume a single width."
            )
        counters = torch.stack([_ensure_counter(r, r.balancing_biases.numel()).to(device) for r in routers], dim=0)

        if get_global_world_size() > 1:
            dist.all_reduce(counters, op=dist.ReduceOp.SUM, group=self.reduce_group)

        # sign(0-0)=0 makes the update a silent no-op, so an all-zero counter must be surfaced.
        if self._first_log and counters.sum() == 0 and is_global_main_process():
            logger.warning(
                "RouterBiasBalancingCallback: all-reduced expert-load counters are entirely ZERO at "
                "the first bias update — no routed tokens were recorded, so bias updates are no-ops. "
                "The forward path is not reaching the balancing-enabled routers' load recording."
            )

        # MoEMetricsCallback is skipped under bias_update; the shared helper keeps moe/* keys identical.
        # It ends in a GPU→CPU sync only on_log consumes, so it runs on the steps the trainer logs —
        # a rank-uniform gate; the all-reduce above stays unconditional because the update needs it.
        step = getattr(state, "global_step", None)
        if self._first_log or step is None or step % self.log_every_n_steps == 0:
            self._pending = compute_moe_load_metrics(counters, exclude_last_slot=self.exclude_last_slot)

        real_counts = counters[:, :-1] if self.exclude_last_slot else counters
        mean_real = real_counts.mean(dim=-1, keepdim=True)
        update = self.update_rate * torch.sign(mean_real - real_counts)

        # strict: ``update``'s first dimension IS ``counters``', which was stacked from ``routers``
        # — a length mismatch means the router list changed under the callback, and silently
        # truncating would leave the tail of the model balancing on a frozen bias.
        for router, upd in zip(routers, update, strict=True):
            if self.exclude_last_slot:
                router.balancing_biases[:-1].add_(upd)
                router.balancing_biases[-1] = _DISCARD_SLOT_BIAS
            else:
                router.balancing_biases.add_(upd)
            # _ensure_counter above assigned every router's counter, so this is unconditional.
            router.expert_load_counter.zero_()

        if self._first_log:
            if is_global_main_process():
                logger.info(
                    f"RouterBiasBalancingCallback: first update applied to {len(routers)} routers "
                    f"(reduce-group per-slot totals, router 0 — under TP/ETP partners recount shared "
                    f"tokens, a uniform scale the sign update is invariant to): {counters[0].tolist()} | "
                    f"metrics: {self._pending}"
                )
            self._first_log = False
