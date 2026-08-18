"""Generic MoE load-distribution metrics (``moe/*``) for any HF MoE model that declares a router.

A forward hook counts each router's OWN selection (a balancing bias or group-limited routing moves it
away from ``topk(router_logits)``, the fallback for a router publishing logits alone); ``on_step_end``
all-reduces the per-step counts and ``on_log`` merges the summary. Turning ``output_router_logits`` on
is never a side effect of asking for metrics — it also couples the router aux loss into the loss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist
from transformers import TrainerCallback

from src.distributed.runtime import current_device, get_global_world_size, is_global_main_process
from src.kernels.histogram import accumulate_bincount
from src.log import warn_once
from src.models.loading.config_levels import (
    get_config_field,
    set_config_field_run_scoped,
)
from src.models.moe_balancing import (
    has_discard_expert_slot,
    router_logits_forced_off,
)

logger = logging.getLogger(__name__)

# Key under which transformers' per-model-class ``_can_record_outputs`` declares its router, and the
# tuple position it captures for a bare-class spec that names no index (its installer's own default
# for every key but ``hidden_states``).
_ROUTER_LOGITS_KEY = "router_logits"
_BARE_SPEC_LOGITS_INDEX = 1


def compute_moe_load_metrics(
    counts: torch.Tensor,
    exclude_last_slot: bool = False,
) -> dict[str, float]:
    """Summarize per-layer expert load counts into the standard ``moe/`` metrics.

    Shared by both collection strategies (router-selection counts, EP ``expert_load_counter``) so the
    keys and units stay identical whichever callback emits them.

    Args:
        counts: ``[num_layers, num_experts]`` token totals per (layer, expert); already all-reduced if distributed.
        exclude_last_slot: drop the trailing column before stats (routers with a discard/null slot, e.g. Zaya).
    """
    if exclude_last_slot and counts.shape[-1] > 1:
        counts = counts[:, :-1]

    counts = counts.to(torch.float32)
    per_layer_totals = counts.sum(dim=-1, keepdim=True).clamp(min=1.0)
    load_fracs = counts / per_layer_totals
    e_eff = load_fracs.shape[-1]
    uniform = 1.0 / max(e_eff, 1)

    per_layer_max = load_fracs.max(dim=-1).values / uniform
    per_layer_min = load_fracs.min(dim=-1).values / uniform
    per_layer_mean = load_fracs.mean(dim=-1)
    per_layer_std = load_fracs.std(dim=-1, unbiased=False)
    per_layer_cv = per_layer_std / per_layer_mean.clamp(min=1e-9)
    per_layer_dead = (counts == 0).float().mean(dim=-1)

    # One D2H sync for all six scalars; six separate .item() calls stall the stream six times.
    load_max, load_min, load_cv, dead_frac, load_max_first, load_max_last = torch.stack(
        (
            per_layer_max.mean(),
            per_layer_min.mean(),
            per_layer_cv.mean(),
            per_layer_dead.mean(),
            per_layer_max[0],
            per_layer_max[-1],
        )
    ).tolist()

    return {
        "moe/load_max": load_max,
        "moe/load_min": load_min,
        "moe/load_cv": load_cv,
        "moe/dead_frac": dead_frac,
        "moe/load_max_first": load_max_first,
        "moe/load_max_last": load_max_last,
        "moe/num_layers": float(counts.shape[0]),
    }


def _extract_router_logits(output) -> tuple[torch.Tensor, ...] | None:
    """Return the tuple of per-layer router logits if present, else None."""
    router_logits = getattr(output, "router_logits", None)
    if router_logits is None and isinstance(output, dict):
        router_logits = output.get("router_logits")
    if router_logits is None:
        return None
    if isinstance(router_logits, torch.Tensor):
        return (router_logits,)
    out = tuple(t for t in router_logits if isinstance(t, torch.Tensor) and t.ndim >= 2)
    return out if out else None


def _selected_experts(router_output, logits_index: int) -> torch.Tensor | None:
    """The expert ids a router selected, taken from its own output, or None if it returns logits alone.

    Identified structurally, not per family: the selection is the one integer tensor sharing the
    logits' token dimensions with a narrower trailing one (``[*tokens, top_k]``), at any position.
    """
    if not isinstance(router_output, (tuple, list)) or not 0 <= logits_index < len(router_output):
        return None
    logits = router_output[logits_index]
    if not isinstance(logits, torch.Tensor):
        return None
    for position, item in enumerate(router_output):
        if position == logits_index or not isinstance(item, torch.Tensor):
            continue
        if item.dtype.is_floating_point or item.dtype == torch.bool or item.dtype.is_complex:
            continue
        if item.ndim == logits.ndim and item.shape[:-1] == logits.shape[:-1] and item.shape[-1] <= logits.shape[-1]:
            return item
    return None


@dataclass(frozen=True)
class _HookedRouter:
    """One router module the callback counts, and how to read its forward output.

    ``logits_index`` is the tuple position transformers' own recorder captures as ``router_logits``.
    ``folds_discard_slot`` marks a router whose indices mask skipped tokens onto a real expert id, so
    only its logits — which keep the trailing discard column — express the routing.
    """

    name: str
    module: torch.nn.Module
    logits_index: int
    folds_discard_slot: bool


def _router_logits_recorders(model) -> list[tuple]:
    """Every ``router_logits`` capture spec declared anywhere in the module tree, de-duplicated.

    ``_can_record_outputs`` is transformers' own registry of which module produces ``router_logits``,
    so it also names the routers to hook. The walk and de-duplication are for composite models, which
    declare it on the sub-model owning the routers and repeat it on the causal-LM wrapper.
    """
    specs: dict[tuple, None] = {}
    for module in model.modules():
        entry = (getattr(module, "_can_record_outputs", None) or {}).get(_ROUTER_LOGITS_KEY)
        if entry is None:
            continue
        for spec in entry if isinstance(entry, (list, tuple)) else (entry,):
            target_class = getattr(spec, "target_class", spec if isinstance(spec, type) else None)
            class_name = getattr(spec, "class_name", spec if isinstance(spec, str) else None)
            index = getattr(spec, "index", _BARE_SPEC_LOGITS_INDEX)
            specs[(target_class, class_name, getattr(spec, "layer_name", None), index)] = None
    return list(specs)


def _hooked_routers(model) -> list[_HookedRouter]:
    """The declared router modules in module-tree order, which is decoder-layer order.

    Matching mirrors transformers' own installer (target class, else a class-name suffix, refined by
    ``layer_name``), so the hooks land on exactly the modules an ``output_router_logits`` capture would.
    """
    specs = _router_logits_recorders(model)
    routers: list[_HookedRouter] = []
    for name, module in model.named_modules():
        for target_class, class_name, layer_name, index in specs:
            if target_class is not None:
                if not isinstance(module, target_class):
                    continue
            elif class_name is None or not name.endswith(class_name):
                continue
            if layer_name is not None and f".{layer_name.strip('.')}." not in f"{name}.":
                continue
            routers.append(_HookedRouter(name, module, index, has_discard_expert_slot(module)))
            break
    return routers


class MoELoadMetricsCallback(TrainerCallback):
    """Base for the callbacks that derive ``moe/*`` from per-expert load counters.

    :class:`MoEMetricsCallback` (hooked routers) and
    :class:`~src.callbacks.router_bias_balancing.RouterBiasBalancingCallback` (the bias-update load
    counter) share the periodic-summary contract, the pending-metrics handoff to the trainer log, and
    ``reduce_group`` — which the PP mixin sets by scanning for THIS type, so a third such callback
    cannot silently miss the stage group. ``reduce_group`` narrows the expert-load all-reduce to a
    subset of the world (``None`` = world); under PP a world reduce would blend stages holding
    different routers.

    Args:
        exclude_last_slot: drop the trailing expert from stats — for routers with a learned
            "discard" slot at the last index (Zaya).
        log_every_n_steps: derive + emit the summary every Nth optimizer step.
    """

    def __init__(self, exclude_last_slot: bool = False, log_every_n_steps: int = 1):
        if log_every_n_steps < 1:
            raise ValueError(f"log_every_n_steps must be >= 1, got {log_every_n_steps}")
        self.exclude_last_slot = bool(exclude_last_slot)
        self.log_every_n_steps = int(log_every_n_steps)
        self.reduce_group: dist.ProcessGroup | None = None
        self._first_log = True
        self._pending: dict[str, float] = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and self._pending:
            logs.update(self._pending)
            self._pending = {}


class MoEMetricsCallback(MoELoadMetricsCallback):
    """Generic MoE expert-load metrics.

    Args (beyond the base's ``exclude_last_slot`` / ``log_every_n_steps``):
        topk: experts selected per token. Default 1 (top-1 routers); pass the model's top-k otherwise.
            Used only where a router exposes no selection of its own and its logits must be re-ranked.
        enable_router_logits: allow the callback to switch ``output_router_logits`` on — pass True
            only from a balancing mode that already requires them, since it costs a
            ``[B*S, num_experts]`` tensor per layer per forward and couples the aux loss into the loss.
    """

    def __init__(
        self,
        topk: int = 1,
        exclude_last_slot: bool = False,
        log_every_n_steps: int = 1,
        enable_router_logits: bool = False,
    ):
        if topk < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")
        super().__init__(exclude_last_slot=exclude_last_slot, log_every_n_steps=log_every_n_steps)
        self.topk = int(topk)
        self.enable_router_logits = bool(enable_router_logits)

        self._counters: list[torch.Tensor | None] = []
        self._hook_handles: list = []
        self._model_type: str | None = None
        self._warned_logits_only: set = set()
        self._warned_empty = False

    def _router_hook(self, position: int, router: _HookedRouter, module, inputs, output):
        """Count what THIS router selected; re-rank its logits only if it returns no selection.

        Gradient checkpointing re-runs the router in the recompute pass; every metric is a per-layer
        share, so the doubled counts cancel and no de-duplication is needed.
        """
        if not module.training:
            return
        with torch.no_grad():
            logits = output[router.logits_index] if isinstance(output, (tuple, list)) else output
            if not isinstance(logits, torch.Tensor):
                return
            indices = None if router.folds_discard_slot else _selected_experts(output, router.logits_index)
            if indices is None and not router.folds_discard_slot:
                warn_once(
                    logger,
                    self._warned_logits_only,
                    type(module),
                    "MoEMetricsCallback: %s (%s, model_type=%s) returns router logits but no selected "
                    "expert ids, so moe/* counts topk(router_logits). That is the model's routing only "
                    "while the router adds no balancing bias and limits no expert groups before its "
                    "own top-k; otherwise the logged load approximates it.",
                    router.name,
                    type(module).__name__,
                    self._model_type,
                )
            if indices is None:
                indices = torch.topk(logits, self.topk, dim=-1).indices
            self._counters[position] = accumulate_bincount(self._counters[position], indices, logits.shape[-1])

    def _hook(self, module, inputs, output):
        """Fallback: per-layer logits off the model output, for a modeling declaring no router."""
        if not module.training:
            return
        router_logits = _extract_router_logits(output)
        if router_logits is None:
            return
        with torch.no_grad():
            if len(self._counters) != len(router_logits):
                self._counters = [None] * len(router_logits)
            for i, logits in enumerate(router_logits):
                if logits is None:
                    continue
                _, top_idx = torch.topk(logits, self.topk, dim=-1)
                self._counters[i] = accumulate_bincount(self._counters[i], top_idx, logits.shape[-1])

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        cfg = getattr(model, "config", None)
        if cfg is None:
            if is_global_main_process():
                logger.warning("MoEMetricsCallback: model has no config; skipping wiring.")
            return
        if router_logits_forced_off(cfg):
            # Re-enabling here would undo the balancing strategy's deliberate off decision.
            if is_global_main_process():
                logger.warning(
                    "MoEMetricsCallback: output_router_logits is forced off by the moe_balancing "
                    "strategy for this model; skipping wiring (no moe/* metrics from this callback). "
                    "Use moe_balancing=bias_update for load metrics via RouterBiasBalancingCallback."
                )
            return
        if not get_config_field(cfg, "output_router_logits", False):
            if not self.enable_router_logits:
                if is_global_main_process():
                    logger.info(
                        "MoEMetricsCallback: output_router_logits is off and no balancing mode requires "
                        "it, so no moe/* metrics are emitted (collecting them would add a "
                        "[B*S, num_experts] tensor per MoE layer per forward and feed the router aux "
                        "loss into the total loss). Set output_router_logits=true in model_init_kwargs "
                        "to opt in, or moe_balancing=aux_loss/bias_update for balancing plus metrics."
                    )
                return
            set_config_field_run_scoped(cfg, "output_router_logits", True)

        self._model_type = getattr(cfg, "model_type", type(cfg).__name__)
        # A second train() on the same callback would otherwise stack a second hook per router and
        # double every count.
        self._remove_hooks()
        routers = _hooked_routers(model)
        if routers:
            self._counters = [None] * len(routers)
            self._hook_handles = [
                router.module.register_forward_hook(partial(self._router_hook, position, router))
                for position, router in enumerate(routers)
            ]
        else:
            self._hook_handles = [model.register_forward_hook(self._hook)]
        if is_global_main_process():
            source = f"{len(routers)} routers" if routers else "the model output's router_logits"
            logger.info(
                f"MoEMetricsCallback wired on {type(model).__name__} via {source} "
                f"(topk={self.topk}, exclude_last_slot={self.exclude_last_slot}, "
                f"log_every={self.log_every_n_steps})"
            )

    def _remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def on_train_end(self, args, state, control, **kwargs):
        self._remove_hooks()

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        counters = self._counters
        # Counters are zeroed every step, logged or not, so each metric covers one optimizer step.
        # This gate is derived from rank-uniform trainer state, so no rank can skip the collective below.
        if state.global_step % self.log_every_n_steps != 0:
            for c in counters:
                if c is not None:
                    c.zero_()
            return

        # Rank-uniform by construction: a rank whose hook captured nothing must not return early or
        # build a differently-shaped payload (it would hang its peers in NCCL), and still joins on its
        # CUDA device — NCCL rejects CPU tensors.
        active = [c for c in counters if c is not None]
        device = active[0].device if active else current_device()
        dims = torch.tensor(
            [len(counters), max((c.shape[0] for c in active), default=0)],
            device=device,
            dtype=torch.long,
        )
        distributed = get_global_world_size() > 1
        if distributed:
            dist.all_reduce(dims, op=dist.ReduceOp.MAX, group=self.reduce_group)
        num_layers, max_e = int(dims[0]), int(dims[1])

        if num_layers == 0 or max_e == 0:
            # Only a hooked callback can diagnose an empty capture; when wiring was declined the reason
            # was already logged at train begin and this warning would name the wrong cause.
            if not self._warned_empty and state.global_step >= 1 and self._hook_handles:
                self._warned_empty = True
                if is_global_main_process():
                    logger.warning(
                        "MoEMetricsCallback captured no routing over the first optimizer step — "
                        "emitting no moe/* metrics. Likely because the routing bypasses the HF router "
                        "module (Expert Parallelism) or the trainer computes log-probs from the "
                        "backbone only (GRPO). For MoE load metrics in those cases use "
                        "moe_balancing=bias_update (metrics come from RouterBiasBalancingCallback)."
                    )
            return

        stacked = torch.zeros(num_layers, max_e, device=device, dtype=torch.float32)
        for i, c in enumerate(counters):
            if c is not None:
                stacked[i, : c.shape[0]] = c

        if distributed:
            dist.all_reduce(stacked, op=dist.ReduceOp.SUM, group=self.reduce_group)

        self._pending = compute_moe_load_metrics(stacked, exclude_last_slot=self.exclude_last_slot)

        if self._first_log and is_global_main_process():
            logger.info(f"MoEMetricsCallback first metrics: L={num_layers}, {self._pending}")
        self._first_log = False

        for c in counters:
            if c is not None:
                c.zero_()
