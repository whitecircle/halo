"""Routing for EP MoE layers: router probe and input dtype, biased top-k, gate weights, routing-replay override."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from halo.models.gradient_checkpointing import active_recompute_scope
from halo.models.moe.config_fields import ROUTER_TOPK_FIELDS
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


class EPRoutingMixin:
    """Routing, mixed into :class:`EPMoELayerBase`, whose construction builds the state annotated here."""

    fp32_router: bool
    top_k: int
    gate: nn.Module
    _ROUTER_ATTR: str | None
    _native_balancing_attr: str | None
    _forced_topk_indices: torch.Tensor | None
    _forced_cursor: int
    _forced_consumed_total: int
    _replay_flip_counts: torch.Tensor | None
    # The family router's gate weights at given top-k indices, which routing replay re-derives live.
    _gate_weights_at: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    # The floor the family's router adds to a top-k weight sum before renormalizing by it: DeepSeek-V3's own
    # 1e-20, 0.0 where the router adds none.
    _TOPK_WEIGHT_NORM_EPS: float = 1e-20

    @classmethod
    def _declared_router(cls, layer: nn.Module) -> nn.Module | None:
        """The block's router under the declared :attr:`_ROUTER_ATTR`, None for an external router; never a
        name probe, which would pick whichever of two present names it tried first."""
        if cls._ROUTER_ATTR is None:
            return None
        router = getattr(layer, cls._ROUTER_ATTR, None)
        if router is None:
            raise AttributeError(
                f"{cls.__name__} declares _ROUTER_ATTR='{cls._ROUTER_ATTR}', but "
                f"{type(layer).__name__} has no such attribute — the family renamed its router "
                f"upstream. Update _ROUTER_ATTR."
            )
        return router

    @classmethod
    def _find_top_k(cls, layer: nn.Module) -> int:
        """Top-k off the block, then its declared router, in :data:`ROUTER_TOPK_FIELDS` order, which puts
        bare ``top_k`` (also the sampling parameter) last."""
        router = None if cls._ROUTER_ATTR is None else getattr(layer, cls._ROUTER_ATTR, None)
        for source in (layer, router):
            for attr in ROUTER_TOPK_FIELDS:
                if source is not None and hasattr(source, attr):
                    return getattr(source, attr)
        raise AttributeError(
            f"Cannot determine top_k from {type(layer).__name__}: none of {ROUTER_TOPK_FIELDS} is on it or its router "
            f"{cls._ROUTER_ATTR!r}. Add the family's spelling to ROUTER_TOPK_FIELDS."
        )

    def _live_router_module(self) -> nn.Module | None:
        """Resolved by name now, so a PEFT ``modules_to_save`` copy is the one returned; None if external."""
        return getattr(self, self._ROUTER_ATTR) if self._ROUTER_ATTR else None

    @property
    def _fp32_router_input(self) -> bool:
        """Whether the router input is upcast to match a live fp32 router weight. Read off the weight, since an FSDP2
        group unshards the router at its bf16 ``param_dtype``, but nothing wraps a one-rank job's model."""
        if not self.fp32_router:
            return False
        router = self._live_router_module()
        weight = next(router.parameters(), None) if router is not None else None
        return weight is not None and weight.dtype == torch.float32

    def _router_input(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.float() if self._fp32_router_input else hidden

    def _linear_router_logits(self, flat: torch.Tensor) -> torch.Tensor:
        """The router's logits as a bare projection, autocast off so the matmul runs at the input dtype, for a router
        whose own forward also selects."""
        with torch.amp.autocast("cuda", enabled=False):
            return F.linear(self._router_input(flat), self.gate.weight)

    def _upcast_router_to_fp32(self) -> None:
        if not self.fp32_router:
            return
        if self._ROUTER_ATTR is None:
            logger.warning(
                f"{type(self).__name__}: fp32_router has NO effect — this family's router is a "
                f"sibling module outside the EP layer (FSDP-managed). Its dtype cannot be raised "
                f"under EP: fp32_non_ep_params would upcast it, and DeepEP's combine is bf16-only. "
                f"The router stays bf16."
            )
            return
        setattr(self, self._ROUTER_ATTR, getattr(self, self._ROUTER_ATTR).float())
        if is_global_main_process():
            logger.info(f"{type(self).__name__}: FP32 router (compute stays BF16)")

    def _biased_route(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(indices, weights)``: :meth:`_biased_topk`, gated by the family router's own :meth:`_gate_weights_at`."""
        indices = self._biased_topk(router_logits)
        return indices, self._gate_weights_at(router_logits, indices)

    def _softmax_gate_weights_at(
        self, router_logits: torch.Tensor, indices: torch.Tensor, *, normalize: bool
    ) -> torch.Tensor:
        """Softmax-over-all-experts gate weights at ``indices``, renormalized when ``normalize`` (each family
        passes its router's own rule, so a rename raises instead of flipping the renorm)."""
        probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        weights = probs.gather(-1, indices)
        if normalize:
            weights = self._renormalize_topk_weights(weights)
        return weights.to(router_logits.dtype)

    def _renormalize_topk_weights(self, weights: torch.Tensor) -> torch.Tensor:
        return weights / (weights.sum(dim=-1, keepdim=True) + self._TOPK_WEIGHT_NORM_EPS)

    def _replay_router_choice(
        self, router_logits: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The router's own ``(indices, weights)``, or with routing replay armed the forced picks gated by
        :meth:`_gate_weights_at` on the live logits, which keeps the router's gradient live."""
        if self._forced_topk_indices is None:
            return indices, weights
        indices = self._apply_routing_replay(indices.long())
        return indices, self._gate_weights_at(router_logits, indices)

    def _select_topk_on_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Top-k indices on :meth:`_selection_scores`, the biased copy of ``scores``; the replay is the caller's."""
        return torch.topk(self._selection_scores(scores), self.top_k, dim=-1, sorted=False)[1]

    def _biased_topk(self, logits: torch.Tensor) -> torch.Tensor:
        """Top-k indices on the softmax probs plus the transient balancing bias: selection only, so a zero
        bias reproduces the unbalanced route."""
        bias = self._transient_balancing_bias(logits)
        if bias is None:
            # Native-adoption mode (GptOss ``router.bias``): the bias is inside the logits, so select on raw logits as
            # the served router does. A large logit gap underflows the losing softmax probs to exact 0.0, and top-k
            # then breaks the tie differently than the logit order.
            _, indices = torch.topk(logits.detach(), self.top_k, dim=-1)
        else:
            probs = F.softmax(logits.float(), dim=-1)
            _, indices = torch.topk(probs.detach() + bias, self.top_k, dim=-1)
        return self._apply_routing_replay(indices)

    def _selection_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """``scores`` plus the declared native correction slot, then the transient balancing bias, as a
        served copy applies them; ``scores`` itself when neither applies. Gate weights stay unbiased."""
        target = self._native_balancing_target()
        if target is None and self._native_balancing_attr and not self._native_slot_absence_is_legal():
            raise AttributeError(
                f"{type(self).__name__} declares the balancing slot "
                f"'{self._native_balancing_attr}' as the bias its own selection applies, but this "
                f"instance carries no such tensor — the family renamed it upstream. Routing on would "
                f"silently drop the pretrained correction bias. Fix the declaration, or override "
                f"_native_slot_absence_is_legal() to answer True for the configuration that omits it."
            )
        if target is not None:
            scores = scores + getattr(*target)
        bias = self._transient_balancing_bias(scores)
        if bias is not None:
            scores = scores + bias
        return scores

    def _apply_routing_replay(self, topk_indices: torch.Tensor) -> torch.Tensor:
        """``topk_indices`` with the armed routing-replay mask's rows swapped in (-1 keeps a token's natural
        pick), counting the flip rate outside the GC recompute. Callers gather gate weights afterwards."""
        forced_full = self._forced_topk_indices
        if forced_full is None:
            return topk_indices
        tokens = topk_indices.size(0)
        scope = active_recompute_scope()
        slot = scope.slot(self, "routing") if scope is not None else {}
        if scope is not None and scope.is_recompute:
            # The scope raises if this frame's original forward never made the matching call.
            forced = slot["forced"]
        else:
            if self._forced_cursor + tokens > forced_full.size(0):
                raise RuntimeError(
                    f"{type(self).__name__}: routing-replay mask exhausted — forward consuming tokens "
                    f"[{self._forced_cursor}, {self._forced_cursor + tokens}) of a {forced_full.size(0)}-token "
                    f"mask. The armed mask must tile the microbatch's chunked forwards exactly."
                )
            forced = forced_full[self._forced_cursor : self._forced_cursor + tokens]
            self._forced_cursor += tokens
            self._forced_consumed_total += tokens
            if self._forced_cursor == forced_full.size(0):
                self._forced_cursor = 0
            slot["forced"] = forced
        if forced.shape != topk_indices.shape:
            raise RuntimeError(
                f"{type(self).__name__}: routing-replay slice shape {tuple(forced.shape)} does not match "
                f"this forward's selection {tuple(topk_indices.shape)}."
            )
        valid = forced[:, :1] >= 0  # [T, 1]; a token is either fully forced or fully natural
        if scope is None or not scope.is_recompute:
            with torch.no_grad():
                flips = (forced.sort(dim=-1).values != topk_indices.sort(dim=-1).values).any(dim=-1)
                counts = torch.stack([(flips & valid.squeeze(1)).sum(), valid.sum()]).float()
                if self._replay_flip_counts is None:
                    self._replay_flip_counts = counts
                else:
                    self._replay_flip_counts += counts
        return torch.where(valid, forced, topk_indices)
