"""``EPMoELayerBase``, the EP MoE layer every family wrapper subclasses.

The base owns construction, the dispatch -> expert compute -> combine forward and the expert-parameter
views; compute, routing, grad sync, expert LoRA, the checkpoint gather and the balancing bias are mixins.
"""

from __future__ import annotations

import contextlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

import torch
import torch.distributed as dist
import torch.nn as nn

from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.expert_parallel.deepep.dispatcher import DeepEPDispatcher
from halo.distributed.expert_parallel.moe_layer.balancing_bias import EPBalancingBiasMixin
from halo.distributed.expert_parallel.moe_layer.compute import EPExpertComputeMixin
from halo.distributed.expert_parallel.moe_layer.expert_lora import EPExpertLoraMixin
from halo.distributed.expert_parallel.moe_layer.gather import EPGatherMixin
from halo.distributed.expert_parallel.moe_layer.grad_sync import EPGradSyncMixin
from halo.distributed.expert_parallel.moe_layer.routing import EPRoutingMixin
from halo.distributed.grad_reduce import SumGradAcrossGroup
from halo.env import env_flag
from halo.hardware import current_device
from halo.models.moe.family_specs import MoEFamilySpec
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# Opt-in overlap of the shared-expert FFN with the latency-bound dispatch all-to-all; one stream per device.
_SHARED_OVERLAP_ENABLED = env_flag("HALO_EP_SHARED_OVERLAP")
_shared_overlap_streams: dict[int, torch.cuda.Stream] = {}


class ReduceFromExpertTP(torch.autograd.Function):
    """All-reduce SUM forward, identity backward, over the combined MoE output in token space.

    Exact because the combine is linear; the backward is identity because :class:`SumGradAcrossGroup`
    all-reduces the input grad before the dispatch.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group):
        out = x.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class EPMoELayerBase(
    EPExpertComputeMixin,
    EPRoutingMixin,
    EPGradSyncMixin,
    EPExpertLoraMixin,
    EPGatherMixin,
    EPBalancingBiasMixin,
    nn.Module,
    ABC,
):
    """Base class for EP MoE layers on DeepEP.

    A family subclasses it, fills in the declarations below, implements ``forward`` and overrides the
    ``_adopt_*`` construction hooks and the compute kernels its expert layout needs.
    """

    # The family's facts; every concrete family layer declares its own.
    FAMILY: ClassVar[MoEFamilySpec]

    # HF class name(s) of the MoE block this wrapper replaces; ``roster.MOE_LAYER_MAP`` claims them.
    HF_MODULE_NAMES: tuple[str, ...] = ()

    # Per-family dotted paths on the original HF MoE module, probed by :meth:`detect_num_experts` first.
    _NUM_EXPERTS_ATTR_PATHS: tuple[str, ...] = ()

    # Must match the family's checkpoint name — it becomes the export key in ``replicated_named_params``.
    # Resolved per instance from ``FAMILY.shared_expert_attrs`` for a family that declares one.
    _shared_expert_attr: str = "shared_experts"

    # True where the architecture always builds the shared expert: an absent or ``None`` slot is then an
    # upstream rename rather than a configuration, and adopting nothing would drop it from every output.
    _SHARED_EXPERT_REQUIRED: bool = False

    # Activation name :meth:`_resolve_activation` falls back to when neither the expert container nor the
    # wrapped block names one — the family's architectural default.
    _DEFAULT_ACTIVATION: str = "silu"

    # Attribute the router hangs off, on the wrapped HF block (:meth:`_find_gate_or_router` reads it
    # there) and on this wrapper, which re-registers it under the same name. Declared once per family so
    # the probe, the fp32 upcast and the grad-sync hook cannot point at different modules. ``None`` = the
    # router is external to the wrapper (Gemma4's sibling module): nothing is adopted here, no routing
    # knob is read, and only expert grad-sync hooks register.
    _ROUTER_ATTR: str | None = "gate"

    def __init__(self, original_layer: nn.Module, ep_config: EPConfig, weights_already_sharded: bool = False):
        """Wrap ``original_layer``'s MoE compute for expert parallelism.

        Every family runs this order and overrides a step's hook, never the sequence: router and expert
        container, EP state, router adoption, routing knobs, shared expert, compute constants, this
        rank's expert weights, the shared tail, the construction line. ``weights_already_sharded``: the
        lazy loader already sliced the experts to this rank's range.
        """
        router = self._find_gate_or_router(original_layer)
        experts = self._find_experts_container(original_layer)
        self._init_ep_state(ep_config, self._detect_hidden_dim(router, experts))
        if router is not None:
            setattr(self, self._ROUTER_ATTR, router)
        self._adopt_routing(original_layer)
        self._adopt_shared_experts(original_layer)
        self._adopt_expert_compute(original_layer, experts)
        self._adopt_expert_params(experts, weights_already_sharded=weights_already_sharded)
        self._finalize_expert_init()
        self._log_init_summary(*self._init_summary_extras(original_layer))

    @classmethod
    def detect_num_experts(cls, layer: nn.Module) -> int:
        """The routed-expert count of an HF MoE block: the first dotted :attr:`_NUM_EXPERTS_ATTR_PATHS` entry,
        then the expert container, holding an ``int`` or a sized value."""
        generic = tuple(p for attr in cls.FAMILY.experts_container_attrs for p in (attr, f"{attr}.num_experts"))
        for path in (*cls._NUM_EXPERTS_ATTR_PATHS, *generic):
            value = layer
            for attr in path.split("."):
                value = getattr(value, attr, None)
            if isinstance(value, int):
                return value
            if hasattr(value, "__len__"):
                return len(value)
        raise AttributeError(f"Cannot detect num_experts from {type(layer).__name__}")

    def expert_named_params(self) -> list:
        """``(name, param)`` for this rank's expert weights, then its expert LoRA adapters.

        Read by grad sync, the fp32 upcast and the checkpoint skip-set. A family's layout is declared in
        ``FAMILY.expert_weight_roots``, not overridden here.
        """
        return self._base_expert_named_params() + self.expert_lora_named_params()

    def replicated_named_params(self) -> list:
        """``(name, param)`` for the fully replicated submodules the router hook DP-averages: the shared
        expert under :attr:`_shared_expert_attr`, empty when that slot is ``None``. Qwen3.5/3.6 overrides it
        for its second gate module.
        """
        return self._submodule_named_params(
            [(self._shared_expert_attr, getattr(self, self._shared_expert_attr, None))]
        )

    @abstractmethod
    def forward(self, hidden_states: torch.Tensor, **kwargs): ...

    @classmethod
    def _find_experts_container(cls, layer: nn.Module) -> nn.Module:
        """The block's expert container, under the first of ``FAMILY.experts_container_attrs`` it carries."""
        for attr in cls.FAMILY.experts_container_attrs:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        raise AttributeError(
            f"{cls.__name__} declares FAMILY.experts_container_attrs={cls.FAMILY.experts_container_attrs}, "
            f"but {type(layer).__name__} carries none of them — the family renamed its expert container "
            f"upstream. Update the family spec."
        )

    @classmethod
    def _detect_hidden_dim(cls, router: nn.Module | None, experts: nn.Module) -> int:
        """The router projection's input width, else (an external or projection-less router) the fused
        experts' ``gate_up_proj [E, 2M, H]``."""
        if router is not None and hasattr(router, "weight"):
            return router.weight.shape[1]
        return cls._require_fused_experts(experts).gate_up_proj.shape[2]

    def _init_ep_state(self, ep_config: EPConfig, hidden_dim: int):
        """Build the EP state every wrapper shares; callable without an HF block, as construction probes
        do."""
        super().__init__()
        self.ep_config = ep_config
        self.num_experts = ep_config.num_experts
        self.experts_per_rank = ep_config.experts_per_rank
        self.ep_rank = ep_config.ep_rank
        self.ep_size = ep_config.ep_size
        self.expert_start = ep_config.expert_start_idx
        self.expert_end = ep_config.expert_end_idx
        self.hidden_dim = hidden_dim

        self.expert_tp_size = ep_config.expert_tp_size
        self.expert_tp_rank = ep_config.expert_tp_rank
        self.expert_tp_group = ep_config.expert_tp_group
        if self.expert_tp_size > 1 and self.expert_tp_group is None:
            raise RuntimeError(
                f"expert_tensor_parallel_size={self.expert_tp_size} requires expert_tp_group to be created in "
                f"EPConfig process group setup. Got expert_tp_group=None."
            )

        self.fp32_router = ep_config.fp32_router
        self.fp32_experts = ep_config.fp32_experts

        self.dispatcher = DeepEPDispatcher(ep_config, self.num_experts, hidden_dim)
        self._grad_sync_hooks = []
        self._hook_synced_expert_params: list = []  # populated only in the in-backward hook regime

        self._use_grouped_mm = ep_config.grouped_gemm_enabled

        # Latched on the first forward: at init the eager loader may still hold this layer on CPU.
        self._activation_warmed = False

        # Not cacheable once experts are FSDP2-managed: the unsharded param's version counter is pinned,
        # so the per-step quant cache would serve step-0 weights forever.
        self._lowp_quant_format = None
        self._lowp_weight_cacheable = True

        # Empty = no expert LoRA, so the hot path is one set-membership check.
        self._expert_lora_attrs: frozenset[str] = frozenset()
        self.expert_lora_scaling: float = 1.0
        self.expert_lora_dropout: nn.Module = nn.Identity()
        # Cleared by disable_expert_adapters: the adapters are not peft-managed.
        self._expert_adapters_enabled: bool = True

        # Routing replay (plain attrs, FSDP2-ignored, not exported): ``[T, top_k]`` global ids, -1 = natural.
        self._forced_topk_indices: torch.Tensor | None = None
        # The cursor wraps over a mask spanning chunked forwards; the total tells "never consumed" from
        # k full passes.
        self._forced_cursor: int = 0
        self._forced_consumed_total: int = 0
        self._capture_routing: bool = False
        self._captured_routing_chunks: list[torch.Tensor] = []
        # [flipped, forced] token counts, device-side; the reader syncs once per logging step.
        self._replay_flip_counts: torch.Tensor | None = None

        # Labels the dispatch / expert-compute / combine spans at no device cost; benchmark_sft_ep.py
        # --comm_profile swaps in EPSpanTimer.time_operation, which syncs per span to time it.
        self._perf: Callable[[str], contextlib.AbstractContextManager] = torch.profiler.record_function

    def _adopt_routing(self, original_layer: nn.Module) -> None:
        """Read the routing knobs the forward consumes; default ``top_k``. An override calls ``super()``
        first and may read the adopted router."""
        self.top_k = self._find_top_k(original_layer)

    def _adopt_shared_experts(self, original_layer: nn.Module) -> None:
        """Adopt the shared expert under the first ``FAMILY.shared_expert_attrs`` spelling the block carries,
        which becomes its export key."""
        for attr in self.FAMILY.shared_expert_attrs:
            if not hasattr(original_layer, attr):
                continue
            shared = getattr(original_layer, attr)
            if shared is None and self._SHARED_EXPERT_REQUIRED:
                continue  # a later declared spelling may still carry the module; else the raise below
            self._shared_expert_attr = attr
            setattr(self, attr, shared)
            return
        if self._SHARED_EXPERT_REQUIRED:
            raise AttributeError(
                f"{type(self).__name__} declares FAMILY.shared_expert_attrs={self.FAMILY.shared_expert_attrs} as "
                f"always built, but {type(original_layer).__name__} carries no such module — the "
                f"family renamed it upstream. A tolerant fallback would silently drop the shared "
                f"expert from every output."
            )
        if self.FAMILY.shared_expert_attrs:
            # Declared but absent (LFM-2 ships none): register the slot so the shared leg reads None.
            self._shared_expert_attr = self.FAMILY.shared_expert_attrs[0]
            setattr(self, self._shared_expert_attr, None)

    def _adopt_expert_compute(self, original_layer: nn.Module, experts: nn.Module) -> None:
        """Adopt the constants the expert kernels read; default the activation. A family reads its extra
        constants here directly, so an upstream rename raises instead of substituting a default."""
        self._resolve_activation(experts, original_layer=original_layer, default=self._DEFAULT_ACTIVATION)

    def _adopt_expert_params(self, experts: nn.Module, weights_already_sharded: bool = False) -> None:
        """Register this rank's expert weights; default the fused-GLU layout."""
        self._init_fused_glu_params(experts, weights_already_sharded=weights_already_sharded)

    def _finalize_expert_init(self) -> None:
        """The shared init tail: device, expert LoRA, fp32 upcasts, grad-sync hooks. The upcast keeps the
        ``requires_grad`` the LoRA freeze set and rebinds each tensor, so the hooks register after it."""
        self.to(self._resident_device())
        self._init_expert_lora()
        self._upcast_router_to_fp32()
        self._upcast_experts_to_fp32()
        self._install_gradient_sync_hooks()

    def _resident_device(self) -> torch.device:
        """Where this layer's state belongs: its parameters' device, so a layer materialized on CPU stays for
        the trainer to move, else this rank's device."""
        device = next((p.device for p in self.parameters()), None)
        return current_device() if device is None or device.type == "meta" else device

    def _init_summary_extras(self, original_layer: nn.Module) -> tuple[str, ...]:
        """This family's extra fields for the construction line."""
        return ()

    def _log_init_summary(self, *extras: str) -> None:
        """Log the construction line: rank-uniform fields at INFO on the main process only (thousands of
        identical lines at scale otherwise), this rank's expert range at DEBUG."""
        logger.debug(
            f"{type(self).__name__}: EP rank {self.ep_rank} owns experts "
            f"{self.expert_start}-{self.expert_end - 1} of {self.num_experts}"
        )
        if not is_global_main_process():
            return
        fields = [f"{self.expert_end - self.expert_start}/{self.num_experts} experts per EP rank"]
        if self.FAMILY.shared_expert_attrs:
            shared = getattr(self, self._shared_expert_attr, None) is not None
            fields.append(f"{self._shared_expert_attr}={'yes' if shared else 'no'}")
        fields += [
            *extras,
            f"grouped_mm={self._grouped_mm_enabled()}",
            f"glu_combine={self._glu_combine_name()}",
            f"fp32_router={self.fp32_router}",
            f"fp32_experts={self.fp32_experts}",
        ]
        logger.info(f"{type(self).__name__}: {', '.join(fields)}")

    def _base_expert_named_params(self) -> list:
        """``(name, param)`` for the ``FAMILY.expert_weight_roots`` this rank holds."""
        return [(n, getattr(self, n)) for n in self.FAMILY.expert_weight_roots if hasattr(self, n)]

    @staticmethod
    def _submodule_named_params(named_modules) -> list:
        """Flatten ``(prefix, submodule)`` pairs into ``(prefix.name, param)``, skipping ``None``
        submodules."""
        out = []
        for prefix, mod in named_modules:
            if mod is not None:
                out += [(f"{prefix}.{n}", p) for n, p in mod.named_parameters()]
        return out

    def _dispatch_compute_combine_shared(
        self,
        flat: torch.Tensor,
        experts: torch.Tensor,
        weights: torch.Tensor,
        input_dtype: torch.dtype,
        orig_shape: torch.Size,
        shared_fn: Callable[[], torch.Tensor] | None,
    ) -> torch.Tensor:
        """The routed output reshaped to ``orig_shape`` plus ``shared_fn()``, which reads only the layer
        input and so runs beside the dispatch on a side stream under ``HALO_EP_SHARED_OVERLAP``."""
        if shared_fn is None:
            return self._dispatch_compute_combine(flat, experts, weights, input_dtype).view(*orig_shape)

        if _SHARED_OVERLAP_ENABLED and flat.is_cuda:
            idx = flat.device.index if flat.device.index is not None else torch.cuda.current_device()
            if idx not in _shared_overlap_streams:
                _shared_overlap_streams[idx] = torch.cuda.Stream(device=idx)
            side = _shared_overlap_streams[idx]
            current = torch.cuda.current_stream()
            side.wait_stream(current)
            with torch.cuda.stream(side):
                shared = shared_fn()
            routed = self._dispatch_compute_combine(flat, experts, weights, input_dtype).view(*orig_shape)
            current.wait_stream(side)
            shared.record_stream(current)  # keep the side-stream tensor alive for the add
            return routed + shared

        routed = self._dispatch_compute_combine(flat, experts, weights, input_dtype).view(*orig_shape)
        return routed + shared_fn()

    def _dispatch_compute_combine(
        self, flat: torch.Tensor, experts: torch.Tensor, weights: torch.Tensor, input_dtype: torch.dtype
    ) -> torch.Tensor:
        """Dispatch -> expert compute -> combine, with the expert-TP sum in token space outside that span:
        inside it, the strided ETP all-reduce couples DeepEP dispatch groups and times out their barrier."""
        # Before the dispatch, not inside the span: a first-call trace between dispatch and combine
        # stalls every peer in DeepEP's barrier.
        self._warm_activation_graphs(flat.device, input_dtype, flat.shape[-1], experts.shape[1])

        if self._capture_routing:
            # Capture the global expert ids before dispatch localizes them; int16 (num_experts < 32768).
            self._captured_routing_chunks.append(experts.detach().to(torch.int16))

        if self.expert_tp_size > 1:
            # Both operands need the ETP backward sum; without either, the router and everything
            # upstream of it trains on a 1/expert_tp_size gradient.
            flat = SumGradAcrossGroup.apply(flat, self.expert_tp_group)
            weights = SumGradAcrossGroup.apply(weights, self.expert_tp_group)

        flat = self._rank_uniform_dispatch_input(flat, weights)

        with self._perf("ep.dispatch"):
            recv_x, recv_topk_idx, recv_topk_weights, handle = self.dispatcher.dispatch(
                flat, experts, weights, owner=self
            )

        with self._perf("ep.expert_compute"):
            output = self._compute_experts(recv_x, recv_topk_idx, recv_topk_weights, input_dtype)

        with self._perf("ep.combine"):
            combined = self.dispatcher.combine(output, recv_topk_weights, handle, owner=self)

        if self.expert_tp_size > 1:
            combined = ReduceFromExpertTP.apply(combined, self.expert_tp_group)

        return combined

    def _rank_uniform_dispatch_input(self, flat: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """``flat`` as a grad-requiring leaf when the experts train but no dispatch operand needs grad, so
        every rank builds the same dispatch/combine backward nodes; ``flat`` itself otherwise."""
        # Without the leaf a rank whose experts got no token joins the graph only through a constant,
        # skips DeepEP collectives its peers enter, and hangs or mispairs them; its grad is discarded.
        if flat.requires_grad or weights.requires_grad:
            return flat
        if not (self.training and torch.is_grad_enabled()) or self.ep_size <= 1:
            return flat
        if not any(param.requires_grad for _, param in self.expert_named_params()):
            return flat
        return flat.detach().requires_grad_()


def named_ep_layers(model: nn.Module) -> list[tuple[str, EPMoELayerBase]]:
    """This model's EP MoE layers as ``(module path, layer)``, in ``named_modules`` order.

    By ``isinstance``, not an attribute probe: a PEFT ``modules_to_save`` wrapper forwards ``__getattr__`` to
    the layer it holds, so a probe would match the wrapper too.
    """
    return [(name, module) for name, module in model.named_modules() if isinstance(module, EPMoELayerBase)]
