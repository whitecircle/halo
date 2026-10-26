"""``EPMoELayerBase``, the EP MoE layer every family wrapper subclasses.

The base owns construction, the dispatch -> expert compute -> combine forward and the expert-parameter
views; compute, routing, grad sync, expert LoRA, the checkpoint gather and the balancing bias are mixins.
"""

from __future__ import annotations

import contextlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn as nn

from src.diagnostics.ep_span_timing import get_ep_span_timer
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.deepep.dispatcher import (
    DeepEPDispatcher,
    ReplayCombineFunction,
    ReplayDispatchFunction,
)
from src.distributed.expert_parallel.expert_gather import EPExpertGatherMixin
from src.distributed.expert_parallel.moe_layer.balancing_bias import EPBalancingBiasMixin
from src.distributed.expert_parallel.moe_layer.compute import EPExpertComputeMixin
from src.distributed.expert_parallel.moe_layer.expert_lora import EPExpertLoraMixin
from src.distributed.expert_parallel.moe_layer.grad_sync import EPGradSyncMixin
from src.distributed.expert_parallel.moe_layer.routing import EPRoutingMixin
from src.distributed.grad_reduce import SumGradAcrossGroup
from src.env import env_flag
from src.hardware import current_device, supports_grouped_mm
from src.models.gradient_checkpointing import active_recompute_scope, in_backward_pass
from src.models.moe_balancing import (
    NativeBalancingSlot,
    native_balancing_bias_attrs,
    register_ep_wrapped_model_types,
    register_legacy_per_layer_config_keys,
    register_native_balancing_slot,
    register_source_config_schema,
)
from src.runtime.ranks import is_global_main_process

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
    EPExpertGatherMixin,
    EPBalancingBiasMixin,
    nn.Module,
    ABC,
):
    """Base class for EP MoE layers on DeepEP.

    A family subclasses it, fills in the declarations below, implements ``forward`` and overrides the
    ``_adopt_*`` construction hooks and the compute kernels its expert layout needs.
    """

    # False where routing replay is unimplementable: Gemma4 (external router) and Zaya (cross-layer state).
    _supports_routing_replay: bool = True

    # HF class name(s) of the MoE block this wrapper replaces; ``patching.build_moe_layer_map`` walks subclasses.
    HF_MODULE_NAMES: tuple[str, ...] = ()

    # ``config.model_type`` spelling(s) served — the family's own first (``docker/vllm/parity/generate.py``
    # reads it), then the composite wrappers and sibling spellings that carry it; unioned so a
    # checkpoint's config.json resolves back off-line.
    HF_MODEL_TYPES: tuple[str, ...] = ()

    # Every expert-weight attribute this family may hold across all config branches (fused vs separate/ETP).
    _EXPERT_WEIGHT_ATTR_ROOTS: tuple[str, ...] = ("gate_up_proj", "gate_proj", "up_proj", "down_proj")

    # Per-family dotted paths on the original HF MoE module, probed by :meth:`detect_num_experts` first.
    _NUM_EXPERTS_ATTR_PATHS: tuple[str, ...] = ()

    # ``(live module, hub)`` pairs for renames transformers applies only inside ``from_pretrained``. The
    # gather rewrites module → hub (the engines ignore unknown names); the lazy loader inverts it.
    _EXPORT_KEY_RENAMES: tuple[tuple[str, str], ...] = ()

    # ``{flat legacy key: (layer_type, per-layer field)}`` for attention geometry the hub config spells
    # one flat key per layer type. transformers folds those into ``per_layer_config`` on load and
    # re-serializes only the folded form, which the pinned rollout server cannot parse, so exported
    # ``config.json`` files are rewritten back to the flat keys (``export_legacy_per_layer_config``).
    # Registered per claimed ``model_type`` into the ``moe_balancing`` leaf by ``__init_subclass__``.
    _LEGACY_PER_LAYER_CONFIG_KEYS: dict[str, tuple[str, str]] = {}

    # Attribute(s) an HF MoE block may hold its expert container under, tried in order. Shared by every
    # probe (container, expert count, the lazy loader's checkpoint-key regexes). A family whose block
    # spells it differently across revisions declares every spelling it accepts; an unknown spelling
    # raises rather than resolving off a name no family claims.
    _EXPERTS_CONTAINER_ATTRS: tuple[str, ...] = ("experts",)

    # Must match the family's checkpoint name — it becomes the export key in ``replicated_named_params``.
    # Resolved per instance from :attr:`_SHARED_EXPERT_ATTRS` for a family that declares one.
    _shared_expert_attr: str = "shared_experts"

    # Attribute(s) the HF block may hold its shared expert under, tried in order; the first spelling the
    # block carries becomes :attr:`_shared_expert_attr` and thus the export key. Empty = the family has
    # no shared expert. A declaring family whose block carries none (LFM-2) registers the slot as
    # ``None``, so the shared leg is skipped rather than raising on the family's own name.
    _SHARED_EXPERT_ATTRS: tuple[str, ...] = ()

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

    # Reduction axis of those tensors as this family writes them: -1 = ``F.linear`` ``[E, N, K]``, GptOss
    # keeps matmul ``[E, K, N]``. The lowp export block-scales along it; the wrong axis corrupts every expert.
    HF_FUSED_EXPERT_CONTRACTION_AXIS: int = -1

    # False where the lazy loader cannot express this family's on-disk expert layout → from_pretrained + patch.
    _supports_lazy_loading: bool = True

    # transformers' conversion-mapping key(s) for this hub checkpoint (``hub_conversion.py``); empty = canonical.
    _HUB_CONVERSION_KEYS: tuple[str, ...] = ()

    # True where transformers' serialization of this config is a schema no pinned serving engine reads:
    # the engines parse the family only through the source repo's ``auto_map`` modules, whose vendor
    # spellings transformers absorbs at load (``attribute_map`` plus ``__post_init__`` kwargs) and never
    # re-emits. Config writes forward the source ``config.json`` and modules instead
    # (``export_source_config_schema``); registered per claimed ``model_type`` by ``__init_subclass__``.
    _EXPORTS_SOURCE_CONFIG_SCHEMA: bool = False

    # False makes ``validate_weight_sync_support`` reject online/env GRPO on every engine: no
    # end-to-end sync into the family's served model has been validated. What one pinned engine
    # cannot serve is the engine client's ``UNSERVABLE_MODEL_TYPES``.
    _supports_weight_sync: bool = True

    # Surfaced verbatim in the refusal above; a family overrides it with its own gap.
    _WEIGHT_SYNC_REFUSAL_REASON: str = (
        "no end-to-end weight sync into this family's served model has been validated on either pinned rollout engine"
    )

    # False where GC recompute on top of EP is architecturally broken (Zaya's cross-layer EDA/CCA state).
    _supports_gradient_checkpointing: bool = True

    # False where fp32 master weights on non-expert params break this family's dispatch; the loader refuses
    # ``fp32_non_ep_params`` rather than letting DeepEP's C++ assert fire after the whole multi-GPU load.
    _supports_fp32_non_ep_params: bool = True

    def __init_subclass__(cls, **kwargs):
        """Register this family's config rewrites and native balancing slot under every ``model_type``
        it claims.

        They land in the ``moe_balancing`` leaf so the offline consumers (config writers, standalone
        conversion tools) read them without importing this package, and a wrapper-less run (no EP, no
        grouped GEMM) exports the same config as a wrapped one.
        """
        super().__init_subclass__(**kwargs)
        if cls._LEGACY_PER_LAYER_CONFIG_KEYS:
            register_legacy_per_layer_config_keys(cls.HF_MODEL_TYPES, cls._LEGACY_PER_LAYER_CONFIG_KEYS)
        if cls._EXPORTS_SOURCE_CONFIG_SCHEMA:
            register_source_config_schema(cls.HF_MODEL_TYPES)
        if attrs := native_balancing_bias_attrs(cls):
            register_native_balancing_slot(
                cls.HF_MODEL_TYPES, NativeBalancingSlot(attrs, cls._NATIVE_BALANCING_CONFIG_FLAG)
            )
        register_ep_wrapped_model_types(cls.HF_MODEL_TYPES)

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
        generic = tuple(p for attr in cls._EXPERTS_CONTAINER_ATTRS for p in (attr, f"{attr}.num_experts"))
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
        :attr:`_EXPERT_WEIGHT_ATTR_ROOTS`, not overridden here.
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
        """The block's expert container, under the first declared :attr:`_EXPERTS_CONTAINER_ATTRS` it
        carries."""
        for attr in cls._EXPERTS_CONTAINER_ATTRS:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        raise AttributeError(
            f"{cls.__name__} declares _EXPERTS_CONTAINER_ATTRS={cls._EXPERTS_CONTAINER_ATTRS}, but "
            f"{type(layer).__name__} carries none of them — the family renamed its expert container "
            f"upstream. Update _EXPERTS_CONTAINER_ATTRS."
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

        self._use_grouped_mm = ep_config.use_grouped_gemm and supports_grouped_mm()

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

        # HALO_EP_PERF_PROFILE=1 → CUDA-event phase timers (per-phase syncs, diagnostic runs only); off → label.
        self._perf: Callable[[str], contextlib.AbstractContextManager]
        if env_flag("HALO_EP_PERF_PROFILE"):
            self._perf = get_ep_span_timer().time_operation
        else:
            self._perf = torch.profiler.record_function

    def _adopt_routing(self, original_layer: nn.Module) -> None:
        """Read the routing knobs the forward consumes; default ``top_k``. An override calls ``super()``
        first and may read the adopted router."""
        self.top_k = self._find_top_k(original_layer)

    def _adopt_shared_experts(self, original_layer: nn.Module) -> None:
        """Adopt the shared expert under the first declared :attr:`_SHARED_EXPERT_ATTRS` spelling the block
        carries, which becomes its export key."""
        for attr in self._SHARED_EXPERT_ATTRS:
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
                f"{type(self).__name__} declares _SHARED_EXPERT_ATTRS={self._SHARED_EXPERT_ATTRS} as "
                f"always built, but {type(original_layer).__name__} carries no such module — the "
                f"family renamed it upstream. A tolerant fallback would silently drop the shared "
                f"expert from every output."
            )
        if self._SHARED_EXPERT_ATTRS:
            # Declared but absent (LFM-2 ships none): register the slot so the shared leg reads None.
            self._shared_expert_attr = self._SHARED_EXPERT_ATTRS[0]
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
        self._setup_gradient_sync()

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
        if self._SHARED_EXPERT_ATTRS:
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
        """``(name, param)`` for the declared :attr:`_EXPERT_WEIGHT_ATTR_ROOTS` this rank holds."""
        return [(n, getattr(self, n)) for n in self._EXPERT_WEIGHT_ATTR_ROOTS if hasattr(self, n)]

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
            recv_x, recv_topk_idx, recv_topk_weights, handle = self._gc_dispatch(flat, experts, weights)

        with self._perf("ep.expert_compute"):
            output = self._compute_experts(recv_x, recv_topk_idx, recv_topk_weights, input_dtype)

        with self._perf("ep.combine"):
            combined = self._gc_combine(output, recv_topk_weights, handle)

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

    def _gc_dispatch(self, flat, experts, weights):
        """Dispatch, replaying this checkpoint frame's cached result in the recompute: a second DeepEP
        dispatch invalidates the handle the original backward holds and silently corrupts every gradient."""
        if self.dispatcher._noop:  # ep1: no transport, nothing to protect
            return self.dispatcher.dispatch(flat, experts, weights)
        scope = active_recompute_scope()
        if scope is None:
            if self.training and in_backward_pass():
                raise RuntimeError(
                    f"{type(self).__name__}: training forward inside backward with no EP checkpoint "
                    f"scope — gradient checkpointing was re-enabled without install_recompute_scopes "
                    f"(TRL's disable_gradient_checkpointing context manager re-installs a bare "
                    f"checkpoint on exit). A second DeepEP dispatch here corrupts every gradient."
                )
            return self.dispatcher.dispatch(flat, experts, weights)

        slot = scope.slot(self, "dispatch")
        if not scope.is_recompute:
            recv_x, recv_topk_idx, recv_topk_weights, handle = self.dispatcher.dispatch(flat, experts, weights)
            # Detached aliases, not clones: DeepEP's copy epilogue lands every result in an
            # allocator-owned tensor (never a view of the shared arena) and nothing downstream writes
            # into them in place, so a clone would only add a recv-buffer memcpy per layer.
            slot.update(
                recv_x=recv_x.detach(),
                recv_topk_idx=recv_topk_idx.detach() if recv_topk_idx is not None else None,
                recv_topk_weights=recv_topk_weights.detach() if recv_topk_weights is not None else None,
                handle=handle,
            )
            return recv_x, recv_topk_idx, recv_topk_weights, handle

        return ReplayDispatchFunction.apply(
            flat,
            experts,
            weights,
            self.dispatcher,
            slot["recv_x"],
            slot["recv_topk_idx"],
            slot["recv_topk_weights"],
            slot["handle"],
        )

    def _gc_combine(self, output, recv_topk_weights, handle):
        """Combine, replaying this checkpoint frame's cached result in the recompute (the combine writes
        the same shared buffer as :meth:`_gc_dispatch`)."""
        scope = active_recompute_scope()
        if scope is None or self.dispatcher._noop:
            return self.dispatcher.combine(output, recv_topk_weights, handle)

        slot = scope.slot(self, "combine")
        if not scope.is_recompute:
            result = self.dispatcher.combine(output, recv_topk_weights, handle)
            slot["combined"] = result.detach()  # allocator-owned, read-only downstream (see _gc_dispatch)
            return result

        return ReplayCombineFunction.apply(output, recv_topk_weights, handle, self.dispatcher, slot["combined"])


class EPSeparateGluMoELayerBase(EPMoELayerBase):
    """EP MoE layer holding its experts as separate gate/up/down 3-D tensors, exported per expert.

    The family's HF checkpoint stores one tensor per expert (:attr:`_HUB_PER_EXPERT_KEYS`) and the
    wrapper keeps the GLU halves apart — Qwen3 splits the pre-fused pair at init, Bailing stacks a
    per-expert ``nn.ModuleList`` — so the gather and its shard-merge inverse are the separate-halves
    pair for both. Declared once here so a fix cannot land in one family and leave the other's gathered
    save out of step with its sharded merge.
    """

    def gather_expert_state_dict(self, device: str = "cpu", merge_lora: bool = False, retain: bool = True) -> dict:
        """These experts are written per-expert (unfused) in the checkpoint."""
        return self._gather_individual_glu_state_dict(device, merge_lora=merge_lora, retain=retain)

    @classmethod
    def merge_shards_to_hf(cls, prefix: str, params: dict) -> dict:
        """Inverse of the gather above: separate 3D shards → per-expert keys (see the base method)."""
        state = cls._merge_individual_glu_shards(params)
        return {f"{prefix}.{key}": tensor for key, tensor in state.items()}


def named_ep_layers(model: nn.Module) -> list[tuple[str, EPMoELayerBase]]:
    """This model's EP MoE layers as ``(module path, layer)``, in ``named_modules`` order.

    By ``isinstance``, not an attribute probe: a PEFT ``modules_to_save`` wrapper forwards ``__getattr__`` to
    the layer it holds, so a probe would match the wrapper too.
    """
    return [(name, module) for name, module in model.named_modules() if isinstance(module, EPMoELayerBase)]
