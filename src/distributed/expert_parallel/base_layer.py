"""Base class for EP MoE layers.

:class:`EPMoELayerBase` owns the machinery shared by every EP family wrapper: the construction
template, DeepEP dispatch + GC dispatch caching, router/expert grad-sync hooks, the per-expert and
grouped-GEMM compute templates, and fused-GLU param init. Per-family wrappers in :mod:`.layers`
override :meth:`forward` and the compute kernels, not the shared :meth:`_compute_experts`.

Its two other halves are mixed in from their own modules: the checkpoint gather and shard merge
(:mod:`~src.distributed.expert_parallel.expert_gather`) and the router-balancing state
(:mod:`~src.distributed.expert_parallel.balancing`).
"""

from __future__ import annotations

import contextlib
import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from src.diagnostics.performance_monitor import get_performance_monitor
from src.distributed.expert_parallel.autograd import (
    MoEGatherPermute,
    MoEScatterUnpermute,
    ReduceFromExpertTP,
    ReplayCombineFunction,
    ReplayDispatchFunction,
)
from src.distributed.expert_parallel.balancing import EPRouterBalancingMixin
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.dispatcher import DeepEPDispatcher
from src.distributed.expert_parallel.expert_gather import EPExpertGatherMixin
from src.distributed.expert_parallel.gc_scope import (
    active_checkpoint_scope,
    in_backward_pass,
)
from src.distributed.expert_parallel.grad_sync import (
    create_expert_grad_hook,
    create_router_grad_hook,
    has_grad_sync_peers,
)
from src.distributed.grad_reduce import SumGradAcrossGroup
from src.distributed.runtime import (
    current_device,
    is_global_main_process,
)
from src.env import env_flag
from src.kernels.fused_glu import FusedGluMul, resolve_fused_glu_mul
from src.kernels.grouped_gemm import GroupedGemmPrecision, grouped_gemm
from src.kernels.histogram import sync_free_bincount
from src.models.moe_balancing import (
    ROUTER_TOPK_FIELDS,
    NativeBalancingSlot,
    native_balancing_bias_attrs,
    register_ep_wrapped_model_types,
    register_legacy_per_layer_config_keys,
    register_native_balancing_slot,
    register_source_config_schema,
)

logger = logging.getLogger(__name__)

# Opt-in overlap of the shared-expert FFN with the latency-bound dispatch all-to-all; one stream per device.
_SHARED_OVERLAP_ENABLED = env_flag("HALO_EP_SHARED_OVERLAP")
_SHARED_OVERLAP_STREAMS: dict[int, torch.cuda.Stream] = {}

# Token counts the expert-activation warmup traces at. Triton specializes a runtime integer argument
# on divisibility by 16, and these kernels take the element count (tokens × local intermediate width)
# as one: 16 tokens is divisible for every width, 1 token is not whenever the width is not — the only
# case a run can present the indivisible class at all. Warming both keeps either compile out of the
# dispatch→combine span.
_ACTIVATION_WARMUP_TOKENS = (16, 1)


def _shared_overlap_stream(device: torch.device) -> torch.cuda.Stream:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    stream = _SHARED_OVERLAP_STREAMS.get(idx)
    if stream is None:
        stream = torch.cuda.Stream(device=idx)
        _SHARED_OVERLAP_STREAMS[idx] = stream
    return stream


def _resolve_expert_count(layer: nn.Module, path: str) -> int | None:
    """Resolve a dotted attribute ``path`` on ``layer`` to an expert count.

    Returns ``None`` when any segment is missing or the resolved value carries no count. An ``int``
    is the count itself; anything sized (``nn.Parameter``, ``nn.ModuleList``, tensor) contributes
    its length (dim 0 for tensors).
    """
    obj = layer
    for attr in path.split("."):
        if not hasattr(obj, attr):
            return None
        obj = getattr(obj, attr)
    if isinstance(obj, int):
        return obj
    if hasattr(obj, "__len__"):
        return len(obj)
    return None


def has_grouped_mm() -> bool:
    """Whether ``F.grouped_mm`` can run here: SM90+ compute capability."""
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability()
    except RuntimeError as e:
        # ``is_available()`` already said yes, so this is a broken CUDA context on THIS rank (a fork
        # re-init), not an old GPU. Never swallowed silently: the answer decides a LAYOUT — GptOss
        # stores gate/up de-interleaved on the grouped path — so a rank that answers differently from
        # its peers enters the expert all-gather with different keys and the job hangs, not errors.
        logger.warning(
            "grouped-GEMM capability probe failed on this rank (%s: %s); falling back to the per-expert "
            "loop path. If peer ranks probe successfully they store expert weights under DIFFERENT "
            "attribute names — set use_grouped_gemm=false uniformly instead of running like this.",
            type(e).__name__,
            e,
        )
        return False
    return (major * 10 + minor) >= 90


class EPMoELayerBase(EPExpertGatherMixin, EPRouterBalancingMixin, nn.Module, ABC):
    """Base class for EP MoE layers using DeepEP.

    Composed of three parts: the construction template, forward and compute kernels below, the
    checkpoint gather/merge (:class:`EPExpertGatherMixin`) and the router-balancing state
    (:class:`EPRouterBalancingMixin`). A family subclasses THIS class and declares into all three.
    """

    # Fused GLU-combine kernel, derived from the resolved activation by ``_resolve_activation``; None = eager.
    _fused_glu_mul: FusedGluMul | None = None

    # False where routing replay is unimplementable: Gemma4 (external router) and Zaya (cross-layer state).
    _supports_routing_replay: bool = True

    # HF class name(s) of the MoE block this wrapper replaces; ``patching.build_moe_layer_map`` walks subclasses.
    HF_MODULE_NAMES: tuple[str, ...] = ()

    # ``config.model_type`` spelling(s) served; unioned so a checkpoint's config.json resolves back off-line.
    HF_MODEL_TYPES: tuple[str, ...] = ()

    # Every expert-weight attribute this family may hold across ALL config branches (fused vs separate/ETP).
    _EXPERT_WEIGHT_ATTR_ROOTS: tuple[str, ...] = ("gate_up_proj", "gate_proj", "up_proj", "down_proj")

    # Per-family dotted paths on the ORIGINAL HF MoE module, probed by :meth:`detect_num_experts` first.
    _NUM_EXPERTS_ATTR_PATHS: tuple[str, ...] = ()

    # ``(live module, hub)`` pairs for renames transformers applies only inside ``from_pretrained``. The
    # gather rewrites module → hub (vLLM silently skips unknown names); the lazy loader inverts it.
    _EXPORT_KEY_RENAMES: tuple[tuple[str, str], ...] = ()

    # ``{flat legacy key: (layer_type, per-layer field)}`` — attention geometry the hub config spells
    # one flat key per layer type, which transformers folds into ``per_layer_config`` on load and
    # re-serializes only folded. The pinned rollout server's transformers refuses the folded form at
    # parse, so every exported ``config.json`` is rewritten to the flat keys
    # (``export_legacy_per_layer_config``), which a reload folds back. Registered per claimed
    # ``model_type`` into the ``moe_balancing`` leaf by ``__init_subclass__``.
    _LEGACY_PER_LAYER_CONFIG_KEYS: dict[str, tuple[str, str]] = {}

    # Attribute(s) an HF MoE block may hold its expert container under, tried in order — one
    # vocabulary per family for every probe (container, expert count, the lazy loader's
    # checkpoint-key regexes, which union it over the roster). A family whose block spells it
    # differently across revisions declares every spelling it accepts; the default is the one name
    # every registered family's current block uses, so an unknown spelling raises rather than
    # resolving off a name no family claims.
    _EXPERTS_CONTAINER_ATTRS: tuple[str, ...] = ("experts",)

    # Must match the family's CHECKPOINT name — it becomes the export key in ``replicated_named_params``.
    # Resolved per instance from :attr:`_SHARED_EXPERT_ATTRS` for a family that declares one.
    _shared_expert_attr: str = "shared_experts"

    # Attribute(s) the HF block may hold its shared expert under, tried in order; the first spelling the
    # block CARRIES is adopted and becomes :attr:`_shared_expert_attr`, i.e. the export key. Empty = the
    # family has no shared expert at all. A declaring family whose block carries none (LFM-2) registers
    # the slot as ``None``, so the shared leg is skipped rather than raising on the family's own name.
    _SHARED_EXPERT_ATTRS: tuple[str, ...] = ()

    # True where the architecture ALWAYS builds the shared expert: an absent or ``None`` slot is then an
    # upstream rename, not a configuration, and adopting nothing would drop it from every output.
    _SHARED_EXPERT_REQUIRED: bool = False

    # Activation name :meth:`_resolve_activation` falls back to when neither the expert container nor the
    # wrapped block names one — the family's architectural default, not a global guess.
    _DEFAULT_ACTIVATION: str = "silu"

    # Attribute the router hangs off — on the wrapped HF block (:meth:`_find_gate_or_router` reads it
    # there) and on this wrapper, which re-registers it under the same name. Declared once per family
    # instead of respelled at each use, so the probe, the fp32 upcast and the grad-sync hook cannot be
    # pointed at different modules. ``None`` = the router is EXTERNAL to the wrapper (Gemma4's sibling
    # module): nothing is adopted here, no routing knob is read, and only expert grad-sync hooks register.
    _ROUTER_ATTR: str | None = "gate"

    # Reduction axis of those tensors as this family WRITES them: -1 = ``F.linear`` ``[E, N, K]``, GptOss
    # keeps matmul ``[E, K, N]``. The lowp export block-scales along it; the wrong axis corrupts every expert.
    HF_FUSED_EXPERT_CONTRACTION_AXIS: int = -1

    # False where the lazy loader cannot express this family's on-disk expert layout → from_pretrained + patch.
    _supports_lazy_loading: bool = True

    # transformers' conversion-mapping key(s) for this hub checkpoint (``hub_conversion.py``); empty = canonical.
    _HUB_CONVERSION_KEYS: tuple[str, ...] = ()

    # True where the live tree exists only behind transformers' load-side conversion and nothing
    # restores the hub namespace from a module-spelled save. The gathered save then applies
    # transformers' own save-side revert to every streamed chunk, so the artifact is the layout the
    # serving engines read. A declaring family's reverse map must not fuse tensors across an EP
    # layer's boundary — non-expert params revert as one chunk, each gathered EP layer as its own —
    # and sharded EP saves are refused: the offline merge streams key by key and cannot revert.
    _EXPORTS_HUB_NAMESPACE: bool = False

    # True where transformers' serialization of this config is a schema NO pinned serving engine
    # reads: the engines parse the family only through the SOURCE repo's ``auto_map`` modules, whose
    # vendor spellings transformers ABSORBS at load (``attribute_map`` plus ``__post_init__`` kwargs)
    # and never re-emits, so the save side cannot reconstruct them. Config writes carry the source
    # ``config.json`` and modules forward instead (``export_source_config_schema``). Registered per
    # claimed ``model_type`` into the ``moe_balancing`` leaf, so a wrapper-less run exports alike.
    _EXPORTS_SOURCE_CONFIG_SCHEMA: bool = False

    # False makes ``validate_weight_sync_support`` reject online/env GRPO instead of letting a sync land
    # nowhere (the export must load into vLLM under the names the trainer sends).
    _supports_weight_sync: bool = True

    # Surfaced verbatim in the refusal above; override where the namespace story is not the reason (Zaya).
    _WEIGHT_SYNC_REFUSAL_REASON: str = (
        "this family is served under a different checkpoint namespace/dtype than the HuggingFace "
        "module tree the trainer holds, so no weight would land where it is read"
    )

    # Spellings NO pinned rollout engine can serve — refused even though the family flag above is True.
    _WEIGHT_SYNC_UNSUPPORTED_MODEL_TYPES: tuple[str, ...] = ()

    # False where GC recompute on top of EP is architecturally broken (Zaya's cross-layer EDA/CCA state).
    _supports_gradient_checkpointing: bool = True

    # False where fp32 master weights on non-expert params break this family's dispatch; the loader refuses
    # ``fp32_non_ep_params`` rather than letting DeepEP's C++ assert fire after the whole multi-GPU load.
    _supports_fp32_non_ep_params: bool = True

    def __init_subclass__(cls, **kwargs):
        """Push this family's export-side declarations into the ``moe_balancing`` leaf under every
        ``model_type`` it claims — the config rewrites and the native balancing slot.

        The offline consumers (the config writers, the standalone conversion tools) read them there
        without importing this package, and a wrapper-less run (no EP, no grouped GEMM) exports the
        same config as a wrapped one. The gather/merge layout contract is enforced by
        :class:`EPExpertGatherMixin`, which declares it.
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

        The construction order every family shares, each step consuming what the previous one
        resolved: the router and the expert container off this family's own declarations, the EP
        state, the router adoption, the routing knobs, the shared expert, the expert-compute
        constants, this rank's expert weights, then the shared tail (device → expert LoRA → fp32
        upcast → grad-sync hooks) and the one construction line. A family customizes a STEP by
        overriding its hook, never by restating the sequence.

        ``weights_already_sharded``: the expert tensors are already sliced to this rank's range (the
        lazy loader), so the layout hooks adopt them instead of slicing dim 0 again.
        """
        router = self._find_gate_or_router(original_layer)
        experts = self._find_experts_container(original_layer)
        self._init_ep_state(ep_config, self._detect_hidden_dim(router, experts))
        if router is not None:
            setattr(self, self._ROUTER_ATTR, router)
        self._init_routing(original_layer)
        self._init_shared_experts(original_layer)
        self._init_expert_compute(original_layer, experts)
        self._init_expert_params(experts, weights_already_sharded=weights_already_sharded)
        self._finalize_expert_init()
        self._log_init_summary(*self._init_summary_extras(original_layer))

    @classmethod
    def _detect_hidden_dim(cls, router: nn.Module | None, experts: nn.Module) -> int:
        """The model hidden size this layer dispatches, off the modules the wrapper adopts.

        The router projection's input width where the family's gate is a bare ``[num_experts,
        hidden]`` matrix; otherwise the fused experts' own ``gate_up_proj [E, 2M, H]``, which is
        where a family whose router is external (Gemma4) or stateful with no projection (Zaya's
        EDA gate) keeps it. A classmethod: it runs before the module state exists.
        """
        if router is not None and hasattr(router, "weight"):
            return router.weight.shape[1]
        return cls._require_fused_experts(experts).gate_up_proj.shape[2]

    def _init_routing(self, original_layer: nn.Module) -> None:
        """Read the routing knobs this family's forward consumes off the wrapped block.

        Default: ``top_k`` alone. A family with more of them (DeepSeek-V3 group limiting, a selection
        function, a correction-bias buffer) overrides this and calls ``super()`` first. The router is
        adopted by then, so an override reads it as ``self.gate`` / ``self.router``.
        """
        self.top_k = self._find_top_k(original_layer)

    def _init_shared_experts(self, original_layer: nn.Module) -> None:
        """Adopt this family's shared expert under the name its CHECKPOINT spells.

        Resolved from :attr:`_SHARED_EXPERT_ATTRS` rather than assigned per family, so the attribute
        the wrapper registers — which is the export key ``replicated_named_params`` emits — is the
        one the block was read from.
        """
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

    def _init_expert_compute(self, original_layer: nn.Module, experts: nn.Module) -> None:
        """Adopt the expert-compute constants this family's kernels read.

        Default: the activation, taken off the expert container with the wrapped block as the name
        fallback (:meth:`_resolve_activation`). A family whose combine carries extra constants (a
        clamp bound, GptOss's ``alpha``) reads them here — directly, so an upstream rename raises
        instead of silently substituting a numeric the model was never trained with.
        """
        self._resolve_activation(experts, original_layer=original_layer, default=self._DEFAULT_ACTIVATION)

    def _init_expert_params(self, experts: nn.Module, weights_already_sharded: bool = False) -> None:
        """Register this rank's slice of the expert weights.

        Default: the fused-GLU layout (:meth:`_init_fused_glu_params`). A family whose HF container
        holds another layout — Qwen3's pre-fused halves, Bailing's per-expert modules, GptOss's
        interleaved pair — overrides this.
        """
        self._init_fused_glu_params(experts, weights_already_sharded=weights_already_sharded)

    def _init_summary_extras(self, original_layer: nn.Module) -> tuple[str, ...]:
        """This family's extra fields for the construction line (see :meth:`_log_init_summary`)."""
        return ()

    def _init_ep_state(self, ep_config: EPConfig, hidden_dim: int):
        """Build the EP plumbing shared by every wrapper: expert ranges, dispatcher, grad-sync config,
        compute precision, LoRA/replay slots and the phase-timing seam.

        Split out of :meth:`__init__` so the template above owns the family-facing sequence while this
        owns the state alone — which is also the only entry point a caller holding no HF MoE block
        (a construction probe) can use."""
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
                f"expert_tp_size={self.expert_tp_size} requires expert_tp_group to be created "
                f"in EPConfig process group setup. Got expert_tp_group=None."
            )

        self.fp32_router = ep_config.fp32_router
        self.fp32_experts = ep_config.fp32_experts

        self.dispatcher = DeepEPDispatcher(ep_config, self.num_experts, hidden_dim)
        self._grad_sync_hooks = []
        self._hook_synced_expert_params: list = []  # populated only in the in-backward hook regime

        self._use_grouped_mm = ep_config.use_grouped_gemm and has_grouped_mm()

        # Latched by _warm_activation_graphs on the first forward (init is too early: the eager loader
        # still has this layer on CPU, and a CPU trace is not the graph the run calls).
        self._activation_warmed = False

        # Low-precision expert compute, set per-layer through set_lowp_compute(). Not cacheable when
        # experts are FSDP2-managed: the unsharded param's version counter is pinned, so the per-step quant
        # cache would serve step-0 weights forever.
        self._lowp_precision = GroupedGemmPrecision.BF16
        self._lowp_weight_cacheable = True

        # Built by _init_expert_lora; empty set = no LoRA, so the hot path is one set-membership check.
        self._expert_lora_attrs: frozenset[str] = frozenset()
        self.expert_lora_scaling: float = 1.0
        self.expert_lora_dropout: nn.Module = nn.Identity()
        # Toggled off by disable_expert_adapters(): not peft-managed, so the KL reference pass must clear them.
        self._expert_adapters_enabled: bool = True

        # Routing replay, plain attrs (FSDP2-ignored, not exported): ``[T, top_k]`` GLOBAL ids, -1 = natural.
        self._forced_topk_indices: torch.Tensor | None = None
        # Cursor over a mask spanning chunked forwards (wraps at exhaustion); the monotonic total lets disarm
        # tell "never consumed" from "k full passes".
        self._forced_cursor: int = 0
        self._forced_consumed_total: int = 0
        self._capture_routing: bool = False
        self._captured_routing_chunks: list[torch.Tensor] = []
        # [flipped, forced] token-count accumulator (device-side adds; reader syncs once per logging step).
        self._replay_flip_counts: torch.Tensor | None = None

        # HALO_EP_PERF_PROFILE=1 → CUDA-event phase timers (per-phase syncs, diagnostic runs only); off → label.
        self._perf: Callable[[str], contextlib.AbstractContextManager]
        if env_flag("HALO_EP_PERF_PROFILE"):
            self._perf = get_performance_monitor().time_operation
        else:
            self._perf = torch.profiler.record_function

    def _glu_combine_name(self) -> str:
        """What this layer's GLU combine ACTUALLY is, for the construction-time summary.

        Read off the latch itself, so the reported name cannot disagree with what runs — including a
        family that binds its own clamp into the latch (``functools.partial``, which carries no
        ``__name__``, or a bound method). Only GptOss, whose interleaved-bias paths never reach
        :meth:`_glu_combine`, names its combine itself.
        """
        combine = getattr(self._fused_glu_mul, "func", self._fused_glu_mul)  # a partial → the kernel it binds
        if combine is None:
            return "eager"
        return getattr(combine, "__name__", type(combine).__name__)

    def _shared_expert_summary_field(self) -> tuple[str, ...]:
        """``('<adopted attr>=yes|no',)`` for a family declaring a shared expert, else empty.

        Keyed on :attr:`_shared_expert_attr` — the name :meth:`_init_shared_experts` actually adopted
        — so the reported field names the module the forward calls and the save exports."""
        if not self._SHARED_EXPERT_ATTRS:
            return ()
        present = getattr(self, self._shared_expert_attr, None) is not None
        return (f"{self._shared_expert_attr}={'yes' if present else 'no'}",)

    def _log_init_summary(self, *extras: str) -> None:
        """Emit this layer's one construction-time line: the fields every family shares, then ``extras``.

        ``grouped_mm`` reports the EFFECTIVE decision (:meth:`_grouped_mm_enabled`) rather than the
        requested flag, so a family whose layout drops it to the per-expert loop says so here instead of
        advertising grouped GEMM it is not running. ``glu_combine`` reports the EFFECTIVE combine
        (:meth:`_glu_combine_name`) for the same reason. The shared-expert field is derived from what
        :meth:`_init_shared_experts` adopted, under the name it adopted it as, so a declaring family
        reports it without restating the line.

        The INFO line is main-process only, like every other per-layer INFO here, so it carries only
        rank-uniform fields: a 92-layer MoE at 512 ranks would otherwise emit ~47k near-identical lines
        at startup. This rank's own expert range is the one rank-specific fact, so it goes to DEBUG on
        every rank instead of being printed for rank 0 and lost everywhere else.
        """
        logger.debug(
            f"{type(self).__name__}: EP rank {self.ep_rank} owns experts "
            f"{self.expert_start}-{self.expert_end - 1} of {self.num_experts}"
        )
        if not is_global_main_process():
            return
        fields = (
            f"{self.expert_end - self.expert_start}/{self.num_experts} experts per EP rank",
            *self._shared_expert_summary_field(),
            *extras,
            f"grouped_mm={self._grouped_mm_enabled()}",
            f"glu_combine={self._glu_combine_name()}",
            f"fp32_router={self.fp32_router}",
            f"fp32_experts={self.fp32_experts}",
        )
        logger.info(f"{type(self).__name__}: {', '.join(fields)}")

    @staticmethod
    def _in_gc_recompute() -> bool:
        """True during the gradient-checkpoint backward recompute of this layer.

        The enclosing :class:`~src.distributed.expert_parallel.gc_scope.EPCheckpointScope` counts the
        passes through the checkpointed body, so this holds for both checkpoint modes — grad mode
        alone would not (non-reentrant runs BOTH passes with grad enabled).
        """
        scope = active_checkpoint_scope()
        return scope is not None and scope.is_recompute

    def _maybe_replace_selection(self, topk_indices: torch.Tensor) -> torch.Tensor:
        """Replace natural top-k indices with the forced routing-replay mask, where one is armed.

        ``_forced_topk_indices`` is ``[T, top_k]`` long (global expert ids, flat token order); ``-1`` keeps
        that token's natural selection. Selection is non-differentiable (``topk`` on detached scores), so
        swapping indices removes nothing from the graph — callers re-gather gate weights from live scores
        at the returned indices. Also accumulates the replay flip-rate (skipped during GC recompute).
        """
        forced_full = self._forced_topk_indices
        if forced_full is None:
            return topk_indices
        tokens = topk_indices.size(0)
        scope = active_checkpoint_scope()
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
        if not self._in_gc_recompute():
            with torch.no_grad():
                flips = (forced.sort(dim=-1).values != topk_indices.sort(dim=-1).values).any(dim=-1)
                counts = torch.stack([(flips & valid.squeeze(1)).sum(), valid.sum()]).float()
                if self._replay_flip_counts is None:
                    self._replay_flip_counts = counts
                else:
                    self._replay_flip_counts += counts
        return torch.where(valid, forced, topk_indices)

    def _deepseek_biased_route(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """DeepSeek-V3 selection for families that re-derive routing from logits: pick top-k on the
        bias-adjusted softmax probs, then gate on the unbiased softmax over the selected logits. Returns
        ``(scores, indices)``, each ``[T, top_k]``."""
        indices = self._biased_topk(logits)
        scores = F.softmax(torch.gather(logits.float(), -1, indices), dim=-1)
        return scores, indices

    def _softmax_gate_weights_at(self, router_logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Renormalized-softmax router weights at ``indices`` — the Qwen3-style gating shared by every
        family whose router is *softmax over all experts → gather → optional top-k renorm*.

        ``norm_topk_prob`` is read off the LIVE router module and defaults to True: Qwen3 makes it a
        required config field, while Qwen3.5's ``Qwen3_5MoeTopKRouter`` carries no such attribute and
        renormalizes unconditionally. Reading the module rather than a cached copy is what keeps routing
        replay bit-identical to the router the forward just called. Not a default for ``_gate_weights_at``
        itself: a family whose gating is not this formula (Bailing's sigmoid, Cohere2's) must still fail
        loudly rather than inherit the wrong weights.
        """
        probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        weights = probs.gather(-1, indices)
        if getattr(self.gate, "norm_topk_prob", True):
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights.to(router_logits.dtype)

    def _gc_dispatch(self, flat, experts, weights):
        """Dispatch, replaying this checkpoint frame's cached result during recompute.

        A second DeepEP dispatch would issue a fresh all-to-all into the same ``ElasticBuffer`` and
        invalidate the handle the ORIGINAL forward's backward node still holds — measured as
        corruption of every gradient in the stage, not an error. So the first pass through the
        checkpointed body caches detached results in its
        :class:`~src.distributed.expert_parallel.gc_scope.EPCheckpointScope` and the recompute
        replays them. Outside a checkpoint (and for the inert ep1 dispatcher) this is a plain dispatch.
        """
        if self.dispatcher._noop:  # ep1: no transport, nothing to protect
            return self.dispatcher.dispatch(flat, experts, weights)
        scope = active_checkpoint_scope()
        if scope is None:
            if self.training and in_backward_pass():
                raise RuntimeError(
                    f"{type(self).__name__}: training forward inside backward with no EP checkpoint "
                    f"scope — gradient checkpointing was re-enabled without install_ep_checkpoint_scopes "
                    f"(TRL's disable_gradient_checkpointing context manager re-installs a bare "
                    f"checkpoint on exit). A second DeepEP dispatch here corrupts every gradient."
                )
            return self.dispatcher.dispatch(flat, experts, weights)

        slot = scope.slot(self, "dispatch")
        if not scope.is_recompute:
            recv_x, recv_topk_idx, recv_topk_weights, handle = self.dispatcher.dispatch(flat, experts, weights)
            # Detached aliases, not clones: DeepEP's copy epilogue already lands every result in an
            # allocator-owned tensor (never a view of the shared arena), and nothing downstream
            # writes into them in place — a clone would only add a full recv-buffer memcpy per layer.
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
        """Combine, replaying this checkpoint frame's cached result during recompute (see
        :meth:`_gc_dispatch` — the combine writes the same shared buffer)."""
        scope = active_checkpoint_scope()
        if scope is None or self.dispatcher._noop:
            return self.dispatcher.combine(output, recv_topk_weights, handle)

        slot = scope.slot(self, "combine")
        if not scope.is_recompute:
            result = self.dispatcher.combine(output, recv_topk_weights, handle)
            slot["combined"] = result.detach()  # allocator-owned, read-only downstream (see _gc_dispatch)
            return result

        return ReplayCombineFunction.apply(output, recv_topk_weights, handle, self.dispatcher, slot["combined"])

    def _setup_gradient_sync(self, router: nn.Module, expert_params: list):
        """Register gradient sync hooks for router, replicated, and expert parameters.

        Every param here is inside this EP module (FSDP ``ignored_params``), so no FSDP reduce-scatter
        reaches them — these hooks are the only sync:

        - **Router + replicated submodules** (shared expert + gate): replicated across DP ranks, computed
          on the LOCAL batch → plain DP average (``create_router_grad_hook``); else they drift across DP ranks.
        - **Expert FFN shards**: EP/ETP-distributed; divisor ``world_size/expert_tp_size``
          (``create_expert_grad_hook``).
        """
        # Truly replicated experts (ep_group_size==1) are FSDP reduce-scattered, so EP hooks would double-
        # sync. NOT pure ETP (ep_size==1, expert_tp_size>1), which still needs them.
        if self.ep_config.experts_fsdp_managed:
            return

        # Deferred sync (every cross-replica topology and PP rank block): the post-backward sweep averages,
        # so no in-backward hook may race the DeepEP combine or re-fire per µbatch.
        if self.ep_config.defer_grad_sync:
            return

        if has_grad_sync_peers(self.ep_config):
            dp_hook = create_router_grad_hook(self.ep_config)
            replicated = list(router.named_parameters()) + self.replicated_named_params()
            for _, param in replicated:
                if param.requires_grad:
                    self._grad_sync_hooks.append(param.register_post_accumulate_grad_hook(dp_hook))

        expert_hook = create_expert_grad_hook(self.ep_config)
        trainable_experts = [param for _, param in expert_params if param.requires_grad]
        # Peers only: without them the hook is a no-op, so the zero-token edge would add graph
        # nodes per forward for a divide that never runs.
        if has_grad_sync_peers(self.ep_config):
            self._hook_synced_expert_params = trainable_experts
        for param in trainable_experts:
            self._grad_sync_hooks.append(param.register_post_accumulate_grad_hook(expert_hook))

    def _expert_hook_grad_edge(self, out: torch.Tensor) -> torch.Tensor:
        """Keep the in-backward expert hooks firing for params absent from this backward's graph.

        ``create_expert_grad_hook`` divides the ACCUMULATED gradient on the sync microbatch, but a
        post-accumulate hook fires only for params in that backward's graph — a rank receiving zero
        tokens (the early returns) or a single expert idling in the weighted loop leaves its
        accumulated gradient undivided, ``world/expert_tp``× too large at the optimizer. A
        zero-valued edge onto every expert param that already accumulated this window keeps the
        divide reachable; params with no accumulated grad stay out, so an all-idle window still
        leaves ``grad=None`` and the optimizer skips them. Empty in the deferred and FSDP-managed
        regimes, whose sweep is structural already.
        """
        if not torch.is_grad_enabled() or not self._hook_synced_expert_params:
            return out
        stubs = [p.reshape(-1)[0] for p in self._hook_synced_expert_params if p.grad is not None]
        if not stubs:
            return out
        # fp32-cast before the stack: the trainable set may mix dtypes (an adapter over a bf16 base).
        return out + (torch.stack([s.float() for s in stubs]).sum() * 0.0).to(out.dtype)

    @property
    def _fp32_router_input(self) -> bool:
        """Whether to upcast the router input to fp32 for the routing matmul.

        ``fp32_router`` keeps the router weight fp32, so the forward upcasts its input to match — but only
        when the router is FSDP-IGNORED (``ep_group_size > 1``, or ep1 without ``fsdp_shard_ep1_experts``).
        When FSDP-MANAGED (ep1 with ``fsdp_shard_ep1_experts``), FSDP casts the weight to bf16 for compute,
        so an fp32 input would mismatch it; return False there (the fp32 master weight is preserved regardless)."""
        router_fsdp_managed = self.ep_config.experts_fsdp_managed
        return self.fp32_router and not router_fsdp_managed

    def _live_router_module(self) -> nn.Module | None:
        """The router/gate submodule resolved LIVE under :attr:`_ROUTER_ATTR` (None where the family's
        router is external to the EP module).

        By-name resolution returns the PEFT ``modules_to_save`` wrapper when one has replaced the router."""
        return getattr(self, self._ROUTER_ATTR, None) if self._ROUTER_ATTR else None

    def _dp_averaged_named_params(self) -> list:
        """``(name, param)`` for the FSDP-ignored submodules this layer DP-averages via the router hook:
        the router/gate plus any full replicated submodules (shared expert). Resolved live so a PEFT
        ``modules_to_save`` copy is included."""
        router = self._live_router_module()
        router_params = list(router.named_parameters()) if router is not None else []
        return router_params + self.replicated_named_params()

    def reattach_router_grad_sync(self) -> int:
        """Re-register the DP-average hook on trainable router/replicated params, for PEFT modules_to_save.

        Construction-time :meth:`_setup_gradient_sync` hooks the ORIGINAL router params; PEFT
        ``modules_to_save`` trains a COPY those hooks never see. Re-attach the hook to the live params.
        Returns the count hooked. No-op unless this layer uses the in-backward router hook (the
        deferred post-backward sweep and the ep1 FSDP-sharded path already cover the copy). Only
        trainable params are touched, so frozen originals are skipped — no double-hooking."""
        if self.ep_config.defer_grad_sync or not has_grad_sync_peers(self.ep_config):
            return 0
        if self.ep_config.experts_fsdp_managed:
            return 0
        dp_hook = create_router_grad_hook(self.ep_config)
        attached = 0
        for _name, param in self._dp_averaged_named_params():
            if param.requires_grad:
                self._grad_sync_hooks.append(param.register_post_accumulate_grad_hook(dp_hook))
                attached += 1
        return attached

    def synced_trainable_param_ids(self) -> set:
        """Param ids this layer keeps DP/EP-consistent — reference set for the trainable-param safety net.

        The ep1 FSDP-sharded path and the deferred post-backward sweep sync EVERY trainable EP param, so
        all count. The in-backward hook path syncs only experts (+ LoRA), the router, and replicated submodules."""
        trainable = {id(p) for p in self.parameters() if p.requires_grad}
        if self.ep_config.experts_fsdp_managed or self.ep_config.defer_grad_sync:
            return trainable
        ids = {id(p) for _n, p in self.expert_named_params() if p.requires_grad}
        ids |= {id(p) for _n, p in self._dp_averaged_named_params() if p.requires_grad}
        return ids

    def _to_device(self):
        """Place parameters, submodules and buffers on this rank's device.

        Follows the wrapped layer's OWN placement when it has one: a lazily-loaded (meta) or
        parameterless layer goes to the local CUDA device, while a layer already materialized on CPU
        stays there — the trainer's ``_move_model_to_device`` moves the whole model afterwards
        anyway. A GPU existing in the process is not a reason to relocate a layer built on CPU: its first
        forward takes CPU inputs.
        """
        device = next((p.device for p in self.parameters()), None)
        if device is None or device.type == "meta":
            device = current_device()
        self.to(device)

    def _select_expert_tokens(
        self, local_idx: int, experts: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find tokens routed to a specific local expert and their routing weights.

        ``experts``/``weights`` are ``[T, top_k]`` (see :meth:`_compute_experts`)."""
        top_k_pos, token_idx = torch.where((experts == local_idx).t())
        return token_idx, weights[token_idx, top_k_pos]

    def _compute_experts_weighted(
        self, tokens: torch.Tensor, experts: torch.Tensor, weights: torch.Tensor, output_dtype: torch.dtype, compute_fn
    ) -> torch.Tensor:
        """Route tokens to local experts, compute, and accumulate weighted outputs.

        Template method: ``compute_fn(local_idx, x)`` returns the expert output
        for one expert. This method handles token selection, weighting, and
        accumulation via index_add_.
        """
        if tokens.shape[0] == 0:
            return self._expert_hook_grad_edge(
                torch.zeros((0, tokens.shape[-1]), device=tokens.device, dtype=output_dtype)
            )

        output = torch.zeros(tokens.shape, device=tokens.device, dtype=output_dtype)

        with torch.amp.autocast("cuda", dtype=output_dtype, enabled=self.fp32_experts):
            for local_idx in range(self.experts_per_rank):
                token_idx, expert_weights = self._select_expert_tokens(local_idx, experts, weights)
                if token_idx.numel() == 0:
                    continue
                out = compute_fn(local_idx, tokens.index_select(0, token_idx))
                expert_output = out.to(output_dtype) * expert_weights.unsqueeze(-1).to(output_dtype)
                output.index_add_(0, token_idx, expert_output)

        # Per-expert params (this loop's regime) miss the sync-microbatch divide when their expert
        # idles in it after accumulating earlier; edges are per-PARAM, so the rank-level wrap on the
        # zero-token return above cannot cover an individual idle expert.
        return self._expert_hook_grad_edge(output)

    @staticmethod
    def _build_inv_map(sorted_token_idx: torch.Tensor, recv_N: int, width: int) -> torch.Tensor:
        """Build the atomic-free permute/unpermute map.

        ``inv_map[r, j]`` = the j-th sorted position whose recv token is ``r``, padded to ``width`` cols
        with sentinel ``N_sorted``. Sync-free (stable argsort + cumulative counts, no host round-trip).
        Consumed by :class:`MoEGatherPermute` / :class:`MoEScatterUnpermute` to replace the bf16 atomic ``index_add_``.
        """
        device = sorted_token_idx.device
        n_sorted = sorted_token_idx.shape[0]
        inv_map = torch.full((recv_N, width), n_sorted, dtype=torch.long, device=device)
        if n_sorted == 0:
            return inv_map
        order = torch.argsort(sorted_token_idx, stable=True)
        rp = sorted_token_idx.index_select(0, order)  # recv positions, grouped & contiguous
        counts = sync_free_bincount(sorted_token_idx, recv_N, dtype=torch.long)
        starts = torch.cumsum(counts, 0) - counts
        slot = torch.arange(n_sorted, device=device) - starts.index_select(0, rp)
        inv_map[rp, slot] = order
        return inv_map

    def _sort_tokens_for_grouped_mm(self, tokens: torch.Tensor, experts: torch.Tensor, weights: torch.Tensor):
        """Sort tokens by expert index and compute offsets for grouped_mm.

        Filters out invalid expert IDs (< 0 or >= experts_per_rank, DeepEP dispatch padding); only valid
        local assignments are in the sorted output.

        Returns:
            sorted_tokens: [N_valid, H] contiguous tokens sorted by expert
            offs: [E_local] int32 cumulative token counts per expert
            sorted_token_idx: [N_valid] original token positions for scatter-back
            sorted_weights: [N_valid] routing weights in sorted order
            sorted_expert_ids: [N_valid] expert ids in sorted order (for bias lookup)
            inv_map: [N, width] atomic-free permute/unpermute map (see _build_inv_map)
        """
        N = tokens.shape[0]
        device = tokens.device

        width = experts.shape[1]  # top_k = max local-expert assignments per token → inv_map width
        flat_indices = experts.reshape(-1)
        flat_weights = weights.reshape(-1)
        flat_token_idx = torch.arange(N, device=device).unsqueeze(1).expand(-1, width).reshape(-1)

        # DeepEP pads unfilled slots with -1; the host-known ep_size gate avoids a device sync at ep_size==1.
        if self.ep_size > 1:
            valid_mask = (flat_indices >= 0) & (flat_indices < self.experts_per_rank)
            valid_idx = valid_mask.nonzero(as_tuple=True)[0]
            flat_indices = flat_indices[valid_idx]
            flat_weights = flat_weights[valid_idx]
            flat_token_idx = flat_token_idx[valid_idx]

        sort_order = torch.argsort(flat_indices, stable=True)
        sorted_expert_ids = flat_indices[sort_order]
        sorted_token_idx = flat_token_idx[sort_order]
        sorted_weights = flat_weights[sort_order]

        # A histogram, not unique_consecutive(return_counts=True): the latter sizes its output from the
        # run count, a host sync per layer per forward.
        expert_counts = sync_free_bincount(sorted_expert_ids, self.experts_per_rank, dtype=torch.long)
        offs = torch.cumsum(expert_counts, dim=0).to(torch.int32)

        # The atomic-free permute beats the bf16 atomic index_add_ only under real duplicate-row contention.
        if width >= self.ep_size:
            inv_map = self._build_inv_map(sorted_token_idx, N, width)
            sorted_tokens = MoEGatherPermute.apply(tokens, sorted_token_idx, inv_map)
        else:
            inv_map = None
            sorted_tokens = tokens.index_select(0, sorted_token_idx)
        return sorted_tokens, offs, sorted_token_idx, sorted_weights, sorted_expert_ids, inv_map

    def _glu_combine(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Combine the GLU halves into the activated intermediate: ``act_fn(gate) * up``.

        The single overridable seam for every base GLU compute path (fused/separate loop, grouped-GEMM,
        ETP separate-GLU); a family with a non-standard combine (DeepseekV4's clamped SwiGLU) overrides
        this instead of re-implementing the compute templates. GptOss owns its interleaved-bias compute
        paths outright and never reaches this seam.

        A fused kernel serves any family whose gate is one the kernels implement (SiLU, tanh-GELU),
        decided by :meth:`_resolve_activation` off the activation itself rather than by a per-family
        flag, so a family opts in by having such a gate rather than by remembering to set one.
        """
        if self._fused_glu_mul is not None:
            return self._fused_glu_mul(gate, up)
        return self.act_fn(gate) * up

    def _warm_expert_activation(self, gate_up: torch.Tensor) -> torch.Tensor:
        """Run this layer's expert activation over a synthetic projection output ``[T, 2M]``.

        The halves are split exactly as the compute templates split them — a strided view under fused
        storage, contiguous under separate/ETP storage — because a compiled graph guards on the stride
        it first traced, and a guard miss recompiles at the call site this warmup exists to keep cold-
        free. A family whose activation is not behind :meth:`_glu_combine` overrides this.
        """
        gate, up = gate_up.chunk(2, dim=-1)
        if not hasattr(self, "gate_up_proj"):
            gate, up = gate.contiguous(), up.contiguous()
        return self._glu_combine(gate, up)

    def _warm_activation_graphs(self, device: torch.device, dtype: torch.dtype) -> None:
        """JIT this layer's expert activation ahead of its first dispatch (once per layer).

        The activation is the one lazily-compiled callable inside the dispatch→combine span: every
        expert combine on the roster is a Triton kernel (the fused GLUs, the clamped SwiGLUs, GptOss),
        and Triton compiles a kernel on its first call per dtype and constexpr set. Left cold it
        compiles BETWEEN the two collectives, while every peer spins in DeepEP's barrier — whose
        budget bounds rank SKEW, and whose clock starts when a rank enters it.

        Both grad modes, because the forward and backward kernels are separate compilations and the
        backward is only built when a grad is first requested. ``enable_grad`` rather than the ambient
        mode: under gradient checkpointing the first forward runs under ``no_grad``, which would leave
        the backward kernel to the recompute — inside the span again. Inputs are leaves, so nothing
        accumulates a warmup gradient, and both token counts of :data:`_ACTIVATION_WARMUP_TOKENS` run
        because Triton specializes each compilation on the element count's divisibility.

        Scoped to a real dispatch group: at ``ep_size == 1`` there is no all-to-all and no barrier for
        a compiling rank to stall, so this would only move the same compilation earlier.
        """
        if self._activation_warmed or self.ep_size <= 1:
            return
        self._activation_warmed = True
        # down_proj is [E, M, H] in matmul convention on every layout (ETP included), so dim 1 is the
        # local intermediate width the activation sees — the fused projection output is twice it.
        width = 2 * self.down_proj.shape[1]
        for tokens in _ACTIVATION_WARMUP_TOKENS:
            # An inference-mode forward cannot build an autograd graph at all, and never runs a backward.
            if not torch.is_inference_mode_enabled():
                # Identity saved-tensor hooks: this runs INSIDE the checkpointed block, and a
                # non-reentrant checkpoint packs everything saved there through its own hooks — the
                # backward below would unpack one, trigger that checkpoint's recompute mid-forward,
                # and replay a dispatch the original pass has not made yet.
                with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
                    gate_up = self._activation_warmup_input(tokens, width, device, dtype, requires_grad=True)
                    torch.autograd.grad(self._warm_expert_activation(gate_up).sum(), gate_up)
            with torch.no_grad():
                self._warm_expert_activation(
                    self._activation_warmup_input(tokens, width, device, dtype, requires_grad=False)
                )

    @staticmethod
    def _activation_warmup_input(
        tokens: int, width: int, device: torch.device, dtype: torch.dtype, *, requires_grad: bool
    ) -> torch.Tensor:
        """One synthetic projection output ``[T, 2M]``, drawn from no generator.

        The warmup runs inside the gradient-checkpointed region, whose recompute restores the RNG to
        region entry: a random draw here would be replayed by the forward and skipped by the
        recompute (it is latched), giving every RNG consumer downstream of the MoE in that block —
        expert-LoRA dropout — a different sample than the loss was computed from. The kernels compile
        on shape and dtype, so the values are irrelevant.
        """
        return torch.zeros(tokens, width, device=device, dtype=dtype, requires_grad=requires_grad)

    def _expert_forward(self, idx: int, x: torch.Tensor) -> torch.Tensor:
        """One local expert's forward for the per-expert loop — the overridable seam
        :meth:`_compute_experts` hands :meth:`_compute_experts_weighted` (GptOss's interleaved
        layouts override it).

        Base = the GLU forward: gate_up_proj → chunk → act_fn → down_proj, weights in matmul
        convention (``gate_up_proj [E, H, 2M]``, ``down_proj [E, M, H]``). Keys on the STORED layout:
        families that keep the halves separate (Qwen3, Bailing) and every family under ETP (where
        slicing the fused halves along dim 2 would split gate from up onto different ranks) run the
        3-projection path instead.
        """
        if not hasattr(self, "gate_up_proj"):
            gate = self._expert_proj_single(idx, x, "gate_proj")
            up = self._expert_proj_single(idx, x, "up_proj")
            activated = self._glu_combine(gate, up)
            return self._expert_proj_single(idx, activated, "down_proj")
        gate_up = self._expert_proj_single(idx, x, "gate_up_proj")
        gate, up = gate_up.chunk(2, dim=-1)
        activated = self._glu_combine(gate, up)
        return self._expert_proj_single(idx, activated, "down_proj")

    def _grouped_mm(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        *,
        offs: torch.Tensor | None = None,
        lowp: bool = True,
    ) -> torch.Tensor:
        """Grouped GEMM for expert FFNs at this layer's compute precision (bf16 unless a low-precision
        mode was enabled via :func:`~src.kernels.lowp.mixed_precision.apply_mixed_precision_compute`).
        ``lowp=False`` forces bf16 for operands that must not be fake-quantized (LoRA adapters).

        No fused bias: the families that carry expert biases add them through ``MoEExpertBiasGather``,
        whose backward is an atomic-free GEMM rather than a bf16 atomic scatter."""
        return grouped_gemm(
            mat_a,
            mat_b,
            offs=offs,
            precision=self._lowp_precision if lowp else GroupedGemmPrecision.BF16,
            weight_cacheable=self._lowp_weight_cacheable,
        )

    # Native grouped-LoRA: adapted ``W [E,K,N]`` gains ``_lora_A [E,K,r]`` / ``_lora_B [E,r,N]``, base frozen.

    def _init_expert_lora(self) -> None:
        """Build grouped LoRA adapters for the selected expert projections.

        Called at the END of each family ``__init__``. For each 3-D expert weight the spec adapts, registers
        ``{attr}_lora_A``/``{attr}_lora_B`` (kaiming A, zeros B ⇒ initial delta 0) and freezes the base.
        Resolving via live ``expert_named_params`` handles per-family + ETP/GptOss-loop variants.
        """
        spec = self.ep_config.expert_lora
        if spec is None:
            return

        self.expert_lora_scaling = spec.scaling
        self.expert_lora_dropout = nn.Dropout(spec.dropout) if spec.dropout > 0 else nn.Identity()

        adapted: list[str] = []
        for name, param in self.expert_named_params():
            if param.dim() != 3 or not spec.adapts(name):
                continue
            num_local, k, n = param.shape
            lora_a = nn.Parameter(torch.empty(num_local, k, spec.r, dtype=param.dtype, device=param.device))
            lora_b = nn.Parameter(torch.zeros(num_local, spec.r, n, dtype=param.dtype, device=param.device))
            # 1/sqrt(K) reproduces PEFT's 2-D kaiming_uniform_(a=sqrt(5)); handing torch the 3-D [E, K, r]
            # tensor would fold r into fan_in, warming expert adapters at a different scale than attention.
            bound = 1.0 / math.sqrt(k)
            nn.init.uniform_(lora_a, -bound, bound)
            setattr(self, f"{name}_lora_A", lora_a)
            setattr(self, f"{name}_lora_B", lora_b)
            param.requires_grad_(False)  # freeze base; only A/B train
            adapted.append(name)

        self._expert_lora_attrs = frozenset(adapted)
        if adapted and is_global_main_process():
            logger.info(
                f"{type(self).__name__}: native grouped-LoRA on experts "
                f"(r={spec.r}, alpha={spec.alpha}, dropout={spec.dropout}, rslora={spec.use_rslora}, "
                f"scaling={spec.scaling:.4g}, projections={sorted(adapted)})"
            )

    @property
    def has_expert_lora(self) -> bool:
        """Whether this layer carries native grouped-LoRA adapters on its experts.

        The one predicate every "is there an adapter to fold?" decision reads, derived from what
        :meth:`_init_expert_lora` actually built — not from the config, which can request adapters a
        family's live layout ends up with none of."""
        return bool(self._expert_lora_attrs)

    def _expert_lora_named_params(self) -> list:
        """``(name, param)`` for this layer's grouped LoRA adapters (empty when no expert LoRA)."""
        out = []
        for attr in self._expert_lora_attrs:
            out.append((f"{attr}_lora_A", getattr(self, f"{attr}_lora_A")))
            out.append((f"{attr}_lora_B", getattr(self, f"{attr}_lora_B")))
        return out

    def _expert_proj(
        self, x: torch.Tensor, weight_attr: str, offs: torch.Tensor, out_dtype: torch.dtype
    ) -> torch.Tensor:
        """Grouped-GEMM expert projection ``x @ W[weight_attr]``, plus the LoRA delta when
        ``weight_attr`` carries an adapter. A no-op set lookup otherwise."""
        out = self._grouped_mm(x, getattr(self, weight_attr).to(out_dtype), offs=offs)
        if self._expert_adapters_enabled and weight_attr in self._expert_lora_attrs:
            # Adapters stay bf16 under lowp: block-scaling along rank r is unrepresentable (r < block size).
            lora_a = getattr(self, f"{weight_attr}_lora_A").to(out_dtype)
            lora_b = getattr(self, f"{weight_attr}_lora_B").to(out_dtype)
            hidden = self._grouped_mm(self.expert_lora_dropout(x), lora_a, offs=offs, lowp=False)
            out = out + self.expert_lora_scaling * self._grouped_mm(hidden, lora_b, offs=offs, lowp=False)
        return out

    def _expert_proj_single(self, idx: int, x: torch.Tensor, weight_attr: str) -> torch.Tensor:
        """Per-expert (loop-path) projection ``x @ W[weight_attr][idx]`` plus the LoRA delta."""
        out = x @ getattr(self, weight_attr)[idx]
        if self._expert_adapters_enabled and weight_attr in self._expert_lora_attrs:
            lora_a = getattr(self, f"{weight_attr}_lora_A")[idx]
            lora_b = getattr(self, f"{weight_attr}_lora_B")[idx]
            out = out + self.expert_lora_scaling * (self.expert_lora_dropout(x) @ lora_a @ lora_b)
        return out

    def _grouped_mm_enabled(self) -> bool:
        """Whether the grouped-GEMM path is active — the gate for the grouped-vs-loop branch in
        :meth:`_compute_experts`. Default: the hardware/config flag. GptOss overrides to also require
        ``expert_tp_size <= 1`` (its interleaved weights can't be de-interleaved once TP-sharded)."""
        return self._use_grouped_mm

    def set_lowp_compute(self, precision: GroupedGemmPrecision, *, weight_cacheable: bool) -> bool:
        """Set this layer's expert-compute precision; return whether the layer can HONOR it.

        The public seam :func:`~src.kernels.lowp.mixed_precision.apply_mixed_precision_compute` writes
        through, so the low-precision request and the "did it apply?" answer come from the owner of
        both facts. False means this layer runs the per-expert loop, where the grouped kernels — and
        with them the fake-quant — never execute.
        """
        self._lowp_precision = precision
        self._lowp_weight_cacheable = weight_cacheable
        return self._grouped_mm_enabled()

    def _compute_experts_with_grouped_mm(
        self, tokens: torch.Tensor, experts: torch.Tensor, weights: torch.Tensor, output_dtype: torch.dtype, compute_fn
    ) -> torch.Tensor:
        """Template for grouped GEMM expert compute (sort → autocast → compute_fn → weight multiply →
        scatter-back).

        ``compute_fn(sorted_tokens, offs, sorted_expert_ids)`` runs the grouped_mm projections + activation
        inside ``torch.amp.autocast`` (stops fp32 promotion from scalar attrs). ``F.grouped_mm`` does not
        auto-cast, so compute_fn must ``.to(output_dtype)`` its weights.
        """
        if tokens.shape[0] == 0:
            return self._expert_hook_grad_edge(
                torch.zeros((0, tokens.shape[-1]), device=tokens.device, dtype=output_dtype)
            )

        sorted_tokens, offs, sorted_token_idx, sorted_weights, sorted_expert_ids, inv_map = (
            self._sort_tokens_for_grouped_mm(tokens, experts, weights)
        )

        with torch.amp.autocast("cuda", dtype=output_dtype):
            expert_out = compute_fn(sorted_tokens, offs, sorted_expert_ids)

        expert_out = expert_out.to(output_dtype) * sorted_weights.unsqueeze(-1).to(output_dtype)
        if inv_map is not None:
            return MoEScatterUnpermute.apply(expert_out, sorted_token_idx, inv_map)
        output = torch.zeros((tokens.shape[0], tokens.shape[-1]), device=tokens.device, dtype=output_dtype)
        output.index_add_(0, sorted_token_idx, expert_out)
        return output

    def _separate_glu_experts_gmm(
        self, tokens: torch.Tensor, experts: torch.Tensor, weights: torch.Tensor, output_dtype: torch.dtype
    ) -> torch.Tensor:
        """Grouped GEMM for experts stored as SEPARATE GLU projections — 3 calls (gate, up, down).

        Used by families storing GLU halves separately (Qwen3, Bailing) and by the separate-storage
        branch of :meth:`_fused_glu_experts_gmm` (every family under ETP).
        """

        def compute(sorted_tokens, offs, _eids):
            gate = self._expert_proj(sorted_tokens, "gate_proj", offs, output_dtype)
            up = self._expert_proj(sorted_tokens, "up_proj", offs, output_dtype)
            activated = self._glu_combine(gate, up)
            return self._expert_proj(activated, "down_proj", offs, output_dtype)

        return self._compute_experts_with_grouped_mm(tokens, experts, weights, output_dtype, compute)

    def _fused_glu_experts_gmm(
        self, tokens: torch.Tensor, experts: torch.Tensor, weights: torch.Tensor, output_dtype: torch.dtype
    ) -> torch.Tensor:
        """Grouped GEMM for fused GLU experts: 2 calls (gate_up + down). Defers to
        :meth:`_separate_glu_experts_gmm` whenever the halves are stored separately — separate-projection
        families (Qwen3, Bailing) and every family under ETP (no fused ``gate_up_proj`` there)."""
        if not hasattr(self, "gate_up_proj"):
            return self._separate_glu_experts_gmm(tokens, experts, weights, output_dtype)

        def compute(sorted_tokens, offs, _eids):
            gate_up = self._expert_proj(sorted_tokens, "gate_up_proj", offs, output_dtype)
            gate, up = gate_up.chunk(2, dim=-1)
            activated = self._glu_combine(gate, up)
            return self._expert_proj(activated, "down_proj", offs, output_dtype)

        return self._compute_experts_with_grouped_mm(tokens, experts, weights, output_dtype, compute)

    def _etp_shard_size(self, intermediate: int) -> int:
        """Per-rank size of an expert intermediate dim sharded across the Expert-TP group.

        Raises if ``intermediate`` is not divisible by ``expert_tp_size`` — a non-divisible split would
        silently drop the top units of every expert (and produce a smaller gathered tensor on save)."""
        if intermediate % self.expert_tp_size != 0:
            raise ValueError(
                f"Expert TP requires the expert intermediate size ({intermediate}) to be divisible by "
                f"expert_tp_size ({self.expert_tp_size}); got remainder {intermediate % self.expert_tp_size}. "
                f"Reduce expert_tp_size."
            )
        return intermediate // self.expert_tp_size

    def _etp_narrow(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """This ETP rank's contiguous slice of ``tensor`` along the expert-intermediate axis ``dim``.

        The ``rank * shard`` offset arithmetic is identical for every expert layout — only WHICH axis
        carries the intermediate dim differs (it moves with the transpose each layout applies), so
        callers pass that and nothing else. Divisibility is checked by :meth:`_etp_shard_size`, so no
        path can slip an indivisible split through. Returns a view; callers materialize it.
        """
        shard = self._etp_shard_size(tensor.shape[dim])
        return tensor.narrow(dim, self.expert_tp_rank * shard, shard)

    def _store_separate_glu_params(self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> None:
        """Shard (under ETP) → transpose to matmul convention → store as ``gate_proj``/``up_proj``/
        ``down_proj``. Inputs are this rank's expert slice in ``F.linear`` convention: ``gate``/``up``
        ``[E_local, M, H]``, ``down`` ``[E_local, H, M]``.

        Shared by families whose experts arrive as separate GLU projections (Qwen3, Bailing); only the
        extraction is family-specific, the ETP shard + transpose + store is identical."""
        if self.expert_tp_size > 1:
            self.gate_proj = nn.Parameter(self._etp_narrow(gate, 1).transpose(1, 2).contiguous())
            self.up_proj = nn.Parameter(self._etp_narrow(up, 1).transpose(1, 2).contiguous())
            self.down_proj = nn.Parameter(self._etp_narrow(down, 2).transpose(1, 2).contiguous())
        else:
            self.gate_proj = nn.Parameter(gate.transpose(1, 2).contiguous())  # [E, H, M]
            self.up_proj = nn.Parameter(up.transpose(1, 2).contiguous())  # [E, H, M]
            self.down_proj = nn.Parameter(down.transpose(1, 2).contiguous())  # [E, M, H]

    @staticmethod
    def _require_fused_experts(experts: nn.Module) -> nn.Module:
        """``experts`` when it carries the fused ``gate_up_proj``/``down_proj`` pair, else raise.

        The fused slice below and the fused hidden-dim probe both read that pair. Skipping a renamed
        tensor instead would register NO expert weight on the wrapper — a layer that trains, saves and
        serves an empty expert set with no error anywhere.
        """
        missing = [name for name in ("gate_up_proj", "down_proj") if not hasattr(experts, name)]
        if missing:
            raise AttributeError(
                f"Expected fused-GLU experts carrying gate_up_proj and down_proj, but "
                f"{type(experts).__name__} carries no {missing} — the family renamed its expert "
                f"tensors upstream."
            )
        return experts

    def _init_fused_glu_params(self, experts: nn.Module, weights_already_sharded: bool = False):
        """Slice fused GLU expert parameters for this rank (stored matmul convention [E, K, N]).

        Non-ETP — fused storage matching the upstream checkpoint (``gate_up_proj [E, H, 2M]`` contiguous
        halves ``[gate | up]``, ``down_proj [E, M, H]``). ETP — gate/up stored separately so the
        intermediate dim shards coherently: slicing the fused halves along dim 2 by ``2M/tp`` would put
        the gate half on rank 0 and up half on rank 1, computing the GLU on misaligned pairs.

        ``weights_already_sharded=True``: experts pre-sliced by the lazy loader, so dim-0 slicing is skipped.
        """
        self._require_fused_experts(experts)
        if weights_already_sharded:
            start, end = 0, self.experts_per_rank
        else:
            start, end = self.expert_start, self.expert_end

        if self.expert_tp_size > 1:
            # Original [E, 2M, H] → matmul [E, H, 2M] with dim 2 = [gate(M) | up(M)]
            full_gate_up = experts.gate_up_proj.data[start:end].transpose(1, 2).contiguous()
            # Split the halves BEFORE sharding so each rank holds matching gate/up positions.
            gate_half, up_half = full_gate_up.chunk(2, dim=2)
            self.gate_proj = nn.Parameter(self._etp_narrow(gate_half, 2).contiguous())
            self.up_proj = nn.Parameter(self._etp_narrow(up_half, 2).contiguous())

            # Original [E, H, M] → matmul [E, M, H] → shard dim 1 to the same intermediate positions.
            full_down = experts.down_proj.data[start:end].transpose(1, 2).contiguous()
            self.down_proj = nn.Parameter(self._etp_narrow(full_down, 1).contiguous())
        else:  # [E, 2M, H] → matmul [E, H, 2M]; [E, H, M] → matmul [E, M, H]
            self.gate_up_proj = nn.Parameter(experts.gate_up_proj.data[start:end].transpose(1, 2).contiguous())
            self.down_proj = nn.Parameter(experts.down_proj.data[start:end].transpose(1, 2).contiguous())

    def expert_named_params(self) -> list:
        """``(name, param)`` tuples for this layer's distributed expert weights + LoRA adapters.

        Single source of truth for expert params (grad-sync, FP32 upcast, checkpoint skip-set). Base weights
        from :meth:`_base_expert_named_params`; LoRA adapters appended here. Don't override this — override
        :meth:`_base_expert_named_params`.
        """
        return self._base_expert_named_params() + self._expert_lora_named_params()

    def _base_expert_named_params(self) -> list:
        """``(name, param)`` for this layer's family base expert weights (no LoRA).

        Derived by hasattr-filtering :attr:`_EXPERT_WEIGHT_ATTR_ROOTS` to the attributes this rank holds;
        the stored layout (fused-GLU, separate/ETP, or GptOss variants) selects the subset automatically.
        Families with extra layouts override the class attribute, not this method."""
        return [(n, getattr(self, n)) for n in self._EXPERT_WEIGHT_ATTR_ROOTS if hasattr(self, n)]

    @staticmethod
    def _submodule_named_params(named_modules) -> list:
        """Flatten ``(prefix, submodule)`` pairs into ``(prefix.param_name, param)`` tuples, skipping
        ``None`` submodules. Helper for per-family ``*_named_params`` overrides."""
        out = []
        for prefix, mod in named_modules:
            if mod is not None:
                out += [(f"{prefix}.{n}", p) for n, p in mod.named_parameters()]
        return out

    def replicated_named_params(self) -> list:
        """``(name, param)`` for this layer's full REPLICATED submodules (a shared expert + its gate) that
        run on the local batch and are FSDP-ignored, so (like the router) they need DP-average grad sync;
        else they drift across DP ranks.

        Derived from :attr:`_shared_expert_attr`, so a family carrying the ordinary single shared
        expert needs no override — it just names the attribute. Empty when that attribute is absent
        or ``None`` (a family with no shared expert; LFM2 sets it ``None``). Only a family whose
        replicated set is a different SHAPE overrides — Qwen3.5/3.6, whose gate is a second module.
        """
        return self._submodule_named_params(
            [(self._shared_expert_attr, getattr(self, self._shared_expert_attr, None))]
        )

    @classmethod
    def _find_gate_or_router(cls, layer: nn.Module) -> nn.Module | None:
        """The HF MoE layer's router module, read under this family's declared :attr:`_ROUTER_ATTR`,
        or ``None`` where the family declares its router EXTERNAL to the wrapper (Gemma4).

        Not a ``("gate", "router")`` probe: a block exposing both would resolve to whichever name the
        probe tried first, silently overriding the family's own declaration.
        """
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
    def _find_experts_container(cls, layer: nn.Module) -> nn.Module:
        """The MoE block's expert container, under the first of this family's declared
        :attr:`_EXPERTS_CONTAINER_ATTRS` the block carries."""
        for attr in cls._EXPERTS_CONTAINER_ATTRS:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        raise AttributeError(
            f"{cls.__name__} declares _EXPERTS_CONTAINER_ATTRS={cls._EXPERTS_CONTAINER_ATTRS}, but "
            f"{type(layer).__name__} carries none of them — the family renamed its expert container "
            f"upstream. Update _EXPERTS_CONTAINER_ATTRS."
        )

    @classmethod
    def detect_num_experts(cls, layer: nn.Module) -> int:
        """Detect the routed-expert count from the original HF MoE module.

        Resolves this family's :attr:`_NUM_EXPERTS_ATTR_PATHS` in order, then the generic probe derived
        from :attr:`_EXPERTS_CONTAINER_ATTRS`. One implementation for every family — the per-arch
        attribute is data.
        """
        generic = tuple(p for attr in cls._EXPERTS_CONTAINER_ATTRS for p in (attr, f"{attr}.num_experts"))
        for path in (*cls._NUM_EXPERTS_ATTR_PATHS, *generic):
            count = _resolve_expert_count(layer, path)
            if count is not None:
                return count
        raise AttributeError(f"Cannot detect num_experts from {type(layer).__name__}")

    def _resolve_activation(self, *sources, original_layer: nn.Module | None = None, default: str = "silu") -> None:
        """Set ``self.act_fn`` from the first source that carries one, and latch the fused-GLU gate.

        ``sources`` are searched in order — normally the HF experts module, then its parent. Only if
        none carries a live ``act_fn`` is the activation rebuilt by NAME, resolved off the wrapped
        block ``original_layer``: its own ``hidden_act`` first, then its ``config``, then ``default``.
        One home for both halves, so a family whose experts module was renamed upstream cannot mean
        ``KeyError`` in one wrapper and a silent SiLU in the next.

        The fused-combine latch is DERIVED here from the resolved activation via
        :func:`resolve_fused_glu_mul`, the toolkit's single answer to "which kernel computes this
        activation". Deriving it means a family opts in by having a gate the kernels implement rather
        than by remembering to set a flag, and any other gate fails closed onto the eager combine.
        """
        for source in sources:
            act_fn = getattr(source, "act_fn", None) if source is not None else None
            if act_fn is not None:
                self.act_fn = act_fn
                break
        else:
            hidden_act = getattr(original_layer, "hidden_act", None)
            if hidden_act is None:
                hidden_act = getattr(getattr(original_layer, "config", None), "hidden_act", None)
            self.act_fn = ACT2FN[hidden_act or default]
        self._fused_glu_mul = resolve_fused_glu_mul(self.act_fn)

    @classmethod
    def _find_top_k(cls, layer: nn.Module) -> int:
        """Router top-k from a MoE layer, then from the router under its declared :attr:`_ROUTER_ATTR`.

        Uses :data:`ROUTER_TOPK_FIELDS` rather than a local pair of names, so this agrees with every
        other consumer (load metrics, the pipeline split's FFN cost model) on both the SPELLINGS and
        their ORDER. Order is load-bearing: the registry puts bare ``top_k`` last because that name is
        also the generation sampling parameter, and it carries ``top_k_experts`` (Gemma4) and
        ``moe_router_topk``, which a two-name probe misses entirely.

        The router falls under the family's own declaration for the same reason
        :meth:`_find_gate_or_router` refuses a ``("gate", "router")`` probe: a block exposing both
        would otherwise resolve to whichever name the probe tried first.
        """
        router = None if cls._ROUTER_ATTR is None else getattr(layer, cls._ROUTER_ATTR, None)
        for source in (layer, router):
            for attr in ROUTER_TOPK_FIELDS:
                if source is not None and hasattr(source, attr):
                    return getattr(source, attr)
        raise AttributeError(f"Cannot determine top_k from {type(layer).__name__}")

    def _upcast_router_to_fp32(self, router_attr: str) -> None:
        """Upcast the router/gate under ``fp32_router`` (no-op otherwise). ``.float()`` covers a bare
        parameter and a full router module (params + floating buffers) alike."""
        if not self.fp32_router:
            return
        setattr(self, router_attr, getattr(self, router_attr).float())
        if is_global_main_process():
            logger.info(f"{type(self).__name__}: FP32 router (compute stays BF16)")

    def _upcast_experts_to_fp32(self, expert_params: list) -> None:
        """Upcast expert weights (incl. grouped-LoRA adapters) under ``fp32_experts`` (no-op otherwise).

        Gated like :attr:`_fp32_router_input`: when the experts are FSDP-MANAGED
        (``fsdp_shard_ep1_experts`` at ``ep_group_size == 1``) the FSDP2 bf16 ``param_dtype`` policy
        casts them for compute anyway, so the flag would silently degrade to fp32-master/bf16-compute
        while mixing fp32 experts into a bf16 ``fully_shard`` group — skip and log instead
        (fp32 master weights in that state need ``fp32_non_ep_params`` WITH
        ``fsdp_shard_ep1_experts: false`` — the managed-experts combination is refused at config time).

        Rebuilding an ``nn.Parameter`` defaults ``requires_grad=True``, so the prior flag is preserved
        explicitly — this runs AFTER ``_init_expert_lora`` froze the LoRA base weights, and re-enabling
        them would silently train the frozen base."""
        if not self.fp32_experts:
            return
        if self.ep_config.experts_fsdp_managed:
            logger.warning(
                f"{type(self).__name__}: fp32_experts has NO effect with fsdp_shard_ep1_experts at "
                f"ep_group_size==1 — FSDP2 manages the replicated experts and its bf16 param_dtype "
                f"policy casts them for compute, so the upcast would only mix fp32 experts into a bf16 "
                f"fully_shard group. Skipping; for fp32 master weights set fp32_non_ep_params: true "
                f"AND fsdp_shard_ep1_experts: false (the managed-experts combination is refused at "
                f"config time)."
            )
            return
        for name, param in expert_params:
            setattr(self, name, nn.Parameter(param.data.float(), requires_grad=param.requires_grad))
        if is_global_main_process():
            logger.info(f"{type(self).__name__}: FP32 experts (compute stays BF16)")

    def _finalize_expert_init(self) -> None:
        """The init tail every family wrapper shares: device placement → expert LoRA → fp32 upcast →
        gradient-sync hooks, keyed off :attr:`_ROUTER_ATTR`.

        Order is load-bearing. LoRA must exist before the fp32 upcast walks the expert params (the
        upcast preserves the ``requires_grad`` the LoRA freeze just set), and both the router and the
        expert params are re-read AFTER their upcast: the upcast rebinds each attribute to a NEW fp32
        tensor, so a reference taken earlier would hook the discarded one.

        A family whose router is external (``_ROUTER_ATTR = None``) registers expert hooks only: the
        sibling router stays FSDP-managed, which is also what DP-syncs it, and its dtype cannot be
        raised under EP at all — DeepEP's combine is bf16-only.
        """
        self._to_device()
        self._init_expert_lora()
        if self._ROUTER_ATTR is not None:
            self._upcast_router_to_fp32(self._ROUTER_ATTR)
        elif self.fp32_router:
            logger.warning(
                f"{type(self).__name__}: fp32_router has NO effect — this family's router is a "
                f"sibling module outside the EP wrapper (FSDP-managed). Its dtype cannot be raised "
                f"under EP: fp32_non_ep_params would upcast it, and DeepEP's combine is bf16-only. "
                f"The router stays bf16."
            )
        self._upcast_experts_to_fp32(self.expert_named_params())
        # An empty placeholder for an external router, so only expert hooks register.
        router = getattr(self, self._ROUTER_ATTR) if self._ROUTER_ATTR is not None else nn.Module()
        self._setup_gradient_sync(router, self.expert_named_params())

    def _compute_experts_gmm(
        self, tokens: torch.Tensor, experts: torch.Tensor, weights: torch.Tensor, output_dtype: torch.dtype
    ) -> torch.Tensor:
        """Grouped-GEMM expert compute — the grouped branch of :meth:`_compute_experts`. Default = the GLU
        path (fused contiguous halves, or 3 calls when stored separately). Override for GptOss's layout."""
        return self._fused_glu_experts_gmm(tokens, experts, weights, output_dtype)

    def _compute_experts(
        self, tokens: torch.Tensor, experts: torch.Tensor, weights: torch.Tensor, output_dtype: torch.dtype
    ) -> torch.Tensor:
        """Compute expert forward on dispatched tokens: grouped GEMM when enabled and non-empty, else the
        per-expert weighted loop. Shared control flow — per-family degrees of freedom are the grouped gate
        (:meth:`_grouped_mm_enabled`) and the compute kernels, not this dispatch.

        ``experts``/``weights`` are ``[T, top_k]`` — local expert ids and gate weights per dispatched
        token, one row per token. Every family's forward passes that shape, DeepEP preserves it
        (``ep_size == 1`` returns the inputs unchanged), and both compute templates index the top-k
        axis, so a flat ``[T]`` selection is not a supported input to either.
        """
        if self._grouped_mm_enabled() and tokens.shape[0] > 0:
            return self._compute_experts_gmm(tokens, experts, weights, output_dtype)
        return self._compute_experts_weighted(tokens, experts, weights, output_dtype, self._expert_forward)

    def _dispatch_compute_combine(
        self, flat: torch.Tensor, experts: torch.Tensor, weights: torch.Tensor, input_dtype: torch.dtype
    ) -> torch.Tensor:
        """Dispatch → expert TP scatter-gather → compute → combine. Shared control flow for all EP layers.

        **Expert-TP all-reduce placement (token space, NOT recv space).** The sum runs OUTSIDE the
        dispatch→combine span (SumGradAcrossGroup pre-dispatch, ReduceFromExpertTP post-combine). Correct because
        the combine is linear in expert outputs, so ``combine(sum) == sum(combine)``.

        MUST stay outside the span: with ep_size>1 AND expert_tp_size>1 the EP group splits into multiple
        DeepEP dispatch groups coupled by strided expert-TP groups. An all-reduce BETWEEN dispatch and
        combine couples the dispatch groups inside DeepEP's intranode combine barrier; under FSDP2
        multi-stream drift they form a circular wait and the barrier times out. At the layer boundary each
        dispatch group runs independently.
        """
        # Before the dispatch, never inside the span: a first-call trace between dispatch and combine
        # stalls every peer in DeepEP's barrier.
        self._warm_activation_graphs(flat.device, input_dtype)

        if self._capture_routing:
            # Capture the GLOBAL expert ids before dispatch localizes them; int16 (num_experts < 32768).
            self._captured_routing_chunks.append(experts.detach().to(torch.int16))

        if self.expert_tp_size > 1:
            # BOTH operands need the ETP backward sum; drop either and the router (plus everything upstream)
            # silently trains on a 1/expert_tp_size gradient.
            flat = SumGradAcrossGroup.apply(flat, self.expert_tp_group)
            weights = SumGradAcrossGroup.apply(weights, self.expert_tp_group)

        with self._perf("ep.dispatch"):
            recv_x, recv_topk_idx, recv_topk_weights, handle = self._gc_dispatch(flat, experts, weights)

        with self._perf("ep.expert_compute"):
            output = self._compute_experts(recv_x, recv_topk_idx, recv_topk_weights, input_dtype)

        with self._perf("ep.combine"):
            combined = self._gc_combine(output, recv_topk_weights, handle)

        if self.expert_tp_size > 1:
            combined = ReduceFromExpertTP.apply(combined, self.expert_tp_group)

        return combined

    def _dispatch_compute_combine_shared(
        self,
        flat: torch.Tensor,
        experts: torch.Tensor,
        weights: torch.Tensor,
        input_dtype: torch.dtype,
        orig_shape: torch.Size,
        shared_fn: Callable[[], torch.Tensor] | None,
    ) -> torch.Tensor:
        """Routed dispatch→compute→combine reshaped to ``orig_shape``, plus the shared-expert output.

        ``shared_fn`` returns the shared-expert output (``None`` for families without one). The shared FFN
        reads only the layer input, so with ``HALO_EP_SHARED_OVERLAP`` set it runs on a side stream
        concurrent with the latency-bound dispatch all-to-all, else sequentially. Numerically identical.
        """
        if shared_fn is None:
            return self._dispatch_compute_combine(flat, experts, weights, input_dtype).view(*orig_shape)

        if _SHARED_OVERLAP_ENABLED and flat.is_cuda:
            side = _shared_overlap_stream(flat.device)
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

    @abstractmethod
    def forward(self, hidden_states: torch.Tensor, **kwargs): ...


class EPSharedExpertsMoELayerBase(EPMoELayerBase):
    """EP MoE layer whose ``forward`` is the standard fp32-router → route → shared-experts path.

    Subclasses supply ``route_tokens_to_experts``; the construction template already adopted the
    ``gate`` and the shared-expert module this forward reads under :attr:`_shared_expert_attr`
    (``None`` where the family has none — LFM-2). ``_router_logits`` and ``forward`` stay overridable
    for a family whose router cannot be called bare (LFM-2) or whose combine scales the result
    (Cohere2). Families with a different router contract (Qwen3.5, Bailing) subclass
    ``EPMoELayerBase`` and keep their own ``forward``.
    """

    def _router_logits(self, flat: torch.Tensor) -> torch.Tensor:
        """Family-native router logits for the flattened tokens.

        The default calls the gate module (nn.Linear convention). Some remote-code routers return
        ``(logits, weights, indices)`` — EP must recompute the latter two after routing replay /
        balancing, but can reuse the family-native (possibly soft-capped) logits. A family whose
        router module cannot be called bare (LFM2's requires the block's ``expert_bias``) overrides
        this with the logits computation alone.
        """
        # Disable autocast so the gate matmul runs at the input dtype (upcast when fp32_router).
        with torch.amp.autocast("cuda", enabled=False):
            router_output = self.gate(flat.float() if self._fp32_router_input else flat)
        return router_output[0] if isinstance(router_output, tuple) else router_output

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        orig_shape = hidden_states.shape
        _B, _S, H = orig_shape
        input_dtype = hidden_states.dtype

        residuals = hidden_states  # shared experts read the layer input, not the routed output
        flat = hidden_states.view(-1, H)

        topk_indices, topk_weights = self.route_tokens_to_experts(self._router_logits(flat).float())

        shared_expert = getattr(self, self._shared_expert_attr, None)
        shared_fn = (lambda: shared_expert(residuals)) if shared_expert is not None else None
        return self._dispatch_compute_combine_shared(
            flat, topk_indices.long(), topk_weights.float(), input_dtype, orig_shape, shared_fn
        )


class EPGroupLimitedMoELayerBase(EPSharedExpertsMoELayerBase):
    """EP MoE layer routing DeepSeek-V3 style: score → selection-only bias → group-limited top-k.

    One routing body for the families that share it (GLM-4 MoE Lite and Laguna, GLM-5 Next,
    Mistral4, Step-3.7). They differ in exactly three declarations: the score function
    (:meth:`_routing_scores`), the ``norm_topk_prob`` fallback for a block that declares none, and
    the floor its weight renormalization adds to the sum. Everything else — which tensor biases
    selection, in what order, and which scores the weights come from — is shared, because a
    mis-ordered bias add changes routing with no shape, dtype or key moving.

    ``n_group: 1`` degenerates to plain top-k, so a family with no group limiting at all (Step-3.7)
    is this class with the knob absent rather than a second implementation.
    """

    # Fallback when neither the block, its router nor its config declares ``norm_topk_prob`` — GLM-4
    # MoE Lite leaves the gate weights un-normalized, the others normalize.
    _NORM_TOPK_PROB_DEFAULT: bool = True

    # Floor added to the top-k weight sum before dividing (DeepSeek-V3's own 1e-20). Step-3.7
    # renormalizes with none; adding one there would change every routed weight it emits.
    _TOPK_WEIGHT_NORM_EPS: float = 1e-20

    # Routing knobs (any spelling) this family's block, router and config genuinely do not declare —
    # Laguna and Step-3.7 carry no group limiting and no ``norm_topk_prob`` at all. Every OTHER knob
    # is REQUIRED: absence means the family renamed it upstream, and the neutral default is then a
    # live wrong number — routed scaling silently 1.0 rather than the block's 2.5, every routed weight
    # under-scaled with no attribute missing — so the resolution below raises instead of defaulting.
    _OPTIONAL_ROUTING_KNOBS: tuple[str, ...] = ()

    def _init_routing(self, original_layer: nn.Module) -> None:
        """Read the six knobs :meth:`_group_limited_topk` consumes: ``n_routed_experts`` /
        ``n_group`` / ``topk_group`` / ``norm_topk_prob`` / ``routed_scaling_factor`` / ``top_k``.

        Writer and reader live in one class, so a family cannot half-populate the contract. Each is
        resolved off the block, then its router, then its config — the three places the roster's
        blocks hang them (5.14 moved most onto the ``*TopkRouter``, and Laguna's remote-code
        revisions put ``norm_topk_prob`` there while omitting the group knobs entirely) — and a knob
        no source declares raises unless the family listed it in :attr:`_OPTIONAL_ROUTING_KNOBS`.
        """
        super()._init_routing(original_layer)  # top_k
        sources = (original_layer, self.gate, getattr(original_layer, "config", None))

        def _resolve(names: str | tuple[str, ...], default, *, required: bool = True):
            spellings = (names,) if isinstance(names, str) else names
            for name in spellings:
                for source in sources:
                    if source is not None and hasattr(source, name):
                        return getattr(source, name)
            # Any spelling opts the knob out: a family whose router only carries the alias must be
            # able to declare it, and the two names are one knob.
            if required and not set(spellings) & set(self._OPTIONAL_ROUTING_KNOBS):
                raise AttributeError(
                    f"{type(self).__name__} resolved none of {spellings} off "
                    f"{type(original_layer).__name__}, its router or its config — the family renamed "
                    f"the knob upstream. Defaulting to {default!r} would silently change every routed "
                    f"weight this layer emits. Declare it in _OPTIONAL_ROUTING_KNOBS if this family "
                    f"genuinely has no such knob."
                )
            return default

        # Not required anywhere: its fallback is read off the live expert container rather than a
        # constant, so a rename cannot turn it into a wrong number.
        self.n_routed_experts = _resolve("n_routed_experts", self.num_experts, required=False)
        # The ``*TopkRouter`` modules spell it ``num_group``, configs ``n_group``; missing it
        # silently disables group-limited routing.
        self.n_group = _resolve(("n_group", "num_group"), 1)
        self.topk_group = _resolve("topk_group", 1)
        self.norm_topk_prob = _resolve("norm_topk_prob", self._NORM_TOPK_PROB_DEFAULT)
        self.routed_scaling_factor = float(_resolve("routed_scaling_factor", 1.0))

    def _routing_scores(self, router_logits: torch.Tensor) -> torch.Tensor:
        """The family's router score function over its logits — the term BOTH selection and the gate
        weights read. Sigmoid is the DeepSeek-V3 form; a softmax-scored family (Mistral4) overrides."""
        return router_logits.sigmoid()

    def route_tokens_to_experts(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(topk_indices, topk_weights)`` — selection on the biased scores, weights on the unbiased."""
        scores = self._routing_scores(router_logits)
        topk_indices, topk_weights = self._group_limited_topk(self._selection_scores(scores), scores)
        self._record_expert_load(topk_indices)
        return topk_indices, topk_weights

    def _group_limited_topk(
        self, scores_for_select: torch.Tensor, gate_scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """DeepSeek-V3 group-limited top-k expert selection.

        ``scores_for_select`` (carries any correction/balancing bias) drives group masking + top-k *index*
        selection; ``gate_scores`` (unbiased) supplies the *weights* gathered at those indices. Returns
        ``(topk_indices, topk_weights)``.
        """
        if self.n_group > 1:
            experts_per_group = self.n_routed_experts // self.n_group
            group_scores = scores_for_select.view(-1, self.n_group, experts_per_group).topk(2, dim=-1)[0].sum(dim=-1)
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1).expand(-1, self.n_group, experts_per_group).reshape(-1, self.n_routed_experts)
            )
            # -inf, not 0.0: scores are sigmoid(logits)+bias (positive), so a 0.0 fill leaves a group beatable.
            scores_for_select = scores_for_select.masked_fill(~score_mask.bool(), float("-inf"))

        topk_indices = torch.topk(scores_for_select, k=self.top_k, dim=-1, sorted=False)[1]
        topk_indices = self._maybe_replace_selection(topk_indices)
        topk_weights = gate_scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + self._TOPK_WEIGHT_NORM_EPS)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class EPSeparateGluMoELayerBase(EPMoELayerBase):
    """EP MoE layer holding its experts as SEPARATE gate/up/down 3-D tensors, exported per expert.

    The two halves of that statement travel together: the family's HF checkpoint stores one tensor
    per expert (:attr:`_HUB_PER_EXPERT_KEYS`), and the wrapper keeps the GLU halves apart — Qwen3
    splits the pre-fused pair at init, Bailing stacks a per-expert ``nn.ModuleList`` — so the gather
    and its shard-merge inverse are the separate-halves pair for both. Declared once here: as a copy
    per family, a fix landing in one silently desynchronizes the other family's gathered save from
    its own sharded merge.
    """

    def gather_expert_state_dict(self, device: str = "cpu", merge_lora: bool = False, retain: bool = True) -> dict:
        """These experts are written per-expert (unfused) in the checkpoint."""
        return self._gather_individual_glu_state_dict(device, merge_lora=merge_lora, retain=retain)

    @classmethod
    def merge_shards_to_hf(cls, prefix: str, params: dict) -> dict:
        """Inverse of the gather above: separate 3D shards → per-expert keys (see the base method)."""
        state = cls._merge_individual_glu_shards(params)
        return {f"{prefix}.{key}": tensor for key, tensor in state.items()}


def find_ep_layers(model: nn.Module) -> list[tuple[str, EPMoELayerBase]]:
    """This model's EP MoE layers as ``(module path, layer)``, in ``named_modules`` order.

    The single spelling of "which modules does the expert gather own": the gathered and sharded EP
    saves, a pipeline stage's export set and the optimizer's expert-replica writer all key off it, and
    a divergent walk would put a layer in one of those lists and not another.
    """
    return [(name, module) for name, module in model.named_modules() if isinstance(module, EPMoELayerBase)]


@contextlib.contextmanager
def disable_expert_adapters(model: nn.Module):
    """Temporarily disable native EP expert-LoRA deltas across every EP layer in ``model``.

    EP-side complement to peft's ``PeftModel.disable_adapter()``: the native grouped expert adapters aren't
    peft-managed, so disabling peft alone leaves them active. Used for the KL reference pass.
    """
    layers = [m for m in model.modules() if isinstance(m, EPMoELayerBase) and m._expert_lora_attrs]
    previous = [m._expert_adapters_enabled for m in layers]
    for layer in layers:
        layer._expert_adapters_enabled = False
    try:
        yield
    finally:
        for layer, prev in zip(layers, previous, strict=True):
            layer._expert_adapters_enabled = prev


def make_disable_adapter_ep_aware(peft_model) -> None:
    """Wrap ``peft_model.disable_adapter`` so it ALSO disables native EP expert adapters.

    Without this, a KL reference pass reverts only the peft-managed attention LoRA, leaving the EP expert
    adapters active so the reference isn't a true frozen base. Idempotent; call once after EP + PEFT setup.
    """
    if getattr(peft_model, "_ep_disable_adapter_patched", False):
        return
    original_disable_adapter = peft_model.disable_adapter

    @contextlib.contextmanager
    def _ep_aware_disable_adapter():
        with original_disable_adapter(), disable_expert_adapters(peft_model):
            yield

    peft_model.disable_adapter = _ep_aware_disable_adapter
    peft_model._ep_disable_adapter_patched = True
