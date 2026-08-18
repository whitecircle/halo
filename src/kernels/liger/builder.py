"""Role-driven Liger applier builder, shared by every family the toolkit covers.

A :class:`LigerFamilySpec` names the classes filling each role; the roles it leaves empty are absent from
the built applier's signature, which the orchestrator reads for per-family defaults. Each role is patched
by subclassing the family's own class, which avoids the constructor variance that blocks Liger's own
class swaps and preserves the original class name that downstream lookups key on.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from types import MethodType, ModuleType

import torch
import torch.nn as nn
from accelerate.logging import get_logger
from liger_kernel.transformers import LigerRMSNorm
from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN
from liger_kernel.transformers.monkey_patch import _patch_rms_norm_module
from liger_kernel.transformers.rope import liger_rotary_pos_emb

from src.kernels.fused_glu import resolve_fused_glu_mul
from src.kernels.liger.cross_entropy import patch_loss_utils_cross_entropy
from src.kernels.liger.lce_forward import build_lce_forward
from src.kernels.liger.remote_modules import patch_remote_modules

logger = get_logger(__name__)

# Marks a class the builder produced, so a second application is a no-op instead of stacking
# subclasses (the orchestrator patches at model load and TRL may re-apply on the instance).
_PATCHED_MARKER = "_halo_liger_patched_role"

# The resolved fused activation-and-multiply, latched per module at construction.
_GLU_MUL_ATTR = "_halo_glu_mul"

# The flag gating the gated MLP of every toolkit-patched family. Liger spells the switch twice
# (`swiglu`/`geglu`) and the orchestrator defaults and overrides both together; which kernel a module
# gets is resolved from its own activation by `resolve_fused_glu_mul`, not from the spelling.
_GLU_FLAG = "swiglu"

# The gates `fla`'s fused gated norm implements. Its kernel dispatches on the string with no else
# branch, so a value it does not know applies no gate at all.
_FLA_GATE_ACTIVATIONS = frozenset({"swish", "silu", "sigmoid"})


@dataclass(frozen=True)
class LigerFamilySpec:
    """The Liger-patchable surface of one model family, by role.

    ``modeling_module`` is a dotted path for a family native to transformers; a remote-code family
    leaves it empty and names ``remote_classes`` instead, which drives the deferred patch in
    :mod:`~src.kernels.liger.remote_modules`.

    A family upstream Liger already covers sets ``delegates_to_upstream``: its applier keeps every
    role it declares, and the spec names only the roles the toolkit adds on top.
    """

    model_types: tuple[str, ...]
    modeling_module: str = ""
    # Norm classes whose forward is exactly Liger's `(offset + w) * x * rsqrt(mean(x²) + eps)`.
    # A family's other norms (weightless, grouped, gated, mean-subtracting) must not be listed:
    # LigerRMSNorm cannot express them and the swap would change the model's numerics.
    rms_norm: tuple[str, ...] = ()
    # "llama" upcasts x to fp32 and casts back before the weight multiply; "gemma" keeps the weight
    # multiply in fp32. `offset` is the constant added to the weight (Gemma-style `(1 + w)`).
    rms_norm_casting_mode: str = "llama"
    rms_norm_offset: float = 0.0
    # Gated norm classes: `norm(x) * weight * act(gate)` over the last dim, as the linear-attention
    # (GDN) blocks apply to their attention output. Served by `fla`'s fused kernel, not Liger's, and
    # patched under the same `rms_norm` flag. A grouped gated norm (Bailing's, which reduces over
    # `hidden_size // group_norm_size`) is a different function and must not be listed.
    gated_rms_norm: tuple[str, ...] = ()
    # MLP classes whose forward is exactly `down(act(gate(x)) * up(x))` over `gate_proj`/`up_proj`/
    # `down_proj`. A clamped, scaled or differently-named GLU must not be listed.
    glu_mlp: tuple[str, ...] = ()
    # `*ForCausalLM` classes whose forward the generic fused loss reproduces exactly.
    causal_lm: tuple[str, ...] = ()
    # Attribute on the causal-LM module holding a scalar applied to the logits before the loss.
    logit_scale_attr: str | None = None
    # Whether Liger's generic rotary computes this family's `apply_rotary_pos_emb`. False for every
    # partial, interleaved, mrope, YARN or per-layer-type rotary.
    rope: bool = False
    # Per-family loss default. True mirrors Liger's own (FLCE on, CE off) for the families whose
    # vocab makes the logits plane the binding memory constraint.
    flce_default: bool = False
    # Remote-code (`trust_remote_code`) families: the class names that identify the modeling module
    # once transformers loads it. Empty for a native family.
    remote_classes: tuple[str, ...] = ()
    # Upstream Liger covers these model types: its applier runs first with every flag, and the roles
    # above are added on top. Which roles it serves is read off Liger's registry, not restated here.
    delegates_to_upstream: bool = False

    def __post_init__(self) -> None:
        if bool(self.modeling_module) == bool(self.remote_classes):
            raise ValueError(
                f"LigerFamilySpec for {self.model_types} must set exactly one of modeling_module "
                f"(native) or remote_classes (trust_remote_code)."
            )
        if self.logit_scale_attr and not self.causal_lm:
            raise ValueError(f"LigerFamilySpec for {self.model_types} declares a logit scale but no causal_lm")
        # A delegating spec needs a module to patch and something to add; a role upstream also claims
        # is caught by :func:`_named_class`, on the class it replaced.
        if self.delegates_to_upstream and not (
            self.modeling_module and (self.rms_norm or self.gated_rms_norm or self.glu_mlp or self.causal_lm)
        ):
            raise ValueError(
                f"LigerFamilySpec for {self.model_types} delegates to upstream Liger but adds no role "
                f"of its own to a native modeling module; drop it and resolve on the upstream registry."
            )


def _upstream_applier(spec: LigerFamilySpec) -> Callable:
    """The upstream Liger applier a delegating spec extends, read off Liger's own registry.

    Every alias of a family maps to one function there; two would mean the spec groups model types
    upstream treats as separate families, and one of them would be patched with the other's applier.
    """
    appliers = {model_type: MODEL_TYPE_TO_APPLY_LIGER_FN.get(model_type) for model_type in spec.model_types}
    missing = sorted(model_type for model_type, applier in appliers.items() if applier is None)
    if missing:
        raise ValueError(
            f"LigerFamilySpec for {spec.model_types} delegates to upstream Liger, which registers no "
            f"applier for {missing}. Declare the family's own roles instead, or drop those model types."
        )
    distinct = set(appliers.values())
    if len(distinct) != 1:
        raise ValueError(
            f"LigerFamilySpec for {spec.model_types} spans {sorted(fn.__qualname__ for fn in distinct)}; "
            f"split it so each spec delegates to one upstream applier."
        )
    return distinct.pop()


def _named_class(module: ModuleType, name: str, spec: LigerFamilySpec) -> type:
    """The class ``name`` in ``module``, raising with the family named if it is absent."""
    cls = getattr(module, name, None)
    if not isinstance(cls, type):
        raise AttributeError(
            f"Liger applier for {spec.model_types[0]}: {module.__name__} defines no class {name!r}. "
            f"The family's modeling module changed; update its LigerFamilySpec."
        )
    # Upstream's applier has already run: a class it replaced is a role both claim, and subclassing
    # the replacement would stack the two swaps. `_rebrand` keeps the toolkit's swaps on the
    # family's own module.
    if spec.delegates_to_upstream and cls.__module__ != module.__name__:
        raise ValueError(
            f"Liger applier for {spec.model_types[0]}: {module.__name__}.{name} is now "
            f"{cls.__module__}.{cls.__qualname__} — upstream claims the class this spec also names. "
            f"Drop that role; two swaps on one class stack silently."
        )
    return cls


def _rebrand(patched: type, original: type, role: str) -> type:
    """Give the subclass the original's identity so name-keyed lookups elsewhere still resolve.

    EP wrapper discovery, the PP spec registry and the module-structure helpers key on
    ``type(module).__name__``; a subclass carrying its own name would drop out of all of them while
    remaining an ``isinstance`` of the original.
    """
    patched.__name__ = original.__name__
    patched.__qualname__ = original.__qualname__
    patched.__module__ = original.__module__
    setattr(patched, _PATCHED_MARKER, role)
    return patched


def _liger_rms_norm_class(original: type, *, offset: float, casting_mode: str) -> type:
    """``original`` with Liger's fused RMSNorm forward and the parameters that select its variant.

    ``in_place`` follows Liger's own convention: the llama casting mode reuses the incoming gradient
    buffer, the gemma one (weight multiply in fp32) does not. Reuse is correct only while no family adds
    a norm output straight into a residual, since ``AddBackward`` hands both inputs the same gradient
    object. Every family here is pre-norm, which
    ``tests/cpu/kernels/test_liger_family_coverage.py`` checks.
    """
    in_place = casting_mode != "gemma"

    class _LigerRMSNorm(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Families spell eps either way; Liger's forward reads `variance_epsilon`. A third
            # spelling would hand `None` to the Triton kernel, which fails inside the launcher
            # without naming the family.
            eps = getattr(self, "variance_epsilon", None)
            if eps is None:
                eps = getattr(self, "eps", None)
            if eps is None:
                raise AttributeError(
                    f"{type(self).__name__} exposes neither `variance_epsilon` nor `eps`; Liger's "
                    f"fused RMSNorm cannot be given an epsilon. Drop it from the family's rms_norm spec."
                )
            self.variance_epsilon = eps
            self.offset = offset
            self.casting_mode = casting_mode
            self.in_place = in_place
            self.row_mode = None

        forward = LigerRMSNorm.forward

    return _rebrand(_LigerRMSNorm, original, "rms_norm")


def _bridge_gated_norm(module: nn.Module) -> None:
    """Give one gated-norm module the attribute spellings ``fla``'s forward reads, or raise.

    The kernel reads ``eps``, ``weight``, ``bias`` and ``activation``; the families spell the epsilon
    ``variance_epsilon`` and register no bias. The activation is passed through unchanged, so the fused
    gate is the one the eager module declared.
    """
    name = type(module).__name__
    eps = getattr(module, "variance_epsilon", None)
    if eps is None:
        eps = getattr(module, "eps", None)
    if eps is None:
        raise AttributeError(f"{name} exposes neither `variance_epsilon` nor `eps` for fla's gated norm")
    activation = getattr(module, "activation", None)
    if activation not in _FLA_GATE_ACTIVATIONS:
        raise ValueError(
            f"{name} gates with {activation!r}, which fla's kernel does not implement — it would apply "
            f"no gate at all. Drop it from the family's gated_rms_norm spec."
        )
    # fla applies a bias when one is present; these norms have none, and gaining one would change
    # the function.
    if getattr(module, "bias", None) is not None:
        raise ValueError(f"{name} carries a bias the eager gated norm never applies")
    module.eps = eps
    module.bias = None


def _fused_gated_rms_norm_class(original: type) -> type:
    """``original`` with ``fla``'s fused gated RMSNorm forward.

    Liger has no gated-norm kernel; `flash-linear-attention`, already a dependency for this roster's
    delta-rule kernels, provides one. It keeps the weight multiply in fp32 where the eager modules round
    first — a deliberate deviation from the bit-for-bit rule, on the more accurate side. Imported inside
    the factory because ``fla`` is slow to import and probes Triton at import time, while the
    orchestrator is on the import path of every run and every CPU test.
    """
    from fla.modules import FusedRMSNormGated  # noqa: PLC0415 — heavy GPU-only kernel dependency

    class _FusedGatedRMSNorm(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _bridge_gated_norm(self)

        def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
            # Keeps the family's own parameter names: its blocks call this positionally, but a
            # keyword caller must not have to use fla's `x`/`g` spelling.
            return FusedRMSNormGated.forward(self, hidden_states, gate)

    return _rebrand(_FusedGatedRMSNorm, original, "gated_rms_norm")


def _fused_glu_forward(self, x: torch.Tensor) -> torch.Tensor:
    """`down(act(gate(x)) * up(x))` with the activation and multiply in one kernel."""
    return self.down_proj(getattr(self, _GLU_MUL_ATTR)(self.gate_proj(x), self.up_proj(x)))


def _fused_glu_mlp_class(original: type) -> type:
    """``original`` with the activation-and-multiply replaced by the toolkit's fused GLU kernel.

    The kernel is chosen by probing the instance's own ``act_fn`` at construction
    (:func:`~src.kernels.fused_glu.resolve_fused_glu_mul`), the same behavioural gate the EP wrappers
    use, so a config that swaps ``hidden_act`` to something the kernels do not compute falls back to
    the family's own forward rather than changing the activation.
    """

    class _FusedGluMLP(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            setattr(self, _GLU_MUL_ATTR, resolve_fused_glu_mul(getattr(self, "act_fn", None)))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if getattr(self, _GLU_MUL_ATTR) is None:
                return original.forward(self, x)
            return _fused_glu_forward(self, x)

    return _rebrand(_FusedGluMLP, original, "glu_mlp")


def _patch_module(module: ModuleType, spec: LigerFamilySpec, flags: dict) -> list[str]:
    """Swap every role the spec fills and the flags enable. Idempotent; returns the roles patched.

    A flag the spec names no class for patches nothing, which covers the roles upstream's applier
    already ran.
    """
    patched: list[str] = []

    if flags.get("rms_norm") and (spec.rms_norm or spec.gated_rms_norm):
        for name in spec.rms_norm:
            original = _named_class(module, name, spec)
            if getattr(original, _PATCHED_MARKER, None) != "rms_norm":
                setattr(
                    module,
                    name,
                    _liger_rms_norm_class(
                        original, offset=spec.rms_norm_offset, casting_mode=spec.rms_norm_casting_mode
                    ),
                )
        for name in spec.gated_rms_norm:
            original = _named_class(module, name, spec)
            if getattr(original, _PATCHED_MARKER, None) != "gated_rms_norm":
                setattr(module, name, _fused_gated_rms_norm_class(original))
        patched.append(f"RMSNorm({', '.join(spec.rms_norm + spec.gated_rms_norm)})")

    if flags.get(_GLU_FLAG) and spec.glu_mlp:
        for name in spec.glu_mlp:
            original = _named_class(module, name, spec)
            if getattr(original, _PATCHED_MARKER, None) != "glu_mlp":
                setattr(module, name, _fused_glu_mlp_class(original))
        patched.append(f"{_GLU_FLAG}({', '.join(spec.glu_mlp)})")

    if flags.get("rope"):
        module.apply_rotary_pos_emb = liger_rotary_pos_emb
        patched.append("RoPE")

    if flags.get("fused_linear_cross_entropy") and spec.causal_lm:
        lce_forward = build_lce_forward(spec.logit_scale_attr)
        for name in spec.causal_lm:
            _named_class(module, name, spec).forward = lce_forward
        patched.append(f"FusedLinearCrossEntropy({', '.join(spec.causal_lm)})")

    return patched


def _patch_instance(model, spec: LigerFamilySpec, flags: dict) -> None:
    """Apply the same roles to an already-built model, for callers that patch post-construction.

    The class swaps above only reach modules constructed afterwards. Matching is by class name, which
    the swap preserves, so this covers both an unpatched model and one whose classes were swapped.
    """
    if flags.get("rms_norm"):
        gated_forward = None
        for module in model.modules():
            if type(module).__name__ in spec.rms_norm:
                _patch_rms_norm_module(
                    module,
                    offset=spec.rms_norm_offset,
                    casting_mode=spec.rms_norm_casting_mode,
                    in_place=spec.rms_norm_casting_mode != "gemma",
                )
            elif (
                type(module).__name__ in spec.gated_rms_norm
                and getattr(type(module), _PATCHED_MARKER, None) != "gated_rms_norm"
            ):
                _bridge_gated_norm(module)
                # Resolved on first need: building it imports `fla`, and every instance is already
                # fused when the class swap ran at load.
                gated_forward = gated_forward or _fused_gated_rms_norm_class(type(module)).forward
                module.forward = MethodType(gated_forward, module)
    if flags.get(_GLU_FLAG):
        for module in model.modules():
            if type(module).__name__ not in spec.glu_mlp:
                continue
            # The class swap already fuses this module unless something bound a forward over it,
            # which upstream's instance patch does on the MLPs a delegating spec names, in the call
            # this one follows. Re-binding puts the toolkit's patch last.
            if getattr(type(module), _PATCHED_MARKER, None) == "glu_mlp" and "forward" not in module.__dict__:
                continue
            fused_mul = getattr(module, _GLU_MUL_ATTR, None) or resolve_fused_glu_mul(getattr(module, "act_fn", None))
            if fused_mul is not None:
                setattr(module, _GLU_MUL_ATTR, fused_mul)
                module.forward = MethodType(_fused_glu_forward, module)
    if flags.get("fused_linear_cross_entropy") and type(model).__name__ in spec.causal_lm:
        model.forward = MethodType(build_lce_forward(spec.logit_scale_attr), model)


class LigerApplier:
    """The callable the orchestrator resolves for one family.

    A callable object rather than a generated function, so the per-family signature the orchestrator reads
    for its FLCE-only, ``rope``-off and loss defaults is built from the spec instead of restated.
    ``upstream`` is set for a delegating spec: its applier runs first, then the spec's own roles on top.
    """

    def __init__(self, spec: LigerFamilySpec):
        self.spec = spec
        self.upstream = _upstream_applier(spec) if spec.delegates_to_upstream else None
        # Resolved once: the signature the orchestrator reads and the defaults a call starts from come
        # from the same source, and under delegation deriving it means inspecting upstream's signature.
        self.defaults = dict(self._declared_parameters())
        self.__signature__ = inspect.Signature(
            [
                inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default)
                for name, default in self.defaults.items()
            ]
        )
        self.__qualname__ = f"apply_liger_kernel_to_{spec.model_types[0]}"
        self.__module__ = __name__

    def _declared_parameters(self):
        """One parameter per role that will run; a role nothing fills is not offered.

        Under delegation the names and defaults are upstream's; the toolkit applies only its own loss
        convention (CE on, FLCE off, the reverse of Liger's, because CE keeps the logits metrics
        readable).
        """
        spec = self.spec
        loss_defaults = {"cross_entropy": not spec.flce_default, "fused_linear_cross_entropy": spec.flce_default}
        if self.upstream is not None:
            for name, parameter in inspect.signature(self.upstream).parameters.items():
                if name != "model":
                    yield name, loss_defaults.get(name, parameter.default)
        else:
            yield "rope", spec.rope
            yield "cross_entropy", loss_defaults["cross_entropy"]
            if spec.causal_lm:
                yield "fused_linear_cross_entropy", loss_defaults["fused_linear_cross_entropy"]
            if spec.rms_norm or spec.gated_rms_norm:
                yield "rms_norm", True
            if spec.glu_mlp:
                yield _GLU_FLAG, True
        yield "model", None

    def __call__(self, **kwargs) -> list[str]:
        spec = self.spec
        flags = dict(self.defaults)
        flags.update(kwargs)
        model = flags.pop("model", None)

        if flags.get("rope") and not spec.rope:
            raise NotImplementedError(
                f"rope is not implemented for {spec.model_types[0]}: its rotary (partial, "
                f"interleaved, mrope, YARN or per-layer-type) is not what Liger's kernel computes. "
                f"Remove rope from liger_kernel_config."
            )
        if flags.get("cross_entropy") and flags.get("fused_linear_cross_entropy"):
            raise ValueError("cross_entropy and fused_linear_cross_entropy cannot both be True.")

        patched: list[str] = []
        if self.upstream is not None:
            # Upstream handles every role it declares, the loss included, hence the skip below. What
            # follows adds only the classes the spec names, which `_named_class` checks are unclaimed.
            self.upstream(model=model, **flags)
            patched.append(f"{self.upstream.__name__}({', '.join(sorted(name for name, on in flags.items() if on))})")

        if spec.remote_classes:
            # The modeling module does not exist until transformers loads the remote file; arm the
            # patch instead of importing it here.
            patch_remote_modules(spec.remote_classes, lambda module: _patch_module(module, spec, flags))
            patched.append(f"armed for the remote module defining {spec.remote_classes[0]}")
        else:
            patched += _patch_module(importlib.import_module(spec.modeling_module), spec, flags)

        if flags.get("cross_entropy") and self.upstream is None:
            patch_loss_utils_cross_entropy()
            patched.append("CrossEntropy")

        if model is not None:
            _patch_instance(model, spec, flags)

        logger.info(f"Liger Kernel applied to {spec.model_types[0]}: {', '.join(patched) or 'nothing'}")
        return patched


def build_liger_appliers(specs: tuple[LigerFamilySpec, ...]) -> dict[str, Callable]:
    """``model_type`` → applier for every spec; a family's aliases share one applier instance."""
    registry: dict[str, Callable] = {}
    for spec in specs:
        applier = LigerApplier(spec)
        for model_type in spec.model_types:
            if model_type in registry:
                raise ValueError(f"Two LigerFamilySpecs claim model_type {model_type!r}")
            registry[model_type] = applier
    return registry
