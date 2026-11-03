"""Role class factories and forwards a Liger family spec swaps in, applied to a modeling module or a built model."""

from __future__ import annotations

import functools
from collections.abc import Callable
from types import MethodType, ModuleType

import torch
import torch.nn as nn
from fla.modules import FusedRMSNormGated
from liger_kernel.transformers import LigerRMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb

from halo.kernels.fused_glu import resolve_fused_glu_mul
from halo.model_loading.liger.cross_entropy import build_lce_forward
from halo.model_loading.liger.family_specs import LigerFamilySpec

# Marks a class a role swap produced, so a second application (TRL re-applies after the load) is a no-op
# instead of stacking subclasses.
_PATCHED_MARKER = "_halo_liger_patched_role"

# The resolved fused activation-and-multiply, latched per module at construction.
_GLU_MUL_ATTR = "_halo_glu_mul"

# Liger's two spellings of the gated-MLP flag; a non-delegating spec offers the first, a delegating one
# whichever upstream declares. The kernel follows the module's own activation, not the spelling.
GLU_FLAGS = ("swiglu", "geglu")

# The gates `fla`'s fused gated norm implements. Its kernel dispatches on the string with no else
# branch, so a value it does not know applies no gate at all.
_FLA_GATE_ACTIVATIONS = frozenset({"swish", "silu", "sigmoid"})


def _named_class(module: ModuleType, name: str, spec: LigerFamilySpec) -> type:
    """The class ``name`` in ``module``, raising with the family named if it is absent."""
    cls = getattr(module, name, None)
    if not isinstance(cls, type):
        raise AttributeError(
            f"Liger applier for {spec.model_types[0]}: {module.__name__} defines no class {name!r}. "
            f"The family's modeling module changed; update its LigerFamilySpec."
        )
    # Upstream's applier has already run: a class it replaced is a role both claim, and subclassing the replacement
    # would stack the swaps. `_rebrand` keeps the toolkit's swaps on the family's own module.
    if spec.delegates_to_upstream and cls.__module__ != module.__name__:
        raise ValueError(
            f"Liger applier for {spec.model_types[0]}: {module.__name__}.{name} is now "
            f"{cls.__module__}.{cls.__qualname__} — upstream claims the class this spec also names. "
            f"Drop that role; two swaps on one class stack silently."
        )
    return cls


def _rebrand(patched: type, original: type, role: str) -> type:
    """Give the subclass the original's identity, since the EP, PP and module-tree lookups key on
    ``type(module).__name__``."""
    patched.__name__ = original.__name__
    patched.__qualname__ = original.__qualname__
    patched.__module__ = original.__module__
    setattr(patched, _PATCHED_MARKER, role)
    return patched


def _norm_epsilon(module: nn.Module) -> float:
    """The epsilon of an RMSNorm or gated-norm module, whichever spelling its family uses."""
    eps = getattr(module, "variance_epsilon", None)
    if eps is None:
        eps = getattr(module, "eps", None)
    if eps is None:
        raise AttributeError(
            f"{type(module).__name__} exposes neither `variance_epsilon` nor `eps`; a fused norm "
            f"cannot be given an epsilon. Drop it from the family's LigerFamilySpec."
        )
    return eps


def native_rms_norm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """``x * rsqrt(mean(x²) + eps) * w`` through torch's fused kernel. Gemma mode multiplies the weight inside it in
    fp32; llama mode casts the normalized activations back first and multiplies in the activation dtype."""
    weight = self.weight if getattr(self, "with_scale", True) else None
    shape, eps = (hidden_states.shape[-1],), _norm_epsilon(self)
    if self.native_rms_casting_mode == "llama":
        normed = torch.nn.functional.rms_norm(hidden_states, shape, None, eps)
        return normed if weight is None else weight * normed
    if weight is not None and weight.dtype != hidden_states.dtype:
        return torch.nn.functional.rms_norm(hidden_states.float(), shape, weight.float(), eps).type_as(hidden_states)
    return torch.nn.functional.rms_norm(hidden_states, shape, weight, eps)


def _native_rms_norm_class(original: type, *, casting_mode: str) -> type:
    class _NativeRMSNorm(original):
        native_rms_casting_mode = casting_mode
        forward = native_rms_norm_forward

    return _rebrand(_NativeRMSNorm, original, "rms_norm")


def _configure_liger_rms_norm(module: nn.Module, *, offset: float, casting_mode: str) -> None:
    """Set the attributes ``LigerRMSNorm.forward`` reads. ``in_place`` (the llama mode reuses the incoming gradient
    buffer) is correct only while no norm output adds straight into a residual, as ``AddBackward`` hands both inputs
    one gradient object; every family here is pre-norm."""
    module.variance_epsilon = _norm_epsilon(module)
    module.offset = offset
    module.casting_mode = casting_mode
    module.in_place = casting_mode != "gemma"
    module.row_mode = None


def _liger_rms_norm_class(original: type, *, offset: float, casting_mode: str) -> type:
    class _LigerRMSNorm(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _configure_liger_rms_norm(self, offset=offset, casting_mode=casting_mode)

        forward = LigerRMSNorm.forward

    return _rebrand(_LigerRMSNorm, original, "rms_norm")


def _rms_norm_class(original: type, spec: LigerFamilySpec) -> type:
    if spec.rms_norm_kernel == "native":
        return _native_rms_norm_class(original, casting_mode=spec.rms_norm_casting_mode)
    return _liger_rms_norm_class(original, offset=spec.rms_norm_offset, casting_mode=spec.rms_norm_casting_mode)


def _bridge_gated_norm(module: nn.Module) -> None:
    """Give a gated-norm module the ``eps`` and ``bias`` spellings ``fla``'s forward reads, or raise."""
    name = type(module).__name__
    eps = _norm_epsilon(module)
    activation = getattr(module, "activation", None)
    if activation not in _FLA_GATE_ACTIVATIONS:
        raise ValueError(
            f"{name} gates with {activation!r}, which fla's kernel does not implement — it would apply "
            f"no gate at all. Drop it from the family's gated_rms_norm spec."
        )
    # fla applies a bias when present; these norms have none, and gaining one would change the function.
    if getattr(module, "bias", None) is not None:
        raise ValueError(f"{name} carries a bias the eager gated norm never applies")
    module.eps = eps
    module.bias = None


def fused_gated_rms_norm_forward(self, hidden_states: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
    """``fla``'s fused gated RMSNorm under the family's parameter names, for a keyword caller."""
    return FusedRMSNormGated.forward(self, hidden_states, gate)


def _fused_gated_rms_norm_class(original: type) -> type:
    """``original`` with ``fla``'s fused gated forward, which keeps the weight multiply in fp32 where the eager modules
    round first: a deliberate deviation from the bit-for-bit rule, on the more accurate side."""

    class _FusedGatedRMSNorm(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _bridge_gated_norm(self)

        forward = fused_gated_rms_norm_forward

    return _rebrand(_FusedGatedRMSNorm, original, "gated_rms_norm")


def _glu_flag_on(flags: dict) -> bool:
    return any(flags.get(name) for name in GLU_FLAGS)


def _fused_glu_forward(self, x: torch.Tensor) -> torch.Tensor:
    """`down(act(gate(x)) * up(x))` with the activation and multiply in one kernel."""
    return self.down_proj(getattr(self, _GLU_MUL_ATTR)(self.gate_proj(x), self.up_proj(x)))


def _fused_glu_mlp_class(original: type) -> type:
    """``original`` with the toolkit's fused GLU kernel, chosen by probing the instance's ``act_fn``; an activation the
    kernels do not compute falls back to the family's own forward."""

    class _FusedGluMLP(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            setattr(self, _GLU_MUL_ATTR, resolve_fused_glu_mul(getattr(self, "act_fn", None)))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if getattr(self, _GLU_MUL_ATTR) is None:
                return original.forward(self, x)
            return _fused_glu_forward(self, x)

    return _rebrand(_FusedGluMLP, original, "glu_mlp")


def apply_fused_modules(module: ModuleType, spec: LigerFamilySpec, flags: dict) -> list[str]:
    """Swap every role the spec fills and the flags enable; idempotent, returns the roles patched."""
    patched: list[str] = []

    if flags.get("rms_norm") and (spec.rms_norm or spec.gated_rms_norm):
        for name in spec.rms_norm:
            original = _named_class(module, name, spec)
            if getattr(original, _PATCHED_MARKER, None) != "rms_norm":
                setattr(module, name, _rms_norm_class(original, spec))
        for name in spec.gated_rms_norm:
            original = _named_class(module, name, spec)
            if getattr(original, _PATCHED_MARKER, None) != "gated_rms_norm":
                setattr(module, name, _fused_gated_rms_norm_class(original))
        patched.append(f"RMSNorm({', '.join(spec.rms_norm + spec.gated_rms_norm)})")

    if _glu_flag_on(flags) and spec.glu_mlp:
        for name in spec.glu_mlp:
            original = _named_class(module, name, spec)
            if getattr(original, _PATCHED_MARKER, None) != "glu_mlp":
                setattr(module, name, _fused_glu_mlp_class(original))
        patched.append(f"GLU({', '.join(spec.glu_mlp)})")

    # Only where the spec declares the rotary is Liger's; under delegation the role is upstream's.
    if flags.get("rope") and spec.rope:
        module.apply_rotary_pos_emb = liger_rotary_pos_emb
        patched.append("RoPE")

    if flags.get("fused_linear_cross_entropy") and spec.causal_lm:
        lce_forward = build_lce_forward(spec.router_aux_loss_in_head)
        for name in spec.causal_lm:
            _named_class(module, name, spec).forward = lce_forward
        patched.append(f"FusedLinearCrossEntropy({', '.join(spec.causal_lm)})")

    return patched


def _bind_forward(module: nn.Module, forward: Callable) -> None:
    """Install the module-level ``forward`` on one instance as a partial, which pickles the function by
    reference where a bound method would unpickle to the class's own ``forward``."""
    module.forward = functools.partial(forward, module)


def apply_fused_modules_to_model(model, spec: LigerFamilySpec, flags: dict) -> None:
    """Apply the same roles to a built model, matched by class name. Every module binding survives
    ``torch.save``/``torch.load``; the fused-loss head's does not."""
    if flags.get("rms_norm"):
        for module in model.modules():
            name = type(module).__name__
            if name in spec.rms_norm:
                if spec.rms_norm_kernel == "native":
                    module.native_rms_casting_mode = spec.rms_norm_casting_mode
                    _bind_forward(module, native_rms_norm_forward)
                else:
                    _configure_liger_rms_norm(
                        module, offset=spec.rms_norm_offset, casting_mode=spec.rms_norm_casting_mode
                    )
                    _bind_forward(module, LigerRMSNorm.forward)
            elif name in spec.gated_rms_norm and getattr(type(module), _PATCHED_MARKER, None) != "gated_rms_norm":
                _bridge_gated_norm(module)
                _bind_forward(module, fused_gated_rms_norm_forward)
    if _glu_flag_on(flags):
        for module in model.modules():
            if type(module).__name__ not in spec.glu_mlp:
                continue
            # The class swap already fuses this module unless something (upstream's instance patch)
            # bound a forward over it; re-binding puts the toolkit's patch last.
            if getattr(type(module), _PATCHED_MARKER, None) == "glu_mlp" and "forward" not in module.__dict__:
                continue
            fused_mul = getattr(module, _GLU_MUL_ATTR, None) or resolve_fused_glu_mul(getattr(module, "act_fn", None))
            if fused_mul is not None:
                setattr(module, _GLU_MUL_ATTR, fused_mul)
                _bind_forward(module, _fused_glu_forward)
    if flags.get("fused_linear_cross_entropy") and type(model).__name__ in spec.causal_lm:
        # A bound method, not `_bind_forward`: the closure cannot be pickled by reference anyway, and
        # `_liger_fused_head_installed` detects the head by the function's module, which a partial hides.
        model.forward = MethodType(build_lce_forward(spec.router_aux_loss_in_head), model)
