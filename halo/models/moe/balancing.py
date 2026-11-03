"""MoE balancing-mode resolution, the balancing-slot contract of exported checkpoints and the per-token weights of the
expert load the bias update balances."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import ClassVar, Literal, Protocol, get_args, runtime_checkable

import torch

from halo.log import warn_once
from halo.models.config_access import set_config_field
from halo.models.moe.family_specs import MOE_FAMILY_SPECS, NativeBalancingSlot, find_moe_family_spec_for_config
from halo.models.moe.router_logits import honors_router_logits_flag
from halo.models.unwrapping import base_transformers_model, plain_module_name, unwrap_framework_wrappers

logger = logging.getLogger(__name__)

BalancingMode = Literal["auto", "none", "aux_loss", "bias_update", "bias_update_transient"]
_VALID_BALANCING_MODES = get_args(BalancingMode)
# Most informed first, as a stage that found a bias-capable router has evidence a stage holding no MoE layer does not.
BALANCING_MODE_PRECEDENCE: tuple[BalancingMode, ...] = ("bias_update", "bias_update_transient", "aux_loss", "none")
assert set(BALANCING_MODE_PRECEDENCE) == set(_VALID_BALANCING_MODES) - {"auto"}, "a balancing mode has no precedence"
# ``bias_update`` requires every balancing router to carry the bias in checkpoint-exported state; the
# ``_transient`` spelling opts into a trainer-only bias where the architecture has no such slot.
BIAS_UPDATE_MODES = ("bias_update", "bias_update_transient")
# The hub-native balancing buffer (Zaya) and the EP base's adoption property.
BALANCING_BIASES_ATTR = "balancing_biases"
# Set by the EP base when it adopts the family's native slot rather than the transient side-buffer.
NATIVE_BALANCING_BIAS_ADOPTED_ATTR = "_native_balancing_bias_adopted"
_BALANCING_KEY_SUFFIXES = frozenset(
    {
        BALANCING_BIASES_ATTR,
        *(attr for spec in MOE_FAMILY_SPECS if spec.balancing_slot for attr in spec.balancing_slot.attrs),
    }
)

# Per model class, so a resolution repeated by a second consumer (the PP split gate) says it once.
_warned_unservable_auto: set[str] = set()
_warned_transient_only_auto: set[str] = set()
_expert_load_weights: ContextVar[torch.Tensor | None] = ContextVar("expert_load_weights", default=None)


@runtime_checkable
class BiasBalancingRouter(Protocol):
    """``SUPPORTS_BIAS_BALANCING``: the layer selects experts itself, so a bias before top-k steers routing.
    ``SEVERS_AUX_LOSS``: the layer bypasses the HF router module, so ``outputs.router_logits`` stays empty."""

    SUPPORTS_BIAS_BALANCING: ClassVar[bool]
    SEVERS_AUX_LOSS: ClassVar[bool]

    def can_adopt_native_balancing(self) -> bool:
        """Whether bias-update balancing on this layer lands in checkpoint-exported state."""


def resolve_balancing_slot(
    module: torch.nn.Module, path: str | None, *, require_tensor: bool = True
) -> tuple[torch.nn.Module, str] | None:
    """``None`` when the owner is absent or, under ``require_tensor``, the slot holds no tensor."""
    if not path:
        return None
    owner_path, _, name = path.rpartition(".")
    try:
        owner = module.get_submodule(owner_path) if owner_path else module
    except AttributeError:
        return None
    if require_tensor and not isinstance(getattr(owner, name, None), torch.Tensor):
        return None
    return owner, name


def balancing_param_keys(model: torch.nn.Module) -> frozenset[str]:
    """Save paths keep these at their trained dtype, as a bf16 round-trip would quantize away 1e-3 sign-steps."""
    keys: set[str] = set()
    for name, module in model.named_modules():
        prefix = f"{name}." if name else ""
        if vars(module).get(NATIVE_BALANCING_BIAS_ADOPTED_ATTR, False):
            keys.update(f"{prefix}{attr}" for attr in type(module).FAMILY.balancing_slot.attrs)
        elif any(buf_name == BALANCING_BIASES_ATTR for buf_name, _ in module.named_buffers(recurse=False)):
            keys.add(f"{prefix}{BALANCING_BIASES_ATTR}")
    return frozenset(keys)


def is_balancing_state_key(key: str) -> bool:
    """The model-free counterpart of :func:`balancing_param_keys`, for a merged-from-shards save."""
    return any(key == suffix or key.endswith(f".{suffix}") for suffix in _BALANCING_KEY_SUFFIXES)


def _native_balancing_tensor(module: torch.nn.Module, slot: NativeBalancingSlot | None) -> torch.Tensor | None:
    native = dict(module.named_buffers(recurse=False)).get(BALANCING_BIASES_ATTR)
    if native is not None:
        return native
    for path in slot.attrs if slot else ():
        target = resolve_balancing_slot(module, path)
        if target is not None:
            return getattr(*target)
    return None


def _materialize_hub_balancing_slot(
    model: torch.nn.Module, module: torch.nn.Module, slot: NativeBalancingSlot, bias: torch.Tensor
) -> torch.Tensor | None:
    """A PEFT merge rebuilds the base tree with the gate still off (LFM-2 ``use_expert_bias: false``) although the run
    materialized and trained the slot. ``None`` for a slot that is not config-gated."""
    if not slot.config_flag:
        return None
    for path in slot.attrs:
        target = resolve_balancing_slot(module, path, require_tensor=False)
        if target is None:
            continue
        owner, name = target
        device = next((p.device for p in module.parameters()), torch.device("cpu"))
        owner.register_buffer(name, torch.zeros_like(bias, dtype=torch.float32, device=device))
        for sub in module.modules():
            if hasattr(sub, slot.config_flag):
                setattr(sub, slot.config_flag, True)
        cfg = getattr(unwrap_framework_wrappers(model), "config", None)
        if cfg is not None:
            set_config_field(cfg, slot.config_flag, True, only_declared=False)
        return getattr(owner, name)
    return None


def apply_balancing_bias_sidecar(
    model: torch.nn.Module, sidecar: dict[str, torch.Tensor]
) -> tuple[list[str], list[str]]:
    """Copy ``router_balancing_biases.pt`` into a HUB model's own slots, returning ``(applied, skipped)`` module names.
    A family with no native slot trained a transient bias no artifact can serve, so its entries are skipped."""
    spec = find_moe_family_spec_for_config(getattr(unwrap_framework_wrappers(model), "config", None))
    slot = spec.balancing_slot if spec else None
    applied: list[str] = []
    skipped: list[str] = []
    for name, bias in sidecar.items():
        target_name = plain_module_name(name)
        try:
            module = model.get_submodule(target_name)
        except AttributeError:
            raise KeyError(
                f"router_balancing_biases.pt names module {name!r}, which does not exist in "
                f"{type(model).__name__} — the sidecar belongs to a different model tree."
            ) from None
        target = _native_balancing_tensor(module, slot)
        if target is None and slot is not None:
            target = _materialize_hub_balancing_slot(model, module, slot, bias)
        if target is None:
            skipped.append(target_name)
            continue
        if target.shape != bias.shape:
            raise ValueError(
                f"router_balancing_biases.pt entry {name!r} has shape {tuple(bias.shape)} but the "
                f"model's balancing slot is {tuple(target.shape)} — refusing a corrupt pairing."
            )
        target.data.copy_(bias.to(device=target.device, dtype=target.dtype))
        applied.append(target_name)
    return applied, skipped


def iter_balancing_routers(model: torch.nn.Module):
    """Both attributes are required so the yield stays unique when an EP layer adopts an inner gate's native buffer."""
    for module in model.modules():
        if hasattr(module, BALANCING_BIASES_ATTR) and hasattr(module, "expert_load_counter"):
            yield module


def has_balancing_routers(model) -> bool:
    return any(True for _ in iter_balancing_routers(model))


def is_transient_balancing_router(module) -> bool:
    """A plain instance attribute, invisible to ``state_dict()``, where an adopted slot reads through the EP base's
    property and a hub router's buffer lives in ``_buffers``."""
    return BALANCING_BIASES_ATTR in getattr(module, "__dict__", {})


def _bias_balancing_routers(model) -> list[BiasBalancingRouter]:
    return [module for module in model.modules() if isinstance(module, BiasBalancingRouter)]


def accepts_bias_balancing(model) -> bool:
    return any(router.SUPPORTS_BIAS_BALANCING for router in _bias_balancing_routers(model))


def accepts_native_balancing_bias(model) -> bool:
    """Where False, only a transient bias is possible and the trained bias would never reach a served copy."""
    return any(router.can_adopt_native_balancing() for router in _bias_balancing_routers(model))


def severs_aux_loss(model) -> bool:
    """Enabling ``output_router_logits`` there would crash the aux-loss reduction on the empty tuple (DeepSeek-V4)."""
    return any(router.SEVERS_AUX_LOSS and router.SUPPORTS_BIAS_BALANCING for router in _bias_balancing_routers(model))


def _liger_fused_head_installed(model) -> bool:
    forward = getattr(base_transformers_model(model), "forward", None)
    return "liger" in (getattr(forward, "__module__", None) or "")


def resolve_balancing_mode(requested: str, model, is_moe: bool) -> BalancingMode:
    """``auto`` selects ``bias_update`` on a router exposing ``balancing_biases`` (Zaya), EP layers that sever the
    aux-loss path (DeepSeek-V4), or a forward that never declares ``output_router_logits`` while a layer accepts the
    bias, but only where the bias exports. Otherwise ``aux_loss`` where the forward honours the flag, else ``none``."""
    if requested not in _VALID_BALANCING_MODES:
        raise ValueError(f"moe_balancing must be one of {_VALID_BALANCING_MODES}, got {requested!r}")
    if requested != "auto":
        return requested  # type: ignore[return-value]
    if not is_moe:
        return "none"
    has_bias_routers = has_balancing_routers(model)
    honors_aux_loss = honors_router_logits_flag(model)
    wants_bias_update = (
        has_bias_routers or severs_aux_loss(model) or (not honors_aux_loss and accepts_bias_balancing(model))
    )
    if wants_bias_update:
        if has_bias_routers or accepts_native_balancing_bias(model):
            return "bias_update"
        warn_once(
            logger,
            _warned_transient_only_auto,
            type(model).__name__,
            f"moe_balancing=auto resolves to none on {type(model).__name__}: only the bias update "
            f"could balance it, but this architecture has no checkpoint slot for a routing bias, so "
            f"the trained bias would never reach a served copy — THIS RUN TRAINS UNBALANCED. Set "
            f"moe_balancing=bias_update_transient to balance during training while accepting that "
            f"every exported checkpoint serves WITHOUT the bias (near-tied top-k picks flip between "
            f"trainer and server).",
        )
        return "none"
    if honors_aux_loss:
        return "aux_loss"
    # A Liger fused loss installs a forward without the parameter; only then is turning it off a remedy.
    remedy = (
        "set fused_linear_cross_entropy: false in liger_kernel_config to keep the family's own head, or "
        if _liger_fused_head_installed(model)
        else ""
    )
    warn_once(
        logger,
        _warned_unservable_auto,
        type(model).__name__,
        f"moe_balancing=auto resolves to none on {type(model).__name__}: its forward does not take "
        f"output_router_logits, so the aux-loss term never reaches the loss, and nothing on this tree "
        f"carries a routing bias (no EP layer, no native balancing_biases router) — THIS RUN "
        f"TRAINS UNBALANCED, a real cost at this expert count. {remedy}Launch under torchrun with "
        f"use_grouped_gemm: true or expert parallelism, where auto resolves to the bias update for "
        f"families whose bias exports, or load the text-only sibling class whose forward honours the "
        f"flag.",
    )
    return "none"


@contextmanager
def expert_load_token_weights(weights: torch.Tensor | None) -> Iterator[None]:
    """Count token ``i`` of every forward inside the block ``weights[i]`` times toward expert load. A tree-packed row
    computes a shared prefix once for every example containing it, so weighting by example count restores the load."""
    token = _expert_load_weights.set(weights)
    try:
        yield
    finally:
        _expert_load_weights.reset(token)


def resolve_expert_load_weights(indices: torch.Tensor) -> torch.Tensor | None:
    weights = _expert_load_weights.get()
    if weights is None:
        return None
    if weights.numel() != indices[..., 0].numel():
        raise ValueError(
            f"expert-load token weights cover {weights.numel()} tokens but the router selected for "
            f"{indices[..., 0].numel()}: the weights do not describe this forward."
        )
    return weights.to(indices.device).reshape(indices.shape[:-1]).unsqueeze(-1).expand_as(indices)
