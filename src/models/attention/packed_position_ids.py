"""Packed-document ``position_ids`` re-injection for families whose forward drops them before flash attention."""

from __future__ import annotations

import functools
import inspect

from accelerate.logging import get_logger
from transformers.models.mistral4 import modeling_mistral4 as m4

logger = get_logger(__name__)


class PositionIdsInjectingRegistry:
    """A view of ``ALL_ATTENTION_FUNCTIONS`` whose flash interfaces re-inject the module's stashed
    ``position_ids``, installed into one modeling module so every other family keeps the original."""

    def __init__(self, inner):
        self._inner = inner

    def get_interface(self, impl, *args, **kwargs):
        fn = self._inner.get_interface(impl, *args, **kwargs)
        if not isinstance(impl, str) or "flash" not in impl:
            return fn

        @functools.wraps(fn)
        def with_position_ids(module, *fargs, **fkwargs):
            position_ids = getattr(module, "_halo_packed_position_ids", None)
            if position_ids is not None and "position_ids" not in fkwargs:
                fkwargs["position_ids"] = position_ids
            return fn(module, *fargs, **fkwargs)

        return with_position_ids

    def __getattr__(self, name):
        return getattr(self._inner, name)

    # Dunders resolve on the type, bypassing __getattr__, so forward the Mapping protocol explicitly.
    def __contains__(self, key):
        return key in self._inner

    def __getitem__(self, key):
        return self._inner[key]

    def __iter__(self):
        return iter(self._inner)

    def __len__(self):
        return len(self._inner)


def install_packed_position_ids_injection(modeling_module, owner_cls, stash_targets) -> bool:
    """Stash ``owner_cls.forward``'s ``position_ids`` on the attention modules ``stash_targets(self)``
    returns, and give ``modeling_module`` a registry view whose flash interfaces re-inject it.

    A forward that never hands ``position_ids`` to the attention interface runs a packed row as one
    dense causal sequence on flash; dense backends form the packed mask at the model level. ``False``
    when already installed.
    """
    if getattr(owner_cls.forward, "_halo_packed_position_ids", False):
        return False
    registry = modeling_module.ALL_ATTENTION_FUNCTIONS

    original_forward = owner_cls.forward
    signature = inspect.signature(original_forward)

    @functools.wraps(original_forward)
    def forward(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        position_ids = bound.arguments.get("position_ids")
        targets = stash_targets(self)
        for target in targets:
            target._halo_packed_position_ids = position_ids
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            for target in targets:
                target._halo_packed_position_ids = None

    forward._halo_packed_position_ids = True
    owner_cls.forward = forward
    if not isinstance(registry, PositionIdsInjectingRegistry):
        modeling_module.ALL_ATTENTION_FUNCTIONS = PositionIdsInjectingRegistry(registry)
    return True


def patch_mistral4_flash_packed_position_ids() -> None:
    """Make Mistral4's flash path see ``position_ids``, so packed documents stay isolated. Its attention
    forward takes ``position_ids`` for the attention scale and passes only ``**kwargs`` on, so the
    stash rides that forward."""
    if install_packed_position_ids_injection(m4, m4.Mistral4Attention, lambda attention: (attention,)):
        logger.info("Patched Mistral4 flash attention to receive position_ids (packed-document isolation)")
