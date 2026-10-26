"""The flash-interface view a modeling module's attention registry takes, and packed-document ``position_ids``
re-injection through it for families whose forward drops them before flash attention."""

from __future__ import annotations

import functools
import inspect
import logging
from abc import ABC, abstractmethod

from transformers.models.mistral4 import modeling_mistral4 as m4

from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


class FlashInterfaceViewBase(ABC):
    """A view of ``ALL_ATTENTION_FUNCTIONS`` that hands out wrapped flash interfaces and everything else
    untouched, installed into one modeling module so every other family keeps the original."""

    def __init__(self, inner):
        self._inner = inner

    @abstractmethod
    def wrap_flash_interface(self, fn):
        """``fn``, a flash attention interface, as this view hands it out."""

    def get_interface(self, impl, *args, **kwargs):
        fn = self._inner.get_interface(impl, *args, **kwargs)
        if not isinstance(impl, str) or "flash" not in impl:
            return fn
        return self.wrap_flash_interface(fn)

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


class PositionIdsInjectingRegistry(FlashInterfaceViewBase):
    """Flash interfaces that re-inject the module's stashed ``position_ids``."""

    def wrap_flash_interface(self, fn):
        @functools.wraps(fn)
        def with_position_ids(module, *fargs, **fkwargs):
            position_ids = getattr(module, "_halo_packed_position_ids", None)
            if position_ids is not None and "position_ids" not in fkwargs:
                fkwargs["position_ids"] = position_ids
            return fn(module, *fargs, **fkwargs)

        return with_position_ids


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
    installed = install_packed_position_ids_injection(m4, m4.Mistral4Attention, lambda attention: (attention,))
    if installed and is_global_main_process():
        logger.info("Patched Mistral4 flash attention to receive position_ids (packed-document isolation)")
