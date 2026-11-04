"""The flash-interface view a modeling module's attention registry takes, the ``position_ids`` stash a forward leaves
on the modules below it, and packed-document re-injection for families whose forward drops ``position_ids``."""

from __future__ import annotations

import functools
import inspect
from abc import ABC, abstractmethod

_STASH_ATTR = "_halo_packed_position_ids"


def get_stashed_position_ids(module):
    return getattr(module, _STASH_ATTR, None)


class FlashInterfaceViewBase(ABC):
    """Installed into one modeling module, so every other family keeps the original registry."""

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
    def wrap_flash_interface(self, fn):
        @functools.wraps(fn)
        def with_position_ids(module, *fargs, **fkwargs):
            position_ids = get_stashed_position_ids(module)
            if position_ids is not None and "position_ids" not in fkwargs:
                fkwargs["position_ids"] = position_ids
            return fn(module, *fargs, **fkwargs)

        return with_position_ids


def is_position_ids_stash_installed(owner_cls) -> bool:
    return getattr(owner_cls.forward, _STASH_ATTR, False)


def forward_names_position_ids(owner_cls) -> bool:
    """Through ``**kwargs`` the stash's signature bind finds none, so every call would stash ``None``."""
    return "position_ids" in inspect.signature(owner_cls.forward).parameters


def install_position_ids_stash(owner_cls, stash_targets) -> bool:
    """Leave ``owner_cls.forward``'s ``position_ids`` on the modules ``stash_targets(self)`` returns, for each call."""
    if is_position_ids_stash_installed(owner_cls):
        return False
    original_forward = owner_cls.forward
    signature = inspect.signature(original_forward)

    @functools.wraps(original_forward)
    def forward(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        position_ids = bound.arguments.get("position_ids")
        targets = stash_targets(self)
        for target in targets:
            setattr(target, _STASH_ATTR, position_ids)
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            for target in targets:
                setattr(target, _STASH_ATTR, None)

    setattr(forward, _STASH_ATTR, True)
    owner_cls.forward = forward
    return True


def install_packed_position_ids_injection(modeling_module, owner_cls, stash_targets) -> bool:
    """A forward that never hands ``position_ids`` to the attention interface runs a packed row as one dense causal
    sequence on flash; dense backends form the packed mask at the model level."""
    if is_position_ids_stash_installed(owner_cls):
        return False
    if not forward_names_position_ids(owner_cls):
        raise RuntimeError(
            f"{owner_cls.__qualname__}.forward takes no position_ids parameter, so the stash that keeps a packed row's "
            f"documents apart would read None: re-derive the patch on this release's forward."
        )
    registry = modeling_module.ALL_ATTENTION_FUNCTIONS
    install_position_ids_stash(owner_cls, stash_targets)
    if not isinstance(registry, PositionIdsInjectingRegistry):
        modeling_module.ALL_ATTENTION_FUNCTIONS = PositionIdsInjectingRegistry(registry)
    return True
