"""Rollout-backend key → weight-sync client, derived from the client subclasses in ``clients``."""

import halo.weight_sync.clients as _clients
from halo.subclass_registry import import_package_modules, iter_subclasses
from halo.weight_sync.clients.base import WeightSyncClientBase

import_package_modules(_clients)


def _client_registry() -> dict[str, type[WeightSyncClientBase]]:
    """Backend key → client class, from the subclasses in ``clients`` only: a test double or a second copy
    of the tree capturing a key would otherwise replace the production client silently."""
    registry: dict[str, type[WeightSyncClientBase]] = {}
    for cls in iter_subclasses(WeightSyncClientBase):
        if not cls.__module__.startswith(f"{_clients.__name__}."):
            continue
        clash = registry.get(cls.BACKEND_KEY)
        if clash is not None and clash is not cls:
            raise RuntimeError(
                f"Two clients in {_clients.__name__} declare BACKEND_KEY {cls.BACKEND_KEY!r}: "
                f"{clash.__module__}.{clash.__qualname__} and {cls.__module__}.{cls.__qualname__}. "
                f"A backend key must identify exactly one client."
            )
        registry[cls.BACKEND_KEY] = cls
    return registry


def rollout_backends() -> list[str]:
    """Every selectable rollout backend."""
    return sorted(_client_registry())


def resolve_weight_sync_client(backend: str) -> type[WeightSyncClientBase]:
    """The weight-sync client class for ``backend``, raising on an unknown one."""
    registry = _client_registry()
    try:
        return registry[backend]
    except KeyError:
        raise ValueError(f"Unknown rollout backend {backend!r}; supported: {sorted(registry)}") from None
