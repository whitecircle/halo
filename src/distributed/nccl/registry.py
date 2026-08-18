"""Rollout-backend key → weight-sync client, derived from the client subclasses shipped in this package."""

from src.distributed.module_registry import iter_subclasses
from src.distributed.nccl.clients.base import BaseWeightSyncClient
from src.distributed.nccl.clients.sglang import (
    SGLangWeightSyncClient,  # noqa: F401 — a shipped client registers by being imported
)
from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient  # noqa: F401


def _client_registry() -> dict[str, type[BaseWeightSyncClient]]:
    """Backend key → client class, derived from the subclasses rather than a parallel table.

    Restricted to the clients shipped in this package: ``__subclasses__()`` also returns classes
    defined outside it (a test double, or this module tree loaded a second time under another name),
    and such a class capturing a backend key would replace the production client with no error.
    """
    registry: dict[str, type[BaseWeightSyncClient]] = {}
    for cls in iter_subclasses(BaseWeightSyncClient):
        if not cls.BACKEND_KEY or not cls.__module__.startswith(f"{__package__}."):
            continue
        clash = registry.get(cls.BACKEND_KEY)
        if clash is not None and clash is not cls:
            raise RuntimeError(
                f"Two clients in {__package__} declare BACKEND_KEY {cls.BACKEND_KEY!r}: "
                f"{clash.__module__}.{clash.__qualname__} and {cls.__module__}.{cls.__qualname__}. "
                f"A backend key must identify exactly one client."
            )
        registry[cls.BACKEND_KEY] = cls
    return registry


def rollout_backends() -> list[str]:
    """Every selectable rollout backend. Tests pin this against the ``rollout_backend`` config
    ``Literal`` so a new client cannot land unselectable."""
    return sorted(_client_registry())


def resolve_weight_sync_client(backend: str) -> type[BaseWeightSyncClient]:
    """The weight-sync client class for ``backend``, raising on an unknown one."""
    registry = _client_registry()
    try:
        return registry[backend]
    except KeyError:
        raise ValueError(f"Unknown rollout backend {backend!r}; supported: {sorted(registry)}") from None
