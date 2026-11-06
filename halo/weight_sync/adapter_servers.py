"""The HTTP side of adapter sync: the same adapter registered on every rollout server or on none, and retired once
nothing runs under it. It holds no weight-sync group; the engines' base is pushed through the pool."""

import concurrent.futures
import logging
from collections.abc import Callable, Iterable
from typing import Any

from halo.weight_sync.clients.base import WeightSyncClientBase

logger = logging.getLogger(__name__)


class AdapterServers:
    """One HTTP-only client per rollout server."""

    def __init__(self, urls: list[str], *, client_cls: type[WeightSyncClientBase], connection_timeout: float):
        self._clients = [client_cls(base_url=url, connection_timeout=connection_timeout) for url in urls]
        self._urls = urls

    def lacking(self, name: str) -> list[str]:
        """The servers that do not hold ``name``: one that restarted since it was loaded lost its registry."""
        loaded = self._on_every_server(lambda client: client.loaded_adapters())
        return [url for url, names in zip(self._urls, loaded, strict=True) if name not in names]

    def load(self, name: str, config: dict[str, Any], payload: str) -> None:
        """All or none: a server that refuses unloads the adapter from the others, so no round meets two policies."""
        outcomes = self._on_every_server(lambda client: self._try(client.load_adapter, name, config, payload))
        failures = [(url, error) for url, error in zip(self._urls, outcomes, strict=True) if error is not None]
        if not failures:
            return
        for client, error in zip(self._clients, outcomes, strict=True):
            if error is None:
                self._try(client.unload_adapter, name)
        raise RuntimeError(
            f"Loading adapter {name!r} failed on {len(failures)}/{len(self._urls)} rollout server(s), and it was "
            f"unloaded from the rest: " + "; ".join(f"{url}: {error}" for url, error in failures)
        )

    def unload(self, names: Iterable[str]) -> None:
        """Retire ``names`` on every server that holds them; raises on a refusal, since an adapter left loaded takes a
        slot the next load needs."""
        retiring = list(names)

        def retire(client: WeightSyncClientBase) -> None:
            held = set(client.loaded_adapters())
            for name in retiring:
                if name in held:
                    client.unload_adapter(name)

        failures = [
            f"{url}: {error}"
            for url, error in zip(self._urls, self._on_every_server(lambda c: self._try(retire, c)), strict=True)
            if error is not None
        ]
        if failures:
            raise RuntimeError(f"Retiring adapters {retiring} failed: " + "; ".join(failures))

    def loaded(self) -> list[list[str]]:
        """Each server's adapters, in server order."""
        return self._on_every_server(lambda client: client.loaded_adapters())

    def close(self) -> None:
        for client in self._clients:
            client.close_http_session()

    @staticmethod
    def _try(step: Callable[..., None], *args: Any) -> str | None:
        """The failure as text, ``None`` on success, so one server's refusal does not hide the others'."""
        try:
            step(*args)
        except Exception as error:
            return f"{type(error).__name__}: {error}"
        return None

    def _on_every_server(self, call: Callable[[WeightSyncClientBase], Any]) -> list[Any]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self._clients)) as executor:
            return list(executor.map(call, self._clients))
