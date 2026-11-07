"""The HTTP side of adapter sync: the same adapter registered on every rollout server or on none, and retired once
nothing runs under it. It holds no weight-sync group; the engines' base is pushed through the pool."""

import concurrent.futures
import logging
from collections.abc import Callable, Iterable
from typing import Any

import requests
from urllib3.util.retry import Retry

from halo.weight_sync.clients.base import ServedBase, WeightSyncClientBase

logger = logging.getLogger(__name__)

# The probes retry a refused connection for about a second: a server that is down is reported at once, where the
# clients' default backoff (4, 8, 16 and 32 s) holds a sync or a step for a minute.
_PROBE_RETRY = Retry(connect=2, read=1, status=0, backoff_factor=0.5, allowed_methods=["GET"])


class AdapterServers:
    """One HTTP-only client per rollout server."""

    def __init__(self, urls: list[str], *, client_cls: type[WeightSyncClientBase], connection_timeout: float):
        self._clients = [
            client_cls(base_url=url, connection_timeout=connection_timeout, retry_policy=_PROBE_RETRY) for url in urls
        ]
        self._urls = urls
        self._unreachable: set[str] = set()

    def probe(self) -> dict[str, list[str] | None]:
        """Each server's adapters, ``None`` for one that does not answer (down or restarting). An outage is logged once
        when it starts and once when it ends, not on every probe."""
        listed = self._on_every_server(self._list_or_none)
        holdings = dict(zip(self._urls, listed, strict=True))
        down = {url for url, names in holdings.items() if names is None}
        for url in sorted(down - self._unreachable):
            logger.warning(f"Rollout server {url} did not answer the adapter probe (down or restarting).")
        for url in sorted(self._unreachable - down):
            logger.info(f"Rollout server {url} answers the adapter probe again.")
        self._unreachable = down
        return holdings

    def lacking(self, name: str) -> list[str]:
        """The servers that do not hold ``name``: one that restarted since it was loaded lost its registry. Raises where a
        server does not answer, since the sync could neither load onto it nor tell what it holds."""
        holdings = self.probe()
        silent = [url for url, names in holdings.items() if names is None]
        if silent:
            raise RuntimeError(
                f"Rollout server(s) {silent} did not answer the adapter probe (connection refused or timed out): "
                f"they are down or still restarting. Restart them before the sync."
            )
        return [url for url, names in holdings.items() if name not in names]

    def served_base(self, url: str) -> ServedBase:
        return self._clients[self._urls.index(url)].served_base()

    def load(self, name: str, config: dict[str, Any], payload: str, urls: Iterable[str] | None = None) -> None:
        """All or none over ``urls`` (every server by default): a server that refuses unloads the adapter from the others,
        so no round meets two policies."""
        chosen = list(self._urls if urls is None else urls)
        clients = [self._clients[self._urls.index(url)] for url in chosen]
        outcomes = self._on_each(clients, lambda client: self._try(client.load_adapter, name, config, payload))
        failures = [(url, error) for url, error in zip(chosen, outcomes, strict=True) if error is not None]
        if not failures:
            return
        for client, error in zip(clients, outcomes, strict=True):
            if error is None:
                self._try(client.unload_adapter, name)
        raise RuntimeError(
            f"Loading adapter {name!r} failed on {len(failures)}/{len(chosen)} rollout server(s), and it was "
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

    @staticmethod
    def _list_or_none(client: WeightSyncClientBase) -> list[str] | None:
        try:
            return client.loaded_adapters()
        except requests.exceptions.RequestException:
            return None

    def _on_every_server(self, call: Callable[[WeightSyncClientBase], Any]) -> list[Any]:
        return self._on_each(self._clients, call)

    @staticmethod
    def _on_each(clients: list[WeightSyncClientBase], call: Callable[[WeightSyncClientBase], Any]) -> list[Any]:
        if not clients:
            return []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(clients)) as executor:
            return list(executor.map(call, clients))
