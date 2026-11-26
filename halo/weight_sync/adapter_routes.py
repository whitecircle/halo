"""Which rollout servers lack which adapter versions, as this process's rollout manager routes by it: a server that
restarted holds none, and an episode leased to a version goes only to servers that hold it."""

import threading
from collections.abc import Callable, Iterable, Mapping


class AdapterRoutes:
    """Negative knowledge only, so a version published after the last probe is routable everywhere. Thread-safe: the
    trainer thread learns from probes while the rollout threads route and report refusals."""

    def __init__(self, is_missing: Callable[[str, str], bool]):
        """``is_missing(error, adapter)``: whether an episode's error says the engine holds no ``adapter``."""
        self.is_missing = is_missing
        self._lock = threading.Lock()
        self._missing: dict[str, set[str]] = {}

    def learn(self, holdings: Mapping[str, Iterable[str] | None], versions: Iterable[str]) -> None:
        """Replace what is known of each probed server: it lacks every version of ``versions`` it did not list. A server
        that did not answer (``None``) lacks them all."""
        tracked = set(versions)
        with self._lock:
            for url, names in holdings.items():
                self._missing[url] = tracked - set(names or ())

    def exclude(self, url: str, version: str) -> None:
        """``url`` answered that it holds no ``version``; it is skipped for that version until a probe finds it."""
        with self._lock:
            self._missing.setdefault(url, set()).add(version)

    def eligible(self, urls: list[str], version: str) -> list[str]:
        with self._lock:
            return [url for url in urls if version not in self._missing.get(url, ())]

    def forget_except(self, versions: Iterable[str]) -> None:
        """Drop what is known of retired versions."""
        live = set(versions)
        with self._lock:
            for url, missing in self._missing.items():
                self._missing[url] = missing & live
