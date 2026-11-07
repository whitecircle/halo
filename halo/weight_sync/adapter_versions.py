"""The adapter versions a process's requests name: the newest one, and which are still held by a request in flight."""

import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager


class AdapterVersions:
    """A request leases the newest version for its whole episode, so a version is retired only once nothing holds it.
    Thread-safe: the trainer thread publishes while the rollout threads lease."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: str | None = None
        self._held: Counter[str] = Counter()

    @property
    def current(self) -> str | None:
        with self._lock:
            return self._current

    def publish(self, name: str) -> None:
        """New leases name ``name``; leases already taken keep their version."""
        with self._lock:
            self._current = name

    @contextmanager
    def lease(self) -> Iterator[str]:
        """The newest version, held for the ``with`` block. Raises before one is published, since a request that names no
        adapter would be served by the frozen base."""
        with self._lock:
            name = self._current
            if name is None:
                raise RuntimeError(
                    "A rollout request was dispatched before any adapter version was published: the engine would "
                    "serve the frozen base model, not the policy."
                )
            self._held[name] += 1
        try:
            yield name
        finally:
            with self._lock:
                self._held[name] -= 1
                if not self._held[name]:
                    del self._held[name]

    def held(self) -> frozenset[str]:
        """The versions some lease still holds."""
        with self._lock:
            return frozenset(self._held)
