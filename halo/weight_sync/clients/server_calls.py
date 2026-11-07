"""A blocking server call run on a background thread, joined later, and the wait over several at once."""

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import requests

logger = logging.getLogger(__name__)

# A group's chunk waits on its declares and its send, so the poll's period is added to every chunk of every sync.
_POLL_PERIOD_S = 0.005


_TIMEOUTS = (TimeoutError, requests.exceptions.Timeout)


@contextmanager
def announced_wait(server: str, what: str, bound_s: float) -> Iterator[None]:
    """Say which server a bounded wait is on, for what and for how long, once at its start, and once more if the
    block ends in a timeout; a wait that ends early says nothing further. The timeout is re-raised."""
    logger.info(f"Waiting on {server} for {what}, for at most {bound_s:.0f}s")
    try:
        yield
    except _TIMEOUTS:
        logger.warning(f"Gave up waiting on {server} for {what} after {bound_s:.0f}s")
        raise


def wait_announced(call: "AsyncCall", server: str, what: str, bound_s: float) -> None:
    """Join ``call`` for at most ``bound_s`` under :func:`announced_wait`. A call still running after the join or
    ended in a timeout has expired and is logged once; it is left running, and no error is raised."""
    try:
        with announced_wait(server, what, bound_s):
            call.join(bound_s)
            if call.is_alive():
                raise TimeoutError(what)
            if isinstance(call.error, _TIMEOUTS):
                raise call.error
    except _TIMEOUTS:
        pass


class AsyncCall:
    def __init__(self, fn, name: str = "server call"):
        self.error: Exception | None = None
        self.result = None
        self.name = name
        self._thread = threading.Thread(target=self._run, args=(fn,), daemon=True)
        self._thread.start()

    def _run(self, fn):
        try:
            self.result = fn()
        except Exception as e:
            self.error = e

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def wait(self, timeout: float):
        """Join within ``timeout`` seconds, re-raising the call's error; a call still running raises TimeoutError."""
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():  # join() returns without signalling a timeout
            raise TimeoutError(f"{self.name} still blocked after {timeout:.0f}s — server wedged?")
        if self.error:
            raise self.error

    def join(self, grace: float) -> "AsyncCall":
        """Wait up to ``grace`` seconds for a call still in flight once the local side failed, which may yet
        record the server's own rejection or hand back a resource only this handle holds."""
        self._thread.join(timeout=grace)
        return self


def wait_for_calls(calls: list[AsyncCall], timeout: float) -> None:
    """Block until every call returned, raising as soon as one fails. Polled, as the halves of a group formation
    complete only together and a serial join would park once the other side failed."""
    deadline = time.monotonic() + timeout
    while True:
        for call in calls:
            if call.error is not None:
                raise call.error
        if not any(call.is_alive() for call in calls):
            return
        if time.monotonic() > deadline:
            blocked = ", ".join(call.name for call in calls if call.is_alive())
            raise TimeoutError(f"Weight-transfer setup still blocked after {timeout:.0f}s: {blocked}")
        time.sleep(_POLL_PERIOD_S)
