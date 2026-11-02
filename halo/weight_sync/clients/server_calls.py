"""A blocking server call run on a background thread, joined later, and the wait over several at once."""

import threading
import time


class AsyncCall:
    """Run a function in a background thread, join later."""

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
        """Join the call within ``timeout`` seconds, re-raising its error; a call still running raises TimeoutError."""
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
    """Block until every call returned, raising as soon as one fails. Polled, not joined in turn: the two
    halves of a group formation complete only together, so a serial join parks once the other side failed."""
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
        time.sleep(0.1)
