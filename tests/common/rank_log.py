"""Log lines for the torchrun test suites: one rank's verdicts or every rank's, rank-tagged, and the messages a
logger emits inside a block."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from halo.runtime.ranks import launcher_global_rank


def log(msg: str, rank: int | None = 0):
    """Print ``msg`` from global rank ``rank`` only, tagged with it."""
    current_rank = launcher_global_rank()
    if current_rank == rank:
        print(f"[Rank {current_rank}] {msg}", flush=True)


def log_all(msg: str):
    """Print ``msg`` from every rank, each line tagged with its rank."""
    print(f"[Rank {launcher_global_rank()}] {msg}", flush=True)


class _MessageLog(logging.Handler):
    """Every message a logger emits while this handler is attached."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextmanager
def logged_messages(logger: logging.Logger) -> Iterator[list[str]]:
    """The messages ``logger`` emits inside the block, filled in as they arrive."""
    handler = _MessageLog()
    logger.addHandler(handler)
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)
