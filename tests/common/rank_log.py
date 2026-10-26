"""Rank-tagged log lines for the torchrun test suites: one rank's verdicts, or every rank's."""

from halo.runtime.ranks import launcher_global_rank


def log(msg: str, rank: int | None = 0):
    """Print ``msg`` from global rank ``rank`` only, tagged with it."""
    current_rank = launcher_global_rank()
    if current_rank == rank:
        print(f"[Rank {current_rank}] {msg}", flush=True)


def log_all(msg: str):
    """Print ``msg`` from every rank, each line tagged with its rank."""
    print(f"[Rank {launcher_global_rank()}] {msg}", flush=True)
