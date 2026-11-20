"""Session users for tests that start the sandbox runner. The runner kills every process of a session's user, so each
xdist worker draws from its own slice of the range."""

import os

from halo.environments.sandbox.runner_session import SESSION_UIDS


def worker_session_uids() -> range:
    workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
    index = int(os.environ.get("PYTEST_XDIST_WORKER", "gw0").removeprefix("gw"))
    users = len(SESSION_UIDS) // workers
    start = SESSION_UIDS.start + index * users
    return range(start, start + users)
