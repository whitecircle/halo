"""Session users for tests that start the sandbox runner. The runner kills every process of a session's user, so each
xdist worker draws from its own slice of the range."""

import os

from halo.environments.sandbox.runner_session import SESSION_UIDS

USERS_PER_WORKER = 1000


def worker_session_uids() -> range:
    index = int(os.environ.get("PYTEST_XDIST_WORKER", "gw0").removeprefix("gw"))
    start = SESSION_UIDS.start + index % (len(SESSION_UIDS) // USERS_PER_WORKER) * USERS_PER_WORKER
    return range(start, start + USERS_PER_WORKER)
