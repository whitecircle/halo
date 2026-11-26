#!/usr/bin/env python
"""SGLang's empty-declare membership probe on a live server: an engine holding the weight-update group answers it as a
no-op, and one that does not answers fast with an error, which a real declare would turn into peers parked in a receive.

Gates SGLang's ``single_group`` topology: if the probe cannot tell the two apart, SGLang stays ``per_server``.
"""

import time

from halo.weight_sync.clients.base import GroupMembershipLost
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from tests.common.harness import gpu_test_main
from tests.common.rank_log import log
from tests.common.weight_sync import SGLANG_SERVER_URL, weight_transfer_port
from tests.common.weight_sync_reinit import CONNECTION_TIMEOUT_S

# An engine without the group fails at once; a probe that waits has become a collective.
_FAST_REFUSAL_S = 10.0


def _probe_outcome(client: SGLangWeightSyncClient) -> tuple[bool, float]:
    """``(group was held, seconds)`` of one probe inside an open session, with the session closed after."""
    client.open_engine_update()
    started = time.monotonic()
    try:
        client.verify_group_membership()
        held = True
    except GroupMembershipLost:
        held = False
    finally:
        seconds = time.monotonic() - started
        client.cancel_engine_update("after the membership probe")
    return held, seconds


def run(ctx) -> dict:
    log(f"SGLang membership probe against {SGLANG_SERVER_URL}")
    client = SGLangWeightSyncClient(
        base_url=SGLANG_SERVER_URL,
        group_port=weight_transfer_port("HALO_TEST_SGLANG_GROUP_PORT"),
        connection_timeout=CONNECTION_TIMEOUT_S,
    )
    client.init_communicator(device=ctx.device)
    try:
        held, held_s = _probe_outcome(client)
        client.destroy_remote_group()
        gone, gone_s = _probe_outcome(client)
    finally:
        client.close_communicator()
    checks = {
        "a_registered_group_answers_the_probe": held,
        "an_unregistered_group_is_reported_lost": not gone,
        "the_refusal_is_fast": gone_s < _FAST_REFUSAL_S,
    }
    return {
        "checks": ctx.broadcast_checks(checks),
        "metrics": {"probe_held_s": held_s, "probe_lost_s": gone_s},
    }


main = gpu_test_main(exact_world_size=1, prefix="sglang_membership_probe", partial_state=False)(run)

if __name__ == "__main__":
    main()
