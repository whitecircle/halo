#!/usr/bin/env python
"""A ``single_group`` weight sync when a server dies. Between syncs the pre-declare check finds it, the group re-forms
and the sync completes with no restart of the healthy servers; mid-stream the failure is global: the trainer returns
within a bound naming the dead server, every server is left paused and unusable, and a restart of all of them is what
lets the next sync through.

``HALO_TEST_SYNC_SERVER_CONTROL`` is an executable called ``<it> restart <index>`` (kill that server's container and
start it again) and ``<it> kill <index>``, ``<it> restart-all``; each returns once the servers it touched serve again.
The index is the position in the backend's server list.
"""

import subprocess
import sys
import time
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM

from halo.env import env_str
from halo.weight_sync.group import WeightSyncGroup
from tests.common.harness import gpu_test_main
from tests.common.on_policy_e2e import probe_top_logprobs
from tests.common.rank_log import log
from tests.common.weight_sync import weight_transfer_port
from tests.common.weight_sync_group import BACKENDS, form_group, parse_server_args, timed_push
from tests.common.weight_sync_reinit import VLLM_REINIT_MODEL

# The dead server's declare is refused at once; a trainer that waits a broadcast's deadline has lost the bound.
MAX_FAILURE_S = 120.0
_CHUNKS_BEFORE_THE_KILL = 1


def _control(*words: str) -> None:
    subprocess.run([env_str("HALO_TEST_SYNC_SERVER_CONTROL"), *words], check=True, timeout=1800)


ARGS, SERVER_URLS = parse_server_args(default_servers=2)


def run(ctx) -> dict:
    client_cls, _, _, port_knob = BACKENDS[ARGS.backend]
    urls = SERVER_URLS[: ARGS.servers]
    last = len(urls) - 1
    port = weight_transfer_port(port_knob)
    model = AutoModelForCausalLM.from_pretrained(VLLM_REINIT_MODEL, dtype=torch.bfloat16).to(ctx.device)
    everything = [(name, t.detach()) for name, t in model.named_parameters()]
    checks: dict[str, bool] = {}

    pool = form_group(client_cls, urls, port, ctx.device)
    try:
        timed_push(pool, everything)
        baseline = [probe_top_logprobs(url, VLLM_REINIT_MODEL) for url in urls]
        _control("restart", str(last))
        timed_push(pool, everything)
        recovered = [probe_top_logprobs(url, VLLM_REINIT_MODEL) for url in urls]
        checks["a_restart_between_syncs_re_forms_and_the_sync_completes"] = all(
            probe == baseline[0] for probe in recovered
        )

        sent = []
        real = WeightSyncGroup._transmit_chunk

        def kill_after_first_chunk(group, chunk, final=False):
            real(group, chunk, final)
            sent.append(len(chunk))
            if len(sent) == _CHUNKS_BEFORE_THE_KILL:
                _control("kill", str(last))

        started = time.monotonic()
        failure = None
        with patch.object(WeightSyncGroup, "_transmit_chunk", kill_after_first_chunk):
            try:
                timed_push(pool, everything)
            except RuntimeError as e:
                failure = e
        seconds = time.monotonic() - started
        checks["the_trainer_returned_within_the_bound"] = failure is not None and seconds < MAX_FAILURE_S
        checks["the_error_names_the_dead_server_first"] = (
            failure is not None
            and str(failure).find(urls[last]) < str(failure).find(urls[0])
            and "RESTART" in str(failure)
        )
        checks["every_server_is_unusable"] = pool.group.is_unusable
    finally:
        pool.close_communicator()

    _control("restart-all")
    after = form_group(client_cls, urls, port, ctx.device)
    try:
        timed_push(after, everything)
        landed = [probe_top_logprobs(url, VLLM_REINIT_MODEL) for url in urls]
        checks["after_restarting_all_the_next_sync_lands_one_policy"] = all(probe == landed[0] for probe in landed)
    finally:
        after.close_communicator()
    return {"checks": ctx.broadcast_checks(checks), "metrics": {"failure_s": seconds}}


main = gpu_test_main(exact_world_size=1, prefix="single_group_failure", partial_state=False)(run)

if __name__ == "__main__":
    if len(SERVER_URLS) < ARGS.servers or not env_str("HALO_TEST_SYNC_SERVER_CONTROL"):
        log(f"SKIP: needs {ARGS.servers} URLs in {BACKENDS[ARGS.backend][1]} and HALO_TEST_SYNC_SERVER_CONTROL")
        sys.exit(0)
    main()
