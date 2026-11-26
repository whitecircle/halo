#!/usr/bin/env python
"""``rollout_sync_topology: auto`` against live servers: one server stays per-server, two or more identical servers form
one group, and either way every server serves the pushed policy.

``--backend`` picks the engine, ``--servers`` how many of its servers (``HALO_TEST_VLLM_SERVER_URLS`` or
``HALO_TEST_SGLANG_SERVER_URLS``, each serving the same checkpoint with the same launch flags).
"""

import sys

import torch
from transformers import AutoModelForCausalLM

from tests.common.harness import gpu_test_main
from tests.common.on_policy_e2e import probe_top_logprobs, served_policy_moved
from tests.common.rank_log import log
from tests.common.weight_sync import weight_transfer_port
from tests.common.weight_sync_group import BACKENDS, form_group, parse_server_args, timed_push
from tests.common.weight_sync_reinit import PERTURBATION, PERTURBED_PARAM, VLLM_REINIT_MODEL

ARGS, SERVER_URLS = parse_server_args()


def run(ctx) -> dict:
    client_cls, _, _, port_knob = BACKENDS[ARGS.backend]
    urls = SERVER_URLS[: ARGS.servers]
    model = AutoModelForCausalLM.from_pretrained(VLLM_REINIT_MODEL, dtype=torch.bfloat16).to(ctx.device)
    everything = [(name, t.detach()) for name, t in model.named_parameters()]
    checks: dict[str, bool] = {}

    pool = form_group(client_cls, urls, weight_transfer_port(port_knob), ctx.device, topology="auto")
    try:
        checks["auto_forms_a_group_exactly_for_two_or_more_servers"] = (pool.group is not None) == (len(urls) > 1)
        timed_push(pool, everything)
        baseline = [probe_top_logprobs(url, VLLM_REINIT_MODEL) for url in urls]
        with torch.no_grad():
            dict(model.named_parameters())[PERTURBED_PARAM].mul_(PERTURBATION)
        timed_push(pool, [(PERTURBED_PARAM, dict(model.named_parameters())[PERTURBED_PARAM].detach())])
        moved = [probe_top_logprobs(url, VLLM_REINIT_MODEL) for url in urls]
        checks["the_push_moved_every_served_policy"] = all(
            served_policy_moved(m, b) for m, b in zip(moved, baseline, strict=True)
        )
        checks["every_server_serves_one_policy"] = all(probe == moved[0] for probe in moved)
    finally:
        pool.close_communicator()
    return {"checks": ctx.broadcast_checks(checks), "metrics": {"servers": len(urls)}}


main = gpu_test_main(exact_world_size=1, prefix="sync_topology_auto", partial_state=False)(run)

if __name__ == "__main__":
    if len(SERVER_URLS) < ARGS.servers:
        log(
            f"SKIP: --servers {ARGS.servers} needs {ARGS.servers} URLs in {BACKENDS[ARGS.backend][1]}, got {SERVER_URLS}"
        )
        sys.exit(0)
    main()
