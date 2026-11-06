#!/usr/bin/env python
"""``rollout_sync_topology: single_group`` against live servers: one group over every server lands the same bytes on
all of them, in the one rank block each, and costs about what one server does. A group's first push also carries
process and communicator warm-up, so the bar is on the second push and the first is reported beside it; run it with
``NCCL_DEBUG=INFO`` to see where a first push's time goes.

``--backend`` picks the engine, ``--servers`` how many of its servers (``HALO_TEST_VLLM_SERVER_URLS`` or
``HALO_TEST_SGLANG_SERVER_URLS``, each serving the same checkpoint on its own GPUs; mix TP sizes in the list for the
mixed-layout row). A push of the checkpoint, a perturbed norm that must move every served policy identically, a push that
skips that tensor and so must leave the perturbation standing (the negative control), then a restore.
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

# The design's bar: N servers within this factor of one.
MAX_SLOWDOWN_VS_ONE_SERVER = 1.5

ARGS, SERVER_URLS = parse_server_args()


def run(ctx) -> dict:
    client_cls, _, _, port_knob = BACKENDS[ARGS.backend]
    urls = SERVER_URLS[: ARGS.servers]
    port = weight_transfer_port(port_knob)
    model = AutoModelForCausalLM.from_pretrained(VLLM_REINIT_MODEL, dtype=torch.bfloat16).to(ctx.device)
    everything = [(name, t.detach()) for name, t in model.named_parameters()]
    rest = [(name, t) for name, t in everything if name != PERTURBED_PARAM]
    original = dict(everything)[PERTURBED_PARAM].clone()
    checks: dict[str, bool] = {}

    pool = form_group(client_cls, urls, port, ctx.device)
    group = pool.group
    try:
        ranks = sorted(rank for offset, world in group.rank_blocks for rank in range(offset, offset + world))
        checks["ranks_are_exactly_the_group"] = ranks == list(range(1, group.world_size))
        first_s = timed_push(pool, everything)
        serve_s = timed_push(pool, everything)
        baseline = [probe_top_logprobs(url, VLLM_REINIT_MODEL) for url in urls]
        checks["servers_serve_one_policy"] = all(probe == baseline[0] for probe in baseline)

        with torch.no_grad():
            dict(model.named_parameters())[PERTURBED_PARAM].mul_(PERTURBATION)
        perturbed_param = dict(model.named_parameters())[PERTURBED_PARAM].detach()
        timed_push(pool, [(PERTURBED_PARAM, perturbed_param)])
        moved = [probe_top_logprobs(url, VLLM_REINIT_MODEL) for url in urls]
        checks["the_push_moved_every_served_policy"] = all(
            served_policy_moved(m, b) for m, b in zip(moved, baseline, strict=False)
        )
        checks["every_server_landed_the_same_bytes"] = all(probe == moved[0] for probe in moved)

        timed_push(pool, rest)
        skipped = [probe_top_logprobs(url, VLLM_REINIT_MODEL) for url in urls]
        checks["a_push_that_skips_a_tensor_leaves_it_standing"] = skipped == moved

        timed_push(pool, [(PERTURBED_PARAM, original)])
        restored = [probe_top_logprobs(url, VLLM_REINIT_MODEL) for url in urls]
        checks["a_restoring_push_returns_every_server_to_the_baseline"] = restored == baseline
    finally:
        pool.close_communicator()

    one = form_group(client_cls, urls[:1], port, ctx.device)
    try:
        one_first_s = timed_push(one, everything)
        one_s = timed_push(one, everything)
    finally:
        one.close_communicator()
    checks["n_servers_cost_at_most_1_5x_one"] = serve_s <= MAX_SLOWDOWN_VS_ONE_SERVER * one_s
    gigabytes = sum(t.nbytes for _, t in everything) / 1e9
    metrics = {
        "servers": len(urls),
        "group_first_sync_s": first_s,
        "group_sync_s": serve_s,
        "one_server_first_sync_s": one_first_s,
        "one_server_sync_s": one_s,
        "gb_per_s_per_unique_byte": gigabytes / serve_s,
    }
    return {"checks": ctx.broadcast_checks(checks), "metrics": metrics}


main = gpu_test_main(exact_world_size=1, prefix="single_group_sync", partial_state=False)(run)

if __name__ == "__main__":
    if len(SERVER_URLS) < ARGS.servers:
        log(
            f"SKIP: --servers {ARGS.servers} needs {ARGS.servers} URLs in {BACKENDS[ARGS.backend][1]}, got {SERVER_URLS}"
        )
        sys.exit(0)
    main()
