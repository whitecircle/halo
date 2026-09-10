#!/usr/bin/env python
"""Weight-sync transport preflight: form the group against a live rollout server and name the transport.

A trainer and a rollout server that disagree on the NCCL net, on the aws-ofi-nccl build, or on
cuMem form their weight-sync group and then hang at the first collective. This runs the toolkit's
own client against the server, pushes one real parameter of the served checkpoint (unchanged, so
the served model is unchanged), and reports what NCCL chose — ``efa`` (``NET/Libfabric`` with
GPUDirect), ``ib`` (NCCL's own InfiniBand transport), ``socket`` (TCP over the host network),
``p2p`` (same-host CUDA IPC), ``shm`` (same-host shared memory) — with the plugin build string, the
libfabric provider and the push rate. With ``--expect`` it is a gate.

    # trainer node, server on another node over EFA
    python scripts/profiling/weight_sync_transport.py --server-url http://10.0.0.7:8000 --expect efa
    # same host, SGLang, machine-readable
    python scripts/profiling/weight_sync_transport.py --server-url http://localhost:30000 \\
        --backend sglang --expect p2p --json

Run it from the trainer container, launched exactly as the trainer would be (same image, devices
and NCCL env), on a GPU the server does not own. The checkpoint read here must be the one the
server loaded (``--model-id`` when the served id is a server-side path or an alias). Exits 1 when
the transport differs from ``--expect``, when the push altered the served model, or when the group
fails to form.
"""

import argparse
import json
import sys

from src.diagnostics.weight_sync_transport import VERDICTS, format_report, run_preflight
from src.distributed.nccl.registry import rollout_backends
from src.log import configure_cli_logging

DEFAULT_CONNECTION_TIMEOUT_S = 120.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server-url", required=True, help="rollout server base URL, e.g. http://10.0.0.7:8000")
    parser.add_argument("--backend", choices=rollout_backends(), default="vllm")
    parser.add_argument("--group-port", type=int, default=0, help="rendezvous port on this host (0 = ephemeral)")
    parser.add_argument("--group-host", default=None, help="address the server dials back to (default: auto)")
    parser.add_argument("--device", default="cuda:0", help="GPU the group forms on; must not be the server's")
    parser.add_argument("--param", default=None, help="checkpoint parameter to push (default: the input embedding)")
    parser.add_argument(
        "--model-id", default=None, help="checkpoint id or path the server loaded (default: the id it advertises)"
    )
    parser.add_argument("--revision", default=None, help="Hub revision of --model-id (default: main)")
    parser.add_argument("--rounds", type=int, default=3, help="pushes to time after the group formed")
    parser.add_argument(
        "--connection-timeout", type=float, default=DEFAULT_CONNECTION_TIMEOUT_S, help="seconds to wait for /health"
    )
    parser.add_argument(
        "--expect", choices=VERDICTS, default=None, help="exit 1 unless the group formed on this transport"
    )
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args()
    configure_cli_logging()

    result = run_preflight(
        server_url=args.server_url,
        backend=args.backend,
        group_port=args.group_port,
        group_host=args.group_host,
        device=args.device,
        param=args.param,
        model_id=args.model_id,
        revision=args.revision,
        rounds=args.rounds,
        connection_timeout=args.connection_timeout,
    )
    print(json.dumps(result, indent=2) if args.json else format_report(result))

    failures = []
    if args.expect and result["verdict"] != args.expect:
        failures.append(f"transport is {result['verdict']}, expected {args.expect}")
    if not result["served_unchanged"]:
        failures.append("the push changed the served model")
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
