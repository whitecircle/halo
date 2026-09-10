#!/usr/bin/env python
"""The shipped rollout-server compose files must serve what the trainer's defaults assume.

These are launch flags, not code, so nothing else in the suite notices when one goes missing — and
each of them fails *silently* at runtime: the run trains, logs a loss, and is quietly wrong.
``--return-tokens-as-token-ids`` is the sharpest case. ``train_on_sampled_tokens`` defaults to
``True`` and recovers the sampled ids from the logprobs, which vLLM only spells out under that flag;
without it every turn falls back to re-tokenizing a chat-template re-render behind a single warning.
"""

import re
import sys

import pytest
import yaml

from tests.common.utils import REPO_ROOT

VLLM_COMPOSE = REPO_ROOT / "docker-compose.vllm.yml"
SGLANG_COMPOSE = REPO_ROOT / "docker-compose.sglang.yml"
VLLM_EFA_OVERLAY = REPO_ROOT / "docker-compose.vllm.efa.yml"
SGLANG_EFA_OVERLAY = REPO_ROOT / "docker-compose.sglang.efa.yml"


def _services(compose_path: str) -> dict:
    with open(compose_path) as fh:
        return yaml.safe_load(fh)["services"]


def _server_command(compose_path: str, service: str) -> str:
    command = _services(compose_path)[service]["command"]
    return command if isinstance(command, str) else " ".join(command)


def _environment(service: dict) -> dict[str, str]:
    """The ``KEY=value`` entries of a service's environment list, value still uninterpolated."""
    env: dict[str, str] = {}
    for entry in service.get("environment", []):
        key, _, value = entry.partition("=")
        env[key] = value
    return env


def test_vllm_compose_requests_token_ids_that_train_on_sampled_tokens_needs():
    command = _server_command(VLLM_COMPOSE, "vllm-server")
    assert "--return-tokens-as-token-ids" in command, (
        "docker-compose.vllm.yml must pass --return-tokens-as-token-ids: train_on_sampled_tokens "
        "defaults on and silently falls back to re-tokenization without it"
    )


def test_vllm_compose_pins_the_only_moe_backend_weight_sync_is_correct_under():
    """FLASHINFER/CUTLASS repack expert weights at load, so a synced update corrupts them."""
    command = _server_command(VLLM_COMPOSE, "vllm-server")
    match = re.search(r"--moe-backend\s+(\S+)", command)
    assert match, "docker-compose.vllm.yml must pass --moe-backend explicitly"
    assert "triton" in match.group(1), f"--moe-backend must default to triton, got {match.group(1)!r}"


def test_vllm_compose_enables_the_weight_transfer_endpoints():
    """Without the nccl weight-transfer config the endpoints the trainer drives do not exist."""
    command = _server_command(VLLM_COMPOSE, "vllm-server")
    assert "--weight-transfer-config" in command
    assert "nccl" in command


def test_both_composes_configure_a_tool_call_parser():
    """Without one the engine returns the model's tool call as plain text.

    The environment then sees no ``tool_calls``, every episode ends unsolved at reward 0, and every
    GRPO group is degenerate — the run trains to completion with a flat zero gradient and no error.
    """
    vllm = _server_command(VLLM_COMPOSE, "vllm-server")
    assert "--tool-call-parser" in vllm and "--enable-auto-tool-choice" in vllm
    sglang = _server_command(SGLANG_COMPOSE, "sglang-server")
    assert "--tool-call-parser" in sglang, (
        "docker-compose.sglang.yml must pass --tool-call-parser: without it tool-using environments "
        "silently score zero for every rollout"
    )


def test_sglang_compose_matches_the_trainers_cumem_and_leaves_the_transports_to_nccl():
    """SGLang sets ``NCCL_CUMEM_ENABLE=0`` process-wide unless it is already set; the trainer's NCCL
    has cuMem on, and the mismatch fails the first cross-container buffer import. The compose file
    pre-sets it; the CUDA-IPC and shared-memory transports and the plugin stay at NCCL's own
    selection, since a same-host group runs over CUDA IPC and the EFA overlay names the plugin."""
    env = _environment(_services(SGLANG_COMPOSE)["sglang-server"])
    assert env.get("NCCL_CUMEM_ENABLE") == "${NCCL_CUMEM_ENABLE:-1}", env
    for forced in ("NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_NET_PLUGIN"):
        assert forced not in env, f"docker-compose.sglang.yml sets {forced}"


def test_efa_overlays_put_every_service_of_their_base_on_the_fabric():
    """An overlay service the base lacks makes compose reject the project; a base service the
    overlay skips stays on sockets, and a trainer on the fabric then hangs against it. Each service
    needs the EFA devices, the pinned memory limit, and the net named so a missing plugin fails
    formation instead of silently falling back to sockets."""
    for base, overlay in ((VLLM_COMPOSE, VLLM_EFA_OVERLAY), (SGLANG_COMPOSE, SGLANG_EFA_OVERLAY)):
        overlay_services = _services(overlay)
        assert set(overlay_services) == set(_services(base)), (base, overlay)
        for name, service in overlay_services.items():
            assert "/dev/infiniband:/dev/infiniband" in service.get("devices", []), (overlay, name)
            assert service.get("ulimits", {}).get("memlock") == -1, (overlay, name)
            env = _environment(service)
            assert env.get("NCCL_NET") == "${NCCL_NET:-Libfabric}", (overlay, name, env)
            assert env.get("NCCL_NET_PLUGIN") == "${NCCL_NET_PLUGIN:-ofi}", (overlay, name, env)
            assert env.get("NCCL_IB_DISABLE") == "${NCCL_IB_DISABLE:-0}", (overlay, name, env)
            # An excluded-only interface list still ranks loopback first, and a server that
            # advertises 127.0.0.1 to a trainer on another node fails formation.
            assert env.get("NCCL_SOCKET_IFNAME", "").startswith("${NCCL_SOCKET_IFNAME:-^lo,"), (overlay, name, env)


def test_every_server_and_trainer_service_passes_nccl_proto_through_bare():
    """A protocol table set on the trainer alone hangs the first collective, on every recipe. The
    entry must be bare: a ``${NCCL_PROTO:-}`` default would hand NCCL an explicit, empty protocol
    list, and a defaulted value would pin the server where the trainer is not."""
    for compose in (VLLM_COMPOSE, SGLANG_COMPOSE):
        for name, service in _services(compose).items():
            assert "NCCL_PROTO" in service.get("environment", []), (compose, name, "no bare NCCL_PROTO pass-through")


def test_the_compose_command_carries_no_yaml_comment_lines():
    """A ``#`` inside a folded block scalar is text, not a comment — it becomes an argv entry.

    Adding the token-ids flag with an explanatory comment inline did exactly this, and compose
    rejected the service with 'invalid command line string'.
    """
    for path, service in ((VLLM_COMPOSE, "vllm-server"), (SGLANG_COMPOSE, "sglang-server")):
        command = _server_command(path, service)
        assert "#" not in command, f"{path}: command contains a '#' YAML did not treat as a comment: {command!r}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
