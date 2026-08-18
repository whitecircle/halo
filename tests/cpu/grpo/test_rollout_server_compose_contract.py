#!/usr/bin/env python
"""The shipped rollout-server compose files must serve what the trainer's defaults assume.

These are launch flags, not code, so nothing else in the suite notices when one goes missing — and
each of them fails *silently* at runtime: the run trains, logs a loss, and is quietly wrong.
``--return-tokens-as-token-ids`` is the sharpest case. ``train_on_sampled_tokens`` defaults to
``True`` and recovers the sampled ids from the logprobs, which vLLM only spells out under that flag;
without it every turn falls back to re-tokenizing a chat-template re-render behind a single warning.
"""

import os
import re
import sys

import pytest
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
VLLM_COMPOSE = os.path.join(PROJECT_ROOT, "docker-compose.vllm.yml")
SGLANG_COMPOSE = os.path.join(PROJECT_ROOT, "docker-compose.sglang.yml")


def _server_command(compose_path: str, service: str) -> str:
    with open(compose_path) as fh:
        compose = yaml.safe_load(fh)
    command = compose["services"][service]["command"]
    return command if isinstance(command, str) else " ".join(command)


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
