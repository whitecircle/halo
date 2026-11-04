#!/usr/bin/env python
"""The shipped rollout-server compose files must serve what the trainer's defaults assume. These are launch flags, so
a missing one fails silently at runtime: the run trains, logs a loss, and is quietly wrong."""

import re

import pytest
import yaml

from tests.common.repo import REPO_ROOT

VLLM_COMPOSE = REPO_ROOT / "docker-compose.vllm.yml"
SGLANG_COMPOSE = REPO_ROOT / "docker-compose.sglang.yml"
VLLM_EFA_OVERLAY = REPO_ROOT / "docker-compose.vllm.efa.yml"
SGLANG_EFA_OVERLAY = REPO_ROOT / "docker-compose.sglang.efa.yml"
MAKEFILE = REPO_ROOT / "Makefile"


def _services(compose_path: str) -> dict:
    with open(compose_path) as fh:
        return yaml.safe_load(fh)["services"]


def _server_command(compose_path: str, service: str) -> str:
    command = _services(compose_path)[service]["command"]
    return command if isinstance(command, str) else " ".join(command)


def _environment(service: dict) -> dict[str, str]:
    env: dict[str, str] = {}
    for entry in service.get("environment", []):
        key, _, value = entry.partition("=")
        env[key] = value
    return env


def _make_build_tag(target: str, dockerfile: str) -> str:
    """The ``-t`` tag of a ``make`` build target, each ``$(NAME)`` filled from the Dockerfile's ``ARG NAME``."""
    recipe = re.search(rf"^{re.escape(target)}:.*\n((?:\t.*\n)+)", MAKEFILE.read_text(encoding="utf-8"), re.MULTILINE)
    assert recipe, f"Makefile has no {target} recipe"
    tag = re.search(r"\s-t\s+(\S+)", recipe.group(1))
    assert tag, f"Makefile {target} tags no image"
    args = dict(re.findall(r"^ARG (\w+)=(\S+)$", (REPO_ROOT / dockerfile).read_text(encoding="utf-8"), re.MULTILINE))
    return re.sub(r"\$\((\w+)\)", lambda name: args[name.group(1)], tag.group(1))


@pytest.mark.parametrize(
    ("compose_path", "service", "target", "dockerfile"),
    [
        (VLLM_COMPOSE, "vllm-server", "build-vllm", "Dockerfile.vllm"),
        (SGLANG_COMPOSE, "sglang-server", "build-sglang", "Dockerfile.sglang"),
        (VLLM_COMPOSE, "training", "build-blackwell", "Dockerfile"),
    ],
)
def test_compose_runs_the_image_make_builds(compose_path, service, target, dockerfile):
    """``make`` is the one build path, so ``up`` finds a locally built image only under its tag; a rename on either
    side sends compose to pull a name no registry serves."""
    spec = _services(compose_path)[service]
    assert "build" not in spec, f"{compose_path.name} {service} builds its own image; make {target} is the build path"
    default = re.fullmatch(r"\$\{\w+:-(\S+)\}", spec["image"])
    assert default, f"{compose_path.name} {service} must take its image as ${{VAR:-default}}, got {spec['image']!r}"
    built = _make_build_tag(target, dockerfile)
    assert default.group(1) == built, (
        f"{compose_path.name} {service} runs {default.group(1)}, make {target} tags {built}"
    )


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


def test_vllm_compose_keeps_prefix_caching_on_by_default():
    """R3 capture refuses to start without prefix caching, and a multi-turn episode re-prefills its whole context every
    turn when it is off. Explicit, so a change in vLLM's default cannot drop it."""
    assert "${VLLM_PREFIX_CACHING_FLAG:---enable-prefix-caching}" in _server_command(VLLM_COMPOSE, "vllm-server")


def test_vllm_compose_turns_strict_tool_calling_off_by_default():
    """vLLM defaults it on, masking every tool-bearing request by a grammar the trainer's log-probs never see."""
    env = _environment(_services(VLLM_COMPOSE)["vllm-server"])
    assert env.get("VLLM_ENFORCE_STRICT_TOOL_CALLING") == "${VLLM_ENFORCE_STRICT_TOOL_CALLING:-0}", env


def test_the_vllm_image_turns_strict_tool_calling_off_for_a_hand_run_server():
    """vLLM defaults it on, and the documented ``docker run`` recipe passes no env: the GLM parsers would then sample
    under a grammar the trainer's log-probs never see."""
    dockerfile = (REPO_ROOT / "Dockerfile.vllm").read_text(encoding="utf-8")
    assert re.search(r"^ENV VLLM_ENFORCE_STRICT_TOOL_CALLING=0$", dockerfile, re.MULTILINE)
    assert "check_tool_calling_grammar_off" in (REPO_ROOT / "docker/vllm/verify_image.py").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("compose_path", "service", "prefix"),
    [(VLLM_COMPOSE, "vllm-server", "VLLM"), (SGLANG_COMPOSE, "sglang-server", "SGLANG")],
    ids=["vllm", "sglang"],
)
def test_each_compose_takes_remote_code_and_further_flags(compose_path, service, prefix):
    """Ling 2.0 and Step-3.7 need the first and a server path differing from the trainer's id ``--served-model-name``.
    With no slot compose cannot serve them, and a hand-run server restates every load-bearing flag or loses it."""
    command = _server_command(compose_path, service)
    assert f"${{{prefix}_TRUST_REMOTE_CODE:+--trust-remote-code}}" in command
    assert command.rstrip().endswith(f"${{{prefix}_EXTRA_ARGS:-}}"), "extra flags must come last, to win a repeat"


def test_vllm_compose_passes_the_model_runner_through_bare():
    """An empty value kills the server at startup (int() parse), and a default of 0 would force the V1
    runner, which R3 refuses and on which thinking budgets misfire; bare, the image's pin holds."""
    assert "VLLM_USE_V2_MODEL_RUNNER" in _services(VLLM_COMPOSE)["vllm-server"].get("environment", [])


def test_vllm_compose_enables_the_weight_transfer_endpoints():
    """Without the nccl weight-transfer config the endpoints the trainer drives do not exist."""
    command = _server_command(VLLM_COMPOSE, "vllm-server")
    assert "--weight-transfer-config" in command
    assert "nccl" in command


def test_both_composes_configure_a_tool_call_parser():
    """Without one the engine returns the tool call as plain text, every episode ends unsolved at reward 0, and the run
    trains on a flat zero gradient with no error. vLLM's pair uses the unset-only ``-`` operator, so only an explicitly
    empty ``VLLM_TOOL_CALLING_FLAGS`` (the ReAct recipes) drops it; ``:+`` would leave it off by default."""
    vllm = _server_command(VLLM_COMPOSE, "vllm-server")
    assert (
        "${VLLM_TOOL_CALLING_FLAGS---enable-auto-tool-choice --tool-call-parser ${VLLM_TOOL_PARSER:-hermes}}" in vllm
    ), vllm
    sglang = _server_command(SGLANG_COMPOSE, "sglang-server")
    assert "--tool-call-parser" in sglang, (
        "docker-compose.sglang.yml must pass --tool-call-parser: without it tool-using environments "
        "silently score zero for every rollout"
    )


def test_sglang_compose_matches_the_trainers_cumem_and_leaves_the_transports_to_nccl():
    """SGLang sets ``NCCL_CUMEM_ENABLE=0`` process-wide unless already set; the trainer's NCCL has cuMem on, and the
    mismatch fails the first cross-container buffer import. Transports and plugin stay at NCCL's own selection, since a
    same-host group runs over CUDA IPC and the EFA overlay names the plugin."""
    env = _environment(_services(SGLANG_COMPOSE)["sglang-server"])
    assert env.get("NCCL_CUMEM_ENABLE") == "${NCCL_CUMEM_ENABLE:-1}", env
    for forced in ("NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_NET_PLUGIN"):
        assert forced not in env, f"docker-compose.sglang.yml sets {forced}"


def test_efa_overlays_put_every_service_of_their_base_on_the_fabric():
    """An overlay service the base lacks makes compose reject the project; a base service the overlay skips stays on
    sockets, and a trainer on the fabric hangs against it. The net is named so a missing plugin fails formation
    instead of silently falling back to sockets."""
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
    """A protocol table set on the trainer alone hangs the first collective. A ``${NCCL_PROTO:-}`` default would hand
    NCCL an explicit, empty protocol list, and a defaulted value would pin the server where the trainer is not."""
    for compose in (VLLM_COMPOSE, SGLANG_COMPOSE):
        for name, service in _services(compose).items():
            assert "NCCL_PROTO" in service.get("environment", []), (compose, name, "no bare NCCL_PROTO pass-through")


@pytest.mark.parametrize(
    ("compose_path", "variable"), [(VLLM_COMPOSE, "HALO_SCRATCH"), (SGLANG_COMPOSE, "SGLANG_MODEL_DIR")]
)
def test_a_host_path_the_compose_file_mounts_has_no_fallback(compose_path, variable):
    """A path's name does not prove its capacity (``/mnt`` shares the small root disk on some hosts), and a bind mount
    of a missing host path creates it."""
    operators = re.findall(rf"\$\{{{variable}([^}}]*)\}}", compose_path.read_text(encoding="utf-8"))
    assert operators, f"{compose_path.name} no longer reads {variable}"
    assert all(op == "" or op.startswith(":?") for op in operators), (
        f"{compose_path.name} defaults {variable}: {operators}"
    )
    assert any(op.startswith(":?") for op in operators), f"{compose_path.name} never requires {variable}"


@pytest.mark.parametrize(
    ("compose_path", "service"),
    [(VLLM_COMPOSE, "vllm-server"), (SGLANG_COMPOSE, "sglang-server")],
    ids=["vllm", "sglang"],
)
def test_each_server_sees_the_chat_templates_the_recipes_pin(compose_path, service):
    """The recipe headers name /jinja-templates/..., and neither image holds a template of its own."""
    assert "./jinja-templates:/jinja-templates:ro" in _services(compose_path)[service]["volumes"]
    assert (REPO_ROOT / "jinja-templates").is_dir()


def test_the_compose_command_carries_no_yaml_comment_lines():
    """A ``#`` inside a folded block scalar is text and becomes an argv entry; compose then rejects the service with
    'invalid command line string'."""
    for path, service in ((VLLM_COMPOSE, "vllm-server"), (SGLANG_COMPOSE, "sglang-server")):
        command = _server_command(path, service)
        assert "#" not in command, f"{path}: command contains a '#' YAML did not treat as a comment: {command!r}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
