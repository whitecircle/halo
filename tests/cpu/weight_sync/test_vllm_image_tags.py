#!/usr/bin/env python
"""``Dockerfile.vllm`` builds two tags: vLLM's release for every family but Inkling 2, and vLLM's Inkling 2 build for
Inkling 2 alone. The client pins each tag's engine build, the image names its build on ``/server_info``, and a family
on the other tag is refused before training."""

import ast
import dataclasses
import importlib.util
import re
import subprocess
import sys
import types

import pytest

from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.models.moe.family_specs import INKLING2
from halo.trainers.grpo.rollout.engine_preflight import ServerRequirements, verify_server_flags
from halo.weight_sync.clients.base import EngineImage
from halo.weight_sync.clients.vllm import _IDENTITY_KEY, VLLMWeightSyncClient
from tests.common.fake_engines import FakeVLLMServer
from tests.common.repo import REPO_ROOT, load_script_module

DOCKERFILE = (REPO_ROOT / "Dockerfile.vllm").read_text(encoding="utf-8")
MAKEFILE = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
PATCHES_DIR = REPO_ROOT / "docker/vllm/patches"
SPLIT_GUARD = "vllm_topp_split_guard_patch"
RELEASE, INKLING2_TAG = VLLMWeightSyncClient.IMAGE, VLLMWeightSyncClient.INKLING2_IMAGE
PATCHES = "vllm_layerwise_reload_patch"


def _arg(name: str) -> str:
    [value] = re.findall(rf"^ARG {name}=(\S+)$", DOCKERFILE, re.MULTILINE)
    return value


def _stage(name: str) -> str:
    """The instructions of one build stage, up to the next ``FROM``."""
    [body] = re.findall(rf"^FROM \S+ AS {re.escape(name)}\n(.*?)(?=^FROM )", DOCKERFILE, re.MULTILINE | re.DOTALL)
    return body


@pytest.fixture
def server():
    fake = FakeVLLMServer()
    yield fake
    fake.close()


def test_the_client_pins_the_tags_and_builds_the_dockerfile_builds():
    """A client pin that drifts from the build's refuses the image the Makefile builds."""
    assert EngineImage(f"halo:vllm-{_arg('VLLM_VERSION')}", _arg("VLLM_VERSION"), "make build-vllm") == RELEASE
    assert (
        EngineImage(
            f"halo:vllm-{_arg('VLLM_INKLING2_VERSION')}", _arg("VLLM_INKLING2_BUILD"), "make build-vllm-inkling2"
        )
        == INKLING2_TAG
    )
    assert f"vLLM {RELEASE.build}" == VLLMWeightSyncClient.ENGINE_RELEASE
    for image in (RELEASE, INKLING2_TAG):
        target = image.make_target.removeprefix("make ")
        assert re.search(rf"^{target}:", MAKEFILE, re.MULTILINE), f"the Makefile has no {target} target"


@pytest.mark.parametrize(
    ("target", "version", "inkling2"),
    [("build-vllm", "VLLM_VERSION", False), ("build-vllm-inkling2", "VLLM_INKLING2_VERSION", True)],
)
def test_each_make_target_builds_its_own_tag(target, version, inkling2):
    """A target that tags the other engine publishes Inkling 2's build under the release's name, or the reverse."""
    recipe = _recipe(target)
    assert f"-t halo:vllm-$({version})" in recipe
    assert ("--build-arg VLLM_ENGINE=inkling2" in recipe) is inkling2
    assert re.search(
        rf"^{version}\s*:= \$\(shell sed -n 's/\^ARG {version}=//p' Dockerfile\.vllm\)$", MAKEFILE, re.MULTILINE
    )
    assert _arg(version)


def _recipe(target: str) -> str:
    [recipe] = re.findall(rf"^{target}:.*\n((?:\t.*\n)+)", MAKEFILE, re.MULTILINE)
    return recipe


def test_the_inkling2_targets_serve_and_dial_one_port():
    """Compose binds ``VLLM_PORT``; a trainer that dials a fixed port misses a server moved off it."""
    serve, train = _recipe("up-vllm-inkling2"), _recipe("train-inkling2")
    assert "VLLM_PORT=$(VLLM_PORT)" in serve and "VLLM_IMAGE=$(INKLING2_VLLM_IMAGE)" in serve
    assert "curl -sf $(VLLM_SERVER_URL)/health" in train and "--rollout_server_url=$(VLLM_SERVER_URL)" in train
    assert not re.search(r"\b8000\b", serve + train), "a port is spelled by hand"
    assert re.search(r"^VLLM_SERVER_URL \?= http://localhost:\$\(VLLM_PORT\)$", MAKEFILE, re.MULTILINE)
    assert "rollout_server_url" in {field.name for field in dataclasses.fields(AsyncRolloutConfig)}


@pytest.mark.parametrize(("server", "trainer"), [("2,3", "0,1"), ("4,5,6,7", "0,1,2,3"), ("7", "0,1,2,3,4,5")])
def test_the_inkling2_targets_size_the_server_and_trainer_by_their_device_lists(server, trainer):
    """A fixed tensor-parallel size or process count beside an overridden device list launches a server or trainer
    that disagrees with the GPUs it sees."""
    overrides = [f"INKLING2_SERVER_DEVICES={server}", f"INKLING2_TRAINER_DEVICES={trainer}", "HALO_SCRATCH=/scratch"]
    serve, train = (
        subprocess.run(
            ["make", "-n", "-C", str(REPO_ROOT), target, *overrides], capture_output=True, text=True, check=True
        ).stdout
        for target in ("up-vllm-inkling2", "train-inkling2")
    )
    assert re.findall(r"VLLM_TP=(\d+)", serve) == [str(len(server.split(",")))]
    assert re.findall(r"VLLM_CUDA_DEVICES=(\S+)", serve) == [server]
    assert re.findall(r"--nproc (\d+)", train) == [str(len(trainer.split(",")))]
    assert re.findall(r"-e CUDA_VISIBLE_DEVICES=(\S+)", train) == [trainer]


def test_each_tag_carries_its_own_engine_specifics():
    """The release pins Model Runner V2 and ships the split top-p guard; the Inkling 2 build has neither a V1 runner
    nor the wiped row, and takes Inkling 2's source edits."""
    release, inkling2 = _stage("server-release"), _stage("server-inkling2")
    shared = DOCKERFILE[DOCKERFILE.index("\nFROM server-${VLLM_ENGINE}\n") :]
    assert "ENV VLLM_USE_V2_MODEL_RUNNER=1" in release
    assert f"COPY docker/vllm/patches/{SPLIT_GUARD}.py /opt/vllm_patches/" in release
    assert "RUN cd /opt/halo && python3 patch_vllm_inkling2.py" in inkling2
    for other in (inkling2, shared):
        assert "VLLM_USE_V2_MODEL_RUNNER" not in other and SPLIT_GUARD not in other
    for other in (release, shared):
        assert "patch_vllm_inkling2" not in other
    assert _arg("VLLM_ENGINE") == "release", "a bare build is not the image every family is served from"
    assert "\nFROM engine-${VLLM_ENGINE} AS " in DOCKERFILE


def test_the_build_asserts_the_engine_build_of_the_tag_it_builds():
    """``inkling-2-small`` is a moving upstream tag: the base is pinned by digest and the build checks the version."""
    assert re.fullmatch(r"vllm/vllm-openai@sha256:[0-9a-f]{64}", _arg("VLLM_INKLING2_BASE"))
    assert _arg("VLLM_INKLING2_BUILD").startswith(f"{_arg('VLLM_INKLING2_VERSION')}+"), "the tag is another build's"
    assert "FROM ${VLLM_INKLING2_BASE} AS engine-inkling2" in DOCKERFILE
    assert "FROM vllm/vllm-openai:v${VLLM_VERSION} AS engine-release" in DOCKERFILE
    runs = re.findall(r"^RUN (?:.*\\\n)*.*$", DOCKERFILE, re.MULTILINE)
    [verify] = [step for step in runs if "verify_image.py" in step]
    assert 'release) build="${VLLM_VERSION}"' in verify and 'inkling2) build="${VLLM_INKLING2_BUILD}"' in verify
    assert 'python3 /tmp/verify_image.py "${VLLM_ENGINE}" "${build}"' in verify
    script = (REPO_ROOT / "docker/vllm/verify_image.py").read_text(encoding="utf-8")
    assert "    check_engine_build(engine, pinned_build)\n" in script and "    check_model_runner_v2,\n" in script


@pytest.mark.parametrize("guard_shipped", [True, False], ids=["release", "inkling2"])
def test_sitecustomize_applies_the_split_guard_only_where_the_image_ships_it(monkeypatch, guard_shipped):
    """The hook table is one file for both tags; the guard's row follows the file the release stage copies."""
    find_spec = importlib.util.find_spec

    def find(name, *args):
        if name == SPLIT_GUARD:
            return object() if guard_shipped else None
        return find_spec(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find)
    before = list(sys.meta_path)
    try:
        hooks = load_script_module("docker/vllm/patches/sitecustomize.py")
    finally:
        sys.meta_path[:] = before
    assert (SPLIT_GUARD in hooks._PATCHES.values()) is guard_shipped
    assert "vllm_layerwise_reload_patch" in hooks._PATCHES.values()
    assert (PATCHES_DIR / f"{SPLIT_GUARD}.py").is_file()


@pytest.mark.parametrize("image", [RELEASE, INKLING2_TAG], ids=["release", "inkling2"])
def test_the_identity_row_the_image_writes_is_the_build_the_client_reads(monkeypatch, server, image):
    """The row is ``<vllm.__version__>|<patches>``; a format the client parses another way refuses every server."""
    envs = types.ModuleType("vllm.envs")
    envs.environment_variables = {}
    engine = types.ModuleType("vllm")
    engine.__version__, engine.envs = image.build, envs
    monkeypatch.setitem(sys.modules, "vllm", engine)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs)
    hooks = types.SimpleNamespace(_PATCHES={"vllm.envs": "vllm_image_identity_patch", "vllm.reload": PATCHES})
    monkeypatch.setitem(sys.modules, "sitecustomize", hooks)
    patch = load_script_module("docker/vllm/patches/vllm_image_identity_patch.py")
    assert patch.IDENTITY_KEY == _IDENTITY_KEY

    patch.apply()
    server.identity = envs.environment_variables[_IDENTITY_KEY]()

    assert server.identity == f"{image.build}|vllm_image_identity_patch,{PATCHES}"
    flags = VLLMWeightSyncClient(base_url=server.url).read_server_flags()
    assert (flags.toolkit_image, flags.engine_build) == (True, image.build)


@pytest.mark.parametrize("identity", [PATCHES, f"|{PATCHES}", f"{RELEASE.build}|", f"0.26.0|{PATCHES}"])
def test_a_row_without_a_pinned_build_is_no_toolkit_image(server, identity):
    """The row's presence proves nothing: it names a build one of the two tags pins, with its patches."""
    server.identity = identity
    assert VLLMWeightSyncClient(base_url=server.url).read_server_flags().toolkit_image is False


def _verify(server, model_types) -> None:
    needs = ServerRequirements(is_moe=True, model_types=model_types)
    verify_server_flags([server.url], needs, backend=VLLMWeightSyncClient.BACKEND_KEY, connection_timeout=5.0)


@pytest.mark.parametrize("model_types", [("qwen3_5_moe",), (), *[(each,) for each in INKLING2.model_types]])
def test_a_family_passes_the_preflight_on_its_own_tag(server, model_types):
    server.identity = f"{VLLMWeightSyncClient.image_for(model_types).build}|{PATCHES}"
    _verify(server, model_types)


@pytest.mark.parametrize(
    ("model_types", "served", "remedy"),
    [
        (("inkling2",), RELEASE, INKLING2_TAG),
        (("inkling2", "inkling2_text"), RELEASE, INKLING2_TAG),
        (("qwen3_5_moe",), INKLING2_TAG, RELEASE),
        (("glm5_next",), INKLING2_TAG, RELEASE),
        ((), INKLING2_TAG, RELEASE),
    ],
)
def test_a_family_on_the_other_tag_is_refused_with_the_image_that_serves_it(server, model_types, served, remedy):
    """The release has no Inkling 2 model, and no other family has run on the Inkling 2 build."""
    server.identity = f"{served.build}|{PATCHES}"
    with pytest.raises(ValueError) as refusal:
        _verify(server, model_types)
    message = str(refusal.value)
    assert f"engine build {served.build!r}" in message
    assert f"Serve from {remedy.tag} ({remedy.make_target})" in message
    assert f"Serve from {served.tag}" not in message


def test_a_server_off_both_tags_is_sent_to_the_image_of_the_run_s_family(server):
    server.toolkit_image = False
    with pytest.raises(ValueError, match=re.escape(f"Serve from {INKLING2_TAG.tag} ({INKLING2_TAG.make_target})")):
        _verify(server, ("inkling2",))


@pytest.mark.parametrize("script", ["environmental.py", "online.py"])
def test_each_script_tells_the_preflight_the_model_s_types(script):
    """Without them every model resolves to the release tag, and Inkling 2 on it passes the preflight."""
    tree = ast.parse((REPO_ROOT / "scripts/training/grpo" / script).read_text(encoding="utf-8"))
    [needs] = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "ServerRequirements"
    ]
    given = {keyword.arg: ast.unparse(keyword.value) for keyword in needs.keywords}
    assert given.get("model_types") == "config_model_types(model.config)"


def test_only_a_family_no_push_reaches_is_served_from_another_tag():
    """Group formation takes the release build alone, which holds while every family on another tag pushes no base."""
    assert set(VLLMWeightSyncClient.FAMILY_IMAGES) == set(INKLING2.model_types)
    assert set(VLLMWeightSyncClient.FAMILY_IMAGES.values()) == {INKLING2_TAG}
    assert INKLING2.weight_sync_refusal is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
