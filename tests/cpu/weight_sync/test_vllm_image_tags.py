#!/usr/bin/env python
"""``Dockerfile.vllm`` builds two tags: vLLM's release for every family but Inkling 2, and vLLM's Inkling 2 build for
Inkling 2 alone."""

import importlib.util
import re
import sys

import pytest

from tests.common.repo import REPO_ROOT, load_script_module

DOCKERFILE = (REPO_ROOT / "Dockerfile.vllm").read_text(encoding="utf-8")
MAKEFILE = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
PATCHES_DIR = REPO_ROOT / "docker/vllm/patches"
SPLIT_GUARD = "vllm_topp_split_guard_patch"


def _arg(name: str) -> str:
    [value] = re.findall(rf"^ARG {name}=(\S+)$", DOCKERFILE, re.MULTILINE)
    return value


def _stage(name: str) -> str:
    """The instructions of one build stage, up to the next ``FROM``."""
    [body] = re.findall(rf"^FROM \S+ AS {re.escape(name)}\n(.*?)(?=^FROM )", DOCKERFILE, re.MULTILINE | re.DOTALL)
    return body


@pytest.mark.parametrize(
    ("target", "version", "inkling2"),
    [("build-vllm", "VLLM_VERSION", False), ("build-vllm-inkling2", "VLLM_INKLING2_VERSION", True)],
)
def test_each_make_target_builds_its_own_tag(target, version, inkling2):
    """A target that tags the other engine publishes Inkling 2's build under the release's name, or the reverse."""
    [recipe] = re.findall(rf"^{target}:.*\n((?:\t.*\n)+)", MAKEFILE, re.MULTILINE)
    assert f"-t halo:vllm-$({version})" in recipe
    assert ("--build-arg VLLM_ENGINE=inkling2" in recipe) is inkling2
    assert re.search(
        rf"^{version}\s*:= \$\(shell sed -n 's/\^ARG {version}=//p' Dockerfile\.vllm\)$", MAKEFILE, re.MULTILINE
    )
    assert _arg(version)


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
