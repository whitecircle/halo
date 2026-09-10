#!/usr/bin/env python
"""Every Halo image installs one EFA userspace from one script, so the weight-sync group can ride EFA.

The stack is wire-sensitive down to rdma-core: the aws-ofi-nccl plugin probes libfabric for in-order
RDMA writes at init and forces NCCL_PROTO=simple when the probe fails, and that probe answers through
rdma-core's EFA provider. Two containers on different builds form the NCCL group with different
protocol tables and hang at the first collective. The pins therefore live in a single script every
Dockerfile runs whole; a Dockerfile that builds its own plugin, keeps its base's rdma-core, or skips
the script reintroduces a silent cross-image mismatch. The script's own build-time checks (plugin
identity, libfabric version, provider presence, resolution) are pinned here too, since they are what
turn a drifted pin into a failed build.

    python tests/cpu/grpo/test_rollout_server_efa_userspace.py
"""

import re

import pytest

from tests.common.utils import REPO_ROOT

SCRIPT = REPO_ROOT / "docker" / "efa" / "install_efa_userspace.sh"
DOCKERFILES = ("Dockerfile", "Dockerfile.vllm", "Dockerfile.sglang")


def _script() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_the_script_pins_installer_libfabric_and_plugin():
    script = _script()
    assert re.search(r"^EFA_INSTALLER_VERSION=\d+\.\d+\.\d+$", script, re.M)
    assert re.search(r"^EFA_INSTALLER_SHA256=[0-9a-f]{64}$", script, re.M), (
        "the installer tarball is not checksum-pinned"
    )
    assert re.search(r"^LIBFABRIC_VERSION=\S+$", script, re.M)
    assert re.search(r"^AWS_OFI_NCCL_COMMIT=[0-9a-f]{40}$", script, re.M), "the plugin is not pinned to a full commit"


def test_every_dockerfile_installs_the_whole_shared_efa_userspace():
    """Whole, not the plugin alone: rdma-core answers the plugin's in-order-write probe, so an image
    keeping its base's rdma-core lands on a different NCCL protocol table than the others."""
    for name in DOCKERFILES:
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert "COPY docker/efa/install_efa_userspace.sh" in text, f"{name} does not copy the shared EFA script"
        run = re.search(r"RUN /tmp/install_efa_userspace\.sh([^\\\n&]*)", text)
        assert run, f"{name} does not run the shared EFA script"
        assert run.group(1).strip() == "", f"{name} runs the script with arguments ({run.group(1).strip()!r})"
        assert "aws-ofi-nccl.git" not in text, f"{name} clones aws-ofi-nccl itself — a second, drifting pin"
        assert "AWS_OFI_NCCL_COMMIT" not in text, f"{name} carries its own plugin commit — a second, drifting pin"


def _live_line(script: str, text: str) -> bool:
    """Whether ``text`` appears on a line that is not commented out."""
    return re.search(r"^\s*(?!#)[^#\n]*" + re.escape(text), script, re.M) is not None


def test_the_script_installs_rdma_core_alongside_libfabric():
    script = _script()
    for deb in (
        '"$debs"/rdma-core/libibverbs1_*.deb',
        '"$debs"/rdma-core/ibverbs-providers_*.deb',
        '"$debs"/rdma-core/librdmacm1_*.deb',
        '"$debs"/libfabric1-aws_${LIBFABRIC_VERSION}_amd64.deb',
    ):
        assert _live_line(script, deb), f"the script no longer installs {deb}"


def test_the_script_verifies_what_it_installed():
    """A pin bump that produces the wrong artifact must fail the build, not the first sync."""
    script = _script()
    assert _live_line(script, 'grep -F "git-${short}"'), "the plugin build is not checked against the pinned commit"
    assert re.search(
        r'^\s*\|\| \{ echo "plugin does not identify as git-\$\{short\}" >&2; exit 1; \}', script, re.M
    ), "a plugin that does not identify as the pinned commit must fail the build"
    assert _live_line(script, 'grep -F "libfabric: ${LIBFABRIC_VERSION}"'), (
        "the installed libfabric version is not checked"
    )
    assert _live_line(script, "for sym in ncclNetPlugin_v12 ncclGinPlugin_v13"), (
        "the GIN entry point DeepEP V2 needs is not checked"
    )
    assert _live_line(script, "libfabric.so.1 => $EFA_PREFIX/lib/"), "the plugin's libfabric resolution is not checked"
    assert _live_line(script, "libefa=$(ldconfig -p | awk '/libefa\\.so\\.1 /"), (
        "the installed rdma-core EFA provider is not looked up in the loader cache"
    )
    # `grep -q` exits early and the producer takes a SIGPIPE, which `pipefail` reports as a failure.
    assert "grep -q" not in script, "a `grep -q` under pipefail fails a passing check"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
