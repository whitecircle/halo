"""Resolve — and after installing, verify — the ``nvidia-nccl-cu13`` pin every image must share.

``uv.lock`` is the single source of truth for the NCCL runtime: the training image and both
inference images (vLLM, SGLang) dynamically link this wheel's ``libnccl.so.2``, and a version skew
hangs the RL weight-sync ``ncclCommInitRank`` on the NCCL timeout with no error. Every Dockerfile
COPYs this beside its ``uv.lock`` so that value is read one way in one place.

``--verify`` checks the wheel metadata AND the library a process actually ends up with: every image
also carries an older system ``libnccl.so.2`` (the NGC base's, the engine base images' own), so a
build that lost the wheel — or an ``LD_LIBRARY_PATH`` that puts the system copy first — still passes
a metadata-only check while dlopening a runtime the other end cannot pair with.

    python3 nccl_pin.py uv.lock            # print the locked version (feeds `pip install ==`)
    python3 nccl_pin.py uv.lock --verify   # assert the installed wheel AND the resolved library
"""

import argparse
import subprocess
import sys
import tomllib
from importlib.metadata import version

PACKAGE = "nvidia-nccl-cu13"

# DeepEP V2 (EPv2, Gin backend) is built against this floor. Any pip step that re-resolves torch
# silently downgrades the wheel to torch's pinned 2.28.9 and leaves ``deep_ep._C`` with an undefined
# Gin symbol at import, so ``--verify`` asserts the floor as well as equality with the lock.
MINIMUM = "2.30.4"

# Resolve the runtime the way a training/serving process does: importing torch maps the copy its
# RPATH names, so the later soname lookup returns that same handle. Run in a child interpreter so
# the resolver mode above stays a stdlib-only lock parse with no torch import.
_RESOLVED_VERSION_PROBE = """
import ctypes, torch  # torch's preload is what decides which libnccl the soname resolves to
assert torch.version.cuda  # a CPU-only torch would link no libnccl at all
value = ctypes.c_int()
ctypes.CDLL("libnccl.so.2").ncclGetVersion(ctypes.byref(value))
print(value.value)
"""


def _version_tuple(value: str) -> tuple[int, ...]:
    """Numeric prefix of a version, for the floor comparison — stdlib only, no ``packaging``."""
    parts = []
    for chunk in value.split("."):
        if not chunk.isdigit():  # a ``.post1`` / local suffix ends the comparable prefix
            break
        parts.append(int(chunk))
    return tuple(parts)


def version_code(value: str) -> int:
    """``ncclGetVersion``'s integer for a wheel version — ``NCCL_VERSION(X,Y,Z)`` since NCCL 2.9."""
    major, minor, patch = (_version_tuple(value) + (0, 0, 0))[:3]
    return major * 10000 + minor * 100 + patch


def locked_version(lock_path: str) -> str:
    """The version ``uv.lock`` pins for :data:`PACKAGE`."""
    with open(lock_path, "rb") as lock:
        packages = tomllib.load(lock)["package"]
    for package in packages:
        if package["name"] == PACKAGE:
            return package["version"]
    raise SystemExit(f"{PACKAGE} is not pinned in {lock_path} — the images cannot agree on an NCCL runtime")


def resolved_version_code() -> int:
    """``ncclGetVersion`` of the ``libnccl.so.2`` this image resolves once torch has preloaded."""
    probe = subprocess.run(
        [sys.executable, "-c", _RESOLVED_VERSION_PROBE], capture_output=True, text=True, check=False
    )
    if probe.returncode != 0:
        raise SystemExit(f"could not resolve libnccl.so.2 in this image:\n{probe.stderr.strip()}")
    return int(probe.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve or verify the shared NCCL pin from uv.lock")
    parser.add_argument("lock_path", help="Path to the uv.lock copied into this build stage")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Assert the installed wheel and the resolved library match the lock instead of printing it",
    )
    args = parser.parse_args()

    locked = locked_version(args.lock_path)
    if not args.verify:
        print(locked)
        return

    installed = version(PACKAGE)
    if installed != locked:
        raise SystemExit(
            f"{PACKAGE} {installed} != training lock {locked} — the two ends would load skewed NCCL runtimes"
        )
    if _version_tuple(installed) < _version_tuple(MINIMUM):
        raise SystemExit(f"{PACKAGE} {installed} is below the DeepEP V2 floor {MINIMUM} — deep_ep._C will not import")
    resolved = resolved_version_code()
    if resolved != version_code(locked):
        raise SystemExit(
            f"libnccl.so.2 resolves to {resolved}, not the locked {locked} ({version_code(locked)}) — an older "
            f"system copy won the soname race and would pair a skewed runtime with the other end"
        )
    print(f"{PACKAGE}: {installed} (== uv.lock, >= {MINIMUM}); resolved libnccl.so.2: {resolved}")


if __name__ == "__main__":
    main()
