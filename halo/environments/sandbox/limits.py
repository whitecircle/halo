"""The limits and the environment every backend gives a program."""

import os

SANDBOX_DEFAULT_MEMORY_MB = 1024
# Build-step wall-clock cap (seconds); bounds an #include bomb in untrusted source.
SANDBOX_DEFAULT_COMPILE_TIMEOUT = 30.0
SANDBOX_DEFAULT_COMPILE_MEMORY_MB = 2048
# Bytes a child may write to one file; bounds disk fill and binary size.
LOCAL_FSIZE_LIMIT = 64 * 1024 * 1024
# RLIMIT_NPROC for the run step. The kernel exempts uid 0, so it binds every bubblewrap run (per user namespace) and
# `local` only in a container run as another user, where it counts the whole trainer+Ray task set.
LOCAL_NPROC_LIMIT = 4096
# RLIMIT_CPU headroom over the wall-clock timeout, so SIGXCPU only fires as the backstop.
RLIMIT_CPU_SLACK_SECONDS = 1.0


# Host-sized math thread pools allocate per-thread buffers at load, which the address-space limit refuses; numpy's
# OpenBLAS then aborts a correct program at import.
_SINGLE_THREADED_MATH_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

# A tampered working directory is the program's own runtime error; an infra fault would void the episode.
TAMPERED_WORKDIR_RETURNCODE = 1


def build_program_env(workdir: str) -> dict[str, str]:
    """A program's environment but ``PATH``, which each backend takes from the host the program runs on."""
    return {
        "HOME": workdir,
        "TMPDIR": workdir,
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        **_SINGLE_THREADED_MATH_ENV,
    }
