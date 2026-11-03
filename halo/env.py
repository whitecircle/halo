"""Readers for the toolkit's env knobs (``HALO_``/``DIST_``/``VLLM_``/``SGLANG_``/``NVLINK_``); launcher, HF and OS
variables are read raw at their owner. The body is stdlib-only, but importing it runs ``halo/__init__.py``."""

import logging
import os
from datetime import timedelta
from typing import TypeGuard, overload

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}

DEFAULT_NCCL_TIMEOUT_MINUTES = 30
# Past this fraction of the NCCL watchdog, a rollout straggler leaves the cancelled rank too little margin to unwind.
WATCHDOG_WARN_FRACTION = 0.8
# Hours, not minutes: store waits outlast one rank's download or packing, not a collective.
DEFAULT_STORE_TIMEOUT_HOURS = 4
# NFS's default ``acdirmax``: a cached negative lookup can hide a peer's new file this long.
DEFAULT_FS_PROBE_TIMEOUT_SECONDS = 60


def _is_set(raw: str | None) -> TypeGuard[str]:
    """Absent and blank both read as unset: a compose pass-through (``- HALO_FOO``) exports an absent var empty."""
    return raw is not None and bool(raw.strip())


def env_flag(name: str, default: bool = False) -> bool:
    """Boolean env var: ``1/true/yes/on`` (any case) → True, anything else → False, unset → ``default``."""
    raw = os.environ.get(name)
    if _is_set(raw):
        return raw.strip().lower() in _TRUE_VALUES
    return default


@overload
def env_int(name: str, default: int) -> int: ...
@overload
def env_int(name: str, default: None) -> int | None: ...
def env_int(name: str, default: int | None) -> int | None:
    """Integer env var; unset → ``default``, and a malformed value warns and falls back to it."""
    raw = os.environ.get(name)
    if _is_set(raw):
        try:
            return int(raw)
        except ValueError:
            logger.warning(f"Ignoring invalid {name}={raw!r} (expected an integer); using {default}.")
    return default


@overload
def env_float(name: str, default: float) -> float: ...
@overload
def env_float(name: str, default: None) -> float | None: ...
def env_float(name: str, default: float | None) -> float | None:
    """Float env var; unset → ``default``, and a malformed value warns and falls back to it."""
    raw = os.environ.get(name)
    if _is_set(raw):
        try:
            return float(raw)
        except ValueError:
            logger.warning(f"Ignoring invalid {name}={raw!r} (expected a number); using {default}.")
    return default


def env_positive_int(name: str, default: int) -> int:
    """Integer env var that must be >= 1 (a ``0`` timeout would expire at once); else warns and falls back."""
    value = env_int(name, default)
    if value is None or value < 1:
        logger.warning(f"Ignoring {name}={value!r} (expected a positive integer); using {default}.")
        return default
    return value


def env_positive_float(name: str, default: float | None) -> float | None:
    """Float env var that must be positive; else warns and falls back."""
    value = env_float(name, default)
    if value is not None and value <= 0:
        logger.warning(f"Ignoring {name}={value!r} (expected a positive number); using {default}.")
        return default
    return value


def nccl_timeout() -> timedelta:
    return timedelta(minutes=env_positive_int("DIST_NCCL_TIMEOUT_MINUTES", DEFAULT_NCCL_TIMEOUT_MINUTES))


def watchdog_bounded_seconds() -> float:
    """Longest a rank may block in one rollout call, since every peer waits at the next collective."""
    return nccl_timeout().total_seconds() * WATCHDOG_WARN_FRACTION


def store_timeout() -> timedelta:
    return timedelta(hours=env_positive_int("DIST_STORE_TIMEOUT_HOURS", DEFAULT_STORE_TIMEOUT_HOURS))


def fs_probe_timeout() -> timedelta:
    """How long a shared-storage rank polls for the filesystem probe's sentinel: at least the attribute-cache bound."""
    return timedelta(seconds=env_positive_int("DIST_FS_PROBE_TIMEOUT_SECONDS", DEFAULT_FS_PROBE_TIMEOUT_SECONDS))


@overload
def env_str(name: str, default: str) -> str: ...
@overload
def env_str(name: str, default: None = None) -> str | None: ...
def env_str(name: str, default: str | None = None) -> str | None:
    """As ``os.environ.get``: an empty value is returned; write ``env_str(name) or default`` to treat it as unset."""
    raw = os.environ.get(name)
    return default if raw is None else raw


# Toolkit scratch only (HF caches follow HF_HOME); ``or`` stops an exported-empty value resolving against the cwd.
HALO_DATA_ROOT = env_str("HALO_DATA_ROOT") or os.path.join(os.path.expanduser("~"), ".cache", "halo")


def data_path(*parts: str) -> str:
    return os.path.join(HALO_DATA_ROOT, *parts)


def torch_trace_dir() -> str:
    """Read at call time so a patched ``HALO_DATA_ROOT`` applies."""
    return data_path("profiling", "torch")


def is_accelerate_fsdp_launch() -> bool:
    return env_flag("ACCELERATE_USE_FSDP")


def is_accelerate_launch() -> bool:
    """``accelerate launch`` sets ``ACCELERATE_MIXED_PRECISION`` for every config; torchrun never does."""
    return env_str("ACCELERATE_MIXED_PRECISION") is not None or is_accelerate_fsdp_launch()
