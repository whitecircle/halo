"""Environment-variable helpers for the toolkit's own knobs (``HALO_``/``DIST_``/``VLLM_``).

Booleans parse through :func:`env_flag` (``1/true/yes/on``, case-insensitive), numbers through
:func:`env_int` / :func:`env_float` (warn and fall back rather than raise mid-run), strings through
:func:`env_str` (absent → ``default``, empty string preserved). :data:`HALO_DATA_ROOT` and
:func:`data_path` resolve the toolkit scratch location. A knob's default is declared next to its
resolver here; every other toolkit constant belongs to the module implementing the behaviour.

Variables the toolkit does not define are read raw at their point of use: launcher rank vars
(``LOCAL_RANK``/``SLURM_*``/``RANK``) in ``src.distributed.runtime``, HuggingFace and OS vars by their
own conventions, and third-party library vars (``EP_*`` for DeepEP, ``TRITON_CACHE_DIR``,
``FLASH_ATTENTION_CUTE_DSL_*``, ``WANDB_*``/``CLEARML_*``, ``CUDA_DEVICE_MAX_CONNECTIONS``) at the
module wiring that integration, so each parse matches the library's own. The ``ACCELERATE_*`` launch
markers are the exception: :func:`is_accelerate_launch` / :func:`is_accelerate_fsdp_launch` read them
here because callers across the toolkit need a consistent answer.

The body is stdlib-only, but importing this module executes ``src/__init__.py`` (torch,
transformers); ``tests/gpu/conftest.py`` reads raw to avoid that.
"""

import logging
import os
from typing import TypeGuard, overload

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}

# Declared here rather than in the distributed helpers so config dataclasses can read it without
# importing torch.
DEFAULT_NCCL_TIMEOUT_MINUTES = 30
# Largest fraction of the NCCL collective watchdog a rollout timeout may reach before it is warned
# about; beyond it a straggler leaves too little margin for the cancelled rank to unwind.
# :func:`watchdog_bounded_seconds` derives unconfigured timeouts from the same ceiling.
WATCHDOG_WARN_FRACTION = 0.8

# Bound for c10d-store coordination waits (main-first downloads, the per-node load throttle). Hours
# rather than minutes: these wait out a single rank's download/packing work, not a collective, so the
# NCCL watchdog scale does not apply.
DEFAULT_STORE_TIMEOUT_HOURS = 4


def _is_set(raw: str | None) -> TypeGuard[str]:
    """Whether a raw env value counts as set: absent and empty/whitespace both read as unset.

    Docker/compose pass-through (``- HALO_FOO``) exports the name with an empty value when the
    variable is absent from the caller's environment, which would otherwise force every
    ``default=True`` flag off. :func:`env_str` is the exception: it mirrors ``os.environ.get`` and
    preserves an explicit empty string.
    """
    return raw is not None and bool(raw.strip())


def env_flag(name: str, default: bool = False) -> bool:
    """Boolean env var: ``1/true/yes/on`` (case-insensitive) → True, anything else → False.

    An unset value (:func:`_is_set`) yields ``default``.
    """
    raw = os.environ.get(name)
    if _is_set(raw):
        return raw.strip().lower() in _TRUE_VALUES
    return default


@overload
def env_int(name: str, default: int) -> int: ...
@overload
def env_int(name: str, default: None) -> int | None: ...
def env_int(name: str, default: int | None) -> int | None:
    """Integer env var; a malformed value warns and falls back to ``default`` (never raises).

    Unset follows :func:`_is_set`, as in :func:`env_flag`: warning on a pass-through's empty value
    would fire on every rank at import. Overloaded like :func:`env_str` so a concrete default narrows
    the result to ``int``, sparing callers an ``or``-guard that would also swallow a legitimate ``0``.
    """
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
    """Float env var; a malformed value warns and falls back to ``default`` (never raises).

    Unset follows :func:`_is_set`, as in :func:`env_int`, and it is overloaded the same way.
    """
    raw = os.environ.get(name)
    if _is_set(raw):
        try:
            return float(raw)
        except ValueError:
            logger.warning(f"Ignoring invalid {name}={raw!r} (expected a number); using {default}.")
    return default


def env_positive_int(name: str, default: int) -> int:
    """Integer env var that must be >= 1; a malformed or non-positive value warns and falls back.

    Timeouts read this rather than :func:`env_int`: a ``0`` would mean "expire immediately" and
    disable the watchdog the knob configures.
    """
    value = env_int(name, default)
    if value is None or value < 1:
        logger.warning(f"Ignoring {name}={value!r} (expected a positive integer); using {default}.")
        return default
    return value


def env_positive_float(name: str, default: float | None) -> float | None:
    """Float env var that must be positive; a malformed or non-positive value warns and falls back.

    As with :func:`env_positive_int`, a ``0`` timeout would expire on entry.
    """
    value = env_float(name, default)
    if value is not None and value <= 0:
        logger.warning(f"Ignoring {name}={value!r} (expected a positive number); using {default}.")
        return default
    return value


def resolve_nccl_timeout_minutes() -> int:
    """NCCL collective-watchdog timeout in minutes: ``DIST_NCCL_TIMEOUT_MINUTES`` → default.

    Used by both the process-group construction and the rollout-timeout validation, which must agree.
    """
    return env_positive_int("DIST_NCCL_TIMEOUT_MINUTES", DEFAULT_NCCL_TIMEOUT_MINUTES)


def watchdog_bounded_seconds() -> float:
    """Longest a rank may block in one rollout call and still leave the peers' collective margin.

    A blocking rollout holds every peer at the next collective, so it must clear the watchdog by
    :data:`WATCHDOG_WARN_FRACTION`, the bound the environmental-GRPO timeout guard warns at. Lowering
    ``DIST_NCCL_TIMEOUT_MINUTES`` lowers the derived rollout timeouts with it.
    """
    return resolve_nccl_timeout_minutes() * 60 * WATCHDOG_WARN_FRACTION


def resolve_store_timeout_hours() -> int:
    """c10d-store coordination-wait bound in hours: ``DIST_STORE_TIMEOUT_HOURS`` → default."""
    return env_positive_int("DIST_STORE_TIMEOUT_HOURS", DEFAULT_STORE_TIMEOUT_HOURS)


@overload
def env_str(name: str, default: str) -> str: ...
@overload
def env_str(name: str, default: None = None) -> str | None: ...
def env_str(name: str, default: str | None = None) -> str | None:
    """String env var: absent → ``default``; a set value (empty string included) is returned as-is.

    Mirrors ``os.environ.get`` semantics (only an *absent* key falls back), so a caller that must
    treat an exported-empty value as unset writes ``env_str(name) or default`` rather than passing the
    default here. Overloaded so a concrete ``default`` narrows the result to ``str``.
    """
    raw = os.environ.get(name)
    return default if raw is None else raw


# Toolkit-owned caches and outputs only; HF caches follow HF_HOME / HF_DATASETS_CACHE. Point this at a
# large mounted volume in production; the home-dir fallback keeps a fresh clone runnable with no setup.
# The ``or`` treats an exported-empty value as unset, since an empty scratch root would resolve every
# data_path() to a relative path.
HALO_DATA_ROOT = env_str("HALO_DATA_ROOT") or os.path.join(os.path.expanduser("~"), ".cache", "halo")


def data_path(*parts: str) -> str:
    """Join ``parts`` under :data:`HALO_DATA_ROOT`, for toolkit scratch (dataset cache, profiler
    traces). Callers may still accept an explicit override that takes precedence."""
    return os.path.join(HALO_DATA_ROOT, *parts)


def torch_trace_dir() -> str:
    """Default scratch directory for torch-profiler artifacts, shared by the CLI flag, the callback
    and the session so traces land where the report tool looks. A function rather than a constant, so
    the args dataclass default and the tests can read it after patching ``HALO_DATA_ROOT``."""
    return data_path("profiling", "torch")


def memory_snapshot_dir() -> str:
    """Default scratch directory for CUDA memory snapshots."""
    return data_path("profiling", "memory")


def is_accelerate_fsdp_launch() -> bool:
    """Whether this process was launched by ``accelerate launch`` with FSDP enabled."""
    return os.environ.get("ACCELERATE_USE_FSDP", "").strip().lower() in _TRUE_VALUES


def is_accelerate_launch() -> bool:
    """Whether this process was launched by ``accelerate launch`` (any distributed_type).

    ``ACCELERATE_MIXED_PRECISION`` is set by the launcher for every config (MULTI_GPU/DDP and
    FSDP alike); torchrun launches never set it.
    """
    return os.environ.get("ACCELERATE_MIXED_PRECISION") is not None or is_accelerate_fsdp_launch()
