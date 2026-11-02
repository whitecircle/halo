"""Same-device fast path for the device context Liger enters around each kernel launch.

Liger 0.8.4 wraps every RMSNorm, SwiGLU, GeGLU and fused CE+TVD launch in ``liger_kernel.ops.utils.device_context``,
a generator context manager that enters ``torch.cuda.device`` on each call so that a tensor on another GPU launches on
its own device. On a tensor already on the current device the switch is a no-op that still costs host time: on one B300,
``LigerRMSNormFunction`` forward and backward at X ``[1, 2048, 4096]`` bf16 takes 337 µs of host time with it and 236 µs
without, and a profile of dense Qwen3-8B CP2 and TP2 training steps (4,096-token rows) puts it at about 11 µs per
RMSNorm forward call. On that release the ops modules that imported the helper are rebound to one that skips the
switch there and enters Liger's own context otherwise. On any other release Liger's own context stays, warned once.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType

import liger_kernel.ops
import torch
from liger_kernel.ops import utils as liger_ops_utils

from halo.log import warn_once
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# The release whose helper the fast path is written against; any other keeps Liger's own context.
VERIFIED_LIGER_VERSION = "0.8.4"

# Importing the package loads every default kernel module, so the scan sees them all.
_OPS_PREFIX = f"{liger_kernel.ops.__name__}."
# Read without requiring the name: a release that drops it must still import this module and run stock.
_LIGER_DEVICE_CONTEXT = getattr(liger_ops_utils, "device_context", None)

_warned_versions: set[str | None] = set()


def same_device_context(device: torch.device | str | int | None) -> contextlib.AbstractContextManager[None]:
    """A null context when ``device`` is the current CUDA device, else Liger's ``device_context(device)``."""
    if isinstance(device, torch.device) and device.type == "cuda" and device.index == torch.cuda.current_device():
        return contextlib.nullcontext()
    return _LIGER_DEVICE_CONTEXT(device)


def _installed_liger_version() -> str | None:
    """The installed liger-kernel release, or ``None`` when no distribution metadata names one (a source tree)."""
    try:
        return version("liger-kernel")
    except PackageNotFoundError:
        return None


def _check_liger_helper() -> None:
    helper = getattr(liger_ops_utils, "device_context", None)
    if (
        helper is None
        or helper is not _LIGER_DEVICE_CONTEXT
        or helper.__module__ != liger_ops_utils.__name__
        or helper.__qualname__ != "device_context"
    ):
        raise RuntimeError(
            f"liger_kernel.ops.utils.device_context is {helper!r}, not Liger's own helper: something rebound or "
            f"removed it, so the toolkit's same-device fast path cannot tell what it would replace."
        )


def pin_liger_device_context() -> tuple[str, ...]:
    """Rebind ``device_context`` to :func:`same_device_context` in every loaded ``liger_kernel.ops`` module that
    imported Liger's helper, and return the modules that carry the fast path. Idempotent.

    On a liger-kernel other than :data:`VERIFIED_LIGER_VERSION`, or one without distribution metadata, nothing is
    rebound: Liger's own context stays, a warning says so once, and the result is empty. On the verified release,
    raises when the helper is not Liger's own, when a module binds the name to anything else, and when no module
    imports the helper.
    """
    installed = _installed_liger_version()
    if installed != VERIFIED_LIGER_VERSION:
        if is_global_main_process():
            found = (
                f"liger-kernel {installed} is installed" if installed else "liger-kernel has no distribution metadata"
            )
            warn_once(
                logger,
                _warned_versions,
                installed,
                f"{found}, and the toolkit's same-device fast path for Liger's device_context is written against "
                f"{VERIFIED_LIGER_VERSION} only, so Liger's own launch context stays: results are unchanged, and every "
                f"Liger launch that enters device_context keeps its device switch, which costs host time. "
                f"Install liger-kernel=={VERIFIED_LIGER_VERSION}, the locked version, for the fast path.",
            )
        return ()
    _check_liger_helper()
    bindings = {
        name: vars(module)["device_context"]
        for name, module in sorted(sys.modules.items())
        if name.startswith(_OPS_PREFIX)
        and isinstance(module, ModuleType)
        and module is not liger_ops_utils
        and "device_context" in vars(module)
    }
    foreign = [name for name, bound in bindings.items() if bound not in (_LIGER_DEVICE_CONTEXT, same_device_context)]
    if foreign:
        raise RuntimeError(
            f"{', '.join(f'{name}.device_context' for name in foreign)} is neither Liger's helper nor the "
            f"toolkit's same-device fast path: something else rebound it."
        )
    if not bindings:
        raise RuntimeError(
            "No liger_kernel.ops module imports liger_kernel.ops.utils.device_context: Liger moved its launch "
            "context, so the toolkit's same-device fast path no longer applies."
        )
    for name in bindings:
        sys.modules[name].device_context = same_device_context
    return tuple(bindings)
