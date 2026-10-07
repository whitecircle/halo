"""The one wrap of transformers' remote-class funnel, the callbacks registered on it, and whole-file
writes into the module cache it loads from.

``dynamic_module_utils.get_class_in_module`` is the funnel every ``trust_remote_code`` class loads
through, and the only seam that can reach a modeling module that does not exist until then. It is
wrapped ONCE here so the compat shims and the deferred Liger patching share it: two independent
wrappers nest, and re-applying either (a second model load, TRL's Liger re-application) then stacks
another. A leaf — stdlib plus ``dynamic_module_utils`` — so both registrants can import it.

Before that import, transformers copies the module into ``HF_MODULES_CACHE``, which a node's ranks
share, by truncating and rewriting the file under a per-process lock. A rank that finds the file while
a peer is still writing it imports a module cut mid-body, which can load without an error (a config
class missing what its ``__init__`` never reached). Importing this module makes every copy
``dynamic_module_utils`` makes land whole, the remote code a save writes beside a checkpoint
included; the loaders and the training entry points import it before their first remote load.
"""

from __future__ import annotations

import contextlib
import functools
import os
import shutil
import sys
import uuid
from collections.abc import Callable
from types import ModuleType, SimpleNamespace

import transformers.dynamic_module_utils

# Callbacks run against each remote modeling module the funnel loads, in registration order.
_HOOKS: list[Callable[[ModuleType], None]] = []

_FUNNEL_WRAPPED_MARKER = "_halo_remote_class_hook"
# Set on the ``shutil`` stand-in bound into ``dynamic_module_utils``.
_WHOLE_COPY_MARKER = "_halo_whole_copy"


def register_remote_class_hook(hook: Callable[[ModuleType], None]) -> None:
    """Run ``hook(module)`` for every remote modeling module transformers loads a class from.

    Idempotent in both directions: a hook already registered is not added twice, and the funnel wrap
    is installed once per process.
    """
    if hook not in _HOOKS:
        _HOOKS.append(hook)
    _wrap_funnel()


def _wrap_funnel() -> None:
    original = transformers.dynamic_module_utils.get_class_in_module
    if getattr(original, _FUNNEL_WRAPPED_MARKER, False):
        return

    @functools.wraps(original)
    def get_class_in_module_hooked(class_name, module_path, **kwargs):
        cls = original(class_name, module_path, **kwargs)
        module = sys.modules.get(cls.__module__)
        if module is not None:
            for hook in _HOOKS:
                hook(module)
        return cls

    setattr(get_class_in_module_hooked, _FUNNEL_WRAPPED_MARKER, True)
    transformers.dynamic_module_utils.get_class_in_module = get_class_in_module_hooked


def _copyfile_whole(src: str | os.PathLike, dst: str | os.PathLike, *, follow_symlinks: bool = True):
    """``shutil.copyfile`` that publishes ``dst`` only once all of it is written.

    The bytes go to a temp file beside ``dst`` with a random name (pids repeat across the containers
    of a shared filesystem), which ``os.replace`` then renames over it: a reader sees the previous file
    or the complete new one, on a local or a network filesystem alike.
    """
    temp = f"{os.fspath(dst)}.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copyfile(src, temp, follow_symlinks=follow_symlinks)
        os.replace(temp, dst)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.remove(temp)
        raise
    return dst


def _write_module_cache_whole() -> None:
    """Route ``dynamic_module_utils``' ``shutil`` through :func:`_copyfile_whole`, once per process.

    Only ``copyfile`` is exposed, so a transformers release that starts calling another ``shutil``
    function there fails on it by name rather than writing the cache in place again.
    """
    bound = getattr(transformers.dynamic_module_utils, "shutil", None)
    if getattr(bound, _WHOLE_COPY_MARKER, False):
        return
    if bound is not shutil:
        raise RuntimeError(
            "transformers.dynamic_module_utils no longer reaches shutil as a module attribute, so its "
            "module-cache copies cannot be made whole; update src/models/patches/remote_code_hooks.py."
        )
    transformers.dynamic_module_utils.shutil = SimpleNamespace(copyfile=_copyfile_whole, **{_WHOLE_COPY_MARKER: True})


_write_module_cache_whole()
