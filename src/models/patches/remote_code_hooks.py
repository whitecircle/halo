"""The one wrap of transformers' remote-class funnel, and the callbacks registered on it.

``dynamic_module_utils.get_class_in_module`` is the funnel every ``trust_remote_code`` class loads
through, and the only seam that can reach a modeling module that does not exist until then. It is
wrapped ONCE here so the compat shims and the deferred Liger patching share it: two independent
wrappers nest, and re-applying either (a second model load, TRL's Liger re-application) then stacks
another. A leaf — stdlib plus ``dynamic_module_utils`` — so both registrants can import it.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable
from types import ModuleType

import transformers.dynamic_module_utils

# Callbacks run against each remote modeling module the funnel loads, in registration order.
_HOOKS: list[Callable[[ModuleType], None]] = []

_FUNNEL_WRAPPED_MARKER = "_halo_remote_class_hook"


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
