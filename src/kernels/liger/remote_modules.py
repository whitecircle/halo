"""Deferred kernel patching for `trust_remote_code` modeling modules.

The module does not exist when Liger is applied — the orchestrator runs before ``from_pretrained`` — so
patches are armed on the shared remote-class funnel hook
(:mod:`src.models.patches.remote_code_hooks`) and matched by the CLASS NAMES a spec declares: a repo
that renames its modeling file stays covered, one that renames the classes fails loudly.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import ModuleType

from accelerate.logging import get_logger

from src.models.patches.remote_code_hooks import register_remote_class_hook

logger = get_logger(__name__)

# (required class names, patch) armed for the modules transformers has not loaded yet.
_ARMED: list[tuple[tuple[str, ...], Callable[[ModuleType], object]]] = []


def _declared_classes(module: ModuleType, class_names: tuple[str, ...]) -> set[str]:
    return {name for name in class_names if isinstance(getattr(module, name, None), type)}


def _defines_all(module: ModuleType, class_names: tuple[str, ...]) -> bool:
    return _declared_classes(module, class_names) == set(class_names)


def _fire(module: ModuleType) -> None:
    """Run every armed patch whose classes this module defines.

    A module carrying SOME of a spec's classes is the drift case — a repo revision that renamed one
    of them. Silence there would leave the run unfused after the applier already logged that it was
    armed, so it warns rather than skipping quietly. Not a raise: an unrelated repo may legitimately
    share one class name, and taking its model load down would be worse than the missed fusion.
    """
    for class_names, patch in _ARMED:
        present = _declared_classes(module, class_names)
        if present == set(class_names):
            patch(module)
            logger.info(f"Liger Kernel applied to remote modeling module {module.__name__}")
        elif present:
            logger.warning(
                f"Liger Kernel NOT applied to {module.__name__}: it defines {sorted(present)} but "
                f"not {sorted(set(class_names) - present)}. This revision has renamed a class the "
                f"family's LigerFamilySpec names — the run trains unfused until the spec is updated."
            )


def patch_remote_modules(class_names: tuple[str, ...], patch: Callable[[ModuleType], object]) -> None:
    """Apply ``patch`` to the remote modeling module defining ``class_names``, now or when it loads.

    Idempotent per ``class_names``: re-applying Liger (model load, then TRL's re-application) must
    not arm the same patch twice. Already-loaded modules are patched immediately — a second run in
    the same process finds the module in ``sys.modules``.
    """
    armed_names = {names for names, _ in _ARMED}
    if class_names not in armed_names:
        _ARMED.append((class_names, patch))
    register_remote_class_hook(_fire)
    for module in list(sys.modules.values()):
        if (
            isinstance(module, ModuleType)
            and module.__name__.startswith("transformers_modules.")
            and _defines_all(module, class_names)
        ):
            patch(module)
