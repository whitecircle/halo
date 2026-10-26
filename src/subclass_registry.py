"""Registries read off a base class's subclass tree: a class registers by subclassing and claiming keys."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from types import ModuleType
from typing import TypeVar

import torch.nn as nn

_T = TypeVar("_T")


def iter_subclasses(base_cls: type[_T]) -> list[type[_T]]:
    """Every descendant of ``base_cls``, each exactly once.

    Diamond-safe: a class reachable by two inheritance paths is yielded once, unlike a plain
    recursive ``__subclasses__()`` walk.
    """
    found: list[type[_T]] = []
    stack = list(base_cls.__subclasses__())
    seen: set[type] = set()
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        found.append(cls)
        stack.extend(cls.__subclasses__())
    return found


def build_class_claim_map(base_cls: type[_T], kind: str, *, attr: str = "HF_MODULE_NAMES") -> dict[str, type[_T]]:
    """Claimed key → claiming class, over ``base_cls``'s whole subclass tree.

    Walks the tree (concrete classes may sit under intermediate bases) reading each class's own
    ``attr`` tuple; an inherited tuple is not re-claimed by the subclass, which would read as a
    duplicate claim. One class may claim several keys.

    Args:
        base_cls: the base whose subclass tree is scanned.
        kind: noun for the error message (e.g. ``"HF MoE class name"``, ``"model_type"``).
        attr: name of the class attribute holding the claimed keys; ``HF_MODULE_NAMES`` (the HF module
            classes a claimant takes over) by default.

    Raises:
        ValueError: two classes claim the same key — the resolution would be order-dependent.
    """
    mapping: dict[str, type[_T]] = {}
    for cls in iter_subclasses(base_cls):
        for key in vars(cls).get(attr, ()):
            existing = mapping.setdefault(key, cls)
            if existing is not cls:
                raise ValueError(f"{kind} '{key}' claimed by both {existing.__name__} and {cls.__name__}")
    return mapping


def swap_registered_modules(
    model: nn.Module,
    class_map: dict[str, type],
    build: Callable[[str, nn.Module, type], nn.Module],
    *,
    descend_into_match: bool,
) -> list[tuple[str, nn.Module]]:
    """Replace every module whose class name a registry claims, and report what was replaced.

    Shared by the EP MoE patcher and the CP attention patcher: walk the module tree, and where
    ``class_map`` claims a child's class name, put ``build(path, child, cls)`` in its place. Returns
    ``(dotted path, replacement)`` per swap, in walk order.

    ``descend_into_match`` decides whether a matched child's own subtree is walked afterwards. EP
    needs it off: Gemma 4 registers the expert tensors on the mapped class itself, so descending would
    re-match inside the block just wrapped. Descent visits the original child, not the replacement, so
    a wrapper holding the module it replaced does not match its own claim.
    """
    swapped: list[tuple[str, nn.Module]] = []

    def walk(module: nn.Module, path: str) -> None:
        for name, child in list(module.named_children()):
            full_path = f"{path}.{name}" if path else name
            claimant = class_map.get(type(child).__name__)
            if claimant is None:
                walk(child, full_path)
                continue
            setattr(module, name, build(full_path, child, claimant))
            swapped.append((full_path, getattr(module, name)))
            if descend_into_match:
                walk(child, full_path)

    walk(model, "")
    return swapped


def import_package_modules(package: ModuleType, *, skip: frozenset[str] = frozenset()) -> None:
    """Import every module directly under ``package`` so its self-registering classes exist.

    Rejects a subpackage: importing one runs only its ``__init__``, so a family shipped as a package
    would register nothing.
    """
    for module in pkgutil.iter_modules(package.__path__):
        if module.name in skip:
            continue
        if module.ispkg:
            raise RuntimeError(f"{package.__name__}.{module.name} is a package; families are single modules")
        importlib.import_module(f"{package.__name__}.{module.name}")
