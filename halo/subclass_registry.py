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
    """Every descendant of ``base_cls``, each once even when two inheritance paths reach it."""
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
    """Claimed key → claiming class over the subclass tree, reading each class's own ``attr`` (an inherited one is no
    second claim). Two classes claiming one key raises, since resolution would be order-dependent."""
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
    """Swap ``build(path, child, cls)`` in for every module ``class_map`` claims; return ``(path, replacement)`` per
    swap. ``descend_into_match`` walks the original child's subtree, so a wrapper does not match its own claim."""
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


def import_package_modules(package: ModuleType) -> None:
    """Import every module directly under ``package`` so its self-registering classes exist; a
    subpackage raises, since importing it would register nothing."""
    for module in pkgutil.iter_modules(package.__path__):
        if module.ispkg:
            raise RuntimeError(f"{package.__name__}.{module.name} is a package; families are single modules")
        importlib.import_module(f"{package.__name__}.{module.name}")
