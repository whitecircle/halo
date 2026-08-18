"""Subclass-tree derivation for the toolkit's self-registering registries.

:func:`iter_subclasses` is the single implementation of that walk. Its widest consumer is
:func:`build_class_claim_map`, which resolves a key each class declares back to the class claiming
it. Three registries read their map through :func:`build_hf_module_name_map` — the EP patcher (HF
MoE block → ``EP*MoELayer``), the CP patcher (HF attention → ``*UlyssesAttention``) and the PP
splitter (HF backbone → ``PPModelSpec``, a spec rather than a wrapper) — and the EP
``config.model_type`` roster through the same builder on its own attribute. Registries whose value
is not a claim map walk the tree directly (the rollout weight-sync clients, keyed by
``BACKEND_KEY``). Either way a new family or backend self-registers by subclassing, on import.
"""

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

    Diamond-safe (a class reachable by two inheritance paths is yielded once), which a plain
    recursive ``__subclasses__()`` walk is not. The one implementation of that traversal: every
    toolkit registry that self-registers by subclassing reads its roster through here.
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


def build_class_claim_map(base_cls: type[_T], attr: str, kind: str) -> dict[str, type[_T]]:
    """Claimed key → claiming class, over ``base_cls``'s whole subclass tree.

    Walks the tree (concrete classes may sit under intermediate bases) reading each class's OWN
    ``attr`` tuple — an inherited tuple is not re-claimed by the subclass, which would read as a
    duplicate claim. One class may fan in several keys. The one implementation behind every
    "which class claims this name" registry, whatever the key is (HF module class name,
    ``config.model_type``).

    Args:
        base_cls: the base whose subclass tree is scanned.
        attr: per-class tuple of claimed keys (e.g. ``"HF_MODULE_NAMES"``, ``"HF_MODEL_TYPES"``).
        kind: noun for the error message (e.g. ``"HF MoE class name"``, ``"model_type"``).

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


def build_hf_module_name_map(base_cls: type[_T], kind: str) -> dict[str, type[_T]]:
    """HF module class name → claiming class, from each claimant's own ``HF_MODULE_NAMES``.

    The patcher-facing spelling of :func:`build_class_claim_map`: ``kind`` is the wrapped module's
    noun (e.g. ``"MoE"``, ``"attention"``, ``"backbone"``). The claimant is a wrapper for EP/CP and
    a split spec for PP.
    """
    return build_class_claim_map(base_cls, "HF_MODULE_NAMES", f"HF {kind} class name")


def swap_registered_modules(
    model: nn.Module,
    class_map: dict[str, type],
    build: Callable[[str, nn.Module, type], nn.Module],
    *,
    descend_into_match: bool,
) -> list[tuple[str, nn.Module]]:
    """Replace every module whose class name a registry claims, and report what was replaced.

    The one tree-walk-and-swap behind the EP MoE patcher and the CP attention patcher: walk the
    module tree, and where ``class_map`` claims a child's class name, put ``build(path, child, cls)``
    in its place. Returns ``(dotted path, replacement)`` per swap, in walk order, so each patcher owns
    only its construction, its zero-hit raise and its logging.

    ``descend_into_match`` decides whether a matched child's own subtree is walked afterwards. It is
    load-bearing for EP: Gemma 4 registers the expert tensors on the mapped class ITSELF, so
    descending would re-match inside the block that was just wrapped. Descent visits the ORIGINAL
    child, not the replacement — a wrapper holding the module it replaced would otherwise match its
    own claim forever.
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

    Refuses a subpackage: importing one runs only its docstring init, so a family shipped as a
    package would register nothing and no error would say so.
    """
    for module in pkgutil.iter_modules(package.__path__):
        if module.name in skip:
            continue
        if module.ispkg:
            raise RuntimeError(f"{package.__name__}.{module.name} is a package; families are single modules")
        importlib.import_module(f"{package.__name__}.{module.name}")
