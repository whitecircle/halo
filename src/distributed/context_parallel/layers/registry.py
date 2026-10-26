"""HF attention class name → Ulysses wrapper class, derived from the wrapper subclass tree. Every
family module is imported first, so a new family registers on import."""

import src.distributed.context_parallel.layers as _layers
from src.distributed.context_parallel.base_layer import UlyssesAttentionBase
from src.subclass_registry import build_class_claim_map, import_package_modules

import_package_modules(_layers, skip=frozenset({"registry"}))


def build_wrapper_class_map() -> dict[str, type[UlyssesAttentionBase]]:
    """HF attention class name → Ulysses wrapper class, from each wrapper's ``HF_MODULE_NAMES``.

    Walks the ``UlyssesAttentionBase`` subclass tree (concrete wrappers may sit under intermediate
    bases). One wrapper may fan in several HF classes. Raises on a duplicate HF name.
    """
    return build_class_claim_map(UlyssesAttentionBase, "HF attention class name")


WRAPPER_CLASS_MAP = build_wrapper_class_map()
