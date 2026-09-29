"""Rosters the tests derive from the tree rather than list by hand."""

import importlib
import pkgutil

import src.trainers
from src.distributed.module_registry import iter_subclasses
from src.trainers.mixins.base import DistributedTrainerMixin


def import_all_trainers() -> None:
    """Import every module under ``src.trainers`` so the ``DistributedTrainerMixin`` subclass tree is complete."""
    for module in pkgutil.walk_packages(src.trainers.__path__, "src.trainers."):
        importlib.import_module(module.name)


def distributed_trainer_classes() -> list[type]:
    """Every ``DistributedTrainerMixin`` subclass defined under ``src.trainers``, sorted by name, after
    importing every trainer module.

    Filtered by module: ``__subclasses__()`` is process-global, and a stub trainer another test module
    defines (pytest imports them all before running any) would otherwise join the roster.
    """
    import_all_trainers()
    return sorted(
        (cls for cls in iter_subclasses(DistributedTrainerMixin) if cls.__module__.startswith("src.trainers")),
        key=lambda cls: cls.__name__,
    )
