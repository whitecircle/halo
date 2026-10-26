"""Rosters the tests derive from the tree rather than list by hand."""

import importlib
import pkgutil

import halo.trainers
from halo.distributed.expert_parallel.families.roster import ep_layer_classes
from halo.subclass_registry import iter_subclasses
from halo.trainers.distributed_trainer import DistributedTrainerMixin


def import_all_trainers() -> None:
    """Import every module under ``halo.trainers`` so the ``DistributedTrainerMixin`` subclass tree is complete."""
    for module in pkgutil.walk_packages(halo.trainers.__path__, "halo.trainers."):
        importlib.import_module(module.name)


def distributed_trainer_classes() -> list[type]:
    """Every ``DistributedTrainerMixin`` subclass defined under ``halo.trainers``, sorted by name, after
    importing every trainer module.

    Filtered by module: ``__subclasses__()`` is process-global, and a stub trainer another test module
    defines (pytest imports them all before running any) would otherwise join the roster.
    """
    import_all_trainers()
    return sorted(
        (cls for cls in iter_subclasses(DistributedTrainerMixin) if cls.__module__.startswith("halo.trainers")),
        key=lambda cls: cls.__name__,
    )


def ep_layer_class_by_model_type() -> dict[str, type]:
    """``config.model_type`` -> the EP layer class whose own ``FAMILY`` claims it, in subclass-walk order."""
    return {
        model_type: cls
        for cls in ep_layer_classes()
        if (family := vars(cls).get("FAMILY"))
        for model_type in family.model_types
    }
