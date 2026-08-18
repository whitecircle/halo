"""Rosters the tests derive from the tree rather than list by hand."""

import importlib
import pkgutil

import src.trainers


def import_all_trainers() -> None:
    """Import every module under ``src.trainers`` so the ``DistributedTrainerMixin`` subclass tree is complete."""
    for module in pkgutil.walk_packages(src.trainers.__path__, "src.trainers."):
        importlib.import_module(module.name)
