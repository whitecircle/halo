"""Imports every family layer module so every EP MoE family is on the subclass tree the EP registries walk."""

import src.distributed.expert_parallel.layers as _layers
from src.subclass_registry import import_package_modules

import_package_modules(_layers, skip=frozenset({"roster"}))
