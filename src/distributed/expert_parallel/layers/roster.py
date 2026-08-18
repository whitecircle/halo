"""Imports every family layer module so the module registry holds the whole roster; a new family
registers by existing."""

import src.distributed.expert_parallel.layers as _layers
from src.distributed.module_registry import import_package_modules

import_package_modules(_layers, skip=frozenset({"roster"}))
