"""Expert Parallelism (EP) machinery: configuration, dispatch, layer wrappers, save/load.

Re-exports nothing: an eager ``__init__`` would pull the DeepEP dispatcher, every family layer wrapper
and the save/load paths into any ``src.distributed`` import, and would close an EP↔CP import cycle.
Import from the owning module (``...expert_parallel.base_layer``, ``.config``, ``.expert_weights``,
``.layers``, ``.saving``, …).
"""
