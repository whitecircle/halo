"""Expert Parallelism (EP) machinery: configuration, dispatch, layer wrappers, save/load.

Deliberately re-exports nothing. The subpackage spans the DeepEP dispatcher, every family layer
wrapper and the save/load paths (which reach into :mod:`src.distributed.context_parallel`), so an
eager ``__init__`` would make importing *any* ``src.distributed`` symbol — a dense/DDP job reading
:class:`~src.distributed.parallelism_config.ParallelismConfig`, a callback reading a class-declared
contract — pull the whole tree, and would close an EP↔CP import cycle. Import from the owning module
(``...expert_parallel.base_layer``, ``.config``, ``.expert_weights``, ``.layers``, ``.saving``, …).
"""
