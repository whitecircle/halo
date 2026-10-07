"""Expert Parallelism (EP) machinery: configuration, dispatch, layer wrappers, lazy loading, FP32 masters.

Deliberately re-exports nothing. The subpackage spans the DeepEP dispatcher and every family layer
wrapper, so an eager ``__init__`` would make importing *any* ``src.distributed`` symbol — a dense/DDP
job reading :class:`~src.distributed.parallelism_config.ParallelismConfig`, a callback reading a
class-declared contract — pull the whole tree. Import from the owning module
(``...expert_parallel.base_layer``, ``.config``, ``.expert_weights``, ``.layers``, ``.lazy_loader``, …);
the EP saver is :mod:`src.distributed.checkpoint.ep_save`.
"""
