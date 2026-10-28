"""Expert Parallelism: the EP config, the DeepEP transport, the EP MoE layer, its per-family subclasses and their
installation.

Deliberately re-exports nothing. The subpackage spans the DeepEP dispatcher and every family's EP
layer, so an eager ``__init__`` would make importing *any* ``halo.distributed`` symbol — a dense/DDP
job reading :class:`~halo.distributed.parallelism_config.ParallelismConfig`, a callback reading a
class-declared contract — pull the whole tree. Import from the owning module
(``...expert_parallel.moe_layer.layer``, ``.config``, ``.families``, …);
the EP saver is :mod:`halo.checkpoint.parallel.ep`.
"""
