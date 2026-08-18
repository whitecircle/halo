"""Distributed runtime and parallelism: rank/world math, filesystem coordination, FSDP2, the
``ParallelismConfig`` validation gate, and the EP/CP/TP/PP implementations in their subpackages.

Nothing is re-exported here; import from the defining module (``src.distributed.parallelism_config``
for the config, ``src.distributed.runtime`` for rank math) so that a job reading one rank does not
pull in the DeepEP dispatcher, the family layer wrappers and the CP save path.
"""
