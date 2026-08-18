"""Distributed runtime and parallelism: rank/world math, filesystem coordination, FSDP2, the
``ParallelismConfig`` gate, and the EP/CP/TP/PP implementations in their subpackages.

Re-exports nothing — import the owning module (``src.distributed.parallelism_config`` for the
config, ``src.distributed.runtime`` for rank math). A package-level facade would make a dense job
that reads one rank load the DeepEP dispatcher, every family layer wrapper and the CP save path.
"""
