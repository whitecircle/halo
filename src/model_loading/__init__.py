"""Sharding-agnostic model construction: the load call, run dtype, processing class, coverage gate
and the lazy safetensors core.

The parallelism-aware loaders live in :mod:`src.distributed.loading`: they dispatch on a
``ParallelismConfig`` and reach into every EP/CP/TP/PP implementation, which this package must not.
"""
