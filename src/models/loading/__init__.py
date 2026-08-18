"""Sharding-agnostic model construction: lazy safetensors I/O, checkpoint coverage, preparation.

The parallelism-aware loaders live in :mod:`src.distributed.loading` — they dispatch on a
``ParallelismConfig`` and reach into every EP/CP/TP/PP implementation, which this package must not.
"""
