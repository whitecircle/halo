"""Sharding-agnostic checkpoint layer: on-disk format, the exported-config contract, the saved-adapter
shape, tool-side directory I/O and the incremental safetensors shard writer. Holds no
``src.distributed`` / ``torch.distributed`` import, so a CPU-only conversion tool pays for none of
them. Re-exports nothing — import the owning module. The parallelism-aware savers live in
:mod:`src.distributed.checkpoint`, above this package.
"""
