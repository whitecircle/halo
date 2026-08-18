"""Model-side toolkit: module-tree introspection, MoE balancing, load-time patches, the
sharding-agnostic loaders, and the task heads transformers does not ship.

Re-exports nothing; import the owning module. The parallelism-aware loaders live in
:mod:`src.distributed.loading`, above this package.
"""
