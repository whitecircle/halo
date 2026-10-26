"""Model-side toolkit: module-tree introspection, composite-config access, MoE balancing, load-time
patches, and the task heads transformers does not ship.

Re-exports nothing; import the owning module. Model construction lives in :mod:`src.model_loading`
and the parallelism-aware loaders in :mod:`src.distributed.loading`, both above this package.
"""
