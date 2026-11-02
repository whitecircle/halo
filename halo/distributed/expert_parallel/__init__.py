"""Expert Parallelism: the EP config, the DeepEP transport, the EP MoE layer, its per-family subclasses and their
installation. Re-exports nothing, so importing one ``halo.distributed`` symbol does not load the EP tree: import
from the owning module (``.moe_layer.layer``, ``.config``, ``.families``; the EP saver is
:mod:`halo.checkpoint.parallel.ep`)."""
