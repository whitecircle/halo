"""Parallelism-aware model construction: the loader dispatcher, VLM entry points and PEFT setup.

``load_distributed_model`` picks the EP / CP / TP / PP loader for a :class:`ParallelismConfig`, so
this package sits ABOVE every parallelism implementation and reaches into all of them; none of them
imports it back. It lives here rather than in :mod:`src.model_loading` for exactly that reason: that
package holds only the sharding-agnostic construction.

Deliberately re-exports nothing: importing the dispatcher pulls the DeepEP dispatcher, every family
layer wrapper and the CP/PP loaders, which a job that merely needs ``setup_peft_model`` must not pay.
Import from the owning module (``.model_loading``, ``.peft_setup``, ``.vlm_setup``).
"""
