"""Parallelism-aware model construction: the loader dispatcher, VLM entry points and PEFT setup.

``load_distributed_model`` picks the EP / CP / TP / PP loader for a :class:`ParallelismConfig`, so
this package sits ABOVE every parallelism implementation and reaches into all of them. It lives here
rather than under ``src/models/`` for exactly that reason — the utility side owns only the
sharding-agnostic leaves (``lazy_safetensors``, checkpoint coverage, model preparation).

Deliberately re-exports nothing: importing the dispatcher pulls the DeepEP dispatcher, every family
layer wrapper and the CP/PP loaders, which a job that merely needs ``setup_peft_model`` must not pay.
Import from the owning module (``.model_loading``, ``.peft_setup``, ``.vlm_setup``).
"""
