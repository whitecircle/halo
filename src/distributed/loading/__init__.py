"""Parallelism-aware model construction: the loader dispatcher, VLM entry points and PEFT setup.

``load_distributed_model`` picks the EP / CP / TP / PP loader for a :class:`ParallelismConfig`.

Nothing is re-exported here: importing the dispatcher pulls in the DeepEP dispatcher, every family
layer wrapper and the CP/PP loaders, which a job that only needs ``setup_peft_model`` should not pay
for. Import from the defining module (``.model_loading``, ``.peft_setup``, ``.vlm_setup``).
"""
