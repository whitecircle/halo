"""The parallelism strategies and their configuration: ``ParallelismConfig``, FSDP2 and the EP/CP/TP/PP
subpackages; ranks, process groups, collectives and barriers live in ``src.runtime``. Nothing is re-exported,
so importing one module does not pull in the DeepEP dispatcher, the family layer wrappers or the CP save path.
"""
