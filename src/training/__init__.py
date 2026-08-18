"""Entry-script plumbing: environment setup, run logging, and the training script runner.

Submodules are imported directly; this package exports no names of its own, so importing one module
does not pull in the model-loading stack ``environment`` and ``script_runner`` reach into.
"""
