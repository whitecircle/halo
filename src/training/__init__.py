"""Entry-script plumbing: environment setup, run logging, and the training script runner.

Re-exports nothing: a package-level facade would drag the model-loading stack ``environment`` and
``script_runner`` reach into every consumer of any module here.
"""
