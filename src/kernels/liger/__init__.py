"""Liger Kernel integration. Coverage is declarative: a family states which of its classes fill each
patchable role in ``families``, ``builder`` turns that into an applier, and ``orchestrator`` resolves one
per ``model_type``. Model types upstream Liger already covers resolve to its applier instead.
"""
