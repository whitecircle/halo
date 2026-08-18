"""Batch-time collators (``collate_fn``) and the factory that selects one for a run.

Nothing is re-exported here; import each collator from its own module, so a pure-config import does
not pull in trl/peft for one constant."""
