"""Batch-time collators (``collate_fn``) and the factory that selects one for a run; the self-distillation
collators render at batch time through the pipeline leaves. Each collator is imported from its own module —
nothing is re-exported here, so a pure-config import does not pay trl/peft for one constant."""
