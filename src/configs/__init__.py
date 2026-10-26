"""Dataclasses a run is configured with, one module per method family: its script arguments and, where the toolkit
defines one, its trainer's ``TrainingArguments`` subclass; plus the arguments every script shares
(``script_arguments``), the field bundles every GRPO method shares (``grpo``) and the plain configs the async GRPO
trainer holds (async rollout in ``grpo_environmental``, environment). The GRPO script arguments and the parallelism
flags live in :mod:`src.args`."""
