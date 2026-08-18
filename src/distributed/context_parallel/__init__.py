"""Context Parallelism (CP) with Ulysses sequence parallelism.

Each GPU holds a sequence chunk; an all-to-all redistributes Q/K/V so every rank
sees the full sequence with a partial set of attention heads (true cross-sequence
attention, unlike simple chunking).

Supported model families: GptOss MoE, Qwen3 / Qwen3 MoE / Qwen3-VL, Qwen3.5/3.6 MoE,
Bailing MoE / Ling 2.0, Cohere2 MoE, GLM4 MoE Lite, and Mistral4 — the last two use the
legacy path (mismatched Q/K vs V head dims).

Deliberately re-exports nothing: :mod:`.loading` pulls the EP loader, so an eager ``__init__``
would close an EP↔CP cycle (EP's :mod:`~src.distributed.expert_parallel.saving` needs CP's state-dict key
mapping). Import from the owning module (``...context_parallel.config``, ``.wrapper``, ``.loading``,
``.validation``, …).
"""
