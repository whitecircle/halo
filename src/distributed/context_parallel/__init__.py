"""Context Parallelism (CP) with Ulysses sequence parallelism.

Each GPU holds a sequence chunk; an all-to-all redistributes Q/K/V so every rank sees the full
sequence with a subset of the attention heads.

Supported families: GptOss MoE, Qwen3 / Qwen3 MoE / Qwen3-VL, Qwen3.5/3.6 MoE, Bailing MoE /
Ling 2.0, Cohere2 MoE, GLM4 MoE Lite and Mistral4; the last two run the legacy path, where the Q/K
and V head dims differ.

Re-exports nothing: :mod:`.loading` imports the EP loader, and an eager ``__init__`` would close an
EP↔CP import cycle. Import from ``.config``, ``.wrapper``, ``.loading``, ``.validation``.
"""
