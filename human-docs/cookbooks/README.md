# Model Cookbooks

One page each for ten of the fifteen shipped MoE families, taking that family
from `docker pull` to a served checkpoint, with LoRA and RL variants where it
supports them.

These are worked examples, not the rulebook. The rules a layout has to satisfy
live in [Parallelism](../parallelism.md), and what each family supports at all
is in the [Supported Matrix](../supported-matrix.md). When a cookbook and the
matrix disagree, the matrix wins.

- [GPT-OSS](halo-gpt-oss-cookbook.md) — `gpt-oss-20b` on UltraChat at EP8, with
  CP/TP/ETP variants, LoRA over attention *and* experts, and RL that keeps the
  pretrained attention sinks live.
- [Qwen3 MoE](halo-qwen3-moe-cookbook.md) — Qwen3-30B-A3B on four GPUs at EP4,
  with CP, TP, and ETP variants on the same ranks.
- [Qwen3.5 / Qwen3.6 MoE](halo-qwen3.5-qwen3.6-moe-cookbook.md) — 35B-A3B at EP8
  with transient bias-update balancing; TP is capped at 2 and the hybrid
  linear-attention layers rule out CP.
- [GLM-4.7-Flash](halo-glm-4.7-flash-cookbook.md) — EP8 at 30,720 tokens from the
  shipped config, plus CP2/TP2/ETP8 and LoRA on GLM's compressed attention.
- [Gemma 4 MoE](halo-gemma4-moe-cookbook.md) — 26B-A4B at EP8 and 32,768 tokens,
  where attention is SDPA-only and no router balancing path exists.
- [Mistral 4 MoE](halo-mistral4-moe-cookbook.md) — Mistral Small 4 119B A6B at
  EP8 and 32,000 tokens, with CP, TP, ETP, multimodal inference, and the required
  FP8-to-BF16 checkpoint conversion.
- [Laguna 2.1](halo-laguna-2.1-cookbook.md) — Laguna S at EP4 on exactly four
  GPUs, Laguna XS on one, and the CJK pad/eos tokens that silently break if
  retyped in ASCII.
- [LFM-2 MoE](halo-lfm2-moe-cookbook.md) — LFM2.5-8B-A1B at EP2, scaled to
  LFM2-24B-A2B at EP4, plus EP+TP over the full-attention layers.
- [ZAYA1](halo-zaya1-cookbook.md) — 8B on a single GPU straight from hub `main`
  (a native transformers family), with bias-update balancing and gradient
  checkpointing off in every mode.
- [Command A+](halo-command-a-plus-cookbook.md) — Cohere2 MoE at EP8, validated at
  full scale on 8× B300; the CP/TP/ETP wrappers are GPU-verified at tiny scale but
  untested on the 200B+ checkpoint.
