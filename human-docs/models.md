# Supported Models

Any HuggingFace `AutoModelForCausalLM` trains under Halo's data-parallel path
with no work — you point `model_name_or_path` at it and launch. The families
below are the ones that additionally carry expert/context/tensor-parallel
wrappers, a tuned recipe, and GPU-validated coverage. Fifteen of them are MoE.

| Family | Hub example | Kind | What to know |
| --- | --- | --- | --- |
| [Qwen3 MoE](cookbooks/halo-qwen3-moe-cookbook.md) | `Qwen/Qwen3-30B-A3B-Instruct-2507` | MoE | broadest mode coverage; reference MoE |
| [Qwen3.5 / Qwen3.6 MoE](cookbooks/halo-qwen3.5-qwen3.6-moe-cookbook.md) | `Qwen/Qwen3.5-35B-A3B` | MoE | hybrid linear attention rules out CP |
| [GPT-OSS](cookbooks/halo-gpt-oss-cookbook.md) | `unsloth/gpt-oss-20b-BF16` | MoE | attention sinks; FA2 refused on-policy |
| [GLM-4 MoE Lite](cookbooks/halo-glm-4.7-flash-cookbook.md) | `zai-org/GLM-4.7-Flash` | MoE | compressed attention; every mode works |
| [Gemma 4 MoE](cookbooks/halo-gemma4-moe-cookbook.md) | `google/gemma-4-26B-A4B-it` | MoE | SDPA only; no router-balancing route |
| [Mistral 4 MoE](cookbooks/halo-mistral4-moe-cookbook.md) | `mistralai/Mistral-Small-4-119B-2603` | MoE | convert the fp8 release to bf16 |
| [Laguna 2.1](cookbooks/halo-laguna-2.1-cookbook.md) | `poolside/Laguna-S-2.1` | MoE | remote code at a pinned revision |
| [LFM-2 MoE](cookbooks/halo-lfm2-moe-cookbook.md) | `LiquidAI/LFM2-24B-A2B` | MoE | short-convolution layers rule out CP |
| [ZAYA1](cookbooks/halo-zaya1-cookbook.md) | `Zyphra/ZAYA1-8B` | MoE | gradient checkpointing never works |
| [Command A+](cookbooks/halo-command-a-plus-cookbook.md) | `CohereLabs/command-a-plus-05-2026-bf16` | MoE | 200B+; pinned revision and FA2 |
| [Bailing / Ling](../agent-docs/models/bailing.md) ↗ | `inclusionAI/Ling-mini-2.0` | MoE | needs `trust_remote_code` and `sdpa` |
| [Inkling](../agent-docs/models/inkling.md) ↗ | `thinkingmachines/Inkling-Small` | MoE | multimodal; pin `sdpa`, no pad token |
| [DeepSeek-V4](../agent-docs/models/deepseek-v4.md) ↗ | `deepseek-ai/DeepSeek-V4-Flash` | MoE | eager attention only; convert to bf16 |
| [GLM-5 Next](../agent-docs/models/glm5-next.md) ↗ | `zai-org/GLM-5.3-Flash` | MoE | convert fp8 to bf16; SDPA only |
| [Step-3.7 Flash](../agent-docs/models/step3p7.md) ↗ | `stepfun-ai/Step-3.7-Flash` | MoE | per-layer head counts rule out TP |
| [Qwen3 dense](../agent-docs/models/qwen3.md) ↗ | `Qwen/Qwen3-4B-Instruct-2507` | dense | the reference dense family; CP and TP |
| Any other HF causal LM | Llama, Mistral, Phi, … | dense | FSDP2 by default; TP needs a `tp_plan` |

The last column is the one surprise per family, not the whole story. Which modes a
family actually supports — EP, CP, TP, ETP and their combinations, LoRA, rollout
weight sync — is the [Supported Matrix](supported-matrix.md), which wins over any
recipe when the two disagree.

## Cookbooks

Ten families have a [cookbook](cookbooks/README.md): a worked path from
`docker pull` to a served checkpoint, with the LoRA and RL variants that family
supports and the exact `halo launch` lines. Start there if your family has one —
the shipped `examples/` config it walks through is the fastest correct start.

## Multimodal checkpoints

Several families are natively multimodal (Gemma 4, Mistral 4, Command A+,
Inkling, GLM-5 Next, Step-3.7 Flash, Qwen3.5/3.6 and Qwen3-VL). Halo loads the
full vision+text model and decides the data path from your *run*, not the
checkpoint: rows carrying images train through the VLM pipeline, and a text-only
dataset on the same checkpoint trains through the text pipeline with packing
still available. `text_only_model: true` loads such a checkpoint through its
text-only sibling when you want that explicitly.

Vision data works for SFT, DPO, KTO, SMPO, reward modeling and both distillation
trainers. Classification, the GRPO trainers and embedding are text-only. Images cannot be
packed, so the VLM path uses standard padding.

## Adding one

A family that trains under plain FSDP2 needs no code. Giving it expert or context
parallelism means one wrapper file that registers itself — usually 40 to 140
lines, and under 40 when an existing family's expert layout matches.
[Model Integration Cost](model-integration-cost.md) is the honest accounting;
[Adding a Model](../agent-docs/models/adding-a-model.md) ↗ is the procedure.
