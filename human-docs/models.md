# Supported models

Any HuggingFace `AutoModelForCausalLM` trains on Halo's data-parallel path with
no extra work: point `model_name_or_path` at it and launch.

The families below also have expert, context or tensor-parallel wrappers, a
tuned recipe and GPU-validated coverage. Fifteen of them are MoE.

| Family | Hub example | Kind | What to know |
| --- | --- | --- | --- |
| [Qwen3 MoE](cookbooks/halo-qwen3-moe-cookbook.md) | `Qwen/Qwen3-30B-A3B-Instruct-2507` | MoE | broadest mode coverage; the reference MoE |
| [Qwen3.5 / Qwen3.6 MoE](cookbooks/halo-qwen3.5-qwen3.6-moe-cookbook.md) | `Qwen/Qwen3.5-35B-A3B` | MoE | hybrid linear attention rules out CP |
| [GPT-OSS](cookbooks/halo-gpt-oss-cookbook.md) | `unsloth/gpt-oss-20b-BF16` | MoE | attention sinks; FA2 refused for on-policy RL |
| [GLM-4 MoE Lite](cookbooks/halo-glm-4.7-flash-cookbook.md) | `zai-org/GLM-4.7-Flash` | MoE | compressed attention; every mode works |
| [Gemma 4 MoE](cookbooks/halo-gemma4-moe-cookbook.md) | `google/gemma-4-26B-A4B-it` | MoE | no flash kernel (SDPA, FlexAttention on sliding layers); no router balancing |
| [Mistral 4 MoE](cookbooks/halo-mistral4-moe-cookbook.md) | `mistralai/Mistral-Small-4-119B-2603` | MoE | convert the fp8 release to bf16 first |
| [Laguna 2.1](cookbooks/halo-laguna-2.1-cookbook.md) | `poolside/Laguna-S-2.1` | MoE | set `attn_implementation: sdpa`; there is no automatic fallback |
| [LFM-2 MoE](cookbooks/halo-lfm2-moe-cookbook.md) | `LiquidAI/LFM2-24B-A2B` | MoE | short-convolution layers rule out CP |
| [ZAYA1](cookbooks/halo-zaya1-cookbook.md) | `Zyphra/ZAYA1-8B` | MoE | trains without gradient checkpointing, always |
| [Command A+](cookbooks/halo-command-a-plus-cookbook.md) | `CohereLabs/command-a-plus-05-2026-bf16` | MoE | 200B+; pinned revision and FA2 |
| [Bailing / Ling](../agent-docs/models/bailing.md) ↗ | `inclusionAI/Ling-mini-2.0` | MoE | needs `trust_remote_code` and `sdpa` |
| [Inkling](../agent-docs/models/inkling.md) ↗ | `thinkingmachines/Inkling-Small` | MoE | multimodal; pin `sdpa`; no pad token |
| [DeepSeek-V4](../agent-docs/models/deepseek-v4.md) ↗ | `deepseek-ai/DeepSeek-V4-Flash` | MoE | eager attention only; one document per row (`packing` and `padding_free` refused); convert to bf16 first |
| [GLM-5 Next](../agent-docs/models/glm5-next.md) ↗ | `zai-org/GLM-5.3-Flash` | MoE | convert fp8 to bf16 first; SDPA only |
| [Step-3.7 Flash](../agent-docs/models/step3p7.md) ↗ | `stepfun-ai/Step-3.7-Flash` | MoE | per-layer head counts rule out TP |
| [Qwen3 dense](../agent-docs/models/qwen3.md) ↗ | `Qwen/Qwen3-4B-Instruct-2507` | dense | the reference dense family; CP and TP |
| Any other HF causal LM | Llama, Mistral, Phi, … | dense | FSDP2 by default; TP needs a `tp_plan` |

The last column is the one thing most likely to surprise you, not the full
story. The modes each family supports (EP, CP, TP, ETP and their combinations,
LoRA, rollout weight sync) are in the [Supported matrix](supported-matrix.md).
When a recipe disagrees with the matrix, the matrix wins.

## Cookbooks

Ten families have a [cookbook](cookbooks/README.md): a worked path from
`docker pull` to a served checkpoint, with the exact `halo launch` lines and the
LoRA and RL variants the family supports. If your family has one, start there.

## Multimodal checkpoints

Gemma 4, Mistral 4, Command A+, Inkling, GLM-5 Next, Step-3.7 Flash,
Qwen3.5/3.6 and Qwen3-VL are natively multimodal. Halo loads the full
vision-and-text model and picks the data path from your dataset, not the
checkpoint:

- Rows with images train through the VLM pipeline, with standard padding.
  Images can't be packed.
- A text-only dataset on the same checkpoint trains through the text pipeline,
  with packing available.

To load a multimodal checkpoint as text-only, set `text_only_model: true`. GLM-5
Next and Step-3.7 Flash have no text-only variant and refuse it.

Image data works for SFT, DPO, KTO, SMPO and both distillation trainers. Reward
modeling takes images only on checkpoints with a multimodal sequence-classification
head, such as Gemma 4 and Qwen3.5/3.6. Classification, the GRPO trainers and
embedding are text-only.

## Adding a family

A family that trains under plain FSDP2 needs no code. Expert or context
parallelism takes one wrapper file that registers itself.
[Model integration cost](model-integration-cost.md) shows what a wrapper costs,
and [Adding a Model](../agent-docs/models/adding-a-model.md) ↗ is the procedure.
