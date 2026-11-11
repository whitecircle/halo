# Supported models

Any HuggingFace `AutoModelForCausalLM` trains on Halo's data-parallel path with
no extra work: point `model_name_or_path` at it and launch.

The families below also have expert, context, tensor or pipeline-parallel
support ([which, per family](supported-matrix.md#model-families)), a tuned
recipe and GPU-validated coverage, except where a row says otherwise. Eighteen
of them are MoE.

| Family | Hub example | Kind | What to know |
| --- | --- | --- | --- |
| [Qwen3 MoE](cookbooks/halo-qwen3-moe-cookbook.md) | `Qwen/Qwen3-30B-A3B-Instruct-2507` | MoE | the reference MoE |
| [Qwen3.5 / Qwen3.6 MoE](cookbooks/halo-qwen3.5-qwen3.6-moe-cookbook.md) | `Qwen/Qwen3.5-35B-A3B` | MoE | hybrid linear attention; [`Qwen/Qwen3.8-2.4T-A95B`](../agent-docs/models/qwen3_5.md#qwen38-24t-a95b) ↗ takes the same path, with no recipe and no GPU run |
| [GPT-OSS](cookbooks/halo-gpt-oss-cookbook.md) | `unsloth/gpt-oss-20b-BF16` | MoE | attention sinks; FA2 refused for on-policy RL |
| [GLM-4 MoE Lite](cookbooks/halo-glm-4.7-flash-cookbook.md) | `zai-org/GLM-4.7-Flash` | MoE | compressed (MLA) attention |
| [Gemma 4 MoE](cookbooks/halo-gemma4-moe-cookbook.md) | `google/gemma-4-26B-A4B-it` | MoE | no flash kernel (SDPA, FlexAttention on sliding layers); no router balancing |
| [Mistral 4 MoE](cookbooks/halo-mistral4-moe-cookbook.md) | `mistralai/Mistral-Small-4-119B-2603` | MoE | convert the fp8 release to bf16 first |
| [Laguna 2.1](cookbooks/halo-laguna-2.1-cookbook.md) | `poolside/Laguna-S-2.1` | MoE | SDPA through `trust_remote_code: true`, and a flash pin is refused |
| [LFM-2 MoE](cookbooks/halo-lfm2-moe-cookbook.md) | `LiquidAI/LFM2-24B-A2B` | MoE | hybrid short-convolution layers |
| [ZAYA1](cookbooks/halo-zaya1-cookbook.md) | `Zyphra/ZAYA1-8B` | MoE | trains without gradient checkpointing, always |
| [Command A+](cookbooks/halo-command-a-plus-cookbook.md) | `CohereLabs/command-a-plus-05-2026-bf16` | MoE | 200B+; pinned revision |
| [Nemotron 3.5 Super VL](cookbooks/halo-nemotron-3.5-super-cookbook.md) | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` | MoE | Mamba-2 layers; EP8 on B300; short runs on the release; no online RL; the text-only [`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](../agent-docs/models/nemotron-h.md#lightning) ↗ has its own recipe, run for 5 steps at EP8 |
| [Bailing / Ling](../agent-docs/models/bailing.md) ↗ | `inclusionAI/Ling-mini-2.0` | MoE | needs `trust_remote_code` |
| [Inkling](../agent-docs/models/inkling.md) ↗ | `thinkingmachines/Inkling-Small` | MoE | multimodal; SDPA; the tokenizer ships no pad token |
| [DeepSeek-V4](../agent-docs/models/deepseek-v4.md) ↗ | `deepseek-ai/DeepSeek-V4-Flash` | MoE | eager attention only; one document per row (`packing` and `padding_free` refused); convert to bf16 first |
| [GLM-5 Next](../agent-docs/models/glm5-next.md) ↗ | `zai-org/GLM-5.3-Flash` | MoE | convert fp8 to bf16 first; SDPA only; LoRA fits one 8-GPU machine, a full fine-tune takes two |
| [Step-3.7 Flash](../agent-docs/models/step3p7.md) ↗ | `stepfun-ai/Step-3.7-Flash` | MoE | composite VLM; SDPA only |
| [Qwen3-Next](../agent-docs/models/qwen3-next.md) ↗ | `Qwen/Qwen3-Next-80B-A3B-Instruct` | MoE | hybrid linear attention; tiny-model GPU tests only |
| [Qwen3.8-Flash-Next](../agent-docs/models/qwen4_exp.md) ↗ | `Qwen/Qwen3.8-Flash-Next` | MoE | experimental: text tower under EP, ETP and PP; 95 GiB n-gram table in host memory; online and async GRPO on vLLM, and on SGLang from the toolkit's image; offline GRPO CPU-checked only |
| [Qwen3.8-27B](../agent-docs/models/qwen3_5.md#qwen38-27b) ↗ | `Qwen/Qwen3.8-27B` | dense | the Qwen3.5 dense path; no recipe and no GPU run |
| [Qwen3 dense](../agent-docs/models/qwen3.md) ↗ | `Qwen/Qwen3-4B-Instruct-2507` | dense | the reference dense family |
| Any other HF causal LM | Llama, Mistral, Phi, … | dense | FSDP2 by default; TP needs a `tp_plan` |

The last column names each family's main constraint. The modes each
family supports are in the [Supported matrix](supported-matrix.md), which wins
when a recipe disagrees.

## Cookbooks

Eleven families have a [cookbook](cookbooks/README.md): a worked path from
`docker pull` to a served checkpoint, with the exact `halo launch` lines and the
LoRA and RL variants the family supports. If your family has one, start there.

## Multimodal checkpoints

Gemma 4, Mistral 4, Command A+, Inkling, GLM-5 Next, Step-3.7 Flash, Nemotron 3.5
Super VL, Qwen3.5/3.6, Qwen3.8-Flash-Next and Qwen3-VL are natively multimodal.
Halo loads the full vision-and-text model and picks the data path from your
dataset, not the checkpoint:

- Rows with images train through the VLM pipeline, with standard padding.
  Images can't be packed.
- A text-only dataset on the same checkpoint trains through the text pipeline,
  with packing available.

To load a multimodal checkpoint as text-only, set `text_only_model: true`, as the
Qwen3.8-Flash-Next and Nemotron 3.5 Super VL recipes do. Qwen3.5/3.6,
Qwen3.8-Flash-Next, Nemotron 3.5 Super VL and Llama 4 have a text-only variant;
the others refuse it.

Image data works for SFT, DPO, KTO, SMPO and both distillation trainers. PaliGemma
takes it through SFT and distillation only, and Nemotron-H refuses it on DPO and
KTO. Reward modeling takes images only on checkpoints with a multimodal
sequence-classification head, such as Gemma 4 and Qwen3.5/3.6. Classification, the
GRPO trainers and embedding are text-only.

## Adding a family

A family that trains under plain FSDP2 needs no code. Context parallelism takes
one wrapper class that registers itself; expert parallelism takes one EP layer
file, the family's spec and a roster import.
[Model integration cost](model-integration-cost.md) shows what an EP layer costs,
and [Adding a Model](../agent-docs/models/adding-a-model.md) ↗ is the procedure.
