# Ministral 3 (dense `mistral3` composites)

`mistralai/Ministral-3-{3B,8B,14B}-Instruct-2512-BF16` are dense vision-language models: a `mistral3` composite holding a Pixtral vision tower, a projector and a `ministral3` text tower (GQA, SwiGLU, YaRN RoPE with the llama-4 logarithmic query scale, `llama_4_scaling_beta` 0.1 from 16,384 positions). The 8B release holds 8.92B parameters, 8.49B of them in the language model and head. The 14B release holds 13.27B and ties its embedding and head.

Per-axis support: the Ministral 3 row of the [compatibility matrix](README.md#compatibility-matrix). The wrapper facts these releases share with Mistral Small 4 (the `AutoModelForImageTextToText` class, the text tower at `model.language_model`, the Liger dispatch) are on [Mistral4 → Model loading](mistral4.md#model-loading).

## Loading

The releases load whole as `Mistral3ForConditionalGeneration` and train on the generic dense path under FSDP2. `text_only_model: true` is refused for `mistral3`, so every recipe loads the vision tower too. Text rows never reach it: its parameters keep `grad is None`, and a full fine-tune saves the tower and projector as they shipped ([Data Parallelism → Shard groups](../parallelism/data-parallelism.md#shard-groups)).

`MISTRAL4` claims the `mistral3` spelling, so the MoE family spec resolves for these dense checkpoints too. SFT, preference and offline GRPO check `has_routed_experts` and train them as dense; online GRPO weight sync is refused (issue #225).

## Recipe settings

- **LoRA targets the language model by regex.** A name list (`q_proj`, …, `down_proj`) also matches Pixtral's attention and feed-forward, 168 more modules on both releases, and `all-linear` reaches the tower and the projector too. The LoRA recipe matches `.*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))`.
- **No packing.** `ministral3` is not in the measured packing table, and the llama-4 query scale reads each token's row position, so packed documents stay off until a flash isolation row passes on GPU (issue #225).
- **`use_liger_kernel: false`.** Liger has no applier for this class, and a run with it on only warns.
- **Completion marker `[/INST]`.** It immediately precedes the assistant content, and the span ends at `</s>`. The hub template adds its default system prompt to every conversation that has none, UltraChat's and Tulu-3's included, and that prompt states the day's date, so the same row renders differently on another day. The template also refuses an empty assistant turn, so the preference recipes read a copy of Tulu-3 without the 940 of its 272,898 rows whose chosen or rejected reply is empty or blank; their headers give the one-line filter.

## Configs

| Config | Method | Base model |
|---|---|---|
| `examples/sft/ministral3/ministral3-8b-ultrachat-full.yaml` | SFT (full) | `mistralai/Ministral-3-8B-Instruct-2512-BF16` |
| `examples/sft/ministral3/ministral3-14b-ultrachat-lora.yaml` | SFT (LoRA) | `mistralai/Ministral-3-14B-Instruct-2512-BF16` |
| `examples/preference/ministral3/smpo-ministral3-8b-tulu3-prefmix.yaml` | SMPO | `mistralai/Ministral-3-8B-Instruct-2512-BF16` |
| `examples/preference/ministral3/dpo-ministral3-8b-tulu3-prefmix.yaml` | DPO | `mistralai/Ministral-3-8B-Instruct-2512-BF16` |
| `examples/grpo/offline/ministral3/offline-grpo-ministral3-8b-gsm8k.yaml` | Offline GRPO | `mistralai/Ministral-3-8B-Instruct-2512-BF16` |

## Status

On 8× H200 (`hopper-1.3.0rc2`), each recipe ran 20 steps under FSDP2. SFT resolves to FA3, and the preference and offline GRPO scripts to SDPA, where cuDNN SDPA is off on this cuDNN and no step was skipped. The SFT runs read the first 4,000 UltraChat rows, and SMPO and DPO the first 2,048 Tulu-3 rows less the 7 with an empty reply. Offline GRPO read 1,400 GSM8K prompts with four completions each, sampled from the 8B release on vLLM and scored with the `accuracy` term. Peak allocated and reserved are rank 0's (`enable_efficiency_metrics`), and the last column is what `nvidia-smi` showed in use across the eight GPUs:

| Recipe | Attention | Loss, first → last | tok/s per rank | Peak allocated | Reserved | In use |
|---|---|---|---|---|---|---|
| SFT, full, 8B | FA3 | 1.385 → 0.964 | 4,172 | 48.9 GiB | 57.1 GiB | 58–62 GiB |
| SFT, LoRA, 14B | FA3 | 1.313 → 0.953 | 2,808 | 56.9 GiB | 64.6 GiB | 66–70 GiB |
| SMPO, 8B | SDPA | 2.041 → 1.404 | 1,481 | 32.0 GiB | 35.4 GiB | 36–38 GiB |
| DPO, 8B | SDPA | 0.690 → 0.491 | 1,855 | 49.2 GiB | 52.6 GiB | 54–55 GiB |
| Offline GRPO, 8B | SDPA | 0.022 → 0.025 | 1,343 | 28.5 GiB | 32.3 GiB | 33–35 GiB |

Every grad norm was finite, and offline GRPO's stayed above zero (0.032 to 0.283). After the full fine-tune, all 222 vision-tower and projector tensors of the save equal the release's bit for bit. The save loads in stock transformers as `Mistral3ForConditionalGeneration` with no missing or unexpected key and generates, at an NLL of 1.555 on 32 UltraChat test rows where the release reads 1.959. The LoRA adapter holds 560 keys over the seven projections of all 40 language-model layers and none in the tower or the projector. The 8B release answers most GSM8K prompts the same way four times, so 1,304 of the 1,400 groups tie and carry no advantage.
