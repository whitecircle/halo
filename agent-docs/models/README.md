# Supported Models

**Any HuggingFace `AutoModelForCausalLM` trains with standard FSDP**, which loads the whole model onto every GPU first, so it must fit one ([Data Parallelism](../parallelism/data-parallelism.md#fsdp2-strategy-by-mode)). Packed rows keep documents apart on the families named in [Collators](../data/collators.md#document-isolation-under-packing). On other families a document can reach the next one (BERT- and BART-family decoders, OPT, BioGPT, GPT-Neo, Moshi), so train those without packing. The matrix below is about *advanced* parallelism — EP for MoE routing, ETP for expert FFN sharding, CP for long sequences, TP for weight sharding, PP for depth across NVLink domains. It is the one family table; `tests/cpu/parallelism/test_docs_limitation_tables.py` derives every EP/ETP/CP/TP/PP cell from the code, keyed on the row's `model_type`.

The **TP column** tracks the Halo-side path: HF-native `tp_plan="auto"` for dense models (Yes when the config ships a `base_model_tp_plan`), and *selective* TP (attention only; embeddings and `lm_head` replicated; experts left to EP) for MoE models. GPT-OSS ships a `base_model_ep_plan` and no `base_model_tp_plan`, so its TP runs through the Halo path.

The registries behind the matrix: the family specs (`MOE_FAMILY_SPECS`, `halo/models/moe/family_specs.py`) and their EP layers under `halo/distributed/expert_parallel/families/`; the Ulysses wrappers (`ULYSSES_ATTENTION_MAP`, `halo/distributed/context_parallel/families.py`) less the sequence-axis layers CP refuses (`halo/distributed/context_parallel/model_rules.py`); `TP_SHARDABLE_ATTENTION_CLASSES` (`halo/distributed/tensor_parallel/parallelize_attention.py`); `PP_FAMILY_SPECS` (`halo/distributed/pipeline_parallel/families.py`).

## Compatibility matrix

| Model Family | `model_type` | EP | ETP | CP | TP | PP | Training Configs |
|---|---|:--:|:--:|:--:|:--:|:--:|:--|
| [Qwen3 (dense)](qwen3.md) | `qwen3` | — | — | Yes | Yes | Yes ¹ | `examples/sft/qwen3/*` |
| [Qwen3 MoE](qwen3.md#qwen3-moe) | `qwen3_moe` | Yes | Yes | Yes | Yes | Yes ¹³ | — |
| [Qwen3-VL (text)](qwen3.md#qwen3-vl) | `qwen3_vl_text` | — | — | Yes ² | No | No | — |
| [Qwen3.5 / Qwen3.6 / Qwen3.8-27B (dense)](qwen3_5.md) | `qwen3_5_text` | — | — | Yes (CP2 SFT) ¹⁶ | Yes | Yes ⁴ | `examples/*/qwen3_5/*9b*` |
| [Qwen3.5 / Qwen3.6 MoE](qwen3_5.md) | `qwen3_5_moe` | Yes | Yes | No | Yes ³ | Yes ⁴ | `examples/sft/qwen3_5/*-a*b-*` |
| [Qwen3-Next](qwen3-next.md) | `qwen3_next` | Yes ¹¹ | Yes ¹¹ | No | No | No | `examples/sft/qwen3_next/*` |
| [Qwen3.8-Flash-Next](qwen4_exp.md) | `qwen4_exp` | Yes ¹² | Yes ¹² | No | No | Yes ¹⁰ ¹³ ¹⁴ | `examples/sft/qwen4_exp/*` |
| [GPT-OSS](gpt-oss.md) | `gpt_oss` | Yes | Yes | Yes | Yes | Yes | `examples/sft/gptoss/*` |
| [GLM-4 MoE Lite](glm4.md) | `glm4_moe_lite` | Yes | Yes | Yes | Yes ⁵ | Yes ¹³ | `examples/sft/glm4/*` |
| [Laguna S / XS 2.1](laguna.md) | `laguna` | Yes | Yes ⁶ | No | No | Yes ¹³ | `examples/sft/laguna/*` |
| [Inkling-Small](inkling.md) | `inkling_text` | Yes | Yes | No | No | Yes ¹⁰ ¹³ | `examples/sft/inkling/*` |
| [Gemma 4 MoE](gemma4.md) | `gemma4` | Yes | Yes | No | No | No | `examples/sft/gemma4/*` |
| [Bailing MoE / Ling](bailing.md) | `bailing_moe` | Yes | Yes | Yes ⁷ | No | No | `examples/sft/bailing/*` |
| [LFM-2 MoE](lfm2.md) | `lfm2_moe` | Yes | Yes | No | Yes | No | `examples/sft/lfm2/*` |
| [Mistral4 MoE](mistral4.md) | `mistral4` | Yes | Yes | Yes ⁸ | Yes | No | `examples/sft/mistral4/*` |
| [DeepSeek-V4](deepseek-v4.md) | `deepseek_v4` | Yes | Yes ⁶ | No | No | Yes | `examples/sft/deepseek_v4/*` |
| [Nemotron-H (Nemotron 3.5 Super VL, Lightning)](nemotron-h.md) | `nemotron_h` | Yes ¹⁵ | Yes ¹⁵ | No | No | Yes ¹⁰ ¹³ | `examples/sft/nemotron_h/*` |
| [Zaya (ZAYA1)](zaya.md) | `zaya` | Yes ⁹ | Yes ⁹ | No | No | No | `examples/sft/zaya/*` |
| [Command A+ (Cohere2 MoE)](cohere2-moe.md) | `cohere2_moe` | Yes | Yes ⁶ | Yes ⁶ | Yes ⁶ | No | `examples/sft/cohere2_moe/*` |
| [GLM-5 Next (GLM-5.3-Flash)](glm5-next.md) | `glm5_next` | Yes | Yes | No | No | Yes ¹⁰ | `examples/sft/glm5_next/*` |
| [Step-3.7 Flash](step3p7.md) | `step3p7` | Yes | Yes | No | No | Yes ¹⁰ | `examples/sft/step3p7/*` |
| Any other HF model with a `tp_plan` (Llama, Mistral, Phi, …) | — | — | — | No | Yes | No | — |

The combined modes follow the columns: EP+CP runs where EP and CP both do, EP+TP where EP and TP do, EP+ETP on every EP family; their topology rules are in [Supported combinations](../parallelism/README.md#supported-combinations). LoRA adapters train under FSDP/DP and on every axis a row runs except TP, EP+TP, PP and hybrid CP, which reject them; expert adapters are also refused at `expert_tp_size > 1` ([PEFT](../optimization/peft.md)). GPU rows train every MoE family's adapters under EP (`ep1` and `ep2`) and, where CP runs, EP+CP; pure-ETP LoRA is verified on every EP family except Command A+, GLM-5 Next, Inkling, Mistral4, Nemotron-H, Qwen3-Next and Qwen3.8-Flash-Next. LoRA ran on release checkpoints for GPT-OSS 20B, Qwen3-30B, Qwen3-0.6B, Ring-mini-linear-2.0, Gemma 4 26B and, for a few steps, Qwen3.8-Flash-Next. The other families' LoRA ran on tiny models. Why a family drops an axis is on its page; trainer support is in [Trainer Compatibility](../reference/trainer-architecture.md#trainer-compatibility).

¹ The tied 0.6B, 1.7B and 4B checkpoints are refused by the tie gate; 8B and up split.

² Text rows only: SFT and SMPO refuse an image run under CP at startup, and the wrapper raises on a batch carrying `pixel_values`.

³ `tensor_parallel_size` must divide the KV heads: at most 2 on 35B-A3B.

⁴ Text towers (`text_only_model: true` for the multimodal checkpoints); packed rows run on the GatedDeltaNet segment markers each stage rebuilds. The MoE tower has no bias slot and the split severs `aux_loss`, so it trains with `moe_balancing: none`.

⁵ Only the MLA expansion projections shard.

⁶ Tiny-model verified only. Laguna and DeepSeek-V4 ETP: pure ETP and EP+ETP train, save and resume in the precompute-resume suites. Command A+: only EP8 ran on the full checkpoint, on Halo 1.0.0 or earlier.

⁷ Ling 2.0 only; Ring-mini-linear-2.0 and Ling 3.0 are refused. EP+CP's only GPU test is a tiny-model LoRA row.

⁸ EP+CP's only GPU test is a tiny-model LoRA row.

⁹ Without gradient checkpointing, in every mode.

¹⁰ Text tower only: image data is refused.

¹¹ Tiny-model GPU tests only: no Qwen3-Next checkpoint is trained at full scale here.

¹² Experimental: EP, pure ETP and EP+ETP pass GPU tests on tiny models; a GPU test also trains, exactly resumes and exports the release cut to four layers under EP8, and at full depth only a few EP8 LoRA steps have run. Text rows, packed rows isolated per document under the toolkit's tiled QSA; the n-gram table stays in host memory and every gathered save writes it ([Qwen3.8-Flash-Next](qwen4_exp.md#n-gram-table)).

¹³ Tiny-model GPU tests, plus a 20-step Qwen3-30B-A3B correctness smoke on Qwen3 MoE: no released checkpoint is otherwise trained under PP here. Per-family balancing under PP: [Model support](../parallelism/pipeline-parallelism.md#model-support).

¹⁴ Full fine-tuning with `moe_balancing: none`: PP refuses adapters, the split severs `aux_loss`, and the router has no bias slot ([Qwen3.8-Flash-Next](qwen4_exp.md#pipeline-parallelism)).

¹⁵ `ep8` recipes, images included, ran 5–20 steps on Super VL on 8× B300, with an exact resume at a 9-layer cut. Lightning ran 5 steps at `ep8`, with an exact resume at a 9-layer cut and a byte-identical full export. Everything else on this row, ETP and pipeline parallelism among it, runs on tiny models ([Nemotron-H](nemotron-h.md#status)).

¹⁶ [Dense hybrid SFT](../parallelism/context-parallelism.md#dense-hybrid-sft) restrictions apply.

## MoE knobs

Every MoE family shares three settings:

- **Distribution** — [EP](../parallelism/expert-parallelism.md) (DeepEP) shards experts across GPUs, orthogonal to data parallelism. Add [ETP](../parallelism/expert-tensor-parallelism.md) (expert-FFN sharding) or [TP](../parallelism/tensor-parallelism.md) (attention) for more headroom.
- **Expert compute** — [Grouped GEMM](../optimization/grouped-gemm.md) (`use_grouped_gemm`, default on at SM90+) batches the per-expert matmuls.
- **Load balancing** — `moe_balancing`: `auto` (default), `aux_loss`, `bias_update` (DeepSeek-V3 auxiliary-loss-free bias, step size `router_balancing_rate`), `bias_update_transient`, or `none`.

`auto` resolves per family ([Callbacks → `auto` resolution per family](../training-methods/callbacks.md#auto-resolution-per-family)); which families export an explicit `bias_update` is the slot table under [BalancingBiasCallback](../training-methods/callbacks.md#balancingbiascallback).

`aux_loss` trains the routers only where the trainer's loss adds the term — SFT, DPO and KTO — and does so with or without gradient checkpointing ([Callbacks](../training-methods/callbacks.md#aux_loss-under-gradient-checkpointing)).

Every expert / router config field — the expert count, the top-k spelling, `router_aux_loss_coef`, `output_router_logits` — lives on the **text** sub-config for a composite (multimodal) family (Qwen3.5/3.6-MoE, Gemma 4, Mistral-4 under its VLM wrapper). `PreTrainedConfig` does not delegate attribute reads, so the toolkit resolves these through `get_config_field` / `set_config_field` over `config` **and** `config.get_text_config()`. Read one field off the wrapper alone and balancing plus `moe/*` metrics silently disappear.

Families spell the count and the width differently (`num_experts`, `num_local_experts`, `num_routed_experts`, `num_moe_experts`, `moe_num_experts`; `num_experts_per_tok`, `top_k_experts`, `moe_router_topk`, …). `resolve_expert_count` and `resolve_router_topk` in `halo/models/moe/config_fields.py` are the one reader every consumer calls — the "is this MoE?" loader gate, the load metrics, and the pipeline split's FFN cost model. Under EP, `detect_num_experts` reads the count off the live expert container instead.

[MoEMetricsCallback](../training-methods/callbacks.md#moemetricscallback) logs per-layer expert load by default (`enable_moe_metrics`).

## Load precision

The training and scoring loaders cast floating parameters to the run dtype after `from_pretrained`
and before parallel wrappers (`cast_parameters_to_run_dtype` in `halo/model_loading/dtype.py`, applied by
the whole-model loads through `load_whole_model`; lazy loaders cast per tensor). This overrides transformers' FP32 module pins — DeepSeek-V4's norms and
hyper-connections, GLM-5 Next's KDA state, Inkling's short convolutions — so FSDP2 sees one dtype
within each shard group. Persistent buffers have a separate policy: a family-pinned buffer stays
FP32 on every loader, including the lazy EP loader's `e_score_correction_bias`, and Zaya's balancing
biases are FP32 by design.

Configured FP32 parameter masters are the exception. The shared selector in
`halo/model_loading/parallel/fp32_masters.py` (selection, coverage check and eager restore) preserves non-EP weights under `fp32_non_ep_params`
and the EP layer's selected router/expert weights under their flags. Eager construction streams
those checkpoint values back before wrapping; lazy EP construction reads and fuses them at FP32
directly. Native dense TP rebuilds its existing TP shards at FP32 before the DP wrap, preserving tied
aliases and never moving a full FP32 matrix onto each GPU. This holds under every parallelism mode,
for a new training run started from a checkpoint and for a resume alike. A full-fine-tune resume
requires every configured master to be restored. See
[Checkpoint precision](../reference/checkpoints.md#what-gets-saved) and the
[eager EP memory/read cost](../parallelism/expert-parallelism.md#model-loading).

A quantized base keeps packed bnb `Params4bit` storage, including floating `bnb_4bit_quant_storage`;
it is never a parameter master. FP8 weights are refused on every training loader that casts (all but the
dense TP loader below): dequantize once to
BF16 with the family's `scripts/before_training/convert_*_bf16.py` converter. Three loads skip the
cast: the conversion tools, which keep the original pins; the deduplication embeddings tool, which
loads at the checkpoint's dtype; and the dense TP loader, which loads straight into DTensors (no dense
family pins a parameter). The reward-scoring tool casts the whole model, buffers included, to its
`--rm_dtype`.

## MTP heads

Qwen3.5/3.6, Qwen3.8-27B, Qwen3-Next, GLM-4 MoE Lite and Nemotron-H train the head their release ships under `mtp_loss_weight` ([MTP heads](../optimization/mtp-heads.md)); other tail-bearing families refuse by name. Without it, every family drops its tail at load and exports carry none, so `method=mtp` refuses them, except Ling-3.0-flash, whose remote class builds and re-exports its tail ([Bailing](bailing.md#mtp-tail)). `convert_to_bf16` and `patch_vocab` write a stored head of these families back; any other family's tail drops, with a warning naming the config field its output keeps. `reattach_vision_tower.py` carries the base's head at its weights, or keeps the one a text-only export trained.

## Per-family pages

Each family page covers its model-specific caveats (attention quirks, checkpoint variants, balancing) and why it drops an axis.

- [Qwen3](qwen3.md) — dense and MoE; reference platform for trainers and tests.
- [Qwen3.5 / Qwen3.6 MoE](qwen3_5.md) — hybrid linear/full attention; the shipped EP config trains 35B-A3B. The dense Qwen3.8-27B runs the dense path, Qwen3.8-2.4T-A95B the MoE one.
- [Qwen3-Next](qwen3-next.md) — Qwen3.5-MoE's block and hybrid attention under a router with a configurable renormalization; per-expert hub checkpoints.
- [Qwen3.8-Flash-Next](qwen4_exp.md) — experimental: Qwen3-Next's router over hyper-connection streams, sparse QSA attention and a 95 GiB n-gram table held in host memory.
- [GPT-OSS](gpt-oss.md) — 20B/120B, GLU experts and attention sinks.
- [GLM-4 MoE Lite](glm4.md) — LoRA-style attention compression, 64 routed + shared experts.
- [Laguna](laguna.md) — Poolside MoE; its EP layer subclasses GLM-4 MoE Lite's, differing in the top-k weight default and the hub↔module key spellings.
- [Inkling](inkling.md) — Thinking Machines multimodal MoE; joint routed+shared router normalization forces a standalone wrapper, and its shipped config runs multi-node EP16.
- [Gemma 4 MoE](gemma4.md) — router outside the EP-wrapped module.
- [Bailing MoE / Ling](bailing.md) — softmax GQA + shared experts; needs `trust_remote_code`.
- [LFM-2 MoE](lfm2.md) — sigmoid routing with optional expert-bias buffer.
- [Mistral4 MoE](mistral4.md) — Mistral3 VLM text backbone, MLA attention.
- [DeepSeek-V4](deepseek-v4.md) — hash-routing bootstrap + top-k MoE, compressed sparse attention, eager-only.
- [Nemotron-H](nemotron-h.md) — NVIDIA Nemotron 3.5 Super VL: Mamba-2, MoE and attention layers, gateless experts on a latent, a composite VLM with a text-only route. Nemotron 3.5 Lightning is the text class alone, with no latent.
- [Zaya](zaya.md) — CCA-augmented MoE with cross-layer router state.
- [Cohere2 MoE](cohere2-moe.md) — Command A+ VLM text backbone; interleaved sliding/NoPE attention, averaged shared expert.
- [GLM-5 Next](glm5-next.md) — GLM-5.3-Flash composite VLM; KDA + DeepSeek sparse attention, hyper-connections, 288 routed experts.
- [Step-3.7 Flash](step3p7.md) — StepFun composite VLM; full/sliding attention interleave with per-layer head counts and per-head sigmoid gates, 288 routed experts.
- [Adding a new model](adding-a-model.md) — per-mode checklists for EP, CP, TP, or vendoring.

Which layout to pick per sequence length and memory budget: [Parallelism → Recommendations](../parallelism/README.md#recommendations).

Parallelism and kernel vocabulary: [Reference Glossary](../reference/glossary.md).
