# Supported Models

**Any HuggingFace `AutoModelForCausalLM` trains with standard FSDP.** The matrix below is about *advanced* parallelism — EP for MoE routing, CP for long sequences, TP for weight sharding, ETP for expert FFN sharding.

The **TP column** tracks the Halo-side path: pure TP for dense models, and *selective* TP (attention only; embeddings and `lm_head` replicated; MoE left to EP) for MoE models in EP+TP mode. Most modern HF models also support native TP via `base_model_tp_plan`; GPT-OSS ships a `base_model_ep_plan` and no `base_model_tp_plan`, so its TP runs through the Halo path. **No** means no working selective-TP path.

The registries behind the matrix: EP wrappers under `src/distributed/expert_parallel/layers/`, CP wrappers (which the CP accept-list is derived from) under `src/distributed/context_parallel/layers/`, TP accept-list in `src/distributed/tensor_parallel/module_types.py`.

## Compatibility matrix

| Model Family | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA ⁷ | Training Configs |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--|
| [Qwen3 (dense)](qwen3.md) | — | Yes | Yes | — | — | — | Yes | `examples/sft/qwen3/qwen3-4b-*` |
| [Qwen3 MoE](qwen3.md#qwen3-moe) | Yes | Yes | Yes | Yes | Yes ⁶ | Yes | Yes | — |
| [Qwen3-VL (text)](qwen3.md#qwen3-vl) | — | Yes | **No** ⁵ | — | — | — | Yes | — |
| [Qwen3.5 / Qwen3.6 MoE](qwen3_5.md) | Yes | **No** ¹ | Yes | Yes | **No** | Yes | Yes | `examples/sft/qwen3_5/*` |
| [GPT-OSS](gpt-oss.md) | Yes | Yes | Yes | Yes | Yes ⁶ | Yes | Yes | `examples/sft/gptoss/*` |
| [GLM-4 MoE Lite](glm4.md) | Yes | Yes | Yes ² | Yes | Yes ⁶ | Yes | Yes | `examples/sft/glm4/*` |
| [Laguna S / XS 2.1](laguna.md) | Yes | **No** | **No** | untested | **No** | **No** | Yes | `examples/sft/laguna/*` |
| [Inkling-Small](inkling.md) | Yes | **No** | **No** | Yes | **No** | **No** | untested | `examples/sft/inkling/*` |
| [Gemma 4 MoE](gemma4.md) | Yes | **No** | **No** | Yes | **No** | **No** | Yes | `examples/sft/gemma4/*` |
| [Bailing MoE / Ling](bailing.md) | Yes | Yes ³ | **No** ³ | Yes | untested ⁶ | **No** | Yes | `examples/sft/ling_mini_2/*` |
| [LFM-2 MoE](lfm2.md) | Yes | **No** | Yes | Yes | **No** | Yes | Yes | — |
| [Mistral4 MoE](mistral4.md) | Yes | Yes | Yes | Yes | Yes ⁶ | Yes | Yes | `examples/sft/mistral4/*` |
| [DeepSeek-V4](deepseek-v4.md) | Yes | **No** ⁸ | **No** ⁸ | untested | **No** | **No** | Yes | `examples/sft/deepseek_v4/*` |
| [Zaya (Zyphra/ZAYA1)](zaya.md) | Yes ⁴ | **No** ⁴ | **No** ⁴ | Yes | **No** | **No** | Yes | `examples/sft/zaya/*` |
| [Cohere2 MoE (Command A+)](cohere2-moe.md) | Yes | Yes ⁹ | Yes ⁹ | Yes ⁹ | Yes ⁶ ⁹ | Yes ⁹ | untested | `examples/sft/cohere2_moe/*` |
| [GLM-5 Next (GLM-5.3-Flash)](glm5-next.md) | Yes | **No** ¹⁰ | **No** ¹⁰ | Yes | **No** | **No** | Yes | `examples/sft/glm5_next/*` |
| [Step-3.7 Flash](step3p7.md) | Yes | **No** ¹¹ | **No** ¹¹ | Yes | **No** | **No** | Yes | `examples/sft/step3p7/*` |
| Any other HF model with `tp_plan` (Llama, Mistral, Phi, …) | — | — | Yes | — | — | — | Yes | — |

**Pipeline parallelism** is not a column above because it is [not yet available in this release](../parallelism/pipeline-parallelism.md) — the config surface and the per-family seams ship (specs, split gates, stage adapters), the schedule engine does not.

Those shipped seams: Zaya and Gemma 4 opt out by declaring `SUPPORTS_PP = False` on their `PPModelSpec`; DeepSeek-V4 and GLM-5 Next split through a family spec that carries their widened hyper-connection stream as the boundary; LFM-2 and Cohere2 MoE are refused by generic gates instead — tied embeddings for both, and for Cohere2 the stage loader's lazy-loading requirement, which its EP layer declares off; Step-3.7 Flash and GLM-5 Next ship only their composite class, which the multimodal gate admits for a run that feeds no images (the vision tower and projector are held by no stage, stashed for the save) and refuses for image data; Ling-3.0-flash is blocked by a config value its checkpoint ships (a live MTP tail layer), which `model_init_kwargs` turns off. Trainer × parallelism support is tracked in [Trainer Compatibility](../reference/trainer-architecture.md#trainer-compatibility).

¹ Qwen3.5/3.6 ship a working CP wrapper for the full-attention block, but every released checkpoint also carries `Qwen3_5MoeGatedDeltaNet` linear-attention layers whose sequence-axis Conv1d and recurrent scan are not CP-shardable. Validation rejects hybrid `layer_types`. See [qwen3_5.md](qwen3_5.md#why-cp-is-blocked-on-real-checkpoints).

² GLM-4 has LoRA-style attention compression; only the expansion projections shard — see [glm4.md](glm4.md#tp).

³ CP covers **Ling 2.0** (standard softmax GQA). `Ring-mini-linear-2.0` is rejected by name — its file reuses the same full-attention class names — and Ling 3.0 (`bailing_hybrid`) pairs KDA linear attention with MLA, so it has no CP path either and additionally needs `attn_implementation: sdpa` and `fp32_non_ep_params: true` (with `fsdp_shard_ep1_experts: false` at `ep_size: 1`); see [bailing.md](bailing.md#ling-30). TP remains unavailable for all of them: a missing DTensor plan, not an architectural blocker.

⁴ Zaya supports **EP without gradient checkpointing**, **EP+ETP without GC**, *or* **plain FSDP2 without GC**. CCA's sequence-axis `Conv1d` rules out CP; no Zaya attention class is in the selective-TP accept-list and upstream ships no `base_model_tp_plan`, so `tensor_parallel_size > 1` raises. See [zaya.md](zaya.md#limitations).

⁵ A dense Qwen3-VL takes the HF-native TP route (`tp_plan="auto"`), and the architecture ships no `base_model_tp_plan` — the plan resolves empty, sharding nothing, so the loader **raises** rather than running `tp_size` full replicas at `1/tp_size` throughput. `Qwen3VLTextAttention` stays in the CP and selective-TP accept-lists, which govern CP and the attention-only path the loader takes for MoE and EP+TP shapes. The MoE variants get no EP: no wrapper claims `Qwen3VLMoeTextSparseMoeBlock`, and `Qwen3VLMoeTextAttention` is in neither registry.

⁶ EP+CP is gated the same way for every family: `_validate_ep_cp` requires node-local EP with `ep_group_size == nvlink_domain_size` and rejects both a smaller `ep` within the domain and cross-domain EP (`ep_scope='global'`). On an 8-GPU node that pins `ep_size` to 8; `cp_size` only has to divide the domain, and the fully orthogonal shape is `cp_size == ep_group_size == nvlink_domain_size`.

⁷ LoRA "Yes" means the family trains with adapters under FSDP/DP, EP, CP, and pure ETP. It is rejected at trainer construction under **TP** and **EP+TP** (adapters are plain tensors outside the TP DTensor graph). Under EP the adapters cover attention (PEFT) *and* the experts (native grouped adapters); the expert half is refused at `expert_tp_size > 1`, leaving attention-only LoRA there. PP rejects both halves. See [PEFT](../optimization/peft.md).

⁸ DeepSeek-V4 is eager-only (`head_dim=512` exceeds every FA kernel; sinks + compressor KV concat rule out SDPA/flex). Its CSA/HCA compressors pool token windows along the sequence axis (no CP), and shared-KV MQA is not shardable (`apply_tp_to_attention_only` raises). `padding_free` is rejected — no varlen kernel. See [deepseek-v4.md](deepseek-v4.md).

⁹ Tiny-model verified on the single-node 8-GPU matrix (ep8 / cp8 / tp8 / ep8+cp2 / ep8+tp2 / etp8 / ep2+etp4) plus EP=2 loss/grad equivalence vs the stock model; only EP=8 is validated at the full Command A+ scale. See [cohere2-moe.md](cohere2-moe.md).

¹⁰ 34 of GLM-5 Next's 45 layers are a KDA linear recurrence (sequence-axis conv + delta-rule scan), so validation rejects its `layer_types` for CP; TP has no sound shard plan (DSA indexer + KDA projections) and `tensor_parallel_size > 1` is rejected. See [glm5-next.md](glm5-next.md#limitations).

¹¹ Step-3.7 Flash's per-layer head counts (64 full / 96 sliding) fit no uniform q/k/v shard plan, so `Step3p7Attention` is outside the selective-TP accept-list and TP shards zero layers (raise); no Ulysses CP wrapper is registered either — nothing architectural, the head counts divide cp 2/4/8. See [step3p7.md](step3p7.md#limitations).

## MoE knobs

Every MoE family shares three settings:

- **Distribution** — [EP](../parallelism/expert-parallelism.md) (DeepEP) shards experts across GPUs, orthogonal to data parallelism. Add [ETP](../parallelism/expert-tensor-parallelism.md) (expert-FFN sharding) or [TP](../parallelism/tensor-parallelism.md) (attention) for more headroom.
- **Expert compute** — [Grouped GEMM](../optimization/grouped-gemm.md) (`use_grouped_gemm`, default on at SM90+) batches the per-expert matmuls.
- **Load balancing** — `moe_balancing`: `auto` (default), `aux_loss`, `bias_update` (DeepSeek-V3 auxiliary-loss-free bias, step size `router_balancing_rate`), `bias_update_transient`, or `none`.

`auto` resolves to `bias_update` for Zaya (native balancing-bias buffer), for DeepSeek-V4, Inkling and Bailing/Ling (their EP wrappers sever the aux-loss path), and for GLM-4 MoE Lite, LFM-2 and — under its EP wrapper — Step-3.7 Flash (no honored `output_router_logits` path, and the bias lands in a checkpoint-persistent tensor). Mistral-4, Cohere2 MoE and the multimodal Qwen3.5/3.6 have no exportable slot at all, so `auto` gives `none` with a warning; the explicit `bias_update_transient` opts into trainer-only balancing there, and exported checkpoints serve without the bias. Gemma 4 also gives `none`: its forward never takes the flag and its EP wrapper accepts no bias, so no mode can balance it. `auto` reaches that same `none` on any tree carrying neither an aux-loss path nor an EP wrapper — the `ep_size=1` + `use_grouped_gemm: false` window for GLM-4 MoE Lite, LFM-2 and Step-3.7 Flash. Everything else resolves to `aux_loss` — including Laguna and GLM-5 Next, whose forwards honor the flag; their exported `e_score_correction_bias` slot still makes explicit `bias_update` legal.

GPT-OSS takes an explicit `bias_update`, adopting its hub `router.bias`, which exports and serves; Qwen3 MoE and text-only Qwen3.5/3.6 take `bias_update_transient`. See [RouterBiasBalancingCallback](../training-methods/callbacks.md#routerbiasbalancingcallback).

Every expert / router config field — the expert count, the top-k spelling, `router_aux_loss_coef`, `output_router_logits` — lives on the **text** sub-config for a composite (multimodal) family (Qwen3.5/3.6-MoE, Gemma 4, Mistral-4 under its VLM wrapper). `PreTrainedConfig` does not delegate attribute reads, so the toolkit resolves these through `get_config_field` / `set_config_field` over `config` **and** `config.get_text_config()`. Read one field off the wrapper alone and balancing plus `moe/*` metrics silently disappear.

Families spell the count and the width differently (`num_experts`, `num_local_experts`, `num_routed_experts`, `num_moe_experts`, `moe_num_experts`; `num_experts_per_tok`, `top_k_experts`, `moe_router_topk`, …). `ROUTER_EXPERT_COUNT_FIELDS` and `ROUTER_TOPK_FIELDS` in `src/models/moe_balancing.py` are the one registry every consumer reads — the "is this MoE?" loader gate, the load metrics, and the pipeline split's FFN cost model. Under EP the wrapper's own `_NUM_EXPERTS_ATTR_PATHS` resolves the count off the live module instead.

[MoEMetricsCallback](../training-methods/callbacks.md#moemetricscallback) logs per-layer expert load by default (`enable_moe_metrics`).

**Hybrid families** — [Qwen3.5 / 3.6](qwen3_5.md) (GatedDeltaNet), [LFM-2](lfm2.md) (short convolution), [GLM-5 Next](glm5-next.md) (KDA) and [Bailing](bailing.md)'s Ling 3.0 (KDA) / Ring (Lightning Attention) interleave linear or convolutional attention with standard attention. Their MoE side runs under EP, but the sequence-axis recurrence cannot be split, so CP is unavailable on released checkpoints.

## Per-family pages

The matrix carries each family's supported modes; the per-family page covers model-specific caveats (attention quirks, checkpoint variants, balancing).

- [Qwen3](qwen3.md) — dense and MoE; reference platform for trainers and tests.
- [Qwen3.5 / Qwen3.6 MoE](qwen3_5.md) — hybrid linear/full attention; the shipped EP config trains 35B-A3B.
- [GPT-OSS](gpt-oss.md) — 20B/120B, GLU experts and attention sinks.
- [GLM-4 MoE Lite](glm4.md) — LoRA-style attention compression, 64 routed + shared experts.
- [Laguna](laguna.md) — Poolside MoE; its EP wrapper subclasses GLM-4 MoE Lite's and overrides the top-k weight default plus the hub↔module key spellings.
- [Inkling](inkling.md) — Thinking Machines multimodal MoE; joint routed+shared router normalization forces a standalone wrapper, and its shipped config runs multi-node EP16.
- [Gemma 4 MoE](gemma4.md) — router outside the EP-wrapped module.
- [Bailing MoE / Ling](bailing.md) — softmax GQA + shared experts; needs `trust_remote_code`.
- [LFM-2 MoE](lfm2.md) — sigmoid routing with optional expert-bias buffer.
- [Mistral4 MoE](mistral4.md) — Mistral3 VLM text backbone, MLA attention.
- [DeepSeek-V4](deepseek-v4.md) — hash-routing bootstrap + top-k MoE, compressed sparse attention, eager-only.
- [Zaya](zaya.md) — CCA-augmented MoE with cross-layer router state.
- [Cohere2 MoE](cohere2-moe.md) — Command A+ VLM text backbone; interleaved sliding/NoPE attention, averaged shared expert.
- [GLM-5 Next](glm5-next.md) — GLM-5.3-Flash composite VLM; KDA + DeepSeek sparse attention, hyper-connections, 288 routed experts.
- [Step-3.7 Flash](step3p7.md) — StepFun composite VLM; full/sliding attention interleave with per-layer head counts and per-head sigmoid gates, 288 routed experts.
- [Adding a new model](adding-a-model.md) — per-mode checklists for EP, CP, TP, or vendoring.

## Choosing a parallelism configuration

| You have | Sequence length | Recommended mode |
|---|---|---|
| Dense that fits per GPU | ≤ 32K | FSDP2, no parallelism flags |
| Dense Qwen3, long context | > 32K | [CP](../parallelism/context-parallelism.md) |
| Dense, doesn't fit per GPU | any | [TP](../parallelism/tensor-parallelism.md) via native `tp_plan` |
| MoE with full coverage (Qwen3 MoE, GPT-OSS, GLM-4, Mistral4, Cohere2 MoE) | ≤ 32K | [EP](../parallelism/expert-parallelism.md), or EP+TP at very large scale |
| Same MoEs, long context | > 32K | EP+CP |
| MoE without CP (Qwen3.5/3.6, LFM-2, Gemma 4, DeepSeek-V4, Laguna, Inkling, Ling 3.0, GLM-5 Next, Step-3.7 Flash) | any | EP; add EP+TP for Qwen3.5/3.6 and LFM-2, or pure ETP (`ep_size=1`) when expert memory is the bottleneck |
| Ling 2.0, long context | > 32K | EP, plus CP once the sequence exceeds one GPU |
| Zaya | any | EP without GC (optionally + ETP), or plain FSDP2 without GC |

Parallelism and kernel vocabulary: [Reference Glossary](../reference/glossary.md).
