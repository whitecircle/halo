# LFM-2 MoE

Liquid AI's `Lfm2MoeForCausalLM` — a hybrid MoE (interleaved short-convolution and full-attention layers) with sigmoid routing and an optional non-trainable `expert_bias` buffer for top-k selection correction. EP, ETP, and selective TP are supported; CP is blocked by the sequence-axis short-conv layers.

| | EP | CP | TP | ETP | PP | EP+TP |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| LFM-2 MoE | Yes | **No** | Yes | Yes | **No** | Yes |

PP is refused at load: `tie_word_embeddings` is `True` — the config-class default, which no released checkpoint overrides — and a tied checkpoint splits stage 0's embedding from the last stage's head with no reconcile, so the tie gate rejects it ([Pipeline Parallelism](../parallelism/pipeline-parallelism.md)). Setting the flag `false` is not a way around it: the released weights carry no `lm_head` tensor, so the untied head would train from random init.

## EP wrapper

`EPLfm2MoELayer` (`src/distributed/expert_parallel/layers/lfm2.py`) replaces `Lfm2MoeSparseMoeBlock`.

- Routing: sigmoid scores, plus optional `expert_bias` for top-k selection only; the selected weights come from the unbiased scores, normalized to sum 1 when `norm_topk_prob` (default), then scaled by `routed_scaling_factor`. `expert_bias` is a buffer (not trainable), so it affects selection but doesn't appear in gradients and needs no grad-sync hook.
- Storage: fused `gate_up_proj` and `down_proj` as 3D tensors. Compute via Grouped GEMM (fused-GLU path).
- No shared experts (the wrapper extends `EPSharedExpertsMoELayerBase` with `shared_experts=None`, so the shared leg is skipped).
- Gathered saves write the per-expert hub layout (`experts.{i}.w{1,3,2}.weight`) via `_PER_EXPERT_UNFUSED_KEYS` — the layout vLLM's per-expert loader reads. See [Serving on vLLM / SGLang](../reference/checkpoints.md#serving-on-vllm-sglang).

## TP and ETP

Selective attention-only TP — `Lfm2MoeAttention` is in `TP_SHARDABLE_ATTENTION_CLASSES` (`src/distributed/tensor_parallel/module_types.py`); `Lfm2MoeSparseMoeBlock` is skipped (handled by EP). Only the full-attention layers shard — the short-conv layers stay replicated, so the per-rank footprint falls by far less than `1/tp_size`.

LFM-2 uses the [fused contiguous-halves ETP layout](../parallelism/expert-tensor-parallelism.md#weight-sharding): `_init_fused_glu_params` splits the `[gate(M) | up(M)]` halves and stores `gate_proj` / `up_proj` / `down_proj` separately when `expert_tp_size > 1`. The non-trainable `expert_bias` buffer is replicated across ETP ranks.

## Packing

Documents stay isolated in both halves of the hybrid stack. The attention layers key on the packed
`position_ids`; for the `Lfm2MoeShortConv` layers the collators additionally emit `seq_idx` for this
family (`select_data_collator` turns it on by model type) and both conv paths honor it, which zeroes
the cross-document term exactly rather than approximately.

## Why CP isn't supported

LFM-2 is architecturally hybrid: `Lfm2MoeDecoderLayer` interleaves `full_attention` blocks with `Lfm2MoeShortConv` layers (an `nn.Conv1d` over the sequence axis via `causal_conv1d_fn`), selected per layer by `config.layer_types`. The short-conv layers mix tokens along the sequence axis, so a Ulysses sequence split would sever the convolution receptive field across CP ranks — the same blocker as Qwen3.5/3.6's `Qwen3_5MoeGatedDeltaNet`. No Ulysses wrapper is provided for the hybrid stack, so CP is unavailable on released LFM-2 checkpoints.

## Configs

No SFT configs ship for LFM-2; the path is exercised by `tests/gpu/trainers/sft/test_sft_lfm2_moe.py` (`LiquidAI/LFM2-24B-A2B`, `--mode fsdp` and `--mode ep`). To train in production, copy an EP config from another MoE family (e.g. `examples/sft/gptoss/gptoss-20b-multinode-ep.yaml`) and swap the model path. The one shipped LFM-2 config is the LFM-2.5 VL self-distillation example, `examples/distillation/lfm2/self-distill-lfm2.5-vl-1.6b.yaml` — a **dense** `LiquidAI/LFM2.5-VL-1.6B` checkpoint (`text_config.model_type: lfm2`, no experts), so none of the MoE columns above apply to it.

**Routing drift**: sigmoid routing under SFT can collapse onto a handful of experts (same failure mode as Bailing). Two mitigations work — freeze the router weights with `freeze_layers_patterns: ["*.feed_forward.gate.weight"]`, or set `moe_balancing: bias_update` (EP path), whose DeepSeek-V3 sign-updates land in the block's own `expert_bias` buffer, adopted whole so the trained bias exports with every checkpoint.

On a `use_expert_bias: false` checkpoint the slot is **materialized** at enable time (a zero buffer in the same sigmoid-score space, a semantic no-op at creation) and `use_expert_bias` flips in the exported config, so serving engines load and apply the trained bias there too.

`moe_balancing: auto` picks `bias_update` here. `Lfm2MoeForCausalLM` has no aux-loss path — no `output_router_logits` / `router_aux_loss_coef` config field, no router-logit output, no aux term in the loss — and a `forward` that declares no `output_router_logits` parameter is the signal `auto` keys on, so it takes the EP-wrapper bias wherever the wrappers apply (`ep_size > 1` or `use_grouped_gemm`, the default under torchrun).

Without the wrappers nothing carries the bias and no aux term can reach the loss, so `auto` resolves to `none` with a warning; freeze the router there. See [Performance & Balancing Flags](../reference/configuration-reference.md#performance-balancing-flags) and [RouterBiasBalancingCallback](../training-methods/callbacks.md#routerbiasbalancingcallback).
