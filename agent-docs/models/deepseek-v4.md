# DeepSeek-V4

`DeepseekV4ForCausalLM` (transformers-native `deepseek_v4`) — MoE with a hash-routing bootstrap: `mlp_layer_types` schedules each MoE layer as `hash_moe` or `moe` (default: the first three `hash_moe`, the rest `moe`). A `hash_moe` layer selects experts via a frozen `tid2eid[input_ids]` table; a `moe` layer uses `sqrtsoftplus` top-k routing with a `noaux_tc` correction bias. Each layer carries a shared expert and clamped SwiGLU experts (`swiglu_limit`).

Around that sit Manifold-Constrained Hyper-Connections (`hc_mult` parallel residual streams) and three attention layer types (sliding window / CSA + Lightning Indexer / HCA) over shared-KV MQA with `head_dim=512` and learnable sinks. V4-Flash: 43 layers, 256 routed experts, top-k=6.

| | EP | CP | TP | ETP | EP+TP | EP+CP |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| DeepSeek-V4 | Yes | No | No | Yes (untested) | No | No |

- **CP** — the CSA/HCA compressors pool non-overlapping token windows along the sequence axis; a CP shard would compress incomplete windows at every chunk boundary. Rejected by class name in `src/distributed/context_parallel/validation.py`.
- **TP** — `DeepseekV4Attention` (shared-KV MQA broadcast to all heads + the compressor branch) is not shardable; `apply_tp_to_attention_only` raises when a model ends up with zero shardable attention layers under `tp_size > 1`.
- **ETP** — the experts use the shared fused-GLU storage, so `expert_tp_size > 1` mechanically works through `_init_fused_glu_params`; not yet validated on V4.
- **PP** — [not yet available in this release](../parallelism/pipeline-parallelism.md). The shipped `DeepSeekV4PPSpec` split contract carries the `hc_mult`-widened hyper-connection stream as the stage boundary, keeps `hc_head` on the last stage, and gives non-first stages a mirrored mid-chain forward. Every `hash_moe` layer must sit on stage 0 — its router consumes `input_ids`, which only stage 0 receives — and the split gate refuses a partition that strands one; the hub's `num_nextn_predict_layers: 1` is metadata and passes the live-MTP gate.
- **RL weight sync** — online and environmental GRPO reject DeepSeek-V4 at trainer construction (`validate_weight_sync_support`; `EPDeepseekV4MoELayer._supports_weight_sync = False`). The sync feeds trainer parameter names straight into vLLM's `model.load_weights`, but vLLM serves V4 from an out-of-tree package whose loader targets DeepSeek's original release checkpoint, not the HuggingFace module tree the toolkit trains — per-expert vs fused experts, fused vs separate attention projections, bare `embed.weight` vs `model.embed_tokens.weight`. Those weights are also fp8/fp4-packed and the o-projection reads a `weight_scale_inv` unconditionally, so the BF16 checkpoint is not servable there either. No key mapping fixes this from the gather side.

## Attention: eager-only

Every non-eager backend is off (`_supports_flash_attn/_supports_sdpa/_supports_flex_attn = False` upstream): FA2/3/4 cap head_dim at 256 (V4 is 512), SDPA drops the learnable sink column, and the compressors concatenate KV entries after the mask is built (no BlockMask resize). `load_distributed_model` forces `attn_implementation="eager"` with a warning (`_model_is_deepseek_v4` in `src/models/patches/attention.py`).

Consequences:

- **No varlen path** — `padding_free` is rejected by the collator factory, which gates on the resolved `_attn_implementation` (only `flash_attention_2/_3/_4` qualify) and so catches DeepSeek-V4's eager attention. `packing` still runs, but materializes a dense mask over the flattened batch (side up to `per_device_train_batch_size * max_length`) instead of consuming `cu_seqlens`.
- **Packed documents are isolated in the masked-attention layers only.** The mask is synthesized from the per-document `position_ids` whenever no cache is live, which training always is (`use_cache=False`). The CSA and HCA compressor layers pool KV across the whole row and so cross document boundaries by construction — an accepted mixer-class cost, the same one the linear-attention families carry ([Document isolation](../data/collators.md#document-isolation-under-packing)).
- The per-rope-type rotary buffers (`{main,compress}_inv_freq`, on the model-level rotary and inside every compressor/indexer) are recomputed in fp32 by the rotary fixer chain `finalize_loaded_model` walks on every load path.

## EP wrapper

`EPDeepseekV4MoELayer` (`src/distributed/expert_parallel/layers/deepseek_v4.py`) replaces `DeepseekV4SparseMoeBlock`. It subclasses `EPMoELayerBase` directly (not the shared-experts forward base): the routers return tuples and hash layers need `input_ids`, which the decoder layer forwards into the MoE block.

- **Top-k layers**: fp32 scores via `sqrtsoftplus(logits)`, selection on `scores + e_score_correction_bias` (+ the balancing bias when enabled), weights gathered from the unbiased scores, normalized (`+1e-20`), scaled by `routed_scaling_factor` (1.5).
- **Hash layers** (`is_hash`): selection is `gate.tid2eid[input_ids]`; the forward raises if `input_ids` is absent (an `inputs_embeds`-only call cannot hash-route). The table must hold distinct experts per token id (DeepEP dispatch asserts distinct top-k on device; the wrapper validates at init and raises).
- **Clamped SwiGLU**: the experts compute `silu(gate.clamp(max=limit)) * up.clamp(±limit)` by latching that combine into the `_glu_combine` seam (one Triton kernel, `fused_clamped_silu_mul`, with the bound as a runtime argument; every base GLU path — fused, grouped-GEMM, ETP separate — routes through it). The fused form hardcodes SiLU, so it is armed by the same behavioral `is_silu_activation` gate as GLM-4 ([Fused SwiGLU](glm4.md#fused-swiglu)); any other `hidden_act` falls back to the generic clamp.
- **Shared expert**: `DeepseekV4MLP` (same clamp), replicated per rank, DP-averaged via the router grad hook, output added after DeepEP combine.
- Expert storage is the standard fused contiguous-halves layout (`gate_up_proj [E, 2M, H]`), so the base gather/save, the lazy fused-expert loader, native grouped expert-LoRA, and the shard-merge transform (the base `merge_shards_to_hf`, reached by resolving `model_type` through `resolve_ep_merge_layer_class`) all apply unchanged.

## Router balancing

Under EP the wrapper re-derives routing from `gate.weight`, so the HF router module never fires and `outputs.router_logits` stays empty — the aux-loss path is severed (`_ep_severs_aux_loss`). `moe_balancing: auto` therefore resolves to `bias_update` (DeepSeek-V3 sign update via [RouterBiasBalancingCallback](../training-methods/callbacks.md#routerbiasbalancingcallback)), landing in the gate's own exported `e_score_correction_bias`, so the trained bias ships with every checkpoint; an explicit `aux_loss` warns and stays off. Hash layers refuse the balancing bias (their selection is frozen); only top-k layers receive it.

## Checkpoint conversion

The hub checkpoints (`deepseek-ai/DeepSeek-V4-Flash[-Base]`) ship fine-grained FP8 dense weights (e4m3, 128×128 blocks, ue8m0 scales) with FP4-packed experts (`config.expert_dtype="fp4"`, two e2m1 nibbles per int8 byte, [1, 32] scale grid). EP requires plain BF16 experts, so convert once:

```bash
HF_HOME=/mnt/hf python scripts/before_training/convert_deepseek_v4_bf16.py \
    --model_id deepseek-ai/DeepSeek-V4-Flash --output_dir /mnt/models/DeepSeek-V4-Flash-BF16
```

Budget ~750 GB disk (~330 GB download cache + ~420 GB BF16 output) and ~420 GB of host RAM: the model is materialized on CPU. The script routes through transformers' dequantizing loader (`FineGrainedFP8Config(dequantize=True)`) and writes a **uniform BF16** checkpoint: transformers' `_keep_in_fp32_modules_strict` would keep the HC/norm modules fp32, whose fp32 outputs crash the eager bf16 forward on a dtype mismatch. Those modules upcast internally, so flattening is safe — the EP loader materializes uniform bf16 the same way.

A plain `from_pretrained` load (non-EP) resurrects the fp32 pins even from the uniform-BF16 checkpoint, and that mix does not wrap: `fully_shard` asserts one original parameter dtype per shard group, so a multi-GPU wrapper-less FSDP2 run fails at wrap — the same failure Ling 3.0's fp32 KDA parameters produce ([Ling 3.0](bailing.md#ling-30)).

Train V4 under EP (every shipped config), where the lazy loader materializes uniform bf16 and `fp32_experts` / `fp32_non_ep_params` are the supported way to hold masters in fp32. A single-GPU run needs none of this and computes the pins in fp32. Where a mixed-precision policy does cast pins, a wrap-time warning lists the affected modules (`_warn_fp32_pins_cast_by_policy`); the fp32-sensitive paths upcast internally anyway, so the cast affects weight-value rounding only.

## Liger

`use_liger_kernel: true` defaults to the fused loss — the 129k-vocab logits are never materialized — plus Liger's scoped cross-entropy on the unfused path. RMSNorm (V4 mixes weighted and weightless variants, several fp32-pinned), SwiGLU (clamped, EP-owned), and RoPE (interleaved partial, per-rope-type buffers) are left eager. The fused loss requires `output_router_logits=False`: it replaces the `lm_head` projection the router aux loss is added after, so the term would silently leave the objective.

## Configs and chat template

Example configs under `examples/sft/deepseek_v4/`:

- `v4-flash-ultrachat-lora-ep.yaml` — EP=8 LoRA on the converted checkpoint: PEFT on `q_a_proj`/`q_b_proj`/`o_b_proj`, native grouped expert-LoRA on `gate_up_proj`/`down_proj`. `merge_expert_lora_on_save: true` folds both halves into a servable checkpoint; `false` writes the mixed adapter, which this toolkit resumes but no merge tool folds.
- `v4-tiny-random-smoke-ep.yaml` — EP=2 pipeline smoke on a tiny random-init checkpoint (materialization recipe in the file).

The hub checkpoint ships **no chat template** (DeepSeek uses an external message encoder) — one is required for conversational SFT. `jinja_templates/deepseek-v4-chat.jinja` renders the tokenizer's native `<｜User｜>`/`<｜Assistant｜>` special tokens; pair it with `assistant_message_template: "<｜Assistant｜>"` and `train_on_completions_only: true` (pad == eos).
