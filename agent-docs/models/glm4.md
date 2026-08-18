# GLM-4 MoE Lite

`Glm4MoeLiteForCausalLM` — MoE with sigmoid routing (group-limited top-k), shared experts, and LoRA-style attention compression (`q_a_proj` / `q_b_proj` + `kv_a_proj_with_mqa` / `kv_b_proj`).

| | EP | CP | TP | ETP | PP | EP+TP | EP+CP |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| GLM-4 MoE Lite | Yes | Yes (MLA path) | Yes (selective) | Yes | — ¹ | Yes | Yes |

¹ Pipeline parallelism is [not yet available in this release](../parallelism/pipeline-parallelism.md). The layer stack is uniform, so the shipped PP split contract carries no constraint of its own.

Its MLA-style attention (256-wide qk/v, 64-dim rope split) triggers the FlashAttention-4 backward NaN on Blackwell, so the loader falls back from FA4 to SDPA (`model_fa4_backward_nan_prone`). See [Flash Attention](../optimization/flash-attention.md#model-specific-handling).

## EP wrapper

`EPGlm4MoELayer` (`src/distributed/expert_parallel/layers/glm4.py`) replaces `Glm4MoeLiteMoE`. Laguna shares the shape and subclasses this wrapper — see [Laguna](laguna.md).

- Routing: sigmoid + group mask + top-k. `topk_group` selects eligible expert *groups*, then `top_k` (=4) experts are chosen across them. `route_tokens_to_experts()` runs this before DeepEP dispatch.
- Routed experts: `Glm4MoeLiteExperts`, fused `gate_up_proj` and `down_proj` 3D tensors.
- Shared experts: `Glm4MoeLiteMLP`, replicated per rank, added after DeepEP combine.
- Activation: SwiGLU on the split fused gate_up; Grouped GEMM on SM90+.

## Fused SwiGLU

`EPMoELayerBase._glu_combine` replaces the activation and the multiply with one Triton kernel on every base compute path including the ETP-sharded one. It is a **roster-wide** seam, not a GLM-4 one: `_resolve_activation` latches whatever `resolve_fused_glu_mul` (`src/kernels/fused_glu.py`) returns for the resolved activation — `fused_silu_mul` for a SiLU gate (GLM-4 Lite, Laguna, Qwen3, Qwen3.5/3.6, Bailing, LFM-2, Mistral4, Cohere2 MoE, Inkling, Zaya, and Step-3.7's unclamped layers), `fused_gelu_tanh_mul` for a tanh-GELU one (Gemma 4), `None` otherwise.

A family opts in by *having* a gate the kernels implement, not by setting a flag. A family whose combine is a variant rebinds the same seam: DeepSeek-V4 (clamp before the activation) and Step-3.7 (its two clamped layers clamp after it) arm their kernels off the same SiLU probe, while GLM-5 Next binds its pre-activation clamp unconditionally — its experts hardcode that gate. GPT-OSS owns its interleaved-bias compute paths and never reaches the seam.

Each kernel hardcodes its activation (`x * sigmoid(x)`; the tanh-GELU approximation), so the probe arms one only when the block's resolved activation computes that function exactly — exact (erf) GELU does not arm the tanh-GeGLU kernel. The probe is **behavioral**: it evaluates the activation on a fixed fp32 vector and demands bitwise equality. A type test would not do — `ACT2FN["silu"]` is a `SiLUActivation` module that is neither `nn.SiLU` nor `F.silu`, so an `isinstance`/identity check disarms the kernel on every real block, costing throughput with no numerical trace. The probe fails closed: an activation it cannot vouch for runs eager rather than getting substituted.

## CP wrapper

`Glm4MoeLiteAttention` → `Glm4MoeLiteUlyssesAttention` (`src/distributed/context_parallel/layers/glm4.py`), which declares only its HF class name. Everything below lives on the shared `MLAUlyssesAttentionBase`, which runs the DeepSeek-V3 MLA geometry on the `[B, H, S, D]` path (`_optimize_attention = False`: flash-attn's optimized GQA path assumes plain per-head QKV of one head dim, which MLA's compressed projections and nope/rope split are not). Two MLA-specific bits:

- **Head dims** — `kv_b_proj` expands the compressed KV into `qk_nope_head_dim + v_head_dim`. Where `v_head_dim` is narrower than `qk_head_dim` the base pads V for the flash kernel and crops the output back; on GLM-4.7-Flash the two match at 256 (192 nope + 64 rope), so the pad and crop are no-ops.
- **Shared rope head** — `kv_a_proj_with_mqa` emits `[kv_lora_rank + qk_rope_head_dim]`; the trailing slots are one rotary K vector broadcast to all KV heads *before* the Ulysses all-to-all, so the head-dim scatter gets a contiguous tensor.

GLM-4 has no llama-4 position scaling, so the base forward is unchanged. RoPE follows `config.rope_interleave` and applies only to the rope half.

## TP

Selective TP via `apply_tp_to_attention_only()` needs a tailored plan for the LoRA-style attention:

- **Sharded** (ColwiseParallel): `q_b_proj`, `kv_b_proj` — the expansion linears.
- **Sharded** (RowwiseParallel): `o_proj` — reduces back over the head dim the expansions split.
- **Replicated**: `q_a_proj`, `kv_a_proj_with_mqa` and their layer norms — low-rank compressions (often dim ≤ `tp_size × num_heads`) where sharding produces degenerate per-rank shapes.

## ETP

The routed experts use fused-GLU contiguous-halves storage, so at `expert_tp_size > 1` `_init_fused_glu_params` splits the halves before sharding and stores `gate_proj` / `up_proj` / `down_proj` separately. Shared experts stay replicated. See [ETP](../parallelism/expert-tensor-parallelism.md#limitations).

## Router balancing and precision

GLM-4 MoE Lite's router is DeepSeek-V3 aux-loss-free (sigmoid + a static `e_score_correction_bias`) and the forward has no aux-loss term, so `aux_loss` adds no balancing. `Glm4MoeLiteForCausalLM.forward` declares no `output_router_logits` parameter, so `auto` resolves to `bias_update` wherever the EP wrappers are present (`ep_size > 1` or `use_grouped_gemm`, the default under torchrun); the shipped EP configs set it explicitly.

[RouterBiasBalancingCallback](../training-methods/callbacks.md#routerbiasbalancingcallback) sign-updates the gate's own `e_score_correction_bias` — the native slot the family's selection arithmetic already consults, adopted whole rather than stacked with a side buffer — so the final bias exports with every checkpoint and a served copy routes as training did. To freeze the router instead, use `freeze_layers_patterns: ["*.mlp.gate.weight"]`.

Router precision comes from `fp32_router`, and `fp32_non_ep_params` (dense attention/embeddings/lm_head) turns it on regardless: that upcast skips the EP-owned router, and a bf16 router beside fp32 dense params would trip FSDP2's uniform-dtype check. The example config sets both.

## Chat templates

Three templates under `jinja-templates/glm/`. The first two render the native non-thinking assistant format (`<|assistant|></think>…`):

- `glm-instruct.jinja` — single-turn (raises on more than one assistant message).
- `glm-chat.jinja` — multi-turn system/user/assistant.
- `glm-native.jinja` — the verbatim upstream `zai-org/GLM-4.7-Flash` template: system messages, tools
  (`<tool_call>` XML arg pairs), thinking via `reasoning_content`/`<think>`, observation turns. Use it
  when the training render must match native serving byte-for-byte (tool data, RL); tool-call
  `arguments` must be a parsed mapping, not the OpenAI JSON-string form.

Neither injects a per-turn `<|endoftext|>` mid-conversation, matching native serving where an assistant turn ends at the next role marker (`<|user|>` / `<|observation|>`, both in `config.eos_token_id`). Completion masking stops each turn at that terminator (`src/data/spans.py`). The final assistant turn ends with an explicit `<|endoftext|>` so it carries an in-document terminator under packing.

Pair them with `assistant_message_template: "<|assistant|>"` — the role token alone marks every assistant turn and matches both the thinking and non-thinking renders, where `<|assistant|></think>` would silently miss a thinking turn. See [Collators](../data/collators.md).

## Configs and serving

GLM-4.7-Flash (47 layers, 64 routed experts, top-k=4) trains under EP=8 at ~30K max length: `examples/sft/glm4/glm-4.7-flash-ultrachat-ep.yaml`.

Gathered EP saves and EP-shard merges (`merge_ep_shards.py`) both write the per-expert hub layout vLLM reads directly — GLM-4 declares `_PER_EXPERT_UNFUSED_KEYS`, so the merge is the structural inverse of the gather. A checkpoint that bypassed the EP save path entirely (`merge_peft_adapters.py` over a plain `from_pretrained` load of the stock hub checkpoint, say) stays in the transformers-native fused layout and needs `scripts/after_training/unfuse_moe_experts.py` first.

MLA on Blackwell needs vLLM `--attention-backend CUTLASS_MLA` or SGLang `--attention-backend triton`. See [Serving on vLLM / SGLang](../reference/checkpoints.md#serving-on-vllm-sglang).
