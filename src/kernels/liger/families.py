"""Per-family Liger coverage — the declarative half of the toolkit's applier registry.

A role is listed only where the kernel reproduces the family's own forward bit-for-bit. Anything else (a
clamped GLU, a gated or grouped norm, a scaled MLP output, a rotary that is not Liger's) is left off,
with the reason recorded here: a different activation or normalization changes the model, not just its
speed. A family upstream Liger covers sets ``delegates_to_upstream`` and names only the roles it adds.
"""

from __future__ import annotations

from src.kernels.liger.builder import LigerFamilySpec

# A multimodal wrapper is listed only where its text tower can be nothing but the family named here
# (`inkling_mm_model`, `step3p7`, `glm5_next`). One whose tower may be either of two families
# (`mistral3`: `mistral` or `mistral4`; `lfm2_vl`; `cohere2_vision`) is resolved by the orchestrator
# through `text_config.model_type` instead, so it is never pinned to the wrong sibling.
#
# Of the families the toolkit patches end to end, three declare no `causal_lm`: GLM-5 Next and
# Step-3.7 define no `*ForCausalLM` at all — their `*ForConditionalGeneration` head is the only one,
# and GLM-5 Next's adds the router aux loss after the projection — and Inkling divides the hidden
# states by `logits_mup_width_multiplier` and truncates the logits to `unpadded_vocab_size` before the
# loss. None of that is what the generic fused loss computes, so they keep the unfused head.
LIGER_FAMILY_SPECS: tuple[LigerFamilySpec, ...] = (
    # Mistral 4, the text tower of `mistral3` checkpoints such as Mistral Small 4. Rotary is
    # interleaved (`rope_interleave`) YARN; the llama-4 log scale is applied to the queries after it.
    LigerFamilySpec(
        model_types=("mistral4",),
        modeling_module="transformers.models.mistral4.modeling_mistral4",
        rms_norm=("Mistral4RMSNorm",),
        glu_mlp=("Mistral4MLP",),
        causal_lm=("Mistral4ForCausalLM",),
    ),
    # Zaya. RMSNorm only besides the loss: the EP wrapper replaces `ZayaSparseMoeBlock`, and the
    # rotary is partial. FLCE by default — the `[B*S, 262272]` logits plane is the binding limit.
    LigerFamilySpec(
        model_types=("zaya",),
        modeling_module="transformers.models.zaya.modeling_zaya",
        rms_norm=("ZayaRMSNorm",),
        causal_lm=("ZayaForCausalLM",),
        flce_default=True,
    ),
    # DeepSeek-V4. No RMSNorm: `_keep_in_fp32_modules_strict` pins the `DeepseekV4RMSNorm` weights to
    # fp32, so the eager norm returns fp32 from a bf16 activation where Liger's kernel stores in the
    # input dtype. No GLU: the experts run a *clamped* SwiGLU (`swiglu_limit`). Rotary is interleaved
    # partial with per-rope-type buffers. The head adds the router aux loss after the projection.
    LigerFamilySpec(
        model_types=("deepseek_v4",),
        modeling_module="transformers.models.deepseek_v4.modeling_deepseek_v4",
        causal_lm=("DeepseekV4ForCausalLM",),
        router_aux_loss_in_head=True,
        flce_default=True,
    ),
    # GLM-4.7-Flash. MLA with a two-way interleaved/plain rotary branch.
    LigerFamilySpec(
        model_types=("glm4_moe_lite",),
        modeling_module="transformers.models.glm4_moe_lite.modeling_glm4_moe_lite",
        rms_norm=("Glm4MoeLiteRMSNorm",),
        glu_mlp=("Glm4MoeLiteMLP",),
        causal_lm=("Glm4MoeLiteForCausalLM",),
        flce_default=True,
    ),
    # Laguna. `LagunaMLP` serves both the dense layers and every block's shared expert, which the EP
    # wrapper adopts unchanged. Rotary is half-width on the full-attention layers and full-width on
    # the sliding ones, through one shared function. The head adds the router aux loss after the
    # projection.
    LigerFamilySpec(
        model_types=("laguna",),
        modeling_module="transformers.models.laguna.modeling_laguna",
        rms_norm=("LagunaRMSNorm",),
        glu_mlp=("LagunaMLP",),
        causal_lm=("LagunaForCausalLM",),
        router_aux_loss_in_head=True,
    ),
    # GLM-5.3-Flash. The two plain norms take Liger's kernel; the GDN blocks' gated norm takes fla's
    # (34 of 45 layers apply it per head on the attention output, eager otherwise — the hub-kernel
    # route its decorator names is inert here). `Glm5NextTextUnweightedRMSNorm` carries no weight, so
    # neither kernel expresses it. `Glm5NextTextMLP` clamps gate and up at `swiglu_limit`. The text
    # tower is NoPE, so there is no rotary to fuse.
    LigerFamilySpec(
        model_types=("glm5_next", "glm5_next_text"),
        modeling_module="transformers.models.glm5_next.modeling_glm5_next",
        rms_norm=("Glm5NextTextRMSNorm", "Glm5NextRMSNorm"),
        gated_rms_norm=("Glm5NextTextRMSNormGated",),
    ),
    # Qwen3.5 / 3.6 (dense and MoE) and Qwen3-Next: upstream Liger owns their norms, rotary and head.
    # Both toolkit roles here are ones it leaves eager — the gated-delta-net blocks' gated norm,
    # applied per head on the attention output of three layers in every four, and (MoE only) the
    # shared-expert MLP. The dense spec declares no `glu_mlp`: upstream's dense applier class-swaps
    # `Qwen3_5MLP` itself, which the patch-time guard in the builder refuses to stack onto.
    LigerFamilySpec(
        model_types=("qwen3_5", "qwen3_5_text"),
        modeling_module="transformers.models.qwen3_5.modeling_qwen3_5",
        gated_rms_norm=("Qwen3_5RMSNormGated",),
        delegates_to_upstream=True,
    ),
    # `Qwen3_5MoeMLP` is the sigmoid-gated shared expert (the gate lives in the block, so the MLP is
    # the canonical GLU body) and the EP wrapper adopts it unchanged; upstream's class-level `swiglu`
    # sets only `Qwen3_5MoeExperts`, which that wrapper replaces. Its instance branch, run when HF
    # Trainer re-applies Liger, binds a kernel-equivalent SwiGLU forward over this class.
    LigerFamilySpec(
        model_types=("qwen3_5_moe", "qwen3_5_moe_text"),
        modeling_module="transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
        gated_rms_norm=("Qwen3_5MoeRMSNormGated",),
        glu_mlp=("Qwen3_5MoeMLP",),
        delegates_to_upstream=True,
    ),
    # `Qwen3NextMLP` serves both the dense layers and every sparse block's shared expert. The family
    # has no EP wrapper, so upstream's `Qwen3NextExperts` swap is its live routed-expert path.
    LigerFamilySpec(
        model_types=("qwen3_next",),
        modeling_module="transformers.models.qwen3_next.modeling_qwen3_next",
        gated_rms_norm=("Qwen3NextRMSNormGated",),
        glu_mlp=("Qwen3NextMLP",),
        delegates_to_upstream=True,
    ),
    # GptOss: upstream owns the rotary (half-width cos/sin, algebraically `rotate_half`) and the head,
    # but applies the llama-cast `LigerRMSNorm` to a norm that multiplies its weight in fp32 before
    # the cast back — Gemma's casting mode — so the norm role is taken over. Its `swiglu` is a no-op
    # upstream (no patch block) and the EP layer runs the clamped GLU through its own fused kernel.
    LigerFamilySpec(
        model_types=("gpt_oss",),
        modeling_module="transformers.models.gpt_oss.modeling_gpt_oss",
        rms_norm=("GptOssRMSNorm",),
        rms_norm_casting_mode="gemma",
        delegates_to_upstream=True,
        upstream_off=("rms_norm",),
    ),
    # Gemma 4: upstream owns the `(1 + w)`-free Gemma-cast norms, the rotary (off, single-tensor
    # signature) and the head; `Gemma4TextMLP` is the dense MLP every decoder layer keeps beside its
    # experts, so the EP wrapper never replaces it. Upstream's `geglu` swaps that class for a
    # tanh-GeGLU that never checks the activation; the toolkit's probes it and survives the wrap.
    # `gemma4` wrappers resolve here through their text tower.
    LigerFamilySpec(
        model_types=("gemma4_text",),
        modeling_module="transformers.models.gemma4.modeling_gemma4",
        glu_mlp=("Gemma4TextMLP",),
        delegates_to_upstream=True,
        upstream_off=("geglu",),
    ),
    # Inkling. `InklingMLP` scales its output by a trained `global_scale`, so the fused GLU would
    # drop a parameter; positions enter as a learned relative-logit bias, so there is no rotary.
    LigerFamilySpec(
        model_types=("inkling_text", "inkling_mm_model"),
        modeling_module="transformers.models.inkling.modeling_inkling",
        rms_norm=("InklingRMSNorm",),
    ),
    # LFM-2, dense and MoE. The only families on the roster whose rotary is Liger's (full head_dim,
    # `rotate_half` over concatenated halves). Their MLPs project through `w1`/`w3`/`w2` with a
    # hardcoded `F.silu` and no `act_fn`, which the fused-GLU forward does not address.
    LigerFamilySpec(
        model_types=("lfm2",),
        modeling_module="transformers.models.lfm2.modeling_lfm2",
        rms_norm=("Lfm2RMSNorm",),
        causal_lm=("Lfm2ForCausalLM",),
        rope=True,
    ),
    LigerFamilySpec(
        model_types=("lfm2_moe",),
        modeling_module="transformers.models.lfm2_moe.modeling_lfm2_moe",
        rms_norm=("Lfm2MoeRMSNorm",),
        causal_lm=("Lfm2MoeForCausalLM",),
        rope=True,
    ),
    # Cohere 2, dense and MoE. The dense family's only norm is `Cohere2LayerNorm` (mean-subtracting,
    # no bias parameter), which LigerRMSNorm cannot express and LigerLayerNorm would need a bias
    # materialized for; the MoE family builds the same `Cohere2MoeLayerNorm` when `rms_norm_eps` is
    # null and a llama-style `Cohere2MoeRMSNorm` when it is set. The rotary is GPT-J-interleaved with
    # `repeat_interleave`d cos/sin, applied on the sliding layers and the dense-prefix layers
    # `force_rope` selects. `logit_scale` rides through the fused loss on the hidden states.
    LigerFamilySpec(
        model_types=("cohere2",),
        modeling_module="transformers.models.cohere2.modeling_cohere2",
        glu_mlp=("Cohere2MLP",),
        causal_lm=("Cohere2ForCausalLM",),
        logit_scale_attr="logit_scale",
    ),
    LigerFamilySpec(
        model_types=("cohere2_moe",),
        modeling_module="transformers.models.cohere2_moe.modeling_cohere2_moe",
        rms_norm=("Cohere2MoeRMSNorm",),
        glu_mlp=("Cohere2MoeMLP",),
        causal_lm=("Cohere2MoeForCausalLM",),
        logit_scale_attr="logit_scale",
    ),
    # Step-3.7 Flash. Gemma-style norm: fp32 statistics, `(1 + w)` scale, zero-init weight.
    # `Step3p7MLP` clamps the activated gate and the up half on its last layers.
    LigerFamilySpec(
        model_types=("step3p7", "step3p5"),
        modeling_module="transformers.models.step3p7.modeling_step3p7",
        rms_norm=("Step3p7RMSNorm",),
        rms_norm_casting_mode="gemma",
        rms_norm_offset=1.0,
    ),
    # Ling / Ring 2.0 (remote code). One spec for both: the linear-attention variant reuses the V2
    # class names for everything it shares. `BailingMoeV2GroupRMSNorm` is deliberately absent — it
    # normalizes over `hidden_size // group_norm_size`, so LigerRMSNorm would reduce the wrong axis.
    # No fused loss: the family's head adds an MTP term whenever `num_nextn_predict_layers > 0`,
    # which a spec role cannot gate on (the shipped checkpoints set it to 0).
    LigerFamilySpec(
        model_types=("bailing_moe", "bailing_moe_linear"),
        remote_classes=("BailingMoeV2RMSNorm", "BailingMoeV2MLP"),
        rms_norm=("BailingMoeV2RMSNorm",),
        glu_mlp=("BailingMoeV2MLP",),
    ),
    # Ling 3.0 (remote code). Its KDA layers already run fla's fused gated norm, short convolutions
    # and delta-rule recurrence, so only the attention/MoE norms and the shared-expert GLU are left.
    LigerFamilySpec(
        model_types=("bailing_hybrid",),
        remote_classes=("BailingMoeV3RMSNorm", "BailingMoeV3MLP"),
        rms_norm=("BailingMoeV3RMSNorm",),
        glu_mlp=("BailingMoeV3MLP",),
    ),
)
