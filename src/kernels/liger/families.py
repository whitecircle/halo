"""Per-family Liger coverage — the declarative half of the toolkit's applier registry.

A role is listed only where the kernel reproduces the family's own forward bit-for-bit. Anything else (a
clamped GLU, a gated or grouped norm, a scaled MLP output, a rotary that is not Liger's) is left off,
with the reason recorded here: a different activation or normalization changes the model, not just its
speed. A family upstream Liger covers sets ``delegates_to_upstream`` and names only the roles it adds.
"""

from __future__ import annotations

from src.kernels.liger.builder import LigerFamilySpec

# Multimodal wrappers are deliberately not listed: the orchestrator resolves them through
# `text_config.model_type`, where the decoder classes these specs name live, so a wrapper whose text
# tower can be the dense or the MoE sibling (`lfm2_vl`, `cohere2_vision`) is not pinned to one.
#
# Of the families the toolkit patches end to end, three declare no `causal_lm`: GLM-5 Next and
# Step-3.7 head their checkpoints with `*ForConditionalGeneration` reading
# `config.text_config.vocab_size`, and Inkling divides the hidden states by
# `logits_mup_width_multiplier` and truncates the logits to `unpadded_vocab_size` before the loss.
# None of that is what the generic fused loss computes, so they keep the unfused head.
LIGER_FAMILY_SPECS: tuple[LigerFamilySpec, ...] = (
    # Mistral 4 (and the mistral3 VLM built on it). Rotary is interleaved YARN + llama-4 log scaling.
    LigerFamilySpec(
        model_types=("mistral4", "mistral3"),
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
    # DeepSeek-V4. No RMSNorm: V4 mixes weighted `DeepseekV4RMSNorm` with the weightless
    # `DeepseekV4UnweightedRMSNorm`, and several norms are pinned fp32 by
    # `_keep_in_fp32_modules_strict`. No GLU: the experts run a *clamped* SwiGLU (`swiglu_limit`).
    # Rotary is interleaved partial with per-rope-type buffers.
    LigerFamilySpec(
        model_types=("deepseek_v4",),
        modeling_module="transformers.models.deepseek_v4.modeling_deepseek_v4",
        causal_lm=("DeepseekV4ForCausalLM",),
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
    # the sliding ones, through one shared function.
    LigerFamilySpec(
        model_types=("laguna",),
        modeling_module="transformers.models.laguna.modeling_laguna",
        rms_norm=("LagunaRMSNorm",),
        glu_mlp=("LagunaMLP",),
        causal_lm=("LagunaForCausalLM",),
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
    # the canonical GLU body) and the EP wrapper adopts it unchanged; upstream's `swiglu` sets only
    # `Qwen3_5MoeExperts`, which that wrapper replaces.
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
    # Cohere 2, dense and MoE. No norm: the live class is `Cohere2*LayerNorm` (mean-subtracting, no
    # bias parameter), which LigerRMSNorm cannot express and LigerLayerNorm would need a bias
    # materialized for. The rotary is GPT-J-interleaved with `repeat_interleave`d cos/sin, and only
    # the sliding layers use it. `logit_scale` rides through the fused loss on the hidden states.
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
    # No fused loss: the family's head adds an MTP term the fused path does not compute.
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
