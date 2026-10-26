"""Per-family Liger coverage, the declarative half of the toolkit's applier registry.

A role is listed only where the kernel reproduces the family's own forward bit-for-bit; anything else is
left off with the reason recorded here, since a different activation or normalization changes the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

# The upstream flags a delegating spec may withhold because it fills that role itself, each mapped to
# the spec field that must then be non-empty.
_WITHHOLDABLE_UPSTREAM_ROLES = {"rms_norm": "rms_norm", "swiglu": "glu_mlp", "geglu": "glu_mlp"}

_RMS_NORM_CASTING_MODES = ("llama", "gemma")

_RmsNormKernel = Literal["liger", "native"]
_RMS_NORM_KERNELS = get_args(_RmsNormKernel)


@dataclass(frozen=True)
class LigerFamilySpec:
    """The Liger-patchable surface of one model family, by role.

    ``modeling_module`` is a family's in-library module; ``remote_classes`` names the classes that identify
    a ``trust_remote_code`` modeling module, and a native family whose hub repos also ship one sets both.
    A family upstream Liger already covers sets ``delegates_to_upstream``: upstream's applier keeps every
    role it declares, and the spec names only the roles the toolkit adds, or takes over via ``upstream_off``.
    """

    model_types: tuple[str, ...]
    modeling_module: str = ""
    # Norm classes whose forward is exactly Liger's `(offset + w) * x * rsqrt(mean(x²) + eps)`.
    # A family's other norms (weightless, grouped, gated, mean-subtracting) must not be listed:
    # LigerRMSNorm cannot express them and the swap would change the model's numerics.
    rms_norm: tuple[str, ...] = ()
    # "llama" upcasts x to fp32 and casts back before the weight multiply; "gemma" keeps the weight
    # multiply in fp32. `offset` is the constant added to the weight (Gemma-style `(1 + w)`).
    rms_norm_casting_mode: str = "llama"
    rms_norm_offset: float = 0.0
    # "liger" swaps in LigerRMSNorm; "native" swaps in torch's fused `F.rms_norm` under the same casting
    # mode (offset 0 only), which also serves weightless norms and is several times cheaper to launch and
    # to run.
    rms_norm_kernel: _RmsNormKernel = "liger"
    # Gated norm classes: `norm(x) * weight * act(gate)` over the last dim, served by `fla`'s fused kernel
    # under the same `rms_norm` flag. A grouped gated norm is a different function and must not be listed.
    gated_rms_norm: tuple[str, ...] = ()
    # MLP classes whose forward is exactly `down(act(gate(x)) * up(x))` over `gate_proj`/`up_proj`/
    # `down_proj`. A clamped, scaled or differently-named GLU must not be listed.
    glu_mlp: tuple[str, ...] = ()
    # `*ForCausalLM` classes whose forward the generic fused loss reproduces exactly.
    causal_lm: tuple[str, ...] = ()
    # Whether the family's head adds the router aux loss after the projection the fused loss
    # replaces. The fused forward then refuses `output_router_logits`, since nothing would add that
    # term to the objective; a head that only forwards the flag passes it through instead.
    router_aux_loss_in_head: bool = False
    # Whether Liger's generic rotary computes this family's `apply_rotary_pos_emb`. False for every
    # partial, interleaved, mrope, YARN or per-layer-type rotary.
    rope: bool = False
    # True mirrors Liger's own loss default (FLCE on, CE off), for a vocab whose logits plane binds memory.
    flce_default: bool = False
    # The class names that identify a `trust_remote_code` modeling module once transformers loads it,
    # every class a role names included. Empty for a family that only loads from transformers.
    remote_classes: tuple[str, ...] = ()
    # Upstream Liger covers these model types: its applier runs first, and the roles above are added on top.
    delegates_to_upstream: bool = False
    # Upstream flags a delegating spec passes as False because it fills that role itself, where upstream's
    # variant is not the family's function or does not survive the EP wrapper.
    upstream_off: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not (self.modeling_module or self.remote_classes):
            raise ValueError(
                f"LigerFamilySpec for {self.model_types} names no module to patch: set modeling_module "
                f"(in-library), remote_classes (trust_remote_code), or both."
            )
        # The remote patch runs only on a module defining every identifying class, which is what lets
        # a revision that renamed one be reported rather than fail its model load.
        unidentified = sorted(
            set(self.rms_norm + self.gated_rms_norm + self.glu_mlp + self.causal_lm) - set(self.remote_classes)
        )
        if self.remote_classes and unidentified:
            raise ValueError(
                f"LigerFamilySpec for {self.model_types} patches {unidentified} on the remote module "
                f"without listing them in remote_classes."
            )
        if self.router_aux_loss_in_head and not self.causal_lm:
            raise ValueError(f"LigerFamilySpec for {self.model_types} declares an in-head aux loss but no causal_lm")
        if self.rms_norm_casting_mode not in _RMS_NORM_CASTING_MODES:
            raise ValueError(
                f"LigerFamilySpec for {self.model_types} sets rms_norm_casting_mode="
                f"{self.rms_norm_casting_mode!r}; expected one of {_RMS_NORM_CASTING_MODES}"
            )
        if self.rms_norm_kernel not in _RMS_NORM_KERNELS:
            raise ValueError(
                f"LigerFamilySpec for {self.model_types} sets rms_norm_kernel={self.rms_norm_kernel!r}; "
                f"expected one of {_RMS_NORM_KERNELS}"
            )
        if self.rms_norm_kernel == "native" and self.rms_norm_offset:
            raise ValueError(
                f"LigerFamilySpec for {self.model_types} pairs the native RMSNorm with offset "
                f"{self.rms_norm_offset}; the native kernel implements offset 0 only"
            )
        if self.upstream_off and not self.delegates_to_upstream:
            raise ValueError(f"LigerFamilySpec for {self.model_types} withholds upstream flags but does not delegate")
        for flag in self.upstream_off:
            role = _WITHHOLDABLE_UPSTREAM_ROLES.get(flag)
            if role is None or not getattr(self, role):
                raise ValueError(
                    f"LigerFamilySpec for {self.model_types} withholds {flag!r} from upstream without "
                    f"filling that role itself; only {sorted(_WITHHOLDABLE_UPSTREAM_ROLES)} can be taken over."
                )
        # A role upstream also claims is caught at patch time, on the class upstream replaced.
        if self.delegates_to_upstream and not (
            self.modeling_module and (self.rms_norm or self.gated_rms_norm or self.glu_mlp or self.causal_lm)
        ):
            raise ValueError(
                f"LigerFamilySpec for {self.model_types} delegates to upstream Liger but adds no role "
                f"of its own to a native modeling module; drop it and resolve on the upstream registry."
            )


# A multimodal wrapper is listed only where its text tower can be nothing but the family named here; one
# whose tower may be either of two siblings resolves through `text_config.model_type` instead. GLM-5 Next
# and Step-3.7 define no `*ForCausalLM`, and Inkling's head truncates the vocabulary, so they keep the
# unfused head.
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
    # DeepSeek-V4. No RMSNorm: no parity test covers the swap. The loaders cast the fp32-pinned
    # `DeepseekV4RMSNorm` weights to the run dtype, where the eager norm returns the input dtype as
    # Liger's kernel does. No GLU: the experts run a *clamped* SwiGLU (`swiglu_limit`). Rotary is interleaved
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
        rms_norm_kernel="native",
        glu_mlp=("Glm4MoeLiteMLP",),
        causal_lm=("Glm4MoeLiteForCausalLM",),
        flce_default=True,
    ),
    # Laguna. `LagunaMLP` serves both the dense layers and every block's shared expert, which the EP
    # wrapper adopts unchanged. Rotary is half-width on the full-attention layers and full-width on
    # the sliding ones, through one shared function. The head adds the router aux loss after the
    # projection. The released repos also ship their own modeling file through `auto_map`, whose norm,
    # MLP and head are the in-library classes verbatim, so a `trust_remote_code` load takes the same swaps.
    LigerFamilySpec(
        model_types=("laguna",),
        modeling_module="transformers.models.laguna.modeling_laguna",
        rms_norm=("LagunaRMSNorm",),
        glu_mlp=("LagunaMLP",),
        causal_lm=("LagunaForCausalLM",),
        router_aux_loss_in_head=True,
        remote_classes=("LagunaRMSNorm", "LagunaMLP", "LagunaForCausalLM"),
    ),
    # GLM-5.3-Flash. The two plain norms take Liger's kernel, the GDN blocks' gated norm fla's.
    # `Glm5NextTextUnweightedRMSNorm` carries no weight, `Glm5NextTextMLP` clamps gate and up at
    # `swiglu_limit`, and the text tower is NoPE.
    LigerFamilySpec(
        model_types=("glm5_next", "glm5_next_text"),
        modeling_module="transformers.models.glm5_next.modeling_glm5_next",
        rms_norm=("Glm5NextTextRMSNorm", "Glm5NextRMSNorm"),
        gated_rms_norm=("Glm5NextTextRMSNormGated",),
    ),
    # Qwen3.5 / 3.6 (dense and MoE) and Qwen3-Next: upstream Liger owns their norms, rotary and head; the
    # toolkit adds the gated-delta-net blocks' gated norm and (MoE only) takes over `swiglu`. The dense
    # spec declares no `glu_mlp`: upstream's dense applier class-swaps `Qwen3_5MLP` itself.
    LigerFamilySpec(
        model_types=("qwen3_5", "qwen3_5_text"),
        modeling_module="transformers.models.qwen3_5.modeling_qwen3_5",
        gated_rms_norm=("Qwen3_5RMSNormGated",),
        delegates_to_upstream=True,
    ),
    # Upstream's `swiglu` would also put `LigerExperts` in place of the routed experts, which the EP
    # wrapper or the model's own experts implementation must run instead. Withheld, so the flag
    # fuses only `Qwen3_5MoeMLP`, the sigmoid-gated shared expert (the gate lives in the block, so
    # the MLP is the canonical GLU body), which the EP wrapper adopts unchanged.
    LigerFamilySpec(
        model_types=("qwen3_5_moe", "qwen3_5_moe_text"),
        modeling_module="transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
        gated_rms_norm=("Qwen3_5MoeRMSNormGated",),
        glu_mlp=("Qwen3_5MoeMLP",),
        delegates_to_upstream=True,
        upstream_off=("swiglu",),
    ),
    # `Qwen3NextMLP` serves both the dense layers and every sparse block's shared expert. `swiglu`
    # is withheld from upstream as above; the family has no EP wrapper, so its routed experts run
    # the model's own experts implementation.
    LigerFamilySpec(
        model_types=("qwen3_next",),
        modeling_module="transformers.models.qwen3_next.modeling_qwen3_next",
        gated_rms_norm=("Qwen3NextRMSNormGated",),
        glu_mlp=("Qwen3NextMLP",),
        delegates_to_upstream=True,
        upstream_off=("swiglu",),
    ),
    # Qwen3 MoE: upstream owns every role but the norms, which take torch's fused kernel in the llama mode
    # (a launch-bound kernel at these shapes, cheaper per call than LigerRMSNorm).
    LigerFamilySpec(
        model_types=("qwen3_moe",),
        modeling_module="transformers.models.qwen3_moe.modeling_qwen3_moe",
        rms_norm=("Qwen3MoeRMSNorm",),
        rms_norm_kernel="native",
        delegates_to_upstream=True,
        upstream_off=("rms_norm",),
    ),
    # GptOss: upstream owns the rotary (half-width cos/sin, algebraically `rotate_half`) and the head,
    # but applies the llama-cast `LigerRMSNorm` to a norm that multiplies its weight in fp32 before
    # the cast back — Gemma's casting mode — so the norm role is taken over, on torch's fused kernel in
    # that mode. Its `swiglu` is a no-op upstream (no patch block) and the EP layer runs the clamped GLU
    # through its own fused kernel.
    LigerFamilySpec(
        model_types=("gpt_oss",),
        modeling_module="transformers.models.gpt_oss.modeling_gpt_oss",
        rms_norm=("GptOssRMSNorm",),
        rms_norm_casting_mode="gemma",
        rms_norm_kernel="native",
        delegates_to_upstream=True,
        upstream_off=("rms_norm",),
    ),
    # Gemma 4: the norms (fp32 normalize and weight multiply, some weightless) take torch's fused kernel.
    # Upstream owns the rotary (off) and the head. `geglu` is taken over for `Gemma4TextMLP`, the dense
    # MLP beside every layer's experts: upstream's swap never checks the activation.
    LigerFamilySpec(
        model_types=("gemma4_text",),
        modeling_module="transformers.models.gemma4.modeling_gemma4",
        rms_norm=("Gemma4RMSNorm",),
        rms_norm_casting_mode="gemma",
        rms_norm_kernel="native",
        glu_mlp=("Gemma4TextMLP",),
        delegates_to_upstream=True,
        upstream_off=("geglu", "rms_norm"),
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
    # `force_rope` selects.
    LigerFamilySpec(
        model_types=("cohere2",),
        modeling_module="transformers.models.cohere2.modeling_cohere2",
        glu_mlp=("Cohere2MLP",),
        causal_lm=("Cohere2ForCausalLM",),
    ),
    LigerFamilySpec(
        model_types=("cohere2_moe",),
        modeling_module="transformers.models.cohere2_moe.modeling_cohere2_moe",
        rms_norm=("Cohere2MoeRMSNorm",),
        glu_mlp=("Cohere2MoeMLP",),
        causal_lm=("Cohere2MoeForCausalLM",),
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
    # and delta-rule recurrence, so only the attention/MoE norms and the GLU MLPs are left.
    LigerFamilySpec(
        model_types=("bailing_hybrid",),
        remote_classes=("BailingMoeV3RMSNorm", "BailingMoeV3MLP"),
        rms_norm=("BailingMoeV3RMSNorm",),
        glu_mlp=("BailingMoeV3MLP",),
    ),
)
