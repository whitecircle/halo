# Liger Kernels

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) provides Triton replacements for RMSNorm,
cross-entropy, SwiGLU, and RoPE. Enabled by default (`use_liger_kernel: true`); set it `false` to disable.
A required dependency, pinned `liger-kernel>=0.8.4,<0.9.0` (`uv.lock` resolves 0.8.4). A model type no
applier covers **warns** that the run trains unfused, and an explicit `liger_kernel_config` key on such a
model raises rather than being dropped.

Why fusion pays, counted in HBM round trips:
[GPU Training Theory §5](../reference/gpu-training-theory.md#5-memory-bound-kernels-and-why-fusion-pays).

## Supported models

The dispatch (`halo/model_loading/liger/dispatch.py`) resolves one applier per `model_type`: the toolkit
registry for the `model_type` or the wrapper's `text_config.model_type`, then upstream Liger's
`MODEL_TYPE_TO_APPLY_LIGER_FN` in the same order. A multimodal wrapper whose text tower a toolkit spec covers
resolves to that spec even where upstream registers the wrapper (`gemma4`), so upstream's wrapper applier never
runs its own text-tower swaps over the roles the spec takes over.

The toolkit registry is built from one declarative spec per family (`halo/model_loading/liger/family_specs.py`): a spec
names the classes filling each role — RMSNorm, GLU MLP, causal-LM head — and only where the kernel computes
the family's function **exactly**. The applier (`halo/model_loading/liger/applier.py`) patches a role by subclassing the family's own class,
preserving its constructor signature and class name.

Upstream Liger 0.8 covers Llama, Qwen2/3, Qwen3 MoE, Qwen3.5/3.6, Mistral, Mixtral, Gemma 2/3/4, GLM-4
(incl. 4V MoE), OLMo 2/3, GptOss, Qwen3-Next, DeepSeek-V3/V4 and VLM wrappers; `qwen3_moe`, `qwen3_5*`,
`qwen3_next`, `gpt_oss` and `gemma4_text` are extended or corrected by a
[delegating spec](#upstream-covered-families-the-toolkit-extends).

Toolkit-covered families. Upstream has none of them except DeepSeek-V4, whose entry the toolkit spec replaces
outright (upstream's applier swaps the RMSNorm the spec leaves eager). ✅ = patched, — = left unfused, with the
reason:

| Model | model_type | RMSNorm | GLU MLP | RoPE | CE | FLCE |
|---|---|:--:|:--:|:--:|:--:|:--:|
| Mistral 4 | `mistral4` (a `mistral3` wrapper resolves through its text tower: `mistral4` here, `mistral` upstream) | ✅ | ✅ | — interleaved YARN; the llama-4 log scale follows it | ✅ | ✅ text-only checkpoints; [forced off under the wrapper](#fused-loss-under-a-multimodal-wrapper) |
| Zaya | `zaya` | ✅ | — EP layer owns the experts | — partial rotary | ✅ | ✅ **default** |
| DeepSeek-V4 | `deepseek_v4` | — no parity test covers the swap; the loaders cast the fp32-pinned weights to the run dtype, where the eager norm matches the kernel's output dtype | — clamped SwiGLU | — interleaved partial | ✅ | ✅ **default** |
| GLM-4.7-Flash | `glm4_moe_lite` | ✅ torch's fused `F.rms_norm` in the llama casting mode (`rms_norm_kernel="native"`) | ✅ | — dual interleave/plain MLA | ✅ | ✅ **default** |
| Laguna | `laguna` | ✅ | ✅ | — half-width on full-attention layers, full on sliding | ✅ | ✅ |
| Qwen3.8-Flash-Next | `qwen4_exp_text` (the `qwen4_exp` wrapper resolves here) | — one class also runs the hyper-connection norms grouped per stream | — the EP layer owns the experts; the shared expert stays eager | — interleaved mrope | ✅ | ✅ text-only; [forced off under the wrapper](#fused-loss-under-a-multimodal-wrapper) |
| Nemotron-H | `nemotron_h` (the `nemotron_h_omni` composite resolves here) | ✅ torch's fused `F.rms_norm` in the Gemma casting mode (`rms_norm_kernel="native"`); not `NemotronHRMSNormGated`, grouped per Mamba group | — no gate half on the experts or `NemotronHMLP` | — its attention takes no rotary | ✅ | ✅ **default**, text-only; [forced off under the composite](#fused-loss-under-a-multimodal-wrapper) |
| GLM-5.3-Flash | `glm5_next`, `glm5_next_text` | ✅ the two plain norms **+ the GDN gated norm** (fla) | — clamped at `swiglu_limit` | — NoPE text tower | ✅ | — no `*ForCausalLM`; the `*ForConditionalGeneration` head adds the router aux loss after the projection |
| Inkling | `inkling_text`, `inkling_mm_model` | ✅ | — trained `global_scale` on the MLP output | — no rotary (learned relative bias) | ✅ | — head divides by `logits_mup_width_multiplier` and truncates to `unpadded_vocab_size` |
| LFM-2 / LFM-2 MoE | `lfm2`, `lfm2_moe` | ✅ | — `w1`/`w3`/`w2`, no `act_fn` | ✅ **the only family whose rotary is Liger's** | ✅ | ✅ text-only; [forced off under `lfm2_vl`](#fused-loss-under-a-multimodal-wrapper) |
| Cohere 2 / Cohere 2 MoE | `cohere2`, `cohere2_moe` | — dense: a mean-subtracting `LayerNorm` with no bias parameter; ✅ MoE `Cohere2MoeRMSNorm`, the live class when `rms_norm_eps` is set (`null` builds the LayerNorm) | ✅ | — GPT-J interleaved, on the sliding and `force_rope` layers | ✅ | ✅ text-only (`logit_scale` folded onto the hidden states); [forced off under `cohere2_vision`](#fused-loss-under-a-multimodal-wrapper) |
| Step-3.7 Flash | `step3p7`, `step3p5` | ✅ Gemma-style (`offset=1.0`, fp32 weight) | — clamps the activated gate | — partial rotary + llama3 scaling | ✅ | — no `*ForCausalLM`; the `*ForConditionalGeneration` head is the only one |
| Ling / Ring 2.0 | `bailing_moe`, `bailing_moe_linear` | ✅ | ✅ dense and shared expert, and the [routed experts](#routed-experts) wherever no EP layer replaces them | — partial rotary | ✅ | — the head computes an MTP term whenever `num_nextn_predict_layers > 0`, weighted by `mtp_loss_scaling_factor`, which a spec cannot gate on |
| Ling 3.0 | `bailing_hybrid` | ✅ | ✅ dense and shared expert, and the [routed experts](#routed-experts) wherever no EP layer replaces them | — interleaved MLA + YARN | ✅ | — same MTP term |

The Bailing families are `trust_remote_code`, so their modeling module does not exist when Liger is applied.
Their patch is registered on the shared hook over `transformers.dynamic_module_utils.get_class_in_module`
(`halo/model_loading/remote_code.py`, wrapped once for every registrant), the funnel every remote
class loads through, and fires on the module defining the declared classes
(`halo/model_loading/liger/applier.py`). Laguna is native to transformers, but its released repos also ship
their own modeling file through `auto_map`, which the shipped recipes load; its spec names that file's
classes too, so either load takes the same swaps.

Ling 3.0's KDA layers already run `fla`'s fused gated norm, short convolutions and delta-rule recurrence;
only the attention/MoE norms and the GLU MLPs are left to fuse.

**Gated norms go to `fla`, not Liger.** A linear-attention (GDN) block applies
`norm(x) * weight * act(gate)` to its attention output, per head: GLM-5.3-Flash in 34 of 45 layers with a
sigmoid gate, Qwen3.5/3.6 and Qwen3-Next in three of every four with a SiLU one.

Liger has no kernel for it, and the `@use_kernel_forward_from_hub("RMSNormGated")` route the classes
advertise is inert here. The spec's `gated_rms_norm` role therefore binds `flash-linear-attention`'s
`FusedRMSNormGated`, already a hard dependency for this roster's delta-rule and short-convolution kernels.
It rides the same `rms_norm` flag, so one knob turns both norm kernels off.

Measured on a B300 over `[1, 8192, 32, 128]` (262144 rows × 128) forward and backward (Blackwell
image, Halo 1.1.0 or earlier): **2.7–6.3×** (1.79–1.98 ms eager → 0.32–0.66 ms fused).

The kernel keeps the reduction, the weight multiply and the gate in fp32, as GLM-5's module does. The Qwen
modules round the normalized activation to storage dtype *before* the weight multiply, so the fused path is
the **more accurate** of the two: worst error 3.3e-3 on `dx` against an fp64 oracle, 0.37–1.00× the eager
module's own (`tests/gpu/kernels/test_liger_family_kernels.py` pins that ratio).

The gate activation string is the family's own and is refused unless `fla` implements it; its kernel
dispatches on the string with no else branch, so an unknown gate would apply none at all.

A **grouped** gated norm is a different function and is deliberately uncovered: Bailing's `*GroupRMSNorm`
reduces over `hidden_size // group_norm_size`, where a last-dim kernel would normalize the wrong axis and
still return the right shape.

**Cross-entropy is scoped, not process-wide — for every family.** The toolkit's CE patch overrides
`transformers.loss.loss_utils`' view of `nn`, its single CE call site, leaving `F.cross_entropy` and every
3-D or soft-target caller untouched (the classification trainer's focal loss calls it twice on one logits
tensor, which Liger's in-place kernel would corrupt).

Upstream Liger's own appliers install CE by rebinding `F.cross_entropy` globally, so the dispatch never
asks them for it: an upstream-resolved family is applied with `cross_entropy=False` and gets the scoped
patch instead, and a delegating spec withholds the flag from upstream the same way. The effective config
still records `cross_entropy: true`; it is the same fused kernel at the same call site.

`layer_norm` is forced off on every upstream applier that declares it, an explicit `true` included (warned), for
the same reason: called before the model is built, upstream's VLM appliers (Gemma 3, PaliGemma, Mllama, InternVL,
SmolVLM) install it by rebinding `torch.nn.LayerNorm`, the class every model the process builds afterwards would
construct. Their vision towers keep torch's LayerNorm, the effective config records `layer_norm: false`, and an
application that still leaves `torch.nn.LayerNorm` rebound raises. No toolkit spec declares the key.

**One fused head for the roster.** Every family declaring a fused loss installs the same forward
(`halo/model_loading/liger/cross_entropy.py`), which applies the head transform the family's `HeadTransformSpec`
declares ([Adding a Model](../models/adding-a-model.md#declare-the-head-transform)): a logit scale (Cohere's
`logit_scale`) folds onto the hidden states. It refuses the shapes it cannot reproduce: a softcap or a vocabulary
cut, an `lm_head` bias, and `output_router_logits` on a family whose head adds the router aux loss *after* the
projection the fused loss replaces (DeepSeek-V4, Laguna, Qwen3.8-Flash-Next — declared `router_aux_loss_in_head`).
A head that casts its logits to fp32 (Nemotron-H, declared `fp32_logits`) keeps the cast on a forward with no labels, so
a log-softmax over them runs in the dtype it does unfused.

There the term would silently leave the objective, so use `moe_balancing: bias_update`, or
`bias_update_transient` on Qwen3.8-Flash-Next, which has no bias slot. A head that only
forwards the flag (Zaya, Mistral 4, LFM-2, Cohere 2) passes it through to the backbone, whose router logits
reach the output as they do unfused.

### Fused loss under a multimodal wrapper

A checkpoint whose `model_type` is a multimodal wrapper — `gemma4`, `lfm2_vl`, `cohere2_vision`, `mistral3`
(every Mistral Small 3.x / 4 checkpoint), `qwen4_exp` — loads as its `*ForConditionalGeneration` class, whose own forward
runs `lm_head` and the loss. The applier resolved through `text_config.model_type` patches the text tower's
`*ForCausalLM`, which that model never instantiates. `nemotron_h_omni` does instantiate it, but calls it with no labels
and scores the loss itself, so the fused path never runs there either.

A fused loss there would report as applied while the logits plane materializes anyway, and trade away the
CE kernel that does reach the decoder. The dispatch therefore forces `fused_linear_cross_entropy` off
whenever it resolved through the text config, warning if the config requested it, and keeps `cross_entropy`
on.

RMSNorm, the fused GLU, RoPE and CE all reach the decoder as usual. For the fused head, train a text-only
checkpoint (`model_type` = the text tower's), or load a wrapper checkpoint that admits it text-only
(`text_only_model: true`, as `qwen4_exp` does): `AutoModelForCausalLM` then builds the text tower's `*ForCausalLM`
on the text sub-config, which is the config Liger is applied to, so the fused loss engages.

A prefix LM (PaliGemma, PaliGemma 2) resolves upstream's own `paligemma` applier, whose fused loss replaces
`PaliGemmaForConditionalGeneration.forward` with one that calls the decoder directly, so the prefix-LM mask
transformers builds from `token_type_ids` is never built (against transformers 5.20 the replacement raises at its
first forward). `liger_prefix_lm_overrides` forces `fused_linear_cross_entropy` off there at both application
sites, warning on a request, and `cross_entropy` serves the loss.

**Not covered:** `glm4_moe` (full GLM-4.5/4.7) has no EP layer, doc page or example in this toolkit, so it
has no spec either.

### Upstream-covered families the toolkit extends

Qwen3 MoE, Qwen3.5 / 3.6, Qwen3-Next, GptOss and Gemma 4 resolve to a toolkit applier that **delegates**: upstream
Liger's applier runs first with every flag it declares, and the spec adds the roles it leaves eager, or takes
over (`upstream_off`) a role upstream gets wrong for the family.

| Model | model_type | Upstream applier patches | Toolkit spec adds or takes over |
|---|---|---|---|
| Qwen3 MoE | `qwen3_moe` | RoPE, SwiGLU (`Qwen3MoeMLP`, `Qwen3MoeExperts`), FLCE | **takes over RMSNorm** with torch's fused `F.rms_norm` in the llama casting mode (`rms_norm_kernel="native"`: fp32 normalize, cast back, weight multiply in the activation dtype, as `Qwen3MoeRMSNorm` does) |
| Qwen3.5 / 3.6 dense | `qwen3_5`, `qwen3_5_text` | RMSNorm, `Qwen3_5MLP`, FLCE | GDN gated norm → `fla` |
| Qwen3.5 / 3.6 MoE | `qwen3_5_moe`, `qwen3_5_moe_text` | RMSNorm, FLCE | GDN gated norm → `fla`; **takes over `swiglu`**: shared-expert `Qwen3_5MoeMLP`, withholding upstream's [routed-expert swap](#routed-experts) |
| Qwen3-Next | `qwen3_next` | RMSNorm, FLCE | GDN gated norm → `fla`; **takes over `swiglu`**: dense + shared-expert `Qwen3NextMLP`, withholding upstream's [routed-expert swap](#routed-experts) |
| GptOss | `gpt_oss` | RoPE, FLCE | **takes over RMSNorm** with torch's fused `F.rms_norm` (`rms_norm_kernel="native"`): `GptOssRMSNorm` multiplies its weight in fp32 before the cast back (Gemma's casting mode); upstream applies the llama-cast `LigerRMSNorm`, a bf16-ULP deviation on every norm |
| Gemma 4 | `gemma4_text` (the `gemma4` wrapper resolves here) | FLCE | **takes over RMSNorm** with torch's fused `F.rms_norm` (`rms_norm_kernel="native"`: fp32 normalize and weight multiply, one cast; covers the weightless `with_scale=False` norms too). At [2048, 2816] bf16 on one B300 it runs fwd+bwd in 51 µs against LigerRMSNorm's gemma mode at 203 µs, and launches in 75 µs against 218 µs, with the same error against fp64. **Takes over GeGLU**: `Gemma4TextMLP` is the dense MLP every decoder layer keeps beside its experts, so the EP layer never replaces it; the toolkit's fused GLU probes its activation and survives EP, where upstream's swap is forced off |

`delegates_to_upstream` makes that a build-time contract: the upstream applier is looked up in liger-kernel's
own registry (a family it stops covering fails at import), the delegating applier re-exports upstream's exact
knob surface, a withheld flag must name a role the spec fills, and a role naming a class upstream has already
replaced raises at patch time rather than stacking two swaps.

Everything the dispatch derives then follows the applier that actually **runs**: the `rope`/`swiglu`
defaults read off its signature, and the EP fused-GLU decision below.

**The shared expert is the toolkit's; the routed experts never run Liger's fused MoE kernel.** Upstream's
`swiglu` would swap the routed-experts class (`Qwen3_5MoeExperts` / `Qwen3NextExperts`) for `LigerExperts`, so
both MoE specs withhold it. The flag fuses the dense and shared-expert MLPs they name in `glu_mlp`, which every
EP layer adopts unchanged, so `swiglu` is **not** forced off under EP for them.

Measured on a 4-layer Qwen3.5-MoE under the EP layer: loss 6.95971 unfused vs 6.95981 fused (1.4e-5
relative), shared-expert gradients at cosine 0.99947 and norm ratio 1.0065
(`tests/gpu/kernels/test_liger_family_kernels.py`).

### Routed experts

Liger's fused MoE kernel, `LigerExperts`, never runs a MoE's routed experts. The EP layer runs them where one
is installed ([below](#fused-glu-under-an-ep-layer)); otherwise the model's own experts implementation does
(transformers' `grouped_mm` by default).

For Ling that implementation is fused too, by Halo rather than Liger: its remote code builds the routed experts
as an `nn.ModuleList` of `BailingMoeV2MLP` / `BailingMoeV3MLP`, the class its spec names in `glu_mlp`, so off
the EP layer the toolkit's class swap runs them on Halo's own SwiGLU kernel (`GluCombine.silu()`), the one the
EP layer runs.

Upstream's MoE appliers (Qwen3 MoE, Mixtral, Qwen3-VL MoE, GLM-4V MoE, HunYuan MoE, Qwen3.5 MoE, Qwen3-Next)
swap the routed-experts class for `LigerExperts` under `swiglu`, and two rules keep it off, since no GPU test here
validates its Blackwell input gradient against fp32 eager experts (the gradient
[Liger-Kernel#1413](https://github.com/linkedin/Liger-Kernel/pull/1413) corrected):

- The Qwen3.5-MoE and Qwen3-Next specs withhold `swiglu` from upstream and serve it on their own MLPs.
- Where Halo does not wrap a MoE's experts and the resolved applier would hand `swiglu` to upstream, the
  dispatch forces it off at load (`liger_routed_expert_overrides`), an explicit `swiglu: true` included
  (warned). That covers MoE at `ep_size: 1` with `use_grouped_gemm: false`, and a family with no EP layer
  class (Mixtral) under any configuration.

Under the force-off those experts run transformers' `grouped_mm`: as an op it is 18% slower than `LigerExperts`
at Qwen3-30B-A3B's expert shape (8192 tokens, 128 experts) on B300, and faster at small shapes (Halo 1.1.0 or earlier). The default
grouped path is unaffected.

The force-off keys on the config having experts, not on what upstream's flag patches, and drops the flag
whole: Llama 4's upstream `swiglu`, which fuses only its dense and shared-expert MLP, goes too, and GLM-4V MoE
loses upstream's dense, shared-expert and vision-MLP SwiGLU along with its `LigerExperts` swap. An applier
whose own `swiglu` default is off (GptOss, whose upstream flag patches nothing) is left alone. Under an EP
layer the upstream swap is inert and the [soft EP gate](#ep--cp--tp--pp-behavior) decides instead.

Both rules hold at both application sites, the trainer's re-application included
([below](#ep--cp--tp--pp-behavior)).

`tests/gpu/kernels/test_liger_routed_experts.py` checks the unwrapped path against fp32 eager experts.

### Fused GLU under an EP layer

Where an EP layer owns the routed experts, their activation is the toolkit's own fused Triton GLU, not
Liger's: SwiGLU (`GluCombine.silu()`) or tanh-GeGLU (`GluCombine.gelu_tanh()`), chosen by probing the layer's
`act_fn` and falling back to the family's eager combine when neither kernel computes it. It runs on the
grouped-GEMM, per-expert-loop and ETP paths alike.

One row-strided kernel pair (`halo/kernels/fused_glu.py`) serves every combine, each a `GluCombine`
(`silu`, `gelu_tanh`, `clamped_silu`, `silu_then_clamp`, `gptoss`): the activation, the clamp placement and
GptOss's `up + 1` are `tl.constexpr`, `alpha` and the bound runtime arguments. On a layer storing one fused
`[gate | up]` projection, a latched `GluCombine` runs its packed form (`GluCombine.packed`), which reads
both halves in place and writes one `[..., 2M]` gradient.

Each wrapped layer logs the executed path at construction (`grouped_mm=True, glu_combine=silu`, or
`glu_combine=eager`, as every layer reads under `HALO_FUSED_GLU=0`); `tests/gpu/kernels/test_fused_glu.py`
checks BF16 and FP32 forward/backward numerics.

## Configuration

Override individual kernels via `liger_kernel_config`:

```yaml
liger_kernel_config:
  cross_entropy: false
  fused_linear_cross_entropy: true
```

Keys: `rope`, `cross_entropy`, `fused_linear_cross_entropy`, `rms_norm`, `swiglu`, `geglu`, `layer_norm`. A key the
resolved applier's signature does not accept is dropped, with a warning naming it — that path runs eager.
`layer_norm` never applies: it is [forced off](#supported-models) wherever an applier declares it.

| Kernel | Default | Notes |
|---|---|---|
| `rope` | On | Fused rotary embedding. Auto-off whenever the resolved applier's own `rope` parameter defaults to `False` — the family's rotary (partial, mrope, YARN) has no Liger kernel. Read off the signature, not a model_type list, so a family added upstream later is covered. An explicit `rope: true` bypasses the auto-off: a toolkit-spec family whose rotary Liger does not compute refuses it with `NotImplementedError` (`halo/model_loading/liger/applier.py`), while an upstream-only applier receives the flag as-is |
| `rms_norm` | On | Fused RMS normalization. Also covers a family's gated (GDN) norm where its spec declares one — one knob, both norm kernels |
| `swiglu` | On | Auto-off where the applier's own default is `False` (GptOss, whose upstream applier has no SwiGLU patch block; Qwen3-VL), and where an **upstream** applier's expert-FFN swap is replaced by an EP layer, which runs Halo's own fused GLU combine instead. A toolkit spec patches the dense and shared-expert MLPs, which survive EP — see below. Forced off, even when requested, wherever upstream's swap would reach [routed experts Halo does not wrap](#routed-experts) |
| `geglu` | On | Gemma 4. Served by the toolkit's spec on the always-on dense `Gemma4TextMLP`, which survives the EP layer, so it is **not** forced off under EP |
| `cross_entropy` | On | Keeps logits for metrics. Always the toolkit's scoped patch. Force-off under TP, CP and PP, and defaulted off when the config explicitly requests `fused_linear_cross_entropy` (the two are mutually exclusive; setting both explicitly raises) |
| `fused_linear_cross_entropy` | **Off** (On for Zaya, DeepSeek-V4, GLM-4.7-Flash and text-only Nemotron-H) | Fuses lm_head + CE, no logits materialization. Mutually exclusive with `cross_entropy`. Force-off under TP and PP, [under a multimodal wrapper](#fused-loss-under-a-multimodal-wrapper) and on a prefix LM (PaliGemma, same section); under CP the [Ulysses wrapper](../parallelism/context-parallelism.md#fused-head-loss) runs it instead of the model's head |

Precedence is generic defaults < per-model defaults < user keys, and user keys win **except** the TP/CP
force-offs ([below](#ep--cp--tp--pp-behavior)), the wrapper and prefix-LM force-offs, the
[routed-experts](#routed-experts) force-off and the `layer_norm` force-off, which overwrite an explicit `true`.

Per-model defaults are read off each applier's own signature (`_PER_MODEL_DEFAULTS`), derived from the spec's
`flce_default`, set by the four large-vocab families above and nothing else.

Two library defaults are overridden: TRL's `use_liger_kernel=False` (toolkit → `True`, via
`_TOOLKIT_DEFAULTS` in `halo/configs/parser.py`), and Liger's `fused_linear_cross_entropy=True,
cross_entropy=False` (toolkit → the reverse, because CE preserves logits for metrics and works with every
trainer).

With `use_liger_kernel: false`, SFT's TRL `chunked_nll` loss type becomes `nll` ([SFT → Configuration](../training-methods/sft.md#configuration)).

### Kernel implementation

Liger's dispatcher ranks a CuTe DSL kernel ahead of Triton for most ops on sm_90+
wherever `cutlass.cute` imports, which it does in both images. Of the Liger entry points the toolkit binds, only the
rotary goes through it to a CuTe kernel; RMSNorm, the GLU MLPs and the fused loss call Triton directly, and
cross-entropy's CuTe kernel ranks behind Triton. RoPE is pinned to Triton: `pin_liger_impls`
(`halo/kernels/jit_cache.py`) sets `LIGER_KERNEL_IMPL_ROPE=nvidia-triton` where the toolkit turns
`CUTE_DSL_ENABLE_TVM_FFI` on and at both Liger application sites, and another explicit value, in that variable or
the legacy `LIGER_KERNEL_BACKEND_ROPE`, raises. Liger's CuTe DSL RoPE serves grad-carrying 2-byte calls with `bsz ×
heads × seq ≥ 131072` and `head_dim ≥ 128`, and under `CUTE_DSL_ENABLE_TVM_FFI`, which the images set for FA4, it
rotates a stale buffer from the second launch of a shape at the latest, backward included.
`tests/gpu/kernels/test_liger_rope_repeated_calls.py` checks the bound rotary over repeated calls at Qwen3-8B's
4096-token shape, against a control that runs the CuTe DSL kernel directly.

## CrossEntropy vs FusedLinearCrossEntropy

**CrossEntropy** (default) replaces only the loss function; the model still materializes the
`[batch, seq, vocab]` logits. Logits stay available for metrics (entropy, token accuracy), it works with
every trainer.

**FusedLinearCrossEntropy** fuses `lm_head` with the loss, never materializing logits ([measured
saving](#benchmarks)). On Halo 1.2.0 it runs at equal speed or faster: Qwen3-8B and Qwen3-30B-A3B at 16k within 0.2%
with 12.5–13.2 GiB lower peak, gpt-oss-20b at 4k–16k 2.7–3.4% faster with 15–16.5 GiB lower, and Qwen3.5-9B at 64k
+4.4% with 133 → 54 GiB (batch 1, full checkpointing, FA2 on both sides; same A/B runs as the [framework comparison](throughput-benchmarks.md#full-parameter-sft-framework-comparison)). Trade-offs: `outputs.logits` is `None` (entropy logging disabled); SFT-only
(other trainers need logits for log-probs).

**Eval.** Liger's `lce_forward` defaults `skip_logits` to `self.training and labels is not None`, so a plain
`model.eval()` forward with labels materializes full fp32 logits: at 75k eval tokens × 155k vocab
(GLM-4.7-Flash) a ~43 GiB allocation the training step never hits.

`DistributedSFTTrainer.prediction_step` injects `skip_logits=True` on a loss-only eval batch carrying labels,
unless something still needs real logits: CP mode, a subclass loss that reads `outputs.logits`, or a forward
with no `skip_logits` parameter.

### Known issue: metrics on fp32 logits

Liger's cross-entropy kernel writes into its input: the gradient on a training step
([Liger-Kernel#272](https://github.com/linkedin/Liger-Kernel/issues/272)), and zeros over the rows labeled -100 when
no gradient is needed ([Liger-Kernel#1594](https://github.com/linkedin/Liger-Kernel/issues/1594)). transformers' loss
calls `logits.float()` first, so bf16 logits hand the kernel a copy and fp32 logits hand it the model's own tensor.

So on a training step under Liger with a forward that returns fp32 logits, such as the
[Nemotron-H composite](../models/nemotron-h.md#images), the logged `mean_token_accuracy` and `entropy` are not
meaningful. Eval metrics that mask label -100, the loss and the gradients are right, while a `compute_metrics` reading
logits at ignored positions sees zeros in eval too.

## EP / CP / TP / PP behavior

**SwiGLU / GeGLU on MoE under EP layers.** Liger's MoE appliers set a class-level fused expert FFN (e.g.
`Qwen3MoeExperts = LigerExperts`) holding every expert on one rank, while `apply_ep_layers`
(`halo/distributed/expert_parallel/parallelize.py`) swaps each layer's sparse-MoE block for an `EPMoELayerBase`
subclass holding only the local shard.

The dispatch therefore sets `swiglu=False` / `geglu=False` when all four of
`ParallelismConfig.needs_ep_layers`, `has_routed_experts(model_config)`, a registered EP layer class for
the family, and the GLU patch being **upstream's** hold.

Each conjunct is load-bearing. `needs_ep_layers` is `ep_group_size > 1 or use_grouped_gemm` and
`use_grouped_gemm` defaults **on**, so the flag alone would strip fused SwiGLU from every dense run. A MoE
family with no EP layer class is never wrapped, so the [routed-experts rules](#routed-experts) govern it
instead: Mixtral's upstream `swiglu` is forced off. A toolkit spec names the dense and shared-expert MLPs,
which survive the wrap (Qwen3.5-MoE, Qwen3-Next).

This is a soft gate: an explicit `swiglu`/`geglu` in `liger_kernel_config` survives it, since the swap is inert
under the EP layer. RoPE, RMSNorm and CE/FLCE stay active under EP — FLCE touches only `lm_head` + the loss,
which EP does not wrap.

**TP, CP and PP force `cross_entropy` and `fused_linear_cross_entropy` off** on the model, at one decision site,
`liger_parallelism_overrides`, logged as a warning when either was explicitly enabled. Under CP a resolved
`fused_linear_cross_entropy` is not refused but handed to the Ulysses wrapper, so it is taken out of the patch
request before the fold and draws no warning.

Under **TP**, the `lm_head` weight is sharded over the vocab dim, so a fused CE computing from it would
softmax a partial vocab slice and produce a wrong loss.

**CP** and **PP** compute the loss *outside* the model's forward and never pass labels into it: the
Ulysses wrapper calls the model with `labels=None` and computes a boundary-aware loss itself, and
PP's last stage chunks its own fp32 CE ([PP fused head](../parallelism/pipeline-parallelism.md#fused-head-and-loss-on-the-last-stage)).
A patched head's `skip_logits` gate never fires, and FLCE would report as applied while the full logits plane
materializes anyway. Under CP the dispatch instead stamps whether FLCE resolves on for the family
(`LIGER_CP_FUSED_LOSS_ATTR`: requested, or the family default, and offered by its applier), and the wrapper then
scores each chunk's hidden states through Liger's fused kernel itself
([CP fused head loss](../parallelism/context-parallelism.md#fused-head-loss)). The stamp, like the applied config, is
run state that no [exported `config.json`](../reference/checkpoints.md#exported-configjson) carries.

`cross_entropy` goes with it: upstream's appliers rebind `torch.nn.functional.cross_entropy` **process-wide**,
the function the external loss calls, and the toolkit's [scoped patch](#supported-models) follows suit.

**The GRPO trainers get a warning, not a force-off.** Their objectives compute per-token log-probs outside
the model's forward and never pass `labels`, so an applied FLCE (explicit, or a [per-model default](#configuration)) is
numerically neutral yet saves nothing.

The load-time patch site cannot know the trainer and a construction-time re-apply would not unpatch
`lce_forward`, so trainers declaring `POLICY_GRADIENT_LOSS` warn at construction, pointing at
`use_chunked_grpo_logprobs`, the lever that removes the plane there
([Online GRPO](../training-methods/grpo/online.md#chunked-log-probs),
[Offline GRPO](../training-methods/grpo/offline.md#memory-chunked-log-probs)).

Liger is applied twice: at model load, and again by `DistributedTrainerMixin` (`halo/trainers/distributed_trainer.py`),
which re-sanitizes the train-time `liger_kernel_config` before HF Trainer can re-apply it on the wrapped
model. A filter present at only one site would be undone by whichever runs second, so both call one
sanitizer, `sanitize_liger_config`.

The dispatch records the **effective** applied config on `model.config`, and
`finalize_liger_after_direct_load` (`halo/model_loading/liger/dispatch.py`) reads it after a
`load_distributed_model` load, or after a direct `from_pretrained` whose config went through
`apply_liger_kernel` (the GPU benchmarks). With FLCE applied it keeps `use_liger_kernel` on and pins
`liger_kernel_config` to the applied dict, so TRL's re-application is an identical re-patch and its entropy
guard knows logits are `None`; otherwise it turns the flag off.

HF Trainer re-applies through liger-kernel's own registry, running the applier registered for the model's
own `model_type` alone, its defaults filling every omitted flag. Every config it re-applies (this pinned dict
and the trainer mixin's re-sanitized one) therefore turns off each flag of that applier the toolkit applier
does not hand upstream (`trl_reapplication_config`). Upstream's instance patch would otherwise bind its
llama-cast norm over GptOss's Gemma-cast one, its GeGLU over Gemma 4's toolkit-fused dense MLP, `LigerExperts`
over the Qwen MoE families' routed experts, or its norm swap and fused head over DeepSeek-V4's. A model loaded
outside Halo's loaders runs those roles eager.

DPO and KTO turn TRL's `use_liger_kernel` off at construction in every mode: TRL's chunked log-prob path
refuses the router aux loss, and DPO's also skips the family's head transform
([DPO](../training-methods/preference/dpo.md)). The model-level kernels stay; single-process DPO loses that
loss path's memory saving.

## GptOss

Upstream Liger's `gpt_oss` applier accepts a `swiglu` flag but has no patch block for it: GptOss's custom
clamped/interleaved expert FFN has no standard-SwiGLU equivalent, so `swiglu` is a no-op at any value and
defaults off. FLCE is supported and runs 2.7–3.4% faster than CE on gpt-oss-20b EP1 at 4k–16k, at 15–16.5 GiB lower peak.

Effective options: `rope`, `cross_entropy`, `fused_linear_cross_entropy`, `rms_norm`. The last is served by
the toolkit's spec with Gemma's casting mode (`GptOssRMSNorm` multiplies its weight in fp32 before the cast
back). Upstream's llama-cast swap rounds before the multiply and moves every norm by a bf16 ULP, on the
family the on-policy RL trainers score against a vLLM copy running the eager norm.

The EP MoE layer's own clamped-SwiGLU is independent of Liger: every path — per-expert loop,
[grouped GEMM](grouped-gemm.md) (default on SM90+) and ETP — runs the same fused Triton kernel
`GluCombine.gptoss` (`halo/kernels/fused_glu.py`), the loop path splitting the halves out of one interleaved
`gate_up` projection.

## Benchmarks

Both measured on the Blackwell image, Halo 1.1.0 or earlier. The host-time figure below is from 1.2.0, on Liger 0.8.4.

**Dense — 1× B300 (SM103), Qwen3-8B, GC on, seq 16384, batch 1:** Liger+CE gives 15,966 vs 11,746
tokens/s/GPU (**+36%**) at 64.6 vs 78.6 GiB peak (**−14 GiB**).

**MoE — 2× B300 (SM103) EP=2, Qwen3-30B-A3B (128 experts, top_k=8), GC on, seq 8192, batch 4,** with the
toolkit spec's torch fused norm on `qwen3_moe`'s norms ([above](#upstream-covered-families-the-toolkit-extends)):

| Configuration | tokens/s/GPU | Peak memory | vs baseline |
|---|:---:|:---:|---|
| No Liger | 13,650 | 168.9 GiB | baseline |
| Liger + CrossEntropy | 18,956 | 141.0 GiB | +39%, −28 GiB |
| Liger + FusedLinearCE | 17,927 | 139.5 GiB | +31%, −29 GiB |

At this shape (Halo 1.1.0 or earlier) FLCE saves only 1.5 GiB more than CE and costs 5% throughput; at 16k on EP1 in 1.2.0 the two are within 0.2%. FLCE's saving is the logits
plane: on Qwen3-8B activations at 32k it is 24.3 GB forward (32.2 vs 56.5 GB), ~2.5 GB at 8k. Reach for plain CE when a model's FLCE patch is unavailable.

**Host time per launch.** Liger 0.8.4 adds host time around each RMSNorm and SwiGLU launch: at most ~1.5%
per step on the most host-bound row measured (dense Qwen3-8B at CP2 on 8× B300, 4,096-token rows), and GPU
kernel time is unchanged.

**Measure MoE throughput at batch ≥ 4.** At batch 1 an EP MoE's DeepEP time varies run
to run, so a batch-1 delta is directional (Liger measures +27% there at seq 16384, [torch.compile](torch-compile.md#benchmark-results)).
Memory numbers are deterministic at any batch.

```bash
# Dense (Qwen3-8B), with and without Liger
torchrun --nproc_per_node=1 \
    tests/gpu/profiling/benchmark_sft_dense.py --model qwen3-8b --seq 16384 [--no_liger]

# MoE (Qwen3-30B-A3B), with/without Liger and FLCE
torchrun --nproc_per_node=2 \
    tests/gpu/profiling/benchmark_sft_ep.py --model qwen3-30b-a3b --ep 2 --seq 8192 --batch_size 4 \
    [--no_liger | --fused_linear_ce]
```

The kernels are Triton and need a CUDA GPU; these runs are on B300.

## Related pages

- [PEFT (LoRA)](peft.md) — FLCE benchmarks with QLoRA
- [Padding-Free Collator](padding-free-collator.md) · [BF16 Optimizer](bf16-optimizer.md) · [Throughput Benchmarks](throughput-benchmarks.md)
- [Configuration Reference](../reference/configuration-reference.md) — `use_liger_kernel`
