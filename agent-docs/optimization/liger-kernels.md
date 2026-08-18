# Liger Kernels

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) provides Triton replacements for RMSNorm,
cross-entropy, SwiGLU, and RoPE. Enabled by default (`use_liger_kernel: true`); set it `false` to disable.
A required dependency, pinned `liger-kernel>=0.8.0,<0.9.0` (`uv.lock` resolves 0.8.0). A model type no
applier covers **warns** that the run trains unfused, and an explicit `liger_kernel_config` key on such a
model raises rather than being dropped.

Why fusion pays, counted in HBM round trips:
[GPU Training Theory §5](../reference/gpu-training-theory.md#5-memory-bound-kernels-and-why-fusion-pays).

## Supported models

The orchestrator (`src/kernels/liger/orchestrator.py`) resolves one applier per `model_type`: the toolkit
registry first, then upstream Liger's `MODEL_TYPE_TO_APPLY_LIGER_FN`, then the wrapper's
`text_config.model_type` — multimodal checkpoints register only their inner text path (Gemma 4 is
`gemma4_text`, with no `gemma4` entry), so kernels still reach the decoder and lm_head. The toolkit registry
is built from one declarative spec per family (`src/kernels/liger/families.py`): a spec names the classes
filling each role — RMSNorm, GLU MLP, causal-LM head — and only where the kernel computes the family's
function **exactly**. The builder patches a role by subclassing the family's own class, preserving its
constructor signature and class name.

Upstream Liger 0.8 covers Llama, Qwen2/3, Qwen3 MoE, Qwen3.5/3.6, Mistral, Mixtral, Gemma 2/3/4, GLM-4
(incl. 4V MoE), OLMo 2/3, GptOss, Qwen3-Next and VLM wrappers; `qwen3_5*` and `qwen3_next` are extended by a
[delegating spec](#upstream-covered-families-the-toolkit-extends).

Toolkit-covered families (upstream has none). ✅ = patched, — = left unfused, with the reason:

| Model | model_type | RMSNorm | GLU MLP | RoPE | CE | FLCE |
|---|---|:--:|:--:|:--:|:--:|:--:|
| Mistral 3 / 4 | `mistral3`, `mistral4` | ✅ | ✅ | — interleaved YARN + llama-4 log scale | ✅ | ✅ |
| Zaya | `zaya` | ✅ | — EP wrapper owns the experts | — partial rotary | ✅ | ✅ **default** |
| DeepSeek-V4 | `deepseek_v4` | — mixes weighted and weightless norms, some fp32-pinned | — clamped SwiGLU | — interleaved partial | ✅ | ✅ **default** |
| GLM-4.7-Flash | `glm4_moe_lite` | ✅ | ✅ | — dual interleave/plain MLA | ✅ | ✅ **default** |
| Laguna | `laguna` | ✅ | ✅ | — half-width on full-attention layers, full on sliding | ✅ | ✅ |
| GLM-5.3-Flash | `glm5_next`, `glm5_next_text` | ✅ the two plain norms **+ the GDN gated norm** (fla) | — clamped at `swiglu_limit` | — NoPE text tower | ✅ | — VLM head reads `text_config.vocab_size` |
| Inkling | `inkling_text`, `inkling_mm_model` | ✅ | — trained `global_scale` on the MLP output | — no rotary (learned relative bias) | ✅ | — head divides by `logits_mup_width_multiplier` and truncates to `unpadded_vocab_size` |
| LFM-2 / LFM-2 MoE | `lfm2`, `lfm2_moe` | ✅ | — `w1`/`w3`/`w2`, no `act_fn` | ✅ **the only family whose rotary is Liger's** | ✅ | ✅ |
| Cohere 2 / Cohere 2 MoE | `cohere2`, `cohere2_moe` | — live norm is a mean-subtracting `LayerNorm` with no bias parameter | ✅ | — GPT-J interleaved, sliding layers only | ✅ | ✅ (`logit_scale` folded onto the hidden states) |
| Step-3.7 Flash | `step3p7`, `step3p5` | ✅ Gemma-style (`offset=1.0`, fp32 weight) | — clamps the activated gate | — partial rotary + llama3 scaling | ✅ | — VLM head reads `text_config.vocab_size` |
| Ling / Ring 2.0 | `bailing_moe`, `bailing_moe_linear` | ✅ | ✅ (dense + shared expert) | — partial rotary | ✅ | — the head adds an MTP loss term |
| Ling 3.0 | `bailing_hybrid` | ✅ | ✅ (dense + shared expert) | — interleaved MLA + YARN | ✅ | — the head adds an MTP loss term |

The Bailing families are `trust_remote_code`, so their modeling module does not exist when Liger is applied.
Their patch is registered on the shared hook over `transformers.dynamic_module_utils.get_class_in_module` (`src/models/patches/remote_code_hooks.py`, wrapped once for every registrant) — the funnel every remote
class loads through — and fires on the module defining the declared classes
(`src/kernels/liger/remote_modules.py`). Ling 3.0's KDA layers already run `fla`'s fused gated norm, short
convolutions and delta-rule recurrence; only the attention/MoE norms and the shared-expert GLU are left to
fuse.

**Gated norms go to `fla`, not Liger.** A linear-attention (GDN) block applies
`norm(x) * weight * act(gate)` to its attention output, per head — GLM-5.3-Flash in 34 of 45 layers with a
sigmoid gate, Qwen3.5/3.6 and Qwen3-Next in three of every four with a SiLU one. Liger has no kernel for it
and the `@use_kernel_forward_from_hub("RMSNormGated")` route the classes advertise is inert here, so the
spec's `gated_rms_norm` role binds `flash-linear-attention`'s `FusedRMSNormGated` — already a hard
dependency for this roster's delta-rule and short-convolution kernels. It rides the same `rms_norm` flag, so
one knob turns both norm kernels off. Measured on a B300 over `[1, 8192, 32, 128]` (262144 rows × 128)
forward and backward: **2.3–4.6×** (1.8–2.1 ms eager → 0.43–0.86 ms fused).

The kernel keeps the reduction, the weight multiply and the gate in fp32, as GLM-5's module does; the Qwen
modules round the normalized activation to storage dtype *before* the weight multiply, so the fused path is
the **more accurate** of the two — worst error 3.3e-3 on `dx` against an fp64 oracle, 0.37–1.00× the eager
module's own (`tests/gpu/kernels/test_liger_family_kernels.py` pins that ratio). The gate activation string
is the family's own and is refused unless `fla` implements it; its kernel dispatches on the string with no
else branch, so an unknown gate would apply none at all. A **grouped** gated norm is a different function and
is deliberately uncovered: Bailing's `*GroupRMSNorm` reduces over `hidden_size // group_norm_size`, where a
last-dim kernel would normalize the wrong axis and still return the right shape.

**Cross-entropy is scoped, not process-wide.** The toolkit's CE patch overrides
`transformers.loss.loss_utils`' view of `nn`, its single CE call site, leaving `F.cross_entropy` and every
3-D or soft-target caller untouched. Upstream Liger's own appliers rebind `F.cross_entropy` globally — as
does a [delegating spec](#upstream-covered-families-the-toolkit-extends), whose CE is upstream's.

**One fused head for the roster.** Every family declaring a fused loss installs the same forward
(`src/kernels/liger/lce_forward.py`). It refuses the two shapes it cannot reproduce: `output_router_logits`
(the family's head adds the router aux loss *after* the projection the fused loss replaces, so the term
would silently leave the objective — use `moe_balancing: bias_update`) and an `lm_head` bias.

**Gemma 4 FLCE caveat:** Liger patches `Gemma4ForCausalLM.forward`, but checkpoints saved as
`Gemma4ForConditionalGeneration` compute `logits = self.lm_head(...)` before calling `loss_function`,
bypassing FLCE. Default Liger CE works; for batch>1 at long seq, use a text-only checkpoint.

**Not covered:** `glm4_moe` (full GLM-4.5/4.7) has no EP wrapper, doc page or example in this toolkit, so it
has no spec either.

### Upstream-covered families the toolkit extends

Qwen3.5 / 3.6 and Qwen3-Next resolve to a toolkit applier that **delegates**: upstream Liger's applier runs
first with every flag it declares, and the spec adds the roles it leaves eager — the gated-delta-net blocks'
gated norm (three layers in every four) and, on the MoE siblings, the shared-expert MLP.

| Model | model_type | Upstream applier patches | Toolkit spec adds |
|---|---|---|---|
| Qwen3.5 / 3.6 dense | `qwen3_5`, `qwen3_5_text` | RMSNorm, `Qwen3_5MLP`, FLCE (its CE branch is broken, [below](#configuration)) | GDN gated norm → `fla` |
| Qwen3.5 / 3.6 MoE | `qwen3_5_moe`, `qwen3_5_moe_text` | RMSNorm, `LigerExperts`, FLCE, CE | GDN gated norm → `fla`, shared-expert `Qwen3_5MoeMLP` |
| Qwen3-Next | `qwen3_next` | RMSNorm, `LigerExperts`, FLCE, CE | GDN gated norm → `fla`, dense + shared-expert `Qwen3NextMLP` |

`delegates_to_upstream` makes that a build-time contract: the upstream applier is looked up in liger-kernel's
own registry (a family it stops covering fails at import), the delegating applier re-exports upstream's exact
knob surface, and a role naming a class upstream has already replaced raises at patch time rather than
stacking two swaps. Everything the orchestrator derives then follows the applier that actually **runs** — the
`cross_entropy` default-off for liger-kernel 0.8.0's broken Qwen3.5 branch, and the EP fused-GLU decision
below. Its CE is upstream's too (the process-wide `F.cross_entropy` rebind), which is inert where it would
matter, since CE is force-disabled under CP/TP/PP.

**The shared expert is the toolkit's, the routed experts are upstream's.** Upstream's `swiglu` sets only the
routed-experts class (`Qwen3_5MoeExperts` / `Qwen3NextExperts`), which the EP wrapper replaces wholesale; the
dense and shared-expert MLPs it leaves eager are what the two MoE specs name in `glu_mlp`, and every wrapper
adopts those unchanged. So `swiglu` is **not** forced off under EP for them. Measured on a 4-layer
Qwen3.5-MoE under the EP wrapper: loss 6.95971 unfused vs 6.95981 fused (1.4e-5 relative), shared-expert
gradients at cosine 0.99947 and norm ratio 1.0065 (`tests/gpu/kernels/test_liger_family_kernels.py`).

### Fused GLU under an EP wrapper

Where an EP wrapper owns the routed experts, their activation is the toolkit's own fused Triton GLU, not
Liger's: SwiGLU (`fused_silu_mul`) or tanh-GeGLU (`fused_gelu_tanh_mul`), chosen by probing the layer's
`act_fn` and falling back to the family's eager combine when neither kernel computes it, and it runs on the
grouped-GEMM, per-expert-loop and ETP paths alike. Each wrapped layer logs the executed path at construction
(`grouped_mm=True, glu_combine=fused_silu_mul`, or `glu_combine=eager`);
`tests/gpu/kernels/test_fused_glu.py` checks BF16 and FP32 forward/backward numerics.

## Configuration

Override individual kernels via `liger_kernel_config`:

```yaml
liger_kernel_config:
  cross_entropy: false
  fused_linear_cross_entropy: true
```

Keys: `rope`, `cross_entropy`, `fused_linear_cross_entropy`, `rms_norm`, `swiglu`, `geglu`. A key the
resolved applier's signature does not accept is dropped, with a warning naming it — that path runs eager.

| Kernel | Default | Notes |
|---|---|---|
| `rope` | On | Fused rotary embedding. Auto-off whenever the resolved applier's own `rope` parameter defaults to `False` — the family's rotary (partial, mrope, YARN) has no Liger kernel, and several such appliers raise `NotImplementedError` if asked. Read off the signature, not a model_type list, so a family added upstream later is covered. An explicit `rope: true` bypasses the auto-off and reaches the applier, which refuses it with `NotImplementedError` rather than patching nothing |
| `rms_norm` | On | Fused RMS normalization. Also covers a family's gated (GDN) norm where its spec declares one — one knob, both norm kernels |
| `swiglu` | On | Auto-off where an **upstream** applier's expert-FFN swap is replaced by an EP wrapper, which runs Halo's own fused GLU combine instead. A toolkit spec patches the dense and shared-expert MLPs, which survive EP — see below |
| `geglu` | On | Gemma4. Same auto-off, same replacement |
| `cross_entropy` | On | Keeps logits for metrics. Force-off under TP, CP and PP, defaulted off for appliers whose CE branch cannot run (below), and defaulted off when the config explicitly requests `fused_linear_cross_entropy` (the two are mutually exclusive; setting both explicitly raises) |
| `fused_linear_cross_entropy` | **Off** (On for Zaya, DeepSeek-V4 and GLM-4.7-Flash) | Fuses lm_head + CE, no logits materialization. Mutually exclusive with `cross_entropy`. Force-off under TP, CP and PP |

Precedence is generic defaults < per-model defaults < user keys, and user keys win **except** the TP/CP/PP
force-offs ([below](#ep-cp-tp-behavior)), which overwrite an explicit `true`. Per-model defaults are read off
each applier's own signature (`_PER_MODEL_DEFAULTS`), derived from the spec's `flce_default` — set by the
three large-vocab families above and nothing else. An applier accepting `fused_linear_cross_entropy` but not
`cross_entropy` also defaults FLCE on, since the generic FLCE-off default would reduce it to a no-op.

**One applier has an unusable CE branch:** liger-kernel 0.8.0's `apply_liger_kernel_to_qwen3_5` (reached by
`qwen3_5` and `qwen3_5_text`) imports `liger_cross_entropy` from `liger_kernel.transformers.cross_entropy`,
where the symbol does not live, so the branch raises `ImportError`. The orchestrator defaults `cross_entropy`
off for it; an explicit `cross_entropy: true` still hits the failure. The check matches every applier a call
runs — the resolved one and, under
[delegation](#upstream-covered-families-the-toolkit-extends), the upstream one it wraps. The Qwen3.5 MoE
applier is a different function and is unaffected.

Two library defaults are overridden: TRL's `use_liger_kernel=False` (toolkit → `True`, via
`_TOOLKIT_DEFAULTS` in `src/training/parser.py`), and Liger's `fused_linear_cross_entropy=True,
cross_entropy=False` (toolkit → the reverse, because CE preserves logits for metrics and works with every
trainer).

## CrossEntropy vs FusedLinearCrossEntropy

**CrossEntropy** (default) replaces only the loss function; the model still materializes the
`[batch, seq, vocab]` logits. Logits stay available for metrics (entropy, token accuracy), it works with
every trainer, and it is slightly faster than FLCE on most models.

**FusedLinearCrossEntropy** fuses `lm_head` with the loss, never materializing logits ([measured
saving](#benchmarks)). Trade-offs: `outputs.logits` is `None` (entropy logging disabled); SFT-only
(other trainers need logits for log-probs).

**Eval.** Liger's `lce_forward` defaults `skip_logits` to `self.training and labels is not None`, so a plain
`model.eval()` forward with labels materializes full fp32 logits — at 75k eval tokens × 155k vocab
(GLM-4.7-Flash) a ~43 GiB allocation the training step never hits. `DistributedSFTTrainer.prediction_step`
injects `skip_logits=True` on a loss-only eval batch carrying labels, unless something still needs real
logits: CP mode, a subclass loss that reads `outputs.logits`, or a forward with no `skip_logits` parameter.

## EP / CP / TP behavior {#ep-cp-tp-behavior}

**SwiGLU / GeGLU on MoE under EP wrappers.** Liger's MoE appliers set a class-level fused expert FFN (e.g.
`Qwen3_5MoeExperts = LigerExperts`) holding every expert on one rank, while `patch_moe_model_for_ep`
(`src/distributed/expert_parallel/patching.py`) swaps each layer's sparse-MoE block for an `EPMoELayerBase`
subclass holding only the local shard. The orchestrator therefore sets `swiglu=False` / `geglu=False` when
all four of `ParallelismConfig.needs_ep_wrappers`, `config_has_experts(model_config)`, a registered EP
wrapper class for the family, and the GLU patch being **upstream's** hold. Each conjunct is load-bearing:
`needs_ep_wrappers` is `ep_group_size > 1 or use_grouped_gemm` and `use_grouped_gemm` defaults **on**, so the
flag alone would strip fused SwiGLU from every dense run; a MoE family with no EP layer class (Qwen3-Next)
is never wrapped and would be left with neither kernel; and a toolkit spec names the dense and shared-expert
MLPs, which survive the wrap. Liger's `LigerExperts` runs only under `use_grouped_gemm: false`.

This is a soft gate: an explicit `swiglu`/`geglu` in `liger_kernel_config` survives it. RoPE, RMSNorm and
CE/FLCE stay active under EP — FLCE touches only `lm_head` + the loss, which EP does not wrap.

**TP, CP and PP force `cross_entropy` and `fused_linear_cross_entropy` off** — one decision site,
`liger_parallelism_overrides`, logged as a warning when either was explicitly enabled. Under **TP**,
`lm_head` is `ColwiseParallel`-sharded into DTensor logits split over the vocab dim, where a fused CE would
softmax a partial vocab slice and produce a wrong loss. **CP and PP** compute their loss *outside* the
model's forward and never pass labels into it (the Ulysses wrapper calls the model with `labels=None` and
computes a boundary-aware fp32 CE itself; a pipeline stage is driven with `input_ids`/`attention_mask` only),
so Liger's `skip_logits` gate never fires and FLCE would report as applied while the full logits plane
materializes anyway. `cross_entropy` goes with it because upstream Liger's appliers install it by rebinding
`torch.nn.functional.cross_entropy` **process-wide** — the exact function both external losses call; the
toolkit's own CE patch is scoped ([above](#supported-models)) but is force-off here too so both branches
behave alike.

Pipeline parallelism ([not yet available in this release](../parallelism/pipeline-parallelism.md)) would
lose nothing: its shipped loss seam runs head + cross-entropy together over one token chunk at a time
(`fused_causal_lm_token_loss`), leaving no full logits plane for FLCE to save.

**The GRPO trainers get a warning, not a force-off.** Their objectives compute per-token log-probs outside
the model's forward and never pass `labels`, so an applied FLCE (explicit, or Zaya's per-model default) is
numerically neutral yet saves nothing. The load-time patch site cannot know the trainer and a
construction-time re-apply would not unpatch `lce_forward`, so trainers declaring
`_loss_outside_model_forward` warn at construction, pointing at `use_chunked_grpo_logprobs` — the lever that
removes the plane there ([Online GRPO](../training-methods/grpo/online-grpo.md#chunked-log-probs),
[Offline GRPO](../training-methods/grpo/offline-grpo.md#memory-chunked-log-probs)).

Liger is applied twice — at model load, and again by `DistributedTrainerMixin` (`src/trainers/mixins/base.py`),
which re-sanitizes the train-time `liger_kernel_config` before HF Trainer can re-apply it on the wrapped
model. Both sites must agree: a filter present in only one is undone by whichever runs second. The
orchestrator records the **effective** applied config on `model.config`, and
`finalize_liger_after_direct_load` (`src/models/loading/model_preparation.py`) reads it after a
`load_distributed_model` load: with FLCE applied it keeps `use_liger_kernel` on and pins
`liger_kernel_config` to the applied dict, so TRL's re-application is an identical re-patch and its entropy
guard knows logits are `None`; otherwise it turns the flag off. Code loading via
`AutoModelForCausalLM.from_pretrained()` instead — the GPU benchmarks — patches through
`apply_liger_kernel_for_direct_loading()`, which applies the toolkit defaults then sets
`use_liger_kernel = False`.

Under FSDP2, TRL's fused Liger preference/GRPO loss (`liger_loss_fn`, `liger_grpo_loss`) is auto-disabled: it
does `input @ weight.t()` against `model.lm_head.weight` outside FSDP2's forward hooks, where the weight is a
sharded DTensor. Model-level kernels stay active.

## GptOss

Upstream Liger's `gpt_oss` applier accepts a `swiglu` flag but has no patch block for it — GptOss's custom
clamped/interleaved expert FFN has no standard-SwiGLU equivalent, so `swiglu` is a no-op at any value. FLCE
is supported, marginally slower than CE. Effective options: `rope`, `cross_entropy`,
`fused_linear_cross_entropy`, `rms_norm`.

The EP MoE layer's own clamped-SwiGLU is independent of Liger: every path — per-expert loop,
[grouped GEMM](grouped-gemm.md) (default on SM90+) and ETP — runs the same fused Triton kernel
`fused_gptoss_glu` (`src/kernels/fused_glu.py`), the loop path splitting the halves out of one interleaved
`gate_up` projection.

## Benchmarks

**Dense — 1× B300 (SM103), Qwen3-8B, GC on, seq 16384, batch 1:** Liger+CE gives 16,396 vs 11,943
tokens/s/GPU (**+37%**) at 64.6 vs 78.6 GB peak (**−14 GB**).

**MoE — 2× B300 (SM103) EP=2, Qwen3-30B-A3B (128 experts, top_k=8), GC on, seq 8192, batch 4:**

| Configuration | tokens/s/GPU | Peak memory | vs baseline |
|---|:---:|:---:|---|
| No Liger | 11,364 | 168.9 GB | baseline |
| Liger + CrossEntropy | 15,961 | 150.2 GB | +40%, −19 GB |
| Liger + FusedLinearCE | 15,469 | 126.6 GB | +36%, −42 GB |

FLCE keeps nearly all of CE's throughput while cutting another −24 GB — the long-sequence lever. The saving
is the logits plane and so scales with sequence: on Qwen3-8B activations at 32k it is 24.3 GB forward
(32.2 vs 56.5 GB), ~2.5 GB at 8k. Reach for plain CE when a model's FLCE patch is unavailable.

**Measure MoE throughput at batch ≥ 4.** At batch 1, EP MoE is communication-bound and the Liger throughput
effect is within run-to-run noise. Memory numbers are deterministic at any batch.

```bash
# Dense (Qwen3-8B), with and without Liger
torchrun --nproc_per_node=1 \
    tests/gpu/profiling/benchmark_sft_dense.py --model qwen3-8b --seq 16384 [--no_liger]

# MoE (Qwen3-30B-A3B), with/without Liger and FLCE
torchrun --nproc_per_node=2 \
    tests/gpu/profiling/benchmark_sft_ep.py --model qwen3-30b-a3b --ep 2 --seq 8192 [--no_liger | --fused_linear_ce]
```

The kernels are Triton, so they need a supported GPU (H100/H200, B200/B300).

## Related pages

- [PEFT (LoRA)](peft.md) — FLCE benchmarks with QLoRA
- [Padding-Free Collator](padding-free-collator.md) · [BF16 Optimizer](bf16-optimizer.md)
- [Throughput Benchmarks](throughput-benchmarks.md)
- [Configuration Reference](../reference/configuration-reference.md) — `use_liger_kernel`
