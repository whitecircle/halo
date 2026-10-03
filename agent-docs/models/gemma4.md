# Gemma 4 MoE

`Gemma4ForConditionalGeneration` — Google's MoE with the router living **outside** the EP-wrapped module, plus KV-shared layers and `attention_k_eq_v` patterns. EP and pure ETP (`ep_size=1`) are supported; TP / CP wrappers are not implemented because of the attention quirks described below.

| | EP | CP | TP | ETP |
|---|:--:|:--:|:--:|:--:|
| Gemma 4 MoE | Yes | **No** | **No** | Yes |

## EP wrapper

`EPGemma4MoELayer` (`src/distributed/expert_parallel/layers/gemma4.py`) replaces `Gemma4TextExperts`.

- **Router is a sibling**, not a child — `Gemma4TextRouter` lives in `Gemma4TextDecoderLayer` next to the experts. The wrapper only replaces the experts; the router stays FSDP-managed.
- Routing input: pre-normalized weights from the sibling router (per-expert-scaled, not raw logits).
- Activation: tanh-GeGLU (fused gate_up), run through the fused Triton kernel (`src/kernels/fused_glu.py`) when the activation is the genuine `gelu_pytorch_tanh`; logged as `glu_combine=fused_gelu_tanh_mul`.
- Norms: `Gemma4RMSNorm` (scaled and weightless) runs torch's fused `F.rms_norm`, which normalizes and multiplies the weight in fp32 and casts once — Gemma's own order. The Liger spec takes the norm role from upstream with `rms_norm_kernel="native"` and `rms_norm_casting_mode="gemma"`.
- Liger: every decoder layer keeps a dense `Gemma4TextMLP` beside its experts, which the wrapper never touches. The toolkit's delegating spec fuses it (`geglu`, probing the activation the same way) so it stays fused under EP, and takes the norms as above; upstream's `gemma4_text` applier serves the fused loss.

    The fused loss is forced off for `Gemma4ForConditionalGeneration` checkpoints, whose own head runs ([Liger Kernels](../optimization/liger-kernels.md#fused-loss-under-a-multimodal-wrapper)).

- Storage: `gate_up_proj [E, H, 2M]`, `down_proj [E, M, H]` (matmul convention; the HF checkpoint is `[E, 2M, H]` / `[E, H, M]`, transposed on load). Compute via Grouped GEMM.

### Precision flags

The router-outside-wrapper layout changes how the precision flags behave:

- `fp32_router` is a no-op — it controls the EP wrapper's internal router, and Gemma 4's lives outside it. Harmless, but it buys nothing: the shipped configs leave it unset and say why.
- `fp32_experts: true` upcasts the precision-sensitive expert weights — use it.
- `fp32_non_ep_params: true` is **refused at model load** under EP, off `config.json` and before the process groups (`_validate_fp32_non_ep_params` in `src/distributed/loading/model_loading.py`, driven by the EP layer's own `_supports_fp32_non_ep_params`). Train Gemma 4 experts-distributed in plain bf16.

    Gemma 4's norms re-emit activations at weight dtype, so the upcast feeds fp32 tokens into DeepEP's 2-byte transport. Without the gate it surfaces as a raw C++ assert at the first dispatch, after the whole multi-GPU load. See [Precision control](../parallelism/expert-parallelism.md#precision-control).

Gemma 4's softmax router has no native aux-loss term (no `router_aux_loss_coef` field, no aux term in the loss) and the EP wrapper has no bias-update path, so no load balancing is available. `auto` resolves to `none` with a warning; **set `moe_balancing: none` explicitly** to state that deliberately. An explicit `aux_loss` finds no coefficient, stays off, and stamps the config so `MoEMetricsCallback` declines to wire itself.

Under `none` nothing is stamped, but `output_router_logits: true` in `model_init_kwargs` cannot buy the `moe/*` load metrics back either: no Gemma 4 config level declares that field, so `_apply_config_overrides` rejects the override at load. See [Gemma 4 has no balancing path](../training-methods/callbacks.md#gemma-4-has-no-balancing-path).

The router's top-k field is spelled `top_k_experts` (not `num_experts_per_tok`) and lives on the text sub-config, along with `num_experts`. Both are covered by the toolkit's shared detection, so `moe/*` metrics and the S-MFU sparsity factor use the real top-k.

## Why TP and CP aren't supported

Gemma 4 attention uses two patterns that need bespoke wrappers:

- **KV-shared layers** — the trailing `num_kv_shared_layers` decoder layers reuse an earlier layer's K/V. Naive TP would clone the cache; a CP wrapper would have to track the shared-from layer's sequence slice.
- **`attention_k_eq_v`** — when set, the K projection is reused as V (one linear), incompatible with the standard ColwiseParallel-on-Q/K/V plan.

No Gemma 4 attention class appears in `TP_SHARDABLE_ATTENTION_CLASSES` (`src/distributed/tensor_parallel/module_types.py`) or in the CP wrapper registry (`src/distributed/context_parallel/layers/`), so neither mode patches Gemma 4 attention.

Pipeline parallelism is [not yet available in this release](../parallelism/pipeline-parallelism.md).

ETP only shards the expert FFN weights (gate/up/down), leaving attention untouched and the sibling `Gemma4TextRouter` FSDP-managed and replicated across ETP ranks. The fused-GLU `gate_up_proj` is split into separate `gate_proj`/`up_proj` shards at init so each rank holds matching gate/up positions on the intermediate dim. See [Expert Tensor Parallelism](../parallelism/expert-tensor-parallelism.md).

## Buffer fixes

`finalize_loaded_model()` (`src/models/patches/buffer_fixes.py`), run by every load path, restores two buffers that meta-device init drops: a `_NON_PERSISTENT_FIXERS` fixer restores `embed_scale` (the `sqrt(hidden_dim)` scaled-word-embedding factor) from the `scalar_embed_scale` the embedding keeps beside it; the `_ROTARY_FIXERS` chain recomputes the per-layer-type RoPE `inv_freq` buffers (`full_attention_inv_freq` + `sliding_attention_inv_freq`) in FP32.

## Export

Every exported `config.json` carries the flat `global_head_dim` / `num_global_key_value_heads` keys
beside the per-layer fields: transformers 5.16 folds the full-attention layers' geometry into
`per_layer_config`, which the pinned vLLM server's transformers refuses, so the EP layer declares
the two keys (`_LEGACY_PER_LAYER_CONFIG_KEYS`) and the export rewrites them back
([Checkpoints](../reference/checkpoints.md#what-gets-saved)).

Both pinned engines read the fused expert pair the gather emits, so RL weight sync runs on either
`rollout_backend`. On SGLang it needs this repo's image: the router folds `scale` into its norm once
behind a latch, so a synced router weight would never reach routing, and `Dockerfile.sglang` patches
that (`docker/sglang/patches/`). Gemma 4 has no routing replay on either engine
([Rollout Servers](../infrastructure/rollout-servers.md#which-families-each-engine-serves)).

## Configs

Gemma-4 26B-A4B (30 EP layers, `Gemma4ForConditionalGeneration`) trains text-only under EP=8 at 32,768 max length. Example: `examples/sft/gemma4/gemma4-26b-a4b-ultrachat-ep.yaml`. The multimodal class loads either way, but a text-only dataset takes the **text** data path, which is what makes `packing` legal here ([SFT — VLMs](../training-methods/sft.md#vision-language-models)).

Async GRPO with Environments: `examples/grpo/environmental/gemma4/vllm/` plus the `sglang/` ep1 siblings; both need `use_chunked_grpo_logprobs` for the 262k vocabulary, and the chunked sweep applies the head's `final_logit_softcapping` ([Memory and Throughput](../training-methods/grpo/async-grpo/performance.md#chunked-log-probs)). Both pinned engines read the fused expert pair the gather emits, so either `rollout_backend` takes the weight sync ([Rollout Servers](../infrastructure/rollout-servers.md#which-families-each-engine-serves)).

**Attention LoRA** (the `-lora-` recipes) adapts the language model alone. The vision and audio towers'
projections carry the same `q_proj`…`o_proj` names but are `Gemma4ClippableLinear`, a wrapper holding
its weight in a child `nn.Linear` that PEFT cannot decompose, so they are excluded from injection with
a warning naming the count ([PEFT](../optimization/peft.md#targets-peft-cannot-adapt)); the vision
MLP spells `gate_proj`/`up_proj`/`down_proj` the same way, so those names collect exclusions there too
(under EP the expert peel takes them first). `all-linear` resolves to the inner `nn.Linear` and adapts
the towers too.

**Long-context attention**: Gemma 4's full-attention layers run at `global_head_dim=512`, which every FlashAttention kernel and cuDNN SDPA reject (FA2 caps at 256; FA4's SM100 kernel overflows tensor memory).

`load_distributed_model` redirects any FlashAttention impl to SDPA for Gemma 4, then `patch_sdpa_for_wide_heads()` forces the mem-efficient SDPA kernel, the only backend handling this head dim, with manual KV repeat (`use_gqa_in_sdpa → False`). That avoids the math kernel's `[B, heads, S, S]` score matrix, which OOMs at seq 32k. Set `attn_implementation: sdpa` to skip the warning.

On CUDA the model is then built with the attention implementation `sdpa_flex_sliding`, which the frozen reference and teacher loader picks too; how it works is in [Flash Attention](../optimization/flash-attention.md#sliding-window-and-wide-head-layers-on-sdpa). On Gemma 4 26B-A4B (one B300, `tests/gpu/profiling/benchmark_gemma4_attention.py`, fwd+bwd per layer; 2026-10-03, commit 0bc3a22a5, Blackwell image):

- **Sliding layers** (head_dim 256, window 1,024) run compiled FlexAttention: 0.55 / 1.91 / 7.86 ms at 2,048 / 8,192 / 32,768 tokens, against 4.65 / 61.8 / 941 ms for mem-efficient SDPA with its dense mask.
- **Global layers** (head_dim 512) run matmul attention while the layer's saved scores fit the 2 GiB budget (16 heads at 4,096 tokens per row), and mem-efficient SDPA past it: 1.86 / 6.73 ms at 2,048 / 4,096 tokens against 6.53 / 19.7 ms, and the same time as SDPA at 8,192 and beyond, where the layer runs it. FlexAttention only fits shared memory at head_dim 512 with its smallest tiles, and those are slower than SDPA.
- The vision and audio towers (bidirectional) keep SDPA. `HALO_FLEX_SLIDING=0` keeps plain SDPA for every layer.

End to end, Gemma 4 26B-A4B full SFT at EP2 on 2× B300 without gradient checkpointing (dd489a700, against plain SDPA): 11,152 → 14,980 cluster tok/s at 2,048 tokens (1.34×), and 4,124 → 16,204 at 16,384 (3.93×) with peak memory 237.6 → 225.0 GiB.

The KV-repeat override is not what makes the global layers legal — transformers 5.16 disables GQA above head_dim 256 itself. It stays because the patch pins mem-efficient as the *only* enabled backend process-wide, where native `enable_gqa` for the 256-dim sliding layers is unverified; the manual repeat is the one measured path. See [Flash Attention](../optimization/flash-attention.md#model-specific-handling).

Gemma 4 never reaches a varlen kernel, so `select_data_collator` rejects **`padding_free`** for it — the gate is the resolved `_attn_implementation`, and only `flash_attention_2/_3/_4` qualify. `packing: true` (what the example config uses) still isolates documents through per-document `position_ids`, at the cost of a dense mask over the flattened batch (side up to `per_device_train_batch_size * max_length`) instead of `cu_seqlens` ([Collators](../data/collators.md)).

## RL chat template

The async GRPO recipes pin `jinja-templates/gemma4/gemma4-reasoning-effort.jinja` with `force_chat_template: true`,
and the servers run the same file (`VLLM_CHAT_TEMPLATE` / `SGLANG_CHAT_TEMPLATE` in the compose files). It
is the hub template reduced to text-only tool use — the `<|think|>` marker opens every system turn, an
assistant turn renders whatever reasoning it carries, the media placeholders and the Gemma-native
`tool_responses` branch dropped — plus a system-turn line stating the episode's `reasoning_effort` and
`reasoning_budget`, which the hub template has no variable for. The hub template keys thinking on an
`enable_thinking` variable that the vLLM server sets from `reasoning_effort` and the trainer's renders do
not, so the pinned template is also what keeps the two prompts identical
([Reasoning budget](../training-methods/grpo/async-grpo/rollouts.md#reasoning-budget)).
