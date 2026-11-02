# Bailing MoE / Ling

`BailingMoeV2ForCausalLM` — Ant Group's MoE with standard softmax GQA attention (16 heads / 4 KV / head_dim 128), sigmoid routing (256 experts in 8 groups; the top-4 groups are chosen, then the top-8 experts within them), and 1 shared expert. Loaded from the `inclusionAI/*` repos under `trust_remote_code=True`.

Per-axis support: [Compatibility matrix](README.md#compatibility-matrix). CP covers Ling 2.0 only ([CP wrapper](#cp-wrapper)).

The shipped config trains `inclusionAI/Ling-mini-2.0` (`model_type: bailing_moe`). The sibling `inclusionAI/Ring-mini-linear-2.0` is a **different** architecture (`BailingMoeLinearV2ForCausalLM`, `model_type: bailing_moe_linear`) that swaps softmax attention for Lightning Attention-2. Both share the `BailingMoeV2SparseMoeBlock` the EP layer targets.

## Ling 3.0

`BailingMoeV3ForCausalLM` (`model_type: bailing_hybrid`) — `inclusionAI/Ling-3.0-tiny` (7.9B, 128 experts) and `inclusionAI/Ling-3.0-flash` (~122B, 512 experts).

The MoE block matches V2's (same per-expert `gate_proj`/`up_proj`/`down_proj` modules, same gate arithmetic: sigmoid, `expert_bias` on selection only, group-limited top-k, renormalization, `routed_scaling_factor`), so `EPBailingMoELayer` claims `BailingMoeV3SparseMoeBlock` too and everything above applies. The V3 block returns `(hidden, router_logits)` where V2 returns a tensor; its decoder layer tests the return for a tuple, so the EP layer's bare tensor is accepted.

Attention is what differs: layers alternate in groups of `layer_group_size`, with most layers `BailingMoeV3KimiDeltaAttention` (KDA linear attention through `fla`) and the rest `BailingMoeV3MultiLatentAttention` (MLA).

Neither layer type keeps a packed row's documents apart. The MLA layers build their mask from `attention_mask` alone, like Ling 2.0's SDPA path, and the KDA op's `cu_seqlens` kwarg is unreachable: the model forward passes its layers no extra kwargs ([Document isolation](../data/collators.md#document-isolation-under-packing)).

Four consequences for a training config:

- **Pin `attn_implementation: sdpa`.** The modeling file declares only the v4-era `_supports_flash_attn_2`, which transformers v5 ignores in favor of `_supports_flash_attn`, and carries no flash attention class for the [template patch](#attention-and-packing) to route, so an auto-selected flash label (FA4 on Blackwell, FA3 on Hopper) is refused at model build. The KDA layers run their own `fla` kernels either way.
- **Set `packing: false`.** Packing off a varlen backend is refused for the whole family ([Attention and packing](#attention-and-packing)), and `padding_free` needs a flash label, so Ling 3.0 trains on padded batches.
- **`fp32_non_ep_params: true` keeps the KDA state in fp32 (optional).** The modeling file declares `A_log` and `dt_bias` in fp32; without the flag the loaders cast them to the run dtype with every other parameter ([Load precision](README.md#load-precision)). With it they train, with the other non-expert parameters, as fp32 masters from their stored fp32 values (compute stays bf16).

    At `expert_parallel_size: 1` pair it with **`fsdp_shard_ep1_experts: false`**: the upcast skips every EP-layer parameter, so FSDP-managed replicated experts would sit bf16 inside the same fp32 shard group and `ParallelismConfig` refuses the combination at config time. Above ep1 the knob has no effect; the experts are FSDP-ignored anyway.

- **Override `rope_scaling`.** The config ships `rope_scaling: null`, which transformers v5 normalizes into a dict carrying no `"factor"`; the MLA layers then read `config.rope_scaling["factor"]` and raise `KeyError`. Pass the minimal replacement and nothing more:

    ```yaml
    model_init_kwargs:
      rope_scaling: {rope_type: default, factor: 1.0, mscale_all_dim: 0}
    ```

    The v5 setter stores this dict verbatim in `rope_parameters`, so adding `rope_theta` or `partial_rotary_factor` to it overrides what the rotary module sets for itself and halves the rotary dimension. The minimal form reproduces stock frequencies exactly, and `mscale_all_dim: 0` keeps `factor` unused.

`expert_swiglu_limit_list` / `share_expert_swiglu_limit_list` appear in the Ling-3.0-flash config but are read nowhere in the shipped modeling code, so no clamp is applied and the wrapper stays equivalent. Both repos need `model_revision` pinned — these workarounds depend on the remote code's exact internals.

## Attention and packing

Ling 2.0's and Ring's modeling files are built on transformers v4's Llama flash template: an `ATTENTION_CLASSES` table keyed by label, a flash class that calls its own mask-only `_flash_attention_forward`, the v4 support flag, and a model that hands the layers a 2D padding mask only when its label is `flash_attention_2`. Unpatched, every v5 flash label is refused at build, and the one the file knows runs a packed row as a single causal sequence.

`patch_v4_flash_attention_template` (`halo/models/attention/remote_flash_attention.py`) runs on every remote modeling module the [remote-class funnel](#remote-code-compatibility) loads and patches any file whose flash call has the template's exact signature:

- the table maps `flash_attention_2/3/4` to the flash class, which keeps its name (the [CP wrapper](#cp-wrapper) claims it by name) and its weights;
- the classes declaring `_supports_flash_attn_2` declare `_supports_flash_attn`, and the model's mask switch holds for every flash label;
- the flash call runs through transformers' flash dispatch with the attention forward's `position_ids`, so a packed row's documents become varlen sequences on FA4 (Blackwell) and FA3 (Hopper).

The file's SDPA and eager paths build their causal mask from `attention_mask` alone and never read `position_ids`, so a packed row attends across its documents there; Ling 3.0's MLA layers do the same. The family spec declares it (`dense_packing_leaks`), so packing on those backends is refused at collator selection for Ling 2.0, Ring and Ling 3.0, as for GPT-OSS ([Document isolation](../data/collators.md#document-isolation-under-packing)). The shipped config leaves `attn_implementation` unset and packs on the auto-selected flash kernel. Ring's Lightning-Attention-2 layers carry state across packed documents on every backend, so packing Ring is refused on flash too unless the run sets `allow_cross_document_linear_attention: true` ([Document isolation](../data/collators.md#document-isolation-under-packing)).

## EP layer

`EPBailingMoELayer` (`halo/distributed/expert_parallel/families/bailing.py`) replaces `BailingMoeV2SparseMoeBlock`.

- Routing: `BailingMoeV2Gate` — sigmoid scoring, group-limited top-k, expert bias, normalization.
- Routed experts: per-expert `BailingMoeV2MLP` modules stacked into 3D tensors at init and transposed to matmul convention (`gate_proj [E, H, M]`, `up_proj [E, H, M]`, `down_proj [E, M, H]`). Compute via 3-call Grouped GEMM.
- Shared experts: replicated per rank and added after DeepEP combine (dispatch overlap is the opt-in `HALO_EP_SHARED_OVERLAP` side stream).
- Checkpoint round-trip: the lazy loader materializes only this rank's expert range and `_adopt_expert_params` stacks it into a 3D slice. On save, the family's `SEPARATE_GLU` expert layout gathers, transposes back to `F.linear` convention, and re-splits to per-expert `experts.{i}.{gate,up,down}_proj.weight`.

## CP wrapper

`BailingMoeV2UlyssesAttention` (`halo/distributed/context_parallel/families.py`) wraps Ling 2.0's attention. The math is plain softmax GQA; what the wrapper carries is the family's spellings: a fused `query_key_value` projection split into Q/K/V, an output projection named `dense`, Q/K RMSNorm applied before a partial rotary, and a softmax scale the module never stores because its own forward computes it inline.

All three eager/sdpa/flash subclasses are claimed: they hold the same weights, and CP replaces the forward either way.

The wrapper sets `REQUIRES_FLASH_ATTN_LABEL = False`: CP calls the flash kernel itself, so the run's label (the auto-selected flash one, or `sdpa`) describes a path CP has replaced; every other family keeps the check. `context_parallel_size` is at most 4 on Ling-mini-2.0, since `cp_size` must divide its 4 KV heads.

`Ring-mini-linear-2.0` is **rejected**, not wrapped: its file reuses Ling 2.0's full-attention class names, so validation matches `BailingMoeV2LinearAttention` by name to avoid wrapping the few full-attention layers while the Lightning-Attention-2 recurrence scans each rank's shard in isolation. (`finalize_loaded_model()` recomputes its slope buffers on every load path.) Ling 3.0 is likewise unavailable — its KDA layers are a linear recurrence and its MLA layers carry no wrapper.

## Why TP doesn't work

No Bailing attention class is registered for TP (`TP_SHARDABLE_ATTENTION_CLASSES`, `halo/distributed/tensor_parallel/parallelize_attention.py`), so TP does not patch Bailing attention. The gap is a missing DTensor plan, not an architectural blocker — the shipped model is standard softmax GQA.

## ETP

`EPBailingMoELayer._adopt_expert_params` stacks the per-expert modules through `_store_separate_glu_params`, which at `expert_tp_size > 1` slices `gate_proj` / `up_proj` along the intermediate dim `M` and `down_proj` to match; the shared expert stays replicated. Use pure ETP (`ep_size=1`) when expert memory is the bottleneck. See [ETP](../parallelism/expert-tensor-parallelism.md#limitations).

## Remote-code compatibility

These checkpoints need `trust_remote_code=True` and call transformers-v5-removed APIs (`is_torch_fx_available`, `ROPE_INIT_FUNCTIONS['default']`). `apply_remote_code_compat_shims()` restores both and the model-loading utilities call it automatically; no vendoring required. The same funnel runs the flash template patch ([Attention and packing](#attention-and-packing)).

Every Bailing/Ling class also ships the v4-era list form of `_tied_weights_keys`, which transformers 5 reads as a `{duplicate: source}` dict and which crashes `save_pretrained` outright. The shims convert it to `{}` (these checkpoints store `lm_head` untied), and the loader refuses a config declaring `tie_word_embeddings: true` that carried the list, rather than training the head untied.

`Ring-mini-linear-2.0` needs a shim of a different kind: its modeling file calls `DynamicLayer()` but imports only `Cache` and `DynamicCache`, so the bare global raises `NameError` on every **cached** forward (`generate`, and any check on a reloaded model that does not pass `use_cache=False`). Training keeps the cache off, so the break appears only after a run finishes.

The API shims patch transformers, which cannot work here (the name is never imported); this one binds the class into the loaded module through `get_class_in_module`, the funnel every remote class passes. Absent names only: a file defining its own is untouched.

A bare `from_pretrained` of it is not a usable model either: transformers 5 re-materializes every non-persistent buffer as `torch.empty_like`, and the file's `_init_weights` overrides transformers' without `super()`, so the Lightning-Attention-2 decay `slope` and the rotary `inv_freq` are never computed. Zeroed pages give finite logits from a model whose RoPE is dead; a reused allocator gives all-NaN logits, `exp(cumsum(g))` overflowing on the garbage slope.

Every toolkit load path repairs both in `finalize_loaded_model()`. A plain `AutoModelForCausalLM` + `PeftModel` reload of a trained adapter has to call it as well.

Ling 3.0's `modeling_bailing_moe_v3` needs one more: the file copies HF's `eager_attention_forward` helper **without** the `ALL_ATTENTION_FUNCTIONS` dispatch, so `attn_implementation: sdpa` changes the mask format while every MLA layer still materializes the `[B, H, S, S]` score plane plus its fp32 softmax copy — ~190 GiB on an 80k-token row, an instant OOM.

The SDPA dispatch shim (bound through the same funnel) wraps the module global with an SDPA route gated on the attention module's own config: `sdpa` runs `F.scaled_dot_product_attention` (linear memory), anything else, including any `output_attentions` call, keeps the file's own eager, and a file carrying a real dispatch is untouched.

The KDA layers run fla's Triton kernels, whose autotuners re-benchmark on shape-derived keys. The causal-conv1d key includes `ceil(tokens/1024)`, so a fresh row width costs that rank tens of seconds of autotuning while its 7 peers spin in the next collective at idle power.

Measured configs persist in the Triton disk cache, which `configure_training_run` anchors on `HF_HOME` (`TRITON_CACHE_DIR`), so the warm-up is paid once per volume and the slow-step pattern appears only against a fresh cache.

On Blackwell the KDA path shares the delta-rule state kernel whose `num_warps=4` configs are excluded there; the Ling 3.0 remote code imports `fla` directly, and the restriction applies when its load finishes ([Qwen3.5 linear attention](qwen3_5.md#linear-attention-qwen3_5moegateddeltanet)).

## Router balancing

Bailing is aux-loss-free by design (`moe_router_enable_expert_bias: true`; Ling 3.0 also declares `topk_method: noaux_tc`): no modeling variant computes a load-balancing loss and the config carries no `router_aux_loss_coef`. The family's native mechanism is the gate's persistent `expert_bias` buffer, added to the sigmoid scores for **selection only** — combine weights stay unbiased.

`EPBailingMoELayer` hands exactly that buffer to `BalancingBiasCallback`: under `moe_balancing: bias_update` (or `auto`, which resolves there whenever the EP layer is applied) the DeepSeek-V3 sign-updates land in `expert_bias` itself, upcast to fp32 at enable so the 1e-3 steps survive the add.

The EP layer exposes it as a live property, so callback updates and the `router_balancing_biases.pt` resume-restore always reach the buffer the gate reads. Because the buffer is part of the checkpoint, a gathered save exports the **final** bias and a served checkpoint routes exactly as training did.

Most EP layers adopt a native slot the same way; the per-family slots and the exceptions are in the [BalancingBiasCallback](../training-methods/callbacks.md#balancingbiascallback) table. `moe/*` load metrics come with the callback.

An explicit `aux_loss` warns and stays off (there is no aux term to enable). Without the EP layer — `use_grouped_gemm: false` at `expert_parallel_size: 1` — nothing counts expert loads, so `bias_update` raises at setup; freezing the gate is the fallback mitigation there:

```yaml
freeze_layers_patterns:
  - "*.mlp.gate.weight"
```

## Serving and weight sync

Only **Ling 2.0** (`bailing_moe`) is servable and syncable on both pinned engines: vLLM 0.32.0 registers `BailingMoeV2ForCausalLM`, SGLang 0.5.22 the same spelling, and gathered saves land in their per-expert expert loaders.

**Ling 3.0** (`bailing_hybrid`) is registered on both pinned engines (`BailingMoeV3ForCausalLM`), but no online update into it has been validated end to end. **Ring**'s checkpoints declare `BailingMoeLinearV2ForCausalLM` where both engines register `BailingMoeV2_5ForCausalLM`, an architecture-string mismatch, so the server cannot load the base model. Both clients list the two spellings in their `UNSERVABLE_MODEL_TYPES`, so RL weight sync is refused at trainer construction on either backend ([Rollout Servers](../infrastructure/rollout-servers.md#which-families-each-engine-serves)).

The SGLang server needs `SGLANG_TRUST_REMOTE_CODE=1` (the repo ships its modeling code), and its routed-experts capturer raises at the first capture for this family, so serve without `SGLANG_ENABLE_R3` and replay routing with `recompute` there.

## Configs

`inclusionAI/Ling-mini-2.0` (16B, 256 experts) trains under EP=8 at 24K max length with `moe_balancing: bias_update`: `examples/sft/bailing/ling-mini-2-ultrachat-ep.yaml`.

The shipped config pins the toolkit's own `jinja-templates/ling/ling-multiturn.jinja` with `force_chat_template: true` (`ling-instruct.jinja` is its single-turn sibling and raises on multi-turn data; `ling3-instruct.jinja` is the Ling 3.0 template), because the hub templates drift across revisions and the training template must match the served one. `ling-native.jinja` is the verbatim upstream `inclusionAI/Ling-mini-2.0` template (system messages + tools) for runs that need the exact hub render.

Ling 3.0 has no shipped example config — start from `ling-mini-2-ultrachat-ep.yaml` and apply the Ling-3.0 settings above. A gathered save of Ling-3.0-flash (~122B) is ~245 GB; check the target volume before launching.
