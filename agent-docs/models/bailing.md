# Bailing MoE / Ling

`BailingMoeV2ForCausalLM` — Ant Group's MoE with standard softmax GQA attention (16 heads / 4 KV / head_dim 128), sigmoid routing (256 experts in 8 groups; the top-4 groups are chosen, then the top-8 experts within them), and 1 shared expert. Loaded from the `inclusionAI/*` repos under `trust_remote_code=True`.

| | EP | CP | TP | ETP | PP |
|---|:--:|:--:|:--:|:--:|:--:|
| Bailing MoE / Ling 2.0 | Yes | Yes | **No** | Yes | — ¹ |
| Ring-mini-linear-2.0, Ling 3.0 | Yes | **No** | **No** | Yes | — ¹ |

¹ Pipeline parallelism is [not yet available in this release](../parallelism/pipeline-parallelism.md). The shipped split contract binds no stage boundary for any variant — only Ling 2.0's layer stack is uniform (Ling 3.0 alternates KDA and MLA attention on a `layer_group_size` period, Ring interleaves 16 linear-attention layers with 4 full-attention ones), but the alternation lives in the layer modules themselves, not in a `layer_types` list that mask selection indexes by position, and a stage is built by slicing the live `ModuleList`, so every layer keeps its own type wherever the cut lands. Its gate is instead multi-token prediction: Ling 2.0 and Ling-3.0-tiny ship `num_nextn_predict_layers: 0`; Ling-3.0-flash ships `1`, which the gate refuses unless set to `0` in `model_init_kwargs`.

The shipped config trains `inclusionAI/Ling-mini-2.0` (`model_type: bailing_moe`). The sibling `inclusionAI/Ring-mini-linear-2.0` is a **different** architecture (`BailingMoeLinearV2ForCausalLM`, `model_type: bailing_moe_linear`) that swaps softmax attention for Lightning Attention-2. Both share the `BailingMoeV2SparseMoeBlock` the EP wrapper targets.

## Ling 3.0

`BailingMoeV3ForCausalLM` (`model_type: bailing_hybrid`) — `inclusionAI/Ling-3.0-tiny` (7.9B, 128 experts) and `inclusionAI/Ling-3.0-flash` (~122B, 512 experts). The MoE block is unchanged from V2 — same per-expert `gate_proj`/`up_proj`/`down_proj` modules, same gate arithmetic (sigmoid, `expert_bias` on selection only, group-limited top-k, renormalization, `routed_scaling_factor`) — so `EPBailingMoELayer` claims `BailingMoeV3SparseMoeBlock` too and everything above applies. The V3 block returns `(hidden, router_logits)` where V2 returns a tensor; its decoder layer tests the return for a tuple, so the wrapper's bare tensor is accepted.

Attention is what differs: layers alternate in groups of `layer_group_size`, with most layers `BailingMoeV3KimiDeltaAttention` (KDA linear attention through `fla`) and the rest `BailingMoeV3MultiLatentAttention` (MLA).

The KDA kernels accept a `cu_seqlens` kwarg (fla convention) that the toolkit collators do not yet emit, so under packing the recurrence scans the whole row and mixes across document boundaries — an emission gap, not an architectural limit. KDA is the majority of the stack, so that term dominates whatever the MLA layers do ([Document isolation](../data/collators.md#document-isolation-under-packing)).

Three consequences for a training config:

- **Pin `attn_implementation: sdpa`.** The modeling file declares only the v4-era `_supports_flash_attn_2`, which transformers v5 ignores in favor of `_supports_flash_attn`, so an auto-selected FA4 is refused at model build. The KDA layers run their own `fla` kernels either way.
- **Set `fp32_non_ep_params: true`.** The KDA layers hold `A_log` and `dt_bias` in fp32 and `from_pretrained` does not unify them, so the model reaches FSDP2 with mixed parameter dtypes and `fully_shard` asserts one original dtype per shard group. Upcasting the non-expert parameters to fp32 masters is what makes the wrap legal; compute stays bf16. At `expert_parallel_size: 1` pair it with **`fsdp_shard_ep1_experts: false`**: the upcast skips every EP-wrapper parameter, so FSDP-managed replicated experts would sit bf16 inside the same fp32 shard group and `ParallelismConfig` refuses the combination at config time. Above ep1 the knob has no effect — the experts are FSDP-ignored anyway.
- **Override `rope_scaling`.** The config ships `rope_scaling: null`, which transformers v5 normalizes into a dict carrying no `"factor"`; the MLA layers then read `config.rope_scaling["factor"]` and raise `KeyError`. Pass the minimal replacement and nothing more:

    ```yaml
    model_init_kwargs:
      rope_scaling: {rope_type: default, factor: 1.0, mscale_all_dim: 0}
    ```

    The v5 setter stores this dict verbatim in `rope_parameters`, so adding `rope_theta` or `partial_rotary_factor` to it overrides what the rotary module sets for itself and halves the rotary dimension. The minimal form reproduces stock frequencies exactly, and `mscale_all_dim: 0` keeps `factor` unused.

`expert_swiglu_limit_list` / `share_expert_swiglu_limit_list` appear in the Ling-3.0-flash config but are read nowhere in the shipped modeling code, so no clamp is applied and the wrapper stays equivalent. Both repos need `model_revision` pinned — these workarounds depend on the remote code's exact internals.

## EP wrapper

`EPBailingMoELayer` (`src/distributed/expert_parallel/layers/bailing.py`) replaces `BailingMoeV2SparseMoeBlock`.

- Routing: `BailingMoeV2Gate` — sigmoid scoring, group-limited top-k, expert bias, normalization.
- Routed experts: per-expert `BailingMoeV2MLP` modules stacked into 3D tensors at init and transposed to matmul convention (`gate_proj [E, H, M]`, `up_proj [E, H, M]`, `down_proj [E, M, H]`). Compute via 3-call Grouped GEMM.
- Shared experts: replicated per rank and added after DeepEP combine (dispatch overlap is the opt-in `HALO_EP_SHARED_OVERLAP` side stream).
- Checkpoint round-trip: the lazy loader materializes only this rank's expert range and `_init_expert_params` stacks it into a 3D slice. On save, `gather_expert_state_dict` (via `_gather_individual_glu_state_dict`) gathers, transposes back to `F.linear` convention, and re-splits to per-expert `experts.{i}.{gate,up,down}_proj.weight`.

## CP wrapper

`BailingMoeV2UlyssesAttention` (`src/distributed/context_parallel/layers/bailing.py`) wraps Ling 2.0's attention. The math is plain softmax GQA; what the wrapper carries is the family's spellings — a fused `query_key_value` projection split into Q/K/V, an output projection named `dense`, Q/K RMSNorm applied before a partial rotary, and a softmax scale the module never stores because its own forward computes it inline. All three eager/sdpa/FA2 subclasses are claimed: they hold the same weights, and CP replaces the forward either way.

The shipped label is always `sdpa` — the remote code declares only the v4-era `_supports_flash_attn_2`, which transformers v5 ignores — so this wrapper sets `REQUIRES_FLASH_ATTN_LABEL = False`. CP calls the flash kernel itself, so that label describes a path CP has replaced; every other family keeps the check.

Two settings a CP run needs:

```yaml
attn_implementation: sdpa       # the auto-detected FA4 is refused at model build
context_parallel_size: 2        # ≤ 4 on Ling-mini-2.0 — cp_size must divide the 4 KV heads
```

The EP lazy loader retries SDPA after that refusal on its own; the CP loader does not, so a CP config that leaves the label unset, or pins a flash one, raises `BailingMoeV2ForCausalLM does not support Flash Attention 2 yet`.

`Ring-mini-linear-2.0` is **rejected**, not wrapped: its file reuses Ling 2.0's full-attention class names, so validation matches `BailingMoeV2LinearAttention` by name to avoid wrapping the few full-attention layers while the Lightning-Attention-2 recurrence scans each rank's shard in isolation. (`finalize_loaded_model()` recomputes its slope buffers on every load path.) Ling 3.0 is likewise unavailable — its KDA layers are a linear recurrence and its MLA layers carry no wrapper.

## Why TP doesn't work

No Bailing attention class is registered for TP (`TP_SHARDABLE_ATTENTION_CLASSES`, `src/distributed/tensor_parallel/module_types.py`), so TP does not patch Bailing attention. The gap is a missing DTensor plan, not an architectural blocker — the shipped model is standard softmax GQA.

## ETP

`EPBailingMoELayer._init_expert_params` stacks the per-expert modules through `_store_separate_glu_params`, which at `expert_tp_size > 1` slices `gate_proj` / `up_proj` along the intermediate dim `M` and `down_proj` to match; the shared expert stays replicated. Use pure ETP (`ep_size=1`) when expert memory is the bottleneck. See [ETP](../parallelism/expert-tensor-parallelism.md#limitations).

## Remote-code compatibility

These checkpoints need `trust_remote_code=True` and call transformers-v5-removed APIs (`is_torch_fx_available`, `ROPE_INIT_FUNCTIONS['default']`). `apply_remote_code_compat_shims()` restores both and the model-loading utilities call it automatically — no vendoring required. Every Bailing/Ling class also ships the v4-era list form of `_tied_weights_keys`, which transformers 5 reads as a `{duplicate: source}` dict and which crashes `save_pretrained` outright; the shims convert it to `{}` (these checkpoints store `lm_head` untied) and the loader refuses a config declaring `tie_word_embeddings: true` that carried the list, rather than training the head untied.

`Ring-mini-linear-2.0` needs a shim of a different kind: its modeling file calls `DynamicLayer()` but imports only `Cache` and `DynamicCache`, so the bare global raises `NameError` on every **cached** forward — `generate`, and any check on a reloaded model that does not pass `use_cache=False`. Training keeps the cache off, so the break appears only after a run finishes. The API shims patch transformers, which cannot work here (the name is never imported); this one binds the class into the loaded module through `get_class_in_module`, the funnel every remote class passes. Absent names only — a file defining its own is untouched.

A bare `from_pretrained` of it is not a usable model either: transformers 5 re-materializes every non-persistent buffer as `torch.empty_like`, and the file's `_init_weights` overrides transformers' without `super()`, so the Lightning-Attention-2 decay `slope` and the rotary `inv_freq` are never computed. Zeroed pages give finite logits from a model whose RoPE is dead; a reused allocator gives all-NaN logits, `exp(cumsum(g))` overflowing on the garbage slope. Every toolkit load path repairs both in `finalize_loaded_model()` — a plain `AutoModelForCausalLM` + `PeftModel` reload of a trained adapter has to call it as well.

Ling 3.0's `modeling_bailing_moe_v3` needs one more: the file copies HF's `eager_attention_forward` helper **without** the `ALL_ATTENTION_FUNCTIONS` dispatch, so `attn_implementation: sdpa` changes the mask format while every MLA layer still materializes the `[B, H, S, S]` score plane plus its fp32 softmax copy — ~190 GiB on a packed 80k row, an instant OOM.

The SDPA dispatch shim (bound through the same funnel) wraps the module global with an SDPA route gated on the attention module's own config: `sdpa` runs `F.scaled_dot_product_attention` (linear memory), anything else — including any `output_attentions` call — keeps the file's own eager, and a file carrying a real dispatch is untouched. SDPA consumes no cu_seqlens, so keep `per_device_train_batch_size: 1` under packing: the flatten makes batch>1 one wider row, doubling the quadratic MLA cost.

The KDA layers run fla's Triton kernels, whose autotuners re-benchmark on shape-derived keys — the causal-conv1d key includes `ceil(tokens/1024)`, so a fresh packed-row width costs that rank tens of seconds of autotuning while its 7 peers spin in the next collective at idle power. Measured configs persist in the Triton disk cache, which `setup_training_environment` anchors on `HF_HOME` (`TRITON_CACHE_DIR`), so the warm-up is paid once per volume and the slow-step pattern appears only against a fresh cache.

## Router balancing

Bailing is aux-loss-free by design (`topk_method: noaux_tc`): no modeling variant computes a load-balancing loss and the config carries no `router_aux_loss_coef`. The family's native mechanism is the gate's persistent `expert_bias` buffer, added to the sigmoid scores for **selection only** — combine weights stay unbiased.

`EPBailingMoELayer` hands exactly that buffer to `RouterBiasBalancingCallback`: under `moe_balancing: bias_update` (or `auto`, which resolves there whenever the wrapper is applied) the DeepSeek-V3 sign-updates land in `expert_bias` itself, upcast to fp32 at enable so the 1e-3 steps survive the add. The wrapper exposes it as a live property, so callback updates and the `router_balancing_biases.pt` resume-restore always reach the buffer the gate reads. Because the buffer is part of the checkpoint, a gathered save exports the **final** bias and a served checkpoint routes exactly as training did.

Most wrapper families adopt a native slot the same way. The exceptions are Qwen3, Qwen3.5/3.6, Mistral4 and Cohere2 MoE, whose routers carry no such slot — strict `bias_update` raises there and only the trainer-only `bias_update_transient` runs, its bias never leaving the run — and Gemma 4, which has no balancing route at all. `moe/*` load metrics come with the callback.

An explicit `aux_loss` warns and stays off (there is no aux term to enable). Without the EP wrapper — `use_grouped_gemm: false` at `expert_parallel_size: 1` — nothing counts expert loads, so `bias_update` raises at setup; freezing the gate is the fallback mitigation there:

```yaml
freeze_layers_patterns:
  - "*.mlp.gate.weight"
```

## Serving and weight sync

Only **Ling 2.0** (`bailing_moe`) is servable and syncable: vLLM 0.26.0 registers `BailingMoeV2ForCausalLM`, and gathered saves land in its per-expert expert loader. **Ling 3.0** (`bailing_hybrid`) has no model class in either pinned engine, and **Ring**'s checkpoints declare `BailingMoeLinearV2ForCausalLM` where both engines register `BailingMoeV2_5ForCausalLM` — an architecture-string mismatch. RL weight sync is refused at trainer construction for both spellings (`_WEIGHT_SYNC_UNSUPPORTED_MODEL_TYPES`): the server cannot load the base model, so no broadcast could land ([Rollout Servers](../infrastructure/rollout-servers.md#weight-sync)).

## Configs

`inclusionAI/Ling-mini-2.0` (16B, 256 experts) trains under EP=8 at 24K max length with `moe_balancing: bias_update`: `examples/sft/ling_mini_2/ling-mini-2-ultrachat-ep.yaml`.

The Ling configs pin the toolkit's own chat templates with `force_chat_template: true` — `jinja-templates/ling/ling-instruct.jinja` / `ling-multiturn.jinja` (Ling-mini-2) and `jinja-templates/ling/ling3-instruct.jinja` (Ling 3.0) — because the hub templates drift across revisions and the training template must match the served one. `ling-native.jinja` is the verbatim upstream `inclusionAI/Ling-mini-2.0` template (system messages + tools) for runs that need the exact hub render.

Ling 3.0 has no shipped example config — start from `ling-mini-2-ultrachat-ep.yaml` and add the three Ling-3.0 settings above. A gathered save of Ling-3.0-flash (~122B) is ~245 GB — check the target volume before launching.
