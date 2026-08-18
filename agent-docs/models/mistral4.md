# Mistral4 (text backbone of Mistral3 VLMs)

`Mistral4ForCausalLM` is the text side of `Mistral3ForConditionalGeneration` — the LM in `mistralai/Mistral-Small-4-*` and any `mistral3` VLM. It is a DeepSeek-V3 clone: softmax + group-topk routing with an always-on shared expert, MLA (multi-head latent attention) with llama-4 logarithmic position scaling, and YARN RoPE.

| | EP | CP | TP | ETP | PP | EP+CP | EP+TP | EP+ETP |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Mistral4 MoE | Yes | Yes (legacy path) | Yes (selective) | Yes (pure, `ep_size=1`) | — ² | Yes ¹ | Yes | Experimental |

² Pipeline parallelism is [not yet available in this release](../parallelism/pipeline-parallelism.md). The decoder layers are homogeneous, so the shipped PP split contract carries no constraint of its own.

EP+ETP (`ep_size>1` and `expert_tp_size>1`) is experimental and node-local; it cannot combine with attention TP (`tp_size>1`). The Mistral4 suite runs `ep_size=2, expert_tp_size=4`; the sibling `ep_size=4, expert_tp_size=2` is a legal shape that this model's matrix does not cover.

`ep4+etp2` clears the racy-EP gate because ETP raises `ep_group_size` to the domain, even though DeepEP still forms the two 4-rank dispatch groups the gate rejects for bare `ep4` — and that validation does not clear bare `ep4` on 8 (`is_racy_single_domain_multigroup_ep`). See [Expert Tensor Parallelism](../parallelism/expert-tensor-parallelism.md).

¹ Node-local EP+CP requires `ep_group_size == nvlink_domain_size`, so on 8-GPU nodes `ep_size=8` exactly; `cp_size` then only has to divide the domain.

Coverage: `tests/gpu/parallelism/test_mistral4_all_parallelism.py` (one node per single-node mode under `pytest -m "gpu and mistral4"`; EP+CP coverage lives in the Cohere2 MoE matrix) and `tests/gpu/parallelism/test_mistral3_vision_smoke.py` (Pixtral + text forward/backward).

## EP wrapper

`EPMistral4MoELayer` (`src/distributed/expert_parallel/layers/mistral4.py`) replaces `Mistral4MoE`.

- Routing: the wrapper calls `Mistral4TopkRouter` and keeps only its logits (the router also returns weights and indices, which balancing and routing replay have to recompute); `route_tokens_to_experts` then does softmax + group-topk inline, mirroring the router's own arithmetic. No `e_score_correction_bias` (unlike GLM-4); `norm_topk_prob` and `routed_scaling_factor` are honored.
- Load balancing: Mistral4 has **no bias slot** (a plain softmax router, no correction-bias tensor), so `moe_balancing: bias_update` **raises** and the explicit `bias_update_transient` is the opt-in — the bias steers training-time routing, resumes from `router_balancing_biases.pt`, and never reaches a checkpoint or a served copy (near-tied top-k picks flip between trainer and server). `Mistral4ForCausalLM.forward` declares no `output_router_logits` parameter either, so `aux_loss` cannot work and `auto` resolves to `none` with a warning naming the transient opt-in. See [RouterBiasBalancingCallback](../training-methods/callbacks.md#routerbiasbalancingcallback).
- Routed experts: `Mistral4Experts`, fused `gate_up_proj [E, 2M, H]` (F.linear convention; matmul-internal form `[E, H, 2M]`) and `down_proj [E, H, M]`. Same layout as Qwen3.5 / GLM-4, so the shared `EPMoELayerBase` fused-GLU helpers cover compute (Grouped GEMM on SM90+).
- Shared expert: `Mistral4MLP`, replicated per rank, summed after DeepEP combine. Intermediate size = `moe_intermediate_size * n_shared_experts`.
- Checkpoints: weights stored in matmul convention internally; `gather_expert_state_dict` transposes back to F.linear convention at save so `from_pretrained` reloads as-is. The public `mistralai/Mistral-Small-4-119B-*` checkpoint ships FP8 expert weights — dequantize once with `scripts/before_training/convert_mistral4_bf16.py` (streams shard-by-shard, handles scalar and per-expert `_scale_inv` layouts) before training.
- Activation: the routed-expert SwiGLU runs through the fused Triton kernel (`src/kernels/fused_glu.py`) when the gate is a genuine SiLU, logged as `glu_combine=fused_silu_mul`.

## Packing

Packed documents stay isolated on both paths, but flash needs a toolkit patch:
`Mistral4Attention.forward` takes `position_ids` as an explicit parameter and forwards only
`**kwargs` to the attention interface, so the varlen detection never sees them.
`patch_mistral4_flash_packed_position_ids` (`src/models/patches/attention.py`) re-injects them
and is applied at load whenever the resolved implementation is a flash one. The dense path builds the
packed mask itself and is clean to grouped-GEMM reduction noise.

## CP wrapper

`Mistral4Attention` → `Mistral4UlyssesAttention` (`src/distributed/context_parallel/layers/mistral4.py`) — legacy `[B, H, S, D]` path (`_optimize_attention = False`).

Three MLA-specific bits the wrapper handles:

- **Mismatched head dims** — `qk_head_dim` (nope + rope) and `v_head_dim` can differ. Flash-attn's native GQA wants one head_dim, so V is padded to `qk_head_dim` before the kernel and the output cropped back.
- **Shared rope head** — `kv_a_proj_with_mqa` produces one rotary K vector broadcast to all heads. The broadcast is materialized *before* the Ulysses all-to-all so the head-dim scatter has a contiguous tensor.
- **Llama-4 position scale** — transformers' own `get_llama_4_attn_scale` multiplies Q after RoPE. The legacy path applies it after the all-to-all, where Q spans the whole sequence, so `UlyssesCPModelWrapper` publishes the full `position_ids` on every patched layer once per forward (no per-layer gather). A `rope_parameters` without `llama_4_scaling_beta` / `original_max_position_embeddings` is refused at construction rather than trained unscaled.

RoPE follows `config.rope_interleave`: rotate-half (`apply_rotary_pos_emb`) or interleaved pairs (`apply_rotary_pos_emb_interleave`).

## TP

Selective TP via `apply_tp_to_attention_only()` — `Mistral4Attention` is in `TP_SHARDABLE_ATTENTION_CLASSES` (`src/distributed/tensor_parallel/module_types.py`). MLA reuses GLM-4's compress/expand layout, so the plan applies verbatim:

- **Sharded** (ColwiseParallel): `q_b_proj`, `kv_b_proj` (head-dim expansions).
- **Replicated**: `q_a_proj`, `kv_a_proj_with_mqa`, `q_a_layernorm`, `kv_a_layernorm` — low-rank compressions where sharding produces degenerate per-rank shapes.
- **Sharded** (RowwiseParallel): `o_proj`.

MoE blocks (`Mistral4MoE`, `Mistral4Experts`, `Mistral4MLP`) are skipped — they belong to EP, not TP.

## ETP

Mistral4's expert layout matches GLM-4 / LFM-2 (contiguous halves `[gate(M) | up(M)]` in matmul-convention `gate_up_proj [E, H, 2M]`), so ETP works automatically via `EPMoELayerBase._init_fused_glu_params`: with `expert_tp_size > 1` the helper splits halves before sharding and stores `gate_proj` / `up_proj` / `down_proj` separately on each ETP rank (intermediate slice `[rank*M/tp : (rank+1)*M/tp]`). On save, gate/up shards are all-gathered, concatenated back into contiguous halves, transposed to F.linear convention. The shared expert (`Mistral4MLP`) stays replicated. See [Expert Tensor Parallelism → Compatibility](../parallelism/expert-tensor-parallelism.md#limitations).

## Liger kernel

Liger upstream ships no `mistral4` applier; the toolkit covers `mistral3` / `mistral4` from its own spec in `src/kernels/liger/families.py`. With `use_liger_kernel: true` it swaps `Mistral4RMSNorm` for Liger's fused RMSNorm, `Mistral4MLP` for the toolkit's fused SwiGLU (the shared expert's `(config, intermediate_size=None)` constructor is inherited, not restated), routes cross-entropy through Liger, and offers the fused loss on `Mistral4ForCausalLM` (off by default — `mistral3` checkpoints load as `Mistral3ForConditionalGeneration`, whose own head runs instead).

RoPE stays unfused — Mistral4 combines YARN with the llama-4 logarithmic scale, which Liger's `liger_rotary_pos_emb` doesn't model. The spec declares `rope=False`, and the orchestrator honors an applier's own `rope` default over its generic default-on, so `rope` stays off unless the user explicitly pins `rope: true` in `liger_kernel_config` — which the applier then refuses with `NotImplementedError`.

## Model loading

Both `Mistral3ForConditionalGeneration` and `Mistral4ForCausalLM` are registered upstream under `AutoModelForImageTextToText`, not `AutoModelForCausalLM`. The loaders resolve the class via `resolve_auto_model_class()` — any `model_type` in transformers' `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES` loads through `AutoModelForImageTextToText`, everything else falls back to `AutoModelForCausalLM`. That keeps the `*ForConditionalGeneration` wrapper intact, so saved checkpoints carry the right `architectures` entry and never drop the vision tower.

| `model_type` | Class |
|---|---|
| `mistral3` | `Mistral3ForConditionalGeneration` (Pixtral vision + Mistral4 text) |
| `mistral4` | `Mistral4ForCausalLM` (text only — `mistral4` *is* the Mistral text decoder, so nothing is lost) |

Pass `model_class=` to override; the same resolver backs `scripts/before_training/patch_vocab.py`.

Inside the multimodal wrapper the text backbone is at `model.model.language_model`. The EP / CP / TP / FSDP plumbing checks that path first (then its `.layers`), falling back to `model.model.layers` for the standalone causal LM, so VLM checkpoints get the same selective TP / FSDP2 treatment.

## Configs

Mistral-Small-4-119B (`mistralai/Mistral-Small-4-119B-2603`) trains under EP at 32K max length. The expert count (128 routed + 1 shared, top-4) divides evenly across 8 and 16 ranks, so EP=8 (16 experts/rank) and EP=16 are the natural fits. Example: `examples/sft/mistral4/mistral-small-4-119b-ultrachat-ep.yaml`.

## Serving

vLLM 0.26.0 knows nothing of this family: `mistral4` appears in neither its architecture registry nor any `model_type` mapping. The composite `Mistral3ForConditionalGeneration` **is** registered, but it builds its text tower through `init_vllm_registered_model` off `text_config`, whose `mistral4` spelling resolves to no class. The public `mistralai/Mistral-Small-4-*` repo serves only because vLLM detects its Mistral-native `params.json` layout and routes the whole model to `PixtralForConditionalGeneration`; a toolkit export is plain HF-format and has no such path. vLLM's generic transformers backend (`--model-impl transformers`) is the one remaining route, neither pinned nor verified here.

**RL weight sync is refused at construction** (`_supports_weight_sync = False` on `EPMistral4MoELayer`): the sync forwards hub-spelled tensors into the server's own loader, and there is none for this family. Online and environmental GRPO on Mistral4 raise at trainer build rather than sync into a server that cannot hold the model ([Rollout Servers](../infrastructure/rollout-servers.md#weight-sync)). SFT, preference and offline methods are unaffected — they never touch a rollout engine.
