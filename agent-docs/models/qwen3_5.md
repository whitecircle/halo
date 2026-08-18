# Qwen3.5 / Qwen3.6 MoE

A separate class hierarchy from Qwen3 (`Qwen3_5MoeForCausalLM`, `Qwen3_5MoeSparseMoeBlock`, `Qwen3_5MoeAttention`). Three traits drive parallelism: a sigmoid-gated shared expert, a double-width `q_proj` fusing query with a sigmoid gate, and a hybrid `layer_types` scheme interleaving full attention with `Qwen3_5MoeGatedDeltaNet` linear-attention layers.

| | EP | CP | TP | ETP | PP | EP+TP |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| Qwen3.5 / 3.6 MoE | Yes | **No** ¹ | Yes | Yes | — ² | Yes |
| Qwen3.5 / 3.6 dense | — | **No** ¹ | Yes | — | — ² | — |

¹ The full-attention block has a working CP wrapper; the hybrid linear-attention layers are the blocker — see [Why CP is blocked](#why-cp-is-blocked-on-real-checkpoints).

² Pipeline parallelism is [not yet available in this release](../parallelism/pipeline-parallelism.md). Its shipped contract admits a text-only run of the multimodal checkpoints — the vision tower and projector are held by no stage, stashed on the save rank for re-emission in every checkpoint, so an export reloads as the composite class — and refuses any image evidence: an image column, embedded image parts, or a collator consuming one. The hub's `mtp_num_hidden_layers` is metadata (the MTP weights are dropped at load) and passes the live-MTP gate. The split contract binds stage boundaries to whole periods of the period-4 `layer_types` pattern, and its collator gate refuses `packing` for this family: PP keeps the packed rows instead of flattening them, and the delta rule's varlen `cu_seq_lens` have no per-row convention ([Collators](../data/collators.md#document-isolation-under-packing)).

## Architecture

Each decoder layer is *either* full softmax attention *or* a linear-attention block, interleaved by config. Both share the same MoE MLP path (`Qwen3_5MoeSparseMoeBlock`): 256 routed experts + 1 sigmoid-gated shared expert, top-k=8.

`Qwen3_5MoeTextConfig` builds `layer_types` from `full_attention_interval` (default 4) — every fourth layer is full attention, so Qwen3.6-35B-A3B (40 layers) gets 10 full + 30 linear.

### Full attention (`Qwen3_5MoeAttention`)

| Feature | Detail |
|---|---|
| `q_proj` | Double-width `num_q_heads * head_dim * 2`, chunked into `(query, gate)`; output becomes `attn_output * sigmoid(gate)` before `o_proj` |
| RoPE | Partial — only the first `head_dim * partial_rotary_factor` channels rotate (default `0.25`) |
| Heads | 16 Q heads, 2 KV heads (GQA factor 8), which caps TP at `tp_size=2` |

> **FlashAttention-4 is unsupported for this family on Blackwell.** FA4's backward emits NaN gradients for the combination head_dim 256 + partial RoPE + output gate + GQA 16:2. `model_fa4_backward_nan_prone` (`src/models/patches/attention.py`) demotes the load to SDPA, keyed on the `qwen3_5*` / `qwen3_next*` `model_type` prefixes. Qwen3.6 is covered by those: it reuses the 3.5 classes and ships under the 3.5 types — `Qwen/Qwen3.6-35B-A3B` declares `qwen3_5_moe` with a `qwen3_5_moe_text` tower. On Hopper the standard detector picks FA3 when installed, else FA2.

### Linear attention (`Qwen3_5MoeGatedDeltaNet`)

A gated delta-rule recurrent attention with a sequence-axis causal Conv1d on the input. The fast path is gated on the `fla` / `causal-conv1d` imports (both pinned in `pyproject.toml` and installed in the images, so it is on by default). The torch fallback matches it only on single-document rows: it takes neither `seq_idx` nor `cu_seq_lens_q`, so a multi-document row mixes through both the conv and the delta-rule scan while attention stays isolated. The collator factory therefore **refuses** `packing` and `padding_free` for this family when either wheel is missing ([Document isolation](../data/collators.md#document-isolation-under-packing)).

These layers need no CP for memory: the recurrent state is independent of `S`, per-layer activations are `O(B·S·d)`, and the Conv1d left-context is a 3-token state. The long-context ceiling comes from the 10 full-attention layers — 33K on a single B200/B300 rank at `num_heads=16, head_dim=256, num_kv_heads=2`.

## Model loading

Qwen3.5/3.6 ship as multimodal: `model_type` `qwen3_5_moe` (or `qwen3_5` dense), top-level class `Qwen3_5MoeForConditionalGeneration`, registered under `AutoModelForImageTextToText`. `resolve_auto_model_class()` keeps the vision tower intact, and the config detection routes the **load** through `load_distributed_model`, adding the processor on top (`src/distributed/loading/vlm_setup.py`). The MoE VLM therefore gets the same EP/TP support as a standalone causal LM — `config_has_experts` finds the experts on the nested `text_config`.

That verdict does not decide the data path: a text-only dataset on these checkpoints is a text run, which is what lets the text SFT recipes below use `packing` ([SFT — VLMs](../training-methods/sft.md#vision-language-models)).

The resolver keys on `model_type` alone, not `architectures`. A `*-patched` checkpoint whose `config.json` carries the text-only `model_type` `qwen3_5_moe_text` is absent from the mapping, falls back to `AutoModelForCausalLM`, and drops the vision tower — regenerate it through the resolver.

## EP wrapper

`EPQwen3_5MoELayer` (`src/distributed/expert_parallel/layers/qwen3_5.py`) replaces `Qwen3_5MoeSparseMoeBlock`. Softmax top-k routing via `Qwen3_5MoeTopKRouter`. The shared expert (`Qwen3_5MoeMLP`, sigmoid-gated output) stays local on every rank and is combined with routed-expert output after DeepEP combine. Routed experts use the fused `gate_up_proj [E, 2M, H]` + `down_proj [E, H, M]` layout, sharded along the expert dimension, SwiGLU, grouped GEMM on SM90+.

The wrapper re-derives selection from the router's own logits — top-k on the bias-adjusted probabilities, gate weights from the unbiased renormalized softmax, which makes a zero bias a no-op.

The architecture has **no bias slot** (the gate is a bare weight), so the trained bias can only be trainer-side: `moe_balancing: bias_update` **raises**, and the explicit `bias_update_transient` is the opt-in. Balancing works during training and resumes exactly, while every exported checkpoint serves without the bias (near-tied top-k picks flip vs training).

On the multimodal checkpoints `aux_loss` cannot work either — `Qwen3_5MoeForConditionalGeneration.forward` declares no `output_router_logits` parameter, so an explicit `aux_loss` raises and `auto` resolves to `none` with a warning naming the transient opt-in. The text-only `Qwen3_5MoeForCausalLM` declares the parameter and stays on `aux_loss` under `auto` ([Callbacks](../training-methods/callbacks.md#moe-balancing-modes)). `text_only_model: true` loads a VLM checkpoint through that CausalLM class deliberately — the vision tower and MTP tail are dropped from the build **and from the export**: the artifact carries no `processor_config.json` and no vision token ids. The two pinned engines differ on that export. **vLLM 0.26.0** registers only `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration`, so serving it there needs `scripts/after_training/reattach_vision_tower.py` first — it re-prefixes the trained text weights to `model.language_model.*` and streams the base's untrained vision tower and MTP tail back in. **SGLang 0.5.17** registers the text-only `Qwen3_5MoeForCausalLM` / `Qwen3_5ForCausalLM` and serves the export unchanged. Image-bearing datasets are refused loudly (the text path would otherwise prune the column silently), and the PP VLM refusal does not apply (there is no tower to strand). `aux_loss` becomes the exported-by-construction balancing.

## CP wrapper

`Qwen3_5MoeUlyssesAttention` (`src/distributed/context_parallel/layers/qwen3_5.py`) replaces `Qwen3_5MoeAttention` / `Qwen3_5Attention` and handles two quirks:

- **Double-width `q_proj`** — the output is viewed as `[..., num_q_heads, head_dim * 2]` and chunked into query + gate. The query goes through Ulysses (q_norm + RoPE + all-to-all + flash-attn + all-to-all back), then `attn_output * sigmoid(gate)` before `o_proj`. The gate stays on the local sequence shard, so no extra communication.
- **Partial RoPE** — only the first `rotary_dim` channels of Q/K rotate; the rest is concatenated unrotated.

Correctness coverage against a synthetic all-full-attention config: `tests/gpu/parallelism/cp/test_qwen3_5_cp_correctness.py`.

## Why CP is blocked on real checkpoints

Every released checkpoint ships hybrid `layer_types`, and both of `Qwen3_5MoeGatedDeltaNet`'s sequence-axis ops cross Ulysses chunk boundaries:

- **`causal_conv1d_fn`** kernel 4 — output `t` depends on `t..t-3`, so each rank's first 3 positions need the previous rank's last 3 input tokens. Without a halo exchange the kernel sees zero-padding.
- **`chunk_gated_delta_rule`** is a recurrent scan — chunk `c+1` must start from chunk `c`'s final state, which a CP rank cannot recover.

`validate_model_for_ulysses` rejects any model whose `layer_types` contains `"linear_attention"`. A halo-exchange wrapper could fix this (`fla` / `causal-conv1d` upstream have no CP-aware variants) but none ships. Workarounds: train at the single-rank ceiling (32K–33K on B200/B300, EP=8, no CP), or train from scratch with `layer_types=['full_attention'] * num_hidden_layers` (discards all pretrained linear-attention weights, so from-init only).

## TP and ETP

Attention-only TP works: `Qwen3_5MoeAttention` and dense `Qwen3_5Attention` are in the TP accept-list, and the MoE block is skipped because EP owns it. The double-width `q_proj` is ColwiseParallel-compatible — the split is per head, so each head keeps its full `head_dim * 2` (query + gate) on one rank. Only the full-attention layers shard: the linear-attention layers stay replicated on every rank, so the per-rank footprint falls by far less than `1/tp_size` (the TP path warns).

ETP is supported: the routed experts use the fused-GLU contiguous-halves layout, so `_init_fused_glu_params` splits the halves before sharding and stores `gate_proj` / `up_proj` / `down_proj` separately when `expert_tp_size > 1`. The sigmoid-gated shared expert stays replicated. See the [ETP guide](../parallelism/expert-tensor-parallelism.md#limitations).

## Configs

The shipped EP config trains Qwen3.5/3.6 35B-A3B (40 layers = 10 full + 30 linear, 256 routed experts, top-k=8) at 33K max length under EP=8 (32 experts/rank); set `fp32_non_ep_params: true` for stable long context.

Pin `attn_implementation: flash_attention_2` with `packing: true` (fixed-length): the M-RoPE varlen path crashes FA2 with `cudaErrorIllegalAddress`, so do not set `padding_free: true` without revalidating. SDPA reaches parity at `b=1`.

Packing isolates the full-attention layers via the packed mask, and `Qwen3_5MoeGatedDeltaNet` receives its boundaries from the collators: the packing/padding-free collators emit `seq_idx` (its causal conv) and `cu_seq_lens_q` (its chunked delta rule) for the family, the two kwargs upstream reads and nothing model-side derives ([Document isolation](../data/collators.md#document-isolation-under-packing)). The same holds for Qwen3-Next.

Examples: `examples/sft/qwen3_5/qwen3.5-35b-a3b-ultrachat-ep.yaml`. `qwen3.5-122b-a10b-ep.yaml` scales the same shape to 122B-A10B at EP=8 single-node, or EP=16 with `ep_scope: global` across two Hopper nodes; it keeps the experts bf16 (`fp32_experts` off) — fp32 expert masters fit the 35B shape but OOM at 122B. VLM SFT on Qwen3.5-9B: `qwen3.5-9b-vl-ocr-olmocr.yaml`, `qwen3.5-9b-vl-docvqa.yaml`. Online GRPO smoke: `examples/grpo/online/qwen3_5/online-grpo-qwen3.6-35b-a3b-smoke.yaml`.

The family also carries the per-method canonical examples: Qwen3.5-9B (dense) for DPO/SMPO/KTO (`examples/preference/qwen3_5/`), reward, classification, and distillation; Qwen3.6-35B-A3B for the GRPO task configs.
