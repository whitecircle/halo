# Cohere2 MoE (Command A+)

`Cohere2MoeForCausalLM` (`model_type: cohere2_moe`) is the text backbone of
`CohereLabs/command-a-plus-05-2026-bf16` — a `Cohere2VisionForConditionalGeneration` VLM
(`cohere2_vision`) with a SigLIP tower. Native in transformers, no `trust_remote_code`.

The block is Cohere's parallel residual: one `input_layernorm`, attention and MLP both read the
normed hidden, `hidden = residual + attn + mlp`. Attention interleaves sliding-window (RoPE,
interleaved GPT-J-style rotary) and full-attention (**NoPE** — no positional embedding) layers on a
period-4 pattern.

The real checkpoint runs GQA 128 Q / 8 KV heads at head_dim 128, 128 routed experts with top-8
sigmoid selection + top-k renorm, 4 shared experts combined by `average` (`(routed + shared) / 2`),
tied embeddings, and a `logit_scale` multiplier on the lm_head output (the config default is
`0.0625`).

| | EP | CP | TP | ETP | PP | EP+CP | EP+TP | EP+ETP |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Cohere2 MoE | Yes | Yes | Yes (selective) | Yes (pure, `ep_size=1`) | **No** | Yes ¹ | Yes | Experimental |

PP is refused loudly: the PP stage loader requires lazy loading, which this family declares off,
and behind that the generic tie gate refuses too — every shipped checkpoint ties `lm_head` to
`embed_tokens`. A non-unit `logit_scale` is a third refusal: it would silently vanish in the stage
head (`_reject_unapplied_logit_scale` in `src/distributed/pipeline_parallel/split.py`). An untied,
unit-scale variant would additionally have to land its stage boundaries on the period-4
`layer_types` pattern.

¹ Node-local EP+CP requires `ep_group_size == nvlink_domain_size`; on an 8-GPU node that pins
`ep_size` to 8, with `cp_size` dividing the domain (EP is orthogonal to DP — ep8+cp2 is a valid
single-node shape).

Coverage: `tests/gpu/parallelism/test_cohere2_moe_all_parallelism.py` (ep8 / cp8 / tp8 / ep8+cp2 /
ep8+tp2 / etp8 / ep2+etp4 on a tiny synthetic checkpoint), plus EP=2 equivalence, SFT and CPU
contract suites (`tests/gpu/parallelism/ep/test_ep_vs_fsdp_cohere2_moe.py`,
`tests/gpu/trainers/sft/test_sft_cohere2_moe.py`, `tests/cpu/models/test_cohere2_moe_support.py`).

## EP wrapper

`EPCohere2MoELayer` (`src/distributed/expert_parallel/layers/cohere2_moe.py`) replaces
`Cohere2MoeSparseMoeBlock`. Layers whose `mlp_layer_types` entry is `dense` instantiate a plain MLP
and are never wrapped.

- Routing: `Cohere2MoeTopKRouter` picks top-k on the RAW logits, then activates only the selected
  scores — softmax over the k scores, or sigmoid with `norm_topk_prob` renorm. The wrapper
  re-derives exactly that from the router's logits (the router module itself is still called, so
  the HF `router_logits` recorder keeps firing).
- Load balancing: the architecture has **no bias slot** and **no aux-loss wiring** (no
  `router_aux_loss_coef`; the loss never reads `router_logits`), so `moe_balancing: bias_update`
  **raises**, `aux_loss` cannot work, and `auto` resolves to `none` with a warning. The explicit
  `bias_update_transient` opts into the trainer-only side-buffer: it steers training-time routing,
  resumes from `router_balancing_biases.pt`, and never reaches a checkpoint or a served copy
  (near-tied top-k picks flip between trainer and server).
- Routed experts: `Cohere2MoeExperts`, fused `gate_up_proj [E, 2I, H]` / `down_proj [E, H, I]` —
  the Qwen3.5/GLM-4 layout, so the shared fused-GLU helpers cover compute (grouped GEMM, 2 calls;
  fused SwiGLU Triton combine).
- Shared experts: one fused `Cohere2MoeMLP` (intermediate = `intermediate_size *
  num_shared_experts`), replicated per rank, combined after DeepEP combine by `sum` or `average`
  (the wrapper scales the summed output by 0.5).
- Checkpoints: the hub stores one tensor per expert (`experts.{i}.{gate,up,down}_proj.weight`);
  transformers fuses on load and reverts on save. The gather emits the fused pair, so
  `scripts/after_training/unfuse_moe_experts.py` can rewrite a gathered save per-expert.

## CP wrapper

`Cohere2MoeAttention` → `Cohere2MoeUlyssesAttention`
(`src/distributed/context_parallel/layers/cohere2_moe.py`), optimized `[B, S, H, D]` path. RoPE is
applied only where the HF layer applies it: sliding-window layers, plus the `force_rope`
dense-prefix override — full-attention layers pass Q/K through untouched (NoPE). The rotary is the
interleaved GPT-J form in fp32, matching `apply_rotary_pos_emb` bit-for-bit in bf16. The 8 KV heads
of the real checkpoint cap `cp_size` at 8; CP requires a flash attention implementation.

## TP

Selective TP: `Cohere2MoeAttention` is in `TP_SHARDABLE_ATTENTION_CLASSES` — standard `q_proj` /
`k_proj` / `v_proj` colwise, `o_proj` rowwise; embeddings and the tied lm_head stay replicated. MoE
blocks belong to EP, not TP.

## Packing

Isolated on every backend with **no toolkit patch**: the model forward feeds `position_ids` into
mask construction and forwards them through layer kwargs to the attention interface, so both the
dense packed mask and the flash varlen path engage. Pinned by
`tests/cpu/models/test_cohere2_moe_support.py` — if transformers drops that plumbing, the spy test
fails and the family needs the Mistral4/Zaya position-ids patch.

A text-only dataset takes the text pipeline, where packing is supported; an image dataset takes the
VLM processor pipeline, which rejects it. The shipped config keeps `packing: false` to stay on its
memory-validated shape.

## Loading and serving

- Lazy loading is off (`_supports_lazy_loading = False`): the Command A+ index spells the vision
  tower `model.vision_tower.vision_model.*`, a `from_pretrained`-only conversion the lazy loader
  does not apply — every load routes through `from_pretrained`.
- vLLM 0.26.0 registers both `Cohere2MoeForCausalLM` and `Cohere2VisionForConditionalGeneration`
  and its loader reads the fused expert pair, so gathered saves serve directly.
- RL weight sync is refused at construction (`_supports_weight_sync = False`): no end-to-end sync
  has been validated against the pinned server — flip only with a validated live-server run.
  SGLang is refused independently (the family declares no fused gather;
  [Rollout Servers](../infrastructure/rollout-servers.md)).
- Liger covers the fused SwiGLU (`Cohere2MoeMLP`, i.e. the dense and shared-expert path), cross-entropy,
  and the fused loss, which folds `config.logit_scale` onto the hidden states so the scaled softmax
  survives the fusion. The norm is left eager: the live class is `Cohere2MoeLayerNorm`, a
  mean-subtracting LayerNorm with no bias parameter. RoPE too — GPT-J interleaved, sliding layers only
  ([Liger Kernels](../optimization/liger-kernels.md#supported-models)).

## Configs

| Config | EP | Notes |
|---|:--:|---|
| `examples/sft/cohere2_moe/command-a-plus-ultrachat-ep.yaml` | 8 | VLM checkpoint, text-only SFT; `max_length: 1024` peaks ~255 of the B300's ~268 GiB at EP=8 — 4096 OOMs when DeepEP's elastic buffer resizes. Pins `model_revision` to the layout those numbers were measured on and `attn_implementation: flash_attention_2` (the family has no forced fallback; FA4 loads too but was not the measured backend) |
