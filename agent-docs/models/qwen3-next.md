# Qwen3-Next

`model_type` `qwen3_next` (`Qwen3NextForCausalLM`; Qwen3-Next-80B-A3B Instruct and Thinking). A text-only causal LM built from Qwen3.5-MoE's parts: the same MoE block (`Qwen3NextSparseMoeBlock`: routed experts plus a sigmoid-gated shared expert), the same period-4 interleave of full attention with `Qwen3NextGatedDeltaNet` linear-attention layers, the same double-width gated `q_proj` with partial RoPE. 80B-A3B has 48 layers (12 full, 36 linear), 512 routed experts at top-10, and 2 KV heads of width 256; 97% of its parameters are routed experts.

Per-axis support: [Compatibility matrix](README.md#compatibility-matrix). EP and ETP run; CP, TP and PP are refused at load. EP and ETP pass the tiny-model GPU tests (EP against the reference model, deterministic expert gradients, export and LoRA merged-save round trips, preference resume at ep2, etp2 and ep2 × etp2); no full-scale checkpoint is trained here.

## EP layer

`EPQwen3NextMoELayer` (`halo/distributed/expert_parallel/families/qwen3_next.py`) subclasses `EPQwen3_5MoELayer` and inherits its whole contract: fused `gate_up_proj [E, 2M, H]` + `down_proj [E, H, M]` routed experts sharded on the expert dimension, grouped GEMM on SM90+, the fused-GLU split under ETP, and the shared expert with its gate replicated and combined after the DeepEP combine ([Qwen3.5](qwen3_5.md#ep-layer)).

The one difference is the router. `Qwen3NextTopKRouter` renormalizes the top-k weights only under its `norm_topk_prob` (`true` on the released checkpoints), where Qwen3.5's router always does. Every gate weight the layer re-derives reads that flag live: routing replay (`_gate_weights_at`) and the balancing-biased route (`_biased_route`, top-k on the biased probabilities, then the router's own gating at those indices). A zero bias reproduces the router for either value of the flag.

At `ep_size: 1` the layer is installed whenever `use_grouped_gemm` is on (the torchrun default), so the experts run Halo's grouped GEMM and FSDP2 shards them (`fsdp_shard_ep1_experts`) instead of running transformers' `grouped_mm` experts.

## Checkpoint layout

The hub checkpoints store the experts one tensor per expert (`mlp.experts.{i}.{gate,up,down}_proj.weight`), which transformers fuses on load. The spec declares that layout (`FUSED_SPLIT_PER_EXPERT` over the GLU projection names), and every writer applies it: the gathered EP save and `merge_ep_shards.py` both emit the hub's per-expert keys, the names `from_pretrained` and both pinned engines' loaders read. The lazy EP loader reads the per-expert keys and fuses each rank's expert slice directly.

`from_pretrained` drops the hub's `mtp.*` multi-token-prediction head, so exports carry none unless the run [trains it](#mtp-head).

## Attention and packing

Attention resolves as on [Qwen3.5](qwen3_5.md#full-attention-qwen3_5moeattention), read off the 256-wide heads: FA4's head_dim-256 path on Blackwell, the padded scripts included, SDPA under `full_determinism`, and FA2 for SFT rows that hold several documents when `attn_implementation` is unset ([Flash Attention](../optimization/flash-attention.md#model-specific-handling)). The shipped recipe pins that FA2 for its packed rows, on Hopper too.

Packed rows isolate as on Qwen3.5: the collators emit `seq_idx` and `cu_seq_lens_q` for the GatedDeltaNet layers, and packing is refused without the `fla` / `causal-conv1d` kernels that read them ([Document isolation](../data/collators.md#document-isolation-under-packing)).

## Balancing

The router is a bare weight with no bias slot. `moe_balancing: auto` resolves to `aux_loss`: the layer calls the real router, so the forward's `output_router_logits` capture still fires under EP. `bias_update` raises, and `bias_update_transient` is the trainer-only opt-in, whose bias reaches no export ([Callbacks](../training-methods/callbacks.md#moe-balancing-modes)).

## Refused modes

- **CP**: 36 of 48 layers are `linear_attention`, whose conv and recurrent scan run along the sequence axis ([Qwen3.5](qwen3_5.md#why-moe-hybrid-cp-is-refused)).
- **TP**: `Qwen3NextAttention` is not in `TP_SHARDABLE_ATTENTION_CLASSES`, so a TP run would shard no layer and is refused.
- **PP**: `Qwen3NextModel` has no `PPFamilySpec` and is refused by name.

## RL weight sync

Online and async GRPO are refused at construction by the spec's `weight_sync_refusal`. Both pinned engines register `Qwen3NextForCausalLM`, and both loaders map the per-expert names the gather emits one key at a time with no per-call coverage assertion, so neither client lists the family in `UNSERVABLE_MODEL_TYPES`; no sync into either has been validated end to end ([Rollout Servers](../infrastructure/rollout-servers.md#which-families-each-engine-serves)).

## Liger

The family's Liger spec keeps upstream's RMSNorm and FLCE, serves the GatedDeltaNet gated norm with `fla`, and takes over `swiglu` for `Qwen3NextMLP`, the dense MLP and the shared expert the EP layer adopts unchanged ([Liger Kernels](../optimization/liger-kernels.md#upstream-covered-families-the-toolkit-extends)).

## MTP head

`mtp_loss_weight` trains the release's `mtp.*` head ([MTP heads](../optimization/mtp-heads.md)). The config declares no tail count, which reads as one layer, as both engines read it.

## Configs

`examples/sft/qwen3_next/qwen3-next-80b-a3b-ultrachat-ep.yaml`: 80B-A3B SFT at EP=8 (64 experts per rank), packed rows on FA2, on `qwen3-multiturn.jinja` ([its first-step loss on Qwen3.6](qwen3_5.md#configs)). Each rank holds about 9.7B expert parameters, so their weights, gradients and AdamWBF16 moments (8 B/param) take about 77 GB (72 GiB) per GPU before any activation: the recipe fits B200/B300, not an 80 GB H100.
