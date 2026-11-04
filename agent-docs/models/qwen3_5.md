# Qwen3.5 / Qwen3.6 MoE

A separate class hierarchy from Qwen3 (`Qwen3_5MoeForCausalLM`, `Qwen3_5MoeSparseMoeBlock`, `Qwen3_5MoeAttention`). Three traits drive parallelism: a sigmoid-gated shared expert, a double-width `q_proj` fusing query with a sigmoid gate, and a hybrid `layer_types` scheme interleaving full attention with `Qwen3_5MoeGatedDeltaNet` linear-attention layers.

Per-axis support (MoE and dense rows): [Compatibility matrix](README.md#compatibility-matrix). Dense text towers have an [unpacked CP2 SFT path](../parallelism/context-parallelism.md#dense-hybrid-sft); MoE hybrid CP remains refused. Pipeline parallelism takes the text towers only: `text_only_model: true` for the multimodal checkpoints, boundaries on the period-4 `layer_types`, packed rows on the GatedDeltaNet segment markers each stage rebuilds, and `moe_balancing: none` (no exportable bias slot, and the split severs `aux_loss`) ([Pipeline Parallelism](../parallelism/pipeline-parallelism.md#model-support)).

## Architecture

Each decoder layer is *either* full softmax attention *or* a linear-attention block, interleaved by config. Both share the same MoE MLP path (`Qwen3_5MoeSparseMoeBlock`): 256 routed experts + 1 sigmoid-gated shared expert, top-k=8.

`Qwen3_5MoeTextConfig` builds `layer_types` from `full_attention_interval` (default 4) — every fourth layer is full attention, so Qwen3.6-35B-A3B (40 layers) gets 10 full + 30 linear.

### Full attention (`Qwen3_5MoeAttention`)

| Feature | Detail |
|---|---|
| `q_proj` | Double-width `num_q_heads * head_dim * 2`, chunked into `(query, gate)`; output becomes `attn_output * sigmoid(gate)` before `o_proj` |
| RoPE | Partial — only the first `head_dim * partial_rotary_factor` channels rotate (default `0.25`) |
| Heads | 16 Q heads, 2 KV heads (GQA factor 8), which caps TP at `tp_size=2` |

> **FlashAttention-4 trains this family on Blackwell on its head_dim-256 path: the general forward kernel and the dedicated head_dim-256 backward.** The partial-rotary `torch.cat` leaves q/k head-major, which that path reads as it is ([Flash Attention](../optimization/flash-attention.md#model-specific-handling)). The dedicated backward's deterministic mode is not validated here, so a `full_determinism` run demotes FA4 to SDPA (`supports_fa4_backward`, `halo/models/attention/selection.py`). SFT multi-document rows take FA2 when `attn_implementation` is unset, and under `full_determinism` even with `flash_attention_4` pinned ([Configs](#configs)); the padded scripts (preference, reward, classification, teacher distillation, GRPO) take FA4 when unpinned, and their shipped recipes pin `sdpa`. The self-distillation recipe trains one row per batch, which FA4's dense kernel runs, so it leaves attention unset. The verdict reads the head widths, so Qwen3.6, which reuses the 3.5 classes and types (`Qwen/Qwen3.6-35B-A3B` declares `qwen3_5_moe` with a `qwen3_5_moe_text` tower), and [Qwen3-Next](qwen3-next.md) resolve alike.

On Hopper the standard detector picks FA3 when installed, else FA2.

### Linear attention (`Qwen3_5MoeGatedDeltaNet`)

A gated delta-rule recurrent attention with a sequence-axis causal Conv1d on the input. The fast path is gated on the `fla` / `causal-conv1d` imports (both pinned in `pyproject.toml` and installed in the images, so it is on by default).

The imports reach the layer through transformers' hub-kernel funnel, whose package resolution attribute-walks a bare package import. `fla` exports nothing at root, so upstream alone silently captures the torch scan, far slower than the kernel.

`halo/kernels/hub_kernel_dispatch.py` imports the mapped submodule chain before decoration so the real kernel is captured, and warns when a capture still lands on the torch body with its package installed; that warning in a training log is a throughput alarm. It sends a CPU call to the torch body, and the `halo` package init re-decorates through it whatever a modeling module imported before `halo` already bound, so import order does not decide which path a forward takes. GLM-5-Next's fla-backed KDA ops ride the same funnel and repair.

On Blackwell the delta rule's state kernel (`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, shared by the gated delta rule and KDA) autotunes only over its `num_warps=2` configs: `finalize_loaded_model` restricts it at the end of every load (`apply_fla_launch_options`, `halo/kernels/fla_launch.py`), so it holds whether the kernel is later reached through this funnel or a remote-code module's own `fla` import. Under the image's Triton 3.6 its `num_warps=4` configs give run-to-run different, at some positions wrong, outputs and gradients on packed (`cu_seqlens`) rows, and on B300 also on larger unpacked batches; the stock autotuner picks them at this family's shapes. Upstream applies the same restriction from fla 0.5.2 (fla #953). Triton keys its autotune cache on the config list, so a pick cached by an earlier run is not reused. `tests/gpu/kernels/test_fla_blackwell_autotune.py` pins it, with a `num_warps=4` control that must still miss.

FLA's kernels for the q/k l2norm inside the delta rule and for the gated output norm (which the Liger spec installs) take `NB = cdiv(tokens × heads, 65536)` as an autotune-key constexpr their bodies never read, so each new value compiles and benchmarks every autotune config: one every 2,048 tokens at Qwen3.5-35B-A3B's 32 value heads, to which q/k are repeated. The binaries and FLA's autotune picks (`FLA_CACHE_RESULTS`, default on) persist in the Triton disk cache anchored on `HF_HOME`, so this is paid once per value per volume, as for [Bailing's KDA kernels](bailing.md#remote-code-compatibility). Picks differ across fresh caches, and so does bf16 rounding ([Docker](../infrastructure/docker.md#baked-in-environment-variables)).

The torch fallback matches the kernels only on single-document rows: it takes neither `seq_idx` nor `cu_seq_lens_q`, so a multi-document row mixes through both the conv and the delta-rule scan while attention stays isolated. The collator factory therefore **refuses** `packing` and `padding_free` for this family when either wheel is missing, and SMPO refuses its own `padding_free` likewise ([Document isolation](../data/collators.md#document-isolation-under-packing)).

These layers need no CP for memory: the recurrent state is independent of `S`, per-layer activations are `O(B·S·d)`, and the Conv1d left-context is a 3-token state. The long-context ceiling comes from the 10 full-attention layers — 33K on a single B200/B300 rank at `num_heads=16, head_dim=256, num_kv_heads=2`.

## Model loading

Qwen3.5/3.6 ship as multimodal: `model_type` `qwen3_5_moe` (or `qwen3_5` dense), top-level class `Qwen3_5MoeForConditionalGeneration`, registered under `AutoModelForImageTextToText`. `resolve_auto_model_class()` keeps the vision tower intact, and the config detection routes the **load** through `load_distributed_model`, adding the processor on top (`halo/entrypoints/model_setup.py`). The MoE VLM therefore gets the same EP/TP support as a standalone causal LM — `has_routed_experts` finds the experts on the nested `text_config`.

That verdict does not decide the data path: a text-only dataset on these checkpoints is a text run, which is what lets the text SFT recipes below use `packing` ([SFT — VLMs](../training-methods/sft.md#vision-language-models)).

The resolver keys on `model_type` alone, not `architectures`. A `*-patched` checkpoint whose `config.json` carries the text-only `model_type` `qwen3_5_moe_text` is absent from the mapping, falls back to `AutoModelForCausalLM`, and drops the vision tower — regenerate it through the resolver.

## EP layer

`EPQwen3_5MoELayer` (`halo/distributed/expert_parallel/families/qwen3_5.py`) replaces `Qwen3_5MoeSparseMoeBlock`. Softmax top-k routing via `Qwen3_5MoeTopKRouter`. The shared expert (`Qwen3_5MoeMLP`, sigmoid-gated output) stays local on every rank and is combined with routed-expert output after DeepEP combine. Routed experts use the fused `gate_up_proj [E, 2M, H]` + `down_proj [E, H, M]` layout, sharded along the expert dimension, SwiGLU, grouped GEMM on SM90+.

The EP layer re-derives selection from the router's own logits — top-k on the bias-adjusted probabilities, gate weights from the unbiased renormalized softmax, which makes a zero bias a no-op.

The architecture has **no bias slot** (the gate is a bare weight), so the trained bias can only be trainer-side: `moe_balancing: bias_update` **raises**, and the explicit `bias_update_transient` is the opt-in. Balancing works during training and resumes exactly, while every exported checkpoint serves without the bias (near-tied top-k picks flip vs training).

`aux_loss` is the exported-by-construction balancing on both classes: `Qwen3_5MoeForConditionalGeneration` and the text-only `Qwen3_5MoeForCausalLM` declare `output_router_logits`, so `auto` resolves to `aux_loss` ([Callbacks](../training-methods/callbacks.md#moe-balancing-modes)).

`text_only_model: true` loads a VLM checkpoint through that CausalLM class deliberately. The vision tower is dropped from the build **and from the export**, and so is the MTP tail unless the run [trains the head](../optimization/mtp-heads.md): the artifact carries the tokenizer but no processor files (`processor_config.json`, `preprocessor_config.json`, `video_preprocessor_config.json`) and no vision token ids. The shipped 122B recipe keeps the multimodal class and sets `bias_update_transient`. A LoRA trained this way addresses `model.layers`, and `merge_peft_adapters.py` reads that off the adapter's keys and loads the base through the same CausalLM class, so the merged checkpoint is this text-only export: the run's tokenizer saved beside the adapter (its chat template and added tokens) and no processor files.

Image-bearing datasets are refused loudly (the text path would otherwise prune the column silently).

Both pinned engines, vLLM 0.32.0 and SGLang 0.5.22, register the text-only `Qwen3_5MoeForCausalLM` / `Qwen3_5ForCausalLM` beside the multimodal classes, and their loaders take the export unchanged (read from the loaders; not served end to end here). `scripts/after_training/reattach_vision_tower.py` rebuilds the multimodal artifact when one is wanted: it re-prefixes the trained text weights to `model.language_model.*`, streams the base's untrained vision tower and MTP tail back in (a head the export trained supersedes the base's), and copies the base's processor files beside the export's tokenizer.

Both classes drop the Hub checkpoints' `mtp.*` MTP head at load, so a run's exports carry none unless it [trains the head](../optimization/mtp-heads.md). `reattach_vision_tower.py` streams the base's back at the base's weights, which lowers acceptance and leaves the output distribution unchanged, and `merge_models.py` leaves a Hub base's `mtp.*` out of a `task_arithmetic`/`ties` merge whose fine-tunes lack it ([Model Merging](../reference/model-merging.md#memory-and-output)).

## CP wrapper

`Qwen3_5MoeUlyssesAttention` (`halo/distributed/context_parallel/families.py`) replaces `Qwen3_5MoeAttention` / `Qwen3_5Attention` and handles two quirks:

- **Double-width `q_proj`** — the output is viewed as `[..., num_q_heads, head_dim * 2]` and chunked into query + gate. The query goes through Ulysses (q_norm + RoPE + all-to-all + flash-attn + all-to-all back), then `attn_output * sigmoid(gate)` before `o_proj`.

    The gate stays on the local sequence shard, so no extra communication.

- **Partial RoPE** — only the first `rotary_dim` channels of Q/K rotate; the rest is concatenated unrotated.

Correctness coverage against a synthetic all-full-attention config: `tests/gpu/parallelism/cp/test_cp_family_vs_reference.py --family qwen3_5_moe_text`.

## Why MoE hybrid CP is refused

MoE checkpoints ship hybrid `layer_types`, and both of `Qwen3_5MoeGatedDeltaNet`'s sequence-axis ops cross Ulysses chunk boundaries:

- **`causal_conv1d_fn`** kernel 4 — output `t` depends on `t..t-3`, so each rank's first 3 positions need the previous rank's last 3 input tokens. Without a halo exchange the kernel sees zero-padding.
- **`chunk_gated_delta_rule`** is a recurrent scan — chunk `c+1` must start from chunk `c`'s final state, which a CP rank cannot recover.

The validator refuses MoE GatedDeltaNet and unclaimed linear-attention layers. Dense CP2 SFT instead redistributes heads before these sequence operations ([Dense hybrid SFT](../parallelism/context-parallelism.md#dense-hybrid-sft)); it does not use FLA's CP context. For MoE, train at the single-rank ceiling (32K–33K on B200/B300, EP=8, no CP), or train from scratch with `layer_types=['full_attention'] * num_hidden_layers` (discards pretrained linear-attention weights).

## TP and ETP

Attention-only TP covers the MoE: `Qwen3_5MoeAttention` is in the TP accept-list, and the MoE block is skipped because EP owns it. The double-width `q_proj` shards colwise: the split is per head, so each head keeps its full `head_dim * 2` (query + gate) on one rank.

Only the full-attention layers shard there: the linear-attention layers stay replicated on every rank, so the per-rank footprint falls by far less than `1/tp_size` (the TP path warns). A dense checkpoint takes HF-native `tp_plan="auto"`, whose plan also shards the MLP ([Tensor Parallelism](../parallelism/tensor-parallelism.md#supported-models)). Its five linear-attention projections are `colwise_gather_output` (transformers 5.20.2): their matmuls shard, but every rank all-gathers each output, so on the linear-attention layers (3 of every 4) the causal conv and the delta rule, with their activations, run full-width on every rank, behind five all-gathers per layer. Where a dense Qwen3.5/3.6 model fits under FSDP2, prefer FSDP2 to TP.

ETP is supported: the routed experts use the fused-GLU contiguous-halves layout, so `_init_fused_glu_params` splits the halves before sharding and stores `gate_proj` / `up_proj` / `down_proj` separately when `expert_tp_size > 1`. The sigmoid-gated shared expert stays replicated. See the [ETP guide](../parallelism/expert-tensor-parallelism.md#limitations).

## Configs

The shipped EP config trains Qwen3.5/3.6 35B-A3B (40 layers = 10 full + 30 linear, 256 routed experts, top-k=8) at 33K max length under EP=8 (32 experts/rank); set `fp32_non_ep_params: true` for stable long context.

SFT with `packing: true` (fixed-length) loads `flash_attention_2` on Blackwell when `attn_implementation` is unset: FA4 runs packed rows only 2.9% (16k) and 4.6% (32k) faster, below the 5% gain a default change needs, so only a pinned `flash_attention_4` keeps them on FA4. Under `full_determinism` that pin loads FA2 too: the SDPA demotion would run a packed row as one dense mask with K/V repeated per query head ([Flash Attention](../optimization/flash-attention.md#model-specific-handling)). Unpacked rows keep FA4; the shipped recipes pin FA2. SDPA matches FA2 at `b=1` on unpacked rows only.

Packing isolates the full-attention layers via the packed mask, and `Qwen3_5MoeGatedDeltaNet` receives its boundaries from the collators and SMPO's padding-free forward, which emit `seq_idx` (its causal conv) and `cu_seq_lens_q` (its chunked delta rule) for the family, the two kwargs upstream reads and nothing model-side derives ([Document isolation](../data/collators.md#document-isolation-under-packing)). The same holds for Qwen3-Next.

[Tree packing](../optimization/tree-packing.md) (`tree_packing: true`) runs on Qwen3.5/3.6 on `flash_attention_2`, where an unset `tree_full_attention` resolves to `gather`: a branch's conv context and recurrent state come from its own path.

Examples under `examples/sft/qwen3_5/`:

- `qwen3.5-35b-a3b-ultrachat-ep.yaml`: [17,225 tok/s/GPU on real conversations](../optimization/throughput-benchmarks.md#recipes-and-real-text).
- `qwen3.5-122b-a10b-ep.yaml` scales the same shape to 122B-A10B at EP=8 on one node. Across two nodes it takes `--expert_parallel_size=16 --ep_scope=global --max_length=8192`: the cross-node dispatch gate refuses its 16,384 tokens per rank. It keeps the experts bf16 (`fp32_experts` off), since fp32 expert masters fit the 35B shape but OOM at 122B.
- VLM SFT on Qwen3.5-9B: `qwen3.5-9b-vl-ocr-olmocr.yaml`, `qwen3.5-9b-vl-docvqa.yaml`.

Online GRPO smoke: `examples/grpo/online/qwen3_5/online-grpo-qwen3.6-35b-a3b-smoke.yaml`. Async GRPO with Environments: `examples/grpo/environmental/qwen3_5/vllm/` (code contests, plus the `-aime-`, `-exam-qa-` and `-react-math-` ep4 task recipes) and the `sglang/` ep1 siblings; both pinned engines read the fused expert pair the gather emits, so either `rollout_backend` takes the weight sync ([Rollout Servers](../infrastructure/rollout-servers.md#which-families-each-engine-serves)).

Chat templates: the shipped SFT configs pin `jinja-templates/qwen3/qwen3-multiturn.jinja` (ultrachat is multi-turn), and the Qwen3.6 MTP recipes `jinja-templates/qwen3/qwen3.6-reasoning-effort.jinja`, as a reply with no think block reads far worse ([MTP heads](../optimization/mtp-heads.md#recipes)). On `qwen3-multiturn.jinja` a first-step loss near 3.3 is that missing block: Qwen3.6-35B reads 3.3 at step 0 and 0.83 after two optimizer steps. Train and serve under one render. `jinja-templates/qwen3/qwen3.5-native.jinja` is the verbatim upstream `Qwen/Qwen3.5-35B-A3B` template — system messages, XML-form tools, thinking, and the VLM vision placeholders — for runs that must match the served render exactly (tool-call `arguments` must be a parsed mapping). Qwen3's own upstream template ships as `qwen3/qwen3-native.jinja`.

The family also carries the per-method canonical examples: Qwen3.5-9B (dense) for DPO/SMPO/KTO (`examples/preference/qwen3_5/`), reward, classification, and distillation; Qwen3.6-35B-A3B for the GRPO task configs.

## Qwen3.8-27B

`Qwen/Qwen3.8-27B` is a dense `qwen3_5` checkpoint (`Qwen3_5ForConditionalGeneration`: 64 layers, hidden 5120, one MTP layer) and loads through the dense Qwen3.5 path unchanged; its `output_gate_type: swish` is the SiLU gate the Qwen3.5 classes hard-code. No recipe ships for it, and no GPU run has trained it.

Its hub chat template, which Qwen3.8-Flash-Next shares:

- With thinking on (the default), `reasoning_effort` must be `xhigh` (the default), `medium` or `low`; any other value raises `TemplateError`, so online GRPO's `reasoning_effort: high`, or `random`, which can draw it, fails the render. `xhigh` and `low` add a reasoning-effort system turn.
- `preserve_thinking` defaults to true: every assistant turn renders a `<think>` block, empty when the message carries no `reasoning_content`.

Pin a Halo template with `force_chat_template` where those renders are not the training target, as the shipped SFT configs do.

## Qwen3.8-2.4T-A95B

`Qwen/Qwen3.8-2.4T-A95B` is a text-only `Qwen3_5MoeForCausalLM`, `model_type` `qwen3_5_moe_text`, which the family spec admits. It has 92 layers (23 full, 69 linear), 512 routed experts at top-10 plus the shared expert, hidden 8192 and 4 KV heads, and the hub stores the experts fused, the layout the spec writes. Its 4.9 TB of bf16 weights come to about 610 GB per GPU at EP=8 (calculated), more than a B300 holds. No recipe ships for it, and no GPU run has trained it.

## RL chat template

The Qwen3.6 code-contests async GRPO recipes pin `jinja-templates/qwen3/qwen3.6-reasoning-effort.jinja` with
`force_chat_template: true`, and the servers run `--chat-template` on the same file
(`VLLM_CHAT_TEMPLATE` in the compose file). It is the hub template reduced to text-only tool use —
thinking always on, carried reasoning always rendered, the vision branch and the
`enable_thinking`/`preserve_thinking` switches dropped — plus a system-block line stating the episode's
`reasoning_effort`, which the hub template has no variable for
([Effort templates](../training-methods/async-grpo/reasoning.md#effort-templates)).
