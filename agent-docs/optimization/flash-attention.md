# Flash Attention & Attention Backends

`load_distributed_model` (`halo/model_loading/parallel/loader.py`) auto-detects the backend per GPU and model: `flash_attention_4` on Blackwell, `flash_attention_3` on Hopper, `flash_attention_2` otherwise. Pass `attn_implementation` only to override it. The detection line names only that GPU default, which the model-level checks below may move; the run's start log names the backend the policy was built with (`Attention:`).

FA2 is the detector's terminal default, with no availability probe — every shipped image carries it, and a host without it raises at model build. A route from FA4 to FA2 for a model FA4 cannot train ([below](#model-specific-handling)) does probe, and refuses on such a host, naming `flex_attention` and `eager`. Context Parallelism hard-requires flash-attn and raises `ImportError` when it is missing.

Why tiling attention is the standard fix for the model's most memory-bound op: [GPU Training Theory §6](../reference/gpu-training-theory.md#6-attention--the-canonical-memory-bound-fix).

## Supported backends

| Backend | GPU | CP | GptOss |
|---------|-----|----|--------|
| `flash_attention_4` | Blackwell SM100+ (CuTe DSL) | Yes | Yes — every sink policy (neutralized, live-frozen via `learnable_sink`, trainable via the sink-gradient rescale) |
| `flash_attention_2` | Ampere+ | Yes | Yes (sinks reset; live sinks rejected) |
| `flash_attention_3` | Hopper SM90 only | Yes | Yes (sinks neutralized; live sinks need an `s_aux` build, trainable sinks rejected) |
| `flex_attention` | torch ≥2.5 | No (auto-switched to the arch's flash kernel: FA4 / FA3 / FA2, [below](#model-specific-handling)) | Yes (LSE renorm; FSDP2 NaN fix auto-applied) |
| `sdpa` | all | No | Yes when sinks reset (reset → dropped sink column is `exp(dtype.min)=0`; loader flips `_supports_sdpa`). Rejected when sinks are kept live. |
| `eager` | all | No | Yes |

GPU support: A100 (SM80) = FA2/flex/sdpa. H100/H200 (SM90) = FA2/FA3/flex/sdpa, no FA4. B200 (SM100) / B300 (SM103) = FA2/FA4/flex/sdpa, no FA3.

The CP column above is what `reject_unsupported_ulysses_model` (`halo/distributed/context_parallel/model_rules.py`) accepts as a config label; it is not which kernel runs. CP's attention call goes through `_flash_attn_func` (`halo/distributed/context_parallel/ulysses_attention.py`), which ignores that label and probes in a fixed order: the architecture's `preferred_flash_impl` (`halo/models/attention/selection.py`: FA3 on Hopper (SM90), FA4 on Blackwell (SM100+)), then FA2.

Both architecture probes precede FA2 because every image ships FA2; an FA2 probe placed first would resolve on every device and leave the arch-matched kernel unreachable.

The one exception is model-level: `_flash_attn_func` takes an `allow_fa4` flag, resolved per wrapper from
`supports_fa4_backward`, so GLM-4.7-Flash runs FA2 on Blackwell under CP in every run
([Model-specific handling](#model-specific-handling)); Qwen3.5/3.6 and Qwen3-Next, whose FA4 a deterministic run
also refuses, take no CP. The probe overrides the configured label, so that flag is the only protection there.
`preferred_flash_impl` is also what the automatic detection and the loader's CP label read, so CP
and the config-level choice can never disagree about a device.

The order matters because CP calls the **non-varlen** forward, the path FA2 serves worst on both architectures.

On B300, CP on FA4 measures 1.2–3× FA2 at ≥32k tokens with equal peak memory, the margin widening with sequence length (+19% at 32k, +63% at 128k on gpt-oss-20b ep8+cp8; +197% on dense Qwen3-4B ep1+cp8 at 128k). FA2 is faster only below ~16k, where per-call overhead dominates.

## Automatic detection

`_detect_attention_impl` (`halo/models/attention/selection.py`) takes `preferred_flash_impl`, which reads only the compute-capability major:

1. `major >= 10` (Blackwell SM100/SM103): `flash_attention_4` if `flash_attn.cute` imports, else `flash_attention_2`. A build that is present but unusable warns before degrading.
2. `major == 9` (Hopper): `flash_attention_3` if `flash_attn_interface` (else `flash_attn_3`) provides the varlen kernel, else `flash_attention_2`.
3. Anything older, or any exception: `flash_attention_2`.

The detection above, the sink probe and the FA4 warm-up resolve kernels through `flash_kernel(attn_impl, varlen=...)`: one module table in transformers' dispatch order (FA3 through `flash_attn_interface` first), FA4 imported only after `ensure_fa4_kernel_cache_env`. A build missing the entry point reads as absent. Context parallelism resolves its kernel through `flash_kernel` too, in `preferred_flash_impl`'s order under its FA4 flag, then FA2 ([above](#supported-backends)); the trainable-sink rescale imports `flash_attn.cute` directly.

The `major >= 10` test also admits consumer Blackwell (SM 12.x, RTX 50-series), where `flash_attn.cute`'s
SM100 kernels do not run. That is not a supported target — the images build for B200/B300 and H100/H200.

`resolve_attn_implementation` then applies the model-level overrides below. One is precision-driven: in
**fp32**, any `flash_attention*` is redirected to `flex_attention` when the model has sinks or an attention
softcap and `sdpa` otherwise — flash kernels are half-precision only. Context Parallelism in fp32 **raises**, since CP runs
attention through FlashAttention unconditionally.

The Blackwell image ships both FA2 (top-level `flash_attn`, inherited from the base image) and FA4
(`flash-attn-4` 4.0.0b37, the `flash_attn.cute` submodule) under one namespace; FA4 is the production
default. Profiling benchmarks leave `--attn_implementation` unset, so they run FA4 on Blackwell too.
transformers dispatches `flash_attention_4` natively through `flash_attn.cute` (causal, sliding-window,
sinks via `s_aux`, softcap, varlen). FA4 serves a softcap on the dense call only: its varlen kernels are not
validated with one, so the varlen entry point
refuses a softcapped call (Gemma 2 27B, below). 4.0.0b37 is beta — validate accuracy-sensitive
runs. Hopper keeps FA2 + FA3.

## FA4 vs FA2 vs SDPA on Blackwell

The tables in this section and the [backend benchmark](#backend-benchmark) are from Halo 1.1.0 or earlier, before FA4 4.0.0b37. The 4.0.0b37 figures quoted under [Model-specific handling](#model-specific-handling) are from 1.2.0.

FA2 is an SM80-style kernel, untuned for Blackwell. FA4 is the Blackwell-native CuTe DSL kernel: **3.6–3.9× faster than FA2 on the isolated attention kernel** (microbench B2×S8192×H32×D128: fwd 2.71→0.70 ms, 3.9×; fwd+bwd 10.9→3.0 ms, 3.6×). Both match an fp32 SDPA reference to ~2e-3 (bf16 floor) across MHA/GQA and head_dim 64/128.

The end-to-end step win is a function of attention's share of the step, which grows with sequence length (attention is O(seq²), the rest ~O(seq)). Qwen3-4B-Instruct-2507, batch 1, GC on, Liger on, tok/s/GPU on one B300 (every cell on the same device), mean of two runs, Blackwell image. Runs repeat within 1.1%, except flex at 4,096 (7.9%):

| SeqLen | FA4 | FA2 | SDPA | flex | FA4/FA2 |
|--------|-----|-----|------|------|---------|
| 4,096 | 23,614 | 19,920 | 25,733 | 20,147 | 1.19× |
| 16,384 | 23,139 | 12,282 | 23,112 | 16,167 | 1.88× |
| 32,768 | 17,504 | 7,497 | 17,047 | 10,607 | 2.33× |

At 4k batch-1 the step is overhead-bound, so FA4 stays close to FA2 (1.19×); the gap opens with length and reaches 2.3× at 32k. SDPA (Blackwell-tuned cuDNN kernel) leads FA4 by 9% at 4k and is within 3% of it from 16k, so SDPA is a fine fallback for plain dense models. flex trails both at every length; it ties FA2 at 4k (within its own 7.9% run spread) and beats it from 16k.

On sparse MoE expert work and the wait for the busiest rank take most of the step, so the kernel speedup is only **+17%** end-to-end. gpt-oss-20b, EP=8, seq 16,384, batch 1, GC on, 8× B300 (Blackwell image, median of 2): FA4 12,382 vs FA2 10,627 tok/s/GPU (1.17×); the FA4 arm is the s16384 row of the ep8 sequence sweep in [Throughput Benchmarks](throughput-benchmarks.md). FA4 is the production default for gpt-oss SFT; SDPA also dispatches once sinks are reset but lacks FA4's native sink/sliding-window/softcap handling.

EP/long-context throughput tables live in [Throughput Benchmarks](throughput-benchmarks.md), which already run the FA4 default.

## Model-specific handling

`load_distributed_model` applies these automatically.

**Nemotron-H** takes FA4 on both towers, or FA2 where no FA4 build imports. The text tower calls 32 query heads over 2 KV heads at head_dim 128, causal, through the varlen kernel across packed documents and padded rows. The Radio vision tower calls 16 heads at head_dim 80, non-causal, through the varlen kernel across the images of a batch. The warm-up compiles both towers' calls, as it does any non-causal tower on FA4. At 8,192 text tokens a call costs 2.2 ms on FA4 and 6.3 ms on FA2, and at 13,322 vision tokens a block under checkpointing 7.3 ms and 18.8 ms. `kernels/test_nemotron_attention_kernels.py` holds both calls against eager through the real modules.

**GptOss** (attention sinks):

1. Disables transformers' auto-fallback to `vllm-flash-attn3` (`ensure_gpt_oss_local_flash`, run once when `halo/models/attention/gpt_oss_sinks.py` is imported, which every toolkit load path does first). transformers reroutes GptOss FA2 and FA3 requests to `kernels-community/vllm-flash-attn3`: SM90-only binaries fetched through the `kernels` package, which the images do not ship. Every GptOss build in the process therefore keeps the local flash build, a reward-model scorer's or a sentence-transformers backbone's included.
2. Applies the run's sink policy (`halo/models/attention/gpt_oss_sinks.py`). The policies — neutralized, live and frozen, trainable — and what each refuses are on [GPT-OSS → Attention sinks](../models/gpt-oss.md#attention-sinks).
3. **Variable-length RL (async GRPO) runs FA4 via a per-row dense forward** (`_dense_last_hidden_state`), for exact RoPE positions and rank-uniform collective counts (`rows_forward_densely`).

    The dense kernel is seqlen-invariant, so the path forwards each row on its own, trimmed to its real contiguous span, with `attention_mask=None`: RoPE positions `[0, len)`, log-probs bit-identical to the unpadded forward. FA4 and the [row-position families](../training-methods/grpo/offline.md#row-positions) are where the trim is *mandatory*; `rows_forward_densely` also takes it on any attention path at `per_device_train_batch_size: 1`, for the cost alone ([Chunked log-probs](../training-methods/async-grpo/performance.md#chunked-log-probs)).

    Per-rank forward counts are all-reduced to a max and padded with tied dummy forwards, and every row — the padding included — stays connected to the loss, so the forward **and** backward collective sequences stay in lockstep. A disconnected row would let autograd prune its backbone backward and desync the FSDP/EP grad collectives across ranks.

Fallback chain for sinks models: requested → `flash_attention_4` (added as a candidate when the sinks are live) → `flash_attention_3` (Hopper, where its build imports) → `flash_attention_2` → `flex_attention` → SDPA (only when sinks are reset) → `eager` (`select_supported_attn_implementation`); under live sinks every sink-dropping candidate is filtered out.

With sinks reset for SFT (the default), an explicit `attn_implementation='sdpa'` is kept — `install_sdpa_for_reset_sinks` flips `_supports_sdpa` because the dropped sink column is `exp(dtype.min)=0`.

With **live** sinks `select_supported_attn_implementation` admits only the sink-carrying impls — FA4 (`learnable_sink`), FA3 with `s_aux` (Hopper), `flex_attention`, `eager` — and **raises** on `flash_attention_2` and `sdpa` ([why](../models/gpt-oss.md#attention-sinks)). The Docker Hopper FA3 build exposes no `s_aux`, so live-sinks runs on Hopper use `flex_attention` or `eager`.

Capability is read from the installed kernel's signature (`_attn_impl_handles_sinks`), probing the module transformers actually dispatches (FA3 via `flash_attn_interface` first, then `flash_attn_3`), so a flash-attn upgrade that adds sink support is picked up automatically.

The probe reads the **varlen** entry point (`flash_kernel`'s default): transformers builds its per-argument capability map from `_flash_varlen_fn` alone and applies it to the dense call too, so a sink argument only the dense signature accepts would never be forwarded.

The resolver takes the run's `SinksPolicy` and records its reset verdict on the config instance, so the nested EP/CP/lazy-loader re-validations (which pass no policy) enforce the same matrix. A from-scratch build resolves as live sinks: it has no pretrained sinks to neutralize.

**Gemma4** (5 full-attention layers at `global_head_dim=512`): FA2/FA3/FA4/cuDNN-SDPA all reject head_dim>256 (FA2 cap 256, cuDNN cap 128 on cu13, FA4's SM100 kernel overflows tensor memory and asserts in `flash_fwd_sm100`); math SDPA materializes `[B, heads, S, S]` (64 GB/layer at 32k → OOM).

The loader therefore redirects any FlashAttention impl — auto-detected or caller-supplied — to SDPA for any model whose widest head exceeds 256 (`has_head_dim_beyond_flash`, off the per-layer `head_dim`), and on SDPA `patch_sdpa_for_wide_heads` forces mem-efficient SDPA (the only backend supporting head_dim=512) and sets `transformers.integrations.sdpa_attention.use_gqa_in_sdpa → False` for manual KV repeat. On that path alone (`HALO_FLEX_SLIDING=0`), Gemma4 32k EP=8 peaks at ~155 GB/rank. On CUDA the model itself is built with `sdpa_flex_sliding`, which moves the sliding and short-context global layers off that kernel ([Gemma 4](../models/gemma4.md)).

**Qwen3.5 / Qwen3.6 / Qwen3-Next** train on FA4's head_dim-256 path, in every script: its general SM100 forward kernel and its dedicated head_dim-256 backward, for a 256-wide query/key head (`uses_fa4_dedicated_head_dim`, which counts MLA's `qk_nope_head_dim + qk_rope_head_dim` too; transformers' flash path pads a narrower value head up to that width). Qwen3.5's partial-rotary `torch.cat` hands that path q/k strided, which it reads as they are (below). The dedicated backward's deterministic mode is not validated here, so a run that asks for one (`FLASH_ATTENTION_DETERMINISTIC=1`, which `full_determinism` sets as the config is parsed, ahead of SFT's row plan and the model load) falls back from FA4 to **SDPA** on a head_dim-256 model FA4 otherwise trains (`supports_fa4_backward`), to FA2 under Context Parallelism, the kernel the Ulysses path then runs, and to FA2 on SFT rows that hold several documents (below).

SFT rows that hold several documents (`packing`, `padding_free`) load **FA2** where FA4 gains little on them or would leave them to SDPA (`default_training_rows_attn_implementation` in `halo/entrypoints/model_setup.py`). On the GatedDeltaNet families that holds when `attn_implementation` is unset and the GPU's pick is FA4, deterministic run or not (keyed on `uses_fa4_dedicated_head_dim`): FA4 runs packed rows only 2.9% faster at 16k and 4.6% at 32k (Qwen3.5-35B-A3B EP8 packed SFT on B300, FA4 4.0.0b37), below the 5% gain a default change needs, so a pinned FA4 keeps them outside `full_determinism` (the request logs that reason and that pin). Under `full_determinism` it holds on every model the run demotes off FA4 (`is_fa4_demoted_by_determinism`), a pinned `attn_implementation: flash_attention_4` included: SDPA runs such a row as one dense `[L, L]` mask with K/V repeated per query head (a one-step packed Qwen3.5-35B-A3B EP8 run: 367 s on SDPA, 162 s on FA2). FA2 consumes the `cu_seq_lens` the collator already emits. Unpacked rows keep FA4, 1.18× FA2 at 16k and 1.33× at 32k on the same model and FA4 release (one round), and padded rows of one document keep a deterministic run's SDPA. One shape keeps the loader's pick: `padding_free` on a Qwen3.5/3.6 decoder, text-only checkpoints and `text_only_model` loads included, whose M-RoPE varlen path faults FA2 (`has_mrope_rotary`, [Qwen3.5](../models/qwen3_5.md#configs)); collator selection refuses `padding_free` on an M-RoPE decoder on every backend.

Qwen3.5's partial-rotary `torch.cat` leaves q/k head-major before the flash transpose, MLA hands FA4 a value sliced off a fused K/V tensor, the Ulysses all-to-all at batch 1 hands it q, k and v each sliced off the one buffer it receives (token stride their summed width), and on every family a forward with the KV cache on leaves k/v head-major. FA4's head_dim-256 path reads those strided q/k/v as they are (Dao-AILab/flash-attention#2666): `tests/gpu/kernels/test_fa4_hd256_strided_inputs.py` holds its output and gradients to FA2's error against fp64 on those layouts, and `test_fa4_hd256_family_parity.py` per head_dim-256 family. The dedicated head_dim-256 backward asserts against softcap and sliding-window attention, so the resolver keeps Gemma 2 and Gemma 3 at head_dim 256 off it, as it keeps GLM-4.7-Flash (next paragraphs).

**Gemma 2 / Gemma 3 at head_dim 256 (2B/9B, 1B/4B/12B)**: auto-fall back from FA4 to **FA2** (`fa4_dedicated_kernel_refusal`), ahead of the deterministic-backward fallback, so a `full_determinism` run lands on FA2 too. FA4 trains a 256-wide query/key head only through its dedicated SM100 2-CTA backward kernel, which asserts on a softcap (Gemma 2's `attn_logit_softcapping`) and on a window (Gemma 3's sliding layers, once a row is longer than the window). The demotion reads the query/key width (`uses_fa4_dedicated_head_dim`, MLA's `qk_nope_head_dim + qk_rope_head_dim` included), the softcap and the declared `layer_types`, not the family: the hd128 27B sizes keep FA4, whose general kernel takes both, and a head_dim-256 model with neither keeps FA4 too.

**GLM-4.7-Flash (GLM-4 MoE Lite)**: multi-head latent attention at a 192 + 64 = 256-wide query/key head and a 256-wide value is kept on **FA2** on Blackwell as a precaution, in every run, a pinned `flash_attention_4`, a deterministic run and Context Parallelism included (`fa4_dedicated_kernel_refusal`, keyed on a `kv_lora_rank` latent at that width, `has_latent_attention`, not on the family). The precaution rests on a training result on FA4 4.0.0b16, not on a kernel defect: on its dedicated kernel (contiguous q/k/v) a run's step-2 gradient norm measured about 4x FA2's, a heavy tail in about 20 tokens of one document, while the kernel matched FA2 on that run's own tensors. No run measures the shipped 4.0.0b37 on this model, so the demotion stays. Latent attention at other widths (Mistral4's 128-wide query/key) keeps FA4's general kernel.

**Gemma 2 27B (head_dim 128) on varlen rows**: a softcap stays off FA4's varlen kernels (`fa4_varlen_refusal`): they are not validated with one. Every training batch but one unpadded row reaches them. Packed and padding-free rows hold several documents, and transformers unpads any batch holding a padding token, so more than one row per batch reaches it too (`rows_reach_varlen_kernel` in `halo/data/collators/collator_rules.py`). The resolver does not see the rows, so the scripts decide before the load. With the YAML pinning nothing, SFT, self-distillation and the embedding backbone the parallel loader loads request **FA2** for a softcapped model on such rows (`default_training_rows_attn_implementation`, a pre-processed corpus judged as it was baked; an fp32 run takes `flex_attention` instead), and the padded scripts request FA2 (Padded workloads, below). One unpadded row per batch (`per_device_train_batch_size: 1`, which `TrainingArguments` defaults to 8, and no packing) keeps FA4's dense kernel, which takes the softcap: the SFT collators hand a softcapped model's one-document rows no varlen set on FA4 ([One-document batches](../data/collators.md#one-document-batches)). Evaluation batches count the same way: a run that evaluates (`eval_strategy` other than `no`) at `per_device_eval_batch_size` above 1, which `TrainingArguments` defaults to 8, takes FA2 too. A reference takes the policy's request, so it runs FA2 beside a policy moved there, and a softcapped teacher moves the padded scripts' shared request as a softcapped student does. A pinned FA4 the load would keep on such rows is refused with the remedy before any weight loads (`reject_fa4_varlen_training`) in SFT, self-distillation, teacher distillation and the embedding parallel-loader path. Any softcapped call that still reaches FA4's varlen entry point, a pinned FA4 on the other padded scripts among them, raises there in the forward, before the kernel launches (`install_fa4_varlen_softcap_refusal`, which `apply_family_patches` installs on every FA4 load of a softcapped model).

**Attention logit softcapping (Gemma 2)**: SDPA is refused for any config setting `attn_logit_softcapping`, requested or defaulted, because transformers' SDPA path never reads the softcap and would train a different model without a word. The validator falls back to FA3 on Hopper, then FA2, then `flex_attention` (which applies the softcap as a score modifier), then `eager`; in fp32 it skips both flash kernels. FA3, the arch's own kernel, comes first on Hopper; a requested FA3 carries the softcap and stays.

**Ling 2.0 and Ring (`bailing_moe`, `bailing_moe_linear`)**: remote code on transformers v4's Llama flash template, whose table, support flag (`_supports_flash_attn_2`, which v5 ignores) and kernel call know only `flash_attention_2`. `patch_remote_llama_flash_template` (`halo/models/attention/remote_flash_attention.py`), run on every remote modeling module the remote-class funnel loads, makes the auto-selected FA4/FA3 build and run varlen over packed documents; the file's own SDPA and eager masks ignore `position_ids`, so packing is refused there ([Bailing → Attention and packing](../models/bailing.md#attention-and-packing)).

**Ling 3.0 (`bailing_hybrid`)**: the file has no flash attention class, only the v4 support flag, so every flash label, and an unset one outside the padded scripts' default (below), is refused at model build; set `attn_implementation: sdpa`. Its MLA mask ignores `position_ids` as Ling 2.0's SDPA path does, so packing is refused and Ling 3.0 trains on padded batches.

**DeepSeek-V4 (`deepseek_v4`)**: eager-only. `head_dim=512` exceeds every FlashAttention kernel's 256 cap, SDPA drops the learnable sink column, and the CSA/HCA compressors concatenate KV entries after the mask is built (no BlockMask resize), so transformers marks all non-eager backends unsupported. `resolve_attn_implementation` forces `attn_implementation="eager"` for any requested backend (the family spec's `eager_only_attention`). There is no varlen path under eager, so the collator factory rejects `padding_free` (see [Padding-Free Collator](padding-free-collator.md)). See [DeepSeek-V4](../models/deepseek-v4.md).

**Qwen3.8-Flash-Next (`qwen4_exp`)**: the class declares no flash or flex support, so the load takes SDPA. Its sparse (QSA) layers run Halo's tiled kernel, masked SDPA per 128-query tile with memory linear in the row ([Qwen3.8-Flash-Next](../models/qwen4_exp.md#attention-and-rows)).

**PaliGemma / PaliGemma 2 (prefix LM)**: image and prompt tokens attend bidirectionally and the trained tokens causally, a mask transformers builds from `token_type_ids` (0 prefix, 1 suffix). transformers also sets the text config's `use_bidirectional_attention` whenever a checkpoint leaves it unset, as every published one does, so each decoder attention module is built non-causal and attends to every token wherever no mask is materialized. No flash kernel reads `token_type_ids`, so the resolver never selects one for the family (`has_prefix_lm_attention` in `declares_flash_attention`), auto-detected or pinned: PaliGemma takes **SDPA**, PaliGemma 2's softcap rules SDPA out and it takes **`flex_attention`**; a pinned flex or eager stands. The VLM collators write the `token_type_ids` that put each row's trained tokens in the causal suffix ([SFT → Vision-language models](../training-methods/sft.md#vision-language-models)), and every training load makes the decoder causal where no mask overrides it (`apply_prefix_lm_causal_decoder`), so a batch without them scores causally.

**GPT-J, GPT-Neo, Falcon and Bark**: their modeling files list one flash class, FA2's, so `resolve_attn_implementation` refuses any other flash label, unset included, naming `flash_attention_2`. **`attention_dropout`**: transformers hands only FA2 its `dropout_p`, so a config with a non-zero value is refused on FA3 and FA4 rather than trained without it. **Live sinks** are read off the built model, under any family: `verify_live_sinks_carried` refuses a backend that drops them (Granite SWA on FA2 or FA3).

**Padded workloads**: the scripts that forward right-padded batches — DPO / SMPO / KTO, teacher distillation, reward modeling, classification, and all three GRPO scripts (offline, online, environmental) — share one default for an unset `attn_implementation` (`default_padded_attn_implementation`; a pinned value wins). On Blackwell they request nothing and take the loader's own pick, FA4 where it applies and every demotion above where it does not: at head_dim 256, FA4 for Qwen3.5/3.6, Qwen3-Next and Gemma 1, FA2 for Gemma 2/3 and GLM-4.7-Flash, never flash for PaliGemma (above). A remote-code checkpoint keeps **SDPA** there, since its class is invisible to the validator (Bailing's refuses every flash label at build). A softcapped one at any width, the policy's or a teacher's, moves the request off the loader's pick too, since its padded batches would reach FA4's varlen kernels: a softcapped policy requests **FA2**, which trains the softcap (SDPA where no FA2 build imports), and a softcapped teacher beside a policy without one leaves the request at SDPA, which the teacher's softcap turns into FA2 at its load. Every other GPU requests **SDPA**, which a softcap turns into FA3 on Hopper. Teacher distillation decides one request on both the student's and the teacher's checkpoint, and the DPO/KTO and GRPO references take the policy's request.

Every one of them takes that default only under `reset_sinks: true`, and offline GRPO and SMPO also skip it under CP, leaving an unset label to the loader's FlashAttention selection, which Ulysses requires. With live gpt-oss sinks (`reset_sinks: false`) the default drops and the model config passes through untouched, since SDPA drops the sink column and would be rejected outright. SFT keeps the auto-selected FA4 (FA2 for GLM-4.7-Flash, for a softcapped model and on the head_dim-256 GatedDeltaNet families' multi-document rows, above): a packed batch takes its varlen path on the set the packing collator builds, whose `max_length` is a host int. On FA4 the GRPO scorers forward each row densely, trimmed to its real span (`rows_forward_densely`, under GptOss above).

Measured on Halo 1.1.0 or earlier, 8× B300, with `benchmark_smpo_ep.py` (the gpt-oss-20b rows from a separate run set; Blackwell image; auto = FA4; mean of 2 runs). Its rows are padded close to `--seq`, so a batch with heavy padding is not measured. The SMPO collator emits no `input_ids`, so the callback's tokens/s is a padded-length estimate; the table gives the step-time speedup of auto over SDPA:

| model | seq | auto vs SDPA |
|---|---|---|
| gpt-oss-20b ep8 | 4096 | **1.36×** |
| gpt-oss-20b ep8 | 8192 | **1.42×** |
| qwen3-30b-a3b ep8 | 4096 | **1.17×** |
| qwen3-30b-a3b ep8 | 8192 | **1.14×** |
| qwen3-8b ep1 (dense) | 4096 | **1.26×** |

FA4 wins every row, on MoE and dense alike.

**Context Parallelism**: an explicit `flex_attention` label auto-switches to the kernel `_flash_attn_func` runs (`preferred_flash_impl` under `supports_fa4_backward`: `flash_attention_4` on Blackwell, `flash_attention_3` on Hopper, else `flash_attention_2`) since Ulysses can't dispatch flex; see [Supported backends](#supported-backends). SDPA/eager unsupported.

## Installation

Flash Attention is not in `pyproject.toml` — it must be built from source against your PyTorch ABI. The
prebuilt images already carry it; this is for a bare host.

```bash
# FA2 — restrict FLASH_ATTN_CUDA_ARCHS to your target arch (the default
# "80;90;100;110;120" builds 5 archs, ~5x compile time).
FLASH_ATTN_CUDA_ARCHS=100 MAX_JOBS=16 NVCC_THREADS=4 \
  pip install --no-cache-dir --no-build-isolation flash-attn

# FA3 (Hopper only)
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper && pip install --no-cache-dir --no-build-isolation .
```

On Hopper (sm_90) under CUDA 13.2, `ptxas` hangs on the 24 `flash_fwd_split_hdim*_*_sm80.cu` kernels. The Docker Hopper build replaces them with throw-stubs (`docker/training/flash_attn_split_stubs_hopper.cpp`) and rewrites `flash_api.cpp`'s two `/*num_splits*/ 0` arguments to `1`, asserting both: with dropout 0 the occupancy heuristic behind them picks split-K whenever `batch*heads*ceil(seq_q/64)` is under ~`0.8*2*SMs` (an unpadded row of a few hundred tokens at batch 1, and a decode step under GQA, whose varlen call takes split-K too), and the dense and varlen forwards then run the one-split kernel instead. The backward never uses split-K. Building FA2 for Hopper outside Docker on CUDA 13.2 needs the same two changes (or CUDA 13.1).

Import FA3 from `flash_attn_interface` (`import flash_attn_3` succeeds but does not guarantee a functional
build). transformers auto-detects FA3 through `is_flash_attn_3_available()`, which reads the
`flash_attn_interface` package metadata; the Docker Hopper build writes that dist-info and a `flash_attn_3`
re-export so both import forms resolve; `flash_kernel` reads `flash_attn_interface` first, as transformers dispatches.

After a PyTorch upgrade the `.so` is ABI-incompatible (`undefined symbol: _ZNK3c106SymInt...`) — rebuild with
`--force-reinstall`. If pip pulls a different torch as a build dependency, reinstall
`torch~=2.11.0` from the cu130 index.

## Usage

A YAML config can pin the backend (`attn_implementation: flash_attention_2`); omit the field for
auto-detection plus the model-specific handling above.

**Mask construction on flash.** Every flash load installs `patch_transformers_packed_sequence_probe`
(`halo/models/attention/backend_setup.py`, from `apply_family_patches`). Without it, transformers' causal and
sliding-window mask builders test `position_ids` for packed documents whenever no padding mask is passed
(`find_packed_sequence_indices`, one device read per mask at the start of every forward), and
`flash_attention_mask` discards the packed mask that test feeds. The skip reads the mask interface off the
registry per model, so an SDPA, eager or flex model in the same process keeps the probe its packed mask needs.

**Gradient checkpointing with EP.** EP MoE layers go through `enable_scoped_gradient_checkpointing`
(`halo/models/gradient_checkpointing.py`), which enables GC and installs the recompute scope its
recompute replays the DeepEP dispatch from. Enabling GC by any other route leaves EP without that scope
and raises.

EP, CP and every MoE force `use_reentrant=True` (a configured `False` is overridden with a warning; a
MoE routes in its recompute, and non-reentrant checkpointing rejects the routing shapes that change
when a nondeterministic attention recompute flips a near-tie pick). Pipeline parallelism is the
exception: a reentrant forward runs under `no_grad`, where FSDP2 registers no pre-backward hooks, so PP
stages run non-reentrant and refuse `use_reentrant: true`
([PP rejected combinations](../parallelism/pipeline-parallelism.md#rejected-combinations)). Under FSDP2,
a layer the reentrant backward recomputes more than once gets its post-backward completed by the trainer
([Data Parallelism](../parallelism/data-parallelism.md#layers-recomputed-more-than-once-per-backward)).

## Backend benchmark

Forward+backward on B300 (SM103), seq 16,384, GC on, BF16: SDPA and FA4 tie for the lead on both Qwen3-8B
dense (~11,080 tok/s/GPU) and Qwen3-30B-A3B MoE (~9,870), with flex at 1.23–1.33× their step time, FA2 at
1.42–1.57×, and eager at 6.55× on dense while the MoE OOMs. Memory is identical across working backends —
weights + activations dominate, not the kernel.

```bash
torchrun --nproc_per_node=1 \
    tests/gpu/profiling/benchmark_attention_implementations.py \
    [--model qwen3-30b-a3b --gradient_checkpointing] [--seq 8192] [--backends flash_attention_2 sdpa]
```

## Choosing a backend

Every row is what the loader picks on its own; the reason for each redirect is in
[Model-specific handling](#model-specific-handling).

| Scenario | Backend |
|----------|---------|
| Blackwell training | `flash_attention_4` |
| Hopper training | `flash_attention_3` (else FA2 when not installed) |
| No flash-attn | FA2 is still requested and the model build raises — set `attn_implementation: sdpa` yourself (CP then unavailable) |
| GptOss | FA4 on Blackwell, FA3 on Hopper |
| Qwen3.5 / Qwen3.6 / Qwen3-Next | FA4 on Blackwell (head_dim-256 path), padded scripts included; under `full_determinism` → SDPA; FA2 for SFT `packing` / `padding_free` rows in either case, except Qwen3.5/3.6 (M-RoPE) `padding_free` |
| GLM-4.7-Flash | → FA2 on Blackwell in every run, CP included (kept there as a precaution, [above](#model-specific-handling)) |
| Gemma 2 / Gemma 3 at head_dim 256 | → FA2 on Blackwell (FA4's head_dim-256 kernel takes no softcap or window); never SDPA under a softcap |
| Gemma 2 27B (head_dim 128) | FA4 at one unpadded row per batch, training and evaluation; → FA2 on packed, padding-free or padded rows (a softcap stays off FA4's varlen kernels) |
| Padded scripts (DPO, SMPO, KTO, RM, classification, distillation, GRPO), unpinned | the loader's pick on Blackwell (FA2 for a softcapped policy; SDPA for remote code and for a softcapped teacher, which lands on FA2); SDPA elsewhere, which a softcap turns into FA3 on Hopper |
| Gemma4 | → `sdpa_flex_sliding` on CUDA: FlexAttention on the sliding layers, matmul attention on the head_dim-512 global layers, past a score-memory budget mem-efficient attention per packed document ([Gemma 4](../models/gemma4.md)) |
| Ling 2.0 · Ring | FA4 on Blackwell, FA3 on Hopper (`patch_remote_llama_flash_template`) |
| Ling 3.0 | no fallback: the flash label fails the model build — set `attn_implementation: sdpa` |
| Laguna | through remote code → SDPA with a warning, and a pinned flash label raises: its flash call never receives the sliding window (`remote_code_flash_loses_window`). In-library classes: FA4 on Blackwell, FA3 on Hopper |
| LFM2-VL | → SDPA (its SigLIP2 tower declares no flash support, and `declares_flash_attention` reads each sub-model's class) |
| DeepSeek-V4 | → eager |
| GLM-5 Next · Inkling | → SDPA (upstream declares no flash support) |
| Step-3.7 Flash | → SDPA (upstream declares no flash support), built as `sdpa_flex_sliding` on CUDA: FlexAttention on the sliding layers and on the full layers' packed or padded rows |
| Qwen3.8-Flash-Next | → SDPA (upstream declares no flash support); its sparse layers run the tiled QSA |
| PaliGemma · PaliGemma 2 | → SDPA · → `flex_attention` (prefix LM: no flash kernel reads its `token_type_ids` mask, [above](#model-specific-handling)) |
| Context Parallelism | FA3 on Hopper, FA4 on Blackwell — FA2 for GLM-4.7-Flash |

## Known issues

**flex_attention + FSDP2 NaN gradients (GptOss).** Loss starts normally then collapses to 0 with NaN `grad_norm`. transformers wraps `flex_attention` in `torch.compile(dynamic=False)` (`WrappedFlexAttention`); the compiled backward produces NaN when GptOss sinks (plain `nn.Parameter`) interact with FSDP2's DTensor-wrapped LSE during post-attention renorm.

`patch_flex_attention_compile()` (`halo/models/attention/backend_setup.py`) bypasses the compile wrapper. `load_distributed_model` applies it whenever flex is the resolved backend and either EP is on (the compile path also deadlocks with EP's all-to-all on seq-length recompiles) or the model is a sinks model running with `reset_sinks: true`.

The bypassed flex runs uncompiled: torch's unfused path holds the full fp32 `[B, heads, L, L]` score
matrix per attention call, about 16 GiB at GPT-OSS's 64 query heads and an 8,192-token row, so under EP
memory bounds the row length, as it does on `eager` (bf16 scores, about half).

**flex_attention gradient overflow on dense Qwen3.** flex produces ~6e18 grad norms on Qwen3-0.6B/8B, even single-GPU without distributed wrapping; losses are fine but overflow is a stability risk. Use the auto-detected flash backend for training, not flex. Upstream PyTorch/transformers issue.

**FA4 first-use compile under multi-rank parallelism.** FA4 JIT-compiles each kernel on first use (~10 s) mid-forward. A cache-miss compile on one rank stalls it while peers race ahead to the next collective (a TP `o_proj` all-reduce, an EP DeepEP dispatch), desyncing the group into a deadlock.

`warm_attention_kernels` (`halo/model_loading/parallel/whole_model.py`, called at the end of `load_distributed_model`, of the frozen reference/teacher loader, of the embedding SentenceTransformer load and of a trainer's path-string model load, `resolve_trainer_model`) runs the rank-local compiles on every rank behind one fence (`barrier_on_exit`), so the training loop only hits warm kernels: `warm_fa4_kernels` (`halo/models/attention/backend_setup.py`) when FA4 is active, for each attention layer's own call (below; MLA's nope + RoPE query/key width) and each non-causal vision tower's, and `warm_flex_sliding_kernels` when the model runs `sdpa_flex_sliding` ([below](#sliding-window-and-wide-head-layers-on-sdpa)). It is a no-op at `world_size == 1`, where no peer waits on a compiling rank and each kernel compiles at its first call. [Tree packing](tree-packing.md) warms the kernel variants it adds behind the same fence (`warm_tree_attention_kernels`). The FA4 and tree warm-ups are best-effort: a failure logs a warning on that rank, which then compiles at first use. Only a failure that reaches the fence, a FlexAttention warm-up's today, raises on every rank.

It covers both entry points — dense `flash_attn_func` and varlen `flash_attn_varlen_func` over a two-document `cu_seqlens` — for each distinct call the model's attention modules make, each in the variants it can reach: windowless always (a row no longer than the window drops it), sliding-window where the layer has one, the layer's logit softcap where it passes one (Gemma 2; FA4 hashes the value into the kernel key), on the dense call only: a softcap stays off FA4's varlen kernels (above), so the warm-up launches no varlen call on a softcapped layer; and a learnable sink sized to the layer's heads when the model carries sinks. Each call passes the `deterministic` flag transformers passes its kernels (`FLASH_ATTENTION_DETERMINISTIC` as set at the load, read by `flash_attention_deterministic`), which FA4's backward keys its kernel on; the Ulysses CP call passes the same flag ([Context Parallelism → Requirements](../parallelism/context-parallelism.md#requirements)). Both warm-ups read that call off each module (`attention_calls` in `halo/models/attention/layout.py`): its own query heads, KV heads from its GQA group count, the query/key width (MLA's `qk_head_dim`), the value width (MLA's `v_head_dim`: FA4 warms the value at the query/key width, which transformers' flash path pads it up to, FlexAttention at its own), the window it passes (its config's where the module keeps none: Mistral, Phi-3) and its softcap. FA4 keys a kernel on the query/KV head ratio, so a per-layer head count gives each layer type its own kernel: Step-3.7's 64 full / 96 sliding heads, whose global `config.num_attention_heads` raises, Laguna's 48 / 72, whose global count is the full layers', and Inkling's sliding layers. `tests/cpu/models/test_attention_warmup_geometry.py` runs a training step of every roster model that declares flash support through transformers' flash path after the warm-up, with and without `FLASH_ATTENTION_DETERMINISTIC=1`, and checks that its forward and backward dispatch no kernel the warm-up skipped; `tests/gpu/kernels/test_attention_warmup_per_layer_heads.py` trains reduced Step-3.7, Laguna and Gemma 2 models after the warm-up (Gemma 2 on its single-document rows) and checks that no FA4 JIT cache and no FlexAttention graph gains an entry, and that no FA4 varlen call carrying a softcap, the warm-up's included, reaches the kernel.

The on-disk cache (`ensure_fa4_kernel_cache_env` in `halo/kernels/jit_cache.py`, anchored on `HF_HOME`) makes the compile a one-time cost across processes. The same call turns on `CUTE_DSL_ENABLE_TVM_FFI` where the environment leaves it unset, and pins Liger's RoPE to Triton, whose CuTe DSL kernel is wrong under it ([Liger → Kernel implementation](liger-kernels.md#kernel-implementation)). The DeepEP `ElasticBuffer` dispatch/combine timeout is long (100 s) and absorbs residual skew. FA2 is precompiled and unaffected.

**FA4 varlen `max_seqlen` (packed / masked batches).** A packed or attention-masked batch unpads to a `cu_seqlens` varlen FA4 call. transformers' `_process_flash_attention_kwargs` coerces the 0-dim `max_seqlen` tensor to `int` only under tracing; in eager it forwards a fresh CUDA tensor each step. On Blackwell FA4 reads no tensor bound on the host, and it compiles its head_dim-256 varlen kernels one way for a host-int bound (a rectangular grid) and another for a tensor one (a flat grid), so the bound's type picks the variant that runs.

`patch_transformers_flash_varlen_int_seqlen` (`halo/models/attention/backend_setup.py`, applied by `apply_family_patches` on every FA4 load) coerces `max_seqlen_{q,k}` to `int` at that boundary, so every call runs the int variant `warm_fa4_kernels` compiles, at one `.item()` per call on a masked batch. A packed flash row arrives with the varlen set already built on the host, by the packing collator ([Collators](../data/collators.md#collator-classes)) or, under pipeline parallelism, by each stage per microbatch ([PP packing](../parallelism/pipeline-parallelism.md#packing)), and so does an SFT batch of one unpadded document ([One-document batches](../data/collators.md#one-document-batches)), so its `max_seqlen` is an int from the start and no layer reads it back; the coercion serves the masked batches and rows that reach attention without the set. `warm_fa4_kernels` warms the varlen backward too, so the first packed step is already compiled.

A padded batch on any flash backend derives its unpad set (token indices, `cu_seqlens`, max length) once per mask, not in every layer: `patch_transformers_flash_unpad_memo` (`halo/models/attention/backend_setup.py`, applied by `apply_family_patches` on every flash load) keeps it on the mask tensor, keyed on its version counter, so a forward derives it once per mask kind and the checkpoint recompute not at all.

## Blackwell notes

vLLM on B200: set `VLLM_ATTENTION_BACKEND=FLASH_ATTN` where FlashInfer JIT-fails on SM 10.0. See [Online GRPO](../training-methods/grpo/online.md#vllm-on-blackwell-b200).

## Sliding-window and wide-head layers on SDPA

On CUDA, a model whose run resolves to `sdpa` and that has sliding-window layers (`has_sliding_window_layers`: Step-3.7, Gemma 4, and Cohere2 MoE or Laguna's in-library classes on a pinned `sdpa`) or a head wider than 256 (`has_head_dim_beyond_flash`; Gemma 4) is built with `sdpa_flex_sliding` (`halo/models/attention/sdpa_flex_sliding.py`), which `apply_family_patches` returns to every loader it patches: the policy loader, the frozen reference/teacher loader, the reward-model scorer, and a pinned-backend embedding SentenceTransformer load or trainer path-string load. Plain `sdpa` stays where every call would carry what the kernels here are not given: sinks models (GptOss), a config with `attn_logit_softcapping` (Gemma 2), and a class transformers does not declare backend-compatible (`is_backend_compatible()`; Inkling, whose every call carries a dense relative bias) or does not map at all (remote code, whose attention need not dispatch through the registry). `HALO_FLEX_SLIDING=0` opts out.

- **Masked causal layers** (every sliding layer, and a full layer no wider than 256 under a packed or padded mask; a query/key or value head narrower than 16, which Inductor's FlexAttention kernel does not take, stays on SDPA) run compiled FlexAttention under a `BlockMask` derived from the dense mask transformers built, so causality, the window, packed-document isolation and padding carry over and the tiles outside the window and across documents are skipped. Under SDPA the same layers compute every tile of the dense mask. Each query's allowed keys in such a mask are one contiguous run, so the mask reduces to the first and last key per query (`key_intervals`, read 256 query rows at a time), and the kernel's mask function compares against those two `[Q]` tensors: no index over the `[Q, K]` grid (whose flattened offset outgrows int32 past about 46k tokens) and no copy of the dense mask is kept. A mask that allows some query a non-contiguous set of keys goes to SDPA. The block masks are built once per mask tensor (every sliding layer of a forward, and its gradient-checkpoint recompute, gets the same one) and are freed with it.
- **One compiled shape family.** Each batch row runs alone, padded to a multiple of 128, always under a mask and with autocast off inside the kernel. Under autocast the inputs first take the autocast dtype, as SDPA's do, so fp32 parameters under bf16 autocast, whose fp32 rotary embedding promotes the queries and keys beside bf16 values, still run the warmed bf16 graph and the tuned tiles. A call either records a graph with every input requiring grad (an input left frozen, as by a LoRA that adapts `q_proj` alone, joins as a detached leaf) or runs under `no_grad`. The kernel therefore compiles twice, a training graph and a forward-only one, for any mix of lengths, batch sizes, grad modes and trained projections. `warm_flex_sliding_kernels` compiles both for each call the model's attention modules make on FlexAttention (head count, read off each module since Step-3.7's and Laguna's vary by layer, query/key and value widths, window and `scaling`, which Dynamo guards on), so no rank compiles mid-run, and FlexAttention takes only that call: the module's heads, its query/key and value widths, its `scaling`, and a mask over exactly the call's queries and keys. Any other call runs SDPA, with a one-time warning. Under tensor parallelism a call carries the rank's share of the heads, which the module does not declare, so a TP run on a pinned `sdpa` (Cohere2 MoE, or a dense sliding-window model on HF-native TP) keeps SDPA on these layers. Deterministic-algorithms mode is a guard too, and HF's `Trainer` turns it on after the load under `full_determinism`, so each rank would compile again at its own first long masked call while its peers wait in the next collective. Where the warm-up ran (world size > 1), every trainer therefore refuses `full_determinism` at construction (`reject_full_determinism_after_warmup`); with `HALO_FLEX_SLIDING=0` the model runs plain SDPA and takes it. The tuned tiles (head_dim 256, 16-bit inputs, SM100+) do not depend on the length, since a length-dependent choice is itself a recompile guard; other shapes use FlexAttention's defaults. The contract exists because Dynamo guards on a batch of one, a tile-multiple length, grad mode, each input's dtype and `requires_grad`, the autocast state and whether a block mask carries a mask tensor: a mid-run compile stalls the other ranks in the next collective, and past Dynamo's recompile limit (8) FlexAttention runs unfused over the full score matrix.
- **Short masked calls** (a query no longer than the window, or than one 128-token tile) go to SDPA: no tile would be skipped. A full layer over one unpadded document gets no mask and keeps SDPA's causal kernel.
- **Causal global layers** run matmul attention where SDPA has only its mem-efficient kernel (head_dim above 256, or the flash backend disabled process-wide, as the Gemma 4 loader does) and the scores saved for backward fit `MATMUL_GLOBAL_BUDGET_BYTES` (2 GiB). The logits stay fp32, as in mem-efficient SDPA: the score GEMM takes the 16-bit inputs and writes fp32, at 16-bit GEMM speed. At Gemma 4's global shape (bf16, `scaling` 1.0, RMS-normed 512-wide heads, unit norm weights, 1,024 tokens) the output's error against fp64 is 1.25x mem-efficient SDPA's, and the query and key gradients' error is half of SDPA's; bf16 logits would put the output's at 18x. Under autocast the GEMM takes the autocast dtype, as SDPA does. With no mask the layer attends as SDPA does: causally from the first key, or every key for a single query or an explicit `is_causal=False`. The budget is per layer: without gradient checkpointing every global layer holds its share. At 4,096 tokens a Gemma 4 26B-A4B global layer saves 1.5 GiB against 0.13 GiB on mem-efficient SDPA, about 7 GiB more across its 5 global layers.
- **Packed global layers past the budget** with heads wider than 256 run mem-efficient attention per document. Where every query of the mask attends exactly its own document, from the document's first token up to itself (what transformers builds for a packed row), the documents' boundaries over every batch row (`causal_documents`, read once per mask tensor) go to the varlen form of the kernel SDPA runs these heads on (`_efficient_attention_forward`/`_backward` with `cu_seqlens`, causal from each document's first query and key), which visits each document's own tiles only. The KV heads are repeated per query group, as SDPA repeats them, and nothing compiles. Any other mask (a padded query, a bidirectional span, a window) goes to SDPA, whose mem-efficient kernel computes every tile of the dense mask.
- **Everything else** goes to the implementation registered as `sdpa`: bidirectional modules, dropout, float masks, `output_attentions`, and calls carrying sinks (`s_aux`), a softcap or a position bias.

`tests/gpu/kernels/test_flex_masked_layers.py` checks a tiny Step-3.7 on packed and padded rows, and a tiny Gemma 4 with 512-wide global layers past the budget on packed rows, against the SDPA model (loss and every gradient, the path counted, a padded row's global layers left on SDPA); at Step-3.7's and Gemma 4's real attention shapes, the output and input gradients against a per-document fp64 reference within 1.5x mem-efficient SDPA's error on the dense mask; packed documents bitwise isolated from each other; and no compile after the warm-up, a call carrying a tensor-parallel rank's share of the heads (left on SDPA) included. `tests/cpu/models/test_attention_warmup_geometry.py` holds the warm-up to that contract at Dynamo's level on CPU, tracing the kernel on Dynamo's eager backend (Inductor's own guards stay the GPU tests') under `error_on_recompile`: on tiny Step-3.7, Laguna and Mistral no packed or single-document row, trained or forward-only, traces a graph after the warm-up, and each row runs FlexAttention on exactly the layers it masks (every layer of a packed row, the sliding ones of a single document). `tests/cpu/models/test_flex_sliding_masked_layers.py` checks the resolver over the tiny roster, that the warm-up's calls are the ones a packed forward makes and each is the call its module declares, that FlexAttention refuses any other call (a rank's share of the heads, another value width or scaling, a mask over other queries or keys), that a packed full layer's block masks hold exactly its tiles, that only a document-causal mask yields documents, and that every tiny model it serves, dense Mistral (a window on the config alone), Cohere2 (NoPE full layers) and OLMo 3 (q/k norm) included, reproduces the SDPA model's loss and every gradient on a packed row and a right-padded batch longer than the window, every layer on uncompiled FlexAttention reading each block mask as the compiled kernel does. `tests/gpu/kernels/test_sdpa_flex_sliding.py` checks Gemma 4 and Mistral (a window on every layer) against the SDPA model, loss and every gradient, on packed rows longer than the window; rows of ~47.5k tokens against SDPA; that one warm-up leaves no compile for later lengths, batch sizes, masks, grad modes, trained projections and autocast, and none for a Gemma 4 forward and backward, fp32 parameters under bf16 autocast included; that the block masks are freed with their mask; that the global layers' error at Gemma 4's shape stays within 1.5x mem-efficient SDPA's, their saved bytes match the budget's count, their unmasked calls match SDPA, and the score GEMM follows autocast; and that `full_determinism` is refused after the warm-up.
