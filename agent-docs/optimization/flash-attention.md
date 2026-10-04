# Flash Attention & Attention Backends

`load_distributed_model` (`src/distributed/loading/model_loading.py`) auto-detects the backend per GPU and model: `flash_attention_4` on Blackwell, `flash_attention_3` on Hopper, `flash_attention_2` otherwise. Pass `attn_implementation` only to override it.

FA2 is the detector's terminal default, with no availability probe — every shipped image carries it, and a host without it raises at model build. Context Parallelism hard-requires flash-attn and raises `ImportError` when it is missing.

Why tiling attention is the standard fix for the model's most memory-bound op: [GPU Training Theory §6](../reference/gpu-training-theory.md#6-attention--the-canonical-memory-bound-fix).

## Supported backends

| Backend | GPU | CP | GptOss |
|---------|-----|----|--------|
| `flash_attention_4` | Blackwell SM100+ (CuTe DSL) | Yes | Yes — every sink policy (neutralized, live-frozen via `learnable_sink`, trainable via the sink-gradient rescale) |
| `flash_attention_2` | Ampere+ | Yes | Yes (sinks reset; live sinks rejected) |
| `flash_attention_3` | Hopper SM90 only | Yes | Yes (sinks neutralized; live sinks need an `s_aux` build, trainable sinks rejected) |
| `flex_attention` | torch ≥2.5 | No (auto-switched to FA4/FA2) | Yes (LSE renorm; FSDP2 NaN fix auto-applied) |
| `sdpa` | all | No | Yes when sinks reset (reset → dropped sink column is `exp(dtype.min)=0`; loader flips `_supports_sdpa`). Rejected when sinks are kept live. |
| `eager` | all | No | Yes |

GPU support: A100 (SM80) = FA2/flex/sdpa. H100/H200 (SM90) = FA2/FA3/flex/sdpa, no FA4. B200 (SM100) / B300 (SM103) = FA2/FA4/flex/sdpa, no FA3.

The CP column above is what `validate_model_for_ulysses` (`src/distributed/context_parallel/validation.py`) accepts as a config label; it is not which kernel runs. CP's attention call goes through `get_flash_attn_func` (`src/distributed/context_parallel/base_layer.py`), which ignores that label and probes in a fixed order: FA3 on Hopper (SM90), FA4 on Blackwell (SM100+), then FA2, then the `kernels-community/flash-attn2` kernel.

Both architecture probes precede FA2 because every image ships FA2; an FA2 probe placed first would resolve on every device and leave the arch-matched kernel unreachable.

The one exception is model-level: `get_flash_attn_func` takes an `allow_fa4` veto, resolved per wrapper from
`model_fa4_backward_nan_prone`, so the FA4-backward-NaN families (Qwen3.5/3.6/Qwen3-Next, GLM-4 MoE Lite)
run FA2 on Blackwell under CP. The probe overrides the configured label, so that veto is their only
protection. Its device predicates are the same `is_hopper_gpu` / `is_blackwell_gpu` the automatic detection
uses, so CP and the config-level choice can never disagree about a device.

The order matters because CP calls the **non-varlen** forward, the path FA2 serves worst on both architectures. On Hopper the image stubs out the split-K kernels FA2 selects there by occupancy heuristic.

On B300, CP on FA4 measures 1.2–3× FA2 at ≥32k tokens with equal peak memory, the margin widening with sequence length (+19% at 32k, +63% at 128k on gpt-oss-20b ep8+cp8; +197% on dense Qwen3-4B ep1+cp8 at 128k). FA2 is faster only below ~16k, where per-call overhead dominates.

## Automatic detection

`_detect_attention_impl` (`src/models/patches/attention.py`) reads only the compute-capability major:

1. `major >= 10` (Blackwell SM100/SM103): `flash_attention_4` if `flash_attn.cute` imports, else `flash_attention_2`. A build that is present but unusable warns before degrading.
2. `major == 9` (Hopper): `flash_attention_3` if `flash_attn_3` imports, else `flash_attention_2`.
3. Anything older, or any exception: `flash_attention_2`.

The `major >= 10` test also admits consumer Blackwell (SM 12.x, RTX 50-series), where `flash_attn.cute`'s
SM100 kernels do not run. That is not a supported target — the images build for B200/B300 and H100/H200.

`resolve_attn_implementation` then applies the model-level overrides below. One is precision-driven: in
**fp32**, any `flash_attention*` is redirected to `flex_attention` when the model has sinks and `sdpa`
otherwise — flash kernels are half-precision only. Context Parallelism in fp32 **raises**, since CP runs
attention through FlashAttention unconditionally.

The Blackwell image ships both FA2 (top-level `flash_attn`, inherited from the base image) and FA4
(`flash-attn-4` 4.0.0b16, the `flash_attn.cute` submodule) under one namespace; FA4 is the production
default. Profiling benchmarks leave `--attn_implementation` unset, so they run FA4 on Blackwell too.
transformers dispatches `flash_attention_4` natively through `flash_attn.cute` (causal, sliding-window,
sinks via `s_aux`, softcap, varlen). 4.0.0b16 is beta — validate accuracy-sensitive runs. Hopper keeps
FA2 + FA3.

## FA4 vs FA2 vs SDPA on Blackwell

FA2 is an SM80-style kernel, untuned for Blackwell. FA4 is the Blackwell-native CuTe DSL kernel: **3.6–3.9× faster than FA2 on the isolated attention kernel** (microbench B2×S8192×H32×D128: fwd 2.71→0.70 ms, 3.9×; fwd+bwd 10.9→3.0 ms, 3.6×). Both match an fp32 SDPA reference to ~2e-3 (bf16 floor) across MHA/GQA and head_dim 64/128.

The end-to-end step win is a function of attention's share of the step, which grows with sequence length (attention is O(seq²), the rest ~O(seq)). Qwen3-4B-Instruct-2507, batch 1, GC on, Liger on, tok/s/GPU on one B300 (every cell on the same device), mean of two runs, measured 2026-10-03 at commit 0bc3a22a5 on the Blackwell image. Runs repeat within 1.1%, except flex at 4,096 (7.9%):

| SeqLen | FA4 | FA2 | SDPA | flex | FA4/FA2 |
|--------|-----|-----|------|------|---------|
| 4,096 | 23,614 | 19,920 | 25,733 | 20,147 | 1.19× |
| 16,384 | 23,139 | 12,282 | 23,112 | 16,167 | 1.88× |
| 32,768 | 17,504 | 7,497 | 17,047 | 10,607 | 2.33× |

At 4k batch-1 the step is overhead-bound, so FA4 stays close to FA2 (1.19×); the gap opens with length and reaches 2.3× at 32k. SDPA (Blackwell-tuned cuDNN kernel) leads FA4 by 9% at 4k and is within 3% of it from 16k, so SDPA is a fine fallback for plain dense models. flex trails both at every length; it ties FA2 at 4k (within its own 7.9% run spread) and beats it from 16k.

On sparse MoE the step is dominated by expert GEMM + DeepEP all-to-all, so the kernel speedup is only **+14%** end-to-end. gpt-oss-20b, EP=8, seq 16,384, batch 1, GC on, 8× B300 (2026-10-03, commit 0bc3a22a5, Blackwell image): FA4 9,981 vs FA2 8,737 tok/s/GPU (1.14×); the FA4 arm is the s16384 row of the ep8 sequence sweep in [Throughput Benchmarks](throughput-benchmarks.md). FA4 is the shipped default for GptOss SFT; SDPA also dispatches once sinks are reset but lacks FA4's native sink/sliding-window/softcap handling.

EP/long-context throughput tables live in [Throughput Benchmarks](throughput-benchmarks.md), which already run the FA4 default.

## Model-specific handling

`load_distributed_model` applies these automatically.

**GptOss** (attention sinks):

1. Disables transformers' auto-fallback to `vllm-flash-attn3` (`_disable_gpt_oss_fa_fallback`). transformers reroutes FA2 requests to `kernels-community/vllm-flash-attn3` (SM90-only), which crashes on Blackwell; the attribute is reset to `None` before `from_pretrained`.
2. Applies the sink policy (`src/models/patches/gpt_oss_sinks.py`). The Blackwell default for GptOss is FA4 under every policy. See [GPT-OSS](../models/gpt-oss.md#attention-sinks).

    **SFT** neutralizes the sinks: FA2 (no `s_aux`) gets `self_attn.sinks = None`; FA3/FA4/flex/eager get `dtype.min`, frozen (preserves shape, ~0 softmax contribution). FA2 then runs end-to-end on B200/B300, equivalent to `flex_attention`.

    **RL** (online/async GRPO, `reset_sinks: false`) keeps the pretrained sinks live and frozen. The rollout engine serves the same sinks, so the trainer reads them via FA4 (`learnable_sink`) to stay on-policy.

    `train_sinks: true` keeps them live and trainable, on FA4 through the sink-gradient rescale or on `eager`; every other implementation raises rather than training nothing. Full fine-tuning only: an adapter run raises (the adapter artifact has no slot for the sinks), and the RL weight sync refuses it.

3. **Variable-length RL (async GRPO) runs FA4 via a per-row dense forward** (`_dense_last_hidden_state`). A padded batch unpads to a `cu_seqlens` **varlen** FA4 call whose cross-row attention would break per-row log-prob identity with the sampling policy.

    The dense kernel is seqlen-invariant, so the path forwards each row on its own, trimmed to its real contiguous span, with `attention_mask=None`: RoPE positions `[0, len)`, log-probs bit-identical to the unpadded forward. FA4 is where the trim is *mandatory*; `rows_forward_densely` also takes it on any attention path at `per_device_train_batch_size: 1`, for the cost alone ([Chunked log-probs](../training-methods/grpo/async-grpo/performance.md#chunked-log-probs)).

    Per-rank forward counts are all-reduced to a max and padded with tied dummy forwards, and every row — the padding included — stays connected to the loss, so the forward **and** backward collective sequences stay in lockstep. A disconnected row would let autograd prune its backbone backward and desync the FSDP/EP grad collectives across ranks.

Fallback chain for sinks models: requested → `flash_attention_4` (added as a candidate when the sinks are live) → `flash_attention_2` → `flex_attention` → SDPA (only when sinks are reset) → `eager` (`validate_attn_implementation`); under live sinks every sink-dropping candidate is filtered out.

With sinks reset for SFT (the default), an explicit `attn_implementation='sdpa'` is kept — `_enable_sink_model_sdpa` flips `_supports_sdpa` because the dropped sink column is `exp(dtype.min)=0`.

With **live** sinks (`reset_sinks: false`, the RL flow) a sink-dropping impl shifts every logprob by nats against the served policy (measured ~−3 nats on gpt-oss-20b under FA2/SDPA), so `validate_attn_implementation` **raises** on `flash_attention_2` and `sdpa` and only admits sink-carrying impls — FA4 (`learnable_sink`), FA3 with `s_aux` (Hopper), `flex_attention`, `eager`. The Docker Hopper FA3 build exposes no `s_aux`, so live-sinks runs on Hopper use `flex_attention` or `eager`.

Capability is read from the installed kernel's signature (`_attn_impl_handles_sinks`), probing the module transformers actually dispatches (FA3 via `flash_attn_interface` first, then `flash_attn_3`), so a flash-attn upgrade that adds sink support is picked up automatically.

The probe reads the **varlen** entry point (`flash_varlen_fn`): transformers builds its per-argument capability map from `_flash_varlen_fn` alone and applies it to the dense call too, so a sink argument only the dense signature accepts would never be forwarded.

The `reset_sinks` decision is recorded on the config instance, so the nested EP/CP/lazy-loader re-validations (which pass no `sinks_reset`) enforce the same matrix.

**Gemma4** (5 full-attention layers at `global_head_dim=512`): FA2/FA3/FA4/cuDNN-SDPA all reject head_dim>256 (FA2 cap 256, cuDNN cap 128 on cu13, FA4's SM100 kernel overflows tensor memory and asserts in `flash_fwd_sm100`); math SDPA materializes `[B, heads, S, S]` (64 GB/layer at 32k → OOM).

The loader therefore redirects any FlashAttention impl — auto-detected or caller-supplied — to SDPA for any model whose widest head exceeds 256 (`head_dim_exceeds_flash`, off the per-layer `head_dim`), and on SDPA `patch_sdpa_for_wide_heads` forces mem-efficient SDPA (the only backend supporting head_dim=512) and sets `transformers.integrations.sdpa_attention.use_gqa_in_sdpa → False` for manual KV repeat. On that path alone (v1.0.0), Gemma4 32k EP=8 peaked at ~155 GB/rank. On CUDA the model itself is built with `sdpa_flex_sliding`, which moves the sliding and short-context global layers off that kernel ([Gemma 4](../models/gemma4.md)).

**Qwen3.5 / Qwen3.6 / Qwen3-Next and GLM-4 MoE Lite (GLM-4.7-Flash)**: auto-fall back from FA4 to **SDPA** (`model_fa4_backward_nan_prone`). The FA4 beta backward emits **NaN gradients** on these models — forward is finite, the first backward goes non-finite and collapses loss to 0 (NaN `grad_norm`).

The shared trigger is a **head_dim-256 attention with partial rotary**: Qwen3.5/3.6/Qwen3-Next pair QK-norm + partial rotary + an attention output gate; GLM-4.7-Flash is MLA-style (`qk_nope_head_dim` 192 + `qk_rope_head_dim` 64 = a 256-wide query/key, `v_head_dim` 256, rope on only the 64-dim split). SDPA keeps fp32 reductions and trains cleanly. gpt-oss is unaffected (head_dim 64) and keeps FA4.

**Bailing / Ling (`bailing_moe`, `bailing_moe_linear`, `bailing_hybrid`)**: the remote code declares only the v4-era `_supports_flash_attn_2`, which transformers v5 ignores in favor of `_supports_flash_attn`, so every flash label is refused at model build and the family runs SDPA.

A Bailing run therefore sets `attn_implementation: sdpa` itself; an unset or flash label raises at model build on every load path (under CP the Ulysses wrapper waives the flash-label check; see [Bailing](../models/bailing.md#cp-wrapper)).

**DeepSeek-V4 (`deepseek_v4`)**: eager-only. `head_dim=512` exceeds every FlashAttention kernel's 256 cap, SDPA drops the learnable sink column, and the CSA/HCA compressors concatenate KV entries after the mask is built (no BlockMask resize), so transformers marks all non-eager backends unsupported. `resolve_attn_implementation` forces `attn_implementation="eager"` for any requested backend (`_model_is_deepseek_v4`). There is no varlen path under eager, so the collator factory rejects `padding_free` (see [Padding-Free Collator](padding-free-collator.md)). See [DeepSeek-V4](../models/deepseek-v4.md).

**Padded workloads**: the scripts that forward right-padded batches — DPO / SMPO / KTO, teacher distillation, reward modeling, classification, and all three GRPO scripts (offline, online, environmental) — default `attn_implementation` to **SDPA** when the YAML sets none (`padded_workload_attn_implementation` / `attn_default="sdpa"`, both overridable).

Every one of them takes that default only under `reset_sinks: true`. With live gpt-oss sinks (`reset_sinks: false`) the default drops and the model config passes through untouched, since SDPA drops the sink column and would be rejected outright. SFT keeps the auto-selected FA4: a packed batch takes its varlen path, kept fast by the `max_seqlen` int-coercion.

That default costs throughput at the lengths these methods actually run. Measured on 8× B300 with `benchmark_smpo_ep.py`, which does *not* apply it (2026-10-04, training code at commit 0bc3a22a5, Blackwell image; auto = FA4; mean of 2 runs). The SMPO collator emits no `input_ids`, so the callback's tokens/s is a padded-length estimate; the table gives the step-time speedup of auto over SDPA:

| model | seq | auto vs SDPA |
|---|---|---|
| gpt-oss-20b ep8 | 4096 | **1.41×** |
| gpt-oss-20b ep8 | 8192 | **1.40×** |
| qwen3-30b-a3b ep8 | 4096 | **1.17×** |
| qwen3-30b-a3b ep8 | 8192 | **1.14×** |
| qwen3-8b ep1 (dense) | 4096 | **1.26×** |

FA4 wins every row, on MoE and dense alike. Set `attn_implementation: flash_attention_4` explicitly in a preference/reward config.

**Context Parallelism**: an explicit `flex_attention` label auto-switches to `flash_attention_4` (Blackwell) / `flash_attention_2` (Hopper) since Ulysses can't dispatch flex — the label is cosmetic, though: see [Supported backends](#supported-backends) for what `get_flash_attn_func` actually runs. SDPA/eager unsupported.

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

On Hopper (sm_90) under CUDA 13.2, `ptxas` hangs on the 24 `flash_fwd_split_hdim*_*_sm80.cu` kernels. The Docker Hopper build replaces them with throw-stubs (`docker/training/flash_attn_split_stubs_hopper.cpp`). Building FA2 for Hopper outside Docker on CUDA 13.2 needs the same workaround (or CUDA 13.1).

`mha_varlen_fwd` gates split-K behind `seqlenq_ngroups_swapped`, so packed/padding-free training and the whole backward never reach these kernels. `mha_fwd` (non-varlen) does **not** gate it: with dropout 0 the occupancy heuristic picks split-K whenever `batch*heads*ceil(seq_q/64)` is under ~`0.8*2*SMs`, so a non-varlen caller must prefer FA3 on sm_90, which `src/distributed/context_parallel/base_layer.py` does.

Import FA3 from `flash_attn_interface` (`import flash_attn_3` succeeds but does not guarantee a functional
build). transformers auto-detects FA3 through `is_flash_attn_3_available()`, which reads the
`flash_attn_interface` package metadata; the Docker Hopper build writes that dist-info and a `flash_attn_3`
re-export so both import forms resolve, while `_detect_attention_impl` keys on `import flash_attn_3`.

After a PyTorch upgrade the `.so` is ABI-incompatible (`undefined symbol: _ZNK3c106SymInt...`) — rebuild with
`--force-reinstall`. If pip pulls a different torch as a build dependency, reinstall
`torch~=2.11.0` from the cu130 index.

## Usage

A YAML config can pin the backend (`attn_implementation: flash_attention_2`); omit the field for
auto-detection plus the model-specific handling above.

**Gradient checkpointing with EP.** EP MoE layers go through `enable_ep_gradient_checkpointing`
(`src/distributed/expert_parallel/patching.py`), which enables GC and installs the checkpoint scope its
recompute replays the DeepEP dispatch from. Enabling GC by any other route leaves EP without that scope
and raises.

Outside pipeline parallelism, EP, CP and every MoE force `use_reentrant=True` (a configured `False` is
overridden with a warning; a MoE routes in its recompute, and non-reentrant checkpointing rejects the
routing shapes that change when a nondeterministic attention recompute flips a near-tie pick). Under PP ([not yet available in this release](../parallelism/pipeline-parallelism.md))
the shipped gates require non-reentrant instead, so the override would not apply there; EP replays its
dispatch from the checkpoint scope in either mode.

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
| Qwen3.5 / Qwen3.6 / GLM-4.7-Flash | → SDPA (FA4 backward NaN) |
| Gemma4 | → `sdpa_flex_sliding` on CUDA: FlexAttention on the sliding layers, matmul attention (mem-efficient SDPA past a score-memory budget) on the head_dim-512 global layers ([Gemma 4](../models/gemma4.md)) |
| Bailing / Ling | no fallback: the flash label fails the model build — set `attn_implementation: sdpa` |
| DeepSeek-V4 | → eager |
| GLM-5 Next · Step-3.7 Flash · Inkling | → SDPA (upstream declares no flash support) |
| Context Parallelism | FA3 on Hopper, FA4 on Blackwell — FA2 for the FA4-NaN families |

## Known issues

**flex_attention + FSDP2 NaN gradients (GptOss).** Loss starts normally then collapses to 0 with NaN `grad_norm`. transformers wraps `flex_attention` in `torch.compile(dynamic=False)` (`WrappedFlexAttention`); the compiled backward produces NaN when GptOss sinks (plain `nn.Parameter`) interact with FSDP2's DTensor-wrapped LSE during post-attention renorm.

`patch_flex_attention_compile()` (`src/models/patches/attention.py`) bypasses the compile wrapper. `load_distributed_model` applies it whenever flex is the resolved backend and either EP is on (the compile path also deadlocks with EP's all-to-all on seq-length recompiles) or the model is a sinks model running with `reset_sinks: true`.

**flex_attention gradient overflow on dense Qwen3.** flex produces ~6e18 grad norms on Qwen3-0.6B/8B, even single-GPU without distributed wrapping; losses are fine but overflow is a stability risk. Use the auto-detected flash backend for training, not flex. Upstream PyTorch/transformers issue.

**FA4 first-use compile under multi-rank parallelism.** FA4 JIT-compiles each kernel on first use (~10 s) mid-forward. A cache-miss compile on one rank stalls it while peers race ahead to the next collective (a TP `o_proj` all-reduce, an EP DeepEP dispatch), desyncing the group into a deadlock.

`warm_attention_kernels` (`src/distributed/loading/warmup.py`, called at the end of `load_distributed_model` and of the frozen reference/teacher loader) runs the rank-local compiles on every rank behind one barrier, so the training loop only hits warm kernels: `warmup_fa4_kernels` (`src/models/patches/attention.py`) when FA4 is active, and `warmup_flex_sliding_kernels` when the model runs `sdpa_flex_sliding` ([below](#sliding-window-and-wide-head-layers-on-sdpa)). It is a no-op at `world_size == 1`, where no peer waits on a compiling rank and each kernel compiles at its first call.

It covers both entry points — dense `flash_attn_func` and varlen `flash_attn_varlen_func` over a two-document `cu_seqlens` — each in the variants the model can reach: sliding-window when the config declares one, and a learnable sink when the model carries sinks.

The on-disk cache (`ensure_fa4_kernel_cache_env`, anchored on `HF_HOME`) makes the compile a one-time cost across processes. The DeepEP `ElasticBuffer` dispatch/combine timeout is long (100 s) and absorbs residual skew. FA2 is precompiled and unaffected.

**FA4 varlen backward JIT-cache key (packed / masked batches).** A packed or attention-masked batch unpads to a `cu_seqlens` varlen FA4 call. transformers' `_process_flash_attention_kwargs` coerces the 0-dim `max_seqlen` tensor to `int` only under tracing; in eager it forwards a fresh CUDA tensor each step, and `flash_attn.cute`'s varlen backward keys its JIT cache on that tensor by *identity*.

The backward then recompiles every step: ~190 s/step vs ~10 s on gpt-oss-20b ep4, flat across sequence length because it is codegen of the backward template, not real attention.

`patch_transformers_flash_varlen_int_seqlen` (`src/models/patches/attention.py`, applied by `load_distributed_model` when FA4 is active) coerces `max_seqlen_{q,k}` to `int` at that boundary, so the key caches by value and the backward compiles once — one `.item()` per call, where the forward already syncs. `warmup_fa4_kernels` warms this varlen backward too, so the first packed step is already compiled.

## Blackwell notes

vLLM on B200: set `VLLM_ATTENTION_BACKEND=FLASH_ATTN` where FlashInfer JIT-fails on SM 10.0. See [Online GRPO](../training-methods/grpo/online-grpo.md#vllm-on-blackwell-b200).

## Sliding-window and wide-head layers on SDPA

On CUDA, a model whose run resolves to `sdpa` and whose widest head exceeds 256 (`head_dim_exceeds_flash`; Gemma 4) is built with `sdpa_flex_sliding` (`src/models/patches/flex_sliding_attention.py`), which `apply_family_attention_patches` returns to the policy loader and the frozen reference/teacher loader alike. Sinks models keep `sdpa`. `HALO_FLEX_SLIDING=0` opts out.

- **Sliding layers** run compiled FlexAttention under a `BlockMask` derived from the dense mask transformers built, so causality, the window, packed-document isolation and padding carry over and the tiles outside the window are skipped. Under SDPA the same layers compute every tile of the dense mask. Each query's allowed keys in such a mask are one contiguous run, so the mask reduces to the first and last key per query (`key_intervals`, read 256 query rows at a time), and the kernel's mask function compares against those two `[Q]` tensors: no index over the `[Q, K]` grid (whose flattened offset outgrows int32 past about 46k tokens) and no copy of the dense mask is kept. A mask that allows some query a non-contiguous set of keys goes to SDPA. The block masks are built once per mask tensor (every sliding layer of a forward, and its gradient-checkpoint recompute, gets the same one) and are freed with it.
- **One compiled shape family.** Each batch row runs alone, padded to a multiple of 128, always under a mask and with autocast off inside the kernel. Under autocast the inputs first take the autocast dtype, as SDPA's do, so fp32 parameters under bf16 autocast, whose fp32 rotary embedding promotes the queries and keys beside bf16 values, still run the warmed bf16 graph and the tuned tiles. A call either records a graph with every input requiring grad (an input left frozen, as by a LoRA that adapts `q_proj` alone, joins as a detached leaf) or runs under `no_grad`. The kernel therefore compiles twice, a training graph and a forward-only one, for any mix of lengths, batch sizes, grad modes and trained projections. `warmup_flex_sliding_kernels` compiles both for each call the model's sliding attention modules make (head count, head width, window and `scaling`, which Dynamo guards on), so no rank compiles mid-run. Deterministic-algorithms mode is a guard too, and HF's `Trainer` turns it on after the load under `full_determinism`, so each rank would compile again at its own first long sliding call while its peers wait in the next collective. Where the warm-up ran (world size > 1), every trainer therefore refuses `full_determinism` at construction (`reject_full_determinism_after_warmup`); with `HALO_FLEX_SLIDING=0` the model runs plain SDPA and takes it. The tuned tiles (head_dim 256, 16-bit inputs, SM100+) do not depend on the length; other shapes use FlexAttention's defaults.
- **Short sliding calls** (a query no longer than the window, or than one 128-token tile) go to SDPA: the window never binds there, so no tile would be skipped.
- **Causal global layers** run matmul attention where SDPA has only its mem-efficient kernel (head_dim above 256, or the flash backend disabled process-wide, as the Gemma 4 loader does) and the scores saved for backward fit `EAGER_GLOBAL_BUDGET_BYTES` (2 GiB). The logits stay fp32, as in mem-efficient SDPA: the score GEMM takes the 16-bit inputs and writes fp32, at 16-bit GEMM speed. At Gemma 4's global shape (bf16, `scaling` 1.0, RMS-normed 512-wide heads, unit norm weights, 1,024 tokens) the output's error against fp64 is 1.25x mem-efficient SDPA's, and the query and key gradients' error is half of SDPA's; bf16 logits would put the output's at 18x. Under autocast the GEMM takes the autocast dtype, as SDPA does. With no mask the layer attends as SDPA does: causally from the first key, or every key for a single query or an explicit `is_causal=False`. The budget is per layer: without gradient checkpointing every global layer holds its share. At 4,096 tokens a Gemma 4 26B-A4B global layer saves 1.5 GiB against 0.13 GiB on mem-efficient SDPA, about 7 GiB more across its 5 global layers.
- **Everything else** goes to the implementation registered as `sdpa`: bidirectional modules, dropout, float masks, `output_attentions`, and calls carrying sinks (`s_aux`), a softcap or a position bias.

`tests/gpu/kernels/test_flex_sliding_attention.py` checks Gemma 4 and Mistral (a window on every layer) against the SDPA model, loss and every gradient, on packed rows longer than the window; rows of ~47.5k tokens against SDPA; that one warm-up leaves no compile for later lengths, batch sizes, masks, grad modes, trained projections and autocast, and none for a Gemma 4 forward and backward, fp32 parameters under bf16 autocast included; that the block masks are freed with their mask; that the global layers' error at Gemma 4's shape stays within 1.5x mem-efficient SDPA's, their saved bytes match the budget's count, their unmasked calls match SDPA, and the score GEMM follows autocast; and that `full_determinism` is refused after the warm-up.
