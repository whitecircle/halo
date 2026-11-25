# Inkling 2

Thinking Machines' `Inkling2ForConditionalGeneration` (`thinkingmachines/Inkling-2-Small`, NVFP4 experts; `-MXFP8`, MXFP8 experts) — a multimodal MoE whose text tower routes every one of its 48 layers to 384 experts with top-8 selection and no shared expert, at hidden 4608.

The hub ships no bf16 release. Its `config.json` declares `quant_method: modelopt`, which transformers 5.23 cannot load, so the release trains only through [frozen quantized experts](#frozen-quantized-experts): LoRA over the shipped NVFP4/MXFP8 experts under EP ≥ 2.

`transformers.models.inkling2` exists from transformers 5.23.0 (the version the image pins). The composite config's `model_type` is `inkling2`, the text config's `inkling2_text`.

Per-axis support: [Compatibility matrix](README.md#compatibility-matrix). EP runs, and ETP runs on bf16 experts; CP, TP, PP and ETP over frozen experts do not ([why](#what-is-refused)).

## Architecture

- **Experts.** Fused `gate_up_proj`/`down_proj` (width 2304). The activation is a softcapped SwiGLU (`Inkling2SoftcappedSwiGLU`): the gate's linear factor is capped at `mlp_gate_softcap` (4), the up half at `mlp_up_softcap` (25), and the product runs in fp32.
- **Router.** `Inkling2TopKRouter` selects top-8 on `sigmoid(logits) + e_score_correction_bias`, gathers the unbiased scores, renormalizes them and scales by `route_scale` (8). The bias steers selection only. Transformers pins the router weight and the bias to fp32.
- **Attention.** Five sliding-window layers (window 1024) to one full layer. Position enters as a learned relative-logits bias (no RoPE), and every attention layer adds a query-conditioned sink logit. No stock kernel takes either term, so on CUDA every load runs the toolkit's `inkling2_relative_flex` (the family spec's `toolkit_attention`; `halo/models/attention/inkling2_relative_flex.py`): compiled FlexAttention whose `score_mod` reads the relative logits at their `[batch, heads, q_len, rel_extent]` size, with the sinks applied from the softmax normalizer. Each batch row runs alone, padded past one 128-token tile, under a block mask compiled from its document starts, window and padding, so no `[q_len, kv_len]` mask is built. Every call records a graph with all inputs requiring grad (a frozen input joins as a detached leaf) or runs under `no_grad`, so the kernel compiles a training and a forward-only graph per attention signature, both in `warm_attention_kernels` at load. Another requested `attn_implementation` is overridden with a warning; `attn_implementation: eager` keeps the eager path, which CPU runs take. Under a KV cache (`model.generate`) the prefill runs the kernel and each decode step the eager attention.
- **Short convolutions.** Depthwise causal convolutions on the projected K/V (dynamic per-token taps on the full layers) and after each attention and MLP sublayer.
- **Attention residuals.** Decoder layers sum into blocks of `n_layers_per_attn_res_block` (6). Each sublayer's input, and the final norm's, is an attention with a learned query over the finished blocks plus the partial one. transformers hands that attention to SDPA as one fp32 head of width 4608 per position, which selects the memory-efficient kernel, whose backward holds a gradient tile of a whole query block per position. The family's patch (`halo/models/families/inkling2.py`) computes it as two matmuls and a softmax over the blocks. One call over eight blocks at 8,192 tokens, forward and backward on one B300: 7.7 GiB peak and 11 ms, against 23.6 GiB and 48 ms through SDPA, with outputs within 1.4e-5 of it (`tests/gpu/models/test_inkling2_block_residual.py`). Every decoder layer's backward runs two of these calls, one at a time.
- **Head.** The logits are softcapped at `final_logit_softcapping` (30) and cut from the padded 201024 columns to `unpadded_vocab_size` (200058).

## EP layer

`EPInkling2MoELayer` (`halo/distributed/expert_parallel/families/inkling2.py`) claims `Inkling2SparseMoeBlock` / `model_type: inkling2`, `inkling2_text`. It is the group-limited template at one group with no shared expert. The router logits run in fp32 even when a loader cast the router weight to the run dtype. The expert combine is the module's own `Inkling2SoftcappedSwiGLU`, so the two softcaps stay in the EP path.

bf16 experts take the base fused-GLU storage, gather and sharded merge. That includes ETP (`tests/gpu/parallelism/ep/test_ep_family_vs_reference.py --family inkling2_text --expert-tp-size 2`, registered as EP+ETP on 4 GPUs).

## Frozen quantized experts

`frozen_quantized_experts: true` keeps each rank's slice of the release's packed experts in their shipped precision as a frozen base, with or without expert LoRA over it. Each expert GEMM dequantizes its bank to bf16, forward and backward; no bf16 copy persists between GEMMs.

Requirements, enforced at config time: `expert_parallel_size` ≥ 2, `expert_tensor_parallel_size: 1`, `ep_lazy_loading: true`, `merge_expert_lora_on_save: false`, `use_peft: true`, and `lowp_precision: bf16` or `lowp_apply_moe_experts: false` (a bank's GEMM reads the packed weights, so the expert fake-quant would never run). MFU and S-MFU count the banks as frozen expert weights. Every family other than Inkling 2 is refused off `config.json`, before the process groups (`reject_unsupported_model`); the loader refuses bf16 expert tensors.

A frozen layer refuses every expert gather — merged save, weight sync, gathered export — so a run saves its adapters. These adapters resume a Halo run and have no merged export: `merge_peft_adapters.py` refuses an adapter that rewrites a tensor the base stores packed (the release's `w13_weight` / `w2_weight`), as a delta folded into packed values gives no weight. The bank buffers are non-persistent: checkpoints never carry them, and every load reads them from the release.

The release layout, per layer and projection (`w13_weight` is the interleaved gate/up, `w2_weight` the down projection):

| Layers | Format | Stored tensors |
|---|---|---|
| 0–39 (`-Small`) | NVFP4 | `uint8` nibbles `[E, N, K/2]`, one scale per 16 elements along `K`, per-expert `_amax`, scalar `_outer_scale_denominator` |
| 40–47 (`-Small`) | MXFP8 | `float8_e4m3fn` `[E, N, K]`, one scale per 32 elements along `K` |

The family spec declares the layout once (`INKLING2.frozen_expert_slots`, `halo/models/moe/family_specs.py`); the loader and the EP layer both read it. `apply_frozen_bank_slots` (`halo/model_loading/parallel/frozen_experts.py`) swaps each meta expert container's bf16 `gate_up_proj` / `down_proj` for buffers sized from the safetensors headers before the lazy plan runs. The packed weight takes the weight's own name, its scale transformers' `<weight>_scale_inv` target (de-interleaved by row with the weight), and NVFP4's sidecars `w13_weight_amax` / `w13_weight_outer_scale_denominator` (and `w2_…`). Each rank reads only its expert rows. The EP layer builds one `FrozenExpertBank` per projection; an NVFP4 bank's per-expert global scale is `amax / outer_scale_denominator`.

The banks take no gradient. Expert adapters take each bank's logical `[E, K, N]` shape in bf16. Every non-expert tensor loads as `from_pretrained` loads it. `tests/gpu/parallelism/ep/test_lazy_load_inkling2_frozen.py` pins a tiny release-layout checkpoint against its bf16 dequantization: the banks, the loss, and an adapter step that leaves the banks unchanged.

## What is refused

- **CP.** The short convolutions run over the sequence axis, so a CP shard severs their receptive field, and the relative-logits bias and sink cannot pass through `flash_attn_func`. Validation rejects the convolution classes by name.
- **ETP over frozen experts.** `frozen_quantized_experts` refuses `expert_tensor_parallel_size` > 1 at config time: a bank is sliced on the expert axis only, so every expert-TP rank would hold the full FFN width. ETP runs on bf16 experts.
- **TP.** The selective-TP planner cannot shard the per-layer head geometry (`swa_*`), the K/V convolutions or the relative-logits projection.
- **PP.** Refused by name: the attention residuals' block states are a stack that one activation at a stage boundary does not carry ([Pipeline Parallelism](../parallelism/pipeline-parallelism.md#model-support)).
- **Packing.** On a packed row (`position_ids` restarting at 0) the family's patches stop every short-convolution tap at its document's start, and `inkling2_relative_flex` reads the documents off `position_ids`, so the on-policy GRPO trainers' `packed_microbatch_tokens` and SFT's `packing` train it ([Document isolation](../data/collators.md#document-isolation-under-packing)). The convolutions isolate one `[1, N]` row: a `[B > 1, N]` `position_ids` that restarts inside a row raises. `padding_free` is refused: it needs a varlen flash kernel.
- **MTP.** The release ships a five-layer draft head under its own `mtp_config`, which transformers' class does not build; `mtp_loss_weight` refuses the family ([MTP heads](../optimization/mtp-heads.md)).
- **Full fine-tuning of the release.** No save exports the frozen banks, so `frozen_quantized_experts` requires `use_peft: true`.
- **Online GRPO, and environmental GRPO with a weight push.** The spec's `weight_sync_refusal`: no full-weight push into either pinned engine has been validated. Environmental GRPO runs on vLLM with [adapter sync](#rollouts), which pushes no weights; SGLang is refused. Offline GRPO needs no server.

## LoRA

Attention adapters (`q_proj`, `k_proj`, `v_proj`, `r_proj`, `o_proj`) train under PEFT; expert adapters train as native grouped adapters ([PEFT](../optimization/peft.md#moe-models--expert-targets-and-full-trained-modules)). On the release both are frozen-bank LoRA. On tiny models `tests/gpu/trainers/lora/test_lora_merged_save_resume.py --family inkling2_text` runs bf16 expert and mixed adapters at ep1 and ep2 through a merged save and an exact resume, and checks that ep2 × cp2 is refused.

Liger covers the weighted RMSNorm (Gemma casting mode) and the cross-entropy. The MLP and head stay eager: the softcapped SwiGLU has no fused kernel, and the head's softcap and vocab cut are the family's declared head transform (`Inkling2HeadTransform`, `halo/models/head_transform.py`), which offline GRPO's chunked log-prob sweep also applies.

## Rollouts

Environmental GRPO trains LoRA on the release and serves it to vLLM as an adapter (`rollout_adapter_sync`). The server loads the release, NVFP4/MXFP8 experts included, and takes each adapter version by path; the trainer never pushes the base and forms no NCCL group with it ([Adapter sync](../infrastructure/rollout-servers.md#adapter-sync)). Two GPUs serve and two train:

```bash
make build-vllm-inkling2  # halo:vllm-0.1.dev25249, the vLLM tag that serves Inkling 2
make up-vllm-inkling2      # download the release once, serve it on GPUs 2,3
make train-inkling2        # the trainer on GPUs 0,1, through halo launch
```

`INKLING2_LORA=expert` on `up-vllm-inkling2` and `train-inkling2` trains and serves LoRA on the routed experts too. The release and the adapters live under `HALO_SCRATCH` (`INKLING2_DIR`, `INKLING2_ADAPTER_DIR`); `INKLING2_SERVER_DEVICES` and `INKLING2_TRAINER_DEVICES` pick the GPUs, and their lengths set the server's tensor-parallel size and the trainer's process count; `VLLM_PORT` (default 8000) moves the server's port and the URL the trainer dials together. The runs are the examples under `examples/grpo/environmental/inkling2/vllm/`:

| Example | LoRA | Server LoRA modules |
|---|---|---|
| `inkling2-small-exam-qa-attention-lora-ep2.yaml` | r16 on the text tower's q/k/v/r/o | `qkvr`, `wo_ud` |
| `inkling2-small-exam-qa-expert-lora-ep2.yaml` | r8 on `o_proj` and the routed experts | all |

The trainer and the server read the release directory and the adapter directory at the same paths: the server loads each version from the path the trainer published it at, and the trainer matches the server's base by its path. A launch outside `make` passes both:

```bash
CUDA_VISIBLE_DEVICES=0,1 halo launch grpo/environmental examples/grpo/environmental/inkling2/vllm/inkling2-small-exam-qa-attention-lora-ep2.yaml \
  --nproc 2 --model_name_or_path=<release dir> --rollout_adapter_dir=<adapter dir>
```

The server is `halo:vllm-0.1.dev25249`, the tag `Dockerfile.vllm` builds on vLLM's Inkling 2 build ([Rollout Servers](../infrastructure/rollout-servers.md#vllm)), run by `docker-compose.vllm.yml`; `make up-vllm-inkling2` sets `VLLM_IMAGE` to it with the Inkling 2 flags. `halo:vllm-0.32.0`, the tag every other family is served from, has no Inkling 2 model.

The configs set four things for this model:

- `pad_token: "<|endoftext|>"`: the release's tokenizer names no pad token, and environmental GRPO pads its batch rows.
- The training shape ([Memory on two B300s](#memory-on-two-b300s)): `per_device_train_batch_size: 1` and `gradient_accumulation_steps: 4`, four trajectories per rank and step, one group of `num_generations: 4`; `packed_microbatch_tokens: 8192`, which packs the step's turn rows into at most four micro-batches; `max_train_row_tokens: 4096`, half that budget, so any two rows share a micro-batch and a rank's eight rows fit the four within it (a row over the cap leaves the batch; uncapped, a round needing more than four is rebalanced into four past the budget); and `use_chunked_grpo_logprobs: true`, which the packing requires.
- Attention LoRA targets the text tower alone (`'.*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|r_proj|o_proj)'`), since the vision tower's attention also names `q_proj` / `k_proj` / `v_proj`. vLLM packs `wq_du` / `wk_dv` / `wv_dv` / `wr_du` into its `qkvr` layer and serves `wo_ud` apart.
- Expert LoRA lists its targets, `[o_proj, gate_up_proj, down_proj]`: a regex string drops the expert targets, and these names stay off the vision tower. Expert and attention adapters share one rank and scaling.

### Memory on two B300s

At EP2 a trainer rank holds its 192 experts of every layer in the release's format beside the replicated non-expert weights: 181.8 GiB allocated before the first step, of the 267.7 a B300 offers. The rest is the micro-batch's. A step's training rows are its turn rows, one per assistant turn (`train_on_sampled_tokens`), so four trajectories of `max_turns: 2` give up to eight, each a prompt and one turn of at most `rollout_max_tokens` (2,048). Measured on the release (memory snapshot at an out-of-memory error, and the allocator's peaks per micro-batch), a micro-batch adds:

- About 1.8 GiB per 1,000 tokens it forwards, held from its forward to its backward under reentrant checkpointing: each decoder layer's input (0.41) and the attention residuals' block states (0.31), and the DeepEP recompute cache, each MoE layer's received tokens and combined output, which the checkpoint recompute replays instead of dispatching again (1.06).
- Inside each layer's backward, the bank dequantized for its GEMM (7.6 GiB for gate/up) and the two attention residuals: 0.9 GiB per 1,000 tokens each with the family's patch, about 3 through SDPA ([Attention residuals](#architecture)).
- 8.9 GiB of FSDP2 all-gather outputs, the non-expert weights kept gathered from the forward to the backward.
- Under the full-logits scorer, 8 bytes per logit held to the backward and up to 16 at the forward's peak: the head upcasts to fp32 before its softcap, whose `tanh` output and result stay. Four rows of 2,049 completion positions hold 12.3 GiB.

A padded micro-batch pays every row the widest prompt plus the widest completion it holds, and its rows' masked tokens too. With all of a step's rows in one padded micro-batch (`per_device_train_batch_size: 4`, `gradient_accumulation_steps: 1`), the full-logits scorer and the SDPA residual, both examples run out of memory within four steps. Packed, a micro-batch forwards only its rows' attended tokens, at most the budget under the row cap, and a row whose completion is masked (a group that settled one reward) forwards its prompt alone. The peaks of micro-batches near the budget are under [Measured](#measured).

### Expert LoRA on the quantized experts

vLLM's quantized experts hold their rows tile-padded and grouped by expert, and the image's bf16 grouped GEMM adds the gate/up delta to the w13 output before the quantizing SwiGLU reads it. The down branch recomputes that SwiGLU in bf16 and adds its delta to the w2 output before the combine. Three Triton kernels carry it: one maps the layout rows to their tokens and experts, one recomputes the activation of the routed rows with the quantizing kernel's arithmetic, and the grouped GEMM masks each row to its adapter slot, so every slot's adapters, stored side by side, take one shrink and one expand per projection. The trainer sends each layer's experts as `mlp.experts.w13_weight` (`lora_A` `[E, r, H]`, `lora_B` `[E, 2I, r]`, rows alternating gate and up as in the checkpoint) and `mlp.experts.w2_weight` (`[E, r, I]`, `[E, H, r]`).

On the MXFP8 experts at Inkling-2-Small's shapes, with two adapter slots (r8, r16) and tokens on each slot and on none, the image's GPU check (`docker run --gpus 1 --entrypoint python3 halo:vllm-0.1.dev25249 /opt/halo/verify_inkling2_expert_lora.py`) finds the output with LoRA at most 1.6% off the reference, the same as the base output alone (1.4%), with deltas of 8% to 37% of the output. Without the down delta the output is 8% to 9% off, which fails the check.

One expert-layer forward on one B300, two r16 slots:

| Tokens | Base | With LoRA |
|---|---|---|
| 8 | 0.42 ms | 0.64 ms (+50%) |
| 256 | 2.02 ms | 2.47 ms (+22%) |
| 2048 | 2.27 ms | 3.26 ms (+44%) |
| 8192 | 3.03 ms | 5.04 ms (+67%) |

The four LoRA grouped GEMMs take most of the added time (0.19 ms at 8 tokens, 1.67 ms at 8192), then the activation recompute (0.28 ms at 8192).

## Measured

The DAPO-Math run below ran without gradient checkpointing on the decoder layers, so its forward and backward time is not a checkpointed run's. The examples' runs ran with it, TRL's default for GRPO.

Each frozen bank dequantizes in one [fused Triton kernel](../optimization/low-precision-moe-kernels.md#frozen-expert-banks), bit-identical to the eager block-scale path: a 16-expert `[4608, 4608]` NVFP4 bank takes 0.69 ms against 11.2 ms eager on one B300.

One full-attention layer at Inkling-2-Small's shape (48 query and 8 KV heads of 128, relative extent 1024), 8,192 tokens in bf16, forward and backward on one B300: `inkling2_relative_flex` takes 41 ms at a 2.6 GiB peak, eager 166 ms at 61 GiB (`tests/gpu/models/test_inkling2_relative_flex.py`). A padded batch makes one kernel call per row: 4 rows of 2,048 tokens take 18 ms and 8 rows of 1,024 tokens 18 ms at a 2.0 GiB peak.

Environmental GRPO on DAPO-Math-17k, expert LoRA r8, 8× B300 (two vLLM TP2 engines, EP4 trainer), 32 prompts × 8 samples of up to 4,096 tokens, `packed_microbatch_tokens: 8192`: a step takes 322 s on-policy (rollout 125 s, adapter sync 15 s, forward and backward about 200 s) and 194 s with `enable_prefetch: true`, whose rollout overlaps the previous step's training, so its samples come from the policy one step back. Both are measured on steps 2–4.

Inkling-2-Small, LoRA r16 on attention q/k/v/o and the experts, 4× B300 at EP4 with `frozen_quantized_experts`: the load takes ~17 s per rank from local disk. Over 150 steps GSM8K held-out loss goes 0.549 → 0.136, ARC-Challenge `acc_norm` 0.777 → 0.779 and HellaSwag (1k) `acc_norm` 0.752 → 0.756.

The attention example on WebInstruct-verified, EP2 on two B300s beside a two-B300 server, 16 steps of 8 rollouts with `drop_degenerate_groups: false`, so every turn row trains: its largest packed micro-batch, 8,172 tokens, peaked at 226.9 GiB allocated (239.4 GiB in `nvidia-smi`) of 267.7. Over steps 2–16 a step took 12.2–52.5 s (the sum of its [`phase_time`](../training-methods/async-grpo/monitoring.md#phase-times) keys): the rollout 4.3–35.9 s, training (the loss forwards and backwards, `loss_forward_s` plus `micro_step_s`) 3.5–14.6 s and the adapter sync 2.4–2.8 s. The expert example, run the same way: its largest micro-batch, 7,703 tokens, peaked at 232.4 GiB allocated (245.5 GiB in `nvidia-smi`), and a step took 24.4–88.1 s, the rollout 8.0–60.4 s, training 3.7–15.3 s and the adapter sync, which publishes the 4.8 GB adapter and loads it on the server, 8.3–19.3 s. Padded micro-batches instead, under the full-logits scorer and the SDPA residual, on the attention example over steps 2–8: at `per_device_train_batch_size` × `gradient_accumulation_steps` of 2 × 2 training took 7.3–24.9 s per step and `nvidia-smi` peaked at 262.5 GiB; at 1 × 4, 11.7–34.8 s and 228.3 GiB.

CPU coverage: `tests/cpu/models/test_inkling2_frozen_experts.py`, `tests/cpu/kernels/test_frozen_bank.py`, `tests/cpu/models/test_inkling2_block_residual_patch.py`. GPU gates: `test_ep_family_vs_reference.py --family inkling2_text` (with and without `--expert-tp-size 2`), `test_lazy_load_inkling2_frozen.py`, `test_lora_merged_save_resume.py --family inkling2_text`, `test_lora_frozen_banks_resume.py`, `test_frozen_bank_fused_dequant.py`, `test_inkling2_relative_flex.py`, `test_inkling2_block_residual.py`.
