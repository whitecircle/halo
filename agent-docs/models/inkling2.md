# Inkling 2

Thinking Machines' `Inkling2ForConditionalGeneration` (`thinkingmachines/Inkling-2-Small`, NVFP4 experts; `-MXFP8`, MXFP8 experts) — a multimodal MoE whose text tower routes every one of its 48 layers to 384 experts with top-8 selection and no shared expert, at hidden 4608.

The hub ships no bf16 release. Its `config.json` declares `quant_method: modelopt`, which transformers 5.23 cannot load, so the release trains only through [frozen quantized experts](#frozen-quantized-experts): LoRA over the shipped NVFP4/MXFP8 experts under EP ≥ 2.

`transformers.models.inkling2` exists from transformers 5.23.0 (the version the image pins). The composite config's `model_type` is `inkling2`, the text config's `inkling2_text`.

Per-axis support: [Compatibility matrix](README.md#compatibility-matrix). EP and ETP run; CP, TP and PP do not ([why](#what-is-refused)).

## Architecture

- **Experts.** Fused `gate_up_proj`/`down_proj` (width 2304). The activation is a softcapped SwiGLU (`Inkling2SoftcappedSwiGLU`): the gate's linear factor is capped at `mlp_gate_softcap` (4), the up half at `mlp_up_softcap` (25), and the product runs in fp32.
- **Router.** `Inkling2TopKRouter` selects top-8 on `sigmoid(logits) + e_score_correction_bias`, gathers the unbiased scores, renormalizes them and scales by `route_scale` (8). The bias steers selection only. Transformers pins the router weight and the bias to fp32.
- **Attention.** Five sliding-window layers (window 1024) to one full layer. Position enters as a learned relative-logits bias (no RoPE), and every attention layer adds a query-conditioned sink logit. The class declares no SDPA, flash or flex support, so every load runs eager attention: any other `attn_implementation` is overridden to `eager` with a warning (the family spec's `eager_only_attention`).
- **Short convolutions.** Depthwise causal convolutions on the projected K/V (dynamic per-token taps on the full layers) and after each attention and MLP sublayer.
- **Attention residuals.** Decoder layers sum into blocks of `n_layers_per_attn_res_block` (6). Each sublayer's input, and the final norm's, is an attention with a learned query over the finished blocks plus the partial one.
- **Head.** The logits are softcapped at `final_logit_softcapping` (30) and cut from the padded 201024 columns to `unpadded_vocab_size` (200058).

## EP layer

`EPInkling2MoELayer` (`halo/distributed/expert_parallel/families/inkling2.py`) claims `Inkling2SparseMoeBlock` / `model_type: inkling2`, `inkling2_text`. It is the group-limited template at one group with no shared expert. The router logits run in fp32 even when a loader cast the router weight to the run dtype. The expert combine is the module's own `Inkling2SoftcappedSwiGLU`, so the two softcaps stay in the EP path.

bf16 experts take the base fused-GLU storage, gather and sharded merge. That includes ETP (`tests/gpu/parallelism/ep/test_ep_family_vs_reference.py --family inkling2_text --expert-tp-size 2`, registered as EP+ETP on 4 GPUs).

## Frozen quantized experts

`frozen_quantized_experts: true` keeps each rank's slice of the release's packed experts in their shipped precision as a frozen base, with or without expert LoRA over it. Each expert GEMM dequantizes its bank to bf16, forward and backward; no bf16 copy persists between GEMMs.

Requirements, enforced at config time: `expert_parallel_size` ≥ 2, `ep_lazy_loading: true` and `merge_expert_lora_on_save: false`. The loader refuses bf16 expert tensors and every family other than Inkling 2.

A frozen layer refuses every expert gather — merged save, weight sync, gathered export — so a run saves its adapters. The bank buffers are non-persistent: checkpoints never carry them, and every load reads them from the release.

The release layout, per layer and projection (`w13_weight` is the interleaved gate/up, `w2_weight` the down projection):

| Layers | Format | Stored tensors |
|---|---|---|
| 0–39 (`-Small`) | NVFP4 | `uint8` nibbles `[E, N, K/2]`, one scale per 16 elements along `K`, per-expert `_amax`, scalar `_outer_scale_denominator` |
| 40–47 (`-Small`) | MXFP8 | `float8_e4m3fn` `[E, N, K]`, one scale per 32 elements along `K` |

`install_frozen_bank_slots` (`halo/model_loading/parallel/frozen_experts.py`) swaps each meta expert container's bf16 `gate_up_proj` / `down_proj` for buffers sized from the safetensors headers before the lazy plan runs. The packed weight takes the weight's own name, its scale transformers' `<weight>_scale_inv` target (de-interleaved by row with the weight), and NVFP4's sidecars `w13_weight_amax` / `w13_weight_outer_scale_denominator` (and `w2_…`). Each rank reads only its expert rows. The EP layer builds one `FrozenExpertBank` per projection; an NVFP4 bank's per-expert global scale is `amax / outer_scale_denominator`.

The banks take no gradient. Expert adapters take each bank's logical `[E, K, N]` shape in bf16. Every non-expert tensor loads as `from_pretrained` loads it. `tests/gpu/parallelism/ep/test_lazy_load_inkling2_frozen.py` pins a tiny release-layout checkpoint against its bf16 dequantization: the banks, the loss, and an adapter step that leaves the banks unchanged.

## What is refused

- **CP.** The short convolutions run over the sequence axis, so a CP shard severs their receptive field, and the relative-logits bias and sink cannot pass through `flash_attn_func`. Validation rejects the convolution classes by name.
- **TP.** The selective-TP planner cannot shard the per-layer head geometry (`swa_*`), the K/V convolutions or the relative-logits projection.
- **PP.** No pipeline spec covers the family.
- **Packing.** The convolutions read no document boundary, so `packing` is refused unless `allow_cross_document_linear_attention: true` accepts the crossing ([Document isolation](../data/collators.md#document-isolation-under-packing)). Eager attention has no varlen kernel, so `padding_free` is refused.
- **MTP.** The release ships a five-layer draft head under its own `mtp_config`, which transformers' class does not build; `mtp_loss_weight` refuses the family ([MTP heads](../optimization/mtp-heads.md)).
- **Weight sync.** The spec's `weight_sync_refusal`: no full-weight push into either pinned engine has been validated, so online GRPO is refused and environmental GRPO runs only with [adapter sync](#rollouts). Offline GRPO needs no server.

## LoRA

Attention adapters (`q_proj`, `k_proj`, `v_proj`, `o_proj`) train under PEFT; expert adapters train as native grouped adapters ([PEFT](../optimization/peft.md#moe-models--expert-targets-and-full-trained-modules)). On the release both are frozen-bank LoRA. On tiny models `tests/gpu/trainers/lora/test_lora_merged_save_resume.py --family inkling2_text` runs bf16 expert and mixed adapters at ep1 and ep2 through a merged save and an exact resume, and checks that ep2 × cp2 is refused.

Liger covers the weighted RMSNorm (Gemma casting mode) and the cross-entropy. The MLP and head stay eager: the softcapped SwiGLU has no fused kernel, and the head's softcap and vocab cut are the family's declared head transform (`Inkling2HeadTransform`, `halo/models/head_transform.py`), which offline GRPO's chunked log-prob sweep also applies.

## Rollouts

Environmental GRPO trains LoRA on the release and serves it to vLLM as an adapter (`rollout_adapter_sync`). The server loads the release, NVFP4/MXFP8 experts included, and takes each adapter version by path; the trainer never pushes the base and forms no NCCL group with it ([Adapter sync](../infrastructure/rollout-servers.md#adapter-sync)). Two GPUs serve and two train:

```bash
make build-vllm           # the vLLM image
make up-vllm-inkling2      # download the release once, serve it on GPUs 2,3
make train-inkling2        # the trainer on GPUs 0,1, through halo launch
```

`INKLING2_LORA=expert` on `up-vllm-inkling2` and `train-inkling2` trains and serves LoRA on the routed experts too. The release and the adapters live under `HALO_SCRATCH` (`INKLING2_DIR`, `INKLING2_ADAPTER_DIR`); `INKLING2_SERVER_DEVICES` and `INKLING2_TRAINER_DEVICES` pick the GPUs. The runs are the examples under `examples/grpo/environmental/inkling2/vllm/`:

| Example | LoRA | Server LoRA modules |
|---|---|---|
| `inkling2-small-exam-qa-attention-lora-ep2.yaml` | r16 on the text tower's q/k/v/r/o | `qkvr`, `wo_ud` |
| `inkling2-small-exam-qa-expert-lora-ep2.yaml` | r8 on `o_proj` and the routed experts | all |

The trainer and the server read the release directory and the adapter directory at the same paths: the server loads each version from the path the trainer published it at, and the trainer matches the server's base by its path. A launch outside `make` passes both:

```bash
CUDA_VISIBLE_DEVICES=0,1 halo launch grpo/environmental examples/grpo/environmental/inkling2/vllm/inkling2-small-exam-qa-attention-lora-ep2.yaml \
  --nproc 2 --model_name_or_path=<release dir> --rollout_adapter_dir=<adapter dir>
```

The server is Halo's vLLM image (`Dockerfile.vllm`), served by `docker-compose.vllm.yml`; `make up-vllm-inkling2` sets its Inkling 2 flags.

The configs set three things for this model:

- `pad_token: "<|endoftext|>"`: the release's tokenizer names no pad token, and environmental GRPO pads its batch rows.
- Attention LoRA targets the text tower alone (`'.*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|r_proj|o_proj)'`), since the vision tower's attention also names `q_proj` / `k_proj` / `v_proj`. vLLM packs `wq_du` / `wk_dv` / `wv_dv` / `wr_du` into its `qkvr` layer and serves `wo_ud` apart.
- Expert LoRA lists its targets, `[o_proj, gate_up_proj, down_proj]`: a regex string drops the expert targets, and these names stay off the vision tower. Expert and attention adapters share one rank and scaling.

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

Inkling-2-Small, LoRA r16 on attention q/k/v/o and the experts, 4× B300 at EP4 with `frozen_quantized_experts`: the load takes ~17 s per rank from local disk; a steady step takes 14.5 s at ~760 tokens/GPU/step, with 86–90% of GPU time in the bf16 decode of the frozen banks, the bottleneck. Over 150 steps GSM8K held-out loss goes 0.549 → 0.136, ARC-Challenge `acc_norm` 0.777 → 0.779 and HellaSwag (1k) `acc_norm` 0.752 → 0.756.

CPU coverage: `tests/cpu/models/test_inkling2_frozen_experts.py`, `tests/cpu/kernels/test_frozen_bank.py`. GPU gates: `test_ep_family_vs_reference.py --family inkling2_text` (with and without `--expert-tp-size 2`), `test_lazy_load_inkling2_frozen.py`, `test_lora_merged_save_resume.py --family inkling2_text`.
