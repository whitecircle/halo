# Supported Matrix

What runs, what doesn't, and what has been validated. The code is the source
of truth; this is the summary.

## Runtime

| Area | Supported | Notes |
| --- | --- | --- |
| Python | `>=3.12,<3.13` | |
| Package manager | uv | `pyproject.toml` + `uv.lock` |
| Primary runtime | Docker image | required for training and tests |
| PyTorch | 2.11.x, `cu130` wheel | image ships the CUDA 13.2 toolkit |
| Transformers | 5.16.x | |
| TRL | 1.6.x | base trainer stack |
| Accelerate | 1.11.x | launcher and FSDP integration |
| PEFT | 0.18.x | LoRA / QLoRA |
| vLLM | 0.26.0, separate container | generation for online and environmental GRPO |
| SGLang | 0.5.17, separate container | alternative rollout engine for environmental GRPO |

## Hardware

| Hardware | Image | Status | Notes |
| --- | --- | --- | --- |
| NVIDIA B200 / B300 | `halo:blackwell` | supported | primary Blackwell target |
| NVIDIA GB200 / GB300 NVL72 | `halo:blackwell` | supported | NVLink-domain support through config/env |
| NVIDIA H100 / H200 | `halo:hopper` | supported | Hopper image target |
| NVIDIA A100 / Ampere | source/local build | partial | FA2 fallback; not a release target |

Pull the prebuilt images from Amazon ECR Public, or build them from source —
both routes are in [Installation](installation.md).

## Attention backends

| Backend | Hardware | Status | Notes |
| --- | --- | --- | --- |
| Flash Attention 4 | Blackwell | supported | auto-selected on Blackwell |
| Flash Attention 3 | Hopper | supported | auto-selected on Hopper |
| Flash Attention 2 | Ampere / Hopper / Blackwell | supported | fallback when FA3/FA4 is absent |
| SDPA | broad | fallback | forced where no FA kernel serves the family |
| eager attention | any | last resort | forced for DeepSeek-V4, the one family SDPA cannot serve either |

Several families are redirected off Flash Attention automatically. Gemma 4's
head_dim of 512 exceeds every FA kernel. Qwen3.5/3.6 and GLM-4 MoE Lite are
demoted from FA4 alone, whose backward produces NaNs at their shapes, so they
keep FA3 on Hopper. GLM-5 Next, Step-3.7 Flash, Inkling and Bailing/Ling declare
no usable flash support upstream.

Context Parallelism picks its own kernel and ignores the configured label: FA3
on Hopper, FA4 on Blackwell, FA2 otherwise. It rejects SDPA, except where a
family's Ulysses wrapper waives the check because its modeling code carries no
flash label at all (Bailing/Ling). The per-family resolution table is in
[Flash Attention](../agent-docs/optimization/flash-attention.md) ↗.

## Optimization defaults

| Feature | Status | Default | Notes |
| --- | --- | --- | --- |
| AdamWBF16 optimizer | supported | auto-enabled when `bf16: true` on FSDP/EP/TP/CP (not accelerate-managed DDP) | stochastic rounding, no fp32 master weights (6 B/param) |
| fp32 master weights | supported | opt-in (`fp32_non_ep_params: true`) | 12 B/param on the non-expert params; experts stay bf16. `bf16_optimizer: false` picks the stock optimizer, not a precision, and is rejected on a MoE whose experts are plain tensors unless `fp32_non_ep_params` is set |
| Liger kernels | supported | on for every method (the parser sets `use_liger_kernel: true`) | fused CE / SwiGLU / RMSNorm paths by model support |
| Grouped GEMM | supported | `use_grouped_gemm: true` on SM90+ | batches the per-expert matmuls, no-op on older hardware |
| Flash Attention backend selection | supported | automatic | by GPU architecture and model constraints |
| FP8 / FP4 MoE training | experimental | off (`lowp_precision: bf16`) | fp8/fp4 can be slower at these MoE shapes |
| FP8 / FP4 export | experimental | explicit conversion | `halo run quantize-to-lowp` |

## Training methods

Every trainer supports EP, TP, ETP, and EP+TP. CP is declare-to-enable
(`_supports_cp`, `False` on the base) and only SFT and SMPO set it: nothing
inspects a trainer's loss, so CP would fail silently, mis-pooling quantities
across sequence shards. What keeps the rest off is `logits_to_keep`, global
log-probability sums, full-sequence pooling, and dual-model / rollout setups.

| Method | Script | EP | CP | TP | ETP | Notes |
| --- | --- | :---: | :---: | :---: | :---: | --- |
| SFT | `scripts/training/sft.py` | Yes | Yes | Yes | Yes | also VLM and continued pretraining |
| SMPO | `scripts/training/preference/smpo.py` | Yes | Yes | Yes | Yes | reference-free preference optimization |
| DPO | `scripts/training/preference/dpo.py` | Yes | No | Yes | Yes | reference-model log-prob sums block CP |
| KTO | `scripts/training/preference/kto.py` | Yes | No | Yes | Yes | Kahneman-Tversky optimization |
| Reward modeling | `scripts/training/preference/rewards.py` | Yes | No | Yes | Yes | full-sequence pooling blocks CP |
| Classification | `scripts/training/classification.py` | Yes | No | Yes | Yes | full-sequence pooling blocks CP |
| Offline GRPO | `scripts/training/offline_grpo.py` | Yes | No | Yes | Yes | trains from scored completions |
| Online GRPO (RLVR) | `scripts/training/online_grpo/rlvr.py` | Yes | No | Yes | Yes | needs a vLLM server; `--use_sdpg=true` runs online SDPG |
| Environmental GRPO | `scripts/training/environmental_grpo.py` | Yes | No | Yes | Yes | needs Ray plus a vLLM or SGLang rollout server |
| Distillation | `scripts/training/distillation/` | Yes | No | Yes | Yes | teacher and self distillation |
| Embedding | `scripts/training/embedding.py` | Yes | No | Yes | Yes | SentenceTransformer trainer |

### Rollout engines

Environmental GRPO serves rollouts from vLLM by default; `rollout_backend:
sglang` switches engines. SGLang is the narrower path and refuses three shapes
at startup rather than mid-run:

- any distributed experts (`expert_parallel_size × expert_tensor_parallel_size > 1`);
- `rollout_max_thinking_tokens`, a vLLM-only request field — steer reasoning
  with the environment's `reasoning_effort` and price it with
  `reasoning_compliance_weight` instead;
- every MoE family except GPT-OSS. SGLang loads experts in the checkpoint-fused
  layout that only the GPT-OSS layer gathers, and 0.5.17's Qwen3-MoE loader
  drops fused expert keys outright (dense Qwen3 is fine).

`routing_replay: rollout` works on either engine; SGLang captures it when the
server runs `--enable-return-routed-experts --moe-runner-backend triton`.
SGLang must be served from this repo's `Dockerfile.sglang` image — the upstream
one ships a different NCCL and cannot form the weight-sync group — and its
weight sync costs the trainer NVLink between its own ranks, so its step runs
slower than vLLM's ([Troubleshooting](troubleshooting.md)). Engine-by-engine
detail: [Rollout Servers](../agent-docs/infrastructure/rollout-servers.md) ↗.

## Parallelism modes

| Mode | Status | Best for | Main limit |
| --- | --- | --- | --- |
| FSDP2 / DP | supported | default dense and small MoE runs | memory scales with model shape |
| HSDP (`use_hsdp: true`) | supported | multi-node DP where 1D FSDP's cross-node shard collectives dominate | pure DP or CP only; no-op on one NVLink domain |
| EP | supported | MoE expert sharding | MoE families need registered wrappers |
| CP | supported | long-context SFT / SMPO | blocked by full-sequence logits / pooling |
| TP | supported | attention and weight sharding | model-family support varies |
| ETP | supported | expert FFN memory reduction | shards experts only, not attention |
| EP + CP | supported for selected model/trainer pairs | MoE plus long context | needs both model and trainer support |
| EP + TP | supported for selected MoE families | MoE plus attention sharding | node / NVLink-domain constraints |
| EP + ETP | experimental | MoE expert memory pressure | node-local, advanced path |
| TP + CP · TP + ETP · ETP + CP · EP + TP + ETP | unsupported | — | rejected at config validation; pick EP+TP or EP+ETP |
| Pipeline parallelism | not yet available | — | the config surface ships, but `pipeline_parallel_size > 1` is rejected at config time |

## Model families

Any HuggingFace `AutoModelForCausalLM` runs under standard FSDP2. The matrix
below is about advanced parallelism. Qwen3 MoE, GPT-OSS, GLM-4 MoE Lite and
Mistral4 carry the broadest coverage; hybrid linear-attention families and MoEs
without a registered CP wrapper drop CP. Source of truth under
`src/distributed/`: `MOE_LAYER_MAP` (EP), the CP wrapper registry in
`context_parallel/layers/` (CP), `TP_SHARDABLE_ATTENTION_CLASSES` (TP).

| Model family | FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA | Notes |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| Qwen3 (dense) | Yes | — | Yes | Yes | — | — | — | Yes | reference dense family |
| Qwen3 MoE | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | broadest coverage |
| Qwen3-VL (text) | Yes | — | Yes | No | — | — | — | Yes | keep `tensor_parallel_size=1` — both variants raise at load under TP |
| Qwen3.5 / Qwen3.6 MoE | Yes | Yes | No | Yes | Yes | No | Yes | Yes | interleaved linear-attention blocks CP; VL checkpoints train too — the MoE-VL wrapper has EP, the dense 9B-VL runs plain FSDP with `sdpa` |
| GPT-OSS | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | interleaved fused experts; the only MoE family SGLang weight sync serves |
| GLM-4 MoE Lite | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | LoRA-style attention compression |
| Command A+ (Cohere2 MoE) | Yes | Yes | Yes | Yes | Yes | untested | Yes | untested | VLM checkpoint; NoPE full-attention layers; averaged shared expert. Only EP is validated on the 200B+ checkpoint — CP/TP/ETP pass the tiny-model 8-GPU matrix. No online/environmental GRPO |
| Laguna S / XS 2.1 | Yes | Yes | No | No | untested | No | No | Yes | sigmoid router and shared expert, native in transformers (released checkpoints still load through remote code at a pinned revision); shipped configs set `attn_implementation: sdpa`, so `padding_free` is rejected and they pack instead |
| Gemma 4 MoE | Yes | Yes | No | No | Yes | No | No | Yes | KV-shared layers block CP/TP; no router-balancing path at all |
| Bailing/Ling | Yes | Yes | Yes | No | Yes | untested | No | Yes | EP covers Ling 2.0, Ling 3.0 and the Ring linear-attention siblings; CP on Ling 2.0 only (needs `sdpa`); no DTensor attention plan. Weight sync refused for Ling 3.0 and the linear siblings |
| LFM-2 MoE | Yes | Yes | No | Yes | Yes | No | Yes | Yes | short-conv layers block CP |
| Mistral4 MoE | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | vLLM 0.26.0 registers no `mistral4` class, so no online / environmental GRPO |
| DeepSeek-V4 | Yes | Yes | No | No | untested | No | No | Yes | shared-KV MQA + CSA/HCA compressor block CP/TP; eager-only, so `padding_free` is rejected (packing works but warns — compressors see across document boundaries); no rollout weight sync |
| Zaya | Yes | Yes | No | No | Yes | No | No | Yes | EP or ETP, always without gradient checkpointing; CCA blocks CP, `num_kv_heads=2` blocks TP; no rollout weight sync |
| Inkling | Yes | Yes | No | No | Yes | No | No | untested | multimodal MoE (276B total / 12B active); short-conv layers and an additive relative-logits bias block CP, no `tp_plan`; no rollout weight sync |
| GLM-5 Next (GLM-5.3-Flash) | Yes | Yes | No | No | Yes | No | No | Yes | composite VLM (321B total / 18B active); KDA linear-attention blocks CP, no TP shard plan; SDPA only; the fp8 release needs a one-time BF16 conversion (`halo run convert-glm5-bf16`); no rollout weight sync |
| Step-3.7 Flash | Yes | Yes | No | No | Yes | No | No | Yes | composite VLM (198B total / ~11B active); per-layer head counts block TP, no CP wrapper; SDPA only; GRPO works on vLLM (serve with `--trust-remote-code`); sharded EP saves are refused — use the gathered save |
| Any other HF causal LM | Yes | — | family-specific | native if `tp_plan` exists | — | — | — | Yes | standard FSDP path; a dense model without a TP plan raises at load instead of sharding |

Three rules cut across the table:

- Every `Yes` in EP+CP carries the same topology rule — EP stays node-local and
  `ep_size × expert_tp_size` equals the NVLink domain size
  ([Parallelism](parallelism.md)).
- ETP has no per-family opt-in: every EP-capable family shards its expert FFNs
  through the same base helper, so `untested` means not yet GPU-validated, not
  unsupported. GPT-OSS is the one behavioral exception — its interleaved expert
  weights cannot be de-interleaved once TP-sharded, so grouped GEMM turns off
  under ETP.
- LoRA `Yes` covers adapters under FSDP/DP, EP, CP, and pure ETP. TP and EP+TP
  reject LoRA outright (adapters are plain tensors outside the TP DTensor
  graph). Under any `expert_tp_size > 1`, adapters on the *expert* projections
  are additionally rejected — the replicated adapter half receives partial
  gradients and drifts across ranks — so keep `lora_target_modules` on
  attention there.

## PEFT and quantization

| Feature | Status | Notes |
| --- | --- | --- |
| LoRA | supported | except under TP / EP+TP — see above |
| QLoRA | supported on DDP / FSDP / CP | rejected under EP and TP; a MoE model also needs `use_grouped_gemm: false`. Rejected by online / environmental GRPO in every mode (vLLM weight sync ships raw 4-bit storage) |
| Adapter-only save / merge | supported | `halo run merge-peft-adapters` produces a standalone checkpoint |
| FP8 / FP4 MoE QAT / export | experimental | simulated backend for QAT; `quantize-to-lowp` for export |
| Muon | supported | optimizer option |
| FlashAdamW | supported | 8-bit quantized optimizer state; opt-in |

## Data

| Data path | Status | Notes |
| --- | --- | --- |
| HuggingFace datasets | supported | default quickstart path |
| local JSON / JSONL | supported | method-specific formats |
| S3 datasets | supported | optional; bring your own credentials, cache-aware |
| offline tokenization | supported | `halo run prepare-dataset`; recommended for larger corpora |
| packed SFT | supported | text-only path |
| VLM packing / padding-free | unsupported | images cannot be packed; VLM inputs use standard padding |
| streaming an infinite corpus | unsupported | the loaders always materialize a map-style `Dataset` — pre-tokenize and shard offline instead |

## Checkpointing

What each mode writes and how to load or resume it:
[Checkpoints & Export](checkpoints.md).

| Mode | Status | Notes |
| --- | --- | --- |
| Standard FSDP checkpoint | supported | exact resume path |
| PEFT adapter checkpoint | supported | adapter-only saves |
| EP gathered save | supported | large models may need sharding |
| EP sharded save | opt-in (`save_sharded_ep: true`) | merge before standalone use; refused for Step-3.7 Flash, whose hub-layout export the offline merge cannot reproduce |
| TP save | supported | full tensors are reconstructed from DTensors at save time; nothing to merge |
| Exact optimizer resume | all torchrun modes | needs a matching topology fingerprint ([Checkpoints](checkpoints.md#resume)) |
| Resume with a changed fingerprint | warm restart | weights and schedule restored, optimizer state reinitialized |

## Limits

- No hosted UI, no hyperparameter search, no built-in Hub upload — the
  inherited `push_to_hub` fields parse but are unguarded and untested
  ([Checkpoints](checkpoints.md)).
- Pipeline parallelism is not yet available.
- Advanced model-family support is explicit, not automatic.
- vLLM and SGLang run outside the training environment, in their own containers.
