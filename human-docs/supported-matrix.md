# Supported Matrix

What runs, what does not, and what has been validated. The code is the source of
truth; where a recipe disagrees with this page, this page wins.

## Runtime

| Area | Supported |
| --- | --- |
| Python | `>=3.12,<3.13`, installed with uv into the image |
| PyTorch | 2.11.x, `cu130` wheel (CUDA 13.2 toolkit in the image) |
| Transformers | 5.16.x |
| TRL | 1.6.x |
| Accelerate | 1.11.x |
| PEFT | 0.18.x |
| vLLM | 0.26.0, separate container: rollouts for both online RL methods |
| SGLang | 0.5.17, separate container: rollouts for async GRPO with environments |

| Hardware | Image | Status |
| --- | --- | --- |
| NVIDIA B200 / B300 | `halo:blackwell` | supported, primary target |
| NVIDIA H100 / H200 | `halo:hopper` | supported |
| NVIDIA GB200 / GB300 NVL72 | none | Grace hosts are aarch64; both images are x86_64-only |
| NVIDIA A100, RTX 3090 / 4090 | `halo:blackwell` | single-GPU LoRA/QLoRA only (torch, FA2 and bitsandbytes carry sm_80–sm_89 kernels; DeepEP, FA3 and FA4 do not run); not validated |
| RTX 50-series (SM 12.x) | `halo:blackwell` | set `attn_implementation: flash_attention_2` or `sdpa`, since the auto-selected FA4 does not run on SM 12.x; not validated |

Pull the prebuilt images or build them from source:
[Installation](installation.md).

## Attention backends

Halo picks the backend: FA4 on Blackwell, FA3 on Hopper, FA2 as the fallback,
SDPA or eager where no flash kernel serves the family.

- **Padded-batch scripts** (preference, reward, classification, teacher
  distillation, GRPO) default to SDPA when the YAML sets none. Exceptions keep
  the hardware pick: SMPO and offline GRPO under CP, and a GPT-OSS run with live
  sinks (`reset_sinks: false`).
- **Gemma 4**: no flash path (FA2 caps head_dim at 256, FA4 overflows tensor
  memory at its 512-wide global layers).
- **Qwen3.5/3.6, GLM-4 MoE Lite**: SDPA on Blackwell (FA4's backward NaNs at
  their shapes), FA3 on Hopper.
- **GLM-5 Next, Step-3.7 Flash, Inkling**: fall back to SDPA on their own.
- **DeepSeek-V4**: eager only.
- **Bailing/Ling, Laguna**: no automatic fallback. Set
  `attn_implementation: sdpa`, as their shipped configs do; on Bailing/Ling any
  flash label, the auto-selected one included, fails the model build.

CP picks its own kernel and ignores the configured label: FA3 on Hopper, FA4 on
Blackwell (FA2 for the families that fall back from FA4), FA2 otherwise. It
rejects an SDPA label, except for Bailing/Ling, so GLM-4 MoE Lite under CP sets
`attn_implementation: flash_attention_2` (its shipped EP config already does).
Per-family details:
[Flash Attention](../agent-docs/optimization/flash-attention.md) ↗.

## Training methods

Every trainer supports EP, TP, ETP and EP+TP. CP is enabled per trainer, only for
SFT, SMPO and offline GRPO; on any other loss it would silently mis-pool across
sequence shards.

| Method | Script | CP | Notes |
| --- | --- | :---: | --- |
| SFT | `scripts/training/sft.py` | Yes | also VLM and continued pretraining |
| SMPO | `scripts/training/preference/smpo.py` | Yes | reference-free preference |
| DPO / KTO | `scripts/training/preference/{dpo,kto}.py` | No | reference log-prob sums block CP |
| Reward modeling | `scripts/training/preference/rewards.py` | No | full-sequence pooling blocks CP |
| Classification | `scripts/training/classification.py` | No | full-sequence pooling blocks CP |
| Offline GRPO | `scripts/training/offline_grpo.py` | Yes | CP is full fine-tuning only: PEFT, native expert LoRA and an explicit `ref_model` are rejected |
| Online GRPO (RLVR) | `scripts/training/online_grpo/rlvr.py` | No | needs vLLM; `--use_sdpg=true` runs online SDPG |
| Async GRPO with environments | `scripts/training/environmental_grpo.py` | No | needs Ray and a vLLM or SGLang server |
| Distillation | `scripts/training/distillation/` | No | teacher and self distillation |
| Embedding | `scripts/training/embedding.py` | No | SentenceTransformer trainer |

### Rollout engines

Async GRPO with environments uses vLLM by default; `rollout_backend: sglang`
switches engines. The engines differ in which families they can take an online
weight update for. The trainer refuses an unsupported pair at construction,
naming the family and the reason.

| Families | vLLM | SGLang |
| --- | :---: | :---: |
| Dense families, GPT-OSS, Qwen3 MoE, Qwen3.5/3.6, GLM-4 MoE Lite, Gemma 4, Ling 2.0, LFM-2 | Yes | Yes |
| Laguna, Step-3.7 Flash | Yes | No (SGLang's loaders assert full coverage per call) |
| Mistral4, Zaya, DeepSeek-V4, Ling 3.0 and Ring's linear checkpoints | No | No |
| Inkling, GLM-5 Next, Cohere2 MoE | No | No (no online RL at all) |

`rollout_max_thinking_tokens` and `carry_reasoning` are vLLM-only. SGLang must
be served from this repo's image. Setup, ports, weight sync and
`routing_replay`: [Rollout Servers](rollout-servers.md).

## Parallelism modes

| Mode | Status | Best for |
| --- | --- | --- |
| FSDP2 / DP | supported | dense and small MoE runs (default) |
| HSDP (`use_hsdp: true`) | supported | multi-node DP; pure DP or CP only, no-op on one domain |
| EP | supported | MoE expert sharding |
| CP | supported | long-context SFT, SMPO, offline GRPO |
| TP | supported | attention and weight sharding; family support varies |
| ETP | experimental | expert FFN memory, experts replicated |
| EP+CP, EP+TP | supported for selected families | MoE plus long context or attention sharding |
| EP+ETP | experimental | expert memory pressure; each ETP group stays inside one NVLink domain, and cross-node EP+ETP forms one ETP group per domain |
| TP+CP, TP+ETP, ETP+CP, any three axes | unsupported | rejected at config validation |
| Pipeline parallelism | not yet available | `pipeline_parallel_size > 1` is rejected at config time |

Layout rules and the shapes rejected before a run starts:
[Parallelism](parallelism.md).

## Model families

Any HuggingFace `AutoModelForCausalLM` runs under FSDP2; this table covers
advanced parallelism. What each family is for:
[Supported Models](models.md).

| Model family | FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA | Notes |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| Qwen3 (dense) | Yes | — | Yes | Yes | — | — | — | Yes | reference dense family |
| Qwen3 MoE | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | broadest coverage |
| Qwen3-VL (text) | Yes | — | Yes | No | — | — | — | Yes | CP on text-only batches (a batch with images raises mid-run); TP raises at load, so keep `tensor_parallel_size=1` |
| Qwen3.5 / Qwen3.6 MoE | Yes | Yes | No | Yes | Yes | No | Yes | Yes | linear attention blocks CP; VL checkpoints train (MoE-VL with EP, dense 9B-VL on plain FSDP with `sdpa`) |
| GPT-OSS | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | trainable attention sinks |
| GLM-4 MoE Lite | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | |
| Command A+ (Cohere2 MoE) | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | only EP validated on the 200B+ checkpoint; the other modes and LoRA pass tiny-model GPU tests |
| Laguna S / XS 2.1 | Yes | Yes | No | No | Yes | No | No | Yes | native in transformers; set `attn_implementation: sdpa` (no automatic fallback), so use `packing`, not `padding_free` |
| Gemma 4 MoE | Yes | Yes | No | No | Yes | No | No | Yes | KV-shared layers block CP/TP; no router balancing; LoRA adapts the language model only |
| Bailing/Ling | Yes | Yes | Yes | No | Yes | partial | No | Yes | EP covers Ling 2.0, Ling 3.0 and Ring; CP on Ling 2.0 only |
| LFM-2 MoE | Yes | Yes | No | Yes | Yes | No | Yes | Yes | short-conv layers block CP |
| Mistral4 MoE | Yes | Yes | Yes | Yes | Yes | partial | Yes | Yes | |
| DeepSeek-V4 | Yes | Yes | No | No | Yes | No | No | Yes | eager only; `packing` and `padding_free` refused; LoRA skips the grouped `o_a_proj` |
| Zaya | Yes | Yes | No | No | Yes | No | No | Yes | always without gradient checkpointing |
| Inkling | Yes | Yes | No | No | Yes | No | No | Yes | multimodal MoE; LoRA passes tiny-model GPU tests only |
| GLM-5 Next (GLM-5.3-Flash) | Yes | Yes | No | No | Yes | No | No | Yes | composite VLM, SDPA only; the fp8 release needs `halo run convert-glm5-bf16` first |
| Step-3.7 Flash | Yes | Yes | No | No | Yes | No | No | Yes | composite VLM, SDPA only; sharded EP saves refused |
| Any other HF causal LM | Yes | — | No | if the model has a `tp_plan` | — | — | — | Yes | a dense model without a TP plan raises under TP |

`partial` marks a shape that only a tiny-model LoRA GPU test runs; validate a
short run before you rely on it. Online RL support per family is in
[Rollout engines](#rollout-engines). Why each family drops a mode:
[Supported Models](models.md) and the
[model pages](../agent-docs/models/README.md) ↗.

Rules that cut across the table:

- **EP+CP** always needs node-local EP with `ep_size` equal to the NVLink
  domain size.
- **ETP** needs no per-family opt-in. GPT-OSS turns grouped GEMM off under ETP.
- **LoRA** `Yes` covers FSDP/DP, EP, CP and pure ETP. TP and EP+TP reject
  adapters. With `expert_tp_size > 1`, adapters on expert projections are also
  rejected, so keep `lora_target_modules` on attention there.

## PEFT, quantization and data

| Feature | Status |
| --- | --- |
| LoRA | supported except under TP and EP+TP; expert-projection adapters also refused when `expert_tensor_parallel_size > 1` |
| QLoRA | DDP, FSDP and CP only, with the 4-bit base replicated rather than sharded; a MoE model also needs `use_grouped_gemm: false`; rejected by both online RL methods and by offline GRPO under CP |
| Adapter merge | `halo run merge-peft-adapters` writes a standalone checkpoint; native EP expert LoRA merges at save time instead (`merge_expert_lora_on_save: true`) |
| FP8 / FP4 MoE QAT and export | experimental: simulated backend for QAT, `quantize-to-lowp` for export |
| Muon, FlashAdamW | supported optimizer options |
| HuggingFace, local and `s3://` datasets | supported; offline tokenize, pack and shard with `halo run prepare-dataset` |
| VLM packing / padding-free | unsupported: images cannot be packed, so VLM inputs use standard padding |
| Streaming an infinite corpus | unsupported: the loaders build a map-style `Dataset`; pre-tokenize and shard offline |

What each mode writes, and which resumes are exact:
[Checkpoints](checkpoints.md).

## Limits

- No hosted training UI and no hyperparameter search.
- No tested end-to-end Hub upload. Upload the final checkpoint yourself
  ([Checkpoints](checkpoints.md#uploading-to-the-huggingface-hub)).
- Pipeline parallelism is not yet available in this release.
- Advanced parallelism support is per family, not automatic.
- vLLM and SGLang run in their own containers, outside the training
  environment.
