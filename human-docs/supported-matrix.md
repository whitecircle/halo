# Supported Matrix

What runs, what doesn't, and what has been validated. The code is the source of
truth; this is the summary, and it wins over any recipe that disagrees with it.

## Runtime

| Area | Supported |
| --- | --- |
| Python | `>=3.12,<3.13`, installed with uv into the image |
| PyTorch | 2.11.x, `cu130` wheel (CUDA 13.2 toolkit in the image) |
| Transformers / TRL / Accelerate / PEFT | 5.16.x / 1.6.x / 1.11.x / 0.18.x |
| vLLM | 0.26.0, separate container — rollouts for both online RL methods |
| SGLang | 0.5.17, separate container — rollouts for async GRPO with environments |

| Hardware | Image | Status |
| --- | --- | --- |
| NVIDIA B200 / B300 | `halo:blackwell` | supported, primary Blackwell target |
| NVIDIA GB200 / GB300 NVL72 | `halo:blackwell` | supported; set `NVLINK_DOMAIN_SIZE=72` |
| NVIDIA H100 / H200 | `halo:hopper` | supported |
| NVIDIA A100 / Ampere | custom-arch build | not a release target — the shipped builds emit no sm_80 kernels |

Pull the prebuilt images or build them from source: [Installation](installation.md).

## Attention backends

Halo picks the backend: FA4 on Blackwell, FA3 on Hopper, FA2 as the fallback,
SDPA or eager where no Flash kernel serves the family. Gemma 4 gets no flash path
at all (FA2 caps head_dim at 256 and FA4 overflows tensor memory at its 512-wide
global layers); Qwen3.5/3.6 and GLM-4 MoE Lite are demoted off FA4 alone (its
backward NaNs at their shapes) and keep FA3 on Hopper; GLM-5 Next, Step-3.7
Flash, Inkling and Bailing/Ling run without one; DeepSeek-V4 needs eager.

Context parallelism picks its own kernel and ignores the configured label — FA3
on Hopper, FA4 on Blackwell, FA2 otherwise — and rejects SDPA except where a
family's wrapper waives the check (Bailing/Ling). Per-family resolution:
[Flash Attention](../agent-docs/optimization/flash-attention.md) ↗.

## Training methods

Every trainer supports EP, TP, ETP and EP+TP. CP is declare-to-enable and only
SFT and SMPO set it: nothing inspects a trainer's loss, so CP would silently
mis-pool across sequence shards. The Notes column says what keeps each of the
others off.

| Method | Script | CP | Notes |
| --- | --- | :---: | --- |
| SFT | `scripts/training/sft.py` | Yes | also VLM and continued pretraining |
| SMPO | `scripts/training/preference/smpo.py` | Yes | reference-free preference |
| DPO / KTO | `scripts/training/preference/{dpo,kto}.py` | No | reference log-prob sums block CP |
| Reward modeling | `scripts/training/preference/rewards.py` | No | full-sequence pooling blocks CP |
| Classification | `scripts/training/classification.py` | No | full-sequence pooling blocks CP |
| Offline GRPO | `scripts/training/offline_grpo.py` | No | trains from scored completions |
| Online GRPO (RLVR) | `scripts/training/online_grpo/rlvr.py` | No | needs vLLM; `--use_sdpg=true` runs online SDPG |
| Async GRPO with environments | `scripts/training/environmental_grpo.py` | No | needs Ray plus a vLLM or SGLang server |
| Distillation | `scripts/training/distillation/` | No | teacher and self distillation |
| Embedding | `scripts/training/embedding.py` | No | SentenceTransformer trainer |

### Rollout engines

Async GRPO with environments serves rollouts from vLLM by default;
`rollout_backend: sglang` switches engines. Both read a family's experts in the
hub layout its gather emits, so what differs is which families each pinned
release can take an online weight update for — the trainer names the family and
the loader reason at construction, rather than failing mid-run.

- **No online RL at all**, whichever engine: Inkling, GLM-5 Next, Cohere2 MoE.
- **Refused by both engines**: Mistral4, Zaya, DeepSeek-V4, Ling 3.0 and Ring's
  linear checkpoints.
- **vLLM only**: Laguna and Step-3.7 Flash — SGLang's loaders assert full
  coverage on every call, which a chunked update cannot satisfy.
- **Both engines**, expert distribution included: everything else — dense
  families, GPT-OSS, Qwen3 MoE, Qwen3.5/3.6, GLM-4 MoE Lite, Gemma 4, Ling 2.0,
  LFM-2.

`rollout_max_thinking_tokens` is vLLM-only and refused under SGLang. SGLang must
be served from this repo's image, not upstream. Engine setup, ports, weight sync
and `routing_replay`: [Rollout Servers](rollout-servers.md).

## Parallelism modes

| Mode | Status | Best for |
| --- | --- | --- |
| FSDP2 / DP | supported | default dense and small MoE runs |
| HSDP (`use_hsdp: true`) | supported | multi-node DP; pure DP or CP only, no-op on one domain |
| EP | supported | MoE expert sharding |
| CP | supported | long-context SFT / SMPO |
| TP | supported | attention and weight sharding; family support varies |
| ETP | experimental | expert FFN memory, experts replicated |
| EP+CP · EP+TP | supported for selected families | MoE plus long context / attention sharding |
| EP+ETP | experimental | MoE expert memory pressure, node-local |
| TP+CP · TP+ETP · ETP+CP · EP+TP+ETP | unsupported | rejected at config validation |
| Pipeline parallelism | not yet available | `pipeline_parallel_size > 1` is rejected at config time |

The layout rules behind these — and the shapes rejected before a run starts —
are in [Parallelism](parallelism.md).

## Model families

Any HuggingFace `AutoModelForCausalLM` runs under FSDP2; this table is about
advanced parallelism. Qwen3 MoE, GPT-OSS, GLM-4 MoE Lite and Mistral4 carry the
broadest coverage; hybrid linear-attention families and MoEs without a registered
CP wrapper drop CP. What each family is for: [Supported Models](models.md).

| Model family | FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA | Notes |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| Qwen3 (dense) | Yes | — | Yes | Yes | — | — | — | Yes | reference dense family |
| Qwen3 MoE | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | broadest coverage |
| Qwen3-VL (text) | Yes | — | Yes | No | — | — | — | Yes | keep `tensor_parallel_size=1` — both variants raise at load under TP |
| Qwen3.5 / Qwen3.6 MoE | Yes | Yes | No | Yes | Yes | No | Yes | Yes | interleaved linear attention blocks CP; VL checkpoints train too (the MoE-VL wrapper has EP, the dense 9B-VL runs plain FSDP with `sdpa`) |
| GPT-OSS | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | interleaved fused experts; trainable attention sinks |
| GLM-4 MoE Lite | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | LoRA-style attention compression |
| Command A+ (Cohere2 MoE) | Yes | Yes | Yes | Yes | Yes | Yes | Yes | untested | only EP is validated on the 200B+ checkpoint; the other modes pass the tiny-model matrix. No online RL, on either engine |
| Laguna S / XS 2.1 | Yes | Yes | No | No | untested | No | No | Yes | native in transformers, released checkpoints still load through remote code at a pinned revision; `sdpa`, so `padding_free` is rejected and they pack instead; weight sync on vLLM only |
| Gemma 4 MoE | Yes | Yes | No | No | Yes | No | No | partial | KV-shared layers block CP/TP; no router-balancing path at all; attention LoRA on the multimodal checkpoint is refused at PEFT setup (open issue), so target the experts or load `text_only_model: true` |
| Bailing/Ling | Yes | Yes | Yes | No | Yes | untested | No | Yes | EP covers Ling 2.0, Ling 3.0 and the Ring siblings; CP on Ling 2.0 only; no DTensor attention plan. Ling 3.0 and Ring's linear spellings take no online weight update |
| LFM-2 MoE | Yes | Yes | No | Yes | Yes | No | Yes | Yes | short-conv layers block CP |
| Mistral4 MoE | Yes | Yes | Yes | Yes | Yes | untested | Yes | Yes | neither engine registers a `mistral4` class, so no online RL |
| DeepSeek-V4 | Yes | Yes | No | No | untested | No | No | Yes | shared-KV MQA and the sparse-attention compressors block CP/TP; eager-only, so `padding_free` is rejected (packing works but warns). No online RL |
| Zaya | Yes | Yes | No | No | Yes | No | No | Yes | EP or ETP, always without gradient checkpointing; CCA blocks CP, its attention class carries no TP plan. No online RL on either engine |
| Inkling | Yes | Yes | No | No | Yes | No | No | untested | multimodal MoE; short-conv layers and a relative-logits bias block CP, and its attention class is not TP-shardable. No online RL |
| GLM-5 Next (GLM-5.3-Flash) | Yes | Yes | No | No | Yes | No | No | Yes | composite VLM; KDA linear attention blocks CP and is not TP-shardable, SDPA only; the fp8 release needs `halo run convert-glm5-bf16` first. No online RL |
| Step-3.7 Flash | Yes | Yes | No | No | Yes | No | No | Yes | composite VLM; per-layer head counts block TP, no CP wrapper, SDPA only; sharded EP saves refused — use the gathered save; online RL on vLLM only |
| Any other HF causal LM | Yes | — | family-specific | native if `tp_plan` exists | — | — | — | Yes | a dense model without a TP plan raises at load instead of sharding |

Three rules cut across the table. Every `Yes` in EP+CP carries the same topology
rule — EP stays node-local and `ep_size × expert_tp_size` equals the NVLink
domain size. ETP has no per-family opt-in (every EP-capable family shards expert
FFNs through the same helper, so `untested` means not yet GPU-validated), with
GPT-OSS the one behavioral exception: its interleaved expert weights cannot be
de-interleaved once TP-sharded, so grouped GEMM turns off under ETP. And LoRA
`Yes` covers FSDP/DP, EP, CP and pure ETP — TP and EP+TP reject adapters
outright, and any `expert_tp_size > 1` additionally rejects adapters on the
*expert* projections, so keep `lora_target_modules` on attention there.

## PEFT, quantization and data

| Feature | Status |
| --- | --- |
| LoRA | supported, except under TP / EP+TP; expert-projection adapters are also refused once `expert_tensor_parallel_size > 1` |
| QLoRA | DDP / FSDP / CP only; a MoE model also needs `use_grouped_gemm: false`. Rejected by both online RL methods in every mode |
| Adapter merge | `halo run merge-peft-adapters` produces a standalone checkpoint |
| FP8 / FP4 MoE QAT and export | experimental; simulated backend for QAT, `quantize-to-lowp` for export |
| Muon, FlashAdamW | supported optimizer options |
| HuggingFace / local / `s3://` datasets | supported; offline tokenize-pack-shard via `halo run prepare-dataset` |
| VLM packing / padding-free | unsupported — images cannot be packed, VLM inputs use standard padding |
| Streaming an infinite corpus | unsupported — the loaders materialize a map-style `Dataset`; pre-tokenize and shard offline |

What each mode writes, and which resumes are exact:
[Checkpoints](checkpoints.md).

## Limits

- No hosted training UI, no hyperparameter search, no built-in model upload to the Hub
  (the inherited `push_to_hub` fields parse but are unguarded and untested).
- Pipeline parallelism is not yet available.
- Advanced model-family support is explicit, not automatic.
- vLLM and SGLang run outside the training environment, in their own containers.
