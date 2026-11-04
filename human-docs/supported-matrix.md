# Supported Matrix

What runs, what does not, and what has been validated. The code is the source of
truth; where a recipe disagrees with this page, this page wins.

## Runtime

| Package | Version in the image |
| --- | --- |
| Python | 3.12, installed with uv into the system interpreter |
| PyTorch | 2.11.0, `cu130` wheel (CUDA 13.2 toolkit) |
| Transformers | 5.20.2 |
| TRL | 1.14.2 |
| Accelerate | 1.11.0 |
| PEFT | 0.18.1 |
| sentence-transformers | 6.1.0 |
| Liger Kernel | 0.8.4 |
| Flash Attention 4 | 4.0.0b37 (beta), Blackwell image only |
| flash-linear-attention | 0.4.2 |
| vLLM | 0.32.0 on Model Runner V2, its own container: rollouts for both online RL methods |
| SGLang | 0.5.22, its own container: rollouts for async GRPO with environments |

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

Leave `attn_implementation` unset unless a recipe pins it. Halo picks FA4 on
Blackwell, FA3 on Hopper and FA2 as the fallback, except:

| Family | Backend |
| --- | --- |
| Qwen3.5/3.6, Qwen3-Next | FA4 on Blackwell, FA3 on Hopper; on Blackwell, packed SFT rows take FA2 unless you pin FA4, and `full_determinism` takes SDPA (FA2 on packed rows) |
| GLM-4 MoE Lite | FA2 on Blackwell, FA3 on Hopper |
| Gemma 2 / Gemma 3 | FA2 on Blackwell at head_dim 256; Gemma 2 27B takes FA4 at one unpadded row per batch and FA2 otherwise; a softcapped model never runs SDPA |
| Gemma 4 | SDPA, with FlexAttention on sliding-window layers; no flash path |
| PaliGemma / PaliGemma 2 | SDPA / `flex_attention`; a pinned flash label falls back with a warning |
| GLM-5 Next, Step-3.7 Flash, Inkling | SDPA, chosen automatically |
| Qwen3.8-Flash-Next | SDPA, with Halo's tiled kernel on the sparse layers |
| DeepSeek-V4 | eager only |
| Laguna with `trust_remote_code: true` | SDPA, and a flash pin is refused |
| Ling 3.0 | set `attn_implementation: sdpa`; the flash label fails the build |
| LFM2-VL | SDPA, chosen automatically |

Padded-batch scripts use a different default on GPUs other than Blackwell
([Writing a config](configuration.md#the-keys-that-matter)). CP picks its own
kernel and ignores the configured label: FA3 on Hopper, FA4 on Blackwell (FA2 for
the families that fall back from FA4), FA2 otherwise. It rejects an explicit SDPA
label, except for Bailing/Ling.

`packing` is refused where a packed row's documents would mix: GPT-OSS and Ling
2.0 / Ring off flash, and on every backend Ling 3.0, DeepSeek-V4 and flat rows of
GLM-5 Next (tree packing keeps GLM-5 Next's documents apart). Ring and Inkling mix
documents on every kernel, so packing them takes
`allow_cross_document_linear_attention: true`. `padding_free` needs a flash
kernel, and Qwen3.5/3.6 refuse it
([Document isolation](../agent-docs/data/collators.md#document-isolation-under-packing) ↗).
Per-family details:
[Flash Attention](../agent-docs/optimization/flash-attention.md#model-specific-handling) ↗.

## Training methods

No trainer refuses EP, TP or ETP. SMPO, KTO, teacher distillation and embedding
are verified under EP and TP only, and self-distillation under EP only
([Trainer Compatibility](../agent-docs/reference/trainer-architecture.md#trainer-compatibility) ↗).
CP and PP are enabled per trainer; CP on any other loss would silently mis-pool
across sequence shards. Under PP every trainer's checkpoint save, resume and
export are verified.

| Method | Script | CP | PP | Notes |
| --- | --- | :---: | :---: | --- |
| SFT | `scripts/training/sft.py` | Yes | Yes | also VLM and continued pretraining |
| SMPO | `scripts/training/preference/smpo.py` | Yes | Yes | reference-free preference |
| DPO / KTO | `scripts/training/preference/{dpo,kto}.py` | No | Yes | reference log-prob sums block CP; PP needs precomputed reference log-probs, and KTO `loss_type: apo_zero_unpaired` |
| Reward modeling | `scripts/training/reward/reward_model.py` | No | Yes | full-sequence pooling blocks CP |
| Classification | `scripts/training/classification.py` | No | Yes | full-sequence pooling blocks CP |
| Offline GRPO | `scripts/training/grpo/offline.py` | Yes | Yes | CP is full fine-tuning only: PEFT, native expert LoRA and an explicit `ref_model` are rejected |
| Online GRPO (RLVR) | `scripts/training/grpo/online.py` | No | No | needs vLLM; `--use_sdpg=true` runs online SDPG |
| Async GRPO with environments | `scripts/training/grpo/environmental.py` | No | No | needs Ray and a vLLM or SGLang server |
| Distillation | `scripts/training/distillation/` | No | No | teacher and self distillation |
| Embedding | `scripts/training/embedding.py` | No | No | SentenceTransformer trainer |

### Rollout engines

Online GRPO and online SDPG generate on vLLM only. Async GRPO with environments
uses vLLM by default; `rollout_backend: sglang` switches engines. The engines
differ in which families they can take an online weight update for. The scripts
refuse an unsupported pair at startup, naming the family and the reason.

The last column is what passed on this release: async GRPO, full fine-tuning on
a real checkpoint, on one 8× B300 node with a TP=1 server, unless it says
otherwise. `ep1` is `expert_parallel_size: 1` on the EP layers, `tp2` is
`tensor_parallel_size: 2`, and replay is `routing_replay: rollout`.

| Family | vLLM 0.32.0 | SGLang 0.5.22 | Replay | Ran on this release |
| --- | :---: | :---: | :---: | --- |
| Dense families | Yes | Yes | No, MoE only | vLLM: online GRPO, SDPG and async GRPO on Qwen3-0.6B. SGLang: a weight push on Qwen3-0.6B, no training loop |
| GPT-OSS | Yes | Yes | Yes | `unsloth/gpt-oss-20b-BF16`. SGLang: every SGLang mode marked Ran below. vLLM: `ep4`, `ep2` with `tp2` or ETP2, `tp2` and `ep2` at DP2, HSDP, online GRPO at `ep2` |
| Qwen3 MoE | Yes | Yes | Yes | Qwen3-30B-A3B. vLLM: online GRPO, SDPG and async GRPO. SGLang: `ep1`, `ep2`, each also with replay |
| Qwen3.5 / Qwen3.6 MoE | Yes | Yes | Yes | Qwen3.6-35B-A3B. vLLM: `ep2`, `tp2`, online GRPO at `ep2`. SGLang: `ep1`, `ep2`. Both: LoRA at `ep1`, replay at `ep2`, also on Qwen3.5-35B-A3B |
| GLM-4 MoE Lite | Yes | Yes | Yes | GLM-4.7-Flash, with and without replay. vLLM: `ep1`, `ep2`, `tp2`. SGLang: `ep1`, `ep2` |
| LFM-2 MoE | Yes | Yes | Yes | LFM2-24B-A2B, with and without replay. vLLM: `ep1`, `ep2`, `tp2`. SGLang: `ep1`, `ep2` |
| Gemma 4 MoE | Yes | Yes | No, on either engine | Gemma 4 26B-A4B on both engines: `ep1`, `ep2`, LoRA at `ep1`. vLLM: online GRPO at `ep2` |
| Ling 2.0 | Yes | Yes | Yes | Ling-mini-2.0. vLLM: `ep1`, `ep2`. Both: replay at `ep2` |
| Laguna | Yes | No (SGLang's loader asserts full coverage per call) | vLLM | Laguna-XS-2.1 on vLLM: `ep1`, `ep2`, replay at `ep2` |
| Step-3.7 Flash | Yes | No (SGLang's loader asserts full coverage per call) | vLLM | vLLM: a weight push on a tiny hub-layout checkpoint, no training loop |
| Mistral4, Zaya, DeepSeek-V4, Ling 3.0 and Ring's linear checkpoints | No | No | No | |
| Inkling, GLM-5 Next, Qwen3-Next, Qwen3.8-Flash-Next, Cohere2 MoE, Nemotron-H | No | No (no online RL at all) | No | |

A MoE with no family in this table has no validated expert layout on either
engine and is refused too. Offline GRPO needs no server, so these refusals do
not apply to it. `rollout_max_thinking_tokens` and `carry_reasoning` are vLLM-only. SGLang must
be served from this repo's image. Setup, ports, weight sync and
`routing_replay`: [Rollout Servers](rollout-servers.md).

A push of unchanged weights leaves the served log-probs bit-identical on every
family that syncs, on each engine that takes it (MoE at `ep1` and `ep2`). Trainer
and engine still disagree on identical weights through bf16 routing flips. Replay
narrows that gap and is experimental:
[numbers](../agent-docs/training-methods/async-grpo/objective.md#trainer-and-engine-noise) ↗.

A dense checkpoint under a spelling a MoE family claims is refused with that
family: LFM2-VL over a dense LFM-2 tower, Mistral 3 over a dense Mistral tower,
Gemma 4 with `enable_moe_block: false`, Command A Vision over dense Cohere2.

**Ran** means the code admits it and a row passed against a live server on this
release, on at least one family; **Yes** means the code admits it. The Offline
GRPO column states what the code admits. A family's own limits from
[Model families](#model-families) still apply.

| Mode | Online GRPO, SDPG (vLLM) | Async GRPO, vLLM | Async GRPO, SGLang | Offline GRPO |
| --- | :---: | :---: | :---: | :---: |
| Plain FSDP2 or one GPU, dense | Ran | Ran | Yes | Yes |
| MoE `ep1`, experts sharded (default) | Ran | Ran | Ran | Yes |
| MoE `ep1`, `fsdp_shard_ep1_experts: false` | Ran | Ran | Yes | Yes |
| ZeRO-3 with no expert group | Ran (dense) | Ran (`ep1`) | Ran (`ep1`) | Yes |
| MoE with `use_grouped_gemm: false` | No | No | No | Yes |
| EP, one group (`expert_parallel_size` = trainer GPUs) | Ran (`ep2`) | Ran (`ep2`, `ep4`) | Ran (`ep2`, `ep4`) | Yes |
| `expert_parallel_size: 2`, several groups in one domain | Ran | Ran | Yes | Yes |
| `expert_parallel_size` above 2, group narrower than the one domain | No | No | No | No |
| Node-local EP groups across domains | Ran (two simulated domains) | Ran (two simulated domains) | Yes | Yes |
| One EP group across domains (`ep_scope: global`) | Yes | Yes | Yes | Yes |
| TP | Ran | Ran | Ran | Yes |
| Pure ETP | Ran (attention LoRA) | Ran (attention LoRA) | Yes | Yes |
| EP+TP | Ran | Ran | Ran | Yes |
| EP+ETP | Ran | Ran | Ran | Yes |
| HSDP | Ran (two simulated domains) | Ran (two simulated domains) | Yes | Yes |
| CP, EP+CP | No | No | No | Full fine-tuning |
| PP, PP+EP, PP+ETP | No | No | No | Full fine-tuning |
| TP+CP, TP+ETP, ETP+CP, EP+TP+ETP | No | No | No | No |
| `accelerate launch`, plain data parallelism | Dense only | Dense only | Dense only | Yes |
| PEFT attention LoRA, off TP and EP+TP | Ran | Ran | Ran | Yes, off CP and PP too |
| Native expert LoRA, off ETP, TP and EP+TP | Ran | Ran | Ran | Yes, off CP and PP too |
| QLoRA | No | No | No | Plain data parallelism only |

| Feature | Online GRPO, SDPG (vLLM) | Async GRPO, vLLM | Async GRPO, SGLang |
| --- | :---: | :---: | :---: |
| Resume | Ran | Ran | Ran |
| Several servers | No, one server | Ran | Yes |
| Prefetch | No | Ran (depth 1, 2) | Ran (depth 1); `max_retries` must cover the syncs a round spans |
| `routing_replay: rollout` | No | Ran | Ran |
| `routing_replay: recompute` | No | Ran | Ran |
| `rollout_max_thinking_tokens` | No | Ran (GPT-OSS, Qwen3.6) | No |
| `carry_reasoning` | No | Yes | No |
| `isr_engine_reference` | No | Yes | Yes |
| Server at TP above 1 | Yes | Ran (`tp2`) | Ran (`tp2`) |
| Server on another machine | Yes | Ran | Ran |

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
| TP+CP, TP+ETP, ETP+CP, PP+TP, PP+CP, any three axes | unsupported | rejected at config validation |
| PP, PP+EP | supported for registered families | depth across NVLink domains; each stage is a whole number of domains, so two nodes at least |
| PP+ETP | experimental (tiny models only) | expert FFN memory inside each stage, experts replicated |

Layout rules and the shapes rejected before a run starts:
[Parallelism](parallelism.md).

## Model families

Any HuggingFace `AutoModelForCausalLM` runs under FSDP2; this table covers
advanced parallelism. What each family is for:
[Supported Models](models.md).

| Model family | EP | ETP | CP | TP | PP | Notes |
| --- | :---: | :---: | :---: | :---: | :---: | --- |
| Qwen3 (dense) | — | — | Yes | Yes | Yes | reference dense family; PP refuses the tied 0.6B, 1.7B and 4B checkpoints |
| Qwen3 MoE | Yes | Yes | Yes | Yes | Yes | broadest coverage; PP on tiny models only, with `moe_balancing: none` |
| Qwen3-VL (text) | — | — | Yes | No | No | CP on text rows only; SFT and SMPO refuse an image run under CP |
| Qwen3.5 / Qwen3.6 / Qwen3.8-27B (dense) | — | — | No | Yes | Yes | the 9B-VL checkpoint trains on plain FSDP with `sdpa`; PP takes the text tower |
| Qwen3.5 / Qwen3.6 MoE | Yes | Yes | No | Yes | Yes | TP up to 2; the MoE-VL checkpoints train with EP; PP takes the text tower, with `moe_balancing: none` |
| Qwen3-Next | Yes | Yes | No | No | No | tiny models only |
| Qwen3.8-Flash-Next | Yes | Yes | No | No | Yes | experimental, text rows: EP8 trains the release cut to four layers, the rest runs on tiny models; full fine-tuning or LoRA; PP takes the text tower, full fine-tuning with `moe_balancing: none`; the 95 GiB n-gram table stays in host memory; `ep_size: 1` loads about 246 GiB on every GPU ([details](../agent-docs/models/qwen4_exp.md) ↗) |
| GPT-OSS | Yes | Yes | Yes | Yes | Yes | trainable attention sinks |
| GLM-4 MoE Lite | Yes | Yes | Yes | Yes | Yes | PP on tiny models only |
| Laguna S / XS 2.1 | Yes | Yes | No | No | Yes | ETP and PP on tiny models only; on SDPA, so use `packing`, not `padding_free` |
| Inkling-Small | Yes | Yes | No | No | Yes | multimodal MoE; LoRA and PP on tiny models only; PP takes the text tower, no packing |
| Gemma 4 MoE | Yes | Yes | No | No | No | no router balancing; LoRA adapts the language model only |
| Bailing MoE / Ling | Yes | Yes | Yes | No | No | EP covers Ling 2.0, Ling 3.0 and Ring; CP on Ling 2.0 only, and EP+CP on a tiny-model LoRA test only |
| LFM-2 MoE | Yes | Yes | No | Yes | No | |
| Mistral4 MoE | Yes | Yes | Yes | Yes | No | EP+CP on a tiny-model LoRA test only |
| DeepSeek-V4 | Yes | Yes | No | No | Yes | eager only; `packing` and `padding_free` refused; ETP on tiny models only; LoRA skips the grouped `o_a_proj` |
| Nemotron-H (Nemotron 3.5 Super VL) | Yes | Yes | No | No | Yes | EP8 on B300; its recipes, image SFT included, ran 5–20 steps on the release, the other modes and image SMPO on tiny models only; no video; no online RL; PP takes the text tower |
| Zaya (ZAYA1) | Yes | Yes | No | No | No | always without gradient checkpointing |
| Command A+ (Cohere2 MoE) | Yes | Yes | Yes | Yes | No | only EP8 ran on the 200B+ checkpoint, on Halo 1.0.0 or earlier; the other modes and LoRA on tiny models |
| GLM-5 Next (GLM-5.3-Flash) | Yes | Yes | No | No | Yes | composite VLM, SDPA only; PP takes the text tower; the fp8 release needs `halo run convert-glm5-bf16` first |
| Step-3.7 Flash | Yes | Yes | No | No | Yes | composite VLM, SDPA only; PP takes the text tower; sharded EP saves refused |
| Any other HF causal LM | — | — | No | if the model has a `tp_plan` | No | a dense model without a TP plan raises under TP |

EP+CP runs where EP and CP both do, EP+TP where EP and TP both do. Online
RL per family: [Rollout engines](#rollout-engines). Why a family drops a
mode: the [model pages](../agent-docs/models/README.md) ↗.

## PEFT, quantization and data

| Feature | Status |
| --- | --- |
| LoRA | supported except under TP, EP+TP and PP; expert-projection adapters also refused when `expert_tensor_parallel_size > 1`. LoRA ran on release checkpoints for GPT-OSS 20B, Qwen3-30B, Qwen3-0.6B, Ring-mini-linear-2.0, Gemma 4 26B and, for a few steps, Qwen3.8-Flash-Next. The other families' LoRA ran on tiny models. |
| QLoRA | DDP, FSDP and CP only, with the 4-bit base replicated; ran on dense Qwen3 and on gpt-oss-20b with stock experts; rejected by both online RL methods and by offline GRPO under CP |
| Adapter merge | `halo run merge-peft-adapters` writes a standalone checkpoint; native EP expert LoRA merges at save time instead (`merge_expert_lora_on_save: true`) |
| FP8 / FP4 MoE QAT and export | experimental: simulated backend for QAT, `quantize-to-lowp` for export |
| Muon, FlashAdamW | supported optimizer options, except under PP |
| HuggingFace, local and `s3://` datasets | supported; offline tokenize, pack and shard with `halo run prepare-dataset` |
| Tree packing (`tree_packing`) | SFT on raw text datasets: GPT-OSS, Qwen3 and Qwen3.5/3.6 (dense and MoE) on FA4 or FA2, GLM-5 Next on SDPA; refused under CP, TP and PP; ETP and EP+ETP are admitted ([Tree Packing](../agent-docs/optimization/tree-packing.md) ↗) |
| MTP head training (`mtp_loss_weight`) | SFT on text rows: Qwen3.5/3.6 (dense and MoE), Qwen3.8-27B, Qwen3-Next and GLM-4 MoE Lite, each as its release ships it, under FSDP2, EP and ETP; refused under CP, TP, PP, HSDP, adapters, quantized loads and `lowp_precision`, and on every other tail by name ([MTP heads](../agent-docs/optimization/mtp-heads.md) ↗) |
| VLM packing / padding-free | unsupported: images cannot be packed, so VLM inputs use standard padding |
| Streaming an infinite corpus | unsupported: the loaders build a map-style `Dataset`; pre-tokenize and shard offline |

What each mode writes, and which resumes are exact:
[Checkpoints](checkpoints.md).

## Validation

The GPU tests run on 8× B300 machines. Two 8-GPU machines ran node-local `ep8`
with DP through a per-node save, resume and evaluation, cross-node `ep16` through
a gathered save and reload, cross-node weight sync on both engines, and the two
pipeline-parallel recipes, the Qwen3-8B one through a per-node save, resume and
merge; the other multi-node shapes run as a two-node simulation on one host.
Checkpoints written by Halo 1.1.0 resume and export; a sharded EP save needs
`merge-ep-shards` first.
The full scope: [Release validation](../agent-docs/infrastructure/ci.md#release-validation) ↗
and [Verified topologies](../agent-docs/parallelism/multi-node.md#verified-topologies) ↗.

## Limits

- No hosted training UI and no hyperparameter search.
- No tested end-to-end Hub upload. Upload the final checkpoint yourself
  ([Checkpoints](checkpoints.md#uploading-to-the-huggingface-hub)).
- Advanced parallelism support is per family, not automatic.
- vLLM and SGLang run in their own containers, outside the training
  environment.
