# Model Cookbooks

One page each for ten of the fifteen shipped MoE families, taking that family
from `docker pull` to inference or a served checkpoint, with LoRA and RL variants
where it supports them.

These are worked examples, not the rulebook. The rules a layout has to satisfy
live in [Parallelism](../parallelism.md), and what each family supports at all
is in the [Supported Matrix](../supported-matrix.md). When a cookbook and the
matrix disagree, the matrix wins.

- [GPT-OSS](halo-gpt-oss-cookbook.md) — `gpt-oss-20b` on UltraChat at EP8, with
  CP/TP/ETP variants, LoRA over attention *and* experts, and RL with live
  attention sinks.
- [Qwen3 MoE](halo-qwen3-moe-cookbook.md) — Qwen3-30B-A3B on four GPUs at EP4,
  with CP, TP, and ETP variants on the same ranks.
- [Qwen3.5 / Qwen3.6 MoE](halo-qwen3.5-qwen3.6-moe-cookbook.md) — 35B-A3B at EP8
  with transient bias-update balancing; TP is capped at 2 and the hybrid
  linear-attention layers rule out CP.
- [GLM-4.7-Flash](halo-glm-4.7-flash-cookbook.md) — EP8 at 30,720 tokens from the
  shipped config, plus CP2/TP2/ETP8 and LoRA on GLM's compressed attention.
- [Gemma 4 MoE](halo-gemma4-moe-cookbook.md) — 26B-A4B at EP8 and 32,768 tokens,
  where attention is SDPA-only and no router balancing path exists.
- [Mistral 4 MoE](halo-mistral4-moe-cookbook.md) — Mistral Small 4 119B A6B at
  EP8 and 32,000 tokens, with CP, TP, ETP, multimodal inference, and the required
  FP8-to-BF16 checkpoint conversion.
- [Laguna 2.1](halo-laguna-2.1-cookbook.md) — Laguna S at EP4 on exactly four
  GPUs, Laguna XS on one, and the CJK pad/eos tokens that silently break if
  retyped in ASCII.
- [LFM-2 MoE](halo-lfm2-moe-cookbook.md) — LFM2.5-8B-A1B at EP2, scaled to
  LFM2-24B-A2B at EP4, plus EP+TP over the full-attention layers.
- [ZAYA1](halo-zaya1-cookbook.md) — 8B on a single GPU straight from hub `main`
  (a native transformers family), with bias-update balancing and gradient
  checkpointing off in every mode.
- [Command A+](halo-command-a-plus-cookbook.md) — Cohere2 MoE at EP8, validated at
  full scale on an 8-GPU B300 node; the CP/TP/ETP wrappers are GPU-verified at tiny scale but
  untested on the 200B+ checkpoint.

## Start the training container

Every cookbook trains inside this container. Export `HF_TOKEN` and `WANDB_API_KEY`
in the host shell, and point `HALO_SCRATCH` at a large volume: `/mnt` is not
guaranteed large, so check with `df -h`.

```bash
git clone --recurse-submodules https://github.com/whitecircle/halo
cd halo
export HALO_IMAGE=public.ecr.aws/whitecircle/halo:blackwell   # :hopper on H100 / H200
export HALO_SCRATCH=/path/to/large/volume
docker pull "$HALO_IMAGE"
mkdir -p "$HALO_SCRATCH/hf" "$HALO_SCRATCH/checkpoints" "$HALO_SCRATCH/tmp"
docker run --rm -it --gpus all --network host --ipc=host --shm-size=128g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_TOKEN -e WANDB_API_KEY \
  -e HF_HOME=/data/hf -e HF_DATASETS_CACHE=/data/hf/datasets \
  -e TMPDIR=/data/tmp -e HALO_DATA_ROOT=/data \
  -v "$(pwd)":/workspace -v "$HALO_SCRATCH":/data -w /workspace \
  "$HALO_IMAGE" bash
```

Training, inference and conversion commands run inside this container, where
`/data` is `$HALO_SCRATCH`: a run that writes `/data/checkpoints/<run>` leaves it
at `$HALO_SCRATCH/checkpoints/<run>` on the host.

## Serve from the host

Rollout and serving containers start on the host from the repo root, never inside
the training container, on GPUs the trainer does not use. Set this up once in each
host shell that starts a server:

```bash
cd halo
export HALO_SCRATCH=/path/to/large/volume   # the same volume the training container mounts
docker pull public.ecr.aws/whitecircle/halo:sglang-0.5.17
docker pull public.ecr.aws/whitecircle/halo:vllm-0.26.0
docker tag public.ecr.aws/whitecircle/halo:vllm-0.26.0 vllm-server:0.26.0
export SGLANG_IMAGE=public.ecr.aws/whitecircle/halo:sglang-0.5.17
export SGLANG_MODEL_DIR="$HALO_SCRATCH"   # SGLang mounts it read-only at the same path
export HF_HOME="$HALO_SCRATCH/hf"         # both servers share the training container's hub cache
```

SGLang then takes host paths (`$HALO_SCRATCH/checkpoints/<run>`). The vLLM service
mounts only the HuggingFace cache: add `- ${HALO_SCRATCH}:/data:ro` under `vllm-server`
`volumes:` in `docker-compose.vllm.yml`, and give vLLM the same `/data` paths the
trainer uses. vLLM answers a request naming any model but the one it serves with a
404, and the trainer names its `model_name_or_path`.

The compose files already pass the MoE backend weight sync needs
(`--moe-backend triton`, `--moe-runner-backend triton`); leave it. A run with
`routing_replay: rollout` also needs `VLLM_ENABLE_R3=1` or `SGLANG_ENABLE_R3=1`
exported where you run compose (it adds `--enable-return-routed-experts`). Plain serving works on any SGLang 0.5.17 image; weight sync needs this
repo's server images. Parsers, ports and the other variables: [Rollout Servers](../rollout-servers.md).
