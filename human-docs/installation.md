# Installation

Halo runs inside a Docker image that holds the whole Python stack: PyTorch 2.11
(CUDA 13), Transformers, TRL, Flash Attention and DeepEP. Don't `pip install`
Halo on the host; the compiled kernels exist only in the image.

The host needs an NVIDIA driver, Docker with the NVIDIA Container Toolkit, and
git. The images are x86_64 only, so Grace-based GB200/GB300 hosts (aarch64) need
an arm64 build, which the Dockerfile does not provide.

## 1. Clone the repo

The configs, scripts and `make` targets live in the repo:

```bash
git clone https://github.com/whitecircle/halo
cd halo
```

## 2. Get the image

Pick the image that matches your GPUs:

| GPU | Image | Attention |
| --- | --- | --- |
| B200 / B300 | `halo:blackwell` | Flash Attention 4 + 2 |
| H100 / H200 | `halo:hopper` | Flash Attention 3 + 2 |
| A100, RTX 3090 / 4090 (single GPU, LoRA/QLoRA) | `halo:blackwell` | Flash Attention 2 / SDPA; no DeepEP; not validated |

Pull a prebuilt image from Amazon ECR Public (no login or AWS account needed) and
give it the local name:

```bash
# Blackwell
docker pull public.ecr.aws/whitecircle/halo:blackwell
docker tag public.ecr.aws/whitecircle/halo:blackwell halo:blackwell

# Hopper
docker pull public.ecr.aws/whitecircle/halo:hopper
docker tag public.ecr.aws/whitecircle/halo:hopper halo:hopper
```

Don't skip the retag: this guide, the `make` targets and the compose files use
the local name. They default to `halo:blackwell`. On Hopper, pass
`IMAGE=halo:hopper` to `make` and set `TRAIN_IMAGE=halo:hopper` for compose.

- `:blackwell` and `:hopper` track the latest release. `:blackwell-1.1.0` and
  `:hopper-1.1.0` pin it.
- There is no `latest` tag, so a Hopper host can't pull a Blackwell image by
  mistake.
- The RL rollout servers are in the same repository, as `:vllm-0.26.0` and
  `:sglang-0.5.17`.

To build from source instead (no token or registry login):

```bash
make build-blackwell     # or: make build-hopper
```

The first build is slow: DeepEP, DeepGEMM (Blackwell) and Flash Attention
(Hopper) compile from source.

## 3. Create a `.env` file

Put your secrets in `.env` at the repo root. `.env.example` is a template:

```bash
HF_TOKEN=hf_...            # gated models and datasets on the HuggingFace Hub
WANDB_API_KEY=...          # Weights & Biases logging
AWS_ACCESS_KEY_ID=...      # only for s3:// datasets
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=...
```

Leave out any key you don't need. A run on a public model with no tracking works
with an empty file.

Halo never loads `.env` itself. A `docker run` you type needs `--env-file .env`.
`make train` and the `training` service in `docker-compose.vllm.yml` pass it for
you.

## 4. Point caches at a large disk

Model weights, dataset caches and checkpoints add up to hundreds of gigabytes,
and the root filesystem is usually small. Find your large volume and check its
size. A path named `/mnt` can still sit on the root disk.

```bash
df -h                             # find the volume with real space
findmnt -T /mnt -no TARGET,AVAIL  # the filesystem that actually holds /mnt
```

Pass that volume to the container through four variables:

- `HF_HOME`: model cache
- `HF_DATASETS_CACHE`: dataset cache
- `TMPDIR`: temp files
- `HALO_DATA_ROOT`: Halo's scratch (S3 dataset cache, profiler output)

The command below assumes the volume is `/mnt`; substitute yours. See
[Environment variables](environment-variables.md).

## 5. Start a container

From the repo root:

```bash
docker run --rm -it --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=128g \
  --cap-add=SYS_PTRACE \
  --env-file .env \
  -e HF_HOME=/mnt/hf \
  -e HF_DATASETS_CACHE=/mnt/hf/datasets \
  -e TMPDIR=/mnt/tmp \
  -e HALO_DATA_ROOT=/mnt \
  -v $(pwd):/workspace \
  -v /mnt:/mnt \
  -v ~/.aws:/root/.aws \
  -w /workspace \
  halo:blackwell bash
```

About the flags:

- `--ipc=host`, `--shm-size=128g` and the two ulimits are required. NCCL and the
  dataloaders fail without them.
- `--cap-add=SYS_PTRACE` lets py-spy attach to a hung run. Without it, that works
  only on hosts where `kernel.yama.ptrace_scope` is 0.
- For an RL run, add `--network host` so the trainer can reach a rollout server
  that compose started on the same host.
- Drop `-v ~/.aws:/root/.aws` unless you read S3 through your AWS profile.
- The repo is mounted at `/workspace`, so edits on the host show up in the
  container.

`make train` runs the same command for you, with `--network host` and without
`--cap-add=SYS_PTRACE`. It mounts the volume named by `HALO_SCRATCH` (default
`/mnt`):

```bash
make train CONFIG=examples/sft/qwen3/qwen3-4b-ultrachat.yaml NPROC=8
```

If your Docker default runtime rejects `--gpus` (for example `sysbox-runc`), add
`DOCKER_RUNTIME=nvidia` to any `make` target.

## 6. Verify

Inside the container:

```bash
nvidia-smi                                                   # GPUs visible
python -c "import torch; print(torch.cuda.is_available())"   # True
halo launch --list                                           # CLI works, methods indexed
```

If a check fails, the usual causes are a missing `--gpus all`, the wrong image
for your GPU architecture, or a missing NVIDIA Container Toolkit. See
[Troubleshooting](troubleshooting.md).

## Next steps

[Quickstart](quickstart.md) launches your first run, and
[Writing a config](configuration.md) shows what to change in it. Image
internals, the multi-container RL setup and registry publishing are in
[Docker](../agent-docs/infrastructure/docker.md) ↗ and
[Installation](../agent-docs/getting-started/installation.md) ↗.
