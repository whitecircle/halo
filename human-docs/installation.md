# Installation

Halo runs inside a Docker image that contains the entire Python stack —
PyTorch 2.11 (CUDA 13), Transformers, TRL, Flash Attention, DeepEP. The host
needs Docker with the NVIDIA Container Toolkit, an NVIDIA driver, and git.
Don't try to `pip install` Halo on the host; the compiled kernels only exist
in the image.

## 1. Get the image

Pick the image that matches your GPUs:

| GPU | Image | Attention stack |
| --- | --- | --- |
| B200 / B300 / GB200 / GB300 | `halo:blackwell` | Flash Attention 4 + 2 |
| H100 / H200 | `halo:hopper` | Flash Attention 3 + 2 |

You still need the repo — configs, scripts, and the `make` targets live there:

```bash
git clone --recurse-submodules https://github.com/whitecircle/halo
cd halo
```

Then pull a prebuilt image from Amazon ECR Public (no login, no AWS account):

```bash
docker pull public.ecr.aws/whitecircle/halo:blackwell   # B200/B300 — FA4 + FA2 + DeepEP
docker pull public.ecr.aws/whitecircle/halo:hopper       # H100/H200 — FA2 + FA3 + DeepEP
docker tag public.ecr.aws/whitecircle/halo:blackwell halo:blackwell
```

Do the retag: the rest of this guide and the `make` targets look for
`halo:blackwell`. The moving tags track the latest release and
`:blackwell-1.0.0` / `:hopper-1.0.0` pin it; there is deliberately no `latest`,
which would let a Hopper host silently pull a Blackwell image. The RL inference
servers live in the same repository as `:vllm-0.26.0` and `:sglang-0.5.17`.

Or build from source. No token, no registry login:

```bash
make build-blackwell     # or: make build-hopper
```

The first build takes a while: DeepEP (and, on Hopper, Flash Attention)
compile from source.

## 2. Create a `.env` file

Put your secrets in a `.env` file at the repo root:

```bash
HF_TOKEN=hf_...            # gated models/datasets on the HuggingFace Hub
WANDB_API_KEY=...          # Weights & Biases logging
AWS_ACCESS_KEY_ID=...      # only if you use s3:// datasets
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=...
```

Nothing reads this file automatically — you pass it to Docker with
`--env-file .env`. Skip any key you don't need; a public-model,
no-tracking run works with an empty file.

## 3. Point caches at a big disk

Model weights, dataset caches, and checkpoints add up to hundreds of gigabytes,
and the root filesystem is usually small. Find your large volume and verify it
actually is one; a path named `/mnt` proves nothing.

```bash
df -h                          # find the volume with real space
findmnt -no TARGET,AVAIL /mnt  # confirm that path is that volume, not the root disk
```

Hand that path to the container through four environment variables: `HF_HOME`
(model cache), `HF_DATASETS_CACHE` (dataset cache), `TMPDIR` (temp files), and
`HALO_DATA_ROOT` (Halo's own scratch — S3 dataset cache, profiler output). The
run command below assumes the volume is `/mnt`; substitute yours. See
[Environment Variables](environment-variables.md).

## 4. Start a container

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

On the flags: `--ipc=host --shm-size=128g` and the ulimits are required, since
NCCL and the dataloaders break without them. `--cap-add=SYS_PTRACE` is optional,
but without it you cannot attach py-spy to a hung run later. Drop
`-v ~/.aws:/root/.aws` unless you want S3 access through your AWS profile. The
repo mounts at `/workspace`, so host-side edits are live in the container.

The `make` targets (`make train`, `make test-cpu`, …) wrap this same invocation
if you'd rather not type it.

## 5. Verify

Inside the container:

```bash
nvidia-smi                                          # GPUs visible
python -c "import torch; print(torch.cuda.is_available())"   # True
halo launch --list                                  # CLI works, methods indexed
```

If any of these fail, see [Troubleshooting](troubleshooting.md) — the usual
suspects are a missing `--gpus all`, the wrong image for your GPU
architecture, or the NVIDIA Container Toolkit not being installed.

## Next

[Quickstart](quickstart.md) launches your first run. For image internals,
multi-container RL setup, and registry publishing, see the reference:
[Docker](../agent-docs/infrastructure/docker.md) ↗ ·
[Installation](../agent-docs/getting-started/installation.md) ↗.
