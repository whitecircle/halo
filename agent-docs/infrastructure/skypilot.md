# SkyPilot Deployment

Automated cluster provisioning for MoE training on AWS (EFA) and Nebius (InfiniBand). Configs live at `launcher-configs/skypilot/{provider}/{model}/{mode}.yaml` — five tasks per provider, tabled below. Adapt to other MoE families by swapping `CONFIG_PATH`.

## Install and authenticate

```bash
pip install "skypilot-nightly[nebius,aws]"   # nightly has best H200 availability
```

Nebius: install the CLI (`curl -sSL https://storage.ai.nebius.cloud/nebius/install.sh | bash`), then `nebius init` and run `nebius-setup.sh` from [nebius-solution-library](https://github.com/nebius/nebius-solution-library)`/skypilot`. AWS: `aws configure`. Verify with `sky check`.

## Launch

Every shipped task needs two edits first, or the launch burns a provisioning cycle before failing:
uncomment `resources.image_id` ([Docker image](#docker-image) — the `setup:` block exits non-zero
when `flash_attn` / `deep_ep` are missing) and set the S3 mount bucket (`--env HALO_BUCKET=<your-bucket>`) to your
bucket ([Storage](#storage)). Each task also pins its provider's region (`us-east-1` on AWS,
`eu-north1` on Nebius) — change it, or delete the key for cross-region failover, to match your quota.

```bash
sky launch -c oss-120b launcher-configs/skypilot/nebius/gpt-oss-120b/crossnode-ep.yaml \
  --secret HF_TOKEN --secret WANDB_API_KEY   # --secret masks tokens in logs

# retarget a task without editing it — every envs: key takes --env
sky launch -c q35-122b launcher-configs/skypilot/nebius/qwen3.5-122b-a10b/crossnode-ep.yaml \
  --secret HF_TOKEN --secret WANDB_API_KEY \
  --env MODEL=/data/checkpoints/q35-122b-sft \
  --env CONFIG_PATH=examples/sft/qwen3_5/qwen3.5-122b-a10b-ep.yaml \
  --env OUTPUT_DIR=/data/checkpoints/q35-122b-sft-v2
```

## Available configs

| Model | Mode | Layout | GPUs | Image |
|---|---|---|---|---|
| gpt-oss-20b | `nodelocal-ep` | 2 nodes, EP=8 node + DP | H100:8 (aws) / H200:8 (nebius) | `halo:hopper` |
| gpt-oss-120b | `crossnode-ep` | 2 nodes, EP=16 global | H200:8 | `halo:hopper` |
| gpt-oss-120b | `nodelocal-ep` | 1 node, EP=8 node | B200:8 | `halo:blackwell` |
| qwen3.5-122b-a10b | `crossnode-ep` | 2 nodes, EP=16 global | H200:8 | `halo:hopper` |
| qwen3.5-122b-a10b | `nodelocal-ep` | 1 node, EP=8 node | B200:8 | `halo:blackwell` |

Each shape exists under both `launcher-configs/skypilot/aws/` (EFA) and `launcher-configs/skypilot/nebius/` (InfiniBand), and drives its run from `envs`: `CONFIG_PATH` (the training config) plus `MODEL`, `EXPERT_PARALLEL_SIZE`, `EXPERT_PARALLEL_SCOPE` and `WANDB_PROJECT`, forwarded as `--model_name_or_path` / `--expert_parallel_size` / `--ep_scope` / `--project_name`. Every `envs:` key takes a launch-time `--env KEY=VALUE`, so `MODEL` (point it at a stage-1 `output_dir` to chain stages), `CONFIG_PATH`, the EP knobs, `OUTPUT_DIR` and `WANDB_PROJECT` retarget without editing the YAML — retarget `WANDB_PROJECT` alongside `MODEL`, or the run logs into the shipped project.

The `aws/` and `nebius/` copy of a shape are **not** interchangeable: beyond `cloud`/`region`/`accelerators` they differ in the RDMA fabric setup below (fabric env, the `/dev/gdrdrv` device, the AWS-only `fi_info` probe), which `--env` cannot supply.

120B/122B `nodelocal-ep` require Blackwell HBM. The gpt-oss tasks default `MODEL` to a BF16-dequantized mirror — EP cannot load the MXFP4 `openai/gpt-oss-*` ([GPT-OSS → Precision and kernels](../models/gpt-oss.md#precision-and-kernels)).

## Cross-node expert parallelism

Cross-node EP distributes MoE experts across nodes — at EP=16 each rank owns `num_experts / 16` experts
(GPT-OSS-120B's 128 experts → 8 per rank, node 0 holding 0–63 and node 1 holding 64–127) — reached via
NVLink within a node and InfiniBand/EFA RDMA between them, fitting models larger than one node.

Set in the SkyPilot YAML `envs`:

```yaml
envs:
  EXPERT_PARALLEL_SIZE: "16"     # EP group size across both nodes
  EXPERT_PARALLEL_SCOPE: "global"  # cross-node (not node-local)
  MAX_LENGTH: "8192"             # cross-node dispatch ceiling (~8k tokens/rank over Gin); pins any overridden config to it
  DIST_SHARED_FILESYSTEM: "1"    # default; "0" for local-only FS
```

The multi-node `run:` blocks are a plain `torchrun --nnodes=$SKYPILOT_NUM_NODES ... scripts/training/sft.py $CONFIG_PATH`
(single-node tasks hardcode `--nnodes=1`) that forwards `--expert_parallel_size` / `--ep_scope` / `--max_length` from these envs — it does not use the
`halo` CLI. `MAX_LENGTH` is a `crossnode-ep` knob: the training configs are tuned for a single NVLink
domain, while cross-node dispatch is capped at 8192 tokens/rank
([DeepEP → AWS EFA](deepep.md#expert-parallelism-over-aws-efa)) and is rejected above it.

### RDMA fabric: AWS EFA vs Nebius InfiniBand

The fabric env differs by cloud. The Nebius (IB) tasks keep the image's InfiniBand defaults. The two
single-node AWS tasks (`gpt-oss-120b/nodelocal-ep`, `qwen3.5-122b-a10b/nodelocal-ep`) set no fabric
env at all — intra-node NVLink only.

The three multi-node AWS (EFA) tasks add the libfabric vars and all set `NCCL_NET_PLUGIN: ofi`
explicitly, because the NGC base's `shinit_v2` sets it only for a shell that sources it. Without the
plugin NCCL loads the HPC-X IB plugin on an EFA host: the cross-node EP tasks abort with "NCCL GIN is
unavailable" once the cluster is already up, and `gpt-oss-20b/nodelocal-ep`'s cross-node DP gradient
sync falls off EFA.

```yaml
envs:                       # AWS EFA
  FI_PROVIDER: efa
  FI_EFA_USE_DEVICE_RDMA: "1"
  NCCL_PROTO: simple        # EFA is unreliable with NCCL's LL/LL128 protocols
  NCCL_NET_PLUGIN: ofi      # every multi-node AWS task (the OFI plugin, not HPC-X)
  NCCL_GIN_TYPE: "2"        # cross-node EP only (proxy GIN)
```

Every task sets `config.docker.run_options: ["--ipc=host", "--ulimit", "memlock=-1", "--ulimit",
"stack=67108864", "--shm-size=128g"]`; only the two `crossnode-ep` AWS tasks prepend
`"--device", "/dev/gdrdrv"` — node-local EP needs neither proxy GIN nor GDRCopy.

The multi-node tasks set `resources.network_tier: best` so SkyPilot provisions the EFA/InfiniBand interfaces; the single-node ones omit it. The host supplies the EFA driver and devices; the image bundles the GIN-capable `aws-ofi-nccl` + GDRCopy that standard NCCL collectives and DeepEP cross-node EP both run on.

GIN-plugin prerequisites and the measured B300 EFA numbers: [DeepEP → Expert parallelism over AWS EFA](../infrastructure/deepep.md#expert-parallelism-over-aws-efa). Per-fabric env and verification (`fi_info -p efa` vs `ibstat`): [Multi-Node → RDMA fabrics](../parallelism/multi-node.md#rdma-fabrics).

### NVL72 racks (GB200/GB300)

On NVL72 the NVLink domain spans the rack (72 GPUs across ~18 OS nodes), so node-local grouping should target the rack: set `NVLINK_DOMAIN_SIZE: "72"` and `EXPERT_PARALLEL_SCOPE: "node"` in `envs`. Standard 8-GPU hosts need no change — the default reads `gpus_per_node`. See [Multi-Node → NVL72](../parallelism/multi-node.md#gb200gb300-nvl72-multi-node-nvlink).

### NCCL collective timeout

Slow loads or large checkpoints can exceed the default. Every shipped `crossnode-ep` task already pins `DIST_NCCL_TIMEOUT_MINUTES: "60"` in `envs`; raise it further there if a load is slower still.

## Storage

Every shipped task declares two mounts:

```yaml
file_mounts:
  /workspace: {source: ., mode: COPY}                  # the repo
  /data:      {source: s3://${HALO_BUCKET}/halo, mode: MOUNT}   # bucket from envs; --env HALO_BUCKET=... retargets
```

`/data` is a **placeholder you must edit** before launching; `HF_HOME`, `HALO_DATA_ROOT` (the toolkit
scratch root) and `OUTPUT_DIR` all resolve under it.

| Mode | Use case | Write |
|------|----------|-------|
| `COPY` | Small datasets (<100GB), read-only | No |
| `MOUNT` | Checkpoints, write-heavy persistence | Yes |
| `MOUNT_CACHED` | Large models/datasets, read-heavy | Yes (async) |

Pre-upload large models (>10GB) to S3 once and mount them `MOUNT_CACHED`, keeping checkpoints on `MOUNT`:

```bash
aws s3 sync models/qwen3.5-122b-a10b s3://my-bucket/models/qwen3.5-122b-a10b/
```

Then point the run at that mount with `--env MODEL=/data/models/...` — the shipped tasks default
`MODEL` to a Hub id and keep only their writes under `/data` (`OUTPUT_DIR: /data/checkpoints/...`).
Alternatively download from HF at runtime with `HF_HOME` on a
`MOUNT_CACHED` bucket so the cache survives restarts. In multi-node training a `MOUNT` checkpoint bucket is
shared across all nodes — any node can save and reload after preemption. Pull results with `aws s3 sync`,
`rsync -Pavz oss-120b:/data/checkpoints/ ./`, or SSH; manage buckets with `sky storage ls` / `delete`.

## Docker image

`resources.image_id` is **not set in the shipped YAMLs — you must uncomment it.** Each carries it as a
commented line under `resources` naming the prebuilt image for that shape, which pulls anonymously (no
AWS account, no registry login). Each task's `setup:` block fails fast when `flash_attn` / `deep_ep` are
not importable, so a from-source build on a bare node is not a fallback; the prebuilt image is required.
The `hopper` image serves H100/H200 (Nebius and AWS p5/p5e) and `blackwell` the B200 single-node tasks.
To run your own build instead, push it to a registry the nodes can reach and point `image_id` there
([Docker](docker.md)).

```yaml
resources:
  image_id: docker:public.ecr.aws/whitecircle/halo:hopper
```

## Cluster management

```bash
sky status                   # list clusters
sky logs <cluster> --follow  # logs
ssh <cluster>                # SSH in
sky stop <cluster>           # stop, preserve state
sky down <cluster>           # terminate
```

Spot instances (AWS): spot with auto-recovery needs managed jobs — `sky jobs launch` with `use_spot: true` (`job_recovery: failover`); on a plain `sky launch` cluster spot preemption is not recovered. Auto-stop idle clusters with `sky launch --idle-minutes-to-autostop 30`. Check pricing with `sky show-gpus H200`.

## Troubleshooting

- **Cross-node EP hangs**: verify RDMA — `ibstat` (Nebius) or `fi_info -p efa` (AWS); on AWS also confirm `/dev/gdrdrv` reached the container.
- **OOM**: reduce `max_length` or enable gradient checkpointing.
- **NCCL errors**: check the NCCL env vars in the YAML; inspect setup with `sky logs <cluster> --status`.

## Resources

- [SkyPilot docs](https://docs.skypilot.co/) · [YAML spec](https://docs.skypilot.co/en/latest/reference/yaml-spec.html) · [Storage](https://docs.skypilot.co/en/latest/reference/storage.html)
- [Nebius SkyPilot integration](https://docs.nebius.com/3p-integrations/skypilot)
- [Multi-Node](../parallelism/multi-node.md) · [Expert Parallelism](../parallelism/expert-parallelism.md)
