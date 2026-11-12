# SkyPilot Deployment

Automated cluster provisioning for MoE training on AWS (EFA) and Nebius (InfiniBand). Four tasks live at `launcher-configs/skypilot/{aws,nebius}/{nodelocal,crossnode}-ep.yaml`, one per cloud fabric and EP topology, each defaulting to GPT-OSS 120B; the [presets](#presets) retarget them at launch, and other MoE families swap `CONFIG_PATH`.

## Install and authenticate

```bash
pip install "skypilot-nightly[nebius,aws]"   # nightly has best H200 availability
```

Nebius: install the CLI (`curl -sSL https://storage.ai.nebius.cloud/nebius/install.sh | bash`), then `nebius init` and run `nebius-setup.sh` from [nebius-solution-library](https://github.com/nebius/nebius-solution-library)`/skypilot`. AWS: `aws configure`. Verify with `sky check`.

## Launch

Every task needs two edits first, or the launch burns a provisioning cycle before failing:
set `resources.image_id` (uncomment it, or pass `--image-id`; [Docker image](#docker-image) — the `setup:`
block exits non-zero when `flash_attn` / `deep_ep` are missing) and set the S3 mount bucket
(`--env HALO_BUCKET=<your-bucket>`) to your bucket ([Storage](#storage)). Each task also pins its
provider's region (`us-east-1` on AWS, `eu-north1` on Nebius) — change it, or delete the key for
cross-region failover, to match your quota.

```bash
sky launch -c oss-120b launcher-configs/skypilot/nebius/crossnode-ep.yaml \
  --secret HF_TOKEN --secret WANDB_API_KEY   # --secret masks tokens in logs
```

As shipped, `nodelocal-ep` runs EP=8 at node scope on one B200:8 node (`halo:blackwell`) and `crossnode-ep` runs EP=16 at global scope on two H200:8 nodes (`halo:hopper`). Each task drives its run from `envs`: `CONFIG_PATH` (the training config) plus `MODEL`, `EXPERT_PARALLEL_SIZE`, `EXPERT_PARALLEL_SCOPE`, `OUTPUT_DIR` and `WANDB_PROJECT`, forwarded as `--model_name_or_path` / `--expert_parallel_size` / `--ep_scope` / `--output_dir` / `--project_name`. Every `envs:` key takes a launch-time `--env KEY=VALUE` and every `resources` field its `sky launch` flag (`--gpus`, `--num-nodes`, `--memory`, `--disk-size`, `--network-tier`, `--image-id`), so no retarget edits the YAML. Point `MODEL` at a stage-1 `output_dir` to chain stages, and retarget `OUTPUT_DIR` and `WANDB_PROJECT` alongside `MODEL`, or the run writes into the shipped checkpoint directory and project.

The two `crossnode-ep` files are **not** interchangeable: beyond `cloud`/`region` they differ in the RDMA fabric setup below (fabric env, the `/dev/gdrdrv` device, the `setup:` probe), and no launch flag supplies the device or the probe. The two `nodelocal-ep` files differ only in the EFA env the AWS `run:` block exports on more than one node.

## Presets

| Model | Task | `--env` | Resources |
|---|---|---|---|
| GPT-OSS 120B | both | none (the default) | as shipped |
| Qwen3.5-122B-A10B | both | `CONFIG_PATH=examples/sft/qwen3_5/qwen3.5-122b-a10b-ep.yaml`, `MODEL=Qwen/Qwen3.5-122B-A10B`, `OUTPUT_DIR=/data/checkpoints/qwen35-122b`, `WANDB_PROJECT=halo-qwen35-122b` | as shipped |
| GPT-OSS 20B | `nodelocal-ep` | `CONFIG_PATH=examples/sft/gptoss/gptoss-20b-multinode-ep.yaml`, `MODEL=openai/gpt-oss-20b`, `OUTPUT_DIR=/data/checkpoints/oss-20b`, `WANDB_PROJECT=halo-20b-nodelocal` | `--num-nodes 2 --network-tier best`, `--gpus H100:8` (AWS) or `H200:8` (Nebius), `--image-id docker:public.ecr.aws/whitecircle/halo:hopper`; `--memory 256+ --disk-size 1500` suffice |

The 120B and 122B models run `nodelocal-ep` on Blackwell only (B200's 180 GB/GPU is the minimum, B300 has headroom; lower `max_length` on OOM) and need the shipped `memory: 512+` and `disk_size: 3000`. Both gpt-oss presets load the release as OpenAI ships it, `openai/gpt-oss-120b` by default and `openai/gpt-oss-20b` for the 20B row, its experts decoded at load ([GPT-OSS → Configs](../models/gpt-oss.md#configs)). The 20B preset keeps EP inside each node and adds cross-node data parallelism over the fabric, hence `--network-tier best` ([RDMA fabric](#rdma-fabric-aws-efa-vs-nebius-infiniband)).

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

Both `run:` blocks are a plain `torchrun --nnodes=$SKYPILOT_NUM_NODES ... scripts/training/sft.py $CONFIG_PATH`
that forwards `--expert_parallel_size` / `--ep_scope` (and, in `crossnode-ep`, `--max_length`) from these envs — it does not use the
`halo` CLI. `MAX_LENGTH` is a `crossnode-ep` knob: the training configs are tuned for a single NVLink
domain, while cross-node dispatch is capped at 8192 tokens/rank
([DeepEP → AWS EFA](deepep.md#expert-parallelism-over-aws-efa)) and is rejected above it.

### RDMA fabric: AWS EFA vs Nebius InfiniBand

The Nebius (IB) tasks keep the image's InfiniBand defaults. A multi-node AWS (EFA) run sets the EFA
launch env: `crossnode-ep` in its `envs`, AWS `nodelocal-ep` in its `run:` block only when
`SKYPILOT_NUM_NODES > 1`, so one node stays NVLink-only. `NCCL_PROTO` is left to the plugin's own probe.

```yaml
envs:                       # AWS crossnode-ep; nodelocal-ep exports the first two on >1 node
  NCCL_NET_PLUGIN: ofi
  NCCL_NET: Libfabric
  NCCL_GIN_TYPE: "2"        # cross-node EP only (proxy GIN)
```

Every task sets `config.docker.run_options: ["--ipc=host", "--ulimit", "memlock=-1", "--ulimit",
"stack=67108864", "--shm-size=128g"]`; only AWS `crossnode-ep` prepends
`"--device", "/dev/gdrdrv"` — node-local EP needs neither proxy GIN nor GDRCopy.

`crossnode-ep` sets `resources.network_tier: best` so SkyPilot provisions the EFA/InfiniBand interfaces; `nodelocal-ep` omits it, so a multi-node `nodelocal-ep` launch passes `--network-tier best`.

On AWS, SkyPilot attaches only one EFA interface unless `aws.use_internal_ips: true` is set in `~/.sky/config.yaml`, which needs a private subnet plus `aws.use_ssm` or `aws.ssh_proxy_command`; without it, all ranks share one NIC. A task file cannot set it.

On Nebius that flag is the only fabric guard: its tasks name no `NCCL_NET`, so a multi-node `nodelocal-ep` launched without it runs the cross-node gradient sync over TCP sockets with no error (AWS's `NCCL_NET=Libfabric` fails the launch instead).

GIN-plugin prerequisites and the measured B300 EFA numbers: [DeepEP → Expert parallelism over AWS EFA](../infrastructure/deepep.md#expert-parallelism-over-aws-efa). Per-fabric env and verification (`fi_info -p efa` vs `ibstat`): [Multi-Node → RDMA fabrics](../parallelism/multi-node.md#rdma-fabrics).

### NVL72 racks (GB200/GB300)

On NVL72 the NVLink domain spans the rack (72 GPUs across ~18 OS nodes), so node-local grouping should target the rack: set `NVLINK_DOMAIN_SIZE: "72"` and `EXPERT_PARALLEL_SCOPE: "node"` in `envs`. Standard 8-GPU hosts need no change — the default reads `gpus_per_node`. See [Multi-Node → NVL72](../parallelism/multi-node.md#gb200gb300-nvl72-multi-node-nvlink). The published images are x86_64-only, so the rack's Grace (aarch64) hosts need an arm64 build first ([Docker](docker.md)).

### NCCL collective timeout

Model loading rarely holds a collective for long: the source agreement and the eager loads join over the c10d store (`DIST_STORE_TIMEOUT_HOURS`), and the default lazy EP load reads on every rank at once ([Multi-Node → Model loading](../parallelism/multi-node.md#model-loading)). A gathered save does: the non-writing ranks wait in its per-layer expert gathers while the save rank streams each flush to disk ([Checkpoints → Multi-node](../reference/checkpoints.md#multi-node)). Every shipped `crossnode-ep` task pins `DIST_NCCL_TIMEOUT_MINUTES: "60"` in `envs`; raise it there if a save is slower still.

## Storage

Every shipped task declares two mounts:

```yaml
file_mounts:
  /workspace: {source: ., mode: COPY}                  # the repo
  /data:      {source: "s3://${HALO_BUCKET}/halo", mode: MOUNT}   # bucket from envs; --env HALO_BUCKET=... retargets
```

`HALO_BUCKET` is the placeholder you must set (`--env HALO_BUCKET=<your-bucket>`); `HF_HOME`,
`HALO_DATA_ROOT` (the toolkit scratch root) and `OUTPUT_DIR` all resolve under `/data`.

| Mode | Use case | Write |
|------|----------|-------|
| `COPY` | Small datasets (<100GB), read-only | No |
| `MOUNT` | Checkpoints, write-heavy persistence | Yes |
| `MOUNT_CACHED` | Large models/datasets, read-heavy | Yes (async) |

Pre-upload large models (>10GB) to S3 once. `/data` is the bucket's `halo/` prefix, so sync below it:

```bash
aws s3 sync models/qwen3.5-122b-a10b s3://<your-bucket>/halo/models/qwen3.5-122b-a10b/
```

Then point the run at it with `--env MODEL=/data/models/qwen3.5-122b-a10b`; the shipped tasks default
`MODEL` to a Hub id and keep only their writes under `/data` (`OUTPUT_DIR: /data/checkpoints/...`).
For `MOUNT_CACHED` reads, add a second `file_mounts` entry for the model prefix and point `MODEL` at
it, keeping checkpoints on the `MOUNT`. Alternatively download from HF at runtime with `HF_HOME` on a
`MOUNT_CACHED` bucket so the cache survives restarts.

In multi-node training a `MOUNT` checkpoint bucket is shared across all nodes; any node can save and
reload after preemption. Pull results with `aws s3 sync`,
`rsync -Pavz oss-120b:/data/checkpoints/ ./`, or SSH; manage buckets with `sky storage ls` / `delete`.

## Docker image

`resources.image_id` is **not set in the shipped YAMLs — uncomment it or pass `--image-id`.** Each carries
it as a commented line under `resources` naming the prebuilt image for its shipped GPUs, which pulls
anonymously (no AWS account, no registry login).

Each task's `setup:` block fails fast when `flash_attn` / `deep_ep` are not importable, so a from-source
build on a bare node is not a fallback; the prebuilt image is required. The `hopper` image serves
H100/H200 (`crossnode-ep`, the GPT-OSS 20B preset) and `blackwell` B200/B300 (`nodelocal-ep` as shipped). To run your own build
instead, push it to a registry the nodes can reach and point `image_id` there ([Docker](docker.md)).

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
- **NCCL errors**: check the NCCL env vars in the YAML and the launch's `--env`; inspect setup with `sky logs <cluster> --status`.

OOM, NCCL timeouts and config rejections: [Troubleshooting](../reference/troubleshooting.md).

## Resources

- [SkyPilot docs](https://docs.skypilot.co/) · [YAML spec](https://docs.skypilot.co/en/latest/reference/yaml-spec.html) · [Storage](https://docs.skypilot.co/en/latest/reference/storage.html)
- [Nebius SkyPilot integration](https://docs.nebius.com/3p-integrations/skypilot)
- [Multi-Node](../parallelism/multi-node.md) · [Expert Parallelism](../parallelism/expert-parallelism.md)
