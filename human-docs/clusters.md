# Clusters and Multi-Node

Multi-node training is one `torchrun` per node, each inside its own container
running the same image. There is no in-repo scheduler; you bring one (SkyPilot
and Nomad templates are included) or start the processes yourself.

## The launch pattern

On every node, from a container started as in
[Installation](installation.md) plus `--network host`:

```bash
torchrun --nnodes=2 --node_rank=$NODE_RANK --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR --master_port=29500 \
    scripts/training/sft.py examples/sft/gptoss/gptoss-20b-multinode-ep.yaml
```

Only `--node_rank` differs between nodes. Everything else — `--nnodes`,
`--nproc_per_node`, the master address and port, the config, every parallelism
flag — must be identical everywhere, or the job dies at startup with a
process-group error.

`MASTER_ADDR` has to be an IP the compute fabric can route (the InfiniBand or
private address, not the public SSH one), and port 29500 has to be reachable
node-to-node. The image must already be present on every node: Flash Attention
and DeepEP are compiled into it, and building from source on a bare node is not
supported.

On SLURM, launch with `--ntasks-per-node=1` and pass `$SLURM_NODEID` as the node
rank — one torchrun per node, not one per GPU.

## Storage: shared or not

`DIST_SHARED_FILESYSTEM=1` (the default) means nodes share a filesystem
(NFS/Lustre) and only rank 0 writes checkpoints and downloads. On per-node local
disk — RunPod pods, ephemeral NVMe — set it to `0`; each node then saves its own
copy and resume works without manual copying.

The variable is an umbrella over a read side (`DIST_INPUT_SHARED_FILESYSTEM`:
downloads, dataset map/pack, HF caches) and a write side
(`DIST_OUTPUT_SHARED_FILESYSTEM`: checkpoints, `run.log`). They want opposite
settings on a flaky NFS/EFS mount, where rank 0 writing the HF cache while
remote ranks read the same inodes surfaces as `Stale file handle`: set the input
side to `0` and leave the output side shared, so checkpoints stay one
authoritative copy.

Either way, one rank going first is a bounded wait —
`DIST_STORE_TIMEOUT_HOURS`, default 4. Raise it when a 100B-scale download or a
whole-corpus pack outlasts that while the other ranks wait.

## Network fabric

InfiniBand works out of the box; the sensible NCCL settings are baked into the
image. Two situations need extra environment:

- **AWS EFA**: add `NCCL_NET_PLUGIN=ofi FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1
  NCCL_PROTO=simple`, and for cross-node expert parallelism also
  `NCCL_GIN_TYPE=2` plus `--device /dev/gdrdrv` on the container. Don't set
  these on an InfiniBand cluster — they make it slower.
- **Multiple NICs**: point `NCCL_SOCKET_IFNAME` at the fast interface so NCCL's
  bootstrap doesn't wander onto the management network.

On GB200/GB300 NVL72 racks, set `NVLINK_DOMAIN_SIZE=72` so Halo knows the NVLink
domain is the rack, not the node.

Cross-node expert parallelism needs `--ep_scope=global` and a real RDMA fabric;
the ready-made template is
`examples/sft/gptoss/gptoss-20b-multinode-ep.yaml`.

## HSDP: fewer collectives over the fabric

Plain FSDP2 shards every parameter across the whole job, so each all-gather and
reduce-scatter crosses the fabric. `--use_hsdp` makes the mesh two-dimensional —
shard within an NVLink domain, replicate across domains — leaving the
cross-domain gradient all-reduce as the only collective that leaves the node. It
covers the standard data-parallel path only (pure DP or CP): EP, TP, and ETP are
rejected at startup, and on a single-domain job the flag is a no-op.

## SkyPilot

`launcher-configs/skypilot/` holds launchable task YAMLs for AWS and Nebius:
GPT-OSS 20B (node-local EP), plus GPT-OSS 120B and Qwen3.5-122B in node-local
and cross-node EP variants. They handle rendezvous, fabric setup, and storage
mounts:

```bash
pip install "skypilot-nightly[aws]"   # or [nebius]
sky launch -c oss-120b launcher-configs/skypilot/aws/gpt-oss-120b/crossnode-ep.yaml \
    --secret HF_TOKEN --secret WANDB_API_KEY
sky logs oss-120b --follow
```

Each YAML ships `resources.image_id` commented out — uncomment it and point it
at a registry holding your image before launching. Details:
[SkyPilot](../agent-docs/infrastructure/skypilot.md) ↗.

## RunPod

No automation, but a manual runbook: pods in one datacenter on the same
InfiniBand fabric, `DIST_SHARED_FILESYSTEM=0`, and gathered (not sharded)
checkpoint saves. Follow [RunPod](../agent-docs/infrastructure/runpod.md) ↗.

## Nomad

`launcher-configs/nomad/` holds batch job specs for a Nomad cluster you already
run: a single-GPU LoRA job, 8-GPU node-local EP, and a two-node EP recipe. Nomad
provisions nothing — the GPU clients, the NVIDIA device plugin, and the scratch
disk are yours — so these sit closer to raw `docker run` than the SkyPilot tasks
do.

```bash
nomad job plan launcher-configs/nomad/qwen3.5-35b-a3b-8gpu-ep.nomad.hcl   # dry run
nomad job run  launcher-configs/nomad/qwen3.5-35b-a3b-8gpu-ep.nomad.hcl
```

One caveat before the two-node job: Nomad has no gang scheduling, so the two
ranks are placed independently and a job can half-place, rank 0 holding 8 GPUs
while rank 1 waits. The spec bounds that with a rendezvous timeout and
`max_run_duration` rather than preventing it. Details:
[Nomad](../agent-docs/infrastructure/nomad.md) ↗.

## When it hangs

NCCL timeouts, rendezvous mistakes, and stragglers have their own section in
[Troubleshooting](troubleshooting.md). Per-topology launch commands:
[Multi-Node](../agent-docs/parallelism/multi-node.md) ↗ ·
[Launch Recipes](../agent-docs/parallelism/launch-recipes.md) ↗.
