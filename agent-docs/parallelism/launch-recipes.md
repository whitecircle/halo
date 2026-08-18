# Multi-Node Launch Recipes

Runnable launches for multi-node training: torchrun and SLURM commands, EP checkpoint save/load, filesystem setup, and sharded data loading. For the process-group layout, node-local vs cross-node scope, and the `ParallelismConfig` API, see [Multi-Node Training](multi-node.md). For SkyPilot cluster provisioning, see [SkyPilot](../infrastructure/skypilot.md).

These recipes assume **N × 8-GPU nodes on an RDMA fabric** (InfiniBand/RoCE or AWS EFA — [RDMA fabrics](multi-node.md#rdma-fabrics)). Launch one `torchrun` per node (it forks the 8 per-GPU ranks); EP all-to-all uses NVLink within a node and RDMA only when `ep_scope="global"` makes the EP group span nodes. A ready-to-run example covering cross-node EP, node-local EP + DP, EP+TP, and the NVL72 variant ships at `examples/sft/gptoss/gptoss-20b-multinode-ep.yaml`.

## torchrun

One `torchrun` per node, incrementing `--node_rank`. The config sets the parallelism layout — it ships `expert_parallel_size: 16` / `ep_scope: global`, cross-node EP across both nodes — and every field overrides on the command line:

- **Node-local EP + DP:** `--expert_parallel_size=8 --ep_scope=node`.
- **PP+EP** (one stage per node) is a planned recipe — pipeline parallelism is [not yet available in this release](pipeline-parallelism.md). For cross-node depth today use cross-node EP, node-local EP + DP, or EP+TP.

```bash
# Node 0 (master): --node_rank=0 ; Node 1: --node_rank=1
torchrun \
    --nnodes=2 \
    --node_rank=0 \
    --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/training/sft.py \
    examples/sft/gptoss/gptoss-20b-multinode-ep.yaml
```

## SLURM

Exactly **one** task per node — `srun` launches one `torchrun` per node, which forks the per-GPU ranks. `--ntasks-per-node=8` would launch 8 torchruns × 8 = 64 procs/node.

A bare `srun --ntasks-per-node=N <script>` — one task per GPU, no `torchrun` — is refused at startup: SLURM declares the world through `SLURM_NTASKS` but supplies no `env://` rendezvous, so `MASTER_ADDR`/`MASTER_PORT` are unset and no process group can be built. Use the recipe below, or export both identically on every task.

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/training/sft.py \
    examples/sft/gptoss/gptoss-20b-multinode-ep.yaml
```

## Other schedulers

There is no scheduler-specific integration. Any orchestrator (Kubernetes, Kubeflow PyTorchJob,
Nomad, Ray, plain SSH) launches the same way: one `torchrun` per node with its `--node_rank`, all
pointed at one rendezvous endpoint. The toolkit reads rank and topology only from the standard
torchrun and `SLURM_*` variables, so map your scheduler's node index onto `--node_rank` — Kubernetes
`JOB_COMPLETION_INDEX`, SkyPilot `SKYPILOT_NODE_RANK`, Nomad `NOMAD_ALLOC_INDEX` within one group (the
shipped specs instead declare one group per rank and set `NODE_RANK` literally —
[Nomad](../infrastructure/nomad.md)).

Managed cloud recipes: [SkyPilot](../infrastructure/skypilot.md) (AWS EFA / Nebius IB) and
[RunPod](../infrastructure/runpod.md) (InfiniBand pods).

## Environment variables

```bash
export DIST_SHARED_FILESYSTEM=1      # default; "0" for per-node local storage
export DIST_NCCL_TIMEOUT_MINUTES=60  # NCCL watchdog; toolkit default 30 (PyTorch's own is 10)
export DIST_STORE_TIMEOUT_HOURS=4    # default; bounds c10d-store waits (main-first download, load gate)
export NCCL_SOCKET_IFNAME=<your fast NIC>  # multi-homed node: `ib0` on IB, the ENA iface on AWS
# The image bakes the InfiniBand / RoCE defaults: NCCL_IB_HCA=mlx5, NCCL_IB_DISABLE=0,
# NCCL_NET_GDR_LEVEL=2 (GPU Direct RDMA for inter-node EP), NCCL_P2P_LEVEL=NVL (NVLink intra-node P2P),
# NCCL_DEBUG=WARN, and CUDA_DEVICE_MAX_CONNECTIONS=1 (DeepEP's free default, +9.7% on ep8; the
# driver latches it at cuInit, so a launch outside the image must export it before the process starts).
# On IB, NCCL_NET_PLUGIN stays unset → HPC-X. Set NCCL_IB_HCA only for a non-default HCA.
# AWS EFA: set the plugin explicitly — the base's shinit_v2 sets it only for a shell that sources
# /etc/shinit_v2 — plus the reliability tuning (would degrade an IB cluster, so it is not baked):
# export NCCL_NET_PLUGIN=ofi
# export FI_PROVIDER=efa
# export FI_EFA_USE_DEVICE_RDMA=1
# export NCCL_PROTO=simple
# Cross-node EP (ep_scope=global) over EFA additionally needs proxy GIN + GDRCopy:
# export NCCL_GIN_TYPE=2             # proxy GIN (EFA has no IBGDA)
# and run the container with `--device /dev/gdrdrv` (host loads the gdrdrv module).
# export NVLINK_DOMAIN_SIZE=72        # GB200/GB300 NVL72 only; leave unset on ≤8-GPU nodes
# export NCCL_DEBUG=INFO              # optional; overrides the image's WARN
```

Per-fabric env and the `aws-ofi-nccl` plugin nuance:
[Multi-Node → RDMA fabrics](multi-node.md#rdma-fabrics).

`init_distributed` pins `DIST_NCCL_TIMEOUT_MINUTES` as the default for every later
`dist.new_group()`. EP/CP/PP subgroups pass it explicitly; the pin is what carries it onto the
DTensor meshes, since `init_device_mesh` takes no timeout — that is the DP/HSDP/TP axes. It does
**not** bound the DeepEP dispatch/combine kernels, which carry the `ElasticBuffer` GPU-side barrier
rather than the PyTorch watchdog.

Inter-node EP runs the DeepEP V2 NCCL **Gin** (RDMA) backend; intra-node EP runs the non-Gin NVLink
path. The dispatcher sets `EP_DISABLE_GIN` from the EP topology (honoring an explicit value).
`EP_SUPPRESS_NCCL_CHECK=1` is baked into both images as an `ENV` and must reach the process
environment — DeepEP latches it at `import deep_ep`, so a Python-level write is too late; the
dispatcher can only warn. Without it DeepEP's duplicate-NCCL guard flags the image's HPC-X transport
plugin as a second NCCL runtime.

Cross-node EP over EFA also rides the GIN-capable `aws-ofi-nccl` + GDRCopy the image bakes. Full
prerequisites: [DeepEP → AWS EFA](../infrastructure/deepep.md#expert-parallelism-over-aws-efa).

## Checkpointing with EP

`save_sharded_ep` picks the save mode, and a sharded save merges before loading:

```bash
python scripts/after_training/merge_ep_shards.py --input_dir checkpoint/ --output_dir merged/
```

The two modes, save-rank memory, and every `save_sharded_ep` rejection live on [Expert Parallelism → Checkpointing & state dict](expert-parallelism.md#checkpointing); resume semantics are on [Checkpoints & Resume](../reference/checkpoints.md).

## File system

The toolkit coordinates file operations (model downloads, dataset loading, directory creation, checkpoint saves) based on whether nodes share a filesystem. Set `DIST_SHARED_FILESYSTEM=1` (default) for shared storage (NFS, Lustre, GPFS) or `0` for per-node local storage; the read and write sides can be split with `DIST_INPUT_SHARED_FILESYSTEM` / `DIST_OUTPUT_SHARED_FILESYSTEM` (e.g. per-node input caches with shared-mount checkpoints on a slow NFS export). Full details: [Filesystem Handling](../data/filesystem-handling.md).

Full-checkpoint sizes scale ~2 bytes/param (bf16): 7B MoE ≈ 14 GB, 20B ≈ 40 GB, 70B ≈ 140 GB; per-rank sharded is roughly `full / ep_size`.

## Sharded data loading

For large-scale multi-node training, pre-process and shard SFT datasets offline so each rank loads only its `1/data_parallel_size` slice. Pre-processing is **SFT-only** — DPO, SMPO, and GRPO require on-the-fly processing.

```bash
python scripts/before_training/prepare_dataset.py \
    --input "s3://my-bucket/raw/my_sft_data" \
    --output "s3://my-bucket/preprocessed/my_sft_data" \
    --model-name "Qwen/Qwen3-8B" \
    --max-length 8192 \
    --num-shards 64 \
    --pack-sequences \
    --conversation-field conversation \
    --assistant-message-template $'<|im_start|>assistant\n'
```

The SFT trainer auto-detects the sharded layout and loads each rank's shards via `ShardedDatasetLoader` keyed on the **data-parallel** rank/size, not global world. Set `num_shards >= data_parallel_size`. Layout, the `num_shards >= data_parallel_size` hard requirement, and shard assignment are on [Data Loading → Pre-processed (sharded) datasets](data-loading.md#pre-processed-sharded-datasets).

## Troubleshooting

**Preflight NVLink health** — EP all-to-all rides NVLink intra-node, so a marginal lane silently caps throughput or hangs a run. `scripts/profiling/nvlink_health.py` reads `nvidia-smi` counters (no GPU allocation, safe alongside a live job), separates hard faults from FEC-absorbed correctable churn, and exits non-zero on any unhealthy link — run it as a gate before a long multi-node launch:

```bash
python scripts/profiling/nvlink_health.py            # summary, exit 1 if unhealthy
python scripts/profiling/nvlink_health.py --per-link  # every link
python scripts/profiling/nvlink_health.py --json      # raw report, for a wrapper to parse
```

**NVSHMEM not found for inter-node EP** — DeepEP V2 still links against NVSHMEM for device linking (`nvidia-nvshmem-cu13` on CUDA 13.x, a transitive dep of `torch 2.11+cu130`; `nvidia-nvshmem-cu12` on CUDA 12.x — never both). Rebuild DeepEP with the full env the build needs: [DeepEP → Build from source](../infrastructure/deepep.md#build-from-source). `DeepEP/` is a git submodule, so a fresh clone needs `git submodule update --init` first.

**RDMA connection timeout** — verify the fabric first (`ibstat` on IB/RoCE, `fi_info -p efa` on EFA — [RDMA fabrics](multi-node.md#rdma-fabrics)), then re-run with `NCCL_DEBUG=INFO` for the transport logs.

**`CP size (16) cannot exceed the NVLink domain (8)`** — set `cp_size <= nvlink_domain_size` (= `gpus_per_node` on a standard cluster, the rack on NVL72). EP+CP also requires node-local EP (`ep_scope=node` with `ep_group_size == nvlink_domain_size`); `ParallelismConfig._validate_ep_cp` rejects cross-node EP (`ep_scope=global`) combined with CP. Cross-node EP combines with DP and TP (not CP) and requires InfiniBand/RDMA for its all-to-all.

**Rendezvous never completes / one node hangs at startup** — every node must pass the identical `--nnodes`, `--master_addr`, `--master_port` and a distinct `--node_rank`, and every rank must reach `init_distributed` in the same order. Symptom-to-cause table for the rest: [Troubleshooting](../reference/troubleshooting.md).
