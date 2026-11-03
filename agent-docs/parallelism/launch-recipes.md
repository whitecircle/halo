# Multi-Node Launch Recipes

Runnable launches for multi-node training: torchrun and SLURM commands, EP checkpoint save/load and filesystem setup. For the process-group layout, node-local vs cross-node scope, and the `ParallelismConfig` API, see [Multi-Node Training](multi-node.md). For SkyPilot cluster provisioning, see [SkyPilot](../infrastructure/skypilot.md).

These recipes assume **N × 8-GPU nodes on an RDMA fabric** (InfiniBand/RoCE or AWS EFA — [RDMA fabrics](multi-node.md#rdma-fabrics)). Launch one `torchrun` per node (it forks the 8 per-GPU ranks); EP all-to-all uses NVLink within a node and RDMA only when `ep_scope="global"` makes the EP group span nodes. A ready-to-run example covering cross-node EP, node-local EP + DP, EP+TP, and the NVL72 variant ships at `examples/sft/gptoss/gptoss-20b-multinode-ep.yaml`; the two pipeline-parallel recipes, one stage per node, launch the same way ([Pipeline Parallelism → Launching](pipeline-parallelism.md#launching)).

## torchrun

One `torchrun` per node, incrementing `--node_rank`. The config sets the parallelism layout — it ships `expert_parallel_size: 16` / `ep_scope: global`, cross-node EP across both nodes — and every parallelism field overrides on the command line:

- **Node-local EP + DP:** `--expert_parallel_size=8 --ep_scope=node --use_grouped_gemm=true`. The config pins the per-expert loop for `ep16`, and at `ep8` that loop is 28% slower (14,402 against 20,066 tok/s/GPU on one 8× B300 host).

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

On one node the launch drops `--nnodes`, `--node_rank` and `--master_*`: `torchrun --nproc_per_node=8 scripts/training/sft.py <config> --expert_parallel_size=8`. Each axis is one `--*_parallel_size` flag, in the YAML or on the CLI, combinable where [Supported combinations](README.md#supported-combinations) allows, and each needs `torchrun` ([Two launchers](data-parallelism.md#two-launchers)): `accelerate launch`, detected through `ACCELERATE_MIXED_PRECISION` or `ACCELERATE_USE_FSDP` (`is_accelerate_launch`), raises on any of them. A plain `python` launch trains on the one GPU it binds however many are visible: `init_training_script` pins `n_gpu` to 1, so HF never wraps the model in `nn.DataParallel`. `halo launch` is single-node only.

On a multi-socket host, add `--numa-binding node` to each `torchrun`: it runs every rank on its GPU's NUMA node, a gain on host-bound TP, CP and small-batch EP runs ([NUMA binding](../optimization/throughput-benchmarks.md#numa-binding)).

The log-writing rank (global rank 0 on a shared output filesystem, each node's local rank 0 otherwise) tees its stdout and stderr to `<output_dir>/log/run.log`, so a detached run keeps its console without a redirect. Redirecting the launch (`> "$D/<job>.log" 2>&1`, as the [detached Docker launch](../infrastructure/docker.md#standard-training-launch) does) also captures the other ranks, such as a crash off rank 0. A log directory or `tee` that cannot be set up is warned on that rank, and the run continues without the file.

## SLURM

Exactly **one** task per node — `srun` launches one `torchrun` per node, which forks the per-GPU ranks. `--ntasks-per-node=8` would launch 8 torchruns × 8 = 64 procs/node.

A bare `srun --ntasks-per-node=N <script>` — one task per GPU, no `torchrun` — is refused at startup: SLURM declares the world through `SLURM_NTASKS` but supplies no `env://` rendezvous, so `MASTER_ADDR`/`MASTER_PORT` are unset and no process group can be built. Use the recipe below, or export both identically on every task.

Ranks must sit `gpus_per_node` per machine in rank order: every node-local EP/CP/TP/ETP group is a contiguous rank block, so `ParallelismConfig` rejects a round-robin placement (`srun --distribution=cyclic` with an exported rendezvous). One `torchrun` per node, or `--distribution=block`, satisfies it.

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

srun bash -c 'torchrun \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/training/sft.py \
    examples/sft/gptoss/gptoss-20b-multinode-ep.yaml'
```

The single quotes make each task expand its own `$SLURM_NODEID`. Unquoted, the batch shell expands it once, to `0`, so every node claims rank 0 and the rendezvous times out with `DistStoreError`.

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
export DIST_STORE_TIMEOUT_HOURS=4    # default; bounds c10d-store waits (main-first download, load gate, checkpoint write joins)
export NCCL_SOCKET_IFNAME=<your fast NIC>  # multi-homed node: `ib0` on IB, the ENA iface on AWS
# The image bakes the IB/RoCE NCCL defaults and CUDA_DEVICE_MAX_CONNECTIONS=1
# (agent-docs/infrastructure/docker.md#baked-in-environment-variables); InfiniBand/RoCE needs no fabric env.
# export NVLINK_DOMAIN_SIZE=72        # GB200/GB300 NVL72 only; leave unset on ≤8-GPU nodes
# export NCCL_DEBUG=INFO              # optional; overrides the image's WARN
```

The per-fabric launch env — AWS EFA's plugin variables, the proxy GIN cross-node EP needs over EFA, a rollout
server on another node: [Multi-Node → RDMA fabrics](multi-node.md#rdma-fabrics).

`DIST_NCCL_TIMEOUT_MINUTES` is the one NCCL watchdog setting: the process group starts before the
config is parsed, so a training config that sets `ddp_timeout` away from its default is refused. Set
it alike on every node. `init_distributed` pins it as the default for every later
`dist.new_group()`. EP/CP/PP subgroups pass it explicitly (`new_group` or `split_group`); the pin is what carries it onto the
DTensor meshes, since `init_device_mesh` takes no timeout — that is the DP/HSDP/TP axes. It does
**not** bound the DeepEP dispatch/combine kernels, which carry the `ElasticBuffer` GPU-side barrier
rather than the PyTorch watchdog.

Inter-node EP runs the DeepEP V2 NCCL **Gin** (RDMA) backend; intra-node EP runs the non-Gin NVLink
path. The dispatcher sets `EP_DISABLE_GIN` from the EP topology (honoring an explicit value).

`EP_SUPPRESS_NCCL_CHECK=1` is baked into both images as an `ENV` and must reach the process
environment: DeepEP latches it at `import deep_ep`, so a Python-level write is too late, and the
dispatcher can only warn. Without it DeepEP's duplicate-NCCL guard flags the image's HPC-X transport
plugin as a second NCCL runtime.

## Checkpointing with EP

`save_sharded_ep` picks the save mode, and a sharded save merges before loading:

```bash
python scripts/after_training/merge_ep_shards.py --input_dir checkpoint/ --output_dir merged/
```

The two modes, save-rank memory, and every `save_sharded_ep` rejection live on [Expert Parallelism → Checkpointing](expert-parallelism.md#checkpointing); resume semantics are on [Checkpoints & Resume](../reference/checkpoints.md).

## File system

Set `DIST_SHARED_FILESYSTEM=1` (default) for shared storage (NFS, Lustre, GPFS) or `0` for per-node local storage, and split the read and write sides with `DIST_INPUT_SHARED_FILESYSTEM` / `DIST_OUTPUT_SHARED_FILESYSTEM` where they differ (per-node input caches beside shared-mount checkpoints on a slow NFS export). What per-node output storage changes for a parallel run: [Multi-Node → Shared vs node-local storage](multi-node.md#shared-vs-node-local-storage); the coordination and the startup probes: [Filesystem Handling](../data/filesystem-handling.md).

Weights take ~2 bytes/param in bf16 (7B MoE ≈ 14 GB, 20B ≈ 40 GB, 70B ≈ 140 GB), the size of a `save_model` export; per-rank sharded is roughly `full / ep_size`. A training checkpoint keeps fp32 masters (`fp32_router`, `fp32_experts`, `fp32_non_ep_params`) at 4 bytes per element and adds the optimizer state (4 bytes/param for AdamWBF16) unless `save_only_model: true`.

## Preflight

EP all-to-all rides NVLink inside a node, so a marginal lane silently caps throughput or hangs a run.
`scripts/profiling/nvlink_health.py` reads `nvidia-smi` counters (no GPU allocation, safe beside a live
job), separates hard faults from FEC-absorbed churn, and exits non-zero on an unhealthy or down link;
run it before a long launch:

```bash
python scripts/profiling/nvlink_health.py            # summary, exit 1 if unhealthy or down
```

Pre-tokenized, per-rank sharded datasets (`num_shards >= data_parallel_size`):
[Data Loading → Pre-processed (sharded) datasets](data-loading.md#pre-processed-sharded-datasets).
Symptoms and fixes for a failed launch (rendezvous, RDMA, config rejections):
[Troubleshooting](../reference/troubleshooting.md).
